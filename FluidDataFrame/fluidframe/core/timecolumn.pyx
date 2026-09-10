# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False

cimport cython
from libc.stdint cimport int8_t, int32_t, int64_t, uint32_t, uint64_t
from libc.string cimport memcpy, memmove, memset
from libc.stdio cimport snprintf
from libc.stdlib cimport bsearch, qsort

from fluidframe.core._base cimport (
    BaseColumn, aligned_malloc, aligned_realloc, os_aligned_free,
    TYPE_TIME, TYPE_INT, TYPE_DOUBLE, TYPE_CHAR256, TYPE_UCS2,
    TYPE_BOOL, TYPE_DATE, TYPE_CATEGORY, TYPE_DATETIME, _resolve_astype_target,
    _c_shift_left, _c_shift_right, _col_pop_back, _col_pop_front, _col_concat,
    _col_drop_row, _col_gather,
    _col_shift_right_2d_int, _col_shift_left_2d, _col_gather_2d, _col_concat_2d
)
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.doublecolumn cimport DoubleColumn
from fluidframe.core.asciicolumn cimport AsciiColumn
from fluidframe.core.ucs2column cimport UCS2Column
from fluidframe.core.categorycolumn cimport CategoryColumn
from fluidframe.core.boolcolumn cimport BoolColumn
from fluidframe.core.datecolumn cimport DateColumn
from fluidframe.core.datetimecolumn cimport DateTimeColumn

# ============================================================
#  底层纯 C 高性能时间运算、LUT 查表格式化与零拷贝解析
# ============================================================

# 00-99 两位字符查找表（LUT 优化：消除格式化中的所有除法/分支）
cdef extern from *:
    """
    static const char DIGIT_PAIRS[200] =
        "00010203040506070809"
        "10111213141516171819"
        "20212223242526272829"
        "30313233343536373839"
        "40414243444546474849"
        "50515253545556575859"
        "60616263646566676869"
        "70717273747576777879"
        "80818283848586878889"
        "90919293949596979899";
    """
    const char DIGIT_PAIRS[200]

cdef inline void _split_hms(int32_t secs, int32_t* h, int32_t* m, int32_t* s) noexcept nogil:
    """提取时分秒（钳制安全范围：0 ~ 86399）"""
    if secs < 0:
        secs = (secs % 86400 + 86400) % 86400
    elif secs >= 86400:
        secs = secs % 86400

    cdef int32_t tot_m = secs / 60
    s[0] = secs - tot_m * 60
    h[0] = tot_m / 60
    m[0] = tot_m - h[0] * 60

cdef inline void _format_time_c(int32_t secs, char* dest) noexcept nogil:
    """纯 C 查表就地格式化为 'HH:MM:SS'（零除法、零分支、零堆分配）"""
    cdef int32_t h, m, s
    _split_hms(secs, &h, &m, &s)

    cdef int32_t h_idx = h * 2
    cdef int32_t m_idx = m * 2
    cdef int32_t s_idx = s * 2

    dest[0] = DIGIT_PAIRS[h_idx]
    dest[1] = DIGIT_PAIRS[h_idx + 1]
    dest[2] = b':'
    dest[3] = DIGIT_PAIRS[m_idx]
    dest[4] = DIGIT_PAIRS[m_idx + 1]
    dest[5] = b':'
    dest[6] = DIGIT_PAIRS[s_idx]
    dest[7] = DIGIT_PAIRS[s_idx + 1]
    dest[8] = 0

cdef inline str _time_to_str(int32_t secs):
    cdef char buf[16]
    _format_time_c(secs, buf)
    return buf.decode('ascii')

cdef int32_t _parse_time_str(str s) except -1:
    """极速解析 'HH:MM:SS', 'H:M:S', 'HH:MM' 等时间格式为自午夜起的秒数"""
    cdef bytes b = s.encode('ascii')
    cdef const char* buf = <const char*>b
    cdef Py_ssize_t n = len(b)
    cdef int32_t parts[3]
    parts[0] = 0; parts[1] = 0; parts[2] = 0
    cdef int32_t cur_val = 0
    cdef int32_t part_idx = 0
    cdef Py_ssize_t i
    cdef int c

    for i in range(n):
        c = <int>buf[i]
        if c >= 48 and c <= 57:
            cur_val = cur_val * 10 + (c - 48)
        elif c == 58:
            if part_idx >= 2:
                raise ValueError(f"Invalid time format: '{s}'")
            parts[part_idx] = cur_val
            cur_val = 0
            part_idx += 1
        elif c == 32 or c == 9 or c == 13 or c == 10:
            continue
        else:
            raise ValueError(f"Invalid character in time string: '{s}'")

    parts[part_idx] = cur_val

    if part_idx == 0:
        raise ValueError(f"Invalid time string: '{s}'")

    if parts[0] < 0 or parts[0] > 23 or parts[1] < 0 or parts[1] > 59 or parts[2] < 0 or parts[2] > 59:
        raise ValueError(f"Time component out of bounds in '{s}'")

    return parts[0] * 3600 + parts[1] * 60 + parts[2]

cdef int _cmp_int32(const void* a, const void* b) noexcept nogil:
    cdef int32_t va = (<const int32_t*>a)[0]
    cdef int32_t vb = (<const int32_t*>b)[0]
    return (va > vb) - (va < vb)


# ============================================================
#  TimeColumn 类实现 (全面 int32_t 与 SIMD 向量化优化)
# ============================================================
@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef class TimeColumn(BaseColumn):

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __cinit__(self, Py_ssize_t init_capacity=16, Py_ssize_t cols=1):
        self.capacity = max(init_capacity, 1)
        self.length = 0
        self.dtype = TYPE_TIME
        self._base_obj = None

        cdef int c = <int>cols if cols > 0 else 1
        self.set_meta_c(c, True, c > 1)

        self.data_c = <int32_t*>aligned_malloc(self.capacity * sizeof(int32_t))
        if not self.data_c:
            raise MemoryError()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def get_size(self):
        return self.capacity * sizeof(int32_t)

    def __dealloc__(self):
        if self.get_owns_data_c() and self.data_c != NULL:
            os_aligned_free(self.data_c)
            self.data_c = NULL

    @property
    def cols(self):
        return self.get_cols_c()

    @property
    def rows(self):
        return self.get_rows_c()

    @property
    def is_2d(self):
        return self.get_is_2d_c()

    @property
    def _owns_data(self):
        return self.get_owns_data_c()

    @_owns_data.setter
    def _owns_data(self, bint val):
        if val:
            self._meta |= 0x00010000
        else:
            self._meta &= ~0x00010000

    @staticmethod
    cdef TimeColumn create_view(int32_t* ptr, Py_ssize_t length, object base_obj):
        cdef TimeColumn col = TimeColumn.__new__(TimeColumn)
        col.dtype = TYPE_TIME
        col.capacity = length
        col.length = length
        col.data_c = ptr
        col._base_obj = base_obj
        col.set_meta_c(1, False, False)
        return col

    @staticmethod
    cdef TimeColumn create_view_2d(int32_t* ptr, Py_ssize_t rows, Py_ssize_t cols, object base_obj):
        cdef TimeColumn col = TimeColumn.__new__(TimeColumn)
        cdef Py_ssize_t total = rows * cols
        col.dtype = TYPE_TIME
        col.capacity = total
        col.length = total
        col.data_c = ptr
        col._base_obj = base_obj
        col.set_meta_c(<int>cols, False, True)
        return col

    cdef inline int get_cols_c(self) noexcept nogil:
        cdef int c = <int>(self._meta & 0x0000FFFF)
        return c if c > 0 else 1

    cdef inline bint get_owns_data_c(self) noexcept nogil:
        return (self._meta & 0x00010000) != 0

    cdef inline bint get_is_2d_c(self) noexcept nogil:
        return (self._meta & 0x00020000) != 0

    cdef inline Py_ssize_t get_rows_c(self) noexcept nogil:
        cdef int c = self.get_cols_c()
        return self.length // c if c > 0 else self.length

    cdef inline void set_meta_c(self, int cols, bint owns_data, bint is_2d) noexcept nogil:
        self._meta = (
            (<unsigned int>cols & 0x0000FFFF) |
            ((<unsigned int>(1 if owns_data else 0)) << 16) |
            ((<unsigned int>(1 if is_2d else 0)) << 17)
        )

    cdef void _reserve(self, Py_ssize_t new_cap):
        if not self.get_owns_data_c():
            raise RuntimeError("Cannot reallocate a view/slice column. Copy it first.")
        cdef int32_t* new_ptr
        if new_cap > self.capacity:
            new_ptr = <int32_t*>aligned_realloc(self.data_c, self.capacity * sizeof(int32_t), new_cap * sizeof(int32_t))
            if not new_ptr:
                raise MemoryError()
            self.data_c = new_ptr
            self.capacity = new_cap

    cdef inline int32_t _resolve_time(self, val) except -1:
        if isinstance(val, str):
            return _parse_time_str(<str>val)
        return <int32_t>val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def append(self, val):
        if not self.get_owns_data_c():
            raise RuntimeError("Cannot append to a view/slice column. Copy it first.")
        if self.get_is_2d_c():
            raise NotImplementedError("append is not supported for 2D columns (rows*cols layout); convert to 1D first")
        cdef int32_t v = _parse_time_str(<str>val) if isinstance(val, str) else <int32_t>val
        if self.length >= self.capacity:
            self._reserve(max(16, self.capacity * 2))
        self.data_c[self.length] = v
        self.length += 1

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def copy(self):
        cdef TimeColumn new_col = TimeColumn(max(self.length, 1), cols=self.cols)
        new_col.length = self.length
        if self.length > 0:
            memcpy(new_col.data_c, self.data_c, self.length * sizeof(int32_t))
        return new_col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_back(self):
        if not self.get_owns_data_c():
            raise RuntimeError("Cannot modify a view/slice column. Copy it first.")
        if self.length == 0:
            raise IndexError("pop from empty column")
        cdef Py_ssize_t c = self.get_cols_c()
        cdef Py_ssize_t r, j
        cdef list vals
        if self.get_is_2d_c():
            r = self.get_rows_c()
            vals = [None] * c
            for j in range(c):
                vals[j] = self.data_c[j * r + (r - 1)]
            _col_shift_left_2d(self.data_c, &self.length, r, c, r - 1, sizeof(int32_t))
            return vals
        cdef int32_t* ptr = <int32_t*>_col_pop_back(
            self.data_c, &self.length, sizeof(int32_t))
        return ptr[0]

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_front(self):
        if not self.get_owns_data_c():
            raise RuntimeError("Cannot modify a view/slice column. Copy it first.")
        if self.length == 0:
            raise IndexError("pop from empty column")
        cdef Py_ssize_t c = self.get_cols_c()
        cdef Py_ssize_t r, j
        cdef list vals
        if self.get_is_2d_c():
            r = self.get_rows_c()
            vals = [None] * c
            for j in range(c):
                vals[j] = self.data_c[j * r]
            _col_shift_left_2d(self.data_c, &self.length, r, c, 0, sizeof(int32_t))
            return vals
        cdef int32_t val = self.data_c[0]
        _col_pop_front(self.data_c, &self.length, sizeof(int32_t))
        return val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def concat(self, TimeColumn other):
        if not self.get_owns_data_c():
            raise RuntimeError("Cannot concat into a view/slice column. Copy it first.")

        cdef Py_ssize_t c = self.get_cols_c()
        cdef Py_ssize_t other_c = other.get_cols_c()
        cdef Py_ssize_t other_rows

        if self.get_is_2d_c() or other.get_is_2d_c():
            if c != other_c:
                if self.length == 0:
                    c = other_c
                else:
                    raise ValueError(f"Cannot concat 2D columns with different column counts: {c} vs {other_c}")
            other_rows = other.get_rows_c()
            if other_rows == 0: return

            self._reserve(self.length + other_rows * c)
            _col_concat_2d(self.data_c, &self.length, other.data_c, other_rows, c, sizeof(int32_t))
            self.set_meta_c(<int>c, True, True)
            return
        if other.length == 0: return
        self._reserve(self.length + other.length)
        _col_concat(self.data_c, &self.length, other.data_c, other.length, sizeof(int32_t))

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def max(self):
        if self.length == 0: raise ValueError("empty column")
        cdef int32_t max_val = self.data_c[0]
        cdef int32_t* ptr = self.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(1, n):
                if ptr[i] > max_val:
                    max_val = ptr[i]
        return max_val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def min(self):
        if self.length == 0: raise ValueError("empty column")
        cdef int32_t min_val = self.data_c[0]
        cdef int32_t* ptr = self.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(1, n):
                if ptr[i] < min_val:
                    min_val = ptr[i]
        return min_val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _reorder(self, list indices):
        cdef Py_ssize_t n = len(indices)
        cdef char err_buf[128]
        if n != self.length:
            snprintf(err_buf, sizeof(err_buf), "Length of indices (%zd) must match column length (%zd)", n, self.length)
            raise ValueError(err_buf.decode('ascii'))
        if n == 0: return

        cdef int32_t* new_data = <int32_t*>aligned_malloc(n * sizeof(int32_t))
        if not new_data: raise MemoryError()
        cdef Py_ssize_t i, idx
        cdef Py_ssize_t* c_indices = <Py_ssize_t*>aligned_malloc(n * sizeof(Py_ssize_t))
        if not c_indices:
            os_aligned_free(new_data)
            raise MemoryError()
        try:
            for i in range(n):
                idx = <Py_ssize_t>indices[i]
                if idx < 0: idx += self.length
                if idx < 0 or idx >= self.length:
                    snprintf(err_buf, sizeof(err_buf), "Index out of bounds: %zd", idx)
                    raise IndexError(err_buf.decode('ascii'))
                c_indices[i] = idx
            _col_gather(new_data, self.data_c, c_indices, n, sizeof(int32_t))
        except Exception:
            os_aligned_free(new_data)
            os_aligned_free(c_indices)
            raise
        os_aligned_free(c_indices)

        if self.get_owns_data_c() and self.data_c != NULL:
            os_aligned_free(self.data_c)
        self.data_c = new_data
        self.capacity = n
        self.length = n
        self.set_meta_c(self.get_cols_c(), True, self.get_is_2d_c())
        self._base_obj = None

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __getitem__(self, Py_ssize_t idx):
        if idx < 0: idx += self.length
        if idx < 0 or idx >= self.length: raise IndexError("Index out of bounds")
        return _time_to_str(self.data_c[idx])

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def unique(self):
        cdef set seen = set()
        cdef list result = []
        cdef Py_ssize_t i, n = self.length
        cdef int32_t val
        for i in range(n):
            val = self.data_c[i]
            if val not in seen:
                seen.add(val)
                result.append(val)
        return result

    # ------------------------------------------------------------
    # 时/分/秒提取 (定点乘法展开，完全自动 SIMD 向量化)
    # ------------------------------------------------------------
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def hour(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int32_t* ptr = self.data_c
        cdef int32_t* res_ptr = <int32_t*>res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = ptr[i] / 3600
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def minute(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int32_t* ptr = self.data_c
        cdef int32_t* res_ptr = <int32_t*>res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = (ptr[i] / 60) % 60
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def second(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int32_t* ptr = self.data_c
        cdef int32_t* res_ptr = <int32_t*>res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = ptr[i] % 60
        return res

    # ------------------------------------------------------------
    # 无分支向量化掩码选择
    # ------------------------------------------------------------
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def where_mask(self, IntColumn mask, object fill):
        cdef Py_ssize_t n = self.length
        if mask.length != n:
            raise ValueError(f"Mask length {mask.length} != column length {n}")
        cdef TimeColumn col = TimeColumn(max(n, 1), cols=self.cols)
        col.length = n
        if n == 0: return col

        cdef int32_t fill_val = self._resolve_time(fill) if fill is not None else 0
        cdef int32_t* col_ptr = col.data_c
        cdef int32_t* self_ptr = self.data_c
        cdef int32_t* mask_ptr = <int32_t*>mask.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                col_ptr[i] = self_ptr[i] if mask_ptr[i] != 0 else fill_val
        return col

    # ------------------------------------------------------------
    # isin 极速优化：释放 GIL + C 级别二分检索（零 Python 装箱）
    # ------------------------------------------------------------
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def isin(self, object values):
        cdef set parsed_vals = set()
        for v in values:
            if isinstance(v, str):
                parsed_vals.add(_parse_time_str(<str>v))
            elif v is not None:
                parsed_vals.add(<int32_t>v)

        cdef Py_ssize_t num_keys = len(parsed_vals)
        cdef Py_ssize_t n = self.length
        cdef BoolColumn res = BoolColumn(max(n, 1))
        res.length = n
        if n == 0:
            return res

        if num_keys == 0:
            memset(res.data_c, 0, n)
            return res

        cdef int32_t* keys_c = <int32_t*>aligned_malloc(num_keys * sizeof(int32_t))
        if not keys_c: raise MemoryError()

        cdef Py_ssize_t ki = 0
        for val in parsed_vals:
            keys_c[ki] = <int32_t>val
            ki += 1

        qsort(keys_c, num_keys, sizeof(int32_t), _cmp_int32)

        cdef int32_t* ptr = self.data_c
        cdef unsigned char* res_ptr = res.data_c
        cdef Py_ssize_t i
        cdef int32_t target

        with nogil:
            for i in range(n):
                target = ptr[i]
                res_ptr[i] = 1 if bsearch(&target, keys_c, num_keys, sizeof(int32_t), _cmp_int32) != NULL else 0

        os_aligned_free(keys_c)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def drop_row(self, Py_ssize_t row_idx):
        if not self.get_owns_data_c():
            raise RuntimeError("Cannot drop row on a view/slice column. Copy it first.")
        cdef Py_ssize_t c = self.get_cols_c()
        cdef Py_ssize_t r = self.get_rows_c()
        if row_idx < 0:
            row_idx += r
        if row_idx < 0 or row_idx >= r:
            raise IndexError("Index out of bounds")
        if self.get_is_2d_c():
            _col_shift_left_2d(self.data_c, &self.length, r, c, row_idx, sizeof(int32_t))
        else:
            _col_drop_row(self.data_c, &self.length, row_idx, sizeof(int32_t))

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def shift_left(self, Py_ssize_t pos):
        if not self.get_owns_data_c():
            raise RuntimeError("Cannot modify a view/slice column. Copy it first.")
        cdef Py_ssize_t c = self.get_cols_c()
        cdef Py_ssize_t r = self.get_rows_c()
        cdef Py_ssize_t col_len
        if pos < 0:
            pos += r
        if pos < 0 or pos >= r:
            raise IndexError("Index out of bounds")
        if self.get_is_2d_c():
            _col_shift_left_2d(self.data_c, &self.length, r, c, pos, sizeof(int32_t))
        else:
            col_len = self.length
            _c_shift_left(self.data_c, pos, col_len, sizeof(int32_t))
            self.length = col_len - 1

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def shift_right(self, Py_ssize_t pos):
        if not self.get_owns_data_c():
            raise RuntimeError("Cannot modify a view/slice column. Copy it first.")
        cdef Py_ssize_t c = self.get_cols_c()
        cdef Py_ssize_t r = self.get_rows_c()
        if pos < 0:
            pos += r
        if pos < 0 or pos > r:
            raise IndexError("Index out of bounds")
        if self.get_is_2d_c():
            if self.length + c > self.capacity:
                self._reserve(max(16, max(self.capacity * 2, self.length + c)))
            _col_shift_right_2d_int(self.data_c, &self.length, r, c, pos, sizeof(int32_t))
        else:
            if self.length >= self.capacity:
                self._reserve(max(16, self.capacity * 2))
            _c_shift_right(self.data_c, pos, self.length, sizeof(int32_t))
            self.length += 1
            (<int32_t*>self.data_c)[pos] = 0

    cdef TimeColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=False):
        cdef TimeColumn new_col
        cdef int32_t* new_data
        cdef Py_ssize_t c = self.get_cols_c()
        cdef Py_ssize_t r = self.get_rows_c()

        if self.get_is_2d_c():
            if not inplace:
                new_col = TimeColumn(max(n * c, 1), cols=c)
                new_col.length = n * c
                _col_gather_2d(new_col.data_c, self.data_c, indices, n, r, c, sizeof(int32_t))
                return new_col
            else:
                new_data = <int32_t*>aligned_malloc(max(n * c, 1) * sizeof(int32_t))
                if not new_data: raise MemoryError()
                _col_gather_2d(new_data, self.data_c, indices, n, r, c, sizeof(int32_t))
                if self.get_owns_data_c() and self.data_c != NULL:
                    os_aligned_free(self.data_c)
                self.data_c = new_data
                self.capacity = n * c
                self.length = n * c
                self.set_meta_c(<int>c, True, True)
                self._base_obj = None
                return self

        if not inplace:
            new_col = TimeColumn(max(n, 1), cols=self.cols)
            new_col.length = n
            _col_gather(new_col.data_c, self.data_c, indices, n, sizeof(int32_t))
            return new_col
        else:
            new_data = <int32_t*>aligned_malloc(max(n, 1) * sizeof(int32_t))
            if not new_data: raise MemoryError()
            _col_gather(new_data, self.data_c, indices, n, sizeof(int32_t))
            if self.get_owns_data_c() and self.data_c != NULL:
                os_aligned_free(self.data_c)
            self.data_c = new_data
            self.capacity = n
            self.length = n
            self.set_meta_c(self.get_cols_c(), True, self.get_is_2d_c())
            self._base_obj = None
            return self

    # ------------------------------------------------------------
    # 纯 C 32位无分支比较算子 (完全向量化)
    # ------------------------------------------------------------
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def _cmp_op(self, val, int op):
        cdef Py_ssize_t i, n = self.length
        cdef IntColumn res = IntColumn(max(n, 1))
        res.length = n

        cdef int32_t* s_ptr = self.data_c
        cdef int32_t* o_ptr
        cdef int32_t* r_ptr = <int32_t*>res.data_c
        cdef char err_buf[128]
        cdef int32_t v
        cdef TimeColumn other

        if isinstance(val, TimeColumn):
            other = <TimeColumn>val
            if other.length != n:
                snprintf(err_buf, sizeof(err_buf), "Length mismatch: %zd vs %zd", n, other.length)
                raise ValueError(err_buf.decode('ascii'))
            o_ptr = other.data_c
            with nogil:
                if op == 0:
                    for i in range(n): r_ptr[i] = <int32_t>(s_ptr[i] < o_ptr[i])
                elif op == 1:
                    for i in range(n): r_ptr[i] = <int32_t>(s_ptr[i] <= o_ptr[i])
                elif op == 2:
                    for i in range(n): r_ptr[i] = <int32_t>(s_ptr[i] > o_ptr[i])
                elif op == 3:
                    for i in range(n): r_ptr[i] = <int32_t>(s_ptr[i] >= o_ptr[i])
                elif op == 4:
                    for i in range(n): r_ptr[i] = <int32_t>(s_ptr[i] == o_ptr[i])
                elif op == 5:
                    for i in range(n): r_ptr[i] = <int32_t>(s_ptr[i] != o_ptr[i])
        else:
            v = self._resolve_time(val)
            with nogil:
                if op == 0:
                    for i in range(n): r_ptr[i] = <int32_t>(s_ptr[i] < v)
                elif op == 1:
                    for i in range(n): r_ptr[i] = <int32_t>(s_ptr[i] <= v)
                elif op == 2:
                    for i in range(n): r_ptr[i] = <int32_t>(s_ptr[i] > v)
                elif op == 3:
                    for i in range(n): r_ptr[i] = <int32_t>(s_ptr[i] >= v)
                elif op == 4:
                    for i in range(n): r_ptr[i] = <int32_t>(s_ptr[i] == v)
                elif op == 5:
                    for i in range(n): r_ptr[i] = <int32_t>(s_ptr[i] != v)
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __lt__(self, val):  return self._cmp_op(val, 0)
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __le__(self, val):  return self._cmp_op(val, 1)
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __gt__(self, val):  return self._cmp_op(val, 2)
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __ge__(self, val):  return self._cmp_op(val, 3)
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __eq__(self, val):  return self._cmp_op(val, 4)
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __ne__(self, val):  return self._cmp_op(val, 5)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __str__(self):
        cdef list items = []
        cdef Py_ssize_t i, n = self.length
        cdef char buf[16]
        for i in range(n):
            _format_time_c(self.data_c[i], buf)
            items.append(buf.decode('ascii'))
        return f"TimeColumn([{', '.join(items)}])"

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __bool__(self):
        return self.length > 0

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __len__(self):
        return self.length

    # ------------------------------------------------------------
    # 类型转换 astype (直接 C 级内存赋值与零堆开销)
    # ------------------------------------------------------------
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def astype(self, object target):
        cdef int target_type = _resolve_astype_target(target)
        cdef Py_ssize_t i, n = self.length
        cdef Py_ssize_t c = self.cols
        cdef Py_ssize_t first_col_n = self.rows if c > 1 else n
        cdef Py_ssize_t safe_cap = max(n, 1)
        cdef Py_ssize_t first_col_cap = max(first_col_n, 1)
        cdef int32_t* ptr = self.data_c
        cdef char str_buf[16]

        cdef IntColumn ic
        cdef DoubleColumn dc
        cdef BoolColumn bc
        cdef UCS2Column uc
        cdef AsciiColumn cc
        cdef CategoryColumn catc
        cdef DateColumn tc
        cdef DateTimeColumn dtc

        if target_type == TYPE_INT:
            ic = IntColumn(safe_cap, cols=self.cols)
            ic.length = n
            with nogil:
                for i in range(n):
                    ic.data_c[i] = ptr[i]
            return ic
        elif target_type == TYPE_DOUBLE:
            dc = DoubleColumn(safe_cap, cols=self.cols)
            dc.length = n
            with nogil:
                for i in range(n):
                    dc.data_c[i] = <double>ptr[i]
            return dc
        elif target_type == TYPE_BOOL:
            bc = BoolColumn(first_col_cap)
            bc.length = first_col_n
            with nogil:
                for i in range(first_col_n):
                    bc.data_c[i] = <unsigned char>(ptr[i] != 0)
            return bc
        elif target_type == TYPE_CHAR256:
            cc = AsciiColumn(first_col_cap)
            cc.length = first_col_n
            with nogil:
                for i in range(first_col_n):
                    _format_time_c(ptr[i], cc.data_c + (i * 256))
            return cc
        elif target_type == TYPE_UCS2:
            uc = UCS2Column(first_col_cap)
            uc.length = first_col_n
            for i in range(first_col_n):
                _format_time_c(ptr[i], str_buf)
                uc._set_item(i, str_buf.decode('ascii'))
            return uc
        elif target_type == TYPE_CATEGORY:
            catc = CategoryColumn(first_col_cap)
            for i in range(first_col_n):
                _format_time_c(ptr[i], str_buf)
                catc.append(str_buf.decode('ascii'))
            return catc
        elif target_type == TYPE_DATE:
            tc = DateColumn(safe_cap, cols=self.cols)
            tc.length = n
            with nogil:
                for i in range(n):
                    tc.data_c[i] = 0
            return tc
        elif target_type == TYPE_DATETIME:
            dtc = DateTimeColumn(safe_cap, cols=self.cols)
            dtc.length = n
            # Time -> DateTime: 秒数转换为微秒数 (1970-01-01 当天对应的时间点)
            with nogil:
                for i in range(n):
                    dtc.data_c[i] = <int64_t>ptr[i] * 1000000LL
            return dtc
        elif target_type == TYPE_TIME:
            return self.copy()

        raise TypeError(f"Failed to cast TIME column to internal type id: {target_type}")