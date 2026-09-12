# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False

cimport cython
from libc.stdint cimport int32_t, uint32_t, int64_t, uint64_t
from libc.string cimport memcpy, memmove, memset
from libc.stdio cimport snprintf
from libc.stdlib cimport malloc, free, qsort, bsearch

from fluidframe.core._base cimport (
    BaseColumn, aligned_malloc, aligned_realloc, os_aligned_free,
    TYPE_DATETIME, TYPE_INT, TYPE_DOUBLE, TYPE_CHAR256, TYPE_UCS2,
    TYPE_BOOL, TYPE_DATE, TYPE_CATEGORY, TYPE_TIME, _resolve_astype_target,
    _c_shift_left, _c_shift_right, _col_pop_back, _col_pop_front, _col_concat,
    _col_drop_row, _col_gather,
    _col_shift_right_2d_int, _col_shift_left_2d, _col_gather_2d, _col_concat_2d,
    _col_gather_zero, _col_gather_2d_zero, _col_memcpy,
    _max_2d_c, _min_2d_c
)
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.doublecolumn cimport DoubleColumn
from fluidframe.core.boolcolumn cimport BoolColumn
from fluidframe.core.asciicolumn cimport AsciiColumn
from fluidframe.core.ucs2column cimport UCS2Column
from fluidframe.core.categorycolumn cimport CategoryColumn
from fluidframe.core.datecolumn cimport DateColumn, _ymd_to_days, _days_to_ymd
from fluidframe.core.timecolumn cimport TimeColumn

# ============================================================
#  时间常数定义 (基于 Unix 1970-01-01 00:00:00 纪元)
# ============================================================

cdef int64_t US_PER_SEC = 1000000
cdef int64_t SEC_PER_DAY = 86400
cdef int64_t US_PER_DAY = 86400000000


cdef inline int64_t _parse_datetime_str(str s) except -1:
    """解析 'YYYY-MM-DD HH:MM:SS' 或 'YYYY-MM-DD' 为 1970 纪元微秒数"""
    cdef bytes b = s.encode('ascii', errors='ignore')
    cdef const unsigned char* buf = <const unsigned char*>b
    cdef int max_len = len(b)
    cdef int digits[20]
    cdef int ndigits = 0
    cdef int i = 0
    cdef int year = 0, month = 1, day = 1, hour = 0, minute = 0, second = 0
    cdef int32_t days
    cdef int64_t us

    while i < max_len and ndigits < 20:
        if buf[i] >= 48 and buf[i] <= 57:
            digits[ndigits] = buf[i] - 48
            ndigits += 1
        i += 1

    if ndigits < 8:
        raise ValueError(f"Invalid datetime string: '{s}'")

    year = digits[0]*1000 + digits[1]*100 + digits[2]*10 + digits[3]
    month = digits[4]*10 + digits[5]
    day = digits[6]*10 + digits[7]
    if ndigits >= 10:
        hour = digits[8]*10 + digits[9]
    if ndigits >= 12:
        minute = digits[10]*10 + digits[11]
    if ndigits >= 14:
        second = digits[12]*10 + digits[13]

    # 直接基于 1970 纪元天数换算
    days = _ymd_to_days(year, month, day)
    us = <int64_t>days * US_PER_DAY + <int64_t>hour * 3600000000LL + <int64_t>minute * 60000000LL + <int64_t>second * US_PER_SEC
    return us

cdef inline void _datetime_to_ymdhms(int64_t us, int* y, int* m, int* d, int* h, int* mi, int* s) noexcept nogil:
    """1970 纪元微秒数分解为 YMD HMS"""
    cdef int64_t total_seconds = us / US_PER_SEC
    cdef int32_t days = <int32_t>(total_seconds / SEC_PER_DAY)
    cdef int rem = <int>(total_seconds - (<int64_t>days * SEC_PER_DAY))
    if rem < 0:
        rem += SEC_PER_DAY
        days -= 1

    h[0] = rem / 3600
    cdef int rem_m = rem - h[0] * 3600
    mi[0] = rem_m / 60
    s[0] = rem_m - mi[0] * 60

    _days_to_ymd(days, y, m, d)

cdef inline void _format_datetime_c(int64_t us, char* dest) noexcept nogil:
    """纯 C 极速格式化 YYYY-MM-DD HH:MM:SS（零堆内存分配）"""
    cdef int y, m, d, h, mi, s
    _datetime_to_ymdhms(us, &y, &m, &d, &h, &mi, &s)
    dest[0] = <char>(48 + ((y / 1000) % 10))
    dest[1] = <char>(48 + ((y / 100) % 10))
    dest[2] = <char>(48 + ((y / 10) % 10))
    dest[3] = <char>(48 + (y % 10))
    dest[4] = b'-'
    dest[5] = <char>(48 + (m / 10))
    dest[6] = <char>(48 + (m % 10))
    dest[7] = b'-'
    dest[8] = <char>(48 + (d / 10))
    dest[9] = <char>(48 + (d % 10))

    if h == 0 and mi == 0 and s == 0:
        dest[10] = 0
    else:
        dest[10] = b' '
        dest[11] = <char>(48 + (h / 10))
        dest[12] = <char>(48 + (h % 10))
        dest[13] = b':'
        dest[14] = <char>(48 + (mi / 10))
        dest[15] = <char>(48 + (mi % 10))
        dest[16] = b':'
        dest[17] = <char>(48 + (s / 10))
        dest[18] = <char>(48 + (s % 10))
        dest[19] = 0

cdef inline str _datetime_to_str(int64_t us):
    cdef char buf[32]
    _format_datetime_c(us, buf)
    return buf.decode('ascii')

cdef int _cmp_int64(const void* a, const void* b) noexcept nogil:
    cdef int64_t va = (<const int64_t*>a)[0]
    cdef int64_t vb = (<const int64_t*>b)[0]
    if va < vb: return -1
    if va > vb: return 1
    return 0

@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef class DateTimeColumn(BaseColumn):

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __cinit__(self, Py_ssize_t init_capacity=16, Py_ssize_t cols=1):
        self.capacity = max(init_capacity, 1)
        self.length = 0
        self.dtype = TYPE_DATETIME
        self._base_obj = None

        self.num_cols = <int>cols if cols > 0 else 1
        self.is_2d = (self.num_cols > 1)
        self.owns_data = True

        self.data_c = <int64_t*>aligned_malloc(self.capacity * sizeof(int64_t))
        if not self.data_c:
            raise MemoryError()

    cdef void* _get_data_ptr(self) noexcept nogil:
        return <void*>self.data_c

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __dealloc__(self):
        if self.owns_data and self.data_c != NULL:
            os_aligned_free(self.data_c)
            self.data_c = NULL

    @property
    def cols(self):
        return self.num_cols

    @property
    def rows(self):
        return self._get_rows_c()

    @property
    def _owns_data(self):
        return self.owns_data

    @_owns_data.setter
    def _owns_data(self, bint val):
        self.owns_data = val

    @staticmethod
    cdef DateTimeColumn create_view(int64_t* ptr, Py_ssize_t length, object base_obj):
        cdef DateTimeColumn col = DateTimeColumn.__new__(DateTimeColumn)
        col.dtype = TYPE_DATETIME
        col.capacity = length
        col.length = length
        col.data_c = ptr
        col._base_obj = base_obj
        col.num_cols = 1
        col.is_2d = False
        col.owns_data = False
        return col

    @staticmethod
    cdef DateTimeColumn create_view_2d(int64_t* ptr, Py_ssize_t rows, Py_ssize_t cols, object base_obj):
        cdef DateTimeColumn col = DateTimeColumn.__new__(DateTimeColumn)
        cdef Py_ssize_t total = rows * cols
        col.dtype = TYPE_DATETIME
        col.capacity = total
        col.length = total
        col.data_c = ptr
        col._base_obj = base_obj
        col.num_cols = <int>cols
        col.is_2d = True
        col.owns_data = False
        return col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def get_size(self):
        return self.capacity * sizeof(int64_t)

    cdef void _reserve(self, Py_ssize_t new_cap):
        if not self.owns_data:
            raise RuntimeError("Cannot reallocate a view/slice column. Copy it first.")
        cdef int64_t* new_ptr
        if new_cap > self.capacity:
            new_ptr = <int64_t*>aligned_realloc(self.data_c, self.capacity * sizeof(int64_t), new_cap * sizeof(int64_t))
            if not new_ptr:
                raise MemoryError()
            self.data_c = new_ptr
            self.capacity = new_cap

    cdef inline int64_t _resolve_dt(self, val) except *:
        if isinstance(val, str):
            return _parse_datetime_str(<str>val)
        return <int64_t>val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def append(self, val):
        if not self.owns_data:
            raise RuntimeError("Cannot append to a view/slice column. Copy it first.")
        if self.is_2d:
            raise NotImplementedError("append is not supported for 2D columns (rows*cols layout); convert to 1D first")
        cdef int64_t v = _parse_datetime_str(<str>val) if isinstance(val, str) else <int64_t>val
        if self.length >= self.capacity:
            self._reserve(max(16, self.capacity * 2))
        self.data_c[self.length] = v
        self.length += 1

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def copy(self):
        cdef DateTimeColumn new_col = DateTimeColumn(max(self.length, 1), cols=self.cols)
        new_col.length = self.length
        _col_memcpy(new_col.data_c, self.data_c, self.length)
        return new_col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_back(self):
        if not self.owns_data:
            raise RuntimeError("Cannot modify a view/slice column. Copy it first.")
        if self.length == 0:
            raise IndexError("pop from empty column")
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r, j
        cdef list vals
        if self.is_2d:
            r = self._get_rows_c()
            vals = [None] * c
            for j in range(c):
                vals[j] = self.data_c[j * r + (r - 1)]
            _col_shift_left_2d(self.data_c, &self.length, r, c, r - 1, sizeof(int64_t))
            return vals
        cdef int64_t* ptr = <int64_t*>_col_pop_back(
            self.data_c, &self.length, sizeof(int64_t))
        return ptr[0]

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_front(self):
        if not self.owns_data:
            raise RuntimeError("Cannot modify a view/slice column. Copy it first.")
        if self.length == 0:
            raise IndexError("pop from empty column")
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r, j
        cdef list vals
        if self.is_2d:
            r = self._get_rows_c()
            vals = [None] * c
            for j in range(c):
                vals[j] = self.data_c[j * r]
            _col_shift_left_2d(self.data_c, &self.length, r, c, 0, sizeof(int64_t))
            return vals
        cdef int64_t val = (<int64_t*>self.data_c)[0]
        _col_pop_front(self.data_c, &self.length, sizeof(int64_t))
        return val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def concat(self, DateTimeColumn other):
        if not self.owns_data:
            raise RuntimeError("Cannot concat into a view/slice column. Copy it first.")

        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t other_c = other.num_cols
        cdef Py_ssize_t other_rows

        if self.is_2d or other.is_2d:
            if c != other_c:
                if self.length == 0:
                    c = other_c
                else:
                    raise ValueError(f"Cannot concat 2D columns with different column counts: {c} vs {other_c}")
            other_rows = other._get_rows_c()
            if other_rows == 0: return

            self._reserve(self.length + other_rows * c)
            _col_concat_2d(self.data_c, &self.length, other.data_c, other_rows, c, sizeof(int64_t))
            self.num_cols = <int>c
            self.is_2d = True
            return
        if other.length == 0: return
        self._reserve(self.length + other.length)
        _col_concat(self.data_c, &self.length, other.data_c, other.length,
                     sizeof(int64_t))

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def max(self):
        if self.length == 0: raise ValueError("empty column")
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()
        cdef int64_t* ptr = <int64_t*>self.data_c
        cdef Py_ssize_t i, j, base
        cdef int64_t max_val

        if not self.is_2d or c <= 1:
            max_val = ptr[0]
            with nogil:
                for i in range(1, self.length):
                    if ptr[i] > max_val:
                        max_val = ptr[i]
            return max_val

        cdef list results = [None] * c
        for j in range(c):
            base = j * r
            max_val = ptr[base]
            for i in range(1, r):
                if ptr[base + i] > max_val:
                    max_val = ptr[base + i]
            results[j] = max_val
        return results

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def min(self):
        if self.length == 0: raise ValueError("empty column")
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()
        cdef int64_t* ptr = <int64_t*>self.data_c
        cdef Py_ssize_t i, j, base
        cdef int64_t min_val

        if not self.is_2d or c <= 1:
            min_val = ptr[0]
            with nogil:
                for i in range(1, self.length):
                    if ptr[i] < min_val:
                        min_val = ptr[i]
            return min_val

        cdef list results = [None] * c
        for j in range(c):
            base = j * r
            min_val = ptr[base]
            for i in range(1, r):
                if ptr[base + i] < min_val:
                    min_val = ptr[base + i]
            results[j] = min_val
        return results

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _reorder_impl(self, list indices):
        cdef Py_ssize_t n = len(indices)
        cdef char err_buf[128]
        if n != self.length:
            snprintf(err_buf, sizeof(err_buf), "Length of indices (%zd) must match column length (%zd)", n, self.length)
            raise ValueError(err_buf.decode('ascii'))
        if n == 0: return

        cdef int64_t* new_data = <int64_t*>aligned_malloc(n * sizeof(int64_t))
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
            _col_gather(new_data, self.data_c, c_indices, n, sizeof(int64_t))
        except Exception:
            os_aligned_free(new_data)
            os_aligned_free(c_indices)
            raise
        os_aligned_free(c_indices)

        if self.owns_data and self.data_c != NULL:
            os_aligned_free(self.data_c)
        self.data_c = new_data
        self.capacity = n
        self._base_obj = None

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __getitem__(self, Py_ssize_t idx):
        if idx < 0: idx += self.length
        if idx < 0 or idx >= self.length: raise IndexError("Index out of bounds")
        return _datetime_to_str((<int64_t*>self.data_c)[idx])

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def unique(self):
        cdef set seen = set()
        cdef list result = []
        cdef Py_ssize_t i, n = self.length
        cdef int64_t* ptr = <int64_t*>self.data_c
        for i in range(n):
            val = ptr[i]
            if val not in seen:
                seen.add(val)
                result.append(val)
        return result

    # ------------------------------------------------------------
    # 字段提取 (时分秒直接算术提取，避免冗余的 YMD 历法计算)
    # ------------------------------------------------------------
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def year(self):
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int64_t* ptr = <int64_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        cdef int32_t days
        with nogil:
            for i in range(n):
                days = <int32_t>(ptr[i] / US_PER_DAY)
                _days_to_ymd(days, &y, &m, &d)
                res_ptr[i] = y
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def month(self):
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int64_t* ptr = <int64_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        cdef int32_t days
        with nogil:
            for i in range(n):
                days = <int32_t>(ptr[i] / US_PER_DAY)
                _days_to_ymd(days, &y, &m, &d)
                res_ptr[i] = m
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def day(self):
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int64_t* ptr = <int64_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        cdef int32_t days
        with nogil:
            for i in range(n):
                days = <int32_t>(ptr[i] / US_PER_DAY)
                _days_to_ymd(days, &y, &m, &d)
                res_ptr[i] = d
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def hour(self):
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int64_t* ptr = <int64_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int64_t secs
        cdef int rem
        with nogil:
            for i in range(n):
                secs = ptr[i] / US_PER_SEC
                rem = <int>(secs % SEC_PER_DAY)
                if rem < 0: rem += SEC_PER_DAY
                res_ptr[i] = rem / 3600
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def minute(self):
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int64_t* ptr = <int64_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int64_t secs
        cdef int rem
        with nogil:
            for i in range(n):
                secs = ptr[i] / US_PER_SEC
                rem = <int>(secs % SEC_PER_DAY)
                if rem < 0: rem += SEC_PER_DAY
                res_ptr[i] = (rem / 60) % 60
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def second(self):
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int64_t* ptr = <int64_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int64_t secs
        cdef int rem
        with nogil:
            for i in range(n):
                secs = ptr[i] / US_PER_SEC
                rem = <int>(secs % SEC_PER_DAY)
                if rem < 0: rem += SEC_PER_DAY
                res_ptr[i] = rem % 60
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def where_mask(self, IntColumn mask, object fill):
        cdef Py_ssize_t n = self.length
        if mask.length != n:
            raise ValueError(f"Mask length {mask.length} != column length {n}")

        cdef DateTimeColumn col = DateTimeColumn(max(n, 1), cols=self.cols)
        col.length = n
        if n == 0: return col

        cdef int64_t fill_val = self._resolve_dt(fill) if fill is not None else 0
        cdef int64_t* col_ptr = <int64_t*>col.data_c
        cdef int64_t* self_ptr = <int64_t*>self.data_c
        cdef int* mask_ptr = mask.data_c
        cdef Py_ssize_t i

        with nogil:
            for i in range(n):
                col_ptr[i] = self_ptr[i] if mask_ptr[i] != 0 else fill_val
        return col

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def isin(self, set values):
        cdef Py_ssize_t n = self.length
        cdef BoolColumn res = BoolColumn(max(n, 1))
        res.num_cols = self.num_cols
        res.is_2d = self.is_2d
        res.length = n
        if n == 0: return res

        cdef Py_ssize_t m = len(values)
        cdef unsigned char* r_ptr = res.data_c
        if m == 0:
            memset(r_ptr, 0, n)
            return res

        cdef int64_t* v_arr = <int64_t*>aligned_malloc(m * sizeof(int64_t))
        if not v_arr: raise MemoryError()

        cdef Py_ssize_t j = 0
        for v in values:
            if v is not None:
                v_arr[j] = self._resolve_dt(v)
                j += 1

        cdef Py_ssize_t valid_m = j
        if valid_m > 0:
            qsort(v_arr, valid_m, sizeof(int64_t), _cmp_int64)

        cdef int64_t* s_ptr = <int64_t*>self.data_c
        cdef Py_ssize_t i
        cdef void* found

        with nogil:
            for i in range(n):
                found = bsearch(&s_ptr[i], v_arr, valid_m, sizeof(int64_t), _cmp_int64)
                r_ptr[i] = 1 if found != NULL else 0

        os_aligned_free(v_arr)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def drop_row(self, Py_ssize_t row_idx):
        if not self.owns_data:
            raise RuntimeError("Cannot drop row on a view/slice column. Copy it first.")
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()
        if row_idx < 0:
            row_idx += r
        if row_idx < 0 or row_idx >= r:
            raise IndexError("Index out of bounds")
        if self.is_2d:
            _col_shift_left_2d(self.data_c, &self.length, r, c, row_idx, sizeof(int64_t))
        else:
            _col_drop_row(self.data_c, &self.length, row_idx, sizeof(int64_t))

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def shift_left(self, Py_ssize_t pos):
        if not self.owns_data:
            raise RuntimeError("Cannot modify a view/slice column. Copy it first.")
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()
        cdef Py_ssize_t col_len
        if pos < 0:
            pos += r
        if pos < 0 or pos >= r:
            raise IndexError("Index out of bounds")
        if self.is_2d:
            _col_shift_left_2d(self.data_c, &self.length, r, c, pos, sizeof(int64_t))
        else:
            col_len = self.length
            _c_shift_left(self.data_c, pos, col_len, sizeof(int64_t))
            self.length = col_len - 1

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def shift_right(self, Py_ssize_t pos):
        if not self.owns_data:
            raise RuntimeError("Cannot modify a view/slice column. Copy it first.")
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()
        if pos < 0:
            pos += r
        if pos < 0 or pos > r:
            raise IndexError("Index out of bounds")
        if self.is_2d:
            if self.length + c > self.capacity:
                self._reserve(max(16, max(self.capacity * 2, self.length + c)))
            _col_shift_right_2d_int(self.data_c, &self.length, r, c, pos, sizeof(int64_t))
        else:
            if self.length >= self.capacity:
                self._reserve(max(16, self.capacity * 2))
            _c_shift_right(self.data_c, pos, self.length, sizeof(int64_t))
            self.length += 1
            (<int64_t*>self.data_c)[pos] = 0

    cdef DateTimeColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=False):
        cdef DateTimeColumn new_col
        cdef int64_t* new_data
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()

        if self.is_2d:
            if not inplace:
                new_col = DateTimeColumn(max(n * c, 1), cols=c)
                new_col.length = n * c
                _col_gather_2d_zero(new_col.data_c, self.data_c, indices, n, r, c, sizeof(int64_t))
                return new_col
            else:
                new_data = <int64_t*>aligned_malloc(max(n * c, 1) * sizeof(int64_t))
                if not new_data: raise MemoryError()
                _col_gather_2d_zero(new_data, self.data_c, indices, n, r, c, sizeof(int64_t))
                if self.owns_data and self.data_c != NULL:
                    os_aligned_free(self.data_c)
                self.data_c = new_data
                self.capacity = n * c
                self.length = n * c
                self.num_cols = <int>c
                self.is_2d = True
                self._base_obj = None
                return self

        if not inplace:
            new_col = DateTimeColumn(max(n, 1), cols=self.num_cols)
            new_col.length = n
            _col_gather_zero(new_col.data_c, self.data_c, indices, n, sizeof(int64_t))
            return new_col
        else:
            new_data = <int64_t*>aligned_malloc(max(n, 1) * sizeof(int64_t))
            if not new_data: raise MemoryError()
            _col_gather_zero(new_data, self.data_c, indices, n, sizeof(int64_t))
            if self.owns_data and self.data_c != NULL:
                os_aligned_free(self.data_c)
            self.data_c = new_data
            self.capacity = n
            self.length = n
            self._base_obj = None
            return self

    # ------------------------------------------------------------
    # 纯 C 64 位整数比较算子 (完全向量化)
    # ------------------------------------------------------------
    cdef IntColumn _cmp_op(self, object val, int op):
        cdef Py_ssize_t i, n = self.length
        cdef IntColumn res = IntColumn(max(n, 1), cols=self.num_cols)
        res.length = n

        cdef int64_t* s_ptr = <int64_t*>self.data_c
        cdef int64_t* o_ptr
        cdef int* r_ptr = res.data_c
        cdef char err_buf[128]
        cdef int64_t v
        cdef DateTimeColumn other

        if isinstance(val, DateTimeColumn):
            other = <DateTimeColumn>val
            if other.length != n:
                snprintf(err_buf, sizeof(err_buf), "Length mismatch: %zd vs %zd", n, other.length)
                raise ValueError(err_buf.decode('ascii'))
            o_ptr = <int64_t*>other.data_c
            with nogil:
                if op == 0:
                    for i in range(n): r_ptr[i] = 1 if s_ptr[i] < o_ptr[i] else 0
                elif op == 1:
                    for i in range(n): r_ptr[i] = 1 if s_ptr[i] <= o_ptr[i] else 0
                elif op == 2:
                    for i in range(n): r_ptr[i] = 1 if s_ptr[i] > o_ptr[i] else 0
                elif op == 3:
                    for i in range(n): r_ptr[i] = 1 if s_ptr[i] >= o_ptr[i] else 0
                elif op == 4:
                    for i in range(n): r_ptr[i] = 1 if s_ptr[i] == o_ptr[i] else 0
                elif op == 5:
                    for i in range(n): r_ptr[i] = 1 if s_ptr[i] != o_ptr[i] else 0
        else:
            v = self._resolve_dt(val)
            with nogil:
                if op == 0:
                    for i in range(n): r_ptr[i] = 1 if s_ptr[i] < v else 0
                elif op == 1:
                    for i in range(n): r_ptr[i] = 1 if s_ptr[i] <= v else 0
                elif op == 2:
                    for i in range(n): r_ptr[i] = 1 if s_ptr[i] > v else 0
                elif op == 3:
                    for i in range(n): r_ptr[i] = 1 if s_ptr[i] >= v else 0
                elif op == 4:
                    for i in range(n): r_ptr[i] = 1 if s_ptr[i] == v else 0
                elif op == 5:
                    for i in range(n): r_ptr[i] = 1 if s_ptr[i] != v else 0
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
        cdef int64_t* ptr = <int64_t*>self.data_c
        for i in range(n):
            items.append(_datetime_to_str(ptr[i]))
        return f"DateTimeColumn([{', '.join(items)}])"

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __bool__(self):
        return self.length > 0

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __len__(self):
        return self.length

    cdef inline void _max_2d_c(self, double* out_buf) noexcept nogil:
        _max_2d_c(<long long*>self.data_c, self._get_rows_c(), self.num_cols, out_buf)

    cdef inline void _min_2d_c(self, double* out_buf) noexcept nogil:
        _min_2d_c(<long long*>self.data_c, self._get_rows_c(), self.num_cols, out_buf)

    # ------------------------------------------------------------
    # astype: 纯 C 单指令转换与截断
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
        cdef int64_t* ptr = <int64_t*>self.data_c
        cdef char str_buf[32]

        cdef IntColumn ic
        cdef DoubleColumn dc
        cdef UCS2Column uc
        cdef AsciiColumn cc
        cdef CategoryColumn catc
        cdef DateColumn tc
        cdef TimeColumn tmc
        cdef int32_t* tc_ptr
        cdef int32_t* tmc_ptr
        cdef int64_t secs
        cdef int rem

        if target_type == TYPE_INT:
            ic = IntColumn(safe_cap, cols=self.cols)
            ic.length = n
            with nogil:
                for i in range(n):
                    ic.data_c[i] = <int>(ptr[i] / US_PER_SEC)
            return ic

        elif target_type == TYPE_DOUBLE:
            dc = DoubleColumn(safe_cap, cols=self.cols)
            dc.length = n
            with nogil:
                for i in range(n):
                    dc.data_c[i] = <double>ptr[i] / 1000000.0
            return dc

        # ------------------------------------------------------------
        # DateTime -> Date (TIMESTAMP)：极致单指令截断，无任何历法开销
        # ------------------------------------------------------------
        elif target_type == TYPE_DATE:
            tc = DateColumn(safe_cap, cols=self.cols)
            tc.length = n
            tc_ptr = <int32_t*>tc.data_c
            with nogil:
                for i in range(n):
                    tc_ptr[i] = <int32_t>(ptr[i] / US_PER_DAY)
            return tc

        # ------------------------------------------------------------
        # DateTime -> Time：秒数取模，单指令提取
        # ------------------------------------------------------------
        elif target_type == TYPE_TIME:
            tmc = TimeColumn(safe_cap, cols=self.cols)
            tmc.length = n
            tmc_ptr = tmc.data_c
            with nogil:
                for i in range(n):
                    secs = ptr[i] / US_PER_SEC
                    rem = <int>(secs % SEC_PER_DAY)
                    tmc_ptr[i] = rem + SEC_PER_DAY if rem < 0 else rem
            return tmc

        elif target_type == TYPE_CHAR256:
            cc = AsciiColumn(first_col_cap)
            cc.length = first_col_n
            with nogil:
                for i in range(first_col_n):
                    _format_datetime_c(ptr[i], cc.data_c + (i * 256))
            return cc

        elif target_type == TYPE_UCS2:
            uc = UCS2Column(first_col_cap)
            uc.length = first_col_n
            for i in range(first_col_n):
                _format_datetime_c(ptr[i], str_buf)
                uc._set_item(i, str_buf.decode('ascii'))
            return uc

        elif target_type == TYPE_CATEGORY:
            catc = CategoryColumn(first_col_cap)
            for i in range(first_col_n):
                _format_datetime_c(ptr[i], str_buf)
                catc.append(str_buf.decode('ascii'))
            return catc

        elif target_type == TYPE_DATETIME:
            return self.copy()

        cdef char err_buf[128]
        snprintf(err_buf, sizeof(err_buf), "Failed to cast DATETIME column to internal type id: %d", target_type)
        raise TypeError(err_buf.decode('ascii'))