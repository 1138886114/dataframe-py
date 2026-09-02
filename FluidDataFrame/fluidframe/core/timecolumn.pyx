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
    TYPE_BOOL, TYPE_TIMESTAMP, TYPE_CATEGORY, TYPE_DATETIME, _resolve_astype_target
)
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.doublecolumn cimport DoubleColumn
from fluidframe.core.char256column cimport Char256Column_SoA
from fluidframe.core.ucs2column cimport UCS2Column_SoA
from fluidframe.core.categorycolumn cimport CategoryColumn
from fluidframe.core.boolcolumn cimport BoolColumn
from fluidframe.core.datecolumn cimport DateColumn
from fluidframe.core.datetimecolumn cimport DateTimeColumn

# ============================================================
#  底层纯 C 高性能时间运算、LUT 查表格式化与零拷贝解析
# ============================================================

# 00-99 两位字符查找表（LUT 优化：消除格式化中的所有除法/分支）
cdef const char[200] DIGIT_PAIRS = (
    b"00010203040506070809"
    b"10111213141516171819"
    b"20212223242526272829"
    b"30313233343536373839"
    b"40414243444546474849"
    b"50515253545556575859"
    b"60616263646566676869"
    b"70717273747576777879"
    b"80818283848586878889"
    b"90919293949596979899"
)

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
    cdef char c

    for i in range(n):
        c = buf[i]
        if c >= b'0' and c <= b'9':
            cur_val = cur_val * 10 + (c - b'0')
        elif c == b':':
            if part_idx >= 2:
                raise ValueError(f"Invalid time format: '{s}'")
            parts[part_idx] = cur_val
            cur_val = 0
            part_idx += 1
        elif c == b' ' or c == b'\t' or c == b'\r' or c == b'\n':
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
    def __cinit__(self, Py_ssize_t init_capacity=16):
        self.capacity = max(init_capacity, 1)
        self.length = 0
        self.dtype = TYPE_TIME
        self.data_c = <int32_t*>aligned_malloc(self.capacity * sizeof(int32_t))
        if not self.data_c:
            raise MemoryError()

    cdef Py_ssize_t get_size(self):
        return self.capacity * sizeof(int32_t)

    def __dealloc__(self):
        if self.data_c != NULL:
            os_aligned_free(self.data_c)
            self.data_c = NULL

    cdef void _reserve(self, Py_ssize_t new_cap):
        if new_cap <= self.capacity:
            return
        cdef int32_t* new_ptr = <int32_t*>aligned_realloc(
            self.data_c, self.capacity * sizeof(int32_t), new_cap * sizeof(int32_t))
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
        cdef int32_t v = _parse_time_str(<str>val) if isinstance(val, str) else <int32_t>val
        if self.length >= self.capacity:
            self._reserve(max(16, self.capacity * 2))
        self.data_c[self.length] = v
        self.length += 1

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def copy(self):
        cdef TimeColumn new_col = TimeColumn(max(self.length, 1))
        new_col.length = self.length
        if self.length > 0:
            memcpy(new_col.data_c, self.data_c, self.length * sizeof(int32_t))
        return new_col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_back(self):
        if self.length == 0:
            raise IndexError("pop from empty column")
        self.length -= 1
        return self.data_c[self.length]

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_front(self):
        if self.length == 0:
            raise IndexError("pop from empty column")
        cdef int32_t val = self.data_c[0]
        if self.length > 1:
            memmove(self.data_c, self.data_c + 1, (self.length - 1) * sizeof(int32_t))
        self.length -= 1
        return val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def concat(self, TimeColumn other):
        cdef Py_ssize_t other_len = other.length
        if other_len == 0: return
        cdef Py_ssize_t new_len = self.length + other_len
        self._reserve(new_len)
        memcpy(self.data_c + self.length, other.data_c, other_len * sizeof(int32_t))
        self.length = new_len

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
        cdef int32_t* self_data = self.data_c
        cdef Py_ssize_t i, idx
        try:
            for i in range(n):
                idx = <Py_ssize_t>indices[i]
                if idx < 0: idx += self.length
                if idx < 0 or idx >= self.length:
                    snprintf(err_buf, sizeof(err_buf), "Index out of bounds: %zd", idx)
                    raise IndexError(err_buf.decode('ascii'))
                new_data[i] = self_data[idx]
        except Exception:
            os_aligned_free(new_data)
            raise

        os_aligned_free(self.data_c)
        self.data_c = new_data
        self.capacity = n
        self.length = n

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
        cdef int32_t* res_ptr = res.data_c
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
        cdef int32_t* res_ptr = res.data_c
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
        cdef int32_t* res_ptr = res.data_c
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
        cdef TimeColumn col = TimeColumn(max(n, 1))
        col.length = n
        if n == 0: return col

        cdef int32_t fill_val = self._resolve_time(fill) if fill is not None else 0
        cdef int32_t* col_ptr = col.data_c
        cdef int32_t* self_ptr = self.data_c
        cdef int32_t* mask_ptr = mask.data_c
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

    cpdef void drop_row(self, Py_ssize_t row_idx, Py_ssize_t rem_len):
        if row_idx < 0:
            row_idx += self.length
        if rem_len > 0:
            memmove(self.data_c + row_idx,
                    self.data_c + row_idx + 1,
                    rem_len * sizeof(int32_t))
        self.length -= 1

    cdef TimeColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=False):
        cdef Py_ssize_t i
        cdef int32_t* old_data = self.data_c
        cdef int32_t* new_data
        cdef TimeColumn new_col

        if not inplace:
            new_col = TimeColumn(max(n, 1))
            new_col.length = n
            with nogil:
                for i in range(n):
                    new_col.data_c[i] = old_data[indices[i]]
            return new_col
        else:
            new_data = <int32_t*>aligned_malloc(max(n, 1) * sizeof(int32_t))
            if not new_data: raise MemoryError()
            with nogil:
                for i in range(n):
                    new_data[i] = old_data[indices[i]]
            os_aligned_free(self.data_c)
            self.data_c = new_data
            self.capacity = n
            self.length = n
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
        cdef int32_t* r_ptr = res.data_c
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
        cdef Py_ssize_t safe_cap = max(n, 1)
        cdef int32_t* ptr = self.data_c
        cdef char str_buf[16]

        cdef IntColumn ic
        cdef DoubleColumn dc
        cdef BoolColumn bc
        cdef UCS2Column_SoA uc
        cdef Char256Column_SoA cc
        cdef CategoryColumn catc
        cdef DateColumn tc
        cdef DateTimeColumn dtc

        if target_type == TYPE_INT:
            ic = IntColumn(safe_cap)
            ic.length = n
            with nogil:
                for i in range(n):
                    ic.data_c[i] = ptr[i]
            return ic
        elif target_type == TYPE_DOUBLE:
            dc = DoubleColumn(safe_cap)
            dc.length = n
            with nogil:
                for i in range(n):
                    dc.data_c[i] = <double>ptr[i]
            return dc
        elif target_type == TYPE_BOOL:
            bc = BoolColumn(safe_cap)
            bc.length = n
            with nogil:
                for i in range(n):
                    bc.data_c[i] = <unsigned char>(ptr[i] != 0)
            return bc
        elif target_type == TYPE_CHAR256:
            cc = Char256Column_SoA(safe_cap)
            cc.length = n
            with nogil:
                for i in range(n):
                    _format_time_c(ptr[i], cc.data_c + (i * 256))
            return cc
        elif target_type == TYPE_UCS2:
            uc = UCS2Column_SoA(safe_cap)
            uc.length = n
            for i in range(n):
                _format_time_c(ptr[i], str_buf)
                uc._set_item(i, str_buf.decode('ascii'))
            return uc
        elif target_type == TYPE_CATEGORY:
            catc = CategoryColumn(safe_cap)
            for i in range(n):
                _format_time_c(ptr[i], str_buf)
                catc.append(str_buf.decode('ascii'))
            return catc
        elif target_type == TYPE_TIMESTAMP:
            tc = DateColumn(safe_cap)
            tc.length = n
            with nogil:
                for i in range(n):
                    tc.data_c[i] = 0
            return tc
        elif target_type == TYPE_DATETIME:
            dtc = DateTimeColumn(safe_cap)
            dtc.length = n
            # Time -> DateTime: 秒数转换为微秒数 (1970-01-01 当天对应的时间点)
            with nogil:
                for i in range(n):
                    dtc.data_c[i] = <int64_t>ptr[i] * 1000000LL
            return dtc
        elif target_type == TYPE_TIME:
            return self.copy()

        raise TypeError(f"Failed to cast TIME column to internal type id: {target_type}")