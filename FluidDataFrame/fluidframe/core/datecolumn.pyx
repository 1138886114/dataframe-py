# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False

cimport cython
from libc.stdint cimport int32_t, uint32_t, int64_t
from libc.string cimport memcpy, memmove, memset
from libc.stdio cimport snprintf
from libc.stdlib cimport malloc, free

from fluidframe.core._base cimport (
    BaseColumn, aligned_malloc, aligned_realloc, os_aligned_free,
    TYPE_DATE, TYPE_INT, TYPE_DOUBLE, TYPE_CHAR256, TYPE_UCS2,
    TYPE_BOOL, TYPE_CATEGORY, TYPE_DATETIME, TYPE_TIME, _resolve_astype_target,
    _c_shift_left
)
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.doublecolumn cimport DoubleColumn
from fluidframe.core.char256column cimport Char256Column_SoA
from fluidframe.core.ucs2column cimport UCS2Column_SoA
from fluidframe.core.categorycolumn cimport CategoryColumn
from fluidframe.core.datetimecolumn cimport DateTimeColumn
from fluidframe.core.timecolumn cimport TimeColumn

# ============================================================
#  静态常量查找表（彻底消除分支预测失败）
# ============================================================

cdef struct NameEntry:
    const char* name
    size_t len

cdef NameEntry[7] _DAY_NAMES = [
    NameEntry(b"Monday", 6),
    NameEntry(b"Tuesday", 7),
    NameEntry(b"Wednesday", 9),
    NameEntry(b"Thursday", 8),
    NameEntry(b"Friday", 6),
    NameEntry(b"Saturday", 8),
    NameEntry(b"Sunday", 6)
]

cdef NameEntry[13] _MONTH_NAMES = [
    NameEntry(b"", 0),
    NameEntry(b"January", 7),
    NameEntry(b"February", 8),
    NameEntry(b"March", 5),
    NameEntry(b"April", 5),
    NameEntry(b"May", 3),
    NameEntry(b"June", 4),
    NameEntry(b"July", 4),
    NameEntry(b"August", 6),
    NameEntry(b"September", 9),
    NameEntry(b"October", 7),
    NameEntry(b"November", 8),
    NameEntry(b"December", 8)
]

cdef int[13] _MONTH_DAYS_NORMAL = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
cdef int[13] _DOY_CUMULATIVE_NORMAL = [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

# ============================================================
#  现代顶级历法算法 (Howard Hinnant Algorithm)
#  专为 1970-01-01 纪元设计，无除法常数表、无循环、零分支
# ============================================================

cdef inline bint _is_leap_year(int y) noexcept nogil:
    return (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)

cdef inline int _days_in_month(int y, int m) noexcept nogil:
    if m == 2:
        return 29 if _is_leap_year(y) else 28
    if m >= 1 and m <= 12:
        return _MONTH_DAYS_NORMAL[m]
    return 30

cdef inline int32_t _ymd_to_days(int y, int m, int d) noexcept nogil:
    """YYYY-MM-DD -> 1970-01-01 起的天数 (Days since Unix Epoch)"""
    cdef int a = (14 - m) // 12
    cdef int y2 = y + 4800 - a
    cdef int m2 = m + 12 * a - 3
    return <int32_t>(d + (153 * m2 + 2) // 5 + 365 * y2 + y2 // 4 - y2 // 100 + y2 // 400 - 32045 - 2440588)

cdef inline void _days_to_ymd(int32_t days, int* y, int* m, int* d) noexcept nogil:
    """1970-01-01 起的天数 -> YYYY-MM-DD"""
    cdef long long daynum = <long long>days + 2440588
    cdef long long a, b, c, d2, e, m2
    a = daynum + 32044
    b = (4 * a + 3) // 146097
    c = a - 146097 * b // 4
    d2 = (4 * c + 3) // 1461
    e = c - 1461 * d2 // 4
    m2 = (5 * e + 2) // 153
    d[0] = <int>(e - (153 * m2 + 2) // 5 + 1)
    m[0] = <int>(m2 + 3 - 12 * (m2 // 10))
    y[0] = <int>(100 * b + d2 - 4800 + m2 // 10)

cdef inline int32_t _parse_date_str(str s) noexcept:
    """极速解析 'YYYYMMDD' 或 'YYYY-MM-DD' 字符串为 Unix 纪元天数"""
    cdef bytes b = s.encode('ascii', errors='ignore')
    cdef const unsigned char* buf = <const unsigned char*>b
    cdef int digits[10]
    cdef int nd = 0, i = 0, n = len(b)
    while i < n and nd < 8:
        if buf[i] >= 48 and buf[i] <= 57:
            digits[nd] = buf[i] - 48
            nd += 1
        i += 1
    if nd < 8:
        return 0
    cdef int y = digits[0] * 1000 + digits[1] * 100 + digits[2] * 10 + digits[3]
    cdef int m = digits[4] * 10 + digits[5]
    cdef int d = digits[6] * 10 + digits[7]
    return _ymd_to_days(y, m, d)


# ============================================================
#  DateColumn 类实现
# ============================================================
@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef class DateColumn(BaseColumn):

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __cinit__(self, Py_ssize_t init_capacity=16):
        self.capacity = max(init_capacity, 1)
        self.length = 0
        self.dtype = TYPE_DATE
        self.data_c = <int32_t*>aligned_malloc(self.capacity * sizeof(int32_t))
        if not self.data_c:
            raise MemoryError()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __dealloc__(self):
        if self.data_c != NULL:
            os_aligned_free(self.data_c)
            self.data_c = NULL

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def get_size(self):
        return self.capacity * sizeof(int32_t)

    cdef void _reserve(self, Py_ssize_t new_cap):
        if new_cap <= self.capacity:
            return
        cdef int32_t* new_ptr = <int32_t*>aligned_realloc(
            self.data_c, self.capacity * sizeof(int32_t), new_cap * sizeof(int32_t))
        if not new_ptr:
            raise MemoryError()
        self.data_c = new_ptr
        self.capacity = new_cap

    cdef inline int32_t _resolve_date(self, date) except *:
        cdef int32_t v
        if isinstance(date, str):
            return _parse_date_str(<str>date)
        v = <int32_t>date
        # YYYYMMDD integers (> 10000000) need conversion to epoch days
        if v > 10000000:
            return _ymd_to_days(v // 10000, (v // 100) % 100, v % 100)
        return v

    cdef inline int32_t _to_ymd_int(self, int32_t epoch_days) noexcept nogil:
        """Convert epoch days back to YYYYMMDD integer for backward compatibility."""
        cdef int y, m, d
        _days_to_ymd(epoch_days, &y, &m, &d)
        return <int32_t>(y * 10000 + m * 100 + d)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def append(self, val):
        cdef int32_t v
        if isinstance(val, str):
            v = _parse_date_str(<str>val)
        else:
            v = <int32_t>val
            if v > 10000000:
                v = _ymd_to_days(v // 10000, (v // 100) % 100, v % 100)

        if self.length >= self.capacity:
            self._reserve(max(16, self.capacity * 2))
        (<int32_t*>self.data_c)[self.length] = v
        self.length += 1

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def copy(self):
        cdef DateColumn new_col = DateColumn(max(self.length, 1))
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
        return self._to_ymd_int((<int32_t*>self.data_c)[self.length])

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_front(self):
        if self.length == 0:
            raise IndexError("pop from empty column")
        cdef int32_t val = (<int32_t*>self.data_c)[0]
        if self.length > 1:
            memmove(self.data_c, <int32_t*>self.data_c + 1, (self.length - 1) * sizeof(int32_t))
        self.length -= 1
        return self._to_ymd_int(val)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def concat(self, DateColumn other):
        cdef Py_ssize_t other_len = other.length
        if other_len == 0: return
        cdef Py_ssize_t new_len = self.length + other_len
        self._reserve(new_len)
        memcpy(<int32_t*>self.data_c + self.length, other.data_c, other_len * sizeof(int32_t))
        self.length = new_len

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def max(self):
        if self.length == 0: raise ValueError("empty column")
        cdef int32_t max_val = (<int32_t*>self.data_c)[0]
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(1, n):
                if ptr[i] > max_val:
                    max_val = ptr[i]
        return self._to_ymd_int(max_val)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def min(self):
        if self.length == 0: raise ValueError("empty column")
        cdef int32_t min_val = (<int32_t*>self.data_c)[0]
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(1, n):
                if ptr[i] < min_val:
                    min_val = ptr[i]
        return self._to_ymd_int(min_val)

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
        cdef int32_t* self_data = <int32_t*>self.data_c
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

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __getitem__(self, Py_ssize_t idx):
        if idx < 0: idx += self.length
        if idx < 0 or idx >= self.length: raise IndexError("Index out of bounds")
        return self._to_ymd_int((<int32_t*>self.data_c)[idx])

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def unique(self):
        cdef set seen = set()
        cdef list result = []
        cdef Py_ssize_t i, n = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        for i in range(n):
            val = self._to_ymd_int(ptr[i])
            if val not in seen:
                seen.add(val)
                result.append(val)
        return result

    # ------------------------------------------------------------
    # 年/月/日/季度 提取
    # ------------------------------------------------------------
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def year(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        with nogil:
            for i in range(n):
                _days_to_ymd(ptr[i], &y, &m, &d)
                res_ptr[i] = y
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def month(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        with nogil:
            for i in range(n):
                _days_to_ymd(ptr[i], &y, &m, &d)
                res_ptr[i] = m
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def day(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        with nogil:
            for i in range(n):
                _days_to_ymd(ptr[i], &y, &m, &d)
                res_ptr[i] = d
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def quarter(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        with nogil:
            for i in range(n):
                _days_to_ymd(ptr[i], &y, &m, &d)
                res_ptr[i] = (m - 1) / 3 + 1
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def strftime(self, str fmt):
        cdef Py_ssize_t i, n = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int y, m, d
        cdef list parts = []

        for i in range(n):
            _days_to_ymd(ptr[i], &y, &m, &d)
            parts.append(fmt.replace('%Y', str(y)).replace('%m', f'{m:02d}').replace('%d', f'{d:02d}'))

        cdef Char256Column_SoA res = Char256Column_SoA(max(n, 1))
        for s in parts:
            res.append(s)
        return res

    # ------------------------------------------------------------
    # 过滤算子 (纯 32 位整数比较，100% AVX2/AVX-512 SIMD 向量化)
    # ------------------------------------------------------------
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def filter_greater(self, date):
        cdef int32_t target = self._resolve_date(date)
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = 1 if ptr[i] > target else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def filter_greater_equal(self, date):
        cdef int32_t target = self._resolve_date(date)
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = 1 if ptr[i] >= target else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def filter_less(self, date):
        cdef int32_t target = self._resolve_date(date)
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = 1 if ptr[i] < target else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def filter_less_equal(self, date):
        cdef int32_t target = self._resolve_date(date)
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = 1 if ptr[i] <= target else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def filter_equal(self, date):
        cdef int32_t target = self._resolve_date(date)
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = 1 if ptr[i] == target else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def filter_not_equal(self, date):
        cdef int32_t target = self._resolve_date(date)
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = 1 if ptr[i] != target else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def filter_between(self, start, end):
        cdef int32_t lo = self._resolve_date(start)
        cdef int32_t hi = self._resolve_date(end)
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = 1 if (ptr[i] >= lo and ptr[i] <= hi) else 0
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def where_mask(self, IntColumn mask, object fill):
        cdef Py_ssize_t n = self.length
        if mask.length != n:
            raise ValueError(f"Mask length {mask.length} != column length {n}")
        cdef DateColumn col = DateColumn(max(n, 1))
        col.length = n
        if n == 0: return col

        cdef int32_t fill_val = self._resolve_date(fill)
        cdef int32_t* col_ptr = <int32_t*>col.data_c
        cdef int32_t* self_ptr = <int32_t*>self.data_c
        cdef int* mask_ptr = mask.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                col_ptr[i] = self_ptr[i] if mask_ptr[i] != 0 else fill_val
        return col

    # ------------------------------------------------------------
    # 加减天数：退化为极致的单指令加法（彻底消灭历法转换）
    # ------------------------------------------------------------
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def add_days(self, int n):
        cdef Py_ssize_t length = self.length
        cdef DateColumn res = DateColumn(max(length, 1))
        res.length = length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int32_t* res_ptr = <int32_t*>res.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(length):
                res_ptr[i] = ptr[i] + n
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sub_days(self, int n):
        return self.add_days(-n)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def add_months(self, int n):
        cdef Py_ssize_t length = self.length
        cdef DateColumn res = DateColumn(max(length, 1))
        res.length = length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int32_t* res_ptr = <int32_t*>res.data_c
        cdef Py_ssize_t i
        cdef int y, m, d, total, new_m, new_y, max_d
        with nogil:
            for i in range(length):
                _days_to_ymd(ptr[i], &y, &m, &d)
                total = (m - 1) + n
                new_y = y + total / 12
                new_m = total % 12 + 1
                if new_m <= 0:
                    new_m += 12
                    new_y -= 1
                max_d = _days_in_month(new_y, new_m)
                if d > max_d:
                    d = max_d
                res_ptr[i] = _ymd_to_days(new_y, new_m, d)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sub_months(self, int n):
        return self.add_months(-n)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def add_years(self, int n):
        cdef Py_ssize_t length = self.length
        cdef DateColumn res = DateColumn(max(length, 1))
        res.length = length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int32_t* res_ptr = <int32_t*>res.data_c
        cdef Py_ssize_t i
        cdef int y, m, d, new_y, max_d
        with nogil:
            for i in range(length):
                _days_to_ymd(ptr[i], &y, &m, &d)
                new_y = y + n
                max_d = _days_in_month(new_y, m)
                if d > max_d:
                    d = max_d
                res_ptr[i] = _ymd_to_days(new_y, m, d)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sub_years(self, int n):
        return self.add_years(-n)

    # ------------------------------------------------------------
    # 星期与历法判定 (1970-01-01 是 Thursday = 3)
    # ------------------------------------------------------------
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def day_of_week(self):
        """返回 0=Monday .. 6=Sunday"""
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = <int>(((ptr[i] + 3) % 7 + 7) % 7)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def is_weekend(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int dow
        with nogil:
            for i in range(n):
                dow = <int>(((ptr[i] + 3) % 7 + 7) % 7)
                res_ptr[i] = 1 if (dow == 5 or dow == 6) else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def day_of_year(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d, doy
        with nogil:
            for i in range(n):
                _days_to_ymd(ptr[i], &y, &m, &d)
                doy = _DOY_CUMULATIVE_NORMAL[m] + d
                if m > 2 and _is_leap_year(y):
                    doy += 1
                res_ptr[i] = doy
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def days_in_month(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        with nogil:
            for i in range(n):
                _days_to_ymd(ptr[i], &y, &m, &d)
                res_ptr[i] = _days_in_month(y, m)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def is_leap_year(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        with nogil:
            for i in range(n):
                _days_to_ymd(ptr[i], &y, &m, &d)
                res_ptr[i] = 1 if _is_leap_year(y) else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def is_month_start(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        with nogil:
            for i in range(n):
                _days_to_ymd(ptr[i], &y, &m, &d)
                res_ptr[i] = 1 if d == 1 else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def is_month_end(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        with nogil:
            for i in range(n):
                _days_to_ymd(ptr[i], &y, &m, &d)
                res_ptr[i] = 1 if d == _days_in_month(y, m) else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def is_year_start(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        with nogil:
            for i in range(n):
                _days_to_ymd(ptr[i], &y, &m, &d)
                res_ptr[i] = 1 if (m == 1 and d == 1) else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def is_year_end(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        with nogil:
            for i in range(n):
                _days_to_ymd(ptr[i], &y, &m, &d)
                res_ptr[i] = 1 if (m == 12 and d == 31) else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def is_quarter_start(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        with nogil:
            for i in range(n):
                _days_to_ymd(ptr[i], &y, &m, &d)
                res_ptr[i] = 1 if ((m == 1 or m == 4 or m == 7 or m == 10) and d == 1) else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def is_quarter_end(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        with nogil:
            for i in range(n):
                _days_to_ymd(ptr[i], &y, &m, &d)
                if m == 3 or m == 6 or m == 9 or m == 12:
                    res_ptr[i] = 1 if d == _days_in_month(y, m) else 0
                else:
                    res_ptr[i] = 0
        return res

    cdef inline int _iso_week_of_year(self, int32_t days, int* iso_y) noexcept nogil:
        cdef int y, m, d
        _days_to_ymd(days, &y, &m, &d)
        cdef int32_t d4jan = _ymd_to_days(y, 1, 4)
        cdef int dow4jan = <int>(((d4jan + 3) % 7 + 7) % 7)
        cdef int32_t week1_start = d4jan - dow4jan

        if days < week1_start:
            y -= 1
            d4jan = _ymd_to_days(y, 1, 4)
            dow4jan = <int>(((d4jan + 3) % 7 + 7) % 7)
            week1_start = d4jan - dow4jan
        else:
            d4jan = _ymd_to_days(y + 1, 1, 4)
            dow4jan = <int>(((d4jan + 3) % 7 + 7) % 7)
            if days >= d4jan - dow4jan:
                y += 1
                week1_start = d4jan - dow4jan

        if iso_y != NULL:
            iso_y[0] = y
        return <int>((days - week1_start) / 7) + 1

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def week_of_year(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = self._iso_week_of_year(ptr[i], NULL)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def day_name(self):
        cdef Char256Column_SoA res = Char256Column_SoA(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int dow

        DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for i in range(n):
            dow = <int>(((ptr[i] + 3) % 7 + 7) % 7)
            res.append(DAY_NAMES[dow])
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def month_name(self):
        cdef Char256Column_SoA res = Char256Column_SoA(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int y, m, d

        MONTH_NAMES = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
        for i in range(n):
            _days_to_ymd(ptr[i], &y, &m, &d)
            if 1 <= m <= 12:
                res.append(MONTH_NAMES[m])
            else:
                res.append('')
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def last_day_of_month(self):
        cdef DateColumn res = DateColumn(max(self.length, 1))
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int32_t* res_ptr = <int32_t*>res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d, max_d
        with nogil:
            for i in range(n):
                _days_to_ymd(ptr[i], &y, &m, &d)
                max_d = _days_in_month(y, m)
                res_ptr[i] = _ymd_to_days(y, m, max_d)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def iso_calendar(self):
        cdef IntColumn res_y = IntColumn(max(self.length, 1))
        cdef IntColumn res_w = IntColumn(max(self.length, 1))
        cdef IntColumn res_d = IntColumn(max(self.length, 1))
        res_y.length = self.length
        res_w.length = self.length
        res_d.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* ry_ptr = res_y.data_c
        cdef int* rw_ptr = res_w.data_c
        cdef int* rd_ptr = res_d.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int iso_y
        with nogil:
            for i in range(n):
                rw_ptr[i] = self._iso_week_of_year(ptr[i], &iso_y)
                ry_ptr[i] = iso_y
                rd_ptr[i] = <int>(((ptr[i] + 3) % 7 + 7) % 7) + 1
        return (res_y, res_w, res_d)

    # ------------------------------------------------------------
    # 日期差值计算 (diff_days 直接减法，SIMD 向量化)
    # ------------------------------------------------------------
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def diff_days(self, date):
        cdef int32_t target = self._resolve_date(date)
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = <int>(ptr[i] - target)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def diff_months(self, date):
        cdef int32_t target = self._resolve_date(date)
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y1, m1, d1, y2, m2, d2
        _days_to_ymd(target, &y2, &m2, &d2)
        with nogil:
            for i in range(n):
                _days_to_ymd(ptr[i], &y1, &m1, &d1)
                res_ptr[i] = (y1 - y2) * 12 + (m1 - m2) - (1 if d1 < d2 else 0)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def diff_years(self, date):
        cdef int32_t target = self._resolve_date(date)
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y1, m1, d1, y2, m2, d2
        _days_to_ymd(target, &y2, &m2, &d2)
        with nogil:
            for i in range(n):
                _days_to_ymd(ptr[i], &y1, &m1, &d1)
                res_ptr[i] = y1 - y2 - (1 if (m1 < m2 or (m1 == m2 and d1 < d2)) else 0)
        return res

    # ------------------------------------------------------------
    # 纯 C 32 位整数比较算子 (完全向量化)
    # ------------------------------------------------------------
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def _cmp_op(self, val, int op):
        cdef Py_ssize_t i, n = self.length
        cdef IntColumn res = IntColumn(max(n, 1))
        res.length = n

        cdef int32_t* s_ptr = <int32_t*>self.data_c
        cdef int32_t* o_ptr
        cdef int* r_ptr = res.data_c
        cdef char err_buf[128]
        cdef int32_t v
        cdef DateColumn other

        if isinstance(val, DateColumn):
            other = <DateColumn>val
            if other.length != n:
                snprintf(err_buf, sizeof(err_buf), "Length mismatch: %zd vs %zd", n, other.length)
                raise ValueError(err_buf.decode('ascii'))
            o_ptr = <int32_t*>other.data_c
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
            v = self._resolve_date(val)
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
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int y, m, d
        cdef char str_buf[16]
        for i in range(n):
            _days_to_ymd(ptr[i], &y, &m, &d)
            snprintf(str_buf, sizeof(str_buf), "%04d-%02d-%02d", y, m, d)
            items.append(str_buf.decode('ascii'))
        return f"DateColumn([{', '.join(items)}])"

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __bool__(self):
        return self.length > 0

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __len__(self):
        return self.length

    cdef DateColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=False):
        cdef Py_ssize_t i
        cdef int32_t* old_data = <int32_t*>self.data_c
        cdef int32_t* new_data
        cdef DateColumn new_col

        if not inplace:
            new_col = DateColumn(max(n, 1))
            new_col.length = n
            with nogil:
                for i in range(n):
                    (<int32_t*>new_col.data_c)[i] = old_data[indices[i]]
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

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def drop_row(self, Py_ssize_t row_idx):
        if row_idx < 0:
            row_idx += self.length
        cdef Py_ssize_t rem_len = self.length - row_idx - 1
        if rem_len > 0:
            memmove(<int32_t*>self.data_c + row_idx,
                    <int32_t*>self.data_c + row_idx + 1,
                    rem_len * sizeof(int32_t))
        self.length -= 1

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def shift_left(self, Py_ssize_t pos):
        cdef Py_ssize_t col_len = self.length
        _c_shift_left(self.data_c, pos, col_len, sizeof(int32_t))
        self.length = col_len - 1

    # ------------------------------------------------------------
    # 类型转换 astype (无 Python 堆分配直写)
    # ------------------------------------------------------------
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def astype(self, object target):
        cdef int target_type = _resolve_astype_target(target)
        cdef Py_ssize_t i, n = self.length
        cdef Py_ssize_t safe_cap = max(n, 1)
        cdef int y, m, d
        cdef char str_buf[16]

        cdef IntColumn ic
        cdef DoubleColumn dc
        cdef UCS2Column_SoA uc
        cdef Char256Column_SoA cc
        cdef CategoryColumn catc
        cdef DateTimeColumn dtc
        cdef TimeColumn tmc
        cdef int32_t* s_ptr = <int32_t*>self.data_c

        if target_type == TYPE_INT:
            ic = IntColumn(safe_cap)
            ic.length = n
            with nogil:
                for i in range(n):
                    _days_to_ymd(s_ptr[i], &y, &m, &d)
                    ic.data_c[i] = y * 10000 + m * 100 + d
            return ic
        elif target_type == TYPE_DOUBLE:
            dc = DoubleColumn(safe_cap)
            dc.length = n
            with nogil:
                for i in range(n):
                    _days_to_ymd(s_ptr[i], &y, &m, &d)
                    dc.data_c[i] = <double>(y * 10000 + m * 100 + d)
            return dc
        elif target_type == TYPE_CHAR256:
            cc = Char256Column_SoA(safe_cap)
            cc.length = n
            with nogil:
                for i in range(n):
                    _days_to_ymd(s_ptr[i], &y, &m, &d)
                    cc.data_c[i * 256 + 0] = <char>(48 + ((y / 1000) % 10))
                    cc.data_c[i * 256 + 1] = <char>(48 + ((y / 100) % 10))
                    cc.data_c[i * 256 + 2] = <char>(48 + ((y / 10) % 10))
                    cc.data_c[i * 256 + 3] = <char>(48 + (y % 10))
                    cc.data_c[i * 256 + 4] = b'-'
                    cc.data_c[i * 256 + 5] = <char>(48 + (m / 10))
                    cc.data_c[i * 256 + 6] = <char>(48 + (m % 10))
                    cc.data_c[i * 256 + 7] = b'-'
                    cc.data_c[i * 256 + 8] = <char>(48 + (d / 10))
                    cc.data_c[i * 256 + 9] = <char>(48 + (d % 10))
                    cc.data_c[i * 256 + 10] = 0
            return cc
        elif target_type == TYPE_UCS2:
            uc = UCS2Column_SoA(safe_cap)
            uc.length = n
            for i in range(n):
                _days_to_ymd(s_ptr[i], &y, &m, &d)
                snprintf(str_buf, sizeof(str_buf), "%04d-%02d-%02d", y, m, d)
                uc._set_item(i, str_buf.decode('ascii'))
            return uc
        elif target_type == TYPE_CATEGORY:
            catc = CategoryColumn(safe_cap)
            for i in range(n):
                _days_to_ymd(s_ptr[i], &y, &m, &d)
                snprintf(str_buf, sizeof(str_buf), "%04d-%02d-%02d", y, m, d)
                catc.append(str_buf.decode('ascii'))
            return catc
        elif target_type == TYPE_DATETIME:
            dtc = DateTimeColumn(safe_cap)
            dtc.length = n
            with nogil:
                for i in range(n):
                    (<int64_t*>dtc.data_c)[i] = <int64_t>s_ptr[i] * 86400000000LL
            return dtc
        elif target_type == TYPE_TIME:
            tmc = TimeColumn(safe_cap)
            tmc.length = n
            with nogil:
                for i in range(n):
                    tmc.data_c[i] = 0
            return tmc
        elif target_type == TYPE_DATE:
            return self.copy()

        raise TypeError(f"Failed to cast TIMESTAMP column to internal type id: {target_type}")