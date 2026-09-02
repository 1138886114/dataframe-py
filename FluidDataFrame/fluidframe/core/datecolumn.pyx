# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False

cimport cython
from libc.string cimport memcpy, memmove, memset
from libc.stdio cimport snprintf
from libc.stdlib cimport malloc, free

from fluidframe.core._base cimport (
    BaseColumn, aligned_malloc, aligned_realloc, os_aligned_free,
    TYPE_TIMESTAMP, TYPE_INT, TYPE_DOUBLE, TYPE_CHAR256, TYPE_UCS2,
    TYPE_BOOL, TYPE_CATEGORY, TYPE_DATETIME, TYPE_TIME, _resolve_astype_target
)
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.doublecolumn cimport DoubleColumn
from fluidframe.core.char256column cimport Char256Column_SoA
from fluidframe.core.ucs2column cimport UCS2Column_SoA
from fluidframe.core.categorycolumn cimport CategoryColumn
from fluidframe.core.datetimecolumn cimport DateTimeColumn
from fluidframe.core.timecolumn cimport TimeColumn

# ============================================================
#  静态查找表与常量结构（彻底消除 if-elif 分支与运行时计算）
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

# ------------------------------------------------------------
# 核心算术优化：单次连除提取 Y/M/D，消除多余的 /100 和 %100 冗余运算
# ------------------------------------------------------------
cdef inline void _split_ymd(long long val, int* y, int* m, int* d) noexcept nogil:
    cdef long long ym = val / 100
    d[0] = <int>(val - ym * 100)
    y[0] = <int>(ym / 100)
    m[0] = <int>(ym - (<long long>y[0]) * 100)

cdef inline bint _is_leap_year(int y) noexcept nogil:
    return (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)

cdef inline int _days_in_month(int y, int m) noexcept nogil:
    if m == 2:
        return 29 if _is_leap_year(y) else 28
    if m >= 1 and m <= 12:
        return _MONTH_DAYS_NORMAL[m]
    return 30

cdef inline long long _date_to_daynum(int y, int m, int d) noexcept nogil:
    """将 YYYY-MM-DD 转换为连续日编号（基于标准天文学算法）"""
    cdef int a, y2, m2
    a = (14 - m) / 12
    y2 = y + 4800 - a
    m2 = m + 12 * a - 3
    return d + (153 * m2 + 2) / 5 + 365 * y2 + y2 / 4 - y2 / 100 + y2 / 400 - 32045

cdef inline void _daynum_to_date(long long daynum, int* y, int* m, int* d) noexcept nogil:
    """将连续日编号逆运算回 YYYY-MM-DD"""
    cdef long long a, b, c, d2, e, m2
    a = daynum + 32044
    b = (4 * a + 3) / 146097
    c = a - 146097 * b / 4
    d2 = (4 * c + 3) / 1461
    e = c - 1461 * d2 / 4
    m2 = (5 * e + 2) / 153
    d[0] = <int>(e - (153 * m2 + 2) / 5 + 1)
    m[0] = <int>(m2 + 3 - 12 * (m2 / 10))
    y[0] = <int>(100 * b + d2 - 4800 + m2 / 10)

cdef inline long long _parse_date_str(str s) noexcept:
    """极速解析 'YYYYMMDD' 或 'YYYY-MM-DD' 为整数 20260902"""
    cdef bytes b = s.encode('ascii', errors='ignore')
    cdef const unsigned char* buf = <const unsigned char*>b
    cdef long long val = 0
    cdef int i = 0, n = len(b)
    while i < n:
        if buf[i] >= 48 and buf[i] <= 57:
            val = val * 10 + (buf[i] - 48)
        i += 1
    return val


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
        self.dtype = TYPE_TIMESTAMP
        self.data_c = <long long*>aligned_malloc(self.capacity * sizeof(long long))
        if not self.data_c:
            raise MemoryError()

    def __dealloc__(self):
        if self.data_c != NULL:
            os_aligned_free(self.data_c)
            self.data_c = NULL

    cdef void _reserve(self, Py_ssize_t new_cap):
        if new_cap <= self.capacity:
            return
        cdef long long* new_ptr = <long long*>aligned_realloc(
            self.data_c, self.capacity * sizeof(long long), new_cap * sizeof(long long))
        if not new_ptr:
            raise MemoryError()
        self.data_c = new_ptr
        self.capacity = new_cap

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def append(self, val):
        cdef long long v = _parse_date_str(<str>val) if isinstance(val, str) else <long long>val
        if self.length >= self.capacity:
            self._reserve(max(16, self.capacity * 2))
        self.data_c[self.length] = v
        self.length += 1

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def copy(self):
        cdef DateColumn new_col = DateColumn(max(self.length, 1))
        new_col.length = self.length
        if self.length > 0:
            memcpy(new_col.data_c, self.data_c, self.length * sizeof(long long))
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
        cdef long long val = self.data_c[0]
        if self.length > 1:
            memmove(self.data_c, self.data_c + 1, (self.length - 1) * sizeof(long long))
        self.length -= 1
        return val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def concat(self, DateColumn other):
        cdef Py_ssize_t other_len = other.length
        if other_len == 0: return
        cdef Py_ssize_t new_len = self.length + other_len
        self._reserve(new_len)
        memcpy(self.data_c + self.length, other.data_c, other_len * sizeof(long long))
        self.length = new_len

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def max(self):
        if self.length == 0: raise ValueError("empty column")
        cdef long long max_val = self.data_c[0]
        cdef long long* ptr = self.data_c
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
        cdef long long min_val = self.data_c[0]
        cdef long long* ptr = self.data_c
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

        cdef long long* new_data = <long long*>aligned_malloc(n * sizeof(long long))
        if not new_data: raise MemoryError()
        cdef long long* self_data = self.data_c
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
        return self.data_c[idx]

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def unique(self):
        cdef set seen = set()
        cdef list result = []
        cdef Py_ssize_t i, n = self.length
        cdef long long val
        for i in range(n):
            val = self.data_c[i]
            if val not in seen:
                seen.add(val)
                result.append(val)
        return result

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def year(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef long long* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = <int>(ptr[i] / 10000)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def month(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef long long* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = <int>((ptr[i] / 100) % 100)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def day(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef long long* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = <int>(ptr[i] % 100)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def quarter(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef long long* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int m
        with nogil:
            for i in range(n):
                m = <int>((ptr[i] / 100) % 100)
                res_ptr[i] = (m - 1) / 3 + 1
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def strftime(self, str fmt):
        cdef bytes b_fmt = fmt.encode('ascii', errors='ignore')
        cdef const char* c_fmt = b_fmt
        cdef Py_ssize_t fmt_len = len(b_fmt)

        cdef Char256Column_SoA res = Char256Column_SoA(max(self.length, 1))
        res.length = self.length

        cdef Py_ssize_t i, n = self.length
        cdef long long* ptr = self.data_c
        cdef int y, m, d
        cdef char* dest
        cdef int f_idx, d_idx
        cdef char ch

        with nogil:
            for i in range(n):
                _split_ymd(ptr[i], &y, &m, &d)
                dest = res.data_c + (i * 256)
                f_idx = 0
                d_idx = 0

                while f_idx < fmt_len and d_idx < 255:
                    ch = c_fmt[f_idx]
                    if ch == b'%' and f_idx + 1 < fmt_len:
                        f_idx += 1
                        ch = c_fmt[f_idx]
                        if ch == b'Y':
                            if d_idx + 4 <= 255:
                                dest[d_idx]   = <char>(48 + ((y / 1000) % 10))
                                dest[d_idx+1] = <char>(48 + ((y / 100) % 10))
                                dest[d_idx+2] = <char>(48 + ((y / 10) % 10))
                                dest[d_idx+3] = <char>(48 + (y % 10))
                                d_idx += 4
                        elif ch == b'm':
                            if d_idx + 2 <= 255:
                                dest[d_idx]   = <char>(48 + (m / 10))
                                dest[d_idx+1] = <char>(48 + (m % 10))
                                d_idx += 2
                        elif ch == b'd':
                            if d_idx + 2 <= 255:
                                dest[d_idx]   = <char>(48 + (d / 10))
                                dest[d_idx+1] = <char>(48 + (d % 10))
                                d_idx += 2
                        else:
                            if d_idx + 2 <= 255:
                                dest[d_idx] = b'%'
                                dest[d_idx+1] = ch
                                d_idx += 2
                    else:
                        dest[d_idx] = ch
                        d_idx += 1
                    f_idx += 1
                dest[d_idx] = 0
        return res

    cdef inline long long _resolve_date(self, date) except *:
        if isinstance(date, str):
            return _parse_date_str(<str>date)
        return <long long>date

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def filter_greater(self, date):
        cdef long long target = self._resolve_date(date)
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef long long* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = 1 if ptr[i] > target else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def filter_greater_equal(self, date):
        cdef long long target = self._resolve_date(date)
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef long long* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = 1 if ptr[i] >= target else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def filter_less(self, date):
        cdef long long target = self._resolve_date(date)
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef long long* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = 1 if ptr[i] < target else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def filter_less_equal(self, date):
        cdef long long target = self._resolve_date(date)
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef long long* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = 1 if ptr[i] <= target else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def filter_equal(self, date):
        cdef long long target = self._resolve_date(date)
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef long long* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = 1 if ptr[i] == target else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def filter_not_equal(self, date):
        cdef long long target = self._resolve_date(date)
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef long long* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = 1 if ptr[i] != target else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def filter_between(self, start, end):
        cdef long long lo = self._resolve_date(start)
        cdef long long hi = self._resolve_date(end)
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef long long* ptr = self.data_c
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

        cdef long long fill_val = self._resolve_date(fill)
        cdef long long* col_ptr = col.data_c
        cdef long long* self_ptr = self.data_c
        cdef int* mask_ptr = mask.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                col_ptr[i] = self_ptr[i] if mask_ptr[i] != 0 else fill_val
        return col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def add_days(self, int n):
        cdef Py_ssize_t length = self.length
        cdef DateColumn res = DateColumn(max(length, 1))
        res.length = length
        cdef long long* ptr = self.data_c
        cdef long long* res_ptr = res.data_c
        cdef Py_ssize_t i
        cdef int y, m, d
        cdef long long daynum
        with nogil:
            for i in range(length):
                _split_ymd(ptr[i], &y, &m, &d)
                daynum = _date_to_daynum(y, m, d) + n
                _daynum_to_date(daynum, &y, &m, &d)
                res_ptr[i] = <long long>y * 10000 + m * 100 + d
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
        cdef long long* ptr = self.data_c
        cdef long long* res_ptr = res.data_c
        cdef Py_ssize_t i
        cdef int y, m, d, total, new_m, new_y, max_d
        with nogil:
            for i in range(length):
                _split_ymd(ptr[i], &y, &m, &d)
                total = (m - 1) + n
                new_y = y + total / 12
                new_m = total % 12 + 1
                if new_m <= 0:
                    new_m += 12
                    new_y -= 1
                max_d = _days_in_month(new_y, new_m)
                if d > max_d:
                    d = max_d
                res_ptr[i] = <long long>new_y * 10000 + new_m * 100 + d
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
        cdef long long* ptr = self.data_c
        cdef long long* res_ptr = res.data_c
        cdef Py_ssize_t i
        cdef int y, m, d, new_y, max_d
        with nogil:
            for i in range(length):
                _split_ymd(ptr[i], &y, &m, &d)
                new_y = y + n
                max_d = _days_in_month(new_y, m)
                if d > max_d:
                    d = max_d
                res_ptr[i] = <long long>new_y * 10000 + m * 100 + d
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sub_years(self, int n):
        return self.add_years(-n)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def day_of_week(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef long long* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        cdef long long daynum
        with nogil:
            for i in range(n):
                _split_ymd(ptr[i], &y, &m, &d)
                daynum = _date_to_daynum(y, m, d)
                res_ptr[i] = <int>(((daynum % 7) + 7) % 7)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def day_of_year(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef long long* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d, doy
        with nogil:
            for i in range(n):
                _split_ymd(ptr[i], &y, &m, &d)
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
        cdef long long* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        with nogil:
            for i in range(n):
                _split_ymd(ptr[i], &y, &m, &d)
                res_ptr[i] = _days_in_month(y, m)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def is_leap_year(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef long long* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = 1 if _is_leap_year(<int>(ptr[i] / 10000)) else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def is_weekend(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef long long* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d, dow
        cdef long long daynum
        with nogil:
            for i in range(n):
                _split_ymd(ptr[i], &y, &m, &d)
                daynum = _date_to_daynum(y, m, d)
                dow = <int>(((daynum % 7) + 7) % 7)
                res_ptr[i] = 1 if (dow == 5 or dow == 6) else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def is_month_start(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef long long* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = 1 if (ptr[i] % 100) == 1 else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def is_month_end(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef long long* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        with nogil:
            for i in range(n):
                _split_ymd(ptr[i], &y, &m, &d)
                res_ptr[i] = 1 if d == _days_in_month(y, m) else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def is_year_start(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef long long* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = 1 if (ptr[i] % 10000) == 101 else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def is_year_end(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef long long* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = 1 if (ptr[i] % 10000) == 1231 else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def is_quarter_start(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef long long* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        with nogil:
            for i in range(n):
                _split_ymd(ptr[i], &y, &m, &d)
                res_ptr[i] = 1 if ((m == 1 or m == 4 or m == 7 or m == 10) and d == 1) else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def is_quarter_end(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef long long* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        with nogil:
            for i in range(n):
                _split_ymd(ptr[i], &y, &m, &d)
                if m == 3 or m == 6 or m == 9 or m == 12:
                    res_ptr[i] = 1 if d == _days_in_month(y, m) else 0
                else:
                    res_ptr[i] = 0
        return res

    cdef inline int _week_of_year_impl(self, int y, int m, int d) noexcept nogil:
        cdef long long daynum = _date_to_daynum(y, m, d)
        cdef long long day4jan = _date_to_daynum(y, 1, 4)
        cdef int dow4jan = <int>(((day4jan % 7) + 7) % 7)
        cdef long long week1_start = day4jan - dow4jan
        cdef long long next_jan4, next_week1_start
        cdef int next_jan4_dow, week, max_week

        if daynum < week1_start:
            return self._week_of_year_impl(y - 1, 12, 31)

        next_jan4 = _date_to_daynum(y + 1, 1, 4)
        next_jan4_dow = <int>(((next_jan4 % 7) + 7) % 7)
        next_week1_start = next_jan4 - next_jan4_dow

        if daynum >= next_week1_start:
            return 1

        max_week = 53 if _is_leap_year(y) else 52
        week = <int>((daynum - week1_start) / 7) + 1
        return max_week if week > max_week else week

    cdef inline int _week_of_year(self, int y, int m, int d) noexcept nogil:
        return self._week_of_year_impl(y, m, d)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def week_of_year(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef long long* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        with nogil:
            for i in range(n):
                _split_ymd(ptr[i], &y, &m, &d)
                res_ptr[i] = self._week_of_year(y, m, d)
        return res

    # ------------------------------------------------------------
    # 查找表零分支优化：day_name & month_name
    # ------------------------------------------------------------
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def day_name(self):
        cdef Char256Column_SoA res = Char256Column_SoA(max(self.length, 1))
        res.length = self.length
        cdef Py_ssize_t i, n = self.length
        cdef long long* ptr = self.data_c
        cdef int y, m, d, dow
        cdef long long daynum
        cdef char* dest

        with nogil:
            for i in range(n):
                _split_ymd(ptr[i], &y, &m, &d)
                daynum = _date_to_daynum(y, m, d)
                dow = <int>(((daynum % 7) + 7) % 7)
                dest = res.data_c + (i * 256)
                memcpy(dest, _DAY_NAMES[dow].name, _DAY_NAMES[dow].len)
                dest[_DAY_NAMES[dow].len] = 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def month_name(self):
        cdef Char256Column_SoA res = Char256Column_SoA(max(self.length, 1))
        res.length = self.length
        cdef Py_ssize_t i, n = self.length
        cdef int m
        cdef char* dest
        cdef long long* ptr = self.data_c

        with nogil:
            for i in range(n):
                m = <int>((ptr[i] / 100) % 100)
                dest = res.data_c + (i * 256)
                if m >= 1 and m <= 12:
                    memcpy(dest, _MONTH_NAMES[m].name, _MONTH_NAMES[m].len)
                    dest[_MONTH_NAMES[m].len] = 0
                else:
                    dest[0] = 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def last_day_of_month(self):
        cdef DateColumn res = DateColumn(max(self.length, 1))
        res.length = self.length
        cdef long long* ptr = self.data_c
        cdef long long* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d, max_d
        with nogil:
            for i in range(n):
                _split_ymd(ptr[i], &y, &m, &d)
                max_d = _days_in_month(y, m)
                res_ptr[i] = <long long>y * 10000 + m * 100 + max_d
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
        cdef long long* ptr = self.data_c
        cdef int* ry_ptr = res_y.data_c
        cdef int* rw_ptr = res_w.data_c
        cdef int* rd_ptr = res_d.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d, dow
        cdef long long daynum
        with nogil:
            for i in range(n):
                _split_ymd(ptr[i], &y, &m, &d)
                daynum = _date_to_daynum(y, m, d)
                dow = <int>(((daynum % 7) + 7) % 7) + 1
                ry_ptr[i] = y
                rw_ptr[i] = self._week_of_year(y, m, d)
                rd_ptr[i] = dow
        return (res_y, res_w, res_d)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def diff_months(self, date):
        cdef long long target = self._resolve_date(date)
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef long long* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y1, m1, d1, y2, m2, d2
        _split_ymd(target, &y2, &m2, &d2)
        with nogil:
            for i in range(n):
                _split_ymd(ptr[i], &y1, &m1, &d1)
                res_ptr[i] = (y1 - y2) * 12 + (m1 - m2) - (1 if d1 < d2 else 0)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def diff_years(self, date):
        cdef long long target = self._resolve_date(date)
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef long long* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y1, m1, d1, y2, m2, d2
        _split_ymd(target, &y2, &m2, &d2)
        with nogil:
            for i in range(n):
                _split_ymd(ptr[i], &y1, &m1, &d1)
                res_ptr[i] = y1 - y2 - (1 if (m1 < m2 or (m1 == m2 and d1 < d2)) else 0)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def diff_days(self, date):
        cdef long long target = self._resolve_date(date)
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef long long* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        cdef long long target_daynum

        _split_ymd(target, &y, &m, &d)
        target_daynum = _date_to_daynum(y, m, d)

        with nogil:
            for i in range(n):
                _split_ymd(ptr[i], &y, &m, &d)
                res_ptr[i] = <int>(_date_to_daynum(y, m, d) - target_daynum)
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def _cmp_op(self, val, int op):
        cdef Py_ssize_t i, n = self.length
        cdef IntColumn res = IntColumn(max(n, 1))
        res.length = n

        cdef long long* s_ptr = self.data_c
        cdef long long* o_ptr
        cdef int* r_ptr = res.data_c
        cdef char err_buf[128]
        cdef long long v
        cdef DateColumn other

        if isinstance(val, DateColumn):
            other = <DateColumn>val
            if other.length != n:
                snprintf(err_buf, sizeof(err_buf), "Length mismatch: %zd vs %zd", n, other.length)
                raise ValueError(err_buf.decode('ascii'))
            o_ptr = other.data_c
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
        for i in range(n):
            items.append(str(self.data_c[i]))
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
        cdef long long* old_data = self.data_c
        cdef long long* new_data
        cdef DateColumn new_col

        if not inplace:
            new_col = DateColumn(max(n, 1))
            new_col.length = n
            with nogil:
                for i in range(n):
                    new_col.data_c[i] = old_data[indices[i]]
            return new_col
        else:
            new_data = <long long*>aligned_malloc(max(n, 1) * sizeof(long long))
            if not new_data: raise MemoryError()
            with nogil:
                for i in range(n):
                    new_data[i] = old_data[indices[i]]
            os_aligned_free(self.data_c)
            self.data_c = new_data
            self.capacity = n
            self.length = n
            return self

    cpdef void drop_row(self, Py_ssize_t row_idx, Py_ssize_t rem_len):
        if row_idx < 0:
            row_idx += self.length
        if rem_len > 0:
            memmove(self.data_c + row_idx,
                    self.data_c + row_idx + 1,
                    rem_len * sizeof(long long))
        self.length -= 1

    # ------------------------------------------------------------
    # astype: 使用纯 C 格式化与零分配直写替换 f-string
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
        cdef long long* s_ptr = self.data_c

        if target_type == TYPE_INT:
            ic = IntColumn(safe_cap)
            ic.length = n
            with nogil:
                for i in range(n):
                    ic.data_c[i] = <int>s_ptr[i]
            return ic
        elif target_type == TYPE_DOUBLE:
            dc = DoubleColumn(safe_cap)
            dc.length = n
            with nogil:
                for i in range(n):
                    dc.data_c[i] = <double>s_ptr[i]
            return dc
        elif target_type == TYPE_CHAR256:
            cc = Char256Column_SoA(safe_cap)
            cc.length = n
            with nogil:
                for i in range(n):
                    _split_ymd(s_ptr[i], &y, &m, &d)
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
                _split_ymd(s_ptr[i], &y, &m, &d)
                snprintf(str_buf, sizeof(str_buf), "%04d-%02d-%02d", y, m, d)
                uc._set_item(i, str_buf.decode('ascii'))
            return uc
        elif target_type == TYPE_CATEGORY:
            catc = CategoryColumn(safe_cap)
            for i in range(n):
                _split_ymd(s_ptr[i], &y, &m, &d)
                snprintf(str_buf, sizeof(str_buf), "%04d-%02d-%02d", y, m, d)
                catc.append(str_buf.decode('ascii'))
            return catc
        elif target_type == TYPE_DATETIME:
            dtc = DateTimeColumn(safe_cap)
            for i in range(n):
                _split_ymd(s_ptr[i], &y, &m, &d)
                snprintf(str_buf, sizeof(str_buf), "%04d-%02d-%02d", y, m, d)
                dtc.append(str_buf.decode('ascii'))
            return dtc
        elif target_type == TYPE_TIME:
            tmc = TimeColumn(safe_cap)
            tmc.length = n
            with nogil:
                for i in range(n):
                    tmc.data_c[i] = 0
            return tmc
        elif target_type == TYPE_TIMESTAMP:
            return self.copy()

        raise TypeError(f"Failed to cast TIMESTAMP column to internal type id: {target_type}")