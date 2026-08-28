# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport cython
from libc.string cimport memcpy, memmove

from fluidframe.core._base cimport BaseColumn, aligned_malloc, aligned_realloc, os_aligned_free, TYPE_TIMESTAMP
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.char256column cimport Char256Column_SoA


# ---------------------------------------------------------------------------
#  C-level date helpers (all nogil)
# ---------------------------------------------------------------------------

cdef inline bint _is_leap_year(int y) noexcept nogil:
    return (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)

cdef inline int _days_in_month(int y, int m) noexcept nogil:
    if m == 2:
        return 29 if _is_leap_year(y) else 28
    if m in (1, 3, 5, 7, 8, 10, 12):
        return 31
    return 30

cdef inline long long _date_to_daynum(int y, int m, int d) noexcept nogil:
    """Convert YYYY-MM-DD to sequential day number (days since 0001-01-01)."""
    cdef int a, y2, m2
    a = (14 - m) / 12
    y2 = y + 4800 - a
    m2 = m + 12 * a - 3
    return d + (153 * m2 + 2) / 5 + 365 * y2 + y2 / 4 - y2 / 100 + y2 / 400 - 32045

cdef inline void _daynum_to_date(long long daynum, int* y, int* m, int* d) noexcept nogil:
    """Convert sequential day number back to YYYY-MM-DD."""
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
    """Parse 'YYYYMMDD' or 'YYYY-MM-DD' to long long YYYYMMDD."""
    cdef bytes b = s.encode('ascii', errors='ignore')
    cdef const unsigned char* buf = <const unsigned char*>b
    cdef long long val = 0
    cdef int i = 0
    while buf[i] >= 48 and buf[i] <= 57:
        val = val * 10 + (buf[i] - 48)
        i += 1
    return val


# ---------------------------------------------------------------------------
#  TimestampColumn
# ---------------------------------------------------------------------------

@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class TimestampColumn(BaseColumn):

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __cinit__(self, Py_ssize_t init_capacity=16):
        self.capacity = init_capacity if init_capacity > 0 else 1
        self.length = 0
        self.dtype = TYPE_TIMESTAMP
        self.data_c = <long long*>aligned_malloc(self.capacity * sizeof(long long))
        if not self.data_c: raise MemoryError()

    def __dealloc__(self):
        if self.data_c is not NULL: os_aligned_free(self.data_c)

    cdef void _reserve(self, Py_ssize_t new_cap):
        cdef long long* new_ptr
        if new_cap > self.capacity:
            new_ptr = <long long*>aligned_realloc(self.data_c, self.capacity * sizeof(long long), new_cap * sizeof(long long))
            if not new_ptr: raise MemoryError()
            self.data_c = new_ptr
            self.capacity = new_cap

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def append(self, val):
        cdef long long v
        if isinstance(val, str):
            v = self._parse(<str>val)
        else:
            v = <long long>val
        if self.length >= self.capacity:
            self._reserve(self.capacity * 2)
        self.data_c[self.length] = v
        self.length += 1

    cdef inline long long _parse(self, str s) noexcept:
        cdef bytes b = s.encode('ascii', errors='ignore')
        cdef const unsigned char* buf = <const unsigned char*>b
        cdef long long val = 0
        cdef int i = 0
        while buf[i] >= 48 and buf[i] <= 57:
            val = val * 10 + (buf[i] - 48)
            i += 1
        return val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def copy(self):
        cdef TimestampColumn new_col = TimestampColumn(self.capacity)
        new_col.length = self.length
        memcpy(new_col.data_c, self.data_c, self.length * sizeof(long long))
        return new_col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_back(self):
        if self.length == 0: raise IndexError("pop from empty column")
        self.length -= 1
        return self.data_c[self.length]

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_front(self):
        if self.length == 0: raise IndexError("pop from empty column")
        cdef long long val = self.data_c[0]
        memmove(self.data_c, self.data_c + 1, (self.length - 1) * sizeof(long long))
        self.length -= 1
        return val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def concat(self, TimestampColumn other):
        self._reserve(self.length + other.length)
        memcpy(self.data_c + self.length, other.data_c, other.length * sizeof(long long))
        self.length += other.length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def max(self):
        if self.length == 0: raise ValueError("empty column")
        cdef long long max_val = self.data_c[0]
        for i in range(1, self.length):
            if self.data_c[i] > max_val: max_val = self.data_c[i]
        return max_val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def min(self):
        if self.length == 0: raise ValueError("empty column")
        cdef long long min_val = self.data_c[0]
        for i in range(1, self.length):
            if self.data_c[i] < min_val: min_val = self.data_c[i]
        return min_val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _reorder(self, list indices):
        cdef Py_ssize_t n = self.length
        cdef long long* new_data = <long long*>aligned_malloc(n * sizeof(long long))
        if not new_data: raise MemoryError()
        cdef Py_ssize_t i, idx
        try:
            for i in range(n):
                idx = <Py_ssize_t>indices[i]
                if idx < 0 or idx >= n: raise IndexError("Index out of bounds")
                new_data[i] = self.data_c[idx]
        except Exception:
            os_aligned_free(new_data)
            raise
        os_aligned_free(self.data_c)
        self.data_c = new_data

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __getitem__(self, Py_ssize_t idx):
        if idx < 0: idx += self.length
        if idx < 0 or idx >= self.length: raise IndexError()
        return self.data_c[idx]

    @cython.binding(False)
    def unique(self):
        cdef set seen = set()
        cdef list result = []
        cdef Py_ssize_t i
        for i in range(self.length):
            val = self.data_c[i]
            if val not in seen:
                seen.add(val)
                result.append(val)
        return result

    @cython.binding(False)
    def year(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                res.data_c[i] = <int>(self.data_c[i] / 10000)
        return res

    @cython.binding(False)
    def month(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                res.data_c[i] = <int>((self.data_c[i] / 100) % 100)
        return res

    @cython.binding(False)
    def day(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                res.data_c[i] = <int>(self.data_c[i] % 100)
        return res

    @cython.binding(False)
    def strftime(self, str fmt):
        cdef bytes b_fmt = fmt.encode('ascii', errors='ignore')
        cdef const char* c_fmt = b_fmt
        cdef Py_ssize_t fmt_len = len(b_fmt)

        cdef Char256Column_SoA res = Char256Column_SoA(self.length)
        res.length = self.length

        cdef Py_ssize_t i
        cdef long long val
        cdef int y, m, d
        cdef char* dest
        cdef int f_idx, d_idx
        cdef char ch, c_percent = 37, c_Y = 89, c_m = 109, c_d = 100, c_zero = 48, c_backslash0 = 0

        with nogil:
            for i in range(self.length):
                val = self.data_c[i]
                y = <int>(val / 10000)
                m = <int>((val / 100) % 100)
                d = <int>(val % 100)

                dest = res.data_c + (i * 256)
                f_idx = 0
                d_idx = 0

                while f_idx < fmt_len and d_idx < 255:
                    ch = c_fmt[f_idx]
                    if ch == c_percent and f_idx + 1 < fmt_len:
                        f_idx += 1
                        ch = c_fmt[f_idx]
                        if ch == c_Y:
                            if d_idx + 4 <= 255:
                                dest[d_idx]   = c_zero + <char>((y / 1000) % 10)
                                dest[d_idx+1] = c_zero + <char>((y / 100) % 10)
                                dest[d_idx+2] = c_zero + <char>((y / 10) % 10)
                                dest[d_idx+3] = c_zero + <char>(y % 10)
                                d_idx += 4
                        elif ch == c_m:
                            if d_idx + 2 <= 255:
                                dest[d_idx]   = c_zero + <char>((m / 10) % 10)
                                dest[d_idx+1] = c_zero + <char>(m % 10)
                                d_idx += 2
                        elif ch == c_d:
                            if d_idx + 2 <= 255:
                                dest[d_idx]   = c_zero + <char>((d / 10) % 10)
                                dest[d_idx+1] = c_zero + <char>(d % 10)
                                d_idx += 2
                        else:
                            if d_idx + 2 <= 255:
                                dest[d_idx] = c_percent
                                dest[d_idx+1] = ch
                                d_idx += 2
                    else:
                        dest[d_idx] = ch
                        d_idx += 1

                    f_idx += 1
                dest[d_idx] = c_backslash0
        return res

    # -----------------------------------------------------------------
    #  Helper: resolve date argument (str or int) to long long YYYYMMDD
    # -----------------------------------------------------------------

    cdef inline long long _resolve_date(self, date) except *:
        if isinstance(date, str):
            return _parse_date_str(<str>date)
        return <long long>date

    # -----------------------------------------------------------------
    #  Filter methods (return IntColumn as boolean mask 0/1)
    # -----------------------------------------------------------------

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def filter_greater(self, date):
        cdef long long target = self._resolve_date(date)
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                res.data_c[i] = 1 if self.data_c[i] > target else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def filter_greater_equal(self, date):
        cdef long long target = self._resolve_date(date)
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                res.data_c[i] = 1 if self.data_c[i] >= target else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def filter_less(self, date):
        cdef long long target = self._resolve_date(date)
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                res.data_c[i] = 1 if self.data_c[i] < target else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def filter_less_equal(self, date):
        cdef long long target = self._resolve_date(date)
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                res.data_c[i] = 1 if self.data_c[i] <= target else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def filter_equal(self, date):
        cdef long long target = self._resolve_date(date)
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                res.data_c[i] = 1 if self.data_c[i] == target else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def filter_not_equal(self, date):
        cdef long long target = self._resolve_date(date)
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                res.data_c[i] = 1 if self.data_c[i] != target else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def filter_between(self, start, end):
        cdef long long lo = self._resolve_date(start)
        cdef long long hi = self._resolve_date(end)
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                res.data_c[i] = 1 if lo <= self.data_c[i] <= hi else 0
        return res

    # -----------------------------------------------------------------
    #  Date arithmetic (return new TimestampColumn)
    # -----------------------------------------------------------------

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def add_days(self, int n):
        cdef Py_ssize_t length = self.length
        cdef TimestampColumn res = TimestampColumn(self.capacity)
        res.length = length
        cdef Py_ssize_t i
        cdef int y, m, d
        cdef long long daynum
        with nogil:
            for i in range(length):
                y = <int>(self.data_c[i] / 10000)
                m = <int>((self.data_c[i] / 100) % 100)
                d = <int>(self.data_c[i] % 100)
                daynum = _date_to_daynum(y, m, d) + n
                _daynum_to_date(daynum, &y, &m, &d)
                res.data_c[i] = y * 10000 + m * 100 + d
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sub_days(self, int n):
        return self.add_days(-n)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def add_months(self, int n):
        cdef Py_ssize_t length = self.length
        cdef TimestampColumn res = TimestampColumn(self.capacity)
        res.length = length
        cdef Py_ssize_t i
        cdef int y, m, d, total, new_m, new_y, max_d
        with nogil:
            for i in range(length):
                y = <int>(self.data_c[i] / 10000)
                m = <int>((self.data_c[i] / 100) % 100)
                d = <int>(self.data_c[i] % 100)
                total = (m - 1) + n
                new_y = y + total / 12
                new_m = total % 12 + 1
                if new_m <= 0:
                    new_m += 12
                    new_y -= 1
                max_d = _days_in_month(new_y, new_m)
                if d > max_d:
                    d = max_d
                res.data_c[i] = new_y * 10000 + new_m * 100 + d
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sub_months(self, int n):
        return self.add_months(-n)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def add_years(self, int n):
        cdef Py_ssize_t length = self.length
        cdef TimestampColumn res = TimestampColumn(self.capacity)
        res.length = length
        cdef Py_ssize_t i
        cdef int y, m, d, new_y, max_d
        with nogil:
            for i in range(length):
                y = <int>(self.data_c[i] / 10000)
                m = <int>((self.data_c[i] / 100) % 100)
                d = <int>(self.data_c[i] % 100)
                new_y = y + n
                max_d = _days_in_month(new_y, m)
                if d > max_d:
                    d = max_d
                res.data_c[i] = new_y * 10000 + m * 100 + d
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sub_years(self, int n):
        return self.add_years(-n)

    # -----------------------------------------------------------------
    #  Date component extraction
    # -----------------------------------------------------------------

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def day_of_week(self):
        """Return day of week: 0=Monday … 6=Sunday (ISO 8601)."""
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        cdef int y, m, d
        cdef long long daynum
        with nogil:
            for i in range(self.length):
                y = <int>(self.data_c[i] / 10000)
                m = <int>((self.data_c[i] / 100) % 100)
                d = <int>(self.data_c[i] % 100)
                daynum = _date_to_daynum(y, m, d)
                # daynum mod 7: 0=Mon 1=Tue ... 6=Sun
                res.data_c[i] = <int>(daynum % 7)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def day_of_year(self):
        """Return day of year: 1–366."""
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        cdef int y, m, d, doy, dm
        cdef int cum[13]
        with nogil:
            cum[0] = 0
            for i in range(1, 13):
                cum[i] = cum[i-1] + _days_in_month(2001, i)  # 2001 is non-leap
            for i in range(self.length):
                y = <int>(self.data_c[i] / 10000)
                m = <int>((self.data_c[i] / 100) % 100)
                d = <int>(self.data_c[i] % 100)
                doy = cum[m - 1] + d
                if m > 2 and _is_leap_year(y):
                    doy += 1
                res.data_c[i] = doy
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def days_in_month(self):
        """Return number of days in each date's month (28/29/30/31)."""
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        cdef int y, m
        with nogil:
            for i in range(self.length):
                y = <int>(self.data_c[i] / 10000)
                m = <int>((self.data_c[i] / 100) % 100)
                res.data_c[i] = _days_in_month(y, m)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def is_leap_year(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        cdef int y
        with nogil:
            for i in range(self.length):
                y = <int>(self.data_c[i] / 10000)
                res.data_c[i] = 1 if _is_leap_year(y) else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def is_weekend(self):
        """Return 1 if Saturday (5) or Sunday (6)."""
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        cdef int y, m, d, dow
        cdef long long daynum
        with nogil:
            for i in range(self.length):
                y = <int>(self.data_c[i] / 10000)
                m = <int>((self.data_c[i] / 100) % 100)
                d = <int>(self.data_c[i] % 100)
                daynum = _date_to_daynum(y, m, d)
                dow = <int>(daynum % 7)
                res.data_c[i] = 1 if dow >= 5 else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def is_month_start(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                res.data_c[i] = 1 if (self.data_c[i] % 100) == 1 else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def is_month_end(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        cdef int y, m, d
        with nogil:
            for i in range(self.length):
                y = <int>(self.data_c[i] / 10000)
                m = <int>((self.data_c[i] / 100) % 100)
                d = <int>(self.data_c[i] % 100)
                res.data_c[i] = 1 if d == _days_in_month(y, m) else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def is_year_start(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                res.data_c[i] = 1 if (self.data_c[i] % 10000) == 101 else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def is_year_end(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                res.data_c[i] = 1 if (self.data_c[i] % 10000) == 1231 else 0
        return res

    # -----------------------------------------------------------------
    #  Date difference
    # -----------------------------------------------------------------

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def diff_days(self, date):
        """Return IntColumn with day difference: self[i] - date."""
        cdef long long target = self._resolve_date(date)
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        cdef int y, m, d
        cdef long long daynum
        with nogil:
            for i in range(self.length):
                y = <int>(self.data_c[i] / 10000)
                m = <int>((self.data_c[i] / 100) % 100)
                d = <int>(self.data_c[i] % 100)
                daynum = _date_to_daynum(y, m, d)
                y = <int>(target / 10000)
                m = <int>((target / 100) % 100)
                d = <int>(target % 100)
                res.data_c[i] = <int>(daynum - _date_to_daynum(y, m, d))
        return res

    # -----------------------------------------------------------------
    #  Comparison operators → return IntColumn (0/1 mask)
    # -----------------------------------------------------------------

    def _cmp_op(self, val, int op):
        cdef Py_ssize_t i
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        if isinstance(val, TimestampColumn):
            other = <TimestampColumn>val
            if other.length != self.length:
                raise ValueError(f"Length mismatch: {self.length} vs {other.length}")
            if op == 0:
                for i in range(self.length): res.data_c[i] = 1 if self.data_c[i] < other.data_c[i] else 0
            elif op == 1:
                for i in range(self.length): res.data_c[i] = 1 if self.data_c[i] <= other.data_c[i] else 0
            elif op == 2:
                for i in range(self.length): res.data_c[i] = 1 if self.data_c[i] > other.data_c[i] else 0
            elif op == 3:
                for i in range(self.length): res.data_c[i] = 1 if self.data_c[i] >= other.data_c[i] else 0
            elif op == 4:
                for i in range(self.length): res.data_c[i] = 1 if self.data_c[i] == other.data_c[i] else 0
            elif op == 5:
                for i in range(self.length): res.data_c[i] = 1 if self.data_c[i] != other.data_c[i] else 0
        else:
            v = <long long>val
            if op == 0:
                for i in range(self.length): res.data_c[i] = 1 if self.data_c[i] < v else 0
            elif op == 1:
                for i in range(self.length): res.data_c[i] = 1 if self.data_c[i] <= v else 0
            elif op == 2:
                for i in range(self.length): res.data_c[i] = 1 if self.data_c[i] > v else 0
            elif op == 3:
                for i in range(self.length): res.data_c[i] = 1 if self.data_c[i] >= v else 0
            elif op == 4:
                for i in range(self.length): res.data_c[i] = 1 if self.data_c[i] == v else 0
            elif op == 5:
                for i in range(self.length): res.data_c[i] = 1 if self.data_c[i] != v else 0
        return res

    def __lt__(self, val):  return self._cmp_op(val, 0)
    def __le__(self, val):  return self._cmp_op(val, 1)
    def __gt__(self, val):  return self._cmp_op(val, 2)
    def __ge__(self, val):  return self._cmp_op(val, 3)
    def __eq__(self, val):  return self._cmp_op(val, 4)
    def __ne__(self, val):  return self._cmp_op(val, 5)

    def __str__(self):
        cdef list items = []
        cdef Py_ssize_t i
        for i in range(self.length):
            items.append(str(self.data_c[i]))
        return f"TimestampColumn([{', '.join(items)}])"

    def __bool__(self):
        return self.length > 0

    def __len__(self):
        return self.length


    cdef TimestampColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=False):
        cdef long long* old_data = self.data_c
        cdef long long* new_data = <long long*>aligned_malloc(n * sizeof(long long))
        if not new_data: raise MemoryError()
        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                new_data[i] = old_data[indices[i]]
        if inplace:
            os_aligned_free(self.data_c)
            self.data_c = new_data
            self.length = n
            return self
        else:
            new_col = TimestampColumn(self.capacity)
            free(new_col.data_c)
            new_col.data_c = new_data
            new_col.length = n
            return new_col


    cpdef void drop_row(self, Py_ssize_t row_idx, Py_ssize_t rem_len):
        if rem_len > 0:
            memmove(self.data_c + row_idx,
                    self.data_c + row_idx + 1,
                    rem_len * sizeof(long long))
        self.length -= 1
