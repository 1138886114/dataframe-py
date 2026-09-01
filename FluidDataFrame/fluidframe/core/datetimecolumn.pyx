# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False

cimport cython
from libc.string cimport memcpy, memmove
from libc.stdio cimport snprintf

from fluidframe.core._base cimport BaseColumn, aligned_malloc, aligned_realloc, os_aligned_free, TYPE_DATETIME, TYPE_INT, TYPE_DOUBLE, TYPE_CHAR256, TYPE_UCS2, TYPE_BOOL, TYPE_TIMESTAMP, TYPE_CATEGORY, TYPE_TIME, _resolve_astype_target
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.doublecolumn cimport DoubleColumn
from fluidframe.core.char256column cimport Char256Column_SoA
from fluidframe.core.ucs2column cimport UCS2Column_SoA
from fluidframe.core.categorycolumn cimport CategoryColumn
from fluidframe.core.datecolumn cimport DateColumn, _date_to_daynum, _daynum_to_date


# Reference epoch: 2024-01-01 00:00:00 (daynum of 2024-01-01)

cdef inline long long _parse_datetime_str(str s):
    """Parse 'YYYY-MM-DD HH:MM:SS' or 'YYYY-MM-DD' or 'YYYYMMDD' to microseconds since epoch."""
    cdef bytes b = s.encode('ascii', errors='ignore')
    cdef const unsigned char* buf = <const unsigned char*>b
    cdef int max_len = len(b)
    cdef int digits[20]
    cdef int ndigits = 0
    cdef int i = 0
    cdef int year = 0, month = 1, day = 1, hour = 0, minute = 0, second = 0
    cdef long long daynum, us

    # Extract all digits
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

    # Calculate day number from epoch
    daynum = _date_to_daynum(year, month, day) - _date_to_daynum(2024, 1, 1)
    us = daynum * 86400000000LL + hour * 3600000000LL + minute * 60000000LL + second * 1000000LL
    return us


cdef inline void _datetime_to_ymdhms(long long us, int* y, int* m, int* d, int* h, int* mi, int* s) noexcept nogil:
    """Convert microseconds since epoch to year, month, day, hour, minute, second."""
    cdef long long total_seconds = us // 1000000LL
    cdef long long daynum = total_seconds // 86400
    cdef int remainder = <int>(total_seconds - daynum * 86400)
    if remainder < 0:
        remainder += 86400
        daynum -= 1
    h[0] = remainder // 3600
    remainder = remainder - (remainder // 3600) * 3600
    mi[0] = remainder // 60
    s[0] = remainder - (remainder // 60) * 60
    # Convert daynum to date
    cdef long long abs_daynum = daynum + _date_to_daynum(2024, 1, 1)
    _daynum_to_date(abs_daynum, y, m, d)


cdef inline str _datetime_to_str(long long us):
    """Format microseconds since epoch as 'YYYY-MM-DD[ HH:MM:SS]'."""
    cdef int y, m, d, h, mi, s
    _datetime_to_ymdhms(us, &y, &m, &d, &h, &mi, &s)
    if h == 0 and mi == 0 and s == 0:
        return f"{y:04d}-{m:02d}-{d:02d}"
    return f"{y:04d}-{m:02d}-{d:02d} {h:02d}:{mi:02d}:{s:02d}"


cdef inline int _dt_extract_field(long long us, int which) noexcept nogil:
    cdef int y, m, d, h, mi, s
    _datetime_to_ymdhms(us, &y, &m, &d, &h, &mi, &s)
    if which == 0: return y
    elif which == 1: return m
    elif which == 2: return d
    elif which == 3: return h
    elif which == 4: return mi
    return s


@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef class DateTimeColumn(BaseColumn):

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __cinit__(self, Py_ssize_t init_capacity=16):
        self.capacity = init_capacity if init_capacity > 0 else 1
        self.length = 0
        self.dtype = TYPE_DATETIME
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
            v = _parse_datetime_str(<str>val)
        else:
            v = <long long>val
        if self.length >= self.capacity:
            self._reserve(self.capacity * 2)
        self.data_c[self.length] = v
        self.length += 1

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def copy(self):
        cdef DateTimeColumn new_col = DateTimeColumn(self.capacity)
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
    def concat(self, DateTimeColumn other):
        self._reserve(self.length + other.length)
        memcpy(self.data_c + self.length, other.data_c, other.length * sizeof(long long))
        self.length += other.length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def max(self):
        if self.length == 0: raise ValueError("empty column")
        cdef long long max_val = self.data_c[0]
        cdef long long* ptr = self.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(1, self.length):
                if ptr[i] > max_val: max_val = ptr[i]
        return max_val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def min(self):
        if self.length == 0: raise ValueError("empty column")
        cdef long long min_val = self.data_c[0]
        cdef long long* ptr = self.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(1, self.length):
                if ptr[i] < min_val: min_val = ptr[i]
        return min_val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _reorder(self, list indices):
        cdef Py_ssize_t n = len(indices)
        cdef char err_buf[128]
        if n != self.length:
            snprintf(err_buf, sizeof(err_buf), "Length of indices (%zd) must match column length (%zd)", n, self.length)
            raise ValueError(err_buf.decode('ascii'))
        cdef long long* new_data = <long long*>aligned_malloc(max(n, 1) * sizeof(long long))
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

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __getitem__(self, Py_ssize_t idx):
        if idx < 0: idx += self.length
        if idx < 0 or idx >= self.length: raise IndexError()
        return _datetime_to_str(self.data_c[idx])

    @cython.binding(False)
    @cython.always_allow_keywords(False)
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

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def year(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef long long* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                res_ptr[i] = _dt_extract_field(ptr[i], 0)
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def month(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef long long* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                res_ptr[i] = _dt_extract_field(ptr[i], 1)
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def day(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef long long* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                res_ptr[i] = _dt_extract_field(ptr[i], 2)
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def hour(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef long long* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                res_ptr[i] = _dt_extract_field(ptr[i], 3)
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def minute(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef long long* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                res_ptr[i] = _dt_extract_field(ptr[i], 4)
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def second(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef long long* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                res_ptr[i] = _dt_extract_field(ptr[i], 5)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def where_mask(self, IntColumn mask, object fill):
        cdef Py_ssize_t n = self.length
        if mask.length != n:
            raise ValueError(f"Mask length {mask.length} != column length {n}")
        cdef DateTimeColumn col = DateTimeColumn(n)
        col.length = n
        cdef long long fill_val = 0
        if isinstance(fill, str):
            fill_val = _parse_datetime_str(<str>fill)
        elif fill is not None:
            fill_val = <long long>fill
        cdef long long* col_ptr = col.data_c
        cdef long long* self_ptr = self.data_c
        cdef int* mask_ptr = mask.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                col_ptr[i] = self_ptr[i] if mask_ptr[i] else fill_val
        return col

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def isin(self, set values):
        cdef Py_ssize_t n = self.length
        cdef Py_ssize_t i
        cdef bint found
        from fluidframe.core.boolcolumn import BoolColumn as PyBoolColumn
        res = PyBoolColumn(n)
        res.length = n
        for i in range(n):
            found = False
            for v in values:
                if isinstance(v, str):
                    if self.data_c[i] == _parse_datetime_str(v):
                        found = True
                        break
                elif self.data_c[i] == <long long>v:
                    found = True
                    break
            res.data_c[i] = 1 if found else 0
        return res

    cpdef void drop_row(self, Py_ssize_t row_idx, Py_ssize_t rem_len):
        if rem_len > 0:
            memmove(self.data_c + row_idx,
                    self.data_c + row_idx + 1,
                    rem_len * sizeof(long long))
        self.length -= 1

    cdef DateTimeColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=False):
        cdef Py_ssize_t i
        cdef long long* old_data = self.data_c
        if not inplace:
            new_col = DateTimeColumn(max(n, 1))
            new_col.length = n
            with nogil:
                for i in range(n):
                    new_col.data_c[i] = old_data[indices[i]]
            return new_col
        else:
            new_data = <long long*>aligned_malloc(max(n, self.capacity) * sizeof(long long))
            if not new_data: raise MemoryError()
            with nogil:
                for i in range(n):
                    new_data[i] = old_data[indices[i]]
            os_aligned_free(self.data_c)
            self.data_c = new_data
            self.capacity = max(n, self.capacity)
            self.length = n
            return self

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def astype(self, object target):
        cdef int target_type = _resolve_astype_target(target)
        cdef Py_ssize_t i, n = self.length
        cdef Py_ssize_t safe_cap = max(n, 1)
        cdef int y, m, d, h, mi, s
        cdef long long* ptr = self.data_c
        cdef IntColumn ic
        cdef DoubleColumn dc
        cdef UCS2Column_SoA uc
        cdef Char256Column_SoA cc
        cdef CategoryColumn catc
        if target_type == TYPE_INT:
            ic = IntColumn(safe_cap)
            ic.length = n
            with nogil:
                for i in range(n):
                    ic.data_c[i] = <int>(ptr[i] / 1000000LL)
            return ic
        elif target_type == TYPE_DOUBLE:
            dc = DoubleColumn(safe_cap)
            dc.length = n
            with nogil:
                for i in range(n):
                    dc.data_c[i] = <double>ptr[i] / 1000000.0
            return dc
        elif target_type == TYPE_UCS2:
            uc = UCS2Column_SoA(safe_cap)
            uc.length = n
            for i in range(n):
                uc._set_item(i, _datetime_to_str(ptr[i]))
            return uc
        elif target_type == TYPE_CHAR256:
            cc = Char256Column_SoA(safe_cap)
            for i in range(n):
                cc.append(_datetime_to_str(ptr[i]))
            return cc
        elif target_type == TYPE_CATEGORY:
            catc = CategoryColumn(safe_cap)
            for i in range(n):
                catc.append(_datetime_to_str(ptr[i]))
            return catc
        elif target_type == TYPE_TIMESTAMP:
            from fluidframe.core.datecolumn import DateColumn as PyDateColumn
            tc = PyDateColumn(safe_cap)
            for i in range(n):
                _datetime_to_ymdhms(ptr[i], &y, &m, &d, &h, &mi, &s)
                tc.append(f"{y:04d}{m:02d}{d:02d}")
            return tc
        elif target_type == TYPE_DATETIME:
            return self.copy()
        elif target_type == TYPE_TIME:
            from fluidframe.core.timecolumn import TimeColumn as PyTimeColumn
            tmc = PyTimeColumn(safe_cap)
            for i in range(n):
                _datetime_to_ymdhms(ptr[i], &y, &m, &d, &h, &mi, &s)
                tmc.append(h * 3600 + mi * 60 + s)
            return tmc
        raise TypeError(f"Failed to cast DATETIME column to internal type id: {target_type}")
