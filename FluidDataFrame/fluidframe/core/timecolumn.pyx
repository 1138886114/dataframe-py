# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport cython
from libc.string cimport memcpy, memmove

from fluidframe.core._base cimport BaseColumn, aligned_malloc, aligned_realloc, os_aligned_free, TYPE_TIME, TYPE_INT, TYPE_DOUBLE, TYPE_CHAR256, TYPE_UCS2, TYPE_BOOL, TYPE_TIMESTAMP, TYPE_CATEGORY, TYPE_DATETIME, _resolve_astype_target
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.doublecolumn cimport DoubleColumn
from fluidframe.core.char256column cimport Char256Column_SoA
from fluidframe.core.ucs2column cimport UCS2Column_SoA
from fluidframe.core.categorycolumn cimport CategoryColumn
from fluidframe.core.boolcolumn cimport BoolColumn
from fluidframe.core.datecolumn cimport DateColumn


cdef int _parse_time_str(str s):
    """Parse 'HH:MM:SS' or 'HH:MM' to seconds since midnight."""
    cdef bytes b = s.encode('ascii', errors='ignore')
    cdef const unsigned char* buf = <const unsigned char*>b
    cdef int max_len = len(b)
    cdef int digits[10]
    cdef int ndigits = 0
    cdef int i = 0
    cdef int hour = 0, minute = 0, second = 0

    while i < max_len and ndigits < 10:
        if buf[i] >= 48 and buf[i] <= 57:
            digits[ndigits] = buf[i] - 48
            ndigits += 1
        i += 1

    if ndigits < 2:
        raise ValueError(f"Invalid time string: '{s}'")

    hour = digits[0]*10 + digits[1]
    if ndigits >= 4:
        minute = digits[2]*10 + digits[3]
    if ndigits >= 6:
        second = digits[4]*10 + digits[5]

    return hour * 3600 + minute * 60 + second


cdef inline void _secs_to_hms(int secs, int* h, int* m, int* s) noexcept nogil:
    h[0] = secs / 3600
    m[0] = (secs % 3600) / 60
    s[0] = secs % 60


cdef inline str _time_to_str(int secs):
    cdef int h, m, s
    _secs_to_hms(secs, &h, &m, &s)
    return f"{h:02d}:{m:02d}:{s:02d}"


@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class TimeColumn(BaseColumn):

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __cinit__(self, Py_ssize_t init_capacity=16):
        self.capacity = init_capacity if init_capacity > 0 else 1
        self.length = 0
        self.dtype = TYPE_TIME
        self.data_c = <int*>aligned_malloc(self.capacity * sizeof(int))
        if not self.data_c: raise MemoryError()

    def __dealloc__(self):
        if self.data_c is not NULL: os_aligned_free(self.data_c)

    cdef void _reserve(self, Py_ssize_t new_cap):
        cdef int* new_ptr
        if new_cap > self.capacity:
            new_ptr = <int*>aligned_realloc(self.data_c, self.capacity * sizeof(int), new_cap * sizeof(int))
            if not new_ptr: raise MemoryError()
            self.data_c = new_ptr
            self.capacity = new_cap

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def append(self, val):
        cdef int v
        if isinstance(val, str):
            v = _parse_time_str(<str>val)
        else:
            v = <int>val
        if self.length >= self.capacity:
            self._reserve(self.capacity * 2)
        self.data_c[self.length] = v
        self.length += 1

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def copy(self):
        cdef TimeColumn new_col = TimeColumn(self.capacity)
        new_col.length = self.length
        memcpy(new_col.data_c, self.data_c, self.length * sizeof(int))
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
        cdef int val = self.data_c[0]
        memmove(self.data_c, self.data_c + 1, (self.length - 1) * sizeof(int))
        self.length -= 1
        return val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def concat(self, TimeColumn other):
        self._reserve(self.length + other.length)
        memcpy(self.data_c + self.length, other.data_c, other.length * sizeof(int))
        self.length += other.length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def max(self):
        if self.length == 0: raise ValueError("empty column")
        cdef int max_val = self.data_c[0]
        for i in range(1, self.length):
            if self.data_c[i] > max_val: max_val = self.data_c[i]
        return max_val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def min(self):
        if self.length == 0: raise ValueError("empty column")
        cdef int min_val = self.data_c[0]
        for i in range(1, self.length):
            if self.data_c[i] < min_val: min_val = self.data_c[i]
        return min_val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _reorder(self, list indices):
        cdef Py_ssize_t n = len(indices)
        if n != self.length:
            raise ValueError("Length of indices must match column length")

        cdef int* new_data = <int*>aligned_malloc(max(n, 1) * sizeof(int))
        if not new_data: raise MemoryError()
        cdef Py_ssize_t i, idx
        try:
            for i in range(n):
                idx = <Py_ssize_t>indices[i]
                if idx < 0: idx += self.length
                if idx < 0 or idx >= self.length:
                    raise IndexError("Index out of bounds")
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
        return _time_to_str(self.data_c[idx])

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
    def hour(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        cdef int h, m, s
        for i in range(self.length):
            _secs_to_hms(self.data_c[i], &h, &m, &s)
            res.data_c[i] = h
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def minute(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        cdef int h, m, s
        for i in range(self.length):
            _secs_to_hms(self.data_c[i], &h, &m, &s)
            res.data_c[i] = m
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def second(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        cdef int h, m, s
        for i in range(self.length):
            _secs_to_hms(self.data_c[i], &h, &m, &s)
            res.data_c[i] = s
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def where_mask(self, IntColumn mask, object fill):
        cdef Py_ssize_t n = self.length
        if mask.length != n:
            raise ValueError(f"Mask length {mask.length} != column length {n}")
        cdef TimeColumn col = TimeColumn(max(n, 1))
        col.length = n
        cdef int fill_val = 0
        if isinstance(fill, str):
            fill_val = _parse_time_str(<str>fill)
        elif fill is not None:
            fill_val = <int>fill
        cdef Py_ssize_t i
        for i in range(n):
            col.data_c[i] = self.data_c[i] if mask.data_c[i] else fill_val
        return col

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def isin(self, set values):
        # 安全性能修复：前置集合解析，消除 O(N*M) 的性能灾难
        cdef set parsed_vals = set()
        for v in values:
            if isinstance(v, str):
                parsed_vals.add(_parse_time_str(v))
            elif v is not None:
                parsed_vals.add(<int>v)

        cdef Py_ssize_t n = self.length
        cdef Py_ssize_t i
        from fluidframe.core.boolcolumn import BoolColumn as PyBoolColumn
        res = PyBoolColumn(max(n, 1))
        res.length = n
        for i in range(n):
            res.data_c[i] = 1 if self.data_c[i] in parsed_vals else 0
        return res

    cpdef void drop_row(self, Py_ssize_t row_idx, Py_ssize_t rem_len):
        if row_idx < 0:
            row_idx += self.length
        if rem_len > 0:
            memmove(self.data_c + row_idx,
                    self.data_c + row_idx + 1,
                    rem_len * sizeof(int))
        self.length -= 1

    cdef TimeColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=False):
        cdef Py_ssize_t i
        if not inplace:
            new_col = TimeColumn(max(n, 1))
            new_col.length = n
            for i in range(n):
                new_col.data_c[i] = self.data_c[indices[i]]
            return new_col
        else:
            new_data = <int*>aligned_malloc(max(max(n, self.capacity), 1) * sizeof(int))
            if not new_data: raise MemoryError()
            for i in range(n):
                new_data[i] = self.data_c[indices[i]]
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
        cdef IntColumn ic
        cdef DoubleColumn dc
        cdef BoolColumn bc
        cdef UCS2Column_SoA uc
        cdef Char256Column_SoA cc
        cdef CategoryColumn catc
        cdef DateColumn tc
        if target_type == TYPE_INT:
            ic = IntColumn(safe_cap)
            ic.length = n
            for i in range(n):
                ic.data_c[i] = self.data_c[i]
            return ic
        elif target_type == TYPE_DOUBLE:
            dc = DoubleColumn(safe_cap)
            dc.length = n
            for i in range(n):
                dc.data_c[i] = <double>self.data_c[i]
            return dc
        elif target_type == TYPE_BOOL:
            bc = BoolColumn(safe_cap)
            bc.length = n
            for i in range(n):
                bc.data_c[i] = 1 if self.data_c[i] != 0 else 0
            return bc
        elif target_type == TYPE_UCS2:
            uc = UCS2Column_SoA(safe_cap)
            uc.length = n
            for i in range(n):
                uc._set_item(i, _time_to_str(self.data_c[i]))
            return uc
        elif target_type == TYPE_CHAR256:
            cc = Char256Column_SoA(safe_cap)
            for i in range(n):
                cc.append(_time_to_str(self.data_c[i]))
            return cc
        elif target_type == TYPE_CATEGORY:
            catc = CategoryColumn(safe_cap)
            for i in range(n):
                catc.append(_time_to_str(self.data_c[i]))
            return catc
        elif target_type == TYPE_TIMESTAMP:
            tc = DateColumn(safe_cap)
            tc.length = n
            for i in range(n):
                tc.data_c[i] = 0
            return tc
        elif target_type == TYPE_TIME:
            return self.copy()
        raise TypeError(f"Failed to cast TIME column to internal type id: {target_type}")