# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport cython
from libc.string cimport memcpy, memmove
from libc.stdlib cimport free

from fluidframe.core._base cimport BaseColumn, aligned_malloc, aligned_realloc
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.char256column cimport Char256Column_SoA

@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class TimestampColumn(BaseColumn):

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __cinit__(self, Py_ssize_t init_capacity=16):
        self.capacity = max(init_capacity, 1)
        self.length = 0
        self.data_c = <long long*>aligned_malloc(self.capacity * sizeof(long long))
        if not self.data_c: raise MemoryError()

    def __dealloc__(self):
        if self.data_c is not NULL: free(self.data_c)

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
            free(new_data)
            raise
        free(self.data_c)
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
