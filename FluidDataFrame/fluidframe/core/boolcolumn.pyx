# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport cython
from libc.string cimport memcpy, memmove
from libc.math cimport NAN
from fluidframe.core._base cimport BaseColumn, aligned_malloc, aligned_realloc, os_aligned_free, TYPE_BOOL
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.doublecolumn cimport DoubleColumn
from fluidframe.core.char256column cimport Char256Column_SoA
from fluidframe.core.ucs2column cimport UCS2Column_SoA
from fluidframe.core.categorycolumn cimport CategoryColumn
from fluidframe.core.datetimecolumn cimport DateTimeColumn
from fluidframe.core.timecolumn cimport TimeColumn


cdef class BoolColumn(BaseColumn):
    def __cinit__(self, Py_ssize_t capacity=0):
        self.dtype = TYPE_BOOL
        self.capacity = capacity if capacity > 0 else 0
        self.length = 0
        if self.capacity > 0:
            self.data_c = <unsigned char*>aligned_malloc(self.capacity * sizeof(unsigned char))
            if not self.data_c: raise MemoryError()
        else:
            self.data_c = NULL

    def __dealloc__(self):
        if self.data_c != NULL:
            os_aligned_free(self.data_c)
            self.data_c = NULL

    cdef void _reserve(self, Py_ssize_t new_cap):
        if new_cap <= self.capacity:
            return
        cdef unsigned char* new_data = <unsigned char*>aligned_realloc(
            self.data_c, self.capacity * sizeof(unsigned char), new_cap * sizeof(unsigned char))
        if not new_data: raise MemoryError()
        self.data_c = new_data
        self.capacity = new_cap

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def append(self, object val):
        if self.length >= self.capacity:
            self._reserve(max(16, self.capacity * 2))
        self.data_c[self.length] = 1 if val else 0
        self.length += 1

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def append_bool(self, bint val):
        if self.length >= self.capacity:
            self._reserve(max(16, self.capacity * 2))
        self.data_c[self.length] = 1 if val else 0
        self.length += 1

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def pop_back(self):
        if self.length == 0: raise IndexError("pop from empty column")
        self.length -= 1
        return bool(self.data_c[self.length])

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def pop_front(self):
        if self.length == 0: raise IndexError("pop from empty column")
        cdef unsigned char val = self.data_c[0]
        memmove(self.data_c, self.data_c + 1, (self.length - 1) * sizeof(unsigned char))
        self.length -= 1
        return bool(val)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def concat(self, BoolColumn other):
        cdef Py_ssize_t new_len = self.length + other.length
        self._reserve(new_len)
        memcpy(self.data_c + self.length, other.data_c, other.length * sizeof(unsigned char))
        self.length = new_len

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def copy(self):
        cdef BoolColumn col = BoolColumn(max(self.capacity, 1))
        col.length = self.length
        if self.length > 0:
            memcpy(col.data_c, self.data_c, self.length * sizeof(unsigned char))
        return col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _reorder(self, list indices):
        cdef Py_ssize_t n = len(indices)
        if n != self.length:
            raise ValueError("Length of indices must match column length")

        cdef unsigned char* new_data = <unsigned char*>aligned_malloc(max(n, 1) * sizeof(unsigned char))
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

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __getitem__(self, Py_ssize_t idx):
        if idx < 0: idx += self.length
        if idx < 0 or idx >= self.length: raise IndexError("Index out of bounds")
        return bool(self.data_c[idx])

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __setitem__(self, Py_ssize_t idx, object val):
        if idx < 0: idx += self.length
        if idx < 0 or idx >= self.length: raise IndexError("Index out of bounds")
        self.data_c[idx] = 1 if val else 0

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __len__(self):
        return self.length

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __bool__(self):
        return self.length > 0

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def isin(self, set values):
        cdef BoolColumn res = BoolColumn(max(self.capacity, 1))
        res.length = self.length
        cdef Py_ssize_t i
        for i in range(self.length):
            res.data_c[i] = 1 if bool(self.data_c[i]) in values else 0
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def where_mask(self, IntColumn mask, object fill):
        cdef Py_ssize_t n = self.length
        if mask.length != n:
            raise ValueError(f"Mask length {mask.length} != column length {n}")
        cdef BoolColumn col = BoolColumn(max(n, 1))
        col.length = n
        cdef unsigned char fill_val = 1 if fill else 0
        cdef Py_ssize_t i
        for i in range(n):
            col.data_c[i] = self.data_c[i] if mask.data_c[i] else fill_val
        return col

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def sum(self):
        cdef Py_ssize_t i, s = 0
        for i in range(self.length):
            s += self.data_c[i]
        return s

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def mean(self):
        if self.length == 0: return NAN
        return <double>self.sum() / self.length

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __and__(self, BoolColumn other):
        cdef Py_ssize_t n = self.length
        if other.length != n:
            raise ValueError(f"Column length {other.length} != {n}")
        cdef BoolColumn res = BoolColumn(max(n, 1))
        res.length = n
        cdef Py_ssize_t i
        for i in range(n):
            res.data_c[i] = self.data_c[i] & other.data_c[i]
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __or__(self, BoolColumn other):
        cdef Py_ssize_t n = self.length
        if other.length != n:
            raise ValueError(f"Column length {other.length} != {n}")
        cdef BoolColumn res = BoolColumn(max(n, 1))
        res.length = n
        cdef Py_ssize_t i
        for i in range(n):
            res.data_c[i] = self.data_c[i] | other.data_c[i]
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __xor__(self, BoolColumn other):
        cdef Py_ssize_t n = self.length
        if other.length != n:
            raise ValueError(f"Column length {other.length} != {n}")
        cdef BoolColumn res = BoolColumn(max(n, 1))
        res.length = n
        cdef Py_ssize_t i
        for i in range(n):
            res.data_c[i] = self.data_c[i] ^ other.data_c[i]
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __invert__(self):
        cdef BoolColumn res = BoolColumn(max(self.capacity, 1))
        res.length = self.length
        cdef Py_ssize_t i
        for i in range(self.length):
            res.data_c[i] = 1 - self.data_c[i]
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __str__(self):
        cdef list items = []
        cdef Py_ssize_t i
        for i in range(self.length):
            items.append(str(bool(self.data_c[i])))
        return f"BoolColumn([{', '.join(items)}])"

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __repr__(self):
        return self.__str__()

    cdef BoolColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=False):
        cdef Py_ssize_t i
        if not inplace:
            new_col = BoolColumn(max(n, 1))
            new_col.length = n
            with nogil:
                for i in range(n):
                    new_col.data_c[i] = self.data_c[indices[i]]
            return new_col
        else:
            new_data = <unsigned char*>aligned_malloc(max(max(n, self.capacity), 1) * sizeof(unsigned char))
            if not new_data: raise MemoryError()
            with nogil:
                for i in range(n):
                    new_data[i] = self.data_c[indices[i]]
            os_aligned_free(self.data_c)
            self.data_c = new_data
            self.capacity = max(n, self.capacity)
            self.length = n
            return self

    cpdef void drop_row(self, Py_ssize_t row_idx, Py_ssize_t rem_len):
        if row_idx < 0:
            row_idx += self.length
        if rem_len > 0:
            memmove(self.data_c + row_idx,
                    self.data_c + row_idx + 1,
                    rem_len * sizeof(unsigned char))
        self.length -= 1

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __eq__(self, other):
        cdef BoolColumn o
        cdef Py_ssize_t i
        if not isinstance(other, BoolColumn):
            return NotImplemented
        o = <BoolColumn>other
        if o.length != self.length:
            return False
        for i in range(self.length):
            if self.data_c[i] != o.data_c[i]:
                return False
        return True

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented: return result
        return not result

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def astype(self, object target):
        cdef str tname
        if isinstance(target, str):
            tname = target.lower()
        else:
            tname = {int: 'int', float: 'double', bool: 'bool', str: 'ucs2'}.get(target, '')
        cdef Py_ssize_t i, n = self.length
        cdef Py_ssize_t cap = max(n, 1)
        cdef IntColumn ic
        cdef DoubleColumn dc
        cdef UCS2Column_SoA uc
        cdef Char256Column_SoA cc
        cdef CategoryColumn catc
        cdef DateTimeColumn dtc
        cdef TimeColumn tmc
        if tname in ('int', 'i4'):
            ic = IntColumn(cap)
            ic.length = n
            for i in range(n):
                ic.data_c[i] = <int>self.data_c[i]
            return ic
        elif tname in ('double', 'float'):
            dc = DoubleColumn(cap)
            dc.length = n
            for i in range(n):
                dc.data_c[i] = <double>self.data_c[i]
            return dc
        elif tname in ('str', 'string', 'ucs2'):
            uc = UCS2Column_SoA(cap)
            uc.length = n
            for i in range(n):
                uc._set_item(i, 'True' if self.data_c[i] else 'False')
            return uc
        elif tname in ('char256', 'ascii'):
            cc = Char256Column_SoA(cap)
            for i in range(n):
                cc.append('True' if self.data_c[i] else 'False')
            return cc
        elif tname == 'category':
            catc = CategoryColumn(cap)
            for i in range(n):
                catc.append('True' if self.data_c[i] else 'False')
            return catc
        elif tname == 'datetime':
            dtc = DateTimeColumn(cap)
            for i in range(n):
                dtc.append('True' if self.data_c[i] else 'False')
            return dtc
        elif tname == 'time':
            tmc = TimeColumn(cap)
            for i in range(n):
                tmc.append(<int>self.data_c[i])
            return tmc
        raise TypeError(f"Cannot convert bool to '{tname}'")