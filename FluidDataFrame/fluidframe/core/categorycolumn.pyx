# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport cython
from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memcpy, memmove
from libc.math cimport NAN
from fluidframe.core._base cimport BaseColumn, aligned_malloc, aligned_realloc, os_aligned_free, TYPE_CATEGORY, TYPE_INT, TYPE_DOUBLE, TYPE_CHAR256, TYPE_UCS2, TYPE_BOOL, TYPE_TIMESTAMP
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.doublecolumn cimport DoubleColumn
from fluidframe.core.boolcolumn cimport BoolColumn
from fluidframe.core.char256column cimport Char256Column_SoA
from fluidframe.core.ucs2column cimport UCS2Column_SoA
from fluidframe.core.timestampcolumn cimport TimestampColumn


cdef class CategoryColumn(BaseColumn):
    def __cinit__(self, Py_ssize_t capacity=0):
        self.dtype = TYPE_CATEGORY
        self.capacity = capacity if capacity > 0 else 0
        self.length = 0
        self.categories = []
        self._cat_to_code = {}
        if self.capacity > 0:
            self.codes_c = <int*>aligned_malloc(self.capacity * sizeof(int))
            if not self.codes_c: raise MemoryError()
        else:
            self.codes_c = NULL

    def __dealloc__(self):
        if self.codes_c != NULL:
            os_aligned_free(self.codes_c)
            self.codes_c = NULL

    cdef void _reserve(self, Py_ssize_t new_cap):
        if new_cap <= self.capacity:
            return
        cdef int* new_data = <int*>aligned_realloc(
            self.codes_c, self.capacity * sizeof(int), new_cap * sizeof(int))
        if not new_data: raise MemoryError()
        self.codes_c = new_data
        self.capacity = new_cap

    cdef int _get_or_add_code(self, str val):
        cdef int code
        if val not in self._cat_to_code:
            code = len(self.categories)
            self.categories.append(val)
            self._cat_to_code[val] = code
        return self._cat_to_code[val]

    def append(self, object val):
        if self.length >= self.capacity:
            self._reserve(max(16, self.capacity * 2))
        cdef str s = str(val) if val is not None else ''
        self.codes_c[self.length] = self._get_or_add_code(s)
        self.length += 1

    def pop_back(self):
        if self.length > 0:
            self.length -= 1

    def concat(self, CategoryColumn other):
        cdef Py_ssize_t new_len = self.length + other.length
        self._reserve(new_len)
        cdef Py_ssize_t i
        cdef int new_code
        for i in range(other.length):
            new_code = self._get_or_add_code(other.categories[other.codes_c[i]])
            self.codes_c[self.length + i] = new_code
        self.length = new_len

    def copy(self):
        cdef CategoryColumn col = CategoryColumn(self.capacity)
        col.length = self.length
        col.categories = list(self.categories)
        col._cat_to_code = dict(self._cat_to_code)
        if self.length > 0:
            memcpy(col.codes_c, self.codes_c, self.length * sizeof(int))
        return col

    @property
    def cat(self):
        return _CategoryAccessor(self)

    def __getitem__(self, Py_ssize_t idx):
        if idx < 0: idx += self.length
        if idx < 0 or idx >= self.length: raise IndexError("Index out of bounds")
        return self.categories[self.codes_c[idx]]

    def __setitem__(self, Py_ssize_t idx, object val):
        if idx < 0: idx += self.length
        if idx < 0 or idx >= self.length: raise IndexError("Index out of bounds")
        cdef str s = str(val) if val is not None else ''
        self.codes_c[idx] = self._get_or_add_code(s)

    def __len__(self):
        return self.length

    def __bool__(self):
        return self.length > 0

    def isin(self, set values):
        cdef BoolColumn res
        res = BoolColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        cdef str val
        for i in range(self.length):
            val = self.categories[self.codes_c[i]]
            res.data_c[i] = 1 if val in values else 0
        return res

    def where_mask(self, IntColumn mask, object fill):
        cdef Py_ssize_t n = self.length
        if mask.length != n:
            raise ValueError(f"Mask length {mask.length} != column length {n}")
        cdef CategoryColumn col = CategoryColumn(n)
        cdef Py_ssize_t i
        cdef str fill_str = str(fill) if fill is not None else ''
        for i in range(n):
            if mask.data_c[i]:
                col.codes_c[i] = col._get_or_add_code(self.categories[self.codes_c[i]])
            else:
                col.codes_c[i] = col._get_or_add_code(fill_str)
        col.length = n
        return col

    def __str__(self):
        cdef list items = []
        cdef Py_ssize_t i
        for i in range(self.length):
            items.append(repr(self.categories[self.codes_c[i]]))
        return f"CategoryColumn([{', '.join(items)}])"

    def __repr__(self):
        return self.__str__()

    cdef CategoryColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=False):
        cdef CategoryColumn col
        if inplace:
            col = self
        else:
            col = CategoryColumn(n)
        col.length = n
        col.categories = list(self.categories)
        col._cat_to_code = dict(self._cat_to_code)
        cdef Py_ssize_t i
        for i in range(n):
            col.codes_c[i] = self.codes_c[indices[i]]
        return col

    cpdef void drop_row(self, Py_ssize_t row_idx, Py_ssize_t rem_len):
        if row_idx < self.length - 1:
            memmove(self.codes_c + row_idx, self.codes_c + row_idx + 1,
                    (self.length - 1 - row_idx) * sizeof(int))
        self.length = rem_len

    def __eq__(self, other):
        cdef CategoryColumn o
        cdef Py_ssize_t i
        if not isinstance(other, CategoryColumn):
            return NotImplemented
        o = <CategoryColumn>other
        if o.length != self.length:
            return False
        if self.categories != o.categories:
            return False
        for i in range(self.length):
            if self.codes_c[i] != o.codes_c[i]:
                return False
        return True

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented: return result
        return not result

    def astype(self, object target):
        cdef str tname
        if isinstance(target, str):
            tname = target.lower()
        else:
            tname = {int: 'int', float: 'double', bool: 'bool', str: 'ucs2'}.get(target, '')
        cdef Py_ssize_t i, n = self.length
        cdef str s
        cdef IntColumn ic
        cdef DoubleColumn dc
        cdef UCS2Column_SoA uc
        cdef Char256Column_SoA cc
        cdef TimestampColumn tc
        if tname in ('int', 'i4'):
            ic = IntColumn(n)
            ic.length = n
            for i in range(n):
                s = self.categories[self.codes_c[i]] if self.codes_c[i] < len(self.categories) else ''
                ic.data_c[i] = int(float(s)) if s else 0
            return ic
        elif tname in ('double', 'float'):
            dc = DoubleColumn(n)
            dc.length = n
            for i in range(n):
                s = self.categories[self.codes_c[i]] if self.codes_c[i] < len(self.categories) else ''
                dc.data_c[i] = float(s) if s else 0.0
            return dc
        elif tname == 'bool':
            bc = BoolColumn(n)
            bc.length = n
            for i in range(n):
                s = self.categories[self.codes_c[i]] if self.codes_c[i] < len(self.categories) else ''
                bc.data_c[i] = 1 if s.lower() in ('true', '1', 'yes') else 0
            return bc
        elif tname in ('str', 'string', 'ucs2'):
            uc = UCS2Column_SoA(n)
            uc.length = n
            for i in range(n):
                s = self.categories[self.codes_c[i]] if self.codes_c[i] < len(self.categories) else ''
                uc._set_item(i, s)
            return uc
        elif tname in ('char256', 'ascii'):
            cc = Char256Column_SoA(n)
            for i in range(n):
                s = self.categories[self.codes_c[i]] if self.codes_c[i] < len(self.categories) else ''
                cc.append(s)
            return cc
        elif tname in ('timestamp', 'date', 'datetime'):
            tc = TimestampColumn(n)
            for i in range(n):
                s = self.categories[self.codes_c[i]] if self.codes_c[i] < len(self.categories) else ''
                if s:
                    tc.append(s)
                else:
                    tc.append(0)
            return tc
        raise TypeError(f"Cannot convert category to '{tname}'")


cdef class _CategoryAccessor:
    cdef CategoryColumn col

    def __cinit__(self, CategoryColumn col):
        self.col = col

    @property
    def codes(self):
        cdef IntColumn res
        res = IntColumn(self.col.capacity)
        res.length = self.col.length
        cdef Py_ssize_t i
        for i in range(self.col.length):
            res.data_c[i] = self.col.codes_c[i]
        return res

    @property
    def categories(self):
        return list(self.col.categories)
