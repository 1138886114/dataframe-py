# distutils: language = c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

cimport cython
from libc.stdlib cimport malloc, free
from fluidframe.core._base cimport aligned_malloc
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.index cimport IndexBase, RangeIndex, MultiIndex
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector


cdef class IndexBase:
    cdef IndexBase _take(self, Py_ssize_t* indices, Py_ssize_t n):
        raise NotImplementedError()

    cdef IndexBase _concat(self, IndexBase other):
        raise NotImplementedError()

    cpdef get_loc(self, object key):
        raise NotImplementedError()

cdef class RangeIndex(IndexBase):
    def __cinit__(self, Py_ssize_t start, Py_ssize_t length, Py_ssize_t step=1):
        self.start = start
        self.length = length
        self.step = step
        self._values = [start + i * step for i in range(length)]

    cdef IndexBase _take(self, Py_ssize_t* indices, Py_ssize_t n):
        cdef list new_vals = []
        cdef Py_ssize_t i
        for i in range(n):
            new_vals.append(self.start + indices[i] * self.step)
        cdef RangeIndex result = RangeIndex(0, 0)
        result._values = new_vals
        result.length = n
        return result

    cpdef get_loc(self, object key):
        cdef Py_ssize_t i
        for i in range(self.length):
            if self._values[i] == key:
                return i
        raise KeyError(key)

cdef class MultiIndex(IndexBase):
    def __cinit__(self, list levels, list codes, list names=None):
        self.levels = levels
        self.codes = codes  # 这是包含多个 IntColumn 的列表
        self.nlevels = len(levels)
        self.names = names if names else [None] * self.nlevels
        if self.nlevels > 0:
            self.length = (<IntColumn>codes[0]).length
        else:
            self.length = 0

    cdef IndexBase _take(self, Py_ssize_t* indices, Py_ssize_t n):
        cdef list new_codes = []
        cdef IntColumn col, new_col
        cdef Py_ssize_t i
        for col in self.codes:
            new_col = IntColumn(n)
            new_col.length = n
            for i in range(n):
                new_col.data_c[i] = col.data_c[indices[i]]
            new_codes.append(new_col)
        return MultiIndex(self.levels, new_codes, self.names)

    cpdef get_loc(self, object key):
        cdef tuple target
        if not isinstance(key, tuple):
            target = (key,)
        else:
            target = <tuple>key
        if len(target) != self.nlevels:
            raise KeyError(key)
        cdef list matches = []
        cdef Py_ssize_t i, j
        cdef bint match
        cdef IntColumn col
        for i in range(self.length):
            match = True
            for j in range(self.nlevels):
                col = <IntColumn>self.codes[j]
                if col.data_c[i] != target[j]:
                    match = False
                    break
            if match:
                matches.append(i)
        if len(matches) == 0:
            raise KeyError(key)
        if len(matches) == 1:
            return matches[0]
        return matches