# cython: language_level=3
cimport cython
from fluidframe.core._base cimport BaseColumn

cdef int _parse_time_str(str s)

@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class TimeColumn(BaseColumn):
    cdef public Py_ssize_t length
    cdef public Py_ssize_t capacity
    cdef int* data_c
    cdef void _reserve(self, Py_ssize_t new_cap)
    cdef TimeColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=*)
    cpdef void drop_row(self, Py_ssize_t row_idx, Py_ssize_t rem_len)
