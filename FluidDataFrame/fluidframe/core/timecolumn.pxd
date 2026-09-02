# cython: language_level=3
cimport cython
from libc.stdint cimport int32_t
from fluidframe.core._base cimport BaseColumn

cdef int32_t _parse_time_str(str s) except -1

@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class TimeColumn(BaseColumn):
    cdef public Py_ssize_t length
    cdef public Py_ssize_t capacity
    cdef int32_t* data_c
    cdef Py_ssize_t get_size(self)
    cdef void _reserve(self, Py_ssize_t new_cap)
    cdef inline int32_t _resolve_time(self, val) except -1
    cdef TimeColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=*)
    cpdef void drop_row(self, Py_ssize_t row_idx, Py_ssize_t rem_len)
