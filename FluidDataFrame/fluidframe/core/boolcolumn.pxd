# cython: language_level=3
cimport cython
from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memcpy, memmove
from fluidframe.core._base cimport BaseColumn

@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class BoolColumn(BaseColumn):
    cdef public Py_ssize_t length
    cdef public Py_ssize_t capacity
    cdef unsigned char* data_c
    cdef void _reserve(self, Py_ssize_t new_cap)
    cdef BoolColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=*)
    cpdef void drop_row(self, Py_ssize_t row_idx, Py_ssize_t rem_len)
