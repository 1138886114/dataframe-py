# cython: language_level=3
cimport cython
from fluidframe.core._base cimport BaseColumn

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class BoolColumn(BaseColumn):
    cdef unsigned char* data_c
    cdef void _reserve(self, Py_ssize_t new_cap)
    cdef BoolColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=*)

