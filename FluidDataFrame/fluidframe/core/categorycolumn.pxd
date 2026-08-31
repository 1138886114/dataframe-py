# cython: language_level=3
cimport cython
from fluidframe.core._base cimport BaseColumn

@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class CategoryColumn(BaseColumn):
    cdef public Py_ssize_t length
    cdef public Py_ssize_t capacity
    cdef int* codes_c
    cdef list categories
    cdef dict _cat_to_code
    cdef void _reserve(self, Py_ssize_t new_cap)
    cdef int _get_or_add_code(self, str val)
    cdef str _resolve(self, Py_ssize_t i)
    cdef CategoryColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=*)
    cpdef void drop_row(self, Py_ssize_t row_idx, Py_ssize_t rem_len)
