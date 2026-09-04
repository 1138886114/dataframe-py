# cython: language_level=3
cimport cython
from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memcpy, memmove, strstr, strlen, strncmp, strncpy
from fluidframe.core._base cimport BaseColumn
from fluidframe.core.intcolumn cimport IntColumn

cdef bint _c_str_contains(const char* source, const char* target) noexcept nogil
cdef void _c_str_replace_inplace(char* buffer, const char* old_sub, const char* new_sub, size_t old_len, size_t new_len) noexcept nogil

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class Char256Column_SoA(BaseColumn):
    cdef public Py_ssize_t length
    cdef public Py_ssize_t capacity
    cdef char* data_c
    cdef void _reserve(self, Py_ssize_t new_cap)
    cdef inline str _get_str(self, Py_ssize_t idx)
    cdef void _set_item(self, Py_ssize_t idx, str val) except *
    cdef Char256Column_SoA take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=*)

