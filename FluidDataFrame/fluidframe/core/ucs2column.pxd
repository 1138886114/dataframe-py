# cython: language_level=3
cimport cython
from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memcpy, memmove
from cpython.unicode cimport (
    PyUnicode_4BYTE_KIND, PyUnicode_FromKindAndData,
    PyUnicode_GET_LENGTH, PyUnicode_READ_CHAR,
    PyUnicode_KIND, PyUnicode_2BYTE_KIND, PyUnicode_DATA,
    Py_UCS2
)
from fluidframe.core._base cimport BaseColumn
from fluidframe.core.intcolumn cimport IntColumn

cdef int _ucs2_len(Py_UCS2* string) noexcept nogil
cdef Py_UCS2* _ucs2_strstr(Py_UCS2* haystack, int h_len, Py_UCS2* needle, int n_len) noexcept nogil

@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class UCS2Column_SoA(BaseColumn):
    cdef public Py_ssize_t length
    cdef public Py_ssize_t capacity
    cdef Py_UCS2* chars_c
    cdef Py_ssize_t* offsets_c
    cdef public Py_ssize_t char_capacity
    cdef public Py_ssize_t char_length
    cdef void _reserve_rows(self, Py_ssize_t new_cap) except *
    cdef void _reserve_chars(self, Py_ssize_t new_char_cap) except *
    cdef inline void _fallback_copy_chars(self, str val, Py_ssize_t str_len)
    cdef UCS2Column_SoA take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=*)
    cpdef void drop_row(self, Py_ssize_t row_idx, Py_ssize_t rem_len)
