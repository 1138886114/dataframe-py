# cython: language_level=3
cimport cython
from libc.stdint cimport int32_t
from fluidframe.core._base cimport BaseColumn

cdef int32_t _parse_time_str(str s) except -1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class TimeColumn(BaseColumn):
    cdef int32_t* data_c
    cdef bint _owns_data
    cdef object _base_obj
    cdef void _reserve(self, Py_ssize_t new_cap)
    cdef inline int32_t _resolve_time(self, val) except -1
    cdef TimeColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=*)
    @staticmethod
    cdef TimeColumn create_view(int32_t* ptr, Py_ssize_t length, object base_obj)

