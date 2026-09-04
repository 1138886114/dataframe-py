# cython: language_level=3
cimport cython
from libc.stdint cimport int64_t, int32_t
from fluidframe.core._base cimport BaseColumn

cdef int64_t _parse_datetime_str(str s) except -1
cdef void _datetime_to_ymdhms(int64_t us, int* y, int* m, int* d, int* h, int* mi, int* s) noexcept nogil

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class DateTimeColumn(BaseColumn):
    cdef public Py_ssize_t length
    cdef public Py_ssize_t capacity
    cdef int64_t* data_c
    cdef void _reserve(self, Py_ssize_t new_cap)
    cdef DateTimeColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=*)
    cpdef void drop_row(self, Py_ssize_t row_idx)
    cpdef void shift_left(self, Py_ssize_t pos)
    cdef inline int64_t _resolve_dt(self, val) except *
