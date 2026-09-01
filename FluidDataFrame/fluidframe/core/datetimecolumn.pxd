# cython: language_level=3
cimport cython
from fluidframe.core._base cimport BaseColumn

cdef long long _parse_datetime_str(str s) except -1
cdef void _datetime_to_ymdhms(long long us, int* y, int* m, int* d, int* h, int* mi, int* s) noexcept nogil

@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class DateTimeColumn(BaseColumn):
    cdef public Py_ssize_t length
    cdef public Py_ssize_t capacity
    cdef long long* data_c
    cdef void _reserve(self, Py_ssize_t new_cap)
    cdef DateTimeColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=*)
    cpdef void drop_row(self, Py_ssize_t row_idx, Py_ssize_t rem_len)
    cdef inline long long _resolve_dt(self, val) except *
