# cython: language_level=3
cimport cython
from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memcpy, memmove
from libc.math cimport sqrt, fabs, NAN
from fluidframe.core._base cimport BaseColumn

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class IntColumn(BaseColumn):
    cdef public Py_ssize_t length
    cdef public Py_ssize_t capacity
    cdef int* data_c
    cdef Py_ssize_t get_size(self)
    cdef void _reserve(self, Py_ssize_t new_cap)
    cdef double _sum_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef double _mean_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef double _min_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef double _max_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef Py_ssize_t _count_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef double _var_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef double _std_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef IntColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=*)
    cpdef IntColumn clip(self, double lower, double upper, bint inplace=*)
    cpdef IntColumn where(self, IntColumn cond, int other, bint inplace=*)
    cpdef IntColumn mask(self, IntColumn cond, int other, bint inplace=*)
    cpdef void drop_row(self, Py_ssize_t row_idx, Py_ssize_t rem_len)
    cdef void to_double_buffer(self, double* out, Py_ssize_t n)
    cdef IntColumn gather_with_sentinel(self, Py_ssize_t* indices, Py_ssize_t n, int sentinel)
    cpdef Py_ssize_t count_nonzero(self)
    cdef void gather_zero_to(self, Py_ssize_t* out)
    cdef void _central_moments(self, double* out_mean, double* out_m2, double* out_m3, double* out_m4) noexcept nogil
    cdef double _quickselect(self, int* data, Py_ssize_t n, Py_ssize_t k) noexcept nogil
