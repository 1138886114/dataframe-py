# cython: language_level=3
cimport cython
from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memcpy, memmove
from libc.math cimport sqrt, fabs, NAN
from fluidframe.core._base cimport BaseColumn
from fluidframe.core.intcolumn cimport IntColumn

cdef double _c_corr(double* a, double* b, Py_ssize_t n) noexcept nogil
cdef double _c_cov(double* a, double* b, Py_ssize_t n) noexcept nogil
cdef double _c_euclidean_dist(double* a, double* b, Py_ssize_t n) noexcept nogil
cdef double _c_manhattan_dist(double* a, double* b, Py_ssize_t n) noexcept nogil
cdef double _c_cosine_dist(double* a, double* b, Py_ssize_t n) noexcept nogil
cdef double _c_cosine_similarity(double* a, double* b, Py_ssize_t n) noexcept nogil
cdef double _c_jaccard_sim(double* a, double* b, Py_ssize_t n) noexcept nogil
cdef double _c_hamming_dist(double* a, double* b, Py_ssize_t n) noexcept nogil

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class DoubleColumn(BaseColumn):
    cdef public Py_ssize_t length
    cdef public Py_ssize_t capacity
    cdef double* data_c
    cdef bint _owns_data
    cdef object _base_obj
    cdef void _reserve(self, Py_ssize_t new_cap)
    cdef double _sum_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef double _mean_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef double _min_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef double _max_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef Py_ssize_t _count_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef double _var_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef double _std_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef DoubleColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=*)
    cpdef DoubleColumn clip(self, double lower, double upper, bint inplace=*)
    cpdef DoubleColumn where(self, IntColumn cond, double other, bint inplace=*)
    cpdef DoubleColumn mask(self, IntColumn cond, double other, bint inplace=*)
    cpdef void drop_row(self, Py_ssize_t row_idx, Py_ssize_t rem_len)
    cdef void to_double_buffer(self, double* out, Py_ssize_t n)
    cdef double corr_with(self, double* other, Py_ssize_t n)
    cdef double cov_with(self, double* other, Py_ssize_t n)
    cdef DoubleColumn gather_with_sentinel(self, Py_ssize_t* indices, Py_ssize_t n, double sentinel)
    cdef void _central_moments(self, double* out_mean, double* out_m2, double* out_m3, double* out_m4) noexcept nogil
    cdef double _quickselect(self, double* data, Py_ssize_t n, Py_ssize_t k) noexcept nogil
    @staticmethod
    cdef DoubleColumn create_view(double* ptr, Py_ssize_t length, object base_obj)
    cdef double _sum_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef double _mean_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef double _min_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef double _max_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef Py_ssize_t _count_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef double _var_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef double _std_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef DoubleColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=*)
    cpdef DoubleColumn clip(self, double lower, double upper, bint inplace=*)
    cpdef DoubleColumn where(self, IntColumn cond, double other, bint inplace=*)
    cpdef DoubleColumn mask(self, IntColumn cond, double other, bint inplace=*)
    cpdef void drop_row(self, Py_ssize_t row_idx, Py_ssize_t rem_len)
    cdef void to_double_buffer(self, double* out, Py_ssize_t n)
    cdef double corr_with(self, double* other, Py_ssize_t n)
    cdef double cov_with(self, double* other, Py_ssize_t n)
    cdef DoubleColumn gather_with_sentinel(self, Py_ssize_t* indices, Py_ssize_t n, double sentinel)
    cdef void _central_moments(self, double* out_mean, double* out_m2, double* out_m3, double* out_m4) noexcept nogil
    cdef double _quickselect(self, double* data, Py_ssize_t n, Py_ssize_t k) noexcept nogil
