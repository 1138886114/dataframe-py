# cython: language_level=3
cimport cython
from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memcpy, memmove
from libc.math cimport sqrt, fabs, NAN
from fluidframe.core._base cimport BaseColumn
from fluidframe.core.intcolumn cimport IntColumn

cdef int _cmp_double_asc(const void* a, const void* b) noexcept nogil

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class DoubleColumn(BaseColumn):
    cdef public Py_ssize_t length
    cdef public Py_ssize_t capacity
    cdef double* data_c
    cdef unsigned int _meta        # 4 字节：紧凑打包 cols, _owns_data, is_2d
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

    cdef void to_double_buffer(self, double* out, Py_ssize_t n)
    cdef double corr_with(self, double* other, Py_ssize_t n)
    cdef double cov_with(self, double* other, Py_ssize_t n)
    cdef DoubleColumn gather_with_sentinel(self, Py_ssize_t* indices, Py_ssize_t n, double sentinel)
    cdef void _central_moments(self, double* out_mean, double* out_m2, double* out_m3, double* out_m4) noexcept nogil
    cdef double _quickselect(self, double* data, Py_ssize_t n, Py_ssize_t k) noexcept nogil
    @staticmethod
    cdef DoubleColumn create_view(double* ptr, Py_ssize_t length, object base_obj)
    @staticmethod
    cdef DoubleColumn create_view_2d(double* ptr, Py_ssize_t rows, Py_ssize_t cols, object base_obj)
    cdef inline int get_cols_c(self) noexcept nogil
    cdef inline bint get_owns_data_c(self) noexcept nogil
    cdef inline bint get_is_2d_c(self) noexcept nogil
    cdef inline Py_ssize_t get_rows_c(self) noexcept nogil
    cdef inline void set_meta_c(self, int cols, bint owns_data, bint is_2d) noexcept nogil
    cdef inline DoubleColumn _make_like_nan(self)
