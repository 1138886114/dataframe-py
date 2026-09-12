# cython: language_level=3
cimport cython
from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memcpy, memmove
from libc.math cimport sqrt, fabs, NAN
from fluidframe.core._base cimport BaseColumn
from fluidframe.core.doublecolumn cimport DoubleColumn

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class IntColumn(BaseColumn):
    cdef int* data_c
    cdef object _base_obj

    cdef void _reserve(self, Py_ssize_t new_cap)
    cdef double _sum_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef double _mean_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef double _min_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef double _max_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef Py_ssize_t _count_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef double _var_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef double _std_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil

    cdef IntColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=*)

    cdef void to_double_buffer(self, double* out, Py_ssize_t n)
    cdef IntColumn gather_with_sentinel(self, Py_ssize_t* indices, Py_ssize_t n, int sentinel)
    cdef Py_ssize_t count_nonzero(self)
    cdef void gather_zero_to(self, Py_ssize_t* out)
    cdef void _central_moments(self, double* out_mean, double* out_m2, double* out_m3, double* out_m4) noexcept nogil
    cdef double _quickselect(self, int* data, Py_ssize_t n, Py_ssize_t k) noexcept nogil

    # 专供 FastDataFrame 内部 C 引擎调用的无锁极速宏
    cdef inline void _max_2d_c(self, double* out_buf) noexcept nogil
    cdef inline void _min_2d_c(self, double* out_buf) noexcept nogil
    cdef inline void _sum_2d_c(self, double* out_buf) noexcept nogil
    cdef inline void _mean_2d_c(self, double* out_buf) noexcept nogil
    cdef inline void _prod_2d_c(self, double* out_buf) noexcept nogil
    cdef inline void _var_2d_c(self, double* out_buf) noexcept nogil
    cdef inline void _std_2d_c(self, double* out_buf) noexcept nogil

    @staticmethod
    cdef IntColumn create_view(int* ptr, Py_ssize_t length, object base_obj)
    @staticmethod
    cdef IntColumn create_view_2d(int* ptr, Py_ssize_t rows, Py_ssize_t cols, object base_obj)

    cdef inline DoubleColumn _make_like_nan_dbl(self)
    cdef IntColumn _cmp_op(self, object val, int op)