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
    cdef double* data_c
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

    cdef void to_double_buffer(self, double* out, Py_ssize_t n)
    cdef DoubleColumn gather_with_sentinel(self, Py_ssize_t* indices, Py_ssize_t n, double sentinel)
    cdef void _central_moments(self, double* out_mean, double* out_m2, double* out_m3, double* out_m4) noexcept nogil
    cdef double _quickselect(self, double* data, Py_ssize_t n, Py_ssize_t k) noexcept nogil
    cdef double quantile_per_subcol(self, double* col_ptr, Py_ssize_t n, double q) noexcept
    @staticmethod
    cdef DoubleColumn create_view(double* ptr, Py_ssize_t length, object base_obj)
    cdef inline DoubleColumn _make_like_nan(self)
    cdef IntColumn _cmp_op(self, object val, int op)
