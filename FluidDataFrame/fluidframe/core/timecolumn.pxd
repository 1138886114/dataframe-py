# cython: language_level=3
cimport cython
from libc.stdint cimport int32_t
from fluidframe.core._base cimport BaseColumn

cdef int32_t _parse_time_str(str s) except -1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class TimeColumn(BaseColumn):
    cdef int32_t* data_c
    cdef unsigned int _meta        # 4 字节：紧凑打包 cols, _owns_data, is_2d
    cdef object _base_obj
    cdef void _reserve(self, Py_ssize_t new_cap)
    cdef inline int32_t _resolve_time(self, val) except -1
    cdef TimeColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=*)
    @staticmethod
    cdef TimeColumn create_view(int32_t* ptr, Py_ssize_t length, object base_obj)
    @staticmethod
    cdef TimeColumn create_view_2d(int32_t* ptr, Py_ssize_t rows, Py_ssize_t cols, object base_obj)
    cdef inline int get_cols_c(self) noexcept nogil
    cdef inline bint get_owns_data_c(self) noexcept nogil
    cdef inline bint get_is_2d_c(self) noexcept nogil
    cdef inline Py_ssize_t get_rows_c(self) noexcept nogil
    cdef inline void set_meta_c(self, int cols, bint owns_data, bint is_2d) noexcept nogil
    cdef inline void _max_2d_c(self, double* out_buf) noexcept nogil
    cdef inline void _min_2d_c(self, double* out_buf) noexcept nogil
