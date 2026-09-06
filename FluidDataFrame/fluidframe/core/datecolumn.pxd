# cython: language_level=3
cimport cython
from libc.stdint cimport int32_t
from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memcpy, memmove
from fluidframe.core._base cimport BaseColumn
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.asciicolumn cimport AsciiColumn

cdef bint _is_leap_year(int y) noexcept nogil
cdef int _days_in_month(int y, int m) noexcept nogil
cdef int32_t _ymd_to_days(int y, int m, int d) noexcept nogil
cdef void _days_to_ymd(int32_t days, int* y, int* m, int* d) noexcept nogil
cdef int32_t _parse_date_str(str s) noexcept

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class DateColumn(BaseColumn):
    cdef int32_t* data_c
    cdef void _reserve(self, Py_ssize_t new_cap)
    cdef inline int32_t _resolve_date(self, date) except *
    cdef inline int32_t _to_ymd_int(self, int32_t epoch_days) noexcept nogil
    cdef inline int _iso_week_of_year(self, int32_t days, int* iso_y) noexcept nogil
    cdef DateColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=*)

