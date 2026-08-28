# cython: language_level=3
cimport cython
from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memcpy, memmove
from fluidframe.core._base cimport BaseColumn
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.char256column cimport Char256Column_SoA

cdef inline bint _is_leap_year(int y) noexcept nogil
cdef inline int _days_in_month(int y, int m) noexcept nogil
cdef inline long long _date_to_daynum(int y, int m, int d) noexcept nogil
cdef inline void _daynum_to_date(long long daynum, int* y, int* m, int* d) noexcept nogil
cdef inline long long _parse_date_str(str s) noexcept

@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class TimestampColumn(BaseColumn):
    cdef public Py_ssize_t length
    cdef public Py_ssize_t capacity
    cdef long long* data_c
    cdef void _reserve(self, Py_ssize_t new_cap)
    cdef inline long long _parse(self, str s) noexcept
    cdef inline long long _resolve_date(self, date) except *
    cdef TimestampColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=*)
    cpdef void drop_row(self, Py_ssize_t row_idx, Py_ssize_t rem_len)
