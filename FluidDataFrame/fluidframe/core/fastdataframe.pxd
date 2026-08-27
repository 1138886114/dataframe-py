# cython: language_level=3
cimport cython
from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memcpy, memmove, strncmp
from libc.math cimport NAN
from cpython.unicode cimport Py_UCS2
from fluidframe.core._base cimport BaseColumn, NumericType, DataType
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.doublecolumn cimport DoubleColumn
from fluidframe.core.char256column cimport Char256Column_SoA
from fluidframe.core.ucs2column cimport UCS2Column_SoA
from fluidframe.core.timestampcolumn cimport TimestampColumn
from fluidframe.core.index cimport IndexBase

cdef void _c_argsort_char256(char* data, Py_ssize_t* indices, Py_ssize_t left, Py_ssize_t right) noexcept nogil
cdef void _c_argsort_numeric(NumericType* data, Py_ssize_t* indices, Py_ssize_t left, Py_ssize_t right) noexcept nogil
cdef void _c_argsort_ucs2(Py_UCS2* chars, Py_ssize_t* offsets, Py_ssize_t* indices, Py_ssize_t left, Py_ssize_t right) noexcept nogil

@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class FastDataFrame:
    cdef public dict columns
    cdef public Py_ssize_t df_length
    cdef public IndexBase index
    cdef IntColumn _reorder_int_column(self, IntColumn col, Py_ssize_t* indices, Py_ssize_t n, bint inplace)
    cdef DoubleColumn _reorder_double_column(self, DoubleColumn col, Py_ssize_t* indices, Py_ssize_t n, bint inplace)
    cdef Char256Column_SoA _reorder_char256_column(self, Char256Column_SoA col, Py_ssize_t* indices, Py_ssize_t n, bint inplace)
    cdef UCS2Column_SoA _reorder_ucs2_column(self, UCS2Column_SoA col, Py_ssize_t* indices, Py_ssize_t n, bint inplace)
    cdef TimestampColumn _reorder_timestamp_column(self, TimestampColumn col, Py_ssize_t* indices, Py_ssize_t n, bint inplace)
    cdef DoubleColumn _diff_double(self, DoubleColumn col, int periods)
    cdef IntColumn _diff_int(self, IntColumn col, int periods)
    cdef void _drop_row_ucs2(self, UCS2Column_SoA col, Py_ssize_t row_idx, Py_ssize_t rem_len)
    cdef FastDataFrame _take(self, Py_ssize_t* indices, Py_ssize_t n)
