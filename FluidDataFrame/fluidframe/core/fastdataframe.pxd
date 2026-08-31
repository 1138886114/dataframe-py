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
from fluidframe.core.timestampcolumn cimport DateColumn
from fluidframe.core.boolcolumn cimport BoolColumn
from fluidframe.core.categorycolumn cimport CategoryColumn
from fluidframe.core.datetimecolumn cimport DateTimeColumn
from fluidframe.core.timecolumn cimport TimeColumn
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
    cdef BaseColumn _reorder_column(self, BaseColumn col, Py_ssize_t* indices, Py_ssize_t n, bint inplace)
    cdef str _type_name(self, BaseColumn col)
    cdef BaseColumn _diff_column(self, BaseColumn col, int periods)
    cdef void _drop_row_column(self, BaseColumn col, Py_ssize_t row_idx, Py_ssize_t rem_len)
    cdef FastDataFrame _take(self, Py_ssize_t* indices, Py_ssize_t n)
    cdef BaseColumn _column_from_list(self, str name, list values)
    cdef void _shift_right(self, BaseColumn col, Py_ssize_t pos)
    cdef void _shift_left(self, BaseColumn col, Py_ssize_t pos)
    cdef double _get_col_value_as_double(self, BaseColumn col, Py_ssize_t i)
    cdef void _col_to_double_buffer(self, BaseColumn col, double* out, Py_ssize_t n)
    cdef Py_ssize_t _col_bytes(self, BaseColumn col)
    cdef str _dtype_summary(self)
