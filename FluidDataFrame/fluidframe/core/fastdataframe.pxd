# cython: language_level=3
cimport cython
from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memcpy, memmove, strncmp
from libc.math cimport NAN
from libc.stdint cimport uint64_t
from cpython.unicode cimport Py_UCS2
from fluidframe.core._base cimport BaseColumn, NumericType, DataType
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.doublecolumn cimport DoubleColumn
from fluidframe.core.asciicolumn cimport AsciiColumn
from fluidframe.core.ucs2column cimport UCS2Column
from fluidframe.core.datecolumn cimport DateColumn
from fluidframe.core.boolcolumn cimport BoolColumn
from fluidframe.core.categorycolumn cimport CategoryColumn
from fluidframe.core.datetimecolumn cimport DateTimeColumn
from fluidframe.core.timecolumn cimport TimeColumn
from fluidframe.core.index cimport IndexBase

cdef struct _ByColPtrs:
    void** data
    void** offsets
    int* types
    int count

cdef struct _GroupHashEntry:
    uint64_t hash
    int group_id
    bint occupied

cdef struct _GroupHashTable:
    _GroupHashEntry* entries
    Py_ssize_t capacity
    Py_ssize_t size

cdef struct _GroupRows:
    Py_ssize_t** rows
    Py_ssize_t* row_sizes
    Py_ssize_t* row_capacities
    Py_ssize_t count

cdef void _c_argsort_char256(char* data, Py_ssize_t* offsets, Py_ssize_t* indices, Py_ssize_t left, Py_ssize_t right) noexcept nogil
cdef void _c_argsort_numeric(NumericType* data, Py_ssize_t* indices, Py_ssize_t left, Py_ssize_t right) noexcept nogil
cdef void _c_argsort_ucs2(Py_UCS2* chars, Py_ssize_t* offsets, Py_ssize_t* indices, Py_ssize_t left, Py_ssize_t right) noexcept nogil

@cython.final
@cython.freelist(16)
cdef class _DataFrameMetricAccessor:
    cdef FastDataFrame _df

@cython.final
@cython.freelist(16)
cdef class _DataFrameGroupByAccessor:
    cdef FastDataFrame _df
    cdef list _by_cols
    cdef _GroupHashTable* _ht
    cdef _GroupRows* _gr
    cdef _ByColPtrs _ptrs
    cdef bint _built
    cdef void _ensure_built(self)
    cdef FastDataFrame _convenience_agg(self, str func_name, object col_name)

@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class FastDataFrame:
    cdef public dict columns
    cdef public Py_ssize_t df_length
    cdef public IndexBase index
    cdef _DataFrameMetricAccessor _metric_cache
    cdef _DataFrameGroupByAccessor _groupby_cache
    cdef BaseColumn _reorder_column(self, BaseColumn col, Py_ssize_t* indices, Py_ssize_t n, bint inplace)
    cdef str _type_name(self, BaseColumn col)
    cdef BaseColumn _diff_column(self, BaseColumn col, int periods)
    cdef void _drop_row_column(self, BaseColumn col, Py_ssize_t row_idx)
    cdef FastDataFrame _take(self, Py_ssize_t* indices, Py_ssize_t n)
    cdef BaseColumn _column_from_list(self, str name, list values)
    cdef void _shift_right(self, BaseColumn col, Py_ssize_t pos)
    cdef void _shift_left(self, BaseColumn col, Py_ssize_t pos)
    cdef double _get_col_value_as_double(self, BaseColumn col, Py_ssize_t i)
    cdef void _col_to_double_buffer(self, BaseColumn col, double* out, Py_ssize_t n)
    cdef inline str _fast_type_name(self, int dt)
    cdef int _display_width(self, str s)
    cdef str _truncate_str(self, str s, int max_display_width=*)
    cdef str _format_value_fast(self, BaseColumn col, Py_ssize_t i)
    cdef tuple _get_header_strings(self)
    cdef list _get_row_strings(self, Py_ssize_t i)
