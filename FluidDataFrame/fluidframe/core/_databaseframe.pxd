from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memcpy, memmove, strncmp
from libc.math cimport NAN
from libc.stdint cimport uint32_t, uint64_t
from cpython.unicode cimport Py_UCS2
cimport cython

from fluidframe.core._columns cimport (
    BaseColumn, NumericType,
    IntColumn, DoubleColumn, AsciiColumn, UCS2Column, DateColumn,
    BoolColumn, CategoryColumn, DateTimeColumn, TimeColumn
)



cdef enum IndexType:
    INDEX_RANGE = 0
    INDEX_INT = 1
    INDEX_MULTI = 2

cdef class IndexBase:
    cdef public Py_ssize_t length
    cdef public IndexType index_type

    cdef IndexBase _take(self, Py_ssize_t* indices, Py_ssize_t n)
    cdef IndexBase _concat(self, IndexBase other)


cdef class IntIndex(IndexBase):
    cdef public IntColumn data

cdef class RangeIndex(IndexBase):
    cdef public Py_ssize_t start
    cdef public Py_ssize_t step

cdef class MultiIndex(IndexBase):
    cdef public list levels          # 每层的唯一值 (list of tuples/strings)
    cdef public list codes           # [IntColumn, IntColumn...] 重用 IntColumn
    cdef public int nlevels
    cdef public list names           # 每层的名字
    cdef public dict _hash_cache     # 用于 O(1) Full-Key 查找 (bytes -> list[int])
    cdef public list _level_maps     # 反向映射字典列表用于 O(1) 编码查找
    cdef void _build_level_maps(self)
    cdef void _build_hash_cache(self)

cdef class _ILocIndexer:
    cdef DataFrame frame

cdef class _LocIndexer:
    cdef DataFrame frame
    cdef DataFrame _apply_mask(self, object mask)
    cdef DataFrame _apply_bool_list(self, list mask)
    cdef DataFrame _get_rows(self, key)


cdef object _wrap_series(str name, object res, IndexBase index)
cdef BaseColumn _col_take(BaseColumn col, Py_ssize_t* indices, Py_ssize_t n, bint inplace)
cdef BaseColumn _column_from_scalar(DataFrame df, str name, object val, Py_ssize_t n)


cdef class Series:
    cdef public str name
    cdef public BaseColumn _col
    cdef public IndexBase _index

# Forward declarations for accessors
cdef class _StringAccessor:
    cdef Series _series
    cdef object _wrap(self, object res)

cdef class _DateTimeAccessor:
    cdef Series _series

cdef class _CategoryAccessor:
    cdef Series _series

cdef _col_row_key(BaseColumn col, Py_ssize_t row)

cdef struct _ByColPtrs:
    void** data
    void** offsets
    int* types
    int* cols
    Py_ssize_t* strides
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

cdef struct ColMeta:
    int dtype
    void* data_ptr
    void* col_obj
    Py_ssize_t* len_ptr


cdef void _c_argsort_char256(char* data, Py_ssize_t* offsets, Py_ssize_t* indices, Py_ssize_t left, Py_ssize_t right) noexcept nogil
cdef void _c_argsort_numeric(NumericType* data, Py_ssize_t* indices, Py_ssize_t left, Py_ssize_t right) noexcept nogil
cdef void _c_argsort_ucs2(Py_UCS2* chars, Py_ssize_t* offsets, Py_ssize_t* indices, Py_ssize_t left, Py_ssize_t right) noexcept nogil

@cython.final
@cython.freelist(16)
cdef class _DataFrameMetricAccessor:
    cdef DataFrame _df

@cython.final
@cython.freelist(16)
cdef class _DataFrameGroupByAccessor:
    cdef DataFrame _df
    cdef list _by_cols
    cdef _GroupHashTable* _ht
    cdef _GroupRows* _gr
    cdef _ByColPtrs _ptrs
    cdef bint _built
    cdef void _ensure_built(self)
    cdef DataFrame _convenience_agg(self, str func_name, object col_name)

@cython.final
@cython.freelist(16)
cdef class _DataFrameLinalgAccessor:
    cdef DataFrame _df

@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class DataFrame:
    cdef public dict _schema
    cdef public tuple _schema_keys
    cdef public list _ordered_cols
    cdef public list _2d_blocks
    cdef public dict _columns
    cdef public uint32_t _fixed_capacity
    cdef public uint32_t df_length
    cdef bint _is_sealed         # 明确标识是否已锁定内存
    cdef uint32_t _num_cols
    cdef ColMeta* _c_cols

    cdef public IndexBase index

    cdef _DataFrameMetricAccessor _metric_cache
    cdef _DataFrameGroupByAccessor _groupby_cache
    cdef _DataFrameLinalgAccessor _linalg_cache
    cdef BaseColumn _reorder_column(self, BaseColumn col, Py_ssize_t* indices, Py_ssize_t n, bint inplace)
    cdef str _type_name(self, BaseColumn col)
    cdef BaseColumn _diff_column(self, BaseColumn col, int periods)
    cdef void _drop_row_column(self, BaseColumn col, Py_ssize_t row_idx)
    cdef void _rollback_pending_writes(self, uint32_t n_cols) noexcept
    cdef void _check_assign_length(self, str name, BaseColumn new_col) except *
    cdef DataFrame _take(self, Py_ssize_t* indices, Py_ssize_t n)
    cdef DataFrame _slice_rows(self, object key)
    cdef BaseColumn _column_from_list(self, str name, list values)
    cdef void _shift_right(self, BaseColumn col, Py_ssize_t pos)
    cdef void _shift_left(self, BaseColumn col, Py_ssize_t pos)
    cdef double _get_col_value_as_double(self, BaseColumn col, Py_ssize_t i)
    cdef void _col_to_double_buffer(self, BaseColumn col, double* out, Py_ssize_t n, Py_ssize_t sub=*)
    cdef inline str _fast_type_name(self, int dt)
    cdef int _display_width(self, str s)
    cdef str _truncate_str(self, str s, int max_display_width=*)
    cdef str _format_value_fast(self, BaseColumn col, Py_ssize_t i)
    cdef tuple _get_header_strings(self)
    cdef list _get_row_strings(self, Py_ssize_t i)
