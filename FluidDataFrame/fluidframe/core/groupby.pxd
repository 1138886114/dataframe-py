# cython: language_level=3
from libc.stdint cimport uint64_t
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from fluidframe.core.fastdataframe cimport FastDataFrame
from fluidframe.core._base cimport BaseColumn
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.doublecolumn cimport DoubleColumn

cdef class FastDataFrameGroupBy:
    cdef FastDataFrame df
    cdef list by_cols

    cdef unordered_map[uint64_t, int]* hash_to_group_id
    cdef vector[vector[Py_ssize_t]]* group_rows
    cdef vector[Py_ssize_t]* first_seen_rows

    cdef int num_groups

    cdef void** _by_col_data
    cdef void** _by_col_offsets
    cdef int* _by_col_types
    cdef int _num_by_cols

    cdef void _build_groups(self)
    cdef void _extract_col_ptrs(self)
    cdef uint64_t _hash_row(self, Py_ssize_t row_idx) nogil
    cdef DoubleColumn _native_agg(self, str col_name, str func)
    cdef IntColumn _native_count(self)
    cdef object _python_agg(self, str col_name, str func_name)
    cdef object _extract_key_column(self, str col_name)
    cdef FastDataFrame _convenience_agg(self, str func_name, str col_name=*)
    cdef IntColumn _native_idx(self, str col_name, bint find_max)
