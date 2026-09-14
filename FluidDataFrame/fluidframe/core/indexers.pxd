# cython: language_level=3
from fluidframe.core.dataframe cimport DataFrame
from fluidframe.core.intcolumn cimport IntColumn

cdef class _ILocIndexer:
    cdef DataFrame frame

cdef class _LocIndexer:
    cdef DataFrame frame
    cdef DataFrame _apply_mask(self, IntColumn mask)
    cdef DataFrame _apply_bool_list(self, list mask)
    cdef DataFrame _get_rows(self, key)