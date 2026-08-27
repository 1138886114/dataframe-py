# cython: language_level=3
from fluidframe.core.fastdataframe cimport FastDataFrame
from fluidframe.core.intcolumn cimport IntColumn

cdef class _ILocIndexer:
    cdef FastDataFrame frame

cdef class _LocIndexer:
    cdef FastDataFrame frame
    cdef FastDataFrame _apply_mask(self, IntColumn mask)
    cdef FastDataFrame _apply_bool_list(self, list mask)
