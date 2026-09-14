# cython: language_level=3

from fluidframe.core._base cimport BaseColumn
from fluidframe.core.index cimport IndexBase

cdef class Series:
    cdef public str name
    cdef public BaseColumn _col
    cdef public IndexBase _index
