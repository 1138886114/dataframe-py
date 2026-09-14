# cython: language_level=3

from fluidframe.core._base cimport BaseColumn
from fluidframe.core.index cimport IndexBase
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.asciicolumn cimport AsciiColumn
from fluidframe.core.ucs2column cimport UCS2Column
from fluidframe.core.categorycolumn cimport CategoryColumn

cdef class Series:
    cdef public str name
    cdef public BaseColumn _col
    cdef public IndexBase _index

# Forward declarations for accessors
cdef class _StringAccessor:
    cdef Series _series

cdef class _DateTimeAccessor:
    cdef Series _series

cdef class _CategoryAccessor:
    cdef Series _series
