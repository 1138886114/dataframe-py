# cython: language_level=3
cimport cython
from fluidframe.core._base cimport BaseColumn
from fluidframe.core.doublecolumn cimport DoubleColumn

@cython.final
cdef class RollingWindow:
    cdef double* data_c
    cdef Py_ssize_t length
    cdef Py_ssize_t window
    cdef bint _owns_data
    cdef object _base_obj
    cdef DoubleColumn _make_result(self)
