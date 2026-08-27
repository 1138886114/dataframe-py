# cython: language_level=3

cdef class BaseColumn:
    cdef public Py_ssize_t length
    cdef public Py_ssize_t capacity


cdef enum DataType:
    TYPE_INT = 0
    TYPE_FLOAT = 1
    TYPE_DOUBLE = 2
    TYPE_CHAR256 = 3
    TYPE_UCS2 = 4
    TYPE_TIMESTAMP = 5


ctypedef fused NumericType:
    int
    double
    long long
