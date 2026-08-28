# cython: language_level=3

cdef void* aligned_malloc(size_t size, size_t alignment=*) nogil
cdef void* aligned_realloc(void* ptr, size_t old_size, size_t new_size, size_t alignment=*) nogil

cdef class BaseColumn:
    cdef public Py_ssize_t length
    cdef public Py_ssize_t capacity
    cdef public DataType dtype


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
