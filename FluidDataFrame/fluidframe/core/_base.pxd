# cython: language_level=3

cdef void* aligned_malloc(size_t size, size_t alignment=*) nogil
cdef void* aligned_realloc(void* ptr, size_t old_size, size_t new_size, size_t alignment=*) nogil
cdef void os_aligned_free(void* ptr) nogil

cdef enum DataType:
    TYPE_INT = 0
    TYPE_FLOAT = 1
    TYPE_DOUBLE = 2
    TYPE_CHAR256 = 3
    TYPE_UCS2 = 4
    TYPE_TIMESTAMP = 5
    TYPE_BOOL = 6
    TYPE_CATEGORY = 7
    TYPE_DATETIME = 8
    TYPE_TIME = 9

cdef int _resolve_astype_target(object target) except -1

cdef class BaseColumn:
    cdef public Py_ssize_t length
    cdef public Py_ssize_t capacity
    cdef public int dtype
    cdef Py_ssize_t get_size(self)


ctypedef fused NumericType:
    int
    long
    double
    long long
