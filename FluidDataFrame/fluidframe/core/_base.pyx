# cython: language_level=3

from libc.string cimport memcpy

cdef extern from "<stdlib.h>" nogil:
    int posix_memalign(void **memptr, size_t alignment, size_t size)
    void free(void *ptr)


cdef void* aligned_malloc(size_t size, size_t alignment=64) nogil:
    cdef void* ptr = NULL
    if posix_memalign(&ptr, alignment, size) == 0:
        return ptr
    return NULL

cdef void* aligned_realloc(void* ptr, size_t old_size, size_t new_size, size_t alignment=64) nogil:
    if new_size == 0:
        free(ptr)
        return NULL
    cdef void* new_ptr = aligned_malloc(new_size, alignment)
    if new_ptr != NULL and ptr != NULL:
        memcpy(new_ptr, ptr, old_size)
        free(ptr)
    return new_ptr


cdef class BaseColumn:
    pass