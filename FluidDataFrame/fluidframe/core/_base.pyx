# cython: language_level=3

from libc.string cimport memcpy

cdef extern from *:
    """
    #include <stdlib.h>

    #if defined(_MSC_VER) || defined(_WIN32)
    #include <malloc.h>
    static inline void* os_aligned_malloc(size_t size, size_t alignment) {
        return _aligned_malloc(size, alignment);
    }
    static inline void os_aligned_free(void* ptr) {
        _aligned_free(ptr);
    }
    #else
    static inline void* os_aligned_malloc(size_t size, size_t alignment) {
        void* ptr = NULL;
        if (posix_memalign(&ptr, alignment, size) == 0) {
            return ptr;
        }
        return NULL;
    }
    static inline void os_aligned_free(void* ptr) {
        free(ptr);
    }
    #endif
    """
    void* os_aligned_malloc(size_t size, size_t alignment) nogil
    void os_aligned_free(void* ptr) nogil

cdef extern from "<stdlib.h>" nogil:
    int posix_memalign(void **memptr, size_t alignment, size_t size)
    void free(void *ptr)


cdef void* aligned_malloc(size_t size, size_t alignment=64) nogil:
    if size == 0:
        return NULL
    cdef size_t ptr_size = sizeof(void*)
    if alignment < ptr_size:
        alignment = ptr_size
    if (alignment & (alignment - 1)) != 0:
        return NULL
    return os_aligned_malloc(size, alignment)


cdef void* aligned_realloc(void* ptr, size_t old_size, size_t new_size, size_t alignment=64) nogil:
    if new_size == 0:
        if ptr != NULL:
            os_aligned_free(ptr)
        return NULL
    if ptr != NULL and new_size == old_size and (<size_t>ptr % alignment) == 0:
        return ptr

    cdef void* new_ptr = aligned_malloc(new_size, alignment)
    if new_ptr != NULL and ptr != NULL:
        if old_size < new_size:
            memcpy(new_ptr, ptr, old_size)
        else:
            memcpy(new_ptr, ptr, new_size)
        os_aligned_free(ptr)
    return new_ptr


cdef class BaseColumn:
    pass


cdef inline int _resolve_astype_target(object target) except -1:
    cdef str tname
    if isinstance(target, str):
        tname = target.lower()
        if tname in ('int', 'i4', 'i8', 'int32', 'int64', 'integer'): return TYPE_INT
        if tname in ('double', 'float', 'f8', 'f4', 'float64'): return TYPE_DOUBLE
        if tname in ('bool', 'boolean'): return TYPE_BOOL
        if tname in ('str', 'string', 'ucs2', 'unicode'): return TYPE_UCS2
        if tname in ('char256', 'ascii'): return TYPE_CHAR256
        if tname in ('timestamp', 'date'): return TYPE_TIMESTAMP
        if tname == 'datetime': return TYPE_DATETIME
        if tname == 'time': return TYPE_TIME
        if tname in ('cat', 'category'): return TYPE_CATEGORY
    else:
        if target is int: return TYPE_INT
        if target is float: return TYPE_DOUBLE
        if target is bool: return TYPE_BOOL
        if target is str: return TYPE_UCS2

    raise TypeError(f"Cannot cast to unsupported target type: {target}")