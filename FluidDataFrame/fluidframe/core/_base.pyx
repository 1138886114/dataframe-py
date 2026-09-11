# cython: language_level=3

from libc.string cimport memcpy, memmove, memset
from libc.math cimport sqrt, fabs, NAN
cimport cython
from cpython.unicode cimport PyUnicode_AsUTF8AndSize

# ============================================================
#  Platform-specific aligned allocation (C header injection)
# ============================================================
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


# ============================================================
#  Aligned memory allocation
# ============================================================
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


# ============================================================
#  Base column class
# ============================================================
cdef class BaseColumn:
    def __cinit__(self):
        self.length = 0
        self.capacity = 0
        self.dtype = -1
        self.is_2d = False
        self.num_cols = 1
        self.owns_data = True

    cdef Py_ssize_t _get_rows_c(self) noexcept nogil:
        """底层无锁获取逻辑行数"""
        if self.is_2d and self.num_cols > 1:
            return self.length // self.num_cols
        return self.length

    cdef void* _get_data_ptr(self) noexcept nogil:
        """
        供子类重写：返回底层 C 数组裸指针。
        """
        return NULL

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def get_size(self):
        return 0

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def copy(self):
        raise NotImplementedError("Subclasses must implement copy()")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def drop_row(self, Py_ssize_t row_idx):
        raise NotImplementedError

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def shift_left(self, Py_ssize_t pos):
        raise NotImplementedError


# ============================================================
#  Astype helper
# ============================================================
cdef inline bint _fast_match(const char* s, const char* expected, Py_ssize_t length) noexcept nogil:
    cdef Py_ssize_t i
    for i in range(length):
        if (s[i] | 0x20) != expected[i]:
            return False
    return True

cdef int _resolve_astype_target(object target) except -1:
    if target is int: return TYPE_INT
    if target is float: return TYPE_DOUBLE
    if target is bool: return TYPE_BOOL
    if target is str: return TYPE_UCS2

    cdef Py_ssize_t length
    cdef const char* c_str

    if isinstance(target, str):
        c_str = PyUnicode_AsUTF8AndSize(target, &length)
        if c_str == NULL:
            raise TypeError(f"Invalid string type: {target}")

        if length == 2:
            if _fast_match(c_str, b"i4", 2): return TYPE_INT
            if _fast_match(c_str, b"i8", 2): return TYPE_INT
            if _fast_match(c_str, b"f4", 2): return TYPE_DOUBLE
            if _fast_match(c_str, b"f8", 2): return TYPE_DOUBLE
        elif length == 3:
            if _fast_match(c_str, b"int", 3): return TYPE_INT
            if _fast_match(c_str, b"str", 3): return TYPE_UCS2
            if _fast_match(c_str, b"cat", 3): return TYPE_CATEGORY
        elif length == 4:
            if _fast_match(c_str, b"bool", 4): return TYPE_BOOL
            if _fast_match(c_str, b"date", 4): return TYPE_DATE
            if _fast_match(c_str, b"time", 4): return TYPE_TIME
            if _fast_match(c_str, b"ucs2", 4): return TYPE_UCS2
            if _fast_match(c_str, b"cate", 4): return TYPE_CATEGORY
        elif length == 5:
            if _fast_match(c_str, b"float", 5): return TYPE_DOUBLE
            if _fast_match(c_str, b"int32", 5): return TYPE_INT
            if _fast_match(c_str, b"int64", 5): return TYPE_INT
            if _fast_match(c_str, b"ascii", 5): return TYPE_CHAR256
        elif length == 6:
            if _fast_match(c_str, b"double", 6): return TYPE_DOUBLE
            if _fast_match(c_str, b"string", 6): return TYPE_UCS2
        elif length == 7:
            if _fast_match(c_str, b"integer", 7): return TYPE_INT
            if _fast_match(c_str, b"float64", 7): return TYPE_DOUBLE
            if _fast_match(c_str, b"boolean", 7): return TYPE_BOOL
            if _fast_match(c_str, b"unicode", 7): return TYPE_UCS2
            if _fast_match(c_str, b"char256", 7): return TYPE_CHAR256
        elif length == 8:
            if _fast_match(c_str, b"datetime", 8): return TYPE_DATETIME
            if _fast_match(c_str, b"category", 8): return TYPE_CATEGORY
        elif length == 9:
            if _fast_match(c_str, b"timestamp", 9): return TYPE_DATE

    raise TypeError(f"Cannot cast to unsupported target type: {target}")
