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


# ============================================================
#  Shift helpers
# ============================================================
cdef void _c_shift_right(void* buf, Py_ssize_t pos, Py_ssize_t col_len, Py_ssize_t elem_size) noexcept nogil:
    cdef Py_ssize_t n = col_len - pos
    if n > 0:
        memmove(<char*>buf + (pos + 1) * elem_size,
                <char*>buf + pos * elem_size,
                n * elem_size)


cdef void _c_shift_left(void* buf, Py_ssize_t pos, Py_ssize_t col_len, Py_ssize_t elem_size) noexcept nogil:
    cdef Py_ssize_t n = col_len - pos - 1
    if n > 0:
        memmove(<char*>buf + pos * elem_size,
                <char*>buf + (pos + 1) * elem_size,
                n * elem_size)


# ============================================================
#  Generic column helpers — void* + elem_size
# ============================================================
cdef void* _col_pop_back(void* data_c, Py_ssize_t* length,
                         Py_ssize_t elem_size) noexcept nogil:
    if length[0] == 0:
        return NULL
    length[0] -= 1
    return <char*>data_c + length[0] * elem_size


cdef void _col_pop_front(void* data_c, Py_ssize_t* length,
                         Py_ssize_t elem_size) noexcept nogil:
    if length[0] > 1:
        memmove(data_c, <char*>data_c + elem_size,
                (length[0] - 1) * elem_size)
    length[0] -= 1


cdef void _col_concat(void* data_c, Py_ssize_t* length,
                      void* other_data, Py_ssize_t other_len,
                      Py_ssize_t elem_size) noexcept nogil:
    if other_len > 0:
        memcpy(<char*>data_c + length[0] * elem_size, other_data,
               other_len * elem_size)
    length[0] += other_len


cdef void _col_drop_row(void* data_c, Py_ssize_t* length,
                        Py_ssize_t row_idx,
                        Py_ssize_t elem_size) noexcept nogil:
    cdef Py_ssize_t rem_len = length[0] - row_idx - 1
    if rem_len > 0:
        memmove(<char*>data_c + row_idx * elem_size,
                <char*>data_c + (row_idx + 1) * elem_size,
                rem_len * elem_size)
    length[0] -= 1


cdef void _col_gather(void* dst, void* src,
                      Py_ssize_t* indices, Py_ssize_t n,
                      Py_ssize_t elem_size) noexcept nogil:
    cdef Py_ssize_t i, idx
    for i in range(n):
        idx = indices[i]
        memcpy(<char*>dst + i * elem_size,
               <char*>src + idx * elem_size,
               elem_size)


cdef Py_ssize_t _col_reserve(void** data_c, Py_ssize_t* capacity,
                             Py_ssize_t new_cap, Py_ssize_t old_size,
                             Py_ssize_t elem_size) except -1 nogil:
    if new_cap <= capacity[0]:
        return 0
    cdef void* new_ptr = aligned_realloc(data_c[0], old_size, new_cap * elem_size)
    if not new_ptr:
        return -1
    data_c[0] = new_ptr
    capacity[0] = new_cap
    return 1


# ============================================================
#  Vectorized math: same-type (int/int, double/double, etc.)
# ============================================================
cdef double _vec_pearson_corr(NumericType* x, NumericType* y, Py_ssize_t n) noexcept nogil:
    if n < 2: return NAN
    cdef double sum_x = 0.0, sum_y = 0.0
    cdef double mean_x, mean_y
    cdef double cov = 0.0, var_x = 0.0, var_y = 0.0
    cdef double xi, yi
    cdef double denom
    cdef Py_ssize_t i
    for i in range(n):
        sum_x += <double>x[i]
        sum_y += <double>y[i]
    mean_x = sum_x / <double>n
    mean_y = sum_y / <double>n
    for i in range(n):
        xi = <double>x[i] - mean_x
        yi = <double>y[i] - mean_y
        cov += xi * yi
        var_x += xi * xi
        var_y += yi * yi
    denom = sqrt(var_x * var_y)
    return cov / denom if denom > 1e-15 else NAN


cdef double _vec_pearson_cov(NumericType* x, NumericType* y, Py_ssize_t n) noexcept nogil:
    if n < 2: return NAN
    cdef double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0
    cdef double val_x, val_y
    cdef Py_ssize_t i
    for i in range(n):
        val_x = <double>x[i]
        val_y = <double>y[i]
        sum_x += val_x
        sum_y += val_y
        sum_xy += val_x * val_y
    return (sum_xy - (sum_x * sum_y) / <double>n) / <double>(n - 1.0)


cdef double _vec_euclidean_dist(NumericType* x, NumericType* y, Py_ssize_t n) noexcept nogil:
    if n == 0: return NAN
    cdef double sum_sq = 0.0
    cdef double diff
    cdef Py_ssize_t i
    for i in range(n):
        diff = <double>x[i] - <double>y[i]
        sum_sq += diff * diff
    return sqrt(sum_sq)


cdef double _vec_manhattan_dist(NumericType* x, NumericType* y, Py_ssize_t n) noexcept nogil:
    if n == 0: return NAN
    cdef double sum_abs = 0.0
    cdef Py_ssize_t i
    for i in range(n):
        sum_abs += fabs(<double>x[i] - <double>y[i])
    return sum_abs


cdef double _vec_cosine_sim(NumericType* x, NumericType* y, Py_ssize_t n) noexcept nogil:
    if n == 0: return NAN
    cdef double sum_xx = 0.0, sum_yy = 0.0, sum_xy = 0.0
    cdef double val_x, val_y
    cdef Py_ssize_t i
    for i in range(n):
        val_x = <double>x[i]
        val_y = <double>y[i]
        sum_xx += val_x * val_x
        sum_yy += val_y * val_y
        sum_xy += val_x * val_y
    cdef double denom = sqrt(sum_xx * sum_yy)
    return sum_xy / denom if denom > 0.0 else 0.0


cdef double _vec_jaccard_sim(NumericType* x, NumericType* y, Py_ssize_t n) noexcept nogil:
    if n == 0: return NAN
    cdef Py_ssize_t intersection = 0
    cdef Py_ssize_t union_cnt = 0
    cdef bint bx, by
    cdef Py_ssize_t i
    for i in range(n):
        bx = (x[i] != 0)
        by = (y[i] != 0)
        if bx and by:
            intersection += 1
        if bx or by:
            union_cnt += 1
    return <double>intersection / <double>union_cnt if union_cnt > 0 else 0.0


cdef double _vec_hamming_dist(NumericType* x, NumericType* y, Py_ssize_t n) noexcept nogil:
    if n == 0: return NAN
    cdef Py_ssize_t diffs = 0
    cdef Py_ssize_t i
    for i in range(n):
        if <double>x[i] != <double>y[i]:
            diffs += 1
    return <double>diffs / <double>n


# ============================================================
#  Vectorized math: int* / NumericType cross-type
# ============================================================
cdef double _int_double_pearson_corr(int* x, NumericType* y, Py_ssize_t n) noexcept nogil:
    if n < 2: return NAN
    cdef double sum_x = 0.0, sum_y = 0.0
    cdef double mean_x, mean_y
    cdef double cov = 0.0, var_x = 0.0, var_y = 0.0
    cdef double xi, yi
    cdef double denom
    cdef Py_ssize_t i
    for i in range(n):
        sum_x += <double>x[i]
        sum_y += <double>y[i]
    mean_x = sum_x / <double>n
    mean_y = sum_y / <double>n
    for i in range(n):
        xi = <double>x[i] - mean_x
        yi = <double>y[i] - mean_y
        cov += xi * yi
        var_x += xi * xi
        var_y += yi * yi
    denom = sqrt(var_x * var_y)
    return cov / denom if denom > 1e-15 else NAN


cdef double _int_double_pearson_cov(int* x, NumericType* y, Py_ssize_t n) noexcept nogil:
    if n < 2: return NAN
    cdef double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0
    cdef double val_x, val_y
    cdef Py_ssize_t i
    for i in range(n):
        val_x = <double>x[i]
        val_y = <double>y[i]
        sum_x += val_x
        sum_y += val_y
        sum_xy += val_x * val_y
    return (sum_xy - (sum_x * sum_y) / <double>n) / <double>(n - 1.0)


cdef double _int_double_euclidean_dist(int* x, NumericType* y, Py_ssize_t n) noexcept nogil:
    if n == 0: return NAN
    cdef double sum_sq = 0.0
    cdef double diff
    cdef Py_ssize_t i
    for i in range(n):
        diff = <double>x[i] - <double>y[i]
        sum_sq += diff * diff
    return sqrt(sum_sq)


cdef double _int_double_manhattan_dist(int* x, NumericType* y, Py_ssize_t n) noexcept nogil:
    if n == 0: return NAN
    cdef double sum_abs = 0.0
    cdef Py_ssize_t i
    for i in range(n):
        sum_abs += fabs(<double>x[i] - <double>y[i])
    return sum_abs


cdef double _int_double_cosine_sim(int* x, NumericType* y, Py_ssize_t n) noexcept nogil:
    if n == 0: return NAN
    cdef double sum_xx = 0.0, sum_yy = 0.0, sum_xy = 0.0
    cdef double val_x, val_y
    cdef Py_ssize_t i
    for i in range(n):
        val_x = <double>x[i]
        val_y = <double>y[i]
        sum_xx += val_x * val_x
        sum_yy += val_y * val_y
        sum_xy += val_x * val_y
    cdef double denom = sqrt(sum_xx * sum_yy)
    return sum_xy / denom if denom > 0.0 else 0.0


cdef double _int_double_jaccard_sim(int* x, NumericType* y, Py_ssize_t n) noexcept nogil:
    if n == 0: return NAN
    cdef Py_ssize_t intersection = 0
    cdef Py_ssize_t union_cnt = 0
    cdef bint bx, by
    cdef Py_ssize_t i
    for i in range(n):
        bx = (x[i] != 0)
        by = (y[i] != 0)
        if bx and by:
            intersection += 1
        if bx or by:
            union_cnt += 1
    return <double>intersection / <double>union_cnt if union_cnt > 0 else 0.0


cdef double _int_double_hamming_dist(int* x, NumericType* y, Py_ssize_t n) noexcept nogil:
    if n == 0: return NAN
    cdef Py_ssize_t diffs = 0
    cdef Py_ssize_t i
    for i in range(n):
        if <double>x[i] != <double>y[i]:
            diffs += 1
    return <double>diffs / <double>n


# ============================================================
#  Vectorized math: double* / NumericType cross-type
# ============================================================
cdef double _double_int_pearson_corr(double* x, NumericType* y, Py_ssize_t n) noexcept nogil:
    if n < 2: return NAN
    cdef double sum_x = 0.0, sum_y = 0.0
    cdef double mean_x, mean_y
    cdef double cov = 0.0, var_x = 0.0, var_y = 0.0
    cdef double xi, yi
    cdef double denom
    cdef Py_ssize_t i
    for i in range(n):
        sum_x += x[i]
        sum_y += <double>y[i]
    mean_x = sum_x / <double>n
    mean_y = sum_y / <double>n
    for i in range(n):
        xi = x[i] - mean_x
        yi = <double>y[i] - mean_y
        cov += xi * yi
        var_x += xi * xi
        var_y += yi * yi
    denom = sqrt(var_x * var_y)
    return cov / denom if denom > 1e-15 else NAN


cdef double _double_int_pearson_cov(double* x, NumericType* y, Py_ssize_t n) noexcept nogil:
    if n < 2: return NAN
    cdef double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0
    cdef double val_x, val_y
    cdef Py_ssize_t i
    for i in range(n):
        val_x = <double>x[i]
        val_y = <double>y[i]
        sum_x += val_x
        sum_y += val_y
        sum_xy += val_x * val_y
    return (sum_xy - (sum_x * sum_y) / <double>n) / <double>(n - 1.0)


cdef double _double_int_euclidean_dist(double* x, NumericType* y, Py_ssize_t n) noexcept nogil:
    if n == 0: return NAN
    cdef double sum_sq = 0.0
    cdef double diff
    cdef Py_ssize_t i
    for i in range(n):
        diff = <double>x[i] - <double>y[i]
        sum_sq += diff * diff
    return sqrt(sum_sq)


cdef double _double_int_manhattan_dist(double* x, NumericType* y, Py_ssize_t n) noexcept nogil:
    if n == 0: return NAN
    cdef double sum_abs = 0.0
    cdef Py_ssize_t i
    for i in range(n):
        sum_abs += fabs(<double>x[i] - <double>y[i])
    return sum_abs


cdef double _double_int_cosine_sim(double* x, NumericType* y, Py_ssize_t n) noexcept nogil:
    if n == 0: return NAN
    cdef double sum_xx = 0.0, sum_yy = 0.0, sum_xy = 0.0
    cdef double val_x, val_y
    cdef Py_ssize_t i
    for i in range(n):
        val_x = <double>x[i]
        val_y = <double>y[i]
        sum_xx += val_x * val_x
        sum_yy += val_y * val_y
        sum_xy += val_x * val_y
    cdef double denom = sqrt(sum_xx * sum_yy)
    return sum_xy / denom if denom > 0.0 else 0.0


cdef double _double_int_jaccard_sim(double* x, NumericType* y, Py_ssize_t n) noexcept nogil:
    if n == 0: return NAN
    cdef Py_ssize_t intersection = 0
    cdef Py_ssize_t union_cnt = 0
    cdef bint bx, by
    cdef Py_ssize_t i
    for i in range(n):
        bx = (x[i] != 0)
        by = (y[i] != 0)
        if bx and by:
            intersection += 1
        if bx or by:
            union_cnt += 1
    return <double>intersection / <double>union_cnt if union_cnt > 0 else 0.0


cdef double _double_int_hamming_dist(double* x, NumericType* y, Py_ssize_t n) noexcept nogil:
    if n == 0: return NAN
    cdef Py_ssize_t diffs = 0
    cdef Py_ssize_t i
    for i in range(n):
        if <double>x[i] != <double>y[i]:
            diffs += 1
    return <double>diffs / <double>n
