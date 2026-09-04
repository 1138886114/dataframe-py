# cython: language_level=3
from libc.math cimport sqrt, fabs, NAN
from libc.string cimport memmove

cdef void* aligned_malloc(size_t size, size_t alignment=*) nogil
cdef void* aligned_realloc(void* ptr, size_t old_size, size_t new_size, size_t alignment=*) nogil
cdef void os_aligned_free(void* ptr) nogil
cdef int _resolve_astype_target(object target) except -1

cdef enum DataType:
    TYPE_INT = 0
    TYPE_FLOAT = 1
    TYPE_DOUBLE = 2
    TYPE_CHAR256 = 3
    TYPE_UCS2 = 4
    TYPE_DATE = 5
    TYPE_BOOL = 6
    TYPE_CATEGORY = 7
    TYPE_DATETIME = 8
    TYPE_TIME = 9



cdef class BaseColumn:
    cdef public Py_ssize_t length
    cdef public Py_ssize_t capacity
    cdef public int dtype
    cpdef void drop_row(self, Py_ssize_t row_idx)
    cpdef void shift_left(self, Py_ssize_t pos)


cdef inline void _c_shift_right(void* buf, Py_ssize_t pos, Py_ssize_t col_len, Py_ssize_t elem_size) noexcept nogil:
    cdef Py_ssize_t n = col_len - pos
    if n > 0:
        memmove(<char*>buf + (pos + 1) * elem_size,
                <char*>buf + pos * elem_size,
                n * elem_size)


cdef inline void _c_shift_left(void* buf, Py_ssize_t pos, Py_ssize_t col_len, Py_ssize_t elem_size) noexcept nogil:
    cdef Py_ssize_t n = col_len - pos - 1
    if n > 0:
        memmove(<char*>buf + pos * elem_size,
                <char*>buf + (pos + 1) * elem_size,
                n * elem_size)


ctypedef fused NumericType:
    int
    long
    double
    long long


cdef inline double _vec_pearson_corr(NumericType* x, NumericType* y, Py_ssize_t n) noexcept nogil:
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


cdef inline double _vec_pearson_cov(NumericType* x, NumericType* y, Py_ssize_t n) noexcept nogil:
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


cdef inline double _vec_euclidean_dist(NumericType* x, NumericType* y, Py_ssize_t n) noexcept nogil:
    if n == 0: return NAN
    cdef double sum_sq = 0.0
    cdef double diff
    cdef Py_ssize_t i
    for i in range(n):
        diff = <double>x[i] - <double>y[i]
        sum_sq += diff * diff
    return sqrt(sum_sq)


cdef inline double _vec_manhattan_dist(NumericType* x, NumericType* y, Py_ssize_t n) noexcept nogil:
    if n == 0: return NAN
    cdef double sum_abs = 0.0
    cdef Py_ssize_t i
    for i in range(n):
        sum_abs += fabs(<double>x[i] - <double>y[i])
    return sum_abs


cdef inline double _vec_cosine_sim(NumericType* x, NumericType* y, Py_ssize_t n) noexcept nogil:
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


cdef inline double _vec_jaccard_sim(NumericType* x, NumericType* y, Py_ssize_t n) noexcept nogil:
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


cdef inline double _vec_hamming_dist(NumericType* x, NumericType* y, Py_ssize_t n) noexcept nogil:
    if n == 0: return NAN
    cdef Py_ssize_t diffs = 0
    cdef Py_ssize_t i
    for i in range(n):
        if <double>x[i] != <double>y[i]:
            diffs += 1
    return <double>diffs / <double>n


cdef inline double _int_double_pearson_corr(int* x, NumericType* y, Py_ssize_t n) noexcept nogil:
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


cdef inline double _int_double_pearson_cov(int* x, NumericType* y, Py_ssize_t n) noexcept nogil:
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


cdef inline double _int_double_euclidean_dist(int* x, NumericType* y, Py_ssize_t n) noexcept nogil:
    if n == 0: return NAN
    cdef double sum_sq = 0.0
    cdef double diff
    cdef Py_ssize_t i
    for i in range(n):
        diff = <double>x[i] - <double>y[i]
        sum_sq += diff * diff
    return sqrt(sum_sq)


cdef inline double _int_double_manhattan_dist(int* x, NumericType* y, Py_ssize_t n) noexcept nogil:
    if n == 0: return NAN
    cdef double sum_abs = 0.0
    cdef Py_ssize_t i
    for i in range(n):
        sum_abs += fabs(<double>x[i] - <double>y[i])
    return sum_abs


cdef inline double _int_double_cosine_sim(int* x, NumericType* y, Py_ssize_t n) noexcept nogil:
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


cdef inline double _int_double_jaccard_sim(int* x, NumericType* y, Py_ssize_t n) noexcept nogil:
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


cdef inline double _int_double_hamming_dist(int* x, NumericType* y, Py_ssize_t n) noexcept nogil:
    if n == 0: return NAN
    cdef Py_ssize_t diffs = 0
    cdef Py_ssize_t i
    for i in range(n):
        if <double>x[i] != <double>y[i]:
            diffs += 1
    return <double>diffs / <double>n


cdef inline double _double_int_pearson_corr(double* x, NumericType* y, Py_ssize_t n) noexcept nogil:
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


cdef inline double _double_int_pearson_cov(double* x, NumericType* y, Py_ssize_t n) noexcept nogil:
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


cdef inline double _double_int_euclidean_dist(double* x, NumericType* y, Py_ssize_t n) noexcept nogil:
    if n == 0: return NAN
    cdef double sum_sq = 0.0
    cdef double diff
    cdef Py_ssize_t i
    for i in range(n):
        diff = <double>x[i] - <double>y[i]
        sum_sq += diff * diff
    return sqrt(sum_sq)


cdef inline double _double_int_manhattan_dist(double* x, NumericType* y, Py_ssize_t n) noexcept nogil:
    if n == 0: return NAN
    cdef double sum_abs = 0.0
    cdef Py_ssize_t i
    for i in range(n):
        sum_abs += fabs(<double>x[i] - <double>y[i])
    return sum_abs


cdef inline double _double_int_cosine_sim(double* x, NumericType* y, Py_ssize_t n) noexcept nogil:
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


cdef inline double _double_int_jaccard_sim(double* x, NumericType* y, Py_ssize_t n) noexcept nogil:
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


cdef inline double _double_int_hamming_dist(double* x, NumericType* y, Py_ssize_t n) noexcept nogil:
    if n == 0: return NAN
    cdef Py_ssize_t diffs = 0
    cdef Py_ssize_t i
    for i in range(n):
        if <double>x[i] != <double>y[i]:
            diffs += 1
    return <double>diffs / <double>n
