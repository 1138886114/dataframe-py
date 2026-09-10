# cython: language_level=3
from libc.math cimport sqrt, fabs, NAN
from libc.string cimport memmove, memcpy, memset
from libc.stdint cimport int32_t, int64_t

# ============================================================
#  Data type enum
# ============================================================
cdef enum DataType:
    TYPE_INT = 0
    TYPE_FLOAT = 2
    TYPE_DOUBLE = 2
    TYPE_CHAR256 = 3
    TYPE_UCS2 = 4
    TYPE_DATE = 5
    TYPE_BOOL = 6
    TYPE_CATEGORY = 7
    TYPE_DATETIME = 8
    TYPE_TIME = 9

# ============================================================
#  Aligned memory allocation
# ============================================================
cdef void* aligned_malloc(size_t size, size_t alignment=*) nogil
cdef void* aligned_realloc(void* ptr, size_t old_size, size_t new_size, size_t alignment=*) nogil
cdef void os_aligned_free(void* ptr) nogil

# ============================================================
#  Base column class
# ============================================================
cdef class BaseColumn:
    cdef public Py_ssize_t length
    cdef public Py_ssize_t capacity
    cdef public int dtype

# ============================================================
#  Astype helper
# ============================================================
cdef int _resolve_astype_target(object target) except -1

# ============================================================
#  Shift helpers (inline for hot-path performance)
# ============================================================
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

# ============================================================
#  Generic column helpers (inline for zero-overhead abstraction)
# ============================================================
cdef inline void* _col_pop_back(void* data_c, Py_ssize_t* length,
                                Py_ssize_t elem_size) noexcept nogil:
    if length[0] == 0:
        return NULL
    length[0] -= 1
    return <char*>data_c + length[0] * elem_size

cdef inline void _col_pop_front(void* data_c, Py_ssize_t* length,
                                Py_ssize_t elem_size) noexcept nogil:
    if length[0] > 1:
        memmove(data_c, <char*>data_c + elem_size,
                (length[0] - 1) * elem_size)
    length[0] -= 1

cdef inline void _col_concat(void* data_c, Py_ssize_t* length,
                             void* other_data, Py_ssize_t other_len,
                             Py_ssize_t elem_size) noexcept nogil:
    if other_len > 0:
        memcpy(<char*>data_c + length[0] * elem_size, other_data,
               other_len * elem_size)
    length[0] += other_len

cdef inline void _col_drop_row(void* data_c, Py_ssize_t* length,
                               Py_ssize_t row_idx,
                               Py_ssize_t elem_size) noexcept nogil:
    cdef Py_ssize_t rem_len = length[0] - row_idx - 1
    if rem_len > 0:
        memmove(<char*>data_c + row_idx * elem_size,
                <char*>data_c + (row_idx + 1) * elem_size,
                rem_len * elem_size)
    length[0] -= 1

cdef inline void _col_gather(void* dst, void* src,
                             Py_ssize_t* indices, Py_ssize_t n,
                             Py_ssize_t elem_size) noexcept nogil:
    cdef Py_ssize_t i, idx
    for i in range(n):
        idx = indices[i]
        memcpy(<char*>dst + i * elem_size,
               <char*>src + idx * elem_size,
               elem_size)

cdef inline void _col_gather_zero(void* dst, void* src,
                                  Py_ssize_t* indices, Py_ssize_t n,
                                  Py_ssize_t elem_size) noexcept nogil:
    cdef Py_ssize_t i, idx
    for i in range(n):
        idx = indices[i]
        if idx < 0:
            memset(<char*>dst + i * elem_size, 0, elem_size)
        else:
            memcpy(<char*>dst + i * elem_size,
                   <char*>src + idx * elem_size,
                   elem_size)

cdef inline void _col_gather_nan(void* dst, void* src,
                                 Py_ssize_t* indices, Py_ssize_t n,
                                 Py_ssize_t elem_size) noexcept nogil:
    cdef Py_ssize_t i, idx
    for i in range(n):
        idx = indices[i]
        if idx < 0:
            (<double*>(<char*>dst + i * elem_size))[0] = NAN
        else:
            memcpy(<char*>dst + i * elem_size,
                   <char*>src + idx * elem_size,
                   elem_size)

# ============================================================
#  2D (rows*cols column-major) row-level helpers
#  A logical row i lives at flat index j*rows + i for sub-column j.
#  These ops need no reallocation: they only memmove within the
#  existing flat buffer (caller must ensure spare capacity for
#  _col_shift_right_2d).
# ============================================================
cdef inline void _col_shift_right_2d_double(void* buf, Py_ssize_t* length,
                                            Py_ssize_t rows, Py_ssize_t cols,
                                            Py_ssize_t row_idx) noexcept nogil:
    """针对 Double 的 2D 插入，填补 NaN"""
    cdef Py_ssize_t j, base_src, base_dst, tail
    cdef double* d_buf = <double*>buf
    for j in range(cols - 1, -1, -1):
        base_src = j * rows
        base_dst = j * (rows + 1)
        tail = rows - row_idx
        if tail > 0:
            memmove(d_buf + base_dst + row_idx + 1,
                    d_buf + base_src + row_idx,
                    tail * sizeof(double))
        if row_idx > 0:
            memmove(d_buf + base_dst,
                    d_buf + base_src,
                    row_idx * sizeof(double))
        # 清空产生的垃圾内存洞，赋默认值 NaN
        d_buf[base_dst + row_idx] = NAN
    length[0] += cols

cdef inline void _col_shift_right_2d_int(void* buf, Py_ssize_t* length,
                                         Py_ssize_t rows, Py_ssize_t cols,
                                         Py_ssize_t row_idx, Py_ssize_t elem_size) noexcept nogil:
    """针对 Int/Date/Time 等整数型的 2D 插入，填补 0 (或者你自定义的 Null Sentinel)"""
    cdef Py_ssize_t j, base_src, base_dst, tail
    for j in range(cols - 1, -1, -1):
        base_src = j * rows
        base_dst = j * (rows + 1)
        tail = rows - row_idx
        if tail > 0:
            memmove(<char*>buf + (base_dst + row_idx + 1) * elem_size,
                    <char*>buf + (base_src + row_idx) * elem_size,
                    tail * elem_size)
        if row_idx > 0:
            memmove(<char*>buf + base_dst * elem_size,
                    <char*>buf + base_src * elem_size,
                    row_idx * elem_size)
        # 将新拉开的一行置为 0
        memset(<char*>buf + (base_dst + row_idx) * elem_size, 0, elem_size)
    length[0] += cols

cdef inline void _col_shift_left_2d(void* buf, Py_ssize_t* length,
                                    Py_ssize_t rows, Py_ssize_t cols,
                                    Py_ssize_t row_idx,
                                    Py_ssize_t elem_size) noexcept nogil:
    """Remove logical row (row_idx in [0, rows)) from every sub-column.
    length shrinks by cols."""
    cdef Py_ssize_t j, base_src, base_dst, tail
    for j in range(cols):
        base_dst = j * (rows - 1)
        base_src = j * rows
        if row_idx > 0 and j > 0:
            memmove(<char*>buf + base_dst * elem_size,
                    <char*>buf + base_src * elem_size,
                    row_idx * elem_size)
        tail = rows - row_idx - 1
        if tail > 0:
            memmove(<char*>buf + (base_dst + row_idx) * elem_size,
                    <char*>buf + (base_src + row_idx + 1) * elem_size,
                    tail * elem_size)
    length[0] -= cols

cdef inline void _col_gather_2d(void* dst, void* src,
                                Py_ssize_t* indices, Py_ssize_t n,
                                Py_ssize_t rows, Py_ssize_t cols,
                                Py_ssize_t elem_size) noexcept nogil:
    """Row-wise gather: dst keeps cols sub-columns of n rows each, one
    per src sub-column (which has rows entries)."""
    cdef Py_ssize_t j
    for j in range(cols):
        _col_gather(<char*>dst + j * n * elem_size,
                    <char*>src + j * rows * elem_size,
                    indices, n, elem_size)

cdef inline void _col_gather_2d_zero(void* dst, void* src,
                                     Py_ssize_t* indices, Py_ssize_t n,
                                     Py_ssize_t rows, Py_ssize_t cols,
                                     Py_ssize_t elem_size) noexcept nogil:
    cdef Py_ssize_t j
    for j in range(cols):
        _col_gather_zero(<char*>dst + j * n * elem_size,
                         <char*>src + j * rows * elem_size,
                         indices, n, elem_size)

cdef inline void _col_gather_2d_nan(void* dst, void* src,
                                    Py_ssize_t* indices, Py_ssize_t n,
                                    Py_ssize_t rows, Py_ssize_t cols,
                                    Py_ssize_t elem_size) noexcept nogil:
    cdef Py_ssize_t j
    for j in range(cols):
        _col_gather_nan(<char*>dst + j * n * elem_size,
                        <char*>src + j * rows * elem_size,
                        indices, n, elem_size)

cdef inline void _col_concat_2d(void* data_c, Py_ssize_t* length,
                                void* other_data, Py_ssize_t other_rows,
                                Py_ssize_t cols, Py_ssize_t elem_size) noexcept nogil:
    """2D column-major concat: each sub-column j has rows = length//cols
    entries in self and other_rows entries in other. Iterating sub-columns
    from high to low, shift the existing self block to its new base
    (j*new_rows) then append other's block. length grows by
    other_rows*cols (caller must reserve capacity first)."""
    cdef Py_ssize_t self_rows = length[0] // cols
    cdef Py_ssize_t new_rows = self_rows + other_rows
    cdef Py_ssize_t j
    if other_rows == 0 or cols == 0:
        return
    for j in range(cols - 1, -1, -1):
        if new_rows != self_rows:
            memmove(<char*>data_c + (j * new_rows) * elem_size,
                    <char*>data_c + (j * self_rows) * elem_size,
                    self_rows * elem_size)
        memcpy(<char*>data_c + (j * new_rows + self_rows) * elem_size,
               <char*>other_data + (j * other_rows) * elem_size,
               other_rows * elem_size)
    length[0] += other_rows * cols

cdef inline Py_ssize_t _col_reserve(void** data_c, Py_ssize_t* capacity,
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
#  Fused numeric type for vectorized math
# ============================================================
ctypedef fused NumericType:
    int
    long
    double
    long long

# ============================================================
#  Vectorized math: same-type (inline for SIMD auto-vectorization)
# ============================================================
cdef inline double _vec_pearson_corr(NumericType* x, NumericType* y, Py_ssize_t n) noexcept nogil:
    if n < 2: return NAN
    cdef double sum_x = 0.0, sum_y = 0.0
    cdef double mean_x, mean_y
    cdef double cov = 0.0, var_x = 0.0, var_y = 0.0
    cdef double xi, yi, denom
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
    cdef double sum_sq = 0.0, diff
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
    cdef Py_ssize_t intersection = 0, union_cnt = 0
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

# ============================================================
#  Vectorized math: int* / NumericType cross-type (inline)
# ============================================================
cdef inline double _int_double_pearson_corr(int* x, NumericType* y, Py_ssize_t n) noexcept nogil:
    if n < 2: return NAN
    cdef double sum_x = 0.0, sum_y = 0.0
    cdef double mean_x, mean_y
    cdef double cov = 0.0, var_x = 0.0, var_y = 0.0
    cdef double xi, yi, denom
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
    cdef double sum_sq = 0.0, diff
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
    cdef Py_ssize_t intersection = 0, union_cnt = 0
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

# ============================================================
#  Vectorized math: double* / NumericType cross-type (inline)
# ============================================================
cdef inline double _double_int_pearson_corr(double* x, NumericType* y, Py_ssize_t n) noexcept nogil:
    if n < 2: return NAN
    cdef double sum_x = 0.0, sum_y = 0.0
    cdef double mean_x, mean_y
    cdef double cov = 0.0, var_x = 0.0, var_y = 0.0
    cdef double xi, yi, denom
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
    cdef double sum_sq = 0.0, diff
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
    cdef Py_ssize_t intersection = 0, union_cnt = 0
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

# ============================================================
#  Fused hot-path helpers (inline, zero-overhead)
# ============================================================
cdef inline Py_ssize_t _col_count_nonzeros(NumericType* ptr, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i, count = 0
    for i in range(n):
        count += (ptr[i] != 0)
    return count

ctypedef fused FixedNum:
    int
    int32_t
    int64_t
    double

cdef inline void _col_memcpy(FixedNum* dst, FixedNum* src, Py_ssize_t n) noexcept nogil:
    if n > 0:
        memcpy(dst, src, n * sizeof(FixedNum))
