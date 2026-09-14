# cython: language_level=3
cimport cython
from libc.math cimport isnan, sqrt, fabs, NAN, fmax
from libc.string cimport memmove, memcpy, memset
from libc.stdint cimport int32_t, int64_t
from cpython.unicode cimport Py_UCS2


# ============================================================
#  Data type enum
# ============================================================
cdef enum DataType:
    TYPE_INT = 0
    TYPE_DOUBLE = 2
    TYPE_CHAR256 = 3
    TYPE_UCS2 = 4
    TYPE_DATE = 5
    TYPE_BOOL = 6
    TYPE_CATEGORY = 7
    TYPE_DATETIME = 8
    TYPE_TIME = 9


# ============================================================
#  Fused numeric type for vectorized math
# ============================================================
ctypedef fused NumericType:
    int
    long
    double
    long long

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
    # ---------------------------------------------------------
    # 核心元数据：使用 readonly。
    # 内部 C 语言可以直接 col.length 极速无锁读取和修改。
    # 外部 Python 代码只能读取 col.length，尝试修改会抛出 AttributeError。
    # ---------------------------------------------------------
    cdef readonly Py_ssize_t length
    cdef readonly Py_ssize_t capacity
    cdef readonly int dtype

    cdef readonly bint is_2d
    cdef readonly int num_cols
    cdef readonly bint owns_data

    # ---------------------------------------------------------
    # C 级别虚函数 (Virtual Methods)：
    # 允许 FastDataFrame 绕过类型强转，直接操作底层数据
    # ---------------------------------------------------------
    cdef Py_ssize_t _get_rows_c(self) noexcept nogil
    cdef void* _get_data_ptr(self) noexcept nogil

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
    """Gather with NaN fill for out-of-range indices.
    Only safe for elem_size == sizeof(double) (float64)."""
    cdef Py_ssize_t i, idx
    cdef double nan_val = NAN
    for i in range(n):
        idx = indices[i]
        if idx < 0:
            memcpy(<char*>dst + i * elem_size, &nan_val, elem_size)
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

# ============================================================
#  Generic element-wise transforms (fused type, zero-overhead)
# ============================================================
cdef inline void _col_abs(NumericType* dst, NumericType* src, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    for i in range(n):
        if NumericType is double:
            dst[i] = fabs(src[i])
        else:
            dst[i] = src[i] if src[i] >= 0 else -src[i]

cdef inline void _col_round(NumericType* dst, NumericType* src,
                            Py_ssize_t n, double factor) noexcept nogil:
    cdef Py_ssize_t i
    cdef double v, scaled, intpart, frac
    if NumericType is double:
        for i in range(n):
            v = src[i]
            scaled = v * factor
            intpart = <double>(<long long>scaled)
            frac = scaled - intpart
            if frac < 0.0:
                frac = -frac
            if frac > 0.5 or (frac == 0.5 and (<long long>intpart) % 2 != 0):
                intpart += (1.0 if scaled >= 0 else -1.0)
            dst[i] = intpart / factor
    else:
        for i in range(n):
            dst[i] = src[i]

cdef inline void _col_ffill_double(double* data, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double last_val = NAN
    for i in range(n):
        if not isnan(data[i]):
            last_val = data[i]
        elif not isnan(last_val):
            data[i] = last_val

cdef inline void _col_bfill_double(double* data, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double next_val = NAN
    for i in range(n - 1, -1, -1):
        if not isnan(data[i]):
            next_val = data[i]
        elif not isnan(next_val):
            data[i] = next_val


# ============================================================
#  Fused-type aggregate helpers (nogil, zero-overhead)
#  Used by IntColumn/DoubleColumn for unified 2D support.
# ============================================================

cdef inline double _c_max(NumericType* ptr, Py_ssize_t n) noexcept nogil:
    """Return max of ptr[0..n). Caller must ensure n > 0."""
    cdef Py_ssize_t i
    cdef NumericType max_val = ptr[0]
    cdef NumericType val
    for i in range(1, n):
        val = ptr[i]
        if NumericType is double:
            if not isnan(val) and (isnan(max_val) or val > max_val):
                max_val = val
        else:
            if val > max_val:
                max_val = val
    return <double>max_val

cdef inline double _c_min(NumericType* ptr, Py_ssize_t n) noexcept nogil:
    """Return min of ptr[0..n). Caller must ensure n > 0."""
    cdef Py_ssize_t i
    cdef NumericType min_val = ptr[0]
    cdef NumericType val
    for i in range(1, n):
        val = ptr[i]
        if NumericType is double:
            if not isnan(val) and (isnan(min_val) or val < min_val):
                min_val = val
        else:
            if val < min_val:
                min_val = val
    return <double>min_val

cdef inline double _c_sum(NumericType* ptr, Py_ssize_t n) noexcept nogil:
    """Return sum of ptr[0..n). Uses int64 accumulator for int types."""
    cdef Py_ssize_t i
    cdef double total = 0.0
    cdef long long total_i = 0
    if NumericType is double:
        for i in range(n):
            if not isnan(ptr[i]):
                total += ptr[i]
        return total
    else:
        for i in range(n):
            total_i += <long long>ptr[i]
        return <double>total_i

cdef inline double _c_prod(NumericType* ptr, Py_ssize_t n) noexcept nogil:
    """Return product of ptr[0..n). Skips NaN for double."""
    cdef Py_ssize_t i
    cdef double total = 1.0
    cdef bint has_valid = False
    for i in range(n):
        if NumericType is double:
            if not isnan(ptr[i]):
                total *= <double>ptr[i]
                has_valid = True
        else:
            total *= <double>ptr[i]
            has_valid = True
    return total if has_valid else NAN

cdef inline Py_ssize_t _c_idxmax(NumericType* ptr, Py_ssize_t n) noexcept nogil:
    """Return index of max in ptr[0..n). Caller must ensure n > 0."""
    cdef Py_ssize_t i, max_idx = 0
    cdef NumericType max_val = ptr[0]
    cdef NumericType val
    for i in range(1, n):
        val = ptr[i]
        if NumericType is double:
            if not isnan(val) and (isnan(max_val) or val > max_val):
                max_val = val; max_idx = i
        else:
            if val > max_val:
                max_val = val; max_idx = i
    return max_idx

cdef inline Py_ssize_t _c_idxmin(NumericType* ptr, Py_ssize_t n) noexcept nogil:
    """Return index of min in ptr[0..n). Caller must ensure n > 0."""
    cdef Py_ssize_t i, min_idx = 0
    cdef NumericType min_val = ptr[0]
    cdef NumericType val
    for i in range(1, n):
        val = ptr[i]
        if NumericType is double:
            if not isnan(val) and (isnan(min_val) or val < min_val):
                min_val = val; min_idx = i
        else:
            if val < min_val:
                min_val = val; min_idx = i
    return min_idx

cdef inline double _c_var(NumericType* ptr, Py_ssize_t n, double mean_val) noexcept nogil:
    """Return sample variance of ptr[0..n) given pre-computed mean."""
    cdef Py_ssize_t i
    cdef double sum_sq = 0.0, diff
    for i in range(n):
        diff = <double>ptr[i] - mean_val
        sum_sq += diff * diff
    return sum_sq / <double>(n - 1) if n > 1 else 0.0

cdef inline double _c_mad(NumericType* ptr, Py_ssize_t n, double mean_val) noexcept nogil:
    """Return mean absolute deviation of ptr[0..n) given pre-computed mean."""
    cdef Py_ssize_t i
    cdef double total_dev = 0.0
    for i in range(n):
        total_dev += fabs(<double>ptr[i] - mean_val)
    return total_dev / <double>n if n > 0 else NAN


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class IntColumn(BaseColumn):
    cdef int* data_c
    cdef object _base_obj

    cdef void _reserve(self, Py_ssize_t new_cap)
    cdef double _sum_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef double _mean_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef double _min_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef double _max_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef Py_ssize_t _count_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef double _var_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef double _std_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil

    cdef IntColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=*)

    cdef void to_double_buffer(self, double* out, Py_ssize_t n)
    cdef IntColumn gather_with_sentinel(self, Py_ssize_t* indices, Py_ssize_t n, int sentinel)
    cdef Py_ssize_t count_nonzero(self)
    cdef void gather_zero_to(self, Py_ssize_t* out)
    cdef void _central_moments(self, double* out_mean, double* out_m2, double* out_m3, double* out_m4) noexcept nogil
    cdef double _quickselect(self, int* data, Py_ssize_t n, Py_ssize_t k) noexcept nogil

    @staticmethod
    cdef IntColumn create_view(int* ptr, Py_ssize_t length, object base_obj)

    cdef inline DoubleColumn _make_like_nan_dbl(self)
    cdef IntColumn _cmp_op(self, object val, int op)

cdef int _cmp_double_asc(const void* a, const void* b) noexcept nogil

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class DoubleColumn(BaseColumn):
    cdef double* data_c
    cdef object _base_obj
    cdef void _reserve(self, Py_ssize_t new_cap)
    cdef double _sum_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef double _mean_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef double _min_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef double _max_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef Py_ssize_t _count_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef double _var_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef double _std_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil
    cdef DoubleColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=*)

    cdef void to_double_buffer(self, double* out, Py_ssize_t n)
    cdef DoubleColumn gather_with_sentinel(self, Py_ssize_t* indices, Py_ssize_t n, double sentinel)
    cdef void _central_moments(self, double* out_mean, double* out_m2, double* out_m3, double* out_m4) noexcept nogil
    cdef double _quickselect(self, double* data, Py_ssize_t n, Py_ssize_t k) noexcept nogil
    cdef double quantile_per_subcol(self, double* col_ptr, Py_ssize_t n, double q) noexcept
    @staticmethod
    cdef DoubleColumn create_view(double* ptr, Py_ssize_t length, object base_obj)
    cdef inline DoubleColumn _make_like_nan(self)
    cdef IntColumn _cmp_op(self, object val, int op)


cdef bint _c_str_contains(const char* source, const char* target) noexcept nogil

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class AsciiColumn(BaseColumn):
    cdef char* data_c                # 变长字符缓冲区
    cdef Py_ssize_t* offsets_c       # 地址偏移数组: offsets[i] = 字符串i在data_c中的起始字节偏移
    cdef public Py_ssize_t byte_capacity  # data_c 的字节容量
    cdef public Py_ssize_t byte_length    # data_c 已使用的字节总数
    cdef void _reserve_rows(self, Py_ssize_t new_cap)
    cdef void _reserve_bytes(self, Py_ssize_t new_byte_cap)
    cdef void append_nolen(self, str val) except *
    cdef void undo_nolen(self) noexcept
    cdef inline str _get_str(self, Py_ssize_t idx)
    cdef void _set_item(self, Py_ssize_t idx, str val) except *
    cdef AsciiColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=*)
    cdef IntColumn _cmp_op(self, object val, int op)

cdef int _ucs2_len(Py_UCS2* string) noexcept nogil
cdef Py_UCS2* _ucs2_strstr(Py_UCS2* haystack, int h_len, Py_UCS2* needle, int n_len) noexcept nogil

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class UCS2Column(BaseColumn):
    cdef Py_UCS2* chars_c
    cdef Py_ssize_t* offsets_c
    cdef public Py_ssize_t char_capacity
    cdef public Py_ssize_t char_length
    cdef void _reserve_rows(self, Py_ssize_t new_cap) except *
    cdef void _reserve_chars(self, Py_ssize_t new_char_cap) except *
    cdef void append_nolen(self, str val) except *
    cdef void undo_nolen(self) noexcept
    cdef UCS2Column take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=*)
    cdef inline void _set_item(self, Py_ssize_t idx, str val)
    cdef inline str _get_str(self, Py_ssize_t idx)
    cdef IntColumn _cmp_op(self, object val, int op)



cdef bint _is_leap_year(int y) noexcept nogil
cdef int _days_in_month(int y, int m) noexcept nogil
cdef int32_t _ymd_to_days(int y, int m, int d) noexcept nogil
cdef void _days_to_ymd(int32_t days, int* y, int* m, int* d) noexcept nogil
cdef int32_t _parse_date_str(str s) noexcept

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class DateColumn(BaseColumn):
    cdef int32_t* data_c
    cdef object _base_obj
    cdef void _reserve(self, Py_ssize_t new_cap)
    cdef inline int32_t _resolve_date(self, date) except *
    cdef inline int32_t _to_ymd_int(self, int32_t epoch_days) noexcept nogil
    cdef inline int _iso_week_of_year(self, int32_t days, int* iso_y) noexcept nogil
    cdef DateColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=*)
    @staticmethod
    cdef DateColumn create_view(int32_t* ptr, Py_ssize_t length, object base_obj)
    cdef IntColumn _cmp_op(self, object val, int op)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class BoolColumn(BaseColumn):
    cdef unsigned char* data_c
    cdef bint _owns_data
    cdef object _base_obj
    cdef void _reserve(self, Py_ssize_t new_cap)
    cdef void append_nolen(self, object val) except *
    cdef BoolColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=*)
    cdef void gather_zero_to(self, Py_ssize_t* out)
    cdef Py_ssize_t count_nonzero(self)
    @staticmethod
    cdef BoolColumn create_view(unsigned char* ptr, Py_ssize_t length, object base_obj)



@cython.boundscheck(False)
@cython.wraparound(False)
cdef class CategoryColumn(BaseColumn):
    cdef int* codes_c
    cdef bint _owns_data
    cdef object _base_obj
    cdef list categories
    cdef dict _cat_to_code
    cdef void _reserve(self, Py_ssize_t new_cap)
    cdef void append_nolen(self, object val) except *
    cdef int _get_or_add_code(self, str val)
    cdef str _resolve(self, Py_ssize_t i)
    cdef CategoryColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=*)
    @staticmethod
    cdef CategoryColumn create_view(int* ptr, Py_ssize_t length, list categories, dict cat_to_code, object base_obj)



cdef int64_t _parse_datetime_str(str s) except -1
cdef void _datetime_to_ymdhms(int64_t us, int* y, int* m, int* d, int* h, int* mi, int* s) noexcept nogil

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class DateTimeColumn(BaseColumn):
    cdef int64_t* data_c
    cdef object _base_obj
    cdef void _reserve(self, Py_ssize_t new_cap)
    cdef DateTimeColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=*)

    cdef inline int64_t _resolve_dt(self, val) except *
    @staticmethod
    cdef DateTimeColumn create_view(int64_t* ptr, Py_ssize_t length, object base_obj)
    cdef IntColumn _cmp_op(self, object val, int op)


cdef int32_t _parse_time_str(str s) except -1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class TimeColumn(BaseColumn):
    cdef int32_t* data_c
    cdef object _base_obj
    cdef void _reserve(self, Py_ssize_t new_cap)
    cdef inline int32_t _resolve_time(self, val) except -1
    cdef TimeColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=*)
    @staticmethod
    cdef TimeColumn create_view(int32_t* ptr, Py_ssize_t length, object base_obj)
    cdef IntColumn _cmp_op(self, object val, int op)
