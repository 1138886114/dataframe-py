# _fastdataframe_linalg.pxi — Linear algebra operations (module-level cdef)
# Included at module level in fastdataframe.pyx (after class definition)
# Replaces standalone linalg.pyx with accessor pattern like _DataFrameMetricAccessor

from libc.string cimport memcpy, memset
from libc.math cimport sqrt, fabs, NAN

from fluidframe.core._base cimport (
    BaseColumn, TYPE_INT, TYPE_DOUBLE,
    aligned_malloc, os_aligned_free,
    _vec_euclidean_dist, _vec_manhattan_dist, _vec_cosine_sim,
    _vec_hamming_dist, _vec_jaccard_sim
)
from fluidframe.core.index cimport RangeIndex

# ============================================================
#  Helper: Check if columns are contiguous in memory
# ============================================================

cdef inline Py_ssize_t _linalg_logical_cols(FastDataFrame df):
    """矩阵逻辑列数：2D 列（is_2d 且 cols>1）按子列展开各计 1，1D 列计 1。"""
    cdef Py_ssize_t total = 0
    cdef bint is2d
    cdef Py_ssize_t c, r
    for name, col in df.columns.items():
        is2d, c, r = _col_shape_probe(col)
        total += c if (is2d and c > 1) else 1
    return total

cdef bint _linalg_is_contiguous_columns(FastDataFrame df):
    """安全修复版：严谨校验多列 DataFrame 底层是否属于同一块物理连续内存"""
    cdef list cols_list = list(df.columns.values())
    cdef Py_ssize_t cols = len(cols_list)
    if cols < 2:
        return False

    cdef BaseColumn col = <BaseColumn>cols_list[0]
    if col.dtype != TYPE_DOUBLE:
        return False

    cdef double* prev_ptr = (<DoubleColumn>col).data_c
    cdef Py_ssize_t rows = df.df_length
    cdef double* curr_ptr
    cdef Py_ssize_t i

    for i in range(1, cols):
        col = <BaseColumn>cols_list[i]
        if col.dtype != TYPE_DOUBLE:
            return False
        curr_ptr = (<DoubleColumn>col).data_c
        if curr_ptr != prev_ptr + rows:
            return False
        prev_ptr = curr_ptr

    return True

cdef double* _linalg_to_contiguous_fast(FastDataFrame df, Py_ssize_t* out_rows, Py_ssize_t* out_cols) except NULL:
    cdef Py_ssize_t rows = df.df_length
    cdef list cols_list = list(df.columns.values())
    cdef Py_ssize_t cols = len(cols_list)

    if cols == 0 or rows == 0:
        out_rows[0] = rows
        out_cols[0] = 0
        return NULL

    cdef Py_ssize_t logical_cols = _linalg_logical_cols(df)
    cdef Py_ssize_t num_elements = rows * logical_cols

    cdef double* data = <double*>aligned_malloc(num_elements * sizeof(double))
    if not data:
        raise MemoryError()

    cdef Py_ssize_t i, j
    cdef BaseColumn col
    if logical_cols != cols:
        os_aligned_free(data)
        return _linalg_to_contiguous(df, out_rows, out_cols)

    try:
        for j in range(cols):
            col = <BaseColumn>cols_list[j]
            if col.dtype != TYPE_DOUBLE:
                os_aligned_free(data)
                return _linalg_to_contiguous(df, out_rows, out_cols)
            with nogil:
                for i in range(rows):
                    data[i * logical_cols + j] = (<DoubleColumn>col).data_c[i]
    except Exception:
        os_aligned_free(data)
        raise

    out_rows[0] = rows
    out_cols[0] = logical_cols
    return data

cdef double* _linalg_to_contiguous(FastDataFrame df, Py_ssize_t* out_rows, Py_ssize_t* out_cols) except NULL:
    cdef Py_ssize_t rows = df.df_length
    cdef list cols_list = list(df.columns.values())
    cdef list name_list = list(df.columns.keys())
    cdef Py_ssize_t n_cols_named = len(cols_list)
    cdef Py_ssize_t logical_cols = _linalg_logical_cols(df)

    if rows == 0 or logical_cols == 0:
        out_rows[0] = rows
        out_cols[0] = logical_cols
        return NULL

    cdef Py_ssize_t num_elements = rows * logical_cols
    cdef double* data = <double*>aligned_malloc(num_elements * sizeof(double))
    if not data: raise MemoryError()

    cdef Py_ssize_t i, j, sub, out_col = 0
    cdef BaseColumn col
    cdef int col_dtype
    cdef int* int_ptr
    cdef double* double_ptr
    cdef bint is2d
    cdef Py_ssize_t c_col, r_col, n_sub

    try:
        for j in range(n_cols_named):
            col = <BaseColumn>cols_list[j]
            col_dtype = col.dtype
            if col_dtype != TYPE_INT and col_dtype != TYPE_DOUBLE:
                raise TypeError(f"Cannot perform linear algebra on non-numeric column '{name_list[j]}'")

            is2d, c_col, r_col = _col_shape_probe(col)
            n_sub = c_col if (is2d and c_col > 1) else 1

            if col_dtype == TYPE_INT:
                int_ptr = (<IntColumn>col).data_c
                with nogil:
                    for sub in range(n_sub):
                        for i in range(rows):
                            data[i * logical_cols + out_col + sub] = <double>int_ptr[sub * r_col + i]
            else:
                double_ptr = (<DoubleColumn>col).data_c
                with nogil:
                    for sub in range(n_sub):
                        for i in range(rows):
                            data[i * logical_cols + out_col + sub] = double_ptr[sub * r_col + i]
            out_col += n_sub
    except Exception:
        os_aligned_free(data)
        raise

    out_rows[0] = rows
    out_cols[0] = logical_cols
    return data


cdef FastDataFrame _linalg_from_contiguous(double* data, Py_ssize_t rows, Py_ssize_t cols):
    """
    消除碎列，生成完美的 2D 连续内存块以匹配 FastDataFrame 核心设计，避免转置惩罚。
    (输入 data 预期为 行优先 C-contiguous)
    """
    cdef FastDataFrame df = FastDataFrame()
    df.df_length = rows
    df.index = RangeIndex(0, rows)

    if rows == 0 or cols == 0:
        return df

    cdef Py_ssize_t total_elements = rows * cols

    # 全量数据申请一块 2D Column，符合 F-contiguous(列优先) 布局
    cdef DoubleColumn block2d = DoubleColumn(total_elements, cols=cols)
    block2d.length = total_elements
    cdef double* b_ptr = block2d.data_c

    cdef Py_ssize_t i, j
    # 矩阵转置：将传入的 C-contiguous (行优先) 转为底层框架期待的 F-contiguous (列优先)
    with nogil:
        for i in range(rows):
            for j in range(cols):
                b_ptr[j * rows + i] = data[i * cols + j]

    cdef list col_names = [chr(65 + i) if i < 26 else f"Col_{i}" for i in range(cols)]

    # 构建零拷贝视图 (Zero-copy views)
    for j in range(cols):
        df.columns[col_names[j]] = DoubleColumn.create_view(b_ptr + j * rows, rows, block2d)

    # 保留对 2D 块的强引用，防止内存被 GC
    if not hasattr(df, '_2d_blocks'):
        df._2d_blocks = []
    df._2d_blocks.append(block2d)

    return df

cdef inline double _linalg_clip_val(double v, double min_v, double max_v) nogil:
    if v < min_v: return min_v
    if v > max_v: return max_v
    return v

# ============================================================
#  Core linalg operations
# ============================================================

cdef FastDataFrame _linalg_dot(FastDataFrame a, FastDataFrame b):
    cdef Py_ssize_t a_rows = a.df_length
    cdef Py_ssize_t a_cols = _linalg_logical_cols(a)
    cdef Py_ssize_t b_rows = b.df_length
    cdef Py_ssize_t b_cols = _linalg_logical_cols(b)

    if a_cols != b_rows:
        raise ValueError(f"Shape mismatch: ({a_rows}, {a_cols}) x ({b_rows}, {b_cols})")
    if a_rows == 0 or b_cols == 0 or a_cols == 0:
        return FastDataFrame()

    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t size_c
    cdef double s
    cdef double* A = NULL
    cdef double* B = NULL
    cdef double* C = NULL

    try:
        A = _linalg_to_contiguous(a, &a_rows, &a_cols)
        B = _linalg_to_contiguous(b, &b_rows, &b_cols)

        size_c = a_rows * b_cols
        C = <double*>aligned_malloc(size_c * sizeof(double))
        if not C: raise MemoryError()
        memset(C, 0, size_c * sizeof(double))

        # O(N^3) 行优先快速乘法（对缓存友好）
        with nogil:
            for i in range(a_rows):
                for k in range(a_cols):
                    s = A[i * a_cols + k]
                    if s != 0:
                        for j in range(b_cols):
                            C[i * b_cols + j] += s * B[k * b_cols + j]

        return _linalg_from_contiguous(C, a_rows, b_cols)
    finally:
        if A != NULL: os_aligned_free(A)
        if B != NULL: os_aligned_free(B)
        if C != NULL: os_aligned_free(C)

cdef FastDataFrame _linalg_transpose(FastDataFrame df):
    cdef Py_ssize_t rows, cols, i, j
    cdef Py_ssize_t size
    cdef double* A = NULL
    cdef double* B = NULL

    if df.df_length == 0 or _linalg_logical_cols(df) == 0:
        return FastDataFrame()
    try:
        A = _linalg_to_contiguous(df, &rows, &cols)
        size = rows * cols
        B = <double*>aligned_malloc(size * sizeof(double))
        if not B: raise MemoryError()

        with nogil:
            for i in range(rows):
                for j in range(cols):
                    B[j * rows + i] = A[i * cols + j]

        return _linalg_from_contiguous(B, cols, rows)
    finally:
        if A != NULL: os_aligned_free(A)
        if B != NULL: os_aligned_free(B)

cdef double _linalg_det(FastDataFrame df):
    cdef Py_ssize_t n = df.df_length
    if n == 0:
        return 1.0

    if _linalg_logical_cols(df) != n:
        raise ValueError("Matrix must be square")

    cdef Py_ssize_t i, j, k, pivot_row
    cdef double* A = NULL
    cdef double* LU = NULL
    cdef double det_val = 1.0
    cdef double pivot, temp, max_val, abs_val
    cdef int sign = 1

    try:
        A = _linalg_to_contiguous(df, &n, &n)
        LU = <double*>aligned_malloc(n * n * sizeof(double))
        if not LU: raise MemoryError()
        memcpy(LU, A, n * n * sizeof(double))

        with nogil:
            for i in range(n):
                pivot_row = i
                max_val = fabs(LU[i * n + i])
                for k in range(i + 1, n):
                    abs_val = fabs(LU[k * n + i])
                    if abs_val > max_val:
                        max_val = abs_val
                        pivot_row = k
                if max_val < 1e-15:
                    det_val = 0.0
                    break
                if pivot_row != i:
                    for j in range(i, n):
                        temp = LU[i * n + j]
                        LU[i * n + j] = LU[pivot_row * n + j]
                        LU[pivot_row * n + j] = temp
                    sign = -sign
                pivot = LU[i * n + i]
                for k in range(i + 1, n):
                    temp = LU[k * n + i] / pivot
                    for j in range(i + 1, n):
                        LU[k * n + j] -= temp * LU[i * n + j]

            if det_val != 0.0:
                for i in range(n):
                    det_val *= LU[i * n + i]
                det_val *= sign

        return det_val
    finally:
        if A != NULL: os_aligned_free(A)
        if LU != NULL: os_aligned_free(LU)

cdef FastDataFrame _linalg_inv(FastDataFrame df):
    cdef Py_ssize_t n = df.df_length
    if n == 0:
        return FastDataFrame()

    if _linalg_logical_cols(df) != n:
        raise ValueError("Matrix must be square")

    cdef Py_ssize_t rows, cols, i, j, k, pivot_row
    cdef double* A = NULL
    cdef double* aug = NULL
    cdef double* result_data = NULL
    cdef double pivot, temp, max_val, abs_val
    cdef int singular = 0

    try:
        A = _linalg_to_contiguous(df, &rows, &cols)
        aug = <double*>aligned_malloc(n * 2 * n * sizeof(double))
        if not aug: raise MemoryError()
        with nogil:
            for i in range(n):
                for j in range(n):
                    aug[i * 2 * n + j] = A[i * n + j]
                    aug[i * 2 * n + n + j] = 1.0 if i == j else 0.0
            for i in range(n):
                pivot_row = i
                max_val = fabs(aug[i * 2 * n + i])
                for k in range(i + 1, n):
                    abs_val = fabs(aug[k * 2 * n + i])
                    if abs_val > max_val:
                        max_val = abs_val
                        pivot_row = k
                if max_val < 1e-15:
                    singular = 1
                    break

                if pivot_row != i:
                    for j in range(i, 2 * n):
                        temp = aug[i * 2 * n + j]
                        aug[i * 2 * n + j] = aug[pivot_row * 2 * n + j]
                        aug[pivot_row * 2 * n + j] = temp

                pivot = aug[i * 2 * n + i]
                for j in range(i, 2 * n):
                    aug[i * 2 * n + j] /= pivot

                for k in range(n):
                    if k == i: continue
                    temp = aug[k * 2 * n + i]
                    for j in range(i, 2 * n):
                        aug[k * 2 * n + j] -= temp * aug[i * 2 * n + j]

        if singular:
            raise ValueError("Matrix is singular")

        result_data = <double*>aligned_malloc(n * n * sizeof(double))
        if not result_data: raise MemoryError()

        with nogil:
            for i in range(n):
                for j in range(n):
                    result_data[i * n + j] = aug[i * 2 * n + n + j]

        return _linalg_from_contiguous(result_data, n, n)
    finally:
        if A != NULL: os_aligned_free(A)
        if aug != NULL: os_aligned_free(aug)
        if result_data != NULL: os_aligned_free(result_data)

cdef FastDataFrame _linalg_solve(FastDataFrame A_df, FastDataFrame b_df):
    cdef Py_ssize_t n = A_df.df_length
    if n == 0:
        return FastDataFrame()

    cdef Py_ssize_t rows, cols, i, j, k, pivot_row
    cdef double* a_data = NULL
    cdef double* b_data = NULL
    cdef double* lu = NULL
    cdef double* x = NULL
    cdef double pivot, temp, max_val, abs_val, sum_val

    if _linalg_logical_cols(A_df) != n:
        raise ValueError("Matrix A must be square")
    if n != b_df.df_length:
        raise ValueError("Shape mismatch: A and b must have same number of rows")
    if _linalg_logical_cols(b_df) != 1:
        raise ValueError("b must have exactly one column")

    try:
        a_data = _linalg_to_contiguous(A_df, &rows, &cols)
        b_data = _linalg_to_contiguous(b_df, &rows, &cols)
        lu = <double*>aligned_malloc(n * n * sizeof(double))
        x = <double*>aligned_malloc(n * sizeof(double))
        if not lu or not x: raise MemoryError()
        memcpy(lu, a_data, n * n * sizeof(double))

        with nogil:
            for i in range(n):
                x[i] = b_data[i]
            for i in range(n):
                pivot_row = i
                max_val = fabs(lu[i * n + i])
                for k in range(i + 1, n):
                    abs_val = fabs(lu[k * n + i])
                    if abs_val > max_val:
                        max_val = abs_val
                        pivot_row = k
                if max_val < 1e-15:
                    raise ValueError("Matrix is singular")
                if pivot_row != i:
                    for j in range(i, n):
                        temp = lu[i * n + j]
                        lu[i * n + j] = lu[pivot_row * n + j]
                        lu[pivot_row * n + j] = temp
                    temp = x[i]
                    x[i] = x[pivot_row]
                    x[pivot_row] = temp
                pivot = lu[i * n + i]
                for k in range(i + 1, n):
                    temp = lu[k * n + i] / pivot
                    for j in range(i + 1, n):
                        lu[k * n + j] -= temp * lu[i * n + j]
                    x[k] -= temp * x[i]
            for i in range(n - 1, -1, -1):
                sum_val = x[i]
                for j in range(i + 1, n):
                    sum_val -= lu[i * n + j] * x[j]
                x[i] = sum_val / lu[i * n + i]

        # solve 输出的是一个 1D 数组(列)
        result = FastDataFrame()
        result.df_length = n
        col = DoubleColumn(n)
        col.length = n
        for i in range(n):
            col.data_c[i] = x[i]
        b_col_name = list(b_df.columns.keys())[0]
        result.columns[b_col_name] = col
        return result
    finally:
        if a_data != NULL: os_aligned_free(a_data)
        if b_data != NULL: os_aligned_free(b_data)
        if lu != NULL: os_aligned_free(lu)
        if x != NULL: os_aligned_free(x)

cdef double _linalg_norm(FastDataFrame df, str ord_type="frobenius"):
    cdef Py_ssize_t rows, cols, n, i
    if df.df_length == 0:
        return 0.0

    cdef double* A = NULL
    cdef double result = 0.0
    cdef double val
    cdef int mode = 0

    if ord_type in ('frobenius', 'fro', '2'):
        mode = 1
    elif ord_type in ('count', 'zero'):
        mode = 2
    else:
        mode = 3

    try:
        A = _linalg_to_contiguous(df, &rows, &cols)
        n = rows * cols
        with nogil:
            if mode == 1:
                result = 0.0
                for i in range(n):
                    result += A[i] * A[i]
                result = sqrt(result)
            elif mode == 2:
                result = 0.0
                for i in range(n):
                    if fabs(A[i]) > 1e-15:
                        result += 1.0
            else:
                result = 0.0
                for i in range(n):
                    val = fabs(A[i])
                    if val > result:
                        result = val
        return result
    finally:
        if A != NULL: os_aligned_free(A)

# ============================================================
#  Pairwise distance / similarity metrics
# ============================================================

cdef FastDataFrame _linalg_euclidean_dist(FastDataFrame df):
    cdef Py_ssize_t rows = 0, cols = 0
    if df.df_length == 0: return FastDataFrame()

    cdef Py_ssize_t i, j
    cdef double* A = NULL
    cdef double* D = NULL
    try:
        A = _linalg_to_contiguous_fast(df, &rows, &cols)
        D = <double*>aligned_malloc(rows * rows * sizeof(double))
        if not D: raise MemoryError()
        with nogil:
            for i in range(rows):
                D[i * rows + i] = 0.0
                for j in range(i + 1, rows):
                    D[i * rows + j] = _vec_euclidean_dist(A + i * cols, A + j * cols, cols)
                    D[j * rows + i] = D[i * rows + j]
        return _linalg_from_contiguous(D, rows, rows)
    finally:
        if A != NULL: os_aligned_free(A)
        if D != NULL: os_aligned_free(D)

cdef FastDataFrame _linalg_manhattan_dist(FastDataFrame df):
    cdef Py_ssize_t rows = 0, cols = 0
    if df.df_length == 0: return FastDataFrame()

    cdef Py_ssize_t i, j
    cdef double* A = NULL
    cdef double* D = NULL
    try:
        A = _linalg_to_contiguous_fast(df, &rows, &cols)
        D = <double*>aligned_malloc(rows * rows * sizeof(double))
        if not D: raise MemoryError()
        with nogil:
            for i in range(rows):
                D[i * rows + i] = 0.0
                for j in range(i + 1, rows):
                    D[i * rows + j] = _vec_manhattan_dist(A + i * cols, A + j * cols, cols)
                    D[j * rows + i] = D[i * rows + j]
        return _linalg_from_contiguous(D, rows, rows)
    finally:
        if A != NULL: os_aligned_free(A)
        if D != NULL: os_aligned_free(D)

cdef FastDataFrame _linalg_cosine_dist(FastDataFrame df):
    cdef Py_ssize_t rows = 0, cols = 0
    if df.df_length == 0: return FastDataFrame()

    cdef Py_ssize_t i, j
    cdef double* A = NULL
    cdef double* D = NULL
    try:
        A = _linalg_to_contiguous_fast(df, &rows, &cols)
        D = <double*>aligned_malloc(rows * rows * sizeof(double))
        if not D: raise MemoryError()
        with nogil:
            for i in range(rows):
                D[i * rows + i] = 0.0
                for j in range(i + 1, rows):
                    D[i * rows + j] = 1.0 - _vec_cosine_sim(A + i * cols, A + j * cols, cols)
                    D[j * rows + i] = D[i * rows + j]
        return _linalg_from_contiguous(D, rows, rows)
    finally:
        if A != NULL: os_aligned_free(A)
        if D != NULL: os_aligned_free(D)

cdef FastDataFrame _linalg_hamming_dist(FastDataFrame df):
    cdef Py_ssize_t rows = 0, cols = 0
    if df.df_length == 0: return FastDataFrame()

    cdef Py_ssize_t i, j
    cdef double* A = NULL
    cdef double* D = NULL
    try:
        A = _linalg_to_contiguous_fast(df, &rows, &cols)
        D = <double*>aligned_malloc(rows * rows * sizeof(double))
        if not D: raise MemoryError()
        with nogil:
            for i in range(rows):
                D[i * rows + i] = 0.0
                for j in range(i + 1, rows):
                    D[i * rows + j] = _vec_hamming_dist(A + i * cols, A + j * cols, cols)
                    D[j * rows + i] = D[i * rows + j]
        return _linalg_from_contiguous(D, rows, rows)
    finally:
        if A != NULL: os_aligned_free(A)
        if D != NULL: os_aligned_free(D)

cdef FastDataFrame _linalg_jaccard_sim(FastDataFrame df):
    cdef Py_ssize_t rows = 0, cols = 0
    if df.df_length == 0: return FastDataFrame()

    cdef Py_ssize_t i, j
    cdef double* A = NULL
    cdef double* D = NULL
    try:
        A = _linalg_to_contiguous_fast(df, &rows, &cols)
        D = <double*>aligned_malloc(rows * rows * sizeof(double))
        if not D: raise MemoryError()
        with nogil:
            for i in range(rows):
                D[i * rows + i] = 1.0
                for j in range(i + 1, rows):
                    D[i * rows + j] = _vec_jaccard_sim(A + i * cols, A + j * cols, cols)
                    D[j * rows + i] = D[i * rows + j]
        return _linalg_from_contiguous(D, rows, rows)
    finally:
        if A != NULL: os_aligned_free(A)
        if D != NULL: os_aligned_free(D)

cdef FastDataFrame _linalg_cosine_similarity(FastDataFrame df):
    cdef Py_ssize_t rows = 0, cols = 0
    if df.df_length == 0: return FastDataFrame()

    cdef Py_ssize_t i, j
    cdef double* A = NULL
    cdef double* D = NULL
    try:
        A = _linalg_to_contiguous_fast(df, &rows, &cols)
        D = <double*>aligned_malloc(rows * rows * sizeof(double))
        if not D: raise MemoryError()
        with nogil:
            for i in range(rows):
                D[i * rows + i] = 1.0
                for j in range(i + 1, rows):
                    D[i * rows + j] = _vec_cosine_sim(A + i * cols, A + j * cols, cols)
                    D[j * rows + i] = D[i * rows + j]
        return _linalg_from_contiguous(D, rows, rows)
    finally:
        if A != NULL: os_aligned_free(A)
        if D != NULL: os_aligned_free(D)

# ============================================================
# _DataFrameLinalgAccessor — lazy-cached accessor
# ============================================================

cdef class _DataFrameLinalgAccessor:

    def __cinit__(self):
        pass

    # ==================== Matrix Operations ====================

    @cython.binding(False)
    def dot(self, FastDataFrame b):
        return _linalg_dot(self._df, b)

    @cython.binding(False)
    def matmul(self, FastDataFrame b):
        return _linalg_dot(self._df, b)

    @cython.binding(False)
    def transpose(self):
        return _linalg_transpose(self._df)

    @cython.binding(False)
    def det(self):
        return _linalg_det(self._df)

    @cython.binding(False)
    def inv(self):
        return _linalg_inv(self._df)

    @cython.binding(False)
    def solve(self, FastDataFrame b):
        return _linalg_solve(self._df, b)

    @cython.binding(False)
    def norm(self, str ord_type="frobenius"):
        return _linalg_norm(self._df, ord_type)

    # ==================== Pairwise Distance ====================

    @cython.binding(False)
    def euclidean_dist(self):
        return _linalg_euclidean_dist(self._df)

    @cython.binding(False)
    def manhattan_dist(self):
        return _linalg_manhattan_dist(self._df)

    @cython.binding(False)
    def cosine_dist(self):
        return _linalg_cosine_dist(self._df)

    @cython.binding(False)
    def hamming_dist(self):
        return _linalg_hamming_dist(self._df)

    @cython.binding(False)
    def jaccard_sim(self):
        return _linalg_jaccard_sim(self._df)

    @cython.binding(False)
    def cosine_similarity(self):
        return _linalg_cosine_similarity(self._df)

    @cython.binding(False)
    def __repr__(self):
        return f"<LinalgAccessor: {len(self._df.columns)} columns, {self._df.df_length} rows>"

cdef inline _DataFrameLinalgAccessor _make_linalg_accessor(FastDataFrame df):
    cdef _DataFrameLinalgAccessor acc = <_DataFrameLinalgAccessor>_DataFrameLinalgAccessor.__new__(_DataFrameLinalgAccessor)
    acc._df = df
    return acc