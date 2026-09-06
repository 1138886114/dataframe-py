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

cdef bint _linalg_is_contiguous_columns(FastDataFrame df):
    cdef list cols_list = list(df.columns.values())
    cdef Py_ssize_t cols = len(cols_list)
    if cols < 2:
        return False
    cdef BaseColumn col0 = <BaseColumn>cols_list[0]
    cdef BaseColumn col1 = <BaseColumn>cols_list[1]
    if col0.dtype != TYPE_DOUBLE or col1.dtype != TYPE_DOUBLE:
        return False
    cdef double* ptr0 = (<DoubleColumn>col0).data_c
    cdef double* ptr1 = (<DoubleColumn>col1).data_c
    cdef Py_ssize_t rows = df.df_length
    return (ptr0 + rows) == ptr1


cdef double* _linalg_to_contiguous_fast(FastDataFrame df, Py_ssize_t* out_rows, Py_ssize_t* out_cols) except NULL:
    cdef Py_ssize_t rows = df.df_length
    cdef list cols_list = list(df.columns.values())
    cdef Py_ssize_t cols = len(cols_list)
    if cols == 0:
        out_rows[0] = 0
        out_cols[0] = 0
        return NULL
    cdef Py_ssize_t num_elements = rows * cols
    if num_elements == 0: num_elements = 1
    cdef double* data = <double*>aligned_malloc(num_elements * sizeof(double))
    if not data: raise MemoryError()
    cdef Py_ssize_t i, j
    cdef BaseColumn col
    try:
        for j in range(cols):
            col = <BaseColumn>cols_list[j]
            if col.dtype != TYPE_DOUBLE:
                os_aligned_free(data)
                return _linalg_to_contiguous(df, out_rows, out_cols)
            with nogil:
                for i in range(rows):
                    data[i * cols + j] = (<DoubleColumn>col).data_c[i]
    except Exception:
        os_aligned_free(data)
        raise
    out_rows[0] = rows
    out_cols[0] = cols
    return data


cdef double* _linalg_to_contiguous(FastDataFrame df, Py_ssize_t* out_rows, Py_ssize_t* out_cols) except NULL:
    cdef Py_ssize_t rows = df.df_length
    cdef list cols_list = list(df.columns.values())
    cdef list name_list = list(df.columns.keys())
    cdef Py_ssize_t cols = len(cols_list)
    cdef Py_ssize_t num_elements = rows * cols
    if num_elements == 0: num_elements = 1
    cdef double* data = <double*>aligned_malloc(num_elements * sizeof(double))
    if not data: raise MemoryError()
    cdef Py_ssize_t i, j
    cdef BaseColumn col
    cdef int col_dtype
    cdef int* int_ptr
    cdef double* double_ptr
    try:
        for j in range(cols):
            col = <BaseColumn>cols_list[j]
            col_dtype = col.dtype
            if col_dtype == TYPE_INT:
                int_ptr = (<IntColumn>col).data_c
                with nogil:
                    for i in range(rows):
                        data[i * cols + j] = <double>int_ptr[i]
            elif col_dtype == TYPE_DOUBLE:
                double_ptr = (<DoubleColumn>col).data_c
                with nogil:
                    for i in range(rows):
                        data[i * cols + j] = double_ptr[i]
            else:
                raise TypeError(f"Cannot perform linear algebra on non-numeric column '{name_list[j]}'")
    except Exception:
        os_aligned_free(data)
        raise
    out_rows[0] = rows
    out_cols[0] = cols
    return data


cdef FastDataFrame _linalg_from_contiguous(double* data, Py_ssize_t rows, Py_ssize_t cols):
    cdef FastDataFrame df = FastDataFrame()
    df.df_length = rows
    df.index = RangeIndex(0, rows)
    cdef list col_names = [chr(65 + i) if i < 26 else str(i) for i in range(cols)]
    cdef Py_ssize_t i, j
    cdef DoubleColumn col
    for j in range(cols):
        col = DoubleColumn(max(rows, 1))
        col.length = rows
        for i in range(rows):
            col.data_c[i] = data[i * cols + j]
        df.columns[col_names[j]] = col
    return df


cdef inline double _linalg_clip_val(double v, double min_v, double max_v) nogil:
    if v < min_v: return min_v
    if v > max_v: return max_v
    return v


# ============================================================
#  Core linalg operations
# ============================================================

cdef FastDataFrame _linalg_dot(FastDataFrame a, FastDataFrame b):
    cdef Py_ssize_t a_rows, a_cols, b_rows, b_cols
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t size_c
    cdef double s
    cdef double* A = NULL
    cdef double* B = NULL
    cdef double* C = NULL
    a_rows = a.df_length
    a_cols = len(list(a.columns.keys()))
    b_rows = b.df_length
    b_cols = len(list(b.columns.keys()))
    if a_cols != b_rows:
        raise ValueError(f"Shape mismatch: ({a_rows}, {a_cols}) x ({b_rows}, {b_cols})")
    try:
        A = _linalg_to_contiguous(a, &a_rows, &a_cols)
        B = _linalg_to_contiguous(b, &b_rows, &b_cols)
        size_c = a_rows * b_cols
        if size_c == 0: size_c = 1
        C = <double*>aligned_malloc(size_c * sizeof(double))
        if not C: raise MemoryError()
        memset(C, 0, size_c * sizeof(double))
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
    try:
        A = _linalg_to_contiguous(df, &rows, &cols)
        size = rows * cols
        if size == 0: size = 1
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
    cdef Py_ssize_t rows, cols, i, j, k, pivot_row
    cdef Py_ssize_t size
    cdef double* A = NULL
    cdef double* LU = NULL
    cdef double det_val = 1.0
    cdef double pivot, temp, max_val, abs_val
    cdef int sign = 1
    if len(list(df.columns.keys())) != n:
        raise ValueError("Matrix must be square")
    try:
        A = _linalg_to_contiguous(df, &rows, &cols)
        size = n * n
        if size == 0: size = 1
        LU = <double*>aligned_malloc(size * sizeof(double))
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
    cdef Py_ssize_t rows, cols, i, j, k, pivot_row
    cdef Py_ssize_t size_aug, size_res
    cdef double* A = NULL
    cdef double* aug = NULL
    cdef double* result_data = NULL
    cdef double pivot, temp, max_val, abs_val
    cdef int singular = 0
    if len(list(df.columns.keys())) != n:
        raise ValueError("Matrix must be square")
    try:
        A = _linalg_to_contiguous(df, &rows, &cols)
        size_aug = n * 2 * n
        if size_aug == 0: size_aug = 1
        aug = <double*>aligned_malloc(size_aug * sizeof(double))
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
        size_res = n * n
        if size_res == 0: size_res = 1
        result_data = <double*>aligned_malloc(size_res * sizeof(double))
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
    cdef Py_ssize_t rows, cols, i, j, k, pivot_row
    cdef double* a_data = NULL
    cdef double* b_data = NULL
    cdef double* lu = NULL
    cdef double* x = NULL
    cdef double pivot, temp, max_val, abs_val, sum_val
    cdef int sign = 1
    cdef list b_cols
    if len(list(A_df.columns.keys())) != n:
        raise ValueError("Matrix A must be square")
    if n != b_df.df_length:
        raise ValueError("Shape mismatch: A and b must have same number of rows")
    b_cols = list(b_df.columns.keys())
    if len(b_cols) != 1:
        raise ValueError("b must have exactly one column")
    try:
        a_data = _linalg_to_contiguous(A_df, &rows, &cols)
        b_data = _linalg_to_contiguous(b_df, &rows, &cols)
        lu = <double*>aligned_malloc(max(n * n, 1) * sizeof(double))
        x = <double*>aligned_malloc(max(n, 1) * sizeof(double))
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
        result = FastDataFrame()
        result.df_length = n
        col = DoubleColumn(max(n, 1))
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
    cdef Py_ssize_t i, j
    cdef double* A = NULL
    cdef double* D = NULL
    try:
        A = _linalg_to_contiguous_fast(df, &rows, &cols)
        D = <double*>aligned_malloc(max(rows * rows, 1) * sizeof(double))
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
    cdef Py_ssize_t i, j
    cdef double* A = NULL
    cdef double* D = NULL
    try:
        A = _linalg_to_contiguous_fast(df, &rows, &cols)
        D = <double*>aligned_malloc(max(rows * rows, 1) * sizeof(double))
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
    cdef Py_ssize_t i, j
    cdef double* A = NULL
    cdef double* D = NULL
    try:
        A = _linalg_to_contiguous_fast(df, &rows, &cols)
        D = <double*>aligned_malloc(max(rows * rows, 1) * sizeof(double))
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
    cdef Py_ssize_t i, j
    cdef double* A = NULL
    cdef double* D = NULL
    try:
        A = _linalg_to_contiguous_fast(df, &rows, &cols)
        D = <double*>aligned_malloc(max(rows * rows, 1) * sizeof(double))
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
    cdef Py_ssize_t i, j
    cdef double* A = NULL
    cdef double* D = NULL
    try:
        A = _linalg_to_contiguous_fast(df, &rows, &cols)
        D = <double*>aligned_malloc(max(rows * rows, 1) * sizeof(double))
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
    cdef Py_ssize_t i, j
    cdef double* A = NULL
    cdef double* D = NULL
    try:
        A = _linalg_to_contiguous_fast(df, &rows, &cols)
        D = <double*>aligned_malloc(max(rows * rows, 1) * sizeof(double))
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
