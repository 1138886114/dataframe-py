# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport cython
from libc.string cimport memcpy, memset
from libc.math cimport sqrt, fabs, NAN
from libc.stdlib cimport malloc, free

from fluidframe.core._base cimport BaseColumn, TYPE_INT, TYPE_DOUBLE, aligned_malloc, os_aligned_free
from fluidframe.core.fastdataframe cimport FastDataFrame
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.doublecolumn cimport DoubleColumn, _c_euclidean_dist, _c_manhattan_dist, _c_cosine_dist, _c_cosine_similarity, _c_jaccard_sim, _c_hamming_dist
from fluidframe.core.index cimport RangeIndex


# ============================================================
#  Conversion: DataFrame <-> Contiguous Row-Major double*
# ============================================================

cdef double* _to_contiguous(FastDataFrame df, Py_ssize_t* out_rows, Py_ssize_t* out_cols) except NULL:
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


cdef FastDataFrame _from_contiguous(double* data, Py_ssize_t rows, Py_ssize_t cols):
    """Convert contiguous row-major double* array to FastDataFrame."""
    cdef FastDataFrame df = FastDataFrame()
    df.df_length = rows
    df.index = RangeIndex(0, rows)
    cdef list col_names = [chr(65 + i) if i < 26 else str(i) for i in range(cols)]
    cdef Py_ssize_t i, j
    cdef DoubleColumn col
    for j in range(cols):
        col = DoubleColumn(rows)
        col.length = rows
        for i in range(rows):
            col.data_c[i] = data[i * cols + j]
        df.columns[col_names[j]] = col
    return df


cdef inline double clip_val(double v, double min_v, double max_v) nogil:
    if v < min_v: return min_v
    if v > max_v: return max_v
    return v

# ============================================================
#  Core linalg operations
# ============================================================

def dot(FastDataFrame a, FastDataFrame b):
    """Matrix multiplication: C = A x B (row-major)."""
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
        A = _to_contiguous(a, &a_rows, &a_cols)
        B = _to_contiguous(b, &b_rows, &b_cols)

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

        return _from_contiguous(C, a_rows, b_cols)
    finally:
        if A != NULL: os_aligned_free(A)
        if B != NULL: os_aligned_free(B)
        if C != NULL: os_aligned_free(C)


def transpose(FastDataFrame df):
    cdef Py_ssize_t rows, cols, i, j
    cdef Py_ssize_t size
    cdef double* A = NULL
    cdef double* B = NULL

    try:
        A = _to_contiguous(df, &rows, &cols)
        size = rows * cols
        if size == 0: size = 1
        B = <double*>aligned_malloc(size * sizeof(double))
        if not B: raise MemoryError()

        with nogil:
            for i in range(rows):
                for j in range(cols):
                    B[j * rows + i] = A[i * cols + j]

        return _from_contiguous(B, cols, rows)
    finally:
        if A != NULL: os_aligned_free(A)
        if B != NULL: os_aligned_free(B)


def det(FastDataFrame df):
    """Determinant via LU decomposition with Partial Pivoting."""
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
        A = _to_contiguous(df, &rows, &cols)
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


def inv(FastDataFrame df):
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
        A = _to_contiguous(df, &rows, &cols)
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

        return _from_contiguous(result_data, n, n)
    finally:
        if A != NULL: os_aligned_free(A)
        if aug != NULL: os_aligned_free(aug)
        if result_data != NULL: os_aligned_free(result_data)


def solve(FastDataFrame A, FastDataFrame b):
    if A.df_length != b.df_length:
        raise ValueError("Shape mismatch")
    return dot(inv(A), b)


def matmul(FastDataFrame a, FastDataFrame b):
    return dot(a, b)


def norm(FastDataFrame df, str ord_type="frobenius"):
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
        A = _to_contiguous(df, &rows, &cols)
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
def euclidean_dist(FastDataFrame df):
    """Pairwise Euclidean distance between rows."""
    cdef Py_ssize_t rows = 0, cols = 0
    cdef Py_ssize_t i, j
    cdef double* A = NULL
    cdef double* D = NULL

    try:
        A = _to_contiguous(df, &rows, &cols)
        D = <double*>aligned_malloc(max(rows * rows, 1) * sizeof(double))
        if not D: raise MemoryError()

        with nogil:
            for i in range(rows):
                D[i * rows + i] = 0.0
                for j in range(i + 1, rows):
                    D[i * rows + j] = _c_euclidean_dist(A + i * cols, A + j * cols, cols)
                    D[j * rows + i] = D[i * rows + j]

        return _from_contiguous(D, rows, rows)
    finally:
        if A != NULL: os_aligned_free(A)
        if D != NULL: os_aligned_free(D)


def manhattan_dist(FastDataFrame df):
    """Pairwise Manhattan distance between rows."""
    cdef Py_ssize_t rows = 0, cols = 0
    cdef Py_ssize_t i, j
    cdef double* A = NULL
    cdef double* D = NULL

    try:
        A = _to_contiguous(df, &rows, &cols)
        D = <double*>aligned_malloc(max(rows * rows, 1) * sizeof(double))
        if not D: raise MemoryError()

        with nogil:
            for i in range(rows):
                D[i * rows + i] = 0.0
                for j in range(i + 1, rows):
                    D[i * rows + j] = _c_manhattan_dist(A + i * cols, A + j * cols, cols)
                    D[j * rows + i] = D[i * rows + j]

        return _from_contiguous(D, rows, rows)
    finally:
        if A != NULL: os_aligned_free(A)
        if D != NULL: os_aligned_free(D)


def cosine_dist(FastDataFrame df):
    """Pairwise cosine distance (1 - cosine_similarity) between rows."""
    cdef Py_ssize_t rows = 0, cols = 0
    cdef Py_ssize_t i, j
    cdef double* A = NULL
    cdef double* D = NULL

    try:
        A = _to_contiguous(df, &rows, &cols)
        D = <double*>aligned_malloc(max(rows * rows, 1) * sizeof(double))
        if not D: raise MemoryError()

        with nogil:
            for i in range(rows):
                D[i * rows + i] = 0.0
                for j in range(i + 1, rows):
                    D[i * rows + j] = _c_cosine_dist(A + i * cols, A + j * cols, cols)
                    D[j * rows + i] = D[i * rows + j]

        return _from_contiguous(D, rows, rows)
    finally:
        if A != NULL: os_aligned_free(A)
        if D != NULL: os_aligned_free(D)


def hamming_dist(FastDataFrame df):
    """Pairwise Hamming distance (fraction of differing elements) between rows."""
    cdef Py_ssize_t rows = 0, cols = 0
    cdef Py_ssize_t i, j
    cdef double* A = NULL
    cdef double* D = NULL

    try:
        A = _to_contiguous(df, &rows, &cols)
        D = <double*>aligned_malloc(max(rows * rows, 1) * sizeof(double))
        if not D: raise MemoryError()

        with nogil:
            for i in range(rows):
                D[i * rows + i] = 0.0
                for j in range(i + 1, rows):
                    D[i * rows + j] = _c_hamming_dist(A + i * cols, A + j * cols, cols)
                    D[j * rows + i] = D[i * rows + j]

        return _from_contiguous(D, rows, rows)
    finally:
        if A != NULL: os_aligned_free(A)
        if D != NULL: os_aligned_free(D)


def jaccard_sim(FastDataFrame df):
    """Pairwise Jaccard similarity for binary data (non-zero = 1)."""
    cdef Py_ssize_t rows = 0, cols = 0
    cdef Py_ssize_t i, j
    cdef double* A = NULL
    cdef double* D = NULL

    try:
        A = _to_contiguous(df, &rows, &cols)
        D = <double*>aligned_malloc(max(rows * rows, 1) * sizeof(double))
        if not D: raise MemoryError()

        with nogil:
            for i in range(rows):
                D[i * rows + i] = 1.0
                for j in range(i + 1, rows):
                    D[i * rows + j] = _c_jaccard_sim(A + i * cols, A + j * cols, cols)
                    D[j * rows + i] = D[i * rows + j]

        return _from_contiguous(D, rows, rows)
    finally:
        if A != NULL: os_aligned_free(A)
        if D != NULL: os_aligned_free(D)


def cosine_similarity(FastDataFrame df):
    """Pairwise cosine similarity between rows."""
    cdef Py_ssize_t rows = 0, cols = 0
    cdef Py_ssize_t i, j
    cdef double* A = NULL
    cdef double* D = NULL

    try:
        A = _to_contiguous(df, &rows, &cols)
        D = <double*>aligned_malloc(max(rows * rows, 1) * sizeof(double))
        if not D: raise MemoryError()

        with nogil:
            for i in range(rows):
                D[i * rows + i] = 1.0
                for j in range(i + 1, rows):
                    D[i * rows + j] = _c_cosine_similarity(A + i * cols, A + j * cols, cols)
                    D[j * rows + i] = D[i * rows + j]

        return _from_contiguous(D, rows, rows)
    finally:
        if A != NULL: os_aligned_free(A)
        if D != NULL: os_aligned_free(D)