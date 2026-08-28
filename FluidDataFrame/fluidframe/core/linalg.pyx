# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport cython
from libc.stdlib cimport malloc, free, calloc
from libc.string cimport memcpy
from libc.math cimport sqrt, fabs, NAN

from fluidframe.core._base cimport BaseColumn, TYPE_INT, TYPE_DOUBLE
from fluidframe.core.fastdataframe cimport FastDataFrame
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.doublecolumn cimport DoubleColumn
from fluidframe.core.index cimport RangeIndex


# ============================================================
#  Conversion: DataFrame <-> Contiguous Row-Major double*
# ============================================================

cdef double* _to_contiguous(FastDataFrame df, Py_ssize_t* out_rows, Py_ssize_t* out_cols) except NULL:
    """Convert DataFrame to a single contiguous row-major double* array.
    Caller must free() the returned pointer."""
    cdef Py_ssize_t rows = df.df_length
    cdef list col_names = list(df.columns.keys())
    cdef Py_ssize_t cols = len(col_names)
    cdef double* data = <double*>calloc(rows * cols, sizeof(double))
    if not data:
        raise MemoryError()
    cdef Py_ssize_t i, j
    cdef BaseColumn col
    for j in range(cols):
        col = df.columns[col_names[j]]
        for i in range(rows):
            if col.dtype == TYPE_INT:
                data[i * cols + j] = <double>(<IntColumn>col).data_c[i]
            elif col.dtype == TYPE_DOUBLE:
                data[i * cols + j] = (<DoubleColumn>col).data_c[i]
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


# ============================================================
#  Core linalg operations
# ============================================================

def dot(FastDataFrame a, FastDataFrame b):
    """Matrix multiplication: C = A x B (row-major)."""
    cdef Py_ssize_t a_rows, a_cols, b_rows, b_cols
    cdef Py_ssize_t i, j, k
    cdef double s
    cdef list res_col_names
    cdef DoubleColumn col
    cdef FastDataFrame result

    a_rows = a.df_length
    a_cols = len(list(a.columns.keys()))
    b_rows = b.df_length
    b_cols = len(list(b.columns.keys()))

    if a_cols != b_rows:
        raise ValueError(f"Shape mismatch: ({a_rows}, {a_cols}) x ({b_rows}, {b_cols})")

    cdef double* A = _to_contiguous(a, &a_rows, &a_cols)
    cdef double* B = _to_contiguous(b, &b_rows, &b_cols)
    cdef double* C = <double*>calloc(a_rows * b_cols, sizeof(double))
    if not C:
        free(A)
        free(B)
        raise MemoryError()

    try:
        for i in range(a_rows):
            for k in range(a_cols):
                s = A[i * a_cols + k]
                for j in range(b_cols):
                    C[i * b_cols + j] += s * B[k * b_cols + j]
    finally:
        free(A)
        free(B)

    result = _from_contiguous(C, a_rows, b_cols)
    free(C)
    return result


def transpose(FastDataFrame df):
    """Matrix transpose: B = A^T."""
    cdef Py_ssize_t rows, cols
    cdef Py_ssize_t i, j

    cdef double* A = _to_contiguous(df, &rows, &cols)
    cdef double* B = <double*>malloc(cols * rows * sizeof(double))
    if not B:
        free(A)
        raise MemoryError()

    try:
        for i in range(rows):
            for j in range(cols):
                B[j * rows + i] = A[i * cols + j]
    finally:
        free(A)

    cdef FastDataFrame result = _from_contiguous(B, cols, rows)
    free(B)
    return result


def det(FastDataFrame df):
    """Determinant via LU decomposition."""
    cdef Py_ssize_t n = df.df_length
    cdef list col_names = list(df.columns.keys())
    if len(col_names) != n:
        raise ValueError("Matrix must be square")

    cdef Py_ssize_t rows, cols
    cdef double* A = _to_contiguous(df, &rows, &cols)

    cdef double* LU = <double*>malloc(n * n * sizeof(double))
    if not LU:
        free(A)
        raise MemoryError()

    cdef Py_ssize_t i, j, k
    cdef double det_val = 1.0
    cdef double pivot, temp

    try:
        memcpy(LU, A, n * n * sizeof(double))

        for i in range(n):
            pivot = LU[i * n + i]
            if fabs(pivot) < 1e-15:
                free(LU)
                free(A)
                return 0.0
            for k in range(i + 1, n):
                temp = LU[k * n + i] / pivot
                for j in range(i + 1, n):
                    LU[k * n + j] -= temp * LU[i * n + j]

        for i in range(n):
            det_val *= LU[i * n + i]
    finally:
        free(LU)
        free(A)

    return det_val


def inv(FastDataFrame df):
    """Matrix inverse via Gauss-Jordan elimination."""
    cdef Py_ssize_t n = df.df_length
    cdef list col_names = list(df.columns.keys())
    cdef Py_ssize_t rows, cols
    cdef Py_ssize_t i, j, k
    cdef double pivot, temp
    cdef double* result_data
    cdef FastDataFrame result

    if len(col_names) != n:
        raise ValueError("Matrix must be square")

    cdef double* A = _to_contiguous(df, &rows, &cols)
    cdef double* aug = <double*>malloc(n * 2 * n * sizeof(double))
    if not aug:
        free(A)
        raise MemoryError()

    try:
        for i in range(n):
            for j in range(n):
                aug[i * 2 * n + j] = A[i * n + j]
                aug[i * 2 * n + n + j] = 1.0 if i == j else 0.0

        for i in range(n):
            pivot = aug[i * 2 * n + i]
            if fabs(pivot) < 1e-15:
                free(aug)
                free(A)
                raise ValueError("Matrix is singular")
            for j in range(2 * n):
                aug[i * 2 * n + j] /= pivot
            for k in range(n):
                if k == i: continue
                temp = aug[k * 2 * n + i]
                for j in range(2 * n):
                    aug[k * 2 * n + j] -= temp * aug[i * 2 * n + j]

        result_data = <double*>malloc(n * n * sizeof(double))
        if not result_data:
            free(aug)
            free(A)
            raise MemoryError()
        for i in range(n):
            for j in range(n):
                result_data[i * n + j] = aug[i * 2 * n + n + j]

        result = _from_contiguous(result_data, n, n)
        free(result_data)
    finally:
        free(aug)
        free(A)

    return result


def solve(FastDataFrame A, FastDataFrame b):
    """Solve linear system Ax = b."""
    if A.df_length != b.df_length:
        raise ValueError("Shape mismatch")
    A_inv = inv(A)
    return dot(A_inv, b)


def matmul(FastDataFrame a, FastDataFrame b):
    """Matrix multiplication (alias for dot)."""
    return dot(a, b)


def norm(FastDataFrame df, str ord_type="frobenius"):
    """Matrix norm.
    ord_type='frobenius' or 'fro' — Frobenius norm (sqrt of sum of squares)
    ord_type='max' or 'inf' — max absolute value
    ord_type='count' or 'zero' — count of non-zero elements
    """
    cdef Py_ssize_t rows, cols
    cdef double* A = _to_contiguous(df, &rows, &cols)
    cdef Py_ssize_t n = rows * cols
    cdef Py_ssize_t i
    cdef double val, result

    try:
        if ord_type in ('frobenius', 'fro', '2'):
            result = 0
            for i in range(n):
                result += A[i] * A[i]
            result = sqrt(result)
        elif ord_type in ('count', 'zero'):
            result = 0
            for i in range(n):
                if fabs(A[i]) > 1e-15:
                    result += 1
        else:
            result = 0
            for i in range(n):
                val = fabs(A[i])
                if val > result:
                    result = val
    finally:
        free(A)

    return result
