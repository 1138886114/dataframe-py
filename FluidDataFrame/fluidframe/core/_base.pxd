# cython: language_level=3
from libc.math cimport sqrt, fabs, NAN
from libc.string cimport memmove, memcpy, memset

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
#  Shift helpers
# ============================================================
cdef void _c_shift_right(void* buf, Py_ssize_t pos, Py_ssize_t col_len, Py_ssize_t elem_size) noexcept nogil
cdef void _c_shift_left(void* buf, Py_ssize_t pos, Py_ssize_t col_len, Py_ssize_t elem_size) noexcept nogil

# ============================================================
#  Generic column helpers — void* + elem_size
#  Used by all column types to eliminate duplicated boilerplate.
# ============================================================
cdef void* _col_pop_back(void* data_c, Py_ssize_t* length, Py_ssize_t elem_size) noexcept nogil
cdef void _col_pop_front(void* data_c, Py_ssize_t* length, Py_ssize_t elem_size) noexcept nogil
cdef void _col_concat(void* data_c, Py_ssize_t* length, void* other_data, Py_ssize_t other_len, Py_ssize_t elem_size) noexcept nogil
cdef void _col_drop_row(void* data_c, Py_ssize_t* length, Py_ssize_t row_idx, Py_ssize_t elem_size) noexcept nogil
cdef void _col_gather(void* dst, void* src, Py_ssize_t* indices, Py_ssize_t n, Py_ssize_t elem_size) noexcept nogil
cdef Py_ssize_t _col_reserve(void** data_c, Py_ssize_t* capacity, Py_ssize_t new_cap, Py_ssize_t old_size, Py_ssize_t elem_size) except -1 nogil

# ============================================================
#  Fused numeric type for vectorized math
# ============================================================
ctypedef fused NumericType:
    int
    long
    double
    long long

# ============================================================
#  Vectorized math: same-type (int/int, double/double, etc.)
# ============================================================
cdef double _vec_pearson_corr(NumericType* x, NumericType* y, Py_ssize_t n) noexcept nogil
cdef double _vec_pearson_cov(NumericType* x, NumericType* y, Py_ssize_t n) noexcept nogil
cdef double _vec_euclidean_dist(NumericType* x, NumericType* y, Py_ssize_t n) noexcept nogil
cdef double _vec_manhattan_dist(NumericType* x, NumericType* y, Py_ssize_t n) noexcept nogil
cdef double _vec_cosine_sim(NumericType* x, NumericType* y, Py_ssize_t n) noexcept nogil
cdef double _vec_jaccard_sim(NumericType* x, NumericType* y, Py_ssize_t n) noexcept nogil
cdef double _vec_hamming_dist(NumericType* x, NumericType* y, Py_ssize_t n) noexcept nogil

# ============================================================
#  Vectorized math: int* / NumericType cross-type
# ============================================================
cdef double _int_double_pearson_corr(int* x, NumericType* y, Py_ssize_t n) noexcept nogil
cdef double _int_double_pearson_cov(int* x, NumericType* y, Py_ssize_t n) noexcept nogil
cdef double _int_double_euclidean_dist(int* x, NumericType* y, Py_ssize_t n) noexcept nogil
cdef double _int_double_manhattan_dist(int* x, NumericType* y, Py_ssize_t n) noexcept nogil
cdef double _int_double_cosine_sim(int* x, NumericType* y, Py_ssize_t n) noexcept nogil
cdef double _int_double_jaccard_sim(int* x, NumericType* y, Py_ssize_t n) noexcept nogil
cdef double _int_double_hamming_dist(int* x, NumericType* y, Py_ssize_t n) noexcept nogil

# ============================================================
#  Vectorized math: double* / NumericType cross-type
# ============================================================
cdef double _double_int_pearson_corr(double* x, NumericType* y, Py_ssize_t n) noexcept nogil
cdef double _double_int_pearson_cov(double* x, NumericType* y, Py_ssize_t n) noexcept nogil
cdef double _double_int_euclidean_dist(double* x, NumericType* y, Py_ssize_t n) noexcept nogil
cdef double _double_int_manhattan_dist(double* x, NumericType* y, Py_ssize_t n) noexcept nogil
cdef double _double_int_cosine_sim(double* x, NumericType* y, Py_ssize_t n) noexcept nogil
cdef double _double_int_jaccard_sim(double* x, NumericType* y, Py_ssize_t n) noexcept nogil
cdef double _double_int_hamming_dist(double* x, NumericType* y, Py_ssize_t n) noexcept nogil
