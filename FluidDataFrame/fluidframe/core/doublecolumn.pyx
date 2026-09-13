# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False

cimport cython
from libc.string cimport memcpy, memmove, memset
from libc.stdio cimport snprintf
from libc.math cimport sqrt, fabs, NAN, fmax
from libc.stdint cimport int32_t, int64_t, uint64_t
from libc.stdlib cimport qsort, bsearch

from fluidframe.core._base cimport (
    BaseColumn, aligned_malloc, aligned_realloc, os_aligned_free,
    TYPE_DOUBLE, TYPE_INT, TYPE_CHAR256, TYPE_UCS2, TYPE_BOOL,
    TYPE_DATE, TYPE_CATEGORY, TYPE_DATETIME, TYPE_TIME,
    _resolve_astype_target, NumericType,
    _vec_pearson_corr, _vec_pearson_cov, _vec_euclidean_dist,
    _vec_manhattan_dist, _vec_cosine_sim, _vec_jaccard_sim, _vec_hamming_dist,
    _c_shift_left, _c_shift_right, _col_pop_back, _col_pop_front, _col_concat,
    _col_drop_row, _col_gather,
    _col_shift_right_2d_double, _col_shift_left_2d, _col_concat_2d,
    _col_gather_nan, _col_gather_2d_nan, _col_count_nonzeros, _col_memcpy,
    _col_abs, _col_round, _col_ffill_double, _col_bfill_double,
    _c_max, _c_min, _c_sum, _c_prod, _c_idxmax, _c_idxmin, _c_var, _c_mad
)
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.boolcolumn cimport BoolColumn
from fluidframe.core.asciicolumn cimport AsciiColumn
from fluidframe.core.ucs2column cimport UCS2Column
from fluidframe.core.datecolumn cimport DateColumn, _ymd_to_days
from fluidframe.core.categorycolumn cimport CategoryColumn
from fluidframe.core.datetimecolumn cimport DateTimeColumn
from fluidframe.core.timecolumn cimport TimeColumn

cdef int64_t US_PER_DAY = 86400000000LL

# ============================================================
#  底层纯 C 数学/度量算子 (零分支与 SIMD 优化)
# ============================================================

cdef int _cmp_double(const void* a, const void* b) noexcept nogil:
    """提供给 qsort 和 bsearch 的严格全序比较函数，安全处理 NaN"""
    cdef double va = (<const double*>a)[0]
    cdef double vb = (<const double*>b)[0]
    cdef bint anan = (va != va)
    cdef bint bnan = (vb != vb)
    if anan and bnan: return 0
    if anan: return -1
    if bnan: return 1
    if va < vb: return -1
    if va > vb: return 1
    return 0

cdef int _cmp_double_asc(const void* a, const void* b) noexcept nogil:
    """升序比较，用于 qsort (NaN 视为最大值，排最后)"""
    cdef double va = (<const double*>a)[0]
    cdef double vb = (<const double*>b)[0]
    cdef bint anan = (va != va)
    cdef bint bnan = (vb != vb)
    if anan and bnan: return 0
    if anan: return 1
    if bnan: return -1
    if va < vb: return -1
    if va > vb: return 1
    return 0


cdef void _col_to_double(object col, double* buf, Py_ssize_t n) except *:
    cdef Py_ssize_t i
    cdef int* int_data
    cdef double* double_data

    if isinstance(col, DoubleColumn):
        double_data = (<DoubleColumn>col).data_c
        if n > 0:
            memcpy(buf, double_data, n * sizeof(double))
    elif isinstance(col, IntColumn):
        int_data = (<IntColumn>col).data_c
        with nogil:
            for i in range(n):
                buf[i] = <double>int_data[i]
    else:
        raise TypeError("Expected IntColumn or DoubleColumn")

@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef class DoubleColumn(BaseColumn):

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __cinit__(self, Py_ssize_t init_capacity=16, Py_ssize_t cols=1):
        self.capacity = max(init_capacity, 1)
        self.length = 0
        self.dtype = TYPE_DOUBLE
        self._base_obj = None

        self.num_cols = <int>cols if cols > 0 else 1
        self.is_2d = (self.num_cols > 1)
        self.owns_data = True

        self.data_c = <double*>aligned_malloc(self.capacity * sizeof(double))
        if not self.data_c:
            raise MemoryError()

    cdef void* _get_data_ptr(self) noexcept nogil:
        return <void*>self.data_c

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __dealloc__(self):
        if self.owns_data and self.data_c != NULL:
            os_aligned_free(self.data_c)
            self.data_c = NULL

    @property
    def cols(self):
        return self.num_cols

    @property
    def rows(self):
        return self._get_rows_c()

    @property
    def _owns_data(self):
        return self.owns_data

    @_owns_data.setter
    def _owns_data(self, bint val):
        self.owns_data = val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def get_size(self):
        return self.capacity * sizeof(double)

    cdef inline DoubleColumn _make_like_nan(self):
        """生成与自身形状与容量完全一致的纯 NaN 结果列"""
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t total = self.length
        cdef DoubleColumn res = DoubleColumn(max(total, 1), cols=c)
        res.length = total
        cdef Py_ssize_t i
        with nogil:
            for i in range(total):
                res.data_c[i] = NAN
        return res

    @staticmethod
    cdef DoubleColumn create_view(double* ptr, Py_ssize_t length, object base_obj):
        """零拷贝 1D 视图"""
        cdef DoubleColumn col = DoubleColumn.__new__(DoubleColumn)
        col.dtype = TYPE_DOUBLE
        col.capacity = length
        col.length = length
        col.data_c = ptr
        col._base_obj = base_obj
        # 1 列, owns_data=False, is_2d=False
        col.num_cols = 1
        col.is_2d = False
        col.owns_data = False
        return col

    

    @staticmethod
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def from_2d(object values, Py_ssize_t rows, Py_ssize_t cols):
        """
        从一维扁平数据结构构建可 owns 的 2D（列优先）DoubleColumn。
        支持 list, tuple 乃至 numpy array 扁平数组。
        """
        cdef Py_ssize_t total = rows * cols
        if len(values) != total:
            raise ValueError(f"len(values)={len(values)} != rows*cols={total}")

        cdef DoubleColumn col = DoubleColumn(max(total, 1), cols=cols)
        col.num_cols = <int>cols
        col.is_2d = True
        col.length = total
        cdef Py_ssize_t i
        cdef double* ptr = col.data_c
        for i in range(total):
            ptr[i] = <double>values[i]
        return col

    @staticmethod
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _load_from_1d_array(object arr):
        """零拷贝加载 1D 连续数组 (支持 NumPy、PyTorch 等任意实现 Buffer 协议的对象)"""
        cdef double[::1] view
        try:
            view = arr
        except (ValueError, TypeError):
            raise ValueError(
                "Input must be a 1D contiguous float64 array. "
                "If using NumPy, please pass np.ascontiguousarray(arr, dtype=np.float64) explicitly."
            )
        cdef Py_ssize_t length = view.shape[0]
        cdef DoubleColumn col = DoubleColumn.__new__(DoubleColumn)
        col.dtype = TYPE_DOUBLE
        col.capacity = length
        col.length = length
        col.owns_data = False
        col._base_obj = arr
        col.data_c = &view[0] if length > 0 else NULL
        return col

    @staticmethod
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _load_from_2d_array(object arr, list col_names):
        """零拷贝加载 2D 列优先数组 (Fortran-contiguous)"""
        cdef double[::1, :] view
        try:
            view = arr
        except (ValueError, TypeError):
            raise ValueError(
                "Input array must be a 2D Fortran-contiguous (column-major) float64 array. "
                "If using NumPy, please pass np.asfortranarray(arr, dtype=np.float64) explicitly."
            )
        cdef Py_ssize_t rows = view.shape[0]
        cdef Py_ssize_t cols = view.shape[1]
        if len(col_names) != cols:
            raise ValueError(f"col_names length ({len(col_names)}) != array columns ({cols})")
        if rows == 0 or cols == 0:
            return {}
        cdef double* raw_ptr = &view[0, 0]
        cdef dict columns = {}
        cdef Py_ssize_t j
        for j in range(cols):
            columns[col_names[j]] = DoubleColumn.create_view(
                raw_ptr + (j * rows),
                rows,
                arr
            )
        return columns

    cdef void _reserve(self, Py_ssize_t new_cap):
        if not self.owns_data:
            raise RuntimeError("Cannot reallocate a view/slice column. Copy it first.")
        cdef double* new_ptr
        if new_cap > self.capacity:
            new_ptr = <double*>aligned_realloc(self.data_c, self.capacity * sizeof(double), new_cap * sizeof(double))
            if not new_ptr:
                raise MemoryError()
            self.data_c = new_ptr
            self.capacity = new_cap

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def append(self, double val):
        if not self.owns_data:
            raise RuntimeError("Cannot append to a view/slice column. Copy it first.")
        if self.is_2d:
            raise NotImplementedError("append is not supported for 2D columns (rows*cols layout); convert to 1D first")
        if self.length >= self.capacity:
            self._reserve(max(16, self.capacity * 2))
        self.data_c[self.length] = val
        self.length += 1

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def copy(self):
        cdef DoubleColumn new_col = DoubleColumn(max(self.length, 1), cols=self.num_cols)
        new_col.length = self.length
        _col_memcpy(new_col.data_c, self.data_c, self.length)
        return new_col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_back(self):
        if not self.owns_data:
            raise RuntimeError("Cannot modify a view/slice column. Copy it first.")
        if self.length == 0:
            raise IndexError("pop from empty column")
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r, j
        cdef list vals
        if self.is_2d:
            r = self._get_rows_c()
            vals = []
            for j in range(c):
                vals.append(self.data_c[j * r + (r - 1)])
            _col_shift_left_2d(self.data_c, &self.length, r, c, r - 1, sizeof(double))
            return vals
        cdef double* ptr = <double*>_col_pop_back(
            self.data_c, &self.length, sizeof(double))
        return ptr[0]

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_front(self):
        if not self.owns_data:
            raise RuntimeError("Cannot modify a view/slice column. Copy it first.")
        if self.length == 0:
            raise IndexError("pop from empty column")
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r, j
        cdef list vals
        if self.is_2d:
            r = self._get_rows_c()
            vals = []
            for j in range(c):
                vals.append(self.data_c[j * r])
            _col_shift_left_2d(self.data_c, &self.length, r, c, 0, sizeof(double))
            return vals
        cdef double val = self.data_c[0]
        _col_pop_front(self.data_c, &self.length, sizeof(double))
        return val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def concat(self, DoubleColumn other):
        if not self.owns_data:
            raise RuntimeError("Cannot concat into a view/slice column. Copy it first.")

        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t other_c = other.num_cols
        cdef Py_ssize_t other_rows

        if self.is_2d or other.is_2d:
            if c != other_c:
                if self.length == 0:
                    c = other_c
                else:
                    raise ValueError(f"Cannot concat 2D columns with different column counts: {c} vs {other_c}")
            other_rows = other._get_rows_c()
            if other_rows == 0: return

            self._reserve(self.length + other_rows * c)
            _col_concat_2d(self.data_c, &self.length, other.data_c, other_rows, c, sizeof(double))
            self.num_cols = <int>c
            self.is_2d = True
            return
        if other.length == 0: return
        self._reserve(self.length + other.length)
        _col_concat(self.data_c, &self.length, other.data_c, other.length,
                     sizeof(double))

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def max(self):
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c()
        if self.length == 0: raise ValueError("empty column")
        cdef double* ptr = self.data_c
        cdef Py_ssize_t j
        cdef list result
        if self.is_2d and c > 1:
            result = []
            for j in range(c):
                result.append(_c_max(&ptr[j * r], r))
            return result
        cdef double max_val = ptr[0]
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(1, n):
                if ptr[i] > max_val:
                    max_val = ptr[i]
        return max_val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def min(self):
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c()
        if self.length == 0: raise ValueError("empty column")
        cdef double* ptr = self.data_c
        cdef Py_ssize_t j
        cdef list result
        if self.is_2d and c > 1:
            result = []
            for j in range(c):
                result.append(_c_min(&ptr[j * r], r))
            return result
        cdef double min_val = ptr[0]
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(1, n):
                if ptr[i] < min_val:
                    min_val = ptr[i]
        return min_val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sum(self):
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c()
        cdef Py_ssize_t n = self.length
        if n == 0: return 0.0
        cdef double* ptr = self.data_c
        cdef Py_ssize_t j
        cdef list result
        if self.is_2d and c > 1:
            result = []
            for j in range(c):
                result.append(_c_sum(&ptr[j * r], r))
            return result
        cdef double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0
        cdef Py_ssize_t i
        cdef Py_ssize_t n_unrolled = n - (n % 4)

        with nogil:
            for i in range(0, n_unrolled, 4):
                s0 += ptr[i]
                s1 += ptr[i+1]
                s2 += ptr[i+2]
                s3 += ptr[i+3]
            s0 = (s0 + s1) + (s2 + s3)
            for i in range(n_unrolled, n):
                s0 += ptr[i]
        return s0

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def mean(self):
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c()
        if self.length == 0: return NAN
        cdef double* ptr = self.data_c
        cdef Py_ssize_t j
        cdef list result
        if self.is_2d and c > 1:
            result = []
            for j in range(c):
                result.append(_c_sum(&ptr[j * r], r) / <double>r)
            return result
        return self.sum() / <double>self.length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def var(self):
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c()
        if self.length < 2: return 0.0
        cdef double* ptr = self.data_c
        cdef Py_ssize_t j
        cdef double m, v
        cdef list result
        if self.is_2d and c > 1:
            result = []
            for j in range(c):
                m = _c_sum(&ptr[j * r], r) / <double>r
                v = _c_var(&ptr[j * r], r, m)
                result.append(v)
            return result
        cdef double m1 = self.mean()
        cdef double sum_sq = 0.0, diff
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                diff = ptr[i] - m1
                sum_sq += diff * diff
        return sum_sq / <double>(n - 1)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def std(self):
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c()
        cdef double* ptr = self.data_c
        cdef Py_ssize_t j
        cdef double m, v
        cdef list result
        if self.is_2d and c > 1:
            result = []
            for j in range(c):
                m = _c_sum(&ptr[j * r], r) / <double>r
                v = _c_var(&ptr[j * r], r, m)
                result.append(sqrt(v))
            return result
        return sqrt(self.var())

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __getitem__(self, Py_ssize_t idx):
        if idx < 0: idx += self.length
        if idx < 0 or idx >= self.length: raise IndexError("Index out of range")
        return self.data_c[idx]

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _count_nonzeros(self):
        return _col_count_nonzeros(self.data_c, self.length)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def idxmax(self):
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c()
        if self.length == 0: raise ValueError("empty column")
        cdef double* ptr = self.data_c
        cdef Py_ssize_t j
        cdef list result
        if self.is_2d and c > 1:
            result = []
            for j in range(c):
                result.append(_c_idxmax(&ptr[j * r], r))
            return result
        cdef Py_ssize_t max_idx = 0
        cdef double max_val = ptr[0]
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(1, n):
                if ptr[i] > max_val:
                    max_val = ptr[i]
                    max_idx = i
        return max_idx

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def idxmin(self):
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c()
        if self.length == 0: raise ValueError("empty column")
        cdef double* ptr = self.data_c
        cdef Py_ssize_t j
        cdef list result
        if self.is_2d and c > 1:
            result = []
            for j in range(c):
                result.append(_c_idxmin(&ptr[j * r], r))
            return result
        cdef Py_ssize_t min_idx = 0
        cdef double min_val = ptr[0]
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(1, n):
                if ptr[i] < min_val:
                    min_val = ptr[i]
                    min_idx = i
        return min_idx

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _argmax(self): return self.idxmax()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _argmin(self): return self.idxmin()

    # ------------------------------------------------------------
    # 工业级三路划分快速选择 (防死循环、防退化、支持 NaN 安全比较)
    # ------------------------------------------------------------
    cdef double _quickselect(self, double* a, Py_ssize_t n, Py_ssize_t k) noexcept nogil:
        cdef Py_ssize_t l = 0, r = n - 1
        cdef Py_ssize_t i, lt, gt
        cdef double pivot, tmp

        while l <= r:
            pivot = a[l + (r - l) // 2]
            lt = l
            gt = r
            i = l
            while i <= gt:
                if a[i] < pivot:
                    tmp = a[lt]; a[lt] = a[i]; a[i] = tmp
                    lt += 1
                    i += 1
                elif a[i] > pivot:
                    tmp = a[i]; a[i] = a[gt]; a[gt] = tmp
                    gt -= 1
                else:
                    i += 1

            if k < lt:
                r = lt - 1
            elif k > gt:
                l = gt + 1
            else:
                return a[k]
        return a[k]

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def median(self):
        if self.length == 0: return NAN
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c()
        cdef double* ptr = self.data_c
        cdef Py_ssize_t j
        cdef list result
        if self.is_2d and c > 1:
            result = []
            for j in range(c):
                result.append(self.quantile_per_subcol(ptr + j * r, r, 0.5))
            return result
        cdef double* temp = <double*>aligned_malloc(self.length * sizeof(double))
        if not temp: raise MemoryError()
        memcpy(temp, self.data_c, self.length * sizeof(double))
        cdef Py_ssize_t mid = self.length // 2
        cdef double v1, v2, res
        try:
            if self.length % 2 == 0:
                v1 = self._quickselect(temp, self.length, mid - 1)
                v2 = self._quickselect(temp, self.length, mid)
                return (v1 + v2) * 0.5
            res = self._quickselect(temp, self.length, mid)
            return res
        finally:
            os_aligned_free(temp)

    cdef double quantile_per_subcol(self, double* col_ptr, Py_ssize_t n, double q) noexcept:
        cdef double* temp = <double*>aligned_malloc(n * sizeof(double))
        if not temp: return NAN
        memcpy(temp, col_ptr, n * sizeof(double))
        cdef double pos = q * (n - 1)
        cdef Py_ssize_t idx = <Py_ssize_t>pos
        cdef double frac = pos - idx
        cdef double v1, v2, result
        try:
            if idx >= n - 1:
                result = self._quickselect(temp, n, n - 1)
            else:
                v1 = self._quickselect(temp, n, idx)
                v2 = self._quickselect(temp, n, idx + 1)
                result = v1 + frac * (v2 - v1)
        finally:
            os_aligned_free(temp)
        return result

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def mad(self):
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c()
        if self.length == 0: return NAN
        cdef double* ptr = self.data_c
        cdef Py_ssize_t j
        cdef list result
        if self.is_2d and c > 1:
            result = []
            for j in range(c):
                result.append(_c_mad(&ptr[j * r], r, _c_sum(&ptr[j * r], r) / <double>r))
            return result
        cdef double m = self.mean()
        cdef double total_dev = 0.0
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                total_dev += fabs(ptr[i] - m)
        return total_dev / <double>n

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def prod(self):
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c()
        if self.length == 0: return NAN
        cdef double* ptr = self.data_c
        cdef Py_ssize_t j
        cdef list result
        if self.is_2d and c > 1:
            result = []
            for j in range(c):
                result.append(_c_prod(&ptr[j * r], r))
            return result
        cdef double total = 1.0
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                total *= ptr[i]
        return total

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _nunique(self):
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()
        if r == 0:
            if not self.is_2d or c <= 1: return 0
            return 0
        cdef double* ptr = self.data_c
        cdef Py_ssize_t i, j, base
        cdef set seen
        seen = set()
        if self.is_2d and c > 1:
            for j in range(c):
                base = j * r
                for i in range(r): seen.add(ptr[base + i])
        else:
            for i in range(r): seen.add(ptr[i])
        return len(seen)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _value_counts(self):
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c()
        cdef double* ptr = self.data_c
        cdef Py_ssize_t i, j, base
        cdef double val
        cdef list result
        cdef dict counts
        if self.is_2d and c > 1:
            result = []
            for j in range(c):
                counts = {}
                base = j * r
                for i in range(r):
                    val = ptr[base + i]
                    counts[val] = counts.get(val, 0) + 1
                result.append(counts)
            return result
        counts = {}
        cdef Py_ssize_t n = self.length
        for i in range(n):
            val = ptr[i]
            counts[val] = counts.get(val, 0) + 1
        return counts

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def mode(self):
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c()
        cdef double* ptr = self.data_c
        cdef Py_ssize_t i, j, base
        cdef double val
        cdef int count, max_count
        cdef list result
        cdef dict counts
        if self.is_2d and c > 1:
            result = []
            for j in range(c):
                counts = {}
                max_count = 0
                base = j * r
                for i in range(r):
                    val = ptr[base + i]
                    count = counts.get(val, 0) + 1
                    counts[val] = count
                    if count > max_count:
                        max_count = count
                result.append([k for k, v in counts.items() if v == max_count])
            return result
        if self.length == 0: return []
        counts = {}
        cdef Py_ssize_t n = self.length
        max_count = 0
        for i in range(n):
            val = ptr[i]
            count = counts.get(val, 0) + 1
            counts[val] = count
            if count > max_count:
                max_count = count
        return [k for k, v in counts.items() if v == max_count]

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def quantile(self, double q):
        if self.length == 0: return NAN
        if q < 0.0 or q > 1.0: raise ValueError("q must be between 0 and 1")
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c()
        cdef double* ptr = self.data_c
        cdef Py_ssize_t j
        cdef list result
        if self.is_2d and c > 1:
            result = []
            for j in range(c):
                result.append(self.quantile_per_subcol(ptr + j * r, r, q))
            return result
        cdef double* temp = <double*>aligned_malloc(self.length * sizeof(double))
        if not temp: raise MemoryError()
        memcpy(temp, self.data_c, self.length * sizeof(double))
        cdef double pos = q * (self.length - 1)
        cdef Py_ssize_t idx = <Py_ssize_t>pos
        cdef double frac = pos - idx
        cdef double v1, v2
        try:
            if idx >= self.length - 1:
                return self._quickselect(temp, self.length, self.length - 1)
            v1 = self._quickselect(temp, self.length, idx)
            v2 = self._quickselect(temp, self.length, idx + 1)
            return v1 + frac * (v2 - v1)
        finally:
            os_aligned_free(temp)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cumsum(self):
        cdef DoubleColumn res = DoubleColumn(max(self.capacity, 1), cols=self.num_cols)
        res.length = self.length
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef double total
        cdef double* ptr = self.data_c
        cdef double* res_ptr = res.data_c
        cdef Py_ssize_t col_idx, offset, i
        with nogil:
            for col_idx in range(c):
                offset = col_idx * r
                total = 0.0
                for i in range(r):
                    total += ptr[offset + i]
                    res_ptr[offset + i] = total
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cummax(self):
        cdef DoubleColumn res = DoubleColumn(max(self.capacity, 1), cols=self.num_cols)
        res.length = self.length
        if self.length == 0: return res
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef double cur_max
        cdef double* ptr = self.data_c
        cdef double* res_ptr = res.data_c
        cdef Py_ssize_t col_idx, offset, i
        with nogil:
            for col_idx in range(c):
                offset = col_idx * r
                if r > 0:
                    cur_max = ptr[offset]
                    res_ptr[offset] = cur_max
                    for i in range(1, r):
                        if ptr[offset + i] > cur_max:
                            cur_max = ptr[offset + i]
                        res_ptr[offset + i] = cur_max
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cummin(self):
        cdef DoubleColumn res = DoubleColumn(max(self.capacity, 1), cols=self.num_cols)
        res.length = self.length
        if self.length == 0: return res
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef double cur_min
        cdef double* ptr = self.data_c
        cdef double* res_ptr = res.data_c
        cdef Py_ssize_t col_idx, offset, i
        with nogil:
            for col_idx in range(c):
                offset = col_idx * r
                if r > 0:
                    cur_min = ptr[offset]
                    res_ptr[offset] = cur_min
                    for i in range(1, r):
                        if ptr[offset + i] < cur_min:
                            cur_min = ptr[offset + i]
                        res_ptr[offset + i] = cur_min
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def diff(self, int periods=1):
        cdef DoubleColumn res = DoubleColumn(max(self.capacity, 1), cols=self.num_cols)
        res.length = self.length
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef double* ptr = self.data_c
        cdef double* res_ptr = res.data_c
        cdef Py_ssize_t col_idx, offset, i
        cdef int p = periods if periods >= 0 else -periods
        with nogil:
            if periods >= 0:
                for col_idx in range(c):
                    offset = col_idx * r
                    for i in range(min(p, r)):
                        res_ptr[offset + i] = NAN
                    for i in range(p, r):
                        res_ptr[offset + i] = ptr[offset + i] - ptr[offset + i - p]
            else:
                for col_idx in range(c):
                    offset = col_idx * r
                    for i in range(max(0, r - p), r):
                        res_ptr[offset + i] = NAN
                    for i in range(r - p):
                        res_ptr[offset + i] = ptr[offset + i] - ptr[offset + i + p]
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _pct_change(self, int periods=1):
        cdef DoubleColumn res = DoubleColumn(max(self.capacity, 1), cols=self.num_cols)
        res.length = self.length
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef double* ptr = self.data_c
        cdef double* res_ptr = res.data_c
        cdef Py_ssize_t col_idx, offset, i
        cdef double prev
        cdef int p = periods if periods >= 0 else -periods
        with nogil:
            if periods >= 0:
                for col_idx in range(c):
                    offset = col_idx * r
                    for i in range(min(p, r)):
                        res_ptr[offset + i] = NAN
                    for i in range(p, r):
                        prev = ptr[offset + i - p]
                        res_ptr[offset + i] = (ptr[offset + i] - prev) / prev if prev != 0.0 else NAN
            else:
                for col_idx in range(c):
                    offset = col_idx * r
                    for i in range(max(0, r - p), r):
                        res_ptr[offset + i] = NAN
                    for i in range(r - p):
                        prev = ptr[offset + i + p]
                        res_ptr[offset + i] = (ptr[offset + i] - prev) / prev if prev != 0.0 else NAN
        return res

    cdef void _central_moments(self, double* out_mean, double* out_m2, double* out_m3, double* out_m4) noexcept nogil:
        cdef Py_ssize_t n = self.length
        if n == 0:
            out_mean[0] = 0.0; out_m2[0] = 0.0; out_m3[0] = 0.0; out_m4[0] = 0.0
            return
        cdef double mean = 0.0, m2 = 0.0, m3 = 0.0, m4 = 0.0
        cdef double delta, delta_n, delta_n2, term1
        cdef Py_ssize_t i, n1
        cdef double* ptr = self.data_c
        for i in range(n):
            n1 = i
            delta = ptr[i] - mean
            mean += delta / (n1 + 1)
            delta_n = ptr[i] - mean
            term1 = delta * delta_n * n1
            delta_n2 = delta_n * delta_n
            m4 += term1 * delta_n2 * ((n1 + 1) * (n1 + 1) - 3 * (n1 + 1) + 3) + 6.0 * delta_n2 * m2 - 4.0 * delta_n * m3
            m3 += term1 * delta_n * (n1 - 2) - 3.0 * delta_n * m2
            m2 += term1
        out_mean[0] = mean
        out_m2[0] = fmax(m2, 0.0)
        out_m3[0] = m3
        out_m4[0] = fmax(m4, 0.0)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def skew(self):
        cdef double m, m2, m3, m4
        cdef double n, var_sample
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c()
        cdef double* ptr = self.data_c
        cdef Py_ssize_t j, i
        cdef double* sub_ptr
        cdef list result
        if self.is_2d and c > 1:
            result = []
            for j in range(c):
                sub_ptr = ptr + j * r
                m = _c_sum(sub_ptr, r) / <double>r
                m2 = _c_var(sub_ptr, r, m) * <double>(r - 1) / <double>r
                if m2 <= 1e-14:
                    result.append(NAN)
                    continue
                n = <double>r
                var_sample = m2 / (n - 1)
                m3 = 0.0
                for i in range(r):
                    m3 += (sub_ptr[i] - m) ** 3
                m3 = m3 / n
                result.append((n * m3) / ((n - 1) * (n - 2) * (var_sample * sqrt(var_sample))))
            return result
        if self.length < 3: return NAN
        self._central_moments(&m, &m2, &m3, &m4)
        if m2 <= 1e-14: return NAN
        n = <double>self.length
        var_sample = m2 / (n - 1)
        return (n * m3) / ((n - 1) * (n - 2) * (var_sample * sqrt(var_sample)))

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def kurt(self):
        cdef double m, m2, m3, m4
        cdef double n, var_sample
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c()
        cdef double* ptr = self.data_c
        cdef Py_ssize_t j, i
        cdef double* sub_ptr
        cdef list result
        if self.is_2d and c > 1:
            result = []
            for j in range(c):
                sub_ptr = ptr + j * r
                m = _c_sum(sub_ptr, r) / <double>r
                m2 = _c_var(sub_ptr, r, m) * <double>(r - 1) / <double>r
                if m2 <= 1e-14:
                    result.append(NAN)
                    continue
                n = <double>r
                var_sample = m2 / (n - 1)
                m4 = 0.0
                for i in range(r):
                    m4 += (sub_ptr[i] - m) ** 4
                m4 = m4 / n
                result.append((m4 / (n - 1)) / (var_sample * var_sample) - 3.0)
            return result
        if self.length < 4: return NAN
        self._central_moments(&m, &m2, &m3, &m4)
        if m2 <= 1e-14: return NAN
        n = <double>self.length
        var_sample = m2 / (n - 1)
        return (m4 / (n - 1)) / (var_sample * var_sample) - 3.0

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sem(self):
        cdef double m, m2, m3, m4
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c()
        cdef double* ptr = self.data_c
        cdef Py_ssize_t j
        cdef double* sub_ptr
        cdef list result
        if self.is_2d and c > 1:
            result = []
            for j in range(c):
                sub_ptr = ptr + j * r
                m = _c_sum(sub_ptr, r) / <double>r
                m2 = _c_var(sub_ptr, r, m) * <double>(r - 1) / <double>r
                result.append(sqrt(m2 / ((<double>(r - 1)) * <double>r)))
            return result
        if self.length < 2: return NAN
        self._central_moments(&m, &m2, &m3, &m4)
        return sqrt(m2 / ((<double>self.length - 1) * <double>self.length))

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def count(self):
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c()
        if self.is_2d and c > 1:
            result = []
            for j in range(c):
                result.append(r)
            return result
        return self.length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cumprod(self):
        cdef DoubleColumn res = DoubleColumn(max(self.capacity, 1), cols=self.num_cols)
        res.length = self.length
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef double total = 1.0
        cdef double* ptr = self.data_c
        cdef double* res_ptr = res.data_c
        cdef Py_ssize_t col_idx, offset, i
        with nogil:
            for col_idx in range(c):
                offset = col_idx * r
                total = 1.0
                for i in range(r):
                    total *= ptr[offset + i]
                    res_ptr[offset + i] = total
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _describe(self):
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c()
        cdef double* ptr = self.data_c
        cdef Py_ssize_t j, i, base
        cdef double total_count, total_sum, total_min, total_max, val
        cdef double total_sq_sum, total_mean, total_std

        if self.is_2d and c > 1:
            total_count = <double>(r * c)
            total_sum = 0.0
            total_sq_sum = 0.0
            total_min = ptr[0]
            total_max = ptr[0]
            with nogil:
                for j in range(c):
                    base = j * r
                    for i in range(r):
                        val = ptr[base + i]
                        total_sum += val
                        total_sq_sum += val * val
                        if val < total_min: total_min = val
                        if val > total_max: total_max = val
            total_mean = total_sum / total_count
            total_std = sqrt((total_sq_sum / total_count) - (total_mean * total_mean)) if total_count > 1 else NAN
            return {
                "count": <int>total_count,
                "mean": total_mean,
                "std": total_std,
                "min": total_min,
                "max": total_max,
            }
        if self.length == 0:
            return {"count": 0, "mean": NAN, "std": NAN, "min": NAN,
                    "25%": NAN, "50%": NAN, "75%": NAN, "max": NAN}
        return {
            "count": self.length,
            "mean": self.mean(),
            "std": self.std(),
            "min": self.min(),
            "25%": self.quantile(0.25),
            "50%": self.quantile(0.50),
            "75%": self.quantile(0.75),
            "max": self.max()
        }

    cdef double _sum_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil:
        cdef double total = 0.0
        cdef Py_ssize_t i
        cdef double* ptr = self.data_c
        for i in range(n):
            total += ptr[indices[i]]
        return total

    cdef double _mean_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil:
        if n == 0: return NAN
        return self._sum_rows(indices, n) / <double>n

    cdef double _min_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil:
        if n == 0: return NAN
        cdef double val, min_val
        cdef double* ptr = self.data_c
        min_val = ptr[indices[0]]
        cdef Py_ssize_t i
        for i in range(1, n):
            val = ptr[indices[i]]
            if val < min_val: min_val = val
        return min_val

    cdef double _max_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil:
        if n == 0: return NAN
        cdef double val, max_val
        cdef double* ptr = self.data_c
        max_val = ptr[indices[0]]
        cdef Py_ssize_t i
        for i in range(1, n):
            val = ptr[indices[i]]
            if val > max_val: max_val = val
        return max_val

    cdef Py_ssize_t _count_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil:
        return n

    cdef double _var_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil:
        if n < 2: return 0.0
        cdef double m = self._mean_rows(indices, n)
        cdef double sum_sq = 0.0, v
        cdef double* ptr = self.data_c
        cdef Py_ssize_t i
        for i in range(n):
            v = ptr[indices[i]] - m
            sum_sq += v * v
        return sum_sq / <double>(n - 1)

    cdef double _std_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil:
        return sqrt(self._var_rows(indices, n))

    # ------------------------------------------------------------
    # 标量算术 (完全向量化)
    # ------------------------------------------------------------
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def add_val(self, double val):
        cdef DoubleColumn res = DoubleColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef double* data_ptr = self.data_c
        cdef double* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = data_ptr[i] + val
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sub_val(self, double val):
        cdef DoubleColumn res = DoubleColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef double* data_ptr = self.data_c
        cdef double* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = data_ptr[i] - val
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def mul_val(self, double val):
        cdef DoubleColumn res = DoubleColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef double* data_ptr = self.data_c
        cdef double* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = data_ptr[i] * val
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def div_val(self, double val):
        cdef DoubleColumn res = DoubleColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef double* data_ptr = self.data_c
        cdef double* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef double inv_val
        with nogil:
            if val != 0.0:
                inv_val = 1.0 / val
                for i in range(n):
                    res_ptr[i] = data_ptr[i] * inv_val
            else:
                for i in range(n):
                    res_ptr[i] = NAN
        return res

    # ------------------------------------------------------------
    # 向量算术
    # ------------------------------------------------------------
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def add(self, object other):
        cdef Py_ssize_t n = self.length
        if other.length != n:
            raise ValueError(f"Column length {other.length} != {n}")
        cdef DoubleColumn res = DoubleColumn(max(n, 1), cols=self.num_cols)
        res.length = n
        cdef double* self_ptr = self.data_c
        cdef double* res_ptr = res.data_c
        cdef double* other_d_ptr
        cdef int* other_i_ptr
        cdef Py_ssize_t i

        if isinstance(other, DoubleColumn):
            other_d_ptr = (<DoubleColumn>other).data_c
            with nogil:
                for i in range(n):
                    res_ptr[i] = self_ptr[i] + other_d_ptr[i]
        elif isinstance(other, IntColumn):
            other_i_ptr = (<IntColumn>other).data_c
            with nogil:
                for i in range(n):
                    res_ptr[i] = self_ptr[i] + <double>other_i_ptr[i]
        else:
            raise TypeError(f"Unsupported type for add: {type(other)}")
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def sub(self, object other):
        cdef Py_ssize_t n = self.length
        if other.length != n:
            raise ValueError(f"Column length {other.length} != {n}")
        cdef DoubleColumn res = DoubleColumn(max(n, 1), cols=self.num_cols)
        res.length = n
        cdef double* self_ptr = self.data_c
        cdef double* res_ptr = res.data_c
        cdef double* other_d_ptr
        cdef int* other_i_ptr
        cdef Py_ssize_t i

        if isinstance(other, DoubleColumn):
            other_d_ptr = (<DoubleColumn>other).data_c
            with nogil:
                for i in range(n):
                    res_ptr[i] = self_ptr[i] - other_d_ptr[i]
        elif isinstance(other, IntColumn):
            other_i_ptr = (<IntColumn>other).data_c
            with nogil:
                for i in range(n):
                    res_ptr[i] = self_ptr[i] - <double>other_i_ptr[i]
        else:
            raise TypeError(f"Unsupported type for sub: {type(other)}")
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def mul(self, object other):
        cdef Py_ssize_t n = self.length
        if other.length != n:
            raise ValueError(f"Column length {other.length} != {n}")
        cdef DoubleColumn res = DoubleColumn(max(n, 1), cols=self.num_cols)
        res.length = n
        cdef double* self_ptr = self.data_c
        cdef double* res_ptr = res.data_c
        cdef double* other_d_ptr
        cdef int* other_i_ptr
        cdef Py_ssize_t i

        if isinstance(other, DoubleColumn):
            other_d_ptr = (<DoubleColumn>other).data_c
            with nogil:
                for i in range(n):
                    res_ptr[i] = self_ptr[i] * other_d_ptr[i]
        elif isinstance(other, IntColumn):
            other_i_ptr = (<IntColumn>other).data_c
            with nogil:
                for i in range(n):
                    res_ptr[i] = self_ptr[i] * <double>other_i_ptr[i]
        else:
            raise TypeError(f"Unsupported type for mul: {type(other)}")
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def div(self, object other):
        cdef Py_ssize_t n = self.length
        if other.length != n:
            raise ValueError(f"Column length {other.length} != {n}")
        cdef DoubleColumn res = DoubleColumn(max(n, 1), cols=self.num_cols)
        res.length = n
        cdef double* self_ptr = self.data_c
        cdef double* res_ptr = res.data_c
        cdef double* other_d_ptr
        cdef int* other_i_ptr
        cdef double val
        cdef Py_ssize_t i

        if isinstance(other, DoubleColumn):
            other_d_ptr = (<DoubleColumn>other).data_c
            with nogil:
                for i in range(n):
                    val = other_d_ptr[i]
                    res_ptr[i] = self_ptr[i] / val if val != 0.0 else NAN
        elif isinstance(other, IntColumn):
            other_i_ptr = (<IntColumn>other).data_c
            with nogil:
                for i in range(n):
                    val = <double>other_i_ptr[i]
                    res_ptr[i] = self_ptr[i] / val if val != 0.0 else NAN
        else:
            raise TypeError(f"Unsupported type for div: {type(other)}")
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __add__(self, object other):
        if isinstance(other, (int, float)): return self.add_val(<double>other)
        if isinstance(other, (DoubleColumn, IntColumn)): return self.add(other)
        return NotImplemented

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __radd__(self, object other): return self.__add__(other)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __sub__(self, object other):
        if isinstance(other, (int, float)): return self.sub_val(<double>other)
        if isinstance(other, (DoubleColumn, IntColumn)): return self.sub(other)
        return NotImplemented

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __rsub__(self, object other):
        cdef Py_ssize_t n = self.length
        cdef DoubleColumn res
        cdef double v
        cdef double* s_ptr
        cdef double* r_ptr
        cdef Py_ssize_t i
        if isinstance(other, (int, float)):
            res = DoubleColumn(max(n, 1), cols=self.num_cols)
            res.length = n
            v = <double>other
            s_ptr = self.data_c
            r_ptr = res.data_c
            with nogil:
                for i in range(n):
                    r_ptr[i] = v - s_ptr[i]
            return res
        return NotImplemented

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __mul__(self, object other):
        if isinstance(other, (int, float)): return self.mul_val(<double>other)
        if isinstance(other, (DoubleColumn, IntColumn)): return self.mul(other)
        return NotImplemented

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __rmul__(self, object other): return self.__mul__(other)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __truediv__(self, object other):
        if isinstance(other, (int, float)): return self.div_val(<double>other)
        if isinstance(other, (DoubleColumn, IntColumn)): return self.div(other)
        return NotImplemented

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __rtruediv__(self, object other):
        cdef Py_ssize_t n = self.length
        cdef DoubleColumn res
        cdef double v, s_val
        cdef double* s_ptr
        cdef double* r_ptr
        cdef Py_ssize_t i
        if isinstance(other, (int, float)):
            res = DoubleColumn(max(n, 1), cols=self.num_cols)
            res.length = n
            v = <double>other
            s_ptr = self.data_c
            r_ptr = res.data_c
            with nogil:
                for i in range(n):
                    s_val = s_ptr[i]
                    r_ptr[i] = v / s_val if s_val != 0.0 else NAN
            return res
        return NotImplemented

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def unique(self):
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c()
        cdef double* ptr = self.data_c
        cdef Py_ssize_t i, j, base, n
        cdef double val
        cdef set seen
        cdef list result
        seen = set()
        result = []
        if self.is_2d and c > 1:
            for j in range(c):
                base = j * r
                for i in range(r):
                    val = ptr[base + i]
                    if val not in seen:
                        seen.add(val)
                        result.append(val)
        else:
            n = self.length
            for i in range(n):
                val = ptr[i]
                if val not in seen:
                    seen.add(val)
                    result.append(val)
        return result

    # ------------------------------------------------------------
    # 纯 C 向量化比较算子 (安全处理 NaN)
    # ------------------------------------------------------------
    cdef IntColumn _cmp_op(self, object val, int op):
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef Py_ssize_t i, n = self.length
        cdef double v
        cdef double* s_ptr = self.data_c
        cdef int* r_ptr = res.data_c
        cdef bint is_nan_val
        cdef char err_buf[128]
        cdef DoubleColumn other
        cdef double* o_ptr

        if isinstance(val, DoubleColumn):
            other = <DoubleColumn>val
            if other.length != n:
                snprintf(err_buf, sizeof(err_buf), "Length mismatch: %zd vs %zd", n, other.length)
                raise ValueError(err_buf.decode('ascii'))
            o_ptr = other.data_c
            with nogil:
                if op == 0:
                    for i in range(n): r_ptr[i] = (s_ptr[i] < o_ptr[i])
                elif op == 1:
                    for i in range(n): r_ptr[i] = (s_ptr[i] <= o_ptr[i])
                elif op == 2:
                    for i in range(n): r_ptr[i] = (s_ptr[i] > o_ptr[i])
                elif op == 3:
                    for i in range(n): r_ptr[i] = (s_ptr[i] >= o_ptr[i])
                elif op == 4:
                    for i in range(n): r_ptr[i] = (s_ptr[i] == o_ptr[i])
                elif op == 5:
                    for i in range(n): r_ptr[i] = (s_ptr[i] != o_ptr[i])
        else:
            v = <double>val
            is_nan_val = (v != v)
            with nogil:
                if is_nan_val:
                    if op == 4:
                        for i in range(n): r_ptr[i] = (s_ptr[i] != s_ptr[i])
                    elif op == 5:
                        for i in range(n): r_ptr[i] = (s_ptr[i] == s_ptr[i])
                    else:
                        for i in range(n): r_ptr[i] = 0
                else:
                    if op == 0:
                        for i in range(n): r_ptr[i] = (s_ptr[i] < v)
                    elif op == 1:
                        for i in range(n): r_ptr[i] = (s_ptr[i] <= v)
                    elif op == 2:
                        for i in range(n): r_ptr[i] = (s_ptr[i] > v)
                    elif op == 3:
                        for i in range(n): r_ptr[i] = (s_ptr[i] >= v)
                    elif op == 4:
                        for i in range(n): r_ptr[i] = (s_ptr[i] == v)
                    elif op == 5:
                        for i in range(n): r_ptr[i] = (s_ptr[i] != v)
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __lt__(self, val): return self._cmp_op(val, 0)
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __le__(self, val): return self._cmp_op(val, 1)
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __gt__(self, val): return self._cmp_op(val, 2)
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __ge__(self, val): return self._cmp_op(val, 3)
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __eq__(self, val): return self._cmp_op(val, 4)
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __ne__(self, val): return self._cmp_op(val, 5)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __and__(self, other):
        if not isinstance(other, IntColumn):
            raise TypeError("Cannot & non-IntColumn mask")
        cdef IntColumn o = <IntColumn>other
        if self.length != o.length:
            raise ValueError(f"Length mismatch: {self.length} vs {o.length}")
        cdef Py_ssize_t n = self.length
        cdef IntColumn res = IntColumn(max(n, 1))
        res.length = n
        cdef double* s_ptr = self.data_c
        cdef int* o_ptr = o.data_c
        cdef int* r_ptr = res.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                r_ptr[i] = (s_ptr[i] != 0.0) & (o_ptr[i] != 0)
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __or__(self, other):
        if not isinstance(other, IntColumn):
            raise TypeError("Cannot | non-IntColumn mask")
        cdef IntColumn o = <IntColumn>other
        if self.length != o.length:
            raise ValueError(f"Length mismatch: {self.length} vs {o.length}")
        cdef Py_ssize_t n = self.length
        cdef IntColumn res = IntColumn(max(n, 1))
        res.length = n
        cdef double* s_ptr = self.data_c
        cdef int* o_ptr = o.data_c
        cdef int* r_ptr = res.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                r_ptr[i] = (s_ptr[i] != 0.0) | (o_ptr[i] != 0)
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __invert__(self):
        cdef Py_ssize_t n = self.length
        cdef IntColumn res = IntColumn(max(n, 1))
        res.length = n
        cdef double* s_ptr = self.data_c
        cdef int* r_ptr = res.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                r_ptr[i] = (s_ptr[i] == 0.0)
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def isin(self, set values):
        cdef Py_ssize_t n = self.length
        cdef IntColumn res = IntColumn(max(n, 1))
        res.length = n
        cdef double* s_ptr = self.data_c
        cdef int* r_ptr = res.data_c
        cdef Py_ssize_t m = len(values)

        if m == 0 or n == 0:
            if n > 0:
                memset(r_ptr, 0, n * sizeof(int))
            return res

        cdef double* v_arr = <double*>aligned_malloc(m * sizeof(double))
        if not v_arr: raise MemoryError()
        cdef Py_ssize_t valid_m = 0
        cdef void* found
        cdef double single_val
        cdef Py_ssize_t i

        try:
            for v in values:
                if v is not None:
                    v_arr[valid_m] = <double>v
                    valid_m += 1

            if valid_m == 0:
                memset(r_ptr, 0, n * sizeof(int))
            elif valid_m == 1:
                single_val = v_arr[0]
                with nogil:
                    for i in range(n):
                        r_ptr[i] = (s_ptr[i] == single_val)
            else:
                qsort(v_arr, valid_m, sizeof(double), _cmp_double)
                with nogil:
                    for i in range(n):
                        found = bsearch(&s_ptr[i], v_arr, valid_m, sizeof(double), _cmp_double)
                        r_ptr[i] = 1 if found != NULL else 0
        finally:
            os_aligned_free(v_arr)

        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def between(self, double low, double high):
        cdef Py_ssize_t n = self.length
        cdef IntColumn res = IntColumn(max(n, 1))
        res.length = n
        cdef double* s_ptr = self.data_c
        cdef int* r_ptr = res.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                r_ptr[i] = (low <= s_ptr[i]) & (s_ptr[i] <= high)
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __str__(self):
        cdef list items = []
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            items.append(str(self.data_c[i]))
        return f"DoubleColumn([{', '.join(items)}])"

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __bool__(self):
        return self.length > 0

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __len__(self):
        return self.length

    cdef DoubleColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=False):
        cdef DoubleColumn new_col
        cdef double* new_data
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()

        if self.is_2d:
            if not inplace:
                new_col = DoubleColumn(max(n * c, 1), cols=c)
                new_col.length = n * c
                _col_gather_2d_nan(new_col.data_c, self.data_c, indices, n, r, c, sizeof(double))
                return new_col
            else:
                new_data = <double*>aligned_malloc(max(n * c, 1) * sizeof(double))
                if not new_data: raise MemoryError()
                _col_gather_2d_nan(new_data, self.data_c, indices, n, r, c, sizeof(double))
                if self.owns_data and self.data_c != NULL:
                    os_aligned_free(self.data_c)
                self.data_c = new_data
                self.capacity = n * c
                self.length = n * c
                self.num_cols = <int>c
                self.is_2d = True
                self._base_obj = None
                return self

        if not inplace:
            new_col = DoubleColumn(max(n, 1))
            new_col.length = n
            _col_gather_nan(new_col.data_c, self.data_c, indices, n, sizeof(double))
            return new_col
        else:
            new_data = <double*>aligned_malloc(max(n, 1) * sizeof(double))
            if not new_data: raise MemoryError()
            _col_gather_nan(new_data, self.data_c, indices, n, sizeof(double))
            if self.owns_data and self.data_c != NULL:
                os_aligned_free(self.data_c)
            self.data_c = new_data
            self.capacity = n
            self.length = n
            self._base_obj = None
            return self

    # ------------------------------------------------------------
    # 无分支 clip 优化 (直接映射 vmaxpd/vminpd)
    # ------------------------------------------------------------
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def clip(self, double lower, double upper, bint inplace=False):
        cdef DoubleColumn col = self if inplace else self.copy()
        cdef Py_ssize_t i, n = col.length
        cdef double v
        cdef double* ptr = col.data_c
        with nogil:
            for i in range(n):
                v = ptr[i]
                v = lower if v < lower else v
                v = upper if v > upper else v
                ptr[i] = v
        return col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def abs(self):
        cdef DoubleColumn new_col = self.copy()
        if new_col.length > 0:
            _col_abs(new_col.data_c, new_col.data_c, new_col.length)
        return new_col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def round(self, int decimals=0):
        cdef DoubleColumn new_col = self.copy()
        cdef double factor
        if new_col.length > 0:
            factor = pow(10.0, <double>decimals)
            _col_round(new_col.data_c, new_col.data_c, new_col.length, factor)
        return new_col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def ffill(self):
        cdef DoubleColumn new_col = self.copy()
        cdef Py_ssize_t i
        if new_col.length > 0:
            if new_col.is_2d and new_col.num_cols > 1:
                for i in range(new_col.num_cols):
                    _col_ffill_double(new_col.data_c + i * new_col._get_rows_c(), new_col._get_rows_c())
            else:
                _col_ffill_double(new_col.data_c, new_col.length)
        return new_col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def bfill(self):
        cdef DoubleColumn new_col = self.copy()
        cdef Py_ssize_t i
        if new_col.length > 0:
            if new_col.is_2d and new_col.num_cols > 1:
                for i in range(new_col.num_cols):
                    _col_bfill_double(new_col.data_c + i * new_col._get_rows_c(), new_col._get_rows_c())
            else:
                _col_bfill_double(new_col.data_c, new_col.length)
        return new_col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def where(self, IntColumn cond, double other, bint inplace=False):
        cdef Py_ssize_t n = self.length
        cdef Py_ssize_t r = self._get_rows_c()
        cdef Py_ssize_t ccols = self.num_cols
        cdef char err_buf[128]
        cdef DoubleColumn col
        cdef Py_ssize_t i, j, base
        cdef double* col_ptr
        cdef double* self_ptr = self.data_c
        cdef int* cond_ptr = cond.data_c

        if self.is_2d and ccols > 1:
            if cond.length != r:
                snprintf(err_buf, sizeof(err_buf), "Condition mask length %zd must match column rows %zd", cond.length, r)
                raise ValueError(err_buf.decode('ascii'))
            col = self if inplace else DoubleColumn(max(n, 1), cols=ccols)
            if not inplace: col.length = n
            col_ptr = col.data_c
            with nogil:
                for j in range(ccols):
                    base = j * r
                    for i in range(r):
                        col_ptr[base + i] = self_ptr[base + i] if cond_ptr[i] != 0 else other
            return col

        if cond.length != n:
            snprintf(err_buf, sizeof(err_buf), "Condition mask length %zd must match column length %zd", cond.length, n)
            raise ValueError(err_buf.decode('ascii'))

        col = self if inplace else DoubleColumn(max(n, 1), cols=self.num_cols)
        if not inplace: col.length = n

        col_ptr = col.data_c
        with nogil:
            for i in range(n):
                col_ptr[i] = self_ptr[i] if cond_ptr[i] != 0 else other
        return col

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def where_mask(self, IntColumn mask, object fill):
        cdef Py_ssize_t n = self.length
        cdef Py_ssize_t r = self._get_rows_c()
        cdef Py_ssize_t ccols = self.num_cols
        cdef char err_buf[128]
        cdef DoubleColumn col
        cdef Py_ssize_t i, j, base
        cdef double fill_val = <double>fill
        cdef double* col_ptr
        cdef double* self_ptr = self.data_c
        cdef int* mask_ptr = mask.data_c

        if self.is_2d and ccols > 1:
            if mask.length != r:
                snprintf(err_buf, sizeof(err_buf), "Mask length %zd must match column rows %zd", mask.length, r)
                raise ValueError(err_buf.decode('ascii'))
            col = DoubleColumn(max(n, 1), cols=ccols)
            col.length = n
            col_ptr = col.data_c
            with nogil:
                for j in range(ccols):
                    base = j * r
                    for i in range(r):
                        col_ptr[base + i] = self_ptr[base + i] if mask_ptr[i] != 0 else fill_val
            return col

        if mask.length != n:
            snprintf(err_buf, sizeof(err_buf), "Mask length %zd != column length %zd", mask.length, n)
            raise ValueError(err_buf.decode('ascii'))

        col = DoubleColumn(max(n, 1), cols=self.num_cols)
        col.length = n
        col_ptr = col.data_c

        with nogil:
            for i in range(n):
                col_ptr[i] = self_ptr[i] if mask_ptr[i] != 0 else fill_val
        return col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def mask(self, IntColumn cond, double other, bint inplace=False):
        cdef Py_ssize_t n = self.length
        cdef Py_ssize_t r = self._get_rows_c()
        cdef Py_ssize_t ccols = self.num_cols
        cdef char err_buf[128]
        cdef DoubleColumn col
        cdef Py_ssize_t i, j, base
        cdef double* col_ptr
        cdef double* self_ptr = self.data_c
        cdef int* cond_ptr = cond.data_c

        if self.is_2d and ccols > 1:
            if cond.length != r:
                snprintf(err_buf, sizeof(err_buf), "Condition mask length %zd must match column rows %zd", cond.length, r)
                raise ValueError(err_buf.decode('ascii'))
            col = self if inplace else DoubleColumn(max(n, 1), cols=ccols)
            if not inplace: col.length = n
            col_ptr = col.data_c
            with nogil:
                for j in range(ccols):
                    base = j * r
                    for i in range(r):
                        col_ptr[base + i] = other if cond_ptr[i] != 0 else self_ptr[base + i]
            return col

        if cond.length != n:
            snprintf(err_buf, sizeof(err_buf), "Condition mask length %zd must match column length %zd", cond.length, n)
            raise ValueError(err_buf.decode('ascii'))

        col = self if inplace else DoubleColumn(max(n, 1), cols=self.num_cols)
        if not inplace: col.length = n

        col_ptr = col.data_c
        with nogil:
            for i in range(n):
                col_ptr[i] = other if cond_ptr[i] != 0 else self_ptr[i]
        return col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def drop_row(self, Py_ssize_t row_idx):
        if not self.owns_data:
            raise RuntimeError("Cannot drop row on a view/slice column. Copy it first.")
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()
        if row_idx < 0:
            row_idx += r
        if row_idx < 0 or row_idx >= r:
            raise IndexError("Row index out of bounds")
        if self.is_2d:
            _col_shift_left_2d(self.data_c, &self.length, r, c, row_idx, sizeof(double))
        else:
            _col_drop_row(self.data_c, &self.length, row_idx, sizeof(double))

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def shift_left(self, Py_ssize_t pos):
        if not self.owns_data:
            raise RuntimeError("Cannot modify a view/slice column. Copy it first.")
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()
        cdef Py_ssize_t col_len
        if pos < 0:
            pos += r
        if pos < 0 or pos >= r:
            raise IndexError("Index out of bounds")
        if self.is_2d:
            _col_shift_left_2d(self.data_c, &self.length, r, c, pos, sizeof(double))
        else:
            col_len = self.length
            _c_shift_left(self.data_c, pos, col_len, sizeof(double))
            self.length = col_len - 1

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def shift_right(self, Py_ssize_t pos):
        if not self.owns_data:
            raise RuntimeError("Cannot modify a view/slice column. Copy it first.")
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()
        if pos < 0:
            pos += r
        if pos < 0 or pos > r:
            raise IndexError("Index out of bounds")
        if self.is_2d:
            if self.length + c > self.capacity:
                self._reserve(max(16, max(self.capacity * 2, self.length + c)))
            _col_shift_right_2d_double(self.data_c, &self.length, r, c, pos)
        else:
            if self.length >= self.capacity:
                self._reserve(max(16, self.capacity * 2))
            _c_shift_right(self.data_c, pos, self.length, sizeof(double))
            self.length += 1
            self.data_c[pos] = NAN

    cdef void to_double_buffer(self, double* out, Py_ssize_t n):
        if n > 0:
            memcpy(out, self.data_c, n * sizeof(double))

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def euclidean_dist(self, other):
        cdef Py_ssize_t n = min(self.length, other.length)
        if n == 0: return NAN
        cdef DoubleColumn temp = DoubleColumn(max(n, 1))
        temp.length = n
        _col_to_double(other, temp.data_c, n)
        return _vec_euclidean_dist(self.data_c, temp.data_c, n)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def manhattan_dist(self, other):
        cdef Py_ssize_t n = min(self.length, other.length)
        if n == 0: return NAN
        cdef DoubleColumn temp = DoubleColumn(max(n, 1))
        temp.length = n
        _col_to_double(other, temp.data_c, n)
        return _vec_manhattan_dist(self.data_c, temp.data_c, n)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cosine_dist(self, other):
        cdef Py_ssize_t n = min(self.length, other.length)
        if n == 0: return NAN
        cdef DoubleColumn temp = DoubleColumn(max(n, 1))
        temp.length = n
        _col_to_double(other, temp.data_c, n)
        return 1.0 - _vec_cosine_sim(self.data_c, temp.data_c, n)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cosine_similarity(self, other):
        cdef Py_ssize_t n = min(self.length, other.length)
        if n == 0: return NAN
        cdef DoubleColumn temp = DoubleColumn(max(n, 1))
        temp.length = n
        _col_to_double(other, temp.data_c, n)
        return _vec_cosine_sim(self.data_c, temp.data_c, n)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def jaccard_sim(self, other):
        cdef Py_ssize_t n = min(self.length, other.length)
        if n == 0: return NAN
        cdef DoubleColumn temp = DoubleColumn(max(n, 1))
        temp.length = n
        _col_to_double(other, temp.data_c, n)
        return _vec_jaccard_sim(self.data_c, temp.data_c, n)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def hamming_dist(self, other):
        cdef Py_ssize_t n = min(self.length, other.length)
        if n == 0: return NAN
        cdef DoubleColumn temp = DoubleColumn(max(n, 1))
        temp.length = n
        _col_to_double(other, temp.data_c, n)
        return _vec_hamming_dist(self.data_c, temp.data_c, n)

    cdef DoubleColumn gather_with_sentinel(self, Py_ssize_t* indices, Py_ssize_t n, double sentinel):
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()
        cdef Py_ssize_t i, j, idx
        cdef Py_ssize_t dst_base, src_base
        cdef double* res_ptr
        cdef double* src_ptr
        cdef DoubleColumn new_col

        if self.is_2d:
            new_col = DoubleColumn(max(n * c, 1), cols=c)
            new_col.length = n * c
            res_ptr = new_col.data_c
            src_ptr = self.data_c
            with nogil:
                for j in range(c):
                    dst_base = j * n
                    src_base = j * r
                    for i in range(n):
                        idx = indices[i]
                        res_ptr[dst_base + i] = src_ptr[src_base + idx] if (idx >= 0 and idx < r) else sentinel
            return new_col

        # 1D 极速处理
        new_col = DoubleColumn(max(n, 1), cols=c)
        new_col.length = n
        res_ptr = new_col.data_c
        src_ptr = self.data_c
        cdef Py_ssize_t my_len = self.length

        with nogil:
            for i in range(n):
                idx = indices[i]
                res_ptr[i] = src_ptr[idx] if (idx >= 0 and idx < my_len) else sentinel
        return new_col

    # ------------------------------------------------------------
    # astype: 纯 C 级别零拷贝/对齐纪元架构转换
    # ------------------------------------------------------------
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def astype(self, object target):
        cdef int target_type = _resolve_astype_target(target)
        cdef Py_ssize_t i, n = self.length
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t first_col_n = self._get_rows_c() if c > 1 else n
        cdef Py_ssize_t safe_cap = max(n, 1)
        cdef Py_ssize_t first_col_cap = max(first_col_n, 1)
        cdef int ival, sz
        cdef char buf[64]

        cdef IntColumn ic
        cdef BoolColumn bc
        cdef UCS2Column uc
        cdef AsciiColumn cc
        cdef DateColumn tc
        cdef CategoryColumn catc
        cdef DateTimeColumn dtc
        cdef TimeColumn tmc

        cdef double* d_ptr = self.data_c
        cdef int32_t* tc_ptr
        cdef int64_t* dt_ptr
        cdef int32_t days
        cdef int32_t* tmc_ptr
        cdef int64_t tmp_dt

        if target_type == TYPE_INT:
            ic = IntColumn(safe_cap, cols=self.cols)
            ic.length = n
            with nogil:
                for i in range(n):
                    ic.data_c[i] = <int>d_ptr[i]
            return ic

        elif target_type == TYPE_BOOL:
            bc = BoolColumn(first_col_cap)
            bc.length = first_col_n
            with nogil:
                for i in range(first_col_n):
                    bc.data_c[i] = (d_ptr[i] != 0.0 and d_ptr[i] == d_ptr[i])
            return bc

        elif target_type == TYPE_UCS2:
            uc = UCS2Column(first_col_cap)
            uc.length = first_col_n
            for i in range(first_col_n):
                if d_ptr[i] != d_ptr[i]:
                    uc._set_item(i, "null")
                else:
                    sz = snprintf(buf, sizeof(buf), "%.6g", d_ptr[i])
                    uc._set_item(i, buf[:sz].decode('ascii'))
            return uc

        elif target_type == TYPE_CHAR256:
            cc = AsciiColumn(first_col_cap)
            cc.length = first_col_n
            with nogil:
                for i in range(first_col_n):
                    if d_ptr[i] != d_ptr[i]:
                        sz = snprintf(cc.data_c + (i * 256), 256, "null")
                    else:
                        sz = snprintf(cc.data_c + (i * 256), 256, "%.6g", d_ptr[i])
            return cc

        # ------------------------------------------------------------
        # Double -> Date (TIMESTAMP)：基于 1970 纪元天数转换
        # ------------------------------------------------------------
        elif target_type == TYPE_DATE:
            tc = DateColumn(safe_cap, cols=self.cols)
            tc.length = n
            tc_ptr = <int32_t*>tc.data_c
            with nogil:
                for i in range(n):
                    ival = <int>d_ptr[i]
                    if ival > 10000000:
                        tc_ptr[i] = _ymd_to_days(ival // 10000, (ival // 100) % 100, ival % 100)
                    else:
                        tc_ptr[i] = <int32_t>ival
            return tc

        elif target_type == TYPE_CATEGORY:
            catc = CategoryColumn(first_col_cap)
            for i in range(first_col_n):
                if d_ptr[i] != d_ptr[i]:
                    catc.append("null")
                else:
                    sz = snprintf(buf, sizeof(buf), "%.6g", d_ptr[i])
                    catc.append(buf[:sz].decode('ascii'))
            return catc

        # ------------------------------------------------------------
        # Double -> DateTime：彻底消除字符串中间层，纯算术微秒换算
        # ------------------------------------------------------------
        elif target_type == TYPE_DATETIME:
            dtc = DateTimeColumn(safe_cap, cols=self.cols)
            dtc.length = n
            dt_ptr = <int64_t*>dtc.data_c
            with nogil:
                for i in range(n):
                    ival = <int>d_ptr[i]
                    if ival > 10000000:
                        days = _ymd_to_days(ival // 10000, (ival // 100) % 100, ival % 100)
                        tmp_dt = <int64_t>days
                        dt_ptr[i] = tmp_dt * <int64_t>86400000000
                    else:
                        tmp_dt = <int64_t>(d_ptr[i] * 1000000.0)
                        dt_ptr[i] = tmp_dt
            return dtc

        elif target_type == TYPE_TIME:
            tmc = TimeColumn(safe_cap, cols=self.cols)
            tmc.length = n
            tmc_ptr = tmc.data_c
            with nogil:
                for i in range(n):
                    tmc_ptr[i] = <int32_t>d_ptr[i]
            return tmc

        elif target_type == TYPE_DOUBLE:
            return self.copy()

        cdef char err_buf[128]
        snprintf(err_buf, sizeof(err_buf), "Failed to cast DOUBLE column to internal type id: %d", target_type)
        raise TypeError(err_buf.decode('ascii'))


    # ============================================================
    # 1. 滚动均值 (roll_mean) - O(N) 自动支持 1D/2D 列优先展开
    # ============================================================
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_mean(self, Py_ssize_t window):
        """
        滚动均值计算。支持一维与二维矩阵。
        时间复杂度: O(rows * cols)，严格常数时间增量滑动。
        """
        if window < 0: raise ValueError("window must be >= 0")
        cdef DoubleColumn res = self._make_like_nan()
        cdef Py_ssize_t r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef Py_ssize_t c = self.num_cols
        if window <= 0 or r < window: return res

        cdef double inv_w = 1.0 / <double>window
        cdef double* src = self.data_c
        cdef double* dst = res.data_c
        cdef Py_ssize_t col_idx, offset, i
        cdef double s

        with nogil:
            for col_idx in range(c):
                offset = col_idx * r
                s = 0.0
                # 初始窗口预热
                for i in range(window):
                    s += src[offset + i]
                dst[offset + window - 1] = s * inv_w

                # 滑动更新
                for i in range(window, r):
                    s += src[offset + i] - src[offset + i - window]
                    dst[offset + i] = s * inv_w
        return res

    # ============================================================
    # 2. 滚动求和 (roll_sum) - O(N)
    # ============================================================
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_sum(self, Py_ssize_t window):
        """滚动累加和"""
        if window < 0: raise ValueError("window must be >= 0")
        cdef DoubleColumn res = self._make_like_nan()
        cdef Py_ssize_t r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef Py_ssize_t c = self.num_cols
        if window <= 0 or r < window: return res

        cdef double* src = self.data_c
        cdef double* dst = res.data_c
        cdef Py_ssize_t col_idx, offset, i
        cdef double s

        with nogil:
            for col_idx in range(c):
                offset = col_idx * r
                s = 0.0
                for i in range(window):
                    s += src[offset + i]
                dst[offset + window - 1] = s

                for i in range(window, r):
                    s += src[offset + i] - src[offset + i - window]
                    dst[offset + i] = s
        return res

    # ============================================================
    # 3. 滚动极值 (roll_min / roll_max) - O(N) 单调队列
    # ============================================================
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_min(self, Py_ssize_t window):
        """滚动最小值 (采用严格连续内存单调双端队列)"""
        if window < 1: raise ValueError("window must be >= 1")
        cdef DoubleColumn res = self._make_like_nan()
        cdef Py_ssize_t r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef Py_ssize_t c = self.num_cols
        if r < window: return res

        # 分配单列复用的队列缓冲 (最大长度为 r)
        cdef Py_ssize_t* q = <Py_ssize_t*>aligned_malloc(r * sizeof(Py_ssize_t))
        if not q: raise MemoryError()

        cdef double* src = self.data_c
        cdef double* dst = res.data_c
        cdef Py_ssize_t col_idx, offset, i, head, tail

        try:
            with nogil:
                for col_idx in range(c):
                    offset = col_idx * r
                    head = 0
                    tail = 0
                    for i in range(r):
                        # 弹出窗口外旧元素
                        while head < tail and q[head] <= i - window:
                            head += 1
                        # 维护单调递增性
                        while head < tail and src[offset + q[tail - 1]] >= src[offset + i]:
                            tail -= 1
                        q[tail] = i
                        tail += 1
                        if i >= window - 1:
                            dst[offset + i] = src[offset + q[head]]
        finally:
            os_aligned_free(q)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_max(self, Py_ssize_t window):
        """滚动最大值"""
        if window < 1: raise ValueError("window must be >= 1")
        cdef DoubleColumn res = self._make_like_nan()
        cdef Py_ssize_t r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef Py_ssize_t c = self.num_cols
        if r < window: return res

        cdef Py_ssize_t* q = <Py_ssize_t*>aligned_malloc(r * sizeof(Py_ssize_t))
        if not q: raise MemoryError()

        cdef double* src = self.data_c
        cdef double* dst = res.data_c
        cdef Py_ssize_t col_idx, offset, i, head, tail

        try:
            with nogil:
                for col_idx in range(c):
                    offset = col_idx * r
                    head = 0
                    tail = 0
                    for i in range(r):
                        while head < tail and q[head] <= i - window:
                            head += 1
                        while head < tail and src[offset + q[tail - 1]] <= src[offset + i]:
                            tail -= 1
                        q[tail] = i
                        tail += 1
                        if i >= window - 1:
                            dst[offset + i] = src[offset + q[head]]
        finally:
            os_aligned_free(q)
        return res

    # ============================================================
    # 4. 滚动方差与标准差 (roll_var / roll_std) - O(N) 稳定滑窗
    # ============================================================
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_var(self, Py_ssize_t window, bint ddof=True):
        """滚动方差 (ddof=True 为样本方差 N-1，False 为总体方差 N)"""
        if window < 2: raise ValueError("window must be >= 2 for variance")
        cdef DoubleColumn res = self._make_like_nan()
        cdef Py_ssize_t r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef Py_ssize_t c = self.num_cols
        if r < window: return res

        cdef double inv_w = 1.0 / <double>window
        cdef double denom = 1.0 / <double>(window - 1) if ddof else inv_w
        cdef double* src = self.data_c
        cdef double* dst = res.data_c
        cdef Py_ssize_t col_idx, offset, i
        cdef double s, sq_s, val, old_val, v

        with nogil:
            for col_idx in range(c):
                offset = col_idx * r
                s = 0.0
                sq_s = 0.0
                for i in range(window):
                    val = src[offset + i]
                    s += val
                    sq_s += val * val
                v = (sq_s - (s * s) * inv_w) * denom
                dst[offset + window - 1] = v if v > 0.0 else 0.0

                for i in range(window, r):
                    val = src[offset + i]
                    old_val = src[offset + i - window]
                    s += val - old_val
                    sq_s += val * val - old_val * old_val
                    v = (sq_s - (s * s) * inv_w) * denom
                    dst[offset + i] = v if v > 0.0 else 0.0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_std(self, Py_ssize_t window, bint ddof=True):
        """滚动标准差"""
        cdef DoubleColumn res = self.roll_var(window, ddof=ddof)
        cdef Py_ssize_t i, n = res.length
        cdef double* ptr = res.data_c
        with nogil:
            for i in range(n):
                if ptr[i] == ptr[i]: # 非 NaN
                    ptr[i] = sqrt(ptr[i])
        return res

    # ============================================================
    # 5. 滚动中位数与分位数 (roll_median / roll_quantile)
    # ============================================================
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_quantile(self, Py_ssize_t window, double q):
        """滚动任意分位数 (q 在 0.0 到 1.0 之间)"""
        if window < 1: raise ValueError("window must be >= 1")
        if q < 0.0 or q > 1.0: raise ValueError("q must be between 0.0 and 1.0")
        cdef DoubleColumn res = self._make_like_nan()
        cdef Py_ssize_t r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef Py_ssize_t c = self.num_cols
        if r < window: return res

        cdef double* buf = <double*>aligned_malloc(window * sizeof(double))
        if not buf: raise MemoryError()

        cdef double pos = q * <double>(window - 1)
        cdef Py_ssize_t i_low = <Py_ssize_t>pos
        cdef double frac = pos - <double>i_low

        cdef double* src = self.data_c
        cdef double* dst = res.data_c
        cdef Py_ssize_t col_idx, offset, i

        try:
            with nogil:
                for col_idx in range(c):
                    offset = col_idx * r
                    for i in range(window - 1, r):
                        memcpy(buf, src + offset + (i - window + 1), window * sizeof(double))
                        qsort(buf, window, sizeof(double), _cmp_double_asc)
                        if i_low >= window - 1:
                            dst[offset + i] = buf[window - 1]
                        else:
                            dst[offset + i] = buf[i_low] + frac * (buf[i_low + 1] - buf[i_low])
        finally:
            os_aligned_free(buf)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_median(self, Py_ssize_t window):
        """滚动中位数"""
        return self.roll_quantile(window, 0.5)

    # ============================================================
    # 6. 自定义权重滤波 (roll_apply_weights) - 极速 FIR / 卷积滤波
    # ============================================================
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _roll_apply_weights(self, object weights):
        """
        滚动权重内积滤波 (用于加权移动平均 WMA、高斯平滑等)
        weights 可以是 list、tuple 或 1D DoubleColumn/NumPy 数组。
        纯 C 指针点积，无 Python 回调开销。
        """
        cdef DoubleColumn w_col
        cdef Py_ssize_t window
        if isinstance(weights, DoubleColumn):
            w_col = <DoubleColumn>weights
            window = w_col.length
        else:
            w_col = DoubleColumn(len(weights))
            for v in weights:
                w_col.append(<double>v)
            window = w_col.length

        if window < 1: raise ValueError("weights length must be >= 1")
        cdef DoubleColumn res = self._make_like_nan()
        cdef Py_ssize_t r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef Py_ssize_t c = self.num_cols
        if r < window: return res

        cdef double* w_ptr = w_col.data_c
        cdef double* src = self.data_c
        cdef double* dst = res.data_c
        cdef Py_ssize_t col_idx, offset, i, k
        cdef double acc

        with nogil:
            for col_idx in range(c):
                offset = col_idx * r
                for i in range(window - 1, r):
                    acc = 0.0
                    for k in range(window):
                        acc += src[offset + (i - window + 1 + k)] * w_ptr[k]
                    dst[offset + i] = acc
        return res

    # ============================================================
    # 7. 纯自定义函数滚动 (roll_custom) - 支持外部 Python 回调
    # ============================================================
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _roll_custom(self, Py_ssize_t window, object func):
        """
        通用滚动自定义滤波。每次滑窗时切片构建一个临时视图传递给 func(slice_col)。
        func 必须返回一个可以转换为 float 的标量数值。
        """
        if window < 1: raise ValueError("window must be >= 1")
        if not callable(func): raise TypeError("func must be a callable object")
        cdef DoubleColumn res = self._make_like_nan()
        cdef Py_ssize_t r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef Py_ssize_t c = self.num_cols
        if r < window: return res

        cdef double* src = self.data_c
        cdef double* dst = res.data_c
        cdef Py_ssize_t col_idx, offset, i
        cdef DoubleColumn window_view
        cdef object ret_val

        for col_idx in range(c):
            offset = col_idx * r
            for i in range(window - 1, r):
                window_view = DoubleColumn.create_view(
                    src + offset + (i - window + 1),
                    window,
                    self
                )
                ret_val = func(window_view)
                dst[offset + i] = <double>ret_val if ret_val is not None else NAN
        return res

    # ============================================================
    # 8. roll_skew / roll_kurt / roll_corr / roll_cov
    # ============================================================
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_skew(self, Py_ssize_t window):
        cdef Py_ssize_t r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef Py_ssize_t c = self.num_cols
        cdef DoubleColumn res = self._make_like_nan()
        if r < window or window < 3: return res
        cdef double* src = self.data_c
        cdef double* dst = res.data_c
        cdef double n_f = <double>window
        cdef Py_ssize_t col_idx, offset, i, j
        cdef double s, mean, m2, m3, val, std_dev
        with nogil:
            for col_idx in range(c):
                offset = col_idx * r
                for i in range(window - 1, r):
                    s = 0.0
                    for j in range(i - window + 1, i + 1):
                        s += src[offset + j]
                    mean = s / n_f
                    m2 = 0.0; m3 = 0.0
                    for j in range(i - window + 1, i + 1):
                        val = src[offset + j] - mean
                        m2 += val * val
                        m3 += val * val * val
                    if m2 > 1e-14:
                        std_dev = sqrt(m2 / n_f)
                        dst[offset + i] = (m3 / n_f) / (std_dev * std_dev * std_dev)
                    else:
                        dst[offset + i] = 0.0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_kurt(self, Py_ssize_t window):
        cdef Py_ssize_t r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef Py_ssize_t c = self.num_cols
        cdef DoubleColumn res = self._make_like_nan()
        if r < window or window < 4: return res
        cdef double* src = self.data_c
        cdef double* dst = res.data_c
        cdef double n_f = <double>window
        cdef Py_ssize_t col_idx, offset, i, j
        cdef double s, mean, m2, m4, val, variance
        with nogil:
            for col_idx in range(c):
                offset = col_idx * r
                for i in range(window - 1, r):
                    s = 0.0
                    for j in range(i - window + 1, i + 1):
                        s += src[offset + j]
                    mean = s / n_f
                    m2 = 0.0; m4 = 0.0
                    for j in range(i - window + 1, i + 1):
                        val = src[offset + j] - mean
                        m2 += val * val
                        m4 += val * val * val * val
                    if m2 > 1e-14:
                        variance = m2 / n_f
                        dst[offset + i] = (m4 / n_f) / (variance * variance) - 3.0
                    else:
                        dst[offset + i] = 0.0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_corr(self, Py_ssize_t window, object other):
        cdef Py_ssize_t r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef Py_ssize_t c = self.num_cols
        cdef DoubleColumn res = self._make_like_nan()
        if r < window: return res
        cdef double* src_x = self.data_c
        cdef double* dst = res.data_c
        cdef double* src_y
        cdef double n_f = <double>window, inv_w = 1.0 / n_f
        cdef Py_ssize_t col_idx, offset, i
        cdef double sum_x, sum_y, sum_xx, sum_yy, sum_xy, x, y, old_x, old_y, vx, vy, denom, cov_xy
        cdef bint other_is_double = (<object>other).dtype == TYPE_DOUBLE
        with nogil:
            for col_idx in range(c):
                offset = col_idx * r
                sum_x = 0.0; sum_y = 0.0; sum_xx = 0.0; sum_yy = 0.0; sum_xy = 0.0
                for i in range(window):
                    x = src_x[offset + i]
                    if other_is_double:
                        y = (<DoubleColumn>other).data_c[offset + i]
                    else:
                        y = <double>((<IntColumn>other).data_c[offset + i])
                    sum_x += x; sum_y += y; sum_xx += x * x; sum_yy += y * y; sum_xy += x * y
                vx = sum_xx - sum_x * sum_x * inv_w
                vy = sum_yy - sum_y * sum_y * inv_w
                denom = sqrt(vx * vy)
                cov_xy = sum_xy - sum_x * sum_y * inv_w
                dst[offset + window - 1] = cov_xy / denom if (denom > 0.0) else 0.0
                for i in range(window, r):
                    x = src_x[offset + i]
                    if other_is_double:
                        y = (<DoubleColumn>other).data_c[offset + i]
                        old_x = src_x[offset + i - window]
                        old_y = (<DoubleColumn>other).data_c[offset + i - window]
                    else:
                        y = <double>((<IntColumn>other).data_c[offset + i])
                        old_x = src_x[offset + i - window]
                        old_y = <double>((<IntColumn>other).data_c[offset + i - window])
                    sum_x += x - old_x; sum_y += y - old_y
                    sum_xx += x * x - old_x * old_x; sum_yy += y * y - old_y * old_y
                    sum_xy += x * y - old_x * old_y
                    vx = sum_xx - sum_x * sum_x * inv_w
                    vy = sum_yy - sum_y * sum_y * inv_w
                    denom = sqrt(vx * vy)
                    cov_xy = sum_xy - sum_x * sum_y * inv_w
                    dst[offset + i] = cov_xy / denom if (denom > 0.0) else 0.0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_cov(self, Py_ssize_t window, object other):
        cdef Py_ssize_t r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef Py_ssize_t c = self.num_cols
        cdef DoubleColumn res = self._make_like_nan()
        if r < window: return res
        cdef double* src_x = self.data_c
        cdef double* dst = res.data_c
        cdef double n_f = <double>window, inv_w = 1.0 / n_f
        cdef Py_ssize_t col_idx, offset, i
        cdef double sum_x, sum_y, sum_xy, x, y, old_x, old_y
        cdef bint other_is_double = (<object>other).dtype == TYPE_DOUBLE
        with nogil:
            for col_idx in range(c):
                offset = col_idx * r
                sum_x = 0.0; sum_y = 0.0; sum_xy = 0.0
                for i in range(window):
                    x = src_x[offset + i]
                    if other_is_double:
                        y = (<DoubleColumn>other).data_c[offset + i]
                    else:
                        y = <double>((<IntColumn>other).data_c[offset + i])
                    sum_x += x; sum_y += y; sum_xy += x * y
                dst[offset + window - 1] = (sum_xy - sum_x * sum_y * inv_w) * inv_w
                for i in range(window, r):
                    x = src_x[offset + i]
                    if other_is_double:
                        y = (<DoubleColumn>other).data_c[offset + i]
                        old_x = src_x[offset + i - window]
                        old_y = (<DoubleColumn>other).data_c[offset + i - window]
                    else:
                        y = <double>((<IntColumn>other).data_c[offset + i])
                        old_x = src_x[offset + i - window]
                        old_y = <double>((<IntColumn>other).data_c[offset + i - window])
                    sum_x += x - old_x; sum_y += y - old_y
                    sum_xy += x * y - old_x * old_y
                    dst[offset + i] = (sum_xy - sum_x * sum_y * inv_w) * inv_w
        return res

    # ============================================================
    # 9. 统一动态分发入口 (roll)
    # ============================================================
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _roll(self, Py_ssize_t window, str method="mean", **kwargs):
        cdef str m = method.lower()
        if m == "mean":
            return self.roll_mean(window)
        elif m == "sum":
            return self.roll_sum(window)
        elif m == "min":
            return self.roll_min(window)
        elif m == "max":
            return self.roll_max(window)
        elif m == "std":
            return self.roll_std(window, ddof=kwargs.get("ddof", True))
        elif m == "var":
            return self.roll_var(window, ddof=kwargs.get("ddof", True))
        elif m == "median":
            return self.roll_median(window)
        elif m == "quantile":
            return self.roll_quantile(window, kwargs.get("q", 0.5))
        elif m == "skew":
            return self.roll_skew(window)
        elif m == "kurt":
            return self.roll_kurt(window)
        else:
            raise ValueError(f"Unsupported rolling method: '{method}'")