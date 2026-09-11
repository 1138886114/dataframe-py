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
from libc.stdlib cimport qsort, bsearch
from libc.stdint cimport int32_t, int64_t
from fluidframe.core._base cimport (
    TYPE_INT, TYPE_DOUBLE, TYPE_CHAR256, TYPE_UCS2, TYPE_BOOL,
    TYPE_DATE, TYPE_CATEGORY, TYPE_DATETIME, TYPE_TIME,
    BaseColumn, aligned_malloc, aligned_realloc, os_aligned_free,
    _resolve_astype_target, NumericType,
    _vec_pearson_corr, _vec_pearson_cov, _vec_euclidean_dist,
    _vec_manhattan_dist, _vec_cosine_sim, _vec_jaccard_sim, _vec_hamming_dist,
    _int_double_pearson_corr, _int_double_pearson_cov, _int_double_euclidean_dist,
    _int_double_manhattan_dist, _int_double_cosine_sim, _int_double_jaccard_sim, _int_double_hamming_dist,
    _c_shift_left, _c_shift_right, _col_pop_back, _col_pop_front, _col_concat,
    _col_drop_row, _col_gather,
    _col_shift_right_2d_int, _col_shift_left_2d, _col_gather_2d, _col_concat_2d,
    _col_gather_zero, _col_gather_2d_zero, _col_count_nonzeros, _col_memcpy,
    _max_2d_c, _min_2d_c, _sum_2d_c, _mean_2d_c, _prod_2d_c, _var_2d_c, _std_2d_c
)
from fluidframe.core.doublecolumn cimport DoubleColumn
from fluidframe.core.boolcolumn cimport BoolColumn
from fluidframe.core.asciicolumn cimport AsciiColumn
from fluidframe.core.ucs2column cimport UCS2Column
from fluidframe.core.datecolumn cimport DateColumn, _ymd_to_days
from fluidframe.core.categorycolumn cimport CategoryColumn
from fluidframe.core.datetimecolumn cimport DateTimeColumn
from fluidframe.core.timecolumn cimport TimeColumn

DEF US_PER_DAY = 86400000000LL

cdef int _cmp_int(const void* a, const void* b) noexcept nogil:
    cdef int va = (<const int*>a)[0]
    cdef int vb = (<const int*>b)[0]
    if va < vb: return -1
    if va > vb: return 1
    return 0

cdef inline bint _binary_search(int* arr, Py_ssize_t n, int target) noexcept nogil:
    cdef Py_ssize_t low = 0, high = n - 1, mid
    cdef int val
    while low <= high:
        mid = (low + high) // 2
        val = arr[mid]
        if val == target:
            return True
        elif val < target:
            low = mid + 1
        else:
            high = mid - 1
    return False

@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef class IntColumn(BaseColumn):

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __cinit__(self, Py_ssize_t init_capacity=16, Py_ssize_t cols=1):
        self.capacity = max(init_capacity, 1)
        self.length = 0
        self.dtype = TYPE_INT

        self.num_cols = <int>cols if cols > 0 else 1
        self.is_2d = (self.num_cols > 1)
        self.owns_data = True
        self._base_obj = None

        self.data_c = <int*>aligned_malloc(self.capacity * sizeof(int))
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

    # ============================================================
    # 对外暴露的兼容性属性 (Properties)
    # ============================================================
    @property
    def cols(self): return self.num_cols

    @property
    def rows(self): return self._get_rows_c()

    @property
    def _owns_data(self): return self.owns_data

    @_owns_data.setter
    def _owns_data(self, bint val): self.owns_data = val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def get_size(self):
        return self.capacity * sizeof(int)

    # ============================================================
    # 创建与初始化
    # ============================================================
    @staticmethod
    cdef IntColumn create_view(int* ptr, Py_ssize_t length, object base_obj):
        cdef IntColumn col = IntColumn.__new__(IntColumn)
        col.dtype = TYPE_INT
        col.capacity = length
        col.length = length
        col.data_c = ptr
        col._base_obj = base_obj
        col.num_cols = 1
        col.is_2d = False
        col.owns_data = False
        return col

    @staticmethod
    cdef IntColumn create_view_2d(int* ptr, Py_ssize_t rows, Py_ssize_t cols, object base_obj):
        cdef IntColumn col = IntColumn.__new__(IntColumn)
        cdef Py_ssize_t total = rows * cols
        col.dtype = TYPE_INT
        col.capacity = total
        col.length = total
        col.data_c = ptr
        col._base_obj = base_obj
        col.num_cols = <int>cols
        col.is_2d = True
        col.owns_data = False
        return col

    cdef inline DoubleColumn _make_like_nan_dbl(self):
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t total = self.length
        cdef DoubleColumn res = DoubleColumn(max(total, 1), cols=c)
        res.length = total
        cdef double* res_ptr = res.data_c
        with nogil:
            for i in range(total):
                res_ptr[i] = NAN
        return res

    @staticmethod
    def from_2d(object values, Py_ssize_t rows, Py_ssize_t cols):
        cdef Py_ssize_t total = rows * cols
        if len(values) != total: raise ValueError(f"len(values)={len(values)} != rows*cols={total}")
        cdef IntColumn col = IntColumn(max(total, 1), cols=cols)
        col.num_cols = <int>cols
        col.is_2d = True
        col.length = total
        cdef Py_ssize_t i
        cdef int* ptr = col.data_c
        for i in range(total):
            ptr[i] = <int>values[i]
        return col

    @staticmethod
    def load_from_1d_array(object arr):
        cdef int[::1] view
        try:
            view = arr
        except (ValueError, TypeError):
            raise ValueError(
                "Input must be a 1D contiguous int32 array. "
                "If using NumPy, please pass np.ascontiguousarray(arr, dtype=np.int32) explicitly."
            )
        cdef Py_ssize_t length = view.shape[0]
        cdef IntColumn col = IntColumn.__new__(IntColumn)
        col.dtype = TYPE_INT
        col.capacity = length
        col.length = length
        col.owns_data = False
        col._base_obj = arr
        col.data_c = &view[0] if length > 0 else NULL
        return col

    @staticmethod
    def load_from_2d_array(object arr, list col_names):
        cdef int[::1, :] view
        try:
            view = arr
        except (ValueError, TypeError):
            raise ValueError(
                "Input array must be a 2D Fortran-contiguous (column-major) int32 array. "
                "If using NumPy, please pass np.asfortranarray(arr, dtype=np.int32) explicitly."
            )
        cdef Py_ssize_t rows = view.shape[0]
        cdef Py_ssize_t cols = view.shape[1]
        if len(col_names) != cols:
            raise ValueError(f"col_names length ({len(col_names)}) != array columns ({cols})")
        if rows == 0 or cols == 0:
            return {}
        cdef int* raw_ptr = &view[0, 0]
        cdef dict columns = {}
        cdef Py_ssize_t j
        for j in range(cols):
            columns[col_names[j]] = IntColumn.create_view(
                raw_ptr + (j * rows),
                rows,
                arr
            )
        return columns

    cdef void _reserve(self, Py_ssize_t new_cap):
        if not self.owns_data:
            raise RuntimeError("Cannot reallocate a view/slice column. Copy it first.")
        cdef int* new_ptr
        if new_cap > self.capacity:
            new_ptr = <int*>aligned_realloc(self.data_c, self.capacity * sizeof(int), new_cap * sizeof(int))
            if not new_ptr: raise MemoryError()
            self.data_c = new_ptr
            self.capacity = new_cap

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def append(self, int val):
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
        cdef IntColumn new_col = IntColumn(max(self.length, 1), cols=self.num_cols)
        new_col.length = self.length
        new_col.is_2d = self.is_2d
        if self.length > 0:
            memcpy(new_col.data_c, self.data_c, self.length * sizeof(int))
        return new_col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_back(self):
        if not self.owns_data: raise RuntimeError("Cannot modify a view/slice.")
        if self.length == 0: raise IndexError("pop from empty column")
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r, j
        cdef list vals
        if self.is_2d:
            r = self._get_rows_c()
            vals = [None] * c
            for j in range(c):
                vals[j] = self.data_c[j * r + (r - 1)]
            _col_shift_left_2d(self.data_c, &self.length, r, c, r - 1, sizeof(int))
            return vals
        cdef int* ptr = <int*>_col_pop_back(self.data_c, &self.length, sizeof(int))
        return ptr[0]

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_front(self):
        if not self.owns_data: raise RuntimeError("Cannot modify a view/slice.")
        if self.length == 0: raise IndexError("pop from empty column")
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r, j
        cdef list vals
        if self.is_2d:
            r = self._get_rows_c()
            vals = [None] * c
            for j in range(c):
                vals[j] = self.data_c[j * r]
            _col_shift_left_2d(self.data_c, &self.length, r, c, 0, sizeof(int))
            return vals
        cdef int val = self.data_c[0]
        _col_pop_front(self.data_c, &self.length, sizeof(int))
        return val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def concat(self, IntColumn other):
        if not self.owns_data: raise RuntimeError("Cannot modify a view/slice.")
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t other_c = other.num_cols
        cdef Py_ssize_t other_rows

        if self.is_2d or other.is_2d:
            if c != other_c:
                if self.length == 0: c = other_c
                else: raise ValueError(f"Cannot concat 2D columns with diff cols: {c} vs {other_c}")
            other_rows = other._get_rows_c()
            if other_rows == 0: return

            self._reserve(self.length + other_rows * c)
            _col_concat_2d(self.data_c, &self.length, other.data_c, other_rows, c, sizeof(int))

            self.num_cols = <int>c
            self.is_2d = True
            return

        if other.length == 0: return
        self._reserve(self.length + other.length)
        _col_concat(self.data_c, &self.length, other.data_c, other.length, sizeof(int))

    # ============================================================
    # Python 层公开接口：智能 2D 感知聚合
    # ============================================================
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def max(self):
        if self.length == 0: raise ValueError("empty column")
        cdef int* ptr = self.data_c
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()
        cdef Py_ssize_t i, j, base
        cdef int max_val

        if not self.is_2d or c <= 1:
            max_val = ptr[0]
            with nogil:
                for i in range(1, r):
                    if ptr[i] > max_val: max_val = ptr[i]
            return max_val

        cdef list results = [None] * c
        with nogil:
            for j in range(c):
                base = j * r
                max_val = ptr[base]
                for i in range(1, r):
                    if ptr[base + i] > max_val: max_val = ptr[base + i]

                with gil:
                    results[j] = max_val
        return results

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def min(self):
        if self.length == 0: raise ValueError("empty column")
        cdef int* ptr = self.data_c
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()
        cdef Py_ssize_t i, j, base
        cdef int min_val

        if not self.is_2d or c <= 1:
            min_val = ptr[0]
            with nogil:
                for i in range(1, r):
                    if ptr[i] < min_val: min_val = ptr[i]
            return min_val

        cdef list results = [None] * c
        with nogil:
            for j in range(c):
                base = j * r
                min_val = ptr[base]
                for i in range(1, r):
                    if ptr[base + i] < min_val: min_val = ptr[base + i]
                with gil:
                    results[j] = min_val
        return results

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sum(self):
        if self.length == 0: return 0
        cdef int* ptr = self.data_c
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()
        cdef Py_ssize_t i, j, base
        cdef int64_t total

        if not self.is_2d or c <= 1:
            total = 0
            with nogil:
                for i in range(r): total += ptr[i]
            return total

        cdef list results = [None] * c
        with nogil:
            for j in range(c):
                base = j * r
                total = 0
                for i in range(r): total += ptr[base + i]
                with gil:
                    results[j] = total
        return results

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def mean(self):
        if self.length == 0: return NAN
        return <double>self.sum() / <double>self.length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def var(self):
        if self.length < 2: return 0.0
        cdef double m = self.mean()
        cdef double sum_sq = 0.0, diff
        cdef int* ptr = self.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                diff = <double>ptr[i] - m
                sum_sq += diff * diff
        return sum_sq / (n - 1)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def std(self):
        return sqrt(self.var())

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def count_nonzeros(self):
        return _col_count_nonzeros(self.data_c, self.length)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def idxmax(self):
        if self.length == 0: raise ValueError("empty column")
        cdef Py_ssize_t max_idx = 0
        cdef int max_val = self.data_c[0]
        cdef int* ptr = self.data_c
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
        if self.length == 0: raise ValueError("empty column")
        cdef Py_ssize_t min_idx = 0
        cdef int min_val = self.data_c[0]
        cdef int* ptr = self.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(1, n):
                if ptr[i] < min_val:
                    min_val = ptr[i]
                    min_idx = i
        return min_idx

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def argmax(self): return self.idxmax()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def argmin(self): return self.idxmin()

    cdef double _quickselect(self, int* a, Py_ssize_t n, Py_ssize_t k) noexcept nogil:
        cdef Py_ssize_t l = 0, r = n - 1
        cdef Py_ssize_t i, lt, gt
        cdef int pivot, tmp

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
                return <double>a[k]
        return <double>a[k]

    cdef inline void _max_2d_c(self, double* out_buf) noexcept nogil:
        _max_2d_c(self.data_c, self.get_rows_c(), self.get_cols_c(), out_buf)

    cdef inline void _min_2d_c(self, double* out_buf) noexcept nogil:
        _min_2d_c(self.data_c, self.get_rows_c(), self.get_cols_c(), out_buf)

    cdef inline void _sum_2d_c(self, double* out_buf) noexcept nogil:
        _sum_2d_c(self.data_c, self.get_rows_c(), self.get_cols_c(), out_buf)

    cdef inline void _mean_2d_c(self, double* out_buf) noexcept nogil:
        _mean_2d_c(self.data_c, self.get_rows_c(), self.get_cols_c(), out_buf)

    cdef inline void _prod_2d_c(self, double* out_buf) noexcept nogil:
        _prod_2d_c(self.data_c, self.get_rows_c(), self.get_cols_c(), out_buf)

    cdef inline void _var_2d_c(self, double* out_buf) noexcept nogil:
        _var_2d_c(self.data_c, self.get_rows_c(), self.get_cols_c(), out_buf)

    cdef inline void _std_2d_c(self, double* out_buf) noexcept nogil:
        _std_2d_c(self.data_c, self.get_rows_c(), self.get_cols_c(), out_buf)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def median(self):
        if self.length == 0: return NAN
        cdef int* temp = <int*>aligned_malloc(self.length * sizeof(int))
        if not temp: raise MemoryError()
        memcpy(temp, self.data_c, self.length * sizeof(int))
        cdef Py_ssize_t mid = self.length // 2
        cdef double v1, v2, result
        try:
            if self.length % 2 == 0:
                v1 = self._quickselect(temp, self.length, mid - 1)
                v2 = self._quickselect(temp, self.length, mid)
                return (v1 + v2) * 0.5
            result = self._quickselect(temp, self.length, mid)
            return result
        finally:
            os_aligned_free(temp)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def mad(self):
        if self.length == 0: return NAN
        cdef double m = self.mean()
        cdef double total_dev = 0.0
        cdef int* ptr = self.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                total_dev += fabs(<double>ptr[i] - m)
        return total_dev / <double>n

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def prod(self):
        if self.length == 0: return NAN
        cdef double total = 1.0
        cdef int* ptr = self.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                total *= <double>ptr[i]
        return total

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def nunique(self):
        if self.length == 0: return 0
        cdef set seen = set()
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            seen.add(self.data_c[i])
        return len(seen)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def value_counts(self):
        cdef dict counts = {}
        cdef Py_ssize_t i, n = self.length
        cdef int val
        for i in range(n):
            val = self.data_c[i]
            counts[val] = counts.get(val, 0) + 1
        return counts

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def mode(self):
        if self.length == 0: return []
        cdef dict counts = {}
        cdef Py_ssize_t i, n = self.length
        cdef int val, count, max_count = 0
        for i in range(n):
            val = self.data_c[i]
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
        cdef int* temp = <int*>aligned_malloc(self.length * sizeof(int))
        if not temp: raise MemoryError()
        memcpy(temp, self.data_c, self.length * sizeof(int))
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
        cdef IntColumn res = IntColumn(max(self.capacity, 1), cols=self.cols)
        res.length = self.length
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef int total
        cdef int* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t col_idx, offset, i
        with nogil:
            for col_idx in range(c):
                offset = col_idx * r
                total = 0
                for i in range(r):
                    total += ptr[offset + i]
                    res_ptr[offset + i] = total
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cummax(self):
        cdef IntColumn res = IntColumn(max(self.capacity, 1), cols=self.cols)
        res.length = self.length
        if self.length == 0: return res
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef int cur_max
        cdef int* ptr = self.data_c
        cdef int* res_ptr = res.data_c
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
        cdef IntColumn res = IntColumn(max(self.capacity, 1), cols=self.cols)
        res.length = self.length
        if self.length == 0: return res
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef int cur_min
        cdef int* ptr = self.data_c
        cdef int* res_ptr = res.data_c
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
    def diff(self, int periods=1):
        cdef IntColumn res = IntColumn(max(self.capacity, 1), cols=self.cols)
        res.length = self.length
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef int* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t col_idx, offset, i
        cdef int p = periods if periods >= 0 else -periods
        with nogil:
            if periods >= 0:
                for col_idx in range(c):
                    offset = col_idx * r
                    for i in range(min(p, r)):
                        res_ptr[offset + i] = 0
                    for i in range(p, r):
                        res_ptr[offset + i] = ptr[offset + i] - ptr[offset + i - p]
            else:
                for col_idx in range(c):
                    offset = col_idx * r
                    for i in range(max(0, r - p), r):
                        res_ptr[offset + i] = 0
                    for i in range(r - p):
                        res_ptr[offset + i] = ptr[offset + i] - ptr[offset + i + p]
        return res

    @cython.binding(False)
    def pct_change(self, int periods=1):
        cdef DoubleColumn res = DoubleColumn(max(self.capacity, 1), cols=self.cols)
        res.length = self.length
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef int* ptr = self.data_c
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
                        prev = <double>ptr[offset + i - p]
                        res_ptr[offset + i] = (<double>ptr[offset + i] - prev) / prev if prev != 0.0 else NAN
            else:
                for col_idx in range(c):
                    offset = col_idx * r
                    for i in range(max(0, r - p), r):
                        res_ptr[offset + i] = NAN
                    for i in range(r - p):
                        prev = <double>ptr[offset + i + p]
                        res_ptr[offset + i] = (<double>ptr[offset + i] - prev) / prev if prev != 0.0 else NAN
        return res

    cdef void _central_moments(self, double* out_mean, double* out_m2, double* out_m3, double* out_m4) noexcept nogil:
        cdef Py_ssize_t n = self.length
        if n == 0:
            out_mean[0] = 0.0; out_m2[0] = 0.0; out_m3[0] = 0.0; out_m4[0] = 0.0
            return
        cdef double mean = 0.0, m2 = 0.0, m3 = 0.0, m4 = 0.0
        cdef double val, delta, delta_n, delta_n2, term1
        cdef Py_ssize_t i, n1
        cdef int* ptr = self.data_c
        for i in range(n):
            n1 = i
            val = <double>ptr[i]
            delta = val - mean
            mean += delta / (n1 + 1)
            delta_n = val - mean
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
        if self.length < 3: return NAN
        cdef double m, m2, m3, m4
        self._central_moments(&m, &m2, &m3, &m4)
        if m2 <= 1e-14: return NAN
        cdef double n = <double>self.length
        cdef double var_sample = m2 / (n - 1)
        return (n * m3) / ((n - 1) * (n - 2) * (var_sample * sqrt(var_sample)))

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def kurt(self):
        if self.length < 4: return NAN
        cdef double m, m2, m3, m4
        self._central_moments(&m, &m2, &m3, &m4)
        if m2 <= 1e-14: return NAN
        cdef double n = <double>self.length
        cdef double var_sample = m2 / (n - 1)
        return (m4 / (n - 1)) / (var_sample * var_sample) - 3.0

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sem(self):
        if self.length < 2: return NAN
        cdef double m, m2, m3, m4
        self._central_moments(&m, &m2, &m3, &m4)
        return sqrt(m2 / ((<double>self.length - 1) * <double>self.length))

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def count(self):
        return self.length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cumprod(self):
        cdef IntColumn res = IntColumn(max(self.capacity, 1), cols=self.cols)
        res.length = self.length
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef int64_t total
        cdef int* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t col_idx, offset, i
        with nogil:
            for col_idx in range(c):
                offset = col_idx * r
                total = 1
                for i in range(r):
                    total *= ptr[offset + i]
                    res_ptr[offset + i] = <int>total
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def describe(self):
        if self.length == 0:
            return {"count": 0, "mean": NAN, "std": NAN, "min": NAN,
                    "25%": NAN, "50%": NAN, "75%": NAN, "max": NAN}
        return {
            "count": self.length,
            "mean": self.mean(),
            "std": self.std(),
            "min": <double>self.min(),
            "25%": self.quantile(0.25),
            "50%": self.quantile(0.50),
            "75%": self.quantile(0.75),
            "max": <double>self.max()
        }

    cdef double _sum_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil:
        cdef double total = 0.0
        cdef int* ptr = self.data_c
        cdef Py_ssize_t i
        for i in range(n):
            total += ptr[indices[i]]
        return total

    cdef double _mean_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil:
        if n == 0: return NAN
        return self._sum_rows(indices, n) / <double>n

    cdef double _min_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil:
        if n == 0: return NAN
        cdef int* ptr = self.data_c
        cdef double val, min_val = <double>ptr[indices[0]]
        cdef Py_ssize_t i
        for i in range(1, n):
            val = <double>ptr[indices[i]]
            if val < min_val: min_val = val
        return min_val

    cdef double _max_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil:
        if n == 0: return NAN
        cdef int* ptr = self.data_c
        cdef double val, max_val = <double>ptr[indices[0]]
        cdef Py_ssize_t i
        for i in range(1, n):
            val = <double>ptr[indices[i]]
            if val > max_val: max_val = val
        return max_val

    cdef Py_ssize_t _count_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil:
        return n

    cdef double _var_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil:
        if n < 2: return 0.0
        cdef double m = self._mean_rows(indices, n)
        cdef double sum_sq = 0.0, v
        cdef int* ptr = self.data_c
        cdef Py_ssize_t i
        for i in range(n):
            v = <double>ptr[indices[i]] - m
            sum_sq += v * v
        return sum_sq / (n - 1)

    cdef double _std_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil:
        return sqrt(self._var_rows(indices, n))

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def add_val(self, int val):
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.cols)
        res.length = self.length
        cdef int* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = ptr[i] + val
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sub_val(self, int val):
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.cols)
        res.length = self.length
        cdef int* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = ptr[i] - val
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def mul_val(self, int val):
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.cols)
        res.length = self.length
        cdef int* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = ptr[i] * val
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def div_val(self, int val):
        cdef DoubleColumn res = DoubleColumn(max(self.length, 1), cols=self.cols)
        res.length = self.length
        cdef int* ptr = self.data_c
        cdef double* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef double inv_val
        with nogil:
            if val != 0:
                inv_val = 1.0 / <double>val
                for i in range(n):
                    res_ptr[i] = <double>ptr[i] * inv_val
            else:
                for i in range(n):
                    res_ptr[i] = NAN
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def add(self, object other):
        cdef Py_ssize_t n = self.length
        if other.length != n:
            raise ValueError(f"Column length {other.length} != {n}")
        cdef IntColumn res
        cdef DoubleColumn dbl
        cdef int* self_ptr = self.data_c
        cdef int* other_i_ptr
        cdef double* other_d_ptr
        cdef Py_ssize_t i

        if isinstance(other, IntColumn):
            res = IntColumn(max(n, 1), cols=self.cols)
            res.length = n
            other_i_ptr = (<IntColumn>other).data_c
            with nogil:
                for i in range(n):
                    res.data_c[i] = self_ptr[i] + other_i_ptr[i]
            return res
        elif isinstance(other, DoubleColumn):
            dbl = DoubleColumn(max(n, 1), cols=self.cols)
            dbl.length = n
            other_d_ptr = (<DoubleColumn>other).data_c
            with nogil:
                for i in range(n):
                    dbl.data_c[i] = <double>self_ptr[i] + other_d_ptr[i]
            return dbl
        raise TypeError(f"Unsupported type for add: {type(other)}")

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def sub(self, object other):
        cdef Py_ssize_t n = self.length
        if other.length != n:
            raise ValueError(f"Column length {other.length} != {n}")
        cdef IntColumn res
        cdef DoubleColumn dbl
        cdef int* self_ptr = self.data_c
        cdef int* other_i_ptr
        cdef double* other_d_ptr
        cdef Py_ssize_t i

        if isinstance(other, IntColumn):
            res = IntColumn(max(n, 1), cols=self.cols)
            res.length = n
            other_i_ptr = (<IntColumn>other).data_c
            with nogil:
                for i in range(n):
                    res.data_c[i] = self_ptr[i] - other_i_ptr[i]
            return res
        elif isinstance(other, DoubleColumn):
            dbl = DoubleColumn(max(n, 1), cols=self.cols)
            dbl.length = n
            other_d_ptr = (<DoubleColumn>other).data_c
            with nogil:
                for i in range(n):
                    dbl.data_c[i] = <double>self_ptr[i] - other_d_ptr[i]
            return dbl
        raise TypeError(f"Unsupported type for sub: {type(other)}")

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def mul(self, object other):
        cdef Py_ssize_t n = self.length
        if other.length != n:
            raise ValueError(f"Column length {other.length} != {n}")
        cdef IntColumn res
        cdef DoubleColumn dbl
        cdef int* self_ptr = self.data_c
        cdef int* other_i_ptr
        cdef double* other_d_ptr
        cdef Py_ssize_t i

        if isinstance(other, IntColumn):
            res = IntColumn(max(n, 1), cols=self.cols)
            res.length = n
            other_i_ptr = (<IntColumn>other).data_c
            with nogil:
                for i in range(n):
                    res.data_c[i] = self_ptr[i] * other_i_ptr[i]
            return res
        elif isinstance(other, DoubleColumn):
            dbl = DoubleColumn(max(n, 1), cols=self.cols)
            dbl.length = n
            other_d_ptr = (<DoubleColumn>other).data_c
            with nogil:
                for i in range(n):
                    dbl.data_c[i] = <double>self_ptr[i] * other_d_ptr[i]
            return dbl
        raise TypeError(f"Unsupported type for mul: {type(other)}")

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def div(self, object other):
        cdef Py_ssize_t n = self.length
        if other.length != n:
            raise ValueError(f"Column length {other.length} != {n}")
        cdef DoubleColumn res = DoubleColumn(max(n, 1), cols=self.cols)
        res.length = n
        cdef int* self_ptr = self.data_c
        cdef int* other_i_ptr
        cdef double* other_d_ptr
        cdef double val
        cdef double* res_ptr = res.data_c
        cdef Py_ssize_t i

        if isinstance(other, IntColumn):
            other_i_ptr = (<IntColumn>other).data_c
            with nogil:
                for i in range(n):
                    val = <double>other_i_ptr[i]
                    res_ptr[i] = <double>self_ptr[i] / val if val != 0.0 else NAN
        elif isinstance(other, DoubleColumn):
            other_d_ptr = (<DoubleColumn>other).data_c
            with nogil:
                for i in range(n):
                    val = other_d_ptr[i]
                    res_ptr[i] = <double>self_ptr[i] / val if val != 0.0 else NAN
        else:
            raise TypeError(f"Unsupported type for div: {type(other)}")
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __add__(self, object other):
        if isinstance(other, int): return self.add_val(other)
        if isinstance(other, (IntColumn, DoubleColumn)): return self.add(other)
        return NotImplemented

    def __radd__(self, object other): return self.__add__(other)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __sub__(self, object other):
        if isinstance(other, int): return self.sub_val(other)
        if isinstance(other, (IntColumn, DoubleColumn)): return self.sub(other)
        return NotImplemented

    def __rsub__(self, object other):
        cdef Py_ssize_t n = self.length
        cdef IntColumn res
        cdef int v
        cdef int* s_ptr
        cdef int* r_ptr
        cdef Py_ssize_t i
        if isinstance(other, int):
            v = <int>other
            res = IntColumn(max(n, 1), cols=self.cols)
            res.length = n
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
        if isinstance(other, int): return self.mul_val(other)
        if isinstance(other, (IntColumn, DoubleColumn)): return self.mul(other)
        return NotImplemented

    def __rmul__(self, object other): return self.__mul__(other)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __truediv__(self, object other):
        if isinstance(other, int): return self.div_val(other)
        if isinstance(other, (IntColumn, DoubleColumn)): return self.div(other)
        return NotImplemented

    def __rtruediv__(self, object other):
        cdef Py_ssize_t n = self.length
        cdef DoubleColumn res
        cdef double v
        cdef int* s_ptr
        cdef double* r_ptr
        cdef Py_ssize_t i
        cdef int s_val
        if isinstance(other, int):
            res = DoubleColumn(max(n, 1), cols=self.cols)
            res.length = n
            v = <double>other
            s_ptr = self.data_c
            r_ptr = res.data_c
            with nogil:
                for i in range(n):
                    s_val = s_ptr[i]
                    r_ptr[i] = v / <double>s_val if s_val != 0 else NAN
            return res
        return NotImplemented

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _reorder(self, list indices):
        cdef Py_ssize_t n = len(indices)
        cdef char err_buf[128]
        if n != self.length:
            snprintf(err_buf, sizeof(err_buf), "Length of indices (%zd) must match column length (%zd)", n, self.length)
            raise ValueError(err_buf.decode('ascii'))
        if n == 0: return

        cdef Py_ssize_t new_cap = max(n, 1)
        cdef int* new_data = <int*>aligned_malloc(new_cap * sizeof(int))
        if not new_data: raise MemoryError()
        cdef Py_ssize_t i, idx
        cdef Py_ssize_t* c_indices = <Py_ssize_t*>aligned_malloc(n * sizeof(Py_ssize_t))
        if not c_indices:
            os_aligned_free(new_data)
            raise MemoryError()
        try:
            for i in range(n):
                idx = <Py_ssize_t>indices[i]
                if idx < 0: idx += self.length
                if idx < 0 or idx >= self.length:
                    snprintf(err_buf, sizeof(err_buf), "Index out of bounds: %zd", idx)
                    raise IndexError(err_buf.decode('ascii'))
                c_indices[i] = idx
            _col_gather(new_data, self.data_c, c_indices, n, sizeof(int))
        except Exception:
            os_aligned_free(new_data)
            raise
        finally:
            os_aligned_free(c_indices)

        if self.get_owns_data_c() and self.data_c != NULL:
            os_aligned_free(self.data_c)
        self.data_c = new_data
        self.capacity = new_cap
        self.length = n
        self.set_meta_c(self.get_cols_c(), True, self.get_is_2d_c())
        self._base_obj = None

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __getitem__(self, Py_ssize_t idx):
        if idx < 0: idx += self.length
        if idx < 0 or idx >= self.length: raise IndexError("Index out of range")
        return self.data_c[idx]

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def unique(self):
        cdef set seen = set()
        cdef list result = []
        cdef Py_ssize_t i, n = self.length
        cdef int val
        for i in range(n):
            val = self.data_c[i]
            if val not in seen:
                seen.add(val)
                result.append(val)
        return result

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _cmp_op(self, object val, int op):
        cdef Py_ssize_t i, n = self.length
        cdef IntColumn res = IntColumn(max(n, 1), cols=self.cols)
        res.length = n
        cdef int v
        cdef int* s_ptr = self.data_c
        cdef int* r_ptr = res.data_c
        cdef int* o_ptr
        cdef char err_buf[128]
        cdef IntColumn other

        if isinstance(val, IntColumn):
            other = <IntColumn>val
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
            v = <int>val
            with nogil:
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
            raise TypeError("Bitwise AND only supported between two IntColumn masks")
        cdef IntColumn b = <IntColumn>other
        if self.length != b.length:
            raise ValueError(f"Length mismatch: {self.length} vs {b.length}")
        cdef Py_ssize_t n = self.length
        cdef IntColumn res = IntColumn(max(n, 1), cols=self.cols)
        res.length = n
        cdef int* a_ptr = self.data_c
        cdef int* b_ptr = b.data_c
        cdef int* r_ptr = res.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                r_ptr[i] = (a_ptr[i] != 0) & (b_ptr[i] != 0)
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __or__(self, other):
        if not isinstance(other, IntColumn):
            raise TypeError("Bitwise OR only supported between two IntColumn masks")
        cdef IntColumn b = <IntColumn>other
        if self.length != b.length:
            raise ValueError(f"Length mismatch: {self.length} vs {b.length}")
        cdef Py_ssize_t n = self.length
        cdef IntColumn res = IntColumn(max(n, 1), cols=self.cols)
        res.length = n
        cdef int* a_ptr = self.data_c
        cdef int* b_ptr = b.data_c
        cdef int* r_ptr = res.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                r_ptr[i] = (a_ptr[i] != 0) | (b_ptr[i] != 0)
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __invert__(self):
        cdef Py_ssize_t n = self.length
        cdef IntColumn res = IntColumn(max(n, 1), cols=self.cols)
        res.length = n
        cdef int* s_ptr = self.data_c
        cdef int* r_ptr = res.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                r_ptr[i] = (s_ptr[i] == 0)
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def isin(self, set values):
        cdef Py_ssize_t n = self.length
        cdef IntColumn res = IntColumn(max(n, 1), cols=self.cols)
        res.length = n
        cdef int* s_ptr = self.data_c
        cdef int* r_ptr = res.data_c
        cdef Py_ssize_t m = len(values)

        if m == 0 or n == 0:
            if n > 0:
                memset(r_ptr, 0, n * sizeof(int))
            return res

        cdef int* v_arr = <int*>aligned_malloc(m * sizeof(int))
        if not v_arr: raise MemoryError()
        cdef Py_ssize_t valid_m = 0
        cdef int single_val
        cdef Py_ssize_t i

        try:
            for v in values:
                if v is not None:
                    v_arr[valid_m] = <int>v
                    valid_m += 1

            if valid_m == 0:
                memset(r_ptr, 0, n * sizeof(int))
            elif valid_m == 1:
                single_val = v_arr[0]
                with nogil:
                    for i in range(n):
                        r_ptr[i] = (s_ptr[i] == single_val)
            else:
                qsort(v_arr, valid_m, sizeof(int), _cmp_int)
                with nogil:
                    for i in range(n):
                        r_ptr[i] = 1 if _binary_search(v_arr, valid_m, s_ptr[i]) else 0
        finally:
            os_aligned_free(v_arr)

        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def between(self, int low, int high):
        cdef Py_ssize_t n = self.length
        cdef IntColumn res = IntColumn(max(n, 1), cols=self.cols)
        res.length = n
        cdef int* s_ptr = self.data_c
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
        return f"IntColumn([{', '.join(items)}])"

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __bool__(self):
        return self.length > 0

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __len__(self):
        return self.length

    cdef IntColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=False):
        cdef IntColumn new_col
        cdef int* new_data
        cdef Py_ssize_t c = self.get_cols_c()
        cdef Py_ssize_t r = self.get_rows_c()

        if self.get_is_2d_c():
            if not inplace:
                new_col = IntColumn(max(n * c, 1), cols=c)
                new_col.length = n * c
                _col_gather_2d_zero(new_col.data_c, self.data_c, indices, n, r, c, sizeof(int))
                return new_col
            else:
                new_data = <int*>aligned_malloc(max(n * c, 1) * sizeof(int))
                if not new_data: raise MemoryError()
                _col_gather_2d_zero(new_data, self.data_c, indices, n, r, c, sizeof(int))
                if self.get_owns_data_c() and self.data_c != NULL:
                    os_aligned_free(self.data_c)
                self.data_c = new_data
                self.capacity = n * c
                self.length = n * c
                self.set_meta_c(<int>c, True, True)
                self._base_obj = None
                return self

        if not inplace:
            new_col = IntColumn(max(n, 1))
            new_col.length = n
            _col_gather_zero(new_col.data_c, self.data_c, indices, n, sizeof(int))
            return new_col
        else:
            new_data = <int*>aligned_malloc(max(n, 1) * sizeof(int))
            if not new_data: raise MemoryError()
            _col_gather_zero(new_data, self.data_c, indices, n, sizeof(int))
            if self.get_owns_data_c() and self.data_c != NULL:
                os_aligned_free(self.data_c)
            self.data_c = new_data
            self.capacity = n
            self.length = n
            self.set_meta_c(self.get_cols_c(), True, self.get_is_2d_c())
            self._base_obj = None
            return self

    cpdef IntColumn clip(self, double lower, double upper, bint inplace=False):
        cdef IntColumn col = self if inplace else self.copy()
        cdef int* ptr = col.data_c
        cdef Py_ssize_t i, n = col.length
        cdef bint has_lower = (lower == lower)
        cdef bint has_upper = (upper == upper)
        cdef int l = <int>lower if has_lower else -2147483648
        cdef int u = <int>upper if has_upper else 2147483647
        cdef int val
        with nogil:
            for i in range(n):
                val = ptr[i]
                if val < l: val = l
                elif val > u: val = u
                ptr[i] = val
        return col

    cpdef IntColumn where(self, IntColumn cond, int other, bint inplace=False):
        cdef Py_ssize_t n = self.length
        cdef Py_ssize_t r = self.get_rows_c()
        cdef Py_ssize_t ccols = self.get_cols_c()
        cdef char err_buf[128]
        cdef IntColumn col
        cdef Py_ssize_t i, j, base
        cdef int* col_ptr
        cdef int* self_ptr = self.data_c
        cdef int* cond_ptr = cond.data_c
        cdef int m

        if self.get_is_2d_c() and ccols > 1:
            if cond.length != r:
                snprintf(err_buf, sizeof(err_buf), "Condition mask length %zd must match column rows %zd", cond.length, r)
                raise ValueError(err_buf.decode('ascii'))
            col = self if inplace else IntColumn(max(n, 1), cols=ccols)
            if not inplace: col.length = n
            col_ptr = col.data_c
            with nogil:
                for j in range(ccols):
                    base = j * r
                    for i in range(r):
                        m = -<int>(cond_ptr[i] != 0)
                        col_ptr[base + i] = (self_ptr[base + i] & m) | (other & ~m)
            return col

        if cond.length != n:
            snprintf(err_buf, sizeof(err_buf), "Condition mask length %zd must match column length %zd", cond.length, n)
            raise ValueError(err_buf.decode('ascii'))

        col = self if inplace else IntColumn(max(n, 1), cols=self.cols)
        if not inplace: col.length = n

        col_ptr = col.data_c
        with nogil:
            for i in range(n):
                m = -<int>(cond_ptr[i] != 0)
                col_ptr[i] = (self_ptr[i] & m) | (other & ~m)
        return col

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def where_mask(self, IntColumn mask, object fill):
        cdef Py_ssize_t n = self.length
        cdef Py_ssize_t r = self.get_rows_c()
        cdef Py_ssize_t ccols = self.get_cols_c()
        cdef char err_buf[128]
        cdef IntColumn col
        cdef Py_ssize_t i, j, base
        cdef int* col_ptr
        cdef int* self_ptr = self.data_c
        cdef int* mask_ptr = mask.data_c
        cdef int m
        cdef int fill_val = <int>fill

        if self.get_is_2d_c() and ccols > 1:
            if mask.length != r:
                snprintf(err_buf, sizeof(err_buf), "Mask length %zd must match column rows %zd", mask.length, r)
                raise ValueError(err_buf.decode('ascii'))
            col = IntColumn(max(n, 1), cols=ccols)
            col.length = n
            col_ptr = col.data_c
            with nogil:
                for j in range(ccols):
                    base = j * r
                    for i in range(r):
                        m = -<int>(mask_ptr[i] != 0)
                        col_ptr[base + i] = (self_ptr[base + i] & m) | (fill_val & ~m)
            return col

        if mask.length != n:
            snprintf(err_buf, sizeof(err_buf), "Mask length %zd != column length %zd", mask.length, n)
            raise ValueError(err_buf.decode('ascii'))

        col = IntColumn(max(n, 1), cols=self.cols)
        col.length = n
        col_ptr = col.data_c

        with nogil:
            for i in range(n):
                m = -<int>(mask_ptr[i] != 0)
                col_ptr[i] = (self_ptr[i] & m) | (fill_val & ~m)
        return col

    cpdef IntColumn mask(self, IntColumn cond, int other, bint inplace=False):
        cdef Py_ssize_t n = self.length
        cdef Py_ssize_t r = self.get_rows_c()
        cdef Py_ssize_t ccols = self.get_cols_c()
        cdef char err_buf[128]
        cdef IntColumn col
        cdef Py_ssize_t i, j, base
        cdef int* col_ptr
        cdef int* self_ptr = self.data_c
        cdef int* cond_ptr = cond.data_c
        cdef int m

        if self.get_is_2d_c() and ccols > 1:
            if cond.length != r:
                snprintf(err_buf, sizeof(err_buf), "Condition mask length %zd must match column rows %zd", cond.length, r)
                raise ValueError(err_buf.decode('ascii'))
            col = self if inplace else IntColumn(max(n, 1), cols=ccols)
            if not inplace: col.length = n
            col_ptr = col.data_c
            with nogil:
                for j in range(ccols):
                    base = j * r
                    for i in range(r):
                        m = -<int>(cond_ptr[i] != 0)
                        col_ptr[base + i] = (other & m) | (self_ptr[base + i] & ~m)
            return col

        if cond.length != n:
            snprintf(err_buf, sizeof(err_buf), "Condition mask length %zd must match column length %zd", cond.length, n)
            raise ValueError(err_buf.decode('ascii'))

        col = self if inplace else IntColumn(max(n, 1), cols=self.cols)
        if not inplace: col.length = n

        col_ptr = col.data_c
        with nogil:
            for i in range(n):
                m = -<int>(cond_ptr[i] != 0)
                col_ptr[i] = (other & m) | (self_ptr[i] & ~m)
        return col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def drop_row(self, Py_ssize_t row_idx):
        if not self.get_owns_data_c():
            raise RuntimeError("Cannot drop row on a view/slice column. Copy it first.")
        cdef Py_ssize_t c = self.get_cols_c()
        cdef Py_ssize_t r = self.get_rows_c()
        if row_idx < 0:
            row_idx += r
        if row_idx < 0 or row_idx >= r:
            raise IndexError("Row index out of bounds")
        if self.get_is_2d_c():
            _col_shift_left_2d(self.data_c, &self.length, r, c, row_idx, sizeof(int))
        else:
            _col_drop_row(self.data_c, &self.length, row_idx, sizeof(int))

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def shift_left(self, Py_ssize_t pos):
        if not self.get_owns_data_c():
            raise RuntimeError("Cannot modify a view/slice column. Copy it first.")
        cdef Py_ssize_t c = self.get_cols_c()
        cdef Py_ssize_t r = self.get_rows_c()
        cdef Py_ssize_t col_len
        if pos < 0:
            pos += r
        if pos < 0 or pos >= r:
            raise IndexError("Index out of bounds")
        if self.get_is_2d_c():
            _col_shift_left_2d(self.data_c, &self.length, r, c, pos, sizeof(int))
        else:
            col_len = self.length
            _c_shift_left(self.data_c, pos, col_len, sizeof(int))
            self.length = col_len - 1

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def shift_right(self, Py_ssize_t pos):
        """Insert a blank slot at logical position pos (any sub-column).
        For 1D columns this mirrors the historical flat shift_right."""
        if not self.get_owns_data_c():
            raise RuntimeError("Cannot modify a view/slice column. Copy it first.")
        cdef Py_ssize_t c = self.get_cols_c()
        cdef Py_ssize_t r = self.get_rows_c()
        if pos < 0:
            pos += r
        if pos < 0 or pos > r:
            raise IndexError("Index out of bounds")
        if self.get_is_2d_c():
            if self.length + c > self.capacity:
                self._reserve(max(16, max(self.capacity * 2, self.length + c)))
            _col_shift_right_2d_int(self.data_c, &self.length, r, c, pos, sizeof(int))
        else:
            if self.length >= self.capacity:
                self._reserve(max(16, self.capacity * 2))
            _c_shift_right(self.data_c, pos, self.length, sizeof(int))
            self.length += 1
            (<int*>self.data_c)[pos] = 0

    cdef void to_double_buffer(self, double* out, Py_ssize_t n):
        cdef int* ptr = self.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                out[i] = <double>ptr[i]

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pearson_corr_with(self, other):
        cdef Py_ssize_t n = min(self.length, other.length)
        if isinstance(other, IntColumn):
            return _vec_pearson_corr(self.data_c, (<IntColumn>other).data_c, n)
        elif isinstance(other, DoubleColumn):
            return _int_double_pearson_corr(self.data_c, (<DoubleColumn>other).data_c, n)
        else:
            raise TypeError(f"Unsupported type: {type(other)}")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pearson_cov_with(self, other):
        cdef Py_ssize_t n = min(self.length, other.length)
        if isinstance(other, IntColumn):
            return _vec_pearson_cov(self.data_c, (<IntColumn>other).data_c, n)
        elif isinstance(other, DoubleColumn):
            return _int_double_pearson_cov(self.data_c, (<DoubleColumn>other).data_c, n)
        else:
            raise TypeError(f"Unsupported type: {type(other)}")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def euclidean_dist(self, other):
        cdef Py_ssize_t n = min(self.length, other.length)
        if isinstance(other, IntColumn):
            return _vec_euclidean_dist(self.data_c, (<IntColumn>other).data_c, n)
        elif isinstance(other, DoubleColumn):
            return _int_double_euclidean_dist(self.data_c, (<DoubleColumn>other).data_c, n)
        else:
            raise TypeError(f"Unsupported type: {type(other)}")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def manhattan_dist(self, other):
        cdef Py_ssize_t n = min(self.length, other.length)
        if isinstance(other, IntColumn):
            return _vec_manhattan_dist(self.data_c, (<IntColumn>other).data_c, n)
        elif isinstance(other, DoubleColumn):
            return _int_double_manhattan_dist(self.data_c, (<DoubleColumn>other).data_c, n)
        else:
            raise TypeError(f"Unsupported type: {type(other)}")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cosine_similarity(self, other):
        cdef Py_ssize_t n = min(self.length, other.length)
        if isinstance(other, IntColumn):
            return _vec_cosine_sim(self.data_c, (<IntColumn>other).data_c, n)
        elif isinstance(other, DoubleColumn):
            return _int_double_cosine_sim(self.data_c, (<DoubleColumn>other).data_c, n)
        else:
            raise TypeError(f"Unsupported type: {type(other)}")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cosine_dist(self, other):
        return 1.0 - self.cosine_similarity(other)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def jaccard_sim(self, other):
        cdef Py_ssize_t n = min(self.length, other.length)
        if isinstance(other, IntColumn):
            return _vec_jaccard_sim(self.data_c, (<IntColumn>other).data_c, n)
        elif isinstance(other, DoubleColumn):
            return _int_double_jaccard_sim(self.data_c, (<DoubleColumn>other).data_c, n)
        else:
            raise TypeError(f"Unsupported type: {type(other)}")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def hamming_dist(self, other):
        cdef Py_ssize_t n = min(self.length, other.length)
        if isinstance(other, IntColumn):
            return _vec_hamming_dist(self.data_c, (<IntColumn>other).data_c, n)
        elif isinstance(other, DoubleColumn):
            return _int_double_hamming_dist(self.data_c, (<DoubleColumn>other).data_c, n)
        else:
            raise TypeError(f"Unsupported type: {type(other)}")

    cdef IntColumn gather_with_sentinel(self, Py_ssize_t* indices, Py_ssize_t n, int sentinel):
        cdef Py_ssize_t c = self.get_cols_c()
        cdef Py_ssize_t r = self.get_rows_c()
        cdef Py_ssize_t i, j, idx
        cdef Py_ssize_t dst_base, src_base
        cdef int* res_ptr
        cdef int* src_ptr
        cdef IntColumn new_col

        if self.get_is_2d_c():
            new_col = IntColumn(max(n * c, 1), cols=c)
            new_col.length = n * c
            new_col.set_meta_c(<int>c, True, True)
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

        new_col = IntColumn(max(n, 1), cols=c)
        new_col.length = n
        res_ptr = new_col.data_c
        src_ptr = self.data_c
        cdef Py_ssize_t my_len = self.length

        with nogil:
            for i in range(n):
                idx = indices[i]
                res_ptr[i] = src_ptr[idx] if (idx >= 0 and idx < my_len) else sentinel
        return new_col

    cpdef Py_ssize_t count_nonzero(self):
        return self.count_nonzeros()

    cdef void gather_zero_to(self, Py_ssize_t* out):
        if out == NULL: return
        cdef Py_ssize_t j = 0
        cdef int* ptr = self.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                if ptr[i] == 0:
                    out[j] = i
                    j += 1

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def astype(self, object target):
        cdef int target_type = _resolve_astype_target(target)
        cdef Py_ssize_t i, n = self.length
        cdef Py_ssize_t c = self.cols
        cdef Py_ssize_t first_col_n = self.rows if c > 1 else n
        cdef Py_ssize_t safe_cap = max(n, 1)
        cdef Py_ssize_t first_col_cap = max(first_col_n, 1)
        cdef DoubleColumn dc
        cdef BoolColumn bc
        cdef UCS2Column uc
        cdef AsciiColumn cc
        cdef DateColumn tc
        cdef CategoryColumn catc
        cdef DateTimeColumn dtc
        cdef TimeColumn tmc
        cdef int ival, sz
        cdef char buf[64]
        cdef int* s_ptr = self.data_c
        cdef int32_t* tc_ptr
        cdef int64_t* dt_ptr
        cdef int32_t days
        cdef int32_t* tmc_ptr
        cdef int64_t tmp_dt

        if target_type == TYPE_DOUBLE:
            dc = DoubleColumn(safe_cap, cols=self.cols)
            dc.length = n
            with nogil:
                for i in range(n):
                    dc.data_c[i] = <double>s_ptr[i]
            return dc

        elif target_type == TYPE_BOOL:
            bc = BoolColumn(first_col_cap)
            bc.length = first_col_n
            with nogil:
                for i in range(first_col_n):
                    bc.data_c[i] = (s_ptr[i] != 0)
            return bc

        elif target_type == TYPE_UCS2:
            uc = UCS2Column(first_col_cap)
            uc.length = first_col_n
            for i in range(first_col_n):
                sz = snprintf(buf, sizeof(buf), "%d", s_ptr[i])
                uc._set_item(i, buf[:sz].decode('ascii'))
            return uc

        elif target_type == TYPE_CHAR256:
            cc = AsciiColumn(first_col_cap)
            for i in range(first_col_n):
                cc.append(str(s_ptr[i]))
            return cc

        elif target_type == TYPE_DATE:
            tc = DateColumn(safe_cap, cols=self.cols)
            tc.length = n
            tc_ptr = <int32_t*>tc.data_c
            with nogil:
                for i in range(n):
                    ival = s_ptr[i]
                    if ival > 10000000:
                        tc_ptr[i] = _ymd_to_days(ival // 10000, (ival // 100) % 100, ival % 100)
                    else:
                        tc_ptr[i] = ival
            return tc

        elif target_type == TYPE_CATEGORY:
            catc = CategoryColumn(first_col_cap)
            for i in range(first_col_n):
                sz = snprintf(buf, sizeof(buf), "%d", s_ptr[i])
                catc.append(buf[:sz].decode('ascii'))
            return catc

        elif target_type == TYPE_DATETIME:
            dtc = DateTimeColumn(safe_cap, cols=self.cols)
            dtc.length = n
            dt_ptr = <int64_t*>dtc.data_c
            with nogil:
                for i in range(n):
                    ival = s_ptr[i]
                    if ival > 10000000:
                        days = _ymd_to_days(ival // 10000, (ival // 100) % 100, ival % 100)
                        tmp_dt = <int64_t>days
                        dt_ptr[i] = tmp_dt * <int64_t>86400000000
                    else:
                        tmp_dt = <int64_t>ival
                        dt_ptr[i] = tmp_dt * <int64_t>1000000
            return dtc

        elif target_type == TYPE_TIME:
            tmc = TimeColumn(safe_cap, cols=self.cols)
            tmc.length = n
            tmc_ptr = tmc.data_c
            with nogil:
                for i in range(n):
                    tmc_ptr[i] = s_ptr[i]
            return tmc

        elif target_type == TYPE_INT:
            return self.copy()

        cdef char err_buf[128]
        snprintf(err_buf, sizeof(err_buf), "Failed to cast INT column to internal type id: %d", target_type)
        raise TypeError(err_buf.decode('ascii'))

    def roll_mean(self, Py_ssize_t window):
        if window < 0:
            raise ValueError("window must be >= 0")
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef DoubleColumn res = self._make_like_nan_dbl()
        if window <= 0 or r < window: return res
        cdef double* dst = res.data_c
        cdef int* src = self.data_c
        cdef double inv_w = 1.0 / <double>window
        cdef Py_ssize_t col_idx, offset, i
        cdef double s
        with nogil:
            for col_idx in range(c):
                offset = col_idx * r
                s = 0.0
                for i in range(window):
                    s += <double>src[offset + i]
                dst[offset + window - 1] = s * inv_w
                for i in range(window, r):
                    s += <double>src[offset + i] - <double>src[offset + i - window]
                    dst[offset + i] = s * inv_w
        return res

    def roll_sum(self, Py_ssize_t window):
        if window < 0:
            raise ValueError("window must be >= 0")
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef DoubleColumn res = self._make_like_nan_dbl()
        if window <= 0 or r < window: return res
        cdef double* dst = res.data_c
        cdef int* src = self.data_c
        cdef Py_ssize_t col_idx, offset, i
        cdef double s
        with nogil:
            for col_idx in range(c):
                offset = col_idx * r
                s = 0.0
                for i in range(window):
                    s += <double>src[offset + i]
                dst[offset + window - 1] = s
                for i in range(window, r):
                    s += <double>src[offset + i] - <double>src[offset + i - window]
                    dst[offset + i] = s
        return res

    def roll_min(self, Py_ssize_t window):
        if window <= 0:
            raise ValueError("window must be > 0")
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef IntColumn res = IntColumn(max(r * c, 1), cols=c)
        res.length = r * c
        cdef int* dst = res.data_c
        cdef int* src = self.data_c
        cdef int sentinel = 2147483647
        cdef Py_ssize_t col_idx, offset, i, head, tail

        cdef Py_ssize_t* q = <Py_ssize_t*>aligned_malloc(r * sizeof(Py_ssize_t))
        if not q: raise MemoryError()
        try:
            with nogil:
                for col_idx in range(c):
                    offset = col_idx * r
                    head = 0; tail = 0
                    for i in range(r):
                        while head < tail and q[head] <= i - window:
                            head += 1
                        while head < tail and src[offset + q[tail - 1]] >= src[offset + i]:
                            tail -= 1
                        q[tail] = i
                        tail += 1
                        if i >= window - 1:
                            dst[offset + i] = src[offset + q[head]]
                        else:
                            dst[offset + i] = sentinel
        finally:
            os_aligned_free(q)
        return res

    def roll_max(self, Py_ssize_t window):
        if window <= 0:
            raise ValueError("window must be > 0")
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef IntColumn res = IntColumn(max(r * c, 1), cols=c)
        res.length = r * c
        cdef int* dst = res.data_c
        cdef int* src = self.data_c
        cdef int sentinel = -2147483647 - 1
        cdef Py_ssize_t col_idx, offset, i, head, tail

        cdef Py_ssize_t* q = <Py_ssize_t*>aligned_malloc(r * sizeof(Py_ssize_t))
        if not q: raise MemoryError()
        try:
            with nogil:
                for col_idx in range(c):
                    offset = col_idx * r
                    head = 0; tail = 0
                    for i in range(r):
                        while head < tail and q[head] <= i - window:
                            head += 1
                        while head < tail and src[offset + q[tail - 1]] <= src[offset + i]:
                            tail -= 1
                        q[tail] = i
                        tail += 1
                        if i >= window - 1:
                            dst[offset + i] = src[offset + q[head]]
                        else:
                            dst[offset + i] = sentinel
        finally:
            os_aligned_free(q)
        return res

    def roll_var(self, Py_ssize_t window, bint ddof=True):
        if window <= 0:
            raise ValueError("window must be > 0")
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef DoubleColumn res = self._make_like_nan_dbl()
        if r < window: return res
        cdef double* dst = res.data_c
        cdef int* src = self.data_c
        cdef double w = <double>window
        cdef Py_ssize_t col_idx, offset, i
        cdef double s, s2, val, val_old, var_val
        with nogil:
            for col_idx in range(c):
                offset = col_idx * r
                s = 0.0; s2 = 0.0
                for i in range(window):
                    val = <double>src[offset + i]
                    s += val; s2 += val * val

                var_val = s2 - (s * s) / w
                if var_val < 0.0: var_val = 0.0

                if ddof:
                    dst[offset + window - 1] = var_val / (w - 1.0)
                else:
                    dst[offset + window - 1] = var_val / w

                for i in range(window, r):
                    val = <double>src[offset + i]
                    val_old = <double>src[offset + i - window]
                    s += val - val_old
                    s2 += val * val - val_old * val_old

                    var_val = s2 - (s * s) / w
                    if var_val < 0.0: var_val = 0.0

                    if ddof:
                        dst[offset + i] = var_val / (w - 1.0)
                    else:
                        dst[offset + i] = var_val / w
        return res

    def roll_std(self, Py_ssize_t window, bint ddof=True):
        if window <= 0:
            raise ValueError("window must be > 0")
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef DoubleColumn var_col = self.roll_var(window, ddof)
        cdef double* v = var_col.data_c
        cdef Py_ssize_t col_idx, offset, i
        with nogil:
            for col_idx in range(c):
                offset = col_idx * r
                for i in range(window - 1, r):
                    if v[offset + i] == v[offset + i]:  # Not NaN check
                        v[offset + i] = sqrt(v[offset + i])
        return var_col

    def roll_median(self, Py_ssize_t window):
        if window <= 0:
            raise ValueError("window must be > 0")
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef DoubleColumn res = self._make_like_nan_dbl()
        if r < window: return res
        cdef double* dst = res.data_c
        cdef int* src = self.data_c
        cdef double med
        cdef Py_ssize_t col_idx, offset, i, j
        cdef int* buf = <int*>aligned_malloc(window * sizeof(int))
        if not buf: raise MemoryError()
        try:
            with nogil:
                for col_idx in range(c):
                    offset = col_idx * r
                    for i in range(window - 1, r):
                        for j in range(window):
                            buf[j] = src[offset + i - window + 1 + j]
                        qsort(buf, window, sizeof(int), _cmp_int)

                        if window & 1:
                            med = <double>buf[window // 2]
                        else:
                            med = (<double>buf[window // 2 - 1] + <double>buf[window // 2]) / 2.0
                        dst[offset + i] = med
        finally:
            os_aligned_free(buf)
        return res

    def roll_quantile(self, Py_ssize_t window, double quantile):
        if window <= 0:
            raise ValueError("window must be > 0")
        if quantile < 0.0 or quantile > 1.0:
            raise ValueError("quantile must be between 0 and 1")
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef DoubleColumn res = self._make_like_nan_dbl()
        if r < window: return res
        cdef double* dst = res.data_c
        cdef int* src = self.data_c
        cdef Py_ssize_t col_idx, offset, i, j, i_low
        cdef double pos = quantile * <double>(window - 1)
        cdef double frac, val_low, val_high
        i_low = <Py_ssize_t>pos
        frac = pos - <double>i_low
        cdef int* buf = <int*>aligned_malloc(window * sizeof(int))
        if not buf: raise MemoryError()
        try:
            with nogil:
                for col_idx in range(c):
                    offset = col_idx * r
                    for i in range(window - 1, r):
                        for j in range(window):
                            buf[j] = src[offset + i - window + 1 + j]
                        qsort(buf, window, sizeof(int), _cmp_int)

                        if i_low >= window - 1:
                            dst[offset + i] = <double>buf[window - 1]
                        else:
                            val_low = <double>buf[i_low]
                            val_high = <double>buf[i_low + 1]
                            dst[offset + i] = val_low + frac * (val_high - val_low)
        finally:
            os_aligned_free(buf)
        return res

    def roll_apply_weights(self, Py_ssize_t window, list weights):
        if window <= 0:
            raise ValueError("window must be > 0")
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef Py_ssize_t wlen = len(weights)
        if wlen != window:
            raise ValueError(f"Length of weights ({wlen}) must match window size ({window})")

        cdef DoubleColumn res = self._make_like_nan_dbl()
        if r < window: return res
        cdef double* dst = res.data_c
        cdef int* src = self.data_c
        cdef Py_ssize_t col_idx, offset, i, j
        cdef double s
        cdef double* wbuf = <double*>aligned_malloc(wlen * sizeof(double))
        if not wbuf: raise MemoryError()

        for i in range(wlen):
            wbuf[i] = <double>weights[i]

        try:
            with nogil:
                for col_idx in range(c):
                    offset = col_idx * r
                    for i in range(window - 1, r):
                        s = 0.0
                        for j in range(window):
                            s += <double>src[offset + i - window + 1 + j] * wbuf[j]
                        dst[offset + i] = s
        finally:
            os_aligned_free(wbuf)
        return res

    def roll_custom(self, Py_ssize_t window, object func):
        if window <= 0:
            raise ValueError("window must be > 0")
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef DoubleColumn res = self._make_like_nan_dbl()
        if r < window: return res
        cdef double* dst = res.data_c
        cdef int* src = self.data_c
        cdef Py_ssize_t col_idx, offset, i, j
        cdef list tmp
        for col_idx in range(c):
            offset = col_idx * r
            for i in range(window - 1, r):
                tmp = []
                for j in range(window):
                    tmp.append(src[offset + i - window + 1 + j])
                dst[offset + i] = <double>func(tmp)
        return res

    def roll_skew(self, Py_ssize_t window):
        if window <= 0:
            raise ValueError("window must be > 0")
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef DoubleColumn res = self._make_like_nan_dbl()
        if r < window or window < 3: return res
        cdef double* dst = res.data_c
        cdef int* src = self.data_c
        cdef double n_f = <double>window
        cdef Py_ssize_t col_idx, offset, i, j
        cdef double s, mean, m2, m3, val, std_dev
        with nogil:
            for col_idx in range(c):
                offset = col_idx * r
                for i in range(window - 1, r):
                    s = 0.0
                    for j in range(i - window + 1, i + 1):
                        s += <double>src[offset + j]
                    mean = s / n_f
                    m2 = 0.0; m3 = 0.0
                    for j in range(i - window + 1, i + 1):
                        val = <double>src[offset + j] - mean
                        m2 += val * val
                        m3 += val * val * val
                    if m2 > 1e-14:
                        std_dev = sqrt(m2 / n_f)
                        dst[offset + i] = (m3 / n_f) / (std_dev * std_dev * std_dev)
                    else:
                        dst[offset + i] = 0.0
        return res

    def roll_kurt(self, Py_ssize_t window):
        if window <= 0:
            raise ValueError("window must be > 0")
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef DoubleColumn res = self._make_like_nan_dbl()
        if r < window or window < 4: return res
        cdef double* dst = res.data_c
        cdef int* src = self.data_c
        cdef double n_f = <double>window
        cdef Py_ssize_t col_idx, offset, i, j
        cdef double s, mean, m2, m4, val, variance
        with nogil:
            for col_idx in range(c):
                offset = col_idx * r
                for i in range(window - 1, r):
                    s = 0.0
                    for j in range(i - window + 1, i + 1):
                        s += <double>src[offset + j]
                    mean = s / n_f
                    m2 = 0.0; m4 = 0.0
                    for j in range(i - window + 1, i + 1):
                        val = <double>src[offset + j] - mean
                        m2 += val * val
                        m4 += val * val * val * val
                    if m2 > 1e-14:
                        variance = m2 / n_f
                        dst[offset + i] = (m4 / n_f) / (variance * variance) - 3.0
                    else:
                        dst[offset + i] = 0.0
        return res

    def roll_corr(self, Py_ssize_t window, object other):
        if window <= 0:
            raise ValueError("window must be > 0")
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef DoubleColumn res = self._make_like_nan_dbl()
        if r < window: return res

        if other.length < self.length:
            raise ValueError("other column must be at least as long as self")

        cdef bint other_is_double = isinstance(other, DoubleColumn)
        cdef bint other_is_int = isinstance(other, IntColumn)
        if not (other_is_double or other_is_int):
            raise TypeError("other must be DoubleColumn or IntColumn")

        cdef double* other_d_ptr = NULL
        cdef int* other_i_ptr = NULL
        if other_is_double:
            other_d_ptr = (<DoubleColumn>other).data_c
        else:
            other_i_ptr = (<IntColumn>other).data_c

        cdef double* dst = res.data_c
        cdef int* src_x = self.data_c
        cdef double n_f = <double>window, inv_w = 1.0 / n_f
        cdef Py_ssize_t col_idx, offset, i
        cdef double sum_x, sum_y, sum_xx, sum_yy, sum_xy, x, y, old_x, old_y, vx, vy, denom, cov_xy

        with nogil:
            for col_idx in range(c):
                offset = col_idx * r
                sum_x = 0.0; sum_y = 0.0; sum_xx = 0.0; sum_yy = 0.0; sum_xy = 0.0
                for i in range(window):
                    x = <double>src_x[offset + i]
                    if other_is_double:
                        y = other_d_ptr[offset + i]
                    else:
                        y = <double>(other_i_ptr[offset + i])

                    sum_x += x; sum_y += y; sum_xx += x * x; sum_yy += y * y; sum_xy += x * y

                vx = fmax(sum_xx - sum_x * sum_x * inv_w, 0.0)
                vy = fmax(sum_yy - sum_y * sum_y * inv_w, 0.0)
                denom = sqrt(vx * vy)
                cov_xy = sum_xy - sum_x * sum_y * inv_w
                dst[offset + window - 1] = cov_xy / denom if (denom > 0.0) else 0.0

                for i in range(window, r):
                    x = <double>src_x[offset + i]
                    old_x = <double>src_x[offset + i - window]

                    if other_is_double:
                        y = other_d_ptr[offset + i]
                        old_y = other_d_ptr[offset + i - window]
                    else:
                        y = <double>(other_i_ptr[offset + i])
                        old_y = <double>(other_i_ptr[offset + i - window])

                    sum_x += x - old_x; sum_y += y - old_y
                    sum_xx += x * x - old_x * old_x; sum_yy += y * y - old_y * old_y
                    sum_xy += x * y - old_x * old_y

                    vx = fmax(sum_xx - sum_x * sum_x * inv_w, 0.0)
                    vy = fmax(sum_yy - sum_y * sum_y * inv_w, 0.0)
                    denom = sqrt(vx * vy)
                    cov_xy = sum_xy - sum_x * sum_y * inv_w
                    dst[offset + i] = cov_xy / denom if (denom > 0.0) else 0.0
        return res

    def roll_cov(self, Py_ssize_t window, object other):
        if window <= 0:
            raise ValueError("window must be > 0")
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef DoubleColumn res = self._make_like_nan_dbl()
        if r < window: return res

        if other.length < self.length:
            raise ValueError("other column must be at least as long as self")

        cdef bint other_is_double = isinstance(other, DoubleColumn)
        cdef bint other_is_int = isinstance(other, IntColumn)
        if not (other_is_double or other_is_int):
            raise TypeError("other must be DoubleColumn or IntColumn")

        cdef double* other_d_ptr = NULL
        cdef int* other_i_ptr = NULL
        if other_is_double:
            other_d_ptr = (<DoubleColumn>other).data_c
        else:
            other_i_ptr = (<IntColumn>other).data_c

        cdef double* dst = res.data_c
        cdef int* src_x = self.data_c
        cdef double n_f = <double>window, inv_w = 1.0 / n_f
        cdef Py_ssize_t col_idx, offset, i
        cdef double sum_x, sum_y, sum_xy, x, y, old_x, old_y

        with nogil:
            for col_idx in range(c):
                offset = col_idx * r
                sum_x = 0.0; sum_y = 0.0; sum_xy = 0.0
                for i in range(window):
                    x = <double>src_x[offset + i]
                    if other_is_double:
                        y = other_d_ptr[offset + i]
                    else:
                        y = <double>(other_i_ptr[offset + i])

                    sum_x += x; sum_y += y; sum_xy += x * y
                dst[offset + window - 1] = (sum_xy - sum_x * sum_y * inv_w) * inv_w

                for i in range(window, r):
                    x = <double>src_x[offset + i]
                    old_x = <double>src_x[offset + i - window]

                    if other_is_double:
                        y = other_d_ptr[offset + i]
                        old_y = other_d_ptr[offset + i - window]
                    else:
                        y = <double>(other_i_ptr[offset + i])
                        old_y = <double>(other_i_ptr[offset + i - window])

                    sum_x += x - old_x; sum_y += y - old_y
                    sum_xy += x * y - old_x * old_y
                    dst[offset + i] = (sum_xy - sum_x * sum_y * inv_w) * inv_w
        return res

    def roll(self, Py_ssize_t window, str method="mean", bint ddof=True, double quantile=0.5, list weights=None, object func=None):
        if method == "mean":
            return self.roll_mean(window)
        elif method == "sum":
            return self.roll_sum(window)
        elif method == "min":
            return self.roll_min(window)
        elif method == "max":
            return self.roll_max(window)
        elif method == "var":
            return self.roll_var(window, ddof)
        elif method == "std":
            return self.roll_std(window, ddof)
        elif method == "median":
            return self.roll_median(window)
        elif method == "quantile":
            return self.roll_quantile(window, quantile)
        elif method == "skew":
            return self.roll_skew(window)
        elif method == "kurt":
            return self.roll_kurt(window)
        elif method == "apply_weights":
            if weights is None:
                raise ValueError("weights must be provided for apply_weights method")
            return self.roll_apply_weights(window, weights)
        elif method == "custom":
            if func is None:
                raise ValueError("func must be provided for custom method")
            return self.roll_custom(window, func)
        else:
            raise ValueError(f"Unknown roll method: {method}")