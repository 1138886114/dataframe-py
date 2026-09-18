# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False

cimport cython
from libc.stdint cimport int8_t, int32_t, int64_t, uint32_t, uint64_t
from libc.string cimport memcpy, memmove, memset, strstr, strlen, strncmp, strcmp, memcmp
from libc.math cimport sqrt, fabs, NAN, fmax, isnan
from libc.stdio cimport snprintf
from libc.stdlib cimport malloc, free, qsort, bsearch
from cpython.unicode cimport (
    PyUnicode_AsUTF8AndSize, PyUnicode_1BYTE_KIND, PyUnicode_2BYTE_KIND,
    PyUnicode_4BYTE_KIND, PyUnicode_FromKindAndData, PyUnicode_GET_LENGTH,
    PyUnicode_READ_CHAR, PyUnicode_KIND, PyUnicode_DATA, Py_UCS1, Py_UCS2,
)
import re
from cpython.number cimport PyNumber_Index

include "_columns_helper.pxi"


# ============================================================
#  Base column class
# ============================================================
cdef class BaseColumn:
    def __cinit__(self):
        self.length = 0
        self.capacity = 0
        self.dtype = -1
        self.is_2d = False
        self.num_cols = 1
        self.owns_data = True

    cdef Py_ssize_t _get_rows_c(self) noexcept nogil:
        """底层无锁获取逻辑行数"""
        if self.is_2d and self.num_cols > 1:
            return self.length // self.num_cols
        return self.length

    cdef void* _get_data_ptr(self) noexcept nogil:
        """
        供子类重写：返回底层 C 数组裸指针。
        """
        return NULL

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def get_size(self):
        return 0

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __contains__(self, object val):
        """元素级成员检测：`val in col`（对标 pandas Series.__contains__）。"""
        cdef Py_ssize_t i
        cdef object getitem = self.__getitem__
        for i in range(self.length):
            if getitem(i) == val:
                return True
        return False

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def copy(self):
        raise NotImplementedError("Subclasses must implement copy()")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def drop_row(self, Py_ssize_t row_idx):
        raise NotImplementedError

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def shift_left(self, Py_ssize_t pos):
        raise NotImplementedError

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def abs(self):
        raise NotImplementedError("Subclasses must implement abs()")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def round(self, int decimals=0):
        raise NotImplementedError("Subclasses must implement round()")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def ffill(self):
        raise NotImplementedError("Subclasses must implement ffill()")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def bfill(self):
        raise NotImplementedError("Subclasses must implement bfill()")


cdef double* _weights_to_buf(object weights, Py_ssize_t win_n) except NULL:
    """把 Python 权重列表拷入 C double 缓冲（调用方负责 os_aligned_free）。"""
    if len(weights) != win_n:
        raise ValueError(f"weights length {len(weights)} != window {win_n}")
    cdef double* buf = <double*>aligned_malloc(win_n * sizeof(double))
    if buf == NULL:
        raise MemoryError()
    cdef Py_ssize_t i
    try:
        for i in range(win_n):
            buf[i] = <double>weights[i]
    except Exception:
        os_aligned_free(buf)
        raise
    return buf


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
    @cython.binding(False)
    @cython.always_allow_keywords(False)
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
    @cython.binding(False)
    @cython.always_allow_keywords(False)
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
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _load_from_1d_array(object arr):
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
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _load_from_2d_array(object arr, list col_names):
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
    def ffill(self):
        """向前填充缺失值。IntColumn 无 NaN 缺失表示（isna 恒为 False），
        语义与 pandas 中不含缺失的整数列一致：各子列内容保持不变（返回拷贝）。"""
        return self.copy()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def bfill(self):
        """向后填充缺失值。语义同 ffill()：整数列无缺失哨兵，返回拷贝。"""
        return self.copy()

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
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()

        if r == 0:
            if not self.is_2d or c <= 1: return NAN
            return [NAN] * c

        cdef int* ptr = self.data_c

        if not self.is_2d or c <= 1:
            return self.sum() / <double>r

        cdef list results = [None] * c
        cdef Py_ssize_t j
        for j in range(c):
            results[j] = _c_sum(ptr + j * r, r) / <double>r
        return results

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def var(self):
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()

        if r < 2:
            if not self.is_2d or c <= 1: return 0.0
            return [0.0] * c

        cdef int* ptr = self.data_c

        if not self.is_2d or c <= 1:
            return _n_accum_var(ptr, r, <double>self.mean())

        cdef list results = [None] * c
        cdef Py_ssize_t j
        cdef double m
        for j in range(c):
            m = _c_sum(ptr + j * r, r) / <double>r
            results[j] = _n_accum_var(ptr + j * r, r, m)
        return results

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def std(self):
        cdef object v = self.var()
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t j
        if not self.is_2d or c <= 1:
            return sqrt(<double>v)

        cdef list results = [None] * c
        for j in range(c):
            results[j] = sqrt(<double>v[j])
        return results

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _count_nonzeros(self):
        return _col_count_nonzeros(self.data_c, self.length)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def idxmax(self):
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()

        if r == 0:
            raise ValueError("empty column")

        cdef int* ptr = self.data_c
        cdef Py_ssize_t i, j, base
        cdef Py_ssize_t max_idx
        cdef int max_val

        if not self.is_2d or c <= 1:
            max_idx = 0
            max_val = ptr[0]
            with nogil:
                for i in range(1, r):
                    if ptr[i] > max_val:
                        max_val = ptr[i]
                        max_idx = i
            return max_idx

        cdef list results = [None] * c
        for j in range(c):
            base = j * r
            max_idx = 0
            max_val = ptr[base]
            with nogil:
                for i in range(1, r):
                    if ptr[base + i] > max_val:
                        max_val = ptr[base + i]
                        max_idx = i
            results[j] = max_idx
        return results

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def idxmin(self):
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()

        if r == 0:
            raise ValueError("empty column")

        cdef int* ptr = self.data_c
        cdef Py_ssize_t i, j, base
        cdef Py_ssize_t min_idx
        cdef int min_val

        if not self.is_2d or c <= 1:
            min_idx = 0
            min_val = ptr[0]
            with nogil:
                for i in range(1, r):
                    if ptr[i] < min_val:
                        min_val = ptr[i]
                        min_idx = i
            return min_idx

        cdef list results = [None] * c
        for j in range(c):
            base = j * r
            min_idx = 0
            min_val = ptr[base]
            with nogil:
                for i in range(1, r):
                    if ptr[base + i] < min_val:
                        min_val = ptr[base + i]
                        min_idx = i
            results[j] = min_idx
        return results

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _argmax(self): return self.idxmax()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _argmin(self): return self.idxmin()

    @cython.binding(False)
    @cython.always_allow_keywords(False)


    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def median(self):
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()

        if r == 0:
            if not self.is_2d or c <= 1: return NAN
            return [NAN] * c

        if not self.is_2d or c <= 1:
            return _n_median(self.data_c, r)

        cdef list results = [None] * c
        cdef Py_ssize_t j
        for j in range(c):
            results[j] = _n_median(self.data_c + j * r, r)
        return results

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def mad(self):
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c()
        if self.length == 0: return NAN
        cdef int* ptr = self.data_c
        cdef Py_ssize_t j
        cdef list result
        if self.is_2d and c > 1:
            result = []
            for j in range(c):
                result.append(_c_mad(ptr + j * r, r, _c_sum(ptr + j * r, r) / <double>r))
            return result
        return _n_accum_mad(ptr, self.length, <double>self.mean())

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def prod(self):
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()

        if r == 0:
            if not self.is_2d or c <= 1: return NAN
            return [NAN] * c

        cdef int* ptr = self.data_c
        cdef Py_ssize_t i, j, base
        cdef double total

        if not self.is_2d or c <= 1:
            total = 1.0
            with nogil:
                for i in range(r):
                    total *= <double>ptr[i]
            return total

        cdef list results = [None] * c
        with nogil:
            for j in range(c):
                base = j * r
                total = 1.0
                for i in range(r):
                    total *= <double>ptr[base + i]
                with gil:
                    results[j] = total
        return results

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _nunique(self):
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()
        if r == 0:
            if not self.is_2d or c <= 1: return 0
            return [0] * c
        cdef int* ptr = self.data_c
        cdef Py_ssize_t i, j, base
        cdef set seen
        cdef int val
        cdef list results
        if self.is_2d and c > 1:
            results = [None] * c
            for j in range(c):
                base = j * r
                seen = set()
                for i in range(r):
                    seen.add(ptr[base + i])
                results[j] = len(seen)
            return results
        seen = set()
        for i in range(r):
            seen.add(ptr[i])
        return len(seen)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _value_counts(self):
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()

        if r == 0:
            if not self.is_2d or c <= 1: return {}
            return [{} for _ in range(c)]

        cdef int* ptr = self.data_c
        cdef Py_ssize_t i, j, base
        cdef dict counts
        cdef int val

        if not self.is_2d or c <= 1:
            counts = {}
            for i in range(r):
                val = ptr[i]
                counts[val] = counts.get(val, 0) + 1
            return counts

        cdef list results = [None] * c
        for j in range(c):
            base = j * r
            counts = {}
            for i in range(r):
                val = ptr[base + i]
                counts[val] = counts.get(val, 0) + 1
            results[j] = counts
        return results

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def mode(self):
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()

        if r == 0:
            if not self.is_2d or c <= 1: return []
            return [[] for _ in range(c)]

        cdef Py_ssize_t i, j, base
        cdef int val, count, max_count
        cdef int* ptr = self.data_c
        cdef dict counts

        if not self.is_2d or c <= 1:
            counts = {}
            max_count = 0
            for i in range(r):
                val = ptr[i]
                count = counts.get(val, 0) + 1
                counts[val] = count
                if count > max_count: max_count = count
            return [k for k, v in counts.items() if v == max_count]

        cdef list results = [None] * c
        for j in range(c):
            base = j * r
            counts = {}
            max_count = 0
            for i in range(r):
                val = ptr[base + i]
                count = counts.get(val, 0) + 1
                counts[val] = count
                if count > max_count: max_count = count
            results[j] = [k for k, v in counts.items() if v == max_count]
        return results

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def quantile(self, double q):
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()
        if r == 0:
            if not self.is_2d or c <= 1: return NAN
            return [NAN] * c

        if q < 0.0 or q > 1.0:
            raise ValueError("q must be between 0 and 1")

        if not self.is_2d or c <= 1:
            return _n_quantile(self.data_c, r, q)

        cdef list results = [None] * c
        cdef Py_ssize_t j
        for j in range(c):
            results[j] = _n_quantile(self.data_c + j * r, r, q)
        return results

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cumsum(self):
        cdef IntColumn res = IntColumn(max(self.capacity, 1), cols=self.cols)
        res.length = self.length
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef int* ptr = self.data_c
        _n_cumsum(ptr, res.data_c, c, r)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cummax(self):
        cdef IntColumn res = IntColumn(max(self.capacity, 1), cols=self.cols)
        res.length = self.length
        if self.length == 0: return res
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef int* ptr = self.data_c
        _n_cummax(ptr, res.data_c, c, r)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cummin(self):
        cdef IntColumn res = IntColumn(max(self.capacity, 1), cols=self.cols)
        res.length = self.length
        if self.length == 0: return res
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef int* ptr = self.data_c
        _n_cummin(ptr, res.data_c, c, r)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def diff(self, int periods=1):
        cdef IntColumn res = IntColumn(max(self.capacity, 1), cols=self.cols)
        res.length = self.length
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef int* ptr = self.data_c
        _n_diff(ptr, res.data_c, c, r, periods)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _pct_change(self, int periods=1):
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

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def skew(self):
        cdef double m, m2, m3, m4
        cdef double n, var_sample
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c()
        cdef int* ptr = self.data_c
        cdef Py_ssize_t j, i
        cdef int* sub_ptr
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
                    m3 += (<double>sub_ptr[i] - m) ** 3
                m3 = m3 / n
                result.append((n * m3) / ((n - 1) * (n - 2) * (var_sample * sqrt(var_sample))))
            return result
        if self.length < 3: return NAN
        _n_central_moments(self.data_c, self.length, &m, &m2, &m3, &m4)
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
        cdef int* ptr = self.data_c
        cdef Py_ssize_t j, i
        cdef int* sub_ptr
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
                    m4 += (<double>sub_ptr[i] - m) ** 4
                m4 = m4 / n
                result.append((m4 / (n - 1)) / (var_sample * var_sample) - 3.0)
            return result
        if self.length < 4: return NAN
        _n_central_moments(self.data_c, self.length, &m, &m2, &m3, &m4)
        if m2 <= 1e-14: return NAN
        n = <double>self.length
        var_sample = m2 / (n - 1)
        return (m4 / (n - 1)) / (var_sample * var_sample) - 3.0

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sem(self):
        cdef double m, m2, m3, m4
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c()
        cdef int* ptr = self.data_c
        cdef Py_ssize_t j
        cdef int* sub_ptr
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
        _n_central_moments(self.data_c, self.length, &m, &m2, &m3, &m4)
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
        cdef IntColumn res = IntColumn(max(self.capacity, 1), cols=self.cols)
        res.length = self.length
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef int* ptr = self.data_c
        _n_cumprod(ptr, res.data_c, c, r)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _describe(self):
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()

        if r == 0:
            if not self.is_2d or c <= 1:
                return {"count": 0, "mean": NAN, "std": NAN, "min": NAN,
                        "25%": NAN, "50%": NAN, "75%": NAN, "max": NAN}
            return {"count": 0, "mean": NAN, "std": NAN, "min": NAN,
                    "25%": NAN, "50%": NAN, "75%": NAN, "max": NAN}

        if not self.is_2d or c <= 1:
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

        cdef object means = self.mean()
        cdef object mins = self.min()
        cdef object maxs = self.max()
        cdef object q25 = self.quantile(0.25)
        cdef object q50 = self.quantile(0.50)
        cdef object q75 = self.quantile(0.75)

        cdef double total_mean = 0.0, global_min = <double>mins[0], global_max = <double>maxs[0]
        cdef double global_q25 = q25[0], global_q50 = q50[0], global_q75 = q75[0]
        cdef double combined_sum = 0.0, combined_sq_sum = 0.0, combined_mean, combined_var
        cdef Py_ssize_t total_count = 0, j, k
        cdef int* ptr = self.data_c
        for j in range(c):
            total_count += r
            total_mean += means[j]
            combined_sum += means[j] * <double>r
            for k in range(r):
                combined_sq_sum += <double>ptr[j * r + k] * <double>ptr[j * r + k]
            if <double>mins[j] < global_min: global_min = <double>mins[j]
            if <double>maxs[j] > global_max: global_max = <double>maxs[j]
            if q25[j] < global_q25: global_q25 = q25[j]
            if q50[j] < global_q50: global_q50 = q50[j]
            if q75[j] < global_q75: global_q75 = q75[j]
        combined_mean = combined_sum / <double>total_count
        combined_var = combined_sq_sum / <double>total_count - combined_mean * combined_mean
        if combined_var < 0.0: combined_var = 0.0
        return {
            "count": total_count,
            "mean": total_mean / <double>c,
            "std": sqrt(combined_var),
            "min": global_min,
            "25%": global_q25,
            "50%": global_q50,
            "75%": global_q75,
            "max": global_max
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
        _n_add_val(ptr, res.data_c, self.length, val)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sub_val(self, int val):
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.cols)
        res.length = self.length
        cdef int* ptr = self.data_c
        _n_sub_val(ptr, res.data_c, self.length, val)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def mul_val(self, int val):
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.cols)
        res.length = self.length
        cdef int* ptr = self.data_c
        _n_mul_val(ptr, res.data_c, self.length, val)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def div_val(self, int val):
        cdef DoubleColumn res = DoubleColumn(max(self.length, 1), cols=self.cols)
        res.length = self.length
        cdef int* ptr = self.data_c
        _n_div_val(ptr, res.data_c, self.length, <double>val)
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def add(self, object other):
        cdef Py_ssize_t n = self.length
        if other.length != n:
            raise ValueError(f"Column length {other.length} != {n}")
        if other.num_cols != self.num_cols:
            raise ValueError(
                f"Shape mismatch: self has {self.num_cols} column(s), other has {other.num_cols}")
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
        if other.num_cols != self.num_cols:
            raise ValueError(
                f"Shape mismatch: self has {self.num_cols} column(s), other has {other.num_cols}")
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
        if other.num_cols != self.num_cols:
            raise ValueError(
                f"Shape mismatch: self has {self.num_cols} column(s), other has {other.num_cols}")
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
        if other.num_cols != self.num_cols:
            raise ValueError(
                f"Shape mismatch: self has {self.num_cols} column(s), other has {other.num_cols}")
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

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __radd__(self, object other): return self.__add__(other)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __sub__(self, object other):
        if isinstance(other, int): return self.sub_val(other)
        if isinstance(other, (IntColumn, DoubleColumn)): return self.sub(other)
        return NotImplemented

    @cython.always_allow_keywords(False)
    @cython.binding(False)
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

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __rmul__(self, object other): return self.__mul__(other)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __truediv__(self, object other):
        if isinstance(other, int): return self.div_val(other)
        if isinstance(other, (IntColumn, DoubleColumn)): return self.div(other)
        return NotImplemented

    @cython.always_allow_keywords(False)
    @cython.binding(False)
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
    def __getitem__(self, Py_ssize_t idx):
        if idx < 0: idx += self.length
        if idx < 0 or idx >= self.length: raise IndexError("Index out of range")
        return self.data_c[idx]

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def unique(self):
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()

        if r == 0:
            if not self.is_2d or c <= 1: return []
            return [[] for _ in range(c)]

        cdef int* ptr = self.data_c
        cdef Py_ssize_t i, j, base
        cdef set seen
        cdef list result
        cdef int val

        if not self.is_2d or c <= 1:
            seen = set()
            result = []
            for i in range(r):
                val = ptr[i]
                if val not in seen:
                    seen.add(val)
                    result.append(val)
            return result

        cdef list results = [None] * c
        for j in range(c):
            base = j * r
            seen = set()
            result = []
            for i in range(r):
                val = ptr[base + i]
                if val not in seen:
                    seen.add(val)
                    result.append(val)
            results[j] = result
        return results

    cdef IntColumn _cmp_op(self, object val, int op):
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
            if other.num_cols != self.num_cols:
                raise ValueError(
                    f"Shape mismatch: self has {self.num_cols} column(s), other has {other.num_cols}")
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
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()

        if self.is_2d:
            if not inplace:
                new_col = IntColumn(max(n * c, 1), cols=c)
                new_col.length = n * c
                new_col.is_2d = self.is_2d
                _col_gather_2d_zero(new_col.data_c, self.data_c, indices, n, r, c, sizeof(int))
                return new_col
            else:
                new_data = <int*>aligned_malloc(max(n * c, 1) * sizeof(int))
                if not new_data: raise MemoryError()
                _col_gather_2d_zero(new_data, self.data_c, indices, n, r, c, sizeof(int))
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
            new_col = IntColumn(max(n, 1))
            new_col.length = n
            _col_gather_zero(new_col.data_c, self.data_c, indices, n, sizeof(int))
            return new_col
        else:
            new_data = <int*>aligned_malloc(max(n, 1) * sizeof(int))
            if not new_data: raise MemoryError()
            _col_gather_zero(new_data, self.data_c, indices, n, sizeof(int))
            if self.owns_data and self.data_c != NULL:
                os_aligned_free(self.data_c)
            self.data_c = new_data
            self.capacity = n
            self.length = n
            self._base_obj = None
            return self

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def clip(self, object lower=None, object upper=None, bint inplace=False):
        cdef IntColumn col = self if inplace else self.copy()
        cdef int* ptr = col.data_c
        cdef int l = <int>lower if lower is not None else -2147483648
        cdef int u = <int>upper if upper is not None else 2147483647
        _n_clip(ptr, col.length, l, u)
        return col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def abs(self):
        cdef IntColumn new_col = self.copy()
        if new_col.length > 0:
            _col_abs(new_col.data_c, new_col.data_c, new_col.length)
        return new_col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def where(self, IntColumn cond, int other, bint inplace=False):
        cdef Py_ssize_t n = self.length
        cdef Py_ssize_t r = self._get_rows_c()
        cdef Py_ssize_t ccols = self.num_cols
        cdef char err_buf[128]
        cdef IntColumn col
        cdef Py_ssize_t i, j, base
        cdef int* col_ptr
        cdef int* self_ptr = self.data_c
        cdef int* cond_ptr = cond.data_c
        cdef int m

        if self.is_2d and ccols > 1:
            if cond.length != r:
                snprintf(err_buf, sizeof(err_buf), "Condition mask length %zd must match column rows %zd", cond.length, r)
                raise ValueError(err_buf.decode('ascii'))
            col = self if inplace else IntColumn(max(n, 1), cols=ccols)
            if not inplace:
                col.length = n
                col.is_2d = True
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

        col = self if inplace else IntColumn(max(n, 1), cols=self.num_cols)
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
        cdef Py_ssize_t r = self._get_rows_c()
        cdef Py_ssize_t ccols = self.num_cols
        cdef char err_buf[128]
        cdef IntColumn col
        cdef Py_ssize_t i, j, base
        cdef int* col_ptr
        cdef int* self_ptr = self.data_c
        cdef int* mask_ptr = mask.data_c
        cdef int m
        cdef int fill_val = <int>fill

        if self.is_2d and ccols > 1:
            if mask.length != r:
                snprintf(err_buf, sizeof(err_buf), "Mask length %zd must match column rows %zd", mask.length, r)
                raise ValueError(err_buf.decode('ascii'))
            col = IntColumn(max(n, 1), cols=ccols)
            col.length = n
            col.is_2d = True
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

        col = IntColumn(max(n, 1), cols=self.num_cols)
        col.length = n
        col_ptr = col.data_c

        with nogil:
            for i in range(n):
                m = -<int>(mask_ptr[i] != 0)
                col_ptr[i] = (self_ptr[i] & m) | (fill_val & ~m)
        return col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def mask(self, IntColumn cond, int other, bint inplace=False):
        cdef Py_ssize_t n = self.length
        cdef Py_ssize_t r = self._get_rows_c()
        cdef Py_ssize_t ccols = self.num_cols
        cdef char err_buf[128]
        cdef IntColumn col
        cdef Py_ssize_t i, j, base
        cdef int* col_ptr
        cdef int* self_ptr = self.data_c
        cdef int* cond_ptr = cond.data_c
        cdef int m

        if self.is_2d and ccols > 1:
            if cond.length != r:
                snprintf(err_buf, sizeof(err_buf), "Condition mask length %zd must match column rows %zd", cond.length, r)
                raise ValueError(err_buf.decode('ascii'))
            col = self if inplace else IntColumn(max(n, 1), cols=ccols)
            if not inplace:
                col.length = n
                col.is_2d = True
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

        col = self if inplace else IntColumn(max(n, 1), cols=self.num_cols)
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
        if not self.owns_data:
            raise RuntimeError("Cannot drop row on a view/slice column. Copy it first.")
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()
        if row_idx < 0:
            row_idx += r
        if row_idx < 0 or row_idx >= r:
            raise IndexError("Row index out of bounds")
        if self.is_2d:
            _col_shift_left_2d(self.data_c, &self.length, r, c, row_idx, sizeof(int))
        else:
            _col_drop_row(self.data_c, &self.length, row_idx, sizeof(int))

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
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()
        cdef Py_ssize_t i, j, idx
        cdef Py_ssize_t dst_base, src_base
        cdef int* res_ptr
        cdef int* src_ptr
        cdef IntColumn new_col

        if self.is_2d:
            new_col = IntColumn(max(n * c, 1), cols=c)
            new_col.length = n * c
            new_col.num_cols = <int>c
            new_col.is_2d = True
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

    cdef Py_ssize_t count_nonzero(self):
        return self._count_nonzeros()

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
        cdef bint is_2d_src = self.is_2d and c > 1
        cdef bint target_supports_2d = target_type in (TYPE_INT, TYPE_DOUBLE, TYPE_DATE, TYPE_DATETIME, TYPE_TIME)
        if is_2d_src and not target_supports_2d:
            raise NotImplementedError(
                f"Cannot astype 2D column ({c} sub-columns) to non-2D type {target}. "
                f"Supported 2D targets: int, float, date, datetime, time.")
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
            for i in range(first_col_n):
                sz = snprintf(buf, sizeof(buf), "%d", s_ptr[i])
                uc.append(buf[:sz].decode('ascii'))
            return uc

        elif target_type == TYPE_ASCII:
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

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_mean(self, Py_ssize_t window, min_periods=None, bint center=False):
        if window < 0:
            raise ValueError("window must be >= 0")
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef DoubleColumn res = self._make_like_nan_dbl()
        if window == 0 or r == 0: return res
        cdef Py_ssize_t mp = _resolve_min_periods(window, min_periods)
        if r < mp: return res
        cdef int* src = self.data_c
        _n_roll_mean(src, res.data_c, c, r, window, mp, center)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_sum(self, Py_ssize_t window, min_periods=None, bint center=False):
        if window < 0:
            raise ValueError("window must be >= 0")
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef DoubleColumn res = self._make_like_nan_dbl()
        if window == 0 or r == 0: return res
        cdef Py_ssize_t mp = _resolve_min_periods(window, min_periods)
        if r < mp: return res
        cdef int* src = self.data_c
        _n_roll_sum(src, res.data_c, c, r, window, mp, center)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_min(self, Py_ssize_t window, min_periods=None, bint center=False):
        if window <= 0:
            raise ValueError("window must be > 0")
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef IntColumn res = IntColumn(max(r * c, 1), cols=c)
        res.length = r * c
        cdef Py_ssize_t mp = _resolve_min_periods(window, min_periods)
        if r == 0: return res
        cdef int* dst = res.data_c
        cdef int* src = self.data_c
        _n_roll_min(src, dst, c, r, window, mp, center)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_max(self, Py_ssize_t window, min_periods=None, bint center=False):
        if window <= 0:
            raise ValueError("window must be > 0")
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef IntColumn res = IntColumn(max(r * c, 1), cols=c)
        res.length = r * c
        cdef Py_ssize_t mp = _resolve_min_periods(window, min_periods)
        if r == 0: return res
        cdef int* dst = res.data_c
        cdef int* src = self.data_c
        _n_roll_max(src, dst, c, r, window, mp, center)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_var(self, Py_ssize_t window, bint ddof=True, min_periods=None, bint center=False):
        if window <= 0:
            raise ValueError("window must be > 0")
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef DoubleColumn res = self._make_like_nan_dbl()
        cdef Py_ssize_t mp = _resolve_min_periods(window, min_periods)
        if r < mp: return res
        cdef int* src = self.data_c
        _n_roll_var(src, res.data_c, c, r, window, mp, center, ddof)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_std(self, Py_ssize_t window, bint ddof=True, min_periods=None, bint center=False):
        if window <= 0:
            raise ValueError("window must be > 0")
        cdef DoubleColumn var_col = self.roll_var(window, ddof, min_periods, center)
        cdef double* v = var_col.data_c
        cdef Py_ssize_t n = var_col.length, i
        with nogil:
            for i in range(n):
                if not isnan(v[i]):
                    v[i] = sqrt(v[i])
        return var_col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_median(self, Py_ssize_t window, min_periods=None, bint center=False):
        if window <= 0:
            raise ValueError("window must be > 0")
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef DoubleColumn res = self._make_like_nan_dbl()
        cdef Py_ssize_t mp = _resolve_min_periods(window, min_periods)
        if r == 0 or r < mp: return res
        cdef int* src = self.data_c
        _n_roll_quantile(src, res.data_c, c, r, window, mp, center, 0.5)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_quantile(self, Py_ssize_t window, double quantile, min_periods=None, bint center=False):
        if window <= 0:
            raise ValueError("window must be > 0")
        if quantile < 0.0 or quantile > 1.0:
            raise ValueError("quantile must be between 0 and 1")
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef DoubleColumn res = self._make_like_nan_dbl()
        cdef Py_ssize_t mp = _resolve_min_periods(window, min_periods)
        if r == 0 or r < mp: return res
        cdef int* src = self.data_c
        _n_roll_quantile(src, res.data_c, c, r, window, mp, center, quantile)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _roll_apply_weights(self, Py_ssize_t window, list weights):
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

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _roll_custom(self, Py_ssize_t window, object func):
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

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_skew(self, Py_ssize_t window, min_periods=None, bint center=False):
        if window <= 0:
            raise ValueError("window must be > 0")
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef DoubleColumn res = self._make_like_nan_dbl()
        cdef Py_ssize_t mp = _resolve_min_periods(window, min_periods)
        if r == 0 or r < mp: return res
        cdef int* src = self.data_c
        _n_roll_skew(src, res.data_c, c, r, window, mp, center)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_kurt(self, Py_ssize_t window, min_periods=None, bint center=False):
        if window <= 0:
            raise ValueError("window must be > 0")
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef DoubleColumn res = self._make_like_nan_dbl()
        cdef Py_ssize_t mp = _resolve_min_periods(window, min_periods)
        if r == 0 or r < mp: return res
        cdef int* src = self.data_c
        _n_roll_kurt(src, res.data_c, c, r, window, mp, center)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_corr(self, Py_ssize_t window, object other, min_periods=None, bint center=False):
        if window <= 0:
            raise ValueError("window must be > 0")
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef DoubleColumn res = self._make_like_nan_dbl()
        cdef Py_ssize_t mp = _resolve_min_periods(window, min_periods)
        if r == 0 or r < mp: return res

        if other.length < self.length:
            raise ValueError("other column must be at least as long as self")

        cdef bint other_is_double = isinstance(other, DoubleColumn)
        cdef bint other_is_int = isinstance(other, IntColumn)
        if not (other_is_double or other_is_int):
            raise TypeError("other must be DoubleColumn or IntColumn")

        cdef int* src = self.data_c
        cdef DoubleColumn x_buf
        if other_is_double:
            x_buf = _int_col_to_double(self)
            _n_roll_corr(x_buf.data_c, (<DoubleColumn>other).data_c, res.data_c, c, r, window, mp, center)
        else:
            _n_roll_corr(src, (<IntColumn>other).data_c, res.data_c, c, r, window, mp, center)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_cov(self, Py_ssize_t window, object other, min_periods=None, bint center=False):
        if window <= 0:
            raise ValueError("window must be > 0")
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef DoubleColumn res = self._make_like_nan_dbl()
        cdef Py_ssize_t mp = _resolve_min_periods(window, min_periods)
        if r == 0 or r < mp: return res

        if other.length < self.length:
            raise ValueError("other column must be at least as long as self")

        cdef bint other_is_double = isinstance(other, DoubleColumn)
        cdef bint other_is_int = isinstance(other, IntColumn)
        if not (other_is_double or other_is_int):
            raise TypeError("other must be DoubleColumn or IntColumn")

        cdef int* src = self.data_c
        cdef DoubleColumn x_buf
        if other_is_double:
            x_buf = _int_col_to_double(self)
            _n_roll_cov(x_buf.data_c, (<DoubleColumn>other).data_c, res.data_c, c, r, window, mp, center)
        else:
            _n_roll_cov(src, (<IntColumn>other).data_c, res.data_c, c, r, window, mp, center)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _roll(self, Py_ssize_t window, str method="mean", bint ddof=True, double quantile=0.5, list weights=None, object func=None):
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
            return self._roll_apply_weights(window, weights)
        elif method == "custom":
            if func is None:
                raise ValueError("func must be provided for custom method")
            return self._roll_custom(window, func)
        else:
            raise ValueError(f"Unknown roll method: {method}")

    # ============================================================
    #  ewm（指数加权移动）列级算子：mean/sum/var/std/cov/corr
    #  仅单参数窗口参数（com/span/halflife/alpha→alpha）在调用侧解析。
    # ============================================================
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def ewm_mean(self, double alpha, Py_ssize_t min_periods=0, bint adjust=True,
                 bint ignore_na=False):
        if alpha <= 0.0 or alpha > 1.0:
            raise ValueError("alpha must be in (0, 1)")
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef DoubleColumn res = self._make_like_nan_dbl()
        if r == 0: return res
        cdef double com = (1.0 - alpha) / alpha
        _n_ewm(self.data_c, res.data_c, c, r, alpha, com, min_periods, adjust, ignore_na, True)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def ewm_sum(self, double alpha, Py_ssize_t min_periods=0, bint adjust=True,
                bint ignore_na=False):
        if not adjust:
            raise NotImplementedError("sum is not implemented with adjust=False")
        if alpha <= 0.0 or alpha > 1.0:
            raise ValueError("alpha must be in (0, 1)")
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef DoubleColumn res = self._make_like_nan_dbl()
        if r == 0: return res
        cdef double com = (1.0 - alpha) / alpha
        _n_ewm(self.data_c, res.data_c, c, r, alpha, com, min_periods, adjust, ignore_na, False)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def ewm_var(self, double alpha, bint bias=False, Py_ssize_t min_periods=0,
                bint adjust=True, bint ignore_na=False):
        if alpha <= 0.0 or alpha > 1.0:
            raise ValueError("alpha must be in (0, 1)")
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef DoubleColumn res = self._make_like_nan_dbl()
        if r == 0: return res
        _n_ewmcov(self.data_c, self.data_c, res.data_c,
                  c, r, alpha, min_periods, adjust, ignore_na, bias)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def ewm_std(self, double alpha, bint bias=False, Py_ssize_t min_periods=0,
                bint adjust=True, bint ignore_na=False):
        cdef DoubleColumn var_col = self.ewm_var(alpha, bias, min_periods, adjust, ignore_na)
        cdef double* v = var_col.data_c
        cdef Py_ssize_t n = var_col.length, i
        with nogil:
            for i in range(n):
                if not isnan(v[i]):
                    v[i] = sqrt(v[i])
        return var_col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def ewm_cov(self, object other, double alpha, bint bias=False,
                Py_ssize_t min_periods=0, bint adjust=True, bint ignore_na=False):
        if alpha <= 0.0 or alpha > 1.0:
            raise ValueError("alpha must be in (0, 1)")
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef DoubleColumn res = self._make_like_nan_dbl()
        if r == 0: return res
        if other.length < self.length:
            raise ValueError("other column must be at least as long as self")
        cdef bint other_is_double = isinstance(other, DoubleColumn)
        cdef bint other_is_int = isinstance(other, IntColumn)
        if not (other_is_double or other_is_int):
            raise TypeError("other must be DoubleColumn or IntColumn")
        cdef int* src = self.data_c
        cdef DoubleColumn x_buf
        if other_is_double:
            x_buf = _int_col_to_double(self)
            _n_ewmcov(x_buf.data_c, (<DoubleColumn>other).data_c, res.data_c,
                      c, r, alpha, min_periods, adjust, ignore_na, bias)
        else:
            _n_ewmcov(src, (<IntColumn>other).data_c, res.data_c,
                      c, r, alpha, min_periods, adjust, ignore_na, bias)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def ewm_corr(self, object other, double alpha, Py_ssize_t min_periods=0,
                 bint adjust=True, bint ignore_na=False):
        if alpha <= 0.0 or alpha > 1.0:
            raise ValueError("alpha must be in (0, 1)")
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef DoubleColumn res = self._make_like_nan_dbl()
        if r == 0: return res
        if other.length < self.length:
            raise ValueError("other column must be at least as long as self")
        cdef bint other_is_double = isinstance(other, DoubleColumn)
        cdef bint other_is_int = isinstance(other, IntColumn)
        if not (other_is_double or other_is_int):
            raise TypeError("other must be DoubleColumn or IntColumn")
        cdef int* src = self.data_c
        cdef DoubleColumn x_buf, y_buf, var_x, var_y
        if other_is_double:
            x_buf = _int_col_to_double(self)
            y_buf = <DoubleColumn>other
        else:
            x_buf = _int_col_to_double(self)
            y_buf = _int_col_to_double(<IntColumn>other)
        var_x = DoubleColumn(max(r * c, 1), cols=c)
        var_x.length = r * c
        var_y = DoubleColumn(max(r * c, 1), cols=c)
        var_y.length = r * c
        _n_ewmcov(x_buf.data_c, x_buf.data_c, var_x.data_c,
                  c, r, alpha, min_periods, adjust, ignore_na, True)
        _n_ewmcov(y_buf.data_c, y_buf.data_c, var_y.data_c,
                  c, r, alpha, min_periods, adjust, ignore_na, True)
        _n_ewmcov(x_buf.data_c, y_buf.data_c, res.data_c,
                  c, r, alpha, min_periods, adjust, ignore_na, True)
        cdef double* cov_ptr = res.data_c
        cdef double* vx = var_x.data_c
        cdef double* vy = var_y.data_c
        cdef double denom
        cdef Py_ssize_t total = r * c, i
        with nogil:
            for i in range(total):
                denom = sqrt(vx[i] * vy[i])
                if not isnan(cov_ptr[i]) and not isnan(denom) and denom != 0.0:
                    cov_ptr[i] = cov_ptr[i] / denom
                else:
                    cov_ptr[i] = NAN
        return res




    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_win_mean(self, Py_ssize_t window, object weights, min_periods=None,
                      bint center=False):
        if window < 1: raise ValueError("window must be >= 1")
        cdef Py_ssize_t mp = _resolve_min_periods(window, min_periods)
        cdef double* wbuf = _weights_to_buf(weights, window)
        try:
            return self._roll_win_sum_mean(window, wbuf, mp, center, True)
        finally:
            os_aligned_free(wbuf)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_win_sum(self, Py_ssize_t window, object weights, min_periods=None,
                     bint center=False):
        if window < 1: raise ValueError("window must be >= 1")
        cdef Py_ssize_t mp = _resolve_min_periods(window, min_periods)
        cdef double* wbuf = _weights_to_buf(weights, window)
        try:
            return self._roll_win_sum_mean(window, wbuf, mp, center, False)
        finally:
            os_aligned_free(wbuf)

    cdef object _roll_win_sum_mean(self, Py_ssize_t window, double* wbuf,
                                   Py_ssize_t mp, bint center, bint avg):
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef DoubleColumn res = self._make_like_nan_dbl()
        if r == 0: return res
        _n_roll_wmean_sum(self.data_c, res.data_c, c, r, wbuf, window, mp, center, avg)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_win_var(self, Py_ssize_t window, object weights, bint ddof=True,
                     min_periods=None, bint center=False):
        if window < 1: raise ValueError("window must be >= 1")
        cdef Py_ssize_t mp = _resolve_min_periods(window, min_periods)
        cdef double* wbuf = _weights_to_buf(weights, window)
        try:
            return self._roll_win_var_buf(window, wbuf, mp, center, ddof)
        finally:
            os_aligned_free(wbuf)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_win_std(self, Py_ssize_t window, object weights, bint ddof=True,
                     min_periods=None, bint center=False):
        if window < 1: raise ValueError("window must be >= 1")
        cdef Py_ssize_t mp = _resolve_min_periods(window, min_periods)
        cdef double* wbuf = _weights_to_buf(weights, window)
        try:
            return self._roll_win_std_buf(window, wbuf, mp, center, ddof)
        finally:
            os_aligned_free(wbuf)

    cdef object _roll_win_var_buf(self, Py_ssize_t window, double* wbuf,
                                   Py_ssize_t mp, bint center, bint ddof):
        cdef Py_ssize_t c = self.cols, r = self.rows if self.rows > 0 else self.length
        cdef DoubleColumn res = self._make_like_nan_dbl()
        if r == 0: return res
        _n_roll_wvar(self.data_c, res.data_c, c, r, wbuf, window, mp, center, 1 if ddof else 0)
        return res

    cdef object _roll_win_std_buf(self, Py_ssize_t window, double* wbuf,
                                   Py_ssize_t mp, bint center, bint ddof):
        cdef DoubleColumn var_col = self._roll_win_var_buf(window, wbuf, mp, center, ddof)
        cdef double* v = var_col.data_c
        cdef Py_ssize_t n = var_col.length, i
        with nogil:
            for i in range(n):
                if not isnan(v[i]):
                    v[i] = sqrt(v[i])
        return var_col


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
    @cython.binding(False)
    @cython.always_allow_keywords(False)
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
        cdef double m
        cdef list result
        if self.is_2d and c > 1:
            result = []
            for j in range(c):
                m = _c_sum(&ptr[j * r], r) / <double>r
                result.append(_n_accum_var(&ptr[j * r], r, m))
            return result
        return _n_accum_var(ptr, self.length, <double>self.mean())

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
    # 工业级三路划分快速选择 → 已迁移至 _n_quickselect fused helper
    # ------------------------------------------------------------

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def median(self):
        if self.length == 0: return NAN
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c()

        if not self.is_2d or c <= 1:
            return _n_median(self.data_c, self.length)

        cdef list result = [None] * c
        cdef Py_ssize_t j
        for j in range(c):
            result[j] = _n_median(self.data_c + j * r, r)
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
        return _n_accum_mad(ptr, self.length, <double>self.mean())

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
            return [0] * c
        cdef double* ptr = self.data_c
        cdef Py_ssize_t i, j, base
        cdef set seen
        cdef list results
        if self.is_2d and c > 1:
            results = [None] * c
            for j in range(c):
                base = j * r
                seen = set()
                for i in range(r): seen.add(ptr[base + i])
                results[j] = len(seen)
            return results
        seen = set()
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

        if not self.is_2d or c <= 1:
            return _n_quantile(self.data_c, self.length, q)

        cdef list result = [None] * c
        cdef Py_ssize_t j
        for j in range(c):
            result[j] = _n_quantile(self.data_c + j * r, r, q)
        return result

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cumsum(self):
        cdef DoubleColumn res = DoubleColumn(max(self.capacity, 1), cols=self.num_cols)
        res.length = self.length
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef double* ptr = self.data_c
        _n_cumsum(ptr, res.data_c, c, r)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cummax(self):
        cdef DoubleColumn res = DoubleColumn(max(self.capacity, 1), cols=self.num_cols)
        res.length = self.length
        if self.length == 0: return res
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef double* ptr = self.data_c
        _n_cummax(ptr, res.data_c, c, r)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cummin(self):
        cdef DoubleColumn res = DoubleColumn(max(self.capacity, 1), cols=self.num_cols)
        res.length = self.length
        if self.length == 0: return res
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef double* ptr = self.data_c
        _n_cummin(ptr, res.data_c, c, r)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def diff(self, int periods=1):
        cdef DoubleColumn res = DoubleColumn(max(self.capacity, 1), cols=self.num_cols)
        res.length = self.length
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef double* ptr = self.data_c
        _n_diff(ptr, res.data_c, c, r, periods)
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
        _n_central_moments(self.data_c, self.length, &m, &m2, &m3, &m4)
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
        _n_central_moments(self.data_c, self.length, &m, &m2, &m3, &m4)
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
        _n_central_moments(self.data_c, self.length, &m, &m2, &m3, &m4)
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
        cdef double* ptr = self.data_c
        _n_cumprod(ptr, res.data_c, c, r)
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
        _n_add_val(data_ptr, res.data_c, self.length, val)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sub_val(self, double val):
        cdef DoubleColumn res = DoubleColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef double* data_ptr = self.data_c
        _n_sub_val(data_ptr, res.data_c, self.length, val)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def mul_val(self, double val):
        cdef DoubleColumn res = DoubleColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef double* data_ptr = self.data_c
        _n_mul_val(data_ptr, res.data_c, self.length, val)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def div_val(self, double val):
        cdef DoubleColumn res = DoubleColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef double* data_ptr = self.data_c
        _n_div_val(data_ptr, res.data_c, self.length, val)
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
        if other.num_cols != self.num_cols:
            raise ValueError(
                f"Shape mismatch: self has {self.num_cols} column(s), other has {other.num_cols}")
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
        if other.num_cols != self.num_cols:
            raise ValueError(
                f"Shape mismatch: self has {self.num_cols} column(s), other has {other.num_cols}")
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
        if other.num_cols != self.num_cols:
            raise ValueError(
                f"Shape mismatch: self has {self.num_cols} column(s), other has {other.num_cols}")
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
        if other.num_cols != self.num_cols:
            raise ValueError(
                f"Shape mismatch: self has {self.num_cols} column(s), other has {other.num_cols}")
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
        cdef list result, results
        if self.is_2d and c > 1:
            results = [None] * c
            for j in range(c):
                base = j * r
                seen = set()
                result = []
                for i in range(r):
                    val = ptr[base + i]
                    if val not in seen:
                        seen.add(val)
                        result.append(val)
                results[j] = result
            return results
        seen = set()
        result = []
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
            if other.num_cols != self.num_cols:
                raise ValueError(
                    f"Shape mismatch: self has {self.num_cols} column(s), other has {other.num_cols}")
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
            is_nan_val = isnan(v)
            with nogil:
                if is_nan_val:
                    if op == 4:
                        for i in range(n): r_ptr[i] = isnan(s_ptr[i])
                    elif op == 5:
                        for i in range(n): r_ptr[i] = not isnan(s_ptr[i])
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
    def clip(self, object lower=None, object upper=None, bint inplace=False):
        cdef DoubleColumn col = self if inplace else self.copy()
        cdef double* ptr = col.data_c
        cdef double lower_val = float('-inf') if lower is None else <double>lower
        cdef double upper_val = float('inf') if upper is None else <double>upper
        _n_clip(ptr, col.length, lower_val, upper_val)
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
        cdef bint is_2d_src = self.is_2d and c > 1
        cdef bint target_supports_2d = target_type in (TYPE_INT, TYPE_DOUBLE, TYPE_DATE, TYPE_DATETIME, TYPE_TIME)
        if is_2d_src and not target_supports_2d:
            raise NotImplementedError(
                f"Cannot astype 2D column ({c} sub-columns) to non-2D type {target}. "
                f"Supported 2D targets: int, float, date, datetime, time.")
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
                    bc.data_c[i] = (not isnan(d_ptr[i]) and d_ptr[i] != 0.0)
            return bc

        elif target_type == TYPE_UCS2:
            uc = UCS2Column(first_col_cap)
            for i in range(first_col_n):
                if isnan(d_ptr[i]):
                    uc.append("null")
                else:
                    sz = snprintf(buf, sizeof(buf), "%.6g", d_ptr[i])
                    uc.append(buf[:sz].decode('ascii'))
            return uc

        elif target_type == TYPE_ASCII:
            cc = AsciiColumn(first_col_cap)
            cc.length = first_col_n
            with nogil:
                for i in range(first_col_n):
                    if isnan(d_ptr[i]):
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
                if isnan(d_ptr[i]):
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
    def roll_mean(self, Py_ssize_t window, min_periods=None, bint center=False):
        """滚动均值计算。支持一维与二维矩阵。
        时间复杂度: O(rows * cols)，严格常数时间增量滑动。"""
        if window < 0: raise ValueError("window must be >= 0")
        cdef DoubleColumn res = self._make_like_nan()
        cdef Py_ssize_t r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef Py_ssize_t c = self.num_cols
        if window == 0 or r == 0: return res
        cdef Py_ssize_t mp = _resolve_min_periods(window, min_periods)
        if r < mp: return res
        cdef double* src = self.data_c
        _n_roll_mean(src, res.data_c, c, r, window, mp, center)
        return res

    # ============================================================
    # 2. 滚动求和 (roll_sum) - O(N)
    # ============================================================
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_sum(self, Py_ssize_t window, min_periods=None, bint center=False):
        """滚动累加和"""
        if window < 0: raise ValueError("window must be >= 0")
        cdef DoubleColumn res = self._make_like_nan()
        cdef Py_ssize_t r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef Py_ssize_t c = self.num_cols
        if window == 0 or r == 0: return res
        cdef Py_ssize_t mp = _resolve_min_periods(window, min_periods)
        if r < mp: return res
        cdef double* src = self.data_c
        _n_roll_sum(src, res.data_c, c, r, window, mp, center)
        return res

    # ============================================================
    # 3. 滚动极值 (roll_min / roll_max) - 单调队列（右对齐）/ 逐窗重扫（居中）
    # ============================================================
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_min(self, Py_ssize_t window, min_periods=None, bint center=False):
        """滚动最小值 (采用严格连续内存单调双端队列)"""
        if window < 1: raise ValueError("window must be >= 1")
        cdef DoubleColumn res = self._make_like_nan()
        cdef Py_ssize_t r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t mp = _resolve_min_periods(window, min_periods)
        if r == 0 or r < mp: return res
        cdef double* src = self.data_c
        cdef double* dst = res.data_c
        _n_roll_min(src, dst, c, r, window, mp, center)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_max(self, Py_ssize_t window, min_periods=None, bint center=False):
        """滚动最大值"""
        if window < 1: raise ValueError("window must be >= 1")
        cdef DoubleColumn res = self._make_like_nan()
        cdef Py_ssize_t r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t mp = _resolve_min_periods(window, min_periods)
        if r == 0 or r < mp: return res
        cdef double* src = self.data_c
        cdef double* dst = res.data_c
        _n_roll_max(src, dst, c, r, window, mp, center)
        return res

    # ============================================================
    # 4. 滚动方差与标准差 (roll_var / roll_std) - O(N) 稳定滑窗
    # ============================================================
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_var(self, Py_ssize_t window, bint ddof=True, min_periods=None, bint center=False):
        """滚动方差 (ddof=True 为样本方差 N-1，False 为总体方差 N)"""
        if window < 2: raise ValueError("window must be >= 2 for variance")
        cdef DoubleColumn res = self._make_like_nan()
        cdef Py_ssize_t r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t mp = _resolve_min_periods(window, min_periods)
        if r == 0 or r < mp: return res
        cdef double* src = self.data_c
        _n_roll_var(src, res.data_c, c, r, window, mp, center, ddof)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_std(self, Py_ssize_t window, bint ddof=True, min_periods=None, bint center=False):
        """滚动标准差"""
        cdef DoubleColumn res = self.roll_var(window, ddof, min_periods, center)
        cdef Py_ssize_t i, n = res.length
        cdef double* ptr = res.data_c
        with nogil:
            for i in range(n):
                if not isnan(ptr[i]):
                    ptr[i] = sqrt(ptr[i])
        return res

    # ============================================================
    # 5. 滚动中位数与分位数 (roll_median / roll_quantile)
    # ============================================================
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_quantile(self, Py_ssize_t window, double q, min_periods=None, bint center=False):
        """滚动任意分位数 (q 在 0.0 到 1.0 之间)"""
        if window < 1: raise ValueError("window must be >= 1")
        if q < 0.0 or q > 1.0: raise ValueError("q must be between 0.0 and 1.0")
        cdef DoubleColumn res = self._make_like_nan()
        cdef Py_ssize_t r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t mp = _resolve_min_periods(window, min_periods)
        if r == 0 or r < mp: return res
        cdef double* src = self.data_c
        _n_roll_quantile(src, res.data_c, c, r, window, mp, center, q)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_median(self, Py_ssize_t window, min_periods=None, bint center=False):
        """滚动中位数"""
        return self.roll_quantile(window, 0.5, min_periods, center)

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
    def roll_skew(self, Py_ssize_t window, min_periods=None, bint center=False):
        if window < 1: raise ValueError("window must be >= 1")
        cdef Py_ssize_t r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef Py_ssize_t c = self.num_cols
        cdef DoubleColumn res = self._make_like_nan()
        cdef Py_ssize_t mp = _resolve_min_periods(window, min_periods)
        if r == 0 or r < mp: return res
        cdef double* src = self.data_c
        _n_roll_skew(src, res.data_c, c, r, window, mp, center)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_kurt(self, Py_ssize_t window, min_periods=None, bint center=False):
        if window < 1: raise ValueError("window must be >= 1")
        cdef Py_ssize_t r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef Py_ssize_t c = self.num_cols
        cdef DoubleColumn res = self._make_like_nan()
        cdef Py_ssize_t mp = _resolve_min_periods(window, min_periods)
        if r == 0 or r < mp: return res
        cdef double* src = self.data_c
        _n_roll_kurt(src, res.data_c, c, r, window, mp, center)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_corr(self, Py_ssize_t window, object other, min_periods=None, bint center=False):
        if window < 1: raise ValueError("window must be >= 1")
        cdef Py_ssize_t r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef Py_ssize_t c = self.num_cols
        cdef DoubleColumn res = self._make_like_nan()
        cdef Py_ssize_t mp = _resolve_min_periods(window, min_periods)
        if r == 0 or r < mp: return res

        cdef bint other_is_double = isinstance(other, DoubleColumn)
        cdef bint other_is_int = isinstance(other, IntColumn)
        if not (other_is_double or other_is_int):
            raise TypeError("other must be DoubleColumn or IntColumn")
        if other.length < self.length:
            raise ValueError("other column must be at least as long as self")

        cdef double* src = self.data_c
        cdef DoubleColumn y_buf
        if other_is_double:
            _n_roll_corr(src, (<DoubleColumn>other).data_c, res.data_c, c, r, window, mp, center)
        else:
            y_buf = _int_col_to_double(<IntColumn>other)
            _n_roll_corr(src, y_buf.data_c, res.data_c, c, r, window, mp, center)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_cov(self, Py_ssize_t window, object other, min_periods=None, bint center=False):
        if window < 1: raise ValueError("window must be >= 1")
        cdef Py_ssize_t r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef Py_ssize_t c = self.num_cols
        cdef DoubleColumn res = self._make_like_nan()
        cdef Py_ssize_t mp = _resolve_min_periods(window, min_periods)
        if r == 0 or r < mp: return res

        cdef bint other_is_double = isinstance(other, DoubleColumn)
        cdef bint other_is_int = isinstance(other, IntColumn)
        if not (other_is_double or other_is_int):
            raise TypeError("other must be DoubleColumn or IntColumn")
        if other.length < self.length:
            raise ValueError("other column must be at least as long as self")

        cdef double* src = self.data_c
        cdef DoubleColumn y_buf
        if other_is_double:
            _n_roll_cov(src, (<DoubleColumn>other).data_c, res.data_c, c, r, window, mp, center)
        else:
            y_buf = _int_col_to_double(<IntColumn>other)
            _n_roll_cov(src, y_buf.data_c, res.data_c, c, r, window, mp, center)
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

    # ============================================================
    #  ewm（指数加权移动）列级算子：mean/sum/var/std/cov/corr
    # ============================================================
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def ewm_mean(self, double alpha, Py_ssize_t min_periods=0, bint adjust=True,
                 bint ignore_na=False):
        if alpha <= 0.0 or alpha > 1.0:
            raise ValueError("alpha must be in (0, 1)")
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef DoubleColumn res = self._make_like_nan()
        if r == 0: return res
        cdef double com = (1.0 - alpha) / alpha
        _n_ewm(self.data_c, res.data_c, c, r, alpha, com, min_periods, adjust, ignore_na, True)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def ewm_sum(self, double alpha, Py_ssize_t min_periods=0, bint adjust=True,
                bint ignore_na=False):
        if not adjust:
            raise NotImplementedError("sum is not implemented with adjust=False")
        if alpha <= 0.0 or alpha > 1.0:
            raise ValueError("alpha must be in (0, 1)")
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef DoubleColumn res = self._make_like_nan()
        if r == 0: return res
        cdef double com = (1.0 - alpha) / alpha
        _n_ewm(self.data_c, res.data_c, c, r, alpha, com, min_periods, adjust, ignore_na, False)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def ewm_var(self, double alpha, bint bias=False, Py_ssize_t min_periods=0,
                bint adjust=True, bint ignore_na=False):
        if alpha <= 0.0 or alpha > 1.0:
            raise ValueError("alpha must be in (0, 1)")
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef DoubleColumn res = self._make_like_nan()
        if r == 0: return res
        _n_ewmcov(self.data_c, self.data_c, res.data_c,
                  c, r, alpha, min_periods, adjust, ignore_na, bias)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def ewm_std(self, double alpha, bint bias=False, Py_ssize_t min_periods=0,
                bint adjust=True, bint ignore_na=False):
        cdef DoubleColumn var_col = self.ewm_var(alpha, bias, min_periods, adjust, ignore_na)
        cdef double* v = var_col.data_c
        cdef Py_ssize_t n = var_col.length, i
        with nogil:
            for i in range(n):
                if not isnan(v[i]):
                    v[i] = sqrt(v[i])
        return var_col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def ewm_cov(self, object other, double alpha, bint bias=False,
                Py_ssize_t min_periods=0, bint adjust=True, bint ignore_na=False):
        if alpha <= 0.0 or alpha > 1.0:
            raise ValueError("alpha must be in (0, 1)")
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef DoubleColumn res = self._make_like_nan()
        if r == 0: return res
        if other.length < self.length:
            raise ValueError("other column must be at least as long as self")
        cdef bint other_is_double = isinstance(other, DoubleColumn)
        cdef bint other_is_int = isinstance(other, IntColumn)
        if not (other_is_double or other_is_int):
            raise TypeError("other must be DoubleColumn or IntColumn")
        cdef double* src = self.data_c
        cdef DoubleColumn y_buf
        if other_is_double:
            _n_ewmcov(src, (<DoubleColumn>other).data_c, res.data_c,
                      c, r, alpha, min_periods, adjust, ignore_na, bias)
        else:
            y_buf = _int_col_to_double(<IntColumn>other)
            _n_ewmcov(src, y_buf.data_c, res.data_c,
                      c, r, alpha, min_periods, adjust, ignore_na, bias)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def ewm_corr(self, object other, double alpha, Py_ssize_t min_periods=0,
                 bint adjust=True, bint ignore_na=False):
        if alpha <= 0.0 or alpha > 1.0:
            raise ValueError("alpha must be in (0, 1)")
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef DoubleColumn res = self._make_like_nan()
        if r == 0: return res
        if other.length < self.length:
            raise ValueError("other column must be at least as long as self")
        cdef bint other_is_double = isinstance(other, DoubleColumn)
        cdef bint other_is_int = isinstance(other, IntColumn)
        if not (other_is_double or other_is_int):
            raise TypeError("other must be DoubleColumn or IntColumn")
        cdef double* src = self.data_c
        cdef DoubleColumn y_buf, var_x, var_y
        if other_is_double:
            y_buf = <DoubleColumn>other
        else:
            y_buf = _int_col_to_double(<IntColumn>other)
        var_x = DoubleColumn(max(r * c, 1), cols=c)
        var_x.length = r * c
        var_y = DoubleColumn(max(r * c, 1), cols=c)
        var_y.length = r * c
        _n_ewmcov(src, src, var_x.data_c, c, r, alpha, min_periods, adjust, ignore_na, True)
        _n_ewmcov(y_buf.data_c, y_buf.data_c, var_y.data_c, c, r, alpha, min_periods, adjust, ignore_na, True)
        _n_ewmcov(src, y_buf.data_c, res.data_c, c, r, alpha, min_periods, adjust, ignore_na, True)
        cdef double* cov_ptr = res.data_c
        cdef double* vx = var_x.data_c
        cdef double* vy = var_y.data_c
        cdef double denom
        cdef Py_ssize_t total = r * c, i
        with nogil:
            for i in range(total):
                denom = sqrt(vx[i] * vy[i])
                if not isnan(cov_ptr[i]) and not isnan(denom) and denom != 0.0:
                    cov_ptr[i] = cov_ptr[i] / denom
                else:
                    cov_ptr[i] = NAN
        return res




# ============================================================
    #  win_type 加权滚动（boxcar/triang）：仅 mean/sum/var/std
    # ============================================================
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_win_mean(self, Py_ssize_t window, object weights, min_periods=None,
                      bint center=False):
        if window < 1: raise ValueError("window must be >= 1")
        cdef Py_ssize_t mp = _resolve_min_periods(window, min_periods)
        cdef double* wbuf = _weights_to_buf(weights, window)
        try:
            return self._roll_win_sum_mean(window, wbuf, mp, center, True)
        finally:
            os_aligned_free(wbuf)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_win_sum(self, Py_ssize_t window, object weights, min_periods=None,
                     bint center=False):
        if window < 1: raise ValueError("window must be >= 1")
        cdef Py_ssize_t mp = _resolve_min_periods(window, min_periods)
        cdef double* wbuf = _weights_to_buf(weights, window)
        try:
            return self._roll_win_sum_mean(window, wbuf, mp, center, False)
        finally:
            os_aligned_free(wbuf)

    cdef object _roll_win_sum_mean(self, Py_ssize_t window, double* wbuf,
                                   Py_ssize_t mp, bint center, bint avg):
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef DoubleColumn res = self._make_like_nan()
        if r == 0: return res
        _n_roll_wmean_sum(self.data_c, res.data_c, c, r, wbuf, window, mp, center, avg)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_win_var(self, Py_ssize_t window, object weights, bint ddof=True,
                     min_periods=None, bint center=False):
        if window < 1: raise ValueError("window must be >= 1")
        cdef Py_ssize_t mp = _resolve_min_periods(window, min_periods)
        cdef double* wbuf = _weights_to_buf(weights, window)
        try:
            return self._roll_win_var_buf(window, wbuf, mp, center, ddof)
        finally:
            os_aligned_free(wbuf)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def roll_win_std(self, Py_ssize_t window, object weights, bint ddof=True,
                     min_periods=None, bint center=False):
        if window < 1: raise ValueError("window must be >= 1")
        cdef Py_ssize_t mp = _resolve_min_periods(window, min_periods)
        cdef double* wbuf = _weights_to_buf(weights, window)
        try:
            return self._roll_win_std_buf(window, wbuf, mp, center, ddof)
        finally:
            os_aligned_free(wbuf)

    cdef object _roll_win_var_buf(self, Py_ssize_t window, double* wbuf,
                                   Py_ssize_t mp, bint center, bint ddof):
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c() if self._get_rows_c() > 0 else self.length
        cdef DoubleColumn res = self._make_like_nan()
        if r == 0: return res
        _n_roll_wvar(self.data_c, res.data_c, c, r, wbuf, window, mp, center, 1 if ddof else 0)
        return res

    cdef object _roll_win_std_buf(self, Py_ssize_t window, double* wbuf,
                                   Py_ssize_t mp, bint center, bint ddof):
        cdef DoubleColumn var_col = self._roll_win_var_buf(window, wbuf, mp, center, ddof)
        cdef double* v = var_col.data_c
        cdef Py_ssize_t n = var_col.length, i
        with nogil:
            for i in range(n):
                if not isnan(v[i]):
                    v[i] = sqrt(v[i])
        return var_col


# ============================================================
#  AsciiColumn class definition
# ============================================================
@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef class AsciiColumn(BaseColumn):

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __cinit__(self, Py_ssize_t init_capacity=16):
        self.capacity = max(init_capacity, 1)
        self.length = 0
        self.dtype = TYPE_ASCII

        self.offsets_c = <Py_ssize_t*>aligned_malloc((self.capacity + 1) * sizeof(Py_ssize_t))
        if not self.offsets_c:
            raise MemoryError()

        self.byte_capacity = max(self.capacity * 64, 4096)
        self.byte_length = 0
        self.data_c = <char*>aligned_malloc(self.byte_capacity)
        if not self.data_c:
            os_aligned_free(self.offsets_c)
            raise MemoryError()
        self.offsets_c[0] = 0

    def __dealloc__(self):
        if self.offsets_c != NULL:
            os_aligned_free(self.offsets_c)
            self.offsets_c = NULL
        if self.data_c != NULL:
            os_aligned_free(self.data_c)
            self.data_c = NULL

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def get_size(self):
        return (self.capacity + 1) * sizeof(Py_ssize_t) + self.byte_capacity

    cdef void _reserve_rows(self, Py_ssize_t new_cap) except *:
        if new_cap <= self.capacity:
            return
        cdef Py_ssize_t* new_offsets = <Py_ssize_t*>aligned_realloc(
            self.offsets_c,
            (self.capacity + 1) * sizeof(Py_ssize_t),
            (new_cap + 1) * sizeof(Py_ssize_t))
        if not new_offsets:
            raise MemoryError()
        self.offsets_c = new_offsets
        self.capacity = new_cap

    cdef void _reserve_bytes(self, Py_ssize_t new_byte_cap) except *:
        if new_byte_cap <= self.byte_capacity:
            return
        cdef char* new_data = <char*>aligned_realloc(
            self.data_c, self.byte_capacity, new_byte_cap)
        if not new_data:
            raise MemoryError()
        self.data_c = new_data
        self.byte_capacity = new_byte_cap

    cdef inline str _get_str(self, Py_ssize_t idx):
        cdef Py_ssize_t start = self.offsets_c[idx]
        cdef Py_ssize_t str_len = self.offsets_c[idx + 1] - start - 1
        if str_len <= 0:
            return ""
        return self.data_c[start:start+str_len].decode('ascii', errors='replace')

    cdef void _set_item(self, Py_ssize_t idx, str val) except *:
        cdef bytes b_val = val.encode('ascii', errors='ignore')
        cdef const char* src = b_val
        cdef Py_ssize_t new_len = len(b_val)
        cdef Py_ssize_t old_start = self.offsets_c[idx]
        cdef Py_ssize_t old_byte_span = self.offsets_c[idx + 1] - old_start
        cdef Py_ssize_t new_byte_span = new_len + 1
        cdef Py_ssize_t diff, oi
        cdef Py_ssize_t tail_len

        diff = new_byte_span - old_byte_span
        if diff > 0:
            if self.byte_length + diff > self.byte_capacity:
                self._reserve_bytes(max(self.byte_capacity * 2, self.byte_length + diff + 4096))

        tail_len = self.byte_length - old_start - old_byte_span
        if tail_len > 0:
            memmove(self.data_c + old_start + new_byte_span,
                    self.data_c + old_start + old_byte_span,
                    tail_len)

        if new_len > 0:
            memcpy(self.data_c + old_start, src, new_len)
        self.data_c[old_start + new_len] = 0

        self.byte_length += diff
        for oi in range(idx + 1, self.length + 1):
            self.offsets_c[oi] += diff

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def append(self, str val):
        cdef bytes b_val = val.encode('ascii', errors='ignore') if val is not None else b''
        cdef Py_ssize_t slen = len(b_val)
        cdef Py_ssize_t need_bytes = slen + 1

        if self.length >= self.capacity:
            self._reserve_rows(max(16, self.capacity * 2))
        if self.byte_length + need_bytes > self.byte_capacity:
            self._reserve_bytes(max(self.byte_capacity * 2, self.byte_length + need_bytes + 4096))

        if slen > 0:
            memcpy(self.data_c + self.byte_length, <const char*>b_val, slen)
        self.data_c[self.byte_length + slen] = 0
        self.byte_length += need_bytes
        self.length += 1
        self.offsets_c[self.length] = self.byte_length

    cdef void append_nolen(self, str val) except *:
        """写入当前行(length)位置的值，但不推进 length（由 add_row 统一提交）。"""
        if self.length >= self.capacity:
            raise BufferError("AsciiColumn row capacity (%d) reached." % self.capacity)
        cdef bytes b_val = val.encode('ascii', errors='ignore') if val is not None else b''
        cdef Py_ssize_t slen = len(b_val)
        cdef Py_ssize_t need_bytes = slen + 1
        if self.byte_length + need_bytes > self.byte_capacity:
            self._reserve_bytes(max(self.byte_capacity * 2, self.byte_length + need_bytes + 4096))
        if slen > 0:
            memcpy(self.data_c + self.byte_length, <const char*>b_val, slen)
        self.data_c[self.byte_length + slen] = 0
        self.byte_length += need_bytes
        self.offsets_c[self.length + 1] = self.byte_length

    cdef void undo_nolen(self) noexcept:
        """回滚一次未提交(未推进 length)的写入，保证重试时不会产生残留字节。"""
        self.byte_length = self.offsets_c[self.length]
        self.offsets_c[self.length + 1] = self.byte_length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def copy(self):
        cdef Py_ssize_t cap = max(self.length, 1)
        cdef AsciiColumn new_col = AsciiColumn(cap)
        new_col._reserve_bytes(self.byte_capacity)
        new_col.length = self.length
        new_col.byte_length = self.byte_length
        memcpy(new_col.offsets_c, self.offsets_c, (self.length + 1) * sizeof(Py_ssize_t))
        if self.byte_length > 0:
            memcpy(new_col.data_c, self.data_c, self.byte_length)
        return new_col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def ffill(self):
        """向前填充空串（本库缺失语义：isna 视空串为缺失，对标 double 的 NaN 填充）。

        存储约定：每串占 content+1 字节（含 NUL 结尾），偏移差=1 表示空串。
        效率说明：两次 O(n) 遍历，一次确定目的偏移表，一次 memcpy 写载荷，
        结果字节区连续紧凑，与 append 的存储布局一致。"""
        cdef Py_ssize_t n = self.length
        cdef AsciiColumn res = AsciiColumn(max(n, 1))
        if n == 0:
            return res
        cdef Py_ssize_t* src_offs = self.offsets_c
        cdef Py_ssize_t* dst_offs = res.offsets_c
        cdef Py_ssize_t* src_map = <Py_ssize_t*>aligned_malloc(max(n, 1) * sizeof(Py_ssize_t))
        if not src_map:
            raise MemoryError()
        cdef Py_ssize_t i, s, span, cur_len, total = 0
        cdef Py_ssize_t last_idx = -1, last_len = 0
        cdef Py_ssize_t si, ss, ss_len
        cdef char* dst_data
        dst_offs[0] = 0
        try:
            for i in range(n):
                s = src_offs[i]
                span = src_offs[i + 1] - s
                if span <= 1 and last_idx >= 0:
                    src_map[i] = last_idx
                    cur_len = last_len
                elif span <= 1:
                    src_map[i] = -1
                    cur_len = 0
                else:
                    last_idx = i
                    last_len = span - 1
                    src_map[i] = i
                    cur_len = span - 1
                dst_offs[i + 1] = dst_offs[i] + cur_len + 1
            total = dst_offs[n]
            if total > 0:
                if total > res.byte_capacity:
                    res._reserve_bytes(total)
                dst_data = res.data_c
                for i in range(n):
                    s = dst_offs[i]
                    si = src_map[i]
                    if si < 0:
                        dst_data[s] = 0
                        continue
                    ss = src_offs[si]
                    ss_len = src_offs[si + 1] - ss - 1
                    if ss_len > 0:
                        memcpy(dst_data + s, self.data_c + ss, ss_len)
                    dst_data[s + ss_len] = 0
                res.byte_length = total
            res.length = n
        finally:
            os_aligned_free(src_map)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def bfill(self):
        """向后填充空串（缺失语义）。与 ffill() 同构，O(n)。"""
        cdef Py_ssize_t n = self.length
        cdef AsciiColumn res = AsciiColumn(max(n, 1))
        if n == 0:
            return res
        cdef Py_ssize_t* src_offs = self.offsets_c
        cdef Py_ssize_t* dst_offs = res.offsets_c
        cdef Py_ssize_t* src_map = <Py_ssize_t*>aligned_malloc(max(n, 1) * sizeof(Py_ssize_t))
        if not src_map:
            raise MemoryError()
        cdef Py_ssize_t i, s, span, cur_len, total = 0
        cdef Py_ssize_t next_idx = n
        cdef Py_ssize_t si, ss, ss_len
        cdef char* dst_data
        dst_offs[0] = 0
        try:
            for i in range(n - 1, -1, -1):
                if src_offs[i + 1] - src_offs[i] > 1:
                    next_idx = i
                src_map[i] = next_idx
            for i in range(n):
                s = src_offs[i]
                span = src_offs[i + 1] - s
                if span <= 1:
                    if src_map[i] < n:
                        si = src_map[i]
                        cur_len = src_offs[si + 1] - src_offs[si] - 1
                    else:
                        src_map[i] = -1
                        cur_len = 0
                else:
                    src_map[i] = i
                    cur_len = span - 1
                dst_offs[i + 1] = dst_offs[i] + cur_len + 1
            total = dst_offs[n]
            if total > 0:
                if total > res.byte_capacity:
                    res._reserve_bytes(total)
                dst_data = res.data_c
                for i in range(n):
                    s = dst_offs[i]
                    si = src_map[i]
                    if si < 0:
                        dst_data[s] = 0
                        continue
                    ss = src_offs[si]
                    ss_len = src_offs[si + 1] - ss - 1
                    if ss_len > 0:
                        memcpy(dst_data + s, self.data_c + ss, ss_len)
                    dst_data[s + ss_len] = 0
                res.byte_length = total
            res.length = n
        finally:
            os_aligned_free(src_map)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_back(self):
        if self.length == 0:
            raise IndexError("pop from empty column")
        cdef str val = self._get_str(self.length - 1)
        self.byte_length = self.offsets_c[self.length - 1]
        self.length -= 1
        return val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_front(self):
        if self.length == 0:
            raise IndexError("pop from empty column")
        cdef str val = self._get_str(0)
        cdef Py_ssize_t str_len = self.offsets_c[1] - self.offsets_c[0]
        cdef Py_ssize_t rem_bytes = self.byte_length - self.offsets_c[1]
        if rem_bytes > 0:
            memmove(self.data_c, self.data_c + self.offsets_c[1], rem_bytes)
        self.byte_length -= str_len
        cdef Py_ssize_t i
        for i in range(self.length):
            self.offsets_c[i] = self.offsets_c[i + 1] - str_len
        self.length -= 1
        return val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def concat(self, AsciiColumn other):
        cdef Py_ssize_t other_len = other.length
        if other_len == 0: return
        cdef Py_ssize_t new_len = self.length + other_len
        self._reserve_rows(new_len)
        self._reserve_bytes(self.byte_length + other.byte_length)

        cdef Py_ssize_t base_offset = self.byte_length
        cdef Py_ssize_t i
        cdef Py_ssize_t* self_offs = self.offsets_c
        cdef Py_ssize_t* other_offs = other.offsets_c

        if other.byte_length > 0:
            memcpy(self.data_c + self.byte_length, other.data_c, other.byte_length)

        for i in range(other_len):
            self_offs[self.length + 1 + i] = base_offset + other_offs[i + 1]

        self.length = new_len
        self.byte_length += other.byte_length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __getitem__(self, Py_ssize_t idx):
        if idx < 0: idx += self.length
        if idx < 0 or idx >= self.length:
            raise IndexError("Index out of bounds")
        return self._get_str(idx)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def contains(self, str substring):
        cdef bytes b_sub = substring.encode('ascii')
        cdef const char* c_sub = b_sub
        cdef Py_ssize_t n = self.length
        cdef int* out_idx = <int*>aligned_malloc(max(n, 1) * sizeof(int))
        if not out_idx: raise MemoryError()

        cdef char* data = self.data_c
        cdef Py_ssize_t* offs = self.offsets_c
        cdef int count = 0
        cdef Py_ssize_t i

        with nogil:
            for i in range(n):
                if _c_str_contains(data + offs[i], c_sub):
                    out_idx[count] = <int>i
                    count += 1
        cdef list res = [out_idx[j] for j in range(count)]
        os_aligned_free(out_idx)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def replace(self, str old_str, str new_str, int count=-1):
        cdef AsciiColumn res = AsciiColumn(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            res.append(self._get_str(i).replace(old_str, new_str, count))
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def replaces(self, *args):
        if len(args) % 2 != 0:
            raise ValueError("Arguments must be in pairs")
        cdef AsciiColumn res = AsciiColumn(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length, k, n_args = len(args)
        cdef str s
        for i in range(n):
            s = self._get_str(i)
            for k in range(0, n_args, 2):
                s = s.replace(str(args[k]), str(args[k + 1]))
            res.append(s)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def extract(self, str pattern, int group=1):
        cdef object compiled = re.compile(pattern)
        cdef AsciiColumn res = AsciiColumn(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        cdef str s, val
        for i in range(n):
            s = self._get_str(i)
            m = compiled.search(s)
            if m is not None:
                try:
                    val = m.group(group)
                except IndexError:
                    val = ''
                res.append(val if val is not None else '')
            else:
                res.append('')
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def extract_twice(self, str pattern):
        cdef object compiled = re.compile(pattern)
        cdef AsciiColumn res1 = AsciiColumn(max(self.length, 1))
        cdef AsciiColumn res2 = AsciiColumn(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        cdef str s
        for i in range(n):
            s = self._get_str(i)
            m = compiled.search(s)
            if m is not None and len(m.groups()) >= 2:
                res1.append(m.group(1) or '')
                res2.append(m.group(2) or '')
            else:
                res1.append('')
                res2.append('')
        return res1, res2

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def len(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                out_ptr[i] = <int>(offs[i + 1] - offs[i] - 1)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def len_list(self):
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i, n = self.length
        cdef list res = []
        for i in range(n):
            res.append(offs[i + 1] - offs[i] - 1)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def startswith(self, prefix):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef char* data = self.data_c
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i, n = self.length
        cdef bytes pbytes
        cdef const char* c_pat
        cdef size_t plen

        if isinstance(prefix, str):
            pbytes = prefix.encode('ascii', errors='ignore')
            c_pat = pbytes
            plen = len(pbytes)
            with nogil:
                for i in range(n):
                    out_ptr[i] = 1 if _c_startswith(data + offs[i], c_pat, plen) else 0
        elif isinstance(prefix, (list, tuple)):
            for i in range(n):
                s = self._get_str(i)
                out_ptr[i] = 1 if any(s.startswith(p) for p in prefix) else 0
        else:
            pbytes = str(prefix).encode('ascii', errors='ignore')
            c_pat = pbytes
            plen = len(pbytes)
            with nogil:
                for i in range(n):
                    out_ptr[i] = 1 if _c_startswith(data + offs[i], c_pat, plen) else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def endswith(self, suffix):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef char* data = self.data_c
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i, n = self.length
        cdef bytes sbytes
        cdef const char* c_pat
        cdef size_t slen

        if isinstance(suffix, str):
            sbytes = suffix.encode('ascii', errors='ignore')
            c_pat = sbytes
            slen = len(sbytes)
            with nogil:
                for i in range(n):
                    out_ptr[i] = 1 if _c_endswith(data + offs[i], c_pat, slen) else 0
        elif isinstance(suffix, (list, tuple)):
            for i in range(n):
                s = self._get_str(i)
                out_ptr[i] = 1 if any(s.endswith(p) for p in suffix) else 0
        else:
            sbytes = str(suffix).encode('ascii', errors='ignore')
            c_pat = sbytes
            slen = len(sbytes)
            with nogil:
                for i in range(n):
                    out_ptr[i] = 1 if _c_endswith(data + offs[i], c_pat, slen) else 0
        return res

    # ------------------------------------------------------------
    # VECTORIZED: lower / upper using Branchless Bitwise Masks
    # ------------------------------------------------------------
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def lower(self):
        cdef AsciiColumn res = self.copy()
        cdef Py_ssize_t i, j, n = self.length
        cdef char* ptr
        cdef unsigned char c
        cdef unsigned char is_up
        cdef Py_ssize_t str_len
        cdef Py_ssize_t* offs = res.offsets_c
        with nogil:
            for i in range(n):
                ptr = res.data_c + offs[i]
                str_len = offs[i + 1] - offs[i] - 1
                for j in range(str_len):
                    c = <unsigned char>ptr[j]
                    is_up = (c >= 65) & (c <= 90)
                    ptr[j] = c | (is_up << 5)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def upper(self):
        cdef AsciiColumn res = self.copy()
        cdef Py_ssize_t i, j, n = self.length
        cdef char* ptr
        cdef unsigned char c
        cdef unsigned char is_low
        cdef Py_ssize_t str_len
        cdef Py_ssize_t* offs = res.offsets_c
        with nogil:
            for i in range(n):
                ptr = res.data_c + offs[i]
                str_len = offs[i + 1] - offs[i] - 1
                for j in range(str_len):
                    c = <unsigned char>ptr[j]
                    is_low = (c >= 97) & (c <= 122)
                    ptr[j] = c & ~(is_low << 5)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isdigit(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        with nogil:
            _c_pred_mask(self.data_c, self.offsets_c, self.length, res.data_c, 0)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isalpha(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        with nogil:
            _c_pred_mask(self.data_c, self.offsets_c, self.length, res.data_c, 1)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isalnum(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        with nogil:
            _c_pred_mask(self.data_c, self.offsets_c, self.length, res.data_c, 2)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isspace(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        with nogil:
            _c_pred_mask(self.data_c, self.offsets_c, self.length, res.data_c, 3)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def islower(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        with nogil:
            _c_case_mask(self.data_c, self.offsets_c, self.length, res.data_c, 0)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isupper(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        with nogil:
            _c_case_mask(self.data_c, self.offsets_c, self.length, res.data_c, 1)
        return res

    # ------------------------------------------------------------
    # O(N) Memory-Safe implementations for stripping & replacing
    # ------------------------------------------------------------
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def strip(self):
        cdef Py_ssize_t n = self.length
        cdef AsciiColumn res = AsciiColumn(max(n, 1))
        res._reserve_bytes(self.byte_length)

        cdef char* src = self.data_c
        cdef char* dst = res.data_c
        cdef Py_ssize_t dst_offset = 0
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i, cur_len, left, right, new_len

        with nogil:
            for i in range(n):
                cur_len = offs[i+1] - offs[i] - 1
                if cur_len == 0:
                    dst[dst_offset] = 0
                    dst_offset += 1
                    res.offsets_c[i+1] = dst_offset
                    continue

                left = 0
                while left < cur_len and _is_space(<unsigned char>src[offs[i] + left]):
                    left += 1

                right = cur_len - 1
                while right >= left and _is_space(<unsigned char>src[offs[i] + right]):
                    right -= 1

                new_len = right - left + 1 if right >= left else 0

                if new_len > 0:
                    memcpy(dst + dst_offset, src + offs[i] + left, new_len)
                dst[dst_offset + new_len] = 0
                dst_offset += new_len + 1
                res.offsets_c[i+1] = dst_offset

        res.length = n
        res.byte_length = dst_offset
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def lstrip(self):
        cdef Py_ssize_t n = self.length
        cdef AsciiColumn res = AsciiColumn(max(n, 1))
        res._reserve_bytes(self.byte_length)

        cdef char* src = self.data_c
        cdef char* dst = res.data_c
        cdef Py_ssize_t dst_offset = 0
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i, cur_len, left, copy_len

        with nogil:
            for i in range(n):
                cur_len = offs[i+1] - offs[i] - 1
                if cur_len == 0:
                    dst[dst_offset] = 0
                    dst_offset += 1
                    res.offsets_c[i+1] = dst_offset
                    continue

                left = 0
                while left < cur_len and _is_space(<unsigned char>src[offs[i] + left]):
                    left += 1

                copy_len = cur_len - left
                if copy_len > 0:
                    memcpy(dst + dst_offset, src + offs[i] + left, copy_len)
                dst[dst_offset + copy_len] = 0
                dst_offset += copy_len + 1
                res.offsets_c[i+1] = dst_offset

        res.length = n
        res.byte_length = dst_offset
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def rstrip(self):
        cdef Py_ssize_t n = self.length
        cdef AsciiColumn res = AsciiColumn(max(n, 1))
        res._reserve_bytes(self.byte_length)

        cdef char* src = self.data_c
        cdef char* dst = res.data_c
        cdef Py_ssize_t dst_offset = 0
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i, cur_len, right, copy_len

        with nogil:
            for i in range(n):
                cur_len = offs[i+1] - offs[i] - 1
                if cur_len == 0:
                    dst[dst_offset] = 0
                    dst_offset += 1
                    res.offsets_c[i+1] = dst_offset
                    continue

                right = cur_len - 1
                while right >= 0 and _is_space(<unsigned char>src[offs[i] + right]):
                    right -= 1

                copy_len = right + 1
                if copy_len > 0:
                    memcpy(dst + dst_offset, src + offs[i], copy_len)
                dst[dst_offset + copy_len] = 0
                dst_offset += copy_len + 1
                res.offsets_c[i+1] = dst_offset

        res.length = n
        res.byte_length = dst_offset
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def removeprefix(self, str prefix):
        cdef bytes b_pre = prefix.encode('ascii')
        cdef const char* c_pre = b_pre
        cdef Py_ssize_t pre_len = len(b_pre)

        cdef Py_ssize_t n = self.length
        cdef AsciiColumn res = AsciiColumn(max(n, 1))
        res._reserve_bytes(self.byte_length)

        cdef char* src = self.data_c
        cdef char* dst = res.data_c
        cdef Py_ssize_t dst_offset = 0
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i, cur_len, copy_len

        with nogil:
            for i in range(n):
                cur_len = offs[i+1] - offs[i] - 1
                if cur_len >= pre_len and strncmp(src + offs[i], c_pre, pre_len) == 0:
                    copy_len = cur_len - pre_len
                    memcpy(dst + dst_offset, src + offs[i] + pre_len, copy_len)
                else:
                    copy_len = cur_len
                    memcpy(dst + dst_offset, src + offs[i], copy_len)

                dst[dst_offset + copy_len] = 0
                dst_offset += copy_len + 1
                res.offsets_c[i+1] = dst_offset

        res.length = n
        res.byte_length = dst_offset
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def removesuffix(self, str suffix):
        cdef bytes b_suf = suffix.encode('ascii')
        cdef const char* c_suf = b_suf
        cdef Py_ssize_t suf_len = len(b_suf)

        cdef Py_ssize_t n = self.length
        cdef AsciiColumn res = AsciiColumn(max(n, 1))
        res._reserve_bytes(self.byte_length)

        cdef char* src = self.data_c
        cdef char* dst = res.data_c
        cdef Py_ssize_t dst_offset = 0
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i, cur_len, copy_len

        with nogil:
            for i in range(n):
                cur_len = offs[i+1] - offs[i] - 1
                if cur_len >= suf_len and strncmp(src + offs[i] + cur_len - suf_len, c_suf, suf_len) == 0:
                    copy_len = cur_len - suf_len
                else:
                    copy_len = cur_len

                if copy_len > 0:
                    memcpy(dst + dst_offset, src + offs[i], copy_len)
                dst[dst_offset + copy_len] = 0
                dst_offset += copy_len + 1
                res.offsets_c[i+1] = dst_offset

        res.length = n
        res.byte_length = dst_offset
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def zfill(self, Py_ssize_t width):
        cdef Py_ssize_t n = self.length
        cdef AsciiColumn res = AsciiColumn(max(n, 1))

        cdef Py_ssize_t req_bytes = 0
        cdef Py_ssize_t i, curr_len
        cdef Py_ssize_t* offs = self.offsets_c
        for i in range(n):
            curr_len = offs[i+1] - offs[i] - 1
            req_bytes += max(curr_len, width) + 1

        res._reserve_bytes(req_bytes)

        cdef char* src = self.data_c
        cdef char* dst = res.data_c
        cdef Py_ssize_t dst_offset = 0
        cdef Py_ssize_t pad

        with nogil:
            for i in range(n):
                curr_len = offs[i+1] - offs[i] - 1
                if curr_len >= width:
                    memcpy(dst + dst_offset, src + offs[i], curr_len + 1)
                    dst_offset += curr_len + 1
                else:
                    pad = width - curr_len
                    memset(dst + dst_offset, b'0', pad)
                    memcpy(dst + dst_offset + pad, src + offs[i], curr_len + 1)
                    dst_offset += pad + curr_len + 1
                res.offsets_c[i+1] = dst_offset

        res.length = n
        res.byte_length = dst_offset
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def find(self, str substring):
        cdef bytes b_sub = substring.encode('ascii')
        cdef const char* c_sub = b_sub
        cdef size_t sub_len = len(b_sub)
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef char* data = self.data_c
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                out_ptr[i] = <int>_c_find(data + offs[i], c_sub, sub_len)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def rfind(self, str substring):
        cdef bytes b_sub = substring.encode('ascii')
        cdef const char* c_sub = b_sub
        cdef size_t sub_len = len(b_sub)
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef char* data = self.data_c
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                out_ptr[i] = <int>_c_rfind(data + offs[i], c_sub, sub_len)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def rindex(self, str substring):
        cdef IntColumn res = self.rfind(substring)
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            if out_ptr[i] == -1:
                raise ValueError(f"substring '{substring}' not found")
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def count(self, str substring):
        cdef bytes b_sub = substring.encode('ascii')
        cdef const char* c_sub = b_sub
        cdef size_t sub_len = len(b_sub)
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef char* data = self.data_c
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                out_ptr[i] = <int>_c_count(data + offs[i], c_sub, sub_len)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def match(self, str pattern):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef object compiled = re.compile(pattern)
        for i in range(n):
            out_ptr[i] = 1 if compiled.match(self._get_str(i)) is not None else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def fullmatch(self, str pattern):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef object compiled = re.compile(pattern)
        for i in range(n):
            out_ptr[i] = 1 if compiled.fullmatch(self._get_str(i)) is not None else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isnumeric(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef str s
        for i in range(n):
            s = self._get_str(i)
            out_ptr[i] = 1 if len(s) > 0 and s.isnumeric() else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isdecimal(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef str s
        for i in range(n):
            s = self._get_str(i)
            out_ptr[i] = 1 if len(s) > 0 and s.isdecimal() else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def title(self):
        cdef AsciiColumn res = AsciiColumn(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            res.append(self._get_str(i).title())
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def capitalize(self):
        cdef AsciiColumn res = AsciiColumn(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            res.append(self._get_str(i).capitalize())
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def swapcase(self):
        cdef AsciiColumn res = AsciiColumn(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            res.append(self._get_str(i).swapcase())
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def casefold(self):
        cdef AsciiColumn res = AsciiColumn(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            res.append(self._get_str(i).casefold())
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def center(self, int width, str fillchar=' '):
        cdef AsciiColumn res = AsciiColumn(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            res.append(self._get_str(i).center(width, fillchar))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def ljust(self, int width, str fillchar=' '):
        cdef AsciiColumn res = AsciiColumn(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            res.append(self._get_str(i).ljust(width, fillchar))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def rjust(self, int width, str fillchar=' '):
        cdef AsciiColumn res = AsciiColumn(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            res.append(self._get_str(i).rjust(width, fillchar))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pad(self, int width, str side='left', str fillchar=' '):
        if side == 'left':
            return self.rjust(width, fillchar)
        elif side == 'right':
            return self.ljust(width, fillchar)
        elif side == 'both':
            return self.center(width, fillchar)
        raise ValueError(f"Invalid side: '{side}'")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def rsplit(self, str sep=None, int maxsplit=-1):
        cdef list res = []
        cdef Py_ssize_t i, n = self.length
        cdef str s
        for i in range(n):
            s = self._get_str(i)
            if maxsplit >= 0:
                res.append(s.rsplit(sep, maxsplit))
            else:
                res.append(s.rsplit(sep))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def partition(self, str sep):
        cdef list res = []
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            res.append(self._get_str(i).partition(sep))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def rpartition(self, str sep):
        cdef list res = []
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            res.append(self._get_str(i).rpartition(sep))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def get(self, int index, str default=''):
        cdef AsciiColumn res = AsciiColumn(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        cdef str s
        for i in range(n):
            s = self._get_str(i)
            if 0 <= index < len(s):
                res.append(s[index])
            else:
                res.append(default)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def slice(self, int start=0, int stop=-1, int step=1):
        cdef AsciiColumn res = AsciiColumn(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        cdef str s
        for i in range(n):
            s = self._get_str(i)
            if stop < 0:
                res.append(s[start::step] if step != 0 else s[start:stop:step])
            else:
                res.append(s[start:stop:step])
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cat(self, other, str sep=''):
        cdef AsciiColumn res = AsciiColumn(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        cdef AsciiColumn other_cc
        if isinstance(other, AsciiColumn):
            other_cc = <AsciiColumn>other
            if other_cc.length != n:
                raise ValueError(f"Length mismatch: {n} vs {other_cc.length}")
            for i in range(n):
                if sep:
                    res.append(self._get_str(i) + sep + other_cc._get_str(i))
                else:
                    res.append(self._get_str(i) + other_cc._get_str(i))
        elif isinstance(other, list):
            if len(other) != n:
                raise ValueError(f"Length mismatch: {n} vs {len(other)}")
            for i in range(n):
                if sep:
                    res.append(self._get_str(i) + sep + str(other[i]))
                else:
                    res.append(self._get_str(i) + str(other[i]))
        else:
            for i in range(n):
                if sep:
                    res.append(self._get_str(i) + sep + str(other))
                else:
                    res.append(self._get_str(i) + str(other))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def prefix(self, str pre):
        cdef AsciiColumn res = AsciiColumn(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            res.append(pre + self._get_str(i))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def suffix(self, str suf):
        cdef AsciiColumn res = AsciiColumn(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            res.append(self._get_str(i) + suf)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def repeat(self, int repeats):
        cdef AsciiColumn res = AsciiColumn(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            res.append(self._get_str(i) * repeats)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def findall(self, str pattern):
        cdef list res = []
        cdef Py_ssize_t i, n = self.length
        cdef object compiled = re.compile(pattern)
        for i in range(n):
            res.append(compiled.findall(self._get_str(i)))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def extractall(self, str pattern, int group=1):
        cdef list res = []
        cdef Py_ssize_t i, n = self.length
        cdef object compiled = re.compile(pattern)
        cdef list matches
        cdef str s, val
        for i in range(n):
            matches = []
            s = self._get_str(i)
            for m in compiled.finditer(s):
                try:
                    val = m.group(group)
                except IndexError:
                    val = ''
                if val is not None:
                    matches.append(val)
            res.append(matches)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def unique(self):
        cdef set seen = set()
        cdef list result = []
        cdef Py_ssize_t i, n = self.length
        cdef str val
        for i in range(n):
            val = self._get_str(i)
            if val not in seen:
                seen.add(val)
                result.append(val)
        return result

    cdef IntColumn _cmp_op(self, val, int op):
        cdef Py_ssize_t i, n = self.length
        cdef IntColumn res = IntColumn(max(n, 1))
        res.length = n
        cdef int* out_ptr = res.data_c
        cdef char* data = self.data_c
        cdef Py_ssize_t* offs = self.offsets_c
        cdef bytes v_bytes
        cdef const char* v_ptr
        cdef char* o_data
        cdef Py_ssize_t* o_offs
        cdef int cmp_res
        cdef AsciiColumn other

        if isinstance(val, AsciiColumn):
            other = <AsciiColumn>val
            if other.length != n:
                raise ValueError(f"Length mismatch: {n} vs {other.length}")
            o_data = other.data_c
            o_offs = other.offsets_c
            with nogil:
                for i in range(n):
                    cmp_res = strcmp(data + offs[i], o_data + o_offs[i])
                    if op == 0:   out_ptr[i] = 1 if cmp_res < 0 else 0
                    elif op == 1: out_ptr[i] = 1 if cmp_res <= 0 else 0
                    elif op == 2: out_ptr[i] = 1 if cmp_res > 0 else 0
                    elif op == 3: out_ptr[i] = 1 if cmp_res >= 0 else 0
                    elif op == 4: out_ptr[i] = 1 if cmp_res == 0 else 0
                    elif op == 5: out_ptr[i] = 1 if cmp_res != 0 else 0
        else:
            v_bytes = str(val).encode('ascii', errors='ignore')
            v_ptr = v_bytes
            with nogil:
                for i in range(n):
                    cmp_res = strcmp(data + offs[i], v_ptr)
                    if op == 0:   out_ptr[i] = 1 if cmp_res < 0 else 0
                    elif op == 1: out_ptr[i] = 1 if cmp_res <= 0 else 0
                    elif op == 2: out_ptr[i] = 1 if cmp_res > 0 else 0
                    elif op == 3: out_ptr[i] = 1 if cmp_res >= 0 else 0
                    elif op == 4: out_ptr[i] = 1 if cmp_res == 0 else 0
                    elif op == 5: out_ptr[i] = 1 if cmp_res != 0 else 0
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __lt__(self, val):  return self._cmp_op(val, 0)
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __le__(self, val):  return self._cmp_op(val, 1)
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __gt__(self, val):  return self._cmp_op(val, 2)
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __ge__(self, val):  return self._cmp_op(val, 3)
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __eq__(self, val):  return self._cmp_op(val, 4)
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __ne__(self, val):  return self._cmp_op(val, 5)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __and__(self, other):
        if not isinstance(other, IntColumn):
            raise TypeError("Cannot & non-IntColumn mask")
        cdef IntColumn o = <IntColumn>other
        if self.length != o.length:
            raise ValueError(f"Length mismatch: {self.length} vs {o.length}")
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef int* other_ptr = o.data_c
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                out_ptr[i] = 1 if (offs[i + 1] - offs[i] - 1) > 0 and other_ptr[i] != 0 else 0
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __or__(self, other):
        if not isinstance(other, IntColumn):
            raise TypeError("Cannot | non-IntColumn mask")
        cdef IntColumn o = <IntColumn>other
        if self.length != o.length:
            raise ValueError(f"Length mismatch: {self.length} vs {o.length}")
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef int* other_ptr = o.data_c
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                out_ptr[i] = 1 if (offs[i + 1] - offs[i] - 1) > 0 or other_ptr[i] != 0 else 0
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __invert__(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                out_ptr[i] = 0 if (offs[i + 1] - offs[i] - 1) > 0 else 1
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def isin(self, set values):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            out_ptr[i] = 1 if self._get_str(i) in values else 0
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def where_mask(self, IntColumn mask, object fill):
        cdef Py_ssize_t n = self.length
        if mask.length != n:
            raise ValueError(f"Mask length {mask.length} != column length {n}")

        cdef AsciiColumn col = AsciiColumn(max(n, 1))
        cdef str fill_str = str(fill) if fill is not None else ""
        cdef int* mask_ptr = mask.data_c
        cdef Py_ssize_t i

        for i in range(n):
            if mask_ptr[i] != 0:
                col.append(self._get_str(i))
            else:
                col.append(fill_str)
        return col

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __str__(self):
        cdef list items = []
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            items.append(repr(self._get_str(i)))
        return f"AsciiColumn([{', '.join(items)}])"

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __bool__(self):
        return self.length > 0

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __len__(self):
        return self.length

    cdef AsciiColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=False):
        cdef Py_ssize_t i, idx
        cdef Py_ssize_t req_cap = max(n, 1)
        cdef AsciiColumn new_col

        cdef Py_ssize_t req_bytes = 0
        for i in range(n):
            idx = indices[i]
            if idx < 0:
                continue
            req_bytes += self.offsets_c[idx + 1] - self.offsets_c[idx]

        cdef Py_ssize_t req_byte_cap = max(req_bytes, 4096)
        cdef Py_ssize_t* new_offsets = <Py_ssize_t*>aligned_malloc((req_cap + 1) * sizeof(Py_ssize_t))
        cdef char* new_data = <char*>aligned_malloc(req_byte_cap)

        if not new_offsets or not new_data:
            if new_offsets: os_aligned_free(new_offsets)
            if new_data: os_aligned_free(new_data)
            raise MemoryError()

        new_offsets[0] = 0
        cdef Py_ssize_t current_byte_len = 0
        cdef Py_ssize_t start, end, str_len
        with nogil:
            for i in range(n):
                idx = indices[i]
                if idx < 0:
                    new_offsets[i + 1] = current_byte_len
                    continue
                start = self.offsets_c[idx]
                end = self.offsets_c[idx + 1]
                str_len = end - start
                if str_len > 0:
                    memcpy(new_data + current_byte_len, self.data_c + start, str_len)
                current_byte_len += str_len
                new_offsets[i + 1] = current_byte_len

        if inplace:
            os_aligned_free(self.offsets_c)
            os_aligned_free(self.data_c)
            self.offsets_c = new_offsets
            self.data_c = new_data
            self.byte_capacity = req_byte_cap
            self.byte_length = current_byte_len
            self.capacity = req_cap
            self.length = n
            return self
        else:
            new_col = AsciiColumn(req_cap)
            os_aligned_free(new_col.offsets_c)
            os_aligned_free(new_col.data_c)
            new_col.offsets_c = new_offsets
            new_col.data_c = new_data
            new_col.byte_capacity = req_byte_cap
            new_col.byte_length = current_byte_len
            new_col.length = n
            return new_col

    # ------------------------------------------------------------
    # Bound Checked Memory Shifting
    # ------------------------------------------------------------
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def drop_row(self, Py_ssize_t row_idx):
        if row_idx < 0:
            row_idx += self.length
        if row_idx < 0 or row_idx >= self.length:
            raise IndexError("Row index out of bounds")

        cdef Py_ssize_t start_offset = self.offsets_c[row_idx]
        cdef Py_ssize_t end_offset = self.offsets_c[row_idx + 1]
        cdef Py_ssize_t char_len = end_offset - start_offset
        cdef Py_ssize_t rem_bytes = self.byte_length - end_offset
        if rem_bytes > 0:
            memmove(self.data_c + start_offset,
                    self.data_c + end_offset,
                    rem_bytes)
        self.byte_length -= char_len
        cdef Py_ssize_t i
        for i in range(row_idx, self.length):
            self.offsets_c[i] = self.offsets_c[i + 1] - char_len
        self.length -= 1

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def shift_left(self, Py_ssize_t pos):
        if pos < 0:
            pos += self.length
        if pos < 0 or pos >= self.length:
            raise IndexError("Index out of bounds")

        cdef Py_ssize_t col_len = self.length
        cdef Py_ssize_t start_offset = self.offsets_c[pos]
        cdef Py_ssize_t end_offset = self.offsets_c[pos + 1]
        cdef Py_ssize_t removed_len = end_offset - start_offset
        cdef Py_ssize_t rem = self.byte_length - end_offset
        cdef Py_ssize_t i
        if rem > 0:
            memmove(self.data_c + start_offset,
                    self.data_c + end_offset,
                    rem)
        self.byte_length -= removed_len
        for i in range(pos, col_len):
            self.offsets_c[i] = self.offsets_c[i + 1] - removed_len
        self.length = col_len - 1

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def astype(self, object target):
        cdef int target_type = _resolve_astype_target(target)
        cdef Py_ssize_t i, n = self.length
        cdef Py_ssize_t safe_cap = max(n, 1)
        cdef str s
        cdef IntColumn ic
        cdef DoubleColumn dc
        cdef BoolColumn bc
        cdef UCS2Column uc
        cdef DateColumn tc
        cdef CategoryColumn catc
        cdef DateTimeColumn dtc
        cdef TimeColumn tmc

        if target_type == TYPE_INT:
            ic = IntColumn(safe_cap)
            ic.length = n
            for i in range(n):
                s = self._get_str(i)
                ic.data_c[i] = int(float(s)) if s else 0
            return ic
        elif target_type == TYPE_DOUBLE:
            dc = DoubleColumn(safe_cap)
            dc.length = n
            for i in range(n):
                s = self._get_str(i)
                dc.data_c[i] = float(s) if s else 0.0
            return dc
        elif target_type == TYPE_BOOL:
            bc = BoolColumn(safe_cap)
            bc.length = n
            for i in range(n):
                s = self._get_str(i).lower()
                bc.data_c[i] = 1 if s in ('true', '1', 'yes') else 0
            return bc
        elif target_type == TYPE_UCS2:
            uc = UCS2Column(safe_cap)
            for i in range(n):
                uc.append(self._get_str(i))
            return uc
        elif target_type == TYPE_DATE:
            tc = DateColumn(safe_cap)
            for i in range(n):
                s = self._get_str(i)
                tc.append(s if s else 0)
            return tc
        elif target_type == TYPE_CATEGORY:
            catc = CategoryColumn(safe_cap)
            for i in range(n):
                catc.append(self._get_str(i))
            return catc
        elif target_type == TYPE_DATETIME:
            dtc = DateTimeColumn(safe_cap)
            for i in range(n):
                s = self._get_str(i)
                dtc.append(s if s else '')
            return dtc
        elif target_type == TYPE_TIME:
            tmc = TimeColumn(safe_cap)
            for i in range(n):
                s = self._get_str(i)
                tmc.append(s if s else 0)
            return tmc
        elif target_type == TYPE_ASCII:
            return self.copy()
        raise TypeError(f"Failed to cast ASCII column to internal type id: {target_type}")

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def str_split(self, sep=None, int maxsplit=-1):
        cdef list res = []
        cdef Py_ssize_t i, n = self.length
        cdef str s
        for i in range(n):
            s = self._get_str(i)
            if maxsplit >= 0:
                res.append(s.split(sep, maxsplit))
            else:
                res.append(s.split(sep))
        return res

    # ------------------------------------------------------------
    # Extreme O(1) Overhead indexing bypassing Python string allocation
    # ------------------------------------------------------------
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def str_getitem(self, key):
        cdef Py_ssize_t n = self.length
        cdef AsciiColumn col = AsciiColumn(max(n, 1))
        cdef Py_ssize_t i, idx, s_len, idx_abs
        cdef Py_ssize_t* offs = self.offsets_c
        cdef char* src = self.data_c
        cdef char* dst
        cdef Py_ssize_t dst_offset = 0

        if isinstance(key, int):
            idx = <Py_ssize_t>key
            col._reserve_bytes(n * 2)
            dst = col.data_c
            with nogil:
                for i in range(n):
                    s_len = offs[i+1] - offs[i] - 1
                    idx_abs = s_len + idx if idx < 0 else idx
                    if 0 <= idx_abs < s_len:
                        dst[dst_offset] = src[offs[i] + idx_abs]
                        dst[dst_offset + 1] = 0
                        dst_offset += 2
                    else:
                        dst[dst_offset] = 0
                        dst_offset += 1
                    col.offsets_c[i+1] = dst_offset
            col.length = n
            col.byte_length = dst_offset
            return col

        elif isinstance(key, slice):
            # Fallback to python indexing for complex slices
            for i in range(n):
                col.append(self._get_str(i)[key])
            return col
        raise TypeError(f"Invalid index type: {type(key)}")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def str_contains_mask(self, object pat):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef object compiled
        if isinstance(pat, str):
            compiled = re.compile(pat)
        elif hasattr(pat, 'search'):
            compiled = pat
        else:
            compiled = re.compile(str(pat))
        for i in range(n):
            out_ptr[i] = 1 if compiled.search(self._get_str(i)) is not None else 0
        return res



@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef class UCS2Column(BaseColumn):

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __cinit__(self, Py_ssize_t init_capacity=64):
        self.capacity = max(init_capacity, 1)
        self.length = 0
        self.dtype = TYPE_UCS2
        self.char_capacity = max(self.capacity * 16, 1024)
        self.char_length = 0

        self.offsets_c = <Py_ssize_t*>aligned_malloc((self.capacity + 1) * sizeof(Py_ssize_t))
        self.chars_c = <Py_UCS2*>aligned_malloc(self.char_capacity * sizeof(Py_UCS2))

        if not self.offsets_c or not self.chars_c:
            raise MemoryError()

        self.offsets_c[0] = 0

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __dealloc__(self):
        if self.offsets_c is not NULL: os_aligned_free(self.offsets_c)
        if self.chars_c is not NULL: os_aligned_free(self.chars_c)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def get_size(self):
        return self.capacity * sizeof(Py_UCS2) + (self.capacity + 1) * sizeof(Py_ssize_t)

    cdef void _reserve_rows(self, Py_ssize_t new_cap) except *:
        cdef Py_ssize_t* new_ptr
        if new_cap > self.capacity:
            new_ptr = <Py_ssize_t*>aligned_realloc(self.offsets_c, (self.capacity + 1) * sizeof(Py_ssize_t), (new_cap + 1) * sizeof(Py_ssize_t))
            if not new_ptr: raise MemoryError()
            self.offsets_c = new_ptr
            self.capacity = new_cap

    cdef void _reserve_chars(self, Py_ssize_t new_char_cap) except *:
        cdef Py_UCS2* new_ptr
        if new_char_cap > self.char_capacity:
            new_ptr = <Py_UCS2*>aligned_realloc(self.chars_c, self.char_capacity * sizeof(Py_UCS2), new_char_cap * sizeof(Py_UCS2))
            if not new_ptr: raise MemoryError()
            self.chars_c = new_ptr
            self.char_capacity = new_char_cap

    cdef inline str _get_str(self, Py_ssize_t idx):
        cdef Py_ssize_t start = self.offsets_c[idx]
        cdef Py_ssize_t str_len = self.offsets_c[idx + 1] - start
        if str_len == 0:
            return ""
        return PyUnicode_FromKindAndData(2, self.chars_c + start, str_len)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def append(self, str val not None):
        cdef Py_ssize_t str_len = PyUnicode_GET_LENGTH(val)

        if self.length >= self.capacity:
            self._reserve_rows(self.capacity * 2)

        if self.char_length + str_len > self.char_capacity:
            self._reserve_chars(max(self.char_capacity * 2, self.char_length + str_len + 1024))

        if str_len > 0:
            _copy_py_str_to_ucs2(val, self.chars_c + self.char_length, str_len)

        self.char_length += str_len
        self.length += 1
        self.offsets_c[self.length] = self.char_length

    cdef void append_nolen(self, str val) except *:
        """写入当前行(length)位置的值，但不推进 length（由 add_row 统一提交）。"""
        if self.length >= self.capacity:
            raise BufferError("UCS2Column row capacity (%d) reached." % self.capacity)
        cdef Py_ssize_t str_len = PyUnicode_GET_LENGTH(val)
        if self.char_length + str_len > self.char_capacity:
            self._reserve_chars(max(self.char_capacity * 2, self.char_length + str_len + 1024))
        if str_len > 0:
            _copy_py_str_to_ucs2(val, self.chars_c + self.char_length, str_len)
        self.char_length += str_len
        self.offsets_c[self.length + 1] = self.char_length

    cdef void undo_nolen(self) noexcept:
        """回滚一次未提交(未推进 length)的写入，保证重试时不会产生残留字符。"""
        self.char_length = self.offsets_c[self.length]
        self.offsets_c[self.length + 1] = self.char_length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def copy(self):
        cdef UCS2Column new_col = UCS2Column(self.capacity)
        new_col._reserve_chars(self.char_capacity)
        new_col.length = self.length
        new_col.char_length = self.char_length
        memcpy(new_col.offsets_c, self.offsets_c, (self.length + 1) * sizeof(Py_ssize_t))
        memcpy(new_col.chars_c, self.chars_c, self.char_length * sizeof(Py_UCS2))
        return new_col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def ffill(self):
        """向前填充空串（本库缺失语义：isna 视空串为缺失，对标 double 的 NaN 填充）。

        效率说明：两次 O(n) 遍历，一次确定目的偏移表，一次 memcpy 写载荷，
        结果字符区连续紧凑，与 append 的存储布局一致。"""
        cdef Py_ssize_t n = self.length
        cdef UCS2Column res = UCS2Column(max(n, 1))
        if n == 0:
            return res
        cdef Py_ssize_t* src_offs = self.offsets_c
        cdef Py_ssize_t* dst_offs = res.offsets_c
        cdef Py_ssize_t* src_map = <Py_ssize_t*>aligned_malloc(max(n, 1) * sizeof(Py_ssize_t))
        if not src_map:
            raise MemoryError()
        cdef Py_ssize_t i, slen, cur_len, total = 0
        cdef Py_ssize_t last_idx = -1, last_len = 0
        cdef Py_ssize_t si, ss, sslen
        cdef Py_UCS2* dst_chars
        dst_offs[0] = 0
        try:
            for i in range(n):
                slen = src_offs[i + 1] - src_offs[i]
                if slen == 0 and last_idx >= 0:
                    src_map[i] = last_idx
                    cur_len = last_len
                elif slen == 0:
                    src_map[i] = -1
                    cur_len = 0
                else:
                    last_idx = i
                    last_len = slen
                    src_map[i] = i
                    cur_len = slen
                dst_offs[i + 1] = dst_offs[i] + cur_len
            total = dst_offs[n]
            if total > 0:
                if total > res.char_capacity:
                    res._reserve_chars(total)
                dst_chars = res.chars_c
                for i in range(n):
                    si = src_map[i]
                    if si < 0:
                        continue
                    ss = src_offs[si]
                    sslen = src_offs[si + 1] - ss
                    if sslen > 0:
                        memcpy(dst_chars + dst_offs[i], self.chars_c + ss, sslen * sizeof(Py_UCS2))
                res.char_length = total
            res.length = n
        finally:
            os_aligned_free(src_map)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def bfill(self):
        """向后填充空串（缺失语义）。与 ffill() 同构，O(n)。"""
        cdef Py_ssize_t n = self.length
        cdef UCS2Column res = UCS2Column(max(n, 1))
        if n == 0:
            return res
        cdef Py_ssize_t* src_offs = self.offsets_c
        cdef Py_ssize_t* dst_offs = res.offsets_c
        cdef Py_ssize_t* src_map = <Py_ssize_t*>aligned_malloc(max(n, 1) * sizeof(Py_ssize_t))
        if not src_map:
            raise MemoryError()
        cdef Py_ssize_t i, slen, cur_len, total = 0
        cdef Py_ssize_t next_idx = n
        cdef Py_ssize_t si, ss, sslen
        cdef Py_UCS2* dst_chars
        dst_offs[0] = 0
        try:
            for i in range(n - 1, -1, -1):
                if src_offs[i + 1] > src_offs[i]:
                    next_idx = i
                src_map[i] = next_idx
            for i in range(n):
                slen = src_offs[i + 1] - src_offs[i]
                if slen == 0:
                    if src_map[i] < n:
                        si = src_map[i]
                        cur_len = src_offs[si + 1] - src_offs[si]
                    else:
                        src_map[i] = -1
                        cur_len = 0
                else:
                    src_map[i] = i
                    cur_len = slen
                dst_offs[i + 1] = dst_offs[i] + cur_len
            total = dst_offs[n]
            if total > 0:
                if total > res.char_capacity:
                    res._reserve_chars(total)
                dst_chars = res.chars_c
                for i in range(n):
                    si = src_map[i]
                    if si < 0:
                        continue
                    ss = src_offs[si]
                    sslen = src_offs[si + 1] - ss
                    if sslen > 0:
                        memcpy(dst_chars + dst_offs[i], self.chars_c + ss, sslen * sizeof(Py_UCS2))
                res.char_length = total
            res.length = n
        finally:
            os_aligned_free(src_map)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def concat(self, UCS2Column other):
        self._reserve_rows(self.length + other.length)
        self._reserve_chars(self.char_length + other.char_length)

        cdef Py_ssize_t i
        cdef Py_ssize_t base_offset = self.char_length
        cdef Py_ssize_t other_len = other.length
        cdef Py_ssize_t* self_offs = self.offsets_c
        cdef Py_ssize_t* other_offs = other.offsets_c

        memcpy(self.chars_c + self.char_length, other.chars_c, other.char_length * sizeof(Py_UCS2))

        for i in range(other_len):
            self_offs[self.length + 1 + i] = base_offset + other_offs[i + 1]

        self.length += other_len
        self.char_length += other.char_length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __getitem__(self, Py_ssize_t idx):
        if idx < 0: idx += self.length
        if idx < 0 or idx >= self.length: raise IndexError()
        return self._get_str(idx)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_back(self):
        if self.length == 0: raise IndexError("pop from empty column")
        self.length -= 1
        cdef Py_ssize_t start = self.offsets_c[self.length]
        cdef Py_ssize_t str_len = self.char_length - start
        self.char_length -= str_len
        if str_len == 0:
            return ""
        return PyUnicode_FromKindAndData(2, self.chars_c + start, str_len)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_front(self):
        if self.length == 0: raise IndexError("pop from empty column")
        cdef Py_ssize_t str_len = self.offsets_c[1] - self.offsets_c[0]
        cdef str val
        if str_len == 0:
            val = ""
        else:
            val = PyUnicode_FromKindAndData(2, self.chars_c + self.offsets_c[0], str_len)
        cdef Py_ssize_t rem_chars = self.char_length - self.offsets_c[1]
        if rem_chars > 0:
            memmove(self.chars_c, self.chars_c + self.offsets_c[1], rem_chars * sizeof(Py_UCS2))
        self.char_length -= str_len
        cdef Py_ssize_t i
        for i in range(self.length):
            self.offsets_c[i] = self.offsets_c[i + 1] - str_len
        self.length -= 1
        return val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def len(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                out_ptr[i] = <int>(offs[i + 1] - offs[i])
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def startswith(self, str prefix):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i, start, str_len
        cdef Py_ssize_t sub_len = PyUnicode_GET_LENGTH(prefix)

        if sub_len == 0:
            for i in range(self.length):
                out_ptr[i] = 1
            return res

        cdef Py_UCS2* n_buf = <Py_UCS2*>aligned_malloc(sub_len * sizeof(Py_UCS2))
        if not n_buf: raise MemoryError()
        _copy_py_str_to_ucs2(prefix, n_buf, sub_len)

        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_UCS2* chars = self.chars_c
        with nogil:
            for i in range(self.length):
                start = offs[i]
                str_len = offs[i+1] - start
                out_ptr[i] = 1 if _ucs2_startswith(chars + start, str_len, n_buf, sub_len) else 0

        os_aligned_free(n_buf)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def endswith(self, str suffix):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i, start, str_len
        cdef Py_ssize_t sub_len = PyUnicode_GET_LENGTH(suffix)

        if sub_len == 0:
            for i in range(self.length):
                out_ptr[i] = 1
            return res

        cdef Py_UCS2* n_buf = <Py_UCS2*>aligned_malloc(sub_len * sizeof(Py_UCS2))
        if not n_buf: raise MemoryError()
        _copy_py_str_to_ucs2(suffix, n_buf, sub_len)

        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_UCS2* chars = self.chars_c
        with nogil:
            for i in range(self.length):
                start = offs[i]
                str_len = offs[i+1] - start
                out_ptr[i] = 1 if _ucs2_endswith(chars + start, str_len, n_buf, sub_len) else 0

        os_aligned_free(n_buf)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def contains(self, object pat):
        """
        判断字符串是否包含指定的正则表达式模式。
        pat: 可以是正则字符串 (如 r"\d+")，也可以是已编译的正则对象 (re.compile)
        """
        cdef IntColumn res = IntColumn(max(self.capacity, 1))
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i
        cdef str s

        cdef object compiled_pat
        if isinstance(pat, str):
            compiled_pat = re.compile(pat)
        elif hasattr(pat, 'search'):
            compiled_pat = pat
        else:
            compiled_pat = re.compile(str(pat))

        for i in range(self.length):
            s = self._get_str(i)
            if compiled_pat.search(s) is not None:
                out_ptr[i] = 1
            else:
                out_ptr[i] = 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def find(self, str substring):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i, start, str_len
        cdef Py_ssize_t sub_len = PyUnicode_GET_LENGTH(substring)

        if sub_len == 0:
            for i in range(self.length):
                out_ptr[i] = 0
            return res

        cdef Py_UCS2* n_buf = <Py_UCS2*>aligned_malloc(sub_len * sizeof(Py_UCS2))
        if not n_buf: raise MemoryError()
        _copy_py_str_to_ucs2(substring, n_buf, sub_len)

        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_UCS2* chars = self.chars_c
        cdef Py_UCS2* match

        with nogil:
            for i in range(self.length):
                start = offs[i]
                str_len = offs[i+1] - start
                match = _ucs2_strstr(chars + start, <int>str_len, n_buf, <int>sub_len)
                out_ptr[i] = <int>(match - (chars + start)) if match != NULL else -1

        os_aligned_free(n_buf)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def replace(self, str old, str new, int count=-1):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            res.append(self._get_str(i).replace(old, new, count))
        return res

    cdef inline void _set_item(self, Py_ssize_t idx, str val):
        """Replace string at given index in-place. O(N) array shift."""
        cdef Py_ssize_t old_start = self.offsets_c[idx]
        cdef Py_ssize_t old_len = self.offsets_c[idx + 1] - old_start
        cdef Py_ssize_t new_len = PyUnicode_GET_LENGTH(val)
        cdef Py_ssize_t diff, oi

        if new_len > old_len:
            diff = new_len - old_len
            if self.char_length + diff > self.char_capacity:
                self._reserve_chars(max(self.char_capacity * 2, self.char_length + diff + 1024))

            if self.char_length - (old_start + old_len) > 0:
                memmove(self.chars_c + old_start + new_len,
                        self.chars_c + old_start + old_len,
                        (self.char_length - old_start - old_len) * sizeof(Py_UCS2))
        elif new_len < old_len:
            if self.char_length - (old_start + old_len) > 0:
                memmove(self.chars_c + old_start + new_len,
                        self.chars_c + old_start + old_len,
                        (self.char_length - old_start - old_len) * sizeof(Py_UCS2))

        if new_len > 0:
            _copy_py_str_to_ucs2(val, self.chars_c + old_start, new_len)

        self.char_length += (new_len - old_len)
        diff = new_len - old_len
        for oi in range(idx + 1, self.length + 1):
            self.offsets_c[oi] += diff

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def lower(self):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        for i in range(self.length):
            s = self._get_str(i)
            res.append(s.lower())
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def upper(self):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self._get_str(i)
            res.append(s.upper())
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def strip(self):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self._get_str(i)
            res.append(s.strip())
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def lstrip(self):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self._get_str(i)
            res.append(s.lstrip())
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def rstrip(self):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self._get_str(i)
            res.append(s.rstrip())
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def count(self, str substring):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i
        cdef str s
        cdef Py_ssize_t sub_len = len(substring)
        cdef Py_ssize_t cnt, pos
        if sub_len == 0:
            for i in range(self.length):
                out_ptr[i] = <int>(len(self._get_str(i)) + 1)
            return res
        for i in range(self.length):
            s = self._get_str(i)
            cnt = 0
            pos = 0
            while True:
                pos = s.find(substring, pos)
                if pos == -1:
                    break
                cnt += 1
                pos += sub_len
            out_ptr[i] = <int>cnt
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isdigit(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self._get_str(i)
            out_ptr[i] = 1 if len(s) > 0 and s.isdigit() else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isalpha(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self._get_str(i)
            out_ptr[i] = 1 if len(s) > 0 and s.isalpha() else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isalnum(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self._get_str(i)
            out_ptr[i] = 1 if len(s) > 0 and s.isalnum() else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isspace(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self._get_str(i)
            out_ptr[i] = 1 if len(s) > 0 and s.isspace() else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def rfind(self, str substring):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i, start, str_len
        cdef Py_ssize_t sub_len = PyUnicode_GET_LENGTH(substring)

        if sub_len == 0:
            for i in range(self.length):
                start = self.offsets_c[i]
                out_ptr[i] = <int>(self.offsets_c[i + 1] - start)
            return res

        cdef Py_UCS2* n_buf = <Py_UCS2*>aligned_malloc(sub_len * sizeof(Py_UCS2))
        if not n_buf: raise MemoryError()
        _copy_py_str_to_ucs2(substring, n_buf, sub_len)

        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_UCS2* chars = self.chars_c
        cdef Py_ssize_t n = self.length
        cdef Py_ssize_t j, k
        cdef bint bmatch

        for i in range(n):
            start = offs[i]
            str_len = offs[i+1] - start
            out_ptr[i] = -1
            if sub_len <= str_len:
                for j in range(str_len - sub_len, -1, -1):
                    bmatch = True
                    for k in range(sub_len):
                        if chars[start + j + k] != n_buf[k]:
                            bmatch = False
                            break
                    if bmatch:
                        out_ptr[i] = <int>j
                        break

        os_aligned_free(n_buf)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def rindex(self, str substring):
        cdef IntColumn res = self.rfind(substring)
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i
        for i in range(self.length):
            if out_ptr[i] == -1:
                raise ValueError(f"substring '{substring}' not found")
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def title(self):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        for i in range(self.length):
            res.append(self._get_str(i).title())
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def capitalize(self):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        for i in range(self.length):
            res.append(self._get_str(i).capitalize())
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def swapcase(self):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        for i in range(self.length):
            res.append(self._get_str(i).swapcase())
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def casefold(self):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        for i in range(self.length):
            res.append(self._get_str(i).casefold())
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def islower(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self._get_str(i)
            out_ptr[i] = 1 if len(s) > 0 and s.islower() else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isupper(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self._get_str(i)
            out_ptr[i] = 1 if len(s) > 0 and s.isupper() else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isnumeric(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self._get_str(i)
            out_ptr[i] = 1 if len(s) > 0 and s.isnumeric() else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isdecimal(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self._get_str(i)
            out_ptr[i] = 1 if len(s) > 0 and s.isdecimal() else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def match(self, str pattern):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i
        cdef object compiled = re.compile(pattern)
        cdef str s
        for i in range(self.length):
            s = self._get_str(i)
            out_ptr[i] = 1 if compiled.match(s) is not None else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def fullmatch(self, str pattern):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i
        cdef object compiled = re.compile(pattern)
        cdef str s
        for i in range(self.length):
            s = self._get_str(i)
            out_ptr[i] = 1 if compiled.fullmatch(s) is not None else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def center(self, int width, str fillchar=' '):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        for i in range(self.length):
            res.append(self._get_str(i).center(width, fillchar))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def ljust(self, int width, str fillchar=' '):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        for i in range(self.length):
            res.append(self._get_str(i).ljust(width, fillchar))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def rjust(self, int width, str fillchar=' '):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        for i in range(self.length):
            res.append(self._get_str(i).rjust(width, fillchar))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pad(self, int width, str side='left', str fillchar=' '):
        if side == 'left':
            return self.rjust(width, fillchar)
        elif side == 'right':
            return self.ljust(width, fillchar)
        elif side == 'both':
            return self.center(width, fillchar)
        raise ValueError(f"Invalid side: '{side}'")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def zfill(self, int width):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        for i in range(self.length):
            res.append(self._get_str(i).zfill(width))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def rsplit(self, str sep=None, int maxsplit=-1):
        cdef list res = []
        cdef Py_ssize_t i
        for i in range(self.length):
            s = self._get_str(i)
            if maxsplit >= 0:
                res.append(s.rsplit(sep, maxsplit))
            else:
                res.append(s.rsplit(sep))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def partition(self, str sep):
        cdef list res = []
        cdef Py_ssize_t i
        for i in range(self.length):
            res.append(self._get_str(i).partition(sep))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def rpartition(self, str sep):
        cdef list res = []
        cdef Py_ssize_t i
        for i in range(self.length):
            res.append(self._get_str(i).rpartition(sep))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def get(self, int index, str default=''):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self._get_str(i)
            if 0 <= index < len(s):
                res.append(s[index])
            else:
                res.append(default)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def slice(self, int start=0, int stop=-1, int step=1):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        for i in range(self.length):
            s = self._get_str(i)
            if stop < 0:
                res.append(s[start::step] if step != 0 else s[start:stop:step])
            else:
                res.append(s[start:stop:step])
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cat(self, other, str sep=''):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i, n = self.length
        if isinstance(other, UCS2Column):
            other_uc = <UCS2Column>other
            if other_uc.length != n:
                raise ValueError(f"Length mismatch: {n} vs {other_uc.length}")
            for i in range(n):
                if sep:
                    res.append(self._get_str(i) + sep + other_uc._get_str(i))
                else:
                    res.append(self._get_str(i) + other_uc._get_str(i))
        elif isinstance(other, list):
            if len(other) != n:
                raise ValueError(f"Length mismatch: {n} vs {len(other)}")
            for i in range(n):
                if sep:
                    res.append(self._get_str(i) + sep + str(other[i]))
                else:
                    res.append(self._get_str(i) + str(other[i]))
        else:
            for i in range(n):
                if sep:
                    res.append(self._get_str(i) + sep + str(other))
                else:
                    res.append(self._get_str(i) + str(other))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def prefix(self, str pre):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        for i in range(self.length):
            res.append(pre + self._get_str(i))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def suffix(self, str suf):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        for i in range(self.length):
            res.append(self._get_str(i) + suf)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def repeat(self, int repeats):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        for i in range(self.length):
            res.append(self._get_str(i) * repeats)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def findall(self, str pattern):
        import re
        cdef list res = []
        cdef Py_ssize_t i
        cdef object compiled = re.compile(pattern)
        for i in range(self.length):
            res.append(compiled.findall(self._get_str(i)))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def extract(self, str pattern, int group=1):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        cdef object compiled = re.compile(pattern)
        cdef str s, val
        for i in range(self.length):
            s = self._get_str(i)
            m = compiled.search(s)
            if m is not None:
                try:
                    val = m.group(group)
                except IndexError:
                    val = ''
                res.append(val if val is not None else '')
            else:
                res.append('')
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def extractall(self, str pattern, int group=1):
        import re
        cdef list res = []
        cdef Py_ssize_t i
        cdef object compiled = re.compile(pattern)
        for i in range(self.length):
            matches = []
            for m in compiled.finditer(self._get_str(i)):
                try:
                    val = m.group(group)
                except IndexError:
                    val = ''
                if val is not None:
                    matches.append(val)
            res.append(matches)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def unique(self):
        cdef set seen = set()
        cdef list result = []
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self._get_str(i)
            if s not in seen:
                seen.add(s)
                result.append(s)
        return result

    cdef IntColumn _cmp_op(self, val, int op):
        cdef Py_ssize_t i
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef char err_buf[128]

        cdef Py_ssize_t* s_offs = self.offsets_c
        cdef Py_UCS2* s_chars = self.chars_c
        cdef Py_UCS2* o_chars
        cdef Py_ssize_t* o_offs
        cdef Py_UCS2* v_buf
        cdef Py_ssize_t v_len, len_s, len_o
        cdef int cmp_res

        if isinstance(val, UCS2Column):
            other = <UCS2Column>val
            if other.length != self.length:
                snprintf(err_buf, sizeof(err_buf), "Length mismatch: %zd vs %zd", self.length, other.length)
                raise ValueError(err_buf.decode('ascii'))

            o_offs = other.offsets_c
            o_chars = other.chars_c

            with nogil:
                for i in range(self.length):
                    len_s = s_offs[i+1] - s_offs[i]
                    len_o = o_offs[i+1] - o_offs[i]
                    cmp_res = _cmp_ucs2_buffers(s_chars + s_offs[i], len_s, o_chars + o_offs[i], len_o)

                    if op == 0:   out_ptr[i] = 1 if cmp_res < 0 else 0
                    elif op == 1: out_ptr[i] = 1 if cmp_res <= 0 else 0
                    elif op == 2: out_ptr[i] = 1 if cmp_res > 0 else 0
                    elif op == 3: out_ptr[i] = 1 if cmp_res >= 0 else 0
                    elif op == 4: out_ptr[i] = 1 if cmp_res == 0 else 0
                    elif op == 5: out_ptr[i] = 1 if cmp_res != 0 else 0
        else:
            v_str = <str>val
            v_len = PyUnicode_GET_LENGTH(v_str)
            v_buf = <Py_UCS2*>aligned_malloc(max(v_len, 1) * sizeof(Py_UCS2))
            if not v_buf: raise MemoryError()

            _copy_py_str_to_ucs2(v_str, v_buf, v_len)

            with nogil:
                for i in range(self.length):
                    len_s = s_offs[i+1] - s_offs[i]
                    cmp_res = _cmp_ucs2_buffers(s_chars + s_offs[i], len_s, v_buf, v_len)

                    if op == 0:   out_ptr[i] = 1 if cmp_res < 0 else 0
                    elif op == 1: out_ptr[i] = 1 if cmp_res <= 0 else 0
                    elif op == 2: out_ptr[i] = 1 if cmp_res > 0 else 0
                    elif op == 3: out_ptr[i] = 1 if cmp_res >= 0 else 0
                    elif op == 4: out_ptr[i] = 1 if cmp_res == 0 else 0
                    elif op == 5: out_ptr[i] = 1 if cmp_res != 0 else 0

            os_aligned_free(v_buf)
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __lt__(self, val):  return self._cmp_op(val, 0)
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __le__(self, val):  return self._cmp_op(val, 1)
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __gt__(self, val):  return self._cmp_op(val, 2)
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __ge__(self, val):  return self._cmp_op(val, 3)
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __eq__(self, val):  return self._cmp_op(val, 4)
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __ne__(self, val):  return self._cmp_op(val, 5)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __and__(self, other):
        if not isinstance(other, IntColumn):
            raise TypeError("Cannot & non-IntColumn mask")
        if self.length != other.length:
            raise ValueError("Length mismatch: %zd vs %zd" % (self.length, other.length))

        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef int* other_ptr = (<IntColumn>other).data_c
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i

        with nogil:
            for i in range(self.length):
                out_ptr[i] = 1 if (offs[i+1] - offs[i]) > 0 and other_ptr[i] else 0
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __or__(self, other):
        if not isinstance(other, IntColumn):
            raise TypeError("Cannot | non-IntColumn mask")
        if self.length != other.length:
            raise ValueError("Length mismatch: %zd vs %zd" % (self.length, other.length))

        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef int* other_ptr = (<IntColumn>other).data_c
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i

        with nogil:
            for i in range(self.length):
                out_ptr[i] = 1 if (offs[i+1] - offs[i]) > 0 or other_ptr[i] else 0
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __invert__(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i

        with nogil:
            for i in range(self.length):
                out_ptr[i] = 0 if (offs[i+1] - offs[i]) > 0 else 1
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def isin(self, set values):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self._get_str(i)
            out_ptr[i] = 1 if s in values else 0
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def where_mask(self, IntColumn mask, object fill):
        cdef Py_ssize_t n = self.length
        if mask.length != n:
            raise ValueError("Mask length %zd != column length %zd" % (mask.length, n))

        cdef UCS2Column col = UCS2Column(n)
        cdef str fill_str = str(fill) if fill is not None else ""
        cdef int* mask_ptr = mask.data_c
        cdef Py_ssize_t i

        for i in range(n):
            if mask_ptr[i]:
                col.append(self._get_str(i))
            else:
                col.append(fill_str)
        return col

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __str__(self):
        cdef list items = []
        cdef Py_ssize_t i
        for i in range(self.length):
            items.append(repr(self._get_str(i)))
        return f"UCS2Column([{', '.join(items)}])"

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __bool__(self):
        return self.length > 0

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __len__(self):
        return self.length

    cdef UCS2Column take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=False):
        cdef Py_ssize_t req_cap = max(n, 1)

        cdef Py_ssize_t req_char_len = 0
        cdef Py_ssize_t i, idx
        for i in range(n):
            idx = indices[i]
            if idx < 0:
                continue
            req_char_len += self.offsets_c[idx + 1] - self.offsets_c[idx]

        cdef Py_ssize_t req_char_cap = max(req_char_len, 1024)
        cdef Py_ssize_t* new_offsets = <Py_ssize_t*>aligned_malloc((req_cap + 1) * sizeof(Py_ssize_t))
        cdef Py_UCS2* new_chars = <Py_UCS2*>aligned_malloc(req_char_cap * sizeof(Py_UCS2))

        if not new_offsets or not new_chars:
            if new_offsets: os_aligned_free(new_offsets)
            if new_chars: os_aligned_free(new_chars)
            raise MemoryError()

        new_offsets[0] = 0
        cdef Py_ssize_t current_char_len = 0
        cdef Py_ssize_t start, end, str_len
        with nogil:
            for i in range(n):
                idx = indices[i]
                if idx < 0:
                    new_offsets[i + 1] = current_char_len
                    continue
                start = self.offsets_c[idx]
                end = self.offsets_c[idx + 1]
                str_len = end - start
                if str_len > 0:
                    memcpy(new_chars + current_char_len, self.chars_c + start, str_len * sizeof(Py_UCS2))
                current_char_len += str_len
                new_offsets[i + 1] = current_char_len

        if inplace:
            os_aligned_free(self.offsets_c)
            os_aligned_free(self.chars_c)
            self.offsets_c = new_offsets
            self.chars_c = new_chars
            self.char_capacity = req_char_cap
            self.char_length = current_char_len
            self.capacity = req_cap
            self.length = n
            return self
        else:
            new_col = UCS2Column(req_cap)
            os_aligned_free(new_col.offsets_c)
            os_aligned_free(new_col.chars_c)
            new_col.offsets_c = new_offsets
            new_col.chars_c = new_chars
            new_col.char_capacity = req_char_cap
            new_col.char_length = current_char_len
            new_col.length = n
            return new_col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def drop_row(self, Py_ssize_t row_idx):
        cdef Py_ssize_t start_offset = self.offsets_c[row_idx]
        cdef Py_ssize_t end_offset = self.offsets_c[row_idx + 1]
        cdef Py_ssize_t char_len = end_offset - start_offset
        cdef Py_ssize_t rem_chars = self.char_length - end_offset
        if rem_chars > 0:
            memmove(self.chars_c + start_offset,
                    self.chars_c + end_offset,
                    rem_chars * sizeof(Py_UCS2))
        self.char_length -= char_len
        cdef Py_ssize_t i
        for i in range(row_idx, self.length):
            self.offsets_c[i] = self.offsets_c[i + 1] - char_len
        self.length -= 1

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def shift_left(self, Py_ssize_t pos):
        cdef Py_ssize_t col_len = self.length
        cdef Py_ssize_t start_offset = self.offsets_c[pos]
        cdef Py_ssize_t end_offset = self.offsets_c[pos + 1]
        cdef Py_ssize_t removed_len = end_offset - start_offset
        cdef Py_ssize_t rem = self.char_length - end_offset
        cdef Py_ssize_t i
        if rem > 0:
            memmove(self.chars_c + start_offset,
                    self.chars_c + end_offset,
                    rem * sizeof(Py_UCS2))
        self.char_length -= removed_len
        for i in range(pos, col_len):
            self.offsets_c[i] = self.offsets_c[i + 1] - removed_len
        self.length = col_len - 1

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def astype(self, object target):
        cdef int target_type = _resolve_astype_target(target)
        cdef Py_ssize_t i, n = self.length
        cdef str s
        cdef Py_ssize_t safe_cap = max(n, 1)

        cdef IntColumn ic
        cdef DoubleColumn dc
        cdef BoolColumn bc
        cdef AsciiColumn cc
        cdef DateColumn tc
        cdef CategoryColumn catc
        cdef DateTimeColumn dtc
        cdef TimeColumn tmc
        cdef int* ic_ptr
        cdef double* dc_ptr
        cdef unsigned char* bc_ptr

        if target_type == TYPE_INT:
            ic = IntColumn(safe_cap)
            ic.length = n
            ic_ptr = ic.data_c
            for i in range(n):
                s = self._get_str(i)
                ic_ptr[i] = int(float(s)) if len(s) > 0 else 0
            return ic

        elif target_type == TYPE_DOUBLE:
            dc = DoubleColumn(safe_cap)
            dc.length = n
            dc_ptr = dc.data_c
            for i in range(n):
                s = self._get_str(i)
                dc_ptr[i] = float(s) if len(s) > 0 else 0.0
            return dc

        elif target_type == TYPE_BOOL:
            bc = BoolColumn(safe_cap)
            bc.length = n
            bc_ptr = bc.data_c
            for i in range(n):
                s = self._get_str(i)
                if len(s) > 0 and (s == 'true' or s == 'True' or s == '1' or s == 'yes' or s == 'Yes'):
                    bc_ptr[i] = 1
                else:
                    bc_ptr[i] = 0
            return bc

        elif target_type == TYPE_ASCII:
            cc = AsciiColumn(safe_cap)
            for i in range(n):
                cc.append(self._get_str(i))
            return cc

        elif target_type == TYPE_DATE:
            tc = DateColumn(safe_cap)
            for i in range(n):
                s = self._get_str(i)
                if len(s) > 0:
                    tc.append(s)
                else:
                    tc.append(0)
            return tc

        elif target_type == TYPE_CATEGORY:
            catc = CategoryColumn(safe_cap)
            for i in range(n):
                catc.append(self._get_str(i))
            return catc

        elif target_type == TYPE_DATETIME:
            dtc = DateTimeColumn(safe_cap)
            for i in range(n):
                s = self._get_str(i)
                if len(s) > 0:
                    dtc.append(s)
                else:
                    dtc.append('')
            return dtc

        elif target_type == TYPE_TIME:
            tmc = TimeColumn(safe_cap)
            for i in range(n):
                s = self._get_str(i)
                if len(s) > 0:
                    tmc.append(s)
                else:
                    tmc.append(0)
            return tmc

        elif target_type == TYPE_UCS2:
            return self.copy()

        cdef char err_buf[128]
        snprintf(err_buf, sizeof(err_buf), "Failed to cast UCS2 column to internal type id: %d", target_type)
        raise TypeError(err_buf.decode('ascii'))

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def str_split(self, sep=None, int maxsplit=-1):
        cdef list res = []
        cdef Py_ssize_t i
        for i in range(self.length):
            s = self._get_str(i)
            if maxsplit >= 0:
                res.append(s.split(sep, maxsplit))
            else:
                res.append(s.split(sep))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def str_getitem(self, key):
        cdef list res = []
        cdef Py_ssize_t i
        cdef str s
        if isinstance(key, int):
            idx = key
            for i in range(self.length):
                s = self._get_str(i)
                if idx < 0:
                    idx_abs = len(s) + idx
                else:
                    idx_abs = idx
                if 0 <= idx_abs < len(s):
                    res.append(s[idx_abs])
                else:
                    res.append('')
            col = UCS2Column(max(self.length, 1))
            for v in res:
                col.append(v)
            return col
        elif isinstance(key, slice):
            for i in range(self.length):
                s = self._get_str(i)
                res.append(s[key])
            col = UCS2Column(max(self.length, 1))
            for v in res:
                col.append(v)
            return col
        raise TypeError(f"Invalid index type: {type(key)}")




# ============================================================
#  DateColumn 类实现
# ============================================================

cdef inline Py_ssize_t _fmt_write_int(char* dst, int v) noexcept nogil:
    """十进制写入（无补零，兼容负号），返回写出的字节数；v == 0 时返回 '0'。"""
    if v == 0:
        dst[0] = 48
        return 1
    cdef bint neg = v < 0
    cdef long t = -<long>v if neg else v
    cdef char tmp[16]
    cdef Py_ssize_t k = 0
    while t > 0:
        tmp[k] = <char>(48 + t % 10)
        k += 1
        t /= 10
    cdef Py_ssize_t w = 0
    if neg:
        dst[0] = 45
        w = 1
    while k > 0:
        dst[w] = tmp[k - 1]
        w += 1
        k -= 1
    return w

cdef inline Py_ssize_t _fmt_int_len(int v) noexcept nogil:
    """十进制表示字节数（含负号），与 _fmt_write_int 输出长度一致。"""
    if v == 0:
        return 1
    cdef bint neg = v < 0
    cdef long t = -<long>v if neg else v
    cdef Py_ssize_t k = 0
    while t > 0:
        k += 1
        t /= 10
    return k + (1 if neg else 0)

cdef inline Py_ssize_t _fmt_expand_len(const char* fmt, Py_ssize_t flen,
                                       int y, int m, int d) noexcept nogil:
    """返回 _fmt_expand 的输出字节数（不写入缓冲区）。"""
    cdef Py_ssize_t p = 0
    cdef Py_ssize_t i = 0
    cdef char c
    while i < flen:
        if fmt[i] == 37 and i + 1 < flen:   # '%'
            c = fmt[i + 1]
            if c == 89:                       # Y
                p += _fmt_int_len(y)
                i += 2
                continue
            elif c == 109:                    # m
                p += 2
                i += 2
                continue
            elif c == 100:                    # d
                p += 2
                i += 2
                continue
        p += 1
        i += 1
    return p

cdef inline Py_ssize_t _fmt_expand(char* dst, const char* fmt, Py_ssize_t flen,
                                   int y, int m, int d) noexcept nogil:
    """单遍展开 strftime 模板：%Y=年（十进制），%m/%d=两位补零，其余字面量原样。"""
    cdef Py_ssize_t p = 0
    cdef Py_ssize_t i = 0
    cdef char c
    while i < flen:
        if fmt[i] == 37 and i + 1 < flen:   # '%'
            c = fmt[i + 1]
            if c == 89:                       # Y
                p += _fmt_write_int(dst + p, y)
                i += 2
                continue
            elif c == 109:                    # m
                dst[p] = <char>(48 + m / 10)
                dst[p + 1] = <char>(48 + m % 10)
                p += 2
                i += 2
                continue
            elif c == 100:                    # d
                dst[p] = <char>(48 + d / 10)
                dst[p + 1] = <char>(48 + d % 10)
                p += 2
                i += 2
                continue
        dst[p] = fmt[i]
        p += 1
        i += 1
    return p

@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef class DateColumn(BaseColumn):

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __cinit__(self, Py_ssize_t init_capacity=16, Py_ssize_t cols=1):
        self.capacity = max(init_capacity, 1)
        self.length = 0
        self.dtype = TYPE_DATE
        self._base_obj = None

        self.num_cols = <int>cols if cols > 0 else 1
        self.is_2d = (self.num_cols > 1)
        self.owns_data = True

        self.data_c = <int32_t*>aligned_malloc(self.capacity * sizeof(int32_t))
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
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _owns_data(self, bint val):
        self.owns_data = val

    @staticmethod
    cdef DateColumn create_view(int32_t* ptr, Py_ssize_t length, object base_obj):
        cdef DateColumn col = DateColumn.__new__(DateColumn)
        col.dtype = TYPE_DATE
        col.capacity = length
        col.length = length
        col.data_c = ptr
        col._base_obj = base_obj
        col.num_cols = 1
        col.is_2d = False
        col.owns_data = False
        return col

    

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def get_size(self):
        return self.capacity * sizeof(int32_t)

    cdef void _reserve(self, Py_ssize_t new_cap):
        if not self.owns_data:
            raise RuntimeError("Cannot reallocate a view/slice column. Copy it first.")
        cdef int32_t* new_ptr
        if new_cap > self.capacity:
            new_ptr = <int32_t*>aligned_realloc(self.data_c, self.capacity * sizeof(int32_t), new_cap * sizeof(int32_t))
            if not new_ptr:
                raise MemoryError()
            self.data_c = new_ptr
            self.capacity = new_cap

    cdef inline int32_t _resolve_date(self, date) except *:
        cdef int32_t v
        if isinstance(date, str):
            return _parse_date_str(<str>date)
        v = <int32_t>date
        # YYYYMMDD integers (> 10000000) need conversion to epoch days
        if v > 10000000:
            return _ymd_to_days(v // 10000, (v // 100) % 100, v % 100)
        return v

    cdef inline int32_t _to_ymd_int(self, int32_t epoch_days) noexcept nogil:
        """Convert epoch days back to YYYYMMDD integer for backward compatibility."""
        cdef int y, m, d
        _days_to_ymd(epoch_days, &y, &m, &d)
        return <int32_t>(y * 10000 + m * 100 + d)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def append(self, val):
        if not self.owns_data:
            raise RuntimeError("Cannot append to a view/slice column. Copy it first.")
        if self.is_2d:
            raise NotImplementedError("append is not supported for 2D columns (rows*cols layout); convert to 1D first")
        cdef int32_t v
        if isinstance(val, str):
            v = _parse_date_str(<str>val)
        else:
            v = <int32_t>val
            if v > 10000000:
                v = _ymd_to_days(v // 10000, (v // 100) % 100, v % 100)

        if self.length >= self.capacity:
            self._reserve(max(16, self.capacity * 2))
        self.data_c[self.length] = v
        self.length += 1

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def copy(self):
        cdef DateColumn new_col = DateColumn(max(self.length, 1), cols=self.num_cols)
        new_col.length = self.length
        _col_memcpy(new_col.data_c, self.data_c, self.length)
        return new_col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def ffill(self):
        """向前填充缺失值。DateColumn 无缺失表示（isna 恒为 False），
        语义与 pandas 中无缺失的日期列一致：各子列内容保持不变（返回拷贝）。"""
        return self.copy()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def bfill(self):
        """向后填充缺失值。语义同 ffill()：日期列无缺失哨兵，返回拷贝。"""
        return self.copy()

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
            vals = [None] * c
            for j in range(c):
                vals[j] = self._to_ymd_int(self.data_c[j * r + (r - 1)])
            _col_shift_left_2d(self.data_c, &self.length, r, c, r - 1, sizeof(int32_t))
            return vals
        cdef int32_t* ptr = <int32_t*>_col_pop_back(self.data_c, &self.length, sizeof(int32_t))
        return self._to_ymd_int(ptr[0])

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
            vals = [None] * c
            for j in range(c):
                vals[j] = self._to_ymd_int(self.data_c[j * r])
            _col_shift_left_2d(self.data_c, &self.length, r, c, 0, sizeof(int32_t))
            return vals

        cdef int32_t val = self.data_c[0]
        _col_pop_front(self.data_c, &self.length, sizeof(int32_t))
        return self._to_ymd_int(val)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def concat(self, DateColumn other):
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
            _col_concat_2d(self.data_c, &self.length, other.data_c, other_rows, c, sizeof(int32_t))
            self.num_cols = <int>c
            self.is_2d = True
            return
        if other.length == 0: return
        self._reserve(self.length + other.length)
        _col_concat(self.data_c, &self.length, other.data_c, other.length,
                     sizeof(int32_t))

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def max(self):
        if self.length == 0: raise ValueError("empty column")
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()
        if not self.is_2d or c <= 1:
            return self._to_ymd_int(_n_max[int32_t](self.data_c, self.length))

        cdef Py_ssize_t j
        cdef int32_t mv
        cdef list results = [None] * c
        for j in range(c):
            mv = _n_max[int32_t](self.data_c + j * r, r)
            results[j] = self._to_ymd_int(mv)
        return results

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def min(self):
        if self.length == 0: raise ValueError("empty column")
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()
        if not self.is_2d or c <= 1:
            return self._to_ymd_int(_n_min[int32_t](self.data_c, self.length))

        cdef Py_ssize_t j
        cdef int32_t mv
        cdef list results = [None] * c
        for j in range(c):
            mv = _n_min[int32_t](self.data_c + j * r, r)
            results[j] = self._to_ymd_int(mv)
        return results

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __getitem__(self, Py_ssize_t idx):
        if idx < 0: idx += self.length
        if idx < 0 or idx >= self.length: raise IndexError("Index out of bounds")
        return self._to_ymd_int((<int32_t*>self.data_c)[idx])

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def unique(self):
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c()
        cdef set seen = set()
        cdef list result = []
        cdef list results
        cdef Py_ssize_t i, n = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        if self.is_2d and c > 1:
            results = [None] * c
            for j in range(c):
                seen = set()
                result = []
                for i in range(r):
                    val = self._to_ymd_int(ptr[j * r + i])
                    if val not in seen:
                        seen.add(val)
                        result.append(val)
                results[j] = result
            return results
        for i in range(n):
            val = self._to_ymd_int(ptr[i])
            if val not in seen:
                seen.add(val)
                result.append(val)
        return result

    # ------------------------------------------------------------
    # 年/月/日/季度 提取
    # ------------------------------------------------------------
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def year(self):
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        with nogil:
            for i in range(n):
                _days_to_ymd(ptr[i], &y, &m, &d)
                res_ptr[i] = y
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def month(self):
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        with nogil:
            for i in range(n):
                _days_to_ymd(ptr[i], &y, &m, &d)
                res_ptr[i] = m
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def day(self):
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        with nogil:
            for i in range(n):
                _days_to_ymd(ptr[i], &y, &m, &d)
                res_ptr[i] = d
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def quarter(self):
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        with nogil:
            for i in range(n):
                _days_to_ymd(ptr[i], &y, &m, &d)
                res_ptr[i] = (m - 1) / 3 + 1
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def strftime(self, str fmt):
        cdef Py_ssize_t i, n = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int y, m, d
        cdef bytes b_fmt = fmt.encode('ascii', errors='ignore')
        cdef const char* f_ptr = b_fmt
        cdef Py_ssize_t f_len = len(b_fmt)

        if n == 0:
            return AsciiColumn(0)

        cdef Py_ssize_t* lens = <Py_ssize_t*>aligned_malloc(n * sizeof(Py_ssize_t))
        if lens == NULL:
            raise MemoryError()
        cdef Py_ssize_t total = 0, need_bytes = 0, acc = 0
        cdef AsciiColumn res
        try:
            with nogil:
                for i in range(n):
                    _days_to_ymd(ptr[i], &y, &m, &d)
                    lens[i] = _fmt_expand_len(f_ptr, f_len, y, m, d)
                    total += lens[i]

            res = AsciiColumn(n)
            res.length = n
            need_bytes = total + n
            if need_bytes > res.byte_capacity:
                res._reserve_bytes(max(res.byte_capacity * 2, need_bytes))
            with nogil:
                for i in range(n):
                    res.offsets_c[i] = acc
                    _days_to_ymd(ptr[i], &y, &m, &d)
                    _fmt_expand(res.data_c + acc, f_ptr, f_len, y, m, d)
                    res.data_c[acc + lens[i]] = 0
                    acc += lens[i] + 1
            res.offsets_c[n] = acc
            res.byte_length = acc
            return res
        finally:
            os_aligned_free(lens)

    # ------------------------------------------------------------
    # 过滤算子 (纯 32 位整数比较，100% AVX2/AVX-512 SIMD 向量化)
    # ------------------------------------------------------------
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def filter_greater(self, date):
        cdef int32_t target = self._resolve_date(date)
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = 1 if ptr[i] > target else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def filter_greater_equal(self, date):
        cdef int32_t target = self._resolve_date(date)
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = 1 if ptr[i] >= target else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def filter_less(self, date):
        cdef int32_t target = self._resolve_date(date)
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = 1 if ptr[i] < target else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def filter_less_equal(self, date):
        cdef int32_t target = self._resolve_date(date)
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = 1 if ptr[i] <= target else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def filter_equal(self, date):
        cdef int32_t target = self._resolve_date(date)
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = 1 if ptr[i] == target else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def filter_not_equal(self, date):
        cdef int32_t target = self._resolve_date(date)
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = 1 if ptr[i] != target else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def filter_between(self, start, end):
        cdef int32_t lo = self._resolve_date(start)
        cdef int32_t hi = self._resolve_date(end)
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = 1 if (ptr[i] >= lo and ptr[i] <= hi) else 0
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def where_mask(self, IntColumn mask, object fill):
        cdef Py_ssize_t n = self.length
        cdef Py_ssize_t r = self._get_rows_c()
        cdef Py_ssize_t ccols = self.num_cols
        cdef DateColumn col
        cdef Py_ssize_t i, j, base
        cdef int32_t fill_val
        cdef int32_t* col_ptr
        cdef int32_t* self_ptr = <int32_t*>self.data_c
        cdef int* mask_ptr = mask.data_c

        if self.is_2d and ccols > 1:
            if mask.length != r:
                raise ValueError(f"Mask length {mask.length} must match column rows {r}")
            col = DateColumn(max(n, 1), cols=ccols)
            col.length = n
            col_ptr = <int32_t*>col.data_c
            fill_val = self._resolve_date(fill)
            with nogil:
                for j in range(ccols):
                    base = j * r
                    for i in range(r):
                        col_ptr[base + i] = self_ptr[base + i] if mask_ptr[i] != 0 else fill_val
            return col

        if mask.length != n:
            raise ValueError(f"Mask length {mask.length} != column length {n}")
        col = DateColumn(max(n, 1), cols=self.cols)
        col.length = n
        if n == 0: return col

        fill_val = self._resolve_date(fill)
        with nogil:
            _n_where_mask(<int32_t*>col.data_c, <int32_t*>self.data_c,
                          mask.data_c, fill_val, n)
        return col

    # ------------------------------------------------------------
    # 加减天数：退化为极致的单指令加法（彻底消灭历法转换）
    # ------------------------------------------------------------
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def add_days(self, int n):
        cdef Py_ssize_t length = self.length
        cdef DateColumn res = DateColumn(max(length, 1), cols=self.num_cols)
        res.length = length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int32_t* res_ptr = <int32_t*>res.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(length):
                res_ptr[i] = ptr[i] + n
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sub_days(self, int n):
        return self.add_days(-n)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def add_months(self, int n):
        cdef Py_ssize_t length = self.length
        cdef DateColumn res = DateColumn(max(length, 1), cols=self.num_cols)
        res.length = length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int32_t* res_ptr = <int32_t*>res.data_c
        cdef Py_ssize_t i
        cdef int y, m, d, total, new_m, new_y, max_d
        with nogil:
            for i in range(length):
                _days_to_ymd(ptr[i], &y, &m, &d)
                total = (m - 1) + n
                new_y = y + total / 12
                new_m = total % 12 + 1
                if new_m <= 0:
                    new_m += 12
                    new_y -= 1
                max_d = _days_in_month(new_y, new_m)
                if d > max_d:
                    d = max_d
                res_ptr[i] = _ymd_to_days(new_y, new_m, d)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sub_months(self, int n):
        return self.add_months(-n)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def add_years(self, int n):
        cdef Py_ssize_t length = self.length
        cdef DateColumn res = DateColumn(max(length, 1), cols=self.num_cols)
        res.length = length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int32_t* res_ptr = <int32_t*>res.data_c
        cdef Py_ssize_t i
        cdef int y, m, d, new_y, max_d
        with nogil:
            for i in range(length):
                _days_to_ymd(ptr[i], &y, &m, &d)
                new_y = y + n
                max_d = _days_in_month(new_y, m)
                if d > max_d:
                    d = max_d
                res_ptr[i] = _ymd_to_days(new_y, m, d)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sub_years(self, int n):
        return self.add_years(-n)

    # ------------------------------------------------------------
    # 星期与历法判定 (1970-01-01 是 Thursday = 3)
    # ------------------------------------------------------------
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def day_of_week(self):
        """返回 0=Monday .. 6=Sunday"""
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = <int>(((ptr[i] + 3) % 7 + 7) % 7)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def is_weekend(self):
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int dow
        with nogil:
            for i in range(n):
                dow = <int>(((ptr[i] + 3) % 7 + 7) % 7)
                res_ptr[i] = 1 if (dow == 5 or dow == 6) else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def day_of_year(self):
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d, doy
        with nogil:
            for i in range(n):
                _days_to_ymd(ptr[i], &y, &m, &d)
                doy = _DOY_CUMULATIVE_NORMAL[m] + d
                if m > 2 and _is_leap_year(y):
                    doy += 1
                res_ptr[i] = doy
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def days_in_month(self):
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        with nogil:
            for i in range(n):
                _days_to_ymd(ptr[i], &y, &m, &d)
                res_ptr[i] = _days_in_month(y, m)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def is_leap_year(self):
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        with nogil:
            for i in range(n):
                _days_to_ymd(ptr[i], &y, &m, &d)
                res_ptr[i] = 1 if _is_leap_year(y) else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def is_month_start(self):
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        with nogil:
            for i in range(n):
                _days_to_ymd(ptr[i], &y, &m, &d)
                res_ptr[i] = 1 if d == 1 else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def is_month_end(self):
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        with nogil:
            for i in range(n):
                _days_to_ymd(ptr[i], &y, &m, &d)
                res_ptr[i] = 1 if d == _days_in_month(y, m) else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def is_year_start(self):
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        with nogil:
            for i in range(n):
                _days_to_ymd(ptr[i], &y, &m, &d)
                res_ptr[i] = 1 if (m == 1 and d == 1) else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def is_year_end(self):
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        with nogil:
            for i in range(n):
                _days_to_ymd(ptr[i], &y, &m, &d)
                res_ptr[i] = 1 if (m == 12 and d == 31) else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def is_quarter_start(self):
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        with nogil:
            for i in range(n):
                _days_to_ymd(ptr[i], &y, &m, &d)
                res_ptr[i] = 1 if ((m == 1 or m == 4 or m == 7 or m == 10) and d == 1) else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def is_quarter_end(self):
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        with nogil:
            for i in range(n):
                _days_to_ymd(ptr[i], &y, &m, &d)
                if m == 3 or m == 6 or m == 9 or m == 12:
                    res_ptr[i] = 1 if d == _days_in_month(y, m) else 0
                else:
                    res_ptr[i] = 0
        return res

    cdef inline int _iso_week_of_year(self, int32_t days, int* iso_y) noexcept nogil:
        cdef int y, m, d
        _days_to_ymd(days, &y, &m, &d)
        cdef int32_t d4jan = _ymd_to_days(y, 1, 4)
        cdef int dow4jan = <int>(((d4jan + 3) % 7 + 7) % 7)
        cdef int32_t week1_start = d4jan - dow4jan

        if days < week1_start:
            y -= 1
            d4jan = _ymd_to_days(y, 1, 4)
            dow4jan = <int>(((d4jan + 3) % 7 + 7) % 7)
            week1_start = d4jan - dow4jan
        else:
            d4jan = _ymd_to_days(y + 1, 1, 4)
            dow4jan = <int>(((d4jan + 3) % 7 + 7) % 7)
            if days >= d4jan - dow4jan:
                y += 1
                week1_start = d4jan - dow4jan

        if iso_y != NULL:
            iso_y[0] = y
        return <int>((days - week1_start) / 7) + 1

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def week_of_year(self):
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = self._iso_week_of_year(ptr[i], NULL)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def day_name(self):
        cdef AsciiColumn res = AsciiColumn(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int dow

        DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for i in range(n):
            dow = <int>(((ptr[i] + 3) % 7 + 7) % 7)
            res.append(DAY_NAMES[dow])
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def month_name(self):
        cdef AsciiColumn res = AsciiColumn(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int y, m, d

        MONTH_NAMES = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
        for i in range(n):
            _days_to_ymd(ptr[i], &y, &m, &d)
            if 1 <= m <= 12:
                res.append(MONTH_NAMES[m])
            else:
                res.append('')
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def last_day_of_month(self):
        cdef DateColumn res = DateColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int32_t* res_ptr = <int32_t*>res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d, max_d
        with nogil:
            for i in range(n):
                _days_to_ymd(ptr[i], &y, &m, &d)
                max_d = _days_in_month(y, m)
                res_ptr[i] = _ymd_to_days(y, m, max_d)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def iso_calendar(self):
        cdef IntColumn res_y = IntColumn(max(self.length, 1), cols=self.num_cols)
        cdef IntColumn res_w = IntColumn(max(self.length, 1), cols=self.num_cols)
        cdef IntColumn res_d = IntColumn(max(self.length, 1), cols=self.num_cols)
        res_y.length = self.length
        res_w.length = self.length
        res_d.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* ry_ptr = res_y.data_c
        cdef int* rw_ptr = res_w.data_c
        cdef int* rd_ptr = res_d.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int iso_y
        with nogil:
            for i in range(n):
                rw_ptr[i] = self._iso_week_of_year(ptr[i], &iso_y)
                ry_ptr[i] = iso_y
                rd_ptr[i] = <int>(((ptr[i] + 3) % 7 + 7) % 7) + 1
        return (res_y, res_w, res_d)

    # ------------------------------------------------------------
    # 日期差值计算 (diff_days 直接减法，SIMD 向量化)
    # ------------------------------------------------------------
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def diff_days(self, date):
        cdef int32_t target = self._resolve_date(date)
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = <int>(ptr[i] - target)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def diff_months(self, date):
        cdef int32_t target = self._resolve_date(date)
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y1, m1, d1, y2, m2, d2
        _days_to_ymd(target, &y2, &m2, &d2)
        with nogil:
            for i in range(n):
                _days_to_ymd(ptr[i], &y1, &m1, &d1)
                res_ptr[i] = (y1 - y2) * 12 + (m1 - m2) - (1 if d1 < d2 else 0)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def diff_years(self, date):
        cdef int32_t target = self._resolve_date(date)
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y1, m1, d1, y2, m2, d2
        _days_to_ymd(target, &y2, &m2, &d2)
        with nogil:
            for i in range(n):
                _days_to_ymd(ptr[i], &y1, &m1, &d1)
                res_ptr[i] = y1 - y2 - (1 if (m1 < m2 or (m1 == m2 and d1 < d2)) else 0)
        return res

    # ------------------------------------------------------------
    # 纯 C 32 位整数比较算子 (完全向量化)
    # ------------------------------------------------------------
    cdef IntColumn _cmp_op(self, val, int op):
        cdef Py_ssize_t n = self.length
        cdef IntColumn res = IntColumn(max(n, 1), cols=self.num_cols)
        res.length = n

        cdef char err_buf[128]
        cdef int32_t v
        cdef DateColumn other

        if isinstance(val, DateColumn):
            other = <DateColumn>val
            if other.length != n:
                snprintf(err_buf, sizeof(err_buf), "Length mismatch: %zd vs %zd", n, other.length)
                raise ValueError(err_buf.decode('ascii'))
            if other.num_cols != self.num_cols:
                raise ValueError(
                    f"Shape mismatch: self has {self.num_cols} column(s), other has {other.num_cols}")
            with nogil:
                _n_cmp_col(<int32_t*>self.data_c, <int32_t*>other.data_c,
                           res.data_c, n, op)
        else:
            v = self._resolve_date(val)
            with nogil:
                _n_cmp_scalar(<int32_t*>self.data_c, res.data_c, v, n, op)
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __lt__(self, val):  return self._cmp_op(val, 0)
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __le__(self, val):  return self._cmp_op(val, 1)
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __gt__(self, val):  return self._cmp_op(val, 2)
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __ge__(self, val):  return self._cmp_op(val, 3)
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __eq__(self, val):  return self._cmp_op(val, 4)
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __ne__(self, val):  return self._cmp_op(val, 5)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __str__(self):
        cdef list items = []
        cdef Py_ssize_t i, n = self.length
        cdef int32_t* ptr = <int32_t*>self.data_c
        cdef int y, m, d
        cdef char str_buf[16]
        for i in range(n):
            _days_to_ymd(ptr[i], &y, &m, &d)
            snprintf(str_buf, sizeof(str_buf), "%04d-%02d-%02d", y, m, d)
            items.append(str_buf.decode('ascii'))
        return f"DateColumn([{', '.join(items)}])"

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __bool__(self):
        return self.length > 0

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __len__(self):
        return self.length

    cdef DateColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=False):
        cdef DateColumn new_col
        cdef int32_t* new_data
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()

        if self.is_2d:
            if not inplace:
                new_col = DateColumn(max(n * c, 1), cols=c)
                new_col.length = n * c
                _col_gather_2d_zero(new_col.data_c, self.data_c, indices, n, r, c, sizeof(int32_t))
                return new_col

            new_data = <int32_t*>aligned_malloc(max(n * c, 1) * sizeof(int32_t))
            if not new_data: raise MemoryError()
            _col_gather_2d_zero(new_data, self.data_c, indices, n, r, c, sizeof(int32_t))
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
            new_col = DateColumn(max(n, 1), cols=self.num_cols)
            new_col.length = n
            _col_gather_zero(new_col.data_c, self.data_c, indices, n, sizeof(int32_t))
            return new_col

        new_data = <int32_t*>aligned_malloc(max(n, 1) * sizeof(int32_t))
        if not new_data: raise MemoryError()
        _col_gather_zero(new_data, self.data_c, indices, n, sizeof(int32_t))
        if self.owns_data and self.data_c != NULL:
            os_aligned_free(self.data_c)
        self.data_c = new_data
        self.capacity = n
        self.length = n
        self._base_obj = None
        return self

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
            _col_shift_left_2d(self.data_c, &self.length, r, c, row_idx, sizeof(int32_t))
        else:
            _col_drop_row(self.data_c, &self.length, row_idx, sizeof(int32_t))

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
            _col_shift_left_2d(self.data_c, &self.length, r, c, pos, sizeof(int32_t))
        else:
            col_len = self.length
            _c_shift_left(self.data_c, pos, col_len, sizeof(int32_t))
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
            _col_shift_right_2d_int(self.data_c, &self.length, r, c, pos, sizeof(int32_t))
        else:
            if self.length >= self.capacity:
                self._reserve(max(16, self.capacity * 2))
            _c_shift_right(self.data_c, pos, self.length, sizeof(int32_t))
            self.length += 1
            (<int32_t*>self.data_c)[pos] = 0


    # ------------------------------------------------------------
    # 类型转换 astype (无 Python 堆分配直写)
    # ------------------------------------------------------------
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def astype(self, object target):
        cdef int target_type = _resolve_astype_target(target)
        cdef Py_ssize_t i, n = self.length
        cdef Py_ssize_t c = self.cols
        cdef bint is_2d_src = self.is_2d and c > 1
        cdef bint target_supports_2d = target_type in (TYPE_INT, TYPE_DOUBLE, TYPE_DATE, TYPE_DATETIME, TYPE_TIME)
        if is_2d_src and not target_supports_2d:
            raise NotImplementedError(
                f"Cannot astype 2D column ({c} sub-columns) to non-2D type {target}. "
                f"Supported 2D targets: int, float, date, datetime, time.")
        cdef Py_ssize_t first_col_n = self.rows if c > 1 else n
        cdef Py_ssize_t safe_cap = max(n, 1)
        cdef Py_ssize_t first_col_cap = max(first_col_n, 1)
        cdef int y, m, d
        cdef char str_buf[16]

        cdef IntColumn ic
        cdef DoubleColumn dc
        cdef UCS2Column uc
        cdef AsciiColumn cc
        cdef CategoryColumn catc
        cdef DateTimeColumn dtc
        cdef TimeColumn tmc
        cdef int32_t* s_ptr = <int32_t*>self.data_c

        if target_type == TYPE_INT:
            ic = IntColumn(safe_cap, cols=self.cols)
            ic.length = n
            with nogil:
                for i in range(n):
                    _days_to_ymd(s_ptr[i], &y, &m, &d)
                    ic.data_c[i] = y * 10000 + m * 100 + d
            return ic
        elif target_type == TYPE_DOUBLE:
            dc = DoubleColumn(safe_cap, cols=self.cols)
            dc.length = n
            with nogil:
                for i in range(n):
                    _days_to_ymd(s_ptr[i], &y, &m, &d)
                    dc.data_c[i] = <double>(y * 10000 + m * 100 + d)
            return dc
        elif target_type == TYPE_ASCII:
            cc = AsciiColumn(first_col_cap)
            cc.length = first_col_n
            with nogil:
                for i in range(first_col_n):
                    _days_to_ymd(s_ptr[i], &y, &m, &d)
                    cc.data_c[i * 256 + 0] = <char>(48 + ((y / 1000) % 10))
                    cc.data_c[i * 256 + 1] = <char>(48 + ((y / 100) % 10))
                    cc.data_c[i * 256 + 2] = <char>(48 + ((y / 10) % 10))
                    cc.data_c[i * 256 + 3] = <char>(48 + (y % 10))
                    cc.data_c[i * 256 + 4] = b'-'
                    cc.data_c[i * 256 + 5] = <char>(48 + (m / 10))
                    cc.data_c[i * 256 + 6] = <char>(48 + (m % 10))
                    cc.data_c[i * 256 + 7] = b'-'
                    cc.data_c[i * 256 + 8] = <char>(48 + (d / 10))
                    cc.data_c[i * 256 + 9] = <char>(48 + (d % 10))
                    cc.data_c[i * 256 + 10] = 0
            return cc
        elif target_type == TYPE_UCS2:
            uc = UCS2Column(first_col_cap)
            for i in range(first_col_n):
                _days_to_ymd(s_ptr[i], &y, &m, &d)
                snprintf(str_buf, sizeof(str_buf), "%04d-%02d-%02d", y, m, d)
                uc.append(str_buf.decode('ascii'))
            return uc
        elif target_type == TYPE_CATEGORY:
            catc = CategoryColumn(first_col_cap)
            for i in range(first_col_n):
                _days_to_ymd(s_ptr[i], &y, &m, &d)
                snprintf(str_buf, sizeof(str_buf), "%04d-%02d-%02d", y, m, d)
                catc.append(str_buf.decode('ascii'))
            return catc
        elif target_type == TYPE_DATETIME:
            dtc = DateTimeColumn(safe_cap, cols=self.cols)
            dtc.length = n
            with nogil:
                for i in range(n):
                    (<int64_t*>dtc.data_c)[i] = <int64_t>s_ptr[i] * 86400000000LL
            return dtc
        elif target_type == TYPE_TIME:
            tmc = TimeColumn(safe_cap, cols=self.cols)
            tmc.length = n
            with nogil:
                for i in range(n):
                    tmc.data_c[i] = 0
            return tmc
        elif target_type == TYPE_DATE:
            return self.copy()

        raise TypeError(f"Failed to cast TIMESTAMP column to internal type id: {target_type}")



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef class BoolColumn(BaseColumn):
    @cython.always_allow_keywords(False)
    def __cinit__(self, Py_ssize_t capacity=0):
        self.dtype = TYPE_BOOL
        self.capacity = capacity if capacity > 0 else 0
        self.length = 0
        self._owns_data = True
        self._base_obj = None
        if self.capacity > 0:
            self.data_c = <unsigned char*>aligned_malloc(self.capacity * sizeof(unsigned char))
            if not self.data_c:
                raise MemoryError()
            memset(self.data_c, 0, self.capacity)
        else:
            self.data_c = NULL

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __dealloc__(self):
        if self._owns_data and self.data_c != NULL:
            os_aligned_free(self.data_c)
            self.data_c = NULL

    @staticmethod
    cdef BoolColumn create_view(unsigned char* ptr, Py_ssize_t length, object base_obj):
        cdef BoolColumn col = BoolColumn.__new__(BoolColumn)
        col.dtype = TYPE_BOOL
        col.capacity = length
        col.length = length
        col.data_c = ptr
        col._owns_data = False
        col._base_obj = base_obj
        return col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def get_size(self):
        return self.capacity * sizeof(unsigned char)

    cdef void _reserve(self, Py_ssize_t new_cap):
        if new_cap <= self.capacity:
            return
        cdef unsigned char* new_data = <unsigned char*>aligned_realloc(
            self.data_c, self.capacity * sizeof(unsigned char), new_cap * sizeof(unsigned char))
        if not new_data:
            raise MemoryError()
        memset(new_data + self.capacity, 0, new_cap - self.capacity)
        self.data_c = new_data
        self.capacity = new_cap

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def append(self, object val):
        if self.length >= self.capacity:
            self._reserve(max(16, self.capacity * 2))
        self.data_c[self.length] = 1 if val else 0
        self.length += 1

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def _append_bool(self, bint val):
        self.append(val)

    cdef void append_nolen(self, object val) except *:
        """写入当前行(length)位置的值，但不推进 length（由 add_row 统一提交）。"""
        if self.length >= self.capacity:
            raise BufferError("BoolColumn row capacity (%d) reached." % self.capacity)
        self.data_c[self.length] = 1 if val else 0

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def pop_back(self):
        if not self._owns_data:
            raise RuntimeError("Cannot modify a view/slice column. Copy it first.")
        if self.length == 0:
            raise IndexError("pop from empty column")
        cdef unsigned char* ptr = <unsigned char*>_col_pop_back(
            self.data_c, &self.length, sizeof(unsigned char))
        return bool(ptr[0])

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def pop_front(self):
        if not self._owns_data:
            raise RuntimeError("Cannot modify a view/slice column. Copy it first.")
        if self.length == 0:
            raise IndexError("pop from empty column")
        cdef unsigned char val = self.data_c[0]
        _col_pop_front(self.data_c, &self.length, sizeof(unsigned char))
        return bool(val)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def concat(self, BoolColumn other):
        if not self._owns_data:
            raise RuntimeError("Cannot concat into a view/slice column. Copy it first.")
        self._reserve(self.length + other.length)
        _col_concat(self.data_c, &self.length, other.data_c, other.length,
                     sizeof(unsigned char))

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def copy(self):
        cdef Py_ssize_t cap = max(self.length, 1)
        cdef BoolColumn col = BoolColumn(cap)
        col.length = self.length
        if self.length > 0:
            memcpy(col.data_c, self.data_c, self.length * sizeof(unsigned char))
        return col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def ffill(self):
        """向前填充缺失值。BoolColumn 无缺失表示（isna 恒为 False），返回拷贝。"""
        return self.copy()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def bfill(self):
        """向后填充缺失值。语义同 ffill()：布尔列无缺失哨兵，返回拷贝。"""
        return self.copy()

    

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __getitem__(self, Py_ssize_t idx):
        if idx < 0: idx += self.length
        if idx < 0 or idx >= self.length: raise IndexError("Index out of bounds")
        return bool(self.data_c[idx])

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __setitem__(self, Py_ssize_t idx, object val):
        if idx < 0: idx += self.length
        if idx < 0 or idx >= self.length: raise IndexError("Index out of bounds")
        self.data_c[idx] = 1 if val else 0

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __len__(self):
        return self.length

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __bool__(self):
        return self.length > 0

    # ------------------------------------------------------------
    # 极致优化 isin：从 O(N) Python Set 查找优化为 O(1) 分支 + 纯 C 向量化
    # ------------------------------------------------------------
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def isin(self, set values):
        cdef Py_ssize_t n = self.length
        cdef BoolColumn res = BoolColumn(max(n, 1))
        res.length = n
        if n == 0:
            return res

        cdef bint has_true = True in values or 1 in values
        cdef bint has_false = False in values or 0 in values
        cdef unsigned char* self_ptr = self.data_c
        cdef unsigned char* res_ptr = res.data_c
        cdef Py_ssize_t i

        if has_true and has_false:
            memset(res_ptr, 1, n)
        elif not has_true and not has_false:
            memset(res_ptr, 0, n)
        elif has_true:
            with nogil:
                for i in range(n):
                    res_ptr[i] = 1 if self_ptr[i] != 0 else 0
        else:
            with nogil:
                for i in range(n):
                    res_ptr[i] = 1 if self_ptr[i] == 0 else 0
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def where_mask(self, IntColumn mask, object fill):
        cdef Py_ssize_t n = self.length
        if mask.length != n:
            raise ValueError(f"Mask length {mask.length} != column length {n}")
        cdef BoolColumn col = BoolColumn(max(n, 1))
        col.length = n
        if n == 0:
            return col

        cdef unsigned char fill_val = 1 if fill else 0
        cdef unsigned char* self_ptr = self.data_c
        cdef unsigned char* col_ptr = col.data_c
        cdef int* mask_ptr = mask.data_c
        cdef Py_ssize_t i

        with nogil:
            for i in range(n):
                col_ptr[i] = self_ptr[i] if mask_ptr[i] != 0 else fill_val
        return col

    # ------------------------------------------------------------
    # 逻辑操作：直接纯 C 循环，编译器会自动生成 256/512-bit AVX 指令
    # 彻底杜绝手写 64-bit 指针在非对齐地址上的 SIGBUS 隐患
    # ------------------------------------------------------------
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __and__(self, BoolColumn other):
        cdef Py_ssize_t n = self.length
        cdef char err_buf[128]
        if other.length != n:
            snprintf(err_buf, sizeof(err_buf), "Column length %zd != %zd", other.length, n)
            raise ValueError(err_buf.decode('ascii'))

        cdef BoolColumn res = BoolColumn(max(n, 1))
        res.length = n
        with nogil:
            _bool_bitwise(self.data_c, other.data_c, res.data_c, n, 0)
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __or__(self, BoolColumn other):
        cdef Py_ssize_t n = self.length
        cdef char err_buf[128]
        if other.length != n:
            snprintf(err_buf, sizeof(err_buf), "Column length %zd != %zd", other.length, n)
            raise ValueError(err_buf.decode('ascii'))

        cdef BoolColumn res = BoolColumn(max(n, 1))
        res.length = n
        with nogil:
            _bool_bitwise(self.data_c, other.data_c, res.data_c, n, 1)
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __xor__(self, BoolColumn other):
        cdef Py_ssize_t n = self.length
        cdef char err_buf[128]
        if other.length != n:
            snprintf(err_buf, sizeof(err_buf), "Column length %zd != %zd", other.length, n)
            raise ValueError(err_buf.decode('ascii'))

        cdef BoolColumn res = BoolColumn(max(n, 1))
        res.length = n
        with nogil:
            _bool_bitwise(self.data_c, other.data_c, res.data_c, n, 2)
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __invert__(self):
        cdef Py_ssize_t n = self.length
        cdef BoolColumn res = BoolColumn(max(n, 1))
        res.length = n
        with nogil:
            _bool_not(self.data_c, res.data_c, n)
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __str__(self):
        cdef list items = []
        cdef Py_ssize_t i
        for i in range(self.length):
            items.append(BOOL_STRS[1 if self.data_c[i] != 0 else 0])
        return f"BoolColumn([{', '.join(items)}])"

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __repr__(self):
        return self.__str__()

    cdef BoolColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=False):
        cdef BoolColumn new_col
        cdef unsigned char* new_data

        if not inplace:
            new_col = BoolColumn(max(n, 1))
            new_col.length = n
            _col_gather_zero(new_col.data_c, self.data_c, indices, n, sizeof(unsigned char))
            return new_col
        else:
            new_data = <unsigned char*>aligned_malloc(max(n, 1) * sizeof(unsigned char))
            if not new_data:
                raise MemoryError()
            _col_gather_zero(new_data, self.data_c, indices, n, sizeof(unsigned char))
            if self._owns_data and self.data_c != NULL:
                os_aligned_free(self.data_c)
            self.data_c = new_data
            self.capacity = n
            self.length = n
            self._owns_data = True
            self._base_obj = None
            return self

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def drop_row(self, Py_ssize_t row_idx):
        if not self._owns_data:
            raise RuntimeError("Cannot drop row on a view/slice column. Copy it first.")
        if row_idx < 0:
            row_idx += self.length
        if row_idx < 0 or row_idx >= self.length:
            raise IndexError("Index out of bounds")
        _col_drop_row(self.data_c, &self.length, row_idx, sizeof(unsigned char))

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def shift_left(self, Py_ssize_t pos):
        cdef Py_ssize_t col_len = self.length
        _c_shift_left(self.data_c, pos, col_len, sizeof(unsigned char))
        self.length = col_len - 1

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __eq__(self, other):
        if not isinstance(other, BoolColumn):
            return NotImplemented
        cdef BoolColumn o = <BoolColumn>other
        cdef char err_buf[128]
        if o.length != self.length:
            snprintf(err_buf, sizeof(err_buf), "Column length %zd != %zd", o.length, self.length)
            raise ValueError(err_buf.decode('ascii'))
        if self.length == 0:
            return True
        return memcmp(self.data_c, o.data_c, self.length * sizeof(unsigned char)) == 0

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def astype(self, object target):
        cdef int target_type = _resolve_astype_target(target)
        cdef Py_ssize_t i, n = self.length
        cdef Py_ssize_t safe_cap = max(n, 1)
        cdef unsigned char* ptr = self.data_c
        cdef IntColumn ic
        cdef DoubleColumn dc
        cdef UCS2Column uc
        cdef AsciiColumn cc
        cdef CategoryColumn catc
        cdef DateTimeColumn dtc
        cdef TimeColumn tmc
        cdef str s_val

        if target_type == TYPE_INT:
            ic = IntColumn(safe_cap)
            ic.length = n
            with nogil:
                for i in range(n):
                    ic.data_c[i] = <int>(ptr[i] != 0)
            return ic
        elif target_type == TYPE_DOUBLE:
            dc = DoubleColumn(safe_cap)
            dc.length = n
            with nogil:
                for i in range(n):
                    dc.data_c[i] = <double>(ptr[i] != 0)
            return dc
        elif target_type == TYPE_UCS2:
            uc = UCS2Column(safe_cap)
            for i in range(n):
                s_val = BOOL_STRS[1 if ptr[i] != 0 else 0]
                uc.append(s_val)
            return uc
        elif target_type == TYPE_ASCII:
            cc = AsciiColumn(safe_cap)
            for i in range(n):
                s_val = BOOL_STRS[1 if ptr[i] != 0 else 0]
                cc.append(s_val)
            return cc
        elif target_type == TYPE_CATEGORY:
            catc = CategoryColumn(safe_cap)
            for i in range(n):
                s_val = BOOL_STRS[1 if ptr[i] != 0 else 0]
                catc.append(s_val)
            return catc
        elif target_type == TYPE_DATETIME:
            dtc = DateTimeColumn(safe_cap)
            for i in range(n):
                s_val = BOOL_STRS[1 if ptr[i] != 0 else 0]
                dtc.append(s_val)
            return dtc
        elif target_type == TYPE_TIME:
            tmc = TimeColumn(safe_cap)
            for i in range(n):
                tmc.append(<int>(ptr[i] != 0))
            return tmc
        elif target_type == TYPE_BOOL:
            return self.copy()
        raise TypeError(f"Failed to cast BOOL column to internal type id: {target_type}")

    cdef void gather_zero_to(self, Py_ssize_t* out):
        if out == NULL: return
        cdef Py_ssize_t j = 0
        cdef unsigned char* ptr = self.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                if ptr[i] == 0:
                    out[j] = i
                    j += 1

    cdef Py_ssize_t count_nonzero(self):
        cdef Py_ssize_t count = 0
        cdef unsigned char* ptr = self.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                if ptr[i] != 0:
                    count += 1
        return count

@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef class CategoryColumn(BaseColumn):
    def __cinit__(self, Py_ssize_t capacity=0):
        self.dtype = TYPE_CATEGORY
        self.capacity = capacity if capacity > 0 else 0
        self.length = 0
        self._owns_data = True
        self._base_obj = None
        self.categories = []
        self._cat_to_code = {}
        if self.capacity > 0:
            self.codes_c = <int*>aligned_malloc(self.capacity * sizeof(int))
            if not self.codes_c:
                raise MemoryError()
        else:
            self.codes_c = NULL

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __dealloc__(self):
        if self._owns_data and self.codes_c != NULL:
            os_aligned_free(self.codes_c)
            self.codes_c = NULL

    @staticmethod
    cdef CategoryColumn create_view(int* ptr, Py_ssize_t length, list categories, dict cat_to_code, object base_obj):
        cdef CategoryColumn col = CategoryColumn.__new__(CategoryColumn)
        col.dtype = TYPE_CATEGORY
        col.capacity = length
        col.length = length
        col.codes_c = ptr
        col._owns_data = False
        col._base_obj = base_obj
        col.categories = categories
        col._cat_to_code = cat_to_code
        return col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def get_size(self):
        return self.capacity * sizeof(int) + len(self.categories) * sizeof(void*)

    cdef void _reserve(self, Py_ssize_t new_cap):
        if new_cap <= self.capacity:
            return
        cdef int* new_data = <int*>aligned_realloc(
            self.codes_c, self.capacity * sizeof(int), new_cap * sizeof(int))
        if not new_data:
            raise MemoryError()
        self.codes_c = new_data
        self.capacity = new_cap

    cdef int _get_or_add_code(self, str val):
        cdef int code
        cdef object existing = self._cat_to_code.get(val)
        if existing is not None:
            return <int>existing
        code = len(self.categories)
        self.categories.append(val)
        self._cat_to_code[val] = code
        return code

    cdef inline str _resolve(self, Py_ssize_t i):
        cdef int code = self.codes_c[i]
        if code >= 0 and code < len(self.categories):
            return self.categories[code]
        return ''

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def append(self, object val):
        if self.length >= self.capacity:
            self._reserve(max(16, self.capacity * 2))
        cdef str s = str(val) if val is not None else ''
        self.codes_c[self.length] = self._get_or_add_code(s)
        self.length += 1

    cdef void append_nolen(self, object val) except *:
        """写入当前行(length)位置的值，但不推进 length（由 add_row 统一提交）。"""
        if self.length >= self.capacity:
            raise BufferError("CategoryColumn row capacity (%d) reached." % self.capacity)
        cdef str s = str(val) if val is not None else ''
        self.codes_c[self.length] = self._get_or_add_code(s)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def pop_back(self):
        if not self._owns_data:
            raise RuntimeError("Cannot modify a view/slice column. Copy it first.")
        if self.length == 0:
            raise IndexError("pop from empty column")
        self.length -= 1
        return self._resolve(self.length)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def pop_front(self):
        if not self._owns_data:
            raise RuntimeError("Cannot modify a view/slice column. Copy it first.")
        if self.length == 0:
            raise IndexError("pop from empty column")
        cdef str val = self._resolve(0)
        if self.length > 1:
            memmove(self.codes_c, self.codes_c + 1, (self.length - 1) * sizeof(int))
        self.length -= 1
        return val

    # ------------------------------------------------------------
    # concat 优化：预映射 O(K_other) 字典，然后 O(N) 纯 C 写入
    # ------------------------------------------------------------
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def concat(self, CategoryColumn other):
        if not self._owns_data:
            raise RuntimeError("Cannot concat into a view/slice column. Copy it first.")
        cdef Py_ssize_t other_len = other.length
        if other_len == 0:
            return

        cdef Py_ssize_t new_len = self.length + other_len
        self._reserve(new_len)

        cdef Py_ssize_t num_other_cats = len(other.categories)
        cdef int* code_map = <int*>aligned_malloc(max(num_other_cats, 1) * sizeof(int))
        if not code_map: raise MemoryError()

        cdef Py_ssize_t j, i
        cdef int* self_codes = self.codes_c + self.length
        cdef int* other_codes = other.codes_c
        try:
            for j in range(num_other_cats):
                code_map[j] = self._get_or_add_code(other.categories[j])

            with nogil:
                for i in range(other_len):
                    self_codes[i] = code_map[other_codes[i]]
            self.length = new_len
        finally:
            os_aligned_free(code_map)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def copy(self):
        cdef CategoryColumn col = CategoryColumn(max(self.length, 1))
        col.length = self.length
        col.categories = list(self.categories)
        col._cat_to_code = dict(self._cat_to_code)
        if self.length > 0:
            memcpy(col.codes_c, self.codes_c, self.length * sizeof(int))
        return col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def ffill(self):
        """向前填充空类别（本库缺失语义：isna 视空类别为缺失）。
        在拷贝上原地改写 code 映射，O(n)。"""
        cdef CategoryColumn col = self.copy()
        cdef Py_ssize_t n = col.length
        if n == 0:
            return col
        cdef int* codes = col.codes_c
        cdef list cats = col.categories
        cdef Py_ssize_t cat_n = len(cats)
        cdef Py_ssize_t i
        cdef int c
        cdef int last_code = -1
        cdef bint has_last = False
        for i in range(n):
            c = codes[i]
            if c < 0 or c >= cat_n or cats[c] == '':
                if has_last:
                    codes[i] = last_code
            else:
                last_code = c
                has_last = True
        return col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def bfill(self):
        """向后填充空类别（缺失语义）。在拷贝上原地改写 code 映射，O(n)。"""
        cdef CategoryColumn col = self.copy()
        cdef Py_ssize_t n = col.length
        if n == 0:
            return col
        cdef int* codes = col.codes_c
        cdef list cats = col.categories
        cdef Py_ssize_t cat_n = len(cats)
        cdef Py_ssize_t i
        cdef int c
        cdef int next_code = -1
        cdef bint has_next = False
        for i in range(n - 1, -1, -1):
            c = codes[i]
            if c < 0 or c >= cat_n or cats[c] == '':
                if has_next:
                    codes[i] = next_code
            else:
                next_code = c
                has_next = True
        return col

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __getitem__(self, Py_ssize_t idx):
        if idx < 0: idx += self.length
        if idx < 0 or idx >= self.length: raise IndexError("Index out of bounds")
        return self._resolve(idx)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __setitem__(self, Py_ssize_t idx, object val):
        if idx < 0: idx += self.length
        if idx < 0 or idx >= self.length: raise IndexError("Index out of bounds")
        cdef str s = str(val) if val is not None else ''
        self.codes_c[idx] = self._get_or_add_code(s)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __len__(self):
        return self.length

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __bool__(self):
        return self.length > 0

    # ------------------------------------------------------------
    # isin 优化：O(K) 预计算命中标记表，O(N) 纯 C 查表，消除全部内层 Python 查找
    # ------------------------------------------------------------
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def isin(self, set values):
        cdef Py_ssize_t n = self.length
        cdef BoolColumn res = BoolColumn(max(n, 1))
        res.length = n
        if n == 0:
            return res

        cdef Py_ssize_t num_cats = len(self.categories)
        cdef unsigned char* match_table = <unsigned char*>malloc(max(num_cats, 1) * sizeof(unsigned char))
        if not match_table: raise MemoryError()

        cdef Py_ssize_t k, i
        cdef int* codes = self.codes_c
        cdef unsigned char* res_ptr = res.data_c
        try:
            for k in range(num_cats):
                match_table[k] = 1 if self.categories[k] in values else 0

            with nogil:
                for i in range(n):
                    res_ptr[i] = match_table[codes[i]]
        finally:
            free(match_table)
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def where_mask(self, IntColumn mask, object fill):
        cdef Py_ssize_t n = self.length
        if mask.length != n:
            raise ValueError(f"Mask length {mask.length} != column length {n}")
        cdef CategoryColumn col = CategoryColumn(max(n, 1))
        col.length = n
        if n == 0:
            return col

        cdef str fill_str = str(fill) if fill is not None else ''
        col.categories = list(self.categories)
        col._cat_to_code = dict(self._cat_to_code)
        cdef int fill_code = col._get_or_add_code(fill_str)

        cdef int* mask_ptr = mask.data_c
        cdef int* self_codes = self.codes_c
        cdef int* col_codes = col.codes_c
        cdef Py_ssize_t i

        with nogil:
            for i in range(n):
                col_codes[i] = self_codes[i] if mask_ptr[i] != 0 else fill_code
        return col

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __str__(self):
        cdef list items = []
        cdef Py_ssize_t i
        for i in range(self.length):
            items.append(repr(self._resolve(i)))
        return f"CategoryColumn([{', '.join(items)}])"

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __repr__(self):
        return self.__str__()

    cdef CategoryColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=False):
        cdef Py_ssize_t i
        cdef int* old_codes = self.codes_c
        cdef int* new_codes
        cdef CategoryColumn new_col

        if not inplace:
            new_col = CategoryColumn(max(n, 1))
            new_col.length = n
            new_col.categories = list(self.categories)
            new_col._cat_to_code = dict(self._cat_to_code)
            with nogil:
                for i in range(n):
                    new_col.codes_c[i] = old_codes[indices[i]] if indices[i] >= 0 else 0
            return new_col
        else:
            new_codes = <int*>aligned_malloc(max(n, 1) * sizeof(int))
            if not new_codes: raise MemoryError()
            with nogil:
                for i in range(n):
                    new_codes[i] = old_codes[indices[i]] if indices[i] >= 0 else 0
            if self._owns_data and self.codes_c != NULL:
                os_aligned_free(self.codes_c)
            self.codes_c = new_codes
            self.capacity = n
            self.length = n
            self._owns_data = True
            self._base_obj = None
            return self

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def drop_row(self, Py_ssize_t row_idx):
        if not self._owns_data:
            raise RuntimeError("Cannot drop row on a view/slice column. Copy it first.")
        if row_idx < 0:
            row_idx += self.length
        cdef Py_ssize_t rem_len = self.length - row_idx - 1
        if rem_len > 0:
            memmove(self.codes_c + row_idx,
                    self.codes_c + row_idx + 1,
                    rem_len * sizeof(int))
        self.length -= 1

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def shift_left(self, Py_ssize_t pos):
        cdef Py_ssize_t col_len = self.length
        _c_shift_left(self.codes_c, pos, col_len, sizeof(int))
        self.length = col_len - 1

    # ------------------------------------------------------------
    # 优化：若 categories 相同直接 memcmp；不同则构建映射比对
    # ------------------------------------------------------------
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __eq__(self, other):
        if not isinstance(other, CategoryColumn):
            return NotImplemented
        cdef CategoryColumn o = <CategoryColumn>other
        if o.length != self.length:
            return False
        if self.length == 0:
            return True

        # 类别列表完全一致时，直接极速内存比对
        if self.categories == o.categories:
            return memcmp(self.codes_c, o.codes_c, self.length * sizeof(int)) == 0

        # 类别列表顺序不同时，先映射 other 的编码
        cdef Py_ssize_t num_o_cats = len(o.categories)
        cdef int* o_to_self = <int*>malloc(max(num_o_cats, 1) * sizeof(int))
        if not o_to_self: raise MemoryError()

        cdef Py_ssize_t j, i
        cdef bint match = True
        cdef int* s_c = self.codes_c
        cdef int* o_c = o.codes_c
        cdef Py_ssize_t n = self.length
        try:
            for j in range(num_o_cats):
                cat = o.categories[j]
                if cat in self._cat_to_code:
                    o_to_self[j] = self._cat_to_code[cat]
                else:
                    o_to_self[j] = -1

            with nogil:
                for i in range(n):
                    if s_c[i] != o_to_self[o_c[i]]:
                        match = False
                        break
        finally:
            free(o_to_self)
        return match

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented: return result
        return not result

    # ------------------------------------------------------------
    # astype 优化：预先转换 K 个 Category 的值（Memoization），消除 N 次解析
    # ------------------------------------------------------------
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def astype(self, object target):
        cdef int target_type = _resolve_astype_target(target)
        cdef Py_ssize_t i, k, n = self.length
        cdef Py_ssize_t num_cats = len(self.categories)
        cdef Py_ssize_t safe_cap = max(n, 1)
        cdef int* codes = self.codes_c
        cdef str s

        cdef IntColumn ic
        cdef DoubleColumn dc
        cdef BoolColumn bc
        cdef UCS2Column uc
        cdef AsciiColumn cc
        cdef DateColumn tc
        cdef DateTimeColumn dtc
        cdef TimeColumn tmc

        # 整数预处理转换
        cdef int* int_lookup
        if target_type == TYPE_INT:
            ic = IntColumn(safe_cap)
            ic.length = n
            if n == 0: return ic

            int_lookup = <int*>malloc(max(num_cats, 1) * sizeof(int))
            if not int_lookup: raise MemoryError()
            try:
                for k in range(num_cats):
                    s = self.categories[k]
                    int_lookup[k] = int(float(s)) if s else 0
                with nogil:
                    for i in range(n):
                        ic.data_c[i] = int_lookup[codes[i]]
            finally:
                free(int_lookup)
            return ic

        # 浮点预处理转换
        cdef double* double_lookup
        if target_type == TYPE_DOUBLE:
            dc = DoubleColumn(safe_cap)
            dc.length = n
            if n == 0: return dc

            double_lookup = <double*>malloc(max(num_cats, 1) * sizeof(double))
            if not double_lookup: raise MemoryError()
            try:
                for k in range(num_cats):
                    s = self.categories[k]
                    double_lookup[k] = float(s) if s else 0.0
                with nogil:
                    for i in range(n):
                        dc.data_c[i] = double_lookup[codes[i]]
            finally:
                free(double_lookup)
            return dc

        # 布尔预处理转换
        cdef unsigned char* bool_lookup
        if target_type == TYPE_BOOL:
            bc = BoolColumn(safe_cap)
            bc.length = n
            if n == 0: return bc

            bool_lookup = <unsigned char*>malloc(max(num_cats, 1) * sizeof(unsigned char))
            if not bool_lookup: raise MemoryError()
            try:
                for k in range(num_cats):
                    s = self.categories[k].lower()
                    bool_lookup[k] = 1 if s in ('true', '1', 'yes') else 0
                with nogil:
                    for i in range(n):
                        bc.data_c[i] = bool_lookup[codes[i]]
            finally:
                free(bool_lookup)
            return bc

        # 字符串/复杂类型转换
        if target_type == TYPE_UCS2:
            uc = UCS2Column(safe_cap)
            for i in range(n):
                uc.append(self._resolve(i))
            return uc
        elif target_type == TYPE_ASCII:
            cc = AsciiColumn(safe_cap)
            for i in range(n):
                cc.append(self._resolve(i))
            return cc
        elif target_type == TYPE_DATE:
            tc = DateColumn(safe_cap)
            for i in range(n):
                s = self._resolve(i)
                tc.append(s if s else 0)
            return tc
        elif target_type == TYPE_DATETIME:
            dtc = DateTimeColumn(safe_cap)
            for i in range(n):
                s = self._resolve(i)
                dtc.append(s if s else '')
            return dtc
        elif target_type == TYPE_TIME:
            tmc = TimeColumn(safe_cap)
            for i in range(n):
                s = self._resolve(i)
                tmc.append(s if s else 0)
            return tmc
        elif target_type == TYPE_CATEGORY:
            return self.copy()

        raise TypeError(f"Failed to cast CATEGORY column to internal type id: {target_type}")



@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef class DateTimeColumn(BaseColumn):

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __cinit__(self, Py_ssize_t init_capacity=16, Py_ssize_t cols=1):
        self.capacity = max(init_capacity, 1)
        self.length = 0
        self.dtype = TYPE_DATETIME
        self._base_obj = None

        self.num_cols = <int>cols if cols > 0 else 1
        self.is_2d = (self.num_cols > 1)
        self.owns_data = True

        self.data_c = <int64_t*>aligned_malloc(self.capacity * sizeof(int64_t))
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
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _owns_data(self, bint val):
        self.owns_data = val

    @staticmethod
    cdef DateTimeColumn create_view(int64_t* ptr, Py_ssize_t length, object base_obj):
        cdef DateTimeColumn col = DateTimeColumn.__new__(DateTimeColumn)
        col.dtype = TYPE_DATETIME
        col.capacity = length
        col.length = length
        col.data_c = ptr
        col._base_obj = base_obj
        col.num_cols = 1
        col.is_2d = False
        col.owns_data = False
        return col

    

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def get_size(self):
        return self.capacity * sizeof(int64_t)

    cdef void _reserve(self, Py_ssize_t new_cap):
        if not self.owns_data:
            raise RuntimeError("Cannot reallocate a view/slice column. Copy it first.")
        cdef int64_t* new_ptr
        if new_cap > self.capacity:
            new_ptr = <int64_t*>aligned_realloc(self.data_c, self.capacity * sizeof(int64_t), new_cap * sizeof(int64_t))
            if not new_ptr:
                raise MemoryError()
            self.data_c = new_ptr
            self.capacity = new_cap

    cdef inline int64_t _resolve_dt(self, val) except *:
        if isinstance(val, str):
            return _parse_datetime_str(<str>val)
        return <int64_t>val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def append(self, val):
        if not self.owns_data:
            raise RuntimeError("Cannot append to a view/slice column. Copy it first.")
        if self.is_2d:
            raise NotImplementedError("append is not supported for 2D columns (rows*cols layout); convert to 1D first")
        cdef int64_t v = _parse_datetime_str(<str>val) if isinstance(val, str) else <int64_t>val
        if self.length >= self.capacity:
            self._reserve(max(16, self.capacity * 2))
        self.data_c[self.length] = v
        self.length += 1

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def copy(self):
        cdef DateTimeColumn new_col = DateTimeColumn(max(self.length, 1), cols=self.cols)
        new_col.length = self.length
        _col_memcpy(new_col.data_c, self.data_c, self.length)
        return new_col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def ffill(self):
        """向前填充缺失值。DateTimeColumn 无缺失表示（isna 恒为 False），
        语义与 pandas 中无缺失的 datetime 列一致：各子列内容保持不变（返回拷贝）。"""
        return self.copy()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def bfill(self):
        """向后填充缺失值。语义同 ffill()：datetime 列无缺失哨兵，返回拷贝。"""
        return self.copy()

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
            vals = [None] * c
            for j in range(c):
                vals[j] = self.data_c[j * r + (r - 1)]
            _col_shift_left_2d(self.data_c, &self.length, r, c, r - 1, sizeof(int64_t))
            return vals
        cdef int64_t* ptr = <int64_t*>_col_pop_back(
            self.data_c, &self.length, sizeof(int64_t))
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
            vals = [None] * c
            for j in range(c):
                vals[j] = self.data_c[j * r]
            _col_shift_left_2d(self.data_c, &self.length, r, c, 0, sizeof(int64_t))
            return vals
        cdef int64_t val = (<int64_t*>self.data_c)[0]
        _col_pop_front(self.data_c, &self.length, sizeof(int64_t))
        return val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def concat(self, DateTimeColumn other):
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
            _col_concat_2d(self.data_c, &self.length, other.data_c, other_rows, c, sizeof(int64_t))
            self.num_cols = <int>c
            self.is_2d = True
            return
        if other.length == 0: return
        self._reserve(self.length + other.length)
        _col_concat(self.data_c, &self.length, other.data_c, other.length,
                     sizeof(int64_t))

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def max(self):
        if self.length == 0: raise ValueError("empty column")
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()
        if not self.is_2d or c <= 1:
            return _n_max[int64_t](<int64_t*>self.data_c, self.length)

        cdef Py_ssize_t j
        cdef int64_t mv
        cdef list results = [None] * c
        for j in range(c):
            mv = _n_max[int64_t](<int64_t*>self.data_c + j * r, r)
            results[j] = mv
        return results

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def min(self):
        if self.length == 0: raise ValueError("empty column")
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()
        if not self.is_2d or c <= 1:
            return _n_min[int64_t](<int64_t*>self.data_c, self.length)

        cdef Py_ssize_t j
        cdef int64_t mv
        cdef list results = [None] * c
        for j in range(c):
            mv = _n_min[int64_t](<int64_t*>self.data_c + j * r, r)
            results[j] = mv
        return results

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __getitem__(self, Py_ssize_t idx):
        if idx < 0: idx += self.length
        if idx < 0 or idx >= self.length: raise IndexError("Index out of bounds")
        return _datetime_to_str((<int64_t*>self.data_c)[idx])

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def unique(self):
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c()
        cdef set seen = set()
        cdef list result = []
        cdef list results
        cdef int64_t* ptr = <int64_t*>self.data_c
        cdef Py_ssize_t i, j, base
        cdef int64_t val
        if self.is_2d and c > 1:
            results = [None] * c
            for j in range(c):
                seen = set()
                result = []
                base = j * r
                for i in range(r):
                    val = ptr[base + i]
                    if val not in seen:
                        seen.add(val)
                        result.append(val)
                results[j] = result
            return results
        seen = set()
        result = []
        for i in range(self.length):
            val = ptr[i]
            if val not in seen:
                seen.add(val)
                result.append(val)
        return result

    # ------------------------------------------------------------
    # 字段提取 (时分秒直接算术提取，避免冗余的 YMD 历法计算)
    # ------------------------------------------------------------
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def year(self):
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int64_t* ptr = <int64_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        cdef int32_t days
        with nogil:
            for i in range(n):
                days = <int32_t>(ptr[i] / US_PER_DAY)
                _days_to_ymd(days, &y, &m, &d)
                res_ptr[i] = y
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def month(self):
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int64_t* ptr = <int64_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        cdef int32_t days
        with nogil:
            for i in range(n):
                days = <int32_t>(ptr[i] / US_PER_DAY)
                _days_to_ymd(days, &y, &m, &d)
                res_ptr[i] = m
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def day(self):
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int64_t* ptr = <int64_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int y, m, d
        cdef int32_t days
        with nogil:
            for i in range(n):
                days = <int32_t>(ptr[i] / US_PER_DAY)
                _days_to_ymd(days, &y, &m, &d)
                res_ptr[i] = d
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def hour(self):
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int64_t* ptr = <int64_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int64_t secs
        cdef int rem
        with nogil:
            for i in range(n):
                secs = ptr[i] / US_PER_SEC
                rem = <int>(secs % SEC_PER_DAY)
                if rem < 0: rem += SEC_PER_DAY
                res_ptr[i] = rem / 3600
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def minute(self):
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int64_t* ptr = <int64_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int64_t secs
        cdef int rem
        with nogil:
            for i in range(n):
                secs = ptr[i] / US_PER_SEC
                rem = <int>(secs % SEC_PER_DAY)
                if rem < 0: rem += SEC_PER_DAY
                res_ptr[i] = (rem / 60) % 60
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def second(self):
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int64_t* ptr = <int64_t*>self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i, n = self.length
        cdef int64_t secs
        cdef int rem
        with nogil:
            for i in range(n):
                secs = ptr[i] / US_PER_SEC
                rem = <int>(secs % SEC_PER_DAY)
                if rem < 0: rem += SEC_PER_DAY
                res_ptr[i] = rem % 60
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def where_mask(self, IntColumn mask, object fill):
        cdef Py_ssize_t n = self.length
        cdef Py_ssize_t r = self._get_rows_c()
        cdef Py_ssize_t ccols = self.num_cols
        cdef DateTimeColumn col
        cdef Py_ssize_t i, j, base
        cdef int64_t fill_val
        cdef int64_t* col_ptr
        cdef int64_t* self_ptr = <int64_t*>self.data_c
        cdef int* mask_ptr = mask.data_c

        if self.is_2d and ccols > 1:
            if mask.length != r:
                raise ValueError(f"Mask length {mask.length} must match column rows {r}")
            col = DateTimeColumn(max(n, 1), cols=ccols)
            col.length = n
            col_ptr = <int64_t*>col.data_c
            fill_val = self._resolve_dt(fill) if fill is not None else 0
            with nogil:
                for j in range(ccols):
                    base = j * r
                    for i in range(r):
                        col_ptr[base + i] = self_ptr[base + i] if mask_ptr[i] != 0 else fill_val
            return col

        if mask.length != n:
            raise ValueError(f"Mask length {mask.length} != column length {n}")

        col = DateTimeColumn(max(n, 1), cols=self.cols)
        col.length = n
        if n == 0: return col

        fill_val = self._resolve_dt(fill) if fill is not None else 0
        with nogil:
            _n_where_mask(<int64_t*>col.data_c, <int64_t*>self.data_c,
                          mask.data_c, fill_val, n)
        return col

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def isin(self, set values):
        cdef Py_ssize_t n = self.length
        cdef BoolColumn res = BoolColumn(max(n, 1))
        res.length = n
        if n == 0: return res

        cdef Py_ssize_t m = len(values)
        cdef unsigned char* r_ptr = res.data_c
        if m == 0:
            memset(r_ptr, 0, n)
            return res

        cdef int64_t* v_arr = <int64_t*>aligned_malloc(m * sizeof(int64_t))
        if not v_arr: raise MemoryError()

        cdef Py_ssize_t j = 0
        for v in values:
            if v is not None:
                v_arr[j] = self._resolve_dt(v)
                j += 1

        cdef Py_ssize_t valid_m = j
        with nogil:
            _n_isin(<int64_t*>self.data_c, v_arr, n, valid_m, <unsigned char*>r_ptr)

        os_aligned_free(v_arr)
        return res

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
            raise IndexError("Index out of bounds")
        if self.is_2d:
            _col_shift_left_2d(self.data_c, &self.length, r, c, row_idx, sizeof(int64_t))
        else:
            _col_drop_row(self.data_c, &self.length, row_idx, sizeof(int64_t))

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
            _col_shift_left_2d(self.data_c, &self.length, r, c, pos, sizeof(int64_t))
        else:
            col_len = self.length
            _c_shift_left(self.data_c, pos, col_len, sizeof(int64_t))
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
            _col_shift_right_2d_int(self.data_c, &self.length, r, c, pos, sizeof(int64_t))
        else:
            if self.length >= self.capacity:
                self._reserve(max(16, self.capacity * 2))
            _c_shift_right(self.data_c, pos, self.length, sizeof(int64_t))
            self.length += 1
            (<int64_t*>self.data_c)[pos] = 0

    cdef DateTimeColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=False):
        cdef DateTimeColumn new_col
        cdef int64_t* new_data
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()

        if self.is_2d:
            if not inplace:
                new_col = DateTimeColumn(max(n * c, 1), cols=c)
                new_col.length = n * c
                _col_gather_2d_zero(new_col.data_c, self.data_c, indices, n, r, c, sizeof(int64_t))
                return new_col
            else:
                new_data = <int64_t*>aligned_malloc(max(n * c, 1) * sizeof(int64_t))
                if not new_data: raise MemoryError()
                _col_gather_2d_zero(new_data, self.data_c, indices, n, r, c, sizeof(int64_t))
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
            new_col = DateTimeColumn(max(n, 1), cols=self.num_cols)
            new_col.length = n
            _col_gather_zero(new_col.data_c, self.data_c, indices, n, sizeof(int64_t))
            return new_col
        else:
            new_data = <int64_t*>aligned_malloc(max(n, 1) * sizeof(int64_t))
            if not new_data: raise MemoryError()
            _col_gather_zero(new_data, self.data_c, indices, n, sizeof(int64_t))
            if self.owns_data and self.data_c != NULL:
                os_aligned_free(self.data_c)
            self.data_c = new_data
            self.capacity = n
            self.length = n
            self._base_obj = None
            return self

    # ------------------------------------------------------------
    # 纯 C 64 位整数比较算子 (完全向量化)
    # ------------------------------------------------------------
    cdef IntColumn _cmp_op(self, object val, int op):
        cdef Py_ssize_t n = self.length
        cdef IntColumn res = IntColumn(max(n, 1), cols=self.num_cols)
        res.length = n

        cdef char err_buf[128]
        cdef int64_t v
        cdef DateTimeColumn other

        if isinstance(val, DateTimeColumn):
            other = <DateTimeColumn>val
            if other.length != n:
                snprintf(err_buf, sizeof(err_buf), "Length mismatch: %zd vs %zd", n, other.length)
                raise ValueError(err_buf.decode('ascii'))
            if other.num_cols != self.num_cols:
                raise ValueError(
                    f"Shape mismatch: self has {self.num_cols} column(s), other has {other.num_cols}")
            with nogil:
                _n_cmp_col(<int64_t*>self.data_c, <int64_t*>other.data_c,
                           res.data_c, n, op)
        else:
            v = self._resolve_dt(val)
            with nogil:
                _n_cmp_scalar(<int64_t*>self.data_c, res.data_c, v, n, op)
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __lt__(self, val):  return self._cmp_op(val, 0)
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __le__(self, val):  return self._cmp_op(val, 1)
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __gt__(self, val):  return self._cmp_op(val, 2)
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __ge__(self, val):  return self._cmp_op(val, 3)
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __eq__(self, val):  return self._cmp_op(val, 4)
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __ne__(self, val):  return self._cmp_op(val, 5)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __str__(self):
        cdef list items = []
        cdef Py_ssize_t i, n = self.length
        cdef int64_t* ptr = <int64_t*>self.data_c
        for i in range(n):
            items.append(_datetime_to_str(ptr[i]))
        return f"DateTimeColumn([{', '.join(items)}])"

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __bool__(self):
        return self.length > 0

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __len__(self):
        return self.length

    # ------------------------------------------------------------
    # astype: 纯 C 单指令转换与截断
    # ------------------------------------------------------------
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def astype(self, object target):
        cdef int target_type = _resolve_astype_target(target)
        cdef Py_ssize_t i, n = self.length
        cdef Py_ssize_t c = self.cols
        cdef bint is_2d_src = self.is_2d and c > 1
        cdef bint target_supports_2d = target_type in (TYPE_INT, TYPE_DOUBLE, TYPE_DATE, TYPE_DATETIME, TYPE_TIME)
        if is_2d_src and not target_supports_2d:
            raise NotImplementedError(
                f"Cannot astype 2D column ({c} sub-columns) to non-2D type {target}. "
                f"Supported 2D targets: int, float, date, datetime, time.")
        cdef Py_ssize_t first_col_n = self.rows if c > 1 else n
        cdef Py_ssize_t safe_cap = max(n, 1)
        cdef Py_ssize_t first_col_cap = max(first_col_n, 1)
        cdef int64_t* ptr = <int64_t*>self.data_c
        cdef char str_buf[32]

        cdef IntColumn ic
        cdef DoubleColumn dc
        cdef UCS2Column uc
        cdef AsciiColumn cc
        cdef CategoryColumn catc
        cdef DateColumn tc
        cdef TimeColumn tmc
        cdef int32_t* tc_ptr
        cdef int32_t* tmc_ptr
        cdef int64_t secs
        cdef int rem

        if target_type == TYPE_INT:
            ic = IntColumn(safe_cap, cols=self.cols)
            ic.length = n
            with nogil:
                for i in range(n):
                    ic.data_c[i] = <int>(ptr[i] / US_PER_SEC)
            return ic

        elif target_type == TYPE_DOUBLE:
            dc = DoubleColumn(safe_cap, cols=self.cols)
            dc.length = n
            with nogil:
                for i in range(n):
                    dc.data_c[i] = <double>ptr[i] / 1000000.0
            return dc

        # ------------------------------------------------------------
        # DateTime -> Date (TIMESTAMP)：极致单指令截断，无任何历法开销
        # ------------------------------------------------------------
        elif target_type == TYPE_DATE:
            tc = DateColumn(safe_cap, cols=self.cols)
            tc.length = n
            tc_ptr = <int32_t*>tc.data_c
            with nogil:
                for i in range(n):
                    tc_ptr[i] = <int32_t>(ptr[i] / US_PER_DAY)
            return tc

        # ------------------------------------------------------------
        # DateTime -> Time：秒数取模，单指令提取
        # ------------------------------------------------------------
        elif target_type == TYPE_TIME:
            tmc = TimeColumn(safe_cap, cols=self.cols)
            tmc.length = n
            tmc_ptr = tmc.data_c
            with nogil:
                for i in range(n):
                    secs = ptr[i] / US_PER_SEC
                    rem = <int>(secs % SEC_PER_DAY)
                    tmc_ptr[i] = rem + SEC_PER_DAY if rem < 0 else rem
            return tmc

        elif target_type == TYPE_ASCII:
            cc = AsciiColumn(first_col_cap)
            cc.length = first_col_n
            with nogil:
                for i in range(first_col_n):
                    _format_datetime_c(ptr[i], cc.data_c + (i * 256))
            return cc

        elif target_type == TYPE_UCS2:
            uc = UCS2Column(first_col_cap)
            for i in range(first_col_n):
                _format_datetime_c(ptr[i], str_buf)
                uc.append(str_buf.decode('ascii'))
            return uc

        elif target_type == TYPE_CATEGORY:
            catc = CategoryColumn(first_col_cap)
            for i in range(first_col_n):
                _format_datetime_c(ptr[i], str_buf)
                catc.append(str_buf.decode('ascii'))
            return catc

        elif target_type == TYPE_DATETIME:
            return self.copy()

        cdef char err_buf[128]
        snprintf(err_buf, sizeof(err_buf), "Failed to cast DATETIME column to internal type id: %d", target_type)
        raise TypeError(err_buf.decode('ascii'))




# ============================================================
#  TimeColumn 类实现 (全面 int32_t 与 SIMD 向量化优化)
# ============================================================
@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef class TimeColumn(BaseColumn):

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __cinit__(self, Py_ssize_t init_capacity=16, Py_ssize_t cols=1):
        self.capacity = max(init_capacity, 1)
        self.length = 0
        self.dtype = TYPE_TIME
        self._base_obj = None

        self.num_cols = <int>cols if cols > 0 else 1
        self.is_2d = (self.num_cols > 1)
        self.owns_data = True

        self.data_c = <int32_t*>aligned_malloc(self.capacity * sizeof(int32_t))
        if not self.data_c:
            raise MemoryError()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def get_size(self):
        return self.capacity * sizeof(int32_t)

    cdef void* _get_data_ptr(self) noexcept nogil:
        return <void*>self.data_c

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
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _owns_data(self, bint val):
        self.owns_data = val

    @staticmethod
    cdef TimeColumn create_view(int32_t* ptr, Py_ssize_t length, object base_obj):
        cdef TimeColumn col = TimeColumn.__new__(TimeColumn)
        col.dtype = TYPE_TIME
        col.capacity = length
        col.length = length
        col.data_c = ptr
        col._base_obj = base_obj
        col.num_cols = 1
        col.is_2d = False
        col.owns_data = False
        return col


    cdef void _reserve(self, Py_ssize_t new_cap):
        if not self.owns_data:
            raise RuntimeError("Cannot reallocate a view/slice column. Copy it first.")
        cdef int32_t* new_ptr
        if new_cap > self.capacity:
            new_ptr = <int32_t*>aligned_realloc(self.data_c, self.capacity * sizeof(int32_t), new_cap * sizeof(int32_t))
            if not new_ptr:
                raise MemoryError()
            self.data_c = new_ptr
            self.capacity = new_cap

    cdef inline int32_t _resolve_time(self, val) except -1:
        if isinstance(val, str):
            return _parse_time_str(<str>val)
        return <int32_t>val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def append(self, val):
        if not self.owns_data:
            raise RuntimeError("Cannot append to a view/slice column. Copy it first.")
        if self.is_2d:
            raise NotImplementedError("append is not supported for 2D columns (rows*cols layout); convert to 1D first")
        cdef int32_t v = _parse_time_str(<str>val) if isinstance(val, str) else <int32_t>val
        if self.length >= self.capacity:
            self._reserve(max(16, self.capacity * 2))
        self.data_c[self.length] = v
        self.length += 1

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def copy(self):
        cdef TimeColumn new_col = TimeColumn(max(self.length, 1), cols=self.cols)
        new_col.length = self.length
        _col_memcpy(new_col.data_c, self.data_c, self.length)
        return new_col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def ffill(self):
        """向前填充缺失值。TimeColumn 无缺失表示（isna 恒为 False），
        语义与 pandas 中无缺失的 time 列一致：各子列内容保持不变（返回拷贝）。"""
        return self.copy()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def bfill(self):
        """向后填充缺失值。语义同 ffill()：time 列无缺失哨兵，返回拷贝。"""
        return self.copy()

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
            vals = [None] * c
            for j in range(c):
                vals[j] = self.data_c[j * r + (r - 1)]
            _col_shift_left_2d(self.data_c, &self.length, r, c, r - 1, sizeof(int32_t))
            return vals
        cdef int32_t* ptr = <int32_t*>_col_pop_back(
            self.data_c, &self.length, sizeof(int32_t))
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
            vals = [None] * c
            for j in range(c):
                vals[j] = self.data_c[j * r]
            _col_shift_left_2d(self.data_c, &self.length, r, c, 0, sizeof(int32_t))
            return vals
        cdef int32_t val = self.data_c[0]
        _col_pop_front(self.data_c, &self.length, sizeof(int32_t))
        return val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def concat(self, TimeColumn other):
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
            _col_concat_2d(self.data_c, &self.length, other.data_c, other_rows, c, sizeof(int32_t))
            self.num_cols = <int>c
            self.is_2d = True
            return
        if other.length == 0: return
        self._reserve(self.length + other.length)
        _col_concat(self.data_c, &self.length, other.data_c, other.length, sizeof(int32_t))

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def max(self):
        if self.length == 0: raise ValueError("empty column")
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()
        if not self.is_2d or c <= 1:
            return _n_max[int32_t](self.data_c, self.length)

        cdef Py_ssize_t j
        cdef int32_t mv
        cdef list results = [None] * c
        for j in range(c):
            mv = _n_max[int32_t](self.data_c + j * r, r)
            results[j] = mv
        return results

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def min(self):
        if self.length == 0: raise ValueError("empty column")
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()
        if not self.is_2d or c <= 1:
            return _n_min[int32_t](self.data_c, self.length)

        cdef Py_ssize_t j
        cdef int32_t mv
        cdef list results = [None] * c
        for j in range(c):
            mv = _n_min[int32_t](self.data_c + j * r, r)
            results[j] = mv
        return results

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __getitem__(self, Py_ssize_t idx):
        if idx < 0: idx += self.length
        if idx < 0 or idx >= self.length: raise IndexError("Index out of bounds")
        return _time_to_str(self.data_c[idx])

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def unique(self):
        cdef Py_ssize_t c = self.num_cols, r = self._get_rows_c()
        cdef set seen = set()
        cdef list result = []
        cdef list results
        cdef int32_t* ptr = self.data_c
        cdef Py_ssize_t i, j, base
        cdef int32_t val
        if self.is_2d and c > 1:
            results = [None] * c
            for j in range(c):
                seen = set()
                result = []
                base = j * r
                for i in range(r):
                    val = ptr[base + i]
                    if val not in seen:
                        seen.add(val)
                        result.append(val)
                results[j] = result
            return results
        seen = set()
        result = []
        for i in range(self.length):
            val = ptr[i]
            if val not in seen:
                seen.add(val)
                result.append(val)
        return result

    # ------------------------------------------------------------
    # 时/分/秒提取 (定点乘法展开，完全自动 SIMD 向量化)
    # ------------------------------------------------------------
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def hour(self):
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int32_t* ptr = self.data_c
        cdef int32_t* res_ptr = <int32_t*>res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = ptr[i] / 3600
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def minute(self):
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int32_t* ptr = self.data_c
        cdef int32_t* res_ptr = <int32_t*>res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = (ptr[i] / 60) % 60
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def second(self):
        cdef IntColumn res = IntColumn(max(self.length, 1), cols=self.num_cols)
        res.length = self.length
        cdef int32_t* ptr = self.data_c
        cdef int32_t* res_ptr = <int32_t*>res.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                res_ptr[i] = ptr[i] % 60
        return res

    # ------------------------------------------------------------
    # 无分支向量化掩码选择
    # ------------------------------------------------------------
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def where_mask(self, IntColumn mask, object fill):
        cdef Py_ssize_t n = self.length
        cdef Py_ssize_t r = self._get_rows_c()
        cdef Py_ssize_t ccols = self.num_cols
        cdef TimeColumn col
        cdef Py_ssize_t i, j, base
        cdef int32_t fill_val
        cdef int32_t* col_ptr
        cdef int32_t* self_ptr = self.data_c
        cdef int* mask_ptr = mask.data_c

        if self.is_2d and ccols > 1:
            if mask.length != r:
                raise ValueError(f"Mask length {mask.length} must match column rows {r}")
            col = TimeColumn(max(n, 1), cols=ccols)
            col.length = n
            col_ptr = col.data_c
            fill_val = self._resolve_time(fill) if fill is not None else 0
            with nogil:
                for j in range(ccols):
                    base = j * r
                    for i in range(r):
                        col_ptr[base + i] = self_ptr[base + i] if mask_ptr[i] != 0 else fill_val
            return col

        if mask.length != n:
            raise ValueError(f"Mask length {mask.length} != column length {n}")
        col = TimeColumn(max(n, 1), cols=self.cols)
        col.length = n
        if n == 0: return col

        fill_val = self._resolve_time(fill) if fill is not None else 0
        with nogil:
            _n_where_mask(col.data_c, self.data_c, mask.data_c, fill_val, n)
        return col

    # ------------------------------------------------------------
    # isin 极速优化：释放 GIL + C 级别二分检索（零 Python 装箱）
    # ------------------------------------------------------------
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def isin(self, object values):
        cdef set parsed_vals = set()
        for v in values:
            if isinstance(v, str):
                parsed_vals.add(_parse_time_str(<str>v))
            elif v is not None:
                parsed_vals.add(<int32_t>v)

        cdef Py_ssize_t num_keys = len(parsed_vals)
        cdef Py_ssize_t n = self.length
        cdef BoolColumn res = BoolColumn(max(n, 1))
        res.length = n
        if n == 0:
            return res

        if num_keys == 0:
            memset(res.data_c, 0, n)
            return res

        cdef int32_t* keys_c = <int32_t*>aligned_malloc(num_keys * sizeof(int32_t))
        if not keys_c: raise MemoryError()

        cdef Py_ssize_t ki = 0
        for val in parsed_vals:
            keys_c[ki] = <int32_t>val
            ki += 1

        with nogil:
            _n_isin(self.data_c, keys_c, n, num_keys, res.data_c)

        os_aligned_free(keys_c)
        return res

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
            raise IndexError("Index out of bounds")
        if self.is_2d:
            _col_shift_left_2d(self.data_c, &self.length, r, c, row_idx, sizeof(int32_t))
        else:
            _col_drop_row(self.data_c, &self.length, row_idx, sizeof(int32_t))

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
            _col_shift_left_2d(self.data_c, &self.length, r, c, pos, sizeof(int32_t))
        else:
            col_len = self.length
            _c_shift_left(self.data_c, pos, col_len, sizeof(int32_t))
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
            _col_shift_right_2d_int(self.data_c, &self.length, r, c, pos, sizeof(int32_t))
        else:
            if self.length >= self.capacity:
                self._reserve(max(16, self.capacity * 2))
            _c_shift_right(self.data_c, pos, self.length, sizeof(int32_t))
            self.length += 1
            (<int32_t*>self.data_c)[pos] = 0

    cdef TimeColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=False):
        cdef TimeColumn new_col
        cdef int32_t* new_data
        cdef Py_ssize_t c = self.num_cols
        cdef Py_ssize_t r = self._get_rows_c()

        if self.is_2d:
            if not inplace:
                new_col = TimeColumn(max(n * c, 1), cols=c)
                new_col.length = n * c
                _col_gather_2d_zero(new_col.data_c, self.data_c, indices, n, r, c, sizeof(int32_t))
                return new_col
            else:
                new_data = <int32_t*>aligned_malloc(max(n * c, 1) * sizeof(int32_t))
                if not new_data: raise MemoryError()
                _col_gather_2d_zero(new_data, self.data_c, indices, n, r, c, sizeof(int32_t))
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
            new_col = TimeColumn(max(n, 1), cols=self.cols)
            new_col.length = n
            _col_gather_zero(new_col.data_c, self.data_c, indices, n, sizeof(int32_t))
            return new_col
        else:
            new_data = <int32_t*>aligned_malloc(max(n, 1) * sizeof(int32_t))
            if not new_data: raise MemoryError()
            _col_gather_zero(new_data, self.data_c, indices, n, sizeof(int32_t))
            if self.owns_data and self.data_c != NULL:
                os_aligned_free(self.data_c)
            self.data_c = new_data
            self.capacity = n
            self.length = n
            self._base_obj = None
            return self

    # ------------------------------------------------------------
    # 纯 C 32位无分支比较算子 (完全向量化)
    # ------------------------------------------------------------
    cdef IntColumn _cmp_op(self, val, int op):
        cdef Py_ssize_t n = self.length
        cdef IntColumn res = IntColumn(max(n, 1), cols=self.num_cols)
        res.length = n

        cdef char err_buf[128]
        cdef int32_t v
        cdef TimeColumn other

        if isinstance(val, TimeColumn):
            other = <TimeColumn>val
            if other.length != n:
                snprintf(err_buf, sizeof(err_buf), "Length mismatch: %zd vs %zd", n, other.length)
                raise ValueError(err_buf.decode('ascii'))
            if other.num_cols != self.num_cols:
                raise ValueError(
                    f"Shape mismatch: self has {self.num_cols} column(s), other has {other.num_cols}")
            with nogil:
                _n_cmp_col(self.data_c, other.data_c, res.data_c, n, op)
        else:
            v = self._resolve_time(val)
            with nogil:
                _n_cmp_scalar(self.data_c, res.data_c, v, n, op)
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __lt__(self, val):  return self._cmp_op(val, 0)
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __le__(self, val):  return self._cmp_op(val, 1)
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __gt__(self, val):  return self._cmp_op(val, 2)
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __ge__(self, val):  return self._cmp_op(val, 3)
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __eq__(self, val):  return self._cmp_op(val, 4)
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __ne__(self, val):  return self._cmp_op(val, 5)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __str__(self):
        cdef list items = []
        cdef Py_ssize_t i, n = self.length
        cdef char buf[16]
        for i in range(n):
            _format_time_c(self.data_c[i], buf)
            items.append(buf.decode('ascii'))
        return f"TimeColumn([{', '.join(items)}])"

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __bool__(self):
        return self.length > 0

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __len__(self):
        return self.length

    # ------------------------------------------------------------
    # 类型转换 astype (直接 C 级内存赋值与零堆开销)
    # ------------------------------------------------------------
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def astype(self, object target):
        cdef int target_type = _resolve_astype_target(target)
        cdef Py_ssize_t i, n = self.length
        cdef Py_ssize_t c = self.cols
        cdef bint is_2d_src = self.is_2d and c > 1
        cdef bint target_supports_2d = target_type in (TYPE_INT, TYPE_DOUBLE, TYPE_DATE, TYPE_DATETIME, TYPE_TIME)
        if is_2d_src and not target_supports_2d:
            raise NotImplementedError(
                f"Cannot astype 2D column ({c} sub-columns) to non-2D type {target}. "
                f"Supported 2D targets: int, float, date, datetime, time.")
        cdef Py_ssize_t first_col_n = self.rows if c > 1 else n
        cdef Py_ssize_t safe_cap = max(n, 1)
        cdef Py_ssize_t first_col_cap = max(first_col_n, 1)
        cdef int32_t* ptr = self.data_c
        cdef char str_buf[16]

        cdef IntColumn ic
        cdef DoubleColumn dc
        cdef BoolColumn bc
        cdef UCS2Column uc
        cdef AsciiColumn cc
        cdef CategoryColumn catc
        cdef DateColumn tc
        cdef DateTimeColumn dtc

        if target_type == TYPE_INT:
            ic = IntColumn(safe_cap, cols=self.cols)
            ic.length = n
            with nogil:
                for i in range(n):
                    ic.data_c[i] = ptr[i]
            return ic
        elif target_type == TYPE_DOUBLE:
            dc = DoubleColumn(safe_cap, cols=self.cols)
            dc.length = n
            with nogil:
                for i in range(n):
                    dc.data_c[i] = <double>ptr[i]
            return dc
        elif target_type == TYPE_BOOL:
            bc = BoolColumn(first_col_cap)
            bc.length = first_col_n
            with nogil:
                for i in range(first_col_n):
                    bc.data_c[i] = <unsigned char>(ptr[i] != 0)
            return bc
        elif target_type == TYPE_ASCII:
            cc = AsciiColumn(first_col_cap)
            cc.length = first_col_n
            with nogil:
                for i in range(first_col_n):
                    _format_time_c(ptr[i], cc.data_c + (i * 256))
            return cc
        elif target_type == TYPE_UCS2:
            uc = UCS2Column(first_col_cap)
            for i in range(first_col_n):
                _format_time_c(ptr[i], str_buf)
                uc.append(str_buf.decode('ascii'))
            return uc
        elif target_type == TYPE_CATEGORY:
            catc = CategoryColumn(first_col_cap)
            for i in range(first_col_n):
                _format_time_c(ptr[i], str_buf)
                catc.append(str_buf.decode('ascii'))
            return catc
        elif target_type == TYPE_DATE:
            tc = DateColumn(safe_cap, cols=self.cols)
            tc.length = n
            with nogil:
                for i in range(n):
                    tc.data_c[i] = 0
            return tc
        elif target_type == TYPE_DATETIME:
            dtc = DateTimeColumn(safe_cap, cols=self.cols)
            dtc.length = n
            # Time -> DateTime: 秒数转换为微秒数 (1970-01-01 当天对应的时间点)
            with nogil:
                for i in range(n):
                    dtc.data_c[i] = <int64_t>ptr[i] * 1000000LL
            return dtc
        elif target_type == TYPE_TIME:
            return self.copy()

        raise TypeError(f"Failed to cast TIME column to internal type id: {target_type}")
