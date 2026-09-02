# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False

cimport cython
from libc.string cimport memcpy, memmove
from libc.stdio cimport snprintf
from libc.math cimport sqrt, fabs, NAN
from libc.stdint cimport uint64_t
from libc.stdlib cimport qsort, bsearch

from fluidframe.core._base cimport (
    BaseColumn, aligned_malloc, aligned_realloc, os_aligned_free,
    TYPE_DOUBLE, TYPE_INT, TYPE_CHAR256, TYPE_UCS2, TYPE_BOOL,
    TYPE_TIMESTAMP, TYPE_CATEGORY, TYPE_DATETIME, TYPE_TIME,
    _resolve_astype_target
)
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.boolcolumn cimport BoolColumn
from fluidframe.core.char256column cimport Char256Column_SoA
from fluidframe.core.ucs2column cimport UCS2Column_SoA
from fluidframe.core.datecolumn cimport DateColumn, _ymd_to_days
from fluidframe.core.categorycolumn cimport CategoryColumn
from fluidframe.core.datetimecolumn cimport DateTimeColumn
from fluidframe.core.timecolumn cimport TimeColumn


cdef int _cmp_double(const void* a, const void* b) noexcept nogil:
    """提供给 qsort 和 bsearch 的比较函数，严格处理 NaN 防止排序崩溃"""
    cdef double va = (<const double*>a)[0]
    cdef double vb = (<const double*>b)[0]
    cdef int anan = (va != va)
    cdef int bnan = (vb != vb)
    if anan and bnan: return 0
    if anan: return -1
    if bnan: return 1
    if va < vb: return -1
    if va > vb: return 1
    return 0


cdef double _c_corr(double* a, double* b, Py_ssize_t n) noexcept nogil:
    if n < 2: return NAN
    cdef double sum_a = 0.0, sum_b = 0.0
    cdef double mean_a, mean_b
    cdef double cov = 0.0, std_a = 0.0, std_b = 0.0
    cdef double xi, xj, denom
    cdef Py_ssize_t k

    # 4路累加求和，打破依赖链加速向量化
    cdef double s_a0 = 0.0, s_a1 = 0.0, s_a2 = 0.0, s_a3 = 0.0
    cdef double s_b0 = 0.0, s_b1 = 0.0, s_b2 = 0.0, s_b3 = 0.0
    cdef Py_ssize_t n_unrolled = n - (n % 4)

    for k in range(0, n_unrolled, 4):
        s_a0 += a[k]
        s_a1 += a[k+1]
        s_a2 += a[k+2]
        s_a3 += a[k+3]
        s_b0 += b[k]
        s_b1 += b[k+1]
        s_b2 += b[k+2]
        s_b3 += b[k+3]
    sum_a = (s_a0 + s_a1) + (s_a2 + s_a3)
    sum_b = (s_b0 + s_b1) + (s_b2 + s_b3)
    for k in range(n_unrolled, n):
        sum_a += a[k]
        sum_b += b[k]

    mean_a = sum_a / n
    mean_b = sum_b / n

    for k in range(n):
        xi = a[k] - mean_a
        xj = b[k] - mean_b
        cov += xi * xj
        std_a += xi * xi
        std_b += xj * xj

    denom = std_a * std_b
    if denom > 0.0:
        return cov / sqrt(denom)
    return NAN


cdef double _c_cov(double* a, double* b, Py_ssize_t n) noexcept nogil:
    if n < 2: return NAN
    cdef double mean_a = 0.0, mean_b = 0.0
    cdef double cov_val = 0.0
    cdef Py_ssize_t k
    for k in range(n):
        mean_a += a[k]
        mean_b += b[k]
    mean_a /= n
    mean_b /= n
    for k in range(n):
        cov_val += (a[k] - mean_a) * (b[k] - mean_b)
    return cov_val / (n - 1)


cdef double _c_euclidean_dist(double* a, double* b, Py_ssize_t n) noexcept nogil:
    if n == 0: return 0.0
    cdef double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0
    cdef double d0, d1, d2, d3
    cdef Py_ssize_t k
    cdef Py_ssize_t n_unrolled = n - (n % 4)

    for k in range(0, n_unrolled, 4):
        d0 = a[k] - b[k]
        d1 = a[k+1] - b[k+1]
        d2 = a[k+2] - b[k+2]
        d3 = a[k+3] - b[k+3]
        s0 += d0 * d0
        s1 += d1 * d1
        s2 += d2 * d2
        s3 += d3 * d3

    cdef double s = (s0 + s1) + (s2 + s3)
    for k in range(n_unrolled, n):
        d0 = a[k] - b[k]
        s += d0 * d0
    return sqrt(s)


cdef double _c_manhattan_dist(double* a, double* b, Py_ssize_t n) noexcept nogil:
    if n == 0: return 0.0
    cdef double s = 0.0
    cdef Py_ssize_t k
    for k in range(n):
        s += fabs(a[k] - b[k])
    return s


cdef double _c_cosine_similarity(double* a, double* b, Py_ssize_t n) noexcept nogil:
    if n == 0: return 0.0
    cdef double dot_val = 0.0, norm_a = 0.0, norm_b = 0.0, prod
    cdef Py_ssize_t k
    for k in range(n):
        dot_val += a[k] * b[k]
        norm_a += a[k] * a[k]
        norm_b += b[k] * b[k]
    prod = sqrt(norm_a * norm_b)
    if prod < 1e-15:
        return 0.0
    cdef double sim = dot_val / prod
    if sim > 1.0: sim = 1.0
    elif sim < -1.0: sim = -1.0
    return sim


cdef double _c_cosine_dist(double* a, double* b, Py_ssize_t n) noexcept nogil:
    if n == 0: return 1.0
    cdef double sim = _c_cosine_similarity(a, b, n)
    return 1.0 - sim


cdef double _c_jaccard_sim(double* a, double* b, Py_ssize_t n) noexcept nogil:
    if n == 0: return 0.0
    cdef Py_ssize_t intersection = 0, union_cnt = 0
    cdef Py_ssize_t k
    cdef int b_i, b_j
    for k in range(n):
        b_i = (a[k] != 0.0)
        b_j = (b[k] != 0.0)
        intersection += (b_i & b_j)
        union_cnt += (b_i | b_j)
    if union_cnt == 0:
        return 0.0
    return <double>intersection / <double>union_cnt


cdef double _c_hamming_dist(double* a, double* b, Py_ssize_t n) noexcept nogil:
    if n == 0: return 0.0
    cdef Py_ssize_t diff_count = 0
    cdef Py_ssize_t k
    for k in range(n):
        diff_count += (a[k] != b[k])
    return <double>diff_count / <double>n


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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef class DoubleColumn(BaseColumn):

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __cinit__(self, Py_ssize_t init_capacity=16):
        self.capacity = max(init_capacity, 1)
        self.length = 0
        self.dtype = TYPE_DOUBLE
        self.data_c = <double*>aligned_malloc(self.capacity * sizeof(double))
        if not self.data_c: raise MemoryError()

    def __dealloc__(self):
        if self.data_c is not NULL:
            os_aligned_free(self.data_c)
            self.data_c = NULL

    cpdef Py_ssize_t get_size(self):
        return self.capacity * sizeof(double)

    cdef void _reserve(self, Py_ssize_t new_cap):
        if new_cap > self.capacity:
            self.data_c = <double*>aligned_realloc(self.data_c, self.capacity * sizeof(double), new_cap * sizeof(double))
            if not self.data_c: raise MemoryError()
            self.capacity = new_cap

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def append(self, double val):
        if self.length >= self.capacity:
            self._reserve(self.capacity * 2)
        self.data_c[self.length] = val
        self.length += 1

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def copy(self):
        cdef DoubleColumn new_col = DoubleColumn(max(self.capacity, 1))
        new_col.length = self.length
        memcpy(new_col.data_c, self.data_c, self.length * sizeof(double))
        return new_col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_back(self):
        if self.length == 0: raise IndexError("pop from empty column")
        self.length -= 1
        return self.data_c[self.length]

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_front(self):
        if self.length == 0: raise IndexError("pop from empty column")
        cdef double val = self.data_c[0]
        self.length -= 1
        if self.length > 0:
            memmove(self.data_c, self.data_c + 1, self.length * sizeof(double))
        return val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def concat(self, DoubleColumn other):
        self._reserve(self.length + other.length)
        memcpy(self.data_c + self.length, other.data_c, other.length * sizeof(double))
        self.length += other.length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def max(self):
        if self.length == 0: raise ValueError("empty column")
        cdef double max_val = self.data_c[0]
        cdef double* ptr = self.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(1, self.length):
                if ptr[i] > max_val: max_val = ptr[i]
        return max_val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def min(self):
        if self.length == 0: raise ValueError("empty column")
        cdef double min_val = self.data_c[0]
        cdef double* ptr = self.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(1, self.length):
                if ptr[i] < min_val: min_val = ptr[i]
        return min_val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sum(self):
        cdef Py_ssize_t n = self.length
        if n == 0: return 0.0
        cdef double* ptr = self.data_c
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
        if self.length == 0: return NAN
        return self.sum() / self.length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def var(self):
        if self.length < 2: return 0.0
        cdef double m = self.mean()
        cdef double sum_sq = 0.0, diff
        cdef double* ptr = self.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                diff = ptr[i] - m
                sum_sq += diff * diff
        return sum_sq / (self.length - 1)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def std(self):
        return sqrt(self.var())

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _reorder(self, list indices):
        cdef Py_ssize_t n = len(indices)
        cdef char err_buf[128]
        if n != self.length:
            snprintf(err_buf, sizeof(err_buf), "Length of indices (%zd) must match column length (%zd)", n, self.length)
            raise ValueError(err_buf.decode('ascii'))

        cdef Py_ssize_t new_cap = max(n, 1)
        cdef double* new_data = <double*>aligned_malloc(new_cap * sizeof(double))
        if not new_data: raise MemoryError()
        cdef Py_ssize_t idx, i
        cdef double* self_data = self.data_c

        try:
            for i in range(n):
                idx = <Py_ssize_t>indices[i]
                if idx < 0: idx += self.length
                if idx < 0 or idx >= self.length:
                    snprintf(err_buf, sizeof(err_buf), "Index out of bounds: %zd", idx)
                    raise IndexError(err_buf.decode('ascii'))
                new_data[i] = self_data[idx]
        except Exception:
            os_aligned_free(new_data)
            raise

        os_aligned_free(self.data_c)
        self.data_c = new_data
        self.capacity = new_cap
        self.length = n

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __getitem__(self, Py_ssize_t idx):
        if idx < 0: idx += self.length
        if idx < 0 or idx >= self.length: raise IndexError("Index out of range")
        return self.data_c[idx]

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def count_nonzeros(self):
        cdef Py_ssize_t count = 0
        cdef Py_ssize_t i
        cdef double* ptr = self.data_c
        with nogil:
            for i in range(self.length):
                count += (ptr[i] != 0.0)
        return count

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def idxmax(self):
        if self.length == 0: raise ValueError("empty column")
        cdef Py_ssize_t max_idx = 0
        cdef double max_val = self.data_c[0]
        cdef Py_ssize_t i
        cdef double* ptr = self.data_c
        with nogil:
            for i in range(1, self.length):
                if ptr[i] > max_val:
                    max_val = ptr[i]
                    max_idx = i
        return max_idx

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def idxmin(self):
        if self.length == 0: raise ValueError("empty column")
        cdef Py_ssize_t min_idx = 0
        cdef double min_val = self.data_c[0]
        cdef Py_ssize_t i
        cdef double* ptr = self.data_c
        with nogil:
            for i in range(1, self.length):
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

    cdef double _quickselect(self, double* data, Py_ssize_t n, Py_ssize_t k) noexcept nogil:
        cdef Py_ssize_t left = 0, right = n - 1, i, j, mid
        cdef double pivot, tmp
        while left < right:
            mid = left + (right - left) // 2
            if data[left] > data[mid]:
                tmp = data[left]; data[left] = data[mid]; data[mid] = tmp
            if data[left] > data[right]:
                tmp = data[left]; data[left] = data[right]; data[right] = tmp
            if data[mid] > data[right]:
                tmp = data[mid]; data[mid] = data[right]; data[right] = tmp

            pivot = data[mid]
            i = left
            j = right
            while True:
                while data[i] < pivot: i += 1
                while data[j] > pivot: j -= 1
                if i >= j: break
                tmp = data[i]; data[i] = data[j]; data[j] = tmp
                i += 1
                j -= 1
            if k <= j:
                right = j
            else:
                left = j + 1
        return data[k]

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def median(self):
        if self.length == 0: return NAN
        cdef double* temp = <double*>aligned_malloc(self.length * sizeof(double))
        if not temp: raise MemoryError()
        memcpy(temp, self.data_c, self.length * sizeof(double))
        cdef Py_ssize_t mid = self.length // 2
        cdef double v1, v2, result
        if self.length % 2 == 0:
            v1 = self._quickselect(temp, self.length, mid - 1)
            v2 = self._quickselect(temp, self.length, mid)
            os_aligned_free(temp)
            return (v1 + v2) * 0.5
        result = self._quickselect(temp, self.length, mid)
        os_aligned_free(temp)
        return result

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def mad(self):
        if self.length == 0: return NAN
        cdef double m = self.mean()
        cdef double total_dev = 0.0
        cdef double* ptr = self.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                total_dev += fabs(ptr[i] - m)
        return total_dev / self.length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def prod(self):
        if self.length == 0: return NAN
        cdef double total = 1.0
        cdef double* ptr = self.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                total *= ptr[i]
        return total

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def nunique(self):
        if self.length == 0: return 0
        cdef set seen = set()
        cdef Py_ssize_t i
        for i in range(self.length):
            seen.add(self.data_c[i])
        return len(seen)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def value_counts(self):
        cdef dict counts = {}
        cdef Py_ssize_t i
        cdef double val
        for i in range(self.length):
            val = self.data_c[i]
            counts[val] = counts.get(val, 0) + 1
        return counts

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def mode(self):
        if self.length == 0: return []
        cdef dict counts = {}
        cdef Py_ssize_t i
        cdef double val
        cdef int count, max_count = 0
        for i in range(self.length):
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
        cdef double* temp = <double*>aligned_malloc(self.length * sizeof(double))
        if not temp: raise MemoryError()
        memcpy(temp, self.data_c, self.length * sizeof(double))
        cdef double pos = q * (self.length - 1)
        cdef Py_ssize_t idx = <Py_ssize_t>pos
        cdef double frac = pos - idx
        cdef double result, v1, v2
        if idx >= self.length - 1:
            result = self._quickselect(temp, self.length, self.length - 1)
            os_aligned_free(temp)
            return result
        v1 = self._quickselect(temp, self.length, idx)
        v2 = self._quickselect(temp, self.length, idx + 1)
        os_aligned_free(temp)
        return v1 + frac * (v2 - v1)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cumsum(self):
        cdef DoubleColumn res = DoubleColumn(max(self.capacity, 1))
        res.length = self.length
        cdef double total = 0.0
        cdef double* ptr = self.data_c
        cdef double* res_ptr = res.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                total += ptr[i]
                res_ptr[i] = total
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cummax(self):
        cdef DoubleColumn res = DoubleColumn(max(self.capacity, 1))
        res.length = self.length
        if self.length == 0: return res
        cdef double cur_max = self.data_c[0]
        cdef double* ptr = self.data_c
        cdef double* res_ptr = res.data_c
        cdef Py_ssize_t i
        with nogil:
            res_ptr[0] = cur_max
            for i in range(1, self.length):
                if ptr[i] > cur_max:
                    cur_max = ptr[i]
                res_ptr[i] = cur_max
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cummin(self):
        cdef DoubleColumn res = DoubleColumn(max(self.capacity, 1))
        res.length = self.length
        if self.length == 0: return res
        cdef double cur_min = self.data_c[0]
        cdef double* ptr = self.data_c
        cdef double* res_ptr = res.data_c
        cdef Py_ssize_t i
        with nogil:
            res_ptr[0] = cur_min
            for i in range(1, self.length):
                if ptr[i] < cur_min:
                    cur_min = ptr[i]
                res_ptr[i] = cur_min
        return res

    @cython.binding(False)
    def diff(self, int periods=1):
        cdef DoubleColumn res = DoubleColumn(max(self.capacity, 1))
        res.length = self.length
        cdef double* ptr = self.data_c
        cdef double* res_ptr = res.data_c
        cdef Py_ssize_t i
        with nogil:
            if periods >= 0:
                for i in range(min(periods, self.length)):
                    res_ptr[i] = NAN
                for i in range(periods, self.length):
                    res_ptr[i] = ptr[i] - ptr[i - periods]
            else:
                periods = -periods
                for i in range(max(0, self.length - periods), self.length):
                    res_ptr[i] = NAN
                for i in range(self.length - periods):
                    res_ptr[i] = ptr[i] - ptr[i + periods]
        return res

    @cython.binding(False)
    def pct_change(self, int periods=1):
        cdef DoubleColumn res = DoubleColumn(max(self.capacity, 1))
        res.length = self.length
        cdef double* ptr = self.data_c
        cdef double* res_ptr = res.data_c
        cdef Py_ssize_t i
        cdef double prev
        with nogil:
            if periods >= 0:
                for i in range(min(periods, self.length)):
                    res_ptr[i] = NAN
                for i in range(periods, self.length):
                    prev = ptr[i - periods]
                    res_ptr[i] = (ptr[i] - prev) / prev if prev != 0.0 else NAN
            else:
                periods = -periods
                for i in range(max(0, self.length - periods), self.length):
                    res_ptr[i] = NAN
                for i in range(self.length - periods):
                    prev = ptr[i + periods]
                    res_ptr[i] = (ptr[i] - prev) / prev if prev != 0.0 else NAN
        return res

    cdef void _central_moments(self, double* out_mean, double* out_m2, double* out_m3, double* out_m4) noexcept nogil:
        """Welford 单趟在线算法计算各阶中心矩"""
        cdef Py_ssize_t n = self.length
        if n == 0:
            out_mean[0] = 0.0
            out_m2[0] = 0.0
            out_m3[0] = 0.0
            out_m4[0] = 0.0
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
            m4 = m4 + term1 * delta_n2 * ((n1 + 1) * (n1 + 1) - 3 * (n1 + 1) + 3) + 6.0 * delta_n2 * m2 - 4.0 * delta_n * m3
            m3 = m3 + term1 * delta_n * (n1 - 2) - 3.0 * delta_n * m2
            m2 = m2 + term1
        out_mean[0] = mean
        out_m2[0] = m2
        out_m3[0] = m3
        out_m4[0] = m4

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def skew(self):
        if self.length < 3: return NAN
        cdef double m, m2, m3, m4
        self._central_moments(&m, &m2, &m3, &m4)
        if m2 == 0.0: return NAN
        cdef double n = <double>self.length
        cdef double var_sample = m2 / (n - 1)
        return (n * m3) / ((n - 1) * (n - 2) * (var_sample * sqrt(var_sample)))

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def kurt(self):
        if self.length < 4: return NAN
        cdef double m, m2, m3, m4
        self._central_moments(&m, &m2, &m3, &m4)
        if m2 == 0.0: return NAN
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
    def kurtosis(self):
        return self.kurt()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cumprod(self):
        cdef DoubleColumn res = DoubleColumn(max(self.capacity, 1))
        res.length = self.length
        cdef double total = 1.0
        cdef double* ptr = self.data_c
        cdef double* res_ptr = res.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                total *= ptr[i]
                res_ptr[i] = total
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
            "min": self.min(),
            "25%": self.quantile(0.25),
            "50%": self.quantile(0.50),
            "75%": self.quantile(0.75),
            "max": self.max()
        }

    # 行索引聚合底层接口
    cdef double _sum_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil:
        cdef double total = 0.0
        cdef Py_ssize_t i
        cdef double* ptr = self.data_c
        for i in range(n):
            total += ptr[indices[i]]
        return total

    cdef double _mean_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil:
        if n == 0: return NAN
        return self._sum_rows(indices, n) / n

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
        return sum_sq / (n - 1)

    cdef double _std_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil:
        return sqrt(self._var_rows(indices, n))

    # 标量算术运算（分支外提优化）
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def add_val(self, double val):
        cdef DoubleColumn res = DoubleColumn(max(self.capacity, 1))
        res.length = self.length
        cdef double* data_ptr = self.data_c
        cdef double* res_ptr = res.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                res_ptr[i] = data_ptr[i] + val
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sub_val(self, double val):
        cdef DoubleColumn res = DoubleColumn(max(self.capacity, 1))
        res.length = self.length
        cdef double* data_ptr = self.data_c
        cdef double* res_ptr = res.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                res_ptr[i] = data_ptr[i] - val
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def mul_val(self, double val):
        cdef DoubleColumn res = DoubleColumn(max(self.capacity, 1))
        res.length = self.length
        cdef double* data_ptr = self.data_c
        cdef double* res_ptr = res.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                res_ptr[i] = data_ptr[i] * val
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def div_val(self, double val):
        cdef DoubleColumn res = DoubleColumn(max(self.capacity, 1))
        res.length = self.length
        cdef double* data_ptr = self.data_c
        cdef double* res_ptr = res.data_c
        cdef Py_ssize_t i
        cdef double inv_val
        with nogil:
            if val != 0.0:
                inv_val = 1.0 / val
                for i in range(self.length):
                    res_ptr[i] = data_ptr[i] * inv_val
            else:
                for i in range(self.length):
                    res_ptr[i] = NAN
        return res

    # 向量算术运算
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def add(self, object other):
        cdef Py_ssize_t n = self.length
        if other.length != n:
            raise ValueError(f"Column length {other.length} != {n}")
        cdef DoubleColumn res = DoubleColumn(max(n, 1))
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
        cdef DoubleColumn res = DoubleColumn(max(n, 1))
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
        cdef DoubleColumn res = DoubleColumn(max(n, 1))
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
        cdef DoubleColumn res = DoubleColumn(max(n, 1))
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
        if isinstance(other, (int, float)):
            return self.add_val(<double>other)
        if isinstance(other, (DoubleColumn, IntColumn)):
            return self.add(other)
        return NotImplemented

    def __radd__(self, object other):
        return self.__add__(other)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __sub__(self, object other):
        if isinstance(other, (int, float)):
            return self.sub_val(<double>other)
        if isinstance(other, (DoubleColumn, IntColumn)):
            return self.sub(other)
        return NotImplemented

    def __rsub__(self, object other):
        cdef Py_ssize_t n = self.length
        cdef DoubleColumn res
        cdef double v
        cdef double* s_ptr
        cdef double* r_ptr
        cdef Py_ssize_t i
        if isinstance(other, (int, float)):
            res = DoubleColumn(max(n, 1))
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
        if isinstance(other, (int, float)):
            return self.mul_val(<double>other)
        if isinstance(other, (DoubleColumn, IntColumn)):
            return self.mul(other)
        return NotImplemented

    def __rmul__(self, object other):
        return self.__mul__(other)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __truediv__(self, object other):
        if isinstance(other, (int, float)):
            return self.div_val(<double>other)
        if isinstance(other, (DoubleColumn, IntColumn)):
            return self.div(other)
        return NotImplemented

    def __rtruediv__(self, object other):
        cdef Py_ssize_t n = self.length
        cdef DoubleColumn res
        cdef double v, s_val
        cdef double* s_ptr
        cdef double* r_ptr
        cdef Py_ssize_t i
        if isinstance(other, (int, float)):
            res = DoubleColumn(max(n, 1))
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
        cdef set seen = set()
        cdef list result = []
        cdef Py_ssize_t i
        cdef double val
        for i in range(self.length):
            val = self.data_c[i]
            if val not in seen:
                seen.add(val)
                result.append(val)
        return result

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _cmp_op(self, object val, int op):
        cdef IntColumn res = IntColumn(max(self.capacity, 1))
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
                    for i in range(n): r_ptr[i] = <int>(s_ptr[i] < o_ptr[i])
                elif op == 1:
                    for i in range(n): r_ptr[i] = <int>(s_ptr[i] <= o_ptr[i])
                elif op == 2:
                    for i in range(n): r_ptr[i] = <int>(s_ptr[i] > o_ptr[i])
                elif op == 3:
                    for i in range(n): r_ptr[i] = <int>(s_ptr[i] >= o_ptr[i])
                elif op == 4:
                    for i in range(n): r_ptr[i] = <int>(s_ptr[i] == o_ptr[i])
                elif op == 5:
                    for i in range(n): r_ptr[i] = <int>(s_ptr[i] != o_ptr[i])
        else:
            v = <double>val
            is_nan_val = (v != v)
            with nogil:
                if is_nan_val:
                    if op == 4:
                        for i in range(n): r_ptr[i] = <int>(s_ptr[i] != s_ptr[i])
                    elif op == 5:
                        for i in range(n): r_ptr[i] = <int>(s_ptr[i] == s_ptr[i])
                    else:
                        for i in range(n): r_ptr[i] = 0
                else:
                    if op == 0:
                        for i in range(n): r_ptr[i] = <int>(s_ptr[i] < v)
                    elif op == 1:
                        for i in range(n): r_ptr[i] = <int>(s_ptr[i] <= v)
                    elif op == 2:
                        for i in range(n): r_ptr[i] = <int>(s_ptr[i] > v)
                    elif op == 3:
                        for i in range(n): r_ptr[i] = <int>(s_ptr[i] >= v)
                    elif op == 4:
                        for i in range(n): r_ptr[i] = <int>(s_ptr[i] == v)
                    elif op == 5:
                        for i in range(n): r_ptr[i] = <int>(s_ptr[i] != v)
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
            raise ValueError(f"Length mismatch: {self.length} vs {other.length}")
        cdef IntColumn res = IntColumn(max(self.capacity, 1))
        res.length = self.length
        cdef Py_ssize_t i
        cdef double* s_ptr = self.data_c
        cdef int* o_ptr = (<IntColumn>other).data_c
        cdef int* r_ptr = res.data_c
        with nogil:
            for i in range(self.length):
                r_ptr[i] = (s_ptr[i] != 0.0) & (o_ptr[i] != 0)
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __or__(self, other):
        if not isinstance(other, IntColumn):
            raise TypeError("Cannot | non-IntColumn mask")
        if self.length != other.length:
            raise ValueError(f"Length mismatch: {self.length} vs {other.length}")
        cdef IntColumn res = IntColumn(max(self.capacity, 1))
        res.length = self.length
        cdef Py_ssize_t i
        cdef double* s_ptr = self.data_c
        cdef int* o_ptr = (<IntColumn>other).data_c
        cdef int* r_ptr = res.data_c
        with nogil:
            for i in range(self.length):
                r_ptr[i] = (s_ptr[i] != 0.0) | (o_ptr[i] != 0)
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __invert__(self):
        cdef IntColumn res = IntColumn(max(self.capacity, 1))
        res.length = self.length
        cdef Py_ssize_t i
        cdef double* s_ptr = self.data_c
        cdef int* r_ptr = res.data_c
        with nogil:
            for i in range(self.length):
                r_ptr[i] = (s_ptr[i] == 0.0)
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def isin(self, set values):
        cdef IntColumn res = IntColumn(max(self.capacity, 1))
        res.length = self.length
        cdef Py_ssize_t i, n = self.length
        cdef double* s_ptr = self.data_c
        cdef int* r_ptr = res.data_c
        cdef Py_ssize_t m = len(values)

        if m == 0:
            with nogil:
                for i in range(n): r_ptr[i] = 0
            return res

        cdef double* v_arr = <double*>aligned_malloc(m * sizeof(double))
        if not v_arr: raise MemoryError()
        cdef Py_ssize_t valid_m = 0
        cdef void* found
        cdef double single_val

        try:
            for v in values:
                if v is not None:
                    v_arr[valid_m] = <double>v
                    valid_m += 1

            if valid_m == 0:
                with nogil:
                    for i in range(n): r_ptr[i] = 0
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
        cdef IntColumn res = IntColumn(max(self.capacity, 1))
        res.length = self.length
        cdef Py_ssize_t i
        cdef double* s_ptr = self.data_c
        cdef int* r_ptr = res.data_c
        with nogil:
            for i in range(self.length):
                r_ptr[i] = (low <= s_ptr[i]) & (s_ptr[i] <= high)
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __str__(self):
        cdef list items = []
        cdef Py_ssize_t i
        for i in range(self.length):
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
        cdef Py_ssize_t i
        cdef double* old_data = self.data_c
        cdef DoubleColumn new_col
        cdef double* new_data

        if not inplace:
            new_col = DoubleColumn(max(n, 1))
            new_col.length = n
            with nogil:
                for i in range(n):
                    new_col.data_c[i] = old_data[indices[i]]
            return new_col
        else:
            new_data = <double*>aligned_malloc(max(n, self.capacity) * sizeof(double))
            if not new_data: raise MemoryError()
            with nogil:
                for i in range(n):
                    new_data[i] = old_data[indices[i]]
            os_aligned_free(self.data_c)
            self.data_c = new_data
            self.capacity = max(n, self.capacity)
            self.length = n
            return self

    cpdef DoubleColumn clip(self, double lower, double upper, bint inplace=False):
        cdef DoubleColumn col = self if inplace else self.copy()
        cdef Py_ssize_t i
        cdef double v
        cdef double* ptr = col.data_c
        with nogil:
            for i in range(col.length):
                v = ptr[i]
                if v < lower: v = lower
                elif v > upper: v = upper
                ptr[i] = v
        return col

    cpdef DoubleColumn where(self, IntColumn cond, double other, bint inplace=False):
        cdef Py_ssize_t n = self.length
        cdef char err_buf[128]
        if cond.length != n:
            snprintf(err_buf, sizeof(err_buf), "Condition mask length %zd must match column length %zd", cond.length, n)
            raise ValueError(err_buf.decode('ascii'))

        cdef DoubleColumn col = self if inplace else DoubleColumn(max(self.capacity, 1))
        if not inplace: col.length = n

        cdef Py_ssize_t i
        cdef double* col_ptr = col.data_c
        cdef double* self_ptr = self.data_c
        cdef int* cond_ptr = cond.data_c

        # 标准三元运算直接映射为高效 SIMD Blend 指令，无类型双关安全隐患
        with nogil:
            for i in range(n):
                col_ptr[i] = self_ptr[i] if cond_ptr[i] != 0 else other
        return col

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def where_mask(self, IntColumn mask, object fill):
        cdef Py_ssize_t n = self.length
        cdef char err_buf[128]
        if mask.length != n:
            snprintf(err_buf, sizeof(err_buf), "Mask length %zd != column length %zd", mask.length, n)
            raise ValueError(err_buf.decode('ascii'))

        cdef DoubleColumn col = DoubleColumn(max(self.capacity, 1))
        col.length = n
        cdef double fill_val = <double>fill
        cdef Py_ssize_t i
        cdef double* col_ptr = col.data_c
        cdef double* self_ptr = self.data_c
        cdef int* mask_ptr = mask.data_c

        with nogil:
            for i in range(n):
                col_ptr[i] = self_ptr[i] if mask_ptr[i] != 0 else fill_val
        return col

    cpdef DoubleColumn mask(self, IntColumn cond, double other, bint inplace=False):
        cdef Py_ssize_t n = self.length
        cdef char err_buf[128]
        if cond.length != n:
            snprintf(err_buf, sizeof(err_buf), "Condition mask length %zd must match column length %zd", cond.length, n)
            raise ValueError(err_buf.decode('ascii'))

        cdef DoubleColumn col = self if inplace else DoubleColumn(max(self.capacity, 1))
        if not inplace: col.length = n

        cdef Py_ssize_t i
        cdef double* col_ptr = col.data_c
        cdef double* self_ptr = self.data_c
        cdef int* cond_ptr = cond.data_c

        with nogil:
            for i in range(n):
                col_ptr[i] = other if cond_ptr[i] != 0 else self_ptr[i]
        return col

    cpdef void drop_row(self, Py_ssize_t row_idx, Py_ssize_t rem_len):
        if row_idx < 0:
            row_idx += self.length
        if row_idx < 0 or row_idx >= self.length:
            raise IndexError("Row index out of bounds")
        if rem_len > 0:
            if row_idx + 1 + rem_len > self.length:
                rem_len = self.length - row_idx - 1
            memmove(self.data_c + row_idx,
                    self.data_c + row_idx + 1,
                    rem_len * sizeof(double))
        self.length -= 1

    cdef void to_double_buffer(self, double* out, Py_ssize_t n):
        if n > 0:
            memcpy(out, self.data_c, n * sizeof(double))

    cdef double corr_with(self, double* other, Py_ssize_t n):
        return _c_corr(self.data_c, other, n)

    cdef double cov_with(self, double* other, Py_ssize_t n):
        return _c_cov(self.data_c, other, n)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pearson_corr_with(self, other):
        cdef Py_ssize_t n = min(self.length, other.length)
        if n < 2: return NAN
        cdef DoubleColumn temp = DoubleColumn(max(n, 1))
        temp.length = n
        _col_to_double(other, temp.data_c, n)
        return self.corr_with(temp.data_c, n)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pearson_cov_with(self, other):
        cdef Py_ssize_t n = min(self.length, other.length)
        if n < 2: return NAN
        cdef DoubleColumn temp = DoubleColumn(max(n, 1))
        temp.length = n
        _col_to_double(other, temp.data_c, n)
        return self.cov_with(temp.data_c, n)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def euclidean_dist(self, other):
        cdef Py_ssize_t n = min(self.length, other.length)
        if n == 0: return NAN
        cdef DoubleColumn temp = DoubleColumn(max(n, 1))
        temp.length = n
        _col_to_double(other, temp.data_c, n)
        return _c_euclidean_dist(self.data_c, temp.data_c, n)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def manhattan_dist(self, other):
        cdef Py_ssize_t n = min(self.length, other.length)
        if n == 0: return NAN
        cdef DoubleColumn temp = DoubleColumn(max(n, 1))
        temp.length = n
        _col_to_double(other, temp.data_c, n)
        return _c_manhattan_dist(self.data_c, temp.data_c, n)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cosine_dist(self, other):
        cdef Py_ssize_t n = min(self.length, other.length)
        if n == 0: return NAN
        cdef DoubleColumn temp = DoubleColumn(max(n, 1))
        temp.length = n
        _col_to_double(other, temp.data_c, n)
        return _c_cosine_dist(self.data_c, temp.data_c, n)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cosine_similarity(self, other):
        cdef Py_ssize_t n = min(self.length, other.length)
        if n == 0: return NAN
        cdef DoubleColumn temp = DoubleColumn(max(n, 1))
        temp.length = n
        _col_to_double(other, temp.data_c, n)
        return _c_cosine_similarity(self.data_c, temp.data_c, n)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def jaccard_sim(self, other):
        cdef Py_ssize_t n = min(self.length, other.length)
        if n == 0: return NAN
        cdef DoubleColumn temp = DoubleColumn(max(n, 1))
        temp.length = n
        _col_to_double(other, temp.data_c, n)
        return _c_jaccard_sim(self.data_c, temp.data_c, n)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def hamming_dist(self, other):
        cdef Py_ssize_t n = min(self.length, other.length)
        if n == 0: return NAN
        cdef DoubleColumn temp = DoubleColumn(max(n, 1))
        temp.length = n
        _col_to_double(other, temp.data_c, n)
        return _c_hamming_dist(self.data_c, temp.data_c, n)

    cdef DoubleColumn gather_with_sentinel(self, Py_ssize_t* indices, Py_ssize_t n, double sentinel):
        cdef DoubleColumn new_col = DoubleColumn(max(n, 1))
        new_col.length = n
        cdef Py_ssize_t i
        cdef double* res_ptr = new_col.data_c
        cdef double* src_ptr = self.data_c
        with nogil:
            for i in range(n):
                res_ptr[i] = src_ptr[indices[i]] if indices[i] >= 0 else sentinel
        return new_col

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def astype(self, object target):
        cdef int target_type = _resolve_astype_target(target)
        cdef Py_ssize_t i, n = self.length
        cdef Py_ssize_t safe_cap = max(n, 1)
        cdef int ival, y, m, d
        cdef char buf[64]
        cdef int sz

        cdef IntColumn ic
        cdef BoolColumn bc
        cdef UCS2Column_SoA uc
        cdef Char256Column_SoA cc
        cdef DateColumn tc
        cdef CategoryColumn catc
        cdef DateTimeColumn dtc
        cdef TimeColumn tmc

        cdef double* d_ptr = self.data_c

        if target_type == TYPE_INT:
            ic = IntColumn(safe_cap)
            ic.length = n
            with nogil:
                for i in range(n):
                    ic.data_c[i] = <int>d_ptr[i]
            return ic

        elif target_type == TYPE_BOOL:
            bc = BoolColumn(safe_cap)
            bc.length = n
            with nogil:
                for i in range(n):
                    bc.data_c[i] = (d_ptr[i] != 0.0 and d_ptr[i] == d_ptr[i])
            return bc

        elif target_type == TYPE_UCS2:
            uc = UCS2Column_SoA(safe_cap)
            uc.length = n
            for i in range(n):
                if d_ptr[i] != d_ptr[i]:
                    uc._set_item(i, "null")
                else:
                    sz = snprintf(buf, sizeof(buf), "%.6g", d_ptr[i])
                    uc._set_item(i, buf[:sz].decode('ascii'))
            return uc

        elif target_type == TYPE_CHAR256:
            cc = Char256Column_SoA(safe_cap)
            for i in range(n):
                if d_ptr[i] != d_ptr[i]:
                    cc.append("null")
                else:
                    sz = snprintf(buf, sizeof(buf), "%.6g", d_ptr[i])
                    cc.append(buf[:sz].decode('ascii'))
            return cc

        elif target_type == TYPE_TIMESTAMP:
            tc = DateColumn(safe_cap)
            tc.length = n
            with nogil:
                for i in range(n):
                    ival = <int>d_ptr[i]
                    tc.data_c[i] = _ymd_to_days(ival // 10000, (ival // 100) % 100, ival % 100)
            return tc

        elif target_type == TYPE_CATEGORY:
            catc = CategoryColumn(safe_cap)
            for i in range(n):
                if d_ptr[i] != d_ptr[i]:
                    catc.append("null")
                else:
                    sz = snprintf(buf, sizeof(buf), "%.6g", d_ptr[i])
                    catc.append(buf[:sz].decode('ascii'))
            return catc

        elif target_type == TYPE_DATETIME:
            dtc = DateTimeColumn(safe_cap)
            for i in range(n):
                ival = <int>d_ptr[i]
                y = ival // 10000
                m = (ival // 100) % 100
                d = ival % 100
                sz = snprintf(buf, sizeof(buf), "%04d-%02d-%02d", y, m, d)
                dtc.append(buf[:sz].decode('ascii'))
            return dtc

        elif target_type == TYPE_TIME:
            tmc = TimeColumn(safe_cap)
            for i in range(n):
                tmc.append(<int>d_ptr[i])
            return tmc

        elif target_type == TYPE_DOUBLE:
            return self.copy()

        cdef char err_buf[128]
        snprintf(err_buf, sizeof(err_buf), "Failed to cast DOUBLE column to internal type id: %d", target_type)
        raise TypeError(err_buf.decode('ascii'))