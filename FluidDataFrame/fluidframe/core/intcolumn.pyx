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
from libc.stdlib cimport qsort, bsearch
from libc.stdint cimport int64_t
from fluidframe.core._base cimport (
    TYPE_INT, TYPE_DOUBLE, TYPE_CHAR256, TYPE_UCS2, TYPE_BOOL,
    TYPE_TIMESTAMP, TYPE_CATEGORY, TYPE_DATETIME, TYPE_TIME,
    BaseColumn, aligned_malloc, aligned_realloc, os_aligned_free,
    _resolve_astype_target
)
from fluidframe.core.doublecolumn cimport DoubleColumn
from fluidframe.core.boolcolumn cimport BoolColumn
from fluidframe.core.char256column cimport Char256Column_SoA
from fluidframe.core.ucs2column cimport UCS2Column_SoA
from fluidframe.core.datecolumn cimport DateColumn, _ymd_to_days
from fluidframe.core.categorycolumn cimport CategoryColumn
from fluidframe.core.datetimecolumn cimport DateTimeColumn
from fluidframe.core.timecolumn cimport TimeColumn


cdef int _cmp_int(const void* a, const void* b) noexcept nogil:
    cdef int va = (<const int*>a)[0]
    cdef int vb = (<const int*>b)[0]
    if va < vb: return -1
    if va > vb: return 1
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef class IntColumn(BaseColumn):

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __cinit__(self, Py_ssize_t init_capacity=16):
        self.capacity = max(init_capacity, 1)
        self.length = 0
        self.dtype = TYPE_INT
        self.data_c = <int*>aligned_malloc(self.capacity * sizeof(int))
        if not self.data_c: raise MemoryError()

    def __dealloc__(self):
        if self.data_c is not NULL:
            os_aligned_free(self.data_c)
            self.data_c = NULL

    cpdef Py_ssize_t get_size(self):
        return self.capacity * sizeof(int)

    cdef void _reserve(self, Py_ssize_t new_cap):
        if new_cap > self.capacity:
            self.data_c = <int*>aligned_realloc(self.data_c, self.capacity * sizeof(int), new_cap * sizeof(int))
            if not self.data_c: raise MemoryError()
            self.capacity = new_cap

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def append(self, int val):
        if self.length >= self.capacity:
            self._reserve(self.capacity * 2)
        self.data_c[self.length] = val
        self.length += 1

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def copy(self):
        cdef IntColumn new_col = IntColumn(max(self.capacity, 1))
        new_col.length = self.length
        memcpy(new_col.data_c, self.data_c, self.length * sizeof(int))
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
        cdef int val = self.data_c[0]
        self.length -= 1
        if self.length > 0:
            memmove(self.data_c, self.data_c + 1, self.length * sizeof(int))
        return val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def concat(self, IntColumn other):
        self._reserve(self.length + other.length)
        memcpy(self.data_c + self.length, other.data_c, other.length * sizeof(int))
        self.length += other.length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def max(self):
        if self.length == 0: raise ValueError("empty column")
        cdef int max_val = self.data_c[0]
        cdef int* ptr = self.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(1, self.length):
                if ptr[i] > max_val: max_val = ptr[i]
        return max_val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def min(self):
        if self.length == 0: raise ValueError("empty column")
        cdef int min_val = self.data_c[0]
        cdef int* ptr = self.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(1, self.length):
                if ptr[i] < min_val: min_val = ptr[i]
        return min_val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sum(self):
        cdef Py_ssize_t n = self.length
        if n == 0:
            return 0

        cdef int* ptr = self.data_c
        cdef int64_t total = 0
        cdef Py_ssize_t i

        cdef int64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0
        cdef Py_ssize_t n_unrolled = n - (n % 4)

        with nogil:
            for i in range(0, n_unrolled, 4):
                s0 += ptr[i]
                s1 += ptr[i + 1]
                s2 += ptr[i + 2]
                s3 += ptr[i + 3]

            total = (s0 + s1) + (s2 + s3)
            for i in range(n_unrolled, n):
                total += ptr[i]
        return total

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def mean(self):
        if self.length == 0: return NAN
        return self.sum() / <double>self.length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def var(self):
        if self.length < 2: return 0.0
        cdef double m = self.mean()
        cdef double sum_sq = 0.0, diff
        cdef int* ptr = self.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                diff = <double>ptr[i] - m
                sum_sq += diff * diff
        return sum_sq / (self.length - 1)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def std(self):
        return sqrt(self.var())

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def count_nonzeros(self):
        cdef Py_ssize_t count = 0
        cdef int* ptr = self.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                count += (ptr[i] != 0)
        return count

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def idxmax(self):
        if self.length == 0: raise ValueError("empty column")
        cdef Py_ssize_t max_idx = 0
        cdef int max_val = self.data_c[0]
        cdef int* ptr = self.data_c
        cdef Py_ssize_t i
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
        cdef int min_val = self.data_c[0]
        cdef int* ptr = self.data_c
        cdef Py_ssize_t i
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

    cdef double _quickselect(self, int* data, Py_ssize_t n, Py_ssize_t k) noexcept nogil:
        """三数中值无死循环快速选择"""
        cdef Py_ssize_t left = 0, right = n - 1, i, j, mid
        cdef int pivot, tmp
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
        return <double>data[k]

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def median(self):
        if self.length == 0: return NAN
        cdef int* temp = <int*>aligned_malloc(self.length * sizeof(int))
        if not temp: raise MemoryError()
        memcpy(temp, self.data_c, self.length * sizeof(int))
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
        cdef int* ptr = self.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                total_dev += fabs(<double>ptr[i] - m)
        return total_dev / self.length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def prod(self):
        if self.length == 0: return NAN
        cdef double total = 1.0
        cdef int* ptr = self.data_c
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
        cdef int val
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
        cdef int val, count, max_count = 0
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
        cdef int* temp = <int*>aligned_malloc(self.length * sizeof(int))
        if not temp: raise MemoryError()
        memcpy(temp, self.data_c, self.length * sizeof(int))
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
        cdef IntColumn res = IntColumn(max(self.capacity, 1))
        res.length = self.length
        cdef int total = 0
        cdef int* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                total += ptr[i]
                res_ptr[i] = total
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cummax(self):
        cdef IntColumn res = IntColumn(max(self.capacity, 1))
        res.length = self.length
        if self.length == 0: return res
        cdef int cur_max = self.data_c[0]
        cdef int* ptr = self.data_c
        cdef int* res_ptr = res.data_c
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
        cdef IntColumn res = IntColumn(max(self.capacity, 1))
        res.length = self.length
        if self.length == 0: return res
        cdef int cur_min = self.data_c[0]
        cdef int* ptr = self.data_c
        cdef int* res_ptr = res.data_c
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
        cdef IntColumn res = IntColumn(max(self.capacity, 1))
        res.length = self.length
        cdef int* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i
        with nogil:
            if periods >= 0:
                for i in range(min(periods, self.length)):
                    res_ptr[i] = 0
                for i in range(periods, self.length):
                    res_ptr[i] = ptr[i] - ptr[i - periods]
            else:
                periods = -periods
                for i in range(max(0, self.length - periods), self.length):
                    res_ptr[i] = 0
                for i in range(self.length - periods):
                    res_ptr[i] = ptr[i] - ptr[i + periods]
        return res

    @cython.binding(False)
    def pct_change(self, int periods=1):
        cdef DoubleColumn res = DoubleColumn(max(self.capacity, 1))
        res.length = self.length
        cdef int* ptr = self.data_c
        cdef double* res_ptr = res.data_c
        cdef Py_ssize_t i
        cdef double prev
        with nogil:
            if periods >= 0:
                for i in range(min(periods, self.length)):
                    res_ptr[i] = NAN
                for i in range(periods, self.length):
                    prev = <double>ptr[i - periods]
                    res_ptr[i] = (<double>ptr[i] - prev) / prev if prev != 0.0 else NAN
            else:
                periods = -periods
                for i in range(max(0, self.length - periods), self.length):
                    res_ptr[i] = NAN
                for i in range(self.length - periods):
                    prev = <double>ptr[i + periods]
                    res_ptr[i] = (<double>ptr[i] - prev) / prev if prev != 0.0 else NAN
        return res

    cdef void _central_moments(self, double* out_mean, double* out_m2, double* out_m3, double* out_m4) noexcept nogil:
        cdef Py_ssize_t n = self.length
        if n == 0:
            out_mean[0] = 0.0
            out_m2[0] = 0.0
            out_m3[0] = 0.0
            out_m4[0] = 0.0
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
        cdef IntColumn res = IntColumn(max(self.capacity, 1))
        res.length = self.length
        cdef int total = 1
        cdef int* ptr = self.data_c
        cdef int* res_ptr = res.data_c
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
        return self._sum_rows(indices, n) / n

    cdef double _min_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil:
        if n == 0: return NAN
        cdef int* ptr = self.data_c
        cdef double val, min_val
        min_val = <double>ptr[indices[0]]
        cdef Py_ssize_t i
        for i in range(1, n):
            val = <double>ptr[indices[i]]
            if val < min_val: min_val = val
        return min_val

    cdef double _max_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil:
        if n == 0: return NAN
        cdef int* ptr = self.data_c
        cdef double val, max_val
        max_val = <double>ptr[indices[0]]
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
        cdef IntColumn res = IntColumn(max(self.capacity, 1))
        res.length = self.length
        cdef int* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                res_ptr[i] = ptr[i] + val
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sub_val(self, int val):
        cdef IntColumn res = IntColumn(max(self.capacity, 1))
        res.length = self.length
        cdef int* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                res_ptr[i] = ptr[i] - val
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def mul_val(self, int val):
        cdef IntColumn res = IntColumn(max(self.capacity, 1))
        res.length = self.length
        cdef int* ptr = self.data_c
        cdef int* res_ptr = res.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                res_ptr[i] = ptr[i] * val
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def div_val(self, int val):
        cdef DoubleColumn res = DoubleColumn(max(self.capacity, 1))
        res.length = self.length
        cdef int* ptr = self.data_c
        cdef double* res_ptr = res.data_c
        cdef Py_ssize_t i
        cdef double inv_val
        with nogil:
            if val != 0:
                inv_val = 1.0 / <double>val
                for i in range(self.length):
                    res_ptr[i] = <double>ptr[i] * inv_val
            else:
                for i in range(self.length):
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
            res = IntColumn(max(n, 1))
            res.length = n
            other_i_ptr = (<IntColumn>other).data_c
            with nogil:
                for i in range(n):
                    res.data_c[i] = self_ptr[i] + other_i_ptr[i]
            return res
        elif isinstance(other, DoubleColumn):
            dbl = DoubleColumn(max(n, 1))
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
            res = IntColumn(max(n, 1))
            res.length = n
            other_i_ptr = (<IntColumn>other).data_c
            with nogil:
                for i in range(n):
                    res.data_c[i] = self_ptr[i] - other_i_ptr[i]
            return res
        elif isinstance(other, DoubleColumn):
            dbl = DoubleColumn(max(n, 1))
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
            res = IntColumn(max(n, 1))
            res.length = n
            other_i_ptr = (<IntColumn>other).data_c
            with nogil:
                for i in range(n):
                    res.data_c[i] = self_ptr[i] * other_i_ptr[i]
            return res
        elif isinstance(other, DoubleColumn):
            dbl = DoubleColumn(max(n, 1))
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
        cdef DoubleColumn res = DoubleColumn(max(n, 1))
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
        if isinstance(other, int):
            return self.add_val(other)
        if isinstance(other, (IntColumn, DoubleColumn)):
            return self.add(other)
        return NotImplemented

    def __radd__(self, object other):
        return self.__add__(other)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __sub__(self, object other):
        if isinstance(other, int):
            return self.sub_val(other)
        if isinstance(other, (IntColumn, DoubleColumn)):
            return self.sub(other)
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
            res = IntColumn(max(n, 1))
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
        if isinstance(other, int):
            return self.mul_val(other)
        if isinstance(other, (IntColumn, DoubleColumn)):
            return self.mul(other)
        return NotImplemented

    def __rmul__(self, object other):
        return self.__mul__(other)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __truediv__(self, object other):
        if isinstance(other, int):
            return self.div_val(other)
        if isinstance(other, (IntColumn, DoubleColumn)):
            return self.div(other)
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
            res = DoubleColumn(max(n, 1))
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

        cdef Py_ssize_t new_cap = max(n, 1)
        cdef int* new_data = <int*>aligned_malloc(new_cap * sizeof(int))
        if not new_data: raise MemoryError()
        cdef Py_ssize_t i, idx
        cdef int* self_data = self.data_c
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
    def unique(self):
        cdef set seen = set()
        cdef list result = []
        cdef Py_ssize_t i
        cdef int val
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
            raise TypeError("Bitwise AND only supported between two IntColumn masks")
        cdef IntColumn b = <IntColumn>other
        if self.length != b.length:
            raise ValueError(f"Length mismatch: {self.length} vs {b.length}")
        cdef IntColumn res = IntColumn(max(self.capacity, 1))
        res.length = self.length
        cdef int* a_ptr = self.data_c
        cdef int* b_ptr = b.data_c
        cdef int* r_ptr = res.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
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
        cdef IntColumn res = IntColumn(max(self.capacity, 1))
        res.length = self.length
        cdef int* a_ptr = self.data_c
        cdef int* b_ptr = b.data_c
        cdef int* r_ptr = res.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                r_ptr[i] = (a_ptr[i] != 0) | (b_ptr[i] != 0)
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __invert__(self):
        cdef IntColumn res = IntColumn(max(self.capacity, 1))
        res.length = self.length
        cdef int* s_ptr = self.data_c
        cdef int* r_ptr = res.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                r_ptr[i] = (s_ptr[i] == 0)
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def isin(self, set values):
        cdef IntColumn res = IntColumn(max(self.capacity, 1))
        res.length = self.length
        cdef Py_ssize_t i, n = self.length
        cdef int* s_ptr = self.data_c
        cdef int* r_ptr = res.data_c
        cdef Py_ssize_t m = len(values)

        if m == 0:
            with nogil:
                for i in range(n): r_ptr[i] = 0
            return res

        cdef int* v_arr = <int*>aligned_malloc(m * sizeof(int))
        if not v_arr: raise MemoryError()
        cdef Py_ssize_t valid_m = 0
        cdef void* found
        cdef int single_val

        try:
            for v in values:
                if v is not None:
                    v_arr[valid_m] = <int>v
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
                qsort(v_arr, valid_m, sizeof(int), _cmp_int)
                with nogil:
                    for i in range(n):
                        found = bsearch(&s_ptr[i], v_arr, valid_m, sizeof(int), _cmp_int)
                        r_ptr[i] = 1 if found != NULL else 0
        finally:
            os_aligned_free(v_arr)

        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def between(self, int low, int high):
        cdef IntColumn res = IntColumn(max(self.capacity, 1))
        res.length = self.length
        cdef int* s_ptr = self.data_c
        cdef int* r_ptr = res.data_c
        cdef Py_ssize_t i
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
        cdef Py_ssize_t i
        cdef int* old_data = self.data_c
        cdef IntColumn new_col
        cdef int* new_data

        if not inplace:
            new_col = IntColumn(max(n, 1))
            new_col.length = n
            with nogil:
                for i in range(n):
                    new_col.data_c[i] = old_data[indices[i]]
            return new_col
        else:
            new_data = <int*>aligned_malloc(max(n, self.capacity) * sizeof(int))
            if not new_data: raise MemoryError()
            with nogil:
                for i in range(n):
                    new_data[i] = old_data[indices[i]]
            os_aligned_free(self.data_c)
            self.data_c = new_data
            self.capacity = max(n, self.capacity)
            self.length = n
            return self

    cpdef IntColumn clip(self, double lower, double upper, bint inplace=False):
        cdef IntColumn col = self if inplace else self.copy()
        cdef int* ptr = col.data_c
        cdef Py_ssize_t i
        cdef int lo = <int>lower
        cdef int hi = <int>upper
        with nogil:
            for i in range(col.length):
                if ptr[i] < lo:
                    ptr[i] = lo
                elif ptr[i] > hi:
                    ptr[i] = hi
        return col

    cpdef IntColumn where(self, IntColumn cond, int other, bint inplace=False):
        cdef Py_ssize_t n = self.length
        cdef char err_buf[128]
        if cond.length != n:
            snprintf(err_buf, sizeof(err_buf), "Condition mask length %zd must match column length %zd", cond.length, n)
            raise ValueError(err_buf.decode('ascii'))

        cdef IntColumn col = self if inplace else IntColumn(max(self.capacity, 1))
        if not inplace: col.length = n

        cdef int* col_ptr = col.data_c
        cdef int* self_ptr = self.data_c
        cdef int* cond_ptr = cond.data_c
        cdef Py_ssize_t i
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

        cdef IntColumn col = IntColumn(max(self.capacity, 1))
        col.length = n
        cdef int fill_val = <int>fill
        cdef int* col_ptr = col.data_c
        cdef int* self_ptr = self.data_c
        cdef int* mask_ptr = mask.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                col_ptr[i] = self_ptr[i] if mask_ptr[i] != 0 else fill_val
        return col

    cpdef IntColumn mask(self, IntColumn cond, int other, bint inplace=False):
        cdef Py_ssize_t n = self.length
        cdef char err_buf[128]
        if cond.length != n:
            snprintf(err_buf, sizeof(err_buf), "Condition mask length %zd must match column length %zd", cond.length, n)
            raise ValueError(err_buf.decode('ascii'))

        cdef IntColumn col = self if inplace else IntColumn(max(self.capacity, 1))
        if not inplace: col.length = n

        cdef int* col_ptr = col.data_c
        cdef int* self_ptr = self.data_c
        cdef int* cond_ptr = cond.data_c
        cdef Py_ssize_t i
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
                    rem_len * sizeof(int))
        self.length -= 1

    cdef void to_double_buffer(self, double* out, Py_ssize_t n):
        cdef int* ptr = self.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                out[i] = <double>ptr[i]

    cdef IntColumn gather_with_sentinel(self, Py_ssize_t* indices, Py_ssize_t n, int sentinel):
        cdef IntColumn new_col = IntColumn(max(n, 1))
        new_col.length = n
        cdef int* res_ptr = new_col.data_c
        cdef int* src_ptr = self.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                res_ptr[i] = src_ptr[indices[i]] if indices[i] >= 0 else sentinel
        return new_col

    cpdef Py_ssize_t count_nonzero(self):
        return self.count_nonzeros()

    cdef void gather_zero_to(self, Py_ssize_t* out):
        """Gather indices where value == 0 into out[]"""
        cdef Py_ssize_t j = 0
        cdef int* ptr = self.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                if ptr[i] == 0:
                    out[j] = i
                    j += 1

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def astype(self, object target):
        cdef int target_type = _resolve_astype_target(target)
        cdef Py_ssize_t i, n = self.length
        cdef Py_ssize_t safe_cap = max(n, 1)
        cdef DoubleColumn dc
        cdef BoolColumn bc
        cdef UCS2Column_SoA uc
        cdef Char256Column_SoA cc
        cdef DateColumn tc
        cdef CategoryColumn catc
        cdef DateTimeColumn dtc
        cdef TimeColumn tmc
        cdef int ival, y, m, d, sz
        cdef char buf[64]
        cdef int* s_ptr = self.data_c

        if target_type == TYPE_DOUBLE:
            dc = DoubleColumn(safe_cap)
            dc.length = n
            with nogil:
                for i in range(n):
                    dc.data_c[i] = <double>s_ptr[i]
            return dc

        elif target_type == TYPE_BOOL:
            bc = BoolColumn(safe_cap)
            bc.length = n
            with nogil:
                for i in range(n):
                    bc.data_c[i] = (s_ptr[i] != 0)
            return bc

        elif target_type == TYPE_UCS2:
            uc = UCS2Column_SoA(safe_cap)
            uc.length = n
            for i in range(n):
                sz = snprintf(buf, sizeof(buf), "%d", s_ptr[i])
                uc._set_item(i, buf[:sz].decode('ascii'))
            return uc

        elif target_type == TYPE_CHAR256:
            cc = Char256Column_SoA(safe_cap)
            for i in range(n):
                sz = snprintf(buf, sizeof(buf), "%d", s_ptr[i])
                cc.append(buf[:sz].decode('ascii'))
            return cc

        elif target_type == TYPE_TIMESTAMP:
            tc = DateColumn(safe_cap)
            tc.length = n
            with nogil:
                for i in range(n):
                    ival = s_ptr[i]
                    if ival > 10000000:
                        tc.data_c[i] = _ymd_to_days(ival // 10000, (ival // 100) % 100, ival % 100)
                    else:
                        tc.data_c[i] = ival
            return tc

        elif target_type == TYPE_CATEGORY:
            catc = CategoryColumn(safe_cap)
            for i in range(n):
                sz = snprintf(buf, sizeof(buf), "%d", s_ptr[i])
                catc.append(buf[:sz].decode('ascii'))
            return catc

        elif target_type == TYPE_DATETIME:
            dtc = DateTimeColumn(safe_cap)
            for i in range(n):
                ival = s_ptr[i]
                y = ival // 10000
                m = (ival // 100) % 100
                d = ival % 100
                sz = snprintf(buf, sizeof(buf), "%04d-%02d-%02d", y, m, d)
                dtc.append(buf[:sz].decode('ascii'))
            return dtc

        elif target_type == TYPE_TIME:
            tmc = TimeColumn(safe_cap)
            for i in range(n):
                tmc.append(s_ptr[i])
            return tmc

        elif target_type == TYPE_INT:
            return self.copy()

        cdef char err_buf[128]
        snprintf(err_buf, sizeof(err_buf), "Failed to cast INT column to internal type id: %d", target_type)
        raise TypeError(err_buf.decode('ascii'))