# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport cython
from libc.string cimport memcpy, memmove
from libc.math cimport sqrt, fabs, NAN
from fluidframe.core._base cimport TYPE_INT, BaseColumn, aligned_malloc, aligned_realloc, os_aligned_free
from fluidframe.core.doublecolumn cimport DoubleColumn


@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
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
        cdef IntColumn new_col = IntColumn(self.capacity)
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
        memmove(self.data_c, self.data_c + 1, (self.length - 1) * sizeof(int))
        self.length -= 1
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
        for i in range(1, self.length):
            if self.data_c[i] > max_val: max_val = self.data_c[i]
        return max_val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def min(self):
        if self.length == 0: raise ValueError("empty column")
        cdef int min_val = self.data_c[0]
        for i in range(1, self.length):
            if self.data_c[i] < min_val: min_val = self.data_c[i]
        return min_val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def mean(self):
        if self.length == 0: return NAN
        cdef double sum_val = 0
        for i in range(self.length):
            sum_val += self.data_c[i]
        return sum_val / self.length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def var(self):
        if self.length < 2: return 0.0
        cdef double m = self.mean()
        cdef double sum_sq = 0
        for i in range(self.length):
            sum_sq += (self.data_c[i] - m) * (self.data_c[i] - m)
        return sum_sq / (self.length - 1)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def std(self):
        return sqrt(self.var())

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def count_nonzeros(self):
        cdef Py_ssize_t count = 0
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                if self.data_c[i] != 0: count += 1
        return count

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sum(self):
        cdef double total = 0
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                total += self.data_c[i]
        return total

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def idxmax(self):
        if self.length == 0: raise ValueError("empty column")
        cdef Py_ssize_t max_idx = 0
        cdef int max_val = self.data_c[0]
        cdef Py_ssize_t i
        with nogil:
            for i in range(1, self.length):
                if self.data_c[i] > max_val:
                    max_val = self.data_c[i]
                    max_idx = i
        return max_idx

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def idxmin(self):
        if self.length == 0: raise ValueError("empty column")
        cdef Py_ssize_t min_idx = 0
        cdef int min_val = self.data_c[0]
        cdef Py_ssize_t i
        with nogil:
            for i in range(1, self.length):
                if self.data_c[i] < min_val:
                    min_val = self.data_c[i]
                    min_idx = i
        return min_idx

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def argmax(self): return self.idxmax()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def argmin(self): return self.idxmin()

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
            return (v1 + v2) / 2.0
        result = self._quickselect(temp, self.length, mid)
        os_aligned_free(temp)
        return result

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def mad(self):
        if self.length == 0: return NAN
        cdef double m = self.mean()
        cdef double total_dev = 0
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                total_dev += fabs(<double>self.data_c[i] - m)
        return total_dev / self.length

    # ==========================================
    # Whole-column statistical methods
    # ==========================================
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def prod(self):
        if self.length == 0: return NAN
        cdef double total = 1
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                total *= self.data_c[i]
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
            if val in counts:
                counts[val] += 1
            else:
                counts[val] = 1
        return counts

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def mode(self):
        if self.length == 0: return []
        cdef dict counts = {}
        cdef Py_ssize_t i
        cdef int val
        cdef int max_count = 0
        for i in range(self.length):
            val = self.data_c[i]
            if val in counts:
                counts[val] += 1
            else:
                counts[val] = 1
            if counts[val] > max_count:
                max_count = counts[val]
        return [k for k, v in counts.items() if v == max_count]

    cdef double _quickselect(self, int* data, Py_ssize_t n, Py_ssize_t k) noexcept nogil:
        """Find k-th smallest element using quickselect with median-of-three pivot."""
        cdef Py_ssize_t left = 0, right = n - 1, i, j, mid
        cdef int pivot, tmp
        while left < right:
            i = left
            j = right
            mid = left + (right - left) // 2
            if data[left] > data[mid]:
                tmp = data[left]
                data[left] = data[mid]
                data[mid] = tmp
            if data[left] > data[right]:
                tmp = data[left]
                data[left] = data[right]
                data[right] = tmp
            if data[mid] > data[right]:
                tmp = data[mid]
                data[mid] = data[right]
                data[right] = tmp
            pivot = data[mid]
            while i <= j:
                while data[i] < pivot:
                    i += 1
                while data[j] > pivot:
                    j -= 1
                if i <= j:
                    tmp = data[i]
                    data[i] = data[j]
                    data[j] = tmp
                    i += 1
                    j -= 1
            if j < k:
                left = i
            if k < i:
                right = j
        return <double>data[k]

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def quantile(self, double q):
        if self.length == 0: return NAN
        if q < 0 or q > 1: raise ValueError("q must be between 0 and 1")
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
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef double total = 0
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                total += self.data_c[i]
                res.data_c[i] = <int>total
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cummax(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        if self.length == 0: return res
        cdef int cur_max = self.data_c[0]
        cdef Py_ssize_t i
        with nogil:
            res.data_c[0] = cur_max
            for i in range(1, self.length):
                if self.data_c[i] > cur_max:
                    cur_max = self.data_c[i]
                res.data_c[i] = cur_max
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cummin(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        if self.length == 0: return res
        cdef int cur_min = self.data_c[0]
        cdef Py_ssize_t i
        with nogil:
            res.data_c[0] = cur_min
            for i in range(1, self.length):
                if self.data_c[i] < cur_min:
                    cur_min = self.data_c[i]
                res.data_c[i] = cur_min
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def diff(self, int periods=1):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        with nogil:
            if periods >= 0:
                for i in range(min(periods, self.length)):
                    res.data_c[i] = 0
                for i in range(periods, self.length):
                    res.data_c[i] = self.data_c[i] - self.data_c[i - periods]
            else:
                periods = -periods
                for i in range(max(0, self.length - periods), self.length):
                    res.data_c[i] = 0
                for i in range(self.length - periods):
                    res.data_c[i] = self.data_c[i] - self.data_c[i + periods]
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pct_change(self, int periods=1):
        cdef DoubleColumn res = DoubleColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        cdef double prev
        with nogil:
            if periods >= 0:
                for i in range(min(periods, self.length)):
                    res.data_c[i] = NAN
                for i in range(periods, self.length):
                    prev = <double>self.data_c[i - periods]
                    res.data_c[i] = (<double>self.data_c[i] - prev) / prev if prev != 0 else NAN
            else:
                periods = -periods
                for i in range(max(0, self.length - periods), self.length):
                    res.data_c[i] = NAN
                for i in range(self.length - periods):
                    prev = <double>self.data_c[i + periods]
                    res.data_c[i] = (<double>self.data_c[i] - prev) / prev if prev != 0 else NAN
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
        cdef Py_ssize_t n1
        for i in range(n):
            n1 = i
            val = <double>self.data_c[i]
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
        if m2 == 0: return NAN
        cdef double n = <double>self.length
        return (n * m3) / ((n - 1) * (n - 2) * (m2 / (n - 1)) ** 1.5)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def kurt(self):
        if self.length < 4: return NAN
        cdef double m, m2, m3, m4
        self._central_moments(&m, &m2, &m3, &m4)
        if m2 == 0: return NAN
        cdef double n = <double>self.length
        cdef double var_sample = m2 / (n - 1)
        cdef double kurt = (m4 / (n - 1)) / (var_sample * var_sample) - 3.0
        return kurt

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sem(self):
        if self.length < 2: return NAN
        cdef double m, m2, m3, m4
        self._central_moments(&m, &m2, &m3, &m4)
        return sqrt(m2 / ((<double>self.length - 1) * <double>self.length))

    # ==========================================
    # Subset aggregation (nogil, for groupby)
    # ==========================================
    cdef double _sum_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil:
        cdef double total = 0
        cdef Py_ssize_t i
        for i in range(n):
            total += self.data_c[indices[i]]
        return total

    cdef double _mean_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil:
        if n == 0: return NAN
        cdef double total = 0
        cdef Py_ssize_t i
        for i in range(n):
            total += self.data_c[indices[i]]
        return total / n

    cdef double _min_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil:
        if n == 0: return NAN
        cdef double val, min_val
        min_val = <double>self.data_c[indices[0]]
        cdef Py_ssize_t i
        for i in range(1, n):
            val = <double>self.data_c[indices[i]]
            if val < min_val: min_val = val
        return min_val

    cdef double _max_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil:
        if n == 0: return NAN
        cdef double val, max_val
        max_val = <double>self.data_c[indices[0]]
        cdef Py_ssize_t i
        for i in range(1, n):
            val = <double>self.data_c[indices[i]]
            if val > max_val: max_val = val
        return max_val

    cdef Py_ssize_t _count_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil:
        return n

    cdef double _var_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil:
        if n < 2: return 0.0
        cdef double m = self._mean_rows(indices, n)
        cdef double sum_sq = 0
        cdef double v
        cdef Py_ssize_t i
        for i in range(n):
            v = <double>self.data_c[indices[i]] - m
            sum_sq += v * v
        return sum_sq / (n - 1)

    cdef double _std_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil:
        return sqrt(self._var_rows(indices, n))

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def add(self, int val):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length): res.data_c[i] = self.data_c[i] + val
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sub(self, int val):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length): res.data_c[i] = self.data_c[i] - val
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def mul(self, int val):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length): res.data_c[i] = self.data_c[i] * val
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def div(self, int val):
        cdef DoubleColumn res = DoubleColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                res.data_c[i] = <double>self.data_c[i] / val if val != 0 else NAN
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _reorder(self, list indices):
        cdef Py_ssize_t n = len(indices)
        if n != self.length:
            raise ValueError("Length of indices must match column length")

        cdef int* new_data = <int*>aligned_malloc(n * sizeof(int))
        if not new_data: raise MemoryError()
        cdef Py_ssize_t i, idx
        try:
            for i in range(n):
                idx = <Py_ssize_t>indices[i]
                if idx < 0 or idx >= self.length:
                    raise IndexError("Index out of bounds")
                new_data[i] = self.data_c[idx]
        except Exception:
            os_aligned_free(new_data)
            raise
        os_aligned_free(self.data_c)
        self.data_c = new_data

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __getitem__(self, Py_ssize_t idx):
        if idx < 0: idx += self.length
        if idx < 0 or idx >= self.length: raise IndexError()
        return self.data_c[idx]

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def unique(self):
        cdef set seen = set()
        cdef list result = []
        cdef Py_ssize_t i
        for i in range(self.length):
            val = self.data_c[i]
            if val not in seen:
                seen.add(val)
                result.append(val)
        return result

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _cmp_op(self, object val, int op):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        cdef int v
        if isinstance(val, IntColumn):
            other = <IntColumn>val
            if other.length != self.length:
                raise ValueError(f"Length mismatch: {self.length} vs {other.length}")
            if op == 0:
                for i in range(self.length): res.data_c[i] = 1 if self.data_c[i] < other.data_c[i] else 0
            elif op == 1:
                for i in range(self.length): res.data_c[i] = 1 if self.data_c[i] <= other.data_c[i] else 0
            elif op == 2:
                for i in range(self.length): res.data_c[i] = 1 if self.data_c[i] > other.data_c[i] else 0
            elif op == 3:
                for i in range(self.length): res.data_c[i] = 1 if self.data_c[i] >= other.data_c[i] else 0
            elif op == 4:
                for i in range(self.length): res.data_c[i] = 1 if self.data_c[i] == other.data_c[i] else 0
            elif op == 5:
                for i in range(self.length): res.data_c[i] = 1 if self.data_c[i] != other.data_c[i] else 0
        else:
            v = <int>val
            if op == 0:
                for i in range(self.length): res.data_c[i] = 1 if self.data_c[i] < v else 0
            elif op == 1:
                for i in range(self.length): res.data_c[i] = 1 if self.data_c[i] <= v else 0
            elif op == 2:
                for i in range(self.length): res.data_c[i] = 1 if self.data_c[i] > v else 0
            elif op == 3:
                for i in range(self.length): res.data_c[i] = 1 if self.data_c[i] >= v else 0
            elif op == 4:
                for i in range(self.length): res.data_c[i] = 1 if self.data_c[i] == v else 0
            elif op == 5:
                for i in range(self.length): res.data_c[i] = 1 if self.data_c[i] != v else 0
        return res

    def __lt__(self, val):  return self._cmp_op(val, 0)
    def __le__(self, val):  return self._cmp_op(val, 1)
    def __gt__(self, val):  return self._cmp_op(val, 2)
    def __ge__(self, val):  return self._cmp_op(val, 3)
    def __eq__(self, val):  return self._cmp_op(val, 4)
    def __ne__(self, val):  return self._cmp_op(val, 5)

    def __and__(self, other):
        if not isinstance(other, IntColumn):
            raise TypeError("Bitwise AND only supported between two IntColumn masks")
        cdef IntColumn a = <IntColumn>self
        cdef IntColumn b = <IntColumn>other
        if a.length != b.length:
            raise ValueError(f"Length mismatch: {a.length} vs {b.length}")
        cdef IntColumn res = IntColumn(a.capacity)
        res.length = a.length
        cdef Py_ssize_t i
        for i in range(a.length):
            res.data_c[i] = 1 if (a.data_c[i] != 0 and b.data_c[i] != 0) else 0
        return res

    def __or__(self, other):
        if not isinstance(other, IntColumn):
            raise TypeError("Bitwise OR only supported between two IntColumn masks")
        cdef IntColumn a = <IntColumn>self
        cdef IntColumn b = <IntColumn>other
        if a.length != b.length:
            raise ValueError(f"Length mismatch: {a.length} vs {b.length}")
        cdef IntColumn res = IntColumn(a.capacity)
        res.length = a.length
        cdef Py_ssize_t i
        for i in range(a.length):
            res.data_c[i] = 1 if (a.data_c[i] != 0 or b.data_c[i] != 0) else 0
        return res

    def __invert__(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        for i in range(self.length):
            res.data_c[i] = 0 if self.data_c[i] != 0 else 1
        return res

    def isin(self, set values):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        for i in range(self.length):
            res.data_c[i] = 1 if self.data_c[i] in values else 0
        return res

    def between(self, int low, int high):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        for i in range(self.length):
            res.data_c[i] = 1 if low <= self.data_c[i] <= high else 0
        return res

    def __str__(self):
        cdef list items = []
        cdef Py_ssize_t i
        for i in range(self.length):
            items.append(str(self.data_c[i]))
        return f"IntColumn([{', '.join(items)}])"

    def __bool__(self):
        return self.length > 0

    def __len__(self):
        return self.length

    cdef IntColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=False):
        cdef Py_ssize_t i
        if not inplace:
            new_col = IntColumn(max(n, 1))
            new_col.length = n
            with nogil:
                for i in range(n):
                    new_col.data_c[i] = self.data_c[indices[i]]
            return new_col
        else:
            new_data = <int*>aligned_malloc(max(n, self.capacity) * sizeof(int))
            if not new_data: raise MemoryError()
            with nogil:
                for i in range(n):
                    new_data[i] = self.data_c[indices[i]]
            os_aligned_free(self.data_c)
            self.data_c = new_data
            self.capacity = max(n, self.capacity)
            self.length = n
            return self

    cpdef IntColumn clip(self, double lower, double upper, bint inplace=False):
        cdef IntColumn col = self if inplace else self.copy()
        cdef Py_ssize_t i
        cdef int lo = <int>lower
        cdef int hi = <int>upper
        for i in range(col.length):
            if col.data_c[i] < lo:
                col.data_c[i] = lo
            elif col.data_c[i] > hi:
                col.data_c[i] = hi
        return col

    cpdef IntColumn where(self, IntColumn cond, int other, bint inplace=False):
        cdef Py_ssize_t n = self.length
        if cond.length != n:
            raise ValueError("Condition mask length must match column length")

        cdef IntColumn col
        if inplace:
            col = self
        else:
            col = IntColumn(self.capacity)
            col.length = n
            memcpy(col.data_c, self.data_c, n * sizeof(int))
        cdef Py_ssize_t i
        for i in range(n):
            if not cond.data_c[i]:
                col.data_c[i] = other
        return col

    cpdef IntColumn mask(self, IntColumn cond, int other, bint inplace=False):
        cdef Py_ssize_t n = self.length
        if cond.length != n:
            raise ValueError("Condition mask length must match column length")

        cdef IntColumn col
        if inplace:
            col = self
        else:
            col = IntColumn(self.capacity)
            col.length = n
            memcpy(col.data_c, self.data_c, n * sizeof(int))
        cdef Py_ssize_t i
        for i in range(n):
            if cond.data_c[i]:
                col.data_c[i] = other
        return col

    cpdef void drop_row(self, Py_ssize_t row_idx, Py_ssize_t rem_len):
        if rem_len > 0:
            memmove(self.data_c + row_idx,
                    self.data_c + row_idx + 1,
                    rem_len * sizeof(int))
        self.length -= 1

    cdef void to_double_buffer(self, double* out, Py_ssize_t n):
        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                out[i] = <double>self.data_c[i]

    cdef IntColumn gather_with_sentinel(self, Py_ssize_t* indices, Py_ssize_t n, int sentinel):
        cdef IntColumn new_col = IntColumn(n)
        new_col.length = n
        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                if indices[i] >= 0:
                    new_col.data_c[i] = self.data_c[indices[i]]
                else:
                    new_col.data_c[i] = sentinel
        return new_col

    cpdef Py_ssize_t count_nonzero(self):
        cdef Py_ssize_t count = 0
        cdef Py_ssize_t i
        for i in range(self.length):
            if self.data_c[i] != 0:
                count += 1
        return count

    cdef void gather_zero_to(self, Py_ssize_t* out):
        """Gather indices where value == 0 into out[]"""
        cdef Py_ssize_t j = 0
        cdef Py_ssize_t i
        for i in range(self.length):
            if self.data_c[i] == 0:
                out[j] = i
                j += 1