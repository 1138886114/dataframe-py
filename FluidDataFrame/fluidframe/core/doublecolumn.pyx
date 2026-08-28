# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport cython
from libc.string cimport memcpy, memmove
from libc.stdlib cimport free
from libc.math cimport sqrt, fabs, NAN

from fluidframe.core._base cimport BaseColumn, aligned_malloc, aligned_realloc, TYPE_DOUBLE, TYPE_INT
from fluidframe.core.intcolumn cimport IntColumn

cdef double _c_corr(double* a, double* b, Py_ssize_t n) noexcept nogil:
    if n == 0: return 0.0
    cdef double mean_i = 0.0, mean_j = 0.0
    cdef double cov = 0.0, std_i = 0.0, std_j = 0.0
    cdef double xi, xj
    cdef Py_ssize_t k
    for k in range(n):
        mean_i += a[k]
        mean_j += b[k]
    mean_i /= n
    mean_j /= n
    for k in range(n):
        xi = a[k] - mean_i
        xj = b[k] - mean_j
        cov += xi * xj
        std_i += xi * xi
        std_j += xj * xj
    if std_i > 0 and std_j > 0:
        return cov / (std_i * std_j) ** 0.5
    return 0.0

cdef double _c_cov(double* a, double* b, Py_ssize_t n) noexcept nogil:
    if n == 0: return 0.0
    cdef double mean_i = 0.0, mean_j = 0.0
    cdef double cov_val = 0.0
    cdef Py_ssize_t k
    for k in range(n):
        mean_i += a[k]
        mean_j += b[k]
    mean_i /= n
    mean_j /= n
    for k in range(n):
        cov_val += (a[k] - mean_i) * (b[k] - mean_j)
    return cov_val / n

@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
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
        if self.data_c is not NULL: free(self.data_c)

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
        cdef DoubleColumn new_col = DoubleColumn(self.capacity)
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
        memmove(self.data_c, self.data_c + 1, (self.length - 1) * sizeof(double))
        self.length -= 1
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
        for i in range(1, self.length):
            if self.data_c[i] > max_val: max_val = self.data_c[i]
        return max_val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def min(self):
        if self.length == 0: raise ValueError("empty column")
        cdef double min_val = self.data_c[0]
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
    def _reorder(self, list indices):
        cdef double* new_data = <double*>aligned_malloc(self.capacity * sizeof(double))
        if not new_data: raise MemoryError()
        cdef Py_ssize_t idx
        try:
            for i in range(self.length):
                idx = indices[i]
                if idx < 0 or idx >= self.length:
                    raise IndexError()
                new_data[i] = self.data_c[idx]
        except Exception:
            free(new_data)
            raise
        self.data_c = new_data

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __getitem__(self, Py_ssize_t idx):
        if idx < 0 or idx >= self.length: raise IndexError()
        return self.data_c[idx]

    @cython.binding(False)
    def count_nonzeros(self):
        cdef Py_ssize_t count = 0
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                if self.data_c[i] != 0: count += 1
        return count

    @cython.binding(False)
    def sum(self):
        cdef double total = 0
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                total += self.data_c[i]
        return total

    @cython.binding(False)
    def idxmax(self):
        if self.length == 0: raise ValueError("empty column")
        cdef Py_ssize_t max_idx = 0
        cdef double max_val = self.data_c[0]
        cdef Py_ssize_t i
        with nogil:
            for i in range(1, self.length):
                if self.data_c[i] > max_val:
                    max_val = self.data_c[i]
                    max_idx = i
        return max_idx

    @cython.binding(False)
    def idxmin(self):
        if self.length == 0: raise ValueError("empty column")
        cdef Py_ssize_t min_idx = 0
        cdef double min_val = self.data_c[0]
        cdef Py_ssize_t i
        with nogil:
            for i in range(1, self.length):
                if self.data_c[i] < min_val:
                    min_val = self.data_c[i]
                    min_idx = i
        return min_idx

    @cython.binding(False)
    def argmax(self): return self.idxmax()

    @cython.binding(False)
    def argmin(self): return self.idxmin()

    @cython.binding(False)
    def median(self):
        if self.length == 0: return NAN
        cdef double* temp = <double*>aligned_malloc(self.length * sizeof(double))
        if not temp: raise MemoryError()
        memcpy(temp, self.data_c, self.length * sizeof(double))
        cdef list py_list = [temp[i] for i in range(self.length)]
        free(temp)
        py_list.sort()
        cdef Py_ssize_t mid = self.length // 2
        if self.length % 2 == 0:
            return (py_list[mid - 1] + py_list[mid]) / 2.0
        return py_list[mid]

    @cython.binding(False)
    def mad(self):
        if self.length == 0: return NAN
        cdef double m = self.mean()
        cdef double total_dev = 0
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                total_dev += fabs(self.data_c[i] - m)
        return total_dev / self.length

    # ==========================================
    # Whole-column statistical methods
    # ==========================================
    @cython.binding(False)
    def prod(self):
        if self.length == 0: return NAN
        cdef double total = 1
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                total *= self.data_c[i]
        return total

    @cython.binding(False)
    def nunique(self):
        if self.length == 0: return 0
        cdef set seen = set()
        cdef Py_ssize_t i
        for i in range(self.length):
            seen.add(self.data_c[i])
        return len(seen)

    @cython.binding(False)
    def value_counts(self):
        cdef dict counts = {}
        cdef Py_ssize_t i
        cdef double val
        for i in range(self.length):
            val = self.data_c[i]
            if val in counts:
                counts[val] += 1
            else:
                counts[val] = 1
        return counts

    @cython.binding(False)
    def mode(self):
        if self.length == 0: return []
        cdef dict counts = {}
        cdef Py_ssize_t i
        cdef double val
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

    @cython.binding(False)
    def quantile(self, double q):
        if self.length == 0: return NAN
        if q < 0 or q > 1: raise ValueError("q must be between 0 and 1")
        cdef double* temp = <double*>aligned_malloc(self.length * sizeof(double))
        if not temp: raise MemoryError()
        memcpy(temp, self.data_c, self.length * sizeof(double))
        cdef list py_list = [temp[i] for i in range(self.length)]
        free(temp)
        py_list.sort()
        cdef double pos = q * (self.length - 1)
        cdef Py_ssize_t idx = <Py_ssize_t>pos
        cdef double frac = pos - idx
        if idx >= self.length - 1:
            return py_list[self.length - 1]
        return py_list[idx] + frac * (py_list[idx + 1] - py_list[idx])

    @cython.binding(False)
    def cumsum(self):
        cdef DoubleColumn res = DoubleColumn(self.capacity)
        res.length = self.length
        cdef double total = 0
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                total += self.data_c[i]
                res.data_c[i] = total
        return res

    @cython.binding(False)
    def cummax(self):
        cdef DoubleColumn res = DoubleColumn(self.capacity)
        res.length = self.length
        if self.length == 0: return res
        cdef double cur_max = self.data_c[0]
        cdef Py_ssize_t i
        with nogil:
            res.data_c[0] = cur_max
            for i in range(1, self.length):
                if self.data_c[i] > cur_max:
                    cur_max = self.data_c[i]
                res.data_c[i] = cur_max
        return res

    @cython.binding(False)
    def cummin(self):
        cdef DoubleColumn res = DoubleColumn(self.capacity)
        res.length = self.length
        if self.length == 0: return res
        cdef double cur_min = self.data_c[0]
        cdef Py_ssize_t i
        with nogil:
            res.data_c[0] = cur_min
            for i in range(1, self.length):
                if self.data_c[i] < cur_min:
                    cur_min = self.data_c[i]
                res.data_c[i] = cur_min
        return res

    @cython.binding(False)
    def diff(self, int periods=1):
        cdef DoubleColumn res = DoubleColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        with nogil:
            for i in range(periods):
                res.data_c[i] = NAN
            for i in range(periods, self.length):
                res.data_c[i] = self.data_c[i] - self.data_c[i - periods]
        return res

    @cython.binding(False)
    def pct_change(self, int periods=1):
        cdef DoubleColumn res = DoubleColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        cdef double prev
        with nogil:
            for i in range(periods):
                res.data_c[i] = NAN
            for i in range(periods, self.length):
                prev = self.data_c[i - periods]
                if prev == 0:
                    res.data_c[i] = NAN
                else:
                    res.data_c[i] = (self.data_c[i] - prev) / prev
        return res

    @cython.binding(False)
    def skew(self):
        if self.length < 3: return NAN
        cdef double m = self.mean()
        cdef double s = self.std()
        if s == 0: return NAN
        cdef double sum_cube = 0
        cdef Py_ssize_t i
        cdef double v
        for i in range(self.length):
            v = (self.data_c[i] - m) / s
            sum_cube += v * v * v
        cdef double n = <double>self.length
        return (n / ((n - 1) * (n - 2))) * sum_cube

    @cython.binding(False)
    def kurt(self):
        if self.length < 4: return NAN
        cdef double m = self.mean()
        cdef double s = self.std()
        if s == 0: return NAN
        cdef double sum_pow4 = 0
        cdef Py_ssize_t i
        cdef double v
        for i in range(self.length):
            v = (self.data_c[i] - m) / s
            sum_pow4 += v * v * v * v
        cdef double n = <double>self.length
        cdef double kurt = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * sum_pow4
        kurt -= 3.0 * (n - 1) * (n - 1) / ((n - 2) * (n - 3))
        return kurt

    @cython.binding(False)
    def sem(self):
        if self.length == 0: return NAN
        return self.std() / sqrt(<double>self.length)

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
        min_val = self.data_c[indices[0]]
        cdef Py_ssize_t i
        for i in range(1, n):
            val = self.data_c[indices[i]]
            if val < min_val: min_val = val
        return min_val

    cdef double _max_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil:
        if n == 0: return NAN
        cdef double val, max_val
        max_val = self.data_c[indices[0]]
        cdef Py_ssize_t i
        for i in range(1, n):
            val = self.data_c[indices[i]]
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
            v = self.data_c[indices[i]] - m
            sum_sq += v * v
        return sum_sq / (n - 1)

    cdef double _std_rows(self, Py_ssize_t* indices, Py_ssize_t n) noexcept nogil:
        return sqrt(self._var_rows(indices, n))

    @cython.binding(False)
    def add(self, double val, double fill_value=0.0):
        cdef DoubleColumn res = DoubleColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length): res.data_c[i] = self.data_c[i] + val
        return res

    @cython.binding(False)
    def sub(self, double val, double fill_value=0.0):
        cdef DoubleColumn res = DoubleColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length): res.data_c[i] = self.data_c[i] - val
        return res

    @cython.binding(False)
    def mul(self, double val, double fill_value=0.0):
        cdef DoubleColumn res = DoubleColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length): res.data_c[i] = self.data_c[i] * val
        return res

    @cython.binding(False)
    def div(self, double val, double fill_value=0.0):
        cdef DoubleColumn res = DoubleColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                res.data_c[i] = self.data_c[i] / val if val != 0 else NAN
        return res

    @cython.binding(False)
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

    # ---- Comparison operators (return IntColumn as 0/1 mask) ----

    @cython.binding(False)
    def _cmp_op(self, object val, int op):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        cdef double v
        cdef bint is_nan_val = False
        if isinstance(val, DoubleColumn):
            other = <DoubleColumn>val
            if other.length != self.length:
                raise ValueError(f"Length mismatch: {self.length} vs {other.length}")
            if op == 0:  # <
                for i in range(self.length): res.data_c[i] = 1 if self.data_c[i] < other.data_c[i] else 0
            elif op == 1:  # <=
                for i in range(self.length): res.data_c[i] = 1 if self.data_c[i] <= other.data_c[i] else 0
            elif op == 2:  # >
                for i in range(self.length): res.data_c[i] = 1 if self.data_c[i] > other.data_c[i] else 0
            elif op == 3:  # >=
                for i in range(self.length): res.data_c[i] = 1 if self.data_c[i] >= other.data_c[i] else 0
            elif op == 4:  # == (NaN != NaN)
                for i in range(self.length):
                    res.data_c[i] = 1 if (self.data_c[i] == other.data_c[i] and self.data_c[i] == self.data_c[i]) else 0
            elif op == 5:  # !=
                for i in range(self.length):
                    res.data_c[i] = 1 if (self.data_c[i] != other.data_c[i] or self.data_c[i] != self.data_c[i]) else 0
        else:
            v = <double>val
            is_nan_val = (v != v)
            if op == 0:
                for i in range(self.length): res.data_c[i] = 1 if self.data_c[i] < v else 0
            elif op == 1:
                for i in range(self.length): res.data_c[i] = 1 if self.data_c[i] <= v else 0
            elif op == 2:
                for i in range(self.length): res.data_c[i] = 1 if self.data_c[i] > v else 0
            elif op == 3:
                for i in range(self.length): res.data_c[i] = 1 if self.data_c[i] >= v else 0
            elif op == 4:
                for i in range(self.length):
                    if is_nan_val:
                        res.data_c[i] = 1 if self.data_c[i] != self.data_c[i] else 0
                    else:
                        res.data_c[i] = 1 if self.data_c[i] == v else 0
            elif op == 5:
                for i in range(self.length):
                    if is_nan_val:
                        res.data_c[i] = 1 if self.data_c[i] == self.data_c[i] else 0
                    else:
                        res.data_c[i] = 1 if self.data_c[i] != v else 0
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

    def __str__(self):
        cdef list items = []
        cdef Py_ssize_t i
        for i in range(self.length):
            items.append(str(self.data_c[i]))
        return f"DoubleColumn([{', '.join(items)}])"

    def __bool__(self):
        return self.length > 0

    def __len__(self):
        return self.length


    cdef DoubleColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=False):
        cdef double* old_data = self.data_c
        cdef double* new_data = <double*>malloc(self.capacity * sizeof(double))
        if not new_data: raise MemoryError()
        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                new_data[i] = old_data[indices[i]]
        if inplace:
            free(self.data_c)
            self.data_c = new_data
            self.length = n
            return self
        else:
            new_col = DoubleColumn(self.capacity)
            free(new_col.data_c)
            new_col.data_c = new_data
            new_col.length = n
            return new_col


    cpdef DoubleColumn clip(self, double lower, double upper, bint inplace=False):
        cdef DoubleColumn col = self if inplace else self.copy()
        cdef Py_ssize_t i
        for i in range(col.length):
            if col.data_c[i] < lower:
                col.data_c[i] = lower
            elif col.data_c[i] > upper:
                col.data_c[i] = upper
        return col


    cpdef DoubleColumn where(self, IntColumn cond, double other, bint inplace=False):
        cdef Py_ssize_t n = self.length
        cdef DoubleColumn col
        if inplace:
            col = self
        else:
            col = DoubleColumn(self.capacity)
            col.length = n
            memcpy(col.data_c, self.data_c, n * sizeof(double))
        cdef Py_ssize_t i
        for i in range(n):
            if not cond.data_c[i]:
                col.data_c[i] = other
        return col

    cpdef DoubleColumn mask(self, IntColumn cond, double other, bint inplace=False):
        cdef Py_ssize_t n = self.length
        cdef DoubleColumn col
        if inplace:
            col = self
        else:
            col = DoubleColumn(self.capacity)
            col.length = n
            memcpy(col.data_c, self.data_c, n * sizeof(double))
        cdef Py_ssize_t i
        for i in range(n):
            if cond.data_c[i]:
                col.data_c[i] = other
        return col


    cpdef void drop_row(self, Py_ssize_t row_idx, Py_ssize_t rem_len):
        if rem_len > 0:
            memmove(self.data_c + row_idx,
                    self.data_c + row_idx + 1,
                    rem_len * sizeof(double))
        self.length -= 1


    cdef void to_double_buffer(self, double* out, Py_ssize_t n):
        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                out[i] = self.data_c[i]


    cdef double corr_with(self, double* other, Py_ssize_t n):
        return _c_corr(self.data_c, other, n)

    cdef double cov_with(self, double* other, Py_ssize_t n):
        return _c_cov(self.data_c, other, n)


    cdef DoubleColumn gather_with_sentinel(self, Py_ssize_t* indices, Py_ssize_t n, double sentinel):
        cdef DoubleColumn new_col = DoubleColumn(n)
        new_col.length = n
        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                if indices[i] >= 0:
                    new_col.data_c[i] = self.data_c[indices[i]]
                else:
                    new_col.data_c[i] = sentinel
        return new_col
