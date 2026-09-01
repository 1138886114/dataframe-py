# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport cython
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libc.math cimport sqrt, fabs, NAN

from fluidframe.core._base cimport BaseColumn, TYPE_INT, TYPE_DOUBLE, aligned_malloc, os_aligned_free
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.doublecolumn cimport DoubleColumn


cdef class RollingWindow:

    def __cinit__(self, object col, Py_ssize_t window):
        cdef Py_ssize_t i
        cdef int col_dtype
        if window < 1:
            raise ValueError("window must be >= 1")
        self.length = col.length
        self.window = window
        self._owns_data = True
        self.data_c = <double*>aligned_malloc(max(self.length, 1) * sizeof(double))
        if not self.data_c:
            raise MemoryError()
        col_dtype = col.dtype
        if col_dtype == TYPE_INT:
            for i in range(self.length):
                self.data_c[i] = <double>(<IntColumn>col).data_c[i]
        elif col_dtype == TYPE_DOUBLE:
            for i in range(self.length):
                self.data_c[i] = (<DoubleColumn>col).data_c[i]
        else:
            for i in range(self.length):
                self.data_c[i] = 0.0

    def __dealloc__(self):
        if self._owns_data and self.data_c != NULL:
            os_aligned_free(self.data_c)

    cdef DoubleColumn _make_result(self):
        cdef DoubleColumn col = DoubleColumn(max(self.length, 1))
        col.length = self.length
        cdef Py_ssize_t i
        for i in range(self.length):
            col.data_c[i] = NAN
        return col

    def mean(self):
        cdef DoubleColumn result = self._make_result()
        cdef Py_ssize_t i, j, end
        cdef double s
        with nogil:
            for i in range(self.window - 1, self.length):
                end = i + 1
                s = 0.0
                for j in range(end - self.window, end):
                    s += self.data_c[j]
                result.data_c[i] = s / <double>self.window
        return result

    def sum(self):
        cdef DoubleColumn result = self._make_result()
        cdef Py_ssize_t i, j, end
        cdef double s
        with nogil:
            for i in range(self.window - 1, self.length):
                end = i + 1
                s = 0.0
                for j in range(end - self.window, end):
                    s += self.data_c[j]
                result.data_c[i] = s
        return result

    def min(self):
        cdef DoubleColumn result = self._make_result()
        cdef Py_ssize_t i, j, end
        cdef double mn, val
        with nogil:
            for i in range(self.window - 1, self.length):
                end = i + 1
                mn = self.data_c[end - self.window]
                for j in range(end - self.window + 1, end):
                    val = self.data_c[j]
                    if val < mn:
                        mn = val
                result.data_c[i] = mn
        return result

    def max(self):
        cdef DoubleColumn result = self._make_result()
        cdef Py_ssize_t i, j, end
        cdef double mx, val
        with nogil:
            for i in range(self.window - 1, self.length):
                end = i + 1
                mx = self.data_c[end - self.window]
                for j in range(end - self.window + 1, end):
                    val = self.data_c[j]
                    if val > mx:
                        mx = val
                result.data_c[i] = mx
        return result

    def std(self):
        cdef DoubleColumn result = self._make_result()
        cdef Py_ssize_t i, j, end
        cdef double s, mean, m2, val
        with nogil:
            for i in range(self.window - 1, self.length):
                end = i + 1
                s = 0.0
                for j in range(end - self.window, end):
                    s += self.data_c[j]
                mean = s / <double>self.window
                m2 = 0.0
                for j in range(end - self.window, end):
                    val = self.data_c[j] - mean
                    m2 += val * val
                result.data_c[i] = sqrt(m2 / <double>self.window)
        return result

    def var(self):
        cdef DoubleColumn result = self._make_result()
        cdef Py_ssize_t i, j, end
        cdef double s, mean, m2, val
        with nogil:
            for i in range(self.window - 1, self.length):
                end = i + 1
                s = 0.0
                for j in range(end - self.window, end):
                    s += self.data_c[j]
                mean = s / <double>self.window
                m2 = 0.0
                for j in range(end - self.window, end):
                    val = self.data_c[j] - mean
                    m2 += val * val
                result.data_c[i] = m2 / <double>self.window
        return result

    def median(self):
        cdef DoubleColumn result = self._make_result()
        cdef Py_ssize_t i, j, end, k
        cdef double* buf = <double*>aligned_malloc(max(self.window, 1) * sizeof(double))
        if not buf:
            raise MemoryError()
        cdef double pivot, temp
        cdef Py_ssize_t left, right, store
        try:
            with nogil:
                for i in range(self.window - 1, self.length):
                    end = i + 1
                    for j in range(self.window):
                        buf[j] = self.data_c[end - self.window + j]
                    left = 0
                    right = self.window - 1
                    while left < right:
                        pivot = buf[right]
                        store = left
                        for j in range(left, right):
                            if buf[j] < pivot:
                                temp = buf[j]
                                buf[j] = buf[store]
                                buf[store] = temp
                                store += 1
                        temp = buf[store]
                        buf[store] = buf[right]
                        buf[right] = temp
                        if store > (left + right) / 2:
                            right = store - 1
                        else:
                            left = store + 1
                    result.data_c[i] = buf[self.window / 2]
        finally:
            os_aligned_free(buf)
        return result

    def quantile(self, double q):
        cdef DoubleColumn result = self._make_result()
        cdef Py_ssize_t i, j, end, idx
        cdef double* buf = <double*>aligned_malloc(max(self.window, 1) * sizeof(double))
        if not buf:
            raise MemoryError()
        cdef double pivot, temp, pos
        cdef Py_ssize_t left, right, store
        try:
            with nogil:
                for i in range(self.window - 1, self.length):
                    end = i + 1
                    for j in range(self.window):
                        buf[j] = self.data_c[end - self.window + j]
                    left = 0
                    right = self.window - 1
                    while left < right:
                        pivot = buf[right]
                        store = left
                        for j in range(left, right):
                            if buf[j] < pivot:
                                temp = buf[j]
                                buf[j] = buf[store]
                                buf[store] = temp
                                store += 1
                        temp = buf[store]
                        buf[store] = buf[right]
                        buf[right] = temp
                        if store > (left + right) / 2:
                            right = store - 1
                        else:
                            left = store + 1
                    pos = q * <double>(self.window - 1)
                    idx = <Py_ssize_t>pos
                    if idx >= self.window - 1:
                        result.data_c[i] = buf[self.window - 1]
                    else:
                        result.data_c[i] = buf[idx] + (pos - <double>idx) * (buf[idx + 1] - buf[idx])
        finally:
            os_aligned_free(buf)
        return result

    def skew(self):
        cdef DoubleColumn result = self._make_result()
        cdef Py_ssize_t i, j, end
        cdef double s, mean, m2, m3, val, n, n_f
        cdef double factor
        with nogil:
            n_f = <double>self.window
            for i in range(self.window - 1, self.length):
                end = i + 1
                s = 0.0
                for j in range(end - self.window, end):
                    s += self.data_c[j]
                mean = s / n_f
                m2 = 0.0
                m3 = 0.0
                for j in range(end - self.window, end):
                    val = self.data_c[j] - mean
                    m2 += val * val
                    m3 += val * val * val
                if m2 > 0:
                    factor = sqrt(n_f) / sqrt(m2)
                    result.data_c[i] = m3 / (m2 * sqrt(m2)) * n_f * factor
                else:
                    result.data_c[i] = 0.0
        return result

    def kurt(self):
        cdef DoubleColumn result = self._make_result()
        cdef Py_ssize_t i, j, end
        cdef double s, mean, m2, m4, val, n_f
        with nogil:
            n_f = <double>self.window
            for i in range(self.window - 1, self.length):
                end = i + 1
                s = 0.0
                for j in range(end - self.window, end):
                    s += self.data_c[j]
                mean = s / n_f
                m2 = 0.0
                m4 = 0.0
                for j in range(end - self.window, end):
                    val = self.data_c[j] - mean
                    m2 += val * val
                    m4 += val * val * val * val
                if m2 > 0:
                    result.data_c[i] = m4 / (m2 * m2) - 3.0
                else:
                    result.data_c[i] = 0.0
        return result

    def corr(self, RollingWindow other):
        cdef Py_ssize_t n = self.window
        cdef DoubleColumn result = self._make_result()
        cdef Py_ssize_t i, j, end
        cdef double n_f = <double>n
        cdef double* buf_a = <double*>aligned_malloc(max(n, 1) * sizeof(double))
        cdef double* buf_b = <double*>aligned_malloc(max(n, 1) * sizeof(double))
        if not buf_a or not buf_b:
            if buf_a: os_aligned_free(buf_a)
            if buf_b: os_aligned_free(buf_b)
            raise MemoryError()
        cdef double mean_a, mean_b, cov, var_a, var_b, val
        try:
            with nogil:
                for i in range(self.window - 1, self.length):
                    end = i + 1
                    for j in range(n):
                        buf_a[j] = self.data_c[end - n + j]
                        buf_b[j] = other.data_c[end - n + j]
                    mean_a = 0.0
                    mean_b = 0.0
                    for j in range(n):
                        mean_a += buf_a[j]
                        mean_b += buf_b[j]
                    mean_a /= n_f
                    mean_b /= n_f
                    cov = 0.0
                    var_a = 0.0
                    var_b = 0.0
                    for j in range(n):
                        val = buf_a[j] - mean_a
                        cov += val * (buf_b[j] - mean_b)
                        var_a += val * val
                        val = buf_b[j] - mean_b
                        var_b += val * val
                    if var_a > 0 and var_b > 0:
                        result.data_c[i] = cov / sqrt(var_a * var_b)
                    else:
                        result.data_c[i] = 0.0
        finally:
            os_aligned_free(buf_a)
            os_aligned_free(buf_b)
        return result

    def cov(self, RollingWindow other):
        cdef Py_ssize_t n = self.window
        cdef DoubleColumn result = self._make_result()
        cdef Py_ssize_t i, j, end
        cdef double* buf_a = <double*>aligned_malloc(max(n, 1) * sizeof(double))
        cdef double* buf_b = <double*>aligned_malloc(max(n, 1) * sizeof(double))
        if not buf_a or not buf_b:
            if buf_a: os_aligned_free(buf_a)
            if buf_b: os_aligned_free(buf_b)
            raise MemoryError()
        cdef double mean_a, mean_b, cov_val, val
        cdef double n_f = <double>n
        try:
            with nogil:
                for i in range(self.window - 1, self.length):
                    end = i + 1
                    for j in range(n):
                        buf_a[j] = self.data_c[end - n + j]
                        buf_b[j] = other.data_c[end - n + j]
                    mean_a = 0.0
                    mean_b = 0.0
                    for j in range(n):
                        mean_a += buf_a[j]
                        mean_b += buf_b[j]
                    mean_a /= n_f
                    mean_b /= n_f
                    cov_val = 0.0
                    for j in range(n):
                        cov_val += (buf_a[j] - mean_a) * (buf_b[j] - mean_b)
                    result.data_c[i] = cov_val / n_f
        finally:
            os_aligned_free(buf_a)
            os_aligned_free(buf_b)
        return result
