# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

cimport cython
from libc.stdlib cimport malloc, free, qsort
from libc.string cimport memcpy
from libc.math cimport sqrt, fabs, NAN

from fluidframe.core._base cimport BaseColumn, TYPE_INT, TYPE_DOUBLE, aligned_malloc, os_aligned_free
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.doublecolumn cimport DoubleColumn


cdef int _cmp_double(const void* a, const void* b) nograd nogil:
    cdef double da = (<double*>a)[0]
    cdef double db = (<double*>b)[0]
    if da < db: return -1
    if da > db: return 1
    return 0


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
        if col_dtype == TYPE_DOUBLE:
            if self.length > 0:
                memcpy(self.data_c, (<DoubleColumn>col).data_c, self.length * sizeof(double))
        elif col_dtype == TYPE_INT:
            with nogil:
                for i in range(self.length):
                    self.data_c[i] = <double>(<IntColumn>col).data_c[i]
        else:
            with nogil:
                for i in range(self.length):
                    self.data_c[i] = 0.0

    def __dealloc__(self):
        if self._owns_data and self.data_c != NULL:
            os_aligned_free(self.data_c)
            self.data_c = NULL

    cdef DoubleColumn _make_result(self):
        cdef DoubleColumn col = DoubleColumn(max(self.length, 1))
        col.length = self.length
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                col.data_c[i] = NAN
        return col

    # ------------------------------------------------------------
    # 1. Sum / Mean: O(N) 滑窗累加器
    # ------------------------------------------------------------
    def sum(self):
        cdef DoubleColumn result = self._make_result()
        cdef Py_ssize_t i, w = self.window, n = self.length
        cdef double s = 0.0
        if n < w: return result

        with nogil:
            for i in range(w):
                s += self.data_c[i]
            result.data_c[w - 1] = s

            for i in range(w, n):
                s += self.data_c[i] - self.data_c[i - w]
                result.data_c[i] = s
        return result

    def mean(self):
        cdef DoubleColumn result = self._make_result()
        cdef Py_ssize_t i, w = self.window, n = self.length
        cdef double s = 0.0, inv_w = 1.0 / <double>w
        if n < w: return result

        with nogil:
            for i in range(w):
                s += self.data_c[i]
            result.data_c[w - 1] = s * inv_w

            for i in range(w, n):
                s += self.data_c[i] - self.data_c[i - w]
                result.data_c[i] = s * inv_w
        return result

    # ------------------------------------------------------------
    # 2. Min / Max: O(N) 严格单调双端队列算法
    # ------------------------------------------------------------
    def min(self):
        cdef DoubleColumn result = self._make_result()
        cdef Py_ssize_t i, head = 0, tail = 0
        cdef Py_ssize_t w = self.window, n = self.length
        if n < w: return result

        cdef Py_ssize_t* q = <Py_ssize_t*>malloc(n * sizeof(Py_ssize_t))
        if not q: raise MemoryError()

        try:
            with nogil:
                for i in range(n):
                    while head < tail and q[head] <= i - w:
                        head += 1
                    while head < tail and self.data_c[q[tail - 1]] >= self.data_c[i]:
                        tail -= 1
                    q[tail] = i
                    tail += 1
                    if i >= w - 1:
                        result.data_c[i] = self.data_c[q[head]]
        finally:
            free(q)
        return result

    def max(self):
        cdef DoubleColumn result = self._make_result()
        cdef Py_ssize_t i, head = 0, tail = 0
        cdef Py_ssize_t w = self.window, n = self.length
        if n < w: return result

        cdef Py_ssize_t* q = <Py_ssize_t*>malloc(n * sizeof(Py_ssize_t))
        if not q: raise MemoryError()

        try:
            with nogil:
                for i in range(n):
                    while head < tail and q[head] <= i - w:
                        head += 1
                    while head < tail and self.data_c[q[tail - 1]] <= self.data_c[i]:
                        tail -= 1
                    q[tail] = i
                    tail += 1
                    if i >= w - 1:
                        result.data_c[i] = self.data_c[q[head]]
        finally:
            free(q)
        return result

    # ------------------------------------------------------------
    # 3. Var / Std: O(N) 双累加器滑窗（带数值稳定处理）
    # ------------------------------------------------------------
    def var(self):
        cdef DoubleColumn result = self._make_result()
        cdef Py_ssize_t i, w = self.window, n = self.length
        cdef double s = 0.0, sq_s = 0.0, val, old_val, v
        cdef double inv_w = 1.0 / <double>w
        if n < w: return result

        with nogil:
            for i in range(w):
                val = self.data_c[i]
                s += val
                sq_s += val * val
            v = (sq_s - (s * s) * inv_w) * inv_w
            result.data_c[w - 1] = v if v > 0.0 else 0.0

            for i in range(w, n):
                val = self.data_c[i]
                old_val = self.data_c[i - w]
                s += val - old_val
                sq_s += val * val - old_val * old_val
                v = (sq_s - (s * s) * inv_w) * inv_w
                result.data_c[i] = v if v > 0.0 else 0.0
        return result

    def std(self):
        cdef DoubleColumn result = self._make_result()
        cdef Py_ssize_t i, w = self.window, n = self.length
        cdef double s = 0.0, sq_s = 0.0, val, old_val, v
        cdef double inv_w = 1.0 / <double>w
        if n < w: return result

        with nogil:
            for i in range(w):
                val = self.data_c[i]
                s += val
                sq_s += val * val
            v = (sq_s - (s * s) * inv_w) * inv_w
            result.data_c[w - 1] = sqrt(v) if v > 0.0 else 0.0

            for i in range(w, n):
                val = self.data_c[i]
                old_val = self.data_c[i - w]
                s += val - old_val
                sq_s += val * val - old_val * old_val
                v = (sq_s - (s * s) * inv_w) * inv_w
                result.data_c[i] = sqrt(v) if v > 0.0 else 0.0
        return result

    # ------------------------------------------------------------
    # 4. Corr / Cov: O(N) 5-累加器在线滑窗
    # ------------------------------------------------------------
    def cov(self, RollingWindow other):
        cdef DoubleColumn result = self._make_result()
        cdef Py_ssize_t i, w = self.window, n = self.length
        cdef double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0
        cdef double x, y, old_x, old_y
        cdef double inv_w = 1.0 / <double>w
        if n < w or other.length != n: return result

        with nogil:
            for i in range(w):
                x = self.data_c[i]; y = other.data_c[i]
                sum_x += x; sum_y += y; sum_xy += x * y
            result.data_c[w - 1] = (sum_xy - sum_x * sum_y * inv_w) * inv_w

            for i in range(w, n):
                x = self.data_c[i]; y = other.data_c[i]
                old_x = self.data_c[i - w]; old_y = other.data_c[i - w]
                sum_x += x - old_x
                sum_y += y - old_y
                sum_xy += x * y - old_x * old_y
                result.data_c[i] = (sum_xy - sum_x * sum_y * inv_w) * inv_w
        return result

    def corr(self, RollingWindow other):
        cdef DoubleColumn result = self._make_result()
        cdef Py_ssize_t i, w = self.window, n = self.length
        cdef double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_xx = 0.0, sum_yy = 0.0
        cdef double x, y, old_x, old_y, vx, vy, cov_xy, denom
        cdef double inv_w = 1.0 / <double>w
        if n < w or other.length != n: return result

        with nogil:
            for i in range(w):
                x = self.data_c[i]; y = other.data_c[i]
                sum_x += x; sum_y += y
                sum_xx += x * x; sum_yy += y * y
                sum_xy += x * y

            vx = sum_xx - sum_x * sum_x * inv_w
            vy = sum_yy - sum_y * sum_y * inv_w
            denom = sqrt(vx * vy)
            cov_xy = sum_xy - sum_x * sum_y * inv_w
            result.data_c[w - 1] = cov_xy / denom if (denom > 0.0) else 0.0

            for i in range(w, n):
                x = self.data_c[i]; y = other.data_c[i]
                old_x = self.data_c[i - w]; old_y = other.data_c[i - w]
                sum_x += x - old_x
                sum_y += y - old_y
                sum_xx += x * x - old_x * old_x
                sum_yy += y * y - old_y * old_y
                sum_xy += x * y - old_x * old_y

                vx = sum_xx - sum_x * sum_x * inv_w
                vy = sum_yy - sum_y * sum_y * inv_w
                denom = sqrt(vx * vy)
                cov_xy = sum_xy - sum_x * sum_y * inv_w
                result.data_c[i] = cov_xy / denom if (denom > 0.0) else 0.0
        return result

    # ------------------------------------------------------------
    # 5. Median / Quantile: 修复插值与排序安全性
    # ------------------------------------------------------------
    def quantile(self, double q):
        if q < 0.0 or q > 1.0:
            raise ValueError("q must be between 0.0 and 1.0")
        cdef DoubleColumn result = self._make_result()
        cdef Py_ssize_t i, j, idx, w = self.window, n = self.length
        if n < w: return result

        cdef double* buf = <double*>malloc(w * sizeof(double))
        if not buf: raise MemoryError()

        cdef double pos = q * <double>(w - 1)
        cdef Py_ssize_t i_low = <Py_ssize_t>pos
        cdef double frac = pos - <double>i_low

        try:
            with nogil:
                for i in range(w - 1, n):
                    memcpy(buf, self.data_c + (i - w + 1), w * sizeof(double))
                    qsort(buf, w, sizeof(double), _cmp_double)
                    if i_low >= w - 1:
                        result.data_c[i] = buf[w - 1]
                    else:
                        result.data_c[i] = buf[i_low] + frac * (buf[i_low + 1] - buf[i_low])
        finally:
            free(buf)
        return result

    def median(self):
        return self.quantile(0.5)

    # ------------------------------------------------------------
    # 6. Skew / Kurt: 修复数学公式与高阶矩
    # ------------------------------------------------------------
    def skew(self):
        cdef DoubleColumn result = self._make_result()
        cdef Py_ssize_t i, j, end, w = self.window, n = self.length
        cdef double s, mean, m2, m3, val, n_f = <double>w
        cdef double std_dev
        if n < w or w < 3: return result

        with nogil:
            for i in range(w - 1, n):
                end = i + 1
                s = 0.0
                for j in range(end - w, end):
                    s += self.data_c[j]
                mean = s / n_f
                m2 = 0.0
                m3 = 0.0
                for j in range(end - w, end):
                    val = self.data_c[j] - mean
                    m2 += val * val
                    m3 += val * val * val

                # 样本二阶中心矩 m2/n，三阶中心矩 m3/n
                if m2 > 1e-14:
                    std_dev = sqrt(m2 / n_f)
                    result.data_c[i] = (m3 / n_f) / (std_dev * std_dev * std_dev)
                else:
                    result.data_c[i] = 0.0
        return result

    def kurt(self):
        cdef DoubleColumn result = self._make_result()
        cdef Py_ssize_t i, j, end, w = self.window, n = self.length
        cdef double s, mean, m2, m4, val, n_f = <double>w
        cdef double variance
        if n < w or w < 4: return result

        with nogil:
            for i in range(w - 1, n):
                end = i + 1
                s = 0.0
                for j in range(end - w, end):
                    s += self.data_c[j]
                mean = s / n_f
                m2 = 0.0
                m4 = 0.0
                for j in range(end - w, end):
                    val = self.data_c[j] - mean
                    m2 += val * val
                    m4 += val * val * val * val

                if m2 > 1e-14:
                    variance = m2 / n_f
                    # 总体超额峰度 Fisher's Kurtosis: (m4 / n) / (variance^2) - 3.0
                    result.data_c[i] = (m4 / n_f) / (variance * variance) - 3.0
                else:
                    result.data_c[i] = 0.0
        return result