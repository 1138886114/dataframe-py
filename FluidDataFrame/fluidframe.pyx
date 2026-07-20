# %%cython
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport cython
from libc.string cimport strncmp
from libc.stdlib cimport malloc, realloc, free
from libc.string cimport strstr, strlen, strncmp, memmove, memcpy, strncat, strncpy
from libc.math cimport sqrt, fabs, NAN

cdef extern from "<ctype.h>" nogil:
    int tolower(int)
    int toupper(int)
    int isdigit(int)
    int isspace(int)
    int isalpha(int)
    int isalnum(int)
    int islower(int)
    int isupper(int)

from cpython.unicode cimport (
    PyUnicode_4BYTE_KIND, PyUnicode_FromKindAndData,
    PyUnicode_GET_LENGTH, PyUnicode_READ_CHAR, PyUnicode_AsUCS4,
    PyUnicode_KIND, PyUnicode_2BYTE_KIND, PyUnicode_DATA,
    Py_UCS2
)
import re

# ==========================================
# 1. 枚举与基础类型定义
# ==========================================
cdef enum DataType:
    TYPE_INT = 0
    TYPE_FLOAT = 1
    TYPE_DOUBLE = 2
    TYPE_CHAR256 = 3
    TYPE_UCS2 = 4
    TYPE_TIMESTAMP = 5


ctypedef fused NumericType:
    int
    double
    long long


class Char256:
    """用于在 Schema 中标记纯 ASCII 256字节定长字符串类型的占位类"""
    pass


class Timestamp:
    """时间戳类型，内部按 YYYYMMDD 整数存储；构造时支持 Timestamp('20160101') 或 Timestamp(20160101)"""
    def __init__(self, val=None):
        pass

@cython.binding(False)
@cython.always_allow_keywords(False)
def _parse_schema(dict schema_dict):
    cdef dict parsed = {}
    for k, v in schema_dict.items():
        if v is int: parsed[k] = TYPE_INT
        elif v is float: parsed[k] = TYPE_DOUBLE
        elif v is Char256: parsed[k] = TYPE_CHAR256
        elif v is str: parsed[k] = TYPE_UCS2
        elif v is Timestamp: parsed[k] = TYPE_TIMESTAMP
        else: raise TypeError(f"Unsupported type: {v}")
    return parsed

# ==========================================
# 2. 基础列类 (统一抽象)
# ==========================================
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class BaseColumn:
    cdef public Py_ssize_t length
    cdef public Py_ssize_t capacity

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __len__(self):
        return self.length

# ==========================================
# 3. 数值型列组件 (Int, Double)
# ==========================================
@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class IntColumn(BaseColumn):
    cdef int* data_c

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __cinit__(self, Py_ssize_t init_capacity=16):
        self.capacity = max(init_capacity, 1)
        self.length = 0
        self.data_c = <int*>malloc(self.capacity * sizeof(int))
        if not self.data_c: raise MemoryError()

    def __dealloc__(self):
        if self.data_c is not NULL:
            free(self.data_c)

    cdef void _reserve(self, Py_ssize_t new_cap):
        if new_cap > self.capacity:
            self.data_c = <int*>realloc(self.data_c, new_cap * sizeof(int))
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
        cdef int* temp = <int*>malloc(self.length * sizeof(int))
        if not temp: raise MemoryError()
        memcpy(temp, self.data_c, self.length * sizeof(int))
        cdef list py_list = [temp[i] for i in range(self.length)]
        free(temp)
        py_list.sort()
        cdef Py_ssize_t mid = self.length // 2
        if self.length % 2 == 0:
            return (py_list[mid - 1] + py_list[mid]) / 2.0
        return py_list[mid]

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

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def add(self, int val, int fill_value=0):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length): res.data_c[i] = self.data_c[i] + val
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sub(self, int val, int fill_value=0):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length): res.data_c[i] = self.data_c[i] - val
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def mul(self, int val, int fill_value=0):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length): res.data_c[i] = self.data_c[i] * val
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def div(self, int val, int fill_value=0):
        cdef DoubleColumn res = DoubleColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                res.data_c[i] = <double>self.data_c[i] / val if val != 0 else NAN
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def diff(self):
        """计算一阶差分，首位用 0 填充"""
        cdef IntColumn res = IntColumn(self.capacity)
        if self.length == 0: return res
        res.append(0)
        for i in range(1, self.length):
            res.append(self.data_c[i] - self.data_c[i-1])
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _reorder(self, list indices):
        cdef int* new_data = <int*>malloc(self.capacity * sizeof(int))
        if not new_data: raise MemoryError()
        cdef Py_ssize_t i
        for i in range(self.length):
            new_data[i] = self.data_c[<Py_ssize_t>indices[i]]
        free(self.data_c)
        self.data_c = new_data

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __getitem__(self, Py_ssize_t idx):
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


@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class DoubleColumn(BaseColumn):
    cdef double* data_c

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __cinit__(self, Py_ssize_t init_capacity=16):
        self.capacity = max(init_capacity, 1)
        self.length = 0
        self.data_c = <double*>malloc(self.capacity * sizeof(double))
        if not self.data_c: raise MemoryError()

    def __dealloc__(self):
        if self.data_c is not NULL: free(self.data_c)

    cdef void _reserve(self, Py_ssize_t new_cap):
        if new_cap > self.capacity:
            self.data_c = <double*>realloc(self.data_c, new_cap * sizeof(double))
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
        cdef double* new_data = <double*>malloc(self.capacity * sizeof(double))
        if not new_data: raise MemoryError()
        cdef Py_ssize_t i
        for i in range(self.length):
            new_data[i] = self.data_c[<Py_ssize_t>indices[i]]
        free(self.data_c)
        self.data_c = new_data

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __getitem__(self, Py_ssize_t idx):
        if idx < 0 or idx >= self.length: raise IndexError()
        return self.data_c[idx]


    # ==========================================
    # 数值型列：高级统计与算术运算
    # ==========================================
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
        cdef double* temp = <double*>malloc(self.length * sizeof(double))
        if not temp: raise MemoryError()
        memcpy(temp, self.data_c, self.length * sizeof(double))
        # 借助 Python 的 sort，或者这里用 C 的 qsort（为保持极速此处略写 qsort wrapper，直接用 Python）
        cdef list py_list = [temp[i] for i in range(self.length)]
        free(temp)
        py_list.sort()
        cdef Py_ssize_t mid = self.length // 2
        if self.length % 2 == 0:
            return (py_list[mid - 1] + py_list[mid]) / 2.0
        return py_list[mid]

    @cython.binding(False)
    def mad(self):
        """平均绝对离差"""
        if self.length == 0: return NAN
        cdef double m = self.mean()
        cdef double total_dev = 0
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                total_dev += fabs(self.data_c[i] - m)
        return total_dev / self.length

    # 元素级四则运算 (支持标量)
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


# ==========================================
# 4. 纯 ASCII 定长 256 存储组件 (Char256)
# ==========================================
cdef inline bint _c_str_contains(char* source, const char* target) noexcept nogil:
    return strstr(source, target) != NULL

cdef inline void _c_str_replace_inplace(char* buffer, const char* old_sub, const char* new_sub) noexcept nogil:
    cdef size_t old_len = strlen(old_sub)
    if old_len == 0: return
    cdef size_t new_len = strlen(new_sub)
    cdef char* match = strstr(buffer, old_sub)
    cdef size_t current_len
    cdef size_t remaining

    while match != NULL:
        current_len = strlen(buffer)
        remaining = current_len - (match - buffer) - old_len
        if current_len - old_len + new_len >= 255: break
        if new_len != old_len:
            memmove(match + new_len, match + old_len, remaining + 1)
        if new_len > 0:
            memcpy(match, new_sub, new_len)
        match = strstr(match + new_len, old_sub)

@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class Char256Column_SoA(BaseColumn):
    cdef char* data_c

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __cinit__(self, Py_ssize_t init_capacity=16):
        self.capacity = max(init_capacity, 1)
        self.length = 0
        self.data_c = <char*>malloc(self.capacity * 256)
        if not self.data_c: raise MemoryError()

    def __dealloc__(self):
        if self.data_c is not NULL: free(self.data_c)

    cdef void _reserve(self, Py_ssize_t new_cap):
        if new_cap > self.capacity:
            self.data_c = <char*>realloc(self.data_c, new_cap * 256)
            if not self.data_c: raise MemoryError()
            self.capacity = new_cap

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def append(self, str val):
        if self.length >= self.capacity:
            self._reserve(self.capacity * 2)
        cdef bytes b_val = val.encode('ascii', errors='ignore')
        cdef char* src = b_val
        cdef char* dest = self.data_c + (self.length * 256)
        strncpy(dest, src, 255)
        dest[255] = b'\0'
        self.length += 1

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def copy(self):
        cdef Char256Column_SoA new_col = Char256Column_SoA(self.capacity)
        new_col.length = self.length
        memcpy(new_col.data_c, self.data_c, self.length * 256)
        return new_col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_back(self):
        if self.length == 0: raise IndexError("pop from empty column")
        self.length -= 1
        cdef char* ptr = self.data_c + (self.length * 256)
        return ptr.decode('ascii')

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_front(self):
        if self.length == 0: raise IndexError("pop from empty column")
        cdef char* ptr = self.data_c
        cdef str val = ptr.decode('ascii')
        memmove(self.data_c, self.data_c + 256, (self.length - 1) * 256)
        self.length -= 1
        return val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def concat(self, Char256Column_SoA other):
        self._reserve(self.length + other.length)
        memcpy(self.data_c + (self.length * 256), other.data_c, other.length * 256)
        self.length += other.length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _reorder(self, list indices):
        cdef char* new_data = <char*>malloc(self.capacity * 256)
        if not new_data: raise MemoryError()
        cdef Py_ssize_t i
        for i in range(self.length):
            memcpy(new_data + (i * 256), self.data_c + (<Py_ssize_t>indices[i] * 256), 256)
        free(self.data_c)
        self.data_c = new_data

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __getitem__(self, Py_ssize_t idx):
        if idx < 0 or idx >= self.length: raise IndexError()
        cdef char* ptr = self.data_c + (idx * 256)
        return ptr.decode('ascii')

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def contains(self, str substring):
        cdef bytes b_sub = substring.encode('ascii')
        cdef const char* c_sub = b_sub
        cdef int* out_idx = <int*>malloc(self.length * sizeof(int))
        if not out_idx: raise MemoryError()
        cdef int count = 0
        cdef Py_ssize_t i

        with nogil:
            for i in range(self.length):
                if _c_str_contains(self.data_c + (i * 256), c_sub):
                    out_idx[count] = i
                    count += 1
        cdef list res = [out_idx[j] for j in range(count)]
        free(out_idx)
        return res

    @cython.binding(False)
    def replace(self, str old_str, str new_str):
        cdef bytes b_old = old_str.encode('ascii')
        cdef bytes b_new = new_str.encode('ascii')
        cdef const char* c_old = b_old
        cdef const char* c_new = b_new
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                _c_str_replace_inplace(self.data_c + (i * 256), c_old, c_new)
        return self


    # ==========================================
    # 字符串操作：Pandas str 风格接口
    # ==========================================
    @cython.binding(False)
    def replaces(self, *args):
        """支持连续替换 df['txt'].replaces('old1','new1', 'old2','new2')"""
        if len(args) % 2 != 0: raise ValueError("Arguments must be in pairs")
        cdef int i
        for i in range(0, len(args), 2):
            self.replace(args[i], args[i+1])
        return self

    @cython.binding(False)
    def extract(self, str pattern):
        """返回匹配的第一个捕获组新列"""
        prog = re.compile(pattern)
        cdef Char256Column_SoA res = Char256Column_SoA(self.capacity)
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = (self.data_c + i * 256).decode('ascii', errors='ignore')
            match = prog.search(s)
            if match and match.groups():
                res.append(match.group(1))
            else:
                res.append("")
        return res

    @cython.binding(False)
    def extract_twice(self, str pattern):
        """返回符合第一、二个捕获组，赋值给两个新列"""
        prog = re.compile(pattern)
        cdef Char256Column_SoA res1 = Char256Column_SoA(self.capacity)
        cdef Char256Column_SoA res2 = Char256Column_SoA(self.capacity)
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = (self.data_c + i * 256).decode('ascii', errors='ignore')
            match = prog.search(s)
            if match and len(match.groups()) >= 2:
                res1.append(match.group(1))
                res2.append(match.group(2))
            else:
                res1.append("")
                res2.append("")
        return res1, res2

    @cython.binding(False)
    def len(self):
        """返回字符串长度 IntColumn"""
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                res.data_c[i] = <int>strlen(self.data_c + i * 256)
        return res

    @cython.binding(False)
    def len_list(self):
        return [strlen(self.data_c + i * 256) for i in range(self.length)]

    @cython.binding(False)
    def startswith(self, str prefix):
        cdef bytes b_pre = prefix.encode('ascii')
        cdef const char* c_pre = b_pre
        cdef Py_ssize_t pre_len = strlen(c_pre)
        cdef IntColumn res = IntColumn(self.capacity) # 用 1/0 表示 True/False
        res.length = self.length
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                res.data_c[i] = 1 if strncmp(self.data_c + i * 256, c_pre, pre_len) == 0 else 0
        return res

    @cython.binding(False)
    def endswith(self, str suffix):
        cdef bytes b_suf = suffix.encode('ascii')
        cdef const char* c_suf = b_suf
        cdef Py_ssize_t suf_len = strlen(c_suf)
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i, curr_len
        cdef char* ptr
        with nogil:
            for i in range(self.length):
                ptr = self.data_c + i * 256
                curr_len = strlen(ptr)
                if curr_len >= suf_len and strncmp(ptr + curr_len - suf_len, c_suf, suf_len) == 0:
                    res.data_c[i] = 1
                else:
                    res.data_c[i] = 0
        return res

    @cython.binding(False)
    def lower(self):
        cdef Char256Column_SoA res = self.copy()
        cdef Py_ssize_t i, j
        cdef char* ptr
        with nogil:
            for i in range(self.length):
                ptr = res.data_c + i * 256
                for j in range(256):
                    if ptr[j] == b'\0': break
                    ptr[j] = tolower(ptr[j])
        return res

    @cython.binding(False)
    def upper(self):
        cdef Char256Column_SoA res = self.copy()
        cdef Py_ssize_t i, j
        cdef char* ptr
        with nogil:
            for i in range(self.length):
                ptr = res.data_c + i * 256
                for j in range(256):
                    if ptr[j] == b'\0': break
                    ptr[j] = toupper(ptr[j])
        return res

    @cython.binding(False)
    def isdigit(self):
        """是否全部是数字"""
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i, j
        cdef char* ptr
        cdef bint valid
        with nogil:
            for i in range(self.length):
                ptr = self.data_c + i * 256
                if ptr[0] == b'\0':
                    res.data_c[i] = 0
                    continue
                valid = 1
                for j in range(256):
                    if ptr[j] == b'\0': break
                    if not isdigit(ptr[j]):
                        valid = 0
                        break
                res.data_c[i] = valid
        return res

    @cython.binding(False)
    def strip(self):
        cdef Char256Column_SoA res = self.copy()
        cdef Py_ssize_t i, left, right
        cdef char* ptr
        with nogil:
            for i in range(self.length):
                ptr = res.data_c + i * 256
                left = 0
                while left < 256 and ptr[left] != b'\0' and isspace(ptr[left]):
                    left += 1
                if left == 256 or ptr[left] == b'\0':
                    ptr[0] = b'\0'
                    continue
                right = 255
                while right >= 0 and (ptr[right] == b'\0' or isspace(ptr[right])):
                    right -= 1
                if left > 0:
                    memmove(ptr, ptr + left, right - left + 1)
                ptr[right - left + 1] = b'\0'
        return res

    @cython.binding(False)
    def lstrip(self):
        cdef Char256Column_SoA res = self.copy()
        cdef Py_ssize_t i, left, remaining
        cdef char* ptr
        with nogil:
            for i in range(self.length):
                ptr = res.data_c + i * 256
                left = 0
                while left < 256 and ptr[left] != b'\0' and isspace(ptr[left]):
                    left += 1
                if left > 0:
                    remaining = 255 - left
                    memmove(ptr, ptr + left, remaining + 1)
                    ptr[remaining] = b'\0'
        return res

    @cython.binding(False)
    def rstrip(self):
        cdef Char256Column_SoA res = self.copy()
        cdef Py_ssize_t i, right
        cdef char* ptr
        with nogil:
            for i in range(self.length):
                ptr = res.data_c + i * 256
                right = 255
                while right >= 0 and (ptr[right] == b'\0' or isspace(ptr[right])):
                    right -= 1
                if right < 255:
                    ptr[right + 1] = b'\0'
        return res

    @cython.binding(False)
    def find(self, str substring):
        cdef bytes b_sub = substring.encode('ascii')
        cdef const char* c_sub = b_sub
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        cdef char* match
        with nogil:
            for i in range(self.length):
                match = strstr(self.data_c + i * 256, c_sub)
                if match != NULL:
                    res.data_c[i] = <int>(match - (self.data_c + i * 256))
                else:
                    res.data_c[i] = -1
        return res

    @cython.binding(False)
    def count(self, str substring):
        cdef bytes b_sub = substring.encode('ascii')
        cdef const char* c_sub = b_sub
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        cdef char* ptr
        cdef char* match
        cdef int cnt
        with nogil:
            for i in range(self.length):
                ptr = self.data_c + i * 256
                cnt = 0
                match = strstr(ptr, c_sub)
                while match != NULL:
                    cnt += 1
                    match = strstr(match + 1, c_sub)
                res.data_c[i] = cnt
        return res

    @cython.binding(False)
    def isalpha(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i, j
        cdef char* ptr
        cdef bint valid
        with nogil:
            for i in range(self.length):
                ptr = self.data_c + i * 256
                if ptr[0] == b'\0':
                    res.data_c[i] = 0
                    continue
                valid = 1
                for j in range(256):
                    if ptr[j] == b'\0': break
                    if not isalpha(ptr[j]):
                        valid = 0
                        break
                res.data_c[i] = valid
        return res

    @cython.binding(False)
    def isalnum(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i, j
        cdef char* ptr
        cdef bint valid
        with nogil:
            for i in range(self.length):
                ptr = self.data_c + i * 256
                if ptr[0] == b'\0':
                    res.data_c[i] = 0
                    continue
                valid = 1
                for j in range(256):
                    if ptr[j] == b'\0': break
                    if not isalnum(ptr[j]):
                        valid = 0
                        break
                res.data_c[i] = valid
        return res

    @cython.binding(False)
    def isspace(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i, j
        cdef char* ptr
        cdef bint valid
        with nogil:
            for i in range(self.length):
                ptr = self.data_c + i * 256
                if ptr[0] == b'\0':
                    res.data_c[i] = 0
                    continue
                valid = 1
                for j in range(256):
                    if ptr[j] == b'\0': break
                    if not isspace(ptr[j]):
                        valid = 0
                        break
                res.data_c[i] = valid
        return res

    @cython.binding(False)
    def islower(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i, j
        cdef char* ptr
        cdef bint has_letter, all_lower
        with nogil:
            for i in range(self.length):
                ptr = self.data_c + i * 256
                if ptr[0] == b'\0':
                    res.data_c[i] = 0
                    continue
                has_letter = 0
                all_lower = 1
                for j in range(256):
                    if ptr[j] == b'\0': break
                    if isalpha(ptr[j]):
                        has_letter = 1
                        if not islower(ptr[j]):
                            all_lower = 0
                            break
                res.data_c[i] = 1 if (has_letter and all_lower) else 0
        return res

    @cython.binding(False)
    def isupper(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i, j
        cdef char* ptr
        cdef bint has_letter, all_upper
        with nogil:
            for i in range(self.length):
                ptr = self.data_c + i * 256
                if ptr[0] == b'\0':
                    res.data_c[i] = 0
                    continue
                has_letter = 0
                all_upper = 1
                for j in range(256):
                    if ptr[j] == b'\0': break
                    if isalpha(ptr[j]):
                        has_letter = 1
                        if not isupper(ptr[j]):
                            all_upper = 0
                            break
                res.data_c[i] = 1 if (has_letter and all_upper) else 0
        return res

    @cython.binding(False)
    def removeprefix(self, str prefix):
        cdef bytes b_pre = prefix.encode('ascii')
        cdef const char* c_pre = b_pre
        cdef Py_ssize_t pre_len = len(b_pre)
        cdef Char256Column_SoA res = self.copy()
        cdef Py_ssize_t i
        cdef char* ptr
        cdef Py_ssize_t curr_len
        with nogil:
            for i in range(self.length):
                ptr = res.data_c + i * 256
                curr_len = strlen(ptr)
                if curr_len >= pre_len and strncmp(ptr, c_pre, pre_len) == 0:
                    memmove(ptr, ptr + pre_len, curr_len - pre_len + 1)
        return res

    @cython.binding(False)
    def removesuffix(self, str suffix):
        cdef bytes b_suf = suffix.encode('ascii')
        cdef const char* c_suf = b_suf
        cdef Py_ssize_t suf_len = len(b_suf)
        cdef Char256Column_SoA res = self.copy()
        cdef Py_ssize_t i
        cdef char* ptr
        cdef Py_ssize_t curr_len
        with nogil:
            for i in range(self.length):
                ptr = res.data_c + i * 256
                curr_len = strlen(ptr)
                if curr_len >= suf_len and strncmp(ptr + curr_len - suf_len, c_suf, suf_len) == 0:
                    ptr[curr_len - suf_len] = b'\0'
        return res

    @cython.binding(False)
    def zfill(self, Py_ssize_t width):
        cdef Char256Column_SoA res = self.copy()
        cdef Py_ssize_t i, j, curr_len, pad
        cdef char* ptr
        with nogil:
            for i in range(self.length):
                ptr = res.data_c + i * 256
                curr_len = strlen(ptr)
                if curr_len >= width or curr_len >= 255:
                    continue
                pad = width - curr_len
                if curr_len + pad > 255:
                    pad = 255 - curr_len
                memmove(ptr + pad, ptr, curr_len + 1)
                for j in range(pad):
                    ptr[j] = b'0'
        return res

    @cython.binding(False)
    def unique(self):
        cdef set seen = set()
        cdef list result = []
        cdef Py_ssize_t i
        for i in range(self.length):
            val = (self.data_c + i * 256).decode('ascii')
            if val not in seen:
                seen.add(val)
                result.append(val)
        return result


# ==========================================
# 5. UCS2 存储引擎 (处理普通 Python 字符串)
# ==========================================
cdef inline int _ucs2_len(Py_UCS2* string) noexcept nogil:
    cdef int i = 0
    while string[i] != 0: i += 1
    return i

cdef inline Py_UCS2* _ucs2_strstr(Py_UCS2* haystack, int h_len, Py_UCS2* needle, int n_len) noexcept nogil:
    if n_len == 0: return haystack
    if h_len < n_len: return NULL
    cdef int i, j
    cdef bint match
    for i in range(h_len - n_len + 1):
        if haystack[i] == needle[0]:
            match = True
            for j in range(1, n_len):
                if haystack[i+j] != needle[j]:
                    match = False
                    break
            if match: return haystack + i
    return NULL


####################################################
#       UCS2 存储引擎
#================================================
# 通过统计业务数据设置如下
# 行数初始容量设为 64
# 初始字符池子：1024
####################################################
@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class UCS2Column_SoA(BaseColumn):
    cdef Py_UCS2* chars_c        # 连续字符池
    cdef Py_ssize_t* offsets_c   # 偏移量数组 (长度为 length + 1)
    cdef public Py_ssize_t char_capacity
    cdef public Py_ssize_t char_length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __cinit__(self, Py_ssize_t init_capacity=64):
        self.capacity = max(init_capacity, 1)
        self.length = 0

        self.char_capacity = max(self.capacity * 16, 1024)
        self.char_length = 0

        self.offsets_c = <Py_ssize_t*>malloc((self.capacity + 1) * sizeof(Py_ssize_t))
        self.chars_c = <Py_UCS2*>malloc(self.char_capacity * sizeof(Py_UCS2))

        if not self.offsets_c or not self.chars_c:
            raise MemoryError()

        self.offsets_c[0] = 0

    def __dealloc__(self):
        if self.offsets_c is not NULL: free(self.offsets_c)
        if self.chars_c is not NULL: free(self.chars_c)

    cdef void _reserve_rows(self, Py_ssize_t new_cap) except *:
        if new_cap > self.capacity:
            self.offsets_c = <Py_ssize_t*>realloc(self.offsets_c, (new_cap + 1) * sizeof(Py_ssize_t))
            if not self.offsets_c: raise MemoryError()
            self.capacity = new_cap

    cdef void _reserve_chars(self, Py_ssize_t new_char_cap) except *:
        if new_char_cap > self.char_capacity:
            self.chars_c = <Py_UCS2*>realloc(self.chars_c, new_char_cap * sizeof(Py_UCS2))
            if not self.chars_c: raise MemoryError()
            self.char_capacity = new_char_cap

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def append(self, str val):
        """支持任意长度的字符串追加，永不截断"""
        cdef Py_ssize_t str_len = PyUnicode_GET_LENGTH(val) if val else 0

        # 1. 检查并扩容行数
        if self.length >= self.capacity:
            self._reserve_rows(self.capacity * 2)

        # 2. 检查并扩容字符池
        if self.char_length + str_len > self.char_capacity:
            self._reserve_chars(max(self.char_capacity * 2, self.char_length + str_len + 1024))

        # 3. 极速填入字符
        if str_len > 0:
            if PyUnicode_KIND(val) == PyUnicode_2BYTE_KIND:
                memcpy(self.chars_c + self.char_length, PyUnicode_DATA(val), str_len * sizeof(Py_UCS2))
            else:
                # 兼容处理 1 byte (ASCII) 和 4 byte (Emoji/生僻字) 的情况
                self._fallback_copy_chars(val, str_len)

        # 4. 更新状态
        self.char_length += str_len
        self.length += 1
        self.offsets_c[self.length] = self.char_length

    cdef inline void _fallback_copy_chars(self, str val, Py_ssize_t str_len):
        """处理非 UCS2 编码字符串的后备方法"""
        cdef Py_ssize_t i
        cdef Py_UCS4 char_code
        for i in range(str_len):
            char_code = PyUnicode_READ_CHAR(val, i)
            if char_code > 0xFFFF:
                self.chars_c[self.char_length + i] = 0x003F # '?'
            else:
                self.chars_c[self.char_length + i] = <Py_UCS2>char_code

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def copy(self):
        cdef UCS2Column_SoA new_col = UCS2Column_SoA(self.capacity)
        new_col._reserve_chars(self.char_capacity)

        new_col.length = self.length
        new_col.char_length = self.char_length

        memcpy(new_col.offsets_c, self.offsets_c, (self.length + 1) * sizeof(Py_ssize_t))
        memcpy(new_col.chars_c, self.chars_c, self.char_length * sizeof(Py_UCS2))
        return new_col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def concat(self, UCS2Column_SoA other):
        self._reserve_rows(self.length + other.length)
        self._reserve_chars(self.char_length + other.char_length)

        cdef Py_ssize_t i
        # 复制字符数据
        memcpy(self.chars_c + self.char_length, other.chars_c, other.char_length * sizeof(Py_UCS2))

        # 调整并复制偏移量
        cdef Py_ssize_t base_offset = self.char_length
        for i in range(other.length):
            self.offsets_c[self.length + 1 + i] = base_offset + other.offsets_c[i + 1]

        self.length += other.length
        self.char_length += other.char_length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _reorder(self, list indices):
        """通过重建连续内存块实现重排，彻底清除碎片"""
        cdef Py_ssize_t n = self.length
        cdef Py_ssize_t* new_offsets = <Py_ssize_t*>malloc((self.capacity + 1) * sizeof(Py_ssize_t))
        cdef Py_UCS2* new_chars = <Py_UCS2*>malloc(self.char_capacity * sizeof(Py_UCS2))
        if not new_offsets or not new_chars: raise MemoryError()

        new_offsets[0] = 0
        cdef Py_ssize_t current_char_len = 0
        cdef Py_ssize_t idx, start, end, str_len
        cdef Py_ssize_t i

        for i in range(n):
            idx = <Py_ssize_t>indices[i]
            start = self.offsets_c[idx]
            end = self.offsets_c[idx + 1]
            str_len = end - start

            if str_len > 0:
                memcpy(new_chars + current_char_len, self.chars_c + start, str_len * sizeof(Py_UCS2))

            current_char_len += str_len
            new_offsets[i + 1] = current_char_len

        free(self.offsets_c)
        free(self.chars_c)
        self.offsets_c = new_offsets
        self.chars_c = new_chars
        self.char_length = current_char_len

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __getitem__(self, Py_ssize_t idx):
        if idx < 0 or idx >= self.length: raise IndexError()
        cdef Py_ssize_t start = self.offsets_c[idx]
        cdef Py_ssize_t str_len = self.offsets_c[idx + 1] - start

        if str_len == 0:
            return ""
        return PyUnicode_FromKindAndData(2, self.chars_c + start, str_len)

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
    def len(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                res.data_c[i] = <int>(self.offsets_c[i + 1] - self.offsets_c[i])
        return res

    @cython.binding(False)
    def startswith(self, str prefix):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self[i]
            res.data_c[i] = 1 if s.startswith(prefix) else 0
        return res

    @cython.binding(False)
    def endswith(self, str suffix):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self[i]
            res.data_c[i] = 1 if s.endswith(suffix) else 0
        return res

    @cython.binding(False)
    def contains(self, str substring):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self[i]
            res.data_c[i] = 1 if substring in s else 0
        return res

    @cython.binding(False)
    def find(self, str substring):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self[i]
            res.data_c[i] = <int>s.find(substring)
        return res

    @cython.binding(False)
    def replace(self, str old, str new):
        cdef UCS2Column_SoA res = UCS2Column_SoA(self.capacity)
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self[i]
            res.append(s.replace(old, new))
        return res

    @cython.binding(False)
    def lower(self):
        cdef UCS2Column_SoA res = UCS2Column_SoA(self.capacity)
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self[i]
            res.append(s.lower())
        return res

    @cython.binding(False)
    def upper(self):
        cdef UCS2Column_SoA res = UCS2Column_SoA(self.capacity)
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self[i]
            res.append(s.upper())
        return res

    @cython.binding(False)
    def strip(self):
        cdef UCS2Column_SoA res = UCS2Column_SoA(self.capacity)
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self[i]
            res.append(s.strip())
        return res

    @cython.binding(False)
    def lstrip(self):
        cdef UCS2Column_SoA res = UCS2Column_SoA(self.capacity)
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self[i]
            res.append(s.lstrip())
        return res

    @cython.binding(False)
    def rstrip(self):
        cdef UCS2Column_SoA res = UCS2Column_SoA(self.capacity)
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self[i]
            res.append(s.rstrip())
        return res

    @cython.binding(False)
    def isdigit(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self[i]
            res.data_c[i] = 1 if s.isdigit() else 0
        return res

    @cython.binding(False)
    def isalpha(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self[i]
            res.data_c[i] = 1 if s.isalpha() else 0
        return res

    @cython.binding(False)
    def isalnum(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self[i]
            res.data_c[i] = 1 if s.isalnum() else 0
        return res

    @cython.binding(False)
    def isspace(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self[i]
            res.data_c[i] = 1 if s.isspace() else 0
        return res

    @cython.binding(False)
    def unique(self):
        cdef set seen = set()
        cdef list result = []
        cdef Py_ssize_t i
        for i in range(self.length):
            s = self[i]
            if s not in seen:
                seen.add(s)
                result.append(s)
        return result


# ==========================================
# 5b. 时间戳列 (Timestamp YYYYMMDD)
# ==========================================
@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class TimestampColumn(BaseColumn):
    cdef long long* data_c

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __cinit__(self, Py_ssize_t init_capacity=16):
        self.capacity = max(init_capacity, 1)
        self.length = 0
        self.data_c = <long long*>malloc(self.capacity * sizeof(long long))
        if not self.data_c: raise MemoryError()

    def __dealloc__(self):
        if self.data_c is not NULL: free(self.data_c)

    cdef void _reserve(self, Py_ssize_t new_cap):
        if new_cap > self.capacity:
            self.data_c = <long long*>realloc(self.data_c, new_cap * sizeof(long long))
            if not self.data_c: raise MemoryError()
            self.capacity = new_cap

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def append(self, val):
        cdef long long v
        if isinstance(val, str):
            v = self._parse(<str>val)
        else:
            v = <long long>val
        if self.length >= self.capacity:
            self._reserve(self.capacity * 2)
        self.data_c[self.length] = v
        self.length += 1

    cdef inline long long _parse(self, str s) noexcept:
        cdef bytes b = s.encode('ascii')
        cdef const unsigned char* buf = <const unsigned char*>b
        cdef long long val = 0
        cdef int i = 0
        while buf[i] >= 48 and buf[i] <= 57:
            val = val * 10 + (buf[i] - 48)
            i += 1
        return val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def copy(self):
        cdef TimestampColumn new_col = TimestampColumn(self.capacity)
        new_col.length = self.length
        memcpy(new_col.data_c, self.data_c, self.length * sizeof(long long))
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
        cdef long long val = self.data_c[0]
        memmove(self.data_c, self.data_c + 1, (self.length - 1) * sizeof(long long))
        self.length -= 1
        return val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def concat(self, TimestampColumn other):
        self._reserve(self.length + other.length)
        memcpy(self.data_c + self.length, other.data_c, other.length * sizeof(long long))
        self.length += other.length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def max(self):
        if self.length == 0: raise ValueError("empty column")
        cdef long long max_val = self.data_c[0]
        for i in range(1, self.length):
            if self.data_c[i] > max_val: max_val = self.data_c[i]
        return max_val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def min(self):
        if self.length == 0: raise ValueError("empty column")
        cdef long long min_val = self.data_c[0]
        for i in range(1, self.length):
            if self.data_c[i] < min_val: min_val = self.data_c[i]
        return min_val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _reorder(self, list indices):
        cdef long long* new_data = <long long*>malloc(self.capacity * sizeof(long long))
        if not new_data: raise MemoryError()
        cdef Py_ssize_t i
        for i in range(self.length):
            new_data[i] = self.data_c[<Py_ssize_t>indices[i]]
        free(self.data_c)
        self.data_c = new_data

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __getitem__(self, Py_ssize_t idx):
        if idx < 0 or idx >= self.length: raise IndexError()
        return self.data_c[idx]

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

    # ---- 时间戳解析 ----
    @cython.binding(False)
    def year(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                res.data_c[i] = <int>(self.data_c[i] / 10000)
        return res

    @cython.binding(False)
    def month(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                res.data_c[i] = <int>((self.data_c[i] / 100) % 100)
        return res

    @cython.binding(False)
    def day(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                res.data_c[i] = <int>(self.data_c[i] % 100)
        return res

    @cython.binding(False)
    def strftime(self, str fmt):
        cdef Char256Column_SoA res = Char256Column_SoA(self.capacity)
        cdef Py_ssize_t i
        cdef long long val
        cdef int y, m, d
        for i in range(self.length):
            val = self.data_c[i]
            y = <int>(val / 10000)
            m = <int>((val / 100) % 100)
            d = <int>(val % 100)
            res.append(fmt.replace('%Y', f'{y:04d}').replace('%m', f'{m:02d}').replace('%d', f'{d:02d}'))
        return res


cdef void _c_argsort_char256(char* data, Py_ssize_t* indices, Py_ssize_t left, Py_ssize_t right) noexcept nogil:
    if left >= right: return
    cdef Py_ssize_t i = left, j = right
    cdef Py_ssize_t pivot_idx = indices[(left + right) // 2]
    # pivot 是指向对应字符串起始位置的指针
    cdef char* pivot_ptr = data + (pivot_idx * 256)
    cdef Py_ssize_t temp

    while i <= j:
        while strncmp(data + (indices[i] * 256), pivot_ptr, 256) < 0: i += 1
        while strncmp(data + (indices[j] * 256), pivot_ptr, 256) > 0: j -= 1
        if i <= j:
            temp = indices[i]
            indices[i] = indices[j]
            indices[j] = temp
            i += 1
            j -= 1

    _c_argsort_char256(data, indices, left, j)
    _c_argsort_char256(data, indices, i, right)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _c_argsort_numeric(NumericType* data, Py_ssize_t* indices, Py_ssize_t left, Py_ssize_t right) noexcept nogil:
    if left >= right:
        return

    cdef Py_ssize_t i = left, j = right
    cdef Py_ssize_t pivot_idx = indices[(left + right) // 2]
    cdef NumericType pivot = data[pivot_idx]
    cdef Py_ssize_t temp

    while i <= j:
        # 使用底层 C 数组的值进行比较
        while data[indices[i]] < pivot: i += 1
        while data[indices[j]] > pivot: j -= 1
        if i <= j:
            # 交换索引
            temp = indices[i]
            indices[i] = indices[j]
            indices[j] = temp
            i += 1
            j -= 1

    _c_argsort_numeric(data, indices, left, j)
    _c_argsort_numeric(data, indices, i, right)


cdef inline int _ucs2_cmp_at(Py_UCS2* chars, Py_ssize_t* offsets, Py_ssize_t i, Py_ssize_t j) noexcept nogil:
    cdef int i_start = offsets[i], i_len = offsets[i+1] - i_start
    cdef int j_start = offsets[j], j_len = offsets[j+1] - j_start
    cdef int k, min_len = i_len if i_len < j_len else j_len
    for k in range(min_len):
        if chars[i_start + k] < chars[j_start + k]: return -1
        if chars[i_start + k] > chars[j_start + k]: return 1
    if i_len < j_len: return -1
    if i_len > j_len: return 1
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _c_argsort_ucs2(Py_UCS2* chars, Py_ssize_t* offsets, Py_ssize_t* indices, Py_ssize_t left, Py_ssize_t right) noexcept nogil:
    if left >= right: return
    cdef Py_ssize_t i = left, j = right
    cdef Py_ssize_t pivot_idx = indices[(left + right) // 2]
    cdef Py_ssize_t temp

    while i <= j:
        while _ucs2_cmp_at(chars, offsets, indices[i], pivot_idx) < 0: i += 1
        while _ucs2_cmp_at(chars, offsets, indices[j], pivot_idx) > 0: j -= 1
        if i <= j:
            temp = indices[i]
            indices[i] = indices[j]
            indices[j] = temp
            i += 1
            j -= 1

    _c_argsort_ucs2(chars, offsets, indices, left, j)
    _c_argsort_ucs2(chars, offsets, indices, i, right)


# ==========================================
# 6. FastDataFrame: 列式极速聚合引擎
# ==========================================
@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class FastDataFrame:
    cdef public dict columns
    cdef public Py_ssize_t df_length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __init__(self, dict schema=None):
        self.columns = {}
        self.df_length = 0

        if schema:
            parsed = _parse_schema(schema)
            for col_name, dtype in parsed.items():
                if dtype == TYPE_INT:
                    self.columns[col_name] = IntColumn()
                elif dtype == TYPE_DOUBLE:
                    self.columns[col_name] = DoubleColumn()
                elif dtype == TYPE_CHAR256:
                    self.columns[col_name] = Char256Column_SoA()
                elif dtype == TYPE_UCS2:
                    self.columns[col_name] = UCS2Column_SoA()
                elif dtype == TYPE_TIMESTAMP:
                    self.columns[col_name] = TimestampColumn()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __getattr__(self, name):
        if name in self.columns: return self.columns[name]
        raise AttributeError(f"FastDataFrame has no column '{name}'")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __getitem__(self, name):
        return self.columns[name]

    @cython.binding(False)
    def append_row(self, **kwargs):
        for col_name, val in kwargs.items():
            self.columns[col_name].append(val)
        self.df_length += 1

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def copy(self):
        """深拷贝整个 DataFrame"""
        new_df = FastDataFrame()
        new_df.df_length = self.df_length
        for col_name, col_obj in self.columns.items():
            new_df.columns[col_name] = col_obj.copy()
        return new_df

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_front(self):
        """移除并返回第一行"""
        if self.df_length == 0: raise IndexError("pop from empty DataFrame")
        row = {}
        for col_name, col_obj in self.columns.items():
            row[col_name] = col_obj.pop_front()
        self.df_length -= 1
        return row

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_back(self):
        """移除并返回最后一行"""
        if self.df_length == 0: raise IndexError("pop from empty DataFrame")
        row = {}
        for col_name, col_obj in self.columns.items():
            row[col_name] = col_obj.pop_back()
        self.df_length -= 1
        return row

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def concat(self, FastDataFrame other):
        """合并另一个同结构的 DataFrame (行追加)"""
        for col_name, col_obj in self.columns.items():
            if col_name in other.columns:
                col_obj.concat(other.columns[col_name])
        self.df_length += other.df_length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def drop(self, str col_name):
        """删除指定列"""
        if col_name in self.columns:
            del self.columns[col_name]

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def argsort_by(self, str col_name, bint reverse=False):
        """
        根据指定列进行极速排序，返回排序后的行索引列表
        """
        if col_name not in self.columns:
            raise KeyError(f"Column '{col_name}' not found")

        cdef BaseColumn col = self.columns[col_name]
        cdef Py_ssize_t n = self.df_length
        if n == 0:
            return []

        cdef Py_ssize_t* indices_c = <Py_ssize_t*>malloc(n * sizeof(Py_ssize_t))
        if not indices_c:
            raise MemoryError()

        cdef Py_ssize_t i
        for i in range(n):
            indices_c[i] = i

        # 注意: nogil 可以在纯 C 执行期间释放 Python 全局锁，允许其他线程运行
        if isinstance(col, IntColumn):
            with nogil:
                _c_argsort_numeric((<IntColumn>col).data_c, indices_c, 0, n - 1)

        elif isinstance(col, DoubleColumn):
            with nogil:
                _c_argsort_numeric((<DoubleColumn>col).data_c, indices_c, 0, n - 1)

        elif isinstance(col, Char256Column_SoA):
            with nogil:
                _c_argsort_char256((<Char256Column_SoA>col).data_c, indices_c, 0, n - 1)

        elif isinstance(col, UCS2Column_SoA):
            with nogil:
                _c_argsort_ucs2((<UCS2Column_SoA>col).chars_c, (<UCS2Column_SoA>col).offsets_c, indices_c, 0, n - 1)

        elif isinstance(col, TimestampColumn):
            with nogil:
                _c_argsort_numeric((<TimestampColumn>col).data_c, indices_c, 0, n - 1)
        else:
            free(indices_c)
            raise TypeError("Sorting not supported for this column type")

        cdef Py_ssize_t temp
        if reverse:
            for i in range(n // 2):
                temp = indices_c[i]
                indices_c[i] = indices_c[n - 1 - i]
                indices_c[n - 1 - i] = temp

        cdef list py_indices = [indices_c[j] for j in range(n)]
        free(indices_c)
        return py_indices


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sort_values(self, str col_name, bint reverse=False, bint inplace=False):
        """
        对整个 DataFrame 进行排序
        :param inplace: 如果为 True，就地修改当前对象；如果为 False，返回排序后的新对象
        """
        cdef Py_ssize_t n = self.df_length
        if n == 0:
            return self if inplace else self._clone_empty()

        cdef list py_indices = self.argsort_by(col_name, reverse=reverse)
        cdef Py_ssize_t* idx_c = <Py_ssize_t*>malloc(n * sizeof(Py_ssize_t))
        if not idx_c: raise MemoryError()
        cdef Py_ssize_t i
        for i in range(n):
            idx_c[i] = py_indices[i]

        cdef FastDataFrame result_df
        if inplace:
            result_df = self
        else:
            result_df = FastDataFrame()
            result_df.df_length = n

        cdef str name
        cdef BaseColumn col
        cdef BaseColumn new_col

        for name, col in self.columns.items():
            if isinstance(col, IntColumn):
                new_col = self._reorder_int_column(<IntColumn>col, idx_c, n, inplace)
            elif isinstance(col, DoubleColumn):
                new_col = self._reorder_double_column(<DoubleColumn>col, idx_c, n, inplace)
            elif isinstance(col, Char256Column_SoA):
                new_col = self._reorder_char256_column(<Char256Column_SoA>col, idx_c, n, inplace)
            elif isinstance(col, UCS2Column_SoA):
                new_col = self._reorder_ucs2_column(<UCS2Column_SoA>col, idx_c, n, inplace)
            elif isinstance(col, TimestampColumn):
                new_col = self._reorder_timestamp_column(<TimestampColumn>col, idx_c, n, inplace)
            else:
                raise NotImplementedError()
            if not inplace:
                result_df.columns[name] = new_col

        free(idx_c)
        if not inplace:
            return result_df


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef IntColumn _reorder_int_column(self, IntColumn col, Py_ssize_t* indices, Py_ssize_t n, bint inplace):
        cdef int* old_data = col.data_c
        cdef int* new_data = <int*>malloc(col.capacity * sizeof(int))
        if not new_data: raise MemoryError()

        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                new_data[i] = old_data[indices[i]]

        if inplace:
            free(col.data_c)
            col.data_c = new_data
            return col
        else:
            new_col = IntColumn(col.capacity)
            free(new_col.data_c)
            new_col.data_c = new_data
            new_col.length = n
            return new_col

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef DoubleColumn _reorder_double_column(self, DoubleColumn col, Py_ssize_t* indices, Py_ssize_t n, bint inplace):
        cdef double* old_data = col.data_c
        cdef double* new_data = <double*>malloc(col.capacity * sizeof(double))
        if not new_data: raise MemoryError()

        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                new_data[i] = old_data[indices[i]]

        if inplace:
            free(col.data_c)
            col.data_c = new_data
            return col
        else:
            new_col = DoubleColumn(col.capacity)
            free(new_col.data_c)
            new_col.data_c = new_data
            new_col.length = n
            return new_col

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef Char256Column_SoA _reorder_char256_column(self, Char256Column_SoA col, Py_ssize_t* indices, Py_ssize_t n, bint inplace):
        cdef char* old_data = col.data_c
        cdef char* new_data = <char*>malloc(col.capacity * 256)
        if not new_data: raise MemoryError()

        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                memcpy(new_data + (i * 256), old_data + (indices[i] * 256), 256)

        if inplace:
            free(col.data_c)
            col.data_c = new_data
            return col
        else:
            new_col = Char256Column_SoA(col.capacity)
            free(new_col.data_c)
            new_col.data_c = new_data
            new_col.length = n
            return new_col

    cdef TimestampColumn _reorder_timestamp_column(self, TimestampColumn col, Py_ssize_t* indices, Py_ssize_t n, bint inplace):
        cdef long long* old_data = col.data_c
        cdef long long* new_data = <long long*>malloc(col.capacity * sizeof(long long))
        if not new_data: raise MemoryError()

        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                new_data[i] = old_data[indices[i]]

        if inplace:
            free(col.data_c)
            col.data_c = new_data
            return col
        else:
            new_col = TimestampColumn(col.capacity)
            free(new_col.data_c)
            new_col.data_c = new_data
            new_col.length = n
            return new_col


    @cython.binding(False)
    @cython.always_allow_keywords(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def diff(self, str col_name, int periods=1):
        """
        计算指定列的差分
        """
        if col_name not in self.columns:
            raise KeyError(f"Column '{col_name}' not found")

        cdef BaseColumn col = self.columns[col_name]
        if periods < 0:
            raise ValueError("periods must be >= 0")

        if isinstance(col, DoubleColumn):
            return self._diff_double(<DoubleColumn>col, periods)
        elif isinstance(col, IntColumn):
            return self._diff_int(<IntColumn>col, periods)
        else:
            raise TypeError(f"diff is only supported for numeric columns, got {type(col)}")


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef DoubleColumn _diff_double(self, DoubleColumn col, int periods):
        cdef Py_ssize_t n = col.length
        cdef DoubleColumn new_col = DoubleColumn(col.capacity)
        new_col.length = n

        cdef double* old_data = col.data_c
        cdef double* new_data = new_col.data_c
        cdef Py_ssize_t i

        with nogil:
            for i in range(n):
                if i < periods:
                    new_data[i] = NAN
                else:
                    new_data[i] = old_data[i] - old_data[i - periods]
        return new_col

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef IntColumn _diff_int(self, IntColumn col, int periods):
        cdef Py_ssize_t n = col.length
        cdef IntColumn new_col = IntColumn(col.capacity)
        new_col.length = n

        cdef int* old_data = col.data_c
        cdef int* new_data = new_col.data_c
        cdef Py_ssize_t i

        with nogil:
            for i in range(n):
                if i < periods:
                    new_data[i] = 0
                else:
                    new_data[i] = old_data[i] - old_data[i - periods]

        return new_col


    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def reset_index(self):
        cdef IntColumn idx_col = IntColumn(self.df_length)
        cdef Py_ssize_t i
        for i in range(self.df_length):
            idx_col.data_c[i] = <int>i
        idx_col.length = self.df_length
        self.columns['index'] = idx_col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __repr__(self):
        return f"<FastDataFrame: {self.df_length} rows, columns: {list(self.columns.keys())}>"


    # ==========================================
    # FastDataFrame 结构操控
    # ==========================================
    @cython.binding(False)
    def drop_col(self, str col_name):
        """等同于 del df['col']"""
        self.drop(col_name)

    @cython.binding(False)
    def sort_index(self, bint ascending=True, bint inplace=False):
        """按照 'index' 列重排"""
        if 'index' not in self.columns:
            raise KeyError("No 'index' column found. Call reset_index() first.")
        return self.sort_values('index', reverse=not ascending, inplace=inplace)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef UCS2Column_SoA _reorder_ucs2_column(self, UCS2Column_SoA col, Py_ssize_t* indices, Py_ssize_t n, bint inplace):
        """重新排布 UCS2 变长字符串的内存池，自动清除被删除项留下的内存碎片"""
        cdef Py_ssize_t* new_offsets = <Py_ssize_t*>malloc((n + 1) * sizeof(Py_ssize_t))
        cdef Py_UCS2* new_chars = <Py_UCS2*>malloc(col.char_capacity * sizeof(Py_UCS2))
        if not new_offsets or not new_chars: raise MemoryError()

        new_offsets[0] = 0
        cdef Py_ssize_t current_char_len = 0
        cdef Py_ssize_t i, idx, start, end, str_len
        cdef UCS2Column_SoA new_col

        # 内存压缩：只提取保留部分的字符数据到新的连续块
        with nogil:
            for i in range(n):
                idx = indices[i]
                start = col.offsets_c[idx]
                end = col.offsets_c[idx + 1]
                str_len = end - start

                if str_len > 0:
                    memcpy(new_chars + current_char_len, col.chars_c + start, str_len * sizeof(Py_UCS2))

                current_char_len += str_len
                new_offsets[i + 1] = current_char_len

        if inplace:
            free(col.offsets_c)
            free(col.chars_c)
            col.offsets_c = new_offsets
            col.chars_c = new_chars
            col.char_length = current_char_len
            col.length = n
            return col
        else:
            new_col = UCS2Column_SoA(col.capacity)
            free(new_col.offsets_c)
            free(new_col.chars_c)
            new_col.offsets_c = new_offsets
            new_col.chars_c = new_chars
            new_col.char_length = current_char_len
            new_col.char_capacity = col.char_capacity
            new_col.length = n
            return new_col

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.binding(False)
    def drop_with_index(self, list drop_indices, bint inplace=False):
        """
        批量删除指定行索引的行。
        采用 C 级别掩码标记机制，完美解决“越删索引越偏”的问题，
        全程仅发生一次内存拷贝重组，时间复杂度 O(N)。
        """
        cdef Py_ssize_t n_orig = self.df_length
        if n_orig == 0:
            return self if inplace else self.copy()

        # 1. 申请 C 级别布尔掩码数组并初始化为 0 (0: 保留, 1: 删除)
        cdef char* drop_mask = <char*>malloc(n_orig * sizeof(char))
        if not drop_mask: raise MemoryError()
        cdef Py_ssize_t i
        for i in range(n_orig):
            drop_mask[i] = 0

        # 2. 标记需要删除的行 (支持负数索引倒数)
        cdef Py_ssize_t idx
        for py_idx in drop_indices:
            idx = <Py_ssize_t>py_idx
            if idx < 0: idx += n_orig
            if idx < 0 or idx >= n_orig:
                free(drop_mask)
                raise IndexError(f"Index {py_idx} out of bounds")
            drop_mask[idx] = 1

        # 3. 统计需要保留的行数
        cdef Py_ssize_t keep_count = 0
        for i in range(n_orig):
            if drop_mask[i] == 0:
                keep_count += 1

        # 如果全删或全不删的捷径处理
        if keep_count == n_orig:
            free(drop_mask)
            return self if inplace else self.copy()
        if keep_count == 0:
            free(drop_mask)
            if inplace:
                # 若原地删除全部，直接重置长度
                self.df_length = 0
                for col_obj in self.columns.values(): col_obj.length = 0
                return self
            else:
                return FastDataFrame() # 返回空表

        # 4. 构建保留行的新连续 C 数组
        cdef Py_ssize_t* keep_idx_c = <Py_ssize_t*>malloc(keep_count * sizeof(Py_ssize_t))
        if not keep_idx_c:
            free(drop_mask)
            raise MemoryError()

        cdef Py_ssize_t j = 0
        for i in range(n_orig):
            if drop_mask[i] == 0:
                keep_idx_c[j] = i
                j += 1
        free(drop_mask) # 掩码用完即毁

        # 5. 调用底层的多态列级重建，并行化更新各列内存
        cdef FastDataFrame result_df
        if inplace:
            result_df = self
            self.df_length = keep_count
        else:
            result_df = FastDataFrame()
            result_df.df_length = keep_count

        cdef str name
        cdef BaseColumn col, new_col
        for name, col in self.columns.items():
            if isinstance(col, IntColumn):
                new_col = self._reorder_int_column(<IntColumn>col, keep_idx_c, keep_count, inplace)
            elif isinstance(col, DoubleColumn):
                new_col = self._reorder_double_column(<DoubleColumn>col, keep_idx_c, keep_count, inplace)
            elif isinstance(col, Char256Column_SoA):
                new_col = self._reorder_char256_column(<Char256Column_SoA>col, keep_idx_c, keep_count, inplace)
            elif isinstance(col, UCS2Column_SoA):
                new_col = self._reorder_ucs2_column(<UCS2Column_SoA>col, keep_idx_c, keep_count, inplace)
            elif isinstance(col, TimestampColumn):
                new_col = self._reorder_timestamp_column(<TimestampColumn>col, keep_idx_c, keep_count, inplace)
            else:
                free(keep_idx_c)
                raise TypeError(f"Drop not supported for column {name}")

            if not inplace:
                result_df.columns[name] = new_col

        free(keep_idx_c)
        if not inplace:
            return result_df

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.binding(False)
    def drop_raw(self, Py_ssize_t row_idx):
        """
        极速硬删除指定行（索引）。
        底层通过 C 的 memmove 将后续所有数据整体前移 1 位，时间复杂度 O(N)
        """
        if row_idx < 0 or row_idx >= self.df_length:
            raise IndexError("Row index out of range")

        cdef str name
        cdef BaseColumn col
        cdef Py_ssize_t rem_len = self.df_length - row_idx - 1

        for name, col in self.columns.items():
            if isinstance(col, IntColumn):
                if rem_len > 0:
                    memmove((<IntColumn>col).data_c + row_idx,
                            (<IntColumn>col).data_c + row_idx + 1,
                            rem_len * sizeof(int))
                (<IntColumn>col).length -= 1

            elif isinstance(col, DoubleColumn):
                if rem_len > 0:
                    memmove((<DoubleColumn>col).data_c + row_idx,
                            (<DoubleColumn>col).data_c + row_idx + 1,
                            rem_len * sizeof(double))
                (<DoubleColumn>col).length -= 1

            elif isinstance(col, Char256Column_SoA):
                if rem_len > 0:
                    memmove((<Char256Column_SoA>col).data_c + (row_idx * 256),
                            (<Char256Column_SoA>col).data_c + ((row_idx + 1) * 256),
                            rem_len * 256)
                (<Char256Column_SoA>col).length -= 1

            elif isinstance(col, UCS2Column_SoA):
                self._drop_row_ucs2(<UCS2Column_SoA>col, row_idx, rem_len)

            elif isinstance(col, TimestampColumn):
                if rem_len > 0:
                    memmove((<TimestampColumn>col).data_c + row_idx,
                            (<TimestampColumn>col).data_c + row_idx + 1,
                            rem_len * sizeof(long long))
                (<TimestampColumn>col).length -= 1

        self.df_length -= 1

    cdef void _drop_row_ucs2(self, UCS2Column_SoA col, Py_ssize_t row_idx, Py_ssize_t rem_len):
        cdef Py_ssize_t start_offset = col.offsets_c[row_idx]
        cdef Py_ssize_t end_offset = col.offsets_c[row_idx + 1]
        cdef Py_ssize_t char_len = end_offset - start_offset

        # 1. 挪动字符池 (清除碎片)
        cdef Py_ssize_t rem_chars = col.char_length - end_offset
        if rem_chars > 0:
            memmove(col.chars_c + start_offset,
                    col.chars_c + end_offset,
                    rem_chars * sizeof(Py_UCS2))
        col.char_length -= char_len

        # 2. 挪动并更新偏移量数组
        cdef Py_ssize_t i
        for i in range(row_idx, col.length):
            col.offsets_c[i] = col.offsets_c[i + 1] - char_len
        col.length -= 1

    @cython.binding(False)
    def argmax(self, str col_name):
        """返回指定列最大值所在的 DataFrame 级别索引"""
        return self.columns[col_name].idxmax()

    @cython.binding(False)
    def argmin(self, str col_name):
        """返回指定列最小值所在的 DataFrame 级别索引"""
        return self.columns[col_name].idxmin()

    @cython.binding(False)
    def unique(self, str col_name):
        """返回指定列的唯一值列表"""
        return self.columns[col_name].unique()

    # ==========================================
    # 持久化：Pickle / CSV / TSV
    # ==========================================
    @cython.binding(False)
    def to_pickle(self, filename):
        import pickle
        cdef dict data = {'df_length': self.df_length, 'columns': {}}
        cdef str name
        cdef BaseColumn col
        cdef list values
        cdef Py_ssize_t i
        for name, col in self.columns.items():
            values = [col[i] for i in range(col.length)]
            if isinstance(col, IntColumn):
                data['columns'][name] = ('int', values)
            elif isinstance(col, DoubleColumn):
                data['columns'][name] = ('float', values)
            elif isinstance(col, Char256Column_SoA):
                data['columns'][name] = ('char256', values)
            elif isinstance(col, UCS2Column_SoA):
                data['columns'][name] = ('ucs2', values)
            elif isinstance(col, TimestampColumn):
                data['columns'][name] = ('timestamp', values)
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    @cython.binding(False)
    def to_csv(self, filename, sep=','):
        import csv
        cdef list col_names = list(self.columns.keys())
        cdef Py_ssize_t i
        cdef str name
        cdef BaseColumn col
        cdef list row
        with open(filename, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter=sep)
            writer.writerow(col_names)
            for i in range(self.df_length):
                row = [str(self.columns[name][i]) for name in col_names]
                writer.writerow(row)

    @cython.binding(False)
    def to_tsv(self, filename):
        self.to_csv(filename, sep='\t')

    # ==========================================
    # 结构操控与元数据
    # ==========================================
    cdef FastDataFrame _take(self, Py_ssize_t* indices, Py_ssize_t n):
        cdef FastDataFrame result = FastDataFrame()
        result.df_length = n
        cdef str name
        cdef BaseColumn col
        for name, col in self.columns.items():
            if isinstance(col, IntColumn):
                result.columns[name] = self._reorder_int_column(<IntColumn>col, indices, n, False)
            elif isinstance(col, DoubleColumn):
                result.columns[name] = self._reorder_double_column(<DoubleColumn>col, indices, n, False)
            elif isinstance(col, Char256Column_SoA):
                result.columns[name] = self._reorder_char256_column(<Char256Column_SoA>col, indices, n, False)
            elif isinstance(col, UCS2Column_SoA):
                result.columns[name] = self._reorder_ucs2_column(<UCS2Column_SoA>col, indices, n, False)
            elif isinstance(col, TimestampColumn):
                result.columns[name] = self._reorder_timestamp_column(<TimestampColumn>col, indices, n, False)
        return result

    @cython.binding(False)
    def head(self, Py_ssize_t n=5):
        if n < 0: raise ValueError("n must be non-negative")
        n = min(n, self.df_length)
        if n == 0: return FastDataFrame()
        cdef Py_ssize_t* idx_c = <Py_ssize_t*>malloc(n * sizeof(Py_ssize_t))
        if not idx_c: raise MemoryError()
        cdef Py_ssize_t i
        for i in range(n): idx_c[i] = i
        cdef FastDataFrame result = self._take(idx_c, n)
        free(idx_c)
        return result

    @cython.binding(False)
    def tail(self, Py_ssize_t n=5):
        if n < 0: raise ValueError("n must be non-negative")
        n = min(n, self.df_length)
        if n == 0: return FastDataFrame()
        cdef Py_ssize_t start = self.df_length - n
        cdef Py_ssize_t* idx_c = <Py_ssize_t*>malloc(n * sizeof(Py_ssize_t))
        if not idx_c: raise MemoryError()
        cdef Py_ssize_t i
        for i in range(n): idx_c[i] = start + i
        cdef FastDataFrame result = self._take(idx_c, n)
        free(idx_c)
        return result

    @property
    def shape(self):
        return (self.df_length, len(self.columns))

    @property
    def col_names(self):
        return list(self.columns.keys())

    @property
    def dtypes(self):
        cdef dict result = {}
        cdef str name
        cdef BaseColumn col
        for name, col in self.columns.items():
            if isinstance(col, IntColumn):
                result[name] = int
            elif isinstance(col, DoubleColumn):
                result[name] = float
            elif isinstance(col, Char256Column_SoA):
                result[name] = Char256
            elif isinstance(col, UCS2Column_SoA):
                result[name] = str
            elif isinstance(col, TimestampColumn):
                result[name] = Timestamp
        return result

    @cython.binding(False)
    def rename(self, dict mapper):
        cdef FastDataFrame result = self.copy()
        cdef str old_name, new_name
        for old_name, new_name in mapper.items():
            if old_name in result.columns:
                result.columns[new_name] = result.columns.pop(old_name)
        return result

    @cython.binding(False)
    def describe(self):
        cdef dict stats = {}
        cdef str name
        cdef BaseColumn col
        for name, col in self.columns.items():
            if isinstance(col, (IntColumn, DoubleColumn)):
                stats[name] = {
                    'count': col.length,
                    'mean': col.mean(),
                    'std': col.std(),
                    'min': col.min(),
                    '50%': col.median(),
                    'max': col.max(),
                }
            elif isinstance(col, (Char256Column_SoA, UCS2Column_SoA)):
                stats[name] = {'count': col.length}
        return stats

    @cython.binding(False)
    def isnull(self):
        cdef FastDataFrame result = FastDataFrame()
        result.df_length = self.df_length
        cdef str name
        cdef BaseColumn col
        cdef IntColumn new_col
        cdef Py_ssize_t i
        cdef double* d_data
        for name, col in self.columns.items():
            new_col = IntColumn(self.df_length)
            new_col.length = self.df_length
            if isinstance(col, DoubleColumn):
                d_data = (<DoubleColumn>col).data_c
                with nogil:
                    for i in range(self.df_length):
                        new_col.data_c[i] = 1 if d_data[i] != d_data[i] else 0
            else:
                with nogil:
                    for i in range(self.df_length):
                        new_col.data_c[i] = 0
            result.columns[name] = new_col
        return result

    @cython.binding(False)
    def fillna(self, double value, col_names=None):
        cdef FastDataFrame result = self.copy()
        cdef str name
        cdef BaseColumn col
        cdef Py_ssize_t i
        cdef double* d_data
        if col_names is None:
            col_names = list(self.columns.keys())
        for name in col_names:
            if name not in result.columns: continue
            col = result.columns[name]
            if isinstance(col, DoubleColumn):
                d_data = (<DoubleColumn>col).data_c
                with nogil:
                    for i in range(self.df_length):
                        if d_data[i] != d_data[i]:
                            d_data[i] = value
        return result

    @cython.binding(False)
    def dropna(self):
        if self.df_length == 0: return self.copy()
        cdef Py_ssize_t n = self.df_length
        cdef char* keep_mask = <char*>malloc(n * sizeof(char))
        if not keep_mask: raise MemoryError()
        cdef Py_ssize_t i
        cdef double* d_data
        cdef str name
        cdef BaseColumn col

        for i in range(n): keep_mask[i] = 1

        for name, col in self.columns.items():
            if isinstance(col, DoubleColumn):
                d_data = (<DoubleColumn>col).data_c
                with nogil:
                    for i in range(n):
                        if keep_mask[i] and d_data[i] != d_data[i]:
                            keep_mask[i] = 0

        cdef Py_ssize_t keep_count = 0
        for i in range(n):
            if keep_mask[i]: keep_count += 1

        if keep_count == n:
            free(keep_mask)
            return self.copy()
        if keep_count == 0:
            free(keep_mask)
            return FastDataFrame()

        cdef Py_ssize_t* keep_idx = <Py_ssize_t*>malloc(keep_count * sizeof(Py_ssize_t))
        if not keep_idx: raise MemoryError()
        cdef Py_ssize_t j = 0
        for i in range(n):
            if keep_mask[i]:
                keep_idx[j] = i
                j += 1
        free(keep_mask)
        cdef FastDataFrame result = self._take(keep_idx, keep_count)
        free(keep_idx)
        return result


# ==========================================
# 7. 持久化函数 (Module-level)
# ==========================================
def from_pickle(filename):
    """从 pickle 文件恢复 FastDataFrame"""
    import pickle
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    df = FastDataFrame()
    df.df_length = data['df_length']
    for col_name, (typ, values) in data['columns'].items():
        if typ == 'int':
            col = IntColumn()
            for v in values:
                col.append(v)
        elif typ == 'float':
            col = DoubleColumn()
            for v in values:
                col.append(v)
        elif typ == 'char256':
            col = Char256Column_SoA()
            for v in values:
                col.append(v)
        elif typ == 'ucs2':
            col = UCS2Column_SoA()
            for v in values:
                col.append(v)
        elif typ == 'timestamp':
            col = TimestampColumn()
            for v in values:
                col.append(v)
        df.columns[col_name] = col
    return df


def read_csv(filename, sep=',', schema=None):
    """读取 CSV 文件，返回 FastDataFrame。自动推断列类型或由 schema 指定"""
    import csv
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=sep)
        headers = next(reader)
        rows = list(reader)

    cdef Py_ssize_t n_rows = len(rows)
    cdef Py_ssize_t n_cols = len(headers)
    cdef Py_ssize_t i, j
    df = FastDataFrame()

    if n_rows == 0:
        for j in range(n_cols):
            df.columns[headers[j]] = UCS2Column_SoA()
        return df

    if schema is not None:
        parsed = _parse_schema(schema)
        for j, name in enumerate(headers):
            py_type = parsed.get(name, str)
            if py_type == TYPE_INT:
                col = IntColumn(n_rows)
                for i in range(n_rows):
                    val = rows[i][j].strip()
                    col.append(int(val) if val else 0)
            elif py_type == TYPE_DOUBLE:
                col = DoubleColumn(n_rows)
                for i in range(n_rows):
                    val = rows[i][j].strip()
                    col.append(float(val) if val else NAN)
            elif py_type == TYPE_CHAR256:
                col = Char256Column_SoA(n_rows)
                for i in range(n_rows):
                    col.append(rows[i][j])
            elif py_type == TYPE_TIMESTAMP:
                col = TimestampColumn(n_rows)
                for i in range(n_rows):
                    col.append(rows[i][j].strip())
            else:
                col = UCS2Column_SoA(n_rows)
                for i in range(n_rows):
                    col.append(rows[i][j])
            df.columns[name] = col
        df.df_length = n_rows
        return df

    # 类型推断
    cdef list types = []
    cdef bint all_int, all_float
    for j in range(n_cols):
        all_int = True
        all_float = True
        for i in range(n_rows):
            val = rows[i][j].strip()
            if val == '':
                continue
            if all_int:
                try:
                    int(val)
                except ValueError:
                    all_int = False
            if all_float:
                try:
                    float(val)
                except ValueError:
                    all_float = False
        if all_int:
            types.append(int)
        elif all_float:
            types.append(float)
        else:
            types.append(str)

    for j, name in enumerate(headers):
        dtype = types[j]
        if dtype is int:
            col = IntColumn(n_rows)
            for i in range(n_rows):
                val = rows[i][j].strip()
                col.append(int(val) if val else 0)
        elif dtype is float:
            col = DoubleColumn(n_rows)
            for i in range(n_rows):
                val = rows[i][j].strip()
                col.append(float(val) if val else NAN)
        else:
            col = UCS2Column_SoA(n_rows)
            for i in range(n_rows):
                col.append(rows[i][j])
        df.columns[name] = col
    df.df_length = n_rows
    return df


def read_tsv(filename, schema=None):
    """读取 TSV 文件（tab 分隔），返回 FastDataFrame"""
    return read_csv(filename, sep='\t', schema=schema)


# ==========================================
# 8. 行列合并与连接
# ==========================================
cdef inline object _default_val(BaseColumn col):
    if isinstance(col, IntColumn): return 0
    if isinstance(col, DoubleColumn): return float('nan')
    if isinstance(col, TimestampColumn): return 0
    return ''


cdef inline BaseColumn _empty_col(BaseColumn col):
    if isinstance(col, IntColumn): return IntColumn()
    if isinstance(col, DoubleColumn): return DoubleColumn()
    if isinstance(col, Char256Column_SoA): return Char256Column_SoA()
    if isinstance(col, TimestampColumn): return TimestampColumn()
    return UCS2Column_SoA()


def concat_outer(list dfs):
    """纵向拼接多个 DataFrame（外连接：取所有列的并集），重置索引"""
    if not dfs: return FastDataFrame()
    all_cols = set()
    for df in dfs:
        all_cols.update(df.columns.keys())
    all_cols = list(all_cols)

    result = FastDataFrame()
    col_types = {}
    for name in all_cols:
        for df in dfs:
            if name in df.columns:
                col_types[name] = df.columns[name]
                break

    for name in all_cols:
        result.columns[name] = _empty_col(col_types[name])

    for df in dfs:
        for i in range(df.df_length):
            for name in all_cols:
                col = result.columns[name]
                if name in df.columns:
                    col.append(df.columns[name][i])
                else:
                    col.append(_default_val(col_types[name]))
        result.df_length += df.df_length

    result.reset_index()
    return result


def concat_inner(list dfs):
    """纵向拼接多个 DataFrame（内连接：仅取所有列的交集），重置索引"""
    if not dfs: return FastDataFrame()
    common = set(dfs[0].columns.keys())
    for df in dfs[1:]:
        common &= set(df.columns.keys())
    common = list(common)

    result = FastDataFrame()
    for name in common:
        result.columns[name] = _empty_col(dfs[0].columns[name])

    for df in dfs:
        for i in range(df.df_length):
            for name in common:
                result.columns[name].append(df.columns[name][i])
        result.df_length += df.df_length

    result.reset_index()
    return result


cdef list _build_pairs(df1, df2, str left_on, str right_on, str how):
    cdef dict right_lookup = {}
    cdef BaseColumn right_key = df2.columns[right_on]
    cdef Py_ssize_t i, j
    cdef object val

    for i in range(df2.df_length):
        val = right_key[i]
        if val not in right_lookup:
            right_lookup[val] = []
        right_lookup[val].append(i)

    cdef BaseColumn left_key = df1.columns[left_on]
    cdef list pairs = []
    cdef set right_matched = set()
    cdef list matches

    for i in range(df1.df_length):
        val = left_key[i]
        matches = right_lookup.get(val)
        if matches:
            for j in matches:
                pairs.append((i, j))
                right_matched.add(j)
        elif how in ('left', 'outer'):
            pairs.append((i, -1))  # -1 = no match

    if how in ('right', 'outer'):
        for j in range(df2.df_length):
            if j not in right_matched:
                pairs.append((-1, j))

    return pairs


def merge_inner(dfs, str left_on, str right_on):
    """内连接：仅保留两表共同拥有匹配键的行"""
    df1, df2 = dfs
    cdef list pairs = _build_pairs(df1, df2, left_on, right_on, 'inner')
    return _build_merge_result(df1, df2, left_on, right_on, pairs)


def merge_left(dfs, str left_on, str right_on):
    """左连接：保留左表所有行，右表无匹配则填充默认值"""
    df1, df2 = dfs
    cdef list pairs = _build_pairs(df1, df2, left_on, right_on, 'left')
    return _build_merge_result(df1, df2, left_on, right_on, pairs)


def merge_right(dfs, str left_on, str right_on):
    """右连接：保留右表所有行，左表无匹配则填充默认值"""
    df1, df2 = dfs
    cdef list pairs = _build_pairs(df1, df2, left_on, right_on, 'right')
    return _build_merge_result(df1, df2, left_on, right_on, pairs)


def merge_outer(dfs, str left_on, str right_on):
    """外连接：保留两表所有行，无匹配则填充默认值"""
    df1, df2 = dfs
    cdef list pairs = _build_pairs(df1, df2, left_on, right_on, 'outer')
    return _build_merge_result(df1, df2, left_on, right_on, pairs)


cdef FastDataFrame _build_merge_result(df1, df2, str left_on, str right_on, list pairs):
    cdef FastDataFrame result = FastDataFrame()
    cdef Py_ssize_t n = len(pairs)
    if n == 0:
        result.df_length = 0
        for name in df1.columns:
            result.columns[name] = _empty_col(df1.columns[name])
        for name in df2.columns:
            if name == right_on and name == left_on:
                continue
            if name not in result.columns:
                result.columns[name] = _empty_col(df2.columns[name])
        return result

    cdef tuple pair
    cdef Py_ssize_t i, j
    cdef BaseColumn src_col, new_col

    for name in df1.columns:
        src_col = df1.columns[name]
        new_col = _empty_col(src_col)
        for pair in pairs:
            i = pair[0]
            if i >= 0:
                new_col.append(src_col[i])
            else:
                new_col.append(_default_val(src_col))
        result.columns[name] = new_col

    for name in df2.columns:
        if name == right_on and name == left_on:
            continue
        col_name = name
        if col_name in result.columns:
            col_name = name + '_right'
        src_col = df2.columns[name]
        new_col = _empty_col(src_col)
        for pair in pairs:
            j = pair[1]
            if j >= 0:
                new_col.append(src_col[j])
            else:
                new_col.append(_default_val(src_col))
        result.columns[col_name] = new_col

    result.df_length = n
    return result