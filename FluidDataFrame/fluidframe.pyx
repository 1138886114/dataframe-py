# %%cython
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport cython
from libc.string cimport strncmp
from libc.stdlib cimport malloc, realloc, free
from libc.string cimport strstr, strlen, strncmp, memmove, memcpy, strncat, strncpy
from libc.math cimport sqrt, NAN
from cpython.unicode cimport (
    PyUnicode_4BYTE_KIND, PyUnicode_FromKindAndData,
    PyUnicode_GET_LENGTH, PyUnicode_READ_CHAR, PyUnicode_AsUCS4,
    Py_UCS2
)

# ==========================================
# 1. 枚举与基础类型定义
# ==========================================
cdef enum DataType:
    TYPE_INT = 0
    TYPE_FLOAT = 1
    TYPE_DOUBLE = 2
    TYPE_CHAR256 = 3
    TYPE_UCS2 = 4


ctypedef fused NumericType:
    int
    double


class Char256:
    """用于在 Schema 中标记纯 ASCII 256字节定长字符串类型的占位类"""
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

@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class UCS2Column_SoA(BaseColumn):
    cdef Py_UCS2* data_c

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __cinit__(self, Py_ssize_t init_capacity=16):
        self.capacity = max(init_capacity, 1)
        self.length = 0
        self.data_c = <Py_UCS2*>malloc(self.capacity * 128 * sizeof(Py_UCS2))
        if not self.data_c: raise MemoryError()

    def __dealloc__(self):
        if self.data_c is not NULL: free(self.data_c)

    cdef void _reserve(self, Py_ssize_t new_cap):
        if new_cap > self.capacity:
            self.data_c = <Py_UCS2*>realloc(self.data_c, new_cap * 128 * sizeof(Py_UCS2))
            if not self.data_c: raise MemoryError()
            self.capacity = new_cap

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def append(self, str val):
        if self.length >= self.capacity:
            self._reserve(self.capacity * 2)

        cdef Py_UCS2* target_ptr = self.data_c + (self.length * 128)
        cdef Py_ssize_t str_len = PyUnicode_GET_LENGTH(val)
        cdef Py_ssize_t copy_len = str_len if str_len < 127 else 127

        cdef Py_ssize_t i
        cdef Py_UCS4 char_code

        for i in range(copy_len):
            char_code = PyUnicode_READ_CHAR(val, i)
            if char_code > 0xFFFF:
                target_ptr[i] = 0x003F # '?'
            else:
                target_ptr[i] = <Py_UCS2>char_code

        target_ptr[copy_len] = 0
        self.length += 1

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def copy(self):
        cdef UCS2Column_SoA new_col = UCS2Column_SoA(self.capacity)
        new_col.length = self.length
        memcpy(new_col.data_c, self.data_c, self.length * 128 * sizeof(Py_UCS2))
        return new_col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_back(self):
        if self.length == 0: raise IndexError("pop from empty column")
        self.length -= 1
        cdef Py_UCS2* ptr = self.data_c + (self.length * 128)
        return PyUnicode_FromKindAndData(2, ptr, _ucs2_len(ptr))

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_front(self):
        if self.length == 0: raise IndexError("pop from empty column")
        cdef Py_UCS2* ptr = self.data_c
        cdef str val = PyUnicode_FromKindAndData(2, ptr, _ucs2_len(ptr))
        memmove(self.data_c, self.data_c + 128, (self.length - 1) * 128 * sizeof(Py_UCS2))
        self.length -= 1
        return val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def concat(self, UCS2Column_SoA other):
        self._reserve(self.length + other.length)
        memcpy(self.data_c + (self.length * 128), other.data_c, other.length * 128 * sizeof(Py_UCS2))
        self.length += other.length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _reorder(self, list indices):
        cdef Py_UCS2* new_data = <Py_UCS2*>malloc(self.capacity * 128 * sizeof(Py_UCS2))
        if not new_data: raise MemoryError()
        cdef Py_ssize_t i
        for i in range(self.length):
            memcpy(new_data + (i * 128), self.data_c + (<Py_ssize_t>indices[i] * 128), 128 * sizeof(Py_UCS2))
        free(self.data_c)
        self.data_c = new_data

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __getitem__(self, Py_ssize_t idx):
        if idx < 0 or idx >= self.length: raise IndexError()
        cdef Py_UCS2* ptr = self.data_c + (idx * 128)
        cdef int char_len = _ucs2_len(ptr)
        return PyUnicode_FromKindAndData(2, ptr, char_len)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def contains_exact(self, str keyword):
        cdef Py_ssize_t kw_len = PyUnicode_GET_LENGTH(keyword)
        cdef Py_UCS2 needle_buf[128]
        cdef Py_ssize_t i

        for i in range(kw_len):
            needle_buf[i] = <Py_UCS2>PyUnicode_READ_CHAR(keyword, i)
        needle_buf[kw_len] = 0

        cdef int* out_idx = <int*>malloc(self.length * sizeof(int))
        if not out_idx: raise MemoryError()
        cdef int count = 0
        cdef Py_UCS2* current_str
        cdef int cur_len

        with nogil:
            for i in range(self.length):
                current_str = self.data_c + (i * 128)
                cur_len = _ucs2_len(current_str)
                if _ucs2_strstr(current_str, cur_len, needle_buf, kw_len) != NULL:
                    out_idx[count] = i
                    count += 1

        cdef list res = [out_idx[j] for j in range(count)]
        free(out_idx)
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
        cdef DoubleColumn new_col = DoubleColumn(init_capacity=col.capacity)
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
        cdef IntColumn new_col = IntColumn(init_capacity=col.capacity)
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
            idx_col.append(i)
        self.columns['index'] = idx_col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __repr__(self):
        return f"<FastDataFrame: {self.df_length} rows, columns: {list(self.columns.keys())}>"