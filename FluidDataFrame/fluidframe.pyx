# %%cython
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport cython
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

class Char256:
    """用于在 Schema 中标记纯 ASCII 256字节定长字符串类型的占位类"""
    pass

@cython.binding(False)
@cython.always_allow_keywords(False)
def _parse_schema(dict schema_dict):
    """将 Python 类型映射到内部的 C 枚举"""
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
            self.capacity = new_cap

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    cpdef void append(self, int val):
        if self.length >= self.capacity:
            self._reserve(self.capacity * 2)
        self.data_c[self.length] = val
        self.length += 1

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    cpdef IntColumn copy(self):
        cdef IntColumn new_col = IntColumn(self.capacity)
        new_col.length = self.length
        memcpy(new_col.data_c, self.data_c, self.length * sizeof(int))
        return new_col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    cpdef int pop_back(self):
        if self.length == 0: raise IndexError("pop from empty column")
        self.length -= 1
        return self.data_c[self.length]

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    cpdef int pop_front(self):
        if self.length == 0: raise IndexError("pop from empty column")
        cdef int val = self.data_c[0]
        memmove(self.data_c, self.data_c + 1, (self.length - 1) * sizeof(int))
        self.length -= 1
        return val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    cpdef void concat(self, IntColumn other):
        self._reserve(self.length + other.length)
        memcpy(self.data_c + self.length, other.data_c, other.length * sizeof(int))
        self.length += other.length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    cpdef int max(self):
        if self.length == 0: raise ValueError("empty column")
        cdef int max_val = self.data_c[0]
        for i in range(1, self.length):
            if self.data_c[i] > max_val: max_val = self.data_c[i]
        return max_val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    cpdef int min(self):
        if self.length == 0: raise ValueError("empty column")
        cdef int min_val = self.data_c[0]
        for i in range(1, self.length):
            if self.data_c[i] < min_val: min_val = self.data_c[i]
        return min_val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    cpdef double mean(self):
        if self.length == 0: return NAN
        cdef double sum_val = 0
        for i in range(self.length):
            sum_val += self.data_c[i]
        return sum_val / self.length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    cpdef double var(self):
        if self.length < 2: return 0.0
        cdef double m = self.mean()
        cdef double sum_sq = 0
        for i in range(self.length):
            sum_sq += (self.data_c[i] - m) * (self.data_c[i] - m)
        return sum_sq / (self.length - 1)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    cpdef double std(self):
        return sqrt(self.var())

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    cpdef IntColumn diff(self):
        """计算一阶差分，首位用 0 填充"""
        cdef IntColumn res = IntColumn(self.capacity)
        if self.length == 0: return res
        res.append(0)
        for i in range(1, self.length):
            res.append(self.data_c[i] - self.data_c[i-1])
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    cpdef void _reorder(self, list indices):
        cdef int* new_data = <int*>malloc(self.capacity * sizeof(int))
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

    def __dealloc__(self):
        if self.data_c is not NULL: free(self.data_c)

    cdef void _reserve(self, Py_ssize_t new_cap):
        if new_cap > self.capacity:
            self.data_c = <double*>realloc(self.data_c, new_cap * sizeof(double))
            self.capacity = new_cap

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    cpdef void append(self, double val):
        if self.length >= self.capacity:
            self._reserve(self.capacity * 2)
        self.data_c[self.length] = val
        self.length += 1

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    cpdef DoubleColumn copy(self):
        cdef DoubleColumn new_col = DoubleColumn(self.capacity)
        new_col.length = self.length
        memcpy(new_col.data_c, self.data_c, self.length * sizeof(double))
        return new_col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    cpdef double pop_back(self):
        if self.length == 0: raise IndexError("pop from empty column")
        self.length -= 1
        return self.data_c[self.length]

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    cpdef double pop_front(self):
        if self.length == 0: raise IndexError("pop from empty column")
        cdef double val = self.data_c[0]
        memmove(self.data_c, self.data_c + 1, (self.length - 1) * sizeof(double))
        self.length -= 1
        return val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    cpdef void concat(self, DoubleColumn other):
        self._reserve(self.length + other.length)
        memcpy(self.data_c + self.length, other.data_c, other.length * sizeof(double))
        self.length += other.length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    cpdef double max(self):
        if self.length == 0: raise ValueError("empty column")
        cdef double max_val = self.data_c[0]
        for i in range(1, self.length):
            if self.data_c[i] > max_val: max_val = self.data_c[i]
        return max_val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    cpdef double min(self):
        if self.length == 0: raise ValueError("empty column")
        cdef double min_val = self.data_c[0]
        for i in range(1, self.length):
            if self.data_c[i] < min_val: min_val = self.data_c[i]
        return min_val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    cpdef double mean(self):
        if self.length == 0: return NAN
        cdef double sum_val = 0
        for i in range(self.length):
            sum_val += self.data_c[i]
        return sum_val / self.length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    cpdef double var(self):
        if self.length < 2: return 0.0
        cdef double m = self.mean()
        cdef double sum_sq = 0
        for i in range(self.length):
            sum_sq += (self.data_c[i] - m) * (self.data_c[i] - m)
        return sum_sq / (self.length - 1)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    cpdef double std(self):
        return sqrt(self.var())

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    cpdef void _reorder(self, list indices):
        cdef double* new_data = <double*>malloc(self.capacity * sizeof(double))
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
            self.capacity = new_cap

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    cpdef void append(self, str val):
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
    cpdef Char256Column_SoA copy(self):
        cdef Char256Column_SoA new_col = Char256Column_SoA(self.capacity)
        new_col.length = self.length
        memcpy(new_col.data_c, self.data_c, self.length * 256)
        return new_col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    cpdef str pop_back(self):
        if self.length == 0: raise IndexError("pop from empty column")
        self.length -= 1
        cdef char* ptr = self.data_c + (self.length * 256)
        return ptr.decode('ascii')

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    cpdef str pop_front(self):
        if self.length == 0: raise IndexError("pop from empty column")
        cdef char* ptr = self.data_c
        cdef str val = ptr.decode('ascii')
        memmove(self.data_c, self.data_c + 256, (self.length - 1) * 256)
        self.length -= 1
        return val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    cpdef void concat(self, Char256Column_SoA other):
        self._reserve(self.length + other.length)
        memcpy(self.data_c + (self.length * 256), other.data_c, other.length * 256)
        self.length += other.length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    cpdef void _reorder(self, list indices):
        cdef char* new_data = <char*>malloc(self.capacity * 256)
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
            self.capacity = new_cap

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    cpdef void append(self, str val):
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
    cpdef UCS2Column_SoA copy(self):
        cdef UCS2Column_SoA new_col = UCS2Column_SoA(self.capacity)
        new_col.length = self.length
        memcpy(new_col.data_c, self.data_c, self.length * 128 * sizeof(Py_UCS2))
        return new_col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    cpdef str pop_back(self):
        if self.length == 0: raise IndexError("pop from empty column")
        self.length -= 1
        cdef Py_UCS2* ptr = self.data_c + (self.length * 128)
        return PyUnicode_FromKindAndData(2, ptr, _ucs2_len(ptr))

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    cpdef str pop_front(self):
        if self.length == 0: raise IndexError("pop from empty column")
        cdef Py_UCS2* ptr = self.data_c
        cdef str val = PyUnicode_FromKindAndData(2, ptr, _ucs2_len(ptr))
        memmove(self.data_c, self.data_c + 128, (self.length - 1) * 128 * sizeof(Py_UCS2))
        self.length -= 1
        return val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    cpdef void concat(self, UCS2Column_SoA other):
        self._reserve(self.length + other.length)
        memcpy(self.data_c + (self.length * 128), other.data_c, other.length * 128 * sizeof(Py_UCS2))
        self.length += other.length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    cpdef void _reorder(self, list indices):
        cdef Py_UCS2* new_data = <Py_UCS2*>malloc(self.capacity * 128 * sizeof(Py_UCS2))
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
        """按行追加数据，底层自动路由至对应列的追加"""
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
    def sort_by(self, str col_name, bint reverse=False):
        if col_name not in self.columns: raise ValueError(f"Column {col_name} not found")

        target_col = self.columns[col_name]
        cdef list temp_list = [target_col[i] for i in range(self.df_length)]

        cdef list indices = sorted(range(self.df_length), key=lambda i: temp_list[i], reverse=reverse)

        for col in self.columns.values():
            col._reorder(indices)

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