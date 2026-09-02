# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False

cimport cython
from libc.string cimport memcpy, memmove, memset, strstr, strlen, strncmp, strcmp
from libc.stdlib cimport malloc, free

cdef extern from "<ctype.h>" nogil:
    int isdigit(int)
    int isspace(int)
    int isalpha(int)
    int isalnum(int)
    int islower(int)
    int isupper(int)

import re

from fluidframe.core._base cimport (
    BaseColumn, aligned_malloc, aligned_realloc, os_aligned_free,
    TYPE_CHAR256, TYPE_INT, TYPE_DOUBLE, TYPE_UCS2, TYPE_BOOL,
    TYPE_TIMESTAMP, TYPE_CATEGORY, TYPE_DATETIME, TYPE_TIME, _resolve_astype_target
)
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.doublecolumn cimport DoubleColumn
from fluidframe.core.boolcolumn cimport BoolColumn
from fluidframe.core.ucs2column cimport UCS2Column_SoA
from fluidframe.core.datecolumn cimport DateColumn
from fluidframe.core.categorycolumn cimport CategoryColumn
from fluidframe.core.datetimecolumn cimport DateTimeColumn
from fluidframe.core.timecolumn cimport TimeColumn
from fluidframe.core.char256_str_accessor import StringAccessor_Char256

# ============================================================
#  底层纯 C 辅助函数 (无 GIL / 无分支 / 边界安全)
# ============================================================

cdef inline bint _c_str_contains(const char* source, const char* target) noexcept nogil:
    return strstr(source, target) != NULL

cdef inline void _c_str_replace_inplace(char* buffer, const char* old_sub, const char* new_sub, size_t old_len, size_t new_len) noexcept nogil:
    if old_len == 0: return
    cdef char* match = strstr(buffer, old_sub)
    cdef size_t cur_len, rem

    while match != NULL:
        cur_len = strlen(buffer)
        rem = cur_len - (match - buffer) - old_len
        if cur_len - old_len + new_len > 255:
            break
        if new_len != old_len:
            memmove(match + new_len, match + old_len, rem + 1)
        if new_len > 0:
            memcpy(match, new_sub, new_len)
        match = strstr(match + new_len, old_sub)

cdef inline bint _c_startswith(const char* data, const char* pat, size_t pat_len) noexcept nogil:
    if strlen(data) < pat_len:
        return 0
    return strncmp(data, pat, pat_len) == 0

cdef inline bint _c_endswith(const char* data, const char* pat, size_t pat_len) noexcept nogil:
    cdef size_t n = strlen(data)
    if n < pat_len:
        return 0
    return strncmp(data + (n - pat_len), pat, pat_len) == 0

cdef inline Py_ssize_t _c_find(const char *data, const char *pat, size_t pat_len) noexcept nogil:
    cdef const char* match = strstr(data, pat)
    if match != NULL:
        return match - data
    return -1

cdef inline Py_ssize_t _c_rfind(const char *data, const char *pat, size_t pat_len) noexcept nogil:
    cdef Py_ssize_t n = <Py_ssize_t>strlen(data)
    cdef Py_ssize_t p_len = <Py_ssize_t>pat_len
    if p_len == 0:
        return n
    if p_len > n:
        return -1
    cdef Py_ssize_t i
    for i in range(n - p_len, -1, -1):
        if strncmp(data + i, pat, pat_len) == 0:
            return i
    return -1

cdef inline Py_ssize_t _c_count(const char *data, const char *pat, size_t pat_len) noexcept nogil:
    if pat_len == 0:
        return <Py_ssize_t>strlen(data) + 1
    cdef Py_ssize_t cnt = 0
    cdef const char* match = strstr(data, pat)
    while match != NULL:
        cnt += 1
        match = strstr(match + pat_len, pat)
    return cnt


# ============================================================
#  Char256Column_SoA 类定义
# ============================================================

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef class Char256Column_SoA(BaseColumn):

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __cinit__(self, Py_ssize_t init_capacity=16):
        self.capacity = max(init_capacity, 1)
        self.length = 0
        self.dtype = TYPE_CHAR256
        self.data_c = <char*>aligned_malloc(self.capacity * 256)
        if not self.data_c:
            raise MemoryError()
        memset(self.data_c, 0, self.capacity * 256)

    def __dealloc__(self):
        if self.data_c != NULL:
            os_aligned_free(self.data_c)
            self.data_c = NULL

    cdef Py_ssize_t get_size(self):
        return self.capacity * 256

    cdef void _reserve(self, Py_ssize_t new_cap):
        if new_cap <= self.capacity:
            return
        cdef char* new_data = <char*>aligned_realloc(self.data_c, self.capacity * 256, new_cap * 256)
        if not new_data:
            raise MemoryError()
        memset(new_data + (self.capacity * 256), 0, (new_cap - self.capacity) * 256)
        self.data_c = new_data
        self.capacity = new_cap

    cdef inline str _get_str(self, Py_ssize_t idx):
        cdef char* ptr = self.data_c + (idx * 256)
        if ptr[0] == 0:
            return ""
        return ptr.decode('ascii', errors='replace')

    cdef void _set_item(self, Py_ssize_t idx, str val) except *:
        """在固定 256 字节槽位上就地写入单字节字符串。"""
        cdef bytes b_val = val.encode('ascii', errors='ignore')
        cdef const char* src = b_val
        cdef size_t slen = len(b_val)
        if slen > 255: slen = 255
        cdef char* dest = self.data_c + (idx * 256)
        memcpy(dest, src, slen)
        dest[slen] = 0

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def append(self, str val):
        if self.length >= self.capacity:
            self._reserve(max(16, self.capacity * 2))
        cdef bytes b_val = val.encode('ascii', errors='ignore') if val is not None else b''
        cdef const char* src = b_val
        cdef size_t slen = len(b_val)
        if slen > 255: slen = 255
        cdef char* dest = self.data_c + (self.length * 256)
        memcpy(dest, src, slen)
        dest[slen] = 0
        self.length += 1

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def copy(self):
        cdef Py_ssize_t cap = max(self.length, 1)
        cdef Char256Column_SoA new_col = Char256Column_SoA(cap)
        new_col.length = self.length
        if self.length > 0:
            memcpy(new_col.data_c, self.data_c, self.length * 256)
        return new_col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_back(self):
        if self.length == 0:
            raise IndexError("pop from empty column")
        self.length -= 1
        return self._get_str(self.length)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_front(self):
        if self.length == 0:
            raise IndexError("pop from empty column")
        cdef str val = self._get_str(0)
        if self.length > 1:
            memmove(self.data_c, self.data_c + 256, (self.length - 1) * 256)
        self.length -= 1
        return val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def concat(self, Char256Column_SoA other):
        cdef Py_ssize_t other_len = other.length
        if other_len == 0: return
        cdef Py_ssize_t new_len = self.length + other_len
        self._reserve(new_len)
        memcpy(self.data_c + (self.length * 256), other.data_c, other_len * 256)
        self.length = new_len

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _reorder(self, list indices):
        cdef Py_ssize_t n = len(indices)
        if n != self.length:
            raise ValueError("Length of indices must match column length")
        if n == 0: return

        cdef char* new_data = <char*>aligned_malloc(n * 256)
        if not new_data:
            raise MemoryError()
        cdef char* self_data = self.data_c
        cdef Py_ssize_t i, idx
        try:
            for i in range(n):
                idx = <Py_ssize_t>indices[i]
                if idx < 0: idx += self.length
                if idx < 0 or idx >= self.length:
                    raise IndexError("Index out of bounds")
                memcpy(new_data + (i * 256), self_data + (idx * 256), 256)
        except Exception:
            os_aligned_free(new_data)
            raise

        os_aligned_free(self.data_c)
        self.data_c = new_data
        self.capacity = n

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
        cdef int count = 0
        cdef Py_ssize_t i

        with nogil:
            for i in range(n):
                if _c_str_contains(data + (i * 256), c_sub):
                    out_idx[count] = <int>i
                    count += 1
        cdef list res = [out_idx[j] for j in range(count)]
        os_aligned_free(out_idx)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def replace(self, str old_str, str new_str):
        cdef bytes b_old = old_str.encode('ascii')
        cdef bytes b_new = new_str.encode('ascii')
        cdef const char* c_old = b_old
        cdef const char* c_new = b_new
        cdef size_t old_len = len(b_old)
        cdef size_t new_len = len(b_new)
        cdef char* data = self.data_c
        cdef Py_ssize_t i, n = self.length

        with nogil:
            for i in range(n):
                _c_str_replace_inplace(data + (i * 256), c_old, c_new, old_len, new_len)
        return self

    @cython.binding(False)
    def replaces(self, *args):
        if len(args) % 2 != 0:
            raise ValueError("Arguments must be in pairs")
        cdef int i
        for i in range(0, len(args), 2):
            self.replace(str(args[i]), str(args[i+1]))
        return self

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def extract(self, str pattern, int group=1):
        cdef object compiled = re.compile(pattern)
        cdef Char256Column_SoA res = Char256Column_SoA(max(self.length, 1))
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
        cdef Char256Column_SoA res1 = Char256Column_SoA(max(self.length, 1))
        cdef Char256Column_SoA res2 = Char256Column_SoA(max(self.length, 1))
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
        cdef char* data = self.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                out_ptr[i] = <int>strlen(data + (i * 256))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def len_list(self):
        cdef char* data = self.data_c
        cdef Py_ssize_t i, n = self.length
        cdef list res = []
        for i in range(n):
            res.append(strlen(data + (i * 256)))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def startswith(self, prefix):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef char* data = self.data_c
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
                    out_ptr[i] = 1 if _c_startswith(data + (i * 256), c_pat, plen) else 0
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
                    out_ptr[i] = 1 if _c_startswith(data + (i * 256), c_pat, plen) else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def endswith(self, suffix):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef char* data = self.data_c
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
                    out_ptr[i] = 1 if _c_endswith(data + (i * 256), c_pat, slen) else 0
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
                    out_ptr[i] = 1 if _c_endswith(data + (i * 256), c_pat, slen) else 0
        return res

    # ------------------------------------------------------------
    # lower / upper: 纯 C 无分支 SIMD 自动向量化
    # ------------------------------------------------------------
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def lower(self):
        cdef Char256Column_SoA res = self.copy()
        cdef Py_ssize_t i, j, n = self.length
        cdef char* ptr
        cdef unsigned char c
        with nogil:
            for i in range(n):
                ptr = res.data_c + (i * 256)
                for j in range(256):
                    c = <unsigned char>ptr[j]
                    if c == 0: break
                    if c >= 65 and c <= 90:
                        ptr[j] = <char>(c + 32)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def upper(self):
        cdef Char256Column_SoA res = self.copy()
        cdef Py_ssize_t i, j, n = self.length
        cdef char* ptr
        cdef unsigned char c
        with nogil:
            for i in range(n):
                ptr = res.data_c + (i * 256)
                for j in range(256):
                    c = <unsigned char>ptr[j]
                    if c == 0: break
                    if c >= 97 and c <= 122:
                        ptr[j] = <char>(c - 32)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isdigit(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef char* data = self.data_c
        cdef Py_ssize_t i, j, n = self.length
        cdef char* ptr
        cdef bint valid
        with nogil:
            for i in range(n):
                ptr = data + (i * 256)
                if ptr[0] == 0:
                    out_ptr[i] = 0
                    continue
                valid = 1
                for j in range(256):
                    if ptr[j] == 0: break
                    if not isdigit(<unsigned char>ptr[j]):
                        valid = 0
                        break
                out_ptr[i] = valid
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def strip(self):
        cdef Char256Column_SoA res = self.copy()
        cdef Py_ssize_t i, left, right, n = self.length
        cdef char* ptr
        cdef size_t cur_len
        with nogil:
            for i in range(n):
                ptr = res.data_c + (i * 256)
                if ptr[0] == 0: continue
                cur_len = strlen(ptr)
                left = 0
                while left < <Py_ssize_t>cur_len and isspace(<unsigned char>ptr[left]):
                    left += 1
                if left == <Py_ssize_t>cur_len:
                    ptr[0] = 0
                    continue
                right = cur_len - 1
                while right >= left and isspace(<unsigned char>ptr[right]):
                    right -= 1
                if left > 0:
                    memmove(ptr, ptr + left, right - left + 1)
                ptr[right - left + 1] = 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def lstrip(self):
        cdef Char256Column_SoA res = self.copy()
        cdef Py_ssize_t i, left, n = self.length
        cdef char* ptr
        cdef size_t cur_len
        with nogil:
            for i in range(n):
                ptr = res.data_c + (i * 256)
                if ptr[0] == 0: continue
                cur_len = strlen(ptr)
                left = 0
                while left < <Py_ssize_t>cur_len and isspace(<unsigned char>ptr[left]):
                    left += 1
                if left > 0:
                    memmove(ptr, ptr + left, cur_len - left + 1)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def rstrip(self):
        cdef Char256Column_SoA res = self.copy()
        cdef Py_ssize_t i, right, n = self.length
        cdef char* ptr
        with nogil:
            for i in range(n):
                ptr = res.data_c + (i * 256)
                if ptr[0] == 0: continue
                right = <Py_ssize_t>strlen(ptr) - 1
                while right >= 0 and isspace(<unsigned char>ptr[right]):
                    right -= 1
                ptr[right + 1] = 0
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
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                out_ptr[i] = <int>_c_find(data + (i * 256), c_sub, sub_len)
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
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                out_ptr[i] = <int>_c_count(data + (i * 256), c_sub, sub_len)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isalpha(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef char* data = self.data_c
        cdef Py_ssize_t i, j, n = self.length
        cdef char* ptr
        cdef bint valid
        with nogil:
            for i in range(n):
                ptr = data + (i * 256)
                if ptr[0] == 0:
                    out_ptr[i] = 0
                    continue
                valid = 1
                for j in range(256):
                    if ptr[j] == 0: break
                    if not isalpha(<unsigned char>ptr[j]):
                        valid = 0
                        break
                out_ptr[i] = valid
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isalnum(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef char* data = self.data_c
        cdef Py_ssize_t i, j, n = self.length
        cdef char* ptr
        cdef bint valid
        with nogil:
            for i in range(n):
                ptr = data + (i * 256)
                if ptr[0] == 0:
                    out_ptr[i] = 0
                    continue
                valid = 1
                for j in range(256):
                    if ptr[j] == 0: break
                    if not isalnum(<unsigned char>ptr[j]):
                        valid = 0
                        break
                out_ptr[i] = valid
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isspace(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef char* data = self.data_c
        cdef Py_ssize_t i, j, n = self.length
        cdef char* ptr
        cdef bint valid
        with nogil:
            for i in range(n):
                ptr = data + (i * 256)
                if ptr[0] == 0:
                    out_ptr[i] = 0
                    continue
                valid = 1
                for j in range(256):
                    if ptr[j] == 0: break
                    if not isspace(<unsigned char>ptr[j]):
                        valid = 0
                        break
                out_ptr[i] = valid
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def islower(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef char* data = self.data_c
        cdef Py_ssize_t i, j, n = self.length
        cdef char* ptr
        cdef bint has_letter, all_lower
        with nogil:
            for i in range(n):
                ptr = data + (i * 256)
                if ptr[0] == 0:
                    out_ptr[i] = 0
                    continue
                has_letter = 0
                all_lower = 1
                for j in range(256):
                    if ptr[j] == 0: break
                    if isalpha(<unsigned char>ptr[j]):
                        has_letter = 1
                        if not islower(<unsigned char>ptr[j]):
                            all_lower = 0
                            break
                out_ptr[i] = 1 if (has_letter and all_lower) else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isupper(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef char* data = self.data_c
        cdef Py_ssize_t i, j, n = self.length
        cdef char* ptr
        cdef bint has_letter, all_upper
        with nogil:
            for i in range(n):
                ptr = data + (i * 256)
                if ptr[0] == 0:
                    out_ptr[i] = 0
                    continue
                has_letter = 0
                all_upper = 1
                for j in range(256):
                    if ptr[j] == 0: break
                    if isalpha(<unsigned char>ptr[j]):
                        has_letter = 1
                        if not isupper(<unsigned char>ptr[j]):
                            all_upper = 0
                            break
                out_ptr[i] = 1 if (has_letter and all_upper) else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def removeprefix(self, str prefix):
        cdef bytes b_pre = prefix.encode('ascii')
        cdef const char* c_pre = b_pre
        cdef Py_ssize_t pre_len = len(b_pre)
        cdef Char256Column_SoA res = self.copy()
        cdef Py_ssize_t i, n = self.length
        cdef char* ptr
        cdef size_t cur_len
        with nogil:
            for i in range(n):
                ptr = res.data_c + (i * 256)
                cur_len = strlen(ptr)
                if cur_len >= <size_t>pre_len and strncmp(ptr, c_pre, pre_len) == 0:
                    memmove(ptr, ptr + pre_len, cur_len - pre_len + 1)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def removesuffix(self, str suffix):
        cdef bytes b_suf = suffix.encode('ascii')
        cdef const char* c_suf = b_suf
        cdef Py_ssize_t suf_len = len(b_suf)
        cdef Char256Column_SoA res = self.copy()
        cdef Py_ssize_t i, n = self.length
        cdef char* ptr
        cdef size_t cur_len
        with nogil:
            for i in range(n):
                ptr = res.data_c + (i * 256)
                cur_len = strlen(ptr)
                if cur_len >= <size_t>suf_len and strncmp(ptr + cur_len - suf_len, c_suf, suf_len) == 0:
                    ptr[cur_len - suf_len] = 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def zfill(self, Py_ssize_t width):
        cdef Char256Column_SoA res = self.copy()
        cdef Py_ssize_t i, j, curr_len, pad, n = self.length
        cdef char* ptr
        with nogil:
            for i in range(n):
                ptr = res.data_c + (i * 256)
                curr_len = <Py_ssize_t>strlen(ptr)
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
    @cython.always_allow_keywords(False)
    def rfind(self, str substring):
        cdef bytes b_sub = substring.encode('ascii')
        cdef const char* c_sub = b_sub
        cdef size_t sub_len = len(b_sub)
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef char* data = self.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                out_ptr[i] = <int>_c_rfind(data + (i * 256), c_sub, sub_len)
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
        cdef Char256Column_SoA res = Char256Column_SoA(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            res.append(self._get_str(i).title())
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def capitalize(self):
        cdef Char256Column_SoA res = Char256Column_SoA(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            res.append(self._get_str(i).capitalize())
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def swapcase(self):
        cdef Char256Column_SoA res = Char256Column_SoA(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            res.append(self._get_str(i).swapcase())
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def casefold(self):
        cdef Char256Column_SoA res = Char256Column_SoA(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            res.append(self._get_str(i).casefold())
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def center(self, int width, str fillchar=' '):
        cdef Char256Column_SoA res = Char256Column_SoA(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            res.append(self._get_str(i).center(width, fillchar))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def ljust(self, int width, str fillchar=' '):
        cdef Char256Column_SoA res = Char256Column_SoA(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            res.append(self._get_str(i).ljust(width, fillchar))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def rjust(self, int width, str fillchar=' '):
        cdef Char256Column_SoA res = Char256Column_SoA(max(self.length, 1))
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
        cdef Char256Column_SoA res = Char256Column_SoA(max(self.length, 1))
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
        cdef Char256Column_SoA res = Char256Column_SoA(max(self.length, 1))
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
        cdef Char256Column_SoA res = Char256Column_SoA(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        cdef Char256Column_SoA other_cc
        if isinstance(other, Char256Column_SoA):
            other_cc = <Char256Column_SoA>other
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
        cdef Char256Column_SoA res = Char256Column_SoA(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            res.append(pre + self._get_str(i))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def suffix(self, str suf):
        cdef Char256Column_SoA res = Char256Column_SoA(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            res.append(self._get_str(i) + suf)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def repeat(self, int repeats):
        cdef Char256Column_SoA res = Char256Column_SoA(max(self.length, 1))
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

    # ------------------------------------------------------------
    # 字符串字典序比较：修复 strncmp 比较到脏内存的漏洞，使用 strcmp
    # ------------------------------------------------------------
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def _cmp_op(self, val, int op):
        cdef Py_ssize_t i, n = self.length
        cdef IntColumn res = IntColumn(max(n, 1))
        res.length = n
        cdef int* out_ptr = res.data_c
        cdef char* data = self.data_c
        cdef bytes v_bytes
        cdef const char* v_ptr
        cdef char* o_data
        cdef int cmp_res
        cdef Char256Column_SoA other

        if isinstance(val, Char256Column_SoA):
            other = <Char256Column_SoA>val
            if other.length != n:
                raise ValueError(f"Length mismatch: {n} vs {other.length}")
            o_data = other.data_c
            with nogil:
                for i in range(n):
                    cmp_res = strcmp(data + (i * 256), o_data + (i * 256))
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
                    cmp_res = strcmp(data + (i * 256), v_ptr)
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
        cdef char* data = self.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                out_ptr[i] = 1 if (data[i * 256] != 0 and other_ptr[i] != 0) else 0
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
        cdef char* data = self.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                out_ptr[i] = 1 if (data[i * 256] != 0 or other_ptr[i] != 0) else 0
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __invert__(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef char* data = self.data_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                out_ptr[i] = 0 if data[i * 256] != 0 else 1
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
        cdef Char256Column_SoA col = Char256Column_SoA(max(n, 1))
        col.length = n
        if n == 0: return col

        cdef int* mask_ptr = mask.data_c
        cdef char* src = self.data_c
        cdef char* dest = col.data_c

        cdef bytes fill_bytes = str(fill).encode('ascii', errors='ignore') if fill is not None else b''
        cdef char fill_buf[256]
        memset(fill_buf, 0, 256)
        cdef size_t flen = len(fill_bytes)
        if flen > 255: flen = 255
        memcpy(fill_buf, <const char*>fill_bytes, flen)

        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                if mask_ptr[i] != 0:
                    memcpy(dest + (i * 256), src + (i * 256), 256)
                else:
                    memcpy(dest + (i * 256), fill_buf, 256)
        return col

    @property
    def str(self):
        return StringAccessor_Char256(self)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __str__(self):
        cdef list items = []
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            items.append(repr(self._get_str(i)))
        return f"Char256Column([{', '.join(items)}])"

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __bool__(self):
        return self.length > 0

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __len__(self):
        return self.length

    cdef Char256Column_SoA take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=False):
        cdef Py_ssize_t i
        cdef char* old_data = self.data_c
        cdef char* new_data
        cdef Char256Column_SoA new_col

        if not inplace:
            new_col = Char256Column_SoA(max(n, 1))
            new_col.length = n
            with nogil:
                for i in range(n):
                    memcpy(new_col.data_c + (i * 256), old_data + (indices[i] * 256), 256)
            return new_col
        else:
            new_data = <char*>aligned_malloc(max(n, 1) * 256)
            if not new_data: raise MemoryError()
            with nogil:
                for i in range(n):
                    memcpy(new_data + (i * 256), old_data + (indices[i] * 256), 256)
            os_aligned_free(self.data_c)
            self.data_c = new_data
            self.capacity = n
            self.length = n
            return self

    cpdef void drop_row(self, Py_ssize_t row_idx, Py_ssize_t rem_len):
        if row_idx < 0:
            row_idx += self.length
        if rem_len > 0:
            memmove(self.data_c + (row_idx * 256),
                    self.data_c + ((row_idx + 1) * 256),
                    rem_len * 256)
        self.length -= 1

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
        cdef UCS2Column_SoA uc
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
            uc = UCS2Column_SoA(safe_cap)
            uc.length = n
            for i in range(n):
                uc._set_item(i, self._get_str(i))
            return uc
        elif target_type == TYPE_TIMESTAMP:
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
        elif target_type == TYPE_CHAR256:
            return self.copy()
        raise TypeError(f"Failed to cast CHAR256 column to internal type id: {target_type}")

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

    def str_getitem(self, key):
        cdef Char256Column_SoA col = Char256Column_SoA(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length, idx, s_len, idx_abs
        cdef str s
        if isinstance(key, int):
            idx = <Py_ssize_t>key
            for i in range(n):
                s = self._get_str(i)
                s_len = len(s)
                idx_abs = s_len + idx if idx < 0 else idx
                if 0 <= idx_abs < s_len:
                    col.append(s[idx_abs])
                else:
                    col.append('')
            return col
        elif isinstance(key, slice):
            for i in range(n):
                s = self._get_str(i)
                col.append(s[key])
            return col
        raise TypeError(f"Invalid index type: {type(key)}")

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