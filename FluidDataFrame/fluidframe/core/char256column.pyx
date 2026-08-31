# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport cython
from libc.string cimport memcpy, memmove, strstr, strlen, strncmp, strncpy, strncat

cdef extern from "<ctype.h>" nogil:
    int tolower(int)
    int toupper(int)
    int isdigit(int)
    int isspace(int)
    int isalpha(int)
    int isalnum(int)
    int islower(int)
    int isupper(int)

import re

from fluidframe.core._base cimport BaseColumn, aligned_malloc, aligned_realloc, os_aligned_free, TYPE_CHAR256, TYPE_INT, TYPE_DOUBLE, TYPE_UCS2, TYPE_BOOL, TYPE_TIMESTAMP, TYPE_CATEGORY, TYPE_DATETIME, TYPE_TIME, _resolve_astype_target
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.doublecolumn cimport DoubleColumn
from fluidframe.core.boolcolumn cimport BoolColumn
from fluidframe.core.ucs2column cimport UCS2Column_SoA
from fluidframe.core.datecolumn cimport DateColumn
from fluidframe.core.categorycolumn cimport CategoryColumn
from fluidframe.core.datetimecolumn cimport DateTimeColumn
from fluidframe.core.timecolumn cimport TimeColumn

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
        if not self.data_c: raise MemoryError()

    def __dealloc__(self):
        if self.data_c is not NULL: os_aligned_free(self.data_c)

    cdef void _reserve(self, Py_ssize_t new_cap):
        if new_cap > self.capacity:
            self.data_c = <char*>aligned_realloc(self.data_c, self.capacity * 256, new_cap * 256)
            if not self.data_c: raise MemoryError()
            self.capacity = new_cap

    cdef inline str _get_str(self, Py_ssize_t idx):
        cdef char* ptr = self.data_c + (idx * 256)
        if ptr[0] == b'\0':
            return ""
        return ptr.decode('ascii')

    cdef void _set_item(self, Py_ssize_t idx, str val) except *:
        """Replace string at given index in-place on the fixed 256-byte row."""
        cdef bytes b_val = val.encode('ascii', errors='ignore')
        cdef char* src = b_val
        cdef char* dest = self.data_c + (idx * 256)
        strncpy(dest, src, 255)
        dest[255] = b'\0'

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
        cdef Py_ssize_t n = len(indices)
        if n != self.length:
            raise ValueError("Length of indices must match column length")

        cdef char* new_data = <char*>aligned_malloc(n * 256)
        if not new_data: raise MemoryError()
        cdef Py_ssize_t i, idx
        try:
            for i in range(n):
                idx = <Py_ssize_t>indices[i]
                if idx < 0 or idx >= n:
                    raise IndexError("Index out of bounds")
                memcpy(new_data + (i * 256), self.data_c + (idx * 256), 256)
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
        return self._get_str(idx)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def contains(self, str substring):
        cdef bytes b_sub = substring.encode('ascii')
        cdef const char* c_sub = b_sub
        cdef int* out_idx = <int*>aligned_malloc(max(self.length, 1) * sizeof(int))
        if not out_idx: raise MemoryError()
        cdef char* data = self.data_c
        cdef int count = 0
        cdef Py_ssize_t i

        with nogil:
            for i in range(self.length):
                if _c_str_contains(data + (i * 256), c_sub):
                    out_idx[count] = i
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
        cdef char* data = self.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                _c_str_replace_inplace(data + (i * 256), c_old, c_new)
        return self

    @cython.binding(False)
    def replaces(self, *args):
        if len(args) % 2 != 0: raise ValueError("Arguments must be in pairs")
        cdef int i
        for i in range(0, len(args), 2):
            self.replace(args[i], args[i+1])
        return self

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def extract(self, str pattern):
        prog = re.compile(pattern)
        cdef Char256Column_SoA res = Char256Column_SoA(self.capacity)
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = (self.data_c + i * 256).decode('ascii', errors='ignore')
            match = prog.search(s)
            if match and match.groups():
                res.append(match.group(1) or "")
            else:
                res.append("")
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def extract_twice(self, str pattern):
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
    @cython.always_allow_keywords(False)
    def len(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef char* data = self.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                out_ptr[i] = <int>strlen(data + i * 256)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def len_list(self):
        cdef char* data = self.data_c
        return [strlen(data + i * 256) for i in range(self.length)]

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def startswith(self, str prefix):
        cdef bytes b_pre = prefix.encode('ascii')
        cdef const char* c_pre = b_pre
        cdef Py_ssize_t pre_len = strlen(c_pre)
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef char* data = self.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                out_ptr[i] = 1 if strncmp(data + i * 256, c_pre, pre_len) == 0 else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def endswith(self, str suffix):
        cdef bytes b_suf = suffix.encode('ascii')
        cdef const char* c_suf = b_suf
        cdef Py_ssize_t suf_len = strlen(c_suf)
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef char* data = self.data_c
        cdef Py_ssize_t i, curr_len
        with nogil:
            for i in range(self.length):
                curr_len = strlen(data + i * 256)
                if curr_len >= suf_len and strncmp(data + i * 256 + curr_len - suf_len, c_suf, suf_len) == 0:
                    out_ptr[i] = 1
                else:
                    out_ptr[i] = 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
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
    @cython.always_allow_keywords(False)
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
    @cython.always_allow_keywords(False)
    def isdigit(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef char* data = self.data_c
        cdef Py_ssize_t i, j
        cdef bint valid
        with nogil:
            for i in range(self.length):
                if data[i * 256] == b'\0':
                    out_ptr[i] = 0
                    continue
                valid = 1
                for j in range(256):
                    if data[i * 256 + j] == b'\0': break
                    if not isdigit(data[i * 256 + j]):
                        valid = 0
                        break
                out_ptr[i] = valid
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
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
                right = strlen(ptr) - 1
                while right >= left and isspace(ptr[right]):
                    right -= 1
                if left > 0:
                    memmove(ptr, ptr + left, right - left + 1)
                ptr[right - left + 1] = b'\0'
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
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
    @cython.always_allow_keywords(False)
    def rstrip(self):
        cdef Char256Column_SoA res = self.copy()
        cdef Py_ssize_t i, right
        cdef char* ptr
        with nogil:
            for i in range(self.length):
                ptr = res.data_c + i * 256
                right = strlen(ptr) - 1
                while right >= 0 and isspace(ptr[right]):
                    right -= 1
                ptr[right + 1] = b'\0'
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def find(self, str substring):
        cdef bytes b_sub = substring.encode('ascii')
        cdef const char* c_sub = b_sub
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef char* data = self.data_c
        cdef Py_ssize_t i
        cdef char* match
        with nogil:
            for i in range(self.length):
                match = strstr(data + i * 256, c_sub)
                if match != NULL:
                    out_ptr[i] = <int>(match - (data + i * 256))
                else:
                    out_ptr[i] = -1
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def count(self, str substring):
        cdef bytes b_sub = substring.encode('ascii')
        cdef size_t sub_len = len(b_sub)

        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef char* data = self.data_c
        cdef Py_ssize_t i

        if sub_len == 0:
            with nogil:
                for i in range(self.length):
                    out_ptr[i] = <int>strlen(data + i * 256) + 1
            return res

        cdef const char* c_sub = b_sub
        cdef char* ptr
        cdef char* match
        cdef int cnt
        with nogil:
            for i in range(self.length):
                ptr = data + i * 256
                cnt = 0
                match = strstr(ptr, c_sub)
                while match != NULL:
                    cnt += 1
                    match = strstr(match + sub_len, c_sub)
                out_ptr[i] = cnt
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isalpha(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef char* data = self.data_c
        cdef Py_ssize_t i, j
        cdef bint valid
        with nogil:
            for i in range(self.length):
                if data[i * 256] == b'\0':
                    out_ptr[i] = 0
                    continue
                valid = 1
                for j in range(256):
                    if data[i * 256 + j] == b'\0': break
                    if not isalpha(data[i * 256 + j]):
                        valid = 0
                        break
                out_ptr[i] = valid
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isalnum(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef char* data = self.data_c
        cdef Py_ssize_t i, j
        cdef bint valid
        with nogil:
            for i in range(self.length):
                if data[i * 256] == b'\0':
                    out_ptr[i] = 0
                    continue
                valid = 1
                for j in range(256):
                    if data[i * 256 + j] == b'\0': break
                    if not isalnum(data[i * 256 + j]):
                        valid = 0
                        break
                out_ptr[i] = valid
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isspace(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef char* data = self.data_c
        cdef Py_ssize_t i, j
        cdef bint valid
        with nogil:
            for i in range(self.length):
                if data[i * 256] == b'\0':
                    out_ptr[i] = 0
                    continue
                valid = 1
                for j in range(256):
                    if data[i * 256 + j] == b'\0': break
                    if not isspace(data[i * 256 + j]):
                        valid = 0
                        break
                out_ptr[i] = valid
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def islower(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef char* data = self.data_c
        cdef Py_ssize_t i, j
        cdef bint has_letter, all_lower
        with nogil:
            for i in range(self.length):
                if data[i * 256] == b'\0':
                    out_ptr[i] = 0
                    continue
                has_letter = 0
                all_lower = 1
                for j in range(256):
                    if data[i * 256 + j] == b'\0': break
                    if isalpha(data[i * 256 + j]):
                        has_letter = 1
                        if not islower(data[i * 256 + j]):
                            all_lower = 0
                            break
                out_ptr[i] = 1 if (has_letter and all_lower) else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isupper(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef char* data = self.data_c
        cdef Py_ssize_t i, j
        cdef bint has_letter, all_upper
        with nogil:
            for i in range(self.length):
                if data[i * 256] == b'\0':
                    out_ptr[i] = 0
                    continue
                has_letter = 0
                all_upper = 1
                for j in range(256):
                    if data[i * 256 + j] == b'\0': break
                    if isalpha(data[i * 256 + j]):
                        has_letter = 1
                        if not isupper(data[i * 256 + j]):
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
    @cython.always_allow_keywords(False)
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
    @cython.always_allow_keywords(False)
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
    @cython.always_allow_keywords(False)
    def unique(self):
        cdef set seen = set()
        cdef list result = []
        cdef Py_ssize_t i
        for i in range(self.length):
            val = self._get_str(i)
            if val not in seen:
                seen.add(val)
                result.append(val)
        return result

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def _cmp_op(self, val, int op):
        cdef Py_ssize_t i
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef char* data = self.data_c
        cdef bytes v_bytes
        cdef char* v_ptr
        cdef char* o_data
        cdef int cmp_res
        if isinstance(val, Char256Column_SoA):
            other = <Char256Column_SoA>val
            if other.length != self.length:
                raise ValueError(f"Length mismatch: {self.length} vs {other.length}")
            o_data = other.data_c
            with nogil:
                for i in range(self.length):
                    cmp_res = strncmp(data + i * 256, o_data + i * 256, 256)
                    if op == 0:   out_ptr[i] = 1 if cmp_res < 0 else 0
                    elif op == 1: out_ptr[i] = 1 if cmp_res <= 0 else 0
                    elif op == 2: out_ptr[i] = 1 if cmp_res > 0 else 0
                    elif op == 3: out_ptr[i] = 1 if cmp_res >= 0 else 0
                    elif op == 4: out_ptr[i] = 1 if cmp_res == 0 else 0
                    elif op == 5: out_ptr[i] = 1 if cmp_res != 0 else 0
        else:
            v_bytes = val.encode('ascii') if isinstance(val, str) else val
            v_ptr = v_bytes
            with nogil:
                for i in range(self.length):
                    cmp_res = strncmp(data + i * 256, v_ptr, 256)
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
        if self.length != other.length:
            raise ValueError(f"Length mismatch: {self.length} vs {other.length}")
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef int* other_ptr = (<IntColumn>other).data_c
        cdef char* data = self.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                out_ptr[i] = 1 if data[i * 256] != 0 and other_ptr[i] else 0
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __or__(self, other):
        if not isinstance(other, IntColumn):
            raise TypeError("Cannot | non-IntColumn mask")
        if self.length != other.length:
            raise ValueError(f"Length mismatch: {self.length} vs {other.length}")
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef int* other_ptr = (<IntColumn>other).data_c
        cdef char* data = self.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                out_ptr[i] = 1 if data[i * 256] != 0 or other_ptr[i] else 0
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __invert__(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef char* data = self.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                out_ptr[i] = 0 if data[i * 256] != 0 else 1
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def isin(self, set values):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i
        for i in range(self.length):
            out_ptr[i] = 1 if self._get_str(i) in values else 0
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def where_mask(self, IntColumn mask, object fill):
        cdef Py_ssize_t n = self.length
        if mask.length != n:
            raise ValueError(f"Mask length {mask.length} != column length {n}")
        cdef Char256Column_SoA col = Char256Column_SoA(n)
        col.length = n
        cdef int* mask_ptr = mask.data_c
        cdef char* src = self.data_c
        cdef char* dest = col.data_c
        cdef bytes fill_bytes = str(fill).encode('ascii', errors='ignore')
        cdef const char* c_fill = fill_bytes
        cdef Py_ssize_t i
        for i in range(n):
            if mask_ptr[i]:
                strncpy(dest + (i * 256), src + (i * 256), 255)
            else:
                strncpy(dest + (i * 256), c_fill, 255)
            dest[i * 256 + 255] = 0
        return col

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __str__(self):
        cdef list items = []
        cdef Py_ssize_t i
        for i in range(self.length):
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
        if not inplace:
            new_col = Char256Column_SoA(max(n, 1))
            new_col.length = n
            with nogil:
                for i in range(n):
                    memcpy(new_col.data_c + (i * 256), self.data_c + (indices[i] * 256), 256)
            return new_col
        else:
            new_data = <char*>aligned_malloc(max(n, self.capacity) * 256)
            if not new_data: raise MemoryError()
            with nogil:
                for i in range(n):
                    memcpy(new_data + (i * 256), self.data_c + (indices[i] * 256), 256)
            os_aligned_free(self.data_c)
            self.data_c = new_data
            self.capacity = max(n, self.capacity)
            self.length = n
            return self

    cpdef void drop_row(self, Py_ssize_t row_idx, Py_ssize_t rem_len):
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
                s = self[i]
                ic.data_c[i] = int(float(s)) if s else 0
            return ic
        elif target_type == TYPE_DOUBLE:
            dc = DoubleColumn(safe_cap)
            dc.length = n
            for i in range(n):
                s = self[i]
                dc.data_c[i] = float(s) if s else 0.0
            return dc
        elif target_type == TYPE_BOOL:
            bc = BoolColumn(safe_cap)
            bc.length = n
            for i in range(n):
                s = self[i].lower() if self[i] else ''
                bc.data_c[i] = 1 if s in ('true', '1', 'yes') else 0
            return bc
        elif target_type == TYPE_UCS2:
            uc = UCS2Column_SoA(safe_cap)
            uc.length = n
            for i in range(n):
                uc._set_item(i, self[i])
            return uc
        elif target_type == TYPE_TIMESTAMP:
            tc = DateColumn(safe_cap)
            for i in range(n):
                s = self[i]
                if s:
                    tc.append(s)
                else:
                    tc.append(0)
            return tc
        elif target_type == TYPE_CATEGORY:
            catc = CategoryColumn(safe_cap)
            for i in range(n):
                catc.append(self[i])
            return catc
        elif target_type == TYPE_DATETIME:
            dtc = DateTimeColumn(safe_cap)
            for i in range(n):
                s = self[i]
                if s:
                    dtc.append(s)
                else:
                    dtc.append('')
            return dtc
        elif target_type == TYPE_TIME:
            tmc = TimeColumn(safe_cap)
            for i in range(n):
                s = self[i]
                if s:
                    tmc.append(s)
                else:
                    tmc.append(0)
            return tmc
        elif target_type == TYPE_CHAR256:
            return self.copy()
        raise TypeError(f"Failed to cast CHAR256 column to internal type id: {target_type}")