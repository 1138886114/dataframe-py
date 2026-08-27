# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport cython
from libc.string cimport memcpy, memmove, strstr, strlen, strncmp, strncpy, strncat
from libc.stdlib cimport free

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

from fluidframe.core._base cimport BaseColumn, aligned_malloc, aligned_realloc
from fluidframe.core.intcolumn cimport IntColumn

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

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __cinit__(self, Py_ssize_t init_capacity=16):
        self.capacity = max(init_capacity, 1)
        self.length = 0
        self.data_c = <char*>aligned_malloc(self.capacity * 256)
        if not self.data_c: raise MemoryError()

    def __dealloc__(self):
        if self.data_c is not NULL: free(self.data_c)

    cdef void _reserve(self, Py_ssize_t new_cap):
        if new_cap > self.capacity:
            self.data_c = <char*>aligned_realloc(self.data_c, self.capacity * 256, new_cap * 256)
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
        cdef char* new_data = <char*>aligned_malloc(self.capacity * 256)
        if not new_data: raise MemoryError()
        cdef Py_ssize_t i
        for i in range(self.length):
            memcpy(new_data + (i * 256), self.data_c + (<Py_ssize_t>indices[i] * 256), 256)
        free(self.data_c)
        self.data_c = new_data

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __getitem__(self, Py_ssize_t idx):
        if idx < 0: idx += self.length
        if idx < 0 or idx >= self.length: raise IndexError()
        cdef char* ptr = self.data_c + (idx * 256)
        return ptr.decode('ascii')

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def contains(self, str substring):
        cdef bytes b_sub = substring.encode('ascii')
        cdef const char* c_sub = b_sub
        cdef int* out_idx = <int*>aligned_malloc(self.length * sizeof(int))
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

    @cython.binding(False)
    def replaces(self, *args):
        if len(args) % 2 != 0: raise ValueError("Arguments must be in pairs")
        cdef int i
        for i in range(0, len(args), 2):
            self.replace(args[i], args[i+1])
        return self

    @cython.binding(False)
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
    def len(self):
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
        cdef IntColumn res = IntColumn(self.capacity)
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
                right = strlen(ptr) - 1
                while right >= left and isspace(ptr[right]):
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
                right = strlen(ptr) - 1
                while right >= 0 and isspace(ptr[right]):
                    right -= 1
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
        cdef size_t sub_len = len(b_sub)
        if sub_len == 0:
            return self.len().add(1)
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
                    match = strstr(match + sub_len, c_sub)
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

    def _cmp_op(self, val, int op):
        cdef Py_ssize_t i
        cdef IntColumn res = IntColumn(self.capacity)
        cdef bytes v_bytes
        cdef char* v_ptr
        res.length = self.length
        if isinstance(val, Char256Column_SoA):
            other = <Char256Column_SoA>val
            if other.length != self.length:
                raise ValueError(f"Length mismatch: {self.length} vs {other.length}")
            if op == 0:
                for i in range(self.length): res.data_c[i] = 1 if strncmp(self.data_c + i * 256, other.data_c + i * 256, 256) < 0 else 0
            elif op == 1:
                for i in range(self.length): res.data_c[i] = 1 if strncmp(self.data_c + i * 256, other.data_c + i * 256, 256) <= 0 else 0
            elif op == 2:
                for i in range(self.length): res.data_c[i] = 1 if strncmp(self.data_c + i * 256, other.data_c + i * 256, 256) > 0 else 0
            elif op == 3:
                for i in range(self.length): res.data_c[i] = 1 if strncmp(self.data_c + i * 256, other.data_c + i * 256, 256) >= 0 else 0
            elif op == 4:
                for i in range(self.length): res.data_c[i] = 1 if strncmp(self.data_c + i * 256, other.data_c + i * 256, 256) == 0 else 0
            elif op == 5:
                for i in range(self.length): res.data_c[i] = 1 if strncmp(self.data_c + i * 256, other.data_c + i * 256, 256) != 0 else 0
        else:
            v_bytes = val.encode('ascii') if isinstance(val, str) else val
            v_ptr = v_bytes
            if op == 0:
                for i in range(self.length): res.data_c[i] = 1 if strncmp(self.data_c + i * 256, v_ptr, 256) < 0 else 0
            elif op == 1:
                for i in range(self.length): res.data_c[i] = 1 if strncmp(self.data_c + i * 256, v_ptr, 256) <= 0 else 0
            elif op == 2:
                for i in range(self.length): res.data_c[i] = 1 if strncmp(self.data_c + i * 256, v_ptr, 256) > 0 else 0
            elif op == 3:
                for i in range(self.length): res.data_c[i] = 1 if strncmp(self.data_c + i * 256, v_ptr, 256) >= 0 else 0
            elif op == 4:
                for i in range(self.length): res.data_c[i] = 1 if strncmp(self.data_c + i * 256, v_ptr, 256) == 0 else 0
            elif op == 5:
                for i in range(self.length): res.data_c[i] = 1 if strncmp(self.data_c + i * 256, v_ptr, 256) != 0 else 0
        return res

    def __lt__(self, val):  return self._cmp_op(val, 0)
    def __le__(self, val):  return self._cmp_op(val, 1)
    def __gt__(self, val):  return self._cmp_op(val, 2)
    def __ge__(self, val):  return self._cmp_op(val, 3)
    def __eq__(self, val):  return self._cmp_op(val, 4)
    def __ne__(self, val):  return self._cmp_op(val, 5)

    def __and__(self, other):
        if not isinstance(other, IntColumn):
            raise TypeError("Cannot & non-IntColumn mask")
        if self.length != other.length:
            raise ValueError(f"Length mismatch: {self.length} vs {other.length}")
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        for i in range(self.length):
            res.data_c[i] = 1 if (self.data_c + i * 256)[0] != 0 and other.data_c[i] else 0
        return res

    def __or__(self, other):
        if not isinstance(other, IntColumn):
            raise TypeError("Cannot | non-IntColumn mask")
        if self.length != other.length:
            raise ValueError(f"Length mismatch: {self.length} vs {other.length}")
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        for i in range(self.length):
            res.data_c[i] = 1 if (self.data_c + i * 256)[0] != 0 or other.data_c[i] else 0
        return res

    def __invert__(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        for i in range(self.length):
            res.data_c[i] = 0 if (self.data_c + i * 256)[0] != 0 else 1
        return res

    def __str__(self):
        cdef list items = []
        cdef Py_ssize_t i
        for i in range(self.length):
            items.append(repr((self.data_c + i * 256).decode('ascii')))
        return f"Char256Column([{', '.join(items)}])"

    def __bool__(self):
        return self.length > 0

    def __len__(self):
        return self.length
