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
    TYPE_DATE, TYPE_CATEGORY, TYPE_DATETIME, TYPE_TIME, _resolve_astype_target,
)
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.doublecolumn cimport DoubleColumn
from fluidframe.core.boolcolumn cimport BoolColumn
from fluidframe.core.ucs2column cimport UCS2Column
from fluidframe.core.datecolumn cimport DateColumn
from fluidframe.core.categorycolumn cimport CategoryColumn
from fluidframe.core.datetimecolumn cimport DateTimeColumn
from fluidframe.core.timecolumn cimport TimeColumn



cdef inline bint _is_lower(unsigned char c) noexcept nogil:
    return c >= 97 and c <= 122

cdef inline bint _is_upper(unsigned char c) noexcept nogil:
    return c >= 65 and c <= 90

cdef inline bint _is_alpha(unsigned char c) noexcept nogil:
    return _is_lower(c) or _is_upper(c)

cdef inline bint _is_digit(unsigned char c) noexcept nogil:
    return c >= 48 and c <= 57

cdef inline bint _is_alnum(unsigned char c) noexcept nogil:
    return _is_alpha(c) or _is_digit(c)

cdef inline bint _is_space(unsigned char c) noexcept nogil:
    return c == 32 or (c >= 9 and c <= 13)

# ============================================================
#  C helper functions (全部保留，提供原始功能支撑)
# ============================================================

cdef inline bint _c_str_contains(const char* source, const char* target) noexcept nogil:
    return strstr(source, target) != NULL

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
#  AsciiColumn class definition
# ============================================================
@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef class AsciiColumn(BaseColumn):

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __cinit__(self, Py_ssize_t init_capacity=16):
        self.capacity = max(init_capacity, 1)
        self.length = 0
        self.dtype = TYPE_CHAR256
        self._str_cache = None

        self.offsets_c = <Py_ssize_t*>aligned_malloc((self.capacity + 1) * sizeof(Py_ssize_t))
        if not self.offsets_c:
            raise MemoryError()

        self.byte_capacity = max(self.capacity * 64, 4096)
        self.byte_length = 0
        self.data_c = <char*>aligned_malloc(self.byte_capacity)
        if not self.data_c:
            os_aligned_free(self.offsets_c)
            raise MemoryError()
        self.offsets_c[0] = 0

    def __dealloc__(self):
        if self.offsets_c != NULL:
            os_aligned_free(self.offsets_c)
            self.offsets_c = NULL
        if self.data_c != NULL:
            os_aligned_free(self.data_c)
            self.data_c = NULL

    def get_size(self):
        return (self.capacity + 1) * sizeof(Py_ssize_t) + self.byte_capacity

    cdef void _reserve_rows(self, Py_ssize_t new_cap) except *:
        if new_cap <= self.capacity:
            return
        cdef Py_ssize_t* new_offsets = <Py_ssize_t*>aligned_realloc(
            self.offsets_c,
            (self.capacity + 1) * sizeof(Py_ssize_t),
            (new_cap + 1) * sizeof(Py_ssize_t))
        if not new_offsets:
            raise MemoryError()
        self.offsets_c = new_offsets
        self.capacity = new_cap

    cdef void _reserve_bytes(self, Py_ssize_t new_byte_cap) except *:
        if new_byte_cap <= self.byte_capacity:
            return
        cdef char* new_data = <char*>aligned_realloc(
            self.data_c, self.byte_capacity, new_byte_cap)
        if not new_data:
            raise MemoryError()
        self.data_c = new_data
        self.byte_capacity = new_byte_cap

    cdef inline str _get_str(self, Py_ssize_t idx):
        cdef Py_ssize_t start = self.offsets_c[idx]
        cdef Py_ssize_t str_len = self.offsets_c[idx + 1] - start - 1
        if str_len <= 0:
            return ""
        return self.data_c[start:start+str_len].decode('ascii', errors='replace')

    cdef void _set_item(self, Py_ssize_t idx, str val) except *:
        cdef bytes b_val = val.encode('ascii', errors='ignore')
        cdef const char* src = b_val
        cdef Py_ssize_t new_len = len(b_val)
        cdef Py_ssize_t old_start = self.offsets_c[idx]
        cdef Py_ssize_t old_byte_span = self.offsets_c[idx + 1] - old_start
        cdef Py_ssize_t new_byte_span = new_len + 1
        cdef Py_ssize_t diff, oi
        cdef Py_ssize_t tail_len

        diff = new_byte_span - old_byte_span
        if diff > 0:
            if self.byte_length + diff > self.byte_capacity:
                self._reserve_bytes(max(self.byte_capacity * 2, self.byte_length + diff + 4096))

        tail_len = self.byte_length - old_start - old_byte_span
        if tail_len > 0:
            memmove(self.data_c + old_start + new_byte_span,
                    self.data_c + old_start + old_byte_span,
                    tail_len)

        if new_len > 0:
            memcpy(self.data_c + old_start, src, new_len)
        self.data_c[old_start + new_len] = 0

        self.byte_length += diff
        for oi in range(idx + 1, self.length + 1):
            self.offsets_c[oi] += diff

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def append(self, str val):
        cdef bytes b_val = val.encode('ascii', errors='ignore') if val is not None else b''
        cdef Py_ssize_t slen = len(b_val)
        cdef Py_ssize_t need_bytes = slen + 1

        if self.length >= self.capacity:
            self._reserve_rows(max(16, self.capacity * 2))
        if self.byte_length + need_bytes > self.byte_capacity:
            self._reserve_bytes(max(self.byte_capacity * 2, self.byte_length + need_bytes + 4096))

        if slen > 0:
            memcpy(self.data_c + self.byte_length, <const char*>b_val, slen)
        self.data_c[self.byte_length + slen] = 0
        self.byte_length += need_bytes
        self.length += 1
        self.offsets_c[self.length] = self.byte_length

    cdef void append_nolen(self, str val) except *:
        """写入当前行(length)位置的值，但不推进 length（由 add_row 统一提交）。"""
        if self.length >= self.capacity:
            raise BufferError("AsciiColumn row capacity (%d) reached." % self.capacity)
        cdef bytes b_val = val.encode('ascii', errors='ignore') if val is not None else b''
        cdef Py_ssize_t slen = len(b_val)
        cdef Py_ssize_t need_bytes = slen + 1
        if self.byte_length + need_bytes > self.byte_capacity:
            self._reserve_bytes(max(self.byte_capacity * 2, self.byte_length + need_bytes + 4096))
        if slen > 0:
            memcpy(self.data_c + self.byte_length, <const char*>b_val, slen)
        self.data_c[self.byte_length + slen] = 0
        self.byte_length += need_bytes
        self.offsets_c[self.length + 1] = self.byte_length

    cdef void undo_nolen(self) noexcept:
        """回滚一次未提交(未推进 length)的写入，保证重试时不会产生残留字节。"""
        self.byte_length = self.offsets_c[self.length]
        self.offsets_c[self.length + 1] = self.byte_length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def copy(self):
        cdef Py_ssize_t cap = max(self.length, 1)
        cdef AsciiColumn new_col = AsciiColumn(cap)
        new_col._reserve_bytes(self.byte_capacity)
        new_col.length = self.length
        new_col.byte_length = self.byte_length
        memcpy(new_col.offsets_c, self.offsets_c, (self.length + 1) * sizeof(Py_ssize_t))
        if self.byte_length > 0:
            memcpy(new_col.data_c, self.data_c, self.byte_length)
        return new_col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_back(self):
        if self.length == 0:
            raise IndexError("pop from empty column")
        cdef str val = self._get_str(self.length - 1)
        self.byte_length = self.offsets_c[self.length - 1]
        self.length -= 1
        return val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_front(self):
        if self.length == 0:
            raise IndexError("pop from empty column")
        cdef str val = self._get_str(0)
        cdef Py_ssize_t str_len = self.offsets_c[1] - self.offsets_c[0]
        cdef Py_ssize_t rem_bytes = self.byte_length - self.offsets_c[1]
        if rem_bytes > 0:
            memmove(self.data_c, self.data_c + self.offsets_c[1], rem_bytes)
        self.byte_length -= str_len
        cdef Py_ssize_t i
        for i in range(self.length):
            self.offsets_c[i] = self.offsets_c[i + 1] - str_len
        self.length -= 1
        return val

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def concat(self, AsciiColumn other):
        cdef Py_ssize_t other_len = other.length
        if other_len == 0: return
        cdef Py_ssize_t new_len = self.length + other_len
        self._reserve_rows(new_len)
        self._reserve_bytes(self.byte_length + other.byte_length)

        cdef Py_ssize_t base_offset = self.byte_length
        cdef Py_ssize_t i
        cdef Py_ssize_t* self_offs = self.offsets_c
        cdef Py_ssize_t* other_offs = other.offsets_c

        if other.byte_length > 0:
            memcpy(self.data_c + self.byte_length, other.data_c, other.byte_length)

        for i in range(other_len):
            self_offs[self.length + 1 + i] = base_offset + other_offs[i + 1]

        self.length = new_len
        self.byte_length += other.byte_length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _reorder(self, list indices):
        cdef Py_ssize_t n = len(indices)
        if n != self.length:
            raise ValueError("Length of indices must match column length")
        if n == 0: return

        cdef Py_ssize_t req_bytes = 0
        cdef Py_ssize_t i, idx
        for i in range(n):
            idx = <Py_ssize_t>indices[i]
            if idx < 0: idx += self.length
            if idx < 0 or idx >= self.length:
                raise IndexError("Index out of bounds")
            req_bytes += self.offsets_c[idx + 1] - self.offsets_c[idx]

        cdef Py_ssize_t req_byte_cap = max(req_bytes, 4096)
        cdef Py_ssize_t* new_offsets = <Py_ssize_t*>aligned_malloc((n + 1) * sizeof(Py_ssize_t))
        if not new_offsets: raise MemoryError()
        cdef char* new_data = <char*>aligned_malloc(req_byte_cap)
        if not new_data:
            os_aligned_free(new_offsets)
            raise MemoryError()

        new_offsets[0] = 0
        cdef Py_ssize_t current_byte_len = 0
        cdef Py_ssize_t start, end, str_len

        try:
            for i in range(n):
                idx = <Py_ssize_t>indices[i]
                if idx < 0: idx += self.length
                start = self.offsets_c[idx]
                end = self.offsets_c[idx + 1]
                str_len = end - start
                if str_len > 0:
                    memcpy(new_data + current_byte_len, self.data_c + start, str_len)
                current_byte_len += str_len
                new_offsets[i + 1] = current_byte_len
        except Exception:
            os_aligned_free(new_offsets)
            os_aligned_free(new_data)
            raise

        os_aligned_free(self.offsets_c)
        os_aligned_free(self.data_c)
        self.offsets_c = new_offsets
        self.data_c = new_data
        self.byte_length = current_byte_len
        self.byte_capacity = req_byte_cap
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
        cdef Py_ssize_t* offs = self.offsets_c
        cdef int count = 0
        cdef Py_ssize_t i

        with nogil:
            for i in range(n):
                if _c_str_contains(data + offs[i], c_sub):
                    out_idx[count] = <int>i
                    count += 1
        cdef list res = [out_idx[j] for j in range(count)]
        os_aligned_free(out_idx)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def replace(self, str old_str, str new_str):
        cdef AsciiColumn res = AsciiColumn(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            res.append(self._get_str(i).replace(old_str, new_str))
        return res

    @cython.binding(False)
    def replaces(self, *args):
        if len(args) % 2 != 0:
            raise ValueError("Arguments must be in pairs")
        cdef int i
        cdef AsciiColumn res = self
        for i in range(0, len(args), 2):
            res = res.replace(str(args[i]), str(args[i+1]))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def extract(self, str pattern, int group=1):
        cdef object compiled = re.compile(pattern)
        cdef AsciiColumn res = AsciiColumn(max(self.length, 1))
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
        cdef AsciiColumn res1 = AsciiColumn(max(self.length, 1))
        cdef AsciiColumn res2 = AsciiColumn(max(self.length, 1))
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
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                out_ptr[i] = <int>(offs[i + 1] - offs[i] - 1)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def len_list(self):
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i, n = self.length
        cdef list res = []
        for i in range(n):
            res.append(offs[i + 1] - offs[i] - 1)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def startswith(self, prefix):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef char* data = self.data_c
        cdef Py_ssize_t* offs = self.offsets_c
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
                    out_ptr[i] = 1 if _c_startswith(data + offs[i], c_pat, plen) else 0
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
                    out_ptr[i] = 1 if _c_startswith(data + offs[i], c_pat, plen) else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def endswith(self, suffix):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef char* data = self.data_c
        cdef Py_ssize_t* offs = self.offsets_c
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
                    out_ptr[i] = 1 if _c_endswith(data + offs[i], c_pat, slen) else 0
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
                    out_ptr[i] = 1 if _c_endswith(data + offs[i], c_pat, slen) else 0
        return res

    # ------------------------------------------------------------
    # VECTORIZED: lower / upper using Branchless Bitwise Masks
    # ------------------------------------------------------------
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def lower(self):
        cdef AsciiColumn res = self.copy()
        cdef Py_ssize_t i, j, n = self.length
        cdef char* ptr
        cdef unsigned char c
        cdef unsigned char is_up
        cdef Py_ssize_t str_len
        cdef Py_ssize_t* offs = res.offsets_c
        with nogil:
            for i in range(n):
                ptr = res.data_c + offs[i]
                str_len = offs[i + 1] - offs[i] - 1
                for j in range(str_len):
                    c = <unsigned char>ptr[j]
                    is_up = (c >= 65) & (c <= 90)
                    ptr[j] = c | (is_up << 5)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def upper(self):
        cdef AsciiColumn res = self.copy()
        cdef Py_ssize_t i, j, n = self.length
        cdef char* ptr
        cdef unsigned char c
        cdef unsigned char is_low
        cdef Py_ssize_t str_len
        cdef Py_ssize_t* offs = res.offsets_c
        with nogil:
            for i in range(n):
                ptr = res.data_c + offs[i]
                str_len = offs[i + 1] - offs[i] - 1
                for j in range(str_len):
                    c = <unsigned char>ptr[j]
                    is_low = (c >= 97) & (c <= 122)
                    ptr[j] = c & ~(is_low << 5)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isdigit(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef char* data = self.data_c
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i, j, n = self.length
        cdef char* ptr
        cdef bint valid
        cdef Py_ssize_t str_len
        with nogil:
            for i in range(n):
                str_len = offs[i + 1] - offs[i] - 1
                if str_len == 0:
                    out_ptr[i] = 0
                    continue
                ptr = data + offs[i]
                valid = 1
                for j in range(str_len):
                    if not _is_digit(<unsigned char>ptr[j]):
                        valid = 0
                        break
                out_ptr[i] = valid
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isalpha(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef char* data = self.data_c
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i, j, n = self.length
        cdef char* ptr
        cdef bint valid
        cdef Py_ssize_t str_len
        with nogil:
            for i in range(n):
                str_len = offs[i + 1] - offs[i] - 1
                if str_len == 0:
                    out_ptr[i] = 0
                    continue
                ptr = data + offs[i]
                valid = 1
                for j in range(str_len):
                    if not _is_alpha(<unsigned char>ptr[j]):
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
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i, j, n = self.length
        cdef char* ptr
        cdef bint valid
        cdef Py_ssize_t str_len
        with nogil:
            for i in range(n):
                str_len = offs[i + 1] - offs[i] - 1
                if str_len == 0:
                    out_ptr[i] = 0
                    continue
                ptr = data + offs[i]
                valid = 1
                for j in range(str_len):
                    if not _is_alnum(<unsigned char>ptr[j]):
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
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i, j, n = self.length
        cdef char* ptr
        cdef bint valid
        cdef Py_ssize_t str_len
        with nogil:
            for i in range(n):
                str_len = offs[i + 1] - offs[i] - 1
                if str_len == 0:
                    out_ptr[i] = 0
                    continue
                ptr = data + offs[i]
                valid = 1
                for j in range(str_len):
                    if not _is_space(<unsigned char>ptr[j]):
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
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i, j, n = self.length
        cdef char* ptr
        cdef bint has_letter, all_lower
        cdef Py_ssize_t str_len
        cdef unsigned char c
        with nogil:
            for i in range(n):
                str_len = offs[i + 1] - offs[i] - 1
                if str_len == 0:
                    out_ptr[i] = 0
                    continue
                ptr = data + offs[i]
                has_letter = 0
                all_lower = 1
                for j in range(str_len):
                    c = <unsigned char>ptr[j]
                    if _is_alpha(c):
                        has_letter = 1
                        if not _is_lower(c):
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
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i, j, n = self.length
        cdef char* ptr
        cdef bint has_letter, all_upper
        cdef Py_ssize_t str_len
        cdef unsigned char c
        with nogil:
            for i in range(n):
                str_len = offs[i + 1] - offs[i] - 1
                if str_len == 0:
                    out_ptr[i] = 0
                    continue
                ptr = data + offs[i]
                has_letter = 0
                all_upper = 1
                for j in range(str_len):
                    c = <unsigned char>ptr[j]
                    if _is_alpha(c):
                        has_letter = 1
                        if not _is_upper(c):
                            all_upper = 0
                            break
                out_ptr[i] = 1 if (has_letter and all_upper) else 0
        return res

    # ------------------------------------------------------------
    # O(N) Memory-Safe implementations for stripping & replacing
    # ------------------------------------------------------------
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def strip(self):
        cdef Py_ssize_t n = self.length
        cdef AsciiColumn res = AsciiColumn(max(n, 1))
        res._reserve_bytes(self.byte_length)

        cdef char* src = self.data_c
        cdef char* dst = res.data_c
        cdef Py_ssize_t dst_offset = 0
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i, cur_len, left, right, new_len

        with nogil:
            for i in range(n):
                cur_len = offs[i+1] - offs[i] - 1
                if cur_len == 0:
                    dst[dst_offset] = 0
                    dst_offset += 1
                    res.offsets_c[i+1] = dst_offset
                    continue

                left = 0
                while left < cur_len and _is_space(<unsigned char>src[offs[i] + left]):
                    left += 1

                right = cur_len - 1
                while right >= left and _is_space(<unsigned char>src[offs[i] + right]):
                    right -= 1

                new_len = right - left + 1 if right >= left else 0

                if new_len > 0:
                    memcpy(dst + dst_offset, src + offs[i] + left, new_len)
                dst[dst_offset + new_len] = 0
                dst_offset += new_len + 1
                res.offsets_c[i+1] = dst_offset

        res.length = n
        res.byte_length = dst_offset
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def lstrip(self):
        cdef Py_ssize_t n = self.length
        cdef AsciiColumn res = AsciiColumn(max(n, 1))
        res._reserve_bytes(self.byte_length)

        cdef char* src = self.data_c
        cdef char* dst = res.data_c
        cdef Py_ssize_t dst_offset = 0
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i, cur_len, left, copy_len

        with nogil:
            for i in range(n):
                cur_len = offs[i+1] - offs[i] - 1
                if cur_len == 0:
                    dst[dst_offset] = 0
                    dst_offset += 1
                    res.offsets_c[i+1] = dst_offset
                    continue

                left = 0
                while left < cur_len and _is_space(<unsigned char>src[offs[i] + left]):
                    left += 1

                copy_len = cur_len - left
                if copy_len > 0:
                    memcpy(dst + dst_offset, src + offs[i] + left, copy_len)
                dst[dst_offset + copy_len] = 0
                dst_offset += copy_len + 1
                res.offsets_c[i+1] = dst_offset

        res.length = n
        res.byte_length = dst_offset
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def rstrip(self):
        cdef Py_ssize_t n = self.length
        cdef AsciiColumn res = AsciiColumn(max(n, 1))
        res._reserve_bytes(self.byte_length)

        cdef char* src = self.data_c
        cdef char* dst = res.data_c
        cdef Py_ssize_t dst_offset = 0
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i, cur_len, right, copy_len

        with nogil:
            for i in range(n):
                cur_len = offs[i+1] - offs[i] - 1
                if cur_len == 0:
                    dst[dst_offset] = 0
                    dst_offset += 1
                    res.offsets_c[i+1] = dst_offset
                    continue

                right = cur_len - 1
                while right >= 0 and _is_space(<unsigned char>src[offs[i] + right]):
                    right -= 1

                copy_len = right + 1
                if copy_len > 0:
                    memcpy(dst + dst_offset, src + offs[i], copy_len)
                dst[dst_offset + copy_len] = 0
                dst_offset += copy_len + 1
                res.offsets_c[i+1] = dst_offset

        res.length = n
        res.byte_length = dst_offset
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def removeprefix(self, str prefix):
        cdef bytes b_pre = prefix.encode('ascii')
        cdef const char* c_pre = b_pre
        cdef Py_ssize_t pre_len = len(b_pre)

        cdef Py_ssize_t n = self.length
        cdef AsciiColumn res = AsciiColumn(max(n, 1))
        res._reserve_bytes(self.byte_length)

        cdef char* src = self.data_c
        cdef char* dst = res.data_c
        cdef Py_ssize_t dst_offset = 0
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i, cur_len, copy_len

        with nogil:
            for i in range(n):
                cur_len = offs[i+1] - offs[i] - 1
                if cur_len >= pre_len and strncmp(src + offs[i], c_pre, pre_len) == 0:
                    copy_len = cur_len - pre_len
                    memcpy(dst + dst_offset, src + offs[i] + pre_len, copy_len)
                else:
                    copy_len = cur_len
                    memcpy(dst + dst_offset, src + offs[i], copy_len)

                dst[dst_offset + copy_len] = 0
                dst_offset += copy_len + 1
                res.offsets_c[i+1] = dst_offset

        res.length = n
        res.byte_length = dst_offset
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def removesuffix(self, str suffix):
        cdef bytes b_suf = suffix.encode('ascii')
        cdef const char* c_suf = b_suf
        cdef Py_ssize_t suf_len = len(b_suf)

        cdef Py_ssize_t n = self.length
        cdef AsciiColumn res = AsciiColumn(max(n, 1))
        res._reserve_bytes(self.byte_length)

        cdef char* src = self.data_c
        cdef char* dst = res.data_c
        cdef Py_ssize_t dst_offset = 0
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i, cur_len, copy_len

        with nogil:
            for i in range(n):
                cur_len = offs[i+1] - offs[i] - 1
                if cur_len >= suf_len and strncmp(src + offs[i] + cur_len - suf_len, c_suf, suf_len) == 0:
                    copy_len = cur_len - suf_len
                else:
                    copy_len = cur_len

                if copy_len > 0:
                    memcpy(dst + dst_offset, src + offs[i], copy_len)
                dst[dst_offset + copy_len] = 0
                dst_offset += copy_len + 1
                res.offsets_c[i+1] = dst_offset

        res.length = n
        res.byte_length = dst_offset
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def zfill(self, Py_ssize_t width):
        cdef Py_ssize_t n = self.length
        cdef AsciiColumn res = AsciiColumn(max(n, 1))

        cdef Py_ssize_t req_bytes = 0
        cdef Py_ssize_t i, curr_len
        cdef Py_ssize_t* offs = self.offsets_c
        for i in range(n):
            curr_len = offs[i+1] - offs[i] - 1
            req_bytes += max(curr_len, width) + 1

        res._reserve_bytes(req_bytes)

        cdef char* src = self.data_c
        cdef char* dst = res.data_c
        cdef Py_ssize_t dst_offset = 0
        cdef Py_ssize_t pad

        with nogil:
            for i in range(n):
                curr_len = offs[i+1] - offs[i] - 1
                if curr_len >= width:
                    memcpy(dst + dst_offset, src + offs[i], curr_len + 1)
                    dst_offset += curr_len + 1
                else:
                    pad = width - curr_len
                    memset(dst + dst_offset, b'0', pad)
                    memcpy(dst + dst_offset + pad, src + offs[i], curr_len + 1)
                    dst_offset += pad + curr_len + 1
                res.offsets_c[i+1] = dst_offset

        res.length = n
        res.byte_length = dst_offset
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
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                out_ptr[i] = <int>_c_find(data + offs[i], c_sub, sub_len)
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
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                out_ptr[i] = <int>_c_rfind(data + offs[i], c_sub, sub_len)
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
    def count(self, str substring):
        cdef bytes b_sub = substring.encode('ascii')
        cdef const char* c_sub = b_sub
        cdef size_t sub_len = len(b_sub)
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef char* data = self.data_c
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                out_ptr[i] = <int>_c_count(data + offs[i], c_sub, sub_len)
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
        cdef AsciiColumn res = AsciiColumn(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            res.append(self._get_str(i).title())
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def capitalize(self):
        cdef AsciiColumn res = AsciiColumn(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            res.append(self._get_str(i).capitalize())
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def swapcase(self):
        cdef AsciiColumn res = AsciiColumn(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            res.append(self._get_str(i).swapcase())
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def casefold(self):
        cdef AsciiColumn res = AsciiColumn(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            res.append(self._get_str(i).casefold())
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def center(self, int width, str fillchar=' '):
        cdef AsciiColumn res = AsciiColumn(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            res.append(self._get_str(i).center(width, fillchar))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def ljust(self, int width, str fillchar=' '):
        cdef AsciiColumn res = AsciiColumn(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            res.append(self._get_str(i).ljust(width, fillchar))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def rjust(self, int width, str fillchar=' '):
        cdef AsciiColumn res = AsciiColumn(max(self.length, 1))
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
        cdef AsciiColumn res = AsciiColumn(max(self.length, 1))
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
        cdef AsciiColumn res = AsciiColumn(max(self.length, 1))
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
        cdef AsciiColumn res = AsciiColumn(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        cdef AsciiColumn other_cc
        if isinstance(other, AsciiColumn):
            other_cc = <AsciiColumn>other
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
        cdef AsciiColumn res = AsciiColumn(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            res.append(pre + self._get_str(i))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def suffix(self, str suf):
        cdef AsciiColumn res = AsciiColumn(max(self.length, 1))
        cdef Py_ssize_t i, n = self.length
        for i in range(n):
            res.append(self._get_str(i) + suf)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def repeat(self, int repeats):
        cdef AsciiColumn res = AsciiColumn(max(self.length, 1))
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

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def _cmp_op(self, val, int op):
        cdef Py_ssize_t i, n = self.length
        cdef IntColumn res = IntColumn(max(n, 1))
        res.length = n
        cdef int* out_ptr = res.data_c
        cdef char* data = self.data_c
        cdef Py_ssize_t* offs = self.offsets_c
        cdef bytes v_bytes
        cdef const char* v_ptr
        cdef char* o_data
        cdef Py_ssize_t* o_offs
        cdef int cmp_res
        cdef AsciiColumn other

        if isinstance(val, AsciiColumn):
            other = <AsciiColumn>val
            if other.length != n:
                raise ValueError(f"Length mismatch: {n} vs {other.length}")
            o_data = other.data_c
            o_offs = other.offsets_c
            with nogil:
                for i in range(n):
                    cmp_res = strcmp(data + offs[i], o_data + o_offs[i])
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
                    cmp_res = strcmp(data + offs[i], v_ptr)
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
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                out_ptr[i] = 1 if (offs[i + 1] - offs[i] - 1) > 0 and other_ptr[i] != 0 else 0
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
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                out_ptr[i] = 1 if (offs[i + 1] - offs[i] - 1) > 0 or other_ptr[i] != 0 else 0
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __invert__(self):
        cdef IntColumn res = IntColumn(max(self.length, 1))
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i, n = self.length
        with nogil:
            for i in range(n):
                out_ptr[i] = 0 if (offs[i + 1] - offs[i] - 1) > 0 else 1
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

        cdef AsciiColumn col = AsciiColumn(max(n, 1))
        cdef str fill_str = str(fill) if fill is not None else ""
        cdef int* mask_ptr = mask.data_c
        cdef Py_ssize_t i

        for i in range(n):
            if mask_ptr[i] != 0:
                col.append(self._get_str(i))
            else:
                col.append(fill_str)
        return col

    @property
    def str(self):
        # 快路径：缓存命中直接返回（均摊实例化开销为 0）
        if self._str_cache is not None:
            return self._str_cache
        # 慢路径：仅首次访问时创建一次（C 级工厂，绕过参数打包）
        self._str_cache = _make_ascii_str_accessor(self)
        return self._str_cache

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

    cdef AsciiColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=False):
        cdef Py_ssize_t i, idx
        cdef Py_ssize_t req_cap = max(n, 1)
        cdef AsciiColumn new_col

        cdef Py_ssize_t req_bytes = 0
        for i in range(n):
            idx = indices[i]
            req_bytes += self.offsets_c[idx + 1] - self.offsets_c[idx]

        cdef Py_ssize_t req_byte_cap = max(req_bytes, 4096)
        cdef Py_ssize_t* new_offsets = <Py_ssize_t*>aligned_malloc((req_cap + 1) * sizeof(Py_ssize_t))
        cdef char* new_data = <char*>aligned_malloc(req_byte_cap)

        if not new_offsets or not new_data:
            if new_offsets: os_aligned_free(new_offsets)
            if new_data: os_aligned_free(new_data)
            raise MemoryError()

        new_offsets[0] = 0
        cdef Py_ssize_t current_byte_len = 0
        cdef Py_ssize_t start, end, str_len
        with nogil:
            for i in range(n):
                idx = indices[i]
                start = self.offsets_c[idx]
                end = self.offsets_c[idx + 1]
                str_len = end - start
                if str_len > 0:
                    memcpy(new_data + current_byte_len, self.data_c + start, str_len)
                current_byte_len += str_len
                new_offsets[i + 1] = current_byte_len

        if inplace:
            os_aligned_free(self.offsets_c)
            os_aligned_free(self.data_c)
            self.offsets_c = new_offsets
            self.data_c = new_data
            self.byte_capacity = req_byte_cap
            self.byte_length = current_byte_len
            self.capacity = req_cap
            self.length = n
            return self
        else:
            new_col = AsciiColumn(req_cap)
            os_aligned_free(new_col.offsets_c)
            os_aligned_free(new_col.data_c)
            new_col.offsets_c = new_offsets
            new_col.data_c = new_data
            new_col.byte_capacity = req_byte_cap
            new_col.byte_length = current_byte_len
            new_col.length = n
            return new_col

    # ------------------------------------------------------------
    # Bound Checked Memory Shifting
    # ------------------------------------------------------------
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def drop_row(self, Py_ssize_t row_idx):
        if row_idx < 0:
            row_idx += self.length
        if row_idx < 0 or row_idx >= self.length:
            raise IndexError("Row index out of bounds")

        cdef Py_ssize_t start_offset = self.offsets_c[row_idx]
        cdef Py_ssize_t end_offset = self.offsets_c[row_idx + 1]
        cdef Py_ssize_t char_len = end_offset - start_offset
        cdef Py_ssize_t rem_bytes = self.byte_length - end_offset
        if rem_bytes > 0:
            memmove(self.data_c + start_offset,
                    self.data_c + end_offset,
                    rem_bytes)
        self.byte_length -= char_len
        cdef Py_ssize_t i
        for i in range(row_idx, self.length):
            self.offsets_c[i] = self.offsets_c[i + 1] - char_len
        self.length -= 1

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def shift_left(self, Py_ssize_t pos):
        if pos < 0:
            pos += self.length
        if pos < 0 or pos >= self.length:
            raise IndexError("Index out of bounds")

        cdef Py_ssize_t col_len = self.length
        cdef Py_ssize_t start_offset = self.offsets_c[pos]
        cdef Py_ssize_t end_offset = self.offsets_c[pos + 1]
        cdef Py_ssize_t removed_len = end_offset - start_offset
        cdef Py_ssize_t rem = self.byte_length - end_offset
        cdef Py_ssize_t i
        if rem > 0:
            memmove(self.data_c + start_offset,
                    self.data_c + end_offset,
                    rem)
        self.byte_length -= removed_len
        for i in range(pos, col_len):
            self.offsets_c[i] = self.offsets_c[i + 1] - removed_len
        self.length = col_len - 1

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
        cdef UCS2Column uc
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
            uc = UCS2Column(safe_cap)
            uc.length = n
            for i in range(n):
                uc._set_item(i, self._get_str(i))
            return uc
        elif target_type == TYPE_DATE:
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

    # ------------------------------------------------------------
    # Extreme O(1) Overhead indexing bypassing Python string allocation
    # ------------------------------------------------------------
    def str_getitem(self, key):
        cdef Py_ssize_t n = self.length
        cdef AsciiColumn col = AsciiColumn(max(n, 1))
        cdef Py_ssize_t i, idx, s_len, idx_abs
        cdef Py_ssize_t* offs = self.offsets_c
        cdef char* src = self.data_c
        cdef char* dst
        cdef Py_ssize_t dst_offset = 0

        if isinstance(key, int):
            idx = <Py_ssize_t>key
            col._reserve_bytes(n * 2)
            dst = col.data_c
            with nogil:
                for i in range(n):
                    s_len = offs[i+1] - offs[i] - 1
                    idx_abs = s_len + idx if idx < 0 else idx
                    if 0 <= idx_abs < s_len:
                        dst[dst_offset] = src[offs[i] + idx_abs]
                        dst[dst_offset + 1] = 0
                        dst_offset += 2
                    else:
                        dst[dst_offset] = 0
                        dst_offset += 1
                    col.offsets_c[i+1] = dst_offset
            col.length = n
            col.byte_length = dst_offset
            return col

        elif isinstance(key, slice):
            # Fallback to python indexing for complex slices
            for i in range(n):
                col.append(self._get_str(i)[key])
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


# ============================================================
# AsciiStringAccessor - pandas-style .str accessor for AsciiColumn
# ============================================================

cdef class AsciiStringAccessor:
    """Pandas-style .str accessor for AsciiColumn.
    All methods delegate to AsciiColumn.
    实例由 AsciiColumn.str property 懒加载缓存（字段声明见 asciicolumn.pxd）。
    """

    def __cinit__(self):
        # 无参 __cinit__：允许工厂函数直接 __new__ 绕过参数打包
        pass

    # ==================== Search / Match ====================

    @cython.always_allow_keywords(False)
    def contains(self, pat):
        return self._col.str_contains_mask(pat)

    def startswith(self, prefix):
        return self._col.startswith(prefix)

    def endswith(self, suffix):
        return self._col.endswith(suffix)

    @cython.always_allow_keywords(False)
    def find(self, str sub):
        return self._col.find(sub)

    @cython.always_allow_keywords(False)
    def rfind(self, str sub):
        return self._col.rfind(sub)

    @cython.always_allow_keywords(False)
    def index(self, str sub):
        return self._col.find(sub)

    @cython.always_allow_keywords(False)
    def rindex(self, str sub):
        return self._col.rindex(sub)

    @cython.always_allow_keywords(False)
    def count(self, str sub):
        return self._col.count(sub)

    @cython.always_allow_keywords(False)
    def match(self, str pattern):
        return self._col.match(pattern)

    @cython.always_allow_keywords(False)
    def fullmatch(self, str pattern):
        return self._col.fullmatch(pattern)

    # ==================== Extract / Findall ====================

    def extract(self, str pattern, int group=1):
        return self._col.extract(pattern)

    def extractall(self, str pattern):
        return self._col.extractall(pattern)

    def findall(self, str pattern):
        return self._col.findall(pattern)

    # ==================== Type Check ====================

    @cython.always_allow_keywords(False)
    def isalpha(self):
        return self._col.isalpha()

    @cython.always_allow_keywords(False)
    def isnumeric(self):
        return self._col.isnumeric()

    @cython.always_allow_keywords(False)
    def isalnum(self):
        return self._col.isalnum()

    @cython.always_allow_keywords(False)
    def isdigit(self):
        return self._col.isdigit()

    @cython.always_allow_keywords(False)
    def isdecimal(self):
        return self._col.isdecimal()

    @cython.always_allow_keywords(False)
    def islower(self):
        return self._col.islower()

    @cython.always_allow_keywords(False)
    def isupper(self):
        return self._col.isupper()

    @cython.always_allow_keywords(False)
    def isspace(self):
        return self._col.isspace()

    # ==================== Case Conversion ====================

    @cython.always_allow_keywords(False)
    def upper(self):
        return self._col.upper()

    @cython.always_allow_keywords(False)
    def lower(self):
        return self._col.lower()

    @cython.always_allow_keywords(False)
    def title(self):
        return self._col.title()

    @cython.always_allow_keywords(False)
    def capitalize(self):
        return self._col.capitalize()

    @cython.always_allow_keywords(False)
    def swapcase(self):
        return self._col.swapcase()

    @cython.always_allow_keywords(False)
    def casefold(self):
        return self._col.casefold()

    # ==================== Strip / Pad ====================

    def strip(self, chars=None):
        return self._col.strip(chars) if chars is not None else self._col.strip()

    def lstrip(self, chars=None):
        return self._col.lstrip(chars) if chars is not None else self._col.lstrip()

    def rstrip(self, chars=None):
        return self._col.rstrip(chars) if chars is not None else self._col.rstrip()

    def center(self, int width, str fillchar=' '):
        return self._col.center(width, fillchar)

    def ljust(self, int width, str fillchar=' '):
        return self._col.ljust(width, fillchar)

    def rjust(self, int width, str fillchar=' '):
        return self._col.rjust(width, fillchar)

    def pad(self, int width, str side='left', str fillchar=' '):
        return self._col.pad(width, side, fillchar)

    def zfill(self, int width):
        return self._col.zfill(width)

    # ==================== Replace ====================

    def replace(self, str old, str new, int count=-1):
        return self._col.replace(old, new)

    # ==================== Split ====================

    def split(self, str sep=None, int maxsplit=-1):
        return self._col.str_split(sep, maxsplit)

    def rsplit(self, str sep=None, int maxsplit=-1):
        return self._col.rsplit(sep, maxsplit)

    @cython.always_allow_keywords(False)
    def partition(self, str sep):
        return self._col.partition(sep)

    @cython.always_allow_keywords(False)
    def rpartition(self, str sep):
        return self._col.rpartition(sep)

    # ==================== Slicing / Access ====================

    def get(self, int index, str default=''):
        return self._col.get(index, default)

    def slice(self, int start=0, int stop=-1, int step=1):
        return self._col.slice(start, stop, step)

    @cython.always_allow_keywords(False)
    def len(self):
        return self._col.len()

    def __getitem__(self, key):
        return self._col.str_getitem(key)

    # ==================== Concat ====================

    def cat(self, other, str sep=''):
        return self._col.cat(other, sep)

    def prefix(self, str pre):
        return self._col.prefix(pre)

    def suffix(self, str suf):
        return self._col.suffix(suf)

    def repeat(self, int repeats):
        return self._col.repeat(repeats)


cdef inline AsciiStringAccessor _make_ascii_str_accessor(AsciiColumn col):
    # C 级工厂：__new__ 无参调用直接命中 tp_new，彻底绕过 Python 参数打包/解包
    # （与 fastdataframe.pyx 的 _make_metric_accessor 同一模式）
    cdef AsciiStringAccessor acc = <AsciiStringAccessor>AsciiStringAccessor.__new__(AsciiStringAccessor)
    # 直接写底层 C 结构体字段（Cython 自动处理 Py_INCREF）
    acc._col = col
    return acc