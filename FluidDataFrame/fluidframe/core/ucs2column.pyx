# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False

cimport cython
from libc.string cimport memcpy, memmove
from libc.stdio cimport snprintf
import re
from fluidframe.core._base cimport BaseColumn, aligned_malloc, aligned_realloc, os_aligned_free, TYPE_UCS2, TYPE_INT, TYPE_DOUBLE, TYPE_CHAR256, TYPE_BOOL, TYPE_DATE, TYPE_CATEGORY, TYPE_DATETIME, TYPE_TIME, _resolve_astype_target
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.doublecolumn cimport DoubleColumn
from fluidframe.core.boolcolumn cimport BoolColumn
from fluidframe.core.asciicolumn cimport AsciiColumn
from fluidframe.core.datecolumn cimport DateColumn
from fluidframe.core.categorycolumn cimport CategoryColumn
from fluidframe.core.datetimecolumn cimport DateTimeColumn
from fluidframe.core.timecolumn cimport TimeColumn


from cpython.unicode cimport (
    PyUnicode_1BYTE_KIND, PyUnicode_2BYTE_KIND, PyUnicode_4BYTE_KIND,
    PyUnicode_FromKindAndData, PyUnicode_GET_LENGTH, PyUnicode_READ_CHAR,
    PyUnicode_KIND, PyUnicode_DATA,
    Py_UCS1, Py_UCS2
)


cdef inline void _copy_py_str_to_ucs2(str val, Py_UCS2* dest, Py_ssize_t length) noexcept:
    """
    极速、安全的摄入器：解析底层 Unicode 物理布局并批量直拷
    超限宽字符 (Emoji等) 会在这里被永久替换为 '?' (0x003F)
    """
    cdef int kind = PyUnicode_KIND(val)
    cdef void* data = PyUnicode_DATA(val)
    cdef Py_ssize_t i
    cdef Py_UCS4 ch

    if kind == PyUnicode_1BYTE_KIND:
        # ASCII / Latin1 数据，直接宽化拷贝
        for i in range(length):
            dest[i] = <Py_UCS2>(<Py_UCS1*>data)[i]
    elif kind == PyUnicode_2BYTE_KIND:
        # BMP 基本多文种平面 (包含中文)，安全且完美对齐，内存直拷
        memcpy(dest, data, length * sizeof(Py_UCS2))
    elif kind == PyUnicode_4BYTE_KIND:
        # 包含了 Emoji 的宽字符字符串，清洗拦截
        for i in range(length):
            ch = (<Py_UCS4*>data)[i]
            dest[i] = <Py_UCS2>ch if ch <= 0xFFFF else 0x003F
    else:
        # 安全降级，理论上 Python3.3+ 不会走到这里
        for i in range(length):
            ch = PyUnicode_READ_CHAR(val, i)
            dest[i] = <Py_UCS2>ch if ch <= 0xFFFF else 0x003F

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

cdef inline bint _ucs2_startswith(Py_UCS2* s, Py_ssize_t s_len, Py_UCS2* prefix, Py_ssize_t p_len) noexcept nogil:
    if p_len == 0: return True
    if s_len < p_len: return False
    cdef Py_ssize_t i
    for i in range(p_len):
        if s[i] != prefix[i]: return False
    return True

cdef inline bint _ucs2_endswith(Py_UCS2* s, Py_ssize_t s_len, Py_UCS2* suffix, Py_ssize_t suf_len) noexcept nogil:
    if suf_len == 0: return True
    if s_len < suf_len: return False
    cdef Py_ssize_t i
    cdef Py_ssize_t offset = s_len - suf_len
    for i in range(suf_len):
        if s[offset + i] != suffix[i]: return False
    return True

cdef inline int _cmp_ucs2_buffers(Py_UCS2* a, Py_ssize_t len_a, Py_UCS2* b, Py_ssize_t len_b) noexcept nogil:
    cdef Py_ssize_t min_len = len_a if len_a < len_b else len_b
    cdef Py_ssize_t i
    for i in range(min_len):
        if a[i] < b[i]: return -1
        if a[i] > b[i]: return 1
    if len_a < len_b: return -1
    if len_a > len_b: return 1
    return 0

@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef class UCS2Column(BaseColumn):

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __cinit__(self, Py_ssize_t init_capacity=64):
        self.capacity = max(init_capacity, 1)
        self.length = 0
        self.dtype = TYPE_UCS2
        self.char_capacity = max(self.capacity * 16, 1024)
        self.char_length = 0
        self._str_cache = None

        self.offsets_c = <Py_ssize_t*>aligned_malloc((self.capacity + 1) * sizeof(Py_ssize_t))
        self.chars_c = <Py_UCS2*>aligned_malloc(self.char_capacity * sizeof(Py_UCS2))

        if not self.offsets_c or not self.chars_c:
            raise MemoryError()

        self.offsets_c[0] = 0

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __dealloc__(self):
        if self.offsets_c is not NULL: os_aligned_free(self.offsets_c)
        if self.chars_c is not NULL: os_aligned_free(self.chars_c)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def get_size(self):
        return self.capacity * sizeof(Py_UCS2) + (self.capacity + 1) * sizeof(Py_ssize_t)

    cdef void _reserve_rows(self, Py_ssize_t new_cap) except *:
        cdef Py_ssize_t* new_ptr
        if new_cap > self.capacity:
            new_ptr = <Py_ssize_t*>aligned_realloc(self.offsets_c, (self.capacity + 1) * sizeof(Py_ssize_t), (new_cap + 1) * sizeof(Py_ssize_t))
            if not new_ptr: raise MemoryError()
            self.offsets_c = new_ptr
            self.capacity = new_cap

    cdef void _reserve_chars(self, Py_ssize_t new_char_cap) except *:
        cdef Py_UCS2* new_ptr
        if new_char_cap > self.char_capacity:
            new_ptr = <Py_UCS2*>aligned_realloc(self.chars_c, self.char_capacity * sizeof(Py_UCS2), new_char_cap * sizeof(Py_UCS2))
            if not new_ptr: raise MemoryError()
            self.chars_c = new_ptr
            self.char_capacity = new_char_cap

    cdef inline str _get_str(self, Py_ssize_t idx):
        cdef Py_ssize_t start = self.offsets_c[idx]
        cdef Py_ssize_t str_len = self.offsets_c[idx + 1] - start
        if str_len == 0:
            return ""
        return PyUnicode_FromKindAndData(2, self.chars_c + start, str_len)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def append(self, str val not None):
        cdef Py_ssize_t str_len = PyUnicode_GET_LENGTH(val)

        if self.length >= self.capacity:
            self._reserve_rows(self.capacity * 2)

        if self.char_length + str_len > self.char_capacity:
            self._reserve_chars(max(self.char_capacity * 2, self.char_length + str_len + 1024))

        if str_len > 0:
            _copy_py_str_to_ucs2(val, self.chars_c + self.char_length, str_len)

        self.char_length += str_len
        self.length += 1
        self.offsets_c[self.length] = self.char_length

    cdef void append_nolen(self, str val) except *:
        """写入当前行(length)位置的值，但不推进 length（由 add_row 统一提交）。"""
        if self.length >= self.capacity:
            raise BufferError("UCS2Column row capacity (%d) reached." % self.capacity)
        cdef Py_ssize_t str_len = PyUnicode_GET_LENGTH(val)
        if self.char_length + str_len > self.char_capacity:
            self._reserve_chars(max(self.char_capacity * 2, self.char_length + str_len + 1024))
        if str_len > 0:
            _copy_py_str_to_ucs2(val, self.chars_c + self.char_length, str_len)
        self.char_length += str_len
        self.offsets_c[self.length + 1] = self.char_length

    cdef void undo_nolen(self) noexcept:
        """回滚一次未提交(未推进 length)的写入，保证重试时不会产生残留字符。"""
        self.char_length = self.offsets_c[self.length]
        self.offsets_c[self.length + 1] = self.char_length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def copy(self):
        cdef UCS2Column new_col = UCS2Column(self.capacity)
        new_col._reserve_chars(self.char_capacity)
        new_col.length = self.length
        new_col.char_length = self.char_length
        memcpy(new_col.offsets_c, self.offsets_c, (self.length + 1) * sizeof(Py_ssize_t))
        memcpy(new_col.chars_c, self.chars_c, self.char_length * sizeof(Py_UCS2))
        return new_col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def concat(self, UCS2Column other):
        self._reserve_rows(self.length + other.length)
        self._reserve_chars(self.char_length + other.char_length)

        cdef Py_ssize_t i
        cdef Py_ssize_t base_offset = self.char_length
        cdef Py_ssize_t other_len = other.length
        cdef Py_ssize_t* self_offs = self.offsets_c
        cdef Py_ssize_t* other_offs = other.offsets_c

        memcpy(self.chars_c + self.char_length, other.chars_c, other.char_length * sizeof(Py_UCS2))

        for i in range(other_len):
            self_offs[self.length + 1 + i] = base_offset + other_offs[i + 1]

        self.length += other_len
        self.char_length += other.char_length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __getitem__(self, Py_ssize_t idx):
        if idx < 0: idx += self.length
        if idx < 0 or idx >= self.length: raise IndexError()
        return self._get_str(idx)

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
    @cython.always_allow_keywords(False)
    def len(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                out_ptr[i] = <int>(offs[i + 1] - offs[i])
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def startswith(self, str prefix):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i, start, str_len
        cdef Py_ssize_t sub_len = PyUnicode_GET_LENGTH(prefix)

        if sub_len == 0:
            for i in range(self.length):
                out_ptr[i] = 1
            return res

        cdef Py_UCS2* n_buf = <Py_UCS2*>aligned_malloc(sub_len * sizeof(Py_UCS2))
        if not n_buf: raise MemoryError()
        _copy_py_str_to_ucs2(prefix, n_buf, sub_len)

        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_UCS2* chars = self.chars_c
        with nogil:
            for i in range(self.length):
                start = offs[i]
                str_len = offs[i+1] - start
                out_ptr[i] = 1 if _ucs2_startswith(chars + start, str_len, n_buf, sub_len) else 0

        os_aligned_free(n_buf)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def endswith(self, str suffix):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i, start, str_len
        cdef Py_ssize_t sub_len = PyUnicode_GET_LENGTH(suffix)

        if sub_len == 0:
            for i in range(self.length):
                out_ptr[i] = 1
            return res

        cdef Py_UCS2* n_buf = <Py_UCS2*>aligned_malloc(sub_len * sizeof(Py_UCS2))
        if not n_buf: raise MemoryError()
        _copy_py_str_to_ucs2(suffix, n_buf, sub_len)

        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_UCS2* chars = self.chars_c
        with nogil:
            for i in range(self.length):
                start = offs[i]
                str_len = offs[i+1] - start
                out_ptr[i] = 1 if _ucs2_endswith(chars + start, str_len, n_buf, sub_len) else 0

        os_aligned_free(n_buf)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def contains(self, object pat):
        """
        判断字符串是否包含指定的正则表达式模式。
        pat: 可以是正则字符串 (如 r"\d+")，也可以是已编译的正则对象 (re.compile)
        """
        cdef IntColumn res = IntColumn(max(self.capacity, 1))
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i
        cdef str s

        cdef object compiled_pat
        if isinstance(pat, str):
            compiled_pat = re.compile(pat)
        elif hasattr(pat, 'search'):
            compiled_pat = pat
        else:
            compiled_pat = re.compile(str(pat))

        for i in range(self.length):
            s = self._get_str(i)
            if compiled_pat.search(s) is not None:
                out_ptr[i] = 1
            else:
                out_ptr[i] = 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def find(self, str substring):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i, start, str_len
        cdef Py_ssize_t sub_len = PyUnicode_GET_LENGTH(substring)

        if sub_len == 0:
            for i in range(self.length):
                out_ptr[i] = 0
            return res

        cdef Py_UCS2* n_buf = <Py_UCS2*>aligned_malloc(sub_len * sizeof(Py_UCS2))
        if not n_buf: raise MemoryError()
        _copy_py_str_to_ucs2(substring, n_buf, sub_len)

        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_UCS2* chars = self.chars_c
        cdef Py_UCS2* match

        with nogil:
            for i in range(self.length):
                start = offs[i]
                str_len = offs[i+1] - start
                match = _ucs2_strstr(chars + start, <int>str_len, n_buf, <int>sub_len)
                out_ptr[i] = <int>(match - (chars + start)) if match != NULL else -1

        os_aligned_free(n_buf)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def replace(self, str old, str new):
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self._get_str(i).replace(old, new)
            self._set_item(i, s)
        return self

    cdef inline void _set_item(self, Py_ssize_t idx, str val):
        """Replace string at given index in-place. O(N) array shift."""
        cdef Py_ssize_t old_start = self.offsets_c[idx]
        cdef Py_ssize_t old_len = self.offsets_c[idx + 1] - old_start
        cdef Py_ssize_t new_len = PyUnicode_GET_LENGTH(val)
        cdef Py_ssize_t diff, oi

        if new_len > old_len:
            diff = new_len - old_len
            if self.char_length + diff > self.char_capacity:
                self._reserve_chars(max(self.char_capacity * 2, self.char_length + diff + 1024))

            if self.char_length - (old_start + old_len) > 0:
                memmove(self.chars_c + old_start + new_len,
                        self.chars_c + old_start + old_len,
                        (self.char_length - old_start - old_len) * sizeof(Py_UCS2))
        elif new_len < old_len:
            if self.char_length - (old_start + old_len) > 0:
                memmove(self.chars_c + old_start + new_len,
                        self.chars_c + old_start + old_len,
                        (self.char_length - old_start - old_len) * sizeof(Py_UCS2))

        if new_len > 0:
            _copy_py_str_to_ucs2(val, self.chars_c + old_start, new_len)

        self.char_length += (new_len - old_len)
        diff = new_len - old_len
        for oi in range(idx + 1, self.length + 1):
            self.offsets_c[oi] += diff

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def lower(self):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self._get_str(i)
            res.append(s.lower())
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def upper(self):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self._get_str(i)
            res.append(s.upper())
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def strip(self):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self._get_str(i)
            res.append(s.strip())
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def lstrip(self):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self._get_str(i)
            res.append(s.lstrip())
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def rstrip(self):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self._get_str(i)
            res.append(s.rstrip())
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def count(self, str substring):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i
        cdef str s
        cdef Py_ssize_t sub_len = len(substring)
        cdef Py_ssize_t cnt, pos
        if sub_len == 0:
            for i in range(self.length):
                out_ptr[i] = <int>(len(self._get_str(i)) + 1)
            return res
        for i in range(self.length):
            s = self._get_str(i)
            cnt = 0
            pos = 0
            while True:
                pos = s.find(substring, pos)
                if pos == -1:
                    break
                cnt += 1
                pos += sub_len
            out_ptr[i] = <int>cnt
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isdigit(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self._get_str(i)
            out_ptr[i] = 1 if len(s) > 0 and s.isdigit() else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isalpha(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self._get_str(i)
            out_ptr[i] = 1 if len(s) > 0 and s.isalpha() else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isalnum(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self._get_str(i)
            out_ptr[i] = 1 if len(s) > 0 and s.isalnum() else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isspace(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self._get_str(i)
            out_ptr[i] = 1 if len(s) > 0 and s.isspace() else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def rfind(self, str substring):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i, start, str_len
        cdef Py_ssize_t sub_len = PyUnicode_GET_LENGTH(substring)

        if sub_len == 0:
            for i in range(self.length):
                start = self.offsets_c[i]
                out_ptr[i] = <int>(self.offsets_c[i + 1] - start)
            return res

        cdef Py_UCS2* n_buf = <Py_UCS2*>aligned_malloc(sub_len * sizeof(Py_UCS2))
        if not n_buf: raise MemoryError()
        _copy_py_str_to_ucs2(substring, n_buf, sub_len)

        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_UCS2* chars = self.chars_c
        cdef Py_ssize_t n = self.length
        cdef Py_ssize_t j, k
        cdef bint bmatch

        for i in range(n):
            start = offs[i]
            str_len = offs[i+1] - start
            out_ptr[i] = -1
            if sub_len <= str_len:
                for j in range(str_len - sub_len, -1, -1):
                    bmatch = True
                    for k in range(sub_len):
                        if chars[start + j + k] != n_buf[k]:
                            bmatch = False
                            break
                    if bmatch:
                        out_ptr[i] = <int>j
                        break

        os_aligned_free(n_buf)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def rindex(self, str substring):
        cdef IntColumn res = self.rfind(substring)
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i
        for i in range(self.length):
            if out_ptr[i] == -1:
                raise ValueError(f"substring '{substring}' not found")
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def title(self):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        for i in range(self.length):
            res.append(self._get_str(i).title())
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def capitalize(self):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        for i in range(self.length):
            res.append(self._get_str(i).capitalize())
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def swapcase(self):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        for i in range(self.length):
            res.append(self._get_str(i).swapcase())
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def casefold(self):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        for i in range(self.length):
            res.append(self._get_str(i).casefold())
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def islower(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self._get_str(i)
            out_ptr[i] = 1 if len(s) > 0 and s.islower() else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isupper(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self._get_str(i)
            out_ptr[i] = 1 if len(s) > 0 and s.isupper() else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isnumeric(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self._get_str(i)
            out_ptr[i] = 1 if len(s) > 0 and s.isnumeric() else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isdecimal(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self._get_str(i)
            out_ptr[i] = 1 if len(s) > 0 and s.isdecimal() else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def match(self, str pattern):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i
        cdef object compiled = re.compile(pattern)
        cdef str s
        for i in range(self.length):
            s = self._get_str(i)
            out_ptr[i] = 1 if compiled.match(s) is not None else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def fullmatch(self, str pattern):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i
        cdef object compiled = re.compile(pattern)
        cdef str s
        for i in range(self.length):
            s = self._get_str(i)
            out_ptr[i] = 1 if compiled.fullmatch(s) is not None else 0
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def center(self, int width, str fillchar=' '):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        for i in range(self.length):
            res.append(self._get_str(i).center(width, fillchar))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def ljust(self, int width, str fillchar=' '):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        for i in range(self.length):
            res.append(self._get_str(i).ljust(width, fillchar))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def rjust(self, int width, str fillchar=' '):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        for i in range(self.length):
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
    def zfill(self, int width):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        for i in range(self.length):
            res.append(self._get_str(i).zfill(width))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def rsplit(self, str sep=None, int maxsplit=-1):
        cdef list res = []
        cdef Py_ssize_t i
        for i in range(self.length):
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
        cdef Py_ssize_t i
        for i in range(self.length):
            res.append(self._get_str(i).partition(sep))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def rpartition(self, str sep):
        cdef list res = []
        cdef Py_ssize_t i
        for i in range(self.length):
            res.append(self._get_str(i).rpartition(sep))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def get(self, int index, str default=''):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self._get_str(i)
            if 0 <= index < len(s):
                res.append(s[index])
            else:
                res.append(default)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def slice(self, int start=0, int stop=-1, int step=1):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        for i in range(self.length):
            s = self._get_str(i)
            if stop < 0:
                res.append(s[start::step] if step != 0 else s[start:stop:step])
            else:
                res.append(s[start:stop:step])
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cat(self, other, str sep=''):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i, n = self.length
        if isinstance(other, UCS2Column):
            other_uc = <UCS2Column>other
            if other_uc.length != n:
                raise ValueError(f"Length mismatch: {n} vs {other_uc.length}")
            for i in range(n):
                if sep:
                    res.append(self._get_str(i) + sep + other_uc._get_str(i))
                else:
                    res.append(self._get_str(i) + other_uc._get_str(i))
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
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        for i in range(self.length):
            res.append(pre + self._get_str(i))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def suffix(self, str suf):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        for i in range(self.length):
            res.append(self._get_str(i) + suf)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def repeat(self, int repeats):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        for i in range(self.length):
            res.append(self._get_str(i) * repeats)
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def findall(self, str pattern):
        import re
        cdef list res = []
        cdef Py_ssize_t i
        cdef object compiled = re.compile(pattern)
        for i in range(self.length):
            res.append(compiled.findall(self._get_str(i)))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def extract(self, str pattern, int group=1):
        cdef UCS2Column res = UCS2Column(self.capacity)
        cdef Py_ssize_t i
        cdef object compiled = re.compile(pattern)
        cdef str s, val
        for i in range(self.length):
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
    def extractall(self, str pattern, int group=1):
        import re
        cdef list res = []
        cdef Py_ssize_t i
        cdef object compiled = re.compile(pattern)
        for i in range(self.length):
            matches = []
            for m in compiled.finditer(self._get_str(i)):
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
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self._get_str(i)
            if s not in seen:
                seen.add(s)
                result.append(s)
        return result

    cdef IntColumn _cmp_op(self, val, int op):
        cdef Py_ssize_t i
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef char err_buf[128]

        cdef Py_ssize_t* s_offs = self.offsets_c
        cdef Py_UCS2* s_chars = self.chars_c
        cdef Py_UCS2* o_chars
        cdef Py_ssize_t* o_offs
        cdef Py_UCS2* v_buf
        cdef Py_ssize_t v_len, len_s, len_o
        cdef int cmp_res

        if isinstance(val, UCS2Column):
            other = <UCS2Column>val
            if other.length != self.length:
                snprintf(err_buf, sizeof(err_buf), "Length mismatch: %zd vs %zd", self.length, other.length)
                raise ValueError(err_buf.decode('ascii'))

            o_offs = other.offsets_c
            o_chars = other.chars_c

            with nogil:
                for i in range(self.length):
                    len_s = s_offs[i+1] - s_offs[i]
                    len_o = o_offs[i+1] - o_offs[i]
                    cmp_res = _cmp_ucs2_buffers(s_chars + s_offs[i], len_s, o_chars + o_offs[i], len_o)

                    if op == 0:   out_ptr[i] = 1 if cmp_res < 0 else 0
                    elif op == 1: out_ptr[i] = 1 if cmp_res <= 0 else 0
                    elif op == 2: out_ptr[i] = 1 if cmp_res > 0 else 0
                    elif op == 3: out_ptr[i] = 1 if cmp_res >= 0 else 0
                    elif op == 4: out_ptr[i] = 1 if cmp_res == 0 else 0
                    elif op == 5: out_ptr[i] = 1 if cmp_res != 0 else 0
        else:
            v_str = <str>val
            v_len = PyUnicode_GET_LENGTH(v_str)
            v_buf = <Py_UCS2*>aligned_malloc(max(v_len, 1) * sizeof(Py_UCS2))
            if not v_buf: raise MemoryError()

            _copy_py_str_to_ucs2(v_str, v_buf, v_len)

            with nogil:
                for i in range(self.length):
                    len_s = s_offs[i+1] - s_offs[i]
                    cmp_res = _cmp_ucs2_buffers(s_chars + s_offs[i], len_s, v_buf, v_len)

                    if op == 0:   out_ptr[i] = 1 if cmp_res < 0 else 0
                    elif op == 1: out_ptr[i] = 1 if cmp_res <= 0 else 0
                    elif op == 2: out_ptr[i] = 1 if cmp_res > 0 else 0
                    elif op == 3: out_ptr[i] = 1 if cmp_res >= 0 else 0
                    elif op == 4: out_ptr[i] = 1 if cmp_res == 0 else 0
                    elif op == 5: out_ptr[i] = 1 if cmp_res != 0 else 0

            os_aligned_free(v_buf)
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
            raise ValueError("Length mismatch: %zd vs %zd" % (self.length, other.length))

        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef int* other_ptr = (<IntColumn>other).data_c
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i

        with nogil:
            for i in range(self.length):
                out_ptr[i] = 1 if (offs[i+1] - offs[i]) > 0 and other_ptr[i] else 0
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __or__(self, other):
        if not isinstance(other, IntColumn):
            raise TypeError("Cannot | non-IntColumn mask")
        if self.length != other.length:
            raise ValueError("Length mismatch: %zd vs %zd" % (self.length, other.length))

        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef int* other_ptr = (<IntColumn>other).data_c
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i

        with nogil:
            for i in range(self.length):
                out_ptr[i] = 1 if (offs[i+1] - offs[i]) > 0 or other_ptr[i] else 0
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __invert__(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t* offs = self.offsets_c
        cdef Py_ssize_t i

        with nogil:
            for i in range(self.length):
                out_ptr[i] = 0 if (offs[i+1] - offs[i]) > 0 else 1
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def isin(self, set values):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef int* out_ptr = res.data_c
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self._get_str(i)
            out_ptr[i] = 1 if s in values else 0
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def where_mask(self, IntColumn mask, object fill):
        cdef Py_ssize_t n = self.length
        if mask.length != n:
            raise ValueError("Mask length %zd != column length %zd" % (mask.length, n))

        cdef UCS2Column col = UCS2Column(n)
        cdef str fill_str = str(fill) if fill is not None else ""
        cdef int* mask_ptr = mask.data_c
        cdef Py_ssize_t i

        for i in range(n):
            if mask_ptr[i]:
                col.append(self._get_str(i))
            else:
                col.append(fill_str)
        return col

    @property
    def _str(self):
        # 快路径：缓存命中直接返回（均摊实例化开销为 0）
        if self._str_cache is not None:
            return self._str_cache
        # 慢路径：仅首次访问时创建一次（C 级工厂，绕过参数打包）
        self._str_cache = _make_ucs2_str_accessor(self)
        return self._str_cache

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __str__(self):
        cdef list items = []
        cdef Py_ssize_t i
        for i in range(self.length):
            items.append(repr(self._get_str(i)))
        return f"UCS2Column([{', '.join(items)}])"

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __bool__(self):
        return self.length > 0

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __len__(self):
        return self.length

    cdef UCS2Column take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=False):
        cdef Py_ssize_t req_cap = max(n, 1)

        cdef Py_ssize_t req_char_len = 0
        cdef Py_ssize_t i, idx
        for i in range(n):
            idx = indices[i]
            if idx < 0:
                continue
            req_char_len += self.offsets_c[idx + 1] - self.offsets_c[idx]

        cdef Py_ssize_t req_char_cap = max(req_char_len, 1024)
        cdef Py_ssize_t* new_offsets = <Py_ssize_t*>aligned_malloc((req_cap + 1) * sizeof(Py_ssize_t))
        cdef Py_UCS2* new_chars = <Py_UCS2*>aligned_malloc(req_char_cap * sizeof(Py_UCS2))

        if not new_offsets or not new_chars:
            if new_offsets: os_aligned_free(new_offsets)
            if new_chars: os_aligned_free(new_chars)
            raise MemoryError()

        new_offsets[0] = 0
        cdef Py_ssize_t current_char_len = 0
        cdef Py_ssize_t start, end, str_len
        with nogil:
            for i in range(n):
                idx = indices[i]
                if idx < 0:
                    new_offsets[i + 1] = current_char_len
                    continue
                start = self.offsets_c[idx]
                end = self.offsets_c[idx + 1]
                str_len = end - start
                if str_len > 0:
                    memcpy(new_chars + current_char_len, self.chars_c + start, str_len * sizeof(Py_UCS2))
                current_char_len += str_len
                new_offsets[i + 1] = current_char_len

        if inplace:
            os_aligned_free(self.offsets_c)
            os_aligned_free(self.chars_c)
            self.offsets_c = new_offsets
            self.chars_c = new_chars
            self.char_capacity = req_char_cap
            self.char_length = current_char_len
            self.capacity = req_cap
            self.length = n
            return self
        else:
            new_col = UCS2Column(req_cap)
            os_aligned_free(new_col.offsets_c)
            os_aligned_free(new_col.chars_c)
            new_col.offsets_c = new_offsets
            new_col.chars_c = new_chars
            new_col.char_capacity = req_char_cap
            new_col.char_length = current_char_len
            new_col.length = n
            return new_col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def drop_row(self, Py_ssize_t row_idx):
        cdef Py_ssize_t start_offset = self.offsets_c[row_idx]
        cdef Py_ssize_t end_offset = self.offsets_c[row_idx + 1]
        cdef Py_ssize_t char_len = end_offset - start_offset
        cdef Py_ssize_t rem_chars = self.char_length - end_offset
        if rem_chars > 0:
            memmove(self.chars_c + start_offset,
                    self.chars_c + end_offset,
                    rem_chars * sizeof(Py_UCS2))
        self.char_length -= char_len
        cdef Py_ssize_t i
        for i in range(row_idx, self.length):
            self.offsets_c[i] = self.offsets_c[i + 1] - char_len
        self.length -= 1

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def shift_left(self, Py_ssize_t pos):
        cdef Py_ssize_t col_len = self.length
        cdef Py_ssize_t start_offset = self.offsets_c[pos]
        cdef Py_ssize_t end_offset = self.offsets_c[pos + 1]
        cdef Py_ssize_t removed_len = end_offset - start_offset
        cdef Py_ssize_t rem = self.char_length - end_offset
        cdef Py_ssize_t i
        if rem > 0:
            memmove(self.chars_c + start_offset,
                    self.chars_c + end_offset,
                    rem * sizeof(Py_UCS2))
        self.char_length -= removed_len
        for i in range(pos, col_len):
            self.offsets_c[i] = self.offsets_c[i + 1] - removed_len
        self.length = col_len - 1

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def astype(self, object target):
        cdef int target_type = _resolve_astype_target(target)
        cdef Py_ssize_t i, n = self.length
        cdef str s
        cdef Py_ssize_t safe_cap = max(n, 1)

        cdef IntColumn ic
        cdef DoubleColumn dc
        cdef BoolColumn bc
        cdef AsciiColumn cc
        cdef DateColumn tc
        cdef CategoryColumn catc
        cdef DateTimeColumn dtc
        cdef TimeColumn tmc
        cdef int* ic_ptr
        cdef double* dc_ptr
        cdef unsigned char* bc_ptr

        if target_type == TYPE_INT:
            ic = IntColumn(safe_cap)
            ic.length = n
            ic_ptr = ic.data_c
            for i in range(n):
                s = self._get_str(i)
                ic_ptr[i] = int(float(s)) if len(s) > 0 else 0
            return ic

        elif target_type == TYPE_DOUBLE:
            dc = DoubleColumn(safe_cap)
            dc.length = n
            dc_ptr = dc.data_c
            for i in range(n):
                s = self._get_str(i)
                dc_ptr[i] = float(s) if len(s) > 0 else 0.0
            return dc

        elif target_type == TYPE_BOOL:
            bc = BoolColumn(safe_cap)
            bc.length = n
            bc_ptr = bc.data_c
            for i in range(n):
                s = self._get_str(i)
                if len(s) > 0 and (s == 'true' or s == 'True' or s == '1' or s == 'yes' or s == 'Yes'):
                    bc_ptr[i] = 1
                else:
                    bc_ptr[i] = 0
            return bc

        elif target_type == TYPE_CHAR256:
            cc = AsciiColumn(safe_cap)
            for i in range(n):
                cc.append(self._get_str(i))
            return cc

        elif target_type == TYPE_DATE:
            tc = DateColumn(safe_cap)
            for i in range(n):
                s = self._get_str(i)
                if len(s) > 0:
                    tc.append(s)
                else:
                    tc.append(0)
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
                if len(s) > 0:
                    dtc.append(s)
                else:
                    dtc.append('')
            return dtc

        elif target_type == TYPE_TIME:
            tmc = TimeColumn(safe_cap)
            for i in range(n):
                s = self._get_str(i)
                if len(s) > 0:
                    tmc.append(s)
                else:
                    tmc.append(0)
            return tmc

        elif target_type == TYPE_UCS2:
            return self.copy()

        cdef char err_buf[128]
        snprintf(err_buf, sizeof(err_buf), "Failed to cast UCS2 column to internal type id: %d", target_type)
        raise TypeError(err_buf.decode('ascii'))

    @cython.binding(False)
    def str_split(self, sep=None, int maxsplit=-1):
        cdef list res = []
        cdef Py_ssize_t i
        for i in range(self.length):
            s = self._get_str(i)
            if maxsplit >= 0:
                res.append(s.split(sep, maxsplit))
            else:
                res.append(s.split(sep))
        return res

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def str_getitem(self, key):
        cdef list res = []
        cdef Py_ssize_t i
        cdef str s
        if isinstance(key, int):
            idx = key
            for i in range(self.length):
                s = self._get_str(i)
                if idx < 0:
                    idx_abs = len(s) + idx
                else:
                    idx_abs = idx
                if 0 <= idx_abs < len(s):
                    res.append(s[idx_abs])
                else:
                    res.append('')
            col = UCS2Column(max(self.length, 1))
            for v in res:
                col.append(v)
            return col
        elif isinstance(key, slice):
            for i in range(self.length):
                s = self._get_str(i)
                res.append(s[key])
            col = UCS2Column(max(self.length, 1))
            for v in res:
                col.append(v)
            return col
        raise TypeError(f"Invalid index type: {type(key)}")


# ============================================================
# UCS2StringAccessor - pandas-style .str accessor for UCS2Column
# ============================================================

cdef class UCS2StringAccessor:
    """Pandas-style .str accessor for UCS2Column.
    All methods delegate to UCS2Column.
    实例由 UCS2Column.str property 懒加载缓存（字段声明见 ucs2column.pxd）。
    """

    def __cinit__(self):
        # 无参 __cinit__：允许工厂函数直接 __new__ 绕过参数打包
        pass

    # ==================== Search / Match ====================

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def contains(self, object pat):
        return self._col.contains(pat)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def startswith(self, prefix):
        return self._col.startswith(prefix)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def endswith(self, suffix):
        return self._col.endswith(suffix)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def find(self, str sub):
        return self._col.find(sub)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def rfind(self, str sub):
        return self._col.rfind(sub)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def index(self, str sub):
        return self._col.find(sub)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def rindex(self, str sub):
        return self._col.rindex(sub)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def count(self, str sub):
        return self._col.count(sub)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def match(self, str pattern):
        return self._col.match(pattern)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def fullmatch(self, str pattern):
        return self._col.fullmatch(pattern)

    # ==================== Extract / Findall ====================

    @cython.binding(False)
    def extract(self, str pattern, int group=1):
        return self._col.extract(pattern, group)

    @cython.binding(False)
    def extractall(self, str pattern, int group=1):
        return self._col.extractall(pattern, group)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def findall(self, str pattern):
        return self._col.findall(pattern)

    # ==================== Type Check ====================

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isalpha(self):
        return self._col.isalpha()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isnumeric(self):
        return self._col.isnumeric()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isalnum(self):
        return self._col.isalnum()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isdigit(self):
        return self._col.isdigit()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isdecimal(self):
        return self._col.isdecimal()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def islower(self):
        return self._col.islower()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isupper(self):
        return self._col.isupper()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isspace(self):
        return self._col.isspace()

    # ==================== Case Conversion ====================

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def upper(self):
        return self._col.upper()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def lower(self):
        return self._col.lower()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def title(self):
        return self._col.title()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def capitalize(self):
        return self._col.capitalize()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def swapcase(self):
        return self._col.swapcase()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def casefold(self):
        return self._col.casefold()

    # ==================== Strip / Pad ====================

    @cython.binding(False)
    def strip(self, chars=None):
        return self._col.strip(chars) if chars is not None else self._col.strip()

    @cython.binding(False)
    def lstrip(self, chars=None):
        return self._col.lstrip(chars) if chars is not None else self._col.lstrip()

    @cython.binding(False)
    def rstrip(self, chars=None):
        return self._col.rstrip(chars) if chars is not None else self._col.rstrip()

    @cython.binding(False)
    def center(self, int width, str fillchar=' '):
        return self._col.center(width, fillchar)

    @cython.binding(False)
    def ljust(self, int width, str fillchar=' '):
        return self._col.ljust(width, fillchar)

    @cython.binding(False)
    def rjust(self, int width, str fillchar=' '):
        return self._col.rjust(width, fillchar)

    @cython.binding(False)
    def pad(self, int width, str side='left', str fillchar=' '):
        return self._col.pad(width, side, fillchar)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def zfill(self, int width):
        return self._col.zfill(width)

    # ==================== Replace ====================

    @cython.binding(False)
    def replace(self, str old, str new, int count=-1):
        return self._col.replace(old, new)

    # ==================== Split ====================

    @cython.binding(False)
    def split(self, str sep=None, int maxsplit=-1):
        return self._col.str_split(sep, maxsplit)

    @cython.binding(False)
    def rsplit(self, str sep=None, int maxsplit=-1):
        return self._col.rsplit(sep, maxsplit)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def partition(self, str sep):
        return self._col.partition(sep)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def rpartition(self, str sep):
        return self._col.rpartition(sep)

    # ==================== Slicing / Access ====================

    @cython.binding(False)
    def get(self, int index, str default=''):
        return self._col.get(index, default)

    @cython.binding(False)
    def slice(self, int start=0, int stop=-1, int step=1):
        return self._col.slice(start, stop, step)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def len(self):
        return self._col.len()

    def __getitem__(self, key):
        return self._col.str_getitem(key)

    # ==================== Concat ====================

    @cython.binding(False)
    def cat(self, other, str sep=''):
        return self._col.cat(other, sep)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def prefix(self, str pre):
        return self._col.prefix(pre)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def suffix(self, str suf):
        return self._col.suffix(suf)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def repeat(self, int repeats):
        return self._col.repeat(repeats)


cdef inline UCS2StringAccessor _make_ucs2_str_accessor(UCS2Column col):
    cdef UCS2StringAccessor acc = <UCS2StringAccessor>UCS2StringAccessor.__new__(UCS2StringAccessor)
    acc._col = col
    return acc