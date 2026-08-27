# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport cython
from libc.string cimport memcpy, memmove
from libc.stdlib cimport free

from fluidframe.core._base cimport BaseColumn
from fluidframe.core.intcolumn cimport IntColumn

from cpython.unicode cimport (
    PyUnicode_4BYTE_KIND, PyUnicode_FromKindAndData,
    PyUnicode_GET_LENGTH, PyUnicode_READ_CHAR,
    PyUnicode_KIND, PyUnicode_2BYTE_KIND, PyUnicode_DATA,
    Py_UCS2
)

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

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __cinit__(self, Py_ssize_t init_capacity=64):
        self.capacity = max(init_capacity, 1)
        self.length = 0

        self.char_capacity = max(self.capacity * 16, 1024)
        self.char_length = 0

        self.offsets_c = <Py_ssize_t*>aligned_malloc((self.capacity + 1) * sizeof(Py_ssize_t))
        self.chars_c = <Py_UCS2*>aligned_malloc(self.char_capacity * sizeof(Py_UCS2))

        if not self.offsets_c or not self.chars_c:
            raise MemoryError()

        self.offsets_c[0] = 0

    def __dealloc__(self):
        if self.offsets_c is not NULL: free(self.offsets_c)
        if self.chars_c is not NULL: free(self.chars_c)

    cdef void _reserve_rows(self, Py_ssize_t new_cap) except *:
        if new_cap > self.capacity:
            self.offsets_c = <Py_ssize_t*>aligned_realloc(self.offsets_c, (new_cap + 1) * sizeof(Py_ssize_t))
            if not self.offsets_c: raise MemoryError()
            self.capacity = new_cap

    cdef void _reserve_chars(self, Py_ssize_t new_char_cap) except *:
        if new_char_cap > self.char_capacity:
            self.chars_c = <Py_UCS2*>aligned_realloc(self.chars_c, new_char_cap * sizeof(Py_UCS2))
            if not self.chars_c: raise MemoryError()
            self.char_capacity = new_char_cap

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def append(self, str val):
        cdef Py_ssize_t str_len = PyUnicode_GET_LENGTH(val) if val else 0

        if self.length >= self.capacity:
            self._reserve_rows(self.capacity * 2)

        if self.char_length + str_len > self.char_capacity:
            self._reserve_chars(max(self.char_capacity * 2, self.char_length + str_len + 1024))

        if str_len > 0:
            if PyUnicode_KIND(val) == PyUnicode_2BYTE_KIND:
                memcpy(self.chars_c + self.char_length, PyUnicode_DATA(val), str_len * sizeof(Py_UCS2))
            else:
                self._fallback_copy_chars(val, str_len)

        self.char_length += str_len
        self.length += 1
        self.offsets_c[self.length] = self.char_length

    cdef inline void _fallback_copy_chars(self, str val, Py_ssize_t str_len):
        cdef Py_ssize_t i
        cdef Py_UCS4 char_code
        for i in range(str_len):
            char_code = PyUnicode_READ_CHAR(val, i)
            if char_code > 0xFFFF:
                self.chars_c[self.char_length + i] = 0x003F
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
        memcpy(self.chars_c + self.char_length, other.chars_c, other.char_length * sizeof(Py_UCS2))

        cdef Py_ssize_t base_offset = self.char_length
        for i in range(other.length):
            self.offsets_c[self.length + 1 + i] = base_offset + other.offsets_c[i + 1]

        self.length += other.length
        self.char_length += other.char_length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _reorder(self, list indices):
        cdef Py_ssize_t n = self.length
        cdef Py_ssize_t* new_offsets = <Py_ssize_t*>aligned_malloc((self.capacity + 1) * sizeof(Py_ssize_t))
        cdef Py_UCS2* new_chars = <Py_UCS2*>aligned_malloc(self.char_capacity * sizeof(Py_UCS2))
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
