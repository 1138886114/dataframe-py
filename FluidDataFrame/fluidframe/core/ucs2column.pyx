# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport cython
from libc.string cimport memcpy, memmove
from libc.stdlib cimport free

from fluidframe.core._base cimport BaseColumn, aligned_malloc, aligned_realloc, TYPE_UCS2, TYPE_INT
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
        self.dtype = TYPE_UCS2
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

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def append(self, str val not None):
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
        cdef Py_ssize_t* new_offsets = <Py_ssize_t*>aligned_malloc((n + 1) * sizeof(Py_ssize_t))
        if not new_offsets: raise MemoryError()
        cdef Py_UCS2* new_chars = <Py_UCS2*>aligned_malloc(self.char_capacity * sizeof(Py_UCS2))
        if not new_chars:
            free(new_offsets)
            raise MemoryError()

        new_offsets[0] = 0
        cdef Py_ssize_t current_char_len = 0
        cdef Py_ssize_t idx, start, end, str_len
        cdef Py_ssize_t i

        try:
            for i in range(n):
                idx = <Py_ssize_t>indices[i]
                if idx < 0 or idx >= n:
                    raise IndexError("Index out of bounds")
                start = self.offsets_c[idx]
                end = self.offsets_c[idx + 1]
                str_len = end - start

                if str_len > 0:
                    memcpy(new_chars + current_char_len, self.chars_c + start, str_len * sizeof(Py_UCS2))

                current_char_len += str_len
                new_offsets[i + 1] = current_char_len
        except Exception:
            free(new_offsets)
            free(new_chars)
            raise

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
        cdef Py_ssize_t i
        cdef str s
        for i in range(self.length):
            s = self[i].replace(old, new)
            self._set_item(i, s)
        return self

    cdef inline void _set_item(self, Py_ssize_t idx, str val):
        """Replace string at given index in-place."""
        cdef Py_ssize_t old_start = self.offsets_c[idx]
        cdef Py_ssize_t old_len = self.offsets_c[idx + 1] - old_start
        cdef Py_ssize_t new_len = len(val)
        cdef Py_ssize_t ci, diff, oi

        if new_len > old_len:
            self._reserve_chars(self.char_capacity + (new_len - old_len))
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
            if PyUnicode_KIND(val) == PyUnicode_2BYTE_KIND:
                memcpy(self.chars_c + old_start, PyUnicode_DATA(val), new_len * sizeof(Py_UCS2))
            else:
                for ci in range(new_len):
                    self.chars_c[old_start + ci] = <Py_UCS2>PyUnicode_READ_CHAR(val, ci)

        self.char_length += (new_len - old_len)
        diff = new_len - old_len
        for oi in range(idx + 1, self.length + 1):
            self.offsets_c[oi] += diff

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
    def count(self, str substring):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        cdef str s
        cdef Py_ssize_t sub_len = len(substring)
        cdef Py_ssize_t cnt, pos
        if sub_len == 0:
            for i in range(self.length):
                res.data_c[i] = <int>(len(self[i]) + 1)
            return res
        for i in range(self.length):
            s = self[i]
            cnt = 0
            pos = 0
            while True:
                pos = s.find(substring, pos)
                if pos == -1:
                    break
                cnt += 1
                pos += sub_len
            res.data_c[i] = <int>cnt
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

    def _cmp_op(self, val, int op):
        cdef Py_ssize_t i
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef str s, v
        cdef int cmp_val
        if isinstance(val, UCS2Column_SoA):
            other = <UCS2Column_SoA>val
            if other.length != self.length:
                raise ValueError(f"Length mismatch: {self.length} vs {other.length}")
            for i in range(self.length):
                s = self[i]
                v = other[i]
                if op == 0:   cmp_val = 1 if s < v else 0
                elif op == 1: cmp_val = 1 if s <= v else 0
                elif op == 2: cmp_val = 1 if s > v else 0
                elif op == 3: cmp_val = 1 if s >= v else 0
                elif op == 4: cmp_val = 1 if s == v else 0
                elif op == 5: cmp_val = 1 if s != v else 0
                res.data_c[i] = cmp_val
        else:
            v = <str>val
            for i in range(self.length):
                s = self[i]
                if op == 0:   cmp_val = 1 if s < v else 0
                elif op == 1: cmp_val = 1 if s <= v else 0
                elif op == 2: cmp_val = 1 if s > v else 0
                elif op == 3: cmp_val = 1 if s >= v else 0
                elif op == 4: cmp_val = 1 if s == v else 0
                elif op == 5: cmp_val = 1 if s != v else 0
                res.data_c[i] = cmp_val
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
            res.data_c[i] = 1 if self[i] != '' and other.data_c[i] else 0
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
            res.data_c[i] = 1 if self[i] != '' or other.data_c[i] else 0
        return res

    def __invert__(self):
        cdef IntColumn res = IntColumn(self.capacity)
        res.length = self.length
        cdef Py_ssize_t i
        for i in range(self.length):
            res.data_c[i] = 0 if self[i] != '' else 1
        return res

    def __str__(self):
        cdef list items = []
        cdef Py_ssize_t i
        for i in range(self.length):
            items.append(repr(self[i]))
        return f"UCS2Column([{', '.join(items)}])"

    def __bool__(self):
        return self.length > 0

    def __len__(self):
        return self.length


    cdef UCS2Column_SoA take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=False):
        cdef Py_ssize_t* new_offsets = <Py_ssize_t*>aligned_malloc((n + 1) * sizeof(Py_ssize_t))
        cdef Py_UCS2* new_chars = <Py_UCS2*>aligned_malloc(self.char_capacity * sizeof(Py_UCS2))
        if not new_offsets or not new_chars: raise MemoryError()
        new_offsets[0] = 0
        cdef Py_ssize_t current_char_len = 0
        cdef Py_ssize_t i, idx, start, end, str_len
        with nogil:
            for i in range(n):
                idx = indices[i]
                start = self.offsets_c[idx]
                end = self.offsets_c[idx + 1]
                str_len = end - start
                if str_len > 0:
                    memcpy(new_chars + current_char_len, self.chars_c + start, str_len * sizeof(Py_UCS2))
                current_char_len += str_len
                new_offsets[i + 1] = current_char_len
        if inplace:
            free(self.offsets_c)
            free(self.chars_c)
            self.offsets_c = new_offsets
            self.chars_c = new_chars
            self.char_length = current_char_len
            self.length = n
            return self
        else:
            new_col = UCS2Column_SoA(self.capacity)
            free(new_col.offsets_c)
            free(new_col.chars_c)
            new_col.offsets_c = new_offsets
            new_col.chars_c = new_chars
            new_col.char_length = current_char_len
            new_col.char_capacity = self.char_capacity
            new_col.length = n
            return new_col


    cpdef void drop_row(self, Py_ssize_t row_idx, Py_ssize_t rem_len):
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
