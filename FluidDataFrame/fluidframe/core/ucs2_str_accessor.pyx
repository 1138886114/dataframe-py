# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False

cimport cython

from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.ucs2column cimport UCS2Column_SoA


cdef class StringAccessor_UCS2:
    """Zero-overhead .str accessor for UCS2Column_SoA.

    All methods are cpdef — callable at C speed from Cython,
    and from Python with minimal overhead.
    """

    cdef UCS2Column_SoA _col

    def __cinit__(self, UCS2Column_SoA col):
        self._col = col

    # ==================== Internal helpers ====================

    cdef inline IntColumn _bool_mask(self):
        cdef IntColumn res = IntColumn(max(self._col.capacity, 1))
        res.length = self._col.length
        return res

    cdef inline UCS2Column_SoA _new_col(self):
        cdef UCS2Column_SoA res = UCS2Column_SoA(max(self._col.capacity, 1))
        return res

    cdef inline UCS2Column_SoA _fill_from_method(self, object method):
        cdef UCS2Column_SoA res = self._new_col()
        cdef Py_ssize_t i
        for i in range(self._col.length):
            res.append(method(self._col[i]))
        return res

    # ==================== Search / Match ====================

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef IntColumn contains(self, object pat):
        return self._col.contains(pat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef IntColumn startswith(self, prefix):
        return self._col.startswith(prefix)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef IntColumn endswith(self, suffix):
        return self._col.endswith(suffix)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef IntColumn find(self, str sub):
        return self._col.find(sub)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef IntColumn rfind(self, str sub):
        return self._col.rfind(sub)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef IntColumn index(self, str sub):
        return self._col.find(sub)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef IntColumn rindex(self, str sub):
        return self._col.rindex(sub)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef IntColumn count(self, str sub):
        return self._col.count(sub)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef IntColumn match(self, str pattern):
        return self._col.match(pattern)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef IntColumn fullmatch(self, str pattern):
        return self._col.fullmatch(pattern)

    # ==================== Extract / Findall ====================

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef UCS2Column_SoA extract(self, str pattern, int group=1):
        return self._col.extract(pattern, group)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef extractall(self, str pattern, int group=1):
        return self._col.extractall(pattern, group)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef findall(self, str pattern):
        return self._col.findall(pattern)

    # ==================== Type Check ====================

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef IntColumn isalpha(self):
        return self._col.isalpha()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef IntColumn isnumeric(self):
        return self._col.isnumeric()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef IntColumn isalnum(self):
        return self._col.isalnum()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef IntColumn isdigit(self):
        return self._col.isdigit()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef IntColumn isdecimal(self):
        return self._col.isdecimal()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef IntColumn islower(self):
        return self._col.islower()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef IntColumn isupper(self):
        return self._col.isupper()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef IntColumn isspace(self):
        return self._col.isspace()

    # ==================== Case Conversion ====================

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef UCS2Column_SoA upper(self):
        return self._fill_from_method(str.upper)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef UCS2Column_SoA lower(self):
        return self._fill_from_method(str.lower)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef UCS2Column_SoA title(self):
        return self._fill_from_method(str.title)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef UCS2Column_SoA capitalize(self):
        return self._fill_from_method(str.capitalize)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef UCS2Column_SoA swapcase(self):
        return self._fill_from_method(str.swapcase)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef UCS2Column_SoA casefold(self):
        return self._fill_from_method(str.casefold)

    # ==================== Strip / Pad ====================

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef UCS2Column_SoA strip(self, chars=None):
        cdef UCS2Column_SoA res = self._new_col()
        cdef Py_ssize_t i
        cdef str s
        for i in range(self._col.length):
            s = self._col[i]
            if chars is not None:
                res.append(s.strip(chars))
            else:
                res.append(s.strip())
        return res

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef UCS2Column_SoA lstrip(self, chars=None):
        cdef UCS2Column_SoA res = self._new_col()
        cdef Py_ssize_t i
        cdef str s
        for i in range(self._col.length):
            s = self._col[i]
            if chars is not None:
                res.append(s.lstrip(chars))
            else:
                res.append(s.lstrip())
        return res

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef UCS2Column_SoA rstrip(self, chars=None):
        cdef UCS2Column_SoA res = self._new_col()
        cdef Py_ssize_t i
        cdef str s
        for i in range(self._col.length):
            s = self._col[i]
            if chars is not None:
                res.append(s.rstrip(chars))
            else:
                res.append(s.rstrip())
        return res

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef UCS2Column_SoA center(self, int width, str fillchar=' '):
        return self._col.center(width, fillchar)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef UCS2Column_SoA ljust(self, int width, str fillchar=' '):
        return self._col.ljust(width, fillchar)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef UCS2Column_SoA rjust(self, int width, str fillchar=' '):
        return self._col.rjust(width, fillchar)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef UCS2Column_SoA pad(self, int width, str side='left', str fillchar=' '):
        return self._col.pad(width, side, fillchar)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef UCS2Column_SoA zfill(self, int width):
        return self._col.zfill(width)

    # ==================== Replace ====================

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef UCS2Column_SoA replace(self, str old, str new, int count=-1):
        return self._col.replace(old, new)

    # ==================== Split ====================

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef split(self, str sep=None, int maxsplit=-1):
        return self._col.str_split(sep, maxsplit)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef rsplit(self, str sep=None, int maxsplit=-1):
        return self._col.rsplit(sep, maxsplit)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef partition(self, str sep):
        return self._col.partition(sep)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef rpartition(self, str sep):
        return self._col.rpartition(sep)

    # ==================== Slicing / Access ====================

    cpdef UCS2Column_SoA get(self, int index, str default=''):
        return self._col.get(index, default)

    cpdef UCS2Column_SoA slice(self, int start=0, int stop=-1, int step=1):
        return self._col.slice(start, stop, step)

    cpdef IntColumn len(self):
        return self._col.len()

    def __getitem__(self, key):
        return self._col.str_getitem(key)

    # ==================== Concat ====================

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef UCS2Column_SoA cat(self, other, str sep=''):
        return self._col.cat(other, sep)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef UCS2Column_SoA prefix(self, str pre):
        return self._col.prefix(pre)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef UCS2Column_SoA suffix(self, str suf):
        return self._col.suffix(suf)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef UCS2Column_SoA repeat(self, int repeats):
        return self._col.repeat(repeats)
