# cython: language_level=3

cimport cython

from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.asciicolumn cimport AsciiColumn


cdef class StringAccessor_Char256:
    """Pandas-style .str accessor for AsciiColumn.
    All methods delegate to AsciiColumn.
    """

    cdef AsciiColumn _col

    def __cinit__(self, AsciiColumn col):
        self._col = col

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
