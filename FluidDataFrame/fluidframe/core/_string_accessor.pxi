# ============================================================
#  String Accessor for Series
#  pandas-style .str accessor for string columns
# ============================================================

@cython.final
cdef class _StringAccessor:
    """Pandas-style .str accessor for Series.
    All methods delegate to the underlying string column.
    """

    def __cinit__(self, Series series):
        self._series = series

    # ==================== Search / Match ====================

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def contains(self, object pat):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).contains(pat)
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).contains(pat)
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def startswith(self, prefix):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).startswith(prefix)
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).startswith(prefix)
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def endswith(self, suffix):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).endswith(suffix)
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).endswith(suffix)
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def find(self, str sub):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).find(sub)
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).find(sub)
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def rfind(self, str sub):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).rfind(sub)
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).rfind(sub)
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def index(self, str sub):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).find(sub)
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).find(sub)
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def rindex(self, str sub):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).rindex(sub)
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).rindex(sub)
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def count(self, str sub):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).count(sub)
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).count(sub)
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def match(self, str pattern):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).match(pattern)
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).match(pattern)
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def fullmatch(self, str pattern):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).fullmatch(pattern)
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).fullmatch(pattern)
        raise TypeError("String accessor requires string column")

    # ==================== Extract / Findall ====================

    @cython.binding(False)
    def extract(self, str pattern, int group=1):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).extract(pattern)
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).extract(pattern, group)
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def extractall(self, str pattern):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).extractall(pattern)
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).extractall(pattern)
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def findall(self, str pattern):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).findall(pattern)
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).findall(pattern)
        raise TypeError("String accessor requires string column")

    # ==================== Type Check ====================

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isalpha(self):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).isalpha()
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).isalpha()
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isnumeric(self):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).isnumeric()
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).isnumeric()
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isalnum(self):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).isalnum()
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).isalnum()
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isdigit(self):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).isdigit()
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).isdigit()
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isdecimal(self):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).isdecimal()
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).isdecimal()
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def islower(self):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).islower()
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).islower()
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isupper(self):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).isupper()
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).isupper()
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isspace(self):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).isspace()
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).isspace()
        raise TypeError("String accessor requires string column")

    # ==================== Case Conversion ====================

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def upper(self):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).upper()
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).upper()
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def lower(self):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).lower()
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).lower()
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def title(self):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).title()
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).title()
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def capitalize(self):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).capitalize()
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).capitalize()
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def swapcase(self):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).swapcase()
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).swapcase()
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def casefold(self):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).casefold()
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).casefold()
        raise TypeError("String accessor requires string column")

    # ==================== Strip / Pad ====================

    @cython.binding(False)
    def strip(self, chars=None):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).strip(chars) if chars is not None else (<AsciiColumn>col).strip()
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).strip(chars) if chars is not None else (<UCS2Column>col).strip()
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    def lstrip(self, chars=None):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).lstrip(chars) if chars is not None else (<AsciiColumn>col).lstrip()
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).lstrip(chars) if chars is not None else (<UCS2Column>col).lstrip()
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    def rstrip(self, chars=None):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).rstrip(chars) if chars is not None else (<AsciiColumn>col).rstrip()
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).rstrip(chars) if chars is not None else (<UCS2Column>col).rstrip()
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    def center(self, int width, str fillchar=' '):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).center(width, fillchar)
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).center(width, fillchar)
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    def ljust(self, int width, str fillchar=' '):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).ljust(width, fillchar)
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).ljust(width, fillchar)
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    def rjust(self, int width, str fillchar=' '):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).rjust(width, fillchar)
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).rjust(width, fillchar)
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    def pad(self, int width, str side='left', str fillchar=' '):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).pad(width, side, fillchar)
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).pad(width, side, fillchar)
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def zfill(self, int width):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).zfill(width)
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).zfill(width)
        raise TypeError("String accessor requires string column")

    # ==================== Replace ====================

    @cython.binding(False)
    def replace(self, str old, str new, int count=-1):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).replace(old, new)
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).replace(old, new)
        raise TypeError("String accessor requires string column")

    # ==================== Split ====================

    @cython.binding(False)
    def split(self, str sep=None, int maxsplit=-1):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).str_split(sep, maxsplit)
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).str_split(sep, maxsplit)
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    def rsplit(self, str sep=None, int maxsplit=-1):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).rsplit(sep, maxsplit)
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).rsplit(sep, maxsplit)
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def partition(self, str sep):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).partition(sep)
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).partition(sep)
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def rpartition(self, str sep):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).rpartition(sep)
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).rpartition(sep)
        raise TypeError("String accessor requires string column")

    # ==================== Slicing / Access ====================

    @cython.binding(False)
    def get(self, int index, str default=''):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).get(index, default)
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).get(index, default)
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    def slice(self, int start=0, int stop=-1, int step=1):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).slice(start, stop, step)
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).slice(start, stop, step)
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def len(self):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).len()
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).len()
        raise TypeError("String accessor requires string column")

    def __getitem__(self, key):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).str_getitem(key)
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).str_getitem(key)
        raise TypeError("String accessor requires string column")

    # ==================== Concat ====================

    @cython.binding(False)
    def cat(self, other, str sep=''):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).cat(other, sep)
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).cat(other, sep)
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def prefix(self, str pre):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).prefix(pre)
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).prefix(pre)
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def suffix(self, str suf):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).suffix(suf)
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).suffix(suf)
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def repeat(self, int repeats):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_CHAR256:
            return (<AsciiColumn>col).repeat(repeats)
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).repeat(repeats)
        raise TypeError("String accessor requires string column")
