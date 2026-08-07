# %%cython --annotate
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memcpy, memmove, memset, memcmp
from libc.math cimport sqrt
import cython
from cpython.long cimport PyLong_FromSsize_t
from cpython.list cimport PyList_New, PyList_SetItem
import inspect
import re


cdef extern from "Python.h":
    bint Py_UNICODE_ISALNUM(Py_UCS4 ch)
    bint Py_UNICODE_ISALPHA(Py_UCS4 ch)
    bint Py_UNICODE_ISDECIMAL(Py_UCS4 ch)
    bint Py_UNICODE_ISDIGIT(Py_UCS4 ch)
    bint Py_UNICODE_ISLOWER(Py_UCS4 ch)
    bint Py_UNICODE_ISNUMERIC(Py_UCS4 ch)
    bint Py_UNICODE_ISSPACE(Py_UCS4 ch)
    bint Py_UNICODE_ISTITLE(Py_UCS4 ch)
    bint Py_UNICODE_ISUPPER(Py_UCS4 ch)
    Py_UCS4 Py_UNICODE_TOLOWER(Py_UCS4 ch)
    Py_UCS4 Py_UNICODE_TOUPPER(Py_UCS4 ch)

from cpython.unicode cimport (
    PyUnicode_FromKindAndData, PyUnicode_GET_LENGTH, PyUnicode_READ_CHAR,
    PyUnicode_KIND, PyUnicode_1BYTE_KIND, PyUnicode_2BYTE_KIND,
    PyUnicode_DATA, Py_UCS2
)

# ============================================
# 1. 前向声明，解决访问器与 NDframe 互相引用的问题
# ============================================
cdef class NDframe
cdef class UCS2Column_SoA
cdef tuple HEADERS = ("idx", "x0", "y0", "x1", "y1", "width", "height", "page", "flags", "text", "text_sep")
cdef tuple HEADER_LENS = (3, 2, 2, 2, 2, 5, 6, 4, 5, 4, 8)


cdef enum FieldID:
    FIELD_idx = 0
    FIELD_x0 = 1
    FIELD_y0 = 2
    FIELD_x1 = 3
    FIELD_y1 = 4
    FIELD_width = 5
    FIELD_height = 6
    FIELD_page = 7
    FIELD_flags = 8


cdef enum FieldTXT:
    FIELD_txt = 10
    FIELD_sep = 11


cdef enum FilterOp:
    OP_GREATER = 0
    OP_LESS = 1
    OP_EQUAL = 2
    OP_BETWEEN = 3

cdef str _field_id_to_name(int field_id):
    if field_id == FIELD_x0: return "x0"
    if field_id == FIELD_y0: return "y0"
    if field_id == FIELD_x1: return "x1"
    if field_id == FIELD_y1: return "y1"
    if field_id == FIELD_width: return "width"
    if field_id == FIELD_height: return "height"
    if field_id == FIELD_idx: return "idx"
    if field_id == FIELD_page: return "page"
    if field_id == FIELD_flags: return "flags"
    if field_id == FIELD_txt: return "text"
    if field_id == FIELD_sep: return "text_sep"
    return "?"


cdef enum StrCheckMode:
    CHK_ISALNUM = 0
    CHK_ISALPHA = 1
    CHK_ISDIGIT = 2
    CHK_ISSPACE = 3
    CHK_ISLOWER = 4
    CHK_ISUPPER = 5
    CHK_ISTITLE = 6
    CHK_ISNUMERIC = 7
    CHK_ISDECIMAL = 8
    CHK_ISCHINESE = 9
    CHK_IS_NUMERIC_PUNCT = 10

cdef enum StrTransformMode:
    TRM_STRIP = 0
    TRM_LSTRIP = 1
    TRM_RSTRIP = 2
    TRM_CAPITALIZE = 3
    TRM_SWAPCASE = 4


cdef inline int _get_field_val(RowC* row, int field) noexcept:
    return (&row.idx)[field]


cdef inline void _add_field_value(RowC* row, int field, int offset) noexcept:
    (&row.idx)[field] += offset


cdef inline object _ensure_regex(object pattern):
    """
    正则参数容错处理宏：
    1. 如果是字符串，自动编译为正则对象。
    2. 如果既不是字符串，也没有 search 方法，抛出明确的 TypeError。
    """
    if isinstance(pattern, str):
        return re.compile(pattern)

    # Python 鸭子类型检测：只要有 search 方法即可（兼容 re, regex, re2 等不同引擎）
    if not hasattr(pattern, 'search'):
        raise TypeError(f"Expected string or compiled regex pattern, got {type(pattern).__name__}")

    return pattern


# ============================================
# 模块级极速匹配 / 比较 / 拷贝帮助函数
# 入口净化（Invariant）：只要保证 UCS2Column_SoA 内部存储的数据 100% 已经是规范的 UCS2 数组。
# ============================================
cdef inline Py_UCS2* _str_to_ucs2_buf(str s, Py_ssize_t* out_len) except NULL:
    """
    利用 PEP 393 内存布局实现 C 级别极速转换：
    1. 2-byte 格式 (中文/BMP)：直接 memcpy，CPU 指令级拷贝！
    2. 1-byte 格式 (ASCII/Latin1)：纯 uint8 -> uint16 指针拓宽，0 宏开销！
    3. 4-byte 格式 (Emoji/生僻字)：降级清洗为 '?'
    """
    cdef Py_ssize_t n = PyUnicode_GET_LENGTH(s) if s is not None else 0
    out_len[0] = n

    cdef Py_UCS2* buf = <Py_UCS2*>malloc((n + 1) * sizeof(Py_UCS2))
    if buf == NULL:
        raise MemoryError()

    if n == 0:
        buf[0] = 0
        return buf

    cdef int kind = PyUnicode_KIND(s)
    cdef void* data = PyUnicode_DATA(s)
    cdef Py_ssize_t i
    cdef Py_UCS4 cp

    # 【分支 1】：最常见的 2-byte 格式 (中文字符串) -> 内存直接 memcpy！
    if kind == PyUnicode_2BYTE_KIND:
        memcpy(buf, data, n * sizeof(Py_UCS2))

    # 【分支 2】：1-byte 格式 (纯英文/数字) -> 无函数调用的指针快速拓宽
    elif kind == PyUnicode_1BYTE_KIND:
        for i in range(n):
            buf[i] = <Py_UCS2>(<unsigned char*>data)[i]

    # 【分支 3】：4-byte 格式 (含有 Emoji/生僻字) -> 降级替换
    else:
        for i in range(n):
            cp = PyUnicode_READ_CHAR(s, i)
            buf[i] = <Py_UCS2>cp if cp <= 0xFFFF else 0x003F

    buf[n] = 0
    return buf


cdef inline bint _substr_contains(Py_UCS2* chars, Py_ssize_t start, Py_ssize_t end,
                                  Py_UCS2* sub_buf, Py_ssize_t sub_len) noexcept:
    """纯 C 内存滑动窗口子串匹配（O(N*M)，无任何 Python 对象创建）"""
    if sub_len == 0:
        return True
    cdef Py_ssize_t str_len = end - start
    if str_len < sub_len:
        return False
    cdef Py_ssize_t j, k
    cdef bint ok
    for j in range(start, end - sub_len + 1):
        if chars[j] != sub_buf[0]:
            continue
        ok = True
        for k in range(1, sub_len):
            if chars[j + k] != sub_buf[k]:
                ok = False
                break
        if ok:
            return True
    return False


cdef inline bint _int_cmp_pass(Py_ssize_t val, int op, Py_ssize_t arg1, Py_ssize_t arg2) noexcept:
    """统一整数比较谓词：把 OP_GREATER/LESS/EQUAL/BETWEEN 的分支收敛到一个函数"""
    if op == OP_GREATER:
        return val > arg1
    elif op == OP_LESS:
        return val < arg1
    elif op == OP_EQUAL:
        return val == arg1
    elif op == OP_BETWEEN:
        return val > arg1 and val < arg2
    return False


cdef inline bint _is_in_chars(Py_UCS2 c, Py_UCS2* chars_buf, Py_ssize_t chars_len) noexcept:
    cdef Py_ssize_t k
    for k in range(chars_len):
        if c == chars_buf[k]:
            return True
    return False


####################################################
#       UCS2 存储引擎
#================================================
# 通过统计业务数据设置如下
# 行数初始容量设为 64
# 初始字符池子：1024
####################################################
@cython.final
@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class UCS2Column_SoA:
    cdef Py_UCS2* chars_c        # 连续字符池
    cdef Py_ssize_t* offsets_c   # 偏移量数组 (长度为 length + 1)

    cdef Py_ssize_t capacity     # 行容量（含已用 length 行）
    cdef Py_ssize_t length       # 当前行数
    cdef Py_ssize_t char_capacity
    cdef Py_ssize_t char_length

    def __cinit__(self, Py_ssize_t init_capacity=64):
        self.capacity = init_capacity if init_capacity > 0 else 1
        self.length = 0

        self.char_capacity = self.capacity * 16
        if self.char_capacity < 1024:
            self.char_capacity = 1024
        self.char_length = 0

        self.offsets_c = <Py_ssize_t*>malloc((self.capacity + 1) * sizeof(Py_ssize_t))
        self.chars_c = <Py_UCS2*>malloc(self.char_capacity * sizeof(Py_UCS2))

        if not self.offsets_c or not self.chars_c:
            if self.offsets_c: free(self.offsets_c)
            if self.chars_c: free(self.chars_c)
            self.offsets_c = NULL
            self.chars_c = NULL
            raise MemoryError()

        self.offsets_c[0] = 0

    def __dealloc__(self):
        if self.offsets_c is not NULL: free(self.offsets_c)
        if self.chars_c is not NULL: free(self.chars_c)

    # ==========================
    # 内部基础内存调度宏
    # ==========================
    cdef inline void _reserve_rows(self, Py_ssize_t new_cap) except *:
        cdef Py_ssize_t* new_ptr
        if new_cap > self.capacity:
            new_ptr = <Py_ssize_t*>realloc(self.offsets_c, (new_cap + 1) * sizeof(Py_ssize_t))
            if not new_ptr: raise MemoryError()
            self.offsets_c = new_ptr
            self.capacity = new_cap

    cdef inline void _reserve_chars(self, Py_ssize_t new_char_cap) except *:
        cdef Py_UCS2* new_ptr
        if new_char_cap > self.char_capacity:
            new_ptr = <Py_UCS2*>realloc(self.chars_c, new_char_cap * sizeof(Py_UCS2))
            if not new_ptr: raise MemoryError()
            self.chars_c = new_ptr
            self.char_capacity = new_char_cap

    # ==========================
    # 高频数据交互 API (全部改为 cdef)
    # ==========================
    cdef inline str _get_str(self, Py_ssize_t idx):
        cdef Py_ssize_t start = self.offsets_c[idx]
        cdef Py_ssize_t str_len = self.offsets_c[idx + 1] - start
        if str_len == 0:
            return ""
        return PyUnicode_FromKindAndData(2, self.chars_c + start, str_len)

    cdef void append(self, str val) except *:
        """纯 C 方法，极速追加"""
        cdef Py_ssize_t str_len = PyUnicode_GET_LENGTH(val) if val is not None else 0

        if self.length >= self.capacity:
            self._reserve_rows(self.capacity * 2)

        if self.char_length + str_len > self.char_capacity:
            self._reserve_chars(max(self.char_capacity * 2, self.char_length + str_len + 1024))
        cdef int kind
        cdef void* data
        if str_len > 0 and val is not None:
            kind = PyUnicode_KIND(val)
            data = PyUnicode_DATA(val)
            if kind == PyUnicode_2BYTE_KIND:
                memcpy(self.chars_c + self.char_length, data, str_len * sizeof(Py_UCS2))
            elif kind == PyUnicode_1BYTE_KIND:
                for i in range(str_len):
                    self.chars_c[self.char_length + i] = <Py_UCS2>(<unsigned char*>data)[i]
            else:
                for i in range(str_len):
                    cp = PyUnicode_READ_CHAR(val, i)
                    self.chars_c[self.char_length + i] = <Py_UCS2>cp if cp <= 0xFFFF else 0x003F

        self.char_length += str_len
        self.length += 1
        self.offsets_c[self.length] = self.char_length

    cdef UCS2Column_SoA copy(self):
        cdef UCS2Column_SoA new_col = UCS2Column_SoA(self.capacity)
        new_col._reserve_chars(self.char_capacity)
        new_col.length = self.length
        new_col.char_length = self.char_length
        if self.length > 0:
            memcpy(new_col.offsets_c, self.offsets_c, (self.length + 1) * sizeof(Py_ssize_t))
        if self.char_length > 0:
            memcpy(new_col.chars_c, self.chars_c, self.char_length * sizeof(Py_UCS2))
        return new_col

    cdef UCS2Column_SoA _slice(self, Py_ssize_t start, Py_ssize_t stop):
        cdef Py_ssize_t new_len = stop - start
        if new_len <= 0:
            return UCS2Column_SoA(16)
        cdef UCS2Column_SoA new_col = UCS2Column_SoA(new_len)
        cdef Py_ssize_t char_start = self.offsets_c[start]
        cdef Py_ssize_t char_stop = self.offsets_c[stop]
        cdef Py_ssize_t char_len = char_stop - char_start

        new_col._reserve_chars(char_len + 1)
        if char_len > 0:
            memcpy(new_col.chars_c, self.chars_c + char_start, char_len * sizeof(Py_UCS2))

        for i in range(new_len + 1):
            new_col.offsets_c[i] = self.offsets_c[start + i] - char_start

        new_col.length = new_len
        new_col.char_length = char_len
        return new_col

    cdef void concat(self, UCS2Column_SoA other) except *:
        if other.length == 0: return
        self._reserve_rows(self.length + other.length)
        self._reserve_chars(self.char_length + other.char_length)

        if other.char_length > 0:
            memcpy(self.chars_c + self.char_length, other.chars_c, other.char_length * sizeof(Py_UCS2))

        cdef Py_ssize_t base_offset = self.char_length
        for i in range(other.length):
            self.offsets_c[self.length + 1 + i] = base_offset + other.offsets_c[i + 1]

        self.length += other.length
        self.char_length += other.char_length

    cdef void _reorder(self, Py_ssize_t* indices) except *:
        cdef Py_ssize_t n = self.length
        if n == 0: return

        cdef Py_ssize_t* new_offsets = <Py_ssize_t*>malloc((self.capacity + 1) * sizeof(Py_ssize_t))
        cdef Py_UCS2* new_chars = <Py_UCS2*>malloc(max(self.char_length, 1024) * sizeof(Py_UCS2))

        if not new_offsets or not new_chars:
            if new_offsets: free(new_offsets)
            if new_chars: free(new_chars)
            raise MemoryError()

        new_offsets[0] = 0
        cdef Py_ssize_t current_char_len = 0
        cdef Py_ssize_t idx, start, end, str_len

        for i in range(n):
            idx = indices[i]
            start = self.offsets_c[idx]
            str_len = self.offsets_c[idx + 1] - start

            if str_len > 0:
                memcpy(new_chars + current_char_len, self.chars_c + start, str_len * sizeof(Py_UCS2))

            current_char_len += str_len
            new_offsets[i + 1] = current_char_len

        free(self.offsets_c)
        free(self.chars_c)
        self.offsets_c = new_offsets
        self.chars_c = new_chars
        self.char_length = current_char_len
        self.char_capacity = max(self.char_length, 1024)

    cdef inline void pop_back(self) noexcept:
        """优化点: O(1) 操作，不创建 Python 字符串，仅回退指针"""
        if self.length > 0:
            self.length -= 1
            self.char_length = self.offsets_c[self.length]

    cdef inline void _remove_at(self, Py_ssize_t idx) noexcept:
        cdef Py_ssize_t start = self.offsets_c[idx]
        cdef Py_ssize_t end = self.offsets_c[idx + 1]
        cdef Py_ssize_t c_len = end - start
        cdef Py_ssize_t tail = self.char_length - end
        if tail > 0:
            memmove(self.chars_c + start, self.chars_c + end, tail * sizeof(Py_UCS2))
        self.char_length -= c_len
        for i in range(idx, self.length):
            self.offsets_c[i] = self.offsets_c[i + 1] - c_len
        self.length -= 1

    cdef list _is_mode(self, int mode):
        cdef list res = PyList_New(self.length)
        cdef Py_ssize_t i, j, start, end
        cdef bint match
        cdef bint (*pred)(Py_UCS2) noexcept

        if mode == CHK_ISALNUM: pred = _pred_isalnum
        elif mode == CHK_ISALPHA: pred = _pred_isalpha
        elif mode == CHK_ISDIGIT: pred = _pred_isdigit
        elif mode == CHK_ISSPACE: pred = _pred_isspace
        elif mode == CHK_ISLOWER: pred = _pred_islower
        elif mode == CHK_ISUPPER: pred = _pred_isupper
        elif mode == CHK_ISTITLE: pred = _pred_istitle
        elif mode == CHK_ISNUMERIC: pred = _pred_isnumeric
        elif mode == CHK_ISDECIMAL: pred = _pred_isdecimal
        elif mode == CHK_ISCHINESE: pred = _pred_is_chinese
        else: pred = _pred_is_numeric_punct

        for i in range(self.length):
            start = self.offsets_c[i]; end = self.offsets_c[i+1]
            if start == end: match = False
            else:
                match = True
                for j in range(start, end):
                    if not pred(self.chars_c[j]): match = False; break
            PyList_SetItem(res, i, match)
        return res


# ==========================================
# 2. 访问器 (Accessor) 定义
# ==========================================
@cython.final
@cython.binding(False)
cdef class StringAccessor:
    """字符串字段访问器：提供类似 pandas 的 df.text.xxx API"""
    cdef NDframe df
    cdef int field

    @cython.always_allow_keywords(False)
    def __cinit__(self, NDframe df, int field):
        self.df = df
        self.field = field

    @property
    def field_name(self):
        return _field_id_to_name(self.field)

    @cython.always_allow_keywords(False)
    def __iter__(self):
        cdef NDframe df = self.df
        cdef UCS2Column_SoA src = df.text_col if self.field == FIELD_txt else df.text_sep_col
        cdef Py_ssize_t i
        for i in range(src.length):
            yield src._get_str(i)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __repr__(self):
        cdef NDframe df = self.df
        cdef Py_ssize_t n = df.length
        cdef Py_ssize_t preview_n = n if n < 5 else 5
        cdef list preview = [None] * preview_n
        cdef Py_ssize_t i
        cdef str v
        cdef UCS2Column_SoA src = df.text_col if self.field == FIELD_txt else df.text_sep_col

        for i in range(preview_n):
            v = src._get_str(i)
            preview[i] = repr(v if len(v) <= 30 else v[:30] + "...")
        if n > 5:
            preview.append("...")

        cdef str fname = self.field_name
        return f"{fname}[{', '.join(preview)}] (n={n})"

    def slice(self, start=None, stop=None, step=None):
        return self.df._str_slice(self.field, start, stop, step)

    @cython.always_allow_keywords(False)
    def count(self, str sub):
        return self.df._str_count(self.field, sub)

    @cython.always_allow_keywords(False)
    def startswith(self, str prefix):
        return self.df._df_str_starts_ends(self.field, prefix, True)

    @cython.always_allow_keywords(False)
    def endswith(self, str suffix):
        return self.df._df_str_starts_ends(self.field, suffix, False)

    def strip(self, str chars=None):
        return self.df._str_transform(self.field, TRM_STRIP, chars)

    def rstrip(self, str chars=None):
        return self.df._str_transform(self.field, TRM_RSTRIP, chars)

    def lstrip(self, str chars=None):
        return self.df._str_transform(self.field, TRM_LSTRIP, chars)

    @cython.always_allow_keywords(False)
    def capitalize(self):
        return self.df._str_transform(self.field, TRM_CAPITALIZE)

    @cython.always_allow_keywords(False)
    def swapcase(self):
        return self.df._str_transform(self.field, TRM_SWAPCASE)

    @cython.always_allow_keywords(False)
    def isalnum(self): return self.df._df_str_is(self.field, CHK_ISALNUM)

    @cython.always_allow_keywords(False)
    def isalpha(self): return self.df._df_str_is(self.field, CHK_ISALPHA)

    @cython.always_allow_keywords(False)
    def isdigit(self): return self.df._df_str_is(self.field, CHK_ISDIGIT)

    @cython.always_allow_keywords(False)
    def isspace(self): return self.df._df_str_is(self.field, CHK_ISSPACE)

    @cython.always_allow_keywords(False)
    def islower(self): return self.df._df_str_is(self.field, CHK_ISLOWER)

    @cython.always_allow_keywords(False)
    def isupper(self): return self.df._df_str_is(self.field, CHK_ISUPPER)

    @cython.always_allow_keywords(False)
    def istitle(self): return self.df._df_str_is(self.field, CHK_ISTITLE)

    @cython.always_allow_keywords(False)
    def isnumeric(self): return self.df._df_str_is(self.field, CHK_ISNUMERIC)

    @cython.always_allow_keywords(False)
    def isdecimal(self): return self.df._df_str_is(self.field, CHK_ISDECIMAL)

    @cython.always_allow_keywords(False)
    def is_chinese(self):
        return self.df._df_is_charset(self.field, True)

    @cython.always_allow_keywords(False)
    def is_numeric_punct(self):
        return self.df._df_is_charset(self.field, False)

    @cython.always_allow_keywords(False)
    def keep_chinese(self):
        return self.df._keep_charset(self.field, True)

    @cython.always_allow_keywords(False)
    def keep_numeric_punct(self):
        return self.df._keep_charset(self.field, False)

    @cython.always_allow_keywords(False)
    def contains(self, object pattern):
        """过滤包含特定模式的行（正则匹配），返回新 NDframe。"""
        pattern = _ensure_regex(pattern)
        return self.df._df_regex(self.field, pattern, True)

    @cython.always_allow_keywords(False)
    def contains_substr(self, str s):
        """过滤字段值包含子串 s 的行（纯 C 滑动窗口子串匹配），返回新 NDframe"""
        return self.df._df_regex(self.field, s, False)

    @cython.always_allow_keywords(False)
    def contains_in_bbox(self, object pattern, int x0, int y0, int x1, int y1):
        """在指定坐标框内进行正则过滤，返回新 NDframe"""
        if x0 < 0: x0 = 0
        if y0 < 0: y0 = 0
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        if x0 > x1 or y0 > y1:
            raise ValueError(f"def contains_in_bbox invalid boundary: x0({x0})>x1({x1}) or y0({y0})>y1({y1})")
        pattern = _ensure_regex(pattern)
        return self.df._df_regex_in_x0y0x1y1(self.field, pattern, x0, y0, x1, y1)

    def replace(self, str old_str, str new_str, str old_str2="", str new_str2=""):
        """就地替换，返回原 df 方便链式调用。支持双字符串替换（可选第二对）"""
        cdef NDframe df = self.df
        if old_str2:
            df._replace_dual(self.field, old_str, new_str, old_str2, new_str2)
        else:
            df._replace(self.field, old_str, new_str)
        return df

    def regex_replace(self, object pattern, str rewrite):
        """就地正则替换"""
        pattern = _ensure_regex(pattern)
        self.df._regex_replace(self.field, pattern, rewrite)
        return self.df

    @cython.always_allow_keywords(False)
    def regex_extract(self, object pattern):
        """正则提取，返回新 NDframe"""
        pattern = _ensure_regex(pattern)
        return self.df._df_regex_extract(self.field, pattern)

    @cython.always_allow_keywords(False)
    def join(self, str separator=""):
        """拼接文本"""
        return self.df._join(self.field, separator)

    # ========================================================
    # 字符串长度相关过滤 (仿照 pandas str.len() + 条件筛选)
    # ========================================================
    @cython.always_allow_keywords(False)
    def strlen_greater(self, int n):
        """返回字符串长度 > n 的行构成的新 NDframe"""
        if n < 0:
            raise ValueError("n must be >= 0")
        return self.df._df_strlen_cmp(self.field, n, OP_GREATER)

    @cython.always_allow_keywords(False)
    def strlen_less(self, int n):
        """返回字符串长度 < n 的行构成的新 NDframe"""
        if n < 0:
            raise ValueError("n must be >= 0")
        return self.df._df_strlen_cmp(self.field, n, OP_LESS)

    @cython.always_allow_keywords(False)
    def strlen_equal(self, n):
        """返回字符串长度 == n 的行构成的新 NDframe"""
        if n < 0:
            raise ValueError("n must be >= 0")
        return self.df._df_strlen_cmp(self.field, n, OP_EQUAL)

    @cython.always_allow_keywords(False)
    def values(self, index=None):
        cdef NDframe df = self.df
        cdef UCS2Column_SoA src = df.text_col if self.field == FIELD_txt else df.text_sep_col
        if index is None:
            return [src._get_str(i) for i in range(src.length)]

        cdef Py_ssize_t idx = int(index)
        if df.length == 0:
            raise IndexError("Cannot get value from an empty NDframe")
        if idx < 0:
            idx += df.length
        if idx < 0 or idx >= df.length:
            raise IndexError("Index out of range")
        return src._get_str(idx)

    @cython.always_allow_keywords(False)
    def len(self):
        """返回每行字符串长度的列表"""
        return self.df._strlen_list(self.field)

    @cython.always_allow_keywords(False)
    def is_in(self, object texts):
        """返回字段值在给定集合中的新 NDframe"""
        return self.df._df_is_in(self.field, texts)

    @cython.always_allow_keywords(False)
    def not_in(self, object texts):
        """返回字段值不在给定集合中的新 NDframe"""
        return self.df._df_not_in(self.field, texts)

    @cython.always_allow_keywords(False)
    def contains_any(self, object texts):
        """检查字段值是否包含给定字符串集合中的任意一个"""
        return self.df._contains_any(self.field, texts)

    @cython.always_allow_keywords(False)
    def is_equal(self, str s):
        """返回字段值恰好等于 s 的行构成的新 NDframe（纯 C memcmp 比较）"""
        return self.df._df_str_equal(self.field, s)

    # ========================================================
    # 正则 + 字符串长度组合筛选 (C++ df_regex_strlen_xxx)
    # ========================================================
    @cython.always_allow_keywords(False)
    def regex_strlen_greater(self, object pattern, int n):
        """正则匹配 && 字符串长度 > n 的行 → 新 NDframe"""
        pattern = _ensure_regex(pattern)
        return self.df._df_regex_strlen_cmp(self.field, pattern, n, OP_GREATER)

    @cython.always_allow_keywords(False)
    def regex_strlen_less(self, object pattern, int n):
        """正则匹配 && 字符串长度 < n 的行 → 新 NDframe"""
        pattern = _ensure_regex(pattern)
        return self.df._df_regex_strlen_cmp(self.field, pattern, n, OP_LESS)

    @cython.always_allow_keywords(False)
    def regex_strlen_equal(self, object pattern, int n):
        """正则匹配 && 字符串长度 == n 的行 → 新 NDframe"""
        pattern = _ensure_regex(pattern)
        return self.df._df_regex_strlen_cmp(self.field, pattern, n, OP_EQUAL)

    # ========================================================
    # 跨字段字符串替换 (C++ replace<field_src, field_dst>)
    # ========================================================
    def replace_to(self, str old_str, str new_str, str target_field="text"):
        """将本字段的 old_str 替换为 new_str，写入 target_field（就地）"""
        cdef int dst_field = FIELD_txt if target_field == "text" else FIELD_sep
        self.df._replace_cross(self.field, dst_field, old_str, new_str)
        return self.df

    def regex_replace_to(self, pattern, str rewrite, str target_field="text"):
        """正则替换：本字段匹配 pattern 替换为 rewrite，写入 target_field（就地）"""
        cdef int dst_field = FIELD_txt if target_field == "text" else FIELD_sep
        pattern = _ensure_regex(pattern)
        self.df._regex_replace_cross(self.field, dst_field, pattern, rewrite)
        return self.df

    @cython.always_allow_keywords(False)
    def drop_regex(self, object pattern):
        """就地删除匹配该正则的行，返回原 df"""
        pattern = _ensure_regex(pattern)
        return self.df._drop_in_regex(self.field, pattern)

    @cython.always_allow_keywords(False)
    def idx_contains(self, object pattern):
        pattern = _ensure_regex(pattern)
        return self.df._idx_regex(self.field, pattern)

    def idx_contains_in_bbox(self, object pattern, int x0, int y0, int x1, int y1):
        pattern = _ensure_regex(pattern)
        if x0 < 0: x0 = 0
        if x1 < 0: x1 = 0
        if y0 < 0: y0 = 0
        if y1 < 0: y1 = 0

        if x0 > x1 or y0 > y1:
            raise ValueError(f"def idx_contains_in_bbox invalid boundary: x0({x0})>x1({x1}) or y0({y0})>y1({y1})")
        return self.df._idx_regex_in_bbox(self.field, pattern, x0, y0, x1, y1)

    @cython.always_allow_keywords(False)
    def idx_strlen_greater(self, int n):
        if n < 0:
            raise ValueError(f"invalid length: {n}")
        return self.df._idx_strlen_cmp(self.field, n, OP_GREATER)

    @cython.always_allow_keywords(False)
    def idx_strlen_less(self, int n):
        if n < 0:
            raise ValueError(f"invalid length: {n}")
        return self.df._idx_strlen_cmp(self.field, n, OP_LESS)

    @cython.always_allow_keywords(False)
    def idx_strlen_equal(self, int n):
        if n < 0:
            raise ValueError(f"invalid length: {n}")
        return self.df._idx_strlen_cmp(self.field, n, OP_EQUAL)

    def idx_regex_strlen_greater(self, object pattern, int n):
        if n < 0:
            raise ValueError(f"invalid length: {n}")
        pattern = _ensure_regex(pattern)
        return self.df._idx_regex_strlen_cmp(self.field, pattern, n, OP_GREATER)

    def idx_regex_strlen_less(self, object pattern, int n):
        if n < 0:
            raise ValueError(f"invalid length: {n}")
        pattern = _ensure_regex(pattern)
        return self.df._idx_regex_strlen_cmp(self.field, pattern, n, OP_LESS)

    def idx_regex_strlen_equal(self, object pattern, int n):
        if n < 0:
            raise ValueError(f"invalid length: {n}")
        pattern = _ensure_regex(pattern)
        return self.df._idx_regex_strlen_cmp(self.field, pattern, n, OP_EQUAL)


@cython.final
@cython.binding(False)
cdef class NumericAccessor:
    """数字字段访问器：提供 df.x0.max() 等 API"""
    cdef NDframe df
    cdef int field_id

    @cython.always_allow_keywords(False)
    def __cinit__(self, NDframe df, int field_id):
        self.df = df
        self.field_id = field_id

    @cython.always_allow_keywords(False)
    def __iter__(self):
        cdef int field_id = self.field_id
        cdef Py_ssize_t n = self.df.length
        cdef RowC* rows = self.df.rows_c
        cdef list result = [0] * n
        cdef Py_ssize_t i
        for i in range(n):
            result[i] = _get_field_val(&rows[i], field_id)
        return iter(result)

    @property
    def field_name(self):
        return _field_id_to_name(self.field_id)

    @cython.always_allow_keywords(False)
    def __repr__(self):
        cdef NDframe df = self.df
        cdef int field_id = self.field_id
        cdef Py_ssize_t n = df.length
        cdef Py_ssize_t preview_n = n if n < 5 else 5
        cdef list preview = [None] * preview_n
        cdef RowC* rows = df.rows_c
        cdef Py_ssize_t i

        for i in range(preview_n):
            preview[i] = str(_get_field_val(&rows[i], field_id))
        if n > 5:
            preview.append("...")
        return f"<Numeric {self.field_name}: [{', '.join(preview)}]>"

    @cython.always_allow_keywords(False)
    def diff(self):
        """
        返回样本的一阶差分列表。
        第 0 个元素固定为 None，其后元素为本行与上一行的差值 (curr - prev)。
        """
        return self.df._diff_col(self.field_id)

    def var(self, int ddof=1):
        """
        计算该整型字段的方差。
        默认 ddof=1 计算样本方差 (Bessel's correction)；ddof=0 计算总体方差。
        """
        return self.df._var_col(self.field_id, ddof)

    def std(self, int ddof=1):
        """
        计算该整型字段的标准差。
        """
        return self.df._std_col(self.field_id, ddof)

    @cython.always_allow_keywords(False)
    def values(self, index=None):
        cdef NDframe df = self.df
        cdef int field_id = self.field_id

        if index is None:
            return df._col_values(field_id)

        cdef Py_ssize_t idx = int(index)
        cdef Py_ssize_t n = df.length

        if n == 0:
            raise IndexError("Cannot get value from an empty NDframe")

        if idx < 0:
            idx += n
        if idx < 0 or idx >= n:
            raise IndexError(f"Index {index} out of range for NDframe with {n} rows")

        return _get_field_val(&df.rows_c[idx], field_id)

    @cython.always_allow_keywords(False)
    def mode(self):
        if self.field_id == FIELD_height:
            return self.df._mode_height()
        return self.df._mode_col(self.field_id)

    @cython.always_allow_keywords(False)
    def max(self):
        return self.df._max_col(self.field_id)

    @cython.always_allow_keywords(False)
    def min(self):
        return self.df._min_col(self.field_id)

    @cython.always_allow_keywords(False)
    def mean(self):
        return self.df._mean_col(self.field_id)

    @cython.always_allow_keywords(False)
    def greater(self, int value):
        """返回该字段值 > value 的行构成的新 NDframe"""
        return self.df._df_filter_by_int(self.field_id, value, 0, OP_GREATER)

    @cython.always_allow_keywords(False)
    def less(self, int value):
        """返回该字段值 < value 的行构成的新 NDframe"""
        return self.df._df_filter_by_int(self.field_id, value, 0, OP_LESS)

    def between(self, int lo, int hi):
        """返回该字段值在 (lo, hi) 区间内的行构成的新 NDframe"""
        if lo > hi:
            raise ValueError("lo must be less than hi")
        return self.df._df_filter_by_int(self.field_id, lo, hi, OP_BETWEEN)

    @cython.always_allow_keywords(False)
    def add(self, int value):
        """就地：每个值 +value"""
        self.df._add_col(self.field_id, value)
        return self.df

    @cython.always_allow_keywords(False)
    def sub(self, int value):
        """就地：每个值 -value（不低于0）"""
        self.df._sub_col(self.field_id, value)
        return self.df

    def sort(self, bint descending=False):
        """按该字段排序（就地），返回原 df 支持链式调用"""
        self.df._sort_col(self.field_id, descending)
        return self.df

    @cython.always_allow_keywords(False)
    def idx_greater(self, int value):
        """返回该字段值 > value 的索引"""
        return self.df._idx_filter_by_int(self.field_id, value, 0, OP_GREATER)

    @cython.always_allow_keywords(False)
    def idx_less(self, int value):
        """返回该字段值 < value 的索引"""
        return self.df._idx_filter_by_int(self.field_id, value, 0, OP_LESS)

    @cython.always_allow_keywords(False)
    def idx_equal(self, int value):
        """返回该字段值 = value 的索引"""
        return self.df._idx_filter_by_int(self.field_id, value, 0, OP_EQUAL)

    def idx_between(self, int lo, int hi):
        """返回该字段值在 [lo, hi) 之间的索引"""
        return self.df._idx_filter_by_int(self.field_id, lo, hi, OP_BETWEEN)


# ==========================================
# 极速字符判定宏 (Inline Functions)
# ==========================================
cdef inline bint _is_chinese_char(Py_UCS4 cp) noexcept:
    """判定是否为中文字符 (覆盖基本、扩展A、扩展B、以及兼容汉字区)"""
    return (
        (0x4E00 <= cp <= 0x9FFF) or   # 核心基本汉字
        (0x3400 <= cp <= 0x4DBF) or   # 扩展 A
        (0x20000 <= cp <= 0x2A6DF) or # 扩展 B
        (0xF900 <= cp <= 0xFAFF)      # 兼容汉字区 (防止繁体特殊字符漏判)
    )

cdef inline bint _is_numeric_punct_char(Py_UCS4 cp) noexcept:
    """判定是否为数字或特定标点 [0-9.,]"""
    return (0x30 <= cp <= 0x39) or (cp == 0x2E) or (cp == 0x2C)


# ==========================================
# 极速字符判定谓词
# ==========================================
cdef inline bint _pred_isalnum(Py_UCS2 c) noexcept: return Py_UNICODE_ISALNUM(c)
cdef inline bint _pred_isalpha(Py_UCS2 c) noexcept: return Py_UNICODE_ISALPHA(c)
cdef inline bint _pred_isdigit(Py_UCS2 c) noexcept: return Py_UNICODE_ISDIGIT(c)
cdef inline bint _pred_isspace(Py_UCS2 c) noexcept: return Py_UNICODE_ISSPACE(c)
cdef inline bint _pred_islower(Py_UCS2 c) noexcept: return Py_UNICODE_ISLOWER(c)
cdef inline bint _pred_isupper(Py_UCS2 c) noexcept: return Py_UNICODE_ISUPPER(c)
cdef inline bint _pred_istitle(Py_UCS2 c) noexcept: return Py_UNICODE_ISTITLE(c) or Py_UNICODE_ISLOWER(c)
cdef inline bint _pred_isnumeric(Py_UCS2 c) noexcept: return Py_UNICODE_ISNUMERIC(c)
cdef inline bint _pred_isdecimal(Py_UCS2 c) noexcept: return Py_UNICODE_ISDECIMAL(c)
cdef inline bint _pred_is_chinese(Py_UCS2 c) noexcept: return _is_chinese_char(c)
cdef inline bint _pred_is_numeric_punct(Py_UCS2 c) noexcept: return _is_numeric_punct_char(c)


cdef int _get_visual_width(str s):
    if not s:
        return 0
    cdef int w = 0
    cdef Py_ssize_t length = PyUnicode_GET_LENGTH(s)
    cdef Py_ssize_t i
    cdef Py_UCS4 cp

    for i in range(length):
        cp = PyUnicode_READ_CHAR(s, i)
        if cp < 0x80:
            w += 1
        elif cp == 0x2014 or cp == 0x2018 or cp == 0x2019 or cp == 0x201C or cp == 0x201D or cp == 0x2026:
            w += 2
        elif cp > 0x2E80:
            w += 2
        else:
            w += 1
    return w

cdef inline int _int_display_width(int n) noexcept:
    if n < 10: return 1
    if n < 100: return 2
    if n < 1000: return 3
    if n < 10000: return 4
    if n < 100000: return 5
    return 6

cdef str _truncate_str(str s, int max_chars=90, int head=79, int tail=8):
    if not s:
        return ""
    if len(s) <= max_chars:
        return s
    return s[:head] + "..." + s[-tail:]


# ==========================================
# UCS2 字典序比较 + 归并排序 (供 text/text_sep 就地排序使用)
# ==========================================
cdef inline int _ucs2_cmp(Py_UCS2* chars, Py_ssize_t* offsets, Py_ssize_t a, Py_ssize_t b) noexcept:
    """比较第 a 行与第 b 行的字符串：-1/0/1。短串在相同时为前缀时排前面 (同 Python str 比较)"""
    cdef Py_ssize_t a_start = offsets[a]
    cdef Py_ssize_t b_start = offsets[b]
    cdef Py_ssize_t a_len = offsets[a + 1] - a_start
    cdef Py_ssize_t b_len = offsets[b + 1] - b_start
    cdef Py_ssize_t min_len = a_len if a_len < b_len else b_len
    cdef Py_ssize_t k
    for k in range(min_len):
        if chars[a_start + k] < chars[b_start + k]: return -1
        if chars[a_start + k] > chars[b_start + k]: return 1
    if a_len < b_len: return -1
    if a_len > b_len: return 1
    return 0

cdef void _mergesort_str(Py_ssize_t* order, Py_ssize_t* tmp, Py_ssize_t lo, Py_ssize_t hi,
                         Py_UCS2* chars, Py_ssize_t* offsets, bint descending) noexcept:
    """迭代归并排序 [lo, hi)：对 order 数组按 (chars, offsets) 指向的字符串排序。
    等值字符串保持原始相对顺序 (稳定)，降序仅翻转比较方向。"""
    if hi - lo < 2:
        return
    cdef Py_ssize_t mid = (lo + hi) // 2
    _mergesort_str(order, tmp, lo, mid, chars, offsets, descending)
    _mergesort_str(order, tmp, mid, hi, chars, offsets, descending)

    cdef Py_ssize_t i = lo, j = mid, k = lo
    cdef int c
    while i < mid and j < hi:
        c = _ucs2_cmp(chars, offsets, order[i], order[j])
        if descending: c = -c
        if c <= 0:
            tmp[k] = order[i]; i += 1
        else:
            tmp[k] = order[j]; j += 1
        k += 1
    while i < mid:
        tmp[k] = order[i]; i += 1; k += 1
    while j < hi:
        tmp[k] = order[j]; j += 1; k += 1
    for k in range(lo, hi):
        order[k] = tmp[k]


# ==========================================
# 3. 核心 NDframe 定义
# ==========================================
cdef struct RowC:
    int idx      # FIELD_idx   = 0
    int x0       # FIELD_x0    = 1
    int y0       # FIELD_y0    = 2
    int x1       # FIELD_x1    = 3
    int y1       # FIELD_y1    = 4
    int width    # FIELD_width = 5
    int height   # FIELD_height= 6
    int page     # FIELD_page  = 7
    int flags    # FIELD_flags = 8


cdef inline int* _row_field_ptr(RowC* row, int field) noexcept:
    return &row.idx + field


@cython.final
@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class NDframe:
    cdef:
        RowC* rows_c
        Py_ssize_t length
        Py_ssize_t _capacity
        UCS2Column_SoA text_col
        UCS2Column_SoA text_sep_col

    @cython.always_allow_keywords(False)
    def __cinit__(self, Py_ssize_t capacity=16):
        self._capacity = max(capacity, 1)
        self.length = 0

        self.rows_c = <RowC*>malloc(<size_t>self._capacity * sizeof(RowC))
        if not self.rows_c:
            raise MemoryError()

        self.text_col = UCS2Column_SoA(self._capacity)
        self.text_sep_col = UCS2Column_SoA(self._capacity)

    def __dealloc__(self):
        """清理内存"""
        if self.rows_c is not NULL:
            free(self.rows_c)

    def __len__(self):
        return self.length

    def __bool__(self):
        return self.length > 0

    @property
    def empty(self):
        return self.length == 0

    @property
    def data(self):
        return self.to_list()


    @cython.always_allow_keywords(False)
    def __iter__(self):
        cdef Py_ssize_t i
        cdef RowC* rows = self.rows_c
        cdef Py_ssize_t n = self.length
        for i in range(n):
            yield {
                "idx": rows[i].idx,
                "x0": rows[i].x0,
                "y0": rows[i].y0,
                "x1": rows[i].x1,
                "y1": rows[i].y1,
                "width": rows[i].width,
                "height": rows[i].height,
                "page": rows[i].page,
                "flags": rows[i].flags,
                "text": self.text_col._get_str(i),
                "text_sep": self.text_sep_col._get_str(i)
            }

    def __repr__(self):
        return f"NDframe({self.length} rows)"

    def __str__(self):
        return f"<NDframe: {self.length} rows>"

    def _repr_html_(self):
        if self.length == 0:
            return "<div><strong>NDframe</strong> is empty.</div>"

        cdef Py_ssize_t preview_n = min(self.length, 60)
        cdef list html = []
        html.append("<div style='max-height: 400px; overflow: auto;'>")
        html.append("<table border='1' style='border-collapse: collapse; text-align: right;'>")

        html.append("<thead><tr style='background-color: #f2f2f2;'>")
        for col in HEADERS:
            html.append(f"<th>{col}</th>")
        html.append("</tr></thead><tbody>")

        cdef Py_ssize_t i
        for i in range(preview_n):
            html.append("<tr>")
            html.append(f"<td>{self.rows_c[i].idx}</td>")
            html.append(f"<td>{self.rows_c[i].x0}</td>")
            html.append(f"<td>{self.rows_c[i].y0}</td>")
            html.append(f"<td>{self.rows_c[i].x1}</td>")
            html.append(f"<td>{self.rows_c[i].y1}</td>")
            html.append(f"<td>{self.rows_c[i].width}</td>")
            html.append(f"<td>{self.rows_c[i].height}</td>")
            html.append(f"<td>{self.rows_c[i].page}</td>")
            html.append(f"<td>{self.rows_c[i].flags}</td>")
            html.append(f"<td style='text-align: left;'>{self.text_col._get_str(i)}</td>")
            html.append(f"<td style='text-align: left;'>{self.text_sep_col._get_str(i)}</td>")
            html.append("</tr>")

        html.append("</tbody></table></div>")
        if self.length > preview_n:
            html.append(f"<p>... showing {preview_n} of {self.length} rows ...</p>")

        return "".join(html)

    # ----------------------------------------------------
    # API 属性映射区
    # ----------------------------------------------------
    @property
    def text(self): return StringAccessor(self, FIELD_txt)

    @property
    def text_sep(self): return StringAccessor(self, FIELD_sep)

    @property
    def x0(self): return NumericAccessor(self, FIELD_x0)

    @property
    def y0(self): return NumericAccessor(self, FIELD_y0)

    @property
    def x1(self): return NumericAccessor(self, FIELD_x1)

    @property
    def y1(self): return NumericAccessor(self, FIELD_y1)

    @property
    def width(self): return NumericAccessor(self, FIELD_width)

    @property
    def height(self): return NumericAccessor(self, FIELD_height)

    @property
    def idx(self): return NumericAccessor(self, FIELD_idx)

    @property
    def page(self): return NumericAccessor(self, FIELD_page)

    @property
    def flags(self): return NumericAccessor(self, FIELD_flags)

    # ----------------------------------------------------
    # pandas-friendly 便捷方法
    # ----------------------------------------------------
    def __getitem__(self, key):
        """严格受限的索引：仅支持 df[3] 或 df[1:5]"""
        cdef Py_ssize_t start, stop, new_len, int_key
        cdef NDframe new_df

        if isinstance(key, slice):
            if key.step not in (None, 1):
                raise ValueError("Only contiguous slices are supported.")

            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else self.length

            if start < 0: start += self.length
            if stop < 0: stop += self.length
            if start < 0: start = 0
            if stop > self.length: stop = self.length
            if start > stop: start = stop

            new_len = stop - start
            new_df = NDframe(new_len if new_len > 0 else 1)

            if new_len > 0:
                memcpy(new_df.rows_c, self.rows_c + start, new_len * sizeof(RowC))
                new_df.text_col = self.text_col._slice(start, stop)
                new_df.text_sep_col = self.text_sep_col._slice(start, stop)
                new_df.length = new_len
            return new_df

        elif isinstance(key, int):
            int_key = <int>key
            if int_key < 0: int_key += <int>self.length
            if int_key < 0 or int_key >= <int>self.length:
                raise IndexError(f"Index {int_key} out of range [0, {self.length})")
            return {
                "idx": self.rows_c[int_key].idx,
                "x0": self.rows_c[int_key].x0,
                "y0": self.rows_c[int_key].y0,
                "x1": self.rows_c[int_key].x1,
                "y1": self.rows_c[int_key].y1,
                "width": self.rows_c[int_key].width,
                "height": self.rows_c[int_key].height,
                "page": self.rows_c[int_key].page,
                "flags": self.rows_c[int_key].flags,
                "text": self.text_col._get_str(int_key),
                "text_sep": self.text_sep_col._get_str(int_key)
            }
        else:
            raise TypeError("Indices must be integers or slices (e.g. df[0], df[1:3])")

    @cython.always_allow_keywords(False)
    def head(self, int n=5):
        """返回前 n 行的新 NDframe"""
        return self[0:n]

    @cython.always_allow_keywords(False)
    def tail(self, int n=5):
        """返回后 n 行的新 NDframe"""
        cdef Py_ssize_t start = self.length - n
        if start < 0: start = 0
        return self[start:]

    @cython.always_allow_keywords(False)
    def pop_back(self):
        """
        删除最后一行（就地）。
        若 NDframe 为空，不执行任何操作。
        """
        if self.length == 0:
            return
        self.length -= 1
        self.text_col.pop_back()
        self.text_sep_col.pop_back()


    @cython.always_allow_keywords(False)
    def bbox(self, Py_ssize_t n=0):
        """
        返回第 n 行的边界框 (x0, y0, x1, y1) 元组。

        参数 n 支持负索引（-1 表示最后一行）。
        默认 n=0，返回第 0 行坐标。
        """
        cdef Py_ssize_t length = self.length
        cdef Py_ssize_t idx = n
        if idx < 0:
            idx += length
        if idx < 0 or idx >= length:
            raise IndexError(f"Index {n} out of range for NDframe with {length} rows")
        cdef RowC* row = &self.rows_c[idx]
        return (row.x0, row.y0, row.x1, row.y1)


    @cython.always_allow_keywords(False)
    def copy(self):
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n > 0:
            memcpy(new_df.rows_c, self.rows_c, n * sizeof(RowC))
            new_df.text_col = self.text_col.copy()
            new_df.text_sep_col = self.text_sep_col.copy()
            new_df.length = n
        return new_df

    cdef void _reserve(self, Py_ssize_t new_capacity) except *:
        if new_capacity <= self._capacity:
            return
        cdef RowC* new_rows = <RowC*>realloc(self.rows_c, new_capacity * sizeof(RowC))
        if not new_rows:
            raise MemoryError()
        self.rows_c = new_rows
        self._capacity = new_capacity

    cdef void _append_c_row(self, RowC* row, str text, str text_sep):
        if self.length >= self._capacity:
            self._reserve(self._capacity * 2)

        self.rows_c[self.length] = row[0]
        self.text_col.append(text)
        self.text_sep_col.append(text_sep)
        self.length += 1

    @cython.always_allow_keywords(False)
    def append(self, int idx, int x0, int y0, int x1, int y1,
               int width, int height, int page, int flags,
               str text, str text_sep=""):
        cdef Py_ssize_t n = self.length
        if n >= self._capacity:
            self._reserve(self._capacity * 2)

        cdef RowC* rows = self.rows_c
        rows[n].idx = idx
        rows[n].x0 = x0
        rows[n].y0 = y0
        rows[n].x1 = x1
        rows[n].y1 = y1
        rows[n].width = width
        rows[n].height = height
        rows[n].page = page
        rows[n].flags = flags

        if text is None: text = ""
        if text_sep is None: text_sep = ""

        self.text_col.append(text)
        self.text_sep_col.append(text_sep)
        self.length = n + 1

    @cython.always_allow_keywords(False)
    def to_list(self):
        cdef Py_ssize_t n = self.length
        if n == 0:
            return []

        cdef RowC* rows = self.rows_c
        cdef list res = [None] * n
        cdef Py_ssize_t i
        cdef RowC* row
        cdef str cur_txt, cur_sep

        for i in range(n):
            row = &rows[i]
            cur_txt = self.text_col._get_str(i)
            cur_sep = self.text_sep_col._get_str(i)
            res[i] = {
                "idx": row.idx,
                "x0": row.x0,
                "y0": row.y0,
                "x1": row.x1,
                "y1": row.y1,
                "width": row.width,
                "height": row.height,
                "page": row.page,
                "flags": row.flags,
                "text": cur_txt,
                "text_sep": cur_sep
            }
        return res


    @classmethod
    def from_dict(cls, dict data):
        cdef object tmp
        cdef list l_x0, l_y0, l_x1, l_y1, l_w, l_h, l_text

        cdef tuple required_keys = ("x0", "y0", "x1", "y1", "width", "height", "text")
        for k in required_keys:
            if k not in data:
                raise ValueError(f"Missing mandatory field: '{k}'")

        tmp = data["x0"]; l_x0 = tmp if type(tmp) is list else list(tmp)
        tmp = data["y0"]; l_y0 = tmp if type(tmp) is list else list(tmp)
        tmp = data["x1"]; l_x1 = tmp if type(tmp) is list else list(tmp)
        tmp = data["y1"]; l_y1 = tmp if type(tmp) is list else list(tmp)
        tmp = data["width"]; l_w = tmp if type(tmp) is list else list(tmp)
        tmp = data["height"]; l_h = tmp if type(tmp) is list else list(tmp)
        tmp = data["text"]; l_text = tmp if type(tmp) is list else list(tmp)

        cdef Py_ssize_t length = len(l_x0)

        cdef list l_page = None
        cdef list l_flags = None
        cdef list l_sep = None

        tmp = data.get("page")
        if tmp is not None: l_page = tmp if type(tmp) is list else list(tmp)

        tmp = data.get("flags")
        if tmp is not None: l_flags = tmp if type(tmp) is list else list(tmp)

        tmp = data.get("text_sep")
        if tmp is not None: l_sep = tmp if type(tmp) is list else list(tmp)

        if not (length == len(l_y0) == len(l_x1) == len(l_y1) ==
                len(l_w) == len(l_h) == len(l_text)):
            raise ValueError("All mandatory arrays must be of the same length")

        if l_page is not None and len(l_page) != length:
            raise ValueError("Optional array 'page' length mismatch")
        if l_flags is not None and len(l_flags) != length:
            raise ValueError("Optional array 'flags' length mismatch")
        if l_sep is not None and len(l_sep) != length:
            raise ValueError("Optional array 'text_sep' length mismatch")

        if length <= 0:
            return cls(16)

        cdef NDframe df = cls(length)
        cdef RowC* rows = df.rows_c
        memset(rows, 0, length * sizeof(RowC))

        cdef Py_ssize_t i
        for i in range(length):
            rows[i].idx    = i
            rows[i].x0     = int(l_x0[i])
            rows[i].y0     = int(l_y0[i])
            rows[i].x1     = int(l_x1[i])
            rows[i].y1     = int(l_y1[i])
            rows[i].width  = int(l_w[i])
            rows[i].height = int(l_h[i])

        if l_page is not None:
            for i in range(length): rows[i].page = int(l_page[i])
        if l_flags is not None:
            for i in range(length): rows[i].flags = int(l_flags[i])

        for i in range(length):
            df.text_col.append(str(l_text[i]) if l_text[i] is not None else "")
            if l_sep is not None:
                df.text_sep_col.append(str(l_sep[i]) if l_sep[i] is not None else "")
            else:
                df.text_sep_col.append("")

        df.length = length
        return df

    def to_string(self, int max_rows=50, str name=None):
        cdef str dataframe_name = name if name else "NDframe"

        if self.length == 0:
            return f"{dataframe_name} is Empty!\n"

        cdef list display_indices = []
        cdef Py_ssize_t i

        if self.length <= max_rows:
            for i in range(self.length):
                display_indices.append(i)
        else:
            half = max_rows // 2
            for i in range(half):
                display_indices.append(i)
            display_indices.append(-1)
            for i in range(self.length - half, self.length):
                display_indices.append(i)

        cdef int widths[11]
        widths[0]=3; widths[1]=2; widths[2]=2; widths[3]=2; widths[4]=2; widths[5]=5
        widths[6]=6; widths[7]=4; widths[8]=5; widths[9]=4; widths[10]=8

        cdef int iw, j
        cdef str s_val, t_val

        cdef list cached_txt = []
        cdef list cached_sep = []
        cdef list cached_txt_w = []
        cdef list cached_sep_w = []

        for i in display_indices:
            if i == -1:
                cached_txt.append("")
                cached_sep.append("")
                cached_txt_w.append(0)
                cached_sep_w.append(0)
                continue

            iw = _int_display_width(self.rows_c[i].idx)
            if iw > widths[0]: widths[0] = iw
            iw = _int_display_width(self.rows_c[i].x0)
            if iw > widths[1]: widths[1] = iw
            iw = _int_display_width(self.rows_c[i].y0)
            if iw > widths[2]: widths[2] = iw
            iw = _int_display_width(self.rows_c[i].x1)
            if iw > widths[3]: widths[3] = iw
            iw = _int_display_width(self.rows_c[i].y1)
            if iw > widths[4]: widths[4] = iw
            iw = _int_display_width(self.rows_c[i].width)
            if iw > widths[5]: widths[5] = iw
            iw = _int_display_width(self.rows_c[i].height)
            if iw > widths[6]: widths[6] = iw
            iw = _int_display_width(self.rows_c[i].page)
            if iw > widths[7]: widths[7] = iw
            iw = _int_display_width(self.rows_c[i].flags)
            if iw > widths[8]: widths[8] = iw

            s_val = _truncate_str(self.text_col._get_str(i))
            iw = _get_visual_width(s_val)
            cached_txt.append(s_val)
            cached_txt_w.append(iw)
            if iw > widths[9]: widths[9] = iw

            t_val = _truncate_str(self.text_sep_col._get_str(i))
            iw = _get_visual_width(t_val)
            cached_sep.append(t_val)
            cached_sep_w.append(iw)
            if iw > widths[10]: widths[10] = iw

        cdef list lines = []
        lines.append(f"[{dataframe_name}] ({self.length} rows):")

        cdef list sep_parts = ["+"]
        for j in range(11): sep_parts.extend(["-" * (widths[j] + 2), "+"])
        cdef str thin_sep = "".join(sep_parts)

        sep_parts = ["+"]
        for j in range(11): sep_parts.extend(["=" * (widths[j] + 2), "+"])
        cdef str thick_sep = "".join(sep_parts)

        lines.append(thin_sep)
        cdef list row_cells = ["|"]
        cdef int pad
        for j in range(11):
            pad = widths[j] - HEADER_LENS[j]
            row_cells.append(f" {HEADERS[j]}{' ' * pad} |")
        lines.append("".join(row_cells))
        lines.append(thick_sep)

        cdef int cache_idx = 0
        for i in display_indices:
            if i == -1:
                row_cells = ["|"]
                for j in range(11):
                    pad = widths[j] - 3
                    if pad >= 0:
                        row_cells.append(f" ...{' ' * pad} |")
                    else:
                        row_cells.append(f" {'.' * widths[j]} |")
                lines.append("".join(row_cells))
                cache_idx += 1
                continue

            row_cells = ["|"]
            row_cells.append(f" {self.rows_c[i].idx:<{widths[0]}} |")
            row_cells.append(f" {self.rows_c[i].x0:<{widths[1]}} |")
            row_cells.append(f" {self.rows_c[i].y0:<{widths[2]}} |")
            row_cells.append(f" {self.rows_c[i].x1:<{widths[3]}} |")
            row_cells.append(f" {self.rows_c[i].y1:<{widths[4]}} |")
            row_cells.append(f" {self.rows_c[i].width:<{widths[5]}} |")
            row_cells.append(f" {self.rows_c[i].height:<{widths[6]}} |")
            row_cells.append(f" {self.rows_c[i].page:<{widths[7]}} |")
            row_cells.append(f" {self.rows_c[i].flags:<{widths[8]}} |")

            pad = widths[9] - cached_txt_w[cache_idx]
            row_cells.append(f" {cached_txt[cache_idx]}{' ' * pad} |")

            pad = widths[10] - cached_sep_w[cache_idx]
            row_cells.append(f" {cached_sep[cache_idx]}{' ' * pad} |")

            lines.append("".join(row_cells))
            cache_idx += 1

        lines.append(thin_sep)
        return "\n".join(lines)

    def print(self, int max_rows=50, str name=None):
        print(self.to_string(max_rows=max_rows, name=name))


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef list _str_slice(self, int field, object start, object stop, object step):
        cdef list res = []
        cdef Py_ssize_t i, n = self.length
        cdef UCS2Column_SoA src = self.text_col if field == FIELD_txt else self.text_sep_col
        cdef slice slc = slice(start, stop, step)

        for i in range(n):
            res.append(src._get_str(i)[slc])
        return res

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int _str_count(self, int field, str sub):
        cdef int total = 0
        cdef Py_ssize_t n = self.length
        if n == 0:
            return 0

        cdef Py_ssize_t i, j, k
        cdef UCS2Column_SoA src = self.text_col if field == FIELD_txt else self.text_sep_col

        cdef Py_ssize_t sub_len = 0
        cdef Py_UCS2* sub_buf = _str_to_ucs2_buf(sub, &sub_len)
        if sub_len == 0:
            free(sub_buf)
            for i in range(n):
                total += (src.offsets_c[i + 1] - src.offsets_c[i]) + 1
            return total

        cdef Py_ssize_t start, end, str_len
        cdef bint match
        cdef Py_UCS2* chars = src.chars_c

        for i in range(n):
            start = src.offsets_c[i]
            end = src.offsets_c[i + 1]
            str_len = end - start

            if str_len >= sub_len:
                j = start
                while j <= end - sub_len:
                    match = True
                    for k in range(sub_len):
                        if chars[j + k] != sub_buf[k]:
                            match = False
                            break
                    if match:
                        total += 1
                        j += sub_len
                    else:
                        j += 1

        free(sub_buf)
        return total

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef NDframe _str_transform(self, int field, int mode, str chars=None):
        cdef Py_ssize_t n = self.length
        if n == 0: return self

        cdef UCS2Column_SoA src = self.text_col if field == FIELD_txt else self.text_sep_col
        cdef Py_ssize_t i, j
        cdef Py_ssize_t start, end
        cdef Py_ssize_t read_start, read_end
        cdef Py_ssize_t write_idx = 0
        cdef Py_UCS2 c
        cdef bint has_chars
        cdef Py_ssize_t strip_len
        cdef Py_UCS2 default_strip[4]
        cdef Py_UCS2* strip_buf

        if mode == TRM_STRIP or mode == TRM_LSTRIP or mode == TRM_RSTRIP:
            if chars is not None and PyUnicode_GET_LENGTH(chars) == 0:
                return self
            has_chars = chars is not None
            if has_chars:
                strip_len = 0
                strip_buf = _str_to_ucs2_buf(chars, &strip_len)
            else:
                strip_len = 4
                strip_buf = default_strip
                default_strip[0] = 32
                default_strip[1] = 9
                default_strip[2] = 10
                default_strip[3] = 13

            for i in range(n):
                read_start = src.offsets_c[i]
                read_end = src.offsets_c[i + 1]

                if mode == TRM_STRIP or mode == TRM_LSTRIP:
                    while read_start < read_end:
                        c = src.chars_c[read_start]
                        if not _is_in_chars(c, strip_buf, strip_len): break
                        read_start += 1

                if mode == TRM_STRIP or mode == TRM_RSTRIP:
                    while read_end > read_start:
                        c = src.chars_c[read_end - 1]
                        if not _is_in_chars(c, strip_buf, strip_len): break
                        read_end -= 1

                src.offsets_c[i] = write_idx
                if read_end > read_start:
                    memmove(src.chars_c + write_idx, src.chars_c + read_start, (read_end - read_start) * sizeof(Py_UCS2))
                    write_idx += (read_end - read_start)

            src.offsets_c[n] = write_idx
            src.char_length = write_idx
            if has_chars: free(strip_buf)

        elif mode == TRM_CAPITALIZE:
            for i in range(n):
                start = src.offsets_c[i]
                end = src.offsets_c[i + 1]
                if start < end:
                    src.chars_c[start] = <Py_UCS2>Py_UNICODE_TOUPPER(src.chars_c[start])
                    for j in range(start + 1, end):
                        src.chars_c[j] = <Py_UCS2>Py_UNICODE_TOLOWER(src.chars_c[j])

        elif mode == TRM_SWAPCASE:
            for i in range(n):
                for j in range(src.offsets_c[i], src.offsets_c[i + 1]):
                    c = src.chars_c[j]
                    if Py_UNICODE_ISLOWER(c):
                        src.chars_c[j] = <Py_UCS2>Py_UNICODE_TOUPPER(c)
                    elif Py_UNICODE_ISUPPER(c):
                        src.chars_c[j] = <Py_UCS2>Py_UNICODE_TOLOWER(c)

        return self

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef NDframe _df_str_starts_ends(self, int field, str sub, bint is_starts):
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n == 0: return new_df

        cdef Py_ssize_t sub_len = 0
        cdef Py_UCS2* sub_buf = _str_to_ucs2_buf(sub, &sub_len)
        if sub_len == 0:
            free(sub_buf)
            return self.copy()

        cdef Py_ssize_t i, j

        cdef UCS2Column_SoA src = self.text_col if field == FIELD_txt else self.text_sep_col
        cdef Py_ssize_t str_start, str_end, str_len, cmp_start
        cdef Py_ssize_t write_idx = 0
        cdef bint match

        for i in range(n):
            str_start = src.offsets_c[i]
            str_end = src.offsets_c[i + 1]
            str_len = str_end - str_start

            if str_len >= sub_len:
                cmp_start = str_start if is_starts else str_end - sub_len
                match = True
                for j in range(sub_len):
                    if src.chars_c[cmp_start + j] != sub_buf[j]:
                        match = False
                        break

                if match:
                    _df_copy_row(new_df, self, i, &write_idx)

        free(sub_buf)
        new_df.length = write_idx
        return new_df

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef NDframe _df_str_is(self, int field, int mode):
        """按 StrCheckMode 谓词过滤全命中行，复用列层 _is_mode 判定"""
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n == 0: return new_df

        cdef UCS2Column_SoA src = self.text_col if field == FIELD_txt else self.text_sep_col
        cdef list matches = src._is_mode(mode)
        cdef Py_ssize_t i, write_idx = 0

        for i in range(n):
            if matches[i]:
                _df_copy_row(new_df, self, i, &write_idx)

        new_df.length = write_idx
        return new_df


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef NDframe _df_is_charset(self, int field, bint is_chinese_mode):
        """按字符集谓词 (中文 / 数字标点) 过滤全命中行，复用列层 _is_mode 判定"""
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n == 0: return new_df

        cdef UCS2Column_SoA src = self.text_col if field == FIELD_txt else self.text_sep_col
        cdef list matches = src._is_mode(
            CHK_ISCHINESE if is_chinese_mode else CHK_IS_NUMERIC_PUNCT)
        cdef Py_ssize_t i, write_idx = 0

        for i in range(n):
            if matches[i]:
                _df_copy_row(new_df, self, i, &write_idx)

        new_df.length = write_idx
        return new_df


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef NDframe _keep_charset(self, int field, bint is_chinese_mode):
        cdef Py_ssize_t n = self.length
        if n == 0: return self
        cdef UCS2Column_SoA src = self.text_col if field == FIELD_txt else self.text_sep_col

        cdef Py_ssize_t i, j
        cdef Py_ssize_t read_start, read_end
        cdef Py_ssize_t write_idx = 0
        cdef Py_UCS2 c

        cdef bint (*pred)(Py_UCS2) noexcept
        if is_chinese_mode:
            pred = _pred_is_chinese
        else:
            pred = _pred_is_numeric_punct

        for i in range(n):
            read_start = src.offsets_c[i]
            read_end = src.offsets_c[i+1]

            src.offsets_c[i] = write_idx

            for j in range(read_start, read_end):
                c = src.chars_c[j]
                if pred(c):
                    src.chars_c[write_idx] = c
                    write_idx += 1

        src.offsets_c[n] = write_idx
        src.char_length = write_idx
        return self

    # ========================================================
    # 聚合与计算功能
    # ========================================================
    cdef int _max_col(self, int field):
        cdef Py_ssize_t n = self.length
        if n == 0 or field < FIELD_idx or field > FIELD_flags:
            return 0
        cdef RowC* rows = self.rows_c
        cdef Py_ssize_t i
        cdef int max_val = _get_field_val(&rows[0], field)
        cdef int v
        for i in range(1, n):
            v = _get_field_val(&rows[i], field)
            if v > max_val: max_val = v
        return max_val


    cdef int _min_col(self, int field):
        cdef Py_ssize_t n = self.length
        if n == 0 or field < FIELD_idx or field > FIELD_flags:
            return 0
        cdef RowC* rows = self.rows_c
        cdef Py_ssize_t i
        cdef int min_val = _get_field_val(&rows[0], field)
        cdef int v
        for i in range(1, n):
            v = _get_field_val(&rows[i], field)
            if v < min_val: min_val = v
        return min_val

    cdef double _mean_col(self, int field):
        cdef Py_ssize_t n = self.length
        if n == 0 or field < FIELD_idx or field > FIELD_flags:
            return 0.0
        cdef RowC* rows = self.rows_c
        cdef Py_ssize_t i
        cdef double sum_val = 0.0
        for i in range(n):
            sum_val += _get_field_val(&rows[i], field)
        return sum_val / <double>n


    cdef inline void _col_copy_row(UCS2Column_SoA dst, UCS2Column_SoA src, Py_ssize_t src_idx) except *:
        cdef Py_ssize_t start = src.offsets_c[src_idx]
        cdef Py_ssize_t c_len = src.offsets_c[src_idx + 1] - start

        if dst.length >= dst.capacity:
            dst._reserve_rows(dst.capacity * 2)

        if dst.char_length + c_len > dst.char_capacity:
            dst._reserve_chars(dst.char_capacity * 2 + c_len + 1024)

        if c_len > 0:
            memcpy(dst.chars_c + dst.char_length, src.chars_c + start, c_len * sizeof(Py_UCS2))

        dst.char_length += c_len
        dst.length += 1
        dst.offsets_c[dst.length] = dst.char_length

    # ========================================================
    # 字符串拼接与替换
    # ========================================================
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef str _join(self, int field, str separator=""):
        cdef Py_ssize_t n = self.length
        if n == 0:
            return ""

        cdef UCS2Column_SoA src = self.text_col if field == FIELD_txt else self.text_sep_col
        if src.char_length == 0:
            return ""

        cdef Py_ssize_t sep_len = 0
        cdef Py_UCS2* sep_buf = _str_to_ucs2_buf(separator, &sep_len)

        if sep_len == 0:
            free(sep_buf)
            return PyUnicode_FromKindAndData(2, src.chars_c, src.char_length)

        cdef Py_ssize_t total_len = src.char_length + (n - 1) * sep_len
        cdef Py_ssize_t i

        cdef Py_UCS2* out_buf = <Py_UCS2*>malloc((total_len + 1) * sizeof(Py_UCS2))
        if not out_buf:
            free(sep_buf)
            raise MemoryError()

        cdef Py_ssize_t write_idx = 0
        cdef Py_ssize_t read_start, row_len

        for i in range(n - 1):
            read_start = src.offsets_c[i]
            row_len = src.offsets_c[i + 1] - read_start
            if row_len > 0:
                memcpy(out_buf + write_idx, src.chars_c + read_start, row_len * sizeof(Py_UCS2))
                write_idx += row_len
            memcpy(out_buf + write_idx, sep_buf, sep_len * sizeof(Py_UCS2))
            write_idx += sep_len

        read_start = src.offsets_c[n - 1]
        row_len = src.offsets_c[n] - read_start
        if row_len > 0:
            memcpy(out_buf + write_idx, src.chars_c + read_start, row_len * sizeof(Py_UCS2))

        cdef str res = PyUnicode_FromKindAndData(2, out_buf, total_len)
        free(sep_buf)
        free(out_buf)
        return res

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _c_replace_engine(self, int src_field, int dst_field,
                                str old0, str new0, str old1, str new1) except *:
        """
        大一统的字符串替换引擎：
        支持单/双重替换，支持同/跨字段写入。具备首字符极速拦截与防内存泄漏机制。
        """
        if self.length == 0 or not old0:
            return

        cdef Py_ssize_t o0_len = 0, n0_len = 0, o1_len = 0, n1_len = 0
        cdef Py_UCS2* o0_buf = _str_to_ucs2_buf(old0, &o0_len)
        if o0_len == 0:
            free(o0_buf)
            return

        cdef Py_UCS2* n0_buf = _str_to_ucs2_buf(new0, &n0_len)

        cdef Py_UCS2* o1_buf = NULL
        cdef Py_UCS2* n1_buf = NULL
        cdef bint is_dual = (old1 is not None and len(old1) > 0)

        if is_dual:
            o1_buf = _str_to_ucs2_buf(old1, &o1_len)
            n1_buf = _str_to_ucs2_buf(new1, &n1_len)

        cdef UCS2Column_SoA src = self.text_col if src_field == FIELD_txt else self.text_sep_col
        cdef UCS2Column_SoA res_col = UCS2Column_SoA(self.length)

        cdef Py_ssize_t i, j, k, start, end
        cdef bint match

        # 优化：提取首字符，用于滑动窗口的 O(1) 快速探针拦截
        cdef Py_UCS2 first_c0 = o0_buf[0]
        cdef Py_UCS2 first_c1 = o1_buf[0] if is_dual else 0
        cdef Py_UCS2 current_c

        try:
            for i in range(self.length):
                start = src.offsets_c[i]
                end = src.offsets_c[i+1]
                res_col.offsets_c[i] = res_col.char_length

                j = start
                while j < end:
                    current_c = src.chars_c[j]
                    match = False

                    # --- 模式 1 匹配 ---
                    # 极速探测：首字符相等 且 剩余长度足够
                    if current_c == first_c0 and end - j >= o0_len:
                        match = True
                        for k in range(1, o0_len):  # 从 1 开始，首字符已验证
                            if src.chars_c[j+k] != o0_buf[k]:
                                match = False
                                break

                    if match:
                        if res_col.char_length + n0_len > res_col.char_capacity:
                            res_col._reserve_chars(max(res_col.char_capacity * 2, res_col.char_length + n0_len + 1024))
                        if n0_len > 0:
                            memcpy(res_col.chars_c + res_col.char_length, n0_buf, n0_len * sizeof(Py_UCS2))
                        res_col.char_length += n0_len
                        j += o0_len
                        continue  # 匹配成功，跳过模式2与单字符拷贝

                    # --- 模式 2 匹配 (仅在开启双重替换时执行) ---
                    if is_dual:
                        match = False
                        if current_c == first_c1 and end - j >= o1_len:
                            match = True
                            for k in range(1, o1_len):
                                if src.chars_c[j+k] != o1_buf[k]:
                                    match = False
                                    break

                        if match:
                            if res_col.char_length + n1_len > res_col.char_capacity:
                                res_col._reserve_chars(max(res_col.char_capacity * 2, res_col.char_length + n1_len + 1024))
                            if n1_len > 0:
                                memcpy(res_col.chars_c + res_col.char_length, n1_buf, n1_len * sizeof(Py_UCS2))
                            res_col.char_length += n1_len
                            j += o1_len
                            continue

                    # --- 均未匹配，直接拷贝 1 个字符 ---
                    if res_col.char_length + 1 > res_col.char_capacity:
                        res_col._reserve_chars(res_col.char_capacity * 2)
                    res_col.chars_c[res_col.char_length] = current_c
                    res_col.char_length += 1
                    j += 1

                res_col.length += 1

            res_col.offsets_c[res_col.length] = res_col.char_length

            if dst_field == FIELD_txt:
                self.text_col = res_col
            else:
                self.text_sep_col = res_col

        finally:
            # 安全：无论是否发生内存溢出异常，都确保 C 堆内存被干净释放
            if o0_buf != NULL: free(o0_buf)
            if n0_buf != NULL: free(n0_buf)
            if o1_buf != NULL: free(o1_buf)
            if n1_buf != NULL: free(n1_buf)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _replace(self, int field, str old_str, str new_str) except *:
        self._c_replace_engine(field, field, old_str, new_str, "", "")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _replace_cross(self, int src_field, int dst_field, str old_str, str new_str) except *:
        self._c_replace_engine(src_field, dst_field, old_str, new_str, "", "")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _replace_dual(self, int field, str old0, str new0, str old1, str new1) except *:
        self._c_replace_engine(field, field, old0, new0, old1, new1)




    # ========================================================
    # 正则表达式支持
    # ========================================================
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _regex_replace(self, int field, object pattern, str rewrite):
        cdef Py_ssize_t i, n = self.length
        if n == 0:
            return

        cdef object sub = pattern.sub

        cdef UCS2Column_SoA src = self.text_col if field == FIELD_txt else self.text_sep_col
        cdef UCS2Column_SoA new_col = UCS2Column_SoA(n)

        cdef str txt, new_txt

        for i in range(n):
            if src.offsets_c[i+1] > src.offsets_c[i]:
                txt = src._get_str(i)
                new_txt = sub(rewrite, txt)
                new_col.append(new_txt)
            else:
                new_col.append("")

        if field == FIELD_txt:
            self.text_col = new_col
        else:
            self.text_sep_col = new_col


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef NDframe _df_regex_extract(self, int field, object pattern):
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n == 0:
            return new_df

        cdef Py_ssize_t i, write_idx = 0
        cdef str txt, g
        cdef object match
        cdef object search = pattern.search

        cdef RowC* src_rows = self.rows_c
        cdef RowC* dst_rows = new_df.rows_c

        cdef UCS2Column_SoA src = self.text_col if field == FIELD_txt else self.text_sep_col
        cdef UCS2Column_SoA n_txt = new_df.text_col
        cdef UCS2Column_SoA n_sep = new_df.text_sep_col

        cdef Py_ssize_t start, str_len

        for i in range(n):
            start = src.offsets_c[i]
            str_len = src.offsets_c[i+1] - start

            if str_len == 0:
                continue

            txt = src._get_str(i)
            match = search(txt)
            if match is None:
                continue

            if match.lastindex is not None and match.lastindex >= 1:
                g = match.group(1)
                if g is not None:
                    dst_rows[write_idx] = src_rows[i]

                    if field == FIELD_txt:
                        n_txt.append(g)
                        _col_copy_row(n_sep, self.text_sep_col, i)
                    else:
                        _col_copy_row(n_txt, self.text_col, i)
                        n_sep.append(g)

                    write_idx += 1

        new_df.length = write_idx
        return new_df


    # ========================================================
    # 核心过滤功能
    # ========================================================
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef NDframe _df_regex(self, int field, object pattern, bint is_regex):
        if is_regex:
            return self._df_regex_re(field, pattern)
        return self._df_regex_pure(field, pattern)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef NDframe _df_regex_re(self, int field, object pattern):
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n == 0: return new_df

        cdef Py_ssize_t i, start, end, write_idx = 0
        cdef object search_fn = pattern.search
        cdef UCS2Column_SoA src = self.text_col if field == FIELD_txt else self.text_sep_col

        for i in range(n):
            start = src.offsets_c[i]
            end = src.offsets_c[i + 1]
            if end > start and search_fn(src._get_str(i)) is not None:
                _df_copy_row(new_df, self, i, &write_idx)

        new_df.length = write_idx
        return new_df

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef NDframe _df_regex_pure(self, int field, object pattern):
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n == 0: return new_df

        cdef Py_ssize_t i, start, end, str_len, write_idx = 0
        cdef Py_ssize_t sub_len = 0
        cdef str pat_str = <str>pattern if isinstance(pattern, str) else str(pattern)
        cdef Py_UCS2* sub_buf = _str_to_ucs2_buf(pat_str, &sub_len)

        cdef UCS2Column_SoA src = self.text_col if field == FIELD_txt else self.text_sep_col
        cdef Py_UCS2* chars = src.chars_c

        for i in range(n):
            start = src.offsets_c[i]
            end = src.offsets_c[i + 1]
            str_len = end - start
            if str_len > 0 and _substr_contains(chars, start, end, sub_buf, sub_len):
                _df_copy_row(new_df, self, i, &write_idx)

        if sub_buf != NULL: free(sub_buf)
        new_df.length = write_idx
        return new_df


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef NDframe _df_regex_in_x0y0x1y1(self, int field, object pattern, int x0, int y0, int x1, int y1):
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n == 0: return new_df

        cdef Py_ssize_t i, start, end, write_idx = 0
        cdef object search_fn = pattern.search
        cdef RowC* src_rows = self.rows_c
        cdef UCS2Column_SoA src = self.text_col if field == FIELD_txt else self.text_sep_col

        for i in range(n):
            if (src_rows[i].x0 > x0 and src_rows[i].x1 < x1 and
                src_rows[i].y0 > y0 and src_rows[i].y1 < y1):
                start = src.offsets_c[i]
                end = src.offsets_c[i + 1]
                if end > start and search_fn(src._get_str(i)) is not None:
                    _df_copy_row(new_df, self, i, &write_idx)

        new_df.length = write_idx
        return new_df


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef NDframe _df_regex_strlen_cmp(self, int field, object pattern, int target_len, int op):
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n == 0: return new_df

        cdef Py_ssize_t i, start, str_len, write_idx = 0
        cdef object search_fn = pattern.search
        cdef UCS2Column_SoA src = self.text_col if field == FIELD_txt else self.text_sep_col

        for i in range(n):
            start = src.offsets_c[i]
            str_len = src.offsets_c[i + 1] - start

            if (_int_cmp_pass(str_len, op, target_len, 0) and str_len > 0
                    and search_fn(src._get_str(i)) is not None):
                _df_copy_row(new_df, self, i, &write_idx)

        new_df.length = write_idx
        return new_df


    # ========================================================
    # 纯坐标过滤
    # ========================================================
    cdef NDframe _in_bbox(self, int x0, int y0, int x1, int y1):
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n == 0: return new_df

        cdef Py_ssize_t i, write_idx = 0
        cdef RowC* src_rows = self.rows_c

        for i in range(n):
            if (src_rows[i].x0 > x0 and src_rows[i].y0 > y0 and
                src_rows[i].x1 < x1 and src_rows[i].y1 < y1):
                _df_copy_row(new_df, self, i, &write_idx)

        new_df.length = write_idx
        return new_df

    cdef NDframe _in_x_range(self, int x0, int x1):
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n == 0:
            return new_df

        cdef Py_ssize_t i, write_idx = 0
        cdef RowC* src_rows = self.rows_c

        for i in range(n):
            if src_rows[i].x0 > x0 and src_rows[i].x1 < x1:
                _df_copy_row(new_df, self, i, &write_idx)
        new_df.length = write_idx
        return new_df

    cdef NDframe _in_y_range(self, int y0, int y1):
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n == 0:
            return new_df

        cdef Py_ssize_t i, write_idx = 0
        cdef RowC* src_rows = self.rows_c

        for i in range(n):
            if src_rows[i].y0 > y0 and src_rows[i].y1 < y1:
                _df_copy_row(new_df, self, i, &write_idx)
        new_df.length = write_idx
        return new_df

    cdef NDframe _greater_point(self, int x0_val, int y0_val):
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n == 0:
            return new_df

        cdef Py_ssize_t i, write_idx = 0
        cdef RowC* src_rows = self.rows_c

        for i in range(n):
            if src_rows[i].x0 > x0_val and src_rows[i].y0 > y0_val:
                _df_copy_row(new_df, self, i, &write_idx)
        new_df.length = write_idx
        return new_df

    cdef NDframe _less_point(self, int x1_val, int y1_val):
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n == 0:
            return new_df

        cdef Py_ssize_t i, write_idx = 0
        cdef RowC* src_rows = self.rows_c

        for i in range(n):
            if src_rows[i].x1 < x1_val and src_rows[i].y1 < y1_val:
                _df_copy_row(new_df, self, i, &write_idx)
        new_df.length = write_idx
        return new_df

    # ========================================================
    # 单字段整数值过滤
    # ========================================================
    cdef NDframe _df_filter_by_int(self, int field, int arg1, int arg2, int op):
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n == 0:
            return new_df

        cdef Py_ssize_t i, write_idx = 0
        cdef RowC* src_rows = self.rows_c

        for i in range(n):
            if _int_cmp_pass(_get_field_val(&src_rows[i], field), op, arg1, arg2):
                _df_copy_row(new_df, self, i, &write_idx)
        new_df.length = write_idx
        return new_df

    # ========================================================
    # 字符串长度过滤
    # ========================================================
    cdef NDframe _df_strlen_cmp(self, int field, int th, int op):
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n == 0:
            return new_df

        cdef Py_ssize_t i, write_idx = 0
        cdef UCS2Column_SoA src = self.text_col if field == FIELD_txt else self.text_sep_col

        for i in range(n):
            if _int_cmp_pass(src.offsets_c[i + 1] - src.offsets_c[i], op, th, 0):
                _df_copy_row(new_df, self, i, &write_idx)
        new_df.length = write_idx
        return new_df

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef list _strlen_list(self, int field):
        cdef Py_ssize_t n = self.length
        if n == 0:
            return []
        cdef list result = [0] * n
        cdef Py_ssize_t i
        cdef UCS2Column_SoA src = self.text_col if field == FIELD_txt else self.text_sep_col
        for i in range(n):
            result[i] = PyLong_FromSsize_t(src.offsets_c[i + 1] - src.offsets_c[i])
        return result

    cdef NDframe _df_str_equal(self, int field, str s):
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n == 0:
            return new_df

        cdef Py_ssize_t s_len = 0
        cdef Py_UCS2* s_buf = _str_to_ucs2_buf(s, &s_len)

        cdef Py_ssize_t i, start, str_len, write_idx = 0
        cdef UCS2Column_SoA src = self.text_col if field == FIELD_txt else self.text_sep_col

        for i in range(n):
            start = src.offsets_c[i]
            str_len = src.offsets_c[i + 1] - start
            if (str_len == s_len and
                    (s_len == 0 or memcmp(src.chars_c + start, s_buf, s_len * sizeof(Py_UCS2)) == 0)):
                _df_copy_row(new_df, self, i, &write_idx)

        free(s_buf)
        new_df.length = write_idx
        return new_df

    cdef NDframe _df_is_in(self, int field, object texts):
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n == 0:
            return new_df

        cdef set text_set = set(texts)
        cdef Py_ssize_t i, write_idx = 0
        cdef UCS2Column_SoA src = self.text_col if field == FIELD_txt else self.text_sep_col

        for i in range(n):
            if src._get_str(i) in text_set:
                _df_copy_row(new_df, self, i, &write_idx)

        new_df.length = write_idx
        return new_df

    cdef NDframe _df_not_in(self, int field, object texts):
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n == 0:
            return new_df

        cdef set text_set = set(texts)
        cdef Py_ssize_t i, write_idx = 0
        cdef UCS2Column_SoA src = self.text_col if field == FIELD_txt else self.text_sep_col

        for i in range(n):
            if src._get_str(i) not in text_set:
                _df_copy_row(new_df, self, i, &write_idx)
        new_df.length = write_idx
        return new_df


    cdef bint _contains_any(self, int field, object texts):
        cdef Py_ssize_t i, n = self.length
        if n == 0:
            return False

        cdef UCS2Column_SoA src = self.text_col if field == FIELD_txt else self.text_sep_col

        cdef str txt, t
        cdef tuple text_tuple = tuple(texts) if not isinstance(texts, tuple) else texts
        cdef Py_ssize_t m = len(text_tuple)

        for i in range(n):
            txt = src._get_str(i)
            if not txt:
                continue
            for j in range(m):
                t = text_tuple[j]
                if t in txt:
                    return True
        return False


    cdef void _regex_replace_cross(self, int src_field, int dst_field, object pattern, str rewrite) except *:
        cdef Py_ssize_t i, n = self.length
        if n == 0:
            return
        cdef UCS2Column_SoA src = self.text_col if src_field == FIELD_txt else self.text_sep_col
        cdef UCS2Column_SoA new_col = UCS2Column_SoA(n)
        cdef object sub = pattern.sub
        cdef str txt, new_txt

        for i in range(n):
            if src.offsets_c[i+1] > src.offsets_c[i]:
                txt = src._get_str(i)
                new_txt = sub(rewrite, txt)
                new_col.append(new_txt)
            else:
                new_col.append("")

        if dst_field == FIELD_txt:
            self.text_col = new_col
        else:
            self.text_sep_col = new_col

    # ========================================================
    # 排序
    # ========================================================
    cdef void _sort_col(self, int field_id, bint descending) except *:
        cdef Py_ssize_t n = self.length
        if n < 2: return

        if field_id == FIELD_txt or field_id == FIELD_sep:
            self._sort_col_str(field_id, descending)
            return

        if field_id < FIELD_idx or field_id > FIELD_flags:
            raise ValueError(f"Unsupported sort field id: {field_id}")

        cdef Py_ssize_t i
        cdef RowC* rows = self.rows_c
        cdef list pairs = [None] * n

        for i in range(n):
            pairs[i] = (_get_field_val(&rows[i], field_id), i)

        pairs.sort(reverse=descending)

        cdef Py_ssize_t* order = <Py_ssize_t*>malloc(n * sizeof(Py_ssize_t))
        if not order: raise MemoryError()
        try:
            for i in range(n):
                order[i] = pairs[i][1]
            self._reorder_rows(order)
        finally:
            free(order)

    cdef void _sort_col_str(self, int field_id, bint descending) except *:
        cdef Py_ssize_t n = self.length
        cdef UCS2Column_SoA col = self.text_col if field_id == FIELD_txt else self.text_sep_col

        cdef Py_ssize_t* order = <Py_ssize_t*>malloc(n * sizeof(Py_ssize_t))
        cdef Py_ssize_t* tmp = <Py_ssize_t*>malloc(n * sizeof(Py_ssize_t))

        if not order or not tmp:
            if order: free(order)
            if tmp: free(tmp)
            raise MemoryError()

        try:
            for i in range(n): order[i] = i
            _mergesort_str(order, tmp, 0, n, col.chars_c, col.offsets_c, descending)
            self._reorder_rows(order)
        finally:
            free(order)
            free(tmp)

    cdef void _reorder_rows(self, Py_ssize_t* indices) except *:
        """纯 C 指针重排，0 Python 对象开销"""
        cdef Py_ssize_t n = self.length
        cdef RowC* rows = self.rows_c

        cdef RowC* temp_rows = <RowC*>malloc(<size_t>n * sizeof(RowC))
        if not temp_rows: raise MemoryError()

        try:
            for i in range(n):
                temp_rows[i] = rows[indices[i]]
            memcpy(rows, temp_rows, <size_t>n * sizeof(RowC))
            self.text_col._reorder(indices)
            self.text_sep_col._reorder(indices)
        finally:
            free(temp_rows)

    @staticmethod
    def concat(NDframe df1, NDframe df2):
        cdef Py_ssize_t len1 = df1.length
        cdef Py_ssize_t len2 = df2.length
        cdef Py_ssize_t total = len1 + len2
        cdef NDframe result = NDframe(total if total > 0 else 1)

        if total == 0:
            return result

        memcpy(result.rows_c, df1.rows_c, len1 * sizeof(RowC))
        memcpy(result.rows_c + len1, df2.rows_c, len2 * sizeof(RowC))

        result.text_col = df1.text_col.copy()
        result.text_col.concat(df2.text_col)
        result.text_sep_col = df1.text_sep_col.copy()
        result.text_sep_col.concat(df2.text_sep_col)

        result.length = total
        return result

    # ========================================================
    # 就地算术操作
    # ========================================================
    cdef void _add_col(self, int field, int value) except *:
        cdef Py_ssize_t i, n = self.length
        if n == 0 or value == 0 or field < FIELD_idx or field > FIELD_flags:
            return
        cdef RowC* rows = self.rows_c
        for i in range(n):
            _add_field_value(&rows[i], field, value)


    cdef void _sub_col(self, int field, int value) except *:
        cdef Py_ssize_t i, n = self.length
        if n == 0 or value == 0 or field < FIELD_idx or field > FIELD_flags:
            return
        cdef RowC* rows = self.rows_c
        cdef int* f
        cdef int v
        for i in range(n):
            f = _row_field_ptr(&rows[i], field)
            v = f[0]
            f[0] = v - value if v > value else 0

    cdef NDframe _df_drop(self, int index):
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n == 0:
            return new_df

        cdef Py_ssize_t i, write_idx = 0
        for i in range(n):
            if i != index:
                _df_copy_row(new_df, self, i, &write_idx)

        new_df.length = write_idx
        return new_df

    cdef NDframe _df_drops(self, object idx_set):
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n == 0:
            return new_df

        cdef set idx_set_c = set(idx_set)
        cdef Py_ssize_t i, write_idx = 0

        for i in range(n):
            if i not in idx_set_c:
                _df_copy_row(new_df, self, i, &write_idx)

        new_df.length = write_idx
        return new_df

    cdef void _apply_if_greater_add(self, int cond_id, int target_id, int threshold, int offset) except *:
        """如果 cond_id > threshold，则 target_id += offset"""
        cdef Py_ssize_t i, n = self.length
        if n == 0 or offset == 0 or cond_id < FIELD_idx or cond_id > FIELD_flags:
            return
        cdef RowC* rows = self.rows_c
        for i in range(n):
            if _get_field_val(&rows[i], cond_id) > threshold:
                _add_field_value(&rows[i], target_id, offset)

    cdef void _apply_if_less_add(self, int cond_id, int target_id, int threshold, int offset) except *:
        """如果 cond_id < threshold，则 target_id += offset"""
        cdef Py_ssize_t i, n = self.length
        if n == 0 or offset == 0 or cond_id < FIELD_idx or cond_id > FIELD_flags:
            return
        cdef RowC* rows = self.rows_c
        for i in range(n):
            if _get_field_val(&rows[i], cond_id) < threshold:
                _add_field_value(&rows[i], target_id, offset)

    cdef void _apply_if_greater_add_y(self, int cond_id, int threshold, int offset) except *:
        """如果 cond_id > threshold，则 y0 += offset, y1 += offset"""
        cdef Py_ssize_t i, n = self.length
        if n == 0 or offset == 0 or cond_id < FIELD_idx or cond_id > FIELD_flags:
            return
        cdef RowC* rows = self.rows_c
        for i in range(n):
            if _get_field_val(&rows[i], cond_id) > threshold:
                rows[i].y0 += offset
                rows[i].y1 += offset

    cdef void _apply_if_less_add_y(self, int cond_id, int threshold, int offset) except *:
        """如果 cond_id < threshold，则 y0 += offset, y1 += offset"""
        cdef Py_ssize_t i, n = self.length
        if n == 0 or offset == 0 or cond_id < FIELD_idx or cond_id > FIELD_flags:
            return
        cdef RowC* rows = self.rows_c
        for i in range(n):
            if _get_field_val(&rows[i], cond_id) < threshold:
                rows[i].y0 += offset
                rows[i].y1 += offset


    cdef list _col_values(self, int field):
        cdef Py_ssize_t n = self.length
        cdef Py_ssize_t i
        cdef RowC* rows = self.rows_c
        cdef list result
        if field == FIELD_txt:
            result = [None] * n
            for i in range(n):
                result[i] = self.text_col._get_str(i)
            return result
        elif field == FIELD_sep:
            result = [None] * n
            for i in range(n):
                result[i] = self.text_sep_col._get_str(i)
            return result
        else:
            result = [0] * n
            for i in range(n):
                result[i] = _get_field_val(&rows[i], field)
            return result


    # ========================================================
    # 就地正则删除
    # ========================================================
    cdef NDframe _drop_in_regex(self, int field, object pattern):
        cdef Py_ssize_t read_idx
        cdef Py_ssize_t write_idx = 0
        cdef Py_ssize_t n = self.length
        cdef str txt
        cdef object search_fn = pattern.search

        cdef RowC* rows = self.rows_c
        cdef UCS2Column_SoA src = self.text_col if field == FIELD_txt else self.text_sep_col
        cdef UCS2Column_SoA n_txt = UCS2Column_SoA(n)
        cdef UCS2Column_SoA n_sep = UCS2Column_SoA(n)

        for read_idx in range(n):
            txt = src._get_str(read_idx)
            if not (txt and search_fn(txt) is not None):
                rows[write_idx] = rows[read_idx]
                _col_copy_row(n_txt, self.text_col, read_idx)
                _col_copy_row(n_sep, self.text_sep_col, read_idx)
                write_idx += 1

        self.length = write_idx
        self.text_col = n_txt
        self.text_sep_col = n_sep
        return self

    # ========================================================
    # 返回索引列表 (idx_xxx 核心实现)
    # ========================================================
    cdef list _idx_regex(self, int field, object pattern):
        cdef Py_ssize_t n = self.length
        cdef list res = []
        cdef Py_ssize_t i
        cdef str txt
        cdef object search_fn = pattern.search
        cdef UCS2Column_SoA src = self.text_col if field == FIELD_txt else self.text_sep_col

        for i in range(n):
            txt = src._get_str(i)
            if txt and search_fn(txt) is not None:
                res.append(i)
        return res

    cdef list _idx_regex_in_bbox(self, int field, object pattern, int x0, int y0, int x1, int y1):
        cdef Py_ssize_t n = self.length
        cdef list res = []
        cdef Py_ssize_t i
        cdef str txt
        cdef object search_fn = pattern.search
        cdef RowC* rows = self.rows_c
        cdef UCS2Column_SoA src = self.text_col if field == FIELD_txt else self.text_sep_col

        for i in range(n):
            if (rows[i].x0 > x0 and rows[i].x1 < x1 and
                rows[i].y0 > y0 and rows[i].y1 < y1):
                txt = src._get_str(i)
                if txt and search_fn(txt) is not None:
                    res.append(i)
        return res

    cdef list _idx_strlen_cmp(self, int field, int n, int op):
        cdef Py_ssize_t length = self.length
        cdef list res = []
        cdef Py_ssize_t i
        cdef UCS2Column_SoA src = self.text_col if field == FIELD_txt else self.text_sep_col

        for i in range(length):
            if _int_cmp_pass(src.offsets_c[i + 1] - src.offsets_c[i], op, n, 0):
                res.append(i)
        return res

    cdef list _idx_regex_strlen_cmp(self, int field, object pattern, int n, int op):
        cdef Py_ssize_t length = self.length
        cdef list res = []
        cdef Py_ssize_t i
        cdef str txt
        cdef object search_fn = pattern.search
        cdef UCS2Column_SoA src = self.text_col if field == FIELD_txt else self.text_sep_col

        for i in range(length):
            txt = src._get_str(i)
            if (txt and _int_cmp_pass(PyUnicode_GET_LENGTH(txt), op, n, 0)
                    and search_fn(txt) is not None):
                res.append(i)
        return res

    cdef list _idx_filter_by_int(self, int field, int arg1, int arg2, int op):
        cdef Py_ssize_t length = self.length
        cdef list res = []
        cdef Py_ssize_t i
        cdef RowC* rows = self.rows_c

        for i in range(length):
            if _int_cmp_pass(_get_field_val(&rows[i], field), op, arg1, arg2):
                res.append(i)
        return res


    # ========================================================
    # 删除 / 合并
    # ========================================================
    cdef NDframe _drop(self, int index):
        cdef Py_ssize_t n = self.length
        if n == 0 or index < 0 or index >= n:
            return self

        cdef Py_ssize_t move_count = n - index - 1
        cdef RowC* rows = self.rows_c
        if move_count > 0:
            memmove(&rows[index], &rows[index + 1], move_count * sizeof(RowC))

        self.text_col._remove_at(index)
        self.text_sep_col._remove_at(index)
        self.length = n - 1
        return self

    cdef NDframe _drop_in_bbox(self, int x0, int y0, int x1, int y1):
        cdef Py_ssize_t n = self.length
        if n == 0:
            return self

        cdef Py_ssize_t read_idx
        cdef Py_ssize_t write_idx = 0
        cdef RowC* rows = self.rows_c
        cdef UCS2Column_SoA n_txt = UCS2Column_SoA(n)
        cdef UCS2Column_SoA n_sep = UCS2Column_SoA(n)

        for read_idx in range(n):
            if not (rows[read_idx].x0 > x0 and rows[read_idx].y0 > y0 and
                    rows[read_idx].x1 < x1 and rows[read_idx].y1 < y1):
                rows[write_idx] = rows[read_idx]
                _col_copy_row(n_txt, self.text_col, read_idx)
                _col_copy_row(n_sep, self.text_sep_col, read_idx)
                write_idx += 1

        self.length = write_idx
        self.text_col = n_txt
        self.text_sep_col = n_sep
        return self

    cdef NDframe _merge_line(self, int width_threshold, int height_threshold, str sep=""):
        cdef Py_ssize_t n = self.length
        if n == 0: return NDframe(1)

        cdef NDframe result = NDframe(n)
        cdef RowC* src_rows = self.rows_c
        cdef RowC* dst_rows = result.rows_c
        cdef UCS2Column_SoA dst_txt = result.text_col
        cdef UCS2Column_SoA dst_sep = result.text_sep_col
        cdef UCS2Column_SoA src_txt = self.text_col
        cdef UCS2Column_SoA src_sep = self.text_sep_col

        cdef Py_ssize_t sep_len = 0
        cdef Py_UCS2* sep_buf = _str_to_ucs2_buf(sep, &sep_len)

        cdef Py_ssize_t i = 0, curr, write_idx = 0
        cdef int min_x0, min_y0, max_x1, max_y1, max_height, new_height
        cdef RowC row

        # 临时缓冲，避免在循环中不断去触碰 dst 的内存伸缩
        cdef Py_ssize_t t_len, s_len, start, slen, part_idx
        cdef Py_UCS2* tmp_buf = <Py_UCS2*>malloc(1024 * 10 * sizeof(Py_UCS2)) # 10K UCS2 缓冲
        if not tmp_buf:
            free(sep_buf)
            raise MemoryError()

        try:
            while i < n:
                min_x0 = src_rows[i].x0; min_y0 = src_rows[i].y0
                max_x1 = src_rows[i].x1; max_y1 = src_rows[i].y1
                max_height = src_rows[i].height
                curr = i

                while curr + 1 < n:
                    if (src_rows[curr + 1].x0 - src_rows[curr].x1 < width_threshold and
                        src_rows[curr + 1].y0 - src_rows[curr].y0 < height_threshold and
                        src_rows[curr + 1].x0 - src_rows[curr].x1 > -width_threshold and
                        src_rows[curr + 1].y0 - src_rows[curr].y0 > -height_threshold):

                        if src_rows[curr + 1].x0 < min_x0: min_x0 = src_rows[curr + 1].x0
                        if src_rows[curr + 1].y0 < min_y0: min_y0 = src_rows[curr + 1].y0
                        if src_rows[curr + 1].x1 > max_x1: max_x1 = src_rows[curr + 1].x1
                        if src_rows[curr + 1].y1 > max_y1: max_y1 = src_rows[curr + 1].y1
                        if src_rows[curr + 1].height > max_height: max_height = src_rows[curr + 1].height
                        curr += 1
                    else:
                        break

                new_height = max_y1 - min_y0
                if new_height < max_height: new_height = max_height

                row.idx = <int>write_idx
                row.x0 = min_x0; row.y0 = min_y0; row.x1 = max_x1; row.y1 = max_y1
                row.width = max_x1 - min_x0
                row.height = new_height
                row.page = src_rows[i].page
                row.flags = src_rows[i].flags

                # --- 纯 C 内存级别组装 text ---
                t_len = 0
                for part_idx in range(i, curr + 1):
                    start = src_txt.offsets_c[part_idx]
                    slen = src_txt.offsets_c[part_idx + 1] - start
                    if slen > 0:
                        memcpy(tmp_buf + t_len, src_txt.chars_c + start, slen * sizeof(Py_UCS2))
                        t_len += slen
                    if part_idx < curr and sep_len > 0:
                        memcpy(tmp_buf + t_len, sep_buf, sep_len * sizeof(Py_UCS2))
                        t_len += sep_len

                # 绕过 append 直接写入 dst_txt，避免构造 Python 字符串
                if dst_txt.length >= dst_txt.capacity: dst_txt._reserve_rows(dst_txt.capacity * 2)
                if dst_txt.char_length + t_len > dst_txt.char_capacity:
                    dst_txt._reserve_chars(dst_txt.char_length + t_len + 1024)
                if t_len > 0: memcpy(dst_txt.chars_c + dst_txt.char_length, tmp_buf, t_len * sizeof(Py_UCS2))
                dst_txt.char_length += t_len
                dst_txt.length += 1
                dst_txt.offsets_c[dst_txt.length] = dst_txt.char_length

                # --- 纯 C 内存级别组装 sep ---
                s_len = 0
                for part_idx in range(i, curr + 1):
                    start = src_sep.offsets_c[part_idx]
                    slen = src_sep.offsets_c[part_idx + 1] - start
                    if slen > 0:
                        memcpy(tmp_buf + s_len, src_sep.chars_c + start, slen * sizeof(Py_UCS2))
                        s_len += slen
                    if part_idx < curr and sep_len > 0:
                        memcpy(tmp_buf + s_len, sep_buf, sep_len * sizeof(Py_UCS2))
                        s_len += sep_len

                if dst_sep.length >= dst_sep.capacity: dst_sep._reserve_rows(dst_sep.capacity * 2)
                if dst_sep.char_length + s_len > dst_sep.char_capacity:
                    dst_sep._reserve_chars(dst_sep.char_length + s_len + 1024)
                if s_len > 0: memcpy(dst_sep.chars_c + dst_sep.char_length, tmp_buf, s_len * sizeof(Py_UCS2))
                dst_sep.char_length += s_len
                dst_sep.length += 1
                dst_sep.offsets_c[dst_sep.length] = dst_sep.char_length

                dst_rows[write_idx] = row
                write_idx += 1
                i = curr + 1

            result.length = write_idx
        finally:
            if tmp_buf != NULL: free(tmp_buf)
            if sep_buf != NULL: free(sep_buf)

        return result

    cdef list _diff_col(self, int field):
        cdef Py_ssize_t n = self.length
        if n == 0 or field < FIELD_idx or field > FIELD_flags:
            return []

        cdef RowC* rows = self.rows_c
        cdef list res = [0] * n
        cdef Py_ssize_t i
        cdef int prev, curr
        prev = _get_field_val(&rows[0], field)
        for i in range(1, n):
            curr = _get_field_val(&rows[i], field)
            res[i] = curr - prev
            prev = curr

        res[0] = None
        return res


    cdef double _var_col(self, int field, int ddof):
        cdef Py_ssize_t n = self.length
        if n <= ddof or field < FIELD_idx or field > FIELD_flags:
            return 0.0

        cdef RowC* rows = self.rows_c
        cdef double mean_val = self._mean_col(field)
        cdef double sum_sq = 0.0
        cdef double d_val
        cdef Py_ssize_t i

        for i in range(n):
            d_val = _get_field_val(&rows[i], field) - mean_val
            sum_sq += d_val * d_val

        return sum_sq / (n - ddof)


    cdef double _std_col(self, int field, int ddof):
        cdef double variance = self._var_col(field, ddof)
        return sqrt(variance)


    def centroid(self):
        cdef Py_ssize_t n = self.length
        if n == 0:
            return (0.0, 0.0)

        cdef RowC* rows = self.rows_c
        cdef double sum_x = 0.0
        cdef double sum_y = 0.0
        cdef Py_ssize_t i

        for i in range(n):
            sum_x += (rows[i].x0 + rows[i].x1) * 0.5
            sum_y += (rows[i].y0 + rows[i].y1) * 0.5

        cdef double inv_n = 1.0 / <double>n
        return (sum_x * inv_n, sum_y * inv_n)


    def centroid_line(self, n):
        """
        返回第 n 行矩形的中心点坐标 (x_center, y_center)。
        支持负索引。
        """
        cdef Py_ssize_t idx = n
        cdef Py_ssize_t length = self.length
        if idx < 0:
            idx += length
        if idx < 0 or idx >= length:
            raise IndexError(f"Index {n} out of range for NDframe with {length} rows")
        cdef RowC* row = &self.rows_c[idx]
        return ((row.x0 + row.x1) * 0.5, (row.y0 + row.y1) * 0.5)

    cdef (int, int) _mode_col(self, int field):
        """
        计算数值字段的众数，返回 (mode_value, count)
        直接遍历 C 数组，避免创建临时列表。
        """
        cdef Py_ssize_t n = self.length
        if n == 0 or field < FIELD_idx or field > FIELD_flags:
            return (0, 0)

        cdef RowC* rows = self.rows_c
        cdef dict freq = {}
        cdef int val
        cdef Py_ssize_t i

        for i in range(n):
            val = _get_field_val(&rows[i], field)
            freq[val] = freq.get(val, 0) + 1

        cdef int max_count = 0
        cdef int mode_val = 0
        for val, cnt in freq.items():
            if cnt > max_count or (cnt == max_count and val < mode_val):
                max_count = cnt
                mode_val = val
        return (mode_val, max_count)

    cdef tuple _mode_height(self):
        """
        专门针对 Height 字段的极致优化：
        利用业务先验 (10 <= height <= 250)，使用 C 数组替代 Python Dict 进行桶排序统计。
        复杂度：时间 O(N)，空间 O(1) (仅占用栈上几十字节内存)。
        """
        if self.length == 0:
            return (0, 0)

        cdef int freq[256]
        cdef int i, h

        for i in range(256):
            freq[i] = 0

        for i in range(self.length):
            h = self.rows_c[i].height
            if 10 <= h <= 250:
                freq[h] += 1

        cdef int max_count = 0
        cdef int mode_val = 0

        for i in range(10, 251):
            if freq[i] > max_count:
                max_count = freq[i]
                mode_val = i

        return (mode_val, max_count)

    # ========================================================
    # 全局坐标 IDX 提取系列
    # ========================================================
    def idx_in_bbox(self, int x0, int y0, int x1, int y1):
        if x0 > x1 or y0 > y1:
            raise ValueError("Invalid x0 y0 x1 y1 values")
        if x0 < 0: x0 = 0
        if x1 < 0: x1 = 0
        if y0 < 0: y0 = 0
        if y1 < 0: y1 = 0

        cdef Py_ssize_t n = self.length
        if n == 0:
            return []
        cdef RowC* rows = self.rows_c
        cdef list res = []
        cdef Py_ssize_t i

        for i in range(n):
            if (rows[i].x0 > x0 and rows[i].y0 > y0 and
                rows[i].x1 < x1 and rows[i].y1 < y1):
                res.append(i)
        return res


    def idx_in_x0x1(self, int x0, int x1):
        if x0 > x1:
            raise ValueError("Invalid x0 and x1 values")
        if x0 < 0: x0 = 0
        if x1 < 0: x1 = 0

        cdef Py_ssize_t n = self.length
        if n == 0:
            return []
        cdef RowC* rows = self.rows_c
        cdef list res = []
        cdef Py_ssize_t i

        for i in range(n):
            if rows[i].x0 > x0 and rows[i].x1 < x1:
                res.append(i)
        return res


    def idx_in_y0y1(self, int y0, int y1):
        if y0 > y1:
            raise ValueError("Invalid y0 or y1")
        if y0 < 0: y0 = 0
        if y1 < 0: y1 = 0

        cdef Py_ssize_t n = self.length
        if n == 0:
            return []
        cdef RowC* rows = self.rows_c
        cdef list res = []
        cdef Py_ssize_t i

        for i in range(n):
            if rows[i].y0 > y0 and rows[i].y1 < y1:
                res.append(i)
        return res

    # ========================================================
    # 基于 IDX 的重建与批量删除
    # ========================================================
    def df_from_idx(self, indices):
        if len(indices) == 0:
            return NDframe(1)

        cdef list idx_list = list(indices)
        cdef Py_ssize_t n_idx = len(idx_list)
        cdef NDframe new_df = NDframe(n_idx if n_idx > 0 else 1)

        cdef RowC* src_rows = self.rows_c
        cdef RowC* dst_rows = new_df.rows_c
        cdef UCS2Column_SoA n_txt = new_df.text_col
        cdef UCS2Column_SoA n_sep = new_df.text_sep_col

        cdef Py_ssize_t i, target
        cdef Py_ssize_t n_self = self.length

        for i in range(n_idx):
            target = <Py_ssize_t>int(idx_list[i])
            if target < 0 or target >= n_self:
                raise IndexError(f"Index {target} out of range [0, {n_self})")
            dst_rows[i] = src_rows[target]
            _col_copy_row(n_txt, self.text_col, target)
            _col_copy_row(n_sep, self.text_sep_col, target)

        new_df.length = n_idx
        return new_df


    def drop_by_idx(self, indices):
        if not indices:
            return self

        cdef set idx_set = set(indices)
        cdef Py_ssize_t read_idx = 0
        cdef Py_ssize_t write_idx = 0
        cdef Py_ssize_t n = self.length
        cdef RowC* rows = self.rows_c
        cdef UCS2Column_SoA n_txt = UCS2Column_SoA(n)
        cdef UCS2Column_SoA n_sep = UCS2Column_SoA(n)

        for read_idx in range(n):
            if read_idx not in idx_set:
                rows[write_idx] = rows[read_idx]
                _col_copy_row(n_txt, self.text_col, read_idx)
                _col_copy_row(n_sep, self.text_sep_col, read_idx)
                write_idx += 1

        self.length = write_idx
        self.text_col = n_txt
        self.text_sep_col = n_sep
        return self


    # ========================================================
    # 纯坐标过滤 — 带参数校验的 def 包装
    # ========================================================
    def filter_in_bbox(self, int x0, int y0, int x1, int y1):
        if x0 < 0: x0 = 0
        if x1 < 0: x1 = 0
        if y0 < 0: y0 = 0
        if y1 < 0: y1 = 0

        if x0 > x1 or y0 > y1:
            raise ValueError(f"def filter_in_bbox invalid boundary: x0({x0})>x1({x1}) or y0({y0})>y1({y1})")
        return self._in_bbox(x0, y0, x1, y1)

    def filter_in_x0x1(self, int x0, int x1):
        if x0 < 0: x0 = 0
        if x1 < 0: x1 = 0
        if x0 > x1:
            raise ValueError(f"x0({x0}) > x1({x1})")
        return self._in_x_range(x0, x1)

    def filter_in_y0y1(self, int y0, int y1):
        if y0 < 0: y0 = 0
        if y1 < 0: y1 = 0
        if y0 > y1:
            raise ValueError(f"y0({y0}) > y1({y1})")
        return self._in_y_range(y0, y1)

    def filter_greater_point(self, int x0, int y0):
        if x0 < 0: x0 = 0
        if y0 < 0: y0 = 0
        return self._greater_point(x0, y0)

    def filter_less_point(self, int x1, int y1):
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        return self._less_point(x1, y1)

    # ========================================================
    # 排序
    # ========================================================
    def sort_by_x0(self, bint descending=False):
        self._sort_col(FIELD_x0, descending)
        return self

    def sort_by_y0(self, bint descending=False):
        self._sort_col(FIELD_y0, descending)
        return self

    def sort_by_x1(self, bint descending=False):
        self._sort_col(FIELD_x1, descending)
        return self

    def sort_by_y1(self, bint descending=False):
        self._sort_col(FIELD_y1, descending)
        return self

    def sort_by_width(self, bint descending=False):
        self._sort_col(FIELD_width, descending)
        return self

    def sort_by_height(self, bint descending=False):
        self._sort_col(FIELD_height, descending)
        return self

    def sort_by_idx(self, bint descending=False):
        self._sort_col(FIELD_idx, descending)
        return self

    def sort_by_page(self, bint descending=False):
        self._sort_col(FIELD_page, descending)
        return self

    def sort_by_flags(self, bint descending=False):
        self._sort_col(FIELD_flags, descending)
        return self

    def sort_by_text(self, bint descending=False):
        """按 text 字段 UCS2 码元字典序就地排序（稳定），返回原 df。
        示例: df.sort_by_text().text.values()"""
        self._sort_col(FIELD_txt, descending)
        return self

    def sort_by_text_sep(self, bint descending=False):
        """按 text_sep 字段 UCS2 码元字典序就地排序（稳定），返回原 df"""
        self._sort_col(FIELD_sep, descending)
        return self

    cdef int _str_to_field_id(self, str field) except? -1:
        fname = field.lower()
        if fname == "x0": return FIELD_x0
        if fname == "y0": return FIELD_y0
        if fname == "x1": return FIELD_x1
        if fname == "y1": return FIELD_y1
        if fname == "width": return FIELD_width
        if fname == "height": return FIELD_height
        if fname == "idx": return FIELD_idx
        if fname == "page": return FIELD_page
        if fname == "flags": return FIELD_flags
        if fname == "text" or fname == "txt": return FIELD_txt
        if fname == "text_sep" or fname == "sep": return FIELD_sep
        return -1

    def sort_by(self, str field="x0", bint descending=False):
        cdef int field_id = self._str_to_field_id(field)
        if field_id < 0:
            raise ValueError(f"Unsupported sort field: {field}")
        self._sort_col(field_id, descending)
        return self

    # ========================================================
    # 索引 / Y轴 / 删除 / 合并
    # ========================================================
    @cython.always_allow_keywords(False)
    def reset_index(self):
        cdef Py_ssize_t i
        for i in range(self.length):
            self.rows_c[i].idx = <int>i
        return self

    def add_y0y1(self, int offset):
        cdef Py_ssize_t i
        for i in range(self.length):
            self.rows_c[i].y0 += offset
            self.rows_c[i].y1 += offset
        return self

    def apply_if_greater_add(self, str cond_field, str target_field, int threshold, int offset):
        cdef int cond_id = self._str_to_field_id(cond_field)
        cdef int target_id = self._str_to_field_id(target_field)
        if cond_id < 0 or target_id < 0 or cond_id >= FIELD_txt or target_id >= FIELD_txt:
            raise ValueError("Invalid numeric fields")
        self._apply_if_greater_add(cond_id, target_id, threshold, offset)
        return self

    def apply_if_less_add(self, str cond_field, str target_field, int threshold, int offset):
        cdef int cond_id = self._str_to_field_id(cond_field)
        cdef int target_id = self._str_to_field_id(target_field)
        if cond_id < 0 or target_id < 0 or cond_id >= FIELD_txt or target_id >= FIELD_txt:
            raise ValueError("Invalid numeric fields")
        self._apply_if_less_add(cond_id, target_id, threshold, offset)
        return self

    def apply_if_greater_add_y(self, str cond_field, int threshold, int offset):
        cdef int cond_id = self._str_to_field_id(cond_field)
        if cond_id < 0 or cond_id >= FIELD_txt:
            raise ValueError("Invalid numeric fields")
        self._apply_if_greater_add_y(cond_id, threshold, offset)
        return self

    def apply_if_less_add_y(self, str cond_field, int threshold, int offset):
        cdef int cond_id = self._str_to_field_id(cond_field)
        if cond_id < 0 or cond_id >= FIELD_txt:
            raise ValueError("Invalid numeric fields")
        self._apply_if_less_add_y(cond_id, threshold, offset)
        return self

    def df_drops(self, indices):
        cdef set idx_set = set(indices)
        return self._df_drops(idx_set)

    def df_drop(self, index):
        cdef int c_idx = int(index)
        if c_idx < 0 or c_idx >= self.length:
            raise IndexError(f"Index {c_idx} out of range [0, {self.length})")
        return self._df_drop(c_idx)

    def drop(self, index):
        cdef int c_idx = int(index)
        if c_idx < 0 or c_idx >= self.length:
            raise IndexError(f"Index {c_idx} out of range [0, {self.length})")
        return self._drop(c_idx)

    def drop_in_bbox(self, int x0, int y0, int x1, int y1):
        if x0 < 0: x0 = 0
        if x1 < 0: x1 = 0
        if y0 < 0: y0 = 0
        if y1 < 0: y1 = 0

        if x0 > x1 or y0 > y1:
            raise ValueError(f"def drop_in_bbox invalid boundary: x0({x0})>x1({x1}) or y0({y0})>y1({y1})")
        return self._drop_in_bbox(x0, y0, x1, y1)

    def merge_line(self, int width_threshold, int height_threshold, str sep=""):
        if width_threshold < 0: width_threshold = 0
        if height_threshold < 0: height_threshold = 0
        return self._merge_line(width_threshold, height_threshold, sep)


cdef void _df_copy_row(NDframe dst, NDframe src, Py_ssize_t src_idx, Py_ssize_t* write_idx) except *:
    """整行克隆：结构体副本 + text/text_sep 两列的纯 C 内存拷贝，写入 dst 的 write_idx 位置"""
    if write_idx[0] >= dst._capacity:
        dst._reserve(dst._capacity * 2)
    dst.rows_c[write_idx[0]] = src.rows_c[src_idx]
    _col_copy_row(dst.text_col, src.text_col, src_idx)
    _col_copy_row(dst.text_sep_col, src.text_sep_col, src_idx)
    write_idx[0] += 1
