# %%cython --annotate
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from libc.stdlib cimport malloc, realloc, free, qsort
from libc.string cimport memcpy, memmove
from libc.stdio cimport printf
from libc.math cimport sqrt
from libc.string cimport memset
import cython
from cpython.long cimport PyLong_FromSsize_t
from cpython.list cimport PyList_Append, PyList_GetItem, PyList_GET_ITEM, PyList_Size, PyList_New, PyList_SetItem, PyList_GET_SIZE
import html as _html
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
    PyUnicode_4BYTE_KIND, PyUnicode_DATA, Py_UCS2
)

# ============================================
# 1. 前向声明，解决访问器与 NDframe 互相引用的问题
# ============================================
cdef class NDframe
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


cdef inline int _get_field_val(RowC* row, int field) noexcept nogil:
    # RowC 的成员顺序与 FieldID 严格一致，可直接通过指针偏移取值
    return (&row.idx)[field]


cdef inline void _add_field_value(RowC* row, int field, int offset) noexcept nogil:
    (&row.idx)[field] += offset


cdef inline bint _int_match(int value, int lo, int hi, int op) noexcept:
    if op == OP_GREATER: return value > lo
    if op == OP_LESS: return value < lo
    if op == OP_EQUAL: return value == lo
    if op == OP_BETWEEN: return value > lo and value < hi
    return False


cdef str _coerce_text(object value):
    """保证文本字段内部始终是 str，避免 C 层按 Unicode 读取非字符串对象。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


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
        cdef list src = df.text_list if self.field == FIELD_txt else df.text_sep_list
        return iter(src)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    def __repr__(self):
        cdef NDframe df = self.df
        cdef Py_ssize_t n = df.length
        cdef Py_ssize_t preview_n = n if n < 5 else 5
        cdef list preview = [None] * preview_n
        cdef Py_ssize_t i
        cdef str v
        cdef list src = df.text_list if self.field == FIELD_txt else df.text_sep_list

        for i in range(preview_n):
            v = src[i]
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
    def contains(self, pattern):
        """正则过滤，返回新 NDframe"""
        pattern = _ensure_regex(pattern)
        return self.df._df_regex(self.field, pattern)

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
        cdef NDframe df = self.df
        return df._df_regex_in_x0y0x1y1(self.field, pattern, x0, y0, x1, y1)

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
        if n < 1:
            raise ValueError("n must be > 0")
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
        cdef int field = self.field
        cdef list src = df.text_list if self.field == FIELD_txt else df.text_sep_list
        if index is None:
            return src[:]
        cdef Py_ssize_t idx = int(index)
        cdef Py_ssize_t n = df.length
        if n == 0:
            raise IndexError("Cannot get value from an empty NDframe")
        if idx < 0:
            idx += n
        if idx < 0 or idx >= n:
            raise IndexError(f"Index {index} out of range for NDframe with {n} rows")
        return <str>PyList_GET_ITEM(src, idx)

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

    # ========================================================
    # 正则 + 字符串长度组合筛选 (C++ df_regex_strlen_xxx)
    # ========================================================
    @cython.always_allow_keywords(False)
    def regex_strlen_greater(self, object pattern, int n):
        """正则匹配 && 字符串长度 > n 的行 → 新 NDframe"""
        if n < 0:
            raise ValueError("n must be >= 0")
        pattern = _ensure_regex(pattern)
        return self.df._df_regex_strlen_cmp(self.field, pattern, n, OP_GREATER)

    @cython.always_allow_keywords(False)
    def regex_strlen_less(self, object pattern, int n):
        """正则匹配 && 字符串长度 < n 的行 → 新 NDframe"""
        if n < 1:
            raise ValueError("n must be > 0")
        pattern = _ensure_regex(pattern)
        return self.df._df_regex_strlen_cmp(self.field, pattern, n, OP_LESS)

    @cython.always_allow_keywords(False)
    def regex_strlen_equal(self, object pattern, int n):
        """正则匹配 && 字符串长度 == n 的行 → 新 NDframe"""
        if n < 0:
            raise ValueError("n must be >= 0")
        pattern = _ensure_regex(pattern)
        return self.df._df_regex_strlen_cmp(self.field, pattern, n, OP_EQUAL)

    # ========================================================
    # 跨字段字符串替换 (C++ replace<field_src, field_dst>)
    # ========================================================
    def replace_to(self, str old_str, str new_str, str target_field="text"):
        """将本字段的 old_str 替换为 new_str，写入 target_field（就地）
        
        target_field: "text" 或 "text_sep"，默认同字段
        例: df.text_sep.replace_to("\\n", " ", "text")  # text_sep 源 → text 目标
        """
        cdef int dst_field = FIELD_txt if target_field == "text" else FIELD_sep
        self.df._replace_cross(self.field, dst_field, old_str, new_str)
        return self.df

    def regex_replace_to(self, pattern, str rewrite, str target_field="text"):
        """正则替换：本字段匹配 pattern 替换为 rewrite，写入 target_field（就地）

        例: df.text.regex_replace_to(re.compile(r'\d+'), '0', 'text_sep')
        """
        pattern = _ensure_regex(pattern)
        cdef int dst_field = FIELD_txt if target_field == "text" else FIELD_sep
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
        if x0 < 0: x0 = 0
        if x1 < 0: x1 = 0
        if y0 < 0: y0 = 0
        if y1 < 0: y1 = 0

        if x0 > x1 or y0 > y1:
            raise ValueError(f"def idx_contains_in_bbox invalid boundary: x0({x0})>x1({x1}) or y0({y0})>y1({y1})")
        pattern = _ensure_regex(pattern)
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
        例如: [10, 15, 12] -> [None, 5, -3]
        """
        return self.df._diff_col(self.field_id)

    def var(self, int ddof=1):
        """
        计算该整型字段的方差。
        默认 ddof=1 计算样本方差 (Bessel's correction)；ddof=0 计算总体方差。
        若样本数 <= ddof，返回 0.0。
        """
        return self.df._var_col(self.field_id, ddof)

    def std(self, int ddof=1):
        """
        计算该整型字段的标准差。
        默认 ddof=1 计算样本标准差；ddof=0 计算总体标准差。
        若样本数 <= ddof，返回 0.0。
        """
        return self.df._std_col(self.field_id, ddof)

    @cython.always_allow_keywords(False)
    def values(self, index=None):
        """
        获取字段的值。
        - 默认 (不传参): 返回该字段所有值的列表
        - 传参 (如 0, 1, -1): 返回指定行的值，支持负数索引 (-1 表示最后一行)
        """
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
        """
        计算该数值字段的众数（出现频率最高的值）。
        返回 (mode_value, count) 元组。
        若多个众数并存，返回其中最小的那个值。
        """
        # 智能路由：如果是 height 字段，直接走 C 数组桶计数的极速通道
        if self.field_id == FIELD_height:
            return self.df._mode_height()

        # 其他字段走原来的字典通用逻辑
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
    # 0x30-0x39: '0'-'9', 0x2E: '.', 0x2C: ','
    return (0x30 <= cp <= 0x39) or (cp == 0x2E) or (cp == 0x2C)


# ==========================================
# 极速字符判定谓词 (统一签名，供 C 循环回调)
# ==========================================
cdef inline bint _pred_isalnum(Py_UCS4 c) noexcept: return Py_UNICODE_ISALNUM(c)
cdef inline bint _pred_isalpha(Py_UCS4 c) noexcept: return Py_UNICODE_ISALPHA(c)
cdef inline bint _pred_isdigit(Py_UCS4 c) noexcept: return Py_UNICODE_ISDIGIT(c)
cdef inline bint _pred_isspace(Py_UCS4 c) noexcept: return Py_UNICODE_ISSPACE(c)
cdef inline bint _pred_islower(Py_UCS4 c) noexcept: return Py_UNICODE_ISLOWER(c)
cdef inline bint _pred_isupper(Py_UCS4 c) noexcept: return Py_UNICODE_ISUPPER(c)
cdef inline bint _pred_istitle(Py_UCS4 c) noexcept: return Py_UNICODE_ISTITLE(c)
cdef inline bint _pred_isnumeric(Py_UCS4 c) noexcept: return Py_UNICODE_ISNUMERIC(c)
cdef inline bint _pred_isdecimal(Py_UCS4 c) noexcept: return Py_UNICODE_ISDECIMAL(c)
cdef inline bint _pred_is_chinese(Py_UCS4 c) noexcept: return _is_chinese_char(c)
cdef inline bint _pred_is_numeric_punct(Py_UCS4 c) noexcept: return _is_numeric_punct_char(c)


cdef inline Py_UCS2* _str_to_ucs2_buf(str s, Py_ssize_t* out_len) except NULL:
    """将 Python str 转为以 0 结尾的 UCS2 缓冲区（调用方负责 free）。"""
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

    if kind == PyUnicode_2BYTE_KIND:
        memcpy(buf, data, n * sizeof(Py_UCS2))
    elif kind == PyUnicode_1BYTE_KIND:
        for i in range(n):
            buf[i] = <Py_UCS2>(<unsigned char*>data)[i]
    else:
        for i in range(n):
            cp = PyUnicode_READ_CHAR(s, i)
            buf[i] = <Py_UCS2>cp if cp <= 0xFFFF else 0x003F

    buf[n] = 0
    return buf


# 用于判断子字符串在某个字符串中存在，纯 C 内存滑动窗口匹配，避免 Python 对象创建
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


cdef inline bint _str_starts_ends(str txt, str sub, bint is_starts) noexcept:
    """纯 C 比较前缀/后缀，语义与 str.startswith/endswith 一致。"""
    cdef Py_ssize_t n = PyUnicode_GET_LENGTH(txt)
    cdef Py_ssize_t m = PyUnicode_GET_LENGTH(sub)
    if m > n:
        return False
    cdef Py_ssize_t offset = 0 if is_starts else n - m
    cdef Py_ssize_t j
    for j in range(m):
        if PyUnicode_READ_CHAR(txt, offset + j) != PyUnicode_READ_CHAR(sub, j):
            return False
    return True


cdef bint _str_all_match(str s, int mode):
    """按 StrCheckMode 逐字符判定，语义对齐 CPython 的 str.isxxx()。"""
    cdef Py_ssize_t i, n = PyUnicode_GET_LENGTH(s)
    if n == 0:
        return False

    cdef Py_UCS4 c
    cdef bint has_cased, previous_is_cased, cased
    cdef bint (*pred)(Py_UCS4) noexcept

    # islower / isupper / istitle 不是简单的“所有字符命中”，需按 cased 规则判断
    if mode == CHK_ISLOWER:
        has_cased = False
        for i in range(n):
            c = PyUnicode_READ_CHAR(s, i)
            if Py_UNICODE_ISLOWER(c):
                has_cased = True
            elif Py_UNICODE_ISUPPER(c) or Py_UNICODE_ISTITLE(c):
                return False
        return has_cased

    if mode == CHK_ISUPPER:
        has_cased = False
        for i in range(n):
            c = PyUnicode_READ_CHAR(s, i)
            if Py_UNICODE_ISUPPER(c):
                has_cased = True
            elif Py_UNICODE_ISLOWER(c) or Py_UNICODE_ISTITLE(c):
                return False
        return has_cased

    if mode == CHK_ISTITLE:
        previous_is_cased = False
        cased = False
        for i in range(n):
            c = PyUnicode_READ_CHAR(s, i)
            if Py_UNICODE_ISUPPER(c) or Py_UNICODE_ISTITLE(c):
                if previous_is_cased:
                    return False
                previous_is_cased = True
                cased = True
            elif Py_UNICODE_ISLOWER(c):
                if not previous_is_cased:
                    return False
                previous_is_cased = True
                cased = True
            else:
                previous_is_cased = False
        return cased

    if mode == CHK_ISALNUM: pred = _pred_isalnum
    elif mode == CHK_ISALPHA: pred = _pred_isalpha
    elif mode == CHK_ISDIGIT: pred = _pred_isdigit
    elif mode == CHK_ISSPACE: pred = _pred_isspace
    elif mode == CHK_ISNUMERIC: pred = _pred_isnumeric
    elif mode == CHK_ISDECIMAL: pred = _pred_isdecimal
    elif mode == CHK_ISCHINESE: pred = _pred_is_chinese
    elif mode == CHK_IS_NUMERIC_PUNCT: pred = _pred_is_numeric_punct
    else: return False

    for i in range(n):
        if not pred(PyUnicode_READ_CHAR(s, i)):
            return False
    return True


cdef str _filter_charset(str s, int mode):
    """保留命中的字符；全命中返回原对象，无命中返回空串，否则构造新串。"""
    cdef Py_ssize_t i, n = PyUnicode_GET_LENGTH(s)
    if n == 0:
        return s

    cdef bint (*pred)(Py_UCS4) noexcept
    if mode == CHK_ISCHINESE: pred = _pred_is_chinese
    elif mode == CHK_IS_NUMERIC_PUNCT: pred = _pred_is_numeric_punct
    else: return s

    cdef Py_UCS2* buf = <Py_UCS2*>malloc(n * sizeof(Py_UCS2))
    if buf == NULL:
        raise MemoryError()

    cdef Py_ssize_t count = 0
    cdef Py_UCS4 cp
    cdef bint has_wide = False

    try:
        for i in range(n):
            cp = PyUnicode_READ_CHAR(s, i)
            if pred(cp):
                if cp > 0xFFFF:
                    has_wide = True
                buf[count] = <Py_UCS2>cp
                count += 1

        if count == n and not has_wide:
            return s
        if count == 0:
            return ""

        if not has_wide:
            return PyUnicode_FromKindAndData(PyUnicode_2BYTE_KIND, buf, count)

        # 含 4 字节字符（如 CJK 扩展 B）时不能用 UCS2 截断，回退逐字符重建
        cdef list parts = [None] * count
        cdef Py_ssize_t k = 0
        for i in range(n):
            cp = PyUnicode_READ_CHAR(s, i)
            if pred(cp):
                parts[k] = s[i]
                k += 1
        return "".join(parts)
    finally:
        free(buf)

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
    """
    返回非负整数的十进制位数。
    基于领域先验：图像坐标 n >= 0 且绝大部分 <= 10000。
    """
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


cdef Py_ssize_t MAX_CAPACITY = <Py_ssize_t>(1 << 40)


cdef struct SortPair:
    int key
    Py_ssize_t idx


cdef int _cmp_sort_asc(const void* a, const void* b) noexcept nogil:
    cdef SortPair* pa = <SortPair*>a
    cdef SortPair* pb = <SortPair*>b
    if pa.key < pb.key: return -1
    if pa.key > pb.key: return 1
    if pa.idx < pb.idx: return -1
    if pa.idx > pb.idx: return 1
    return 0


cdef int _cmp_sort_desc(const void* a, const void* b) noexcept nogil:
    cdef SortPair* pa = <SortPair*>a
    cdef SortPair* pb = <SortPair*>b
    if pa.key > pb.key: return -1
    if pa.key < pb.key: return 1
    if pa.idx < pb.idx: return -1
    if pa.idx > pb.idx: return 1
    return 0


@cython.final
@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class NDframe:
    cdef:
        RowC* rows_c
        Py_ssize_t length
        Py_ssize_t _capacity
        list text_list
        list text_sep_list

    @cython.always_allow_keywords(False)
    def __cinit__(self, Py_ssize_t capacity=16):
        if capacity < 1:
            capacity = 1
        if capacity > MAX_CAPACITY:
            raise MemoryError("NDframe capacity too large")
        self._capacity = capacity
        self.length = 0

        self.rows_c = <RowC*>malloc(<size_t>self._capacity * sizeof(RowC))
        if not self.rows_c:
            raise MemoryError()

        self.text_list = []
        self.text_sep_list = []

    def __dealloc__(self):
        """清理内存"""
        if self.rows_c is not NULL:
            free(self.rows_c)

    def __len__(self):
        return self.length

    @cython.always_allow_keywords(False)
    def len(self):
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
        cdef list txt = self.text_list
        cdef list sep = self.text_sep_list
        cdef Py_ssize_t n = self.length
        for i in range(n):
            cur_txt = <str>PyList_GET_ITEM(txt, i)
            cur_sep = <str>PyList_GET_ITEM(sep, i)
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
                "text": cur_txt,
                "text_sep": cur_sep
            }

    def __repr__(self):
        return f"NDframe({self.length} rows)"

    def __str__(self):
        return f"<NDframe: {self.length} rows>"

    def _repr_html_(self):
        """
        IPython / Jupyter / PyCharm SciView 富文本协议。
        在 Notebook 或交互式终端中输出极其精美的 HTML 表格，0 外部依赖。
        """
        if self.length == 0:
            return "<div><strong>NDframe</strong> is empty.</div>"

        cdef Py_ssize_t preview_n = min(self.length, 60) # 限制渲染行数防卡顿
        cdef list parts = []
        parts.append("<div style='max-height: 400px; overflow: auto;'>")
        parts.append("<table border='1' style='border-collapse: collapse; text-align: right;'>")

        parts.append("<thead><tr style='background-color: #f2f2f2;'>")
        for col in HEADERS:
            parts.append(f"<th>{_html.escape(col)}</th>")
        parts.append("</tr></thead><tbody>")

        cdef Py_ssize_t i
        for i in range(preview_n):
            parts.append("<tr>")
            parts.append(f"<td>{self.rows_c[i].idx}</td>")
            parts.append(f"<td>{self.rows_c[i].x0}</td>")
            parts.append(f"<td>{self.rows_c[i].y0}</td>")
            parts.append(f"<td>{self.rows_c[i].x1}</td>")
            parts.append(f"<td>{self.rows_c[i].y1}</td>")
            parts.append(f"<td>{self.rows_c[i].width}</td>")
            parts.append(f"<td>{self.rows_c[i].height}</td>")
            parts.append(f"<td>{self.rows_c[i].page}</td>")
            parts.append(f"<td>{self.rows_c[i].flags}</td>")
            parts.append(f"<td style='text-align: left;'>{_html.escape(str(self.text_list[i]))}</td>")
            parts.append(f"<td style='text-align: left;'>{_html.escape(str(self.text_sep_list[i]))}</td>")
            parts.append("</tr>")

        parts.append("</tbody></table></div>")
        if self.length > preview_n:
            parts.append(f"<p>... showing {preview_n} of {self.length} rows ...</p>")

        return "".join(parts)

    # ----------------------------------------------------
    # API 属性映射区：极其优雅的调用入口
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
        cdef Py_ssize_t start, stop, new_len, ii
        cdef NDframe new_df
        cdef int int_key

        if isinstance(key, slice):
            if key.step not in (None, 1):
                raise ValueError("Only contiguous slices are supported (e.g., df[1:5]). Step is not allowed.")

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
                new_df.text_list = self.text_list[start:stop]
                new_df.text_sep_list = self.text_sep_list[start:stop]
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
                "text": <str>PyList_GET_ITEM(self.text_list, int_key),
                "text_sep": <str>PyList_GET_ITEM(self.text_sep_list, int_key)
            }
        else:
            raise TypeError("Indices must be integers or slices (e.g. df[0], df[1:3])")

    @cython.always_allow_keywords(False)
    def head(self, ):
        """返回前 10 行的新 NDframe"""
        Py_ssize_t n = 10
        if n > self.length:
            n = self.length
        return self[0:n]

    @cython.always_allow_keywords(False)
    def tail(self):
        """返回后 10 行的新 NDframe"""
        Py_ssize_t n = 10
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
        self.text_list.pop()
        self.text_sep_list.pop()

    @cython.always_allow_keywords(False)
    def bbox(self, int n=0):
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
        """深拷贝，返回完全独立的新 NDframe"""
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n > 0:
            memcpy(new_df.rows_c, self.rows_c, n * sizeof(RowC))
            new_df.text_list = self.text_list[:]
            new_df.text_sep_list = self.text_sep_list[:]
            new_df.length = n
        return new_df

    cdef void _reserve(self, Py_ssize_t new_capacity) except *:
        if new_capacity <= self._capacity:
            return
        if new_capacity < 0 or new_capacity > MAX_CAPACITY:
            raise MemoryError("NDframe capacity too large")
        cdef RowC* new_rows = <RowC*>realloc(self.rows_c, <size_t>new_capacity * sizeof(RowC))
        if not new_rows:
            raise MemoryError()
        self.rows_c = new_rows
        self._capacity = new_capacity

    cdef void _grow(self) except *:
        if self._capacity > MAX_CAPACITY // 2:
            raise MemoryError("NDframe capacity too large")
        self._reserve(self._capacity * 2)

    cdef void _append_c_row(self, RowC* row, str text, str text_sep):
        """内部快速追加方法"""
        if self.length >= self._capacity:
            self._grow()

        self.rows_c[self.length] = row[0]
        PyList_Append(self.text_list, text)
        PyList_Append(self.text_sep_list, text_sep)
        self.length += 1

    @cython.always_allow_keywords(False)
    def append(self, int idx, int x0, int y0, int x1, int y1,
               int width, int height, int page, int flags,
               str text, str text_sep=""):
        cdef Py_ssize_t n = self.length
        if n >= self._capacity:
            self._grow()

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

        self.text_list.append(text)
        self.text_sep_list.append(text_sep)
        self.length = n + 1

    @cython.always_allow_keywords(False)
    def to_list(self):
        """转换为可打印的 Python 格式"""
        cdef Py_ssize_t n = self.length
        if n == 0:
            return []

        cdef RowC* rows = self.rows_c
        cdef list txt = self.text_list
        cdef list sep = self.text_sep_list
        cdef list res = [None] * n
        cdef Py_ssize_t i
        cdef RowC* row

        for i in range(n):
            row = &rows[i]
            cur_txt = <str>PyList_GET_ITEM(txt, i)
            cur_sep = <str>PyList_GET_ITEM(sep, i)
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

        # 强制检查必须字段，并且兼容 numpy.ndarray / tuple / pandas.Series
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

        # 可选字段也需要提前获取并转换为 list
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
            rows[i].x0     = l_x0[i]
            rows[i].y0     = l_y0[i]
            rows[i].x1     = l_x1[i]
            rows[i].y1     = l_y1[i]
            rows[i].width  = l_w[i]
            rows[i].height = l_h[i]

        if l_page is not None:
            for i in range(length): rows[i].page = l_page[i]
        if l_flags is not None:
            for i in range(length): rows[i].flags = l_flags[i]

        cdef list clean_text = [None] * length
        cdef list clean_sep = [None] * length
        for i in range(length):
            clean_text[i] = _coerce_text(l_text[i])
            clean_sep[i] = _coerce_text(l_sep[i]) if l_sep is not None else ""

        df.text_list = clean_text
        df.text_sep_list = clean_sep

        df.length = length
        return df

    def print(self, str name=None):
        cdef str dataframe_name = name if name else "NDframe"
        if self.length == 0:
            return f"{dataframe_name} is Empty!\n"
        print(f"[{dataframe_name}] ({self.length} rows):")
        print(self.to_string())

    @cython.always_allow_keywords(False)
    def to_string(self):
        """构造并返回对齐的完美 ASCII 表格，支持超大行数自动折叠"""
        cdef Py_ssize_t max_rows = 80
        cdef Py_ssize_t half = max_rows // 2

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

            iw = _int_display_width(self.rows_c[i].idx);    widths[0] = iw if iw > widths[0] else widths[0]
            iw = _int_display_width(self.rows_c[i].x0);     widths[1] = iw if iw > widths[1] else widths[1]
            iw = _int_display_width(self.rows_c[i].y0);     widths[2] = iw if iw > widths[2] else widths[2]
            iw = _int_display_width(self.rows_c[i].x1);     widths[3] = iw if iw > widths[3] else widths[3]
            iw = _int_display_width(self.rows_c[i].y1);     widths[4] = iw if iw > widths[4] else widths[4]
            iw = _int_display_width(self.rows_c[i].width);  widths[5] = iw if iw > widths[5] else widths[5]
            iw = _int_display_width(self.rows_c[i].height); widths[6] = iw if iw > widths[6] else widths[6]
            iw = _int_display_width(self.rows_c[i].page);   widths[7] = iw if iw > widths[7] else widths[7]
            iw = _int_display_width(self.rows_c[i].flags);  widths[8] = iw if iw > widths[8] else widths[8]

            s_val = _truncate_str(self.text_list[i])
            iw = _get_visual_width(s_val)
            cached_txt.append(s_val)
            cached_txt_w.append(iw)
            widths[9] = iw if iw > widths[9] else widths[9]

            t_val = _truncate_str(self.text_sep_list[i])
            iw = _get_visual_width(t_val)
            cached_sep.append(t_val)
            cached_sep_w.append(iw)
            widths[10] = iw if iw > widths[10] else widths[10]

        cdef list lines = []
        cdef list sep_parts
        sep_parts = ["+"]
        for j in range(11): sep_parts.extend(["-" * (widths[j] + 2), "+"])
        cdef str thin_sep = "".join(sep_parts)

        sep_parts = ["+"]
        for j in range(11): sep_parts.extend(["=" * (widths[j] + 2), "+"])
        cdef str thick_sep = "".join(sep_parts)

        cdef list row_cells
        cdef int pad

        lines.append(thin_sep)

        row_cells = ["|"]
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


    cdef list _str_slice(self, int field, object start, object stop, object step):
        cdef list res = []
        cdef Py_ssize_t i, n = self.length
        cdef list src = self.text_list if field == FIELD_txt else self.text_sep_list
        for i in range(n):
            res.append(src[i][start:stop:step])
        return res

    cdef int _str_count(self, int field, str sub):
        cdef int total = 0
        cdef Py_ssize_t i, n = self.length
        cdef list src = self.text_list if field == FIELD_txt else self.text_sep_list
        for i in range(n):
            if src[i]: total += src[i].count(sub)
        return total

    cdef NDframe _str_transform(self, int field, int mode, str chars=None):
        """就地操作：执行 strip, capitalize 等"""
        cdef Py_ssize_t i, n = self.length
        cdef list src = self.text_list if field == FIELD_txt else self.text_sep_list

        if n == 0:
            return self
        if mode == TRM_STRIP:
            for i in range(n):
                if src[i]:
                    src[i] = src[i].strip(chars)
        elif mode == TRM_LSTRIP:
            for i in range(n):
                if src[i]:
                    src[i] = src[i].lstrip(chars)
        elif mode == TRM_RSTRIP:
            for i in range(n):
                if src[i]:
                    src[i] = src[i].rstrip(chars)
        elif mode == TRM_CAPITALIZE:
            for i in range(n):
                if src[i]:
                    src[i] = src[i].capitalize()
        elif mode == TRM_SWAPCASE:
            for i in range(n):
                if src[i]:
                    src[i] = src[i].swapcase()
        return self


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef NDframe _df_str_starts_ends(self, int field, str sub, bint is_starts):
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n == 0: return new_df

        cdef Py_ssize_t i, write_idx = 0
        cdef str txt
        cdef RowC* src_rows = self.rows_c
        cdef RowC* dst_rows = new_df.rows_c
        cdef list src_txt = self.text_list
        cdef list src_sep = self.text_sep_list

        cdef list dst_txt = new_df.text_list
        cdef list dst_sep = new_df.text_sep_list

        cdef list target_list = src_txt if field == FIELD_txt else src_sep

        for i in range(n):
            txt = <str>target_list[i]
            if txt and _str_starts_ends(txt, sub, is_starts):
                dst_rows[write_idx] = src_rows[i]
                PyList_Append(dst_txt, src_txt[i])
                PyList_Append(dst_sep, src_sep[i])
                write_idx += 1
        new_df.length = write_idx
        return new_df

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef NDframe _df_str_is(self, int field, int mode):
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n == 0: return new_df

        if mode < CHK_ISALNUM or mode > CHK_IS_NUMERIC_PUNCT:
            return new_df

        cdef Py_ssize_t i, write_idx = 0
        cdef str txt
        cdef RowC* src_rows = self.rows_c
        cdef RowC* dst_rows = new_df.rows_c
        cdef list src_txt = self.text_list
        cdef list src_sep = self.text_sep_list

        cdef list dst_txt = new_df.text_list
        cdef list dst_sep = new_df.text_sep_list

        cdef list target_list = src_txt if field == FIELD_txt else src_sep

        for i in range(n):
            txt = <str>target_list[i]
            if txt and _str_all_match(txt, mode):
                dst_rows[write_idx] = src_rows[i]
                PyList_Append(dst_txt, src_txt[i])
                PyList_Append(dst_sep, src_sep[i])
                write_idx += 1

        new_df.length = write_idx
        return new_df

    cdef NDframe _df_is_charset(self, int field, bint is_chinese_mode):
        return self._df_str_is(
            field, CHK_ISCHINESE if is_chinese_mode else CHK_IS_NUMERIC_PUNCT)

    cdef NDframe _keep_charset(self, int field, bint is_chinese_mode):
        cdef Py_ssize_t i, n = self.length
        cdef list src = self.text_list if field == FIELD_txt else self.text_sep_list
        cdef int mode = CHK_ISCHINESE if is_chinese_mode else CHK_IS_NUMERIC_PUNCT

        for i in range(n):
            if src[i]:
                src[i] = _filter_charset(src[i], mode)
        return self

    # ========================================================
    # 聚合与计算功能
    # ========================================================
    cdef int _max_col(self, int field):
        """寻找最大值"""
        cdef Py_ssize_t n = self.length
        if n == 0:
            return 0
        if field < FIELD_idx or field > FIELD_flags:
            printf("warning: unsupported field!\n")
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
        """寻找最小值"""
        cdef Py_ssize_t n = self.length
        if n == 0:
            return 0
        if field < FIELD_idx or field > FIELD_flags:
            printf("warning: unsupported field!\n")
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
        if n == 0:
            return 0.0
        if field < FIELD_idx or field > FIELD_flags:
            printf("warning: only support integral fields!\n")
            return -1.0
        cdef RowC* rows = self.rows_c
        cdef Py_ssize_t i
        cdef double sum_val = 0.0

        for i in range(n):
            sum_val += _get_field_val(&rows[i], field)
        return sum_val / <double>n

    # ========================================================
    # 字符串拼接与替换
    # ========================================================
    cdef str _join(self, int field, str separator=""):
        if field == FIELD_txt:
            return separator.join(self.text_list)
        return separator.join(self.text_sep_list)

    cdef void _replace(self, int field, str old_str, str new_str):
        if old_str == "":
            return
        cdef Py_ssize_t n = self.length
        if n == 0:
            return
        cdef list lst = self.text_list if field == FIELD_txt else self.text_sep_list
        cdef Py_ssize_t i
        cdef str txt
        cdef object replace = str.replace

        for i in range(n):
            txt = lst[i]
            if txt and old_str in txt:
                lst[i] = replace(txt, old_str, new_str)


    cdef void _replace_dual(self, int field, str old0, str new0, str old1, str new1):
        cdef Py_ssize_t n = self.length
        if n == 0:
            return

        if (not old0) and (not old1):
            return
        cdef list lst = self.text_list if field == FIELD_txt else self.text_sep_list
        cdef Py_ssize_t i
        cdef str txt
        cdef object replace = str.replace

        if old0 and old1:
            for i in range(n):
                txt = lst[i]
                if txt:
                    if old0 in txt:
                        txt = replace(txt, old0, new0)
                    if old1 in txt:
                        txt = replace(txt, old1, new1)
                    lst[i] = txt
        elif old0:
            for i in range(n):
                txt = lst[i]
                if txt and old0 in txt:
                    lst[i] = replace(txt, old0, new0)
        elif old1:
            for i in range(n):
                txt = lst[i]
                if txt and old1 in txt:
                    lst[i] = replace(txt, old1, new1)

    # ========================================================
    # 正则表达式支持
    # ========================================================
    cdef void _regex_replace(self, int field, pattern, str rewrite):
        """正则就地替换 (pattern 需要是 re.compile 的对象)"""
        cdef Py_ssize_t i, n = self.length
        if n == 0:
            return
        cdef object sub = pattern.sub  # 绑定方法
        cdef list txt_list = None
        cdef list sep_list = None
        if field == FIELD_txt:
            txt_list = self.text_list
            for i in range(n):
                if txt_list[i]:
                    txt_list[i] = sub(rewrite, txt_list[i])
        elif field == FIELD_sep:
            sep_list = self.text_sep_list
            for i in range(n):
                if sep_list[i]:
                    sep_list[i] = sub(rewrite, sep_list[i])


    cdef NDframe _df_regex_extract(self, int field, object pattern):
        """正则提取（第一个捕获组）并返回新 NDframe"""
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n == 0:
            return new_df

        cdef Py_ssize_t i, write_idx = 0
        cdef str txt, new_txt
        cdef object match
        cdef object search = pattern.search

        cdef RowC* src_rows = self.rows_c
        cdef RowC* dst_rows = new_df.rows_c
        cdef list src_txt = self.text_list
        cdef list src_sep = self.text_sep_list

        cdef list dst_txt = new_df.text_list
        cdef list dst_sep = new_df.text_sep_list

        if field == FIELD_txt:
            for i in range(n):
                txt = src_txt[i]
                if not txt:
                    continue
                match = search(txt)
                if match is None:
                    continue
                if match.lastindex is not None and match.lastindex >= 1:
                    g = match.group(1)
                    if g is not None:
                        dst_rows[write_idx] = src_rows[i]
                        PyList_Append(dst_txt, g)
                        PyList_Append(dst_sep, src_sep[i])
                        write_idx += 1

        elif field == FIELD_sep:
            for i in range(n):
                txt = src_sep[i]
                if not txt:
                    continue
                match = search(txt)
                if match is None:
                    continue
                if match.lastindex is not None and match.lastindex >= 1:
                    g = match.group(1)
                    if g is not None:
                        dst_rows[write_idx] = src_rows[i]
                        PyList_Append(dst_txt, src_txt[i])
                        PyList_Append(dst_sep, g)
                        write_idx += 1
        else:
            printf("field must be FIELD_txt or FIELD_sep\n")
            return new_df
        new_df.length = write_idx
        return new_df


    # ========================================================
    # 核心过滤功能
    # ========================================================
    cdef NDframe _df_regex(self, int field, pattern):
        """根据正则过滤并返回新 DF"""
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if self.length == 0:
            return new_df

        cdef Py_ssize_t i
        cdef Py_ssize_t write_idx = 0
        cdef str txt
        cdef object search_fn = pattern.search

        cdef RowC* src_rows = self.rows_c
        cdef RowC* dst_rows = new_df.rows_c
        cdef list src_txt = self.text_list
        cdef list src_sep = self.text_sep_list

        cdef list dst_txt = new_df.text_list
        cdef list dst_sep = new_df.text_sep_list

        if field == FIELD_txt:
            for i in range(n):
                txt = <str>src_txt[i]
                if txt and search_fn(txt) is not None:
                    dst_rows[write_idx] = src_rows[i]
                    PyList_Append(dst_txt, txt)
                    PyList_Append(dst_sep, src_sep[i])
                    write_idx += 1
        elif field == FIELD_sep:
            for i in range(n):
                txt = <str>src_sep[i]
                if txt and search_fn(txt) is not None:
                    dst_rows[write_idx] = src_rows[i]
                    PyList_Append(dst_txt, src_txt[i])
                    PyList_Append(dst_sep, txt)

                    write_idx += 1
        else:
            printf("field must be FIELD_txt or FIELD_sep\n")
            return new_df
        new_df.length = write_idx
        return new_df

    cdef NDframe _df_regex_in_x0y0x1y1(self, int field, pattern, int x0, int y0, int x1, int y1):
        """坐标与字符串混合过滤"""
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if self.length == 0:
            return new_df

        cdef Py_ssize_t i, write_idx = 0
        cdef str txt
        cdef object search_fn = pattern.search

        cdef RowC* src_rows = self.rows_c
        cdef RowC* dst_rows = new_df.rows_c
        cdef list src_txt = self.text_list
        cdef list src_sep = self.text_sep_list

        cdef list dst_txt = new_df.text_list
        cdef list dst_sep = new_df.text_sep_list

        cdef list target_list = src_txt if field == FIELD_txt else src_sep

        for i in range(n):
            if (src_rows[i].x0 > x0 and src_rows[i].x1 < x1 and
                src_rows[i].y0 > y0 and src_rows[i].y1 < y1):
                txt = <str>target_list[i]
                if txt and search_fn(txt) is not None:
                    dst_rows[write_idx] = src_rows[i]
                    PyList_Append(dst_txt, src_txt[i])
                    PyList_Append(dst_sep, src_sep[i])
                    write_idx += 1

        new_df.length = write_idx
        return new_df

    # ========================================================
    # 纯坐标过滤 (返回值 NDframe，参照 C++ df_in_xxx 系列)
    # ========================================================
    cdef NDframe _in_bbox(self, int x0, int y0, int x1, int y1):
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n == 0: return new_df

        cdef Py_ssize_t i, write_idx = 0
        cdef RowC* src_rows = self.rows_c
        cdef RowC* dst_rows = new_df.rows_c
        cdef list src_txt = self.text_list
        cdef list src_sep = self.text_sep_list

        cdef list dst_txt = new_df.text_list
        cdef list dst_sep = new_df.text_sep_list

        for i in range(n):
            if (src_rows[i].x0 > x0 and src_rows[i].y0 > y0 and
                src_rows[i].x1 < x1 and src_rows[i].y1 < y1):
                dst_rows[write_idx] = src_rows[i]
                PyList_Append(dst_txt, src_txt[i])
                PyList_Append(dst_sep, src_sep[i])
                write_idx += 1

        new_df.length = write_idx
        return new_df

    cdef NDframe _in_x_range(self, int x0, int x1):
        """返回 x 方向在范围内的行: x0 > bx0 && x1 < bx1"""
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n == 0:
            return new_df

        cdef Py_ssize_t i, write_idx = 0
        cdef RowC* src_rows = self.rows_c
        cdef RowC* dst_rows = new_df.rows_c
        cdef list src_txt = self.text_list
        cdef list src_sep = self.text_sep_list

        cdef list dst_txt = new_df.text_list
        cdef list dst_sep = new_df.text_sep_list

        for i in range(n):
            if src_rows[i].x0 > x0 and src_rows[i].x1 < x1:
                dst_rows[write_idx] = src_rows[i]
                PyList_Append(dst_txt, src_txt[i])
                PyList_Append(dst_sep, src_sep[i])
                write_idx += 1
        new_df.length = write_idx
        return new_df

    cdef NDframe _in_y_range(self, int y0, int y1):
        """返回 y 方向在范围内的行: y0 > by0 && y1 < by1"""
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n == 0:
            return new_df

        cdef Py_ssize_t i, write_idx = 0
        cdef RowC* src_rows = self.rows_c
        cdef RowC* dst_rows = new_df.rows_c
        cdef list src_txt = self.text_list
        cdef list src_sep = self.text_sep_list

        cdef list dst_txt = new_df.text_list
        cdef list dst_sep = new_df.text_sep_list
        for i in range(n):
            if src_rows[i].y0 > y0 and src_rows[i].y1 < y1:
                dst_rows[write_idx] = src_rows[i]
                PyList_Append(dst_txt, src_txt[i])
                PyList_Append(dst_sep, src_sep[i])
                write_idx += 1
        new_df.length = write_idx
        return new_df

    cdef NDframe _greater_point(self, int x0_val, int y0_val):
        """返回左上角坐标均大于阈值的行: x0 > vx && y0 > vy"""
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n == 0:
            return new_df

        cdef Py_ssize_t i, write_idx = 0
        cdef RowC* src_rows = self.rows_c
        cdef RowC* dst_rows = new_df.rows_c
        cdef list src_txt = self.text_list
        cdef list src_sep = self.text_sep_list

        cdef list dst_txt = new_df.text_list
        cdef list dst_sep = new_df.text_sep_list

        for i in range(n):
            if src_rows[i].x0 > x0_val and src_rows[i].y0 > y0_val:
                dst_rows[write_idx] = src_rows[i]
                PyList_Append(dst_txt, src_txt[i])
                PyList_Append(dst_sep, src_sep[i])
                write_idx += 1
        new_df.length = write_idx
        return new_df

    cdef NDframe _less_point(self, int x1_val, int y1_val):
        """返回右下角坐标均小于阈值的行: x1 < vx && y1 < vy"""
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n == 0:
            return new_df

        cdef Py_ssize_t i, write_idx = 0
        cdef RowC* src_rows = self.rows_c
        cdef RowC* dst_rows = new_df.rows_c
        cdef list src_txt = self.text_list
        cdef list src_sep = self.text_sep_list

        cdef list dst_txt = new_df.text_list
        cdef list dst_sep = new_df.text_sep_list

        for i in range(n):
            if src_rows[i].x1 < x1_val and src_rows[i].y1 < y1_val:
                dst_rows[write_idx] = src_rows[i]
                PyList_Append(dst_txt, src_txt[i])
                PyList_Append(dst_sep, src_sep[i])
                write_idx += 1
        new_df.length = write_idx
        return new_df

    # ========================================================
    # 单字段整数值过滤 (NumericAccessor 底层实现)
    # ========================================================
    cdef NDframe _df_filter_by_int(self, int field, int arg1, int arg2, int op):
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n == 0:
            return new_df

        cdef Py_ssize_t i, write_idx = 0
        cdef int val
        cdef RowC* src_rows = self.rows_c
        cdef RowC* dst_rows = new_df.rows_c
        cdef list src_txt = self.text_list
        cdef list src_sep = self.text_sep_list

        cdef list dst_txt = new_df.text_list
        cdef list dst_sep = new_df.text_sep_list

        if op == OP_GREATER:
            for i in range(n):
                val = _get_field_val(&src_rows[i], field)
                if val > arg1:
                    dst_rows[write_idx] = src_rows[i]
                    PyList_Append(dst_txt, src_txt[i])
                    PyList_Append(dst_sep, src_sep[i])
                    write_idx += 1
        elif op == OP_LESS:
            for i in range(n):
                val = _get_field_val(&src_rows[i], field)
                if val < arg1:
                    dst_rows[write_idx] = src_rows[i]
                    PyList_Append(dst_txt, src_txt[i])
                    PyList_Append(dst_sep, src_sep[i])
                    write_idx += 1
        elif op == OP_BETWEEN:
            for i in range(n):
                val = _get_field_val(&src_rows[i], field)
                if val > arg1 and val < arg2:
                    dst_rows[write_idx] = src_rows[i]
                    PyList_Append(dst_txt, src_txt[i])
                    PyList_Append(dst_sep, src_sep[i])
                    write_idx += 1
        new_df.length = write_idx
        return new_df

    # ========================================================
    # 字符串长度过滤 (StringAccessor 底层实现)
    # ========================================================
    cdef NDframe _df_strlen_cmp(self, int field, int th, int op):
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n == 0:
            return new_df

        cdef Py_ssize_t i, write_idx = 0
        cdef int ln
        cdef RowC* src_rows = self.rows_c
        cdef RowC* dst_rows = new_df.rows_c
        cdef list src_txt = self.text_list
        cdef list src_sep = self.text_sep_list

        cdef list dst_txt = new_df.text_list
        cdef list dst_sep = new_df.text_sep_list

        cdef list target_list = src_txt if field == FIELD_txt else src_sep

        if op == OP_GREATER:
            for i in range(n):
                ln = len(target_list[i])
                if ln > th:
                    dst_rows[write_idx] = src_rows[i]
                    PyList_Append(dst_txt, src_txt[i])
                    PyList_Append(dst_sep, src_sep[i])
                    write_idx += 1
        elif op == OP_LESS:
            for i in range(n):
                ln = len(target_list[i])
                if ln < th:
                    dst_rows[write_idx] = src_rows[i]
                    PyList_Append(dst_txt, src_txt[i])
                    PyList_Append(dst_sep, src_sep[i])
                    write_idx += 1
        elif op == OP_EQUAL:
            for i in range(n):
                ln = len(target_list[i])
                if ln == th:
                    dst_rows[write_idx] = src_rows[i]
                    PyList_Append(dst_txt, src_txt[i])
                    PyList_Append(dst_sep, src_sep[i])
                    write_idx += 1
        else:
            printf("ERROR: Invalid operation\n")
            return new_df
        new_df.length = write_idx
        return new_df

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef list _strlen_list(self, int field):
        """返回字符串字段每行长度的列表"""
        cdef Py_ssize_t n = self.length
        if n == 0:
            return []
        cdef list result = [0] * n
        cdef Py_ssize_t i
        cdef list src_txt = self.text_list
        cdef list src_sep = self.text_sep_list

        if field == FIELD_txt:
            for i in range(n):
                result[i] = PyLong_FromSsize_t(PyUnicode_GET_LENGTH(src_txt[i]))
        else:
            for i in range(n):
                result[i] = PyLong_FromSsize_t(PyUnicode_GET_LENGTH(src_sep[i]))
        return result

    cdef NDframe _df_is_in(self, int field, object texts):
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n == 0:
            return new_df

        cdef set text_set = set(texts)
        cdef Py_ssize_t i, write_idx = 0
        cdef RowC* src_rows = self.rows_c
        cdef RowC* dst_rows = new_df.rows_c
        cdef list src_txt = self.text_list
        cdef list src_sep = self.text_sep_list

        cdef list dst_txt = new_df.text_list
        cdef list dst_sep = new_df.text_sep_list

        cdef list target_list = src_txt if field == FIELD_txt else src_sep

        for i in range(n):
            if target_list[i] in text_set:
                dst_rows[write_idx] = src_rows[i]
                PyList_Append(dst_txt, src_txt[i])
                PyList_Append(dst_sep, src_sep[i])
                write_idx += 1

        new_df.length = write_idx
        return new_df

    cdef NDframe _df_not_in(self, int field, object texts):
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n == 0:
            return new_df

        cdef set text_set = set(texts)
        cdef Py_ssize_t i, write_idx = 0
        cdef RowC* src_rows = self.rows_c
        cdef RowC* dst_rows = new_df.rows_c
        cdef list src_txt = self.text_list
        cdef list src_sep = self.text_sep_list

        cdef list dst_txt = new_df.text_list
        cdef list dst_sep = new_df.text_sep_list

        if field == FIELD_txt:
            for i in range(n):
                if src_txt[i] not in text_set:
                    dst_rows[write_idx] = src_rows[i]
                    PyList_Append(dst_txt, src_txt[i])
                    PyList_Append(dst_sep, src_sep[i])
                    write_idx += 1
        elif field == FIELD_sep:
            for i in range(n):
                if src_sep[i] not in text_set:
                    dst_rows[write_idx] = src_rows[i]
                    PyList_Append(dst_txt, src_txt[i])
                    PyList_Append(dst_sep, src_sep[i])
                    write_idx += 1
        else:
            return new_df
        new_df.length = write_idx
        return new_df


    cdef bint _contains_any(self, int field, object texts):
        """检查字段值中是否包含给定集合中的任意一个字符串（C 级子串匹配）"""
        cdef Py_ssize_t i, n = self.length
        if n == 0:
            return False

        cdef list text_list
        if field == FIELD_txt:
            text_list = self.text_list
        elif field == FIELD_sep:
            text_list = self.text_sep_list
        else:
            return False

        cdef tuple text_tuple = tuple(texts) if not isinstance(texts, tuple) else texts
        cdef Py_ssize_t m = len(text_tuple)
        if m == 0:
            return False

        cdef Py_UCS2** sub_bufs = <Py_UCS2**>malloc(m * sizeof(Py_UCS2*))
        cdef Py_ssize_t* sub_lens = <Py_ssize_t*>malloc(m * sizeof(Py_ssize_t))
        cdef bint* sub_ok = <bint*>malloc(m * sizeof(bint))
        if not sub_bufs or not sub_lens or not sub_ok:
            if sub_bufs: free(sub_bufs)
            if sub_lens: free(sub_lens)
            if sub_ok: free(sub_ok)
            raise MemoryError()

        cdef Py_UCS2* row_buf = NULL
        cdef Py_UCS2* new_row
        cdef Py_ssize_t row_cap = 0
        cdef Py_ssize_t row_len
        cdef Py_ssize_t p, j
        cdef int kind
        cdef void* data
        cdef unsigned char* src1
        cdef str txt, t
        cdef bint found = False

        for p in range(m):
            sub_bufs[p] = NULL
            sub_ok[p] = False

        try:
            for p in range(m):
                t = <str>text_tuple[p]
                sub_lens[p] = PyUnicode_GET_LENGTH(t)
                if PyUnicode_KIND(t) != PyUnicode_4BYTE_KIND:
                    sub_bufs[p] = _str_to_ucs2_buf(t, &sub_lens[p])
                    sub_ok[p] = True

            for i in range(n):
                txt = <str>text_list[i]
                if not txt:
                    continue

                row_len = PyUnicode_GET_LENGTH(txt)
                kind = PyUnicode_KIND(txt)

                if kind == PyUnicode_4BYTE_KIND:
                    # 4-byte (Emoji/生僻字) 无法无损放入 UCS2，回退到 Python 精确匹配
                    for p in range(m):
                        if text_tuple[p] in txt:
                            found = True
                            break
                    if found:
                        break
                    continue

                if row_len + 1 > row_cap:
                    row_cap = row_len + 1
                    new_row = <Py_UCS2*>realloc(row_buf, row_cap * sizeof(Py_UCS2))
                    if new_row == NULL:
                        raise MemoryError()
                    row_buf = new_row

                data = PyUnicode_DATA(txt)
                if kind == PyUnicode_2BYTE_KIND:
                    memcpy(row_buf, data, row_len * sizeof(Py_UCS2))
                else:
                    src1 = <unsigned char*>data
                    for j in range(row_len):
                        row_buf[j] = <Py_UCS2>src1[j]

                for p in range(m):
                    if sub_ok[p]:
                        if _substr_contains(row_buf, 0, row_len, sub_bufs[p], sub_lens[p]):
                            found = True
                            break
                    elif text_tuple[p] in txt:
                        found = True
                        break
                if found:
                    break

            return found
        finally:
            if row_buf != NULL:
                free(row_buf)
            for p in range(m):
                if sub_bufs[p] != NULL:
                    free(sub_bufs[p])
            free(sub_bufs)
            free(sub_lens)
            free(sub_ok)

    # ========================================================
    # 正则 + 字符串长度组合筛选 (cdef 实现)
    # ========================================================
    cdef NDframe _df_regex_strlen_cmp(self, int field, object pattern, int threshold, int op):
        """正则匹配 && 字符串长度比较 -> 新 NDframe"""
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n == 0:
            return new_df

        cdef Py_ssize_t i, write_idx = 0
        cdef str txt
        cdef int ln
        cdef RowC* src_rows = self.rows_c
        cdef RowC* dst_rows = new_df.rows_c
        cdef list src_txt = self.text_list
        cdef list src_sep = self.text_sep_list
        cdef object search_fn = pattern.search

        cdef list dst_txt = new_df.text_list
        cdef list dst_sep = new_df.text_sep_list

        if field == FIELD_txt:
            if op == OP_GREATER:
                for i in range(n):
                    txt = <str>src_txt[i]
                    if (txt is not None and
                        PyUnicode_GET_LENGTH(txt) > threshold and
                        search_fn(txt) is not None):
                        dst_rows[write_idx] = src_rows[i]
                        PyList_Append(dst_txt, txt)
                        PyList_Append(dst_sep, src_sep[i])
                        write_idx += 1
            elif op == OP_LESS:
                for i in range(n):
                    txt = <str>src_txt[i]
                    if (txt is not None and
                        PyUnicode_GET_LENGTH(txt) < threshold and
                        search_fn(txt) is not None):
                        dst_rows[write_idx] = src_rows[i]
                        PyList_Append(dst_txt, txt)
                        PyList_Append(dst_sep, src_sep[i])
                        write_idx += 1
            elif op == OP_EQUAL:
                for i in range(n):
                    txt = <str>src_txt[i]
                    if (txt is not None and
                        PyUnicode_GET_LENGTH(txt) == threshold and
                        search_fn(txt) is not None):
                        dst_rows[write_idx] = src_rows[i]
                        PyList_Append(dst_txt, txt)
                        PyList_Append(dst_sep, src_sep[i])
                        write_idx += 1
        elif field == FIELD_sep:
            if op == OP_GREATER:
                for i in range(n):
                    txt = <str>src_sep[i]
                    if (txt is not None and
                        PyUnicode_GET_LENGTH(txt) > threshold and
                        search_fn(txt) is not None):
                        dst_rows[write_idx] = src_rows[i]
                        PyList_Append(dst_txt, src_txt[i])
                        PyList_Append(dst_sep, txt)
                        write_idx += 1
            elif op == OP_LESS:
                for i in range(n):
                    txt = <str>src_sep[i]
                    if (txt is not None and
                        PyUnicode_GET_LENGTH(txt) < threshold and
                        search_fn(txt) is not None):
                        dst_rows[write_idx] = src_rows[i]
                        PyList_Append(dst_txt, src_txt[i])
                        PyList_Append(dst_sep, txt)
                        write_idx += 1
            elif op == OP_EQUAL:
                for i in range(n):
                    txt = <str>src_sep[i]
                    if (txt is not None and
                        PyUnicode_GET_LENGTH(txt) == threshold and
                        search_fn(txt) is not None):
                        dst_rows[write_idx] = src_rows[i]
                        PyList_Append(dst_txt, src_txt[i])
                        PyList_Append(dst_sep, txt)
                        write_idx += 1
        else:
            return new_df

        new_df.length = write_idx
        return new_df


    cdef void _replace_cross(self, int src_field, int dst_field, str old_str, str new_str) except *:
        """将 src_field 中 old_str 替换为 new_str，结果写入 dst_field（就地）"""
        cdef Py_ssize_t i, n = self.length
        if n == 0:
            return

        cdef list src_list, dst_list
        cdef object replace_method = str.replace

        # 确定源列表和目标列表
        if src_field == FIELD_txt:
            src_list = self.text_list
        else:
            src_list = self.text_sep_list

        if dst_field == FIELD_sep:
            dst_list = self.text_sep_list
        else:
            dst_list = self.text_list

        for i in range(n):
            if src_list[i]:
                dst_list[i] = replace_method(src_list[i], old_str, new_str)


    cdef void _regex_replace_cross(self, int src_field, int dst_field, object pattern, str rewrite) except *:
        """将 src_field 中 pattern 替换为 rewrite，结果写入 dst_field（就地）"""
        cdef Py_ssize_t i, n = self.length
        if n == 0:
            return

        cdef list src_list, dst_list
        cdef object sub = pattern.sub

        if src_field == FIELD_txt:
            src_list = self.text_list
        else:
            src_list = self.text_sep_list

        if dst_field == FIELD_sep:
            dst_list = self.text_sep_list
        else:
            dst_list = self.text_list

        for i in range(n):
            if src_list[i]:
                dst_list[i] = sub(rewrite, src_list[i])

    # ========================================================
    # 排序
    # ========================================================
    cdef void _sort_col(self, int field_id, bint descending) except *:
        """核心排序实现：按 int field_id 排序（就地）"""
        cdef Py_ssize_t n = self.length
        if n < 2:
            return
        if field_id < FIELD_idx or field_id > FIELD_flags:
            return

        cdef Py_ssize_t i
        cdef RowC* rows = self.rows_c
        cdef list pairs = [None] * n   # 预分配列表

        for i in range(n):
            pairs[i] = (_get_field_val(&rows[i], field_id), i)

        pairs.sort(reverse=descending)
        cdef list indices = [idx for _, idx in pairs]

        cdef RowC* temp_rows = <RowC*>malloc(<size_t>n * sizeof(RowC))
        if not temp_rows:
            raise MemoryError()

        cdef list temp_text = [None] * n
        cdef list temp_sep  = [None] * n
        cdef list src_text = self.text_list
        cdef list src_sep  = self.text_sep_list

        try:
            for i in range(n):
                idx = indices[i]
                temp_rows[i] = rows[idx]
                temp_text[i] = src_text[idx]
                temp_sep[i]  = src_sep[idx]

            memcpy(rows, temp_rows, <size_t>n * sizeof(RowC))
            self.text_list = temp_text
            self.text_sep_list = temp_sep
        finally:
            free(temp_rows)

    @staticmethod
    def concat(NDframe df1, NDframe df2):
        """拼接两个 NDframe，返回新 NDframe

        Usage: result = NDframe.concat(df_a, df_b)
        """
        cdef Py_ssize_t len1 = df1.length
        cdef Py_ssize_t len2 = df2.length
        cdef Py_ssize_t total = len1 + len2
        cdef NDframe result = NDframe(total if total > 0 else 1)

        if total == 0:
            return result

        # 批量复制 C 数组
        memcpy(result.rows_c, df1.rows_c, len1 * sizeof(RowC))
        memcpy(result.rows_c + len1, df2.rows_c, len2 * sizeof(RowC))

        # 合并 Python 列表（C 级拼接，高效）
        result.text_list = df1.text_list + df2.text_list
        result.text_sep_list = df1.text_sep_list + df2.text_sep_list

        result.length = total
        return result

    # ========================================================
    # 就地算术操作 (对应 C++ sub/add)
    # ========================================================
    cdef void _add_col(self, int field, int value) except *:
        """对该字段所有值加法"""
        cdef Py_ssize_t i, n = self.length
        if n == 0 or value == 0:
            return
        if field < FIELD_idx or field > FIELD_flags:
            return
        cdef RowC* rows = self.rows_c

        for i in range(n):
            _add_field_value(&rows[i], field, value)


    cdef void _sub_col(self, int field, int value) except *:
        """对该字段所有值减法（不低于0）"""
        cdef Py_ssize_t i, n = self.length
        if n == 0 or value == 0:
            return
        if field < FIELD_idx or field > FIELD_flags:
            return
        cdef RowC* rows = self.rows_c
        cdef int v

        for i in range(n):
            v = _get_field_val(&rows[i], field)
            if v > value:
                (&rows[i].idx)[field] = v - value
            else:
                (&rows[i].idx)[field] = 0

    cdef NDframe _df_drop(self, int index):
        """返回不含 index 行的新 NDframe"""
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n == 0:
            return new_df

        cdef Py_ssize_t i, write_idx = 0
        cdef RowC* src_rows = self.rows_c
        cdef RowC* dst_rows = new_df.rows_c
        cdef list src_txt = self.text_list
        cdef list src_sep = self.text_sep_list
        cdef list new_text = [None] * n
        cdef list new_sep = [None] * n

        for i in range(n):
            if i != index:
                dst_rows[write_idx] = src_rows[i]
                new_text[write_idx] = src_txt[i]
                new_sep[write_idx]  = src_sep[i]
                write_idx += 1

        new_df.length = write_idx
        if write_idx < n:
            del new_text[write_idx:]
            del new_sep[write_idx:]
        new_df.text_list = new_text
        new_df.text_sep_list = new_sep
        return new_df

    cdef NDframe _df_drops(self, object idx_set):
        """返回不含 idx_set 中索引的新 NDframe"""
        cdef Py_ssize_t n = self.length
        cdef NDframe new_df = NDframe(n if n > 0 else 1)
        if n == 0:
            return new_df

        cdef set idx_set_c = set(idx_set)   # 转换为 set 加速 in 测试
        cdef Py_ssize_t i, write_idx = 0
        cdef RowC* src_rows = self.rows_c
        cdef RowC* dst_rows = new_df.rows_c
        cdef list src_txt = self.text_list
        cdef list src_sep = self.text_sep_list
        cdef list new_text = [None] * n
        cdef list new_sep = [None] * n

        for i in range(n):
            if i not in idx_set_c:
                dst_rows[write_idx] = src_rows[i]
                new_text[write_idx] = src_txt[i]
                new_sep[write_idx]  = src_sep[i]
                write_idx += 1

        new_df.length = write_idx
        if write_idx < n:
            del new_text[write_idx:]
            del new_sep[write_idx:]
        new_df.text_list = new_text
        new_df.text_sep_list = new_sep
        return new_df

    cdef void _apply_if_greater_add(self, int cond_id, int target_id, int threshold, int offset) except *:
        cdef Py_ssize_t i, n = self.length
        if n == 0 or offset == 0:
            return
        if cond_id < FIELD_idx or cond_id > FIELD_flags:
            return
        cdef RowC* rows = self.rows_c

        for i in range(n):
            if _get_field_val(&rows[i], cond_id) > threshold:
                _add_field_value(&rows[i], target_id, offset)

    cdef void _apply_if_less_add(self, int cond_id, int target_id, int threshold, int offset) except *:
        """如果 cond_id < threshold，则 target_id += offset"""
        cdef Py_ssize_t i, n = self.length
        if n == 0 or offset == 0:
            return
        if cond_id < FIELD_idx or cond_id > FIELD_flags:
            return
        cdef RowC* rows = self.rows_c

        for i in range(n):
            if _get_field_val(&rows[i], cond_id) < threshold:
                _add_field_value(&rows[i], target_id, offset)

    cdef void _apply_if_greater_add_y(self, int cond_id, int threshold, int offset) except *:
        cdef Py_ssize_t i, n = self.length
        if n == 0 or offset == 0:
            return
        if cond_id < FIELD_idx or cond_id > FIELD_flags:
            return
        cdef RowC* rows = self.rows_c

        for i in range(n):
            if _get_field_val(&rows[i], cond_id) > threshold:
                rows[i].y0 += offset
                rows[i].y1 += offset

    cdef void _apply_if_less_add_y(self, int cond_id, int threshold, int offset) except *:
        """如果 cond_id < threshold，则 y0 += offset, y1 += offset"""
        cdef Py_ssize_t i, n = self.length
        if n == 0 or offset == 0:
            return
        if cond_id < FIELD_idx or cond_id > FIELD_flags:
            return
        cdef RowC* rows = self.rows_c

        for i in range(n):
            if _get_field_val(&rows[i], cond_id) < threshold:
                rows[i].y0 += offset
                rows[i].y1 += offset


    cdef list _col_values(self, int field):
        """返回该字段所有值的列表"""
        cdef Py_ssize_t n = self.length
        cdef Py_ssize_t i
        cdef RowC* rows = self.rows_c
        cdef list result = None
        if field == FIELD_txt:
            return self.text_list[:]          # 返回拷贝
        elif field == FIELD_sep:
            return self.text_sep_list[:]      # 返回拷贝
        else:
            result = [0] * n
            for i in range(n):
                result[i] = _get_field_val(&rows[i], field)
            return result


    # ========================================================
    # 就地正则删除
    # ========================================================
    cdef NDframe _drop_in_regex(self, int field, object pattern):
        cdef Py_ssize_t read_idx = 0
        cdef Py_ssize_t write_idx = 0
        cdef Py_ssize_t n = self.length
        cdef str txt
        cdef object search_fn = pattern.search

        cdef RowC* rows = self.rows_c
        cdef list txt_list = self.text_list
        cdef list sep_list = self.text_sep_list

        if field == FIELD_txt:
            for read_idx in range(n):
                txt = txt_list[read_idx]
                if not (txt and search_fn(txt) is not None):
                    if read_idx != write_idx:
                        rows[write_idx] = rows[read_idx]
                        txt_list[write_idx] = txt_list[read_idx]
                        sep_list[write_idx] = sep_list[read_idx]
                    write_idx += 1
        else:
            for read_idx in range(n):
                txt = sep_list[read_idx]
                if not (txt and search_fn(txt) is not None):
                    if read_idx != write_idx:
                        rows[write_idx] = rows[read_idx]
                        txt_list[write_idx] = txt_list[read_idx]
                        sep_list[write_idx] = sep_list[read_idx]
                    write_idx += 1

        if write_idx < n:
            del txt_list[write_idx:]
            del sep_list[write_idx:]

        self.length = write_idx
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

        cdef list txt_list = self.text_list
        cdef list sep_list = self.text_sep_list

        if field == FIELD_txt:
            for i in range(n):
                txt = txt_list[i]
                if txt and search_fn(txt) is not None:
                    res.append(i)
        else:
            for i in range(n):
                txt = sep_list[i]
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

        cdef list txt_list = self.text_list
        cdef list sep_list = self.text_sep_list

        if field == FIELD_txt:
            for i in range(n):
                if (rows[i].x0 > x0 and rows[i].x1 < x1 and
                    rows[i].y0 > y0 and rows[i].y1 < y1):
                    txt = txt_list[i]
                    if txt and search_fn(txt) is not None:
                        res.append(i)
        else:
            for i in range(n):
                if (rows[i].x0 > x0 and rows[i].x1 < x1 and
                    rows[i].y0 > y0 and rows[i].y1 < y1):
                    txt = sep_list[i]
                    if txt and search_fn(txt) is not None:
                        res.append(i)
        return res

    cdef list _idx_strlen_cmp(self, int field, int n, int op):
        cdef Py_ssize_t length = self.length
        cdef list res = []
        cdef Py_ssize_t i

        cdef list txt_list = self.text_list
        cdef list sep_list = self.text_sep_list

        if field == FIELD_txt:
            if op == OP_GREATER:
                for i in range(length):
                    if PyUnicode_GET_LENGTH(txt_list[i]) > n:
                        res.append(i)
            elif op == OP_LESS:
                for i in range(length):
                    if PyUnicode_GET_LENGTH(txt_list[i]) < n:
                        res.append(i)
            elif op == OP_EQUAL:
                for i in range(length):
                    if PyUnicode_GET_LENGTH(txt_list[i]) == n:
                        res.append(i)
        else:
            if op == OP_GREATER:
                for i in range(length):
                    if PyUnicode_GET_LENGTH(sep_list[i]) > n:
                        res.append(i)
            elif op == OP_LESS:
                for i in range(length):
                    if PyUnicode_GET_LENGTH(sep_list[i]) < n:
                        res.append(i)
            elif op == OP_EQUAL:
                for i in range(length):
                    if PyUnicode_GET_LENGTH(sep_list[i]) == n:
                        res.append(i)
        return res

    cdef list _idx_regex_strlen_cmp(self, int field, object pattern, int n, int op):
        cdef Py_ssize_t length = self.length
        cdef list res = []
        cdef Py_ssize_t i
        cdef str txt
        cdef object search_fn = pattern.search

        cdef list txt_list = self.text_list
        cdef list sep_list = self.text_sep_list

        if field == FIELD_txt:
            if op == OP_GREATER:
                for i in range(length):
                    txt = txt_list[i]
                    if txt is not None and PyUnicode_GET_LENGTH(txt) > n and search_fn(txt) is not None:
                        res.append(i)
            elif op == OP_LESS:
                for i in range(length):
                    txt = txt_list[i]
                    if txt is not None and PyUnicode_GET_LENGTH(txt) < n and search_fn(txt) is not None:
                        res.append(i)
            elif op == OP_EQUAL:
                for i in range(length):
                    txt = txt_list[i]
                    if txt is not None and PyUnicode_GET_LENGTH(txt) == n and search_fn(txt) is not None:
                        res.append(i)
        else:
            if op == OP_GREATER:
                for i in range(length):
                    txt = sep_list[i]
                    if txt is not None and PyUnicode_GET_LENGTH(txt) > n and search_fn(txt) is not None:
                        res.append(i)
            elif op == OP_LESS:
                for i in range(length):
                    txt = sep_list[i]
                    if txt is not None and PyUnicode_GET_LENGTH(txt) < n and search_fn(txt) is not None:
                        res.append(i)
            elif op == OP_EQUAL:
                for i in range(length):
                    txt = sep_list[i]
                    if txt is not None and PyUnicode_GET_LENGTH(txt) == n and search_fn(txt) is not None:
                        res.append(i)
        return res

    cdef list _idx_filter_by_int(self, int field, int arg1, int arg2, int op):
        cdef Py_ssize_t length = self.length
        cdef list res = []
        cdef Py_ssize_t i
        cdef int val
        cdef RowC* rows = self.rows_c

        if op == OP_GREATER:
            for i in range(length):
                if _get_field_val(&rows[i], field) > arg1:
                    res.append(i)
        elif op == OP_LESS:
            for i in range(length):
                if _get_field_val(&rows[i], field) < arg1:
                    res.append(i)
        elif op == OP_EQUAL:
            for i in range(length):
                if _get_field_val(&rows[i], field) == arg1:
                    res.append(i)
        elif op == OP_BETWEEN:
            for i in range(length):
                val = _get_field_val(&rows[i], field)
                if val > arg1 and val < arg2:
                    res.append(i)
        return res


    # ========================================================
    # 删除 / 合并 (cdef 核心实现)
    # ========================================================
    cdef NDframe _drop(self, int index):
        """根据索引删除单行（就地），返回 self"""
        cdef Py_ssize_t n = self.length
        if n == 0 or index < 0 or index >= n:
            return self

        cdef Py_ssize_t move_count = n - index - 1
        cdef RowC* rows = self.rows_c
        if move_count > 0:
            memmove(&rows[index], &rows[index + 1], move_count * sizeof(RowC))

        self.text_list.pop(index)
        self.text_sep_list.pop(index)
        self.length = n - 1
        return self

    cdef NDframe _drop_in_bbox(self, int x0, int y0, int x1, int y1):
        """删除边界框内的行（就地），返回 self"""
        cdef Py_ssize_t n = self.length
        if n == 0:
            return self

        cdef Py_ssize_t read_idx = 0
        cdef Py_ssize_t write_idx = 0
        cdef RowC* rows = self.rows_c
        cdef list txt_list = self.text_list
        cdef list sep_list = self.text_sep_list

        for read_idx in range(n):
            if not (rows[read_idx].x0 > x0 and rows[read_idx].y0 > y0 and
                    rows[read_idx].x1 < x1 and rows[read_idx].y1 < y1):
                if read_idx != write_idx:
                    rows[write_idx] = rows[read_idx]
                    txt_list[write_idx] = txt_list[read_idx]
                    sep_list[write_idx] = sep_list[read_idx]
                write_idx += 1

        if write_idx < n:
            del txt_list[write_idx:]
            del sep_list[write_idx:]

        self.length = write_idx
        return self

    cdef NDframe _merge_line(self, int width_threshold, int height_threshold, str sep=""):
        """cdef 实现：合并满足相邻条件的行，返回新 NDframe"""
        cdef Py_ssize_t n = self.length
        if n == 0:
            return NDframe(1)

        cdef NDframe result = NDframe(n)
        cdef RowC* src_rows = self.rows_c
        cdef RowC* dst_rows = result.rows_c
        cdef list src_txt = self.text_list
        cdef list src_sep = self.text_sep_list
        cdef list dst_txt = [None] * n
        cdef list dst_sep = [None] * n

        cdef Py_ssize_t i = 0
        cdef Py_ssize_t write_idx = 0
        cdef Py_ssize_t curr
        cdef int min_x0, min_y0, max_x1, max_y1, max_height
        cdef int new_width, new_height
        cdef list merged_text_parts
        cdef list merged_sep_parts
        cdef RowC row
        cdef str part_txt, part_sep

        while i < n:
            min_x0 = src_rows[i].x0
            min_y0 = src_rows[i].y0
            max_x1 = src_rows[i].x1
            max_y1 = src_rows[i].y1
            max_height = src_rows[i].height

            merged_text_parts = [src_txt[i]]
            merged_sep_parts = [src_sep[i]]
            curr = i

            while curr + 1 < n:
                # 检查相邻条件
                if (src_rows[curr + 1].x0 - src_rows[curr].x1 < width_threshold and
                    src_rows[curr + 1].y0 - src_rows[curr].y0 < height_threshold and
                    src_rows[curr + 1].x0 - src_rows[curr].x1 > -width_threshold and
                    src_rows[curr + 1].y0 - src_rows[curr].y0 > -height_threshold):
                    if src_rows[curr + 1].x0 < min_x0:
                        min_x0 = src_rows[curr + 1].x0
                    if src_rows[curr + 1].y0 < min_y0:
                        min_y0 = src_rows[curr + 1].y0
                    if src_rows[curr + 1].x1 > max_x1:
                        max_x1 = src_rows[curr + 1].x1
                    if src_rows[curr + 1].y1 > max_y1:
                        max_y1 = src_rows[curr + 1].y1
                    if src_rows[curr + 1].height > max_height:
                        max_height = src_rows[curr + 1].height

                    merged_text_parts.append(src_txt[curr + 1])
                    merged_sep_parts.append(src_sep[curr + 1])
                    curr += 1
                else:
                    break

            # 合并后的文本
            merged_text = sep.join(merged_text_parts)
            merged_sep = sep.join(merged_sep_parts)

            new_width = max_x1 - min_x0
            new_height = max_y1 - min_y0
            if new_height < max_height:
                new_height = max_height

            row.idx = <int>write_idx
            row.x0 = min_x0
            row.y0 = min_y0
            row.x1 = max_x1
            row.y1 = max_y1
            row.width = new_width
            row.height = new_height
            row.page = src_rows[i].page
            row.flags = src_rows[i].flags

            # 直接赋值到结果 DF
            dst_rows[write_idx] = row
            dst_txt[write_idx] = merged_text
            dst_sep[write_idx] = merged_sep
            write_idx += 1

            i = curr + 1

        # 更新结果 DF 的长度并截断多余元素
        result.length = write_idx
        if write_idx < n:
            del dst_txt[write_idx:]
            del dst_sep[write_idx:]
        result.text_list = dst_txt
        result.text_sep_list = dst_sep
        return result

    cdef list _diff_col(self, int field):
        """C 级别的一阶差分实现"""
        cdef Py_ssize_t n = self.length
        if n == 0:
            return []
        if field < FIELD_idx or field > FIELD_flags:
            printf("warning: unsupported field for diff!\n")
            return []

        cdef RowC* rows = self.rows_c
        cdef list res = [0] * n   # 预分配，第一个元素稍后设为 None
        cdef Py_ssize_t i
        cdef int prev = _get_field_val(&rows[0], field)
        cdef int curr

        for i in range(1, n):
            curr = _get_field_val(&rows[i], field)
            res[i] = curr - prev
            prev = curr

        res[0] = None
        return res


    cdef double _var_col(self, int field, int ddof):
        """C 级别的方差计算实现（两趟遍历算法确保数值稳定）"""
        cdef Py_ssize_t n = self.length
        if n <= ddof:
            return 0.0

        if field < FIELD_idx or field > FIELD_flags:
            printf("warning: unsupported field for variance!\n")
            return 0.0

        cdef RowC* rows = self.rows_c
        cdef double mean_val = self._mean_col(field)   # 调用均值方法（已优化）
        cdef double sum_sq = 0.0
        cdef double d_val
        cdef Py_ssize_t i

        for i in range(n):
            d_val = _get_field_val(&rows[i], field) - mean_val
            sum_sq += d_val * d_val

        return sum_sq / (n - ddof)


    cdef double _std_col(self, int field, int ddof):
        """C 级别的标准差实现"""
        cdef double variance = self._var_col(field, ddof)
        return sqrt(variance)


    def centroid(self):
        """
        计算所有矩形中心点的平均值（质心）。
        返回 (x_center, y_center) 浮点元组。
        若 NDframe 为空，返回 (0.0, 0.0)。
        """
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
        if n == 0:
            return (0, 0)

        if field < FIELD_idx or field > FIELD_flags:
            return (0, 0)

        cdef RowC* rows = self.rows_c
        cdef dict freq = {}
        cdef int val
        cdef Py_ssize_t i

        for i in range(n):
            val = _get_field_val(&rows[i], field)
            freq[val] = freq.get(val, 0) + 1

        # 寻找众数（若多个众数，返回最小的值）
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

        # 业务规定高度在 10~250 之间，分配 256 个桶足以防备任何极端越界的脏数据
        cdef int freq[256]
        cdef int i, h

        # 1. 数组清零 (在栈上分配的数组初始值不确定，必须清零)
        for i in range(256):
            freq[i] = 0

        # 2. 遍历 C 结构体数组，直接统计频率 (0 内存分配，0 对象创建，CPU 缓存极度友好)
        for i in range(self.length):
            h = self.rows_c[i].height
            # 过滤脏数据：只统计 10 到 100 之间的有效字符高度
            if 10 <= h <= 250:
                freq[h] += 1

        # 3. 寻找众数
        cdef int max_count = 0
        cdef int mode_val = 0

        # 从 10 遍历到 250。
        # 注意：这里使用 freq[i] > max_count (严格大于)。
        # 因为 i 是从小到大遍历的，如果出现相同最大频率，后遍历的值无法替换前面的值，
        # 自动满足了“若多个众数并存，返回其中最小的那个值”的业务要求。
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
        """
        根据给定的索引列表（或集合），极速生成一个新的 NDframe。
        支持无序或重复索引。
        """
        if len(indices) == 0:
            return NDframe(1)

        cdef list idx_list = list(indices)
        cdef Py_ssize_t n_idx = len(idx_list)
        cdef NDframe new_df = NDframe(n_idx if n_idx > 0 else 1)

        cdef RowC* src_rows = self.rows_c
        cdef RowC* dst_rows = new_df.rows_c
        cdef list src_txt = self.text_list
        cdef list src_sep = self.text_sep_list
        cdef list dst_txt = [None] * n_idx
        cdef list dst_sep = [None] * n_idx

        cdef Py_ssize_t i, target
        cdef Py_ssize_t n_self = self.length

        for i in range(n_idx):
            target = <Py_ssize_t>int(idx_list[i])
            if target < 0 or target >= n_self:
                raise IndexError(f"Index {target} out of range [0, {n_self})")
            dst_rows[i] = src_rows[target]
            dst_txt[i] = src_txt[target]
            dst_sep[i] = src_sep[target]

        new_df.length = n_idx
        new_df.text_list = dst_txt
        new_df.text_sep_list = dst_sep
        return new_df


    def drop_by_idx(self, indices):
        """
        就地删除指定的一批索引，返回原 df。（比 df_drops 返回新对象更省内存）
        """
        if not indices:  # 空集合/列表，直接返回
            return self

        cdef set idx_set = set(indices)
        cdef Py_ssize_t read_idx = 0
        cdef Py_ssize_t write_idx = 0
        cdef Py_ssize_t n = self.length
        cdef RowC* rows = self.rows_c
        cdef list txt_list = self.text_list
        cdef list sep_list = self.text_sep_list

        for read_idx in range(n):
            if read_idx not in idx_set:
                if read_idx != write_idx:
                    rows[write_idx] = rows[read_idx]
                    txt_list[write_idx] = txt_list[read_idx]
                    sep_list[write_idx] = sep_list[read_idx]
                write_idx += 1

        if write_idx < n:
            del txt_list[write_idx:]
            del sep_list[write_idx:]

        self.length = write_idx
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
    # 排序 (便捷方法 + 字符串字段名合一路由)
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

    cdef int _str_to_field_id(self, str field) except -1:
        """将字段名字符串转为 int field_id"""
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
        """按字段名字符串排序（就地），返回原 df 支持链式调用"""
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
        """重置 idx 字段为从 0 开始的递增序列（就地），返回原 df"""
        cdef Py_ssize_t i
        for i in range(self.length):
            self.rows_c[i].idx = <int>i
        return self

    def add_y0y1(self, int offset):
        """y0 和 y1 同步增加 offset（就地），返回原 df"""
        cdef Py_ssize_t i
        for i in range(self.length):
            self.rows_c[i].y0 += offset
            self.rows_c[i].y1 += offset
        return self

    def apply_if_greater_add(self, str cond_field, str target_field, int threshold, int offset):
        """如果 cond_field > threshold，则 target_field += offset（就地），返回原 df"""
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

    @cython.always_allow_keywords(False)
    def df_drops(self, list indices):
        """删除指定索引列表的行，返回新 NDframe（保留不在这些索引的行）"""
        cdef set idx_set = set(indices)
        return self._df_drops(idx_set)

    @cython.always_allow_keywords(False)
    def df_drop(self, int index):
        cdef int c_idx = int(index)
        if c_idx < 0 or c_idx >= self.length:
            raise IndexError(f"Index {c_idx} out of range [0, {self.length})")
        return self._df_drop(c_idx)

    @cython.always_allow_keywords(False)
    def drop(self, int index):
        cdef int c_idx = int(index)
        if c_idx < 0 or c_idx >= self.length:
            raise IndexError(f"Index {c_idx} out of range [0, {self.length})")
        return self._drop(c_idx)

    @cython.always_allow_keywords(False)
    def drop_in_bbox(self, int x0, int y0, int x1, int y1):
        if x0 < 0: x0 = 0
        if x1 < 0: x1 = 0
        if y0 < 0: y0 = 0
        if y1 < 0: y1 = 0

        if x0 > x1 or y0 > y1:
            raise ValueError(f"def drop_in_bbox invalid boundary: x0({x0})>x1({x1}) or y0({y0})>y1({y1})")
        return self._drop_in_bbox(x0, y0, x1, y1)

    @cython.always_allow_keywords(False)
    def merge_line(self, int width_threshold, int height_threshold, str sep=""):
        if width_threshold < 0: width_threshold = 0
        if height_threshold < 0: height_threshold = 0
        return self._merge_line(width_threshold, height_threshold, sep)
