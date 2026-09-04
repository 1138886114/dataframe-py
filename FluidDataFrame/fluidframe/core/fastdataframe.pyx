# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False

cimport cython
from libc.string cimport memcpy, memmove, strncmp, memset, strncpy
from libc.stdio cimport snprintf
from libc.stdint cimport uint32_t, int32_t, int64_t
from libc.time cimport time_t, time
from libc.math cimport NAN, sqrt, log, cos, isnan
from cpython.unicode cimport Py_UCS2, PyUnicode_KIND, PyUnicode_READ_CHAR, PyUnicode_2BYTE_KIND, PyUnicode_DATA
from fluidframe.core._base cimport (
    BaseColumn, NumericType, DataType, aligned_malloc, os_aligned_free,
    TYPE_INT, TYPE_DOUBLE, TYPE_CHAR256, TYPE_UCS2, TYPE_DATE,
    TYPE_BOOL, TYPE_CATEGORY, TYPE_DATETIME, TYPE_TIME
)

cdef extern from *:
    """
    #include <stdlib.h>
    #include <time.h>
    #include <math.h>

    #if defined(_MSC_VER) || defined(_WIN32)
        #define THREAD_LOCAL __declspec(thread)
    #else
        #define THREAD_LOCAL __thread
    #endif

    static THREAD_LOCAL unsigned long long _xorshift128plus_state[2] = {
        0x8a5cd789635d2dffULL,
        0x121fd2155c472f96ULL
    };

    static inline void _xorshift128plus_seed(unsigned long long seed) {
        unsigned long long z = (seed + 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        _xorshift128plus_state[0] = z ^ (z >> 31);

        z = (seed + 0x2 * 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        _xorshift128plus_state[1] = z ^ (z >> 31);

        if (_xorshift128plus_state[0] == 0 && _xorshift128plus_state[1] == 0) {
            _xorshift128plus_state[0] = 1ULL;
        }
    }

    static inline unsigned long long _xorshift128plus_next(void) {
        unsigned long long s1 = _xorshift128plus_state[0];
        unsigned long long s0 = _xorshift128plus_state[1];
        _xorshift128plus_state[0] = s0;
        s1 ^= s1 << 23;
        s1 ^= s1 >> 17;
        s1 ^= s0;
        s1 ^= s0 >> 26;
        _xorshift128plus_state[1] = s1;
        return _xorshift128plus_state[0] + _xorshift128plus_state[1];
    }

    static inline double _random_uniform(void) {
        return (double)(_xorshift128plus_next() >> 11) / (double)(1ULL << 53);
    }

    static inline double _random_gauss(void) {
        double u1 = _random_uniform();
        double u2 = _random_uniform();
        if (u1 < 1e-15) u1 = 1e-15;
        return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
    }

    // 严格检查目标缓冲区的剩余容量，防止栈溢出
    static inline size_t _csv_write_field(char* dest, size_t dest_cap, size_t pos, const char* field, size_t field_len, char sep) {
        size_t i;
        int need_quote = 0;

        if (pos + (field_len * 2) + 2 >= dest_cap) {
            return (size_t)-1;
        }

        for (i = 0; i < field_len; i++) {
            char c = field[i];
            if (c == sep || c == '\\n' || c == '\\r' || c == '"') {
                need_quote = 1;
                break;
            }
        }

        if (need_quote) {
            dest[pos++] = '"';
            for (i = 0; i < field_len; i++) {
                if (field[i] == '"') {
                    dest[pos++] = '"';
                    dest[pos++] = '"';
                } else {
                    dest[pos++] = field[i];
                }
            }
            dest[pos++] = '"';
        } else {
            for (i = 0; i < field_len; i++) {
                dest[pos++] = field[i];
            }
        }
        return pos;
    }

    #define _FMT_KB 1024LL
    #define _FMT_MB (1024LL * 1024LL)
    #define _FMT_GB (1024LL * 1024LL * 1024LL)

    static inline void _c_fmt_bytes(char *buf, size_t cap, Py_ssize_t n) {
        if (n < _FMT_KB) {
            snprintf(buf, cap, "%zd B", n);
        } else if (n < _FMT_MB) {
            snprintf(buf, cap, "%.1f KiB", (double)n / _FMT_KB);
        } else if (n < _FMT_GB) {
            snprintf(buf, cap, "%.1f MiB", (double)n / _FMT_MB);
        } else {
            snprintf(buf, cap, "%.1f GiB", (double)n / _FMT_GB);
        }
    }
    """
    void _xorshift128plus_seed(unsigned long long seed) nogil
    double _random_uniform() nogil
    double _random_gauss() nogil
    size_t _csv_write_field(char* dest, size_t dest_cap, size_t pos, const char* field, size_t field_len, char sep) nogil
    void _c_fmt_bytes(char* buf, size_t cap, Py_ssize_t n) nogil


from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.doublecolumn cimport DoubleColumn
from fluidframe.core._base cimport (
    _vec_pearson_corr, _vec_pearson_cov, _vec_euclidean_dist,
    _vec_manhattan_dist, _vec_cosine_sim, _vec_jaccard_sim, _vec_hamming_dist
)
from fluidframe.core.char256column cimport Char256Column_SoA
from fluidframe.core.ucs2column cimport UCS2Column_SoA
from fluidframe.core.datecolumn cimport DateColumn, _days_to_ymd
from fluidframe.core.boolcolumn cimport BoolColumn
from fluidframe.core.categorycolumn cimport CategoryColumn
from fluidframe.core.datetimecolumn cimport DateTimeColumn
from fluidframe.core.timecolumn cimport TimeColumn
from fluidframe.core.index cimport IndexBase, RangeIndex, IntIndex, MultiIndex, INDEX_RANGE, INDEX_INT, INDEX_MULTI
from fluidframe.core.indexers import _ILocIndexer, _LocIndexer
from fluidframe._schema_types import Char256, Date, Bool, Category, DateTime, Time, _parse_schema


class _RollingWrapper:
    """轻量级滚动窗口包装器，委托给列的 roll_xxx 方法"""

    _SUPPORTED_TYPES = {TYPE_INT, TYPE_DOUBLE}

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __init__(self, col, window):
        if col.dtype not in self._SUPPORTED_TYPES:
            type_name = _fast_type_name(col.dtype)
            raise TypeError(f"Rolling window operations are not supported for {type_name} columns")
        self._col = col
        self._window = window
        self.length = col.length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def mean(self): return self._col.roll_mean(self._window)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sum(self): return self._col.roll_sum(self._window)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def min(self): return self._col.roll_min(self._window)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def max(self): return self._col.roll_max(self._window)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def std(self, ddof=False): return self._col.roll_std(self._window, ddof)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def var(self, ddof=False): return self._col.roll_var(self._window, ddof)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def median(self): return self._col.roll_median(self._window)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def quantile(self, q=0.5): return self._col.roll_quantile(self._window, q)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def skew(self): return self._col.roll_skew(self._window)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def kurt(self): return self._col.roll_kurt(self._window)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def corr(self, other): return self._col.roll_corr(self._window, other._col if isinstance(other, _RollingWrapper) else other)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cov(self, other): return self._col.roll_cov(self._window, other._col if isinstance(other, _RollingWrapper) else other)


cdef void _c_argsort_char256(char* data, Py_ssize_t* indices, Py_ssize_t left, Py_ssize_t right) noexcept nogil:
    if left >= right: return
    cdef Py_ssize_t i = left, j = right
    cdef Py_ssize_t pivot_idx = indices[(left + right) // 2]
    cdef char* pivot_ptr = data + (pivot_idx * 256)
    cdef Py_ssize_t temp

    while i <= j:
        while strncmp(data + (indices[i] * 256), pivot_ptr, 256) < 0: i += 1
        while strncmp(data + (indices[j] * 256), pivot_ptr, 256) > 0: j -= 1
        if i <= j:
            temp = indices[i]; indices[i] = indices[j]; indices[j] = temp
            i += 1
            j -= 1

    if left < j: _c_argsort_char256(data, indices, left, j)
    if i < right: _c_argsort_char256(data, indices, i, right)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _c_argsort_numeric(NumericType* data, Py_ssize_t* indices, Py_ssize_t left, Py_ssize_t right) noexcept nogil:
    if left >= right: return

    cdef Py_ssize_t mid = left + ((right - left) >> 1)
    cdef Py_ssize_t temp
    if data[indices[left]] > data[indices[mid]]:
        temp = indices[left]; indices[left] = indices[mid]; indices[mid] = temp
    if data[indices[left]] > data[indices[right]]:
        temp = indices[left]; indices[left] = indices[right]; indices[right] = temp
    if data[indices[mid]] > data[indices[right]]:
        temp = indices[mid]; indices[mid] = indices[right]; indices[right] = temp

    cdef Py_ssize_t pivot_idx = indices[mid]
    cdef NumericType pivot = data[pivot_idx]
    cdef Py_ssize_t i = left, j = right

    while i <= j:
        while data[indices[i]] < pivot: i += 1
        while data[indices[j]] > pivot: j -= 1
        if i <= j:
            temp = indices[i]; indices[i] = indices[j]; indices[j] = temp
            i += 1
            j -= 1

    if left < j: _c_argsort_numeric(data, indices, left, j)
    if i < right: _c_argsort_numeric(data, indices, i, right)


cdef inline int _ucs2_cmp_at(Py_UCS2* chars, Py_ssize_t* offsets, Py_ssize_t i, Py_ssize_t j) noexcept nogil:
    cdef int i_start = offsets[i], i_len = offsets[i+1] - i_start
    cdef int j_start = offsets[j], j_len = offsets[j+1] - j_start
    cdef int k, min_len = i_len if i_len < j_len else j_len
    for k in range(min_len):
        if chars[i_start + k] < chars[j_start + k]: return -1
        if chars[i_start + k] > chars[j_start + k]: return 1
    if i_len < j_len: return -1
    if i_len > j_len: return 1
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _c_argsort_ucs2(Py_UCS2* chars, Py_ssize_t* offsets, Py_ssize_t* indices, Py_ssize_t left, Py_ssize_t right) noexcept nogil:
    if left >= right: return
    cdef Py_ssize_t i = left, j = right
    cdef Py_ssize_t pivot_idx = indices[(left + right) // 2]
    cdef Py_ssize_t temp

    while i <= j:
        while _ucs2_cmp_at(chars, offsets, indices[i], pivot_idx) < 0: i += 1
        while _ucs2_cmp_at(chars, offsets, indices[j], pivot_idx) > 0: j -= 1
        if i <= j:
            temp = indices[i]; indices[i] = indices[j]; indices[j] = temp
            i += 1
            j -= 1

    if left < j: _c_argsort_ucs2(chars, offsets, indices, left, j)
    if i < right: _c_argsort_ucs2(chars, offsets, indices, i, right)


cdef inline void _c_shift_right(void* buf, Py_ssize_t pos, Py_ssize_t col_len, Py_ssize_t elem_size) noexcept nogil:
    cdef Py_ssize_t n = col_len - pos
    if n > 0:
        memmove(<char*>buf + (pos + 1) * elem_size,
                <char*>buf + pos * elem_size,
                n * elem_size)


cdef dict _TYPE_NAMES = {
    TYPE_INT: 'int', TYPE_DOUBLE: 'float', TYPE_CHAR256: 'char256',
    TYPE_UCS2: 'ucs2', TYPE_DATE: 'date', TYPE_BOOL: 'bool',
    TYPE_CATEGORY: 'category', TYPE_DATETIME: 'datetime', TYPE_TIME: 'time',
}

cdef dict _TYPE_PYCLASSES = {
    TYPE_INT: int, TYPE_DOUBLE: float, TYPE_CHAR256: Char256,
    TYPE_UCS2: str, TYPE_DATE: Date, TYPE_BOOL: bool,
    TYPE_CATEGORY: Category, TYPE_DATETIME: DateTime, TYPE_TIME: Time,
}


@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class FastDataFrame:

    cdef BaseColumn _reorder_column(self, BaseColumn col, Py_ssize_t* indices, Py_ssize_t n, bint inplace):
        if col.dtype == TYPE_INT:
            return (<IntColumn>col).take_by_indices(indices, n, inplace)
        elif col.dtype == TYPE_DOUBLE:
            return (<DoubleColumn>col).take_by_indices(indices, n, inplace)
        elif col.dtype == TYPE_CHAR256:
            return (<Char256Column_SoA>col).take_by_indices(indices, n, inplace)
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column_SoA>col).take_by_indices(indices, n, inplace)
        elif col.dtype == TYPE_DATE:
            return (<DateColumn>col).take_by_indices(indices, n, inplace)
        elif col.dtype == TYPE_BOOL:
            return (<BoolColumn>col).take_by_indices(indices, n, inplace)
        elif col.dtype == TYPE_CATEGORY:
            return (<CategoryColumn>col).take_by_indices(indices, n, inplace)
        elif col.dtype == TYPE_DATETIME:
            return (<DateTimeColumn>col).take_by_indices(indices, n, inplace)
        elif col.dtype == TYPE_TIME:
            return (<TimeColumn>col).take_by_indices(indices, n, inplace)
        raise TypeError(f"Unsupported column dtype: {col.dtype}")

    cdef str _type_name(self, BaseColumn col):
        return _TYPE_NAMES.get(col.dtype, 'unknown')

    cdef BaseColumn _diff_column(self, BaseColumn col, int periods):
        if col.dtype == TYPE_INT:
            return (<IntColumn>col).diff(periods)
        elif col.dtype == TYPE_DOUBLE:
            return (<DoubleColumn>col).diff(periods)
        raise TypeError(f"diff not supported for column type {self._type_name(col)}")

    cdef void _drop_row_column(self, BaseColumn col, Py_ssize_t row_idx):
        col.drop_row(row_idx)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __init__(self, dict schema=None):
        self.columns = {}
        self.df_length = 0
        self.index = RangeIndex(0, 0)

        if schema:
            parsed = _parse_schema(schema)
            for col_name, dtype in parsed.items():
                if dtype == TYPE_INT:
                    self.columns[col_name] = IntColumn()
                elif dtype == TYPE_DOUBLE:
                    self.columns[col_name] = DoubleColumn()
                elif dtype == TYPE_CHAR256:
                    self.columns[col_name] = Char256Column_SoA()
                elif dtype == TYPE_UCS2:
                    self.columns[col_name] = UCS2Column_SoA()
                elif dtype == TYPE_DATE:
                    self.columns[col_name] = DateColumn()
                elif dtype == TYPE_BOOL:
                    self.columns[col_name] = BoolColumn()
                elif dtype == TYPE_CATEGORY:
                    self.columns[col_name] = CategoryColumn()
                elif dtype == TYPE_DATETIME:
                    self.columns[col_name] = DateTimeColumn()
                elif dtype == TYPE_TIME:
                    self.columns[col_name] = TimeColumn()

    def __len__(self):
        """Return the number of rows in the DataFrame."""
        return self.df_length

    cdef inline str _fast_type_name(self, int dt):
        """纯 C 级别的类型映射，零装箱开销"""
        if dt == TYPE_INT: return 'i64'
        elif dt == TYPE_DOUBLE: return 'f64'
        elif dt == TYPE_CHAR256: return 'str'
        elif dt == TYPE_UCS2: return 'str'
        elif dt == TYPE_DATE: return 'date'
        elif dt == TYPE_DATETIME: return 'datetime'
        elif dt == TYPE_TIME: return 'time'
        elif dt == TYPE_BOOL: return 'bool'
        elif dt == TYPE_CATEGORY: return 'cat'
        return 'unknown'

    cdef int _display_width(self, str s):
        """计算 UTF-8 字符串的真实控制台显示宽度 (兼容中文/全角字符)"""
        cdef int w = 0
        cdef Py_UCS4 c
        for ch in s:
            c = ord(ch)
            if (c >= 0x4E00 and c <= 0x9FFF) or \
               (c >= 0x3000 and c <= 0x3FFF) or \
               (c >= 0xFF00 and c <= 0xFFEF):
                w += 2
            else:
                w += 1
        return w

    cdef str _truncate_str(self, str s, int max_display_width=40):
        cdef int w = 0
        cdef Py_ssize_t i = 0
        cdef Py_UCS4 c
        if self._display_width(s) <= max_display_width:
            return s

        for ch in s:
            c = ord(ch)
            w += 2 if ((c >= 0x4E00 and c <= 0x9FFF) or (c >= 0x3000 and c <= 0x3FFF) or (c >= 0xFF00 and c <= 0xFFEF)) else 1
            if w > max_display_width - 3:
                break
            i += 1
        return s[:i] + "..."

    cdef str _format_value_fast(self, BaseColumn col, Py_ssize_t i):
        cdef int dt = col.dtype
        cdef char buf[128]
        cdef int sz = 0
        cdef double dval
        cdef long long tval
        cdef int h, mm, s
        cdef object val
        cdef int32_t* date_ptr
        cdef int dy, dm, dd

        if dt == TYPE_INT:
            sz = snprintf(buf, sizeof(buf), "%d", (<IntColumn>col).data_c[i])
            return buf[:sz].decode('ascii')

        elif dt == TYPE_DOUBLE:
            dval = (<DoubleColumn>col).data_c[i]
            if dval != dval:
                return "null"
            sz = snprintf(buf, sizeof(buf), "%.6g", dval)
            return buf[:sz].decode('ascii')

        elif dt == TYPE_DATE:
            date_ptr = <int32_t*>(<DateColumn>col).data_c
            _days_to_ymd(date_ptr[i], &dy, &dm, &dd)
            sz = snprintf(buf, sizeof(buf), "%04d-%02d-%02d", dy, dm, dd)
            return buf[:sz].decode('ascii')

        elif dt == TYPE_TIME:
            tval = (<TimeColumn>col).data_c[i]
            h = <int>(tval / 3600)
            mm = <int>((tval % 3600) / 60)
            s = <int>(tval % 60)
            sz = snprintf(buf, sizeof(buf), "%02d:%02d:%02d", h, mm, s)
            return buf[:sz].decode('ascii')

        elif dt == TYPE_BOOL:
            return "true" if (<BoolColumn>col).data_c[i] else "false"

        val = col[i]
        if val is None or val == '': return "null"
        if dt in (TYPE_CHAR256, TYPE_UCS2, TYPE_CATEGORY):
            return f'"{str(val).replace(chr(10), "\\n").replace(chr(13), "\\r")}"'
        return str(val)

    cdef tuple _get_header_strings(self):
        cdef list names = []
        cdef list types = []
        cdef MultiIndex mi
        cdef int j

        if self.index is None or self.index.index_type == INDEX_RANGE:
            names.append("")
            types.append("[range]")
        elif self.index.index_type == INDEX_INT:
            names.append("index")
            types.append("[i64]")
        elif self.index.index_type == INDEX_MULTI:
            mi = <MultiIndex>self.index
            for j in range(mi.nlevels):
                n = mi.names[j] if mi.names[j] is not None else f"level_{j}"
                names.append(self._truncate_str(str(n)))
                types.append("[multi]")

        for col_name, col in self.columns.items():
            names.append(self._truncate_str(str(col_name)))
            types.append(f"[{self._fast_type_name(col.dtype)}]")

        return names, types

    cdef list _get_row_strings(self, Py_ssize_t i):
        cdef list row_strs = []
        cdef RangeIndex ri
        cdef IntIndex ii
        cdef MultiIndex mi
        cdef int j, code

        if self.index is None:
            row_strs.append(str(i))
        elif self.index.index_type == INDEX_RANGE:
            ri = <RangeIndex>self.index
            row_strs.append(str(ri.start + i * ri.step))
        elif self.index.index_type == INDEX_INT:
            ii = <IntIndex>self.index
            row_strs.append(str(ii.data.data_c[i]))
        elif self.index.index_type == INDEX_MULTI:
            mi = <MultiIndex>self.index
            for j in range(mi.nlevels):
                code = (<IntColumn>mi.codes[j]).data_c[i]
                row_strs.append(self._truncate_str(str(mi.levels[j][code])))

        for col_name, col in self.columns.items():
            row_strs.append(self._truncate_str(self._format_value_fast(col, i)))

        return row_strs

    @cython.binding(False)
    def to_string(self, Py_ssize_t max_rows=80):
        cdef Py_ssize_t n = self.df_length
        cdef list row_indices = []
        cdef Py_ssize_t half

        if n <= max_rows:
            row_indices = list(range(n))
        else:
            half = max_rows // 2
            row_indices = list(range(half)) + [-1] + list(range(n - half, n))

        names, types = self._get_header_strings()
        cdef int num_display_cols = len(names)

        cdef list table_data = []
        cdef Py_ssize_t r
        for r in row_indices:
            if r == -1:
                table_data.append(["..."] * num_display_cols)
            else:
                table_data.append(self._get_row_strings(r))

        cdef list col_widths = [0] * num_display_cols
        cdef int c, max_w, current_w
        for c in range(num_display_cols):
            max_w = self._display_width(names[c])
            current_w = self._display_width(types[c])
            if current_w > max_w: max_w = current_w
            for row in table_data:
                current_w = self._display_width(row[c])
                if current_w > max_w: max_w = current_w
            col_widths[c] = max_w + 2

        cdef list lines = []
        lines.append(f"shape: ({self.df_length}, {len(self.columns)})")

        if num_display_cols > 0:
            def make_border(left, mid, right, fill):
                return left + mid.join([fill * col_widths[k] for k in range(num_display_cols)]) + right

            lines.append(make_border("┌", "┬", "┐", "─"))

            def format_cell(text, width, align_right):
                dw = self._display_width(text)
                padding = width - dw
                return (" " * padding + text) if align_right else (text + " " * padding)

            row_str = "│"
            for c in range(num_display_cols):
                row_str += " " + format_cell(names[c], col_widths[c] - 2, False) + " " + ("│" if c == num_display_cols - 1 else "┆")
            lines.append(row_str)

            row_str = "│"
            for c in range(num_display_cols):
                row_str += " " + format_cell(types[c], col_widths[c] - 2, False) + " " + ("│" if c == num_display_cols - 1 else "┆")
            lines.append(row_str)

            lines.append(make_border("╞", "╪", "╡", "═"))

            for row_idx, row in enumerate(table_data):
                row_str = "│"
                for c in range(num_display_cols):
                    val = row[c]
                    is_num = val != "..." and types[c] in ("[i64]", "[f64]", "[range]", "[multi]")
                    row_str += " " + format_cell(val, col_widths[c] - 2, is_num) + " " + ("│" if c == num_display_cols - 1 else "┆")
                lines.append(row_str)

            lines.append(make_border("└", "┴", "┘", "─"))
        else:
            lines.append("Empty DataFrame")

        return "\n".join(lines)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def print(self, Py_ssize_t max_rows=80):
        print(self.to_string(max_rows))

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __str__(self):
        return self.to_string()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __repr__(self):
        return self.to_string()

    @property
    def loc(self):
        return _LocIndexer(self)

    @property
    def iloc(self):
        return _ILocIndexer(self)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __getattr__(self, name):
        if name in self.columns: return self.columns[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __dir__(self):
        cdef list base_dir = dir(type(self))
        return sorted(set(base_dir + list(self.columns.keys())))

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __getitem__(self, name):
        if isinstance(name, IntColumn):
            return self.loc[name]
        return self.columns[name]

    @cython.binding(False)
    def eval(self, str expr, bint inplace=False):
        """Evaluate expression against columns.

        Security: Only supports basic arithmetic and comparison operations.
        Function calls, imports, and attribute access are blocked.
        """
        cdef FastDataFrame target_df = self if inplace else self.copy()
        cdef bint is_assignment = False
        cdef str target_col = ""
        cdef str calc_expr = expr

        if "=" in expr:
            parts = expr.split("=", 1)
            if not parts[0].strip().endswith(('>', '<', '!', '=')):
                target_col = parts[0].strip()
                calc_expr = parts[1].strip()
                is_assignment = True

        # Security check: block dangerous patterns
        import re
        dangerous_patterns = [
            r'__\w+__',  # dunder attributes
            r'\bimport\b',  # import statements
            r'\bfrom\b',  # from imports
            r'\bexec\b',  # exec calls
            r'\beval\b',  # nested eval
            r'\bcompile\b',  # compile calls
            r'\bglobals\b',  # globals access
            r'\blocals\b',  # locals access
            r'\bgetattr\b',  # getattr calls
            r'\bsetattr\b',  # setattr calls
            r'\bdelattr\b',  # delattr calls
            r'\b__import__\b',  # __import__ calls
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, calc_expr, re.IGNORECASE):
                raise ValueError(f"Expression contains blocked pattern: {pattern}")

        cdef dict env = dict(self.columns)
        try:
            res_col = eval(calc_expr, {"__builtins__": {}}, env)
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression '{calc_expr}': {e}")

        if is_assignment:
            target_df.columns[target_col] = res_col
            return target_df
        else:
            return res_col

    @cython.binding(False)
    def query(self, str expr):
        mask = self.eval(expr)
        if not type(mask).__name__ in ('IntColumn', 'BoolColumn'):
            raise TypeError("Query expression must evaluate to a boolean mask (IntColumn or BoolColumn)")
        return self.loc[mask]

    @cython.binding(False)
    def append_row(self, **kwargs):
        for col_name in self.columns:
            if col_name not in kwargs:
                raise ValueError(f"Missing value for column: {col_name}")

        cdef Py_ssize_t original_df_length = self.df_length
        cdef Py_ssize_t original_index_length = self.index.length if self.index is not None else 0

        try:
            for col_name, val in kwargs.items():
                self.columns[col_name].append(val)
            self.df_length += 1

            if self.index is not None:
                if self.index.index_type == INDEX_RANGE:
                    self.index.length = self.df_length
                elif self.index.index_type == INDEX_INT:
                    (<IntIndex>self.index).append(<int>(self.df_length - 1))
                elif self.index.index_type == INDEX_MULTI:
                    raise NotImplementedError("Appending rows to MultiIndex is not supported via append_row yet.")
        except Exception:
            self.df_length = original_df_length
            for col_name, col_obj in self.columns.items():
                while col_obj.length > original_df_length:
                    col_obj.pop_back()
            if self.index is not None:
                if self.index.index_type == INDEX_RANGE:
                    self.index.length = original_index_length
                elif self.index.index_type == INDEX_INT:
                    while (<IntIndex>self.index).length > original_index_length:
                        (<IntIndex>self.index).pop_back()
            raise

    def append(self, object row, bint ignore_index=False):
        cdef FastDataFrame new_df = self.copy()
        cdef list col_names
        if isinstance(row, dict):
            new_df.append_row(**row)
        elif isinstance(row, list):
            col_names = list(new_df.columns.keys())
            if len(row) != len(col_names):
                raise ValueError(f"Length of row {len(row)} != number of columns {len(col_names)}")
            kwargs = {col_names[i]: row[i] for i in range(len(col_names))}
            new_df.append_row(**kwargs)
        else:
            raise TypeError(f"Cannot append row from {type(row)}")
        if ignore_index:
            new_df.index = RangeIndex(0, new_df.df_length)
        return new_df

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def copy(self):
        cdef Py_ssize_t n, i
        cdef Py_ssize_t* idx_c
        cdef FastDataFrame new_df = FastDataFrame()
        new_df.df_length = self.df_length
        for col_name, col_obj in self.columns.items():
            new_df.columns[col_name] = col_obj.copy()

        if self.index is not None:
            if self.index.index_type == INDEX_RANGE:
                new_df.index = RangeIndex((<RangeIndex>self.index).start,
                                          self.index.length,
                                          (<RangeIndex>self.index).step)
            elif self.index.index_type == INDEX_INT:
                n = self.index.length
                idx_c = <Py_ssize_t*>aligned_malloc(max(n, 1) * sizeof(Py_ssize_t))
                if not idx_c: raise MemoryError()
                try:
                    for i in range(n): idx_c[i] = i
                    new_df.index = self.index._take(idx_c, n)
                finally:
                    os_aligned_free(idx_c)
            else:
                new_df.index = self.index
        return new_df

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_front(self):
        if self.df_length == 0: raise IndexError("pop from empty DataFrame")
        row = {}
        for col_name, col_obj in self.columns.items():
            row[col_name] = col_obj.pop_front()
        self.df_length -= 1
        if self.index is not None and self.index.length > 0:
            if self.index.index_type == INDEX_RANGE:
                (<RangeIndex>self.index).start += (<RangeIndex>self.index).step
            elif self.index.index_type == INDEX_INT:
                (<IntIndex>self.index).pop_back()
            self.index.length -= 1
        return row

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_back(self):
        if self.df_length == 0: raise IndexError("pop from empty DataFrame")
        row = {}
        for col_name, col_obj in self.columns.items():
            row[col_name] = col_obj.pop_back()
        self.df_length -= 1
        if self.index is not None and self.index.length > 0:
            if self.index.index_type == INDEX_INT:
                (<IntIndex>self.index).pop_back()
            self.index.length -= 1
        return row

    @cython.binding(False)
    def concat(self, FastDataFrame other):
        if set(self.columns.keys()) != set(other.columns.keys()):
            raise ValueError("Schema mismatch during concat")
        cdef FastDataFrame result = self.copy()
        for col_name, col_obj in result.columns.items():
            col_obj.concat(other.columns[col_name])
        result.df_length = self.df_length + other.df_length
        return result

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def drop(self, str col_name):
        if col_name not in self.columns:
            raise KeyError(f"Column '{col_name}' not found")
        del self.columns[col_name]

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def argsort_by(self, str col_name, bint reverse=False):
        if col_name not in self.columns:
            raise KeyError(f"Column '{col_name}' not found")

        cdef BaseColumn col = self.columns[col_name]
        cdef Py_ssize_t n = self.df_length
        if n == 0:
            return []

        cdef Py_ssize_t* indices_c = <Py_ssize_t*>aligned_malloc(max(n, 1) * sizeof(Py_ssize_t))
        if not indices_c: raise MemoryError()
        cdef Py_ssize_t i
        cdef Py_ssize_t temp

        try:
            for i in range(n): indices_c[i] = i

            if col.dtype == TYPE_INT:
                with nogil:
                    _c_argsort_numeric((<IntColumn>col).data_c, indices_c, 0, n - 1)
            elif col.dtype == TYPE_DOUBLE:
                with nogil:
                    _c_argsort_numeric((<DoubleColumn>col).data_c, indices_c, 0, n - 1)
            elif col.dtype == TYPE_CHAR256:
                with nogil:
                    _c_argsort_char256((<Char256Column_SoA>col).data_c, indices_c, 0, n - 1)
            elif col.dtype == TYPE_UCS2:
                with nogil:
                    _c_argsort_ucs2((<UCS2Column_SoA>col).chars_c, (<UCS2Column_SoA>col).offsets_c, indices_c, 0, n - 1)
            elif col.dtype == TYPE_DATE:
                with nogil:
                    _c_argsort_numeric(<int*>(<DateColumn>col).data_c, indices_c, 0, n - 1)
            elif col.dtype == TYPE_DATETIME:
                with nogil:
                    _c_argsort_numeric(<long long*>(<DateTimeColumn>col).data_c, indices_c, 0, n - 1)
            elif col.dtype == TYPE_TIME:
                with nogil:
                    _c_argsort_numeric(<int*>(<TimeColumn>col).data_c, indices_c, 0, n - 1)
            else:
                raise TypeError("Sorting not supported for this column type")

            if reverse:
                for i in range(n // 2):
                    temp = indices_c[i]
                    indices_c[i] = indices_c[n - 1 - i]
                    indices_c[n - 1 - i] = temp

            return [indices_c[j] for j in range(n)]
        finally:
            os_aligned_free(indices_c)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sort_values(self, str col_name, bint reverse=False, bint inplace=False):
        if col_name not in self.columns:
            raise KeyError(f"Column '{col_name}' not found")

        cdef Py_ssize_t n = self.df_length
        if n == 0:
            return self if inplace else self.copy()

        cdef list py_indices = self.argsort_by(col_name, reverse=reverse)
        cdef Py_ssize_t* idx_c = <Py_ssize_t*>aligned_malloc(max(n, 1) * sizeof(Py_ssize_t))
        if not idx_c: raise MemoryError()
        cdef Py_ssize_t i
        cdef FastDataFrame result_df
        cdef str name
        cdef BaseColumn col, new_col

        try:
            for i in range(n): idx_c[i] = py_indices[i]
            if inplace:
                result_df = self
            else:
                result_df = FastDataFrame()
                result_df.df_length = n

            for name, col in self.columns.items():
                new_col = self._reorder_column(col, idx_c, n, inplace)
                if not inplace:
                    result_df.columns[name] = new_col

            if not inplace and self.index is not None:
                result_df.index = self.index._take(idx_c, n)
            elif inplace and self.index is not None:
                self.index = self.index._take(idx_c, n)

            return result_df
        finally:
            os_aligned_free(idx_c)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def diff(self, str col_name, int periods=1):
        if col_name not in self.columns:
            raise KeyError(f"Column '{col_name}' not found")
        cdef BaseColumn col = self.columns[col_name]
        return self._diff_column(col, periods)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def reset_index(self):
        cdef IntColumn idx_col = IntColumn(self.df_length)
        cdef Py_ssize_t i
        idx_col.length = self.df_length
        for i in range(self.df_length):
            idx_col.data_c[i] = <int>i
        self.columns['index'] = idx_col
        self.index = RangeIndex(0, self.df_length)

    @cython.binding(False)
    def sort_index(self, bint ascending=True, bint inplace=False):
        if 'index' not in self.columns:
            raise KeyError("No 'index' column found. Call reset_index() first.")
        return self.sort_values('index', reverse=not ascending, inplace=inplace)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.binding(False)
    def drop_with_index(self, list drop_indices, bint inplace=False):
        cdef Py_ssize_t n_orig = self.df_length
        if n_orig == 0:
            return self if inplace else self.copy()

        cdef char* drop_mask = <char*>aligned_malloc(n_orig * sizeof(char))
        if not drop_mask: raise MemoryError()
        memset(drop_mask, 0, n_orig)

        cdef Py_ssize_t idx, i, j
        cdef Py_ssize_t keep_count = 0
        cdef Py_ssize_t* keep_idx_c = NULL
        cdef FastDataFrame result_df
        cdef str name
        cdef BaseColumn col, new_col

        try:
            for py_idx in drop_indices:
                idx = <Py_ssize_t>py_idx
                if idx < 0: idx += n_orig
                if idx < 0 or idx >= n_orig:
                    raise IndexError(f"Index {py_idx} out of bounds")
                drop_mask[idx] = 1

            for i in range(n_orig):
                if drop_mask[i] == 0:
                    keep_count += 1

            if keep_count == n_orig:
                return self if inplace else self.copy()

            if keep_count == 0:
                if inplace:
                    self.df_length = 0
                    for col_obj in self.columns.values(): col_obj.length = 0
                    return self
                else:
                    return FastDataFrame()

            keep_idx_c = <Py_ssize_t*>aligned_malloc(keep_count * sizeof(Py_ssize_t))
            if not keep_idx_c: raise MemoryError()

            j = 0
            for i in range(n_orig):
                if drop_mask[i] == 0:
                    keep_idx_c[j] = i
                    j += 1

            if inplace:
                result_df = self
                self.df_length = keep_count
            else:
                result_df = FastDataFrame()
                result_df.df_length = keep_count

            for name, col in self.columns.items():
                new_col = self._reorder_column(col, keep_idx_c, keep_count, inplace)
                if not inplace:
                    result_df.columns[name] = new_col

            if self.index is not None:
                if inplace:
                    self.index = self.index._take(keep_idx_c, keep_count)
                else:
                    result_df.index = self.index._take(keep_idx_c, keep_count)

            return result_df
        finally:
            if drop_mask: os_aligned_free(drop_mask)
            if keep_idx_c: os_aligned_free(keep_idx_c)

    def drop_index(self, object labels, bint inplace=False):
        if self.index is None:
            raise ValueError("DataFrame has no index set")
        cdef list drop_positions = []
        if isinstance(labels, list):
            for label in labels:
                drop_positions.append(self.index.get_loc(label))
        else:
            drop_positions.append(self.index.get_loc(labels))
        return self.drop_with_index(drop_positions, inplace=inplace)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.binding(False)
    def drop_raw(self, Py_ssize_t row_idx):
        if row_idx < 0: row_idx += self.df_length
        if row_idx < 0 or row_idx >= self.df_length:
            raise IndexError("Row index out of range")

        cdef str name
        cdef BaseColumn col
        cdef Py_ssize_t rem_len_idx, ii
        cdef IntColumn tmp_col, idx_col
        cdef RangeIndex ri

        for name, col in self.columns.items():
            self._drop_row_column(col, row_idx)

        self.df_length -= 1
        if self.index is not None and self.index.length > 0:
            if self.index.index_type == INDEX_RANGE:
                ri = <RangeIndex>self.index
                if row_idx == 0:
                    ri.start += ri.step
                elif row_idx == self.index.length - 1:
                    pass
                else:
                    # When deleting from middle, convert to IntIndex
                    rem_len_idx = self.index.length - row_idx - 1
                    tmp_col = IntColumn(self.index.length)
                    tmp_col.length = self.index.length
                    for ii in range(self.index.length):
                        tmp_col.data_c[ii] = <int>(ri.start + ii * ri.step)
                    memmove(tmp_col.data_c + row_idx,
                            tmp_col.data_c + row_idx + 1,
                            rem_len_idx * sizeof(int))
                    tmp_col.length = self.index.length - 1
                    self.index = IntIndex(tmp_col)
                    return
            elif self.index.index_type == INDEX_INT:
                idx_col = (<IntIndex>self.index).data
                rem_len_idx = self.index.length - row_idx - 1
                if rem_len_idx > 0:
                    memmove(idx_col.data_c + row_idx,
                            idx_col.data_c + row_idx + 1,
                            rem_len_idx * sizeof(int))
                idx_col.length -= 1
            self.index.length -= 1

    @cython.binding(False)
    def argmax(self, str col_name): return self.columns[col_name].idxmax()

    @cython.binding(False)
    def argmin(self, str col_name): return self.columns[col_name].idxmin()

    @cython.binding(False)
    def unique(self, str col_name): return self.columns[col_name].unique()

    @cython.binding(False)
    def to_pickle(self, filename):
        import pickle
        cdef dict data = {'df_length': self.df_length, 'columns': {}}
        for name, col_obj in self.columns.items():
            col_len = col_obj.length
            values = [col_obj[i] for i in range(col_len)]
            data['columns'][name] = (self._type_name(col_obj), values)
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    @cython.binding(False)
    def to_csv(self, filename, sep=','):
        cdef list col_names = list(self.columns.keys())
        cdef list col_objs = [self.columns[name] for name in col_names]
        cdef Py_ssize_t i, j
        cdef Py_ssize_t num_cols = len(col_names)
        cdef size_t buf_pos = 0
        cdef size_t write_res
        cdef char sep_c = <char>ord(sep) if len(sep) == 1 else <char>ord(',')
        cdef char line_buf[65536]
        cdef size_t buf_cap = sizeof(line_buf)
        cdef bytes b_field

        with open(filename, 'wb') as f:
            for j in range(num_cols):
                b_field = col_names[j].encode('utf-8')
                if j > 0:
                    line_buf[buf_pos] = sep_c
                    buf_pos += 1
                write_res = _csv_write_field(line_buf, buf_cap, buf_pos, <const char*>b_field, len(b_field), sep_c)
                if write_res == <size_t>-1:
                    f.write(line_buf[:buf_pos])
                    buf_pos = 0
                    write_res = _csv_write_field(line_buf, buf_cap, buf_pos, <const char*>b_field, len(b_field), sep_c)
                buf_pos = write_res

            line_buf[buf_pos] = 10
            buf_pos += 1
            f.write(line_buf[:buf_pos])

            for i in range(self.df_length):
                buf_pos = 0
                for j in range(num_cols):
                    b_field = str(col_objs[j][i]).encode('utf-8')
                    if j > 0:
                        line_buf[buf_pos] = sep_c
                        buf_pos += 1
                    write_res = _csv_write_field(line_buf, buf_cap, buf_pos, <const char*>b_field, len(b_field), sep_c)
                    if write_res == <size_t>-1:
                        f.write(line_buf[:buf_pos])
                        buf_pos = 0
                        write_res = _csv_write_field(line_buf, buf_cap, buf_pos, <const char*>b_field, len(b_field), sep_c)
                    buf_pos = write_res

                line_buf[buf_pos] = 10
                buf_pos += 1
                f.write(line_buf[:buf_pos])

    @cython.binding(False)
    def to_tsv(self, filename):
        self.to_csv(filename, sep='\t')

    cdef FastDataFrame _take(self, Py_ssize_t* indices, Py_ssize_t n):
        cdef FastDataFrame result = FastDataFrame()
        result.df_length = n
        cdef str name
        cdef BaseColumn col
        for name, col in self.columns.items():
            result.columns[name] = self._reorder_column(col, indices, n, False)

        if self.index is not None:
            result.index = self.index._take(indices, n)
        return result

    @cython.binding(False)
    def head(self, Py_ssize_t n=5):
        if n < 0: raise ValueError("n must be non-negative")
        n = min(n, self.df_length)
        if n == 0:
            result = FastDataFrame()
            for col_name, col_obj in self.columns.items():
                result.columns[col_name] = col_obj.copy()
                result.columns[col_name].length = 0
            return result

        cdef Py_ssize_t* idx_c = <Py_ssize_t*>aligned_malloc(max(n, 1) * sizeof(Py_ssize_t))
        if not idx_c: raise MemoryError()
        cdef Py_ssize_t i
        try:
            for i in range(n): idx_c[i] = i
            return self._take(idx_c, n)
        finally:
            os_aligned_free(idx_c)

    @cython.binding(False)
    def tail(self, Py_ssize_t n=5):
        if n < 0: raise ValueError("n must be non-negative")
        n = min(n, self.df_length)
        if n == 0: return FastDataFrame()
        cdef Py_ssize_t start = self.df_length - n
        cdef Py_ssize_t* idx_c = <Py_ssize_t*>aligned_malloc(max(n, 1) * sizeof(Py_ssize_t))
        if not idx_c: raise MemoryError()
        cdef Py_ssize_t i
        try:
            for i in range(n): idx_c[i] = start + i
            return self._take(idx_c, n)
        finally:
            os_aligned_free(idx_c)

    @property
    def shape(self):
        return (self.df_length, len(self.columns))

    @property
    def col_names(self):
        return list(self.columns.keys())

    @property
    def dtypes(self):
        cdef dict result = {}
        cdef str name
        cdef BaseColumn col
        for name, col in self.columns.items():
            result[name] = _TYPE_PYCLASSES.get(col.dtype, None)
        return result

    @cython.binding(False)
    def rename(self, dict mapper):
        cdef FastDataFrame result = self.copy()
        cdef str old_name, new_name
        for old_name, new_name in mapper.items():
            if old_name in result.columns:
                result.columns[new_name] = result.columns.pop(old_name)
        return result

    @cython.binding(False)
    def describe(self):
        cdef dict stats = {}
        cdef str name, val, top_val
        cdef Py_ssize_t col_len, i, max_freq, freq_val
        cdef long long ts_min, ts_max
        cdef dict freq_map
        for name, col_obj in self.columns.items():
            col_len = col_obj.length
            if col_obj.dtype == TYPE_INT or col_obj.dtype == TYPE_DOUBLE:
                stats[name] = {
                    'count': col_len,
                    'mean': col_obj.mean(),
                    'std': col_obj.std(),
                    'min': col_obj.min(),
                    '50%': col_obj.median(),
                    'max': col_obj.max(),
                }
            elif col_obj.dtype == TYPE_CHAR256 or col_obj.dtype == TYPE_UCS2:
                freq_map = {}
                for i in range(col_len):
                    val = str(col_obj[i])
                    freq_map[val] = freq_map.get(val, 0) + 1
                top_val = ''
                max_freq = 0
                for val, freq_val in freq_map.items():
                    if freq_val > max_freq:
                        max_freq = freq_val
                        top_val = val
                stats[name] = {
                    'count': col_len,
                    'unique': len(freq_map),
                    'top': top_val,
                    'freq': max_freq,
                }
            elif (col_obj.dtype == TYPE_DATE or col_obj.dtype == TYPE_DATETIME
                  or col_obj.dtype == TYPE_TIME):
                if col_len > 0:
                    ts_min = col_obj.min()
                    ts_max = col_obj.max()
                else:
                    ts_min = ts_max = 0
                stats[name] = {
                    'count': col_len,
                    'min': ts_min,
                    'max': ts_max,
                }
            elif col_obj.dtype == TYPE_BOOL or col_obj.dtype == TYPE_CATEGORY:
                freq_map = {}
                for i in range(col_len):
                    val = str(col_obj[i])
                    freq_map[val] = freq_map.get(val, 0) + 1
                top_val = ''
                max_freq = 0
                for val, freq_val in freq_map.items():
                    if freq_val > max_freq:
                        max_freq = freq_val
                        top_val = val
                stats[name] = {
                    'count': col_len,
                    'unique': len(freq_map),
                    'top': top_val,
                    'freq': max_freq,
                }
        return stats

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _numeric_agg(self, str method):
        """统一反射调度数值聚合，保证与派生列兼容"""
        cdef dict result = {}
        cdef str name
        cdef object col
        for name, col in self.columns.items():
            if col.dtype == TYPE_INT or col.dtype == TYPE_DOUBLE:
                result[name] = getattr(col, method)()
        return result

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _numeric_agg_arg(self, str method, arg):
        cdef dict result = {}
        cdef str name
        cdef object col
        for name, col in self.columns.items():
            if col.dtype == TYPE_INT or col.dtype == TYPE_DOUBLE:
                result[name] = getattr(col, method)(arg)
        return result

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def mean(self): return self._numeric_agg('mean')

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def median(self): return self._numeric_agg('median')

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def mode(self):
        cdef dict result = {}
        cdef str name
        cdef object col
        for name, col in self.columns.items():
            if col.dtype == TYPE_INT or col.dtype == TYPE_DOUBLE:
                result[name] = col.mode()
        return result

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def std(self): return self._numeric_agg('std')

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def var(self): return self._numeric_agg('var')

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def max(self): return self._numeric_agg('max')

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def min(self): return self._numeric_agg('min')

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sum(self): return self._numeric_agg('sum')

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def count(self):
        cdef dict result = {}
        cdef str name
        cdef object col
        for name, col in self.columns.items():
            result[name] = col.length
        return result

    @cython.binding(False)
    def euclidean_dist_withcols(self, col1, col2):
        cdef str name1 = col1 if isinstance(col1, str) else None
        cdef str name2 = col2 if isinstance(col2, str) else None
        cdef DoubleColumn c1, c2
        c1 = <DoubleColumn>self.columns[name1] if name1 is not None else <DoubleColumn>col1
        c2 = <DoubleColumn>self.columns[name2] if name2 is not None else <DoubleColumn>col2
        return c1.euclidean_dist(c2)

    @cython.binding(False)
    def manhattan_dist_withcols(self, col1, col2):
        cdef str name1 = col1 if isinstance(col1, str) else None
        cdef str name2 = col2 if isinstance(col2, str) else None
        cdef DoubleColumn c1, c2
        c1 = <DoubleColumn>self.columns[name1] if name1 is not None else <DoubleColumn>col1
        c2 = <DoubleColumn>self.columns[name2] if name2 is not None else <DoubleColumn>col2
        return c1.manhattan_dist(c2)

    @cython.binding(False)
    def cosine_dist_withcols(self, col1, col2):
        cdef str name1 = col1 if isinstance(col1, str) else None
        cdef str name2 = col2 if isinstance(col2, str) else None
        cdef DoubleColumn c1, c2
        c1 = <DoubleColumn>self.columns[name1] if name1 is not None else <DoubleColumn>col1
        c2 = <DoubleColumn>self.columns[name2] if name2 is not None else <DoubleColumn>col2
        return c1.cosine_dist(c2)

    @cython.binding(False)
    def cosine_similarity_withcols(self, col1, col2):
        cdef str name1 = col1 if isinstance(col1, str) else None
        cdef str name2 = col2 if isinstance(col2, str) else None
        cdef DoubleColumn c1, c2
        c1 = <DoubleColumn>self.columns[name1] if name1 is not None else <DoubleColumn>col1
        c2 = <DoubleColumn>self.columns[name2] if name2 is not None else <DoubleColumn>col2
        return c1.cosine_similarity(c2)

    @cython.binding(False)
    def jaccard_sim_withcols(self, col1, col2):
        cdef str name1 = col1 if isinstance(col1, str) else None
        cdef str name2 = col2 if isinstance(col2, str) else None
        cdef DoubleColumn c1, c2
        c1 = <DoubleColumn>self.columns[name1] if name1 is not None else <DoubleColumn>col1
        c2 = <DoubleColumn>self.columns[name2] if name2 is not None else <DoubleColumn>col2
        return c1.jaccard_sim(c2)

    @cython.binding(False)
    def hamming_dist_withcols(self, col1, col2):
        cdef str name1 = col1 if isinstance(col1, str) else None
        cdef str name2 = col2 if isinstance(col2, str) else None
        cdef DoubleColumn c1, c2
        c1 = <DoubleColumn>self.columns[name1] if name1 is not None else <DoubleColumn>col1
        c2 = <DoubleColumn>self.columns[name2] if name2 is not None else <DoubleColumn>col2
        return c1.hamming_dist(c2)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def quantile(self, double q): return self._numeric_agg_arg('quantile', q)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def skew(self): return self._numeric_agg('skew')

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def kurt(self): return self._numeric_agg('kurt')

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cumsum(self): return self._numeric_agg('cumsum')

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cumprod(self): return self._numeric_agg('cumprod')

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cummax(self): return self._numeric_agg('cummax')

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cummin(self): return self._numeric_agg('cummin')

    @cython.binding(False)
    def isnull(self):
        cdef FastDataFrame result = FastDataFrame()
        result.df_length = self.df_length
        cdef str name
        cdef BaseColumn col
        cdef IntColumn new_col
        cdef Py_ssize_t i, n = self.df_length
        cdef double* d_data
        cdef char* c_data
        cdef Py_ssize_t start, end, str_len

        for name, col in self.columns.items():
            new_col = IntColumn(max(n, 1))
            new_col.length = n
            if col.dtype == TYPE_DOUBLE:
                d_data = (<DoubleColumn>col).data_c
                with nogil:
                    for i in range(n):
                        new_col.data_c[i] = isnan(d_data[i])
            elif col.dtype == TYPE_CHAR256:
                c_data = (<Char256Column_SoA>col).data_c
                for i in range(n):
                    str_len = (<Char256Column_SoA>col).offsets_c[i + 1] - (<Char256Column_SoA>col).offsets_c[i]
                    new_col.data_c[i] = 1 if str_len == 0 else 0
            elif col.dtype == TYPE_UCS2:
                for i in range(n):
                    start = (<UCS2Column_SoA>col).offsets_c[i]
                    end = (<UCS2Column_SoA>col).offsets_c[i + 1]
                    new_col.data_c[i] = 1 if end == start else 0
            else:
                memset(new_col.data_c, 0, n * sizeof(int))
            result.columns[name] = new_col
        return result

    @cython.binding(False)
    def fillna(self, double value, col_names=None):
        cdef FastDataFrame result = self.copy()
        cdef str name
        cdef BaseColumn col
        cdef Py_ssize_t i, n = self.df_length
        cdef double* d_data
        if col_names is None:
            col_names = list(self.columns.keys())
        for name in col_names:
            if name not in result.columns: continue
            col = result.columns[name]
            if col.dtype == TYPE_DOUBLE:
                d_data = (<DoubleColumn>col).data_c
                with nogil:
                    for i in range(n):
                        if isnan(d_data[i]):
                            d_data[i] = value
        return result

    @cython.binding(False)
    def dropna(self):
        if self.df_length == 0: return self.copy()
        cdef Py_ssize_t n = self.df_length
        cdef char* keep_mask = <char*>aligned_malloc(n * sizeof(char))
        if not keep_mask: raise MemoryError()
        memset(keep_mask, 1, n)

        cdef Py_ssize_t i, j, keep_count = 0
        cdef double* d_data
        cdef str name
        cdef BaseColumn col
        cdef Py_ssize_t* keep_idx = NULL
        cdef Py_ssize_t start, end, str_len

        try:
            for name, col in self.columns.items():
                if col.dtype == TYPE_DOUBLE:
                    d_data = (<DoubleColumn>col).data_c
                    with nogil:
                        for i in range(n):
                            if keep_mask[i] and isnan(d_data[i]):
                                keep_mask[i] = 0
                elif col.dtype == TYPE_CHAR256:
                    for i in range(n):
                        if keep_mask[i]:
                            str_len = (<Char256Column_SoA>col).offsets_c[i + 1] - (<Char256Column_SoA>col).offsets_c[i]
                            if str_len == 0:
                                keep_mask[i] = 0
                elif col.dtype == TYPE_UCS2:
                    for i in range(n):
                        if keep_mask[i]:
                            start = (<UCS2Column_SoA>col).offsets_c[i]
                            end = (<UCS2Column_SoA>col).offsets_c[i + 1]
                            if start == end:
                                keep_mask[i] = 0

            for i in range(n):
                keep_count += keep_mask[i]

            if keep_count == n:
                return self.copy()
            if keep_count == 0:
                return FastDataFrame()

            keep_idx = <Py_ssize_t*>aligned_malloc(max(keep_count, 1) * sizeof(Py_ssize_t))
            if not keep_idx: raise MemoryError()

            j = 0
            for i in range(n):
                if keep_mask[i]:
                    keep_idx[j] = i
                    j += 1

            return self._take(keep_idx, keep_count)
        finally:
            if keep_mask: os_aligned_free(keep_mask)
            if keep_idx: os_aligned_free(keep_idx)

    @cython.binding(False)
    def groupby(self, by_cols):
        from fluidframe.core.groupby import FastDataFrameGroupBy
        if isinstance(by_cols, str):
            by_cols = [by_cols]
        cdef list by_list = list(by_cols)
        for cn in by_list:
            if cn not in self.columns:
                raise KeyError(f"Column '{cn}' not found")
        return FastDataFrameGroupBy(self, by_list)

    @staticmethod
    def _make_empty_column(int dtype):
        if dtype == TYPE_INT: return IntColumn()
        elif dtype == TYPE_DOUBLE: return DoubleColumn()
        elif dtype == TYPE_CHAR256: return Char256Column_SoA()
        elif dtype == TYPE_UCS2: return UCS2Column_SoA()
        elif dtype == TYPE_DATE: return DateColumn()
        elif dtype == TYPE_BOOL: return BoolColumn()
        elif dtype == TYPE_CATEGORY: return CategoryColumn()
        elif dtype == TYPE_DATETIME: return DateTimeColumn()
        elif dtype == TYPE_TIME: return TimeColumn()
        raise ValueError(f"Unknown dtype: {dtype}")

    @staticmethod
    def _populate_column(int dtype, Py_ssize_t n, list values):
        cdef Py_ssize_t i
        if dtype == TYPE_INT:
            int_col = IntColumn(n)
            int_col.length = n
            for i in range(n): int_col.data_c[i] = <int>values[i]
            return int_col
        elif dtype == TYPE_DOUBLE:
            dbl_col = DoubleColumn(n)
            dbl_col.length = n
            for i in range(n): dbl_col.data_c[i] = <double>values[i]
            return dbl_col
        elif dtype == TYPE_CHAR256:
            c256_col = Char256Column_SoA(n)
            for i in range(n): c256_col.append(<str>values[i])
            return c256_col
        elif dtype == TYPE_UCS2:
            ucs2_col = UCS2Column_SoA(n)
            for i in range(n): ucs2_col.append(<str>values[i])
            return ucs2_col
        elif dtype == TYPE_DATE:
            ts_col = DateColumn(n)
            ts_col.length = n
            for i in range(n): ts_col.data_c[i] = <int32_t>values[i]
            return ts_col
        elif dtype == TYPE_BOOL:
            bool_col = BoolColumn(n)
            bool_col.length = n
            for i in range(n): bool_col.data_c[i] = 1 if values[i] else 0
            return bool_col
        elif dtype == TYPE_CATEGORY:
            cat_col = CategoryColumn(n)
            for i in range(n): cat_col.append(<str>values[i])
            return cat_col
        elif dtype == TYPE_DATETIME:
            dtc_col = DateTimeColumn(n)
            for i in range(n): dtc_col.append(<str>values[i])
            return dtc_col
        elif dtype == TYPE_TIME:
            tmc_col = TimeColumn(n)
            for i in range(n): tmc_col.append(<str>values[i])
            return tmc_col
        raise ValueError(f"Unknown dtype: {dtype}")

    @staticmethod
    def _rename_columns(FastDataFrame df, columns):
        if columns is not None:
            col_names = list(df.columns.keys())
            new_names = list(columns)
            if len(new_names) != len(col_names):
                raise ValueError(f"columns length {len(new_names)} != {len(col_names)}")
            new_cols = {}
            for i in range(len(new_names)):
                new_cols[new_names[i]] = df.columns[col_names[i]]
            df.columns = new_cols

    @staticmethod
    def from_dict(dict data, dict schema, columns=None):
        cdef dict parsed = _parse_schema(schema) if schema else {}
        cdef FastDataFrame df = FastDataFrame()
        cdef str col_name
        cdef list values
        cdef Py_ssize_t n = 0
        cdef Py_ssize_t i
        cdef int dtype
        cdef list col_names, new_names
        cdef dict new_cols
        cdef IntColumn int_col
        cdef DoubleColumn dbl_col
        cdef Char256Column_SoA c256_col
        cdef UCS2Column_SoA ucs2_col
        cdef DateColumn ts_col
        cdef BoolColumn bool_col
        cdef CategoryColumn cat_col
        cdef DateTimeColumn dtc_col
        cdef TimeColumn tmc_col

        for col_name, values in data.items():
            if col_name not in parsed:
                raise ValueError(f"Column '{col_name}' not in schema")
            if n == 0:
                n = len(values)
            if len(values) != n:
                raise ValueError(f"Column '{col_name}' length {len(values)} != expected {n}")

        if n == 0:
            return df

        df.df_length = n
        df.index = RangeIndex(0, n)

        for col_name in data:
            df.columns[col_name] = FastDataFrame._populate_column(parsed[col_name], n, data[col_name])

        FastDataFrame._rename_columns(df, columns)

        return df

    @staticmethod
    def from_list(list rows, dict schema, columns=None):
        if not schema:
            raise ValueError("schema is required and must not be empty")
        cdef dict parsed = _parse_schema(schema)
        cdef FastDataFrame df = FastDataFrame()
        cdef Py_ssize_t n_rows = len(rows)
        cdef Py_ssize_t n_cols = len(parsed)
        cdef Py_ssize_t i, j
        cdef list col_order = list(parsed.keys())
        cdef int dtype
        cdef str col_name
        cdef list new_names, col_names
        cdef dict new_cols

        if n_rows == 0:
            for col_name in col_order:
                df.columns[col_name] = FastDataFrame._make_empty_column(parsed[col_name])
            return df

        df.df_length = n_rows
        df.index = RangeIndex(0, n_rows)

        for j in range(n_cols):
            col_name = col_order[j]
            col_values = [rows[i][j] for i in range(n_rows)]
            df.columns[col_name] = FastDataFrame._populate_column(parsed[col_name], n_rows, col_values)

        FastDataFrame._rename_columns(df, columns)

        return df

    @staticmethod
    def from_arrays(list arrays, dict schema, columns=None):
        if not schema:
            raise ValueError("schema is required and must not be empty")
        cdef dict parsed = _parse_schema(schema)
        cdef FastDataFrame df = FastDataFrame()
        cdef Py_ssize_t n_cols = len(arrays)
        cdef Py_ssize_t n_rows = 0
        cdef Py_ssize_t i, j
        cdef list col_order = list(schema.keys())
        cdef int dtype
        cdef str col_name
        cdef list new_names, col_names
        cdef dict new_cols

        if n_cols == 0:
            return df

        n_rows = len(arrays[0])
        for j in range(n_cols):
            if len(arrays[j]) != n_rows:
                raise ValueError(f"Array {j} length {len(arrays[j])} != expected {n_rows}")

        if n_rows == 0:
            for col_name in col_order:
                df.columns[col_name] = FastDataFrame._make_empty_column(parsed[col_name])
            return df

        df.df_length = n_rows
        df.index = RangeIndex(0, n_rows)

        for j in range(n_cols):
            col_name = col_order[j]
            df.columns[col_name] = FastDataFrame._populate_column(parsed[col_name], n_rows, arrays[j])

        FastDataFrame._rename_columns(df, columns)

        return df

    @staticmethod
    def arange(Py_ssize_t n):
        cdef FastDataFrame df = FastDataFrame()
        if n <= 0: return df
        df.df_length = n
        df.index = RangeIndex(0, n)
        cdef IntColumn col = IntColumn(max(n, 1))
        col.length = n
        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                col.data_c[i] = <int>i
        df.columns['0'] = col
        return df

    def reshape(self, Py_ssize_t rows, Py_ssize_t cols):
        cdef Py_ssize_t total_elements = self.df_length * len(self.columns)
        if total_elements != rows * cols:
            raise ValueError(f"Cannot reshape {total_elements} elements into ({rows}, {cols})")

        cdef FastDataFrame result = FastDataFrame()
        result.df_length = rows
        result.index = RangeIndex(0, rows)

        cdef list col_names = [chr(65 + i) if i < 26 else f"C{i}" for i in range(cols)]
        cdef list all_vals = []
        cdef str name
        cdef BaseColumn col_obj
        cdef Py_ssize_t i, r_idx, c_idx

        for i in range(self.df_length):
            for name in self.columns:
                col_obj = self.columns[name]
                all_vals.append(col_obj[i])

        cdef DoubleColumn col
        for c_idx in range(cols):
            col = DoubleColumn(max(rows, 1))
            col.length = rows
            for r_idx in range(rows):
                col.data_c[r_idx] = <double>all_vals[r_idx * cols + c_idx]
            result.columns[col_names[c_idx]] = col

        return result

    @staticmethod
    def randn(Py_ssize_t rows, Py_ssize_t cols):
        cdef FastDataFrame df = FastDataFrame()
        if rows <= 0 or cols <= 0: return df
        df.df_length = rows
        df.index = RangeIndex(0, rows)
        cdef list col_names = [chr(65 + i) if i < 26 else f"C{i}" for i in range(cols)]
        cdef Py_ssize_t c_idx, r_idx
        cdef DoubleColumn col
        _xorshift128plus_seed(<unsigned long long>time(NULL) ^ 12345ULL)
        for c_idx in range(cols):
            col = DoubleColumn(max(rows, 1))
            col.length = rows
            for r_idx in range(rows):
                col.data_c[r_idx] = _random_gauss()
            df.columns[col_names[c_idx]] = col
        return df

    @staticmethod
    def rand(Py_ssize_t rows, Py_ssize_t cols):
        cdef FastDataFrame df = FastDataFrame()
        if rows <= 0 or cols <= 0: return df
        df.df_length = rows
        df.index = RangeIndex(0, rows)
        cdef list col_names = [chr(65 + i) if i < 26 else f"C{i}" for i in range(cols)]
        cdef Py_ssize_t c_idx, r_idx
        cdef DoubleColumn col
        _xorshift128plus_seed(<unsigned long long>time(NULL) ^ 54321ULL)
        for c_idx in range(cols):
            col = DoubleColumn(max(rows, 1))
            col.length = rows
            for r_idx in range(rows):
                col.data_c[r_idx] = _random_uniform()
            df.columns[col_names[c_idx]] = col
        return df

    @staticmethod
    def zeros(Py_ssize_t rows, Py_ssize_t cols):
        cdef FastDataFrame df = FastDataFrame()
        if rows <= 0 or cols <= 0: return df
        df.df_length = rows
        df.index = RangeIndex(0, rows)
        cdef list col_names = [chr(65 + i) if i < 26 else f"C{i}" for i in range(cols)]
        cdef Py_ssize_t c_idx
        cdef DoubleColumn col
        for c_idx in range(cols):
            col = DoubleColumn(max(rows, 1))
            col.length = rows
            memset(col.data_c, 0, rows * sizeof(double))
            df.columns[col_names[c_idx]] = col
        return df

    @staticmethod
    def ones(Py_ssize_t rows, Py_ssize_t cols):
        cdef FastDataFrame df = FastDataFrame()
        if rows <= 0 or cols <= 0: return df
        df.df_length = rows
        df.index = RangeIndex(0, rows)
        cdef list col_names = [chr(65 + i) if i < 26 else f"C{i}" for i in range(cols)]
        cdef Py_ssize_t c_idx, r_idx
        cdef DoubleColumn col
        for c_idx in range(cols):
            col = DoubleColumn(max(rows, 1))
            col.length = rows
            with nogil:
                for r_idx in range(rows):
                    col.data_c[r_idx] = 1.0
            df.columns[col_names[c_idx]] = col
        return df

    def to_numpy(self):
        cdef list result = []
        cdef list row
        cdef Py_ssize_t i, j
        cdef list col_names = list(self.columns.keys())
        cdef list col_objs = [self.columns[name] for name in col_names]
        cdef Py_ssize_t num_cols = len(col_names)
        for i in range(self.df_length):
            row = []
            for j in range(num_cols):
                row.append(col_objs[j][i])
            result.append(row)
        return result

    cdef BaseColumn _column_from_list(self, str name, list values):
        cdef Py_ssize_t n = len(values)
        cdef Py_ssize_t i
        cdef IntColumn int_col
        cdef DoubleColumn dbl_col
        cdef Char256Column_SoA c256_col
        cdef UCS2Column_SoA ucs2_col
        cdef BoolColumn bool_col
        if n == 0:
            return IntColumn(0)

        cdef object sample = None
        for i in range(n):
            if values[i] is not None:
                sample = values[i]
                break

        if sample is None:
            dbl_col = DoubleColumn(max(n, 1))
            dbl_col.length = n
            for i in range(n): dbl_col.data_c[i] = NAN
            return dbl_col

        if isinstance(sample, bool):
            bool_col = BoolColumn(max(n, 1))
            bool_col.length = n
            for i in range(n): bool_col.data_c[i] = 1 if values[i] else 0
            return bool_col
        elif isinstance(sample, int):
            int_col = IntColumn(max(n, 1))
            int_col.length = n
            for i in range(n): int_col.data_c[i] = <int>(values[i] if values[i] is not None else 0)
            return int_col
        elif isinstance(sample, float):
            dbl_col = DoubleColumn(max(n, 1))
            dbl_col.length = n
            for i in range(n):
                dbl_col.data_c[i] = <double>values[i] if values[i] is not None else NAN
            return dbl_col
        elif isinstance(sample, str):
            try:
                (<str>sample).encode('ascii')
                c256_col = Char256Column_SoA(max(n, 1))
                for i in range(n):
                    c256_col.append(<str>(values[i] if values[i] is not None else ''))
                return c256_col
            except UnicodeEncodeError:
                ucs2_col = UCS2Column_SoA(max(n, 1))
                for i in range(n):
                    ucs2_col.append(<str>(values[i] if values[i] is not None else ''))
                return ucs2_col
        raise TypeError(f"Unsupported type: {type(sample)}")

    cdef double _get_col_value_as_double(self, BaseColumn col, Py_ssize_t i):
        if col.dtype == TYPE_INT:
            return <double>(<IntColumn>col).data_c[i]
        elif col.dtype == TYPE_DOUBLE:
            return (<DoubleColumn>col).data_c[i]
        return 0.0

    def __setitem__(self, str name, object value):
        cdef Py_ssize_t i
        cdef IntColumn int_col
        cdef DoubleColumn dbl_col
        cdef BoolColumn bool_col

        if isinstance(value, BaseColumn):
            if self.df_length > 0 and value.length != self.df_length:
                raise ValueError(f"Column length {value.length} != df_length {self.df_length}")
            self.columns[name] = value
            if self.df_length == 0 and value.length > 0:
                self.df_length = value.length
                if self.index is None or self.index.length == 0:
                    self.index = RangeIndex(0, self.df_length)
        elif isinstance(value, list):
            if self.df_length > 0 and len(value) != self.df_length:
                raise ValueError(f"List length {len(value)} != df_length {self.df_length}")
            self.columns[name] = self._column_from_list(name, value)
            if self.df_length == 0 and len(value) > 0:
                self.df_length = len(value)
                if self.index is None or self.index.length == 0:
                    self.index = RangeIndex(0, self.df_length)
        elif isinstance(value, bool):
            if self.df_length == 0: raise ValueError("Cannot fill scalar into empty DataFrame")
            bool_col = BoolColumn(max(self.df_length, 1))
            bool_col.length = self.df_length
            for i in range(self.df_length): bool_col.data_c[i] = 1 if value else 0
            self.columns[name] = bool_col
        elif isinstance(value, (int, float)):
            if self.df_length == 0: raise ValueError("Cannot fill scalar into empty DataFrame")
            if isinstance(value, int):
                int_col = IntColumn(max(self.df_length, 1))
                int_col.length = self.df_length
                for i in range(self.df_length): int_col.data_c[i] = <int>value
                self.columns[name] = int_col
            else:
                dbl_col = DoubleColumn(max(self.df_length, 1))
                dbl_col.length = self.df_length
                for i in range(self.df_length): dbl_col.data_c[i] = <double>value
                self.columns[name] = dbl_col
        else:
            raise TypeError(f"Cannot set column from {type(value)}")

    def insert_column(self, str col_name, object col_data, Py_ssize_t pos=-1):
        cdef BaseColumn col
        if isinstance(col_data, BaseColumn):
            col = col_data
        elif isinstance(col_data, list):
            col = self._column_from_list(col_name, col_data)
        else:
            raise TypeError(f"Cannot create column from {type(col_data)}")

        if self.df_length == 0 and col.length > 0:
            self.df_length = col.length

        cdef list names = list(self.columns.keys())
        cdef Py_ssize_t n = len(names)

        if pos < 0 or pos >= n:
            self.columns[col_name] = col
        else:
            new_cols = {}
            for i in range(n):
                if i == pos:
                    new_cols[col_name] = col
                new_cols[names[i]] = self.columns[names[i]]
            self.columns = new_cols

    def insert_row(self, Py_ssize_t pos, **kwargs):
        if pos < 0 or pos > self.df_length:
            raise IndexError(f"Position {pos} out of range [0, {self.df_length}]")

        for col_name in self.columns:
            if col_name not in kwargs:
                raise ValueError(f"Missing value for column '{col_name}'")

        cdef Py_ssize_t orig_len = self.df_length
        cdef Py_ssize_t ucs2_val_len, ucs2_start, _ci, _oi
        cdef UCS2Column_SoA ucs2_obj
        cdef str ucs2_val
        cdef list shifted_cols = []
        try:
            for col_name, col_obj in self.columns.items():
                val = kwargs[col_name]
                self._shift_right(col_obj, pos)
                shifted_cols.append(col_name)
                if col_obj.dtype == TYPE_INT:
                    (<IntColumn>col_obj).data_c[pos] = <int>val
                elif col_obj.dtype == TYPE_DOUBLE:
                    (<DoubleColumn>col_obj).data_c[pos] = <double>val
                elif col_obj.dtype == TYPE_CHAR256:
                    bval = (<str>val).encode('utf-8')[:255] if isinstance(val, str) else str(val).encode('utf-8')[:255]
                    memset((<Char256Column_SoA>col_obj).data_c + (pos * 256), 0, 256)
                    memcpy((<Char256Column_SoA>col_obj).data_c + (pos * 256), <char*>bval, len(bval))
                elif col_obj.dtype == TYPE_UCS2:
                    ucs2_val = <str>val if isinstance(val, str) else str(val)
                    ucs2_obj = <UCS2Column_SoA>col_obj
                    ucs2_val_len = len(ucs2_val)
                    ucs2_start = ucs2_obj.offsets_c[pos]
                    if ucs2_val_len > 0:
                        ucs2_obj._reserve_chars(ucs2_obj.char_capacity + ucs2_val_len)
                        if ucs2_obj.char_length - ucs2_start > 0:
                            memmove(ucs2_obj.chars_c + ucs2_start + ucs2_val_len,
                                    ucs2_obj.chars_c + ucs2_start,
                                    (ucs2_obj.char_length - ucs2_start) * sizeof(Py_UCS2))
                        if PyUnicode_KIND(ucs2_val) == PyUnicode_2BYTE_KIND:
                            memcpy(ucs2_obj.chars_c + ucs2_start,
                                   PyUnicode_DATA(ucs2_val),
                                   ucs2_val_len * sizeof(Py_UCS2))
                        else:
                            for _ci in range(ucs2_val_len):
                                ucs2_obj.chars_c[ucs2_start + _ci] = <Py_UCS2>PyUnicode_READ_CHAR(ucs2_val, _ci)
                        for _oi in range(pos + 1, ucs2_obj.length + 1):
                            ucs2_obj.offsets_c[_oi] += ucs2_val_len
                        ucs2_obj.char_length += ucs2_val_len
                elif col_obj.dtype == TYPE_DATE:
                    (<DateColumn>col_obj).data_c[pos] = <int32_t>val
                elif col_obj.dtype == TYPE_BOOL:
                    (<BoolColumn>col_obj).data_c[pos] = 1 if val else 0
                elif col_obj.dtype == TYPE_CATEGORY:
                    (<CategoryColumn>col_obj).codes_c[pos] = (<CategoryColumn>col_obj)._get_or_add_code(str(val) if val is not None else '')
                elif col_obj.dtype == TYPE_DATETIME:
                    (<DateTimeColumn>col_obj).data_c[pos] = 0
                elif col_obj.dtype == TYPE_TIME:
                    (<TimeColumn>col_obj).data_c[pos] = <int>val if isinstance(val, int) else 0
            self.df_length += 1
            if self.index is not None:
                if self.index.index_type == INDEX_RANGE:
                    self.index.length = self.df_length
                elif self.index.index_type == INDEX_INT:
                    self.index.length = self.df_length
        except Exception:
            self.df_length = orig_len
            for cn in shifted_cols:
                self._shift_left(self.columns[cn], pos)
            raise

    def duplicate_column(self, str src_name, str new_name=None):
        if src_name not in self.columns:
            raise KeyError(f"Column '{src_name}' not found")
        if new_name is None:
            new_name = src_name + "_dup"
        self.columns[new_name] = self.columns[src_name].copy()

    def duplicate_rows(self, list indices=None):
        if indices is None:
            return self.copy()
        cdef Py_ssize_t n = len(indices)
        cdef Py_ssize_t* idx_c = <Py_ssize_t*>aligned_malloc(max(n, 1) * sizeof(Py_ssize_t))
        if not idx_c: raise MemoryError()
        cdef Py_ssize_t i
        for i in range(n):
            idx_c[i] = indices[i]
        cdef FastDataFrame result = self._take(idx_c, n)
        os_aligned_free(idx_c)
        return result

    @property
    def T(self):
        cdef Py_ssize_t rows = self.df_length
        cdef Py_ssize_t cols = len(self.columns)
        if rows == 0 or cols == 0:
            return FastDataFrame()

        cdef list col_names = list(self.columns.keys())
        cdef FastDataFrame result = FastDataFrame()
        result.df_length = cols
        result.index = RangeIndex(0, cols)

        cdef Py_ssize_t i, j
        cdef DoubleColumn new_col
        for i in range(rows):
            new_col = DoubleColumn(max(cols, 1))
            new_col.length = cols
            for j in range(cols):
                new_col.data_c[j] = self._get_col_value_as_double(self.columns[col_names[j]], i)
            result.columns[col_names[i]] = new_col
        return result

    def reindex(self, list new_index):
        cdef Py_ssize_t new_len = len(new_index)
        cdef dict old_pos = {}
        cdef Py_ssize_t i, old_idx
        if self.index is not None and self.index.length > 0:
            old_vals = self.index.values()
            for i in range(len(old_vals)):
                old_pos[old_vals[i]] = i

        cdef FastDataFrame result = FastDataFrame()
        result.df_length = new_len
        result.index = RangeIndex(0, new_len)

        for col_name, col_obj in self.columns.items():
            new_col_list = []
            for i in range(new_len):
                if new_index[i] in old_pos:
                    old_idx = old_pos[new_index[i]]
                    new_col_list.append(col_obj[old_idx])
                else:
                    new_col_list.append(None if col_obj.dtype in (TYPE_CHAR256, TYPE_UCS2) else (NAN if col_obj.dtype == TYPE_DOUBLE else 0))
            result.columns[col_name] = self._column_from_list(col_name, new_col_list)
        return result

    def duplicated(self, list subset=None, str keep='first'):
        cdef Py_ssize_t n = self.df_length
        cdef IntColumn result = IntColumn(max(n, 1))
        result.length = n
        if n == 0:
            return result

        if subset is None:
            subset = list(self.columns.keys())

        cdef dict seen = {}
        cdef Py_ssize_t i
        cdef tuple row_key

        if keep == 'first':
            for i in range(n):
                row_key = tuple(self.columns[c][i] for c in subset)
                if row_key in seen:
                    result.data_c[i] = 1
                else:
                    seen[row_key] = i
                    result.data_c[i] = 0
        elif keep == 'last':
            for i in range(n):
                row_key = tuple(self.columns[c][i] for c in subset)
                if row_key in seen:
                    result.data_c[seen[row_key]] = 1
                seen[row_key] = i
            for i in range(n):
                row_key = tuple(self.columns[c][i] for c in subset)
                if seen[row_key] == i:
                    result.data_c[i] = 0
        else:
            for i in range(n):
                result.data_c[i] = 0
        return result

    def drop_duplicates(self, list subset=None, str keep='first', bint inplace=False):
        cdef IntColumn mask = self.duplicated(subset, keep)
        cdef Py_ssize_t n = self.df_length
        cdef Py_ssize_t keep_count = n - mask.count_nonzero()

        if keep_count == n:
            return self if inplace else self.copy()

        cdef Py_ssize_t* keep_idx = <Py_ssize_t*>aligned_malloc(max(keep_count, 1) * sizeof(Py_ssize_t))
        if not keep_idx: raise MemoryError()
        mask.gather_zero_to(keep_idx)

        cdef FastDataFrame result = self._take(keep_idx, keep_count)
        os_aligned_free(keep_idx)
        if inplace:
            self.columns = result.columns
            self.df_length = result.df_length
            self.index = result.index
            return self
        return result

    def merge(self, FastDataFrame other, str on=None, str how='inner'):
        if on is not None:
            if on not in self.columns:
                raise KeyError(f"Column '{on}' not found in left DataFrame")
            if on not in other.columns:
                raise KeyError(f"Column '{on}' not found in right DataFrame")
        else:
            common = set(self.columns.keys()) & set(other.columns.keys())
            if not common:
                raise ValueError("No common columns to merge on")
            on = sorted(common)[0]

        cdef dict left_map = {}
        cdef dict right_map = {}
        cdef Py_ssize_t i, j

        for i in range(self.df_length):
            key = self.columns[on][i]
            if key not in left_map:
                left_map[key] = []
            left_map[key].append(i)

        for j in range(other.df_length):
            key = other.columns[on][j]
            if key not in right_map:
                right_map[key] = []
            right_map[key].append(j)

        cdef list left_indices = []
        cdef list right_indices = []

        if how == 'inner':
            for key in left_map:
                if key in right_map:
                    for li in left_map[key]:
                        for ri in right_map[key]:
                            left_indices.append(li)
                            right_indices.append(ri)
        elif how == 'left':
            for key in left_map:
                rights = right_map.get(key, [])
                if not rights:
                    for li in left_map[key]:
                        left_indices.append(li)
                        right_indices.append(-1)
                else:
                    for li in left_map[key]:
                        for ri in rights:
                            left_indices.append(li)
                            right_indices.append(ri)
        elif how == 'right':
            for key in right_map:
                lefts = left_map.get(key, [])
                if not lefts:
                    for ri in right_map[key]:
                        left_indices.append(-1)
                        right_indices.append(ri)
                else:
                    for li in lefts:
                        for ri in right_map[key]:
                            left_indices.append(li)
                            right_indices.append(ri)
        elif how == 'outer':
            all_keys = set(left_map.keys()) | set(right_map.keys())
            for key in all_keys:
                lefts = left_map.get(key, [])
                rights = right_map.get(key, [])
                if not lefts:
                    for ri in rights:
                        left_indices.append(-1)
                        right_indices.append(ri)
                elif not rights:
                    for li in lefts:
                        left_indices.append(li)
                        right_indices.append(-1)
                else:
                    for li in lefts:
                        for ri in rights:
                            left_indices.append(li)
                            right_indices.append(ri)

        cdef Py_ssize_t result_len = len(left_indices)
        cdef FastDataFrame result = FastDataFrame()
        result.df_length = result_len
        result.index = RangeIndex(0, result_len)

        cdef list left_col_names = [n for n in self.columns.keys() if n != on]
        cdef list right_col_names = [n for n in other.columns.keys() if n != on]

        cdef Py_ssize_t* left_idx_c = <Py_ssize_t*>aligned_malloc(max(result_len, 1) * sizeof(Py_ssize_t))
        cdef Py_ssize_t* right_idx_c = <Py_ssize_t*>aligned_malloc(max(result_len, 1) * sizeof(Py_ssize_t))
        if not left_idx_c or not right_idx_c:
            if left_idx_c: os_aligned_free(left_idx_c)
            if right_idx_c: os_aligned_free(right_idx_c)
            raise MemoryError()
        cdef BaseColumn src_col

        try:
            for i in range(result_len):
                left_idx_c[i] = left_indices[i]
                right_idx_c[i] = right_indices[i]

            for cn in [on] + left_col_names:
                src_col = self.columns[cn]
                if src_col.dtype == TYPE_INT:
                    result.columns[cn] = (<IntColumn>src_col).gather_with_sentinel(left_idx_c, result_len, 0)
                elif src_col.dtype == TYPE_DOUBLE:
                    result.columns[cn] = (<DoubleColumn>src_col).gather_with_sentinel(left_idx_c, result_len, NAN)
                else:
                    result.columns[cn] = self._reorder_column(src_col, left_idx_c, result_len, False)

            for cn in right_col_names:
                src_col = other.columns[cn]
                out_name = cn + '_y' if cn in result.columns else cn
                if src_col.dtype == TYPE_INT:
                    result.columns[out_name] = (<IntColumn>src_col).gather_with_sentinel(right_idx_c, result_len, 0)
                elif src_col.dtype == TYPE_DOUBLE:
                    result.columns[out_name] = (<DoubleColumn>src_col).gather_with_sentinel(right_idx_c, result_len, NAN)
                else:
                    result.columns[out_name] = other._reorder_column(src_col, right_idx_c, result_len, False)

            return result
        finally:
            os_aligned_free(left_idx_c)
            os_aligned_free(right_idx_c)

    def sample(self, Py_ssize_t n=0, double frac=0.0, bint replace=False, int seed=-1):
        if seed >= 0:
            _xorshift128plus_seed(<unsigned long long>seed)
        else:
            _xorshift128plus_seed(<unsigned long long>time(NULL) ^ <unsigned long long>id(self))
        cdef Py_ssize_t total = self.df_length
        if total == 0:
            return FastDataFrame()

        cdef Py_ssize_t sample_size
        if frac > 0:
            sample_size = <Py_ssize_t>(total * frac)
            if sample_size == 0: sample_size = 1
        elif n > 0:
            sample_size = n
        else:
            sample_size = 1

        if not replace and sample_size > total:
            raise ValueError(f"Sample size {sample_size} > population {total}")

        cdef Py_ssize_t* idx_c = <Py_ssize_t*>aligned_malloc(max(sample_size, 1) * sizeof(Py_ssize_t))
        if not idx_c: raise MemoryError()
        cdef Py_ssize_t i, j, tmp
        cdef Py_ssize_t* shuffle_buf

        if replace:
            for i in range(sample_size):
                idx_c[i] = <Py_ssize_t>(_random_uniform() * total)
                if idx_c[i] >= total:
                    idx_c[i] = total - 1
        else:
            shuffle_buf = <Py_ssize_t*>aligned_malloc(total * sizeof(Py_ssize_t))
            if not shuffle_buf:
                os_aligned_free(idx_c)
                raise MemoryError()
            for i in range(total):
                shuffle_buf[i] = i
            for i in range(sample_size):
                j = i + <Py_ssize_t>(_random_uniform() * (total - i))
                if j >= total: j = total - 1
                tmp = shuffle_buf[i]
                shuffle_buf[i] = shuffle_buf[j]
                shuffle_buf[j] = tmp
                idx_c[i] = shuffle_buf[i]
            os_aligned_free(shuffle_buf)

        cdef FastDataFrame result = self._take(idx_c, sample_size)
        os_aligned_free(idx_c)
        return result

    def nsmallest(self, Py_ssize_t n, str col_name):
        cdef FastDataFrame sorted_df = self.sort_values(col_name, reverse=False)
        return sorted_df.head(n)

    def nlargest(self, Py_ssize_t n, str col_name):
        cdef FastDataFrame sorted_df = self.sort_values(col_name, reverse=True)
        return sorted_df.head(n)

    def pipe(self, object func, *args, **kwargs):
        return func(self, *args, **kwargs)

    def assign(self, **kwargs):
        cdef FastDataFrame result = self.copy()
        for name, value in kwargs.items():
            result[name] = value
        return result

    def corr(self):
        cdef list num_cols = []
        for name, col in self.columns.items():
            if col.dtype == TYPE_INT or col.dtype == TYPE_DOUBLE:
                num_cols.append(name)

        cdef Py_ssize_t n_cols = len(num_cols)
        cdef Py_ssize_t n_rows = self.df_length
        if n_cols == 0 or n_rows == 0:
            return FastDataFrame()

        cdef double** data = <double**>aligned_malloc(max(n_cols, 1) * sizeof(double*))
        if not data: raise MemoryError()
        cdef Py_ssize_t i, j
        for i in range(n_cols): data[i] = NULL

        cdef DoubleColumn c_i, c_j
        cdef double val
        cdef FastDataFrame result

        try:
            for i in range(n_cols):
                data[i] = <double*>aligned_malloc(max(n_rows, 1) * sizeof(double))
                if not data[i]: raise MemoryError()
                self._col_to_double_buffer(self.columns[num_cols[i]], data[i], n_rows)

            result = FastDataFrame()
            result.df_length = n_cols
            result.index = RangeIndex(0, n_cols)
            for i in range(n_cols):
                c_i = DoubleColumn(max(n_cols, 1))
                c_i.length = n_cols
                result.columns[num_cols[i]] = c_i

            for i in range(n_cols):
                c_i = <DoubleColumn>result.columns[num_cols[i]]
                c_i.data_c[i] = 1.0
                for j in range(i + 1, n_cols):
                    c_j = <DoubleColumn>result.columns[num_cols[j]]
                    val = _vec_pearson_corr(data[i], data[j], n_rows)
                    c_i.data_c[j] = val
                    c_j.data_c[i] = val
            return result
        finally:
            for i in range(n_cols):
                if data[i] != NULL: os_aligned_free(data[i])
            os_aligned_free(data)

    def cov(self):
        cdef list num_cols = []
        for name, col in self.columns.items():
            if col.dtype == TYPE_INT or col.dtype == TYPE_DOUBLE:
                num_cols.append(name)

        cdef Py_ssize_t n_cols = len(num_cols)
        cdef Py_ssize_t n_rows = self.df_length
        if n_cols == 0 or n_rows == 0:
            return FastDataFrame()

        cdef double** data = <double**>aligned_malloc(max(n_cols, 1) * sizeof(double*))
        if not data: raise MemoryError()
        cdef Py_ssize_t i, j
        for i in range(n_cols): data[i] = NULL

        cdef DoubleColumn c_i, c_j
        cdef double val
        cdef FastDataFrame result

        try:
            for i in range(n_cols):
                data[i] = <double*>aligned_malloc(max(n_rows, 1) * sizeof(double))
                if not data[i]: raise MemoryError()
                self._col_to_double_buffer(self.columns[num_cols[i]], data[i], n_rows)

            result = FastDataFrame()
            result.df_length = n_cols
            result.index = RangeIndex(0, n_cols)
            for i in range(n_cols):
                c_i = DoubleColumn(max(n_cols, 1))
                c_i.length = n_cols
                result.columns[num_cols[i]] = c_i

            for i in range(n_cols):
                c_i = <DoubleColumn>result.columns[num_cols[i]]
                c_i.data_c[i] = _vec_pearson_cov(data[i], data[i], n_rows)
                for j in range(i + 1, n_cols):
                    c_j = <DoubleColumn>result.columns[num_cols[j]]
                    val = _vec_pearson_cov(data[i], data[j], n_rows)
                    c_i.data_c[j] = val
                    c_j.data_c[i] = val
            return result
        finally:
            for i in range(n_cols):
                if data[i] != NULL: os_aligned_free(data[i])
            os_aligned_free(data)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def corrwith(self, other):
        cdef dict result = {}
        cdef str name
        cdef BaseColumn col
        cdef Py_ssize_t n = self.df_length
        cdef DoubleColumn temp, other_buf
        if n == 0: return result

        other_buf = DoubleColumn(max(n, 1))
        other_buf.length = n
        self._col_to_double_buffer(other, other_buf.data_c, n)

        for name, col in self.columns.items():
            if col.dtype == TYPE_INT or col.dtype == TYPE_DOUBLE:
                temp = DoubleColumn(max(n, 1))
                temp.length = n
                self._col_to_double_buffer(col, temp.data_c, n)
                result[name] = _vec_pearson_corr(temp.data_c, other_buf.data_c, n)
        return result

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def corrcoef(self, col_a, col_b):
        cdef Py_ssize_t n = min(col_a.length, col_b.length)
        if n < 2: return NAN
        cdef DoubleColumn buf_a = DoubleColumn(max(n, 1))
        cdef DoubleColumn buf_b = DoubleColumn(max(n, 1))
        buf_a.length = n
        buf_b.length = n
        self._col_to_double_buffer(col_a, buf_a.data_c, n)
        self._col_to_double_buffer(col_b, buf_b.data_c, n)
        return _vec_pearson_corr(buf_a.data_c, buf_b.data_c, n)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cov_withcols(self, col_a, col_b):
        cdef Py_ssize_t n = min(col_a.length, col_b.length)
        if n < 2: return NAN
        cdef DoubleColumn buf_a = DoubleColumn(max(n, 1))
        cdef DoubleColumn buf_b = DoubleColumn(max(n, 1))
        buf_a.length = n
        buf_b.length = n
        self._col_to_double_buffer(col_a, buf_a.data_c, n)
        self._col_to_double_buffer(col_b, buf_b.data_c, n)
        return _vec_pearson_cov(buf_a.data_c, buf_b.data_c, n)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pearson_sim(self):
        cdef list num_names = []
        cdef list num_cols = []
        cdef str name
        cdef BaseColumn col
        cdef Py_ssize_t n_cols, n_rows, i, j
        cdef double* buf_i = NULL
        cdef double* buf_j = NULL
        cdef DoubleColumn result_col
        cdef FastDataFrame result

        for name, col in self.columns.items():
            if col.dtype == TYPE_INT or col.dtype == TYPE_DOUBLE:
                num_names.append(name)
                num_cols.append(col)

        n_cols = len(num_names)
        if n_cols < 2:
            raise ValueError("Need at least 2 numeric columns")
        n_rows = self.df_length

        result = FastDataFrame()
        result.df_length = n_cols
        result.index = RangeIndex(0, n_cols)

        buf_i = <double*>aligned_malloc(max(n_rows, 1) * sizeof(double))
        buf_j = <double*>aligned_malloc(max(n_rows, 1) * sizeof(double))
        if not buf_i or not buf_j:
            if buf_i: os_aligned_free(buf_i)
            if buf_j: os_aligned_free(buf_j)
            raise MemoryError()

        try:
            for i in range(n_cols):
                result_col = DoubleColumn(max(n_cols, 1))
                result_col.length = n_cols
                self._col_to_double_buffer(<BaseColumn>num_cols[i], buf_i, n_rows)
                for j in range(n_cols):
                    if i == j:
                        result_col.data_c[j] = 1.0
                    else:
                        self._col_to_double_buffer(<BaseColumn>num_cols[j], buf_j, n_rows)
                        result_col.data_c[j] = _vec_pearson_corr(buf_i, buf_j, n_rows)
                result.columns[num_names[i]] = result_col
            return result
        finally:
            os_aligned_free(buf_i)
            os_aligned_free(buf_j)

    def rolling(self, Py_ssize_t window, str col_name):
        if window <= 0:
            raise ValueError(f"Window size must be positive, got {window}")
        if col_name not in self.columns:
            raise KeyError(f"Column '{col_name}' not found")
        cdef BaseColumn col = self.columns[col_name]
        cdef int col_dtype = col.dtype
        if col_dtype != TYPE_INT and col_dtype != TYPE_DOUBLE:
            raise TypeError(f"Rolling window requires numeric column, got '{col_name}'")
        return _RollingWrapper(col, window)

    def value_counts(self, str col_name):
        if col_name not in self.columns:
            raise KeyError(f"Column '{col_name}' not found")
        cdef BaseColumn col = self.columns[col_name]
        cdef dict counts = {}
        cdef Py_ssize_t i
        for i in range(self.df_length):
            val = col[i]
            counts[val] = counts.get(val, 0) + 1
        sorted_items = sorted(counts.items(), key=lambda x: -x[1])
        result = FastDataFrame()
        result.df_length = len(sorted_items)
        result.index = RangeIndex(0, len(sorted_items))
        key_col = self._column_from_list(col_name, [k for k, v in sorted_items])
        cnt_col = IntColumn(len(sorted_items))
        cnt_col.length = len(sorted_items)
        for i in range(len(sorted_items)):
            cnt_col.data_c[i] = sorted_items[i][1]
        result.columns[col_name] = key_col
        result.columns['count'] = cnt_col
        return result

    def nunique(self, str col_name):
        if col_name not in self.columns:
            raise KeyError(f"Column '{col_name}' not found")
        cdef set unique_vals = set()
        cdef Py_ssize_t i
        cdef BaseColumn col = self.columns[col_name]
        for i in range(self.df_length):
            unique_vals.add(col[i])
        return len(unique_vals)

    def clip_column(self, str col_name, double lower, double upper, bint inplace=False):
        cdef BaseColumn col = self.columns[col_name]
        if col.dtype == TYPE_INT:
            new_col = (<IntColumn>col).clip(lower, upper, inplace)
            if not inplace:
                self.columns[col_name] = new_col
        elif col.dtype == TYPE_DOUBLE:
            new_col = (<DoubleColumn>col).clip(lower, upper, inplace)
            if not inplace:
                self.columns[col_name] = new_col
        else:
            raise TypeError(f"clip not supported for column type {self._type_name(col)}")
        if inplace:
            return self

    def where_column(self, str col_name, str cond_col, object other):
        cdef BaseColumn src = self.columns[col_name]
        cdef BaseColumn cond = self.columns[cond_col]
        if src.dtype == TYPE_DOUBLE and cond.dtype == TYPE_INT:
            return (<DoubleColumn>src).where(<IntColumn>cond, <double>other)
        elif src.dtype == TYPE_INT and cond.dtype == TYPE_INT:
            return (<IntColumn>src).where(<IntColumn>cond, <int>other)
        raise TypeError(f"where not supported for column type {self._type_name(src)}")

    def mask_column(self, str col_name, str cond_col, object other):
        cdef BaseColumn src = self.columns[col_name]
        cdef BaseColumn cond = self.columns[cond_col]
        if src.dtype == TYPE_DOUBLE and cond.dtype == TYPE_INT:
            return (<DoubleColumn>src).mask(<IntColumn>cond, <double>other)
        elif src.dtype == TYPE_INT and cond.dtype == TYPE_INT:
            return (<IntColumn>src).mask(<IntColumn>cond, <int>other)
        raise TypeError(f"mask not supported for column type {self._type_name(src)}")

    def where_row(self, IntColumn mask, object fill=0):
        cdef Py_ssize_t n = self.df_length
        if mask.length != n:
            raise ValueError(f"Mask length {mask.length} != DataFrame length {n}")
        cdef FastDataFrame result = FastDataFrame()
        result.df_length = n
        result.index = self.index
        cdef str name
        cdef BaseColumn col
        for name, col in self.columns.items():
            result.columns[name] = col.where_mask(mask, fill)
        return result

    def _resolve_cols(self, tuple col_names):
        cdef list cols = []
        cdef str name
        for name in col_names:
            if name not in self.columns:
                raise KeyError(f"Column '{name}' not found")
            cols.append(self.columns[name])
        return cols

    def add(self, tuple col_names):
        if len(col_names) < 2: raise ValueError("Need at least 2 column names")
        cdef list cols = self._resolve_cols(col_names)
        cdef BaseColumn result = cols[0].add(cols[1])
        cdef Py_ssize_t j
        for j in range(2, len(cols)):
            result = result.add(cols[j])
        return result

    def sub(self, tuple col_names):
        if len(col_names) < 2: raise ValueError("Need at least 2 column names")
        cdef list cols = self._resolve_cols(col_names)
        cdef BaseColumn result = cols[0].sub(cols[1])
        cdef Py_ssize_t j
        for j in range(2, len(cols)):
            result = result.sub(cols[j])
        return result

    def mul(self, tuple col_names):
        if len(col_names) < 2: raise ValueError("Need at least 2 column names")
        cdef list cols = self._resolve_cols(col_names)
        cdef BaseColumn result = cols[0].mul(cols[1])
        cdef Py_ssize_t j
        for j in range(2, len(cols)):
            result = result.mul(cols[j])
        return result

    def div(self, tuple col_names):
        if len(col_names) < 2: raise ValueError("Need at least 2 column names")
        cdef list cols = self._resolve_cols(col_names)
        cdef BaseColumn result = cols[0].div(cols[1])
        cdef Py_ssize_t j
        for j in range(2, len(cols)):
            result = result.div(cols[j])
        return result

    def info(self):
        cdef Py_ssize_t total_cols = len(self.columns)
        cdef Py_ssize_t mem = self.memory_usage()
        cdef char bytes_buf[16]
        cdef list lines = []
        cdef Py_ssize_t idx = 0
        cdef Py_ssize_t col_bytes
        cdef str type_name, name
        cdef object col

        lines.append("<class 'FastDataFrame'>")
        lines.append(f"RangeIndex: {self.df_length} entries")
        lines.append(f"Data columns (total {total_cols} columns):")
        lines.append(" #   Column    Non-Null  Dtype     Mem Usage")
        lines.append("---  ------    --------  ----     ---------")

        for name, col in self.columns.items():
            type_name = self._type_name(col)
            col_bytes = col.get_size()
            _c_fmt_bytes(bytes_buf, sizeof(bytes_buf), col_bytes)
            lines.append(f" {idx}   {name:<10}{col.length:<10}{type_name:<10}{bytes_buf.decode('ascii')}")
            idx += 1

        _c_fmt_bytes(bytes_buf, sizeof(bytes_buf), mem)
        lines.append(f"dtypes: {self._dtype_summary()}")
        lines.append(f"memory usage: {bytes_buf.decode('ascii')}")
        return '\n'.join(lines)

    cdef str _dtype_summary(self):
        cdef dict counts = {}
        cdef str name, tname
        for name, col in self.columns.items():
            tname = self._type_name(col)
            counts[tname] = counts.get(tname, 0) + 1
        return ', '.join(f'{v} {k}' for k, v in counts.items())

    def memory_usage(self):
        cdef Py_ssize_t total = 0
        cdef BaseColumn col
        for col in self.columns.values():
            total += col.get_size()
        return total

    def apply(self, object func, str axis='columns'):
        if axis == 'columns':
            result = FastDataFrame()
            result.df_length = self.df_length
            result.index = RangeIndex(0, self.df_length)
            for col_name, col_obj in self.columns.items():
                new_vals = []
                for i in range(self.df_length):
                    new_vals.append(func(col_obj[i]))
                result.columns[col_name] = self._column_from_list(col_name, new_vals)
            return result
        elif axis == 'index':
            result_vals = []
            for i in range(self.df_length):
                row = {}
                for col_name, col_obj in self.columns.items():
                    row[col_name] = col_obj[i]
                result_vals.append(func(row))
            return result_vals
        raise ValueError(f"axis must be 'columns' or 'index', got '{axis}'")

    def select_dtypes(self, list include_types=None, list exclude_types=None):
        cdef FastDataFrame result = FastDataFrame()
        cdef dict dtype_map = {TYPE_INT: 'int', TYPE_DOUBLE: 'float', TYPE_CHAR256: 'char256',
                               TYPE_UCS2: 'ucs2', TYPE_DATE: 'date',
                               TYPE_BOOL: 'bool', TYPE_CATEGORY: 'category',
                               TYPE_DATETIME: 'datetime', TYPE_TIME: 'time'}
        for name, col in self.columns.items():
            tn = dtype_map.get(col.dtype, 'unknown')
            if include_types is not None and tn not in include_types:
                continue
            if exclude_types is not None and tn in exclude_types:
                continue
            result.columns[name] = col
        result.df_length = self.df_length
        result.index = self.index
        return result

    def astype(self, dict dtype_dict):
        cdef FastDataFrame result = FastDataFrame()
        cdef str col_name, target
        for col_name, col_obj in self.columns.items():
            if col_name in dtype_dict:
                target = dtype_dict[col_name]
                result.columns[col_name] = col_obj.astype(target)
            else:
                result.columns[col_name] = col_obj.copy() if hasattr(col_obj, 'copy') else col_obj
        result.df_length = self.df_length
        if self.index is not None:
            if self.index.index_type == INDEX_RANGE:
                result.index = RangeIndex((<RangeIndex>self.index).start,
                                         self.index.length,
                                         (<RangeIndex>self.index).step)
            elif self.index.index_type == INDEX_INT:
                result.index = self.index.copy()
            else:
                result.index = self.index.copy()
        return result

    cdef void _shift_right(self, BaseColumn col, Py_ssize_t pos):
        cdef Py_ssize_t col_len
        cdef UCS2Column_SoA ucs2_col
        cdef IntColumn ic
        cdef DoubleColumn dc
        cdef Char256Column_SoA cc
        cdef DateColumn tc
        cdef BoolColumn bool_col
        cdef CategoryColumn cat_col
        cdef DateTimeColumn dtc_col
        cdef TimeColumn tmc_col

        if col.dtype == TYPE_UCS2:
            ucs2_col = <UCS2Column_SoA>col
            col_len = ucs2_col.length
            if col_len >= ucs2_col.capacity: ucs2_col._reserve(ucs2_col.capacity * 2)
            if pos < col_len:
                memmove(&ucs2_col.offsets_c[pos + 1],
                        &ucs2_col.offsets_c[pos],
                        (col_len - pos) * sizeof(Py_ssize_t))
            ucs2_col.length = col_len + 1
        elif col.dtype == TYPE_INT:
            ic = <IntColumn>col
            col_len = ic.length
            if col_len >= ic.capacity: ic._reserve(ic.capacity * 2)
            _c_shift_right(ic.data_c, pos, col_len, sizeof(int))
            ic.length = col_len + 1
        elif col.dtype == TYPE_DOUBLE:
            dc = <DoubleColumn>col
            col_len = dc.length
            if col_len >= dc.capacity: dc._reserve(dc.capacity * 2)
            _c_shift_right(dc.data_c, pos, col_len, sizeof(double))
            dc.length = col_len + 1
        elif col.dtype == TYPE_CHAR256:
            cc = <Char256Column_SoA>col
            col_len = cc.length
            if col_len >= cc.capacity: cc._reserve(cc.capacity * 2)
            _c_shift_right(cc.data_c, pos, col_len, 256)
            cc.length = col_len + 1
        elif col.dtype == TYPE_DATE:
            tc = <DateColumn>col
            col_len = tc.length
            if col_len >= tc.capacity: tc._reserve(tc.capacity * 2)
            _c_shift_right(tc.data_c, pos, col_len, sizeof(int32_t))
            tc.length = col_len + 1
        elif col.dtype == TYPE_BOOL:
            bool_col = <BoolColumn>col
            col_len = bool_col.length
            if col_len >= bool_col.capacity: bool_col._reserve(bool_col.capacity * 2)
            _c_shift_right(bool_col.data_c, pos, col_len, sizeof(unsigned char))
            bool_col.length = col_len + 1
        elif col.dtype == TYPE_CATEGORY:
            cat_col = <CategoryColumn>col
            col_len = cat_col.length
            if col_len >= cat_col.capacity: cat_col._reserve(cat_col.capacity * 2)
            _c_shift_right(cat_col.codes_c, pos, col_len, sizeof(int))
            cat_col.length = col_len + 1
        elif col.dtype == TYPE_DATETIME:
            dtc_col = <DateTimeColumn>col
            col_len = dtc_col.length
            if col_len >= dtc_col.capacity: dtc_col._reserve(dtc_col.capacity * 2)
            _c_shift_right(dtc_col.data_c, pos, col_len, sizeof(long long))
            dtc_col.length = col_len + 1
        elif col.dtype == TYPE_TIME:
            tmc_col = <TimeColumn>col
            col_len = tmc_col.length
            if col_len >= tmc_col.capacity: tmc_col._reserve(tmc_col.capacity * 2)
            _c_shift_right(tmc_col.data_c, pos, col_len, sizeof(int))
            tmc_col.length = col_len + 1

    cdef void _shift_left(self, BaseColumn col, Py_ssize_t pos):
        col.shift_left(pos)

    cdef void _col_to_double_buffer(self, BaseColumn col, double* out, Py_ssize_t n):
        if col.dtype == TYPE_INT:
            (<IntColumn>col).to_double_buffer(out, n)
        elif col.dtype == TYPE_DOUBLE:
            (<DoubleColumn>col).to_double_buffer(out, n)
        else:
            raise TypeError(f"Cannot convert column of type {_fast_type_name(col.dtype)} to double buffer")