# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False

cimport cython
from libc.string cimport memcpy, memmove, strncmp, memset, strncpy
from libc.stdio cimport snprintf
from libc.stdint cimport uint32_t, int32_t, int64_t, uint64_t
from libc.time cimport time_t, time
from libc.math cimport NAN, sqrt, log, cos, isnan
from libc.stdlib cimport abs as c_abs
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
from fluidframe.core.asciicolumn cimport AsciiColumn
from fluidframe.core.ucs2column cimport UCS2Column
from fluidframe.core.datecolumn cimport DateColumn, _days_to_ymd, _ymd_to_days
from fluidframe.core.boolcolumn cimport BoolColumn
from fluidframe.core.categorycolumn cimport CategoryColumn
from fluidframe.core.datetimecolumn cimport DateTimeColumn
from fluidframe.core.timecolumn cimport TimeColumn
from fluidframe.core.index cimport IndexBase, RangeIndex, IntIndex, MultiIndex, INDEX_RANGE, INDEX_INT, INDEX_MULTI
from fluidframe.core.indexers import _ILocIndexer, _LocIndexer
from fluidframe._schema_types import Char256, Date, Bool, Category, DateTime, Time, _parse_schema


class _RollingWrapper:
    """Lightweight rolling window wrapper, delegates to column roll_xxx methods"""

    _SUPPORTED_TYPES = {TYPE_INT, TYPE_DOUBLE}
    _TYPE_LABELS = {0: 'int', 2: 'float', 3: 'char256', 4: 'ucs2', 5: 'date', 6: 'bool', 7: 'category', 8: 'datetime', 9: 'time'}

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __init__(self, col, window):
        if col.dtype not in self._SUPPORTED_TYPES:
            type_name = self._TYPE_LABELS.get(col.dtype, 'unknown')
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


cdef inline int _cstrncmp(const char* a, Py_ssize_t a_len, const char* b, Py_ssize_t b_len) noexcept nogil:
    cdef Py_ssize_t min_len = a_len if a_len < b_len else b_len
    cdef int r = strncmp(a, b, min_len)
    if r != 0:
        return r
    if a_len < b_len:
        return -1
    elif a_len > b_len:
        return 1
    return 0

cdef void _c_argsort_char256(char* data, Py_ssize_t* offsets, Py_ssize_t* indices, Py_ssize_t left, Py_ssize_t right) noexcept nogil:
    if left >= right: return

    cdef Py_ssize_t stack[64]
    cdef Py_ssize_t top = -1
    cdef Py_ssize_t l, r, i, j, temp
    cdef Py_ssize_t pivot_idx, cur_len
    cdef char* pivot_ptr
    cdef Py_ssize_t pivot_len

    top += 1; stack[top] = left
    top += 1; stack[top] = right

    while top >= 0:
        r = stack[top]; top -= 1
        l = stack[top]; top -= 1
        if l >= r:
            continue

        pivot_idx = indices[(l + r) // 2]
        pivot_ptr = data + offsets[pivot_idx]
        pivot_len = offsets[pivot_idx + 1] - offsets[pivot_idx] - 1
        i = l; j = r

        while i <= j:
            while i <= j:
                cur_len = offsets[indices[i] + 1] - offsets[indices[i]] - 1
                if _cstrncmp(data + offsets[indices[i]], cur_len, pivot_ptr, pivot_len) >= 0:
                    break
                i += 1
            while i <= j:
                cur_len = offsets[indices[j] + 1] - offsets[indices[j]] - 1
                if _cstrncmp(data + offsets[indices[j]], cur_len, pivot_ptr, pivot_len) <= 0:
                    break
                j -= 1
            if i <= j:
                temp = indices[i]; indices[i] = indices[j]; indices[j] = temp
                i += 1; j -= 1

        if l < j:
            if j - l < r - i:
                top += 1; stack[top] = l
                top += 1; stack[top] = j
                if i < right:
                    top += 1; stack[top] = i
                    top += 1; stack[top] = r
            else:
                if i < r:
                    top += 1; stack[top] = i
                    top += 1; stack[top] = r
                if l < j:
                    top += 1; stack[top] = l
                    top += 1; stack[top] = j
        elif i < r:
            top += 1; stack[top] = i
            top += 1; stack[top] = r


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _c_argsort_numeric(NumericType* data, Py_ssize_t* indices, Py_ssize_t left, Py_ssize_t right) noexcept nogil:
    if left >= right: return

    cdef Py_ssize_t stack[64]
    cdef Py_ssize_t top = -1
    cdef Py_ssize_t l, r, i, j, temp
    cdef Py_ssize_t mid_idx
    cdef NumericType pivot

    top += 1; stack[top] = left
    top += 1; stack[top] = right

    while top >= 0:
        r = stack[top]; top -= 1
        l = stack[top]; top -= 1
        if l >= r:
            continue

        mid_idx = l + ((r - l) >> 1)
        if data[indices[l]] > data[indices[mid_idx]]:
            temp = indices[l]; indices[l] = indices[mid_idx]; indices[mid_idx] = temp
        if data[indices[l]] > data[indices[r]]:
            temp = indices[l]; indices[l] = indices[r]; indices[r] = temp
        if data[indices[mid_idx]] > data[indices[r]]:
            temp = indices[mid_idx]; indices[mid_idx] = indices[r]; indices[r] = temp

        pivot = data[indices[mid_idx]]
        i = l; j = r

        while i <= j:
            while data[indices[i]] < pivot: i += 1
            while data[indices[j]] > pivot: j -= 1
            if i <= j:
                temp = indices[i]; indices[i] = indices[j]; indices[j] = temp
                i += 1; j -= 1

        if l < j:
            if j - l < r - i:
                top += 1; stack[top] = l
                top += 1; stack[top] = j
                if i < right:
                    top += 1; stack[top] = i
                    top += 1; stack[top] = r
            else:
                if i < r:
                    top += 1; stack[top] = i
                    top += 1; stack[top] = r
                if l < j:
                    top += 1; stack[top] = l
                    top += 1; stack[top] = j
        elif i < r:
            top += 1; stack[top] = i
            top += 1; stack[top] = r


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

    cdef Py_ssize_t stack[64]
    cdef Py_ssize_t top = -1
    cdef Py_ssize_t l, r, i, j, temp
    cdef Py_ssize_t pivot_idx

    top += 1; stack[top] = left
    top += 1; stack[top] = right

    while top >= 0:
        r = stack[top]; top -= 1
        l = stack[top]; top -= 1
        if l >= r:
            continue

        pivot_idx = indices[(l + r) // 2]
        i = l; j = r

        while i <= j:
            while _ucs2_cmp_at(chars, offsets, indices[i], pivot_idx) < 0: i += 1
            while _ucs2_cmp_at(chars, offsets, indices[j], pivot_idx) > 0: j -= 1
            if i <= j:
                temp = indices[i]; indices[i] = indices[j]; indices[j] = temp
                i += 1; j -= 1

        if l < j:
            if j - l < r - i:
                top += 1; stack[top] = l
                top += 1; stack[top] = j
                if i < right:
                    top += 1; stack[top] = i
                    top += 1; stack[top] = r
            else:
                if i < r:
                    top += 1; stack[top] = i
                    top += 1; stack[top] = r
                if l < j:
                    top += 1; stack[top] = l
                    top += 1; stack[top] = j
        elif i < r:
            top += 1; stack[top] = i
            top += 1; stack[top] = r


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

cdef inline void _fast_insert_val(int dtype, void* data_ptr, void* col_obj,
                                  uint32_t row_idx, object val) except *:
    cdef int ival
    cdef int32_t days
    cdef int64_t tmp_dt

    if dtype == TYPE_INT:
        (<int*>data_ptr)[row_idx] = <int>val
    elif dtype == TYPE_DOUBLE:
        (<double*>data_ptr)[row_idx] = <double>val
    elif dtype == TYPE_DATE:
        if isinstance(val, int) and val > 10000000:
            ival = <int>val
            days = _ymd_to_days(ival // 10000, (ival // 100) % 100, ival % 100)
            (<int32_t*>data_ptr)[row_idx] = days
        else:
            (<int32_t*>data_ptr)[row_idx] = <int32_t>val
    elif dtype == TYPE_DATETIME:
        if isinstance(val, int) and val > 10000000:
            ival = <int>val
            days = _ymd_to_days(ival // 10000, (ival // 100) % 100, ival % 100)
            tmp_dt = <int64_t>days
            (<int64_t*>data_ptr)[row_idx] = tmp_dt * 86400000000LL
        else:
            tmp_dt = <int64_t>val
            (<int64_t*>data_ptr)[row_idx] = tmp_dt * 1000000LL
    elif dtype == TYPE_TIME:
        (<int32_t*>data_ptr)[row_idx] = <int32_t>val
    elif dtype == TYPE_CHAR256:
        (<AsciiColumn>col_obj).append_nolen(val)
    elif dtype == TYPE_UCS2:
        (<UCS2Column>col_obj).append_nolen(val)
    elif dtype == TYPE_BOOL:
        (<BoolColumn>col_obj).append_nolen(val)
    elif dtype == TYPE_CATEGORY:
        (<CategoryColumn>col_obj).append_nolen(val)


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
            return (<AsciiColumn>col).take_by_indices(indices, n, inplace)
        elif col.dtype == TYPE_UCS2:
            return (<UCS2Column>col).take_by_indices(indices, n, inplace)
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

    def __cinit__(self, *args, **kwargs):
        self._c_cols = NULL
        self._fixed_capacity = 0
        self.df_length = 0
        self._num_cols = 0
        self._is_sealed = False

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __init__(self, dict schema=None):
        self.columns = {}
        self.df_length = 0
        self.index = RangeIndex(0, 0)
        self._metric_cache = None
        self._groupby_cache = None
        self._linalg_cache = None

        self._2d_blocks = []          # 持有巨型2D数据块的强引用，防止被 GC
        self._ordered_cols = []
        self._schema_keys = ()

        if schema:
            parsed = _parse_schema(schema)
            for col_name, dtype in parsed.items():
                if dtype == TYPE_INT:
                    self.columns[col_name] = IntColumn()
                elif dtype == TYPE_DOUBLE:
                    self.columns[col_name] = DoubleColumn()
                elif dtype == TYPE_CHAR256:
                    self.columns[col_name] = AsciiColumn()
                elif dtype == TYPE_UCS2:
                    self.columns[col_name] = UCS2Column()
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
            self._schema = parsed
            self._schema_keys = tuple(parsed.keys())

    def __dealloc__(self):
        if self._c_cols != NULL:
            free(self._c_cols)


    def __len__(self):
        return self.df_length

    def __eq__(self, other):
        if not isinstance(other, FastDataFrame):
            return NotImplemented
        cdef FastDataFrame rhs = <FastDataFrame>other
        if self.df_length != rhs.df_length:
            return False
        if set(self.columns.keys()) != set(rhs.columns.keys()):
            return False
        cdef Py_ssize_t i, n = self.df_length
        cdef str name
        cdef BaseColumn col_l, col_r
        for name in self.columns:
            col_l = self.columns[name]
            col_r = rhs.columns[name]
            if col_l.dtype != col_r.dtype:
                return False
            for i in range(n):
                if _col_row_key(col_l, i) != _col_row_key(col_r, i):
                    return False
        return True

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def reserve_with_capacity(self, uint32_t capacity):
        if self._fixed_capacity > 0:
            raise RuntimeError("Capacity has already been reserved and cannot be expanded.")
        if self.df_length > 0:
            raise RuntimeError("Cannot reserve capacity on a non-empty DataFrame.")
        if capacity == 0:
            raise ValueError("Capacity must be strictly positive.")

        cdef dict type_groups = {
            TYPE_INT: [], TYPE_DOUBLE: [], TYPE_DATE: [],
            TYPE_DATETIME: [], TYPE_TIME: []
        }
        cdef dict other_cols = {}

        # 1. 提取同构类型以备二维内存合并
        for col_name, dtype in self._schema.items():
            if dtype in type_groups:
                type_groups[dtype].append(col_name)
            else:
                other_cols[col_name] = dtype

        self.columns = {}
        self._fixed_capacity = capacity

        cdef uint32_t c, j
        cdef list names

        # 2. 生成多列二维连续缓存块（Column Vectorization Chunking）
        if type_groups[TYPE_INT]:
            names = type_groups[TYPE_INT]
            c = <uint32_t>len(names)
            blk_int = IntColumn(capacity * c, cols=c)
            blk_int.length = capacity * c
            self._2d_blocks.append(blk_int)
            for j, name in enumerate(names):
                self.columns[name] = IntColumn.create_view(blk_int.data_c + j * capacity, 0, blk_int)
            for j, name in enumerate(names):
                (<IntColumn>self.columns[name]).capacity = capacity

        if type_groups[TYPE_DOUBLE]:
            names = type_groups[TYPE_DOUBLE]
            c = <uint32_t>len(names)
            blk_dbl = DoubleColumn(capacity * c, cols=c)
            blk_dbl.length = capacity * c
            self._2d_blocks.append(blk_dbl)
            for j, name in enumerate(names):
                self.columns[name] = DoubleColumn.create_view(blk_dbl.data_c + j * capacity, 0, blk_dbl)
            for j, name in enumerate(names):
                (<DoubleColumn>self.columns[name]).capacity = capacity

        if type_groups[TYPE_DATE]:
            names = type_groups[TYPE_DATE]
            c = <uint32_t>len(names)
            blk_date = DateColumn(capacity * c, cols=c)
            blk_date.length = capacity * c
            self._2d_blocks.append(blk_date)
            for j, name in enumerate(names):
                self.columns[name] = DateColumn.create_view(<int32_t*>blk_date.data_c + j * capacity, 0, blk_date)
            for j, name in enumerate(names):
                (<DateColumn>self.columns[name]).capacity = capacity

        if type_groups[TYPE_DATETIME]:
            names = type_groups[TYPE_DATETIME]
            c = <uint32_t>len(names)
            blk_dt = DateTimeColumn(capacity * c, cols=c)
            blk_dt.length = capacity * c
            self._2d_blocks.append(blk_dt)
            for j, name in enumerate(names):
                self.columns[name] = DateTimeColumn.create_view(<int64_t*>blk_dt.data_c + j * capacity, 0, blk_dt)
            for j, name in enumerate(names):
                (<DateTimeColumn>self.columns[name]).capacity = capacity

        if type_groups[TYPE_TIME]:
            names = type_groups[TYPE_TIME]
            c = <uint32_t>len(names)
            blk_time = TimeColumn(capacity * c, cols=c)
            blk_time.length = capacity * c
            self._2d_blocks.append(blk_time)
            for j, name in enumerate(names):
                self.columns[name] = TimeColumn.create_view(<int32_t*>blk_time.data_c + j * capacity, 0, blk_time)
            for j, name in enumerate(names):
                (<TimeColumn>self.columns[name]).capacity = capacity

        # 3. 处理离散的非等宽字段列
        for name, dtype in other_cols.items():
            if dtype == TYPE_CHAR256:
                self.columns[name] = AsciiColumn(capacity)
            elif dtype == TYPE_UCS2:
                self.columns[name] = UCS2Column(capacity)
            elif dtype == TYPE_BOOL:
                self.columns[name] = BoolColumn(capacity)
            elif dtype == TYPE_CATEGORY:
                self.columns[name] = CategoryColumn(capacity)

        # ------------------------------------------------------------
        # 4. 构建底层的 C Pointer Mapping Arrays（核心硬件加速）
        # ------------------------------------------------------------
        self._ordered_cols = [self.columns[k] for k in self._schema_keys]
        self._num_cols = <uint32_t>len(self._schema_keys)

        self._c_cols = <ColMeta*>malloc(self._num_cols * sizeof(ColMeta))
        if not self._c_cols:
            raise MemoryError("Failed to allocate C array for column metadata.")

        cdef BaseColumn col
        cdef Py_ssize_t i

        for i, key in enumerate(self._schema_keys):
            col = self._ordered_cols[i]
            self._c_cols[i].dtype = col.dtype
            self._c_cols[i].col_obj = <void*>col
            self._c_cols[i].len_ptr = &(col.length)

            if col.dtype == TYPE_INT:
                self._c_cols[i].data_ptr = (<IntColumn>col).data_c
            elif col.dtype == TYPE_DOUBLE:
                self._c_cols[i].data_ptr = (<DoubleColumn>col).data_c
            elif col.dtype == TYPE_DATE:
                self._c_cols[i].data_ptr = (<DateColumn>col).data_c
            elif col.dtype == TYPE_DATETIME:
                self._c_cols[i].data_ptr = (<DateTimeColumn>col).data_c
            elif col.dtype == TYPE_TIME:
                self._c_cols[i].data_ptr = (<TimeColumn>col).data_c
            else:
                self._c_cols[i].data_ptr = NULL

        self._fixed_capacity = capacity
        self._is_sealed = True   # 锁定扩容状态

    @cython.binding(False)
    def add_row(self, *args, **kwargs):
        """
        极限推流行。因为使用了直接寻址指针 _c_cols[i].data_ptr，
        在执行这段代码时几乎不触发 Python API，直接对内存打数据。

        支持两种调用方式：
          df.add_row(1, 12, 53, 45, 67, 95.0, 'Alice', 'senior', 20230115, 1)
          df.add_row(id=1, x0=12, ..., flags=1)
        任一行写入中途失败都会被回滚，保证各列 length 始终与 df_length 一致。
        """
        if not self._is_sealed:
            raise RuntimeError("Must call reserve_with_capacity before pushing rows.")

        cdef uint32_t row_idx = self.df_length
        if row_idx >= self._fixed_capacity:
            raise BufferError(f"DataFrame maximum capacity ({self._fixed_capacity}) exceeded.")

        cdef uint32_t i
        cdef uint32_t n_cols = self._num_cols

        if kwargs:
            if args:
                raise TypeError("Cannot mix positional args and keyword args in add_row.")
            for i in range(n_cols):
                if self._schema_keys[i] not in kwargs:
                    raise ValueError(f"Missing value for column: {self._schema_keys[i]}")
            try:
                for i in range(n_cols):
                    _fast_insert_val(
                        self._c_cols[i].dtype,
                        self._c_cols[i].data_ptr,
                        self._c_cols[i].col_obj,
                        row_idx,
                        kwargs[self._schema_keys[i]]
                    )
            except Exception:
                self._rollback_pending_writes(n_cols)
                raise
        else:
            if len(args) != n_cols:
                raise ValueError(f"Expected {n_cols} elements, got {len(args)}.")
            try:
                for i in range(n_cols):
                    _fast_insert_val(
                        self._c_cols[i].dtype,
                        self._c_cols[i].data_ptr,
                        self._c_cols[i].col_obj,
                        row_idx,
                        args[i]
                    )
            except Exception:
                self._rollback_pending_writes(n_cols)
                raise

        self.df_length += 1
        for i in range(n_cols):
            self._c_cols[i].len_ptr[0] = self.df_length

    cdef void _rollback_pending_writes(self, uint32_t n_cols) noexcept:
        cdef uint32_t i
        for i in range(n_cols):
            if self._c_cols[i].dtype == TYPE_CHAR256:
                (<AsciiColumn>self._c_cols[i].col_obj).undo_nolen()
            elif self._c_cols[i].dtype == TYPE_UCS2:
                (<UCS2Column>self._c_cols[i].col_obj).undo_nolen()

    def add_rows_from_list(self, list rows):
        """
        如果你有一个 list of tuples (如示例中)，直接调用这个方法，能省去大量 Python 函数调用栈的开销。
        """
        cdef tuple r
        for r in rows:
            self.add_row(*r)

    cdef inline str _fast_type_name(self, int dt):
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
        cdef bint is2d = False
        cdef Py_ssize_t cols = 1
        cdef Py_ssize_t rows = col.length
        cdef Py_ssize_t j
        cdef list parts
        cdef uint64_t dbits

        if dt == TYPE_INT:
            is2d = (<IntColumn>col).get_is_2d_c()
            cols = (<IntColumn>col).get_cols_c()
            rows = (<IntColumn>col).get_rows_c()
        elif dt == TYPE_DOUBLE:
            is2d = (<DoubleColumn>col).get_is_2d_c()
            cols = (<DoubleColumn>col).get_cols_c()
            rows = (<DoubleColumn>col).get_rows_c()
        elif dt == TYPE_DATE:
            is2d = (<DateColumn>col).get_is_2d_c()
            cols = (<DateColumn>col).get_cols_c()
            rows = (<DateColumn>col).get_rows_c()
        elif dt == TYPE_DATETIME:
            is2d = (<DateTimeColumn>col).get_is_2d_c()
            cols = (<DateTimeColumn>col).get_cols_c()
            rows = (<DateTimeColumn>col).get_rows_c()
        elif dt == TYPE_TIME:
            is2d = (<TimeColumn>col).get_is_2d_c()
            cols = (<TimeColumn>col).get_cols_c()
            rows = (<TimeColumn>col).get_rows_c()

        # 2D 列：单元格 = 整行向量（各子列取值），与 _col_row_key 语义一致
        if is2d and cols > 1:
            parts = []
            for j in range(cols):
                if dt == TYPE_INT:
                    sz = snprintf(buf, sizeof(buf), "%d", (<IntColumn>col).data_c[j * rows + i])
                    parts.append(buf[:sz].decode('ascii'))
                elif dt == TYPE_DOUBLE:
                    dval = (<DoubleColumn>col).data_c[j * rows + i]
                    memcpy(&dbits, &dval, sizeof(double))
                    if (dbits & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL and (dbits & 0x000FFFFFFFFFFFFFULL) != 0:
                        parts.append("null")
                    else:
                        sz = snprintf(buf, sizeof(buf), "%.6g", dval)
                        parts.append(buf[:sz].decode('ascii'))
                elif dt == TYPE_DATE:
                    date_ptr = <int32_t*>(<DateColumn>col).data_c
                    _days_to_ymd(date_ptr[j * rows + i], &dy, &dm, &dd)
                    sz = snprintf(buf, sizeof(buf), "%04d-%02d-%02d", dy, dm, dd)
                    parts.append(buf[:sz].decode('ascii'))
                elif dt == TYPE_TIME:
                    tval = (<TimeColumn>col).data_c[j * rows + i]
                    h = <int>(tval / 3600)
                    mm = <int>((tval % 3600) / 60)
                    s = <int>(tval % 60)
                    sz = snprintf(buf, sizeof(buf), "%02d:%02d:%02d", h, mm, s)
                    parts.append(buf[:sz].decode('ascii'))
                else:
                    parts.append(str(col[j * rows + i]))
            return "[" + ", ".join(parts) + "]"

        if dt == TYPE_INT:
            sz = snprintf(buf, sizeof(buf), "%d", (<IntColumn>col).data_c[i])
            return buf[:sz].decode('ascii')

        elif dt == TYPE_DOUBLE:
            dval = (<DoubleColumn>col).data_c[i]
            memcpy(&dbits, &dval, sizeof(double))
            if (dbits & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL and (dbits & 0x000FFFFFFFFFFFFFULL) != 0:
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
    @cython.always_allow_keywords(False)
    def __setitem__(self, name, value):
        self.columns[name] = value

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

    cdef BaseColumn _column_from_list(self, str name, list values):
        return _column_from_list_impl(self, name, values)

    cdef double _get_col_value_as_double(self, BaseColumn col, Py_ssize_t i):
        if col.dtype == TYPE_INT:
            return <double>(<IntColumn>col).data_c[i]
        elif col.dtype == TYPE_DOUBLE:
            return (<DoubleColumn>col).data_c[i]
        return 0.0

    @cython.binding(False)
    def groupby(self, by_cols):
        if self._groupby_cache is not None:
            self._groupby_cache = None
        if isinstance(by_cols, str):
            by_cols = [by_cols]
        cdef list by_list = list(by_cols)
        for cn in by_list:
            if cn not in self.columns:
                raise KeyError(f"Column '{cn}' not found")
        cdef _DataFrameGroupByAccessor acc = _make_groupby_accessor(self, by_list)
        self._groupby_cache = acc
        return acc

    cdef void _shift_right(self, BaseColumn col, Py_ssize_t pos):
        if col.dtype == TYPE_INT:
            (<IntColumn>col).shift_right(pos)
            return
        elif col.dtype == TYPE_DOUBLE:
            (<DoubleColumn>col).shift_right(pos)
            return
        elif col.dtype == TYPE_DATE:
            (<DateColumn>col).shift_right(pos)
            return
        elif col.dtype == TYPE_DATETIME:
            (<DateTimeColumn>col).shift_right(pos)
            return
        elif col.dtype == TYPE_TIME:
            (<TimeColumn>col).shift_right(pos)
            return

        cdef Py_ssize_t col_len
        cdef UCS2Column ucs2_col
        cdef IntColumn ic
        cdef DoubleColumn dc
        cdef AsciiColumn cc
        cdef DateColumn tc
        cdef BoolColumn bool_col
        cdef CategoryColumn cat_col
        cdef DateTimeColumn dtc_col
        cdef TimeColumn tmc_col

        if col.dtype == TYPE_UCS2:
            ucs2_col = <UCS2Column>col
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
            cc = <AsciiColumn>col
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

    cdef void _col_to_double_buffer(self, BaseColumn col, double* out, Py_ssize_t n, Py_ssize_t sub=0) except *:
        # sub = 2D 列的子列下标（列优先布局 data_c[sub*rows + i]），1D 列恒为 0
        cdef Py_ssize_t r, i
        cdef int* int_ptr
        cdef double* double_ptr

        if col.dtype == TYPE_INT:
            int_ptr = (<IntColumn>col).data_c
            r = (<IntColumn>col).get_rows_c()
            if n > r: n = r

            with nogil:
                for i in range(n):
                    out[i] = <double>int_ptr[sub * r + i]

        elif col.dtype == TYPE_DOUBLE:
            double_ptr = (<DoubleColumn>col).data_c
            r = (<DoubleColumn>col).get_rows_c()
            if n > r: n = r
            memcpy(out, double_ptr + (sub * r), n * sizeof(double))
        else:
            raise TypeError(f"Cannot convert column of type {self._fast_type_name(col.dtype)} to double buffer")

    # ─── Thin delegate methods ───

    # CRUD
    def eval(self, str expr, bint inplace=False):
        return _eval_impl(self, expr, inplace)

    def query(self, str expr):
        return _query_impl(self, expr)

    def append_row(self, **kwargs):
        return _append_row_impl(self, kwargs)

    def append(self, row, bint ignore_index=False):
        return _append_impl(self, row, ignore_index)

    def copy(self):
        return _copy_impl(self)

    def pop_front(self):
        return _pop_front_impl(self)

    def pop_back(self):
        return _pop_back_impl(self)

    def rename(self, dict mapper):
        return _rename_impl(self, mapper)

    def reset_index(self):
        return _reset_index_impl(self)

    def reindex(self, list new_index):
        return _reindex_impl(self, new_index)

    def insert_row(self, Py_ssize_t pos, **kwargs):
        return _insert_row_impl(self, pos, kwargs)

    def drop(self, col_names):
        if isinstance(col_names, str):
            col_names = [col_names]
        cdef str name
        for name in col_names:
            if name not in self.columns:
                raise KeyError(f"Column '{name}' not found")
        for name in col_names:
            del self.columns[name]

    def drop_with_index(self, list drop_indices, bint inplace=False):
        return _drop_with_index_impl(self, drop_indices, inplace)

    def drop_index(self, object labels, bint inplace=False):
        return _drop_index_impl(self, labels, inplace)

    def drop_raw(self, Py_ssize_t row_idx):
        return _drop_raw_impl(self, row_idx)

    def insert_column(self, str col_name, object col_data, Py_ssize_t pos=-1):
        return _insert_column_impl(self, col_name, col_data, pos)

    def duplicate_column(self, str src_name, str new_name=None):
        _duplicate_column_impl(self, src_name, new_name)

    def duplicate_rows(self, list indices=None):
        return _duplicate_rows_impl(self, indices)

    # Aggregation
    def mean(self, col_names=None):
        return _numeric_agg_impl(self, 'mean', col_names)

    def sum(self, col_names=None):
        return _numeric_agg_impl(self, 'sum', col_names)

    def min(self, col_names=None):
        return _numeric_agg_impl(self, 'min', col_names)

    def max(self, col_names=None):
        return _numeric_agg_impl(self, 'max', col_names)

    def std(self, col_names=None):
        return _numeric_agg_impl(self, 'std', col_names)

    def var(self, col_names=None):
        return _numeric_agg_impl(self, 'var', col_names)

    def median(self, col_names=None):
        return _numeric_agg_impl(self, 'median', col_names)

    def quantile(self, q=0.5, col_names=None):
        return _numeric_agg_arg_impl(self, 'quantile', col_names, q)

    def mode(self, col_names=None):
        return _mode_impl(self, col_names)

    def count(self, col_names=None):
        return _count_impl(self, col_names)

    def describe(self):
        return _describe_impl(self)

    def value_counts(self, str col_name):
        return _value_counts_impl(self, col_name)

    def nunique(self, str col_name=None):
        return _nunique_impl(self, col_name)

    # Cumulative aggregations
    def cumsum(self):
        return _numeric_agg_impl(self, 'cumsum')

    def cumprod(self):
        return _numeric_agg_impl(self, 'cumprod')

    def cummax(self):
        return _numeric_agg_impl(self, 'cummax')

    def cummin(self):
        return _numeric_agg_impl(self, 'cummin')

    def skew(self):
        return _numeric_agg_impl(self, 'skew')

    def kurt(self):
        return _numeric_agg_impl(self, 'kurt')

    # Column-level delegates
    def argmax(self, str col_name):
        return self.columns[col_name].idxmax()

    def argmin(self, str col_name):
        return self.columns[col_name].idxmin()

    def unique(self, str col_name):
        return self.columns[col_name].unique()

    # I/O
    def __str__(self):
        return _to_string_impl(self)

    def __repr__(self):
        return _to_string_impl(self)

    def to_string(self):
        return _to_string_impl(self)

    def print(self, max_rows=80):
        print(self.to_string())

    def to_pickle(self, str path):
        return _to_pickle_impl(self, path)

    def to_csv(self, str path, str sep=','):
        return _to_csv_impl(self, path, sep)

    def to_tsv(self, filename):
        return self.to_csv(filename, sep='\t')

    def to_dict(self):
        return _to_dict_impl(self)

    def to_json(self, bint orient_records=True):
        return _to_json_impl(self, orient_records)

    def to_numpy(self):
        return _to_numpy_impl(self)

    def info(self):
        return _info_impl(self)

    def _dtype_summary(self):
        return _dtype_summary_impl(self)

    def memory_usage(self):
        return _memory_usage_impl(self)

    # Merge
    def concat(self, FastDataFrame other):
        return _concat_impl(self, other)

    def merge(self, FastDataFrame right, object on=None, str how='inner'):
        if isinstance(on, str):
            on = [on]
        return _merge_impl(self, right, on, how)

    def duplicated(self, list subset=None, str keep='first'):
        return _duplicated_impl(self, subset, keep)

    def drop_duplicates(self, list subset=None, str keep='first', bint inplace=False):
        return _drop_duplicates_impl(self, subset, keep, inplace)

    # Transform
    def argsort_by(self, str col_name, bint reverse=False):
        return _argsort_by_impl(self, col_name, reverse)

    def sort_values(self, str col_name, bint reverse=False, bint inplace=False):
        return _sort_values_impl(self, col_name, reverse, inplace)

    def sort_index(self, bint ascending=True, bint inplace=False):
        if 'index' not in self.columns:
            raise KeyError("No 'index' column found. Call reset_index() first.")
        return self.sort_values('index', reverse=not ascending, inplace=inplace)

    def diff(self, str col_name, int periods=1):
        return _diff_impl(self, col_name, periods)

    def apply(self, object func, str axis='columns'):
        return _apply_impl(self, func, axis)

    def clip_column(self, str col_name, double lower, double upper, bint inplace=False):
        return _clip_column_impl(self, col_name, lower, upper, inplace)

    def where_column(self, str col_name, str cond_col, object other):
        return _where_column_impl(self, col_name, cond_col, other)

    def mask_column(self, str col_name, str cond_col, object other):
        return _mask_column_impl(self, col_name, cond_col, other)

    def where_row(self, IntColumn mask, object fill=0):
        return _where_row_impl(self, mask, fill)

    def isnull(self):
        return _isnull_impl(self)

    def fillna(self, double value, col_names=None):
        return _fillna_impl(self, value, col_names)

    def dropna(self):
        return _dropna_impl(self)

    def abs(self):
        return _abs_impl(self)

    def round(self, int decimals=0):
        return _round_impl(self, decimals)

    def all(self):
        return _all_impl(self)

    def any(self):
        return _any_impl(self)

    def pct_change(self, int periods=1):
        return _pct_change_impl(self, periods)

    def replace(self, to_replace, value):
        return _replace_impl(self, to_replace, value)

    def ffill(self):
        return _ffill_impl(self)

    def bfill(self):
        return _bfill_impl(self)

    def head(self, Py_ssize_t n=5):
        return _head_impl(self, n)

    def tail(self, Py_ssize_t n=5):
        return _tail_impl(self, n)

    def sample(self, Py_ssize_t n=0, double frac=0.0, bint replace=False, int seed=-1):
        return _sample_impl(self, n, frac, replace, seed)

    def nsmallest(self, Py_ssize_t n, str col_name):
        return _nsmallest_impl(self, n, col_name)

    def nlargest(self, Py_ssize_t n, str col_name):
        return _nlargest_impl(self, n, col_name)

    def pipe(self, object func, *args, **kwargs):
        return _pipe_impl(self, func, args, kwargs)

    def assign(self, **kwargs):
        return _assign_impl(self, kwargs)

    def select_dtypes(self, list include_types=None, list exclude_types=None):
        return _select_dtypes_impl(self, include_types, exclude_types)

    def astype(self, dict dtype_dict):
        return _astype_impl(self, dtype_dict)

    def add(self, tuple col_names):
        return _add_impl(self, col_names)

    def sub(self, tuple col_names):
        return _sub_impl(self, col_names)

    def mul(self, tuple col_names):
        return _mul_impl(self, col_names)

    def div(self, tuple col_names):
        return _div_impl(self, col_names)

    def reshape(self, Py_ssize_t rows, Py_ssize_t cols):
        return _reshape_impl(self, rows, cols)

    @property
    def T(self):
        return _T_impl(self)

    # Metrics
    @property
    def metric(self):
        # 快路径：缓存命中时零实例化开销（C 级字段读取 + None 判断）
        if self._metric_cache is not None:
            return self._metric_cache
        # 慢路径：仅首次访问时创建一次，此后均摊为 0
        self._metric_cache = _make_metric_accessor(self)
        return self._metric_cache

    def euclidean_dist_withcols(self, col1, col2):
        return _euclidean_dist_withcols_impl(self, col1, col2)

    def manhattan_dist_withcols(self, col1, col2):
        return _manhattan_dist_withcols_impl(self, col1, col2)

    def cosine_dist_withcols(self, col1, col2):
        return _cosine_dist_withcols_impl(self, col1, col2)

    def cosine_similarity_withcols(self, col1, col2):
        return _cosine_similarity_withcols_impl(self, col1, col2)

    def jaccard_sim_withcols(self, col1, col2):
        return _jaccard_sim_withcols_impl(self, col1, col2)

    def hamming_dist_withcols(self, col1, col2):
        return _hamming_dist_withcols_impl(self, col1, col2)

    # Linear Algebra
    @property
    def linalg(self):
        if self._linalg_cache is not None:
            return self._linalg_cache
        self._linalg_cache = _make_linalg_accessor(self)
        return self._linalg_cache

    def dot(self, FastDataFrame b):
        return _linalg_dot(self, b)

    def matmul(self, FastDataFrame b):
        return _linalg_dot(self, b)

    def transpose(self):
        return _linalg_transpose(self)

    def det(self):
        return _linalg_det(self)

    def inv(self):
        return _linalg_inv(self)

    def solve(self, FastDataFrame b):
        return _linalg_solve(self, b)

    def norm(self, str ord_type="frobenius"):
        return _linalg_norm(self, ord_type)

    def euclidean_dist_matrix(self):
        return _linalg_euclidean_dist(self)

    def manhattan_dist_matrix(self):
        return _linalg_manhattan_dist(self)

    def cosine_dist_matrix(self):
        return _linalg_cosine_dist(self)

    def hamming_dist_matrix(self):
        return _linalg_hamming_dist(self)

    def jaccard_sim_matrix(self):
        return _linalg_jaccard_sim(self)

    def cosine_similarity_matrix(self):
        return _linalg_cosine_similarity(self)

    def corr(self):
        return _corr_impl(self)

    def cov(self):
        return _cov_impl(self)

    def corrwith(self, other):
        return _corrwith_impl(self, other)

    def corrcoef(self, col_a, col_b):
        return _corrcoef_impl(self, col_a, col_b)

    def cov_withcols(self, col_a, col_b):
        return _cov_withcols_impl(self, col_a, col_b)

    def pearson_sim(self):
        return _pearson_sim_impl(self)

    # Factory
    @staticmethod
    def from_dict(data, schema=None, columns=None):
        return _from_dict_impl(data, schema, columns)

    @staticmethod
    def from_list(rows, schema, columns=None):
        return _from_list_impl(rows, schema, columns)

    @staticmethod
    def from_arrays(arrays, schema, columns=None):
        return _from_arrays_impl(arrays, schema, columns)

    @staticmethod
    def arange(n):
        return _arange_impl(n)

    @staticmethod
    def zeros(*args):
        if len(args) == 1:
            return _zeros_impl(args[0], 1)
        return _zeros_impl(args[0], args[1])

    @staticmethod
    def ones(*args):
        if len(args) == 1:
            return _ones_impl(args[0], 1)
        return _ones_impl(args[0], args[1])

    @staticmethod
    def randn(*args):
        if len(args) == 1:
            return _randn_impl(args[0], 1)
        return _randn_impl(args[0], args[1])

    @staticmethod
    def rand(*args):
        if len(args) == 1:
            return _rand_impl(args[0], 1)
        return _rand_impl(args[0], args[1])

    # Rolling
    def rolling(self, Py_ssize_t window, str col_name):
        return _rolling_impl(self, window, col_name)

    # DataFrame-level shift (delegates to cdef _shift_impl in .pxi)
    def shift(self, Py_ssize_t periods=1, str axis='index'):
        return _shift_impl(self, periods, axis)

    # DataFrame-level diff for all numeric columns
    def diff_all(self, int periods=1):
        return _diff_column_all_impl(self, periods)

    # DataFrame-level rank
    def rank(self, str col_name, str method='average', bint ascending=True, str na_option='keep'):
        return _rank_impl(self, col_name, method, ascending, na_option)

    # DataFrame-level cut
    def cut(self, str col_name, list bins, bint include_lowest=False, str right='right'):
        return _cut_impl(self, col_name, bins, include_lowest, right)

    # DataFrame-level qcut
    def qcut(self, str col_name, int q=4, str labels='True', double precision=3):
        return _qcut_impl(self, col_name, q, labels, precision)

    def pivot(self, str index, str columns, str values):
        return _pivot_impl(self, index, columns, values)

    def melt(self, list id_vars=None, list value_vars=None, str var_name='variable', str value_name='value'):
        return _melt_impl(self, id_vars, value_vars, var_name, value_name)

    def expanding_sum(self, str col_name):
        return _expanding_sum_impl(self, col_name)

    def ewm_mean(self, str col_name, double span):
        return _ewm_mean_impl(self, col_name, span)


include "../api/_fastdataframe_crud.pxi"
include "../api/_fastdataframe_transform.pxi"
include "../api/_fastdataframe_agg.pxi"
include "../api/_fastdataframe_io.pxi"
include "../api/_fastdataframe_merge.pxi"
include "../api/_fastdataframe_metric.pxi"
include "../api/_fastdataframe_factory.pxi"
include "../api/_fastdataframe_rolling.pxi"
include "../api/_fastdataframe_shift.pxi"
include "../api/_fastdataframe_stats.pxi"
include "../api/_fastdataframe_groupby.pxi"
include "../api/_fastdataframe_linalg.pxi"
include "../api/_fastdataframe_standalone.pxi"


# _DataFrameMetricAccessor — 懒加载缓存的 metric 访问器
# ============================================================
# 用法: df.metric.euclidean('a', 'b') / df.metric.corr() 等。
# 实例由 FastDataFrame.metric property 首次访问时创建并缓存，
# 实例化开销均摊为 0（参见 _make_metric_accessor）。
cdef class _DataFrameMetricAccessor:
    """Metric accessor with lazy caching on the parent FastDataFrame."""

    def __cinit__(self):
        # 无参 __cinit__：允许工厂函数直接 __new__ 绕过参数打包
        pass

    # ==================== Distance ====================

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def euclidean(self, col1, col2):
        return _euclidean_dist_withcols_impl(self._df, col1, col2)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def manhattan(self, col1, col2):
        return _manhattan_dist_withcols_impl(self._df, col1, col2)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def cosine_dist(self, col1, col2):
        return _cosine_dist_withcols_impl(self._df, col1, col2)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def cosine_similarity(self, col1, col2):
        return _cosine_similarity_withcols_impl(self._df, col1, col2)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def jaccard(self, col1, col2):
        return _jaccard_sim_withcols_impl(self._df, col1, col2)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def hamming(self, col1, col2):
        return _hamming_dist_withcols_impl(self._df, col1, col2)

    # ==================== Correlation / Covariance ====================

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def corr(self):
        return _corr_impl(self._df)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def cov(self):
        return _cov_impl(self._df)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def corrwith(self, other):
        return _corrwith_impl(self._df, other)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def corrcoef(self, col_a, col_b):
        return _corrcoef_impl(self._df, col_a, col_b)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def cov_withcols(self, col_a, col_b):
        return _cov_withcols_impl(self._df, col_a, col_b)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def pearson(self):
        return _pearson_sim_impl(self._df)


cdef inline _DataFrameMetricAccessor _make_metric_accessor(FastDataFrame df):
    # C 级工厂：__new__ 无参调用直接命中 tp_new，彻底绕过 Python 参数打包/解包
    cdef _DataFrameMetricAccessor acc = <_DataFrameMetricAccessor>_DataFrameMetricAccessor.__new__(_DataFrameMetricAccessor)
    # 直接写底层 C 结构体字段（Cython 自动处理 Py_INCREF）
    acc._df = df
    return acc
