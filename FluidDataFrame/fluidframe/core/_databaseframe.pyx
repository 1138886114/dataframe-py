# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False

from libc.string cimport memcpy, memmove, strncmp, strncpy, memset
from libc.stdio cimport snprintf
from libc.stdint cimport int32_t, int64_t, uint32_t, uint64_t
from libc.time cimport time_t, time
from libc.math cimport NAN, sqrt, log, cos, isnan, INFINITY, fabs, exp
from libc.stdlib cimport abs as c_abs, qsort
from cpython.unicode cimport Py_UCS2, PyUnicode_KIND, PyUnicode_READ_CHAR, PyUnicode_2BYTE_KIND, PyUnicode_DATA
from cpython.number cimport PyNumber_Index

cimport cython
from fluidframe.core._columns cimport (
    BaseColumn, NumericType, DataType,
    aligned_malloc, aligned_realloc, os_aligned_free,
    _resolve_astype_target,
    _vec_pearson_corr, _vec_pearson_cov,
    _vec_euclidean_dist, _vec_manhattan_dist, _vec_cosine_sim, _vec_jaccard_sim, _vec_hamming_dist,
    _int_double_euclidean_dist, _int_double_manhattan_dist,
    _int_double_cosine_sim, _int_double_jaccard_sim, _int_double_hamming_dist,
    _c_shift_left, _c_shift_right, _col_pop_back, _col_pop_front, _col_concat,
    _col_drop_row, _col_gather_zero, _col_gather_nan,
    _col_shift_right_2d_double, _col_shift_right_2d_int, _col_shift_left_2d,
    _col_concat_2d, _col_gather_2d_zero, _col_gather_2d_nan,
    _col_count_nonzeros, _col_memcpy, _col_abs, _col_round, _cmp_double_asc,
    _col_ffill_double, _col_bfill_double, _c_max, _c_min, _c_sum, _c_prod,
    _c_idxmax, _c_idxmin, _c_var, _c_mad,
    IntColumn, DoubleColumn, AsciiColumn, UCS2Column, DateColumn, BoolColumn,
    CategoryColumn, DateTimeColumn, TimeColumn,
    _days_to_ymd, _ymd_to_days, _parse_date_str, _parse_datetime_str, _parse_time_str,
    TYPE_INT, TYPE_DOUBLE, TYPE_ASCII, TYPE_UCS2, TYPE_DATE, TYPE_BOOL,
    TYPE_CATEGORY, TYPE_DATETIME, TYPE_TIME,
)

# ============================================================
#  Python-level imports used across DataFrame implementation
# ============================================================
from fluidframe._schema_types import Ascii, Date, Bool, Category, DateTime, Time, _parse_schema


# ============================================================
#  模块级常量（DataFrame 内部使用）
# ============================================================
# 属性化白名单：列上同名零参方法在 .dt 下以属性返回值列。
cdef list _DT_ATTRS = [
    'year', 'month', 'day', 'hour', 'minute', 'second', 'quarter',
    'week_of_year', 'day_of_week', 'day_of_year', 'days_in_month',
    'is_leap_year', 'is_weekend',
    'is_month_start', 'is_month_end', 'is_year_start', 'is_year_end',
    'is_quarter_start', 'is_quarter_end',
    'day_name', 'month_name',
]


cdef dict _TYPE_NAMES = {
    TYPE_INT: 'int', TYPE_DOUBLE: 'float', TYPE_ASCII: 'ascii',
    TYPE_UCS2: 'ucs2', TYPE_DATE: 'date', TYPE_BOOL: 'bool',
    TYPE_CATEGORY: 'category', TYPE_DATETIME: 'datetime', TYPE_TIME: 'time',
}

cdef dict _TYPE_PYCLASSES = {
    TYPE_INT: int, TYPE_DOUBLE: float, TYPE_ASCII: Ascii,
    TYPE_UCS2: str, TYPE_DATE: Date, TYPE_BOOL: bool,
    TYPE_CATEGORY: Category, TYPE_DATETIME: DateTime, TYPE_TIME: Time,
}


# ============================================================
#  C 头文件声明（extern 注入）
# ============================================================
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


# ============================================================
#  _RollingWindow（滚动窗口访问器：DataFrame/Series 双模式）
# ============================================================
cdef double* _win_weights_buf(int kind, Py_ssize_t win_n):
    """按 win_type 生成窗口权重 C 缓冲（调用方负责 os_aligned_free）。
    triang：奇窗 1-|2i-(M-1)|/(M+1)；偶窗 1-|2i-(M-1)|/M。"""
    cdef double* buf = <double*>aligned_malloc(win_n * sizeof(double))
    if buf == NULL:
        raise MemoryError()
    cdef Py_ssize_t i, d
    cdef double denom
    if kind == WIN_BOXCAR:
        for i in range(win_n):
            buf[i] = 1.0
    elif kind == WIN_TRIANG:
        denom = <double>(win_n + 1) if (win_n % 2 == 1) else <double>win_n
        for i in range(win_n):
            d = 2 * i - (win_n - 1)
            if d < 0: d = -d
            buf[i] = 1.0 - <double>d / denom
    else:
        os_aligned_free(buf)
        raise ValueError(f"Unsupported win_type kind: {kind}")
    return buf


cdef class _RollingWindow:
    """滚动窗口访问器，委托列级 roll_xxx 内核。
    mode-1（Series）：绑定单列，算子返回同长 Series；
    mode-2（DataFrame）：逐数值列滚动，算子返回同 schema 的整表 DataFrame。"""

    _SUPPORTED_TYPES = {TYPE_INT, TYPE_DOUBLE}

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __cinit__(self, object obj, Py_ssize_t window, min_periods=None,
                  bint center=False, bint is_df=False, str name=None,
                  IndexBase index=None, object win_type=None):
        self._window = window
        self._min_periods = min_periods
        self._center = center
        self._is_df = is_df
        self._win_kind = WIN_NONE
        self._is_win = False
        self._mp = 0
        self._wbuf = NULL
        if win_type is not None:
            if win_type == 'boxcar':
                self._win_kind = WIN_BOXCAR
            elif win_type == 'triang':
                self._win_kind = WIN_TRIANG
            else:
                raise ValueError(f"Invalid win_type '{win_type}'; supported: 'boxcar', 'triang'")
            self._is_win = True
            if min_periods is not None:
                if isinstance(min_periods, int):
                    self._mp = <Py_ssize_t>min_periods
                else:
                    self._mp = <Py_ssize_t>PyNumber_Index(min_periods)
            else:
                self._mp = window
            if self._mp < 1:
                self._mp = 1
            if self._mp > window:
                raise ValueError(f"min_periods {min_periods} must be <= window {window}")
            self._wbuf = _win_weights_buf(self._win_kind, window)
        if is_df:
            self._df = <DataFrame>obj
            self._index = self._df.index
            self._name = ''
        else:
            self._col = <BaseColumn>obj
            self._index = index
            self._name = name if name is not None else '_roll'
            if self._col.dtype not in self._SUPPORTED_TYPES:
                type_name = _TYPE_NAMES.get(self._col.dtype, 'unknown')
                raise TypeError(f"Rolling window operations are not supported for {type_name} columns")

    def __dealloc__(self):
        if self._wbuf != NULL:
            os_aligned_free(self._wbuf)
            self._wbuf = NULL

    cdef object _call_col(self, BaseColumn col, str method, list args):
        """单列滚动算子调用（min_periods/center 固定透传）。"""
        cdef object fn = getattr(col, method)
        return fn(self._window, *args, min_periods=self._min_periods, center=self._center)

    cdef object _call_col_w(self, BaseColumn col, str method, list args):
        """win_type 加权模式：直接调用列级 cdef 核（复用 cinit 缓存的 C 权重缓冲）。"""
        cdef double* wbuf = self._wbuf
        cdef Py_ssize_t win = self._window, mp = self._mp
        cdef bint center = self._center
        if col.dtype == TYPE_INT:
            if method == 'roll_win_mean':
                return (<IntColumn>col)._roll_win_sum_mean(win, wbuf, mp, center, True)
            if method == 'roll_win_sum':
                return (<IntColumn>col)._roll_win_sum_mean(win, wbuf, mp, center, False)
            if method == 'roll_win_var':
                return (<IntColumn>col)._roll_win_var_buf(win, wbuf, mp, center, <bint>args[0])
            return (<IntColumn>col)._roll_win_std_buf(win, wbuf, mp, center, <bint>args[0])
        cdef DoubleColumn dcol = <DoubleColumn>col
        if method == 'roll_win_mean':
            return dcol._roll_win_sum_mean(win, wbuf, mp, center, True)
        if method == 'roll_win_sum':
            return dcol._roll_win_sum_mean(win, wbuf, mp, center, False)
        if method == 'roll_win_var':
            return dcol._roll_win_var_buf(win, wbuf, mp, center, <bint>args[0])
        return dcol._roll_win_std_buf(win, wbuf, mp, center, <bint>args[0])

    cdef void _check_win_type(self, str method):
        """加权模式下仅放行 mean/sum/var/std；其余算子按 pandas Window 语义抛 AttributeError。"""
        if self._is_win:
            raise AttributeError(f"'Window' object has no attribute '{method}'")

    cdef object _wrap(self, object res):
        """Series 模式：BaseColumn 结果包成带索引的 Series。"""
        return _wrap_series(self._name, res, self._index)

    cdef object _apply(self, str method, list args, bint weighted):
        """df 模式：逐数值列滚动并回装整表 DataFrame（weighted 走 win_type 加权核）；
        无数值列则报错。"""
        if not self._is_df:
            if weighted:
                return self._wrap(self._call_col_w(self._col, method, args))
            return self._wrap(self._call_col(self._col, method, args))
        cdef DataFrame result = DataFrame()
        result.df_length = self._df.df_length
        result.index = self._df.index
        cdef BaseColumn col
        cdef bint found_numeric = False
        for col_name, col in self._df._columns.items():
            if col.dtype != TYPE_INT and col.dtype != TYPE_DOUBLE:
                continue
            found_numeric = True
            if weighted:
                result._columns[col_name] = self._call_col_w(col, method, args)
            else:
                result._columns[col_name] = self._call_col(col, method, args)
        if not found_numeric:
            raise TypeError(f"Rolling {method} requires at least one numeric column")
        return result

    cdef BaseColumn _resolve_other(self, object other):
        """把 Series / _RollingWindow 统一解析为底层列对象。"""
        if isinstance(other, _RollingWindow):
            return (<_RollingWindow>other)._col
        if isinstance(other, Series):
            return (<Series>other)._col
        if isinstance(other, BaseColumn):
            return <BaseColumn>other
        raise TypeError("corr/cov 'other' must be a Series, rolling window or column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def mean(self):
        if self._is_win:
            return self._apply('roll_win_mean', [], True)
        return self._apply('roll_mean', [], False)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sum(self):
        if self._is_win:
            return self._apply('roll_win_sum', [], True)
        return self._apply('roll_sum', [], False)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def min(self):
        self._check_win_type('min')
        return self._apply('roll_min', [], False)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def max(self):
        self._check_win_type('max')
        return self._apply('roll_max', [], False)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def std(self, bint ddof=True):
        if self._is_win:
            return self._apply('roll_win_std', [ddof], True)
        return self._apply('roll_std', [ddof], False)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def var(self, bint ddof=True):
        if self._is_win:
            return self._apply('roll_win_var', [ddof], True)
        return self._apply('roll_var', [ddof], False)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def median(self):
        self._check_win_type('median')
        return self._apply('roll_median', [], False)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def quantile(self, double q=0.5):
        self._check_win_type('quantile')
        return self._apply('roll_quantile', [q], False)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def skew(self):
        self._check_win_type('skew')
        return self._apply('roll_skew', [], False)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def kurt(self):
        self._check_win_type('kurt')
        return self._apply('roll_kurt', [], False)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def corr(self, object other):
        self._check_win_type('corr')
        if self._is_df:
            return self._corr_cov_df('roll_corr', other)
        return self._wrap(self._call_col(self._col, 'roll_corr', [self._resolve_other(other)]))

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cov(self, object other):
        self._check_win_type('cov')
        if self._is_df:
            return self._corr_cov_df('roll_cov', other)
        return self._wrap(self._call_col(self._col, 'roll_cov', [self._resolve_other(other)]))

    cdef object _corr_cov_df(self, str method, object other):
        """df 模式双列滚动：other 为 Series→每列与之配对；DataFrame→按列名对齐。"""
        cdef DataFrame result = DataFrame()
        result.df_length = self._df.df_length
        result.index = self._df.index
        cdef BaseColumn col, other_col
        cdef bint found_numeric = False
        cdef dict other_cols
        if isinstance(other, DataFrame):
            other_cols = (<DataFrame>other)._columns
            for col_name, col in self._df._columns.items():
                if col.dtype != TYPE_INT and col.dtype != TYPE_DOUBLE:
                    continue
                if col_name not in other_cols:
                    raise KeyError(f"Column '{col_name}' not found in 'other'")
                found_numeric = True
                result._columns[col_name] = self._call_col_single(col, method,
                                                                  <BaseColumn>other_cols[col_name])
        else:
            other_col = self._resolve_other(other)
            for col_name, col in self._df._columns.items():
                if col.dtype != TYPE_INT and col.dtype != TYPE_DOUBLE:
                    continue
                found_numeric = True
                result._columns[col_name] = self._call_col_single(col, method, other_col)
        if not found_numeric:
            raise TypeError(f"Rolling {method} requires at least one numeric column")
        return result

    cdef object _call_col_single(self, BaseColumn col, str method, BaseColumn other):
        return getattr(col, method)(self._window, other,
                                    min_periods=self._min_periods, center=self._center)


cdef inline double _resolve_ewm_alpha(com=None, span=None, halflife=None, alpha=None):
    """pandas 式指数加权参数解析：com/span/halflife/alpha 四选一，返回 alpha。
    com → 1/(1+com)；span → 2/(span+1)；halflife → 1-exp(-ln2/halflife)；alpha 直传。
    仅支持 method='single'（times 参数本框架暂不支持）。"""
    cdef Py_ssize_t provided = (com is not None) + (span is not None) + \
                               (halflife is not None) + (alpha is not None)
    if provided > 1:
        raise ValueError("comass, span, halflife, and alpha are mutually exclusive")
    if provided == 0:
        raise ValueError("Must pass one of comass, span, halflife, or alpha")
    cdef object param = com if com is not None else (span if span is not None
                                                     else (halflife if halflife is not None else alpha))
    if isinstance(param, str):
        raise ValueError(f"Invalid {type(param).__name__} argument: {param}")
    cdef double value = <double>param
    if com is not None:
        if value <= 0.0:
            raise ValueError("com must satisfy: com > 0")
        return 1.0 / (1.0 + value)
    if span is not None:
        if value <= 1.0:
            raise ValueError("span must satisfy: span > 1")
        return 2.0 / (value + 1.0)
    if halflife is not None:
        if value <= 0.0:
            raise ValueError("halflife must satisfy: halflife > 0")
        return 1.0 - exp(-0.6931471805599453 / value)
    if not (0.0 < value <= 1.0):
        raise ValueError("alpha must satisfy: 0 < alpha <= 1")
    return value


cdef class _EWMWindow:
    """指数加权窗口访问器（pandas 式签名），委托列级 ewm_xxx 算子。
    mode-1（Series）：绑定单列，算子返回同长 Series；
    mode-2（DataFrame）：逐数值列 ewm，算子返回同 schema 的整表 DataFrame。"""

    _SUPPORTED_TYPES = {TYPE_INT, TYPE_DOUBLE}

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __cinit__(self, object obj, com=None, span=None, halflife=None, alpha=None,
                  Py_ssize_t min_periods=0, bint adjust=True, bint ignore_na=False,
                  bint is_df=False, str name=None, IndexBase index=None):
        cdef object param
        self._min_periods = min_periods
        self._adjust = adjust
        self._ignore_na = ignore_na
        if is_df:
            self._df = <DataFrame>obj
            self._index = self._df.index
            self._name = ''
            self._is_df = True
        else:
            self._col = <BaseColumn>obj
            self._index = index
            self._name = name if name is not None else '_ewm'
            self._is_df = False
            if self._col.dtype not in self._SUPPORTED_TYPES:
                type_name = _TYPE_NAMES.get(self._col.dtype, 'unknown')
                raise TypeError(f"EWM operations are not supported for {type_name} columns")
        self._alpha = _resolve_ewm_alpha(com, span, halflife, alpha)

    cdef object _wrap(self, object res):
        return _wrap_series(self._name, res, self._index)

    cdef object _apply(self, str method, list args):
        if not self._is_df:
            return self._wrap(getattr(self._col, method)(self._alpha, *args,
                                                         min_periods=self._min_periods,
                                                         adjust=self._adjust,
                                                         ignore_na=self._ignore_na))
        cdef DataFrame result = DataFrame()
        result.df_length = self._df.df_length
        result.index = self._df.index
        cdef BaseColumn col
        cdef bint found_numeric = False
        for col_name, col in self._df._columns.items():
            if col.dtype != TYPE_INT and col.dtype != TYPE_DOUBLE:
                continue
            found_numeric = True
            result._columns[col_name] = getattr(col, method)(self._alpha, *args,
                                                             min_periods=self._min_periods,
                                                             adjust=self._adjust,
                                                             ignore_na=self._ignore_na)
        if not found_numeric:
            raise TypeError(f"EWM {method} requires at least one numeric column")
        return result

    cdef BaseColumn _resolve_other(self, object other):
        if isinstance(other, _EWMWindow):
            return (<_EWMWindow>other)._col
        if isinstance(other, Series):
            return (<Series>other)._col
        if isinstance(other, BaseColumn):
            return <BaseColumn>other
        raise TypeError("corr/cov 'other' must be a Series, EWM window or column")

    cdef object _corr_cov_df(self, str method, object other):
        cdef DataFrame result = DataFrame()
        result.df_length = self._df.df_length
        result.index = self._df.index
        cdef BaseColumn col, other_col
        cdef bint found_numeric = False
        cdef dict other_cols
        cdef bint is_corr = method == 'ewm_corr'
        if isinstance(other, DataFrame):
            other_cols = (<DataFrame>other)._columns
            for col_name, col in self._df._columns.items():
                if col.dtype != TYPE_INT and col.dtype != TYPE_DOUBLE:
                    continue
                if col_name not in other_cols:
                    raise KeyError(f"Column '{col_name}' not found in 'other'")
                found_numeric = True
                result._columns[col_name] = self._call_col_other(col, method,
                                                                 <BaseColumn>other_cols[col_name], is_corr)
        else:
            other_col = self._resolve_other(other)
            for col_name, col in self._df._columns.items():
                if col.dtype != TYPE_INT and col.dtype != TYPE_DOUBLE:
                    continue
                found_numeric = True
                result._columns[col_name] = self._call_col_other(col, method, other_col, is_corr)
        if not found_numeric:
            raise TypeError(f"EWM {method} requires at least one numeric column")
        return result

    cdef object _call_col_other(self, BaseColumn col, str method, BaseColumn other, bint is_corr):
        if is_corr:
            return getattr(col, method)(other, self._alpha,
                                        min_periods=self._min_periods,
                                        adjust=self._adjust,
                                        ignore_na=self._ignore_na)
        return getattr(col, method)(other, self._alpha, False,
                                    min_periods=self._min_periods,
                                    adjust=self._adjust,
                                    ignore_na=self._ignore_na)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def mean(self):
        return self._apply('ewm_mean', [])

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sum(self):
        return self._apply('ewm_sum', [])

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def var(self, bint bias=False): return self._apply('ewm_var', [bias])

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def std(self, bint bias=False): return self._apply('ewm_std', [bias])

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cov(self, object other):
        if self._is_df:
            return self._corr_cov_df('ewm_cov', other)
        return self._wrap(getattr(self._col, 'ewm_cov')(self._resolve_other(other), self._alpha,
                                                        bias=False,
                                                        min_periods=self._min_periods,
                                                        adjust=self._adjust,
                                                        ignore_na=self._ignore_na))

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def corr(self, object other):
        if self._is_df:
            return self._corr_cov_df('ewm_corr', other)
        return self._wrap(getattr(self._col, 'ewm_corr')(self._resolve_other(other), self._alpha,
                                                         min_periods=self._min_periods,
                                                         adjust=self._adjust,
                                                         ignore_na=self._ignore_na))


# ============================================================
#  Index 类型体系（IndexBase / IntIndex / RangeIndex / MultiIndex）
# ============================================================
cdef class IndexBase:
    cdef IndexBase _take(self, Py_ssize_t* indices, Py_ssize_t n):
        raise NotImplementedError()

    cdef IndexBase _concat(self, IndexBase other):
        raise NotImplementedError()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def get_loc(self, object key):
        raise NotImplementedError()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def values(self):
        raise NotImplementedError()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __getitem__(self, object key):
        if not isinstance(key, int):
            raise TypeError("Index keys must be integers")
        cdef Py_ssize_t i = key
        cdef list vals = self.values()
        if i < 0: i += self.length
        if i < 0 or i >= self.length:
            raise IndexError("Index out of range")
        return vals[i]

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __len__(self):
        return self.length


cdef class IntIndex(IndexBase):
    def __cinit__(self, IntColumn col):
        self.data = col
        self.length = col.length
        self.index_type = INDEX_INT

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def values(self):
        cdef list result = []
        cdef Py_ssize_t i
        for i in range(self.length):
            result.append(self.data.data_c[i])
        return result

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __getitem__(self, object key):
        if not isinstance(key, int):
            raise TypeError("Index keys must be integers")
        cdef Py_ssize_t i = key
        if i < 0: i += self.length
        if i < 0 or i >= self.length:
            raise IndexError("Index out of range")
        return self.data.data_c[i]

    cdef IndexBase _take(self, Py_ssize_t* indices, Py_ssize_t n):
        cdef IntColumn new_col = IntColumn(n)
        new_col.length = n
        cdef int* old_data = self.data.data_c
        cdef int* new_data = new_col.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                new_data[i] = old_data[indices[i]]
        return IntIndex(new_col)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def get_loc(self, object key):
        cdef int target = <int>key
        cdef list matches = []
        cdef Py_ssize_t i
        cdef int* data_ptr = self.data.data_c
        for i in range(self.length):
            if data_ptr[i] == target:
                matches.append(i)
        if not matches: raise KeyError(key)
        if len(matches) == 1: return matches[0]
        return matches

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def append(self, int val):
        self.data.append(val)
        self.length = self.data.length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_back(self):
        if self.length == 0:
            raise IndexError("pop from empty Index")
        cdef int val = self.data.pop_back()
        self.length = self.data.length
        return val

    cdef IndexBase _concat(self, IndexBase other):
        if not isinstance(other, IntIndex):
            raise TypeError("Cannot concat IntIndex with non-IntIndex")
        cdef IntIndex o = <IntIndex>other
        cdef IntColumn new_col = IntColumn(self.length + o.length)
        new_col.length = self.length + o.length
        cdef Py_ssize_t i
        cdef int* src1 = self.data.data_c
        cdef int* src2 = o.data.data_c
        cdef int* dst = new_col.data_c
        with nogil:
            for i in range(self.length):
                dst[i] = src1[i]
            for i in range(o.length):
                dst[self.length + i] = src2[i]
        return IntIndex(new_col)


cdef class RangeIndex(IndexBase):
    def __cinit__(self, Py_ssize_t start, Py_ssize_t length, Py_ssize_t step=1):
        self.index_type = INDEX_RANGE
        self.start = start
        self.length = length
        self.step = step
        if step == 0: raise ValueError("Step cannot be zero.")

    cdef IndexBase _take(self, Py_ssize_t* indices, Py_ssize_t n):
        cdef IntColumn new_col = IntColumn(n)
        new_col.length = n
        cdef int* out_data = new_col.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                out_data[i] = <int>(self.start + indices[i] * self.step)
        return IntIndex(new_col)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def get_loc(self, object key):
        if not isinstance(key, int): raise KeyError(key)
        cdef Py_ssize_t val = <Py_ssize_t>key
        cdef Py_ssize_t diff = val - self.start

        # O(1) 数学验证
        if diff % self.step != 0: raise KeyError(key)
        cdef Py_ssize_t idx = diff // self.step
        if idx < 0 or idx >= self.length: raise KeyError(key)
        return idx

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def values(self):
        cdef list result = []
        cdef Py_ssize_t i
        for i in range(self.length):
            result.append(self.start + i * self.step)
        return result

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __getitem__(self, object key):
        if not isinstance(key, int):
            raise TypeError("Index keys must be integers")
        cdef Py_ssize_t i = key
        if i < 0: i += self.length
        if i < 0 or i >= self.length:
            raise IndexError("Index out of range")
        return self.start + i * self.step

    cdef IndexBase _concat(self, IndexBase other):
        cdef RangeIndex o
        cdef IntColumn new_col
        cdef list v1, v2
        cdef Py_ssize_t i
        cdef int* dst
        if isinstance(other, RangeIndex):
            o = <RangeIndex>other
            if self.step == o.step and self.start + self.length * self.step == o.start:
                return RangeIndex(self.start, self.length + o.length, self.step)
        new_col = IntColumn(self.length + other.length)
        new_col.length = self.length + other.length
        v1 = self.values()
        v2 = other.values()
        dst = new_col.data_c
        for i in range(self.length):
            dst[i] = <int>v1[i]
        for i in range(other.length):
            dst[self.length + i] = <int>v2[i]
        return IntIndex(new_col)

cdef class MultiIndex(IndexBase):
    def __cinit__(self, list levels, list codes, list names=None):
        self.index_type = INDEX_MULTI
        self.levels = levels
        self.codes = codes
        self.nlevels = len(levels)
        self.names = names if names else [None] * self.nlevels
        self.length = (<IntColumn>codes[0]).length if len(codes) > 0 else 0
        self._hash_cache = None
        self._level_maps = None

    cdef void _build_level_maps(self):
        """延迟构建：将 values 转换为 codes 的 O(1) 反向映射"""
        self._level_maps = []
        for level in self.levels:
            self._level_maps.append({val: idx for idx, val in enumerate(level)})

    cdef void _build_hash_cache(self):
        """黑魔法：直接使用 C 内存字节流作为哈希键，无 Python 对象冗余开销"""
        cdef Py_ssize_t i, j
        cdef int n_levels = self.nlevels

        cdef int** ptrs = <int**>aligned_malloc(n_levels * sizeof(int*))
        cdef int* buf = <int*>aligned_malloc(n_levels * sizeof(int))
        if not ptrs or not buf:
            if ptrs: os_aligned_free(ptrs)
            if buf: os_aligned_free(buf)
            raise MemoryError()

        cdef Py_ssize_t bytes_len = n_levels * sizeof(int)
        cdef bytes b_key

        try:
            for j in range(n_levels):
                ptrs[j] = (<IntColumn>self.codes[j]).data_c

            self._hash_cache = {}

            for i in range(self.length):
                for j in range(n_levels):
                    buf[j] = ptrs[j][i]

                # 将 int 数组内存强制转换为 bytes
                b_key = (<char*>buf)[:bytes_len]

                if b_key not in self._hash_cache:
                    self._hash_cache[b_key] = [i]
                else:
                    (<list>self._hash_cache[b_key]).append(i)
        finally:
            os_aligned_free(ptrs)
            os_aligned_free(buf)

    cdef IndexBase _take(self, Py_ssize_t* indices, Py_ssize_t n):
        cdef list new_codes = []
        cdef IntColumn col, new_col
        cdef Py_ssize_t i
        for col in self.codes:
            new_col = IntColumn(n)
            new_col.length = n
            with nogil:
                for i in range(n):
                    new_col.data_c[i] = col.data_c[indices[i]]
            new_codes.append(new_col)
        return MultiIndex(self.levels, new_codes, self.names)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __getitem__(self, object key):
        if not isinstance(key, int):
            raise TypeError("Index keys must be integers")
        cdef Py_ssize_t i = key
        if i < 0: i += self.length
        if i < 0 or i >= self.length:
            raise IndexError("Index out of range")
        cdef list result = []
        cdef int code
        for j in range(self.nlevels):
            code = (<IntColumn>self.codes[j]).data_c[i]
            result.append(self.levels[j][code])
        return tuple(result) if self.nlevels > 1 else result[0]

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def get_loc(self, object key):
        cdef tuple target = key if isinstance(key, tuple) else (key,)
        cdef Py_ssize_t target_len = len(target)
        if target_len > self.nlevels: raise KeyError(key)

        if self._level_maps is None:
            self._build_level_maps()

        cdef int* target_buf = <int*>aligned_malloc(target_len * sizeof(int))
        if not target_buf: raise MemoryError()

        cdef int level_idx
        cdef object val
        cdef dict level_map
        cdef list matches = []
        cdef Py_ssize_t i, j
        cdef bint match
        cdef int** ptrs
        cdef bytes target_b_key

        try:
            for level_idx in range(target_len):
                level_map = self._level_maps[level_idx]
                val = target[level_idx]
                if val not in level_map:
                    raise KeyError(val)
                target_buf[level_idx] = level_map[val]

            if target_len < self.nlevels:
                ptrs = <int**>aligned_malloc(target_len * sizeof(int*))
                if not ptrs: raise MemoryError()
                try:
                    for j in range(target_len):
                        ptrs[j] = (<IntColumn>self.codes[j]).data_c

                    # C语言级裸奔：内存吞吐极限
                    for i in range(self.length):
                        match = True
                        for j in range(target_len):
                            if ptrs[j][i] != target_buf[j]:
                                match = False
                                break
                        if match:
                            matches.append(i)
                finally:
                    os_aligned_free(ptrs)

                if not matches: raise KeyError(key)
                return matches[0] if len(matches) == 1 else matches

            if self._hash_cache is None:
                self._build_hash_cache()

            target_b_key = (<char*>target_buf)[:target_len * sizeof(int)]
            matches = self._hash_cache.get(target_b_key)

            if matches is None:
                raise KeyError(key)

            return matches[0] if len(matches) == 1 else matches
        finally:
            os_aligned_free(target_buf)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def values(self):
        cdef list result = []
        cdef Py_ssize_t i, j
        cdef int code_val
        cdef IntColumn col
        for i in range(self.length):
            row = []
            for j in range(self.nlevels):
                col = <IntColumn>self.codes[j]
                code_val = col.data_c[i]
                if code_val < 0 or code_val >= len(self.levels[j]):
                    row.append(None)
                else:
                    row.append(self.levels[j][code_val])
            result.append(tuple(row))
        return result

    cdef IndexBase _concat(self, IndexBase other):
        cdef MultiIndex o
        cdef Py_ssize_t total, i, j
        cdef list new_codes
        cdef IntColumn col, new_col, o_col
        cdef int* src1
        cdef int* src2
        cdef int* dst
        if not isinstance(other, MultiIndex):
            raise TypeError("Cannot concat MultiIndex with non-MultiIndex")
        o = <MultiIndex>other
        if self.nlevels != o.nlevels:
            raise ValueError("MultiIndex nlevels mismatch")
        total = self.length + o.length
        new_codes = []
        for j in range(self.nlevels):
            col = <IntColumn>self.codes[j]
            o_col = <IntColumn>o.codes[j]
            new_col = IntColumn(total)
            new_col.length = total
            src1 = col.data_c
            src2 = o_col.data_c
            dst = new_col.data_c
            with nogil:
                for i in range(self.length):
                    dst[i] = src1[i]
                for i in range(o.length):
                    dst[self.length + i] = src2[i]
            new_codes.append(new_col)
        return MultiIndex(self.levels, new_codes, self.names)


cdef class StringIndex(IndexBase):
    """字符串标签索引（用于聚合结果 Series：索引为列名，等长于值列）。"""
    def __cinit__(self, list labels):
        self.labels = labels
        self.length = len(labels)
        self.index_type = INDEX_STRING

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def values(self):
        return list(self.labels)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __getitem__(self, object key):
        if not isinstance(key, int):
            raise TypeError("Index keys must be integers")
        cdef Py_ssize_t i = key
        if i < 0: i += self.length
        if i < 0 or i >= self.length:
            raise IndexError("Index out of range")
        return self.labels[i]

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def get_loc(self, object key):
        cdef Py_ssize_t i
        cdef object label
        for i in range(self.length):
            label = self.labels[i]
            if label == key:
                return i
        raise KeyError(key)

    cdef IndexBase _take(self, Py_ssize_t* indices, Py_ssize_t n):
        cdef list new_labels = []
        cdef Py_ssize_t i
        for i in range(n):
            new_labels.append(self.labels[indices[i]])
        return StringIndex(new_labels)

    cdef IndexBase _concat(self, IndexBase other):
        if not isinstance(other, StringIndex):
            raise TypeError("Cannot concat StringIndex with non-StringIndex")
        cdef StringIndex o = <StringIndex>other
        return StringIndex(self.labels + o.labels)


# ============================================================
#  索引器（_ILocIndexer / _LocIndexer）
# ============================================================


cdef class _ILocIndexer:
    def __cinit__(self, DataFrame frame):
        self.frame = frame

    def __getitem__(self, key):
        cdef Py_ssize_t n_orig = self.frame.df_length
        if n_orig == 0:
            raise IndexError("Cannot index empty DataFrame")

        cdef Py_ssize_t* indices_c = NULL
        cdef Py_ssize_t n_new = 0
        cdef Py_ssize_t i, idx
        cdef DataFrame result
        cdef Py_ssize_t nsub, rstart, rstop, rstep, ri, r, k
        cdef list cnames
        cdef str cname
        cdef BaseColumn colsel
        cdef Py_ssize_t* sidx = NULL

        # 解析用户输入的类型
        if isinstance(key, int):
            # 获取单行 (df.iloc[5])
            idx = <Py_ssize_t>key
            if idx < 0: idx += n_orig
            if idx < 0 or idx >= n_orig: raise IndexError("Index out of bounds")

            indices_c = <Py_ssize_t*>aligned_malloc(sizeof(Py_ssize_t))
            if not indices_c: raise MemoryError()
            indices_c[0] = idx
            n_new = 1

        elif isinstance(key, slice):
            # 获取切片 (df.iloc[2:10])
            start, stop, step = key.indices(n_orig)
            n_new = len(range(start, stop, step))
            if n_new <= 0:
                indices_c = <Py_ssize_t*>aligned_malloc(sizeof(Py_ssize_t))
                try:
                    return self.frame._take(indices_c, 0)
                finally:
                    os_aligned_free(indices_c)

            indices_c = <Py_ssize_t*>aligned_malloc(n_new * sizeof(Py_ssize_t))
            if not indices_c: raise MemoryError()

            i = 0
            for idx in range(start, stop, step):
                indices_c[i] = idx
                i += 1

        elif isinstance(key, list):
            # 获取列表 (df.iloc[[0, 2, 4]])
            n_new = len(key)
            if n_new == 0:
                indices_c = <Py_ssize_t*>aligned_malloc(sizeof(Py_ssize_t))
                try:
                    return self.frame._take(indices_c, 0)
                finally:
                    os_aligned_free(indices_c)

            indices_c = <Py_ssize_t*>aligned_malloc(n_new * sizeof(Py_ssize_t))
            if not indices_c: raise MemoryError()

            try:
                for i in range(n_new):
                    idx = <Py_ssize_t>key[i]
                    if idx < 0: idx += n_orig
                    if idx < 0 or idx >= n_orig: raise IndexError("Index out of bounds")
                    indices_c[i] = idx
            except Exception:
                os_aligned_free(indices_c)
                raise
        elif isinstance(key, tuple):
            cnames = list(self.frame._columns.keys())
            if len(key) != 2:
                raise TypeError("iloc tuple index must have 2 elements (rows, cols)")
            row_key, col_key = key
            if isinstance(col_key, str):
                cname = col_key
            elif isinstance(col_key, int):
                if col_key < 0:
                    col_key = col_key + len(cnames)
                if col_key < 0 or col_key >= len(cnames):
                    raise IndexError("Column index out of range")
                cname = cnames[col_key]
            else:
                raise TypeError("iloc column selector must be str or int")
            if cname not in self.frame._columns:
                raise KeyError(f"Column '{cname}' not found")
            colsel = self.frame._columns[cname]
            if isinstance(row_key, slice):
                rstart, rstop, rstep = row_key.indices(n_orig)
                nsub = len(range(rstart, rstop, rstep))
            elif isinstance(row_key, int):
                ri = <Py_ssize_t>row_key
                if ri < 0: ri += n_orig
                if ri < 0 or ri >= n_orig: raise IndexError("Index out of bounds")
                nsub = 1
                rstart, rstop, rstep = ri, ri + 1, 1
            elif isinstance(row_key, list):
                nsub = len(row_key)
            else:
                raise TypeError("iloc row selector must be int, slice, or list")

            if nsub <= 0:
                sub = _col_take(colsel, <Py_ssize_t*>NULL, 0, False)
                return Series(cname, sub, self.frame.index._take(<Py_ssize_t*>NULL, 0))

            sidx = <Py_ssize_t*>aligned_malloc(nsub * sizeof(Py_ssize_t))
            if not sidx: raise MemoryError()
            try:
                if isinstance(row_key, slice):
                    k = 0
                    for r in range(rstart, rstop, rstep):
                        sidx[k] = r
                        k += 1
                elif isinstance(row_key, int):
                    sidx[0] = rstart
                else:
                    k = 0
                    for r in row_key:
                        r = <Py_ssize_t>r
                        if r < 0: r += n_orig
                        if r < 0 or r >= n_orig: raise IndexError("Index out of bounds")
                        sidx[k] = r
                        k += 1
                sub = _col_take(colsel, sidx, nsub, False)
                return Series(cname, sub, self.frame.index._take(sidx, nsub))
            finally:
                os_aligned_free(sidx)

        else:
            raise TypeError("iloc requires integer, slice, list of integers, or (rows, cols) tuple")

        # 安全地调用底层 C 提取方法
        try:
            result = self.frame._take(indices_c, n_new)
            return result
        finally:
            os_aligned_free(indices_c)


cdef class _LocIndexer:

    def __cinit__(self, DataFrame frame):
        self.frame = frame

    def __setitem__(self, key, object values):
        cdef Py_ssize_t n_orig = self.frame.df_length
        if n_orig == 0:
            raise IndexError("Cannot setitem on empty DataFrame")

        cdef Py_ssize_t idx, i
        cdef list col_names = list(self.frame._columns.keys())
        cdef Py_ssize_t n_cols = len(col_names)
        cdef str col_name
        cdef BaseColumn col

        if isinstance(key, int):
            idx = <Py_ssize_t>key
            if idx < 0: idx += n_orig
            if idx < 0 or idx >= n_orig:
                raise IndexError("Index out of bounds")

            if isinstance(values, list):
                if len(values) != n_cols:
                    raise ValueError(f"Length of values {len(values)} != number of columns {n_cols}")
                for i in range(n_cols):
                    col_name = col_names[i]
                    col = self.frame._columns[col_name]
                    _fast_set_cell(col, col.dtype, idx, values[i])
            elif isinstance(values, dict):
                for col_name, val in values.items():
                    if col_name not in self.frame._columns:
                        raise KeyError(f"Column '{col_name}' not found")
                    col = self.frame._columns[col_name]
                    _fast_set_cell(col, col.dtype, idx, val)
            else:
                raise TypeError(f"Cannot set row from {type(values)}")

        else:
            raise TypeError(f"Unsupported loc key type for setitem: {type(key).__name__}")

    def __getitem__(self, key):
        cdef Py_ssize_t n_orig = self.frame.df_length
        cdef object loc_res
        cdef DataFrame row_result
        cdef DataFrame result
        cdef list col_names
        cdef str col_name

        # Case 0: tuple (rows, cols) — df.loc[mask, ['name', 'age']]
        if isinstance(key, tuple) and len(key) == 2:
            row_key, col_key = key
            row_result = self._get_rows(row_key)
            if isinstance(col_key, list):
                result = DataFrame()
                result.df_length = row_result.df_length
                result.index = row_result.index
                for col_name in col_key:
                    if col_name not in row_result._columns:
                        raise KeyError(f"Column '{col_name}' not found")
                    result._columns[col_name] = row_result._columns[col_name]
                return result
            elif isinstance(col_key, str):
                if col_key not in row_result._columns:
                    raise KeyError(f"Column '{col_key}' not found")
                return Series(col_key, row_result._columns[col_key], row_result.index)
            raise TypeError(f"Unsupported column key type: {type(col_key).__name__}")

        return self._get_rows(key)

    cdef DataFrame _get_rows(self, key):
        cdef Py_ssize_t n_orig = self.frame.df_length
        cdef object loc_res
        cdef BaseColumn key_col

        # Callable filter — df.loc[lambda df: df.col > 5]
        if callable(key):
            mask = key(self.frame)
            return self._apply_mask(mask)

        # Series 布尔掩码（封装了 IntColumn 0/1 或 BoolColumn 掩码列）
        if isinstance(key, Series):
            key_col = (<Series>key)._col
            if key_col.dtype != TYPE_BOOL and key_col.dtype != TYPE_INT:
                raise TypeError(
                    "DataFrame mask must be a boolean Series (compare a column, e.g. df['a'] > 5)")
            return self._apply_mask(key)

        # 裸掩码列 — df.loc[mask_int_col] / df.loc[mask_bool_col]
        if isinstance(key, IntColumn):
            return self._apply_mask(key)
        if isinstance(key, BoolColumn):
            return self._apply_mask(key)

        # list of booleans — df.loc[[True, False, True]]
        if isinstance(key, list):
            return self._apply_bool_list(key)

        # tuple key — index-based lookup via index.get_loc()
        if self.frame.index is not None:
            loc_res = self.frame.index.get_loc(key)
            return self.frame.iloc[loc_res]

        raise TypeError(f"Unsupported loc key type: {type(key).__name__}")

    cdef DataFrame _apply_mask(self, object mask):
        cdef BaseColumn mask_col
        cdef Py_ssize_t n_orig = self.frame.df_length
        if isinstance(mask, Series):
            mask_col = (<Series>mask)._col
        elif isinstance(mask, IntColumn):
            mask_col = <BaseColumn>mask
        elif isinstance(mask, BoolColumn):
            mask_col = <BaseColumn>mask
        else:
            raise TypeError("Mask must be a boolean Series, IntColumn (0/1), or BoolColumn")

        if mask_col.length != n_orig:
            raise ValueError(f"Mask length {mask_col.length} != DataFrame length {n_orig}")
        if mask_col.dtype != TYPE_INT and mask_col.dtype != TYPE_BOOL:
            raise TypeError("Mask column must be int (0/1) or bool")

        cdef Py_ssize_t keep_count = 0
        cdef Py_ssize_t i
        cdef int* int_ptr
        cdef unsigned char* bool_ptr
        if mask_col.dtype == TYPE_INT:
            int_ptr = (<IntColumn>mask_col).data_c
            for i in range(n_orig):
                if int_ptr[i] != 0:
                    keep_count += 1
        else:
            bool_ptr = (<BoolColumn>mask_col).data_c
            for i in range(n_orig):
                if bool_ptr[i] != 0:
                    keep_count += 1

        if keep_count == 0:
            result = DataFrame()
            result.df_length = 0
            for col_name, col_obj in self.frame._columns.items():
                result._columns[col_name] = col_obj.copy()
                result._columns[col_name].length = 0
            return result

        cdef Py_ssize_t* keep_idx = <Py_ssize_t*>aligned_malloc(keep_count * sizeof(Py_ssize_t))
        if not keep_idx: raise MemoryError()
        cdef Py_ssize_t j = 0
        if mask_col.dtype == TYPE_INT:
            for i in range(n_orig):
                if int_ptr[i] != 0:
                    keep_idx[j] = i
                    j += 1
        else:
            for i in range(n_orig):
                if bool_ptr[i] != 0:
                    keep_idx[j] = i
                    j += 1

        try:
            return self.frame._take(keep_idx, keep_count)
        finally:
            os_aligned_free(keep_idx)

    cdef DataFrame _apply_bool_list(self, list mask):
        cdef Py_ssize_t n_orig = self.frame.df_length
        if len(mask) != n_orig:
            raise ValueError(f"Mask length {len(mask)} != DataFrame length {n_orig}")

        cdef Py_ssize_t keep_count = 0
        cdef Py_ssize_t i
        for i in range(n_orig):
            if mask[i]:
                keep_count += 1

        if keep_count == 0:
            result = DataFrame()
            result.df_length = 0
            for col_name, col_obj in self.frame._columns.items():
                result._columns[col_name] = col_obj.copy()
                result._columns[col_name].length = 0
            return result

        cdef Py_ssize_t* keep_idx = <Py_ssize_t*>aligned_malloc(keep_count * sizeof(Py_ssize_t))
        if not keep_idx: raise MemoryError()
        cdef Py_ssize_t j = 0
        for i in range(n_orig):
            if mask[i]:
                keep_idx[j] = i
                j += 1

        try:
            return self.frame._take(keep_idx, keep_count)
        finally:
            os_aligned_free(keep_idx)


# ============================================================
#  Series 访问器（.str / .dt / .cat）
# ============================================================
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
    cdef object _wrap(self, object res):
        """访问器结果若是单列 → 包成随身 Series 视图；否则原样返回。"""
        if isinstance(res, BaseColumn):
            return Series(self._series.name, <BaseColumn>res, self._series._index)
        return res

    # ==================== Search / Match ====================

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def contains(self, object pat):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).str_contains_mask(pat))
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).contains(pat))
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def startswith(self, prefix):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).startswith(prefix))
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).startswith(prefix))
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def endswith(self, suffix):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).endswith(suffix))
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).endswith(suffix))
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def find(self, str sub):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).find(sub))
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).find(sub))
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def rfind(self, str sub):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).rfind(sub))
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).rfind(sub))
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def index(self, str sub):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).find(sub))
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).find(sub))
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def rindex(self, str sub):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).rindex(sub))
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).rindex(sub))
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def count(self, str sub):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).count(sub))
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).count(sub))
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def match(self, str pattern):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).match(pattern))
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).match(pattern))
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def fullmatch(self, str pattern):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).fullmatch(pattern))
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).fullmatch(pattern))
        raise TypeError("String accessor requires string column")

    # ==================== Extract / Findall ====================

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def extract(self, str pattern, int group=1):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).extract(pattern))
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).extract(pattern, group))
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def extractall(self, str pattern):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).extractall(pattern))
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).extractall(pattern))
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def findall(self, str pattern):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).findall(pattern))
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).findall(pattern))
        raise TypeError("String accessor requires string column")

    # ==================== Type Check ====================

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isalpha(self):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).isalpha())
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).isalpha())
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isnumeric(self):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).isnumeric())
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).isnumeric())
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isalnum(self):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).isalnum())
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).isalnum())
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isdigit(self):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).isdigit())
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).isdigit())
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isdecimal(self):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).isdecimal())
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).isdecimal())
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def islower(self):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).islower())
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).islower())
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isupper(self):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).isupper())
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).isupper())
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isspace(self):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).isspace())
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).isspace())
        raise TypeError("String accessor requires string column")

    # ==================== Case Conversion ====================

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def upper(self):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).upper())
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).upper())
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def lower(self):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).lower())
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).lower())
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def title(self):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).title())
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).title())
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def capitalize(self):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).capitalize())
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).capitalize())
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def swapcase(self):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).swapcase())
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).swapcase())
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def casefold(self):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).casefold())
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).casefold())
        raise TypeError("String accessor requires string column")

    # ==================== Strip / Pad ====================

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def strip(self, chars=None):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).strip(chars) if chars is not None else (<AsciiColumn>col).strip())
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).strip(chars) if chars is not None else (<UCS2Column>col).strip())
        raise TypeError("String accessor requires string column")

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def lstrip(self, chars=None):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).lstrip(chars) if chars is not None else (<AsciiColumn>col).lstrip())
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).lstrip(chars) if chars is not None else (<UCS2Column>col).lstrip())
        raise TypeError("String accessor requires string column")

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def rstrip(self, chars=None):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).rstrip(chars) if chars is not None else (<AsciiColumn>col).rstrip())
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).rstrip(chars) if chars is not None else (<UCS2Column>col).rstrip())
        raise TypeError("String accessor requires string column")

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def center(self, int width, str fillchar=' '):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).center(width, fillchar))
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).center(width, fillchar))
        raise TypeError("String accessor requires string column")

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def ljust(self, int width, str fillchar=' '):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).ljust(width, fillchar))
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).ljust(width, fillchar))
        raise TypeError("String accessor requires string column")

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def rjust(self, int width, str fillchar=' '):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).rjust(width, fillchar))
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).rjust(width, fillchar))
        raise TypeError("String accessor requires string column")

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def pad(self, int width, str side='left', str fillchar=' '):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).pad(width, side, fillchar))
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).pad(width, side, fillchar))
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def zfill(self, int width):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).zfill(width))
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).zfill(width))
        raise TypeError("String accessor requires string column")

    # ==================== Replace ====================

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def replace(self, str old, str new, int count=-1):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).replace(old, new, count))
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).replace(old, new, count))
        raise TypeError("String accessor requires string column")

    # ==================== Split ====================

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def split(self, str sep=None, int maxsplit=-1):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).str_split(sep, maxsplit))
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).str_split(sep, maxsplit))
        raise TypeError("String accessor requires string column")

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def rsplit(self, str sep=None, int maxsplit=-1):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).rsplit(sep, maxsplit))
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).rsplit(sep, maxsplit))
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def partition(self, str sep):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).partition(sep))
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).partition(sep))
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def rpartition(self, str sep):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).rpartition(sep))
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).rpartition(sep))
        raise TypeError("String accessor requires string column")

    # ==================== Slicing / Access ====================

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def get(self, int index, str default=''):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).get(index, default))
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).get(index, default))
        raise TypeError("String accessor requires string column")

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def slice(self, int start=0, int stop=-1, int step=1):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).slice(start, stop, step))
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).slice(start, stop, step))
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def len(self):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).len())
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).len())
        raise TypeError("String accessor requires string column")

    def __getitem__(self, key):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).str_getitem(key))
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).str_getitem(key))
        raise TypeError("String accessor requires string column")

    # ==================== Concat ====================

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def cat(self, other, str sep=''):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).cat(other, sep))
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).cat(other, sep))
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def prefix(self, str pre):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).prefix(pre))
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).prefix(pre))
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def suffix(self, str suf):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).suffix(suf))
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).suffix(suf))
        raise TypeError("String accessor requires string column")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def repeat(self, int repeats):
        cdef BaseColumn col = self._series._col
        if col.dtype == TYPE_ASCII:
            return self._wrap((<AsciiColumn>col).repeat(repeats))
        elif col.dtype == TYPE_UCS2:
            return self._wrap((<UCS2Column>col).repeat(repeats))
        raise TypeError("String accessor requires string column")

@cython.final
cdef class _DateTimeAccessor:
    """pandas 风格 .dt 访问器。
    由 Series.dt property 懒加载缓存；属性将列上的零参日期访问器
    （year()/month()/...）转为返回值列的形式。
    """

    def __cinit__(self, Series series):
        self._series = series

    def __getattr__(self, object name):
        cdef object fn
        cdef BaseColumn col = self._series._col
        if name.startswith('_'):
            raise AttributeError(f"'DateTimeAccessor' object has no attribute '{name}'")
        if name not in _DT_ATTRS:
            raise AttributeError(f"'DateTimeAccessor' object has no attribute '{name}'")
        fn = getattr(col, name, None)
        if fn is None or not callable(fn):
            raise AttributeError(
                f"`.dt.{name}` not supported for column type {type(col).__name__}")
        return _wrap_series(self._series.name, fn(), self._series._index)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def weekday(self):
        return _wrap_series(self._series.name, self._series._col.day_of_week(), self._series._index)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def dayofweek(self):
        return _wrap_series(self._series.name, self._series._col.day_of_week(), self._series._index)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def dayofyear(self):
        return _wrap_series(self._series.name, self._series._col.day_of_year(), self._series._index)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def weekofyear(self):
        return _wrap_series(self._series.name, self._series._col.week_of_year(), self._series._index)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def strftime(self, str fmt):
        return _wrap_series(self._series.name, getattr(self._series._col, 'strftime')(fmt), self._series._index)

# ============================================================
#  Category Accessor for Series
#  pandas-style .cat accessor for categorical columns
# ============================================================

@cython.final
cdef class _CategoryAccessor:
    """pandas 风格 .cat 访问器。
    由 Series.cat property 懒加载缓存。
    """

    def __cinit__(self, Series series):
        self._series = series

    @property
    def codes(self):
        """返回底层的整数编码 (IntColumn)。使用底层内存拷贝，速度极快。"""
        cdef CategoryColumn col = <CategoryColumn>self._series._col
        cdef IntColumn res = IntColumn(max(col.length, 1))
        res.length = col.length
        if col.length > 0:
            with nogil:
                memcpy(res.data_c, col.codes_c, col.length * sizeof(int))
        return res

    @property
    def categories(self):
        """返回当前所有的不重复类别列表。"""
        cdef CategoryColumn col = <CategoryColumn>self._series._col
        return list(col.categories)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def rename_categories(self, mapper, bint inplace=False):
        cdef CategoryColumn col = <CategoryColumn>self._series._col
        cdef CategoryColumn target = col if inplace else col.copy()
        cdef list new_cats = list(target.categories)
        cdef int i

        if isinstance(mapper, dict):
            for i in range(len(new_cats)):
                if new_cats[i] in mapper:
                    new_cats[i] = str(mapper[new_cats[i]])
        elif isinstance(mapper, list):
            if len(mapper) != len(new_cats):
                raise ValueError(f"新类别列表的长度 ({len(mapper)}) 必须等于原类别长度 ({len(new_cats)})")
            for i in range(len(mapper)):
                new_cats[i] = str(mapper[i])
        else:
            raise TypeError("mapper 必须是字典或列表")

        if len(new_cats) != len(set(new_cats)):
            raise ValueError("rename_categories 不能产生重复的类别")

        target.categories = new_cats
        target._cat_to_code = {cat: idx for idx, cat in enumerate(new_cats)}

        if not inplace:
            return target

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def add_categories(self, list new_categories, bint inplace=False):
        cdef CategoryColumn col = <CategoryColumn>self._series._col
        cdef CategoryColumn target = col if inplace else col.copy()
        cdef str c, c_str

        for c in new_categories:
            c_str = str(c)
            if c_str not in target._cat_to_code:
                target._cat_to_code[c_str] = len(target.categories)
                target.categories.append(c_str)

        if not inplace:
            return target

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def remove_categories(self, list removals, bint inplace=False):
        cdef set rem_set = set([str(x) for x in removals])
        cdef CategoryColumn col = <CategoryColumn>self._series._col
        cdef list new_cats = [c for c in col.categories if c not in rem_set]
        return self.set_categories(new_cats, inplace=inplace)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def set_categories(self, list new_categories, bint inplace=False):
        cdef CategoryColumn col = <CategoryColumn>self._series._col
        cdef CategoryColumn target = col if inplace else col.copy()
        cdef dict final_cat_to_code = {}
        cdef list final_categories = []
        cdef str c, c_str
        cdef int n_new = 0

        for c in new_categories:
            c_str = str(c)
            if c_str not in final_cat_to_code:
                final_cat_to_code[c_str] = n_new
                final_categories.append(c_str)
                n_new += 1

        if '' not in final_cat_to_code:
            final_cat_to_code[''] = n_new
            final_categories.append('')
            n_new += 1

        cdef int missing_code = final_cat_to_code['']
        cdef int old_len = len(target.categories)
        cdef int* code_map = <int*>aligned_malloc(max(old_len, 1) * sizeof(int))
        if not code_map: raise MemoryError()

        cdef int old_code
        cdef Py_ssize_t i
        cdef str old_cat
        try:
            for old_code in range(old_len):
                old_cat = target.categories[old_code]
                code_map[old_code] = final_cat_to_code.get(old_cat, missing_code)

            with nogil:
                for i in range(target.length):
                    target.codes_c[i] = code_map[target.codes_c[i]]
        finally:
            os_aligned_free(code_map)

        target.categories = final_categories
        target._cat_to_code = final_cat_to_code

        if not inplace:
            return target

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def as_ordered(self, bint inplace=False):
        cdef CategoryColumn col = <CategoryColumn>self._series._col
        cdef CategoryColumn target = col if inplace else col.copy()
        if not inplace:
            return target

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def as_unordered(self, bint inplace=False):
        cdef CategoryColumn col = <CategoryColumn>self._series._col
        cdef CategoryColumn target = col if inplace else col.copy()
        if not inplace:
            return target


# ============================================================
#  Series 类
# ============================================================
@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class Series:
    """轻量级 Series 实现，单列代理，知道自己的名字（name），绑定了一个索引（index）。"""

    def __cinit__(self, str name, BaseColumn col, IndexBase index):
        self.name = name
        self._col = col
        self._index = index

    # =================================================================
    #  Properties
    # =================================================================
    
    @property
    def dtype(self):
        """返回底层列类型"""
        return self._col.dtype
    
    @property
    def is_2d(self):
        """返回是否2D布局"""
        return self._col.is_2d
    
    @property
    def length(self):
        """返回Series长度（逻辑行数），向后兼容"""
        if self._col.is_2d and self._col.num_cols > 1:
            return self._col._get_rows_c()
        return self._col.length
    
    @property
    def index(self):
        """返回索引对象"""
        return self._index
    
    @index.setter
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def index(self, value):
        self._index = value
    
    @property
    def values(self):
        """返回底层数据的Python列表（2D 列返回行键元组，与 _col_to_keys/value_counts 一致）"""
        return _col_to_keys(self._col)

    @property
    def shape(self):
        """Series 形状，对标 pandas：单列 (len,)"""
        return (len(self),)

    # =================================================================
    #  迭代 / 包含 / 属性透传
    # =================================================================

    def __iter__(self):
        for i in range(len(self)):
            yield _col_row_key(self._col, i)

    def __contains__(self, val):
        for i in range(len(self)):
            if _col_row_key(self._col, i) == val:
                return True
        return False

    def __getattr__(self, name):
        """透传列级方法/属性的临时包装视图：
        - 调用列方法且返回 BaseColumn → 包成 Series（支持 s.col.startswith().len() 链）
        - 返回标量/列表等非列 → 原样返回（如 .cols、.rows、.is_2d）
        """
        cdef BaseColumn col = self._col
        if col is None or (name.startswith('__') and name.endswith('__')):
            raise AttributeError(f"'Series' object has no attribute '{name}'")
        attr = getattr(col, name, None)
        if callable(attr):
            @cython.binding(False)
            @cython.always_allow_keywords(False)
            def _forward(*args, **kwargs):
                cdef BaseColumn rcol
                cdef Py_ssize_t rlen
                res = attr(*args, **kwargs)
                if isinstance(res, BaseColumn):
                    rcol = <BaseColumn>res
                    if rcol.is_2d and rcol.num_cols > 1:
                        rlen = rcol._get_rows_c()
                    else:
                        rlen = rcol.length
                    if self._index is not None and (<IndexBase>self._index).length == rlen:
                        return Series(self.name, rcol, self._index)
                    return Series(self.name, rcol, RangeIndex(0, rlen))
                return res
            return _forward
        if attr is None:
            raise AttributeError(f"'Series' object has no attribute '{name}'")
        return attr

    # =================================================================
    #  基本信息
    # =================================================================
    
    def __len__(self):
        """返回Series长度（逻辑行数）"""
        if self._col.is_2d and self._col.num_cols > 1:
            return self._col._get_rows_c()
        return self._col.length
    
    def __repr__(self):
        """字符串表示"""
        n = len(self)
        max_display = 30
        lines = []
        lines.append(f"Series(name='{self.name}')")
        lines.append(f"dtype: {self.dtype}")
        lines.append(f"length: {n}")
        
        if n <= max_display:
            for i in range(n):
                idx_val = self._index[i] if self._index else i
                lines.append(f"  {idx_val}: {_col_row_key(self._col, i)}")
        else:
            for i in range(10):
                idx_val = self._index[i] if self._index else i
                lines.append(f"  {idx_val}: {_col_row_key(self._col, i)}")
            lines.append(f"  ... ({n - 20} rows omitted)")
            for i in range(n - 10, n):
                idx_val = self._index[i] if self._index else i
                lines.append(f"  {idx_val}: {_col_row_key(self._col, i)}")
        
        return "\n".join(lines)

    def __str__(self):
        return self.__repr__()

    # =================================================================
    #  标量提取与切片
    # =================================================================
    
    def __getitem__(self, key):
        """支持切片、标量、布尔掩码（视图过滤）"""
        cdef Py_ssize_t n = len(self)
        cdef BaseColumn m
        cdef list keep
        cdef Py_ssize_t i, k, total
        cdef Py_ssize_t* keep_c

        # 布尔掩码筛选（Series 封装 IntColumn 0/1 或 BoolColumn）
        if isinstance(key, Series):
            m = (<Series>key)._col
            if m.dtype != TYPE_BOOL and m.dtype != TYPE_INT:
                raise TypeError("Series boolean mask required")
            if m.length != n:
                raise ValueError(f"Boolean Series length {m.length} != Series length {n}")
            keep = []
            for i in range(n):
                if m[i]:
                    keep.append(i)
            k = len(keep)
            if k == 0:
                new_col = self._col.copy()
                new_col.length = 0
                return Series(self.name, new_col, self._index)
            keep_c = <Py_ssize_t*>aligned_malloc(k * sizeof(Py_ssize_t))
            if not keep_c: raise MemoryError()
            try:
                for i in range(k):
                    keep_c[i] = keep[i]
                new_col = _col_take(self._col, keep_c, k, False)
                new_index = self._index._take(keep_c, k)
                return Series(self.name, new_col, new_index)
            finally:
                os_aligned_free(keep_c)

        # 整数索引（逻辑行；2D 列返回行键元组，与 _col_to_keys/factorize/value_counts 一致）
        if isinstance(key, int):
            total = len(self)
            if key < 0:
                key = total + key
            if key < 0 or key >= total:
                raise IndexError("index out of range")
            return _col_row_key(self._col, key)

        # 切片
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            indices = list(range(start, stop, step))
            k = len(indices)
            if k == 0:
                new_col = self._col.copy()
                new_col.length = 0
                return Series(self.name, new_col, self._index)
            keep_c = <Py_ssize_t*>aligned_malloc(k * sizeof(Py_ssize_t))
            if not keep_c: raise MemoryError()
            try:
                for i in range(k):
                    keep_c[i] = indices[i]
                new_col = _col_take(self._col, keep_c, k, False)
                new_index = self._index._take(keep_c, k)
                return Series(self.name, new_col, new_index)
            finally:
                os_aligned_free(keep_c)

        # 字符串标签：索引为 StringIndex（聚合结果 Series，标签为列名）时按标签查找
        if isinstance(key, str):
            if self._index is None or self._index.index_type != INDEX_STRING:
                raise KeyError(key)
            try:
                pos = self._index.get_loc(key)
            except KeyError:
                raise KeyError(key)
            if isinstance(pos, list):
                pos = pos[0]
            return self._col[pos]

        raise TypeError(f"invalid index type: {type(key)}")

    def __setitem__(self, key, value):
        """标量赋值（2D 列禁止行级赋值，避免只写子列 0 的静默错误）"""
        if isinstance(key, int):
            if self._col.is_2d and self._col.num_cols > 1:
                raise ValueError("Cannot setitem on a 2D Series row; assign on the underlying 2D column")
            if key < 0:
                key = self._col.length + key
            if key < 0 or key >= self._col.length:
                raise IndexError("index out of range")
            self._col[key] = value
        else:
            raise TypeError(f"invalid index type: {type(key)}")

    # =================================================================
    #  算术运算
    # =================================================================
    
    def __add__(self, other):
        if isinstance(other, Series):
            if self._index is (<Series>other)._index:
                new_col = self._col + (<Series>other)._col
            else:
                aligned_other = other.reindex(self._index)
                new_col = self._col + aligned_other._col
            return Series(self.name, new_col, self._index)
        else:
            new_col = self._col + other
            return Series(self.name, new_col, self._index)

    def __sub__(self, other):
        if isinstance(other, Series):
            if self._index is (<Series>other)._index:
                new_col = self._col - (<Series>other)._col
            else:
                aligned_other = other.reindex(self._index)
                new_col = self._col - aligned_other._col
            return Series(self.name, new_col, self._index)
        else:
            new_col = self._col - other
            return Series(self.name, new_col, self._index)

    def __mul__(self, other):
        if isinstance(other, Series):
            if self._index is (<Series>other)._index:
                new_col = self._col * (<Series>other)._col
            else:
                aligned_other = other.reindex(self._index)
                new_col = self._col * aligned_other._col
            return Series(self.name, new_col, self._index)
        else:
            new_col = self._col * other
            return Series(self.name, new_col, self._index)

    def __truediv__(self, other):
        if isinstance(other, Series):
            if self._index is (<Series>other)._index:
                new_col = self._col / (<Series>other)._col
            else:
                aligned_other = other.reindex(self._index)
                new_col = self._col / aligned_other._col
            return Series(self.name, new_col, self._index)
        else:
            new_col = self._col / other
            return Series(self.name, new_col, self._index)

    def __floordiv__(self, other):
        if isinstance(other, Series):
            if self._index is (<Series>other)._index:
                new_col = self._col // (<Series>other)._col
            else:
                aligned_other = other.reindex(self._index)
                new_col = self._col // aligned_other._col
            return Series(self.name, new_col, self._index)
        else:
            new_col = self._col // other
            return Series(self.name, new_col, self._index)

    def __mod__(self, other):
        if isinstance(other, Series):
            if self._index is (<Series>other)._index:
                new_col = self._col % (<Series>other)._col
            else:
                aligned_other = other.reindex(self._index)
                new_col = self._col % aligned_other._col
            return Series(self.name, new_col, self._index)
        else:
            new_col = self._col % other
            return Series(self.name, new_col, self._index)

    def __neg__(self):
        return Series(self.name, -self._col, self._index)

    def __pos__(self):
        return self

    def __abs__(self):
        return Series(self.name, self._col.abs(), self._index)

    # =================================================================
    #  比较运算
    # =================================================================
    
    def __eq__(self, other):
        if isinstance(other, Series):
            if self._index is not (<Series>other)._index:
                other = (<Series>other).reindex(self._index)
            res = self._col == (<Series>other)._col
        else:
            res = self._col == other
        return _wrap_series(self.name, res, self._index)

    def __ne__(self, other):
        if isinstance(other, Series):
            if self._index is not (<Series>other)._index:
                other = (<Series>other).reindex(self._index)
            res = self._col != (<Series>other)._col
        else:
            res = self._col != other
        return _wrap_series(self.name, res, self._index)

    def __lt__(self, other):
        if isinstance(other, Series):
            if self._index is not (<Series>other)._index:
                other = (<Series>other).reindex(self._index)
            res = self._col < (<Series>other)._col
        else:
            res = self._col < other
        return _wrap_series(self.name, res, self._index)

    def __le__(self, other):
        if isinstance(other, Series):
            if self._index is not (<Series>other)._index:
                other = (<Series>other).reindex(self._index)
            res = self._col <= (<Series>other)._col
        else:
            res = self._col <= other
        return _wrap_series(self.name, res, self._index)

    def __gt__(self, other):
        if isinstance(other, Series):
            if self._index is not (<Series>other)._index:
                other = (<Series>other).reindex(self._index)
            res = self._col > (<Series>other)._col
        else:
            res = self._col > other
        return _wrap_series(self.name, res, self._index)

    def __ge__(self, other):
        if isinstance(other, Series):
            if self._index is not (<Series>other)._index:
                other = (<Series>other).reindex(self._index)
            res = self._col >= (<Series>other)._col
        else:
            res = self._col >= other
        return _wrap_series(self.name, res, self._index)

    def __and__(self, other):
        if isinstance(other, Series):
            new_col = self._col & (<Series>other)._col
        else:
            new_col = self._col & other
        return Series(self.name, new_col, self._index)

    def __or__(self, other):
        if isinstance(other, Series):
            new_col = self._col | (<Series>other)._col
        else:
            new_col = self._col | other
        return Series(self.name, new_col, self._index)

    def __xor__(self, other):
        if isinstance(other, Series):
            new_col = self._col ^ (<Series>other)._col
        else:
            new_col = self._col ^ other
        return Series(self.name, new_col, self._index)

    def __invert__(self):
        return Series(self.name, ~self._col, self._index)

    # =================================================================
    #  命名算术（pandas 风格：s.add(other) 等价于 s + other）
    # =================================================================

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def add(self, other):
        """加法：Series + other（other 可为 Series/列/标量）"""
        o = (<Series>other)._col if isinstance(other, Series) else other
        if isinstance(o, (IntColumn, DoubleColumn)):
            res = self._col.add(o)
        else:
            res = self._col.add_val(o)
        return Series(self.name, res, self._index)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sub(self, other):
        """减法：Series - other"""
        o = (<Series>other)._col if isinstance(other, Series) else other
        if isinstance(o, (IntColumn, DoubleColumn)):
            res = self._col.sub(o)
        else:
            res = self._col.sub_val(o)
        return Series(self.name, res, self._index)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def mul(self, other):
        """乘法：Series * other"""
        o = (<Series>other)._col if isinstance(other, Series) else other
        if isinstance(o, (IntColumn, DoubleColumn)):
            res = self._col.mul(o)
        else:
            res = self._col.mul_val(o)
        return Series(self.name, res, self._index)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def div(self, other):
        """除法：Series / other"""
        o = (<Series>other)._col if isinstance(other, Series) else other
        if isinstance(o, (IntColumn, DoubleColumn)):
            res = self._col.div(o)
        else:
            res = self._col.div_val(o)
        return Series(self.name, res, self._index)

    # =================================================================
    #  单列统计
    # =================================================================
    
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sum(self):
        """求和"""
        return self._col.sum()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def mean(self):
        """均值"""
        return self._col.mean()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def std(self):
        """标准差"""
        return self._col.std()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def var(self):
        """方差"""
        return self._col.var()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def min(self):
        """最小值"""
        return self._col.min()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def max(self):
        """最大值"""
        return self._col.max()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def median(self):
        """中位数"""
        return self._col.median()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def prod(self):
        """连乘积"""
        return self._col.prod()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def count(self):
        """计数（非空/非NaN）"""
        return self._col.count()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def nunique(self):
        """唯一值数量（2D 列返回每子列的计数列表）"""
        return self._col._nunique()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def unique(self):
        """唯一值"""
        return self._col.unique()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def value_counts(self):
        """值计数，返回 Series：以值为索引（StringIndex）、频次为值，按频次降序。"""
        cdef list keys = _col_to_keys(self._col)
        cdef Py_ssize_t n = len(keys)
        cdef Py_ssize_t i
        cdef object v
        cdef dict counts = {}
        for i in range(n):
            v = keys[i]
            counts[v] = counts.get(v, 0) + 1
        cdef list sorted_items = sorted(counts.items(), key=lambda x: -x[1])
        cdef list key_list = [k for k, v in sorted_items]
        cdef IntColumn cnt_col = IntColumn(len(sorted_items))
        cnt_col.length = len(sorted_items)
        cdef int* cnt_ptr = cnt_col.data_c
        for i in range(len(sorted_items)):
            cnt_ptr[i] = sorted_items[i][1]
        return Series(self.name, cnt_col, StringIndex(key_list))

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def argmax(self):
        """最大值索引"""
        return self._col.idxmax()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def argmin(self):
        """最小值索引"""
        return self._col.idxmin()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def quantile(self, q):
        """分位数"""
        return self._col.quantile(q)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def skew(self):
        """偏度"""
        return self._col.skew()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def kurt(self):
        """峰度"""
        return self._col.kurt()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def mad(self):
        """平均绝对偏差"""
        return self._col.mad()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sem(self):
        """标准误"""
        return self._col.sem()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def mode(self):
        """众数：返回以值为列的 Series（pandas 语义，按 Mode 值升序）"""
        cdef list vals = self._col.mode()
        cdef Py_ssize_t n = len(vals)
        cdef DataFrame tmp = DataFrame()
        cdef BaseColumn col = _column_from_list_impl(tmp, self.name, vals)
        return Series(self.name, col, RangeIndex(0, n))

    # =================================================================
    #  单列变换
    # =================================================================
    
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def fillna(self, value):
        """填充NaN/None值"""
        return Series(self.name, self._col.fillna(value), self._index)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def ffill(self):
        """向前填充"""
        return Series(self.name, self._col.ffill(), self._index)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def bfill(self):
        """向后填充"""
        return Series(self.name, self._col.bfill(), self._index)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def astype(self, dtype):
        """类型转换"""
        return Series(self.name, self._col.astype(dtype), self._index)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def abs(self):
        """绝对值"""
        return Series(self.name, self._col.abs(), self._index)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def clip(self, lower=None, upper=None):
        """截断值"""
        return Series(self.name, self._col.clip(lower, upper), self._index)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def round(self, decimals):
        """四舍五入"""
        return Series(self.name, self._col.round(decimals), self._index)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def diff(self, periods=1):
        """差分"""
        return Series(self.name, self._col.diff(periods), self._index)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pct_change(self, periods=1):
        """百分比变化"""
        return Series(self.name, self._col.pct_change(periods), self._index)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cumsum(self):
        """累积和"""
        return Series(self.name, self._col.cumsum(), self._index)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cumprod(self):
        """累积积"""
        return Series(self.name, self._col.cumprod(), self._index)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cummax(self):
        """累积最大值"""
        return Series(self.name, self._col.cummax(), self._index)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cummin(self):
        """累积最小值"""
        return Series(self.name, self._col.cummin(), self._index)

    # =================================================================
    #  排序与索引
    # =================================================================
    
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sort_values(self, ascending=True):
        """按值排序"""
        return self._col.sort_values(ascending)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def head(self, n=5):
        """前n行"""
        return self._col.head(n)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def tail(self, n=5):
        """后n行"""
        return self._col.tail(n)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def nsmallest(self, n):
        """最小的n个值"""
        return self._col.nsmallest(n)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def nlargest(self, n):
        """最大的n个值"""
        return self._col.nlargest(n)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def idxmax(self):
        """最大值索引"""
        return self._col.idxmax()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def idxmin(self):
        """最小值索引"""
        return self._col.idxmin()

    # =================================================================
    #  布尔判断
    # =================================================================
    
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isna(self):
        """判断是否为NaN/None"""
        return Series(self.name, _series_isna(self._col), self._index)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def notna(self):
        """判断是否非NaN/None"""
        return Series(self.name, _series_notna(self._col), self._index)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isin(self, values):
        """判断是否在值列表中"""
        return Series(self.name, self._col.isin(values), self._index)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def between(self, low, high):
        """判断是否在区间内"""
        return Series(self.name, self._col.between(low, high), self._index)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def all(self):
        """判断是否全部为True"""
        n = len(self)
        if n == 0:
            return True
        for i in range(n):
            if not self._col[i]:
                return False
        return True

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def any(self):
        """判断是否有True"""
        n = len(self)
        for i in range(n):
            if self._col[i]:
                return True
        return False

    # =================================================================
    #  条件替换
    # =================================================================
    
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def where(self, cond, other):
        """条件替换：当cond为True时保留原值，否则替换为other"""
        n = len(self)
        new_vals = [None] * n
        for i in range(n):
            if isinstance(cond, Series):
                c = cond[i]
            else:
                c = cond
            if c:
                new_vals[i] = self._col[i]
            else:
                new_vals[i] = other
        
        from fluidframe.core.dataframe import _column_from_list_impl
        new_col = _column_from_list_impl(None, self.name, new_vals)
        return Series(self.name, new_col, self._index)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def mask(self, cond, other):
        """掩码替换：当cond为True时替换为other，否则保留原值"""
        n = len(self)
        new_vals = [None] * n
        for i in range(n):
            if isinstance(cond, Series):
                c = cond[i]
            else:
                c = cond
            if c:
                new_vals[i] = other
            else:
                new_vals[i] = self._col[i]
        
        from fluidframe.core.dataframe import _column_from_list_impl
        new_col = _column_from_list_impl(None, self.name, new_vals)
        return Series(self.name, new_col, self._index)

    # =================================================================
    #  映射与转换
    # =================================================================
    
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def map(self, func):
        """映射函数"""
        n = len(self)
        new_vals = [None] * n
        for i in range(n):
            new_vals[i] = func(self._col[i])
        
        from fluidframe.core.dataframe import _column_from_list_impl
        new_col = _column_from_list_impl(None, self.name, new_vals)
        return Series(self.name, new_col, self._index)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def factorize(self):
        """分类编码（对标 pandas Series.factorize）：返回
        (codes, uniques)，uniques 按首次出现顺序、codes 与行数等长。"""
        return _series_factorize(self._col)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def apply(self, func):
        """应用函数"""
        return self.map(func)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def replace(self, to_replace, value):
        """替换值"""
        n = len(self)
        new_vals = [None] * n
        for i in range(n):
            val = self._col[i]
            if val == to_replace:
                new_vals[i] = value
            else:
                new_vals[i] = val
        
        from fluidframe.core.dataframe import _column_from_list_impl
        new_col = _column_from_list_impl(None, self.name, new_vals)
        return Series(self.name, new_col, self._index)

    # =================================================================
    #  索引操作
    # =================================================================
    
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def reindex(self, new_index):
        """重新索引"""
        from fluidframe.core.dataframe import _column_from_list_impl
        
        # 创建映射：旧索引值 -> 新索引值的位置
        old_idx_map = {}
        old_len = len(self)
        for i in range(old_len):
            old_idx_map[self._index[i]] = i
        
        new_len = len(new_index)
        new_vals = [None] * new_len
        for i in range(new_len):
            idx_val = new_index[i]
            if idx_val in old_idx_map:
                new_vals[i] = self._col[old_idx_map[idx_val]]
            else:
                new_vals[i] = None  # NaN for missing
        
        new_col = _column_from_list_impl(None, self.name, new_vals)
        return Series(self.name, new_col, new_index)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def reset_index(self):
        """重置索引"""
        from fluidframe.core.index import RangeIndex
        return Series(self.name, self._col, RangeIndex(len(self)))

    # =================================================================
    #  访问器（使用pxi文件中的访问器类）
    # =================================================================
    
    @property
    def str(self):
        """字符串访问器，返回_StringAccessor实例"""
        cdef int dt = self._col.dtype
        if dt != TYPE_ASCII and dt != TYPE_UCS2:
            raise AttributeError("Can only use .str accessor with string values!")
        return _StringAccessor(self)

    @property
    def dt(self):
        """日期时间访问器，返回_DateTimeAccessor实例"""
        cdef int dt = self._col.dtype
        if dt not in (TYPE_DATE, TYPE_DATETIME, TYPE_TIME):
            raise AttributeError("Can only use .dt accessor with datetimelike values!")
        return _DateTimeAccessor(self)

    @property
    def cat(self):
        """分类访问器，返回_CategoryAccessor实例"""
        if self._col.dtype != TYPE_CATEGORY:
            raise AttributeError("Can only use .cat accessor with categorical values!")
        return _CategoryAccessor(self)

    # =================================================================
    #  滚动窗口
    # =================================================================

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def rolling(self, Py_ssize_t window, min_periods=None, center=False, win_type=None):
        """滚动窗口访问器：返回 _RollingWindow（.mean()/.sum()/... 窗口算子）。
        win_type='boxcar'/'triang' 时仅支持 .mean()/.sum()/.var()/.std()（pandas Window 语义）。"""
        if window <= 0:
            raise ValueError(f"Window size must be positive, got {window}")
        return _RollingWindow(self._col, window, min_periods, center, False, self.name, self._index, win_type)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def expanding(self, Py_ssize_t min_periods=1, axis=0):
        """扩展窗口访问器：等价于 rolling(window=行数, min_periods=min_periods)。
        复用 _RollingWindow（window=max(r, mp) 以支持 mp>r 时全 NaN 的 pandas 语义）。"""
        if axis != 0 and axis != 'index':
            raise NotImplementedError("expanding supports axis=0 ('index') only")
        cdef Py_ssize_t r = <Py_ssize_t>self.length
        return _RollingWindow(self._col, max(r, min_periods), min_periods, False, False, self.name, self._index)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def ewm(self, com=None, span=None, halflife=None, alpha=None,
            Py_ssize_t min_periods=0, bint adjust=True, bint ignore_na=False):
        """指数加权窗口访问器（pandas 式）：参数四选一（com/span/halflife/alpha），
        支持 .mean()/.sum()/.var(bias)/.std(bias)/.cov(other)/.corr(other)。"""
        return _EWMWindow(self._col, com, span, halflife, alpha, min_periods,
                          adjust, ignore_na, False, self.name, self._index)

    # =================================================================
    #  距离度量（与另一列 / Series 逐元素计算）
    # =================================================================

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def euclidean_dist(self, other):
        """欧氏距离：与其他 Series/列 逐元素计算"""
        o = (<Series>other)._col if isinstance(other, Series) else other
        return self._col.euclidean_dist(o)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def manhattan_dist(self, other):
        """曼哈顿距离"""
        o = (<Series>other)._col if isinstance(other, Series) else other
        return self._col.manhattan_dist(o)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cosine_similarity(self, other):
        """余弦相似度"""
        o = (<Series>other)._col if isinstance(other, Series) else other
        return self._col.cosine_similarity(o)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cosine_dist(self, other):
        """余弦距离"""
        o = (<Series>other)._col if isinstance(other, Series) else other
        return self._col.cosine_dist(o)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def jaccard_sim(self, other):
        """Jaccard 相似度"""
        o = (<Series>other)._col if isinstance(other, Series) else other
        return self._col.jaccard_sim(o)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def hamming_dist(self, other):
        """汉明距离"""
        o = (<Series>other)._col if isinstance(other, Series) else other
        return self._col.hamming_dist(o)

    # =================================================================
    #  导出
    # =================================================================
    
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def to_list(self):
        """转换为Python列表"""
        return self.values

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def tolist(self):
        """pandas 风格别名：转换为 Python 列表"""
        return self.values

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def to_numpy(self):
        """导出为一维 Python 列表（本库 to_numpy 约定为 Python 容器）"""
        return self.values

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def to_dict(self):
        """转换为字典（索引 -> 值）"""
        result = {}
        n = len(self)
        for i in range(n):
            result[self._index[i]] = self._col[i]
        return result

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def to_pandas(self):
        """转换为pandas Series（如果pandas可用）"""
        try:
            import pandas as pd
            return pd.Series(self.values, index=list(self._index), name=self.name)
        except ImportError:
            raise ImportError("pandas is required for to_pandas()")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def to_frame(self, name=None):
        """转换为单列 DataFrame（对标 pandas Series.to_frame）"""
        cdef DataFrame result = DataFrame()
        cdef str cname = name if name is not None else self.name
        result._columns[cname] = self._col
        result.df_length = len(self)
        result.index = self._index
        return result

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def shift(self, int periods=1, fill_value=None):
        """移动数据（语义同 df.shift）：periods>0 下移，空位填 NaN/fill_value。"""
        n = len(self)
        if periods == 0:
            return self
        cdef BaseColumn col = _shift_column_impl(self._col, periods, n)
        cdef Py_ssize_t k, i
        if fill_value is not None:
            if col.is_2d and col.num_cols > 1:
                raise ValueError("fill_value not supported for 2D Series.shift")
            k = periods if periods >= 0 else -periods
            if periods > 0:
                for i in range(k):
                    _fast_set_cell(col, col.dtype, i, fill_value)
            else:
                for i in range(n - k, n):
                    _fast_set_cell(col, col.dtype, i, fill_value)
        cdef IndexBase idx = self._index
        return Series(self.name, col, idx)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def rank(self, str method='average', bint ascending=True, str na_option='keep'):
        """排名（pandas 语义）：average/min/max/first/dense。"""
        cdef BaseColumn col = self._col
        cdef Py_ssize_t n = len(self)
        if col.is_2d and col.num_cols > 1:
            return Series(self.name, _rank_row_keys(col, n, method, ascending), self._index)
        if col.dtype == TYPE_INT:
            return Series(self.name, _rank_col((<IntColumn>col).data_c, n, method, ascending), self._index)
        elif col.dtype == TYPE_DOUBLE:
            return Series(self.name, _rank_col((<DoubleColumn>col).data_c, n, method, ascending), self._index)
        raise TypeError(f"rank not supported for column type {col.dtype}")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def describe(self, object percentiles=None):
        """数值统计概览（对标 pandas Series.describe）：count/mean/std/min/分位/max。"""
        cdef BaseColumn col = self._col
        if col.dtype != TYPE_INT and col.dtype != TYPE_DOUBLE:
            raise TypeError(f"describe only supports numeric columns, got dtype {col.dtype}")
        if col.is_2d and col.num_cols > 1:
            raise ValueError("describe not supported for 2D column")
        cdef list names = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        cdef list vals = [
            col.count(), col.mean(), col.std(), col.min(),
            col.quantile(0.25), col.quantile(0.5), col.quantile(0.75), col.max(),
        ]
        return _agg_series(dict(zip(names, vals)))


# ============================================================
#  Series 级实现助手（供 Series 类调用）
# ============================================================
# ============================================================
# 列级 value_counts / isna / notna / map / factorize 的快速实现：
# 由 Series 类（Series.factorize 等）调用，语义与 df.value_counts /
# df.isnull / df.apply 完全对齐（2D 列走行级键，见 crud._col_to_keys）。
# ============================================================

@cython.binding(False)
@cython.always_allow_keywords(False)
def _series_value_counts(BaseColumn col, str key_name='value'):
    """列级 value_counts：行级键计数，2D 列键为子列元组，结果保持 cols 维度。"""
    cdef list keys = _col_to_keys(col)
    cdef Py_ssize_t n = len(keys)
    cdef Py_ssize_t i
    cdef object v

    cdef dict counts = {}
    for i in range(n):
        v = keys[i]
        counts[v] = counts.get(v, 0) + 1

    cdef list sorted_items = sorted(counts.items(), key=lambda x: -x[1])
    cdef DataFrame result = DataFrame()
    result.df_length = len(sorted_items)
    result.index = RangeIndex(0, len(sorted_items))
    cdef list key_list = [k for k, v in sorted_items]

    if _col_is_2d(col):
        result._columns[key_name] = _col_from_keys_2d(col, key_list)
    else:
        result._columns[key_name] = _column_from_list_impl(result, key_name, key_list)

    cdef IntColumn cnt_col = IntColumn(len(sorted_items))
    cnt_col.length = len(sorted_items)
    cdef int* cnt_ptr = cnt_col.data_c
    for i in range(len(sorted_items)):
        cnt_ptr[i] = sorted_items[i][1]
    result._columns['count'] = cnt_col
    return result


@cython.binding(False)
@cython.always_allow_keywords(False)
def _series_isna(BaseColumn col):
    """列级缺失掩码，语义与 df.isnull 一致：double=NaN、字符串=空串、类别=''，其余恒 False。
    2D 输入保持 cols 维度（IntColumn 掩码）。"""
    cdef bint is2d
    cdef Py_ssize_t c_out, total, i
    cdef Py_ssize_t* offset_ptr
    cdef double* d_ptr
    cdef CategoryColumn cat_col
    cdef int* out_ptr

    is2d, c_out, _ = _col_shape_probe(col)
    total = col.length
    cdef IntColumn new_col = IntColumn(max(total, 1), cols=c_out)
    new_col.length = total
    out_ptr = new_col.data_c

    if col.dtype == TYPE_DOUBLE:
        d_ptr = (<DoubleColumn>col).data_c
        with nogil:
            for i in range(total):
                out_ptr[i] = 1 if isnan(d_ptr[i]) else 0
    elif col.dtype == TYPE_ASCII:
        offset_ptr = (<AsciiColumn>col).offsets_c
        with nogil:
            for i in range(total):
                out_ptr[i] = 1 if offset_ptr[i + 1] == offset_ptr[i] else 0
    elif col.dtype == TYPE_UCS2:
        offset_ptr = (<UCS2Column>col).offsets_c
        with nogil:
            for i in range(total):
                out_ptr[i] = 1 if offset_ptr[i + 1] == offset_ptr[i] else 0
    elif col.dtype == TYPE_CATEGORY:
        cat_col = <CategoryColumn>col
        for i in range(total):
            out_ptr[i] = 1 if cat_col._resolve(i) == '' else 0
    else:
        memset(out_ptr, 0, total * sizeof(int))
    return new_col


@cython.binding(False)
@cython.always_allow_keywords(False)
def _series_notna(BaseColumn col):
    """列级非缺失掩码：_series_isna 的结果取反。"""
    cdef object m = _series_isna(col)
    cdef IntColumn mask_col = <IntColumn>m
    cdef int* p = mask_col.data_c
    cdef Py_ssize_t i
    with nogil:
        for i in range(mask_col.length):
            p[i] = 1 - p[i]
    return mask_col


@cython.binding(False)
@cython.always_allow_keywords(False)
def _series_factorize(BaseColumn col):
    """列级 factorize（对标 pandas Series.factorize）：返回
    (codes, uniques)，uniques 按首次出现顺序、codes 与行数等长。
    2D 列按行键（子列元组）编码。"""
    cdef list keys = _col_to_keys(col)
    cdef Py_ssize_t n = len(keys)
    cdef Py_ssize_t i
    cdef object k, code
    cdef dict seen = {}
    cdef list uniques = []
    cdef list codes_list = [None] * n
    for i in range(n):
        k = keys[i]
        code = seen.get(k, -1)
        if code == -1:
            code = len(uniques)
            seen[k] = code
            uniques.append(k)
        codes_list[i] = code
    return codes_list, uniques


@cython.binding(False)
@cython.always_allow_keywords(False)
def _series_map(BaseColumn col, object func):
    """列级元素映射（对标 pandas Series.map）：逐存储单元应用 func。
    2D 列按列主序逐 cell 应用，数值结果保持 cols 维度。"""
    cdef Py_ssize_t total = col.length
    cdef Py_ssize_t i
    cdef Py_ssize_t rows
    cdef list out = [None] * total
    for i in range(total):
        out[i] = func(col[i])
    if _col_is_2d(col):
        rows = col._get_rows_c()
        return _column_from_2d_list_impl(out, rows, col.num_cols)
    return _column_from_list_impl(DataFrame(), '', out)


# ============================================================
#  DataFrame 类（核心数据结构）
# ============================================================
@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class DataFrame:

    cdef BaseColumn _reorder_column(self, BaseColumn col, Py_ssize_t* indices, Py_ssize_t n, bint inplace):
        if col.dtype == TYPE_INT:
            return (<IntColumn>col).take_by_indices(indices, n, inplace)
        elif col.dtype == TYPE_DOUBLE:
            return (<DoubleColumn>col).take_by_indices(indices, n, inplace)
        elif col.dtype == TYPE_ASCII:
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
        self._columns = {}
        self.df_length = 0
        self.index = RangeIndex(0, 0)
        self._metric_cache = None
        self._groupby_cache = None
        self._linalg_cache = None

        self._2d_blocks = []          # 持有巨型2D数据块的强引用，防止被 GC
        self._ordered_cols = []
        self._schema_keys = ()

        cdef object sv
        cdef BaseColumn scol
        cdef Series ss
        cdef Py_ssize_t vlen
        cdef str col_name
        cdef dict parsed
        cdef bint has_values

        if schema:
            has_values = False
            for sv in schema.values():
                if isinstance(sv, (Series, BaseColumn, list)):
                    has_values = True
                    break
            if has_values:
                vlen = -1
                for col_name, sv in schema.items():
                    if isinstance(sv, Series):
                        ss = <Series>sv
                        scol = ss._col
                    elif isinstance(sv, BaseColumn):
                        scol = <BaseColumn>sv
                    elif isinstance(sv, list):
                        scol = _column_from_list_impl(self, col_name, sv)
                    else:
                        raise TypeError(
                            f"Unsupported column value for '{col_name}': {type(sv).__name__}")
                    if vlen < 0:
                        vlen = scol.length
                    elif scol.length != vlen:
                        raise ValueError(
                            f"All columns must have the same length, got {scol.length} and {vlen}")
                    self._columns[col_name] = scol
                self.df_length = vlen
                self.index = RangeIndex(0, vlen)
                self._schema_keys = tuple(schema.keys())
                return

            parsed = _parse_schema(schema)
            for col_name, dtype in parsed.items():
                if dtype == TYPE_INT:
                    self._columns[col_name] = IntColumn()
                elif dtype == TYPE_DOUBLE:
                    self._columns[col_name] = DoubleColumn()
                elif dtype == TYPE_ASCII:
                    self._columns[col_name] = AsciiColumn()
                elif dtype == TYPE_UCS2:
                    self._columns[col_name] = UCS2Column()
                elif dtype == TYPE_DATE:
                    self._columns[col_name] = DateColumn()
                elif dtype == TYPE_BOOL:
                    self._columns[col_name] = BoolColumn()
                elif dtype == TYPE_CATEGORY:
                    self._columns[col_name] = CategoryColumn()
                elif dtype == TYPE_DATETIME:
                    self._columns[col_name] = DateTimeColumn()
                elif dtype == TYPE_TIME:
                    self._columns[col_name] = TimeColumn()
            self._schema = parsed
            self._schema_keys = tuple(parsed.keys())

    def __dealloc__(self):
        if self._c_cols != NULL:
            free(self._c_cols)


    def __len__(self):
        return self.df_length

    def __eq__(self, other):
        if not isinstance(other, DataFrame):
            return NotImplemented
        cdef DataFrame rhs = <DataFrame>other
        if self.df_length != rhs.df_length:
            return False
        if set(self._columns.keys()) != set(rhs._columns.keys()):
            return False
        cdef DataFrame result = DataFrame()
        cdef Py_ssize_t i, n = self.df_length
        cdef str name
        cdef BaseColumn col_l, col_r
        cdef BoolColumn bool_col
        for name in self._columns:
            col_l = self._columns[name]
            col_r = rhs._columns[name]
            bool_col = BoolColumn(n)
            bool_col.length = n
            for i in range(n):
                if _col_row_key(col_l, i) == _col_row_key(col_r, i):
                    bool_col.data_c[i] = 1
                else:
                    bool_col.data_c[i] = 0
            result._columns[name] = bool_col
        result.df_length = n
        return result

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

        self._columns = {}
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
                self._columns[name] = IntColumn.create_view(blk_int.data_c + j * capacity, 0, blk_int)
            for j, name in enumerate(names):
                (<IntColumn>self._columns[name]).capacity = capacity

        if type_groups[TYPE_DOUBLE]:
            names = type_groups[TYPE_DOUBLE]
            c = <uint32_t>len(names)
            blk_dbl = DoubleColumn(capacity * c, cols=c)
            blk_dbl.length = capacity * c
            self._2d_blocks.append(blk_dbl)
            for j, name in enumerate(names):
                self._columns[name] = DoubleColumn.create_view(blk_dbl.data_c + j * capacity, 0, blk_dbl)
            for j, name in enumerate(names):
                (<DoubleColumn>self._columns[name]).capacity = capacity

        if type_groups[TYPE_DATE]:
            names = type_groups[TYPE_DATE]
            c = <uint32_t>len(names)
            blk_date = DateColumn(capacity * c, cols=c)
            blk_date.length = capacity * c
            self._2d_blocks.append(blk_date)
            for j, name in enumerate(names):
                self._columns[name] = DateColumn.create_view(<int32_t*>blk_date.data_c + j * capacity, 0, blk_date)
            for j, name in enumerate(names):
                (<DateColumn>self._columns[name]).capacity = capacity

        if type_groups[TYPE_DATETIME]:
            names = type_groups[TYPE_DATETIME]
            c = <uint32_t>len(names)
            blk_dt = DateTimeColumn(capacity * c, cols=c)
            blk_dt.length = capacity * c
            self._2d_blocks.append(blk_dt)
            for j, name in enumerate(names):
                self._columns[name] = DateTimeColumn.create_view(<int64_t*>blk_dt.data_c + j * capacity, 0, blk_dt)
            for j, name in enumerate(names):
                (<DateTimeColumn>self._columns[name]).capacity = capacity

        if type_groups[TYPE_TIME]:
            names = type_groups[TYPE_TIME]
            c = <uint32_t>len(names)
            blk_time = TimeColumn(capacity * c, cols=c)
            blk_time.length = capacity * c
            self._2d_blocks.append(blk_time)
            for j, name in enumerate(names):
                self._columns[name] = TimeColumn.create_view(<int32_t*>blk_time.data_c + j * capacity, 0, blk_time)
            for j, name in enumerate(names):
                (<TimeColumn>self._columns[name]).capacity = capacity

        # 3. 处理离散的非等宽字段列
        for name, dtype in other_cols.items():
            if dtype == TYPE_ASCII:
                self._columns[name] = AsciiColumn(capacity)
            elif dtype == TYPE_UCS2:
                self._columns[name] = UCS2Column(capacity)
            elif dtype == TYPE_BOOL:
                self._columns[name] = BoolColumn(capacity)
            elif dtype == TYPE_CATEGORY:
                self._columns[name] = CategoryColumn(capacity)

        # ------------------------------------------------------------
        # 4. 构建底层的 C Pointer Mapping Arrays（核心硬件加速）
        # ------------------------------------------------------------
        self._ordered_cols = [self._columns[k] for k in self._schema_keys]
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

    @cython.always_allow_keywords(False)
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
        cdef tuple schema_keys = self._schema_keys

        if kwargs:
            if args:
                raise TypeError("Cannot mix positional args and keyword args in add_row.")
            for i in range(n_cols):
                if schema_keys[i] not in kwargs:
                    raise ValueError(f"Missing value for column: {schema_keys[i]}")
            try:
                for i in range(n_cols):
                    _fast_insert_val(
                        self._c_cols[i].dtype,
                        self._c_cols[i].data_ptr,
                        self._c_cols[i].col_obj,
                        row_idx,
                        kwargs[schema_keys[i]]
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
            if self._c_cols[i].dtype == TYPE_ASCII:
                (<AsciiColumn>self._c_cols[i].col_obj).undo_nolen()
            elif self._c_cols[i].dtype == TYPE_UCS2:
                (<UCS2Column>self._c_cols[i].col_obj).undo_nolen()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
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
        elif dt == TYPE_ASCII: return 'str'
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
            is2d = (<IntColumn>col).is_2d
            cols = (<IntColumn>col).num_cols
            rows = (<IntColumn>col)._get_rows_c()
        elif dt == TYPE_DOUBLE:
            is2d = (<DoubleColumn>col).is_2d
            cols = (<DoubleColumn>col).num_cols
            rows = (<DoubleColumn>col)._get_rows_c()
        elif dt == TYPE_DATE:
            is2d = (<DateColumn>col).is_2d
            cols = (<DateColumn>col).num_cols
            rows = (<DateColumn>col)._get_rows_c()
        elif dt == TYPE_DATETIME:
            is2d = (<DateTimeColumn>col).is_2d
            cols = (<DateTimeColumn>col).num_cols
            rows = (<DateTimeColumn>col)._get_rows_c()
        elif dt == TYPE_TIME:
            is2d = (<TimeColumn>col).is_2d
            cols = (<TimeColumn>col).num_cols
            rows = (<TimeColumn>col)._get_rows_c()

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
        if dt in (TYPE_ASCII, TYPE_UCS2, TYPE_CATEGORY):
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

        for col_name, col in self._columns.items():
            names.append(self._truncate_str(str(col_name)))
            types.append(f"[{self._fast_type_name(col.dtype)}]")

        return names, types

    cdef list _get_row_strings(self, Py_ssize_t i):
        cdef list row_strs
        cdef RangeIndex ri
        cdef IntIndex ii
        cdef MultiIndex mi
        cdef int j, code, pos, n_index_cells

        if self.index is None or self.index.index_type == INDEX_RANGE or self.index.index_type == INDEX_INT:
            n_index_cells = 1
        else:
            mi = <MultiIndex>self.index
            n_index_cells = mi.nlevels

        row_strs = [None] * (n_index_cells + len(self._columns))
        pos = 0

        if self.index is None:
            row_strs[pos] = str(i)
            pos += 1
        elif self.index.index_type == INDEX_RANGE:
            ri = <RangeIndex>self.index
            row_strs[pos] = str(ri.start + i * ri.step)
            pos += 1
        elif self.index.index_type == INDEX_INT:
            ii = <IntIndex>self.index
            row_strs[pos] = str(ii.data.data_c[i])
            pos += 1
        elif self.index.index_type == INDEX_MULTI:
            mi = <MultiIndex>self.index
            for j in range(mi.nlevels):
                code = (<IntColumn>mi.codes[j]).data_c[i]
                row_strs[pos] = self._truncate_str(str(mi.levels[j][code]))
                pos += 1

        for col_name, col in self._columns.items():
            row_strs[pos] = self._truncate_str(self._format_value_fast(col, i))
            pos += 1

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
        if name in self._columns:
            return Series(name, self._columns[name], self.index)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __dir__(self):
        cdef list base_dir = dir(type(self))
        return sorted(set(base_dir + list(self._columns.keys())))

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __getitem__(self, object name):
        cdef DataFrame result
        cdef str col_name
        cdef dict columns = self._columns
        if isinstance(name, str):
            if name in columns:
                return Series(name, columns[name], self.index)
            raise KeyError(f"Column '{name}' not found")
        if isinstance(name, list):
            result = DataFrame()
            result.df_length = self.df_length
            for col_name in name:
                if col_name not in columns:
                    raise KeyError(f"Column '{col_name}' not found")
                result._columns[col_name] = columns[col_name]
            if self.index is not None:
                result.index = self.index
            return result
        if isinstance(name, slice):
            return self._slice_rows(name)
        if isinstance(name, Series):
            c = (<Series>name)._col
            if c.dtype == TYPE_BOOL or c.dtype == TYPE_INT:
                return self.loc[name]
            raise TypeError("DataFrame boolean mask must be a boolean Series")
        if isinstance(name, IntColumn):
            return self.loc[name]
        if isinstance(name, BoolColumn):
            return self.loc[name]
        if name in columns:
            return Series(name, columns[name], self.index)
        raise KeyError(f"Column '{name}' not found")

    cdef DataFrame _slice_rows(self, object key):
        cdef Py_ssize_t n = self.df_length
        cdef Py_ssize_t start, stop, step, count, i
        start, stop, step = key.indices(n)
        count = 0
        if step > 0:
            if start < stop:
                count = (stop - start + step - 1) // step
        else:
            if start > stop:
                count = (start - stop - step - 1) // (-step)

        if count == 0:
            result = DataFrame()
            result.df_length = 0
            for col_name, col_obj in self._columns.items():
                result._columns[col_name] = col_obj.copy()
                result._columns[col_name].length = 0
            if self.index is not None:
                result.index = self.index._take(<Py_ssize_t*>NULL, 0)
            return result

        cdef Py_ssize_t* idx = <Py_ssize_t*>aligned_malloc(count * sizeof(Py_ssize_t))
        if not idx: raise MemoryError()
        i = 0
        for j in range(start, stop, step):
            idx[i] = j
            i += 1
        try:
            return self._take(idx, count)
        finally:
            os_aligned_free(idx)

    cdef DataFrame _cum_results(self, str method):
        cdef DataFrame result = DataFrame()
        result.df_length = self.df_length
        result.index = self.index
        cdef str name
        cdef object col
        cdef object res
        cdef BaseColumn new_col
        for name, col in self._columns.items():
            if col.dtype == TYPE_INT or col.dtype == TYPE_DOUBLE:
                res = getattr(col, method)()
                if isinstance(res, BaseColumn):
                    new_col = <BaseColumn>res
                    result._columns[name] = new_col
        return result

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def get_series(self, name):
        """返回Series（单列代理），知道自己的名字和索引"""
        from fluidframe.core.series import Series
        return Series(name, self._columns[name], self.index)

    def __iter__(self):
        return iter(self._columns)

    def __contains__(self, name):
        return name in self._columns

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __setitem__(self, object name, object value):
        cdef BaseColumn new_col
        cdef Series s
        if not isinstance(name, str):
            raise TypeError(f"DataFrame column names must be strings, got {type(name).__name__}")

        if isinstance(value, Series):
            s = <Series>value
            if self.df_length > 0 and s._index is not self.index:
                s = s.reindex(self.index)
            new_col = s._col
        elif isinstance(value, BaseColumn):
            new_col = <BaseColumn>value
        elif isinstance(value, list):
            new_col = _column_from_list_impl(self, name, value)
            self._check_assign_length(name, new_col)
        else:
            new_col = _column_from_scalar(self, name, value, self.df_length)
            self._check_assign_length(name, new_col)

        self._columns[name] = new_col

    cdef void _check_assign_length(self, str name, BaseColumn new_col) except *:
        # 长度防爆校验：允许 2D 列 rows 数匹配的特例
        if self.df_length > 0:
            if new_col.length != self.df_length:
                if new_col.is_2d and new_col._get_rows_c() == self.df_length:
                    return
                raise ValueError(
                    f"Length of assignment column {new_col.length} != DataFrame length {self.df_length}")
        elif new_col.length > 0:
            self.df_length = new_col.length
            self.index = RangeIndex(0, new_col.length)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __delitem__(self, object name):
        del self._columns[name]

    @property
    def shape(self):
        return (self.df_length, len(self._columns))

    @property
    def values(self):
        """导出为二维 List of Lists（与 to_numpy 一致，本库约定）"""
        return _to_numpy_impl(self)

    @property
    def size(self):
        """元素总数：行 × 列"""
        return self.df_length * len(self._columns)

    @property
    def ndim(self):
        """始终为 2（DataFrame 语义）"""
        return 2

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def get(self, key, default=None):
        """按列名取 Series，不存在时返回 default（pandas df.get 语义）"""
        if key in self._columns:
            return self.get_series(key)
        return default

    @property
    def columns(self):
        return list(self._columns.keys())

    @property
    def col_names(self):
        import warnings
        warnings.warn("col_names is deprecated, use columns instead", DeprecationWarning, stacklevel=2)
        return list(self._columns.keys())

    @property
    def dtypes(self):
        cdef dict result = {}
        cdef str name
        cdef BaseColumn col
        for name, col in self._columns.items():
            result[name] = _TYPE_PYCLASSES.get(col.dtype, None)
        return result

    cdef DataFrame _take(self, Py_ssize_t* indices, Py_ssize_t n):
        cdef DataFrame result = DataFrame()
        result.df_length = n
        cdef str name
        cdef BaseColumn col
        for name, col in self._columns.items():
            result._columns[name] = self._reorder_column(col, indices, n, False)

        if self.index is not None:
            result.index = self.index._take(indices, n)
        return result

    cdef double _get_col_value_as_double(self, BaseColumn col, Py_ssize_t i):
        if col.dtype == TYPE_INT:
            return <double>(<IntColumn>col).data_c[i]
        elif col.dtype == TYPE_DOUBLE:
            return (<DoubleColumn>col).data_c[i]
        return 0.0

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def groupby(self, by_cols):
        if self._groupby_cache is not None:
            self._groupby_cache = None
        if isinstance(by_cols, str):
            by_cols = [by_cols]
        cdef list by_list = list(by_cols)
        for cn in by_list:
            if cn not in self._columns:
                raise KeyError(f"Column '{cn}' not found")
        cdef _DataFrameGroupByAccessor acc = _make_groupby_accessor(self, by_list)
        self._groupby_cache = acc
        return acc

    cdef void _shift_right(self, BaseColumn col, Py_ssize_t pos):
        col.shift_right(pos)

    cdef void _shift_left(self, BaseColumn col, Py_ssize_t pos):
        col.shift_left(pos)

    cdef void _col_to_double_buffer(self, BaseColumn col, double* out, Py_ssize_t n, Py_ssize_t sub=0) except *:
        # sub = 2D 列的子列下标（列优先布局 data_c[sub*rows + i]），1D 列恒为 0
        cdef Py_ssize_t r, i
        cdef int* int_ptr
        cdef double* double_ptr

        if col.dtype == TYPE_INT:
            int_ptr = (<IntColumn>col).data_c
            r = (<IntColumn>col)._get_rows_c()
            if n > r: n = r

            with nogil:
                for i in range(n):
                    out[i] = <double>int_ptr[sub * r + i]

        elif col.dtype == TYPE_DOUBLE:
            double_ptr = (<DoubleColumn>col).data_c
            r = (<DoubleColumn>col)._get_rows_c()
            if n > r: n = r
            memcpy(out, double_ptr + (sub * r), n * sizeof(double))
        else:
            raise TypeError(f"Cannot convert column of type {self._fast_type_name(col.dtype)} to double buffer")

    # ─── Thin delegate methods ───

    # CRUD
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def eval(self, str expr, bint inplace=False):
        return _eval_impl(self, expr, inplace)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def query(self, str expr):
        return _query_impl(self, expr)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def append_row(self, **kwargs):
        return _append_row_impl(self, kwargs)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def append(self, row, bint ignore_index=False):
        return _append_impl(self, row, ignore_index)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def copy(self):
        return _copy_impl(self)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_front(self):
        return _pop_front_impl(self)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_back(self):
        return _pop_back_impl(self)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def rename(self, mapper=None, columns=None, index=None, bint inplace=False):
        if columns is not None:
            mapper = columns
        elif mapper is None:
            raise ValueError("Must specify either 'mapper' or 'columns'")
        if callable(mapper):
            result = _copy_impl(self) if not inplace else self
            new_columns = {}
            for old_name in list(result._columns.keys()):
                new_name = mapper(old_name)
                if new_name != old_name:
                    new_columns[new_name] = result._columns.pop(old_name)
            result._columns.update(new_columns)
            return result
        else:
            return _rename_impl(self, mapper)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def reset_index(self):
        return _reset_index_impl(self)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def set_index(self, str col_name, bint drop=True):
        return _set_index_impl(self, col_name, drop)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def get_dummies(self, col_names=None, str prefix='', bint drop_first=False):
        return _get_dummies_impl(self, col_names, prefix, drop_first)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def reindex(self, list new_index):
        return _reindex_impl(self, new_index)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def insert_row(self, Py_ssize_t pos, **kwargs):
        return _insert_row_impl(self, pos, kwargs)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def drop(self, labels=None, axis=0, columns=None, index=None):
        cdef str name
        cdef str col_name
        cdef DataFrame new_df
        if columns is not None:
            labels = columns
            axis = 1
        elif index is not None:
            labels = index
            axis = 0
        if labels is None:
            raise ValueError("Must specify either 'labels' or 'index'/'columns'")
        if isinstance(labels, str):
            labels = [labels]
        if axis == 0 and all(isinstance(x, str) for x in labels):
            axis = 1
        if axis == 0:
            return self.drop_with_index(labels)
        else:
            for name in labels:
                if name not in self._columns:
                    raise KeyError(f"Column '{name}' not found")
            new_df = DataFrame()
            new_df.df_length = self.df_length
            for col_name, col_obj in self._columns.items():
                if col_name not in labels:
                    new_df._columns[col_name] = col_obj
            if self.index is not None:
                new_df.index = self.index
            return new_df

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def drop_with_index(self, list drop_indices, bint inplace=False):
        return _drop_with_index_impl(self, drop_indices, inplace)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def drop_index(self, object labels, bint inplace=False):
        return _drop_index_impl(self, labels, inplace)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def drop_raw(self, Py_ssize_t row_idx):
        return _drop_raw_impl(self, row_idx)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def insert(self, Py_ssize_t loc, str column, object value, bint allow_duplicates=False):
        return _insert_column_impl(self, column, value, loc)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def insert_column(self, str col_name, object col_data, Py_ssize_t pos=-1):
        return _insert_column_impl(self, col_name, col_data, pos)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def duplicate_column(self, str src_name, str new_name=None):
        _duplicate_column_impl(self, src_name, new_name)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def duplicate_rows(self, list indices=None):
        return _duplicate_rows_impl(self, indices)

    # Aggregation
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def mean(self, col_names=None):
        return _numeric_agg_impl(self, 'mean', col_names)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def sum(self, col_names=None):
        return _numeric_agg_impl(self, 'sum', col_names)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def min(self, col_names=None):
        return _numeric_agg_impl(self, 'min', col_names)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def max(self, col_names=None):
        return _numeric_agg_impl(self, 'max', col_names)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def std(self, col_names=None):
        return _numeric_agg_impl(self, 'std', col_names)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def var(self, col_names=None):
        return _numeric_agg_impl(self, 'var', col_names)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def median(self, col_names=None):
        return _numeric_agg_impl(self, 'median', col_names)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def quantile(self, q=0.5, col_names=None):
        return _numeric_agg_arg_impl(self, 'quantile', col_names, q)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def mode(self, col_names=None):
        return _mode_impl(self, col_names)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def prod(self, col_names=None):
        return _numeric_agg_impl(self, 'prod', col_names)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def sem(self, col_names=None):
        return _numeric_agg_impl(self, 'sem', col_names)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def mad(self, col_names=None):
        return _numeric_agg_impl(self, 'mad', col_names)

    # 逐子列聚合（2D 列每个子列独立聚合，1D 列与普通聚合一致）
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def mean_withcol(self, col_names=None):
        return _numeric_agg_withcol_impl(self, 'mean', col_names)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def sum_withcol(self, col_names=None):
        return _numeric_agg_withcol_impl(self, 'sum', col_names)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def min_withcol(self, col_names=None):
        return _numeric_agg_withcol_impl(self, 'min', col_names)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def max_withcol(self, col_names=None):
        return _numeric_agg_withcol_impl(self, 'max', col_names)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def std_withcol(self, col_names=None):
        return _numeric_agg_withcol_impl(self, 'std', col_names)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def var_withcol(self, col_names=None):
        return _numeric_agg_withcol_impl(self, 'var', col_names)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def median_withcol(self, col_names=None):
        return _numeric_agg_withcol_impl(self, 'median', col_names)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def quantile_withcol(self, q=0.5, col_names=None):
        return _numeric_agg_arg_withcol_impl(self, 'quantile', col_names, q)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def mode_withcol(self, col_names=None):
        return _numeric_agg_withcol_impl(self, 'mode', col_names)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def prod_withcol(self, col_names=None):
        return _numeric_agg_withcol_impl(self, 'prod', col_names)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def sem_withcol(self, col_names=None):
        return _numeric_agg_withcol_impl(self, 'sem', col_names)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def mad_withcol(self, col_names=None):
        return _numeric_agg_withcol_impl(self, 'mad', col_names)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def describe_withcol(self):
        return _describe_withcol_impl(self)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def count(self, col_names=None):
        return _count_impl(self, col_names)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def describe(self, object percentiles=None):
        return _describe_impl(self, percentiles)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def value_counts(self, str col_name):
        return _value_counts_impl(self, col_name)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def nunique(self, str col_name=None):
        return _nunique_impl(self, col_name)

    # Cumulative aggregations
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cumsum(self):
        return self._cum_results('cumsum')

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cumprod(self):
        return self._cum_results('cumprod')

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cummax(self):
        return self._cum_results('cummax')

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cummin(self):
        return self._cum_results('cummin')

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def skew(self):
        return _numeric_agg_impl(self, 'skew')

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def kurt(self):
        return _numeric_agg_impl(self, 'kurt')

    # Column-level delegates
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def argmax(self, str col_name):
        return self._columns[col_name].idxmax()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def argmin(self, str col_name):
        return self._columns[col_name].idxmin()

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def idxmax(self, str col_name=None):
        """逐列最大值的行位置；指定列名时返回该列标量。"""
        if col_name is not None:
            return self._columns[col_name].idxmax()
        cdef dict result = {}
        cdef str name
        cdef object col
        for name, col in self._columns.items():
            if col.dtype != TYPE_INT and col.dtype != TYPE_DOUBLE:
                continue
            if col.is_2d and col.num_cols > 1:
                raise ValueError(f"Cannot compute idxmax for 2D column '{name}'")
            result[name] = col.idxmax()
        return _agg_series(result)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def idxmin(self, str col_name=None):
        """逐列最小值的行位置；指定列名时返回该列标量。"""
        if col_name is not None:
            return self._columns[col_name].idxmin()
        cdef dict result = {}
        cdef str name
        cdef object col
        for name, col in self._columns.items():
            if col.dtype != TYPE_INT and col.dtype != TYPE_DOUBLE:
                continue
            if col.is_2d and col.num_cols > 1:
                raise ValueError(f"Cannot compute idxmin for 2D column '{name}'")
            result[name] = col.idxmin()
        return _agg_series(result)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def unique(self, str col_name):
        return self._columns[col_name].unique()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def factorize(self, str col_name):
        return _series_factorize(self._columns[col_name])

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def isin(self, str col_name, set values):
        return Series(col_name, self._columns[col_name].isin(values), self.index)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def between(self, str col_name, low, high):
        return Series(col_name, self._columns[col_name].between(low, high), self.index)

    # ============================================================
    # P2: pandas 兼容方法
    # ============================================================
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop(self, str col_name):
        """弹出指定列并返回 Series（原地删除），对标 pandas DataFrame.pop。"""
        if col_name not in self._columns:
            raise KeyError(f"Column '{col_name}' not found")
        cdef BaseColumn col = self._columns.pop(col_name)
        return Series(col_name, col, self.index)

    @property
    def empty(self):
        """判断 DataFrame 是否为空，对标 pandas DataFrame.empty。"""
        return self.df_length == 0 or len(self._columns) == 0

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def update(self, DataFrame other):
        """用另一个 DataFrame 的值原地更新（按列名对齐），对标 pandas DataFrame.update。"""
        cdef str col_name
        cdef Py_ssize_t n
        for col_name in other._columns:
            if col_name in self._columns:
                src = other._columns[col_name]
                dst = self._columns[col_name]
                if src.length != dst.length:
                    raise ValueError(
                        f"Column '{col_name}': replacement has length {src.length}, "
                        f"existing has length {dst.length}")
                if src.dtype != dst.dtype:
                    raise TypeError(
                        f"Column '{col_name}': source dtype {src.dtype} != dest dtype {dst.dtype}")
                n = dst.length
                if dst.dtype == TYPE_INT:
                    memcpy((<IntColumn>dst).data_c, (<IntColumn>src).data_c, n * sizeof(int))
                elif dst.dtype == TYPE_DOUBLE:
                    memcpy((<DoubleColumn>dst).data_c, (<DoubleColumn>src).data_c, n * sizeof(double))
                else:
                    raise NotImplementedError(
                        f"update() not supported for column type {dst.dtype}. "
                        f"Only INT and DOUBLE columns are supported.")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def to_records(self, bint index=True):
        """转为 list of dicts（records 格式），对标 pandas DataFrame.to_records 的简化版。"""
        result = []
        cdef list col_names = list(self._columns.keys())
        cdef Py_ssize_t r, c_idx
        cdef dict row
        for r in range(self.df_length):
            row = {}
            for c_idx, cn in enumerate(col_names):
                row[cn] = self._columns[cn][r]
            if index and self.index is not None:
                row['index'] = self.index[r]
            result.append(row)
        return result

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def items(self):
        """按列迭代，返回 (column_name, Series) 对，对标 pandas DataFrame.items。"""
        cdef str col_name
        cdef list col_names = list(self._columns.keys())
        for col_name in col_names:
            yield col_name, Series(col_name, self._columns[col_name], self.index)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def iterrows(self):
        """按行迭代，返回 (index, Series) 对，对标 pandas DataFrame.iterrows。"""
        cdef list col_names = list(self._columns.keys())
        cdef Py_ssize_t r
        cdef dict row_dict
        cdef BaseColumn col
        for r in range(self.df_length):
            row_dict = {}
            for cn in col_names:
                col = self._columns[cn]
                row_dict[cn] = col[r]
            idx_label = self.index[r] if self.index is not None else r
            yield idx_label, row_dict

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def itertuples(self):
        """按行迭代，返回 namedtuple，对标 pandas DataFrame.itertuples。"""
        from collections import namedtuple
        cdef list col_names = list(self._columns.keys())
        cdef Py_ssize_t r
        cdef dict row_dict
        cdef BaseColumn col
        Tup = namedtuple('Pandas', ['index'] + col_names)
        for r in range(self.df_length):
            row_dict = {}
            for cn in col_names:
                col = self._columns[cn]
                row_dict[cn] = col[r]
            idx_label = self.index[r] if self.index is not None else r
            yield Tup(index=idx_label, **row_dict)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def join(self, DataFrame other, str how='left'):
        """按索引连接另一个 DataFrame，对标 pandas DataFrame.join。"""
        if how not in ('left', 'inner', 'outer'):
            raise ValueError(f"join how must be 'left', 'inner', or 'outer', got '{how}'")
        cdef DataFrame result = self.copy()
        cdef str cn
        for cn in other._columns:
            if cn not in result._columns:
                result.insert_column(cn, other._columns[cn])
            else:
                raise ValueError(f"Column '{cn}' already exists in both DataFrames. "
                                 f"Use merge() with on= parameter for column-based join.")
        return result

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def compare(self, DataFrame other, align_axis=1):
        """对比两个 DataFrame 的差异，返回差异值，对标 pandas DataFrame.compare 的简化版。"""
        if self.columns != other.columns:
            raise ValueError("Can only compare identically-labeled DataFrame objects (same columns)")
        cdef dict diffs = {}
        cdef str cn
        cdef Py_ssize_t r
        for cn in self.columns:
            col_a = self._columns[cn]
            col_b = other._columns[cn]
            if col_a.length != col_b.length:
                raise ValueError(f"Column '{cn}': lengths differ ({col_a.length} vs {col_b.length})")
            diff_rows = []
            for r in range(col_a.length):
                if col_a[r] != col_b[r]:
                    diff_rows.append((r, col_a[r], col_b[r]))
            if diff_rows:
                diffs[cn] = diff_rows
        return diffs

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def combine_first(self, DataFrame other):
        """用另一个 DataFrame 填充本 DataFrame 的缺失值，对标 pandas DataFrame.combine_first。"""
        cdef str cn
        cdef Py_ssize_t r, n
        result = self.copy()
        for cn in other._columns:
            if cn not in result._columns:
                result.insert_column(cn, other._columns[cn])
            else:
                dst_col = result._columns[cn]
                src_col = other._columns[cn]
                n = min(dst_col.length, src_col.length)
                vals = [dst_col[r] for r in range(dst_col.length)]
                src_vals = [src_col[r] for r in range(n)]
                for r in range(n):
                    v = vals[r]
                    if v is None or (isinstance(v, float) and v != v):
                        vals[r] = src_vals[r]
                dtype = dst_col.dtype
                if dtype == TYPE_INT:
                    new_col = IntColumn(len(vals))
                    for r in range(len(vals)):
                        new_col.append(int(vals[r]))
                elif dtype == TYPE_DOUBLE:
                    new_col = DoubleColumn(len(vals))
                    for r in range(len(vals)):
                        new_col.append(float(vals[r]))
                elif dtype == TYPE_ASCII:
                    new_col = AsciiColumn()
                    for v in vals:
                        new_col.append(str(v) if v is not None else '')
                elif dtype == TYPE_UCS2:
                    new_col = UCS2Column()
                    for v in vals:
                        new_col.append(str(v) if v is not None else '')
                elif dtype == TYPE_DATE:
                    new_col = DateColumn()
                    for v in vals:
                        new_col.append(v)
                elif dtype == TYPE_BOOL:
                    new_col = BoolColumn()
                    for v in vals:
                        new_col.append(v)
                elif dtype == TYPE_CATEGORY:
                    new_col = CategoryColumn()
                    for v in vals:
                        new_col.append(str(v) if v is not None else '')
                elif dtype == TYPE_DATETIME:
                    new_col = DateTimeColumn()
                    for v in vals:
                        new_col.append(v)
                elif dtype == TYPE_TIME:
                    new_col = TimeColumn()
                    for v in vals:
                        new_col.append(v)
                else:
                    new_col = dst_col
                result._columns[cn] = new_col
        return result

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def interpolate(self, str method='linear'):
        """缺失值插值，支持 'linear' 方法，对标 pandas DataFrame.interpolate 的简化版。"""
        if method != 'linear':
            raise NotImplementedError(f"interpolate method '{method}' not supported, use 'linear'")
        cdef str cn
        cdef list col_names = list(self._columns.keys())
        cdef Py_ssize_t n, i, j, last_valid, next_valid
        cdef double last_val, next_val, alpha
        cdef double* ptr
        cdef DoubleColumn dc
        result = self.copy()
        for cn in col_names:
            col = result._columns[cn]
            if col.dtype != TYPE_DOUBLE:
                continue
            dc = <DoubleColumn>col
            n = dc.length
            if n == 0:
                continue
            ptr = dc.data_c
            last_valid = -1
            for i in range(n):
                if not isnan(ptr[i]):
                    last_valid = i
                    break
            if last_valid == -1:
                continue
            i = last_valid + 1
            while i < n:
                if isnan(ptr[i]):
                    next_valid = -1
                    for j in range(i + 1, n):
                        if not isnan(ptr[j]):
                            next_valid = j
                            break
                    if next_valid == -1:
                        for j in range(i, n):
                            ptr[j] = ptr[last_valid]
                        break
                    last_val = ptr[last_valid]
                    next_val = ptr[next_valid]
                    for j in range(i, next_valid):
                        alpha = <double>(j - last_valid) / <double>(next_valid - last_valid)
                        ptr[j] = last_val + alpha * (next_val - last_val)
                    last_valid = next_valid
                    i = next_valid + 1
                else:
                    last_valid = i
                    i += 1
        return result

    # I/O
    def __str__(self):
        return _to_string_impl(self)

    def __repr__(self):
        return _to_string_impl(self)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def to_string(self):
        return _to_string_impl(self)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def print(self, max_rows=80):
        print(self.to_string())

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def to_html(self, Py_ssize_t max_rows=80):
        return _to_html_impl(self, max_rows)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def to_excel(self, object filename, str sheet_name='Sheet1'):
        return _to_excel_impl(self, filename, sheet_name)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def to_pickle(self, str path):
        return _to_pickle_impl(self, path)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def to_csv(self, str path, str sep=',', str na_rep='', str float_format=None,
               list columns=None, bint header=True, str line_terminator='\n'):
        return _to_csv_impl(self, path, sep, na_rep, float_format, columns, header, line_terminator)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def to_tsv(self, filename):
        return self.to_csv(filename, sep='\t')

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def to_dict(self):
        return _to_dict_impl(self)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def to_json(self, bint orient_records=True):
        return _to_json_impl(self, orient_records)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def to_numpy(self):
        return _to_numpy_impl(self)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def info(self):
        return _info_impl(self)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _dtype_summary(self):
        return _dtype_summary_impl(self)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def memory_usage(self):
        return _memory_usage_impl(self)

    # Merge
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def concat(self, DataFrame other):
        return _concat_impl(self, other)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def merge(self, DataFrame right, object on=None, str how='inner'):
        if isinstance(on, str):
            on = [on]
        return _merge_impl(self, right, on, how)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def duplicated(self, list subset=None, str keep='first'):
        return _wrap_series('_mask', _duplicated_impl(self, subset, keep), self.index)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def drop_duplicates(self, list subset=None, str keep='first', bint inplace=False):
        return _drop_duplicates_impl(self, subset, keep, inplace)

    # Transform
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def argsort_by(self, str col_name, bint reverse=False):
        return _argsort_by_impl(self, col_name, reverse)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def sort_values(self, str col_name, bint ascending=True, bint inplace=False):
        return _sort_values_impl(self, col_name, not ascending, inplace)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def sort_index(self, bint ascending=True, bint inplace=False):
        if 'index' not in self._columns:
            raise KeyError("No 'index' column found. Call reset_index() first.")
        return self.sort_values('index', ascending=ascending, inplace=inplace)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def diff(self, str col_name, int periods=1):
        return _wrap_series(col_name, _diff_impl(self, col_name, periods), self.index)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def apply(self, object func, str axis='columns', bint raw=False):
        return _apply_impl(self, func, axis, raw)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def applymap(self, object func):
        return _apply_impl(self, func, 'columns', False)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def clip_column(self, str col_name, object lower=None, object upper=None, bint inplace=False):
        return _clip_column_impl(self, col_name, lower, upper, inplace)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def clip(self, object lower=None, object upper=None, bint inplace=False):
        return _clip_all_impl(self, lower, upper, inplace)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def where_column(self, str col_name, str cond_col, object other):
        return _wrap_series(col_name, _where_column_impl(self, col_name, cond_col, other), self.index)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def mask_column(self, str col_name, str cond_col, object other):
        return _wrap_series(col_name, _mask_column_impl(self, col_name, cond_col, other), self.index)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def where_row(self, IntColumn mask, object fill=0):
        return _where_row_impl(self, mask, fill)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isnull(self):
        return _isnull_impl(self)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isna(self):
        return _isnull_impl(self)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def notna(self):
        return _notna_impl(self)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def fillna(self, value=None, method=None, col_names=None):
        if method is not None:
            if method == 'ffill':
                return self.ffill()
            elif method == 'bfill':
                return self.bfill()
            else:
                raise ValueError(f"Invalid fill method: {method}")
        if value is None:
            raise ValueError("Must specify either 'value' or 'method'")
        return _fillna_impl(self, value, col_names)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def dropna(self, axis=0, str how='any', int thresh=-1, list subset=None):
        """删除含缺失值的行/列，对标 pandas DataFrame.dropna。
        axis=0: 删除行, axis=1: 删除列。
        how='any': 任一缺失即删, 'all': 全部缺失才删。
        thresh: 需要的非缺失值数量。
        subset: 只检查指定列。
        """
        if axis == 1 or axis == 'columns':
            return _dropna_columns_impl(self, how)
        cdef list check_cols = subset if subset else list(self._columns.keys())
        cdef Py_ssize_t n = self.df_length
        cdef set drop_rows = set()
        cdef Py_ssize_t r, valid_count
        cdef str cn
        cdef BaseColumn col
        for cn in check_cols:
            if cn not in self._columns:
                raise KeyError(f"Column '{cn}' not found")
        for r in range(n):
            valid_count = 0
            has_na = False
            for cn in check_cols:
                col = self._columns[cn]
                val = col[r]
                if val is None or (isinstance(val, float) and val != val):
                    has_na = True
                else:
                    valid_count += 1
            if how == 'any' and has_na:
                drop_rows.add(r)
            elif how == 'all' and valid_count == 0:
                drop_rows.add(r)
            if thresh > 0 and valid_count < thresh:
                drop_rows.add(r)
        if not drop_rows:
            return self.copy()
        keep = [i for i in range(n) if i not in drop_rows]
        cdef Py_ssize_t keep_count = len(keep)
        cdef Py_ssize_t* keep_c = <Py_ssize_t*>aligned_malloc(max(keep_count, 1) * sizeof(Py_ssize_t))
        if not keep_c: raise MemoryError()
        cdef Py_ssize_t ki
        try:
            for ki in range(keep_count):
                keep_c[ki] = keep[ki]
            return self._take(keep_c, keep_count)
        finally:
            os_aligned_free(keep_c)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def abs(self):
        return _abs_impl(self)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def round(self, int decimals=0):
        return _round_impl(self, decimals)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def all(self):
        return _all_impl(self)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def any(self):
        return _any_impl(self)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def pct_change(self, int periods=1):
        return _pct_change_impl(self, periods)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def replace(self, to_replace, value):
        return _replace_impl(self, to_replace, value)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def ffill(self):
        return _ffill_impl(self)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def bfill(self):
        return _bfill_impl(self)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def head(self, Py_ssize_t n=5):
        return _head_impl(self, n)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def tail(self, Py_ssize_t n=5):
        return _tail_impl(self, n)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def sample(self, object n=None, object frac=None, bint replace=False, int seed=-1):
        return _sample_impl(self, n, frac, replace, seed)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def nsmallest(self, Py_ssize_t n, str col_name):
        return _nsmallest_impl(self, n, col_name)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def nlargest(self, Py_ssize_t n, str col_name):
        return _nlargest_impl(self, n, col_name)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def pipe(self, object func, *args, **kwargs):
        return _pipe_impl(self, func, args, kwargs)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def assign(self, **kwargs):
        return _assign_impl(self, kwargs)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def select_dtypes(self, list include_types=None, list exclude_types=None):
        return _select_dtypes_impl(self, include_types, exclude_types)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def astype(self, dtype):
        if isinstance(dtype, str):
            dtype_dict = {name: dtype for name in self._columns}
        else:
            dtype_dict = dtype
        return _astype_impl(self, dtype_dict)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def add(self, tuple col_names):
        return _wrap_series('_result', _add_impl(self, col_names), self.index)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sub(self, tuple col_names):
        return _wrap_series('_result', _sub_impl(self, col_names), self.index)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def mul(self, tuple col_names):
        return _wrap_series('_result', _mul_impl(self, col_names), self.index)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def div(self, tuple col_names):
        return _wrap_series('_result', _div_impl(self, col_names), self.index)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def reshape(self, Py_ssize_t rows, Py_ssize_t cols):
        return _reshape_impl(self, rows, cols)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def explode(self, str col_name):
        return _explode_impl(self, col_name)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def stack(self):
        return _stack_impl(self)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def unstack(self):
        return _unstack_impl(self)

    @property
    @cython.binding(False)
    @cython.always_allow_keywords(False)
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

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def euclidean_dist_withcols(self, col1, col2):
        return _metric_dist_withcols_impl(self, col1, col2, PW_EUCLIDEAN)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def manhattan_dist_withcols(self, col1, col2):
        return _metric_dist_withcols_impl(self, col1, col2, PW_MANHATTAN)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cosine_dist_withcols(self, col1, col2):
        return _metric_dist_withcols_impl(self, col1, col2, PW_COSINE_DIST)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cosine_similarity_withcols(self, col1, col2):
        return _metric_dist_withcols_impl(self, col1, col2, PW_COSINE_SIM)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def jaccard_sim_withcols(self, col1, col2):
        return _metric_dist_withcols_impl(self, col1, col2, PW_JACCARD)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def hamming_dist_withcols(self, col1, col2):
        return _metric_dist_withcols_impl(self, col1, col2, PW_HAMMING)

    # Linear Algebra
    @property
    def linalg(self):
        if self._linalg_cache is not None:
            return self._linalg_cache
        self._linalg_cache = _make_linalg_accessor(self)
        return self._linalg_cache

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def dot(self, DataFrame b):
        return _linalg_dot(self, b)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def matmul(self, DataFrame b):
        return _linalg_dot(self, b)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def transpose(self):
        return _linalg_transpose(self)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def det(self):
        return _linalg_det(self)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def inv(self):
        return _linalg_inv(self)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def solve(self, DataFrame b):
        return _linalg_solve(self, b)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def norm(self, str ord_type="frobenius"):
        return _linalg_norm(self, ord_type)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def euclidean_dist_matrix(self):
        return _linalg_pairwise(self, PW_EUCLIDEAN)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def manhattan_dist_matrix(self):
        return _linalg_pairwise(self, PW_MANHATTAN)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cosine_dist_matrix(self):
        return _linalg_pairwise(self, PW_COSINE_DIST)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def hamming_dist_matrix(self):
        return _linalg_pairwise(self, PW_HAMMING)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def jaccard_sim_matrix(self):
        return _linalg_pairwise(self, PW_JACCARD)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cosine_similarity_matrix(self):
        return _linalg_pairwise(self, PW_COSINE_SIM)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def corr(self):
        return _corr_cov_impl(self, True)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cov(self):
        return _corr_cov_impl(self, False)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def corrwith(self, other):
        return _corrwith_impl(self, other)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def corrcoef(self, col_a, col_b):
        return _corrcoef_cov_withcols_impl(self, col_a, col_b, True)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def cov_withcols(self, col_a, col_b):
        return _corrcoef_cov_withcols_impl(self, col_a, col_b, False)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pearson_sim(self):
        return _pearson_sim_impl(self)

    # Factory
    @staticmethod
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def from_dict(data, schema=None, columns=None):
        return _from_dict_impl(data, schema, columns)

    @staticmethod
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def from_list(rows, schema, columns=None):
        return _from_list_impl(rows, schema, columns)

    @staticmethod
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def from_arrays(arrays, schema, columns=None):
        return _from_arrays_impl(arrays, schema, columns)

    @staticmethod
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def arange(n):
        return _arange_impl(n)

    @staticmethod
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def zeros(*args):
        if len(args) == 1:
            return _zeros_impl(args[0], 1)
        return _zeros_impl(args[0], args[1])

    @staticmethod
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def ones(*args):
        if len(args) == 1:
            return _ones_impl(args[0], 1)
        return _ones_impl(args[0], args[1])

    @staticmethod
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def randn(*args):
        if len(args) == 1:
            return _randn_impl(args[0], 1)
        return _randn_impl(args[0], args[1])

    @staticmethod
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def rand(*args):
        if len(args) == 1:
            return _rand_impl(args[0], 1)
        return _rand_impl(args[0], args[1])

    # Rolling
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def rolling(self, Py_ssize_t window, min_periods=None, center=False, axis=0, win_type=None):
        """滚动窗口访问器：返回整表 DataFrame 滚动算子（pandas 式签名）。
        win_type='boxcar'/'triang' 时仅支持 .mean()/.sum()/.var()/.std()（pandas Window 语义）。"""
        if window <= 0:
            raise ValueError(f"Window size must be positive, got {window}")
        if axis != 0 and axis != 'index':
            raise NotImplementedError("rolling supports axis=0 ('index') only")
        return _RollingWindow(self, window, min_periods, center, True, None, None, win_type)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def expanding(self, Py_ssize_t min_periods=1, axis=0):
        """扩展窗口访问器：等价于 rolling(window=行数, min_periods=min_periods)，
        复用 _RollingWindow 逐列滚动（window=max(r, mp) 以支持 mp>r 时全 NaN）。"""
        if axis != 0 and axis != 'index':
            raise NotImplementedError("expanding supports axis=0 ('index') only")
        return _RollingWindow(self, max(self.df_length, min_periods), min_periods, False, True)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def ewm(self, com=None, span=None, halflife=None, alpha=None,
            Py_ssize_t min_periods=0, bint adjust=True, bint ignore_na=False):
        """指数加权窗口访问器（pandas 式）：参数四选一（com/span/halflife/alpha），
        逐数值列应用 .mean()/.sum()/.var(bias)/.std(bias)/.cov(other)/.corr(other)。"""
        return _EWMWindow(self, com, span, halflife, alpha, min_periods,
                          adjust, ignore_na, True)

    # DataFrame-level shift (delegates to cdef _shift_impl)
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def shift(self, Py_ssize_t periods=1, str axis='index'):
        return _shift_impl(self, periods, axis)

    # DataFrame-level diff for all numeric columns
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def diff_all(self, int periods=1):
        return _diff_column_all_impl(self, periods)

    # DataFrame-level rank
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def rank(self, str col_name, str method='average', bint ascending=True, str na_option='keep'):
        return _wrap_series(col_name, _rank_impl(self, col_name, method, ascending, na_option), self.index)

    # DataFrame-level cut
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def cut(self, str col_name, list bins, bint include_lowest=False, str right='right'):
        return _wrap_series(col_name, _cut_impl(self, col_name, bins, include_lowest, right), self.index)

    # DataFrame-level qcut
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def qcut(self, str col_name, int q=4, str labels='True', double precision=3):
        return _wrap_series(col_name, _qcut_impl(self, col_name, q, labels, precision), self.index)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pivot(self, str index, str columns, str values):
        return _pivot_impl(self, index, columns, values)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def melt(self, list id_vars=None, list value_vars=None, str var_name='variable', str value_name='value'):
        return _melt_impl(self, id_vars, value_vars, var_name, value_name)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def expanding_sum(self, str col_name):
        """单列运行和：数值语义与 pandas `expanding(min_periods=1).sum()` 一致
        （NaN 跳过累计，有效观测为 0 时输出 NaN）。"""
        cdef BaseColumn col = self._columns[col_name]
        if col.dtype != TYPE_INT and col.dtype != TYPE_DOUBLE:
            type_name = _TYPE_NAMES.get(col.dtype, 'unknown')
            raise TypeError(f"Expanding sum is not supported for {type_name} columns")
        return _wrap_series(col_name, _expanding_sum_impl(self, col_name), self.index)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def ewm_mean(self, str col_name, double span):
        """单列 EWMA 均值：数值语义与 pandas `ewm(span=span).mean()` 一致
        （adjust=True, min_periods=0，首值等于原值）。"""
        cdef BaseColumn col = self._columns[col_name]
        cdef DoubleColumn res
        cdef double alpha = 2.0 / (span + 1.0)
        if col.dtype == TYPE_INT:
            res = (<IntColumn>col).ewm_mean(alpha, 0, True, False)
        elif col.dtype == TYPE_DOUBLE:
            res = (<DoubleColumn>col).ewm_mean(alpha, 0, True, False)
        else:
            type_name = _TYPE_NAMES.get(col.dtype, 'unknown')
            raise TypeError(f"EWM mean is not supported for {type_name} columns")
        return _wrap_series(col_name, res, self.index)


# Python-callable wrapper for _column_from_list_impl
@cython.binding(False)
@cython.always_allow_keywords(False)
def column_from_list(DataFrame df, str name, list values):
    """Python-callable wrapper for _column_from_list_impl"""
    return _column_from_list_impl(df, name, values)


# ============================================================
#  C 底层结构体 / 枚举类型（保留于 pyx）
#  只为被三个 accessor 类在 cdef 字段中直接引用：移动后的所有函数
#  实现（含 C 哈希表 / 动态数组 / 聚合）已迁至 _databaseframe_helper.pxi。
# ============================================================
cdef struct _RankItem:
    Py_ssize_t idx
    double val

cdef enum _AggFunc:
    _AGG_SUM = 0
    _AGG_MEAN = 1
    _AGG_MIN = 2
    _AGG_MAX = 3
    _AGG_VAR = 4
    _AGG_STD = 5

cdef struct _GroupHashEntry:
    uint64_t hash
    int group_id
    bint occupied

cdef struct _GroupHashTable:
    _GroupHashEntry* entries
    Py_ssize_t capacity
    Py_ssize_t size

cdef struct _GroupRows:
    Py_ssize_t** rows
    Py_ssize_t* row_sizes
    Py_ssize_t* row_capacities
    Py_ssize_t count

cdef struct _ByColPtrs:
    void** data
    void** offsets
    int* types
    int* cols            # 每列子列数：2D 列 = cols，其余 = 1
    Py_ssize_t* strides  # 每列行步长（2D 列 data_c[sub*stride + row]），1D 列 = length
    int count

# ============================================================
# _DataFrameGroupByAccessor — lazy-cached accessor
# ============================================================

cdef class _DataFrameGroupByAccessor:

    def __cinit__(self):
        pass

    def __dealloc__(self):
        if self._ht != NULL: _ght_free(self._ht)
        if self._gr != NULL: _group_rows_free(self._gr)
        if self._ptrs.data != NULL: _free_by_col_ptrs(self._ptrs)

    cdef void _ensure_built(self):
        if self._built:
            return
        self._ht = _ght_new(max(self._df.df_length * 2, 16))
        self._gr = _group_rows_new()
        self._ptrs = _extract_by_col_ptrs(self._df, self._by_cols)
        _gb_build_groups(self._df, self._by_cols, self._ht, self._gr, self._ptrs)
        self._built = True

    cdef DataFrame _result_shell(self):
        """构造结果 DataFrame 骨架：列数由聚合行决定，并挂载所有分组键列。"""
        cdef DataFrame res = DataFrame()
        res.df_length = self._gr.count
        self._attach_keys(res)
        return res

    cdef void _attach_keys(self, DataFrame res):
        """把分组键列挂到结果上（若聚合结果中已存在同名列则跳过）。"""
        cdef str cn
        for cn in self._by_cols:
            if cn not in res._columns:
                res._columns[cn] = _gb_extract_key_column(self._df, self._gr, cn)

    cdef list _target_cols(self):
        """计算需要聚合的目标列：所有非分组键的数值列名。"""
        cdef list target = []
        cdef str cn
        cdef BaseColumn c
        for cn in self._df._columns:
            if cn not in self._by_cols:
                c = self._df._columns[cn]
                if c.dtype == TYPE_INT or c.dtype == TYPE_DOUBLE:
                    target.append(cn)
        return target

    cdef DataFrame _idx_agg(self, str col_name, bint find_max):
        """idxmax/idxmin 共用的 C 级分派壳。"""
        self._ensure_built()
        cdef DataFrame res = self._result_shell()
        cdef str cn
        if col_name is not None:
            res._columns[col_name] = _gb_native_idx(self._df, self._gr, col_name, find_max)
        else:
            for cn in self._target_cols():
                res._columns[cn] = _gb_native_idx(self._df, self._gr, cn, find_max)
        return res

    cdef object _dispatch_agg(self, str col_name, str func_name):
        """把聚合函数名映射到 C 级 / Python 级实现（count 特殊为计数字段）。"""
        if func_name in ('sum', 'mean', 'min', 'max', 'var', 'std'):
            return _gb_native_agg(self._df, self._gr, col_name, func_name)
        elif func_name == 'count':
            return _gb_native_count(self._gr)
        else:
            return _gb_python_agg(self._df, self._gr, col_name, func_name)

    cdef DataFrame _convenience_agg(self, str func_name, object col_name):
        self._ensure_built()
        cdef DataFrame res
        cdef dict agg
        cdef str cn
        if col_name is not None:
            res = self._result_shell()
            res._columns[col_name] = self._dispatch_agg(col_name, func_name)
            return res
        else:
            agg = {}
            for cn in self._target_cols():
                agg[cn] = func_name
            return self.agg(agg)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def size(self):
        self._ensure_built()
        return self._gr.count

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def agg(self, dict agg_dict):
        self._ensure_built()
        cdef DataFrame res = self._result_shell()
        cdef str col_name, func_name
        for col_name, func_name in agg_dict.items():
            res._columns[col_name] = self._dispatch_agg(col_name, func_name)
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def sum(self, str col_name=None):
        return self._convenience_agg('sum', col_name)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def mean(self, str col_name=None):
        return self._convenience_agg('mean', col_name)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def count(self):
        self._ensure_built()
        cdef DataFrame res = self._result_shell()
        res._columns['__count__'] = _gb_native_count(self._gr)
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def min(self, str col_name=None):
        return self._convenience_agg('min', col_name)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def max(self, str col_name=None):
        return self._convenience_agg('max', col_name)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def median(self, str col_name=None):
        return self._convenience_agg('median', col_name)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def quantile(self, str col_name=None):
        return self._convenience_agg('quantile', col_name)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def var(self, str col_name=None):
        return self._convenience_agg('var', col_name)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def std(self, str col_name=None):
        return self._convenience_agg('std', col_name)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def apply(self, object func):
        self._ensure_built()
        cdef list sub_dfs = []
        cdef int g_id
        cdef Py_ssize_t sz
        cdef Py_ssize_t* indices_c
        cdef DataFrame sub_df
        cdef object func_res

        for g_id in range(self._gr.count):
            sz = self._gr.row_sizes[g_id]
            indices_c = <Py_ssize_t*>aligned_malloc(max(sz, 1) * sizeof(Py_ssize_t))
            if not indices_c: raise MemoryError()
            for i in range(sz):
                indices_c[i] = self._gr.rows[g_id][i]
            sub_df = self._df._take(indices_c, sz)
            os_aligned_free(indices_c)
            func_res = func(sub_df)
            sub_dfs.append(func_res)

        if len(sub_dfs) == 0:
            return DataFrame()
        if len(sub_dfs) == 1:
            return sub_dfs[0]

        cdef DataFrame result = sub_dfs[0]
        for i in range(1, len(sub_dfs)):
            result = result.concat(sub_dfs[i])
        return result

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def idxmax(self, str col_name=None):
        return self._idx_agg(col_name, True)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def idxmin(self, str col_name=None):
        return self._idx_agg(col_name, False)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def transform(self, object func):
        self._ensure_built()
        cdef Py_ssize_t n = self._df.df_length
        cdef DataFrame result = DataFrame()
        result.df_length = n
        result.index = self._df.index

        cdef str cn
        cdef list col_names = []
        cdef dict col_values = {}

        for cn in self._df._columns:
            col_names.append(cn)
            col_values[cn] = [None] * n

        cdef int g_id
        cdef Py_ssize_t sz, row_idx, i

        for g_id in range(self._gr.count):
            sz = self._gr.row_sizes[g_id]
            indices_c = <Py_ssize_t*>aligned_malloc(max(sz, 1) * sizeof(Py_ssize_t))
            if not indices_c: raise MemoryError()
            try:
                for i in range(sz):
                    indices_c[i] = self._gr.rows[g_id][i]
                sub_df = self._df._take(indices_c, sz)
                transformed = func(sub_df)
                if isinstance(transformed, DataFrame):
                    for cn in transformed._columns:
                        if cn not in col_values:
                            col_names.append(cn)
                            col_values[cn] = [None] * n
                    for cn in col_names:
                        if cn in transformed._columns:
                            src_col = transformed._columns[cn]
                            for i in range(sz):
                                row_idx = self._gr.rows[g_id][i]
                                col_values[cn][row_idx] = src_col[i]
            finally:
                os_aligned_free(indices_c)

        for cn in col_names:
            values = col_values[cn]
            nv = len(values)
            if nv == 0:
                result._columns[cn] = IntColumn(0)
                continue
            sample = None
            for vi in range(nv):
                if values[vi] is not None:
                    sample = values[vi]
                    break
            if sample is None:
                dbl = DoubleColumn(max(nv, 1))
                dbl.length = nv
                for vi in range(nv):
                    dbl.data_c[vi] = NAN
                result._columns[cn] = dbl
            elif isinstance(sample, bool):
                bc = BoolColumn(max(nv, 1))
                bc.length = nv
                for vi in range(nv):
                    bc.data_c[vi] = 1 if values[vi] else 0
                result._columns[cn] = bc
            elif isinstance(sample, int):
                ic = IntColumn(max(nv, 1))
                ic.length = nv
                for vi in range(nv):
                    ic.data_c[vi] = <int>(values[vi] if values[vi] is not None else 0)
                result._columns[cn] = ic
            elif isinstance(sample, float):
                dc = DoubleColumn(max(nv, 1))
                dc.length = nv
                for vi in range(nv):
                    dc.data_c[vi] = <double>values[vi] if values[vi] is not None else NAN
                result._columns[cn] = dc
            elif isinstance(sample, str):
                ac = AsciiColumn(max(nv, 1))
                for vi in range(nv):
                    ac.append(str(values[vi] if values[vi] is not None else ''))
                result._columns[cn] = ac

        return result

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def filter(self, object func):
        self._ensure_built()
        cdef DataFrame result = DataFrame()
        cdef list keep_rows = []
        cdef int g_id
        cdef Py_ssize_t sz, i

        for g_id in range(self._gr.count):
            sz = self._gr.row_sizes[g_id]
            indices_c = <Py_ssize_t*>aligned_malloc(max(sz, 1) * sizeof(Py_ssize_t))
            if not indices_c: raise MemoryError()
            try:
                for i in range(sz):
                    indices_c[i] = self._gr.rows[g_id][i]
                sub_df = self._df._take(indices_c, sz)
                if func(sub_df):
                    for i in range(sz):
                        keep_rows.append(self._gr.rows[g_id][i])
            finally:
                os_aligned_free(indices_c)

        if len(keep_rows) == 0:
            result.df_length = 0
            result.index = RangeIndex(0, 0)
            return result

        cdef Py_ssize_t* keep_c = <Py_ssize_t*>aligned_malloc(max(len(keep_rows), 1) * sizeof(Py_ssize_t))
        if not keep_c: raise MemoryError()
        try:
            for i in range(len(keep_rows)):
                keep_c[i] = keep_rows[i]
            return self._df._take(keep_c, len(keep_rows))
        finally:
            os_aligned_free(keep_c)

    @cython.binding(False)
    def __repr__(self):
        self._ensure_built()
        return f"<GroupBy: {self._gr.count} groups, by={self._by_cols}>"



# ============================================================
# _DataFrameLinalgAccessor — lazy-cached accessor
# ============================================================

cdef class _DataFrameLinalgAccessor:

    def __cinit__(self):
        pass

    # ==================== Matrix Operations ====================

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def dot(self, DataFrame b):
        return _linalg_dot(self._df, b)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def matmul(self, DataFrame b):
        return _linalg_dot(self._df, b)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def transpose(self):
        return _linalg_transpose(self._df)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def det(self):
        return _linalg_det(self._df)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def inv(self):
        return _linalg_inv(self._df)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def solve(self, DataFrame b):
        return _linalg_solve(self._df, b)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def norm(self, str ord_type="frobenius"):
        return _linalg_norm(self._df, ord_type)

    # ==================== Pairwise Distance ====================

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def euclidean_dist(self):
        return _linalg_pairwise(self._df, PW_EUCLIDEAN)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def manhattan_dist(self):
        return _linalg_pairwise(self._df, PW_MANHATTAN)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def cosine_dist(self):
        return _linalg_pairwise(self._df, PW_COSINE_DIST)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def hamming_dist(self):
        return _linalg_pairwise(self._df, PW_HAMMING)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def jaccard_sim(self):
        return _linalg_pairwise(self._df, PW_JACCARD)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def cosine_similarity(self):
        return _linalg_pairwise(self._df, PW_COSINE_SIM)

    @cython.binding(False)
    def __repr__(self):
        return f"<LinalgAccessor: {len(self._df._columns)} columns, {self._df.df_length} rows>"


# ============================================================
#  _DataFrameMetricAccessor（懒加载 metric 访问器）
# ============================================================

# _DataFrameMetricAccessor — 懒加载缓存的 metric 访问器
# ============================================================
# 用法: df.metric.euclidean('a', 'b') / df.metric.corr() 等。
# 实例由 DataFrame.metric property 首次访问时创建并缓存，
# 实例化开销均摊为 0（参见 _make_metric_accessor）。
cdef class _DataFrameMetricAccessor:
    """Metric accessor with lazy caching on the parent DataFrame."""

    def __cinit__(self):
        # 无参 __cinit__：允许工厂函数直接 __new__ 绕过参数打包
        pass

    # ==================== Distance ====================

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def euclidean(self, col1, col2):
        return _metric_dist_withcols_impl(self._df, col1, col2, PW_EUCLIDEAN)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def manhattan(self, col1, col2):
        return _metric_dist_withcols_impl(self._df, col1, col2, PW_MANHATTAN)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def cosine_dist(self, col1, col2):
        return _metric_dist_withcols_impl(self._df, col1, col2, PW_COSINE_DIST)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def cosine_similarity(self, col1, col2):
        return _metric_dist_withcols_impl(self._df, col1, col2, PW_COSINE_SIM)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def jaccard(self, col1, col2):
        return _metric_dist_withcols_impl(self._df, col1, col2, PW_JACCARD)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def hamming(self, col1, col2):
        return _metric_dist_withcols_impl(self._df, col1, col2, PW_HAMMING)

    # ==================== Correlation / Covariance ====================

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def corr(self):
        return _corr_cov_impl(self._df, True)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def cov(self):
        return _corr_cov_impl(self._df, False)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def corrwith(self, other):
        return _corrwith_impl(self._df, other)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def corrcoef(self, col_a, col_b):
        return _corrcoef_cov_withcols_impl(self._df, col_a, col_b, True)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def cov_withcols(self, col_a, col_b):
        return _corrcoef_cov_withcols_impl(self._df, col_a, col_b, False)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def pearson(self):
        return _pearson_sim_impl(self._df)


# ============================================================
#  Standalone 公开 API（from_pickle / read_csv* / concat* / to_* / where）
# ============================================================

import pickle as _pickle
import csv as _csv
from fluidframe._schema_types import Ascii as _Ascii, Date as _Date, Bool as _Bool, \
    Category as _Category, DateTime as _DateTime, Time as _Time, _parse_schema as _parse_schema


class _SafeUnpickler(_pickle.Unpickler):
    """Safe unpickler that restricts deserialization to built-in types."""
    _ALLOWED_BUILTINS = frozenset({
        'dict', 'list', 'tuple', 'int', 'float', 'str', 'bytes',
        'NoneType', 'bool', 'set',
    })

    def __init__(self, file, allow_raw_pickle=False):
        super().__init__(file)
        self._allow_raw_pickle = allow_raw_pickle

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def find_class(self, module, name):
        if self._allow_raw_pickle:
            return super().find_class(module, name)
        if module == 'builtins' and name in self._ALLOWED_BUILTINS:
            return super().find_class(module, name)
        full = f"{module}.{name}"
        raise _pickle.UnpicklingError(
            f"Deserialization of '{full}' is not allowed. "
            f"Use allow_raw_pickle=True to bypass (unsafe)."
        )


@cython.binding(False)
@cython.always_allow_keywords(False)
def from_pickle(filename, bint allow_raw_pickle=False):
    """从 pickle 文件恢复 DataFrame。
    2D 列条目为 (typ, values, rows, cols)，重建后保持二维维度。"""
    cdef Py_ssize_t rows = 0, cols = 0
    cdef Py_ssize_t total, i
    cdef IntColumn icol
    cdef DoubleColumn fcol
    cdef DateColumn dcol
    cdef DateTimeColumn tcol
    cdef TimeColumn qcol
    cdef int32_t* dptr
    cdef int64_t* tptr
    with open(filename, 'rb') as f:
        data = _SafeUnpickler(f, allow_raw_pickle).load()
    df = DataFrame()
    df.df_length = data['df_length']
    for col_name, entry in data['columns'].items():
        if len(entry) == 4:
            typ, values, rows, cols = entry
        else:
            typ, values = entry
        is_2d = cols > 1
        total = len(values)
        if typ == 'int':
            if is_2d:
                col = IntColumn.from_2d(<object>values, rows, cols)
            else:
                col = IntColumn()
                for v in values:
                    col.append(v)
        elif typ == 'float':
            if is_2d:
                col = DoubleColumn.from_2d(<object>values, rows, cols)
            else:
                col = DoubleColumn()
                for v in values:
                    col.append(v)
        elif typ == 'ascii':
            col = AsciiColumn()
            for v in values:
                col.append(v)
        elif typ == 'ucs2':
            col = UCS2Column()
            for v in values:
                col.append(v)
        elif typ == 'date':
            if is_2d:
                if total != rows * cols:
                    raise ValueError(f"DateColumn 2D size mismatch: expected {rows * cols}, got {total}")
                dcol = DateColumn(max(total, 1), cols=cols)
                dcol.length = total
                dcol.num_cols = <int>cols
                dcol.is_2d = True
                dcol.owns_data = True
                dptr = <int32_t*>dcol.data_c
                val_list = <list>values if type(values) is list else list(values)
                for i in range(total):
                    dptr[i] = <int32_t>val_list[i]
                col = dcol
            else:
                col = DateColumn()
                for v in values:
                    col.append(v)
        elif typ == 'bool':
            col = BoolColumn()
            for v in values:
                col.append(v)
        elif typ == 'category':
            col = CategoryColumn()
            for v in values:
                col.append(v)
        elif typ == 'datetime':
            if is_2d:
                if total != rows * cols:
                    raise ValueError(f"DateTimeColumn 2D size mismatch: expected {rows * cols}, got {total}")
                tcol = DateTimeColumn(max(total, 1), cols=cols)
                tcol.length = total
                tcol.num_cols = <int>cols
                tcol.is_2d = True
                tcol.owns_data = True
                tptr = <int64_t*>tcol.data_c
                val_list = <list>values if type(values) is list else list(values)
                for i in range(total):
                    tptr[i] = <int64_t>val_list[i]
                col = tcol
            else:
                col = DateTimeColumn()
                for v in values:
                    col.append(v)
        elif typ == 'time':
            if is_2d:
                if total != rows * cols:
                    raise ValueError(f"TimeColumn 2D size mismatch: expected {rows * cols}, got {total}")
                qcol = TimeColumn(max(total, 1), cols=cols)
                qcol.length = total
                qcol.num_cols = <int>cols
                qcol.is_2d = True
                qcol.owns_data = True
                dptr = <int32_t*>qcol.data_c
                val_list = <list>values if type(values) is list else list(values)
                for i in range(total):
                    dptr[i] = <int32_t>val_list[i]
                col = qcol
            else:
                col = TimeColumn()
                for v in values:
                    col.append(v)
        df._columns[col_name] = col
    return df


@cython.binding(False)
@cython.always_allow_keywords(False)
def read_csv(filename, dict schema, sep=','):
    """读取 CSV 文件，返回 DataFrame。必须提供 schema 指定列类型"""
    if not schema:
        raise ValueError("schema is required and must not be empty")
    cdef dict parsed = _parse_schema(schema)
    with open(filename, 'r', encoding='utf-8') as f:
        reader = _csv.reader(f, delimiter=sep)
        headers = next(reader)
        rows = list(reader)

    cdef Py_ssize_t n_rows = len(rows)
    cdef Py_ssize_t n_cols = len(headers)
    cdef Py_ssize_t i, j
    df = DataFrame()

    if n_rows == 0:
        for j in range(n_cols):
            name = headers[j]
            dtype = parsed.get(name, str)
            if dtype == TYPE_INT:
                df._columns[name] = IntColumn()
            elif dtype == TYPE_DOUBLE:
                df._columns[name] = DoubleColumn()
            elif dtype == TYPE_ASCII:
                df._columns[name] = AsciiColumn()
            elif dtype == TYPE_DATE:
                df._columns[name] = DateColumn()
            elif dtype == TYPE_BOOL:
                df._columns[name] = BoolColumn()
            elif dtype == TYPE_CATEGORY:
                df._columns[name] = CategoryColumn()
            elif dtype == TYPE_DATETIME:
                df._columns[name] = DateTimeColumn()
            elif dtype == TYPE_TIME:
                df._columns[name] = TimeColumn()
            else:
                df._columns[name] = UCS2Column()
        return df

    for j, name in enumerate(headers):
        py_type = parsed.get(name, str)
        if py_type == TYPE_INT:
            col = IntColumn(n_rows)
            for i in range(n_rows):
                val = rows[i][j].strip()
                col.append(int(val) if val else 0)
        elif py_type == TYPE_DOUBLE:
            col = DoubleColumn(n_rows)
            for i in range(n_rows):
                val = rows[i][j].strip()
                col.append(float(val) if val else NAN)
        elif py_type == TYPE_ASCII:
            col = AsciiColumn(n_rows)
            for i in range(n_rows):
                col.append(rows[i][j])
        elif py_type == TYPE_DATE:
            col = DateColumn(n_rows)
            for i in range(n_rows):
                col.append(rows[i][j].strip())
        elif py_type == TYPE_BOOL:
            col = BoolColumn(n_rows)
            for i in range(n_rows):
                val = rows[i][j].strip().lower()
                col.append(val in ('true', '1', 'yes'))
        elif py_type == TYPE_CATEGORY:
            col = CategoryColumn(n_rows)
            for i in range(n_rows):
                col.append(rows[i][j])
        elif py_type == TYPE_DATETIME:
            col = DateTimeColumn(n_rows)
            for i in range(n_rows):
                col.append(rows[i][j].strip())
        elif py_type == TYPE_TIME:
            col = TimeColumn(n_rows)
            for i in range(n_rows):
                col.append(rows[i][j].strip())
        else:
            col = UCS2Column(n_rows)
            for i in range(n_rows):
                col.append(rows[i][j])
        df._columns[name] = col
    df.df_length = n_rows
    return df


@cython.binding(False)
@cython.always_allow_keywords(False)
def read_tsv(filename, dict schema):
    """读取 TSV 文件（tab 分隔），返回 DataFrame"""
    return read_csv(filename, schema=schema, sep='\t')


@cython.binding(False)
@cython.always_allow_keywords(False)
def read_json(filename, dict schema):
    """读取 JSON 文件（records 格式），返回 DataFrame"""
    import json as _json
    if not schema:
        raise ValueError("schema is required and must not be empty")
    cdef dict parsed = _parse_schema(schema)
    with open(filename, 'r', encoding='utf-8') as f:
        records = _json.load(f)
    if not isinstance(records, list):
        raise ValueError("JSON must be a list of records (records orient)")
    cdef Py_ssize_t n_rows = len(records)
    cdef list headers = list(parsed.keys())
    cdef Py_ssize_t n_cols = len(headers)
    df = DataFrame()
    cdef Py_ssize_t i, j

    for j, name in enumerate(headers):
        py_type = parsed.get(name, str)
        if py_type == TYPE_INT:
            col = IntColumn(max(n_rows, 1))
            col.length = n_rows
            for i in range(n_rows):
                col.data_c[i] = <int>records[i].get(name, 0)
            df._columns[name] = col
        elif py_type == TYPE_DOUBLE:
            col_d = DoubleColumn(max(n_rows, 1))
            col_d.length = n_rows
            for i in range(n_rows):
                val = records[i].get(name)
                col_d.data_c[i] = <double>val if val is not None else NAN
            df._columns[name] = col_d
        elif py_type == TYPE_ASCII:
            col_a = AsciiColumn(max(n_rows, 1))
            for i in range(n_rows):
                col_a.append(str(records[i].get(name, '')))
            df._columns[name] = col_a
        elif py_type == TYPE_DATE:
            col_dt = DateColumn(max(n_rows, 1))
            col_dt.length = n_rows
            for i in range(n_rows):
                col_dt.data_c[i] = <int>records[i].get(name, 0)
            df._columns[name] = col_dt
        elif py_type == TYPE_BOOL:
            col_b = BoolColumn(max(n_rows, 1))
            col_b.length = n_rows
            for i in range(n_rows):
                col_b.data_c[i] = 1 if records[i].get(name, False) else 0
            df._columns[name] = col_b
        else:
            col_u = UCS2Column(max(n_rows, 1))
            for i in range(n_rows):
                col_u.append(str(records[i].get(name, '')))
            df._columns[name] = col_u

    df.df_length = n_rows
    df.index = RangeIndex(0, n_rows)
    return df


@cython.binding(False)
@cython.always_allow_keywords(False)
def concat_outer(list dfs):
    """纵向拼接多个 DataFrame（外连接：取所有列的并集），重置索引。
    2D 列：使用列级 concat 保持二维维度结构不变。"""
    if not dfs: return DataFrame()
    all_cols = set()
    for df in dfs:
        all_cols.update(df._columns.keys())
    all_cols = list(all_cols)

    result = DataFrame()
    col_refs = {}
    for name in all_cols:
        for df in dfs:
            if name in df._columns:
                col_refs[name] = df._columns[name]
                break

    for name in all_cols:
        result._columns[name] = _empty_col_for(col_refs[name])

    for df in dfs:
        for name in all_cols:
            if name in df._columns:
                result._columns[name].concat(df._columns[name])
            else:
                ref = col_refs[name]
                is2d, c, r = _col_shape_probe(ref)
                fill_rows = df.df_length
                if fill_rows == 0:
                    continue
                total = fill_rows * c if (is2d and c > 1) else fill_rows
                if ref.dtype == TYPE_INT:
                    fill_col = IntColumn()
                    for _ in range(total): fill_col.append(0)
                elif ref.dtype == TYPE_DOUBLE:
                    fill_col = DoubleColumn()
                    for _ in range(total): fill_col.append(NAN)
                elif ref.dtype == TYPE_DATE:
                    fill_col = DateColumn()
                    for _ in range(total): fill_col.append(0)
                elif ref.dtype == TYPE_DATETIME:
                    fill_col = DateTimeColumn()
                    for _ in range(total): fill_col.append(0)
                elif ref.dtype == TYPE_TIME:
                    fill_col = TimeColumn()
                    for _ in range(total): fill_col.append(0)
                elif ref.dtype == TYPE_BOOL:
                    fill_col = BoolColumn()
                    for _ in range(total): fill_col.append(False)
                elif ref.dtype == TYPE_ASCII:
                    fill_col = AsciiColumn()
                    for _ in range(total): fill_col.append('')
                elif ref.dtype == TYPE_UCS2:
                    fill_col = UCS2Column()
                    for _ in range(total): fill_col.append('')
                elif ref.dtype == TYPE_CATEGORY:
                    fill_col = CategoryColumn()
                    for _ in range(total): fill_col.append(None)
                if is2d and c > 1:
                    _set_2d_shape(fill_col, c)
                result._columns[name].concat(fill_col)

        result.df_length += df.df_length

    result.reset_index()
    return result


@cython.binding(False)
@cython.always_allow_keywords(False)
def concat_inner(list dfs):
    """纵向拼接多个 DataFrame（内连接：仅取所有列的交集），重置索引。
    2D 列：使用列级 concat 保持二维维度结构不变。"""
    if not dfs: return DataFrame()
    common = set(dfs[0]._columns.keys())
    for df in dfs[1:]:
        common &= set(df._columns.keys())
    common = list(common)

    result = DataFrame()
    for name in common:
        result._columns[name] = _empty_col_for(dfs[0]._columns[name])

    for df in dfs:
        for name in common:
            result._columns[name].concat(df._columns[name])
        result.df_length += df.df_length

    result.reset_index()
    return result


@cython.binding(False)
@cython.always_allow_keywords(False)
def concat(list dfs, bint ignore_index=False):
    cdef DataFrame result = concat_outer(dfs)
    if ignore_index:
        result.index = RangeIndex(0, result.df_length)
    return result


@cython.binding(False)
@cython.always_allow_keywords(False)
def to_numeric(column, errors='coerce'):
    """Convert column to numeric values. Like pd.to_numeric.
    2D 输入保持同维度输出（is_2d/rows/cols 不变，逐单元转换）。"""
    cdef Py_ssize_t n = column.length
    cdef bint is2d
    cdef Py_ssize_t c = 1, r
    is2d, c, r = _col_shape_probe(column)

    cdef bint is_2d_c = is2d and c > 1
    cdef Py_ssize_t alloc_size = (r * c) if is_2d_c else r
    cdef DoubleColumn res = DoubleColumn(max(alloc_size, 1))
    res.length = alloc_size
    if is_2d_c:
        res.num_cols = <int>c
        res.is_2d = True
        res.owns_data = True

    cdef Py_ssize_t i, j, idx = 0
    cdef bint raise_err = (errors == 'raise')

    if is_2d_c:
        # 2D 扁平缓冲为列主序（先 col0 全部行，再 col1…）：cell = column[j * r + i]
        for j in range(c):
            for i in range(r):
                idx = j * r + i
                s = str(column[idx])
                try:
                    res.data_c[idx] = float(s)
                except (ValueError, TypeError):
                    if raise_err:
                        raise ValueError(f"Could not convert '{s}' to numeric")
                    res.data_c[idx] = NAN
                idx += 1
    else:
        for i in range(r):
            s = str(column[i])
            try:
                res.data_c[i] = float(s)
            except (ValueError, TypeError):
                if raise_err:
                    raise ValueError(f"Could not convert '{s}' to numeric")
                res.data_c[i] = NAN
    return res


@cython.binding(False)
@cython.always_allow_keywords(False)
def to_datetime(column, str fmt='', errors='coerce'):
    """Convert column to DateTimeColumn. Like pd.to_datetime.
    2D 输入保持同维度输出。"""
    cdef bint is2d
    cdef Py_ssize_t c = 1, r
    is2d, c, r = _col_shape_probe(column)

    cdef bint is_2d_c = is2d and c > 1
    cdef Py_ssize_t alloc = r * c if is_2d_c else r
    cdef DateTimeColumn res = DateTimeColumn(max(alloc, 1))
    cdef bint raise_err = (errors == 'raise')
    cdef Py_ssize_t i, j, idx = 0

    if is_2d_c:
        res.length = alloc
        res.num_cols = <int>c
        res.is_2d = True
        res.owns_data = True
        for j in range(c):
            for i in range(r):
                idx = j * r + i
                s = str(column[idx])
                try:
                    res.data_c[idx] = _parse_datetime_str(s)
                except (ValueError, TypeError):
                    if raise_err:
                        raise ValueError(f"Could not convert '{s}' to datetime")
                    res.data_c[idx] = 0
    else:
        for i in range(r):
            s = str(column[i])
            try:
                res.append(s)
            except (ValueError, TypeError):
                if raise_err:
                    raise ValueError(f"Could not convert '{s}' to datetime")
                res.append(0)
    return res


@cython.binding(False)
@cython.always_allow_keywords(False)
def to_date(column, str fmt='', errors='coerce'):
    """Convert column to DateColumn (YYYYMMDD).
    2D 输入保持同维度输出。"""
    cdef bint is2d
    cdef Py_ssize_t c = 1, r
    is2d, c, r = _col_shape_probe(column)

    cdef bint is_2d_c = is2d and c > 1
    cdef Py_ssize_t alloc = r * c if is_2d_c else r
    cdef DateColumn res = DateColumn(max(alloc, 1))
    cdef bint raise_err = (errors == 'raise')
    cdef Py_ssize_t i, j, idx = 0

    if is_2d_c:
        res.length = alloc
        res.num_cols = <int>c
        res.is_2d = True
        res.owns_data = True
        for j in range(c):
            for i in range(r):
                idx = j * r + i
                s = str(column[idx])
                try:
                    res.data_c[idx] = _parse_date_str(s)
                except (ValueError, TypeError):
                    if raise_err:
                        raise ValueError(f"Could not convert '{s}' to date")
                    res.data_c[idx] = 0
    else:
        for i in range(r):
            s = str(column[i])
            try:
                res.append(s)
            except (ValueError, TypeError):
                if raise_err:
                    raise ValueError(f"Could not convert '{s}' to date")
                res.append(0)
    return res


@cython.binding(False)
@cython.always_allow_keywords(False)
def to_time(column, str fmt='', errors='coerce'):
    """Convert column to TimeColumn (seconds since midnight).
    2D 输入保持同维度输出。"""
    cdef bint is2d
    cdef Py_ssize_t c = 1, r
    is2d, c, r = _col_shape_probe(column)

    cdef bint is_2d_c = is2d and c > 1
    cdef Py_ssize_t alloc = r * c if is_2d_c else r
    cdef TimeColumn res = TimeColumn(max(alloc, 1))
    cdef bint raise_err = (errors == 'raise')
    cdef Py_ssize_t i, j, idx = 0

    if is_2d_c:
        res.length = alloc
        res.num_cols = <int>c
        res.is_2d = True
        res.owns_data = True
        for j in range(c):
            for i in range(r):
                idx = j * r + i
                s = str(column[idx])
                try:
                    res.data_c[idx] = _parse_time_str(s)
                except (ValueError, TypeError):
                    if raise_err:
                        raise ValueError(f"Could not convert '{s}' to time")
                    res.data_c[idx] = 0
    else:
        for i in range(r):
            s = str(column[i])
            try:
                res.append(s)
            except (ValueError, TypeError):
                if raise_err:
                    raise ValueError(f"Could not convert '{s}' to time")
                res.append(0)
    return res


@cython.binding(False)
@cython.always_allow_keywords(False)
def to_timedelta(values, str unit='D'):
    """Create a column of timedelta-like values (stored as int days). Like pd.to_timedelta."""
    n = len(values)
    res = IntColumn(n)
    res.length = n
    if unit == 'D':
        factor = 1.0
    elif unit == 'h':
        factor = 1.0 / 24.0
    elif unit == 'm':
        factor = 1.0 / 1440.0
    elif unit == 's':
        factor = 1.0 / 86400.0
    else:
        raise ValueError(f"Unsupported unit: '{unit}'")
    for i in range(n):
        res.data_c[i] = <int>(<double>values[i] * factor)
    return res


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.binding(False)
@cython.always_allow_keywords(False)
def where(IntColumn cond, object true_val, object false_val):
    """numpy.where 等效替代。根据 cond 掩码列选择填充 true_val 或 false_val。"""
    cdef Py_ssize_t n = cond.length
    cdef Py_ssize_t i
    cdef int* cond_ptr = cond.data_c
    cdef double v_t, v_f
    cdef double* res_ptr
    cdef DoubleColumn res_dbl
    cdef IntColumn inv
    cdef int* inv_ptr

    if isinstance(true_val, str) and isinstance(false_val, str):
        res_str = UCS2Column(max(n, 1))
        res_str.length = n
        for i in range(n):
            if cond_ptr[i]:
                res_str._set_item(i, true_val)
            else:
                res_str._set_item(i, false_val)
        return res_str

    elif isinstance(true_val, (int, float)) and isinstance(false_val, (int, float)):
        res_dbl = DoubleColumn(max(n, 1))
        res_dbl.length = n
        v_t = <double>true_val
        v_f = <double>false_val
        res_ptr = res_dbl.data_c
        with nogil:
            for i in range(n):
                res_ptr[i] = v_t if cond_ptr[i] else v_f
        return res_dbl

    elif hasattr(true_val, 'where_mask'):
        return true_val.where_mask(cond, false_val)

    elif hasattr(false_val, 'where_mask'):
        inv = IntColumn(max(n, 1))
        inv.length = n
        inv_ptr = inv.data_c
        with nogil:
            for i in range(n):
                inv_ptr[i] = 0 if cond_ptr[i] else 1
        return false_val.where_mask(inv, true_val)

    raise TypeError(f"Unsupported types for where: {type(true_val)}, {type(false_val)}")

# ============================================================
#  _databaseframe_helper.pxi（模块级 cdef/cpdef 辅助函数全集）
# ============================================================
include "_databaseframe_helper.pxi"
