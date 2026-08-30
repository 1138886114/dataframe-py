# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport cython
from libc.string cimport memcpy, memmove, strncmp, memset, strncpy
from libc.stdint cimport uint32_t
from libc.time cimport time_t, time
from libc.math cimport NAN, sqrt, log, cos
from cpython.unicode cimport Py_UCS2, PyUnicode_KIND, PyUnicode_READ_CHAR, PyUnicode_2BYTE_KIND, PyUnicode_DATA

cdef extern from *:
    """
    #include <stdlib.h>
    #include <time.h>
    #include <math.h>

    // 使用跨平台宏定义线程局部变量，防范 nogil 下的多线程数据竞争
    #if defined(_MSC_VER) || defined(_WIN32)
        #define THREAD_LOCAL __declspec(thread)
    #else
        #define THREAD_LOCAL __thread
    #endif

    // 赋予非零的默认状态，防止用户未调用 seed() 时产生全 0 死锁
    static THREAD_LOCAL unsigned long long _xorshift128plus_state[2] = {
        0x8a5cd789635d2dffULL,
        0x121fd2155c472f96ULL
    };

    static inline void _xorshift128plus_seed(unsigned long long seed) {
        // 使用 SplitMix64 算法处理种子，避免相近种子产生相近序列
        unsigned long long z = (seed + 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        _xorshift128plus_state[0] = z ^ (z >> 31);

        z = (seed + 0x2 * 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        _xorshift128plus_state[1] = z ^ (z >> 31);

        // 确保状态绝对不全为 0
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

    // 引入 dest_cap 容量参数，堵死缓冲区溢出漏洞
    static inline size_t _csv_write_field(char* dest, size_t dest_cap, size_t pos, const char* field, size_t field_len, char sep) {
        size_t i;
        int need_quote = 0;

        if (pos + (field_len * 2) + 2 >= dest_cap) {
            return (size_t)-1;  // 代表溢出错误
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


from fluidframe.core._base cimport BaseColumn, NumericType, DataType, aligned_malloc, os_aligned_free
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.doublecolumn cimport DoubleColumn, _c_corr, _c_cov
from fluidframe.core.char256column cimport Char256Column_SoA
from fluidframe.core.ucs2column cimport UCS2Column_SoA
from fluidframe.core.timestampcolumn cimport TimestampColumn
from fluidframe.core.index cimport IndexBase, RangeIndex, IntIndex, MultiIndex, INDEX_RANGE, INDEX_INT, INDEX_MULTI
from fluidframe.core.indexers import _ILocIndexer, _LocIndexer
from fluidframe.core._base cimport TYPE_INT, TYPE_DOUBLE, TYPE_CHAR256, TYPE_UCS2, TYPE_TIMESTAMP

from fluidframe._schema_types import Char256, Timestamp, _parse_schema


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
            temp = indices[i]
            indices[i] = indices[j]
            indices[j] = temp
            i += 1
            j -= 1

    _c_argsort_char256(data, indices, left, j)
    _c_argsort_char256(data, indices, i, right)


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

    _c_argsort_numeric(data, indices, left, j)
    _c_argsort_numeric(data, indices, i, right)


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
            temp = indices[i]
            indices[i] = indices[j]
            indices[j] = temp
            i += 1
            j -= 1

    _c_argsort_ucs2(chars, offsets, indices, left, j)
    _c_argsort_ucs2(chars, offsets, indices, i, right)


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
        elif col.dtype == TYPE_TIMESTAMP:
            return (<TimestampColumn>col).take_by_indices(indices, n, inplace)
        raise TypeError(f"Unsupported column dtype: {col.dtype}")

    cdef str _type_name(self, BaseColumn col):
        if col.dtype == TYPE_INT: return 'int'
        elif col.dtype == TYPE_DOUBLE: return 'float'
        elif col.dtype == TYPE_CHAR256: return 'char256'
        elif col.dtype == TYPE_UCS2: return 'ucs2'
        elif col.dtype == TYPE_TIMESTAMP: return 'timestamp'
        return 'unknown'

    cdef BaseColumn _diff_column(self, BaseColumn col, int periods):
        if col.dtype == TYPE_INT:
            return (<IntColumn>col).diff(periods)
        elif col.dtype == TYPE_DOUBLE:
            return (<DoubleColumn>col).diff(periods)
        raise TypeError(f"diff not supported for column type {self._type_name(col)}")

    cdef void _drop_row_column(self, BaseColumn col, Py_ssize_t row_idx, Py_ssize_t rem_len):
        if col.dtype == TYPE_INT:
            (<IntColumn>col).drop_row(row_idx, rem_len)
        elif col.dtype == TYPE_DOUBLE:
            (<DoubleColumn>col).drop_row(row_idx, rem_len)
        elif col.dtype == TYPE_CHAR256:
            (<Char256Column_SoA>col).drop_row(row_idx, rem_len)
        elif col.dtype == TYPE_UCS2:
            (<UCS2Column_SoA>col).drop_row(row_idx, rem_len)
        elif col.dtype == TYPE_TIMESTAMP:
            (<TimestampColumn>col).drop_row(row_idx, rem_len)

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
                elif dtype == TYPE_TIMESTAMP:
                    self.columns[col_name] = TimestampColumn()

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

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def copy(self):
        cdef Py_ssize_t n, i
        cdef Py_ssize_t* idx_c
        new_df = FastDataFrame()
        new_df.df_length = self.df_length
        for col_name, col_obj in self.columns.items():
            new_df.columns[col_name] = col_obj.copy()

        if self.index is not None:
            if self.index.index_type == INDEX_RANGE:
                new_idx = RangeIndex((<RangeIndex>self.index).start,
                                     self.index.length,
                                     (<RangeIndex>self.index).step)
                new_df.index = new_idx
            elif self.index.index_type == INDEX_INT:
                n = self.index.length
                idx_c = <Py_ssize_t*>aligned_malloc(max(n, 1) * sizeof(Py_ssize_t))
                if not idx_c: raise MemoryError()
                for i in range(n):
                    idx_c[i] = i
                try:
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

        try:
            for col_name, col_obj in self.columns.items():
                col_obj.concat(other.columns[col_name])
            self.df_length += other.df_length
        except Exception:
            raise RuntimeError("Concat failed halfway, dataframe is corrupted.")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def drop(self, str col_name):
        if col_name in self.columns:
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
        if not indices_c:
            raise MemoryError()

        cdef Py_ssize_t i
        for i in range(n):
            indices_c[i] = i

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

        elif col.dtype == TYPE_TIMESTAMP:
            with nogil:
                _c_argsort_numeric((<TimestampColumn>col).data_c, indices_c, 0, n - 1)
        else:
            os_aligned_free(indices_c)
            raise TypeError("Sorting not supported for this column type")

        cdef Py_ssize_t temp
        if reverse:
            for i in range(n // 2):
                temp = indices_c[i]
                indices_c[i] = indices_c[n - 1 - i]
                indices_c[n - 1 - i] = temp

        cdef list py_indices = [indices_c[j] for j in range(n)]
        os_aligned_free(indices_c)
        return py_indices


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sort_values(self, str col_name, bint reverse=False, bint inplace=False):
        if col_name not in self.columns:
            raise KeyError(f"Column '{col_name}' not found")

        cdef Py_ssize_t n = self.df_length
        if n == 0:
            if inplace:
                return self
            return self.copy()

        cdef list py_indices = self.argsort_by(col_name, reverse=reverse)
        cdef Py_ssize_t* idx_c = <Py_ssize_t*>aligned_malloc(max(n, 1) * sizeof(Py_ssize_t))
        if not idx_c: raise MemoryError()
        cdef Py_ssize_t i
        for i in range(n):
            idx_c[i] = py_indices[i]

        cdef FastDataFrame result_df
        if inplace:
            result_df = self
        else:
            result_df = FastDataFrame()
            result_df.df_length = n

        cdef str name
        cdef BaseColumn col, new_col
        for name, col in self.columns.items():
            new_col = self._reorder_column(col, idx_c, n, inplace)
            if not inplace:
                result_df.columns[name] = new_col

        if not inplace and self.index is not None:
            result_df.index = self.index._take(idx_c, n)

        os_aligned_free(idx_c)
        if not inplace:
            return result_df

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def diff(self, str col_name, int periods=1):
        if col_name not in self.columns:
            raise KeyError(f"Column '{col_name}' not found")

        cdef BaseColumn col = self.columns[col_name]
        if periods < 0:
            raise ValueError("periods must be >= 0")

        if col.dtype == TYPE_DOUBLE or col.dtype == TYPE_INT:
            return self._diff_column(col, periods)
        else:
            raise TypeError(f"diff is only supported for numeric columns, got {type(col)}")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def reset_index(self):
        cdef IntColumn idx_col = IntColumn(self.df_length)
        cdef Py_ssize_t i
        cdef Py_ssize_t n_old = self.index.length if self.index is not None else 0
        cdef Py_ssize_t n_fill = self.df_length if n_old >= self.df_length else n_old
        cdef list old_values = self.index.values() if n_old > 0 else None
        for i in range(n_fill):
            idx_col.data_c[i] = <int>old_values[i]
        for i in range(n_fill, self.df_length):
            idx_col.data_c[i] = <int>i
        idx_col.length = self.df_length
        self.columns['index'] = idx_col
        self.index = RangeIndex(0, self.df_length)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __repr__(self):
        return f"<FastDataFrame: {self.df_length} rows, columns: {list(self.columns.keys())}>"

    @cython.binding(False)
    def drop_col(self, str col_name):
        self.drop(col_name)

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
        cdef RangeIndex new_idx
        if n_orig == 0:
            return self if inplace else self.copy()

        cdef char* drop_mask = <char*>aligned_malloc(n_orig * sizeof(char))
        if not drop_mask: raise MemoryError()
        cdef Py_ssize_t i
        for i in range(n_orig):
            drop_mask[i] = 0

        cdef Py_ssize_t idx
        for py_idx in drop_indices:
            idx = <Py_ssize_t>py_idx
            if idx < 0: idx += n_orig
            if idx < 0 or idx >= n_orig:
                os_aligned_free(drop_mask)
                raise IndexError(f"Index {py_idx} out of bounds")
            drop_mask[idx] = 1

        cdef Py_ssize_t keep_count = 0
        for i in range(n_orig):
            if drop_mask[i] == 0:
                keep_count += 1

        if keep_count == n_orig:
            os_aligned_free(drop_mask)
            return self if inplace else self.copy()

        if keep_count == 0:
            os_aligned_free(drop_mask)
            if inplace:
                self.df_length = 0
                for col_obj in self.columns.values(): col_obj.length = 0
                return self
            else:
                return FastDataFrame()

        cdef Py_ssize_t* keep_idx_c = <Py_ssize_t*>aligned_malloc(keep_count * sizeof(Py_ssize_t))
        if not keep_idx_c:
            os_aligned_free(drop_mask)
            raise MemoryError()

        cdef Py_ssize_t j = 0
        for i in range(n_orig):
            if drop_mask[i] == 0:
                keep_idx_c[j] = i
                j += 1
        os_aligned_free(drop_mask)

        cdef FastDataFrame result_df
        if inplace:
            result_df = self
            self.df_length = keep_count
        else:
            result_df = FastDataFrame()
            result_df.df_length = keep_count

        cdef str name
        cdef BaseColumn col, new_col
        for name, col in self.columns.items():
            new_col = self._reorder_column(col, keep_idx_c, keep_count, inplace)
            if not inplace:
                result_df.columns[name] = new_col

        if self.index is not None:
            if inplace:
                self.index = self.index._take(keep_idx_c, keep_count)
            else:
                result_df.index = self.index._take(keep_idx_c, keep_count)

        os_aligned_free(keep_idx_c)
        if not inplace:
            return result_df

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.binding(False)
    def drop_raw(self, Py_ssize_t row_idx):
        if row_idx < 0 or row_idx >= self.df_length:
            raise IndexError("Row index out of range")

        cdef str name
        cdef BaseColumn col
        cdef Py_ssize_t rem_len = self.df_length - row_idx - 1
        cdef Py_ssize_t rem_len_idx, ii
        cdef IntColumn tmp_col, idx_col
        cdef RangeIndex ri

        for name, col in self.columns.items():
            self._drop_row_column(col, row_idx, rem_len)

        self.df_length -= 1
        if self.index is not None and self.index.length > 0:
            if self.index.index_type == INDEX_RANGE:
                ri = <RangeIndex>self.index
                if row_idx == 0:
                    ri.start += ri.step
                elif row_idx == self.index.length - 1:
                    pass
                else:
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
                if rem_len > 0:
                    memmove(idx_col.data_c + row_idx,
                            idx_col.data_c + row_idx + 1,
                            rem_len * sizeof(int))
                idx_col.length -= 1
            self.index.length -= 1

    @cython.binding(False)
    def argmax(self, str col_name):
        return self.columns[col_name].idxmax()

    @cython.binding(False)
    def argmin(self, str col_name):
        return self.columns[col_name].idxmin()

    @cython.binding(False)
    def unique(self, str col_name):
        return self.columns[col_name].unique()

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
        cdef int buf_pos
        cdef char sep_c = <char>ord(sep) if len(sep) == 1 else <char>ord(',')
        cdef char line_buf[65536]
        cdef bytes b_field
        cdef const char* c_field
        cdef int field_len
        cdef int buf_cap = 65536

        with open(filename, 'wb') as f:
            buf_pos = 0
            for j in range(num_cols):
                b_name = col_names[j].encode('utf-8')
                c_field = <const char*>b_name
                field_len = len(b_name)
                if j > 0:
                    line_buf[buf_pos] = sep_c
                    buf_pos += 1
                buf_pos = _csv_write_field(line_buf, buf_cap, buf_pos, c_field, field_len, sep_c)
            line_buf[buf_pos] = 10
            buf_pos += 1
            f.write(line_buf[:buf_pos])

            for i in range(self.df_length):
                buf_pos = 0
                for j in range(num_cols):
                    val_str = str(col_objs[j][i])
                    b_field = val_str.encode('utf-8')
                    c_field = <const char*>b_field
                    field_len = len(b_field)
                    if j > 0:
                        line_buf[buf_pos] = sep_c
                        buf_pos += 1
                    buf_pos = _csv_write_field(line_buf, buf_cap, buf_pos, c_field, field_len, sep_c)
                    if buf_pos > 65000:
                        f.write(line_buf[:buf_pos])
                        buf_pos = 0
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
            for i in range(n):
                idx_c[i] = i
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
        for i in range(n): idx_c[i] = start + i
        cdef FastDataFrame result = self._take(idx_c, n)
        os_aligned_free(idx_c)
        return result

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
            if col.dtype == TYPE_INT:
                result[name] = int
            elif col.dtype == TYPE_DOUBLE:
                result[name] = float
            elif col.dtype == TYPE_CHAR256:
                result[name] = Char256
            elif col.dtype == TYPE_UCS2:
                result[name] = str
            elif col.dtype == TYPE_TIMESTAMP:
                result[name] = Timestamp
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
            elif col_obj.dtype == TYPE_TIMESTAMP:
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
        return stats

    @cython.binding(False)
    def isnull(self):
        cdef FastDataFrame result = FastDataFrame()
        result.df_length = self.df_length
        cdef str name
        cdef BaseColumn col
        cdef IntColumn new_col
        cdef Py_ssize_t i
        cdef double* d_data
        for name, col in self.columns.items():
            new_col = IntColumn(self.df_length)
            new_col.length = self.df_length
            if col.dtype == TYPE_DOUBLE:
                d_data = (<DoubleColumn>col).data_c
                with nogil:
                    for i in range(self.df_length):
                        new_col.data_c[i] = 1 if d_data[i] != d_data[i] else 0
            else:
                with nogil:
                    for i in range(self.df_length):
                        new_col.data_c[i] = 0
            result.columns[name] = new_col
        return result

    @cython.binding(False)
    def fillna(self, double value, col_names=None):
        cdef FastDataFrame result = self.copy()
        cdef str name
        cdef BaseColumn col
        cdef Py_ssize_t i
        cdef double* d_data
        if col_names is None:
            col_names = list(self.columns.keys())
        for name in col_names:
            if name not in result.columns: continue
            col = result.columns[name]
            if col.dtype == TYPE_DOUBLE:
                d_data = (<DoubleColumn>col).data_c
                with nogil:
                    for i in range(self.df_length):
                        if d_data[i] != d_data[i]:
                            d_data[i] = value
        return result

    @cython.binding(False)
    def dropna(self):
        if self.df_length == 0: return self.copy()
        cdef Py_ssize_t n = self.df_length
        cdef char* keep_mask = <char*>aligned_malloc(n * sizeof(char))
        if not keep_mask: raise MemoryError()
        cdef Py_ssize_t i
        cdef double* d_data
        cdef str name
        cdef BaseColumn col

        for i in range(n): keep_mask[i] = 1

        for name, col in self.columns.items():
            if col.dtype == TYPE_DOUBLE:
                d_data = (<DoubleColumn>col).data_c
                with nogil:
                    for i in range(n):
                        if keep_mask[i] and d_data[i] != d_data[i]:
                            keep_mask[i] = 0

        cdef Py_ssize_t keep_count = 0
        for i in range(n):
            if keep_mask[i]: keep_count += 1

        if keep_count == n:
            os_aligned_free(keep_mask)
            return self.copy()
        if keep_count == 0:
            os_aligned_free(keep_mask)
            return FastDataFrame()

        cdef Py_ssize_t* keep_idx = <Py_ssize_t*>aligned_malloc(max(keep_count, 1) * sizeof(Py_ssize_t))
        if not keep_idx:
            os_aligned_free(keep_mask)
            raise MemoryError()

        cdef Py_ssize_t j = 0
        for i in range(n):
            if keep_mask[i]:
                keep_idx[j] = i
                j += 1
        os_aligned_free(keep_mask)
        cdef FastDataFrame result = self._take(keep_idx, keep_count)
        os_aligned_free(keep_idx)
        return result

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
    def from_dict(dict data, dict schema, columns=None):
        cdef dict parsed = _parse_schema(schema) if schema else {}
        cdef FastDataFrame df = FastDataFrame()
        cdef str col_name
        cdef list values
        cdef Py_ssize_t n = 0
        cdef Py_ssize_t i, j
        cdef int dtype
        cdef list col_names
        cdef list new_names
        cdef dict new_cols

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
            values = data[col_name]
            dtype = parsed[col_name]
            if dtype == TYPE_INT:
                int_col = IntColumn(n)
                int_col.length = n
                for i in range(n):
                    int_col.data_c[i] = <int>values[i]
                df.columns[col_name] = int_col
            elif dtype == TYPE_DOUBLE:
                dbl_col = DoubleColumn(n)
                dbl_col.length = n
                for i in range(n):
                    dbl_col.data_c[i] = <double>values[i]
                df.columns[col_name] = dbl_col
            elif dtype == TYPE_CHAR256:
                c256_col = Char256Column_SoA(n)
                c256_col.length = n
                for i in range(n):
                    c256_col.append(<str>values[i])
                df.columns[col_name] = c256_col
            elif dtype == TYPE_UCS2:
                ucs2_col = UCS2Column_SoA(n)
                ucs2_col.length = n
                for i in range(n):
                    ucs2_col.append(<str>values[i])
                df.columns[col_name] = ucs2_col
            elif dtype == TYPE_TIMESTAMP:
                ts_col = TimestampColumn(n)
                ts_col.length = n
                for i in range(n):
                    ts_col.data_c[i] = <long long>values[i]
                df.columns[col_name] = ts_col

        if columns is not None:
            col_names = list(df.columns.keys())
            new_names = list(columns)
            if len(new_names) != len(col_names):
                raise ValueError(f"columns length {len(new_names)} != {len(col_names)}")
            new_cols = {}
            for i in range(len(new_names)):
                new_cols[new_names[i]] = df.columns[col_names[i]]
            df.columns = new_cols

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
                dtype = parsed[col_name]
                if dtype == TYPE_INT:
                    df.columns[col_name] = IntColumn()
                elif dtype == TYPE_DOUBLE:
                    df.columns[col_name] = DoubleColumn()
                elif dtype == TYPE_CHAR256:
                    df.columns[col_name] = Char256Column_SoA()
                elif dtype == TYPE_UCS2:
                    df.columns[col_name] = UCS2Column_SoA()
                elif dtype == TYPE_TIMESTAMP:
                    df.columns[col_name] = TimestampColumn()
            return df

        df.df_length = n_rows
        df.index = RangeIndex(0, n_rows)

        for j in range(n_cols):
            col_name = col_order[j]
            dtype = parsed[col_name]
            if dtype == TYPE_INT:
                int_col = IntColumn(n_rows)
                int_col.length = n_rows
                for i in range(n_rows):
                    int_col.data_c[i] = <int>rows[i][j]
                df.columns[col_name] = int_col
            elif dtype == TYPE_DOUBLE:
                dbl_col = DoubleColumn(n_rows)
                dbl_col.length = n_rows
                for i in range(n_rows):
                    dbl_col.data_c[i] = <double>rows[i][j]
                df.columns[col_name] = dbl_col
            elif dtype == TYPE_CHAR256:
                c256_col = Char256Column_SoA(n_rows)
                c256_col.length = n_rows
                for i in range(n_rows):
                    c256_col.append(<str>rows[i][j])
                df.columns[col_name] = c256_col
            elif dtype == TYPE_UCS2:
                ucs2_col = UCS2Column_SoA(n_rows)
                ucs2_col.length = n_rows
                for i in range(n_rows):
                    ucs2_col.append(<str>rows[i][j])
                df.columns[col_name] = ucs2_col
            elif dtype == TYPE_TIMESTAMP:
                ts_col = TimestampColumn(n_rows)
                ts_col.length = n_rows
                for i in range(n_rows):
                    ts_col.data_c[i] = <long long>rows[i][j]
                df.columns[col_name] = ts_col

        if columns is not None:
            col_names = list(df.columns.keys())
            new_names = list(columns)
            if len(new_names) != len(col_names):
                raise ValueError(f"columns length {len(new_names)} != {len(col_names)}")
            new_cols = {}
            for i in range(len(new_names)):
                new_cols[new_names[i]] = df.columns[col_names[i]]
            df.columns = new_cols

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
                dtype = parsed[col_name]
                if dtype == TYPE_INT:
                    df.columns[col_name] = IntColumn()
                elif dtype == TYPE_DOUBLE:
                    df.columns[col_name] = DoubleColumn()
                elif dtype == TYPE_CHAR256:
                    df.columns[col_name] = Char256Column_SoA()
                elif dtype == TYPE_UCS2:
                    df.columns[col_name] = UCS2Column_SoA()
                elif dtype == TYPE_TIMESTAMP:
                    df.columns[col_name] = TimestampColumn()
            return df

        df.df_length = n_rows
        df.index = RangeIndex(0, n_rows)

        for j in range(n_cols):
            col_name = col_order[j]
            dtype = parsed[col_name]
            if dtype == TYPE_INT:
                int_col = IntColumn(n_rows)
                int_col.length = n_rows
                for i in range(n_rows):
                    int_col.data_c[i] = <int>arrays[j][i]
                df.columns[col_name] = int_col
            elif dtype == TYPE_DOUBLE:
                dbl_col = DoubleColumn(n_rows)
                dbl_col.length = n_rows
                for i in range(n_rows):
                    dbl_col.data_c[i] = <double>arrays[j][i]
                df.columns[col_name] = dbl_col
            elif dtype == TYPE_CHAR256:
                c256_col = Char256Column_SoA(n_rows)
                c256_col.length = n_rows
                for i in range(n_rows):
                    c256_col.append(<str>arrays[j][i])
                df.columns[col_name] = c256_col
            elif dtype == TYPE_UCS2:
                ucs2_col = UCS2Column_SoA(n_rows)
                ucs2_col.length = n_rows
                for i in range(n_rows):
                    ucs2_col.append(<str>arrays[j][i])
                df.columns[col_name] = ucs2_col
            elif dtype == TYPE_TIMESTAMP:
                ts_col = TimestampColumn(n_rows)
                ts_col.length = n_rows
                for i in range(n_rows):
                    ts_col.data_c[i] = <long long>arrays[j][i]
                df.columns[col_name] = ts_col

        if columns is not None:
            col_names = list(df.columns.keys())
            new_names = list(columns)
            if len(new_names) != len(col_names):
                raise ValueError(f"columns length {len(new_names)} != {len(col_names)}")
            new_cols = {}
            for i in range(len(new_names)):
                new_cols[new_names[i]] = df.columns[col_names[i]]
            df.columns = new_cols

        return df

    @staticmethod
    def arange(Py_ssize_t n):
        cdef FastDataFrame df = FastDataFrame()
        if n <= 0:
            return df
        df.df_length = n
        df.index = RangeIndex(0, n)
        cdef IntColumn col = IntColumn(n)
        col.length = n
        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                col.data_c[i] = <int>i
        df.columns['0'] = col
        return df

    def reshape(self, Py_ssize_t rows, Py_ssize_t cols):
        if self.df_length != rows * cols:
            raise ValueError(f"Cannot reshape {self.df_length} elements into ({rows}, {cols})")

        cdef FastDataFrame result = FastDataFrame()
        result.df_length = rows
        result.index = RangeIndex(0, rows)

        cdef list col_names = [chr(65 + i) for i in range(cols)]

        cdef list all_vals = []
        cdef str name
        cdef BaseColumn col_obj
        for i in range(self.df_length):
            for name in self.columns:
                col_obj = self.columns[name]
                all_vals.append(col_obj[i])

        for c_idx in range(cols):
            col = DoubleColumn(rows)
            col.length = rows
            for r_idx in range(rows):
                col.data_c[r_idx] = <double>all_vals[r_idx * cols + c_idx]
            result.columns[col_names[c_idx]] = col

        return result

    @staticmethod
    def randn(Py_ssize_t rows, Py_ssize_t cols):
        cdef FastDataFrame df = FastDataFrame()
        if rows <= 0 or cols <= 0:
            return df
        df.df_length = rows
        df.index = RangeIndex(0, rows)
        cdef list col_names = [chr(65 + i) for i in range(cols)]
        cdef Py_ssize_t c_idx, r_idx
        _xorshift128plus_seed(<unsigned long long>time(NULL) ^ 12345)
        for c_idx in range(cols):
            col = DoubleColumn(rows)
            col.length = rows
            for r_idx in range(rows):
                col.data_c[r_idx] = _random_gauss()
            df.columns[col_names[c_idx]] = col
        return df

    @staticmethod
    def rand(Py_ssize_t rows, Py_ssize_t cols):
        cdef FastDataFrame df = FastDataFrame()
        if rows <= 0 or cols <= 0:
            return df
        df.df_length = rows
        df.index = RangeIndex(0, rows)
        cdef list col_names = [chr(65 + i) for i in range(cols)]
        cdef Py_ssize_t c_idx, r_idx
        _xorshift128plus_seed(<unsigned long long>time(NULL) ^ 54321)
        for c_idx in range(cols):
            col = DoubleColumn(rows)
            col.length = rows
            for r_idx in range(rows):
                col.data_c[r_idx] = _random_uniform()
            df.columns[col_names[c_idx]] = col
        return df

    @staticmethod
    def zeros(Py_ssize_t rows, Py_ssize_t cols):
        cdef FastDataFrame df = FastDataFrame()
        if rows <= 0 or cols <= 0:
            return df
        df.df_length = rows
        df.index = RangeIndex(0, rows)
        cdef list col_names = [chr(65 + i) for i in range(cols)]
        cdef Py_ssize_t c_idx
        for c_idx in range(cols):
            col = DoubleColumn(rows)
            col.length = rows
            df.columns[col_names[c_idx]] = col
        return df

    @staticmethod
    def ones(Py_ssize_t rows, Py_ssize_t cols):
        cdef FastDataFrame df = FastDataFrame()
        if rows <= 0 or cols <= 0:
            return df
        df.df_length = rows
        df.index = RangeIndex(0, rows)
        cdef list col_names = [chr(65 + i) for i in range(cols)]
        cdef Py_ssize_t c_idx, r_idx
        for c_idx in range(cols):
            col = DoubleColumn(rows)
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
        if n == 0:
            return IntColumn(0)

        cdef object sample = None
        for i in range(n):
            if values[i] is not None:
                sample = values[i]
                break

        if sample is None:
            dbl_col = DoubleColumn(n)
            dbl_col.length = n
            for i in range(n):
                dbl_col.data_c[i] = NAN
            return dbl_col

        if isinstance(sample, int):
            int_col = IntColumn(n)
            int_col.length = n
            for i in range(n):
                int_col.data_c[i] = <int>(values[i] if values[i] is not None else 0)
            return int_col
        elif isinstance(sample, float):
            dbl_col = DoubleColumn(n)
            dbl_col.length = n
            for i in range(n):
                if values[i] is not None:
                    dbl_col.data_c[i] = <double>values[i]
                else:
                    dbl_col.data_c[i] = NAN
            return dbl_col
        elif isinstance(sample, str):
            try:
                (<str>sample).encode('ascii')
                c256_col = Char256Column_SoA(n)
                for i in range(n):
                    c256_col.append(<str>(values[i] if values[i] is not None else ''))
                return c256_col
            except UnicodeEncodeError:
                ucs2_col = UCS2Column_SoA(n)
                for i in range(n):
                    ucs2_col.append(<str>(values[i] if values[i] is not None else ''))
                return ucs2_col
        raise TypeError(f"Unsupported type: {type(sample)}")

    cdef void _shift_right(self, BaseColumn col, Py_ssize_t pos):
        cdef Py_ssize_t rem, col_len
        cdef Py_ssize_t start_offset, new_offsets_len, i
        cdef UCS2Column_SoA ucs2_col
        cdef Py_ssize_t* new_offsets
        cdef IntColumn ic
        cdef DoubleColumn dc
        cdef Char256Column_SoA cc
        cdef TimestampColumn tc
        if col.dtype == TYPE_INT:
            ic = <IntColumn>col
            col_len = ic.length
            rem = col_len - pos
            if col_len >= ic.capacity:
                ic._reserve(ic.capacity * 2)
            if rem > 0:
                memmove(ic.data_c + pos + 1,
                        ic.data_c + pos,
                        rem * sizeof(int))
            ic.length = col_len + 1
        elif col.dtype == TYPE_DOUBLE:
            dc = <DoubleColumn>col
            col_len = dc.length
            rem = col_len - pos
            if col_len >= dc.capacity:
                dc._reserve(dc.capacity * 2)
            if rem > 0:
                memmove(dc.data_c + pos + 1,
                        dc.data_c + pos,
                        rem * sizeof(double))
            dc.length = col_len + 1
        elif col.dtype == TYPE_CHAR256:
            cc = <Char256Column_SoA>col
            col_len = cc.length
            rem = col_len - pos
            if col_len >= cc.capacity:
                cc._reserve(cc.capacity * 2)
            if rem > 0:
                memmove(cc.data_c + ((pos + 1) * 256),
                        cc.data_c + (pos * 256),
                        rem * 256)
            cc.length = col_len + 1
        elif col.dtype == TYPE_UCS2:
            ucs2_col = <UCS2Column_SoA>col
            col_len = ucs2_col.length
            if col_len >= ucs2_col.capacity:
                ucs2_col._reserve_rows(ucs2_col.capacity * 2)
            start_offset = ucs2_col.offsets_c[pos]

            for i in range(col_len, pos, -1):
                ucs2_col.offsets_c[i + 1] = ucs2_col.offsets_c[i]
            ucs2_col.offsets_c[pos + 1] = start_offset
            ucs2_col.length = col_len + 1

        elif col.dtype == TYPE_TIMESTAMP:
            tc = <TimestampColumn>col
            col_len = tc.length
            rem = col_len - pos
            if col_len >= tc.capacity:
                tc._reserve(tc.capacity * 2)
            if rem > 0:
                memmove(tc.data_c + pos + 1,
                        tc.data_c + pos,
                        rem * sizeof(long long))
            tc.length = col_len + 1

    cdef void _shift_left(self, BaseColumn col, Py_ssize_t pos):
        """Remove element at pos by shifting data left (inverse of _shift_right)."""
        cdef Py_ssize_t rem, col_len
        cdef Py_ssize_t start_offset, new_offsets_len, i, removed_len
        cdef UCS2Column_SoA ucs2_col
        cdef Py_ssize_t* new_offsets
        cdef IntColumn ic
        cdef DoubleColumn dc
        cdef Char256Column_SoA cc
        cdef TimestampColumn tc
        if col.dtype == TYPE_INT:
            ic = <IntColumn>col
            col_len = ic.length
            rem = col_len - pos - 1
            if rem > 0:
                memmove(ic.data_c + pos,
                        ic.data_c + pos + 1,
                        rem * sizeof(int))
            ic.length = col_len - 1
        elif col.dtype == TYPE_DOUBLE:
            dc = <DoubleColumn>col
            col_len = dc.length
            rem = col_len - pos - 1
            if rem > 0:
                memmove(dc.data_c + pos,
                        dc.data_c + pos + 1,
                        rem * sizeof(double))
            dc.length = col_len - 1
        elif col.dtype == TYPE_CHAR256:
            cc = <Char256Column_SoA>col
            col_len = cc.length
            rem = col_len - pos - 1
            if rem > 0:
                memmove(cc.data_c + (pos * 256),
                        cc.data_c + ((pos + 1) * 256),
                        rem * 256)
            cc.length = col_len - 1
        elif col.dtype == TYPE_UCS2:
            ucs2_col = <UCS2Column_SoA>col
            col_len = ucs2_col.length
            start_offset = ucs2_col.offsets_c[pos]
            removed_len = ucs2_col.offsets_c[pos + 1] - start_offset
            rem = ucs2_col.char_length - ucs2_col.offsets_c[pos + 1]
            if rem > 0:
                memmove(ucs2_col.chars_c + start_offset,
                        ucs2_col.chars_c + ucs2_col.offsets_c[pos + 1],
                        rem * sizeof(Py_UCS2))
            ucs2_col.char_length -= removed_len
            for i in range(pos, col_len):
                ucs2_col.offsets_c[i] = ucs2_col.offsets_c[i + 1] - removed_len
            ucs2_col.length = col_len - 1
        elif col.dtype == TYPE_TIMESTAMP:
            tc = <TimestampColumn>col
            col_len = tc.length
            rem = col_len - pos - 1
            if rem > 0:
                memmove(tc.data_c + pos,
                        tc.data_c + pos + 1,
                        rem * sizeof(long long))
            tc.length = col_len - 1

    cdef double _get_col_value_as_double(self, BaseColumn col, Py_ssize_t i):
        if col.dtype == TYPE_INT:
            return <double>(<IntColumn>col).data_c[i]
        elif col.dtype == TYPE_DOUBLE:
            return (<DoubleColumn>col).data_c[i]
        return 0.0

    cdef void _col_to_double_buffer(self, BaseColumn col, double* out, Py_ssize_t n):
        if col.dtype == TYPE_INT:
            (<IntColumn>col).to_double_buffer(out, n)
        elif col.dtype == TYPE_DOUBLE:
            (<DoubleColumn>col).to_double_buffer(out, n)


    def __setitem__(self, str name, object value):
        cdef Py_ssize_t i
        cdef IntColumn int_col
        cdef DoubleColumn dbl_col

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
        elif isinstance(value, (int, float)):
            if self.df_length == 0:
                raise ValueError("Cannot fill scalar into empty DataFrame")
            if isinstance(value, int):
                int_col = IntColumn(self.df_length)
                int_col.length = self.df_length
                for i in range(self.df_length):
                    int_col.data_c[i] = <int>value
                self.columns[name] = int_col
            else:
                dbl_col = DoubleColumn(self.df_length)
                dbl_col.length = self.df_length
                for i in range(self.df_length):
                    dbl_col.data_c[i] = <double>value
                self.columns[name] = dbl_col
        else:
            raise TypeError(f"Cannot set column from {type(value)}")

    def __setitem_with_colname__(self, str col_name, object value):
        self[col_name] = value


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
        cdef Py_UCS2* ucs2_chars_ptr
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
                elif col_obj.dtype == TYPE_TIMESTAMP:
                    (<TimestampColumn>col_obj).data_c[pos] = <long long>val
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
        for j in range(cols):
            new_col = DoubleColumn(rows)
            new_col.length = rows
            for i in range(rows):
                new_col.data_c[i] = self._get_col_value_as_double(self.columns[col_names[i]], j)
            result.columns[col_names[j]] = new_col
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
                    new_col_list.append(None if col_obj.dtype == TYPE_CHAR256 or col_obj.dtype == TYPE_UCS2 else (NAN if col_obj.dtype == TYPE_DOUBLE else 0))
            result.columns[col_name] = self._column_from_list(col_name, new_col_list)
        return result

    def duplicated(self, list subset=None, str keep='first'):
        cdef Py_ssize_t n = self.df_length
        cdef IntColumn result = IntColumn(n)
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

        for i in range(result_len):
            left_idx_c[i] = left_indices[i]
            right_idx_c[i] = right_indices[i]

        cdef BaseColumn src_col, new_col
        for cn in [on] + left_col_names:
            src_col = self.columns[cn]
            if src_col.dtype == TYPE_INT:
                result.columns[cn] = (<IntColumn>src_col).gather_with_sentinel(left_idx_c, result_len, 0)
            elif src_col.dtype == TYPE_DOUBLE:
                result.columns[cn] = (<DoubleColumn>src_col).gather_with_sentinel(left_idx_c, result_len, NAN)
            else:
                result.columns[cn] = src_col.copy()

        for cn in right_col_names:
            src_col = other.columns[cn]
            if cn in result.columns:
                out_name = cn + '_y'
            else:
                out_name = cn
            if src_col.dtype == TYPE_INT:
                result.columns[out_name] = (<IntColumn>src_col).gather_with_sentinel(right_idx_c, result_len, 0)
            elif src_col.dtype == TYPE_DOUBLE:
                result.columns[out_name] = (<DoubleColumn>src_col).gather_with_sentinel(right_idx_c, result_len, NAN)
            else:
                result.columns[out_name] = src_col.copy()

        os_aligned_free(left_idx_c)
        os_aligned_free(right_idx_c)

        if on in result.columns and on not in left_col_names:
            pass
        return result


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
        for i in range(n_cols):
            data[i] = NULL

        try:
            for i in range(n_cols):
                data[i] = <double*>aligned_malloc(max(n_rows, 1) * sizeof(double))
                if not data[i]:
                    raise MemoryError()
                self._col_to_double_buffer(self.columns[num_cols[i]], data[i], n_rows)

            result = FastDataFrame()
            result.df_length = n_cols
            result.index = RangeIndex(0, n_cols)
            for i in range(n_cols):
                dbl_col = DoubleColumn(n_cols)
                dbl_col.length = n_cols
                result.columns[num_cols[i]] = dbl_col

            for i in range(n_cols):
                for j in range(n_cols):
                    if i == j:
                        (<DoubleColumn>result.columns[num_cols[i]]).data_c[j] = 1.0
                    else:
                        (<DoubleColumn>result.columns[num_cols[i]]).data_c[j] = _c_corr(data[i], data[j], n_rows)
            return result
        finally:
            if data != NULL:
                for i in range(n_cols):
                    if data[i] != NULL:
                        os_aligned_free(data[i])
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
        for i in range(n_cols):
            data[i] = NULL

        try:
            for i in range(n_cols):
                data[i] = <double*>aligned_malloc(max(n_rows, 1) * sizeof(double))
                if not data[i]:
                    raise MemoryError()
                self._col_to_double_buffer(self.columns[num_cols[i]], data[i], n_rows)

            result = FastDataFrame()
            result.df_length = n_cols
            result.index = RangeIndex(0, n_cols)
            for i in range(n_cols):
                dbl_col = DoubleColumn(n_cols)
                dbl_col.length = n_cols
                result.columns[num_cols[i]] = dbl_col

            for i in range(n_cols):
                for j in range(n_cols):
                    (<DoubleColumn>result.columns[num_cols[i]]).data_c[j] = _c_cov(data[i], data[j], n_rows)
            return result
        finally:
            if data != NULL:
                for i in range(n_cols):
                    if data[i] != NULL:
                        os_aligned_free(data[i])
                os_aligned_free(data)


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

    def where_index(self, IntColumn mask):
        return self.loc[mask]

    def where_all(self, IntColumn mask, object fill=0):
        return self.where_row(mask, fill)


    def sort_values_by(self, str col_name, bint reverse=False):
        return self.sort_values(col_name, reverse=reverse)

    def sort_values_inplace(self, str col_name, bint reverse=False):
        return self.sort_values(col_name, reverse=reverse, inplace=True)


    def info(self):
        cdef Py_ssize_t total_cols = len(self.columns)
        cdef Py_ssize_t mem = self.memory_usage()
        cdef char bytes_buf[16]
        cdef list lines = []
        cdef Py_ssize_t idx = 0
        cdef Py_ssize_t col_bytes
        cdef str type_name
        cdef str name
        cdef object col

        lines.append("<class 'FastDataFrame'>")
        lines.append(f"RangeIndex: {self.df_length} entries")
        lines.append(f"Data columns (total {total_cols} columns):")
        lines.append(" #   Column    Non-Null  Dtype     Mem Usage")
        lines.append("---  ------    --------  ----     ---------")

        for name, col in self.columns.items():
            type_name = self._type_name(col)
            col_bytes = col.length * self._col_bytes(col)
            _c_fmt_bytes(bytes_buf, sizeof(bytes_buf), col_bytes)
            lines.append(f" {idx}   {name:<10}{col.length:<10}{type_name:<10}{bytes_buf.decode('ascii')}")
            idx += 1

        _c_fmt_bytes(bytes_buf, sizeof(bytes_buf), mem)
        lines.append(f"dtypes: {self._dtype_summary()}")
        lines.append(f"memory usage: {bytes_buf.decode('ascii')}")
        return '\n'.join(lines)

    cdef Py_ssize_t _col_bytes(self, BaseColumn col):
        if col.dtype == TYPE_INT: return sizeof(int)
        elif col.dtype == TYPE_DOUBLE: return sizeof(double)
        elif col.dtype == TYPE_CHAR256: return 256
        elif col.dtype == TYPE_TIMESTAMP: return sizeof(long long)
        elif col.dtype == TYPE_UCS2: return 4
        return 0

    cdef str _dtype_summary(self):
        cdef dict counts = {}
        cdef str name, tname
        for name, col in self.columns.items():
            tname = self._type_name(col)
            counts[tname] = counts.get(tname, 0) + 1
        return ', '.join(f'{v} {k}' for k, v in counts.items())

    def memory_usage(self):
        cdef Py_ssize_t total = 0
        for name, col in self.columns.items():
            if col.dtype == TYPE_INT:
                total += col.capacity * sizeof(int)
            elif col.dtype == TYPE_DOUBLE:
                total += col.capacity * sizeof(double)
            elif col.dtype == TYPE_CHAR256:
                total += col.capacity * 256
            elif col.dtype == TYPE_TIMESTAMP:
                total += col.capacity * sizeof(long long)
            elif col.dtype == TYPE_UCS2:
                total += col.capacity * sizeof(Py_UCS2) + (col.capacity + 1) * sizeof(Py_ssize_t)
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
                               TYPE_UCS2: 'ucs2', TYPE_TIMESTAMP: 'timestamp'}
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