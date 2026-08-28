# distutils: language = c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport cython
from libc.stdlib cimport malloc, free
from libc.stdint cimport uint64_t
from libc.math cimport NAN, INFINITY
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from cpython.unicode cimport Py_UCS2

from fluidframe.core._base cimport (
    BaseColumn, TYPE_INT, TYPE_DOUBLE, TYPE_CHAR256, TYPE_UCS2, TYPE_TIMESTAMP
)
from fluidframe.core.fastdataframe cimport FastDataFrame
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.doublecolumn cimport DoubleColumn
from fluidframe.core.char256column cimport Char256Column_SoA
from fluidframe.core.ucs2column cimport UCS2Column_SoA
from fluidframe.core.timestampcolumn cimport TimestampColumn

cdef enum AggFunc:
    AGG_SUM = 0
    AGG_MEAN = 1
    AGG_MIN = 2
    AGG_MAX = 3
    AGG_VAR = 4
    AGG_STD = 5

cdef AggFunc _resolve_agg_func(str func) except? AGG_SUM:
    if func == 'sum': return AGG_SUM
    elif func == 'mean': return AGG_MEAN
    elif func == 'min': return AGG_MIN
    elif func == 'max': return AGG_MAX
    elif func == 'var': return AGG_VAR
    elif func == 'std': return AGG_STD
    else: raise ValueError(f"Unknown aggregation function: {func}")


cdef uint64_t _fnv1a_bytes(char* data, Py_ssize_t length) noexcept nogil:
    cdef uint64_t h = 14695981039346656037ULL
    cdef Py_ssize_t i
    for i in range(length):
        h ^= <uint64_t><unsigned char>data[i]
        h *= 1099511628211ULL
    return h

cdef uint64_t _fnv1a_ucs2(Py_UCS2* data, Py_ssize_t length) noexcept nogil:
    cdef uint64_t h = 14695981039346656037ULL
    cdef Py_ssize_t i
    for i in range(length):
        h ^= <uint64_t>data[i]
        h *= 1099511628211ULL
    return h

cdef uint64_t _mix_hash(uint64_t h, uint64_t val) noexcept nogil:
    h ^= val
    h *= 1099511628211ULL
    return h


cdef class FastDataFrameGroupBy:

    cdef void _extract_col_ptrs(self):
        cdef int n = len(self.by_cols)
        cdef str col_name
        cdef BaseColumn col
        cdef int idx
        if n == 0:
            return
        self._num_by_cols = n
        self._by_col_data = <void**>malloc(n * sizeof(void*))
        self._by_col_offsets = <void**>malloc(n * sizeof(void*))
        self._by_col_types = <int*>malloc(n * sizeof(int))
        if not self._by_col_data or not self._by_col_offsets or not self._by_col_types:
            raise MemoryError()
        for idx in range(n):
            col_name = self.by_cols[idx]
            col = self.df.columns[col_name]
            self._by_col_types[idx] = col.dtype
            self._by_col_offsets[idx] = NULL
            if col.dtype == TYPE_INT:
                self._by_col_data[idx] = <void*>(<IntColumn>col).data_c
            elif col.dtype == TYPE_DOUBLE:
                self._by_col_data[idx] = <void*>(<DoubleColumn>col).data_c
            elif col.dtype == TYPE_CHAR256:
                self._by_col_data[idx] = <void*>(<Char256Column_SoA>col).data_c
            elif col.dtype == TYPE_UCS2:
                self._by_col_data[idx] = <void*>(<UCS2Column_SoA>col).chars_c
                self._by_col_offsets[idx] = <void*>(<UCS2Column_SoA>col).offsets_c
            elif col.dtype == TYPE_TIMESTAMP:
                self._by_col_data[idx] = <void*>(<TimestampColumn>col).data_c

    cdef uint64_t _hash_row(self, Py_ssize_t row_idx) nogil:
        cdef uint64_t h = 14695981039346656037ULL
        cdef int col_idx
        cdef uint64_t val
        cdef int* i_data
        cdef double* d_data
        cdef long long* t_data
        cdef char* c_data
        cdef Py_UCS2* u_data
        cdef Py_ssize_t start, end

        for col_idx in range(self._num_by_cols):
            if self._by_col_types[col_idx] == TYPE_INT:
                i_data = <int*>self._by_col_data[col_idx]
                val = <uint64_t>i_data[row_idx]
            elif self._by_col_types[col_idx] == TYPE_DOUBLE:
                d_data = <double*>self._by_col_data[col_idx]
                val = (<uint64_t*>(&d_data[row_idx]))[0]
            elif self._by_col_types[col_idx] == TYPE_TIMESTAMP:
                t_data = <long long*>self._by_col_data[col_idx]
                val = <uint64_t>t_data[row_idx]
            elif self._by_col_types[col_idx] == TYPE_CHAR256:
                c_data = <char*>self._by_col_data[col_idx]
                val = _fnv1a_bytes(c_data + row_idx * 256, 256)
            elif self._by_col_types[col_idx] == TYPE_UCS2:
                u_data = <Py_UCS2*>self._by_col_data[col_idx]
                start = (<Py_ssize_t*>self._by_col_offsets[col_idx])[row_idx]
                end = (<Py_ssize_t*>self._by_col_offsets[col_idx])[row_idx + 1]
                val = _fnv1a_ucs2(u_data + start, end - start)
            else:
                val = 0
            h = _mix_hash(h, val)

        return h

    cdef void _build_groups(self):
        cdef Py_ssize_t i
        cdef uint64_t h
        cdef int g_id
        cdef Py_ssize_t n = self.df.df_length

        for i in range(n):
            h = self._hash_row(i)

            if self.hash_to_group_id.find(h) == self.hash_to_group_id.end():
                g_id = self.num_groups
                self.hash_to_group_id[0][h] = g_id
                self.first_seen_rows.push_back(i)
                self.group_rows.push_back(vector[Py_ssize_t]())
                self.num_groups += 1
            else:
                g_id = self.hash_to_group_id[0][h]

            self.group_rows[0][g_id].push_back(i)

    cdef DoubleColumn _native_agg(self, str col_name, str func):
        """Generic native aggregation dispatcher for numeric columns."""
        cdef DoubleColumn res_col = DoubleColumn(self.num_groups)
        res_col.length = self.num_groups
        cdef BaseColumn orig_col = self.df.columns[col_name]
        cdef int g_id
        cdef Py_ssize_t sz
        cdef AggFunc agg_type = _resolve_agg_func(func)

        for g_id in range(self.num_groups):
            sz = self.group_rows[0][g_id].size()
            if orig_col.dtype == TYPE_DOUBLE:
                if agg_type == AGG_SUM:
                    res_col.data_c[g_id] = (<DoubleColumn>orig_col)._sum_rows(self.group_rows[0][g_id].data(), sz)
                elif agg_type == AGG_MEAN:
                    res_col.data_c[g_id] = (<DoubleColumn>orig_col)._mean_rows(self.group_rows[0][g_id].data(), sz)
                elif agg_type == AGG_MIN:
                    res_col.data_c[g_id] = (<DoubleColumn>orig_col)._min_rows(self.group_rows[0][g_id].data(), sz)
                elif agg_type == AGG_MAX:
                    res_col.data_c[g_id] = (<DoubleColumn>orig_col)._max_rows(self.group_rows[0][g_id].data(), sz)
                elif agg_type == AGG_VAR:
                    res_col.data_c[g_id] = (<DoubleColumn>orig_col)._var_rows(self.group_rows[0][g_id].data(), sz)
                elif agg_type == AGG_STD:
                    res_col.data_c[g_id] = (<DoubleColumn>orig_col)._std_rows(self.group_rows[0][g_id].data(), sz)
            elif orig_col.dtype == TYPE_INT:
                if agg_type == AGG_SUM:
                    res_col.data_c[g_id] = (<IntColumn>orig_col)._sum_rows(self.group_rows[0][g_id].data(), sz)
                elif agg_type == AGG_MEAN:
                    res_col.data_c[g_id] = (<IntColumn>orig_col)._mean_rows(self.group_rows[0][g_id].data(), sz)
                elif agg_type == AGG_MIN:
                    res_col.data_c[g_id] = (<IntColumn>orig_col)._min_rows(self.group_rows[0][g_id].data(), sz)
                elif agg_type == AGG_MAX:
                    res_col.data_c[g_id] = (<IntColumn>orig_col)._max_rows(self.group_rows[0][g_id].data(), sz)
                elif agg_type == AGG_VAR:
                    res_col.data_c[g_id] = (<IntColumn>orig_col)._var_rows(self.group_rows[0][g_id].data(), sz)
                elif agg_type == AGG_STD:
                    res_col.data_c[g_id] = (<IntColumn>orig_col)._std_rows(self.group_rows[0][g_id].data(), sz)
        return res_col

    cdef IntColumn _native_count(self):
        cdef IntColumn res_col = IntColumn(self.num_groups)
        res_col.length = self.num_groups
        cdef int g_id
        for g_id in range(self.num_groups):
            res_col.data_c[g_id] = <int>self.group_rows[0][g_id].size()
        return res_col

    cdef object _python_agg(self, str col_name, str func_name):
        cdef list results = []
        cdef int g_id
        cdef Py_ssize_t sz, row_idx, i
        cdef BaseColumn orig_col = self.df.columns[col_name]
        cdef list vals

        for g_id in range(self.num_groups):
            sz = self.group_rows[0][g_id].size()
            vals = []
            for i in range(sz):
                row_idx = self.group_rows[0][g_id][i]
                vals.append(orig_col[row_idx])
            if func_name == 'sum':
                results.append(sum(vals))
            elif func_name == 'mean':
                results.append(sum(vals) / len(vals) if vals else NAN)
            elif func_name == 'min':
                results.append(min(vals) if vals else NAN)
            elif func_name == 'max':
                results.append(max(vals) if vals else NAN)
            elif func_name == 'count':
                results.append(len(vals))
            else:
                results.append(vals)

        cdef DoubleColumn res_col = DoubleColumn(self.num_groups)
        res_col.length = self.num_groups
        for g_id in range(self.num_groups):
            res_col.data_c[g_id] = <double>results[g_id] if results[g_id] is not None else NAN
        return res_col

    cdef object _extract_key_column(self, str col_name):
        cdef BaseColumn orig = self.df.columns[col_name]
        cdef int i
        cdef Py_ssize_t row_idx
        cdef IntColumn res_int
        cdef DoubleColumn res_dbl
        cdef Py_ssize_t* fs
        cdef FastDataFrame tmp

        if orig.dtype == TYPE_INT:
            res_int = IntColumn(self.num_groups)
            res_int.length = self.num_groups
            for i in range(self.num_groups):
                row_idx = self.first_seen_rows[0][i]
                res_int.data_c[i] = (<IntColumn>orig).data_c[row_idx]
            return res_int
        elif orig.dtype == TYPE_DOUBLE:
            res_dbl = DoubleColumn(self.num_groups)
            res_dbl.length = self.num_groups
            for i in range(self.num_groups):
                row_idx = self.first_seen_rows[0][i]
                res_dbl.data_c[i] = (<DoubleColumn>orig).data_c[row_idx]
            return res_dbl
        else:
            fs = <Py_ssize_t*>malloc(self.num_groups * sizeof(Py_ssize_t))
            if not fs: raise MemoryError()
            for i in range(self.num_groups):
                fs[i] = self.first_seen_rows[0][i]
            tmp = self.df._take(fs, self.num_groups)
            free(fs)
            return tmp.columns[col_name]

    # ==========================================
    # Python-facing __init__ and __dealloc__
    # ==========================================
    def __cinit__(self, FastDataFrame df, list by_cols):
        self.df = df
        self.by_cols = by_cols

        self.hash_to_group_id = new unordered_map[uint64_t, int]()
        self.group_rows = new vector[vector[Py_ssize_t]]()
        self.first_seen_rows = new vector[Py_ssize_t]()
        self.num_groups = 0

        self._by_col_data = NULL
        self._by_col_offsets = NULL
        self._by_col_types = NULL
        self._num_by_cols = 0

        self._extract_col_ptrs()
        self._build_groups()

    def __dealloc__(self):
        if self.hash_to_group_id is not NULL: del self.hash_to_group_id
        if self.group_rows is not NULL: del self.group_rows
        if self.first_seen_rows is not NULL: del self.first_seen_rows
        if self._by_col_data is not NULL: free(self._by_col_data)
        if self._by_col_offsets is not NULL: free(self._by_col_offsets)
        if self._by_col_types is not NULL: free(self._by_col_types)

    def size(self):
        return self.num_groups

    @cython.binding(False)
    def agg(self, dict agg_dict):
        cdef FastDataFrame res = FastDataFrame()
        res.df_length = self.num_groups

        cdef Py_ssize_t* fs_c = <Py_ssize_t*>malloc(self.num_groups * sizeof(Py_ssize_t))
        cdef int i
        for i in range(self.num_groups):
            fs_c[i] = self.first_seen_rows[0][i]
        cdef FastDataFrame keys_df = self.df._take(fs_c, self.num_groups)
        free(fs_c)

        cdef str col_name, func_name
        for col_name, func_name in agg_dict.items():
            if func_name in ('sum', 'mean', 'min', 'max', 'var', 'std'):
                res.columns[col_name] = self._native_agg(col_name, func_name)
            elif func_name == 'count':
                res.columns[col_name] = self._native_count()
            else:
                res.columns[col_name] = self._python_agg(col_name, func_name)

        for col_name in self.by_cols:
            if col_name not in res.columns:
                res.columns[col_name] = self._extract_key_column(col_name)

        return res

    cdef FastDataFrame _convenience_agg(self, str func_name, str col_name=None):
        """Shared implementation for sum/mean/min/max/var/std convenience methods."""
        cdef FastDataFrame res
        cdef dict agg
        cdef str cn
        cdef BaseColumn c
        if col_name is not None:
            res = FastDataFrame()
            res.df_length = self.num_groups
            res.columns[col_name] = self._native_agg(col_name, func_name)
            for cn in self.by_cols:
                res.columns[cn] = self._extract_key_column(cn)
            return res
        else:
            agg = {}
            for cn in self.df.columns:
                if cn not in self.by_cols:
                    c = self.df.columns[cn]
                    if c.dtype == TYPE_INT or c.dtype == TYPE_DOUBLE:
                        agg[cn] = func_name
            return self.agg(agg)

    @cython.binding(False)
    def sum(self, str col_name=None):
        return self._convenience_agg('sum', col_name)

    @cython.binding(False)
    def mean(self, str col_name=None):
        return self._convenience_agg('mean', col_name)

    @cython.binding(False)
    def count(self):
        cdef FastDataFrame res = FastDataFrame()
        res.df_length = self.num_groups
        res.columns['__count__'] = self._native_count()
        for cn in self.by_cols:
            res.columns[cn] = self._extract_key_column(cn)
        return res

    @cython.binding(False)
    def min(self, str col_name=None):
        return self._convenience_agg('min', col_name)

    @cython.binding(False)
    def max(self, str col_name=None):
        return self._convenience_agg('max', col_name)

    @cython.binding(False)
    def var(self, str col_name=None):
        return self._convenience_agg('var', col_name)

    @cython.binding(False)
    def std(self, str col_name=None):
        return self._convenience_agg('std', col_name)

    @cython.binding(False)
    def apply(self, object func):
        cdef list sub_dfs = []
        cdef int g_id
        cdef Py_ssize_t sz
        cdef Py_ssize_t* indices_c
        cdef FastDataFrame sub_df
        cdef object func_res

        for g_id in range(self.num_groups):
            sz = self.group_rows[0][g_id].size()
            indices_c = <Py_ssize_t*>malloc(sz * sizeof(Py_ssize_t))
            for i in range(sz):
                indices_c[i] = self.group_rows[0][g_id][i]
            sub_df = self.df._take(indices_c, sz)
            free(indices_c)
            func_res = func(sub_df)
            sub_dfs.append(func_res)

        if len(sub_dfs) == 0:
            return FastDataFrame()
        if len(sub_dfs) == 1:
            return sub_dfs[0]

        cdef FastDataFrame result = sub_dfs[0]
        for i in range(1, len(sub_dfs)):
            result.concat(sub_dfs[i])
        return result

    @cython.binding(False)
    def __repr__(self):
        return f"<FastDataFrameGroupBy: {self.num_groups} groups, by={self.by_cols}>"
