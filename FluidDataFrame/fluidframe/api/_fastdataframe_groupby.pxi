# _fastdataframe_groupby.pxi — GroupBy accessor pattern (module-level cdef)
# Included at module level in fastdataframe.pyx (after class definition)
# Replaces standalone groupby.pyx with accessor pattern like _DataFrameMetricAccessor

from libc.stdint cimport uint64_t, int32_t
from libc.math cimport NAN, INFINITY, sqrt
from cpython.unicode cimport Py_UCS2
from fluidframe.core._base cimport aligned_malloc, aligned_realloc, os_aligned_free

cdef enum _AggFunc:
    _AGG_SUM = 0
    _AGG_MEAN = 1
    _AGG_MIN = 2
    _AGG_MAX = 3
    _AGG_VAR = 4
    _AGG_STD = 5

cdef _AggFunc _resolve_agg_func(str func) except? _AGG_SUM:
    if func == 'sum': return _AGG_SUM
    elif func == 'mean': return _AGG_MEAN
    elif func == 'min': return _AGG_MIN
    elif func == 'max': return _AGG_MAX
    elif func == 'var': return _AGG_VAR
    elif func == 'std': return _AGG_STD
    else: raise ValueError(f"Unknown aggregation function: {func}")

# ============================================================
# Pure C hash table (open addressing) replacing unordered_map
# ============================================================

cdef struct _GroupHashEntry:
    uint64_t hash
    int group_id
    bint occupied

cdef struct _GroupHashTable:
    _GroupHashEntry* entries
    Py_ssize_t capacity
    Py_ssize_t size

cdef _GroupHashTable* _ght_new(Py_ssize_t initial_capacity) nogil:
    cdef _GroupHashTable* ht = <_GroupHashTable*>aligned_malloc(sizeof(_GroupHashTable))
    if not ht: return NULL
    ht.capacity = initial_capacity
    ht.size = 0
    ht.entries = <_GroupHashEntry*>aligned_malloc(initial_capacity * sizeof(_GroupHashEntry))
    if not ht.entries:
        os_aligned_free(ht)
        return NULL
    cdef Py_ssize_t i
    for i in range(initial_capacity):
        ht.entries[i].occupied = 0
    return ht

cdef void _ght_free(_GroupHashTable* ht) nogil:
    if ht != NULL:
        if ht.entries != NULL: os_aligned_free(ht.entries)
        os_aligned_free(ht)

cdef int _ght_get_or_insert(_GroupHashTable* ht, uint64_t h) nogil:
    if ht.size >= ht.capacity:
        return -1
    cdef Py_ssize_t idx = <Py_ssize_t>(h % <uint64_t>ht.capacity)
    cdef Py_ssize_t probe_count = 0

    while probe_count < ht.capacity:
        if not ht.entries[idx].occupied:
            ht.entries[idx].hash = h
            ht.entries[idx].group_id = ht.size
            ht.entries[idx].occupied = 1
            ht.size += 1
            return ht.size - 1
        if ht.entries[idx].hash == h:
            return ht.entries[idx].group_id
        idx = (idx + 1) % ht.capacity
        probe_count += 1
    return -1

# ============================================================
# Pure C dynamic arrays replacing vector
# ============================================================

cdef struct _GroupRows:
    Py_ssize_t** rows
    Py_ssize_t* row_sizes
    Py_ssize_t* row_capacities
    Py_ssize_t count

cdef _GroupRows* _group_rows_new() nogil:
    cdef _GroupRows* gr = <_GroupRows*>aligned_malloc(sizeof(_GroupRows))
    if not gr: return NULL
    gr.count = 0
    gr.rows = NULL
    gr.row_sizes = NULL
    gr.row_capacities = NULL
    return gr

cdef void _group_rows_free(_GroupRows* gr) nogil:
    if gr == NULL: return
    cdef Py_ssize_t i
    if gr.rows != NULL:
        for i in range(gr.count):
            if gr.rows[i] != NULL: os_aligned_free(gr.rows[i])
        os_aligned_free(gr.rows)
    if gr.row_sizes != NULL: os_aligned_free(gr.row_sizes)
    if gr.row_capacities != NULL: os_aligned_free(gr.row_capacities)
    os_aligned_free(gr)

cdef int _group_rows_add_group(_GroupRows* gr) nogil:
    """Add a new group. Returns group_id."""
    cdef Py_ssize_t new_count = gr.count + 1
    cdef Py_ssize_t* new_sizes
    cdef Py_ssize_t* new_caps
    cdef Py_ssize_t** new_rows

    new_sizes = <Py_ssize_t*>aligned_realloc(gr.row_sizes, max(gr.count, 1) * sizeof(Py_ssize_t), new_count * sizeof(Py_ssize_t))
    if not new_sizes: return -1

    new_caps = <Py_ssize_t*>aligned_realloc(gr.row_capacities, max(gr.count, 1) * sizeof(Py_ssize_t), new_count * sizeof(Py_ssize_t))
    if not new_caps:
        os_aligned_free(new_sizes)
        return -1

    new_rows = <Py_ssize_t**>aligned_realloc(gr.rows, max(gr.count, 1) * sizeof(Py_ssize_t*), new_count * sizeof(Py_ssize_t*))
    if not new_rows:
        os_aligned_free(new_sizes)
        os_aligned_free(new_caps)
        return -1

    gr.row_sizes = new_sizes
    gr.row_capacities = new_caps
    gr.rows = new_rows

    gr.row_sizes[gr.count] = 0
    gr.row_capacities[gr.count] = 4
    gr.rows[gr.count] = <Py_ssize_t*>aligned_malloc(4 * sizeof(Py_ssize_t))
    if not gr.rows[gr.count]: return -1
    gr.count = new_count
    return gr.count - 1

cdef int _group_rows_append(_GroupRows* gr, int group_id, Py_ssize_t row_idx) nogil:
    """Append a row index to a group."""
    cdef Py_ssize_t new_cap
    cdef Py_ssize_t* new_ptr
    if gr.row_sizes[group_id] >= gr.row_capacities[group_id]:
        new_cap = gr.row_capacities[group_id] * 2
        new_ptr = <Py_ssize_t*>aligned_realloc(gr.rows[group_id], new_cap * sizeof(Py_ssize_t), gr.row_capacities[group_id] * sizeof(Py_ssize_t))
        if not new_ptr: return -1
        gr.rows[group_id] = new_ptr
        gr.row_capacities[group_id] = new_cap
    gr.rows[group_id][gr.row_sizes[group_id]] = row_idx
    gr.row_sizes[group_id] += 1
    return 0

# ============================================================
# FNV-1a hashing (same as original)
# ============================================================

cdef uint64_t _gb_fnv1a_bytes(char* data, Py_ssize_t length) noexcept nogil:
    cdef uint64_t h = 14695981039346656037ULL
    cdef Py_ssize_t i
    for i in range(length):
        h ^= <uint64_t><unsigned char>data[i]
        h *= 1099511628211ULL
    return h

cdef uint64_t _gb_fnv1a_ucs2(Py_UCS2* data, Py_ssize_t length) noexcept nogil:
    cdef uint64_t h = 14695981039346656037ULL
    cdef Py_ssize_t i
    for i in range(length):
        h ^= <uint64_t>data[i]
        h *= 1099511628211ULL
    return h

cdef uint64_t _gb_mix_hash(uint64_t h, uint64_t val) noexcept nogil:
    h ^= val
    h *= 1099511628211ULL
    return h

# ============================================================
# Column pointer extraction for fast hashing/comparison
# ============================================================

cdef struct _ByColPtrs:
    void** data
    void** offsets
    int* types
    int* cols          # 每列子列数：2D 列 = cols，其余 = 1
    Py_ssize_t* strides  # 每列行步长（2D 列 data_c[sub*stride + row]），1D 列 = length
    int count

cdef _ByColPtrs _extract_by_col_ptrs(FastDataFrame df, list by_cols):
    cdef _ByColPtrs ptrs
    cdef int n = len(by_cols)
    ptrs.count = n
    ptrs.data = <void**>aligned_malloc(max(n, 1) * sizeof(void*))
    ptrs.offsets = <void**>aligned_malloc(max(n, 1) * sizeof(void*))
    ptrs.types = <int*>aligned_malloc(max(n, 1) * sizeof(int))
    ptrs.cols = <int*>aligned_malloc(max(n, 1) * sizeof(int))
    ptrs.strides = <Py_ssize_t*>aligned_malloc(max(n, 1) * sizeof(Py_ssize_t))
    if not ptrs.data or not ptrs.offsets or not ptrs.types or not ptrs.cols or not ptrs.strides:
        if ptrs.data: os_aligned_free(ptrs.data)
        if ptrs.offsets: os_aligned_free(ptrs.offsets)
        if ptrs.types: os_aligned_free(ptrs.types)
        if ptrs.cols: os_aligned_free(ptrs.cols)
        if ptrs.strides: os_aligned_free(ptrs.strides)
        raise MemoryError()
    cdef str col_name
    cdef BaseColumn col
    cdef int idx
    cdef bint is2d
    cdef Py_ssize_t c_col, r_col
    for idx in range(n):
        col_name = by_cols[idx]
        col = df.columns[col_name]
        ptrs.types[idx] = col.dtype
        ptrs.offsets[idx] = NULL
        # 2D 列（is_2d 且 cols>1）：按完整行向量参与分组；其余列子列数=1
        is2d, c_col, r_col = _col_shape_probe(col)
        if is2d and c_col > 1 and col.dtype != TYPE_CHAR256 and col.dtype != TYPE_UCS2:
            ptrs.cols[idx] = <int>c_col
            ptrs.strides[idx] = r_col
        else:
            ptrs.cols[idx] = 1
            ptrs.strides[idx] = df.df_length
        if col.dtype == TYPE_INT:
            ptrs.data[idx] = <void*>(<IntColumn>col).data_c
        elif col.dtype == TYPE_DOUBLE:
            ptrs.data[idx] = <void*>(<DoubleColumn>col).data_c
        elif col.dtype == TYPE_CHAR256:
            ptrs.data[idx] = <void*>(<AsciiColumn>col).data_c
            ptrs.offsets[idx] = <void*>(<AsciiColumn>col).offsets_c
        elif col.dtype == TYPE_UCS2:
            ptrs.data[idx] = <void*>(<UCS2Column>col).chars_c
            ptrs.offsets[idx] = <void*>(<UCS2Column>col).offsets_c
        elif col.dtype == TYPE_DATE:
            ptrs.data[idx] = <void*>(<DateColumn>col).data_c
        elif col.dtype == TYPE_BOOL:
            ptrs.data[idx] = <void*>(<BoolColumn>col).data_c
        elif col.dtype == TYPE_CATEGORY:
            ptrs.data[idx] = <void*>(<CategoryColumn>col).codes_c
        elif col.dtype == TYPE_DATETIME:
            ptrs.data[idx] = <void*>(<DateTimeColumn>col).data_c
        elif col.dtype == TYPE_TIME:
            ptrs.data[idx] = <void*>(<TimeColumn>col).data_c
    return ptrs

cdef void _free_by_col_ptrs(_ByColPtrs ptrs) nogil:
    if ptrs.data != NULL: os_aligned_free(ptrs.data)
    if ptrs.offsets != NULL: os_aligned_free(ptrs.offsets)
    if ptrs.types != NULL: os_aligned_free(ptrs.types)
    if ptrs.cols != NULL: os_aligned_free(ptrs.cols)
    if ptrs.strides != NULL: os_aligned_free(ptrs.strides)

# ============================================================
# Fast row hashing and comparison using extracted pointers
# ============================================================

cdef uint64_t _gb_hash_row(_ByColPtrs ptrs, Py_ssize_t row_idx) nogil:
    cdef uint64_t h = 14695981039346656037ULL
    cdef int col_idx, sub, n_sub
    cdef uint64_t val
    cdef int* i_data
    cdef double* d_data
    cdef long long* t_data
    cdef char* c_data
    cdef unsigned char* b_data
    cdef Py_UCS2* u_data
    cdef Py_ssize_t start, end, stride

    for col_idx in range(ptrs.count):
        n_sub = ptrs.cols[col_idx]
        stride = ptrs.strides[col_idx]
        if ptrs.types[col_idx] == TYPE_INT:
            i_data = <int*>ptrs.data[col_idx]
            for sub in range(n_sub):
                h = _gb_mix_hash(h, <uint64_t>i_data[sub * stride + row_idx])
        elif ptrs.types[col_idx] == TYPE_DOUBLE:
            d_data = <double*>ptrs.data[col_idx]
            for sub in range(n_sub):
                val = (<uint64_t*>(&d_data[sub * stride + row_idx]))[0]
                h = _gb_mix_hash(h, val)
        elif ptrs.types[col_idx] == TYPE_DATE:
            t_data = <long long*>ptrs.data[col_idx]
            for sub in range(n_sub):
                h = _gb_mix_hash(h, <uint64_t>t_data[sub * stride + row_idx])
        elif ptrs.types[col_idx] == TYPE_CHAR256:
            c_data = <char*>ptrs.data[col_idx]
            start = (<Py_ssize_t*>ptrs.offsets[col_idx])[row_idx]
            end = (<Py_ssize_t*>ptrs.offsets[col_idx])[row_idx + 1]
            h = _gb_mix_hash(h, _gb_fnv1a_bytes(c_data + start, end - start))
        elif ptrs.types[col_idx] == TYPE_UCS2:
            u_data = <Py_UCS2*>ptrs.data[col_idx]
            start = (<Py_ssize_t*>ptrs.offsets[col_idx])[row_idx]
            end = (<Py_ssize_t*>ptrs.offsets[col_idx])[row_idx + 1]
            h = _gb_mix_hash(h, _gb_fnv1a_ucs2(u_data + start, end - start))
        elif ptrs.types[col_idx] == TYPE_BOOL:
            b_data = <unsigned char*>ptrs.data[col_idx]
            for sub in range(n_sub):
                h = _gb_mix_hash(h, <uint64_t>b_data[sub * stride + row_idx])
        elif ptrs.types[col_idx] == TYPE_CATEGORY:
            i_data = <int*>ptrs.data[col_idx]
            for sub in range(n_sub):
                h = _gb_mix_hash(h, <uint64_t>i_data[sub * stride + row_idx])
        elif ptrs.types[col_idx] == TYPE_DATETIME:
            t_data = <long long*>ptrs.data[col_idx]
            for sub in range(n_sub):
                h = _gb_mix_hash(h, <uint64_t>t_data[sub * stride + row_idx])
        elif ptrs.types[col_idx] == TYPE_TIME:
            i_data = <int*>ptrs.data[col_idx]
            for sub in range(n_sub):
                h = _gb_mix_hash(h, <uint64_t>i_data[sub * stride + row_idx])
        else:
            h = _gb_mix_hash(h, 0)
    return h

cdef bint _gb_rows_equal(_ByColPtrs ptrs, Py_ssize_t row1, Py_ssize_t row2) noexcept nogil:
    cdef int col_idx, sub, n_sub
    cdef int* i_data
    cdef double* d_data
    cdef long long* t_data
    cdef char* c_data
    cdef unsigned char* b_data
    cdef Py_ssize_t start1, end1, start2, end2, k, stride

    for col_idx in range(ptrs.count):
        n_sub = ptrs.cols[col_idx]
        stride = ptrs.strides[col_idx]
        if ptrs.types[col_idx] == TYPE_INT:
            i_data = <int*>ptrs.data[col_idx]
            for sub in range(n_sub):
                if i_data[sub * stride + row1] != i_data[sub * stride + row2]: return False
        elif ptrs.types[col_idx] == TYPE_DOUBLE:
            d_data = <double*>ptrs.data[col_idx]
            for sub in range(n_sub):
                if d_data[sub * stride + row1] != d_data[sub * stride + row2]: return False
        elif ptrs.types[col_idx] == TYPE_DATE:
            t_data = <long long*>ptrs.data[col_idx]
            for sub in range(n_sub):
                if t_data[sub * stride + row1] != t_data[sub * stride + row2]: return False
        elif ptrs.types[col_idx] == TYPE_CHAR256:
            c_data = <char*>ptrs.data[col_idx]
            start1 = (<Py_ssize_t*>ptrs.offsets[col_idx])[row1]
            end1 = (<Py_ssize_t*>ptrs.offsets[col_idx])[row1 + 1]
            start2 = (<Py_ssize_t*>ptrs.offsets[col_idx])[row2]
            end2 = (<Py_ssize_t*>ptrs.offsets[col_idx])[row2 + 1]
            if (end1 - start1) != (end2 - start2): return False
            for k in range(end1 - start1):
                if c_data[start1 + k] != c_data[start2 + k]: return False
        elif ptrs.types[col_idx] == TYPE_UCS2:
            start1 = (<Py_ssize_t*>ptrs.offsets[col_idx])[row1]
            end1 = (<Py_ssize_t*>ptrs.offsets[col_idx])[row1 + 1]
            start2 = (<Py_ssize_t*>ptrs.offsets[col_idx])[row2]
            end2 = (<Py_ssize_t*>ptrs.offsets[col_idx])[row2 + 1]
            if (end1 - start1) != (end2 - start2): return False
            i_data = <int*>ptrs.data[col_idx]
            for k in range(end1 - start1):
                if (<Py_UCS2*>i_data)[start1 + k] != (<Py_UCS2*>i_data)[start2 + k]: return False
        elif ptrs.types[col_idx] == TYPE_BOOL:
            b_data = <unsigned char*>ptrs.data[col_idx]
            for sub in range(n_sub):
                if b_data[sub * stride + row1] != b_data[sub * stride + row2]: return False
        elif ptrs.types[col_idx] == TYPE_CATEGORY:
            i_data = <int*>ptrs.data[col_idx]
            for sub in range(n_sub):
                if i_data[sub * stride + row1] != i_data[sub * stride + row2]: return False
        elif ptrs.types[col_idx] == TYPE_DATETIME:
            t_data = <long long*>ptrs.data[col_idx]
            for sub in range(n_sub):
                if t_data[sub * stride + row1] != t_data[sub * stride + row2]: return False
        elif ptrs.types[col_idx] == TYPE_TIME:
            i_data = <int*>ptrs.data[col_idx]
            for sub in range(n_sub):
                if i_data[sub * stride + row1] != i_data[sub * stride + row2]: return False
    return True

# ============================================================
# Group building
# ============================================================

cdef int _gb_build_groups(FastDataFrame df, list by_cols,
                            _GroupHashTable* ht, _GroupRows* gr,
                            _ByColPtrs ptrs) except -1:
    cdef Py_ssize_t n = df.df_length
    cdef Py_ssize_t i
    cdef uint64_t h
    cdef int g_id, ret
    cdef Py_ssize_t first_seen_row
    cdef bint found
    cdef int existing_g

    for i in range(n):
        h = _gb_hash_row(ptrs, i)
        g_id = _ght_get_or_insert(ht, h)

        if g_id == -1:
            raise MemoryError("GroupBy hash table capacity exceeded limit")
        if g_id >= gr.count:
            ret = _group_rows_add_group(gr)
            if ret < 0: raise MemoryError("Failed to add group in GroupBy")
        first_seen_row = gr.rows[g_id][0] if gr.row_sizes[g_id] > 0 else -1
        if first_seen_row >= 0 and not _gb_rows_equal(ptrs, i, first_seen_row):
            found = False
            for existing_g in range(gr.count):
                if gr.row_sizes[existing_g] > 0:
                    if _gb_rows_equal(ptrs, i, gr.rows[existing_g][0]):
                        ret = _group_rows_append(gr, existing_g, i)
                        if ret < 0: raise MemoryError("Failed to append row in GroupBy")
                        found = True
                        break
            if not found:
                g_id = gr.count
                ret = _group_rows_add_group(gr)
                if ret < 0: raise MemoryError("Failed to add group in GroupBy")
                ret = _group_rows_append(gr, g_id, i)
                if ret < 0: raise MemoryError("Failed to append row in GroupBy")
        else:
            ret = _group_rows_append(gr, g_id, i)
            if ret < 0: raise MemoryError("Failed to append row in GroupBy")

# ============================================================
# Aggregation helpers
# ============================================================

cdef inline double _gb_agg_int_sub(int* ptr, Py_ssize_t stride, int sub, Py_ssize_t* idxs, Py_ssize_t sz, _AggFunc agg) noexcept nogil:
    cdef double val = 0.0, d, mean = 0.0, M2 = 0.0, delta, delta2
    cdef Py_ssize_t i, base = sub * stride

    if agg == _AGG_SUM:
        for i in range(sz): val += ptr[base + idxs[i]]
        return val
    if agg == _AGG_MEAN:
        if sz == 0: return NAN
        for i in range(sz): val += ptr[base + idxs[i]]
        return val / <double>sz
    if agg == _AGG_MIN:
        if sz == 0: return NAN
        val = <double>ptr[base + idxs[0]]
        for i in range(1, sz):
            d = <double>ptr[base + idxs[i]]
            if d < val: val = d
        return val
    if agg == _AGG_MAX:
        if sz == 0: return NAN
        val = <double>ptr[base + idxs[0]]
        for i in range(1, sz):
            d = <double>ptr[base + idxs[i]]
            if d > val: val = d
        return val
    if agg == _AGG_VAR or agg == _AGG_STD:
        if sz < 2: return NAN
        # 采用 Welford 算法实现单趟内存访问，省掉一次间接指针查询，速度翻倍
        for i in range(sz):
            val = <double>ptr[base + idxs[i]]
            delta = val - mean
            mean += delta / (i + 1)
            delta2 = val - mean
            M2 += delta * delta2

        if agg == _AGG_VAR: return M2 / <double>(sz - 1)
        return sqrt(M2 / <double>(sz - 1))

    return NAN

cdef inline double _gb_agg_double_sub(double* ptr, Py_ssize_t stride, int sub, Py_ssize_t* idxs, Py_ssize_t sz, _AggFunc agg) noexcept nogil:
    cdef double val = 0.0, d, mean = 0.0, M2 = 0.0, delta, delta2
    cdef Py_ssize_t i, base = sub * stride

    if agg == _AGG_SUM:
        for i in range(sz): val += ptr[base + idxs[i]]
        return val

    if agg == _AGG_MEAN:
        if sz == 0: return NAN
        for i in range(sz): val += ptr[base + idxs[i]]
        return val / <double>sz

    # 使用 Welford 处理 VAR/STD
    if agg == _AGG_VAR or agg == _AGG_STD:
        if sz < 2: return NAN
        for i in range(sz):
            val = ptr[base + idxs[i]]
            delta = val - mean
            mean += delta / (i + 1)
            delta2 = val - mean
            M2 += delta * delta2

        if agg == _AGG_VAR: return M2 / <double>(sz - 1)
        return sqrt(M2 / <double>(sz - 1))

    if agg == _AGG_MIN:
        if sz == 0: return NAN
        val = ptr[base + idxs[0]]
        for i in range(1, sz):
            d = ptr[base + idxs[i]]
            if d < val: val = d
        return val

    if agg == _AGG_MAX:
        if sz == 0: return NAN
        val = ptr[base + idxs[0]]
        for i in range(1, sz):
            d = ptr[base + idxs[i]]
            if d > val: val = d
        return val
    return NAN

cdef DoubleColumn _gb_native_agg(FastDataFrame df, _GroupRows* gr, str col_name, str func):
    cdef BaseColumn orig_col = df.columns[col_name]
    cdef bint is2d
    cdef Py_ssize_t c_col, r_col
    is2d, c_col, r_col = _col_shape_probe(orig_col)

    cdef Py_ssize_t out_n = gr.count
    cdef _AggFunc agg_type = _resolve_agg_func(func)
    cdef int g_id, sub
    cdef Py_ssize_t sz
    cdef DoubleColumn res_col
    cdef double* out_ptr

    if is2d and c_col > 1:
        res_col = DoubleColumn(max(out_n * c_col, 1), cols=c_col)
        res_col.length = out_n * c_col
        if orig_col.dtype == TYPE_DOUBLE:
            d_ptr = (<DoubleColumn>orig_col).data_c
            for sub in range(c_col):
                for g_id in range(out_n):
                    res_col.data_c[sub * out_n + g_id] = _gb_agg_double_sub(
                        d_ptr, r_col, sub, gr.rows[g_id], gr.row_sizes[g_id], agg_type)
        else:
            i_ptr = (<IntColumn>orig_col).data_c
            for sub in range(c_col):
                for g_id in range(out_n):
                    res_col.data_c[sub * out_n + g_id] = _gb_agg_int_sub(
                        i_ptr, r_col, sub, gr.rows[g_id], gr.row_sizes[g_id], agg_type)
        return res_col

    res_col = DoubleColumn(max(out_n, 1))
    res_col.length = out_n
    out_ptr = res_col.data_c

    if orig_col.dtype == TYPE_DOUBLE:
        col_dbl = <DoubleColumn>orig_col
        if agg_type == _AGG_SUM:
            for g_id in range(out_n): out_ptr[g_id] = col_dbl._sum_rows(gr.rows[g_id], gr.row_sizes[g_id])
        elif agg_type == _AGG_MEAN:
            for g_id in range(out_n): out_ptr[g_id] = col_dbl._mean_rows(gr.rows[g_id], gr.row_sizes[g_id])
        elif agg_type == _AGG_MIN:
            for g_id in range(out_n): out_ptr[g_id] = col_dbl._min_rows(gr.rows[g_id], gr.row_sizes[g_id])
        elif agg_type == _AGG_MAX:
            for g_id in range(out_n): out_ptr[g_id] = col_dbl._max_rows(gr.rows[g_id], gr.row_sizes[g_id])
        elif agg_type == _AGG_VAR:
            for g_id in range(out_n): out_ptr[g_id] = col_dbl._var_rows(gr.rows[g_id], gr.row_sizes[g_id])
        elif agg_type == _AGG_STD:
            for g_id in range(out_n): out_ptr[g_id] = col_dbl._std_rows(gr.rows[g_id], gr.row_sizes[g_id])
    elif orig_col.dtype == TYPE_INT:
        col_int = <IntColumn>orig_col
        if agg_type == _AGG_SUM:
            for g_id in range(out_n): out_ptr[g_id] = col_int._sum_rows(gr.rows[g_id], gr.row_sizes[g_id])
        elif agg_type == _AGG_MEAN:
            for g_id in range(out_n): out_ptr[g_id] = col_int._mean_rows(gr.rows[g_id], gr.row_sizes[g_id])
        elif agg_type == _AGG_MIN:
            for g_id in range(out_n): out_ptr[g_id] = col_int._min_rows(gr.rows[g_id], gr.row_sizes[g_id])
        elif agg_type == _AGG_MAX:
            for g_id in range(out_n): out_ptr[g_id] = col_int._max_rows(gr.rows[g_id], gr.row_sizes[g_id])
        elif agg_type == _AGG_VAR:
            for g_id in range(out_n): out_ptr[g_id] = col_int._var_rows(gr.rows[g_id], gr.row_sizes[g_id])
        elif agg_type == _AGG_STD:
            for g_id in range(out_n): out_ptr[g_id] = col_int._std_rows(gr.rows[g_id], gr.row_sizes[g_id])

    return res_col

cdef IntColumn _gb_native_count(_GroupRows* gr):
    cdef IntColumn res_col = IntColumn(max(gr.count, 1))
    res_col.length = gr.count
    cdef int g_id
    for g_id in range(gr.count):
        res_col.data_c[g_id] = <int>gr.row_sizes[g_id]
    return res_col

cdef object _gb_extract_key_column(FastDataFrame df, _GroupRows* gr, str col_name):
    cdef BaseColumn orig = df.columns[col_name]
    cdef int i
    cdef IntColumn res_int
    cdef DoubleColumn res_dbl
    cdef Py_ssize_t* fs
    cdef FastDataFrame tmp

    # 使用 shape_probe
    cdef bint is2d
    cdef Py_ssize_t c_col, r_col
    is2d, c_col, r_col = _col_shape_probe(orig)

    # 如果是 2D 类型 或者 是不支持直拷的复杂类型，走安全的 `_take` API
    if (is2d and c_col > 1) or (orig.dtype != TYPE_INT and orig.dtype != TYPE_DOUBLE):
        fs = <Py_ssize_t*>aligned_malloc(max(gr.count, 1) * sizeof(Py_ssize_t))
        if not fs: raise MemoryError()
        for i in range(gr.count):
            fs[i] = gr.rows[i][0]
        tmp = df._take(fs, gr.count)
        os_aligned_free(fs)
        return tmp.columns[col_name]

    # 1D 的极速直拷路径
    if orig.dtype == TYPE_INT:
        res_int = IntColumn(max(gr.count, 1))
        res_int.length = gr.count
        for i in range(gr.count):
            res_int.data_c[i] = (<IntColumn>orig).data_c[gr.rows[i][0]]
        return res_int
    elif orig.dtype == TYPE_DOUBLE:
        res_dbl = DoubleColumn(max(gr.count, 1))
        res_dbl.length = gr.count
        for i in range(gr.count):
            res_dbl.data_c[i] = (<DoubleColumn>orig).data_c[gr.rows[i][0]]
        return res_dbl

cdef IntColumn _gb_native_idx(FastDataFrame df, _GroupRows* gr, str col_name, bint find_max):
    # 2D 列取主子列（sub=0）比较：idxmax/idxmin 不确定向量级比较，沿用 _get_col_value_as_double 的 sub0 约定
    cdef IntColumn res_col = IntColumn(max(gr.count, 1))
    res_col.length = gr.count
    cdef BaseColumn orig_col = df.columns[col_name]
    cdef int g_id
    cdef Py_ssize_t sz, row_idx, best_row, i
    cdef double best_val, cur_val
    for g_id in range(gr.count):
        sz = gr.row_sizes[g_id]
        if sz == 0:
            res_col.data_c[g_id] = 0
            continue
        best_row = gr.rows[g_id][0]
        best_val = df._get_col_value_as_double(orig_col, best_row)
        for i in range(1, sz):
            row_idx = gr.rows[g_id][i]
            cur_val = df._get_col_value_as_double(orig_col, row_idx)
            if find_max:
                if cur_val > best_val:
                    best_val = cur_val
                    best_row = row_idx
            else:
                if cur_val < best_val:
                    best_val = cur_val
                    best_row = row_idx
        res_col.data_c[g_id] = <int>best_row
    return res_col

cdef object _gb_python_compute(list vals, str func_name):
    """单子列 Python 级聚合计算（维持原 median/quantile 的语义不变）"""
    cdef list sorted_vals
    cdef Py_ssize_t mid
    if func_name == 'sum':
        return sum(vals)
    elif func_name == 'mean':
        return sum(vals) / len(vals) if vals else NAN
    elif func_name == 'min':
        return min(vals) if vals else NAN
    elif func_name == 'max':
        return max(vals) if vals else NAN
    elif func_name == 'count':
        return len(vals)
    elif func_name == 'median':
        if not vals:
            return NAN
        sorted_vals = sorted(vals)
        mid = len(sorted_vals) // 2
        if len(sorted_vals) % 2 == 0:
            return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
        return sorted_vals[mid]
    elif func_name == 'quantile':
        if not vals:
            return NAN
        sorted_vals = sorted(vals)
        mid = len(sorted_vals) // 2
        if len(sorted_vals) % 2 == 0:
            return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
        return sorted_vals[mid]
    else:
        raise ValueError(f"Unsupported python aggregation function: {func_name}")

cdef object _gb_python_agg(FastDataFrame df, _GroupRows* gr, str col_name, str func_name):
    cdef list results = []
    cdef int g_id, sub
    cdef Py_ssize_t sz, row_idx, i, out_n = gr.count
    cdef BaseColumn orig_col = df.columns[col_name]
    cdef list vals
    cdef bint is2d
    cdef Py_ssize_t c_col, r_col
    cdef DoubleColumn res_col

    is2d, c_col, r_col = _col_shape_probe(orig_col)

    cdef double* d_ptr = NULL
    cdef int* i_ptr = NULL
    if orig_col.dtype == TYPE_DOUBLE:
        d_ptr = (<DoubleColumn>orig_col).data_c
    elif orig_col.dtype == TYPE_INT:
        i_ptr = (<IntColumn>orig_col).data_c

    if is2d and c_col > 1:
        res_col = DoubleColumn(max(out_n * c_col, 1), cols=c_col)
        res_col.length = out_n * c_col
        for sub in range(c_col):
            results = []
            for g_id in range(out_n):
                sz = gr.row_sizes[g_id]
                vals = []
                for i in range(sz):
                    row_idx = gr.rows[g_id][i]
                    if d_ptr != NULL:
                        vals.append(d_ptr[sub * r_col + row_idx])
                    elif i_ptr != NULL:
                        vals.append(<double>i_ptr[sub * r_col + row_idx])
                    else:
                        vals.append(_col_row_key(orig_col, row_idx)[sub])
                results.append(_gb_python_compute(vals, func_name))
            for g_id in range(out_n):
                res_col.data_c[sub * out_n + g_id] = <double>results[g_id] if results[g_id] is not None else NAN
        return res_col

    res_col = DoubleColumn(max(out_n, 1))
    res_col.length = out_n
    for g_id in range(out_n):
        sz = gr.row_sizes[g_id]
        vals = []
        for i in range(sz):
            row_idx = gr.rows[g_id][i]
            vals.append(orig_col[row_idx])
        results.append(_gb_python_compute(vals, func_name))
    for g_id in range(out_n):
        res_col.data_c[g_id] = <double>results[g_id] if results[g_id] is not None else NAN
    return res_col

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

    cdef FastDataFrame _convenience_agg(self, str func_name, object col_name):
        self._ensure_built()
        cdef FastDataFrame res
        cdef dict agg
        cdef str cn
        cdef BaseColumn c
        if col_name is not None:
            res = FastDataFrame()
            res.df_length = self._gr.count
            if func_name in ('sum', 'mean', 'min', 'max', 'var', 'std'):
                res.columns[col_name] = _gb_native_agg(self._df, self._gr, col_name, func_name)
            else:
                res.columns[col_name] = _gb_python_agg(self._df, self._gr, col_name, func_name)
            for cn in self._by_cols:
                res.columns[cn] = _gb_extract_key_column(self._df, self._gr, cn)
            return res
        else:
            agg = {}
            for cn in self._df.columns:
                if cn not in self._by_cols:
                    c = self._df.columns[cn]
                    if c.dtype == TYPE_INT or c.dtype == TYPE_DOUBLE:
                        agg[cn] = func_name
            return self.agg(agg)

    @cython.binding(False)
    def size(self):
        self._ensure_built()
        return self._gr.count

    @cython.binding(False)
    def agg(self, dict agg_dict):
        self._ensure_built()
        cdef FastDataFrame res = FastDataFrame()
        res.df_length = self._gr.count
        cdef str col_name, func_name
        for col_name, func_name in agg_dict.items():
            if func_name in ('sum', 'mean', 'min', 'max', 'var', 'std'):
                res.columns[col_name] = _gb_native_agg(self._df, self._gr, col_name, func_name)
            elif func_name == 'count':
                res.columns[col_name] = _gb_native_count(self._gr)
            else:
                res.columns[col_name] = _gb_python_agg(self._df, self._gr, col_name, func_name)
        for col_name in self._by_cols:
            if col_name not in res.columns:
                res.columns[col_name] = _gb_extract_key_column(self._df, self._gr, col_name)
        return res

    @cython.binding(False)
    def sum(self, str col_name=None):
        return self._convenience_agg('sum', col_name)

    @cython.binding(False)
    def mean(self, str col_name=None):
        return self._convenience_agg('mean', col_name)

    @cython.binding(False)
    def count(self):
        self._ensure_built()
        cdef FastDataFrame res = FastDataFrame()
        res.df_length = self._gr.count
        res.columns['__count__'] = _gb_native_count(self._gr)
        for cn in self._by_cols:
            res.columns[cn] = _gb_extract_key_column(self._df, self._gr, cn)
        return res

    @cython.binding(False)
    def min(self, str col_name=None):
        return self._convenience_agg('min', col_name)

    @cython.binding(False)
    def max(self, str col_name=None):
        return self._convenience_agg('max', col_name)

    @cython.binding(False)
    def median(self, str col_name=None):
        return self._convenience_agg('median', col_name)

    @cython.binding(False)
    def quantile(self, str col_name=None):
        return self._convenience_agg('quantile', col_name)

    @cython.binding(False)
    def var(self, str col_name=None):
        return self._convenience_agg('var', col_name)

    @cython.binding(False)
    def std(self, str col_name=None):
        return self._convenience_agg('std', col_name)

    @cython.binding(False)
    def apply(self, object func):
        self._ensure_built()
        cdef list sub_dfs = []
        cdef int g_id
        cdef Py_ssize_t sz
        cdef Py_ssize_t* indices_c
        cdef FastDataFrame sub_df
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
            return FastDataFrame()
        if len(sub_dfs) == 1:
            return sub_dfs[0]

        cdef FastDataFrame result = sub_dfs[0]
        for i in range(1, len(sub_dfs)):
            result = result.concat(sub_dfs[i])
        return result

    @cython.binding(False)
    def idxmax(self, str col_name=None):
        self._ensure_built()
        cdef FastDataFrame res = FastDataFrame()
        res.df_length = self._gr.count
        cdef str cn
        cdef BaseColumn c
        if col_name is not None:
            res.columns[col_name] = _gb_native_idx(self._df, self._gr, col_name, True)
        else:
            for cn in self._df.columns:
                if cn not in self._by_cols:
                    c = self._df.columns[cn]
                    if c.dtype == TYPE_INT or c.dtype == TYPE_DOUBLE:
                        res.columns[cn] = _gb_native_idx(self._df, self._gr, cn, True)
        for cn in self._by_cols:
            res.columns[cn] = _gb_extract_key_column(self._df, self._gr, cn)
        return res

    @cython.binding(False)
    def idxmin(self, str col_name=None):
        self._ensure_built()
        cdef FastDataFrame res = FastDataFrame()
        res.df_length = self._gr.count
        cdef str cn
        cdef BaseColumn c
        if col_name is not None:
            res.columns[col_name] = _gb_native_idx(self._df, self._gr, col_name, False)
        else:
            for cn in self._df.columns:
                if cn not in self._by_cols:
                    c = self._df.columns[cn]
                    if c.dtype == TYPE_INT or c.dtype == TYPE_DOUBLE:
                        res.columns[cn] = _gb_native_idx(self._df, self._gr, cn, False)
        for cn in self._by_cols:
            res.columns[cn] = _gb_extract_key_column(self._df, self._gr, cn)
        return res

    @cython.binding(False)
    def transform(self, object func):
        self._ensure_built()
        cdef Py_ssize_t n = self._df.df_length
        cdef FastDataFrame result = FastDataFrame()
        result.df_length = n
        result.index = self._df.index

        cdef str cn
        cdef list col_names = []
        cdef dict col_values = {}

        for cn in self._df.columns:
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
                if isinstance(transformed, FastDataFrame):
                    for cn in transformed.columns:
                        if cn not in col_values:
                            col_names.append(cn)
                            col_values[cn] = [None] * n
                    for cn in col_names:
                        if cn in transformed.columns:
                            src_col = transformed.columns[cn]
                            for i in range(sz):
                                row_idx = self._gr.rows[g_id][i]
                                col_values[cn][row_idx] = src_col[i]
            finally:
                os_aligned_free(indices_c)

        for cn in col_names:
            values = col_values[cn]
            nv = len(values)
            if nv == 0:
                result.columns[cn] = IntColumn(0)
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
                result.columns[cn] = dbl
            elif isinstance(sample, bool):
                bc = BoolColumn(max(nv, 1))
                bc.length = nv
                for vi in range(nv):
                    bc.data_c[vi] = 1 if values[vi] else 0
                result.columns[cn] = bc
            elif isinstance(sample, int):
                ic = IntColumn(max(nv, 1))
                ic.length = nv
                for vi in range(nv):
                    ic.data_c[vi] = <int>(values[vi] if values[vi] is not None else 0)
                result.columns[cn] = ic
            elif isinstance(sample, float):
                dc = DoubleColumn(max(nv, 1))
                dc.length = nv
                for vi in range(nv):
                    dc.data_c[vi] = <double>values[vi] if values[vi] is not None else NAN
                result.columns[cn] = dc
            elif isinstance(sample, str):
                ac = AsciiColumn(max(nv, 1))
                for vi in range(nv):
                    ac.append(str(values[vi] if values[vi] is not None else ''))
                result.columns[cn] = ac

        return result

    @cython.binding(False)
    def filter(self, object func):
        self._ensure_built()
        cdef FastDataFrame result = FastDataFrame()
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


cdef inline _DataFrameGroupByAccessor _make_groupby_accessor(FastDataFrame df, list by_cols):
    cdef _DataFrameGroupByAccessor acc = <_DataFrameGroupByAccessor>_DataFrameGroupByAccessor.__new__(_DataFrameGroupByAccessor)
    acc._df = df
    acc._by_cols = by_cols
    acc._ht = NULL
    acc._gr = NULL
    acc._ptrs.data = NULL
    acc._ptrs.offsets = NULL
    acc._ptrs.types = NULL
    acc._ptrs.count = 0
    acc._built = False
    return acc
