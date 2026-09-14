# _fastdataframe_agg.pxi — Aggregation (module-level cdef)
# Included at module level in fastdataframe.pyx (after class definition)

cdef _numeric_agg_impl(FastDataFrame self, str method, col_names=None):
    cdef dict result = {}
    cdef str name
    cdef object col
    if col_names is None:
        col_names = list(self._columns.keys())
    for name in col_names:
        if name not in self._columns: continue
        col = self._columns[name]
        if col.dtype == TYPE_INT or col.dtype == TYPE_DOUBLE:
            result[name] = getattr(col, method)()
    return result

cdef _numeric_agg_arg_impl(FastDataFrame self, str method, col_names, arg):
    cdef dict result = {}
    cdef str name
    cdef object col
    if col_names is None:
        col_names = list(self._columns.keys())
    for name in col_names:
        if name not in self._columns: continue
        col = self._columns[name]
        if col.dtype == TYPE_INT or col.dtype == TYPE_DOUBLE:
            result[name] = getattr(col, method)(arg)
    return result

cdef _mode_impl(FastDataFrame self, col_names=None):
    cdef dict result = {}
    cdef str name
    cdef object col
    if col_names is None:
        col_names = list(self._columns.keys())
    for name in col_names:
        if name not in self._columns: continue
        col = self._columns[name]
        if col.dtype == TYPE_INT or col.dtype == TYPE_DOUBLE:
            result[name] = col.mode()
    return result


cdef _numeric_agg_withcol_impl(FastDataFrame self, str method, col_names=None):
    cdef dict result = {}
    cdef str name
    cdef object col
    cdef bint is2d
    cdef Py_ssize_t c, r, k, i, offset
    cdef DoubleColumn buf
    cdef int dtype
    cdef int32_t* i_ptr
    cdef double* d_ptr
    cdef double* buf_ptr
    cdef object meth
    cdef list res_list

    if col_names is None:
        col_names = list(self._columns.keys())

    for name in col_names:
        if name not in self._columns:
            continue
        col = self._columns[name]
        dtype = col.dtype

        if dtype == TYPE_INT or dtype == TYPE_DOUBLE:
            is2d, c, r = _col_shape_probe(col)
            if is2d and c > 1:
                res_list = []
                if r > 0:
                    buf = DoubleColumn(r)
                    buf.length = r
                    buf_ptr = buf.data_c
                    meth = getattr(buf, method)

                    if dtype == TYPE_INT:
                        i_ptr = <int32_t*>(<IntColumn>col).data_c
                        for k in range(c):
                            offset = k * r
                            for i in range(r):
                                buf_ptr[i] = <double>i_ptr[offset + i]
                            res_list.append(meth())
                    else:
                        d_ptr = <double*>(<DoubleColumn>col).data_c
                        for k in range(c):
                            offset = k * r
                            for i in range(r):
                                buf_ptr[i] = d_ptr[offset + i]
                            res_list.append(meth())
                else:
                    # 空列边界情况保护：避免 r=0 且 2D 时返回单标量，破坏维度结构
                    for k in range(c):
                        res_list.append(getattr(col, method)())
                result[name] = res_list
            else:
                result[name] = getattr(col, method)()

    return result


cdef _numeric_agg_arg_withcol_impl(FastDataFrame self, str method, col_names, arg):
    cdef dict result = {}
    cdef str name
    cdef object col
    cdef bint is2d
    cdef Py_ssize_t c, r, k, i, offset
    cdef DoubleColumn buf
    cdef int dtype
    cdef int32_t* i_ptr
    cdef double* d_ptr
    cdef double* buf_ptr
    cdef object meth
    cdef list res_list

    if col_names is None:
        col_names = list(self._columns.keys())

    for name in col_names:
        if name not in self._columns:
            continue
        col = self._columns[name]
        dtype = col.dtype

        if dtype == TYPE_INT or dtype == TYPE_DOUBLE:
            is2d, c, r = _col_shape_probe(col)
            if is2d and c > 1:
                res_list = []
                if r > 0:
                    buf = DoubleColumn(r)
                    buf.length = r
                    buf_ptr = buf.data_c
                    meth = getattr(buf, method)

                    if dtype == TYPE_INT:
                        i_ptr = <int32_t*>(<IntColumn>col).data_c
                        for k in range(c):
                            offset = k * r
                            for i in range(r):
                                buf_ptr[i] = <double>i_ptr[offset + i]
                            res_list.append(meth(arg))
                    else:
                        d_ptr = <double*>(<DoubleColumn>col).data_c
                        for k in range(c):
                            offset = k * r
                            for i in range(r):
                                buf_ptr[i] = d_ptr[offset + i]
                            res_list.append(meth(arg))
                else:
                    for k in range(c):
                        res_list.append(getattr(col, method)(arg))
                result[name] = res_list
            else:
                result[name] = getattr(col, method)(arg)

    return result


cdef _describe_withcol_impl(FastDataFrame self):
    cdef dict stats = {}
    cdef str name
    cdef object col
    cdef bint is2d
    cdef Py_ssize_t c, r, k, i, offset
    cdef DoubleColumn buf
    cdef int dtype
    cdef int32_t* i_ptr
    cdef double* d_ptr
    cdef double* buf_ptr
    cdef list stats_list

    for name, col in self._columns.items():
        dtype = col.dtype
        if dtype == TYPE_INT or dtype == TYPE_DOUBLE:
            is2d, c, r = _col_shape_probe(col)
            if is2d and c > 1:
                stats_list = []
                if r > 0:
                    buf = DoubleColumn(r)
                    buf.length = r
                    buf_ptr = buf.data_c

                    if dtype == TYPE_INT:
                        i_ptr = <int32_t*>(<IntColumn>col).data_c
                        for k in range(c):
                            offset = k * r
                            for i in range(r):
                                buf_ptr[i] = <double>i_ptr[offset + i]
                            stats_list.append({
                                'count': r,
                                'mean': buf.mean(),
                                'std': buf.std(),
                                'min': buf.min(),
                                '50%': buf.median(),
                                'max': buf.max(),
                            })
                    else:
                        d_ptr = <double*>(<DoubleColumn>col).data_c
                        for k in range(c):
                            offset = k * r
                            for i in range(r):
                                buf_ptr[i] = d_ptr[offset + i]
                            stats_list.append({
                                'count': r,
                                'mean': buf.mean(),
                                'std': buf.std(),
                                'min': buf.min(),
                                '50%': buf.median(),
                                'max': buf.max(),
                            })
                else:
                    for k in range(c):
                        stats_list.append({
                            'count': 0,
                            'mean': col.mean(),
                            'std': col.std(),
                            'min': col.min(),
                            '50%': col.median(),
                            'max': col.max(),
                        })
                stats[name] = stats_list
            else:
                stats[name] = {
                    'count': r,
                    'mean': col.mean(),
                    'std': col.std(),
                    'min': col.min(),
                    '50%': col.median(),
                    'max': col.max(),
                }
    return stats

cdef _count_impl(FastDataFrame self, col_names=None):
    cdef dict result = {}
    cdef str name
    cdef BaseColumn col
    cdef Py_ssize_t count, i, total
    cdef int* i_ptr
    cdef double* d_ptr
    cdef unsigned char* bool_ptr
    if col_names is None:
        col_names = list(self._columns.keys())
    for name in col_names:
        if name in self._columns:
            col = self._columns[name]
            total = col.length
            count = 0
            if col.dtype == TYPE_INT:
                i_ptr = (<IntColumn>col).data_c
                for i in range(total):
                    if i_ptr[i] != 0:
                        count += 1
            elif col.dtype == TYPE_DOUBLE:
                d_ptr = (<DoubleColumn>col).data_c
                for i in range(total):
                    if not isnan(d_ptr[i]):
                        count += 1
            elif col.dtype == TYPE_BOOL:
                bool_ptr = (<BoolColumn>col).data_c
                for i in range(total):
                    if bool_ptr[i] != 0:
                        count += 1
            else:
                count = total
            result[name] = count
    return result

cdef inline double _qs_double(double* a, Py_ssize_t n, Py_ssize_t k) noexcept nogil:
    cdef Py_ssize_t l = 0
    cdef Py_ssize_t r = n - 1
    cdef Py_ssize_t i, lt, gt
    cdef double pivot, tmp
    while l <= r:
        pivot = a[l + (r - l) // 2]
        lt = l
        gt = r
        i = l
        while i <= gt:
            if a[i] < pivot:
                tmp = a[lt]; a[lt] = a[i]; a[i] = tmp
                lt += 1; i += 1
            elif a[i] > pivot:
                tmp = a[i]; a[i] = a[gt]; a[gt] = tmp
                gt -= 1
            else:
                i += 1
        if k < lt:
            r = lt - 1
        elif k > gt:
            l = gt + 1
        else:
            return a[k]
    return a[k]


cdef _describe_impl(FastDataFrame self, object percentiles):
    cdef dict stats = {}
    cdef str name, top_val
    cdef Py_ssize_t col_len, i, jc, valid_cnt, s_mid
    cdef bint is2d
    cdef Py_ssize_t c_probe, r_probe, base

    cdef double s_sum, s_min, s_max, s_mean, s_m2, s_diff, val, s_med
    cdef double* s_tmp = NULL
    cdef double* d_ptr
    cdef int* i_ptr

    cdef int32_t* i32p
    cdef int64_t* i64p
    cdef int32_t v32_min, v32_max
    cdef int64_t v64_min, v64_max

    cdef list s_means, s_stds, s_mins, s_meds, s_maxs, d_mins, d_maxs
    cdef list keys
    cdef dict freq_map
    cdef Py_ssize_t max_freq, freq_val
    
    cdef list pcts
    cdef list pcts_sorted
    cdef dict pcts_dict
    cdef double idx, frac
    cdef Py_ssize_t low, high
    
    # Handle percentiles parameter
    if percentiles is None:
        pcts = [0.25, 0.5, 0.75]
    elif isinstance(percentiles, (list, tuple)):
        pcts = list(percentiles)
    else:
        pcts = [float(percentiles)]
    
    # Validate percentiles
    for p in pcts:
        if p < 0 or p > 1:
            raise ValueError(f"Percentiles must be between 0 and 1, got {p}")
    
    # Sort percentiles
    pcts_sorted = sorted(pcts)

    for name, col_obj in self._columns.items():
        col_len = col_obj.length
        _probe_shape(col_obj, &is2d, &c_probe, &r_probe)

        # =========================================================
        # 1. 数值类型处理 (Int / Double)
        # =========================================================
        if col_obj.dtype == TYPE_INT or col_obj.dtype == TYPE_DOUBLE:
            s_means = []; s_stds = []; s_mins = []; s_meds = []; s_maxs = []

            s_tmp = <double*>aligned_malloc(max(r_probe, 1) * sizeof(double))
            if not s_tmp: raise MemoryError()

            d_ptr = (<DoubleColumn>col_obj).data_c if col_obj.dtype == TYPE_DOUBLE else NULL
            i_ptr = (<IntColumn>col_obj).data_c if col_obj.dtype == TYPE_INT else NULL

            try:
                for jc in range(c_probe):
                    valid_cnt = 0
                    base = jc * r_probe

                    with nogil:
                        for i in range(r_probe):
                            if d_ptr != NULL:
                                val = d_ptr[base + i]
                                if not isnan(val):
                                    s_tmp[valid_cnt] = val
                                    valid_cnt += 1
                            else:
                                s_tmp[valid_cnt] = <double>i_ptr[base + i]
                                valid_cnt += 1

                    if valid_cnt == 0:
                        s_means.append(NAN); s_stds.append(NAN)
                        s_mins.append(NAN); s_maxs.append(NAN); s_meds.append(NAN)
                        continue

                    s_sum = s_tmp[0]
                    s_min = s_tmp[0]
                    s_max = s_tmp[0]

                    with nogil:
                        for i in range(1, valid_cnt):
                            val = s_tmp[i]
                            s_sum += val
                            if val < s_min: s_min = val
                            if val > s_max: s_max = val

                        s_mean = s_sum / <double>valid_cnt

                        if valid_cnt >= 2:
                            s_m2 = 0.0
                            for i in range(valid_cnt):
                                s_diff = s_tmp[i] - s_mean
                                s_m2 += s_diff * s_diff

                    if valid_cnt < 2:
                        s_stds.append(0.0)
                    else:
                        s_stds.append(sqrt(s_m2 / <double>(valid_cnt - 1)))

                    s_means.append(s_mean)
                    s_mins.append(s_min)
                    s_maxs.append(s_max)
                    
                    # Calculate percentiles
                    pcts_dict = {}
                    for p in pcts_sorted:
                        if valid_cnt == 0:
                            pcts_dict[f'{p*100:.0f}%'] = NAN
                        elif valid_cnt == 1:
                            pcts_dict[f'{p*100:.0f}%'] = s_tmp[0]
                        else:
                            idx = p * <double>(valid_cnt - 1)
                            low = <Py_ssize_t>idx
                            high = low + 1
                            if high >= valid_cnt:
                                pcts_dict[f'{p*100:.0f}%'] = s_tmp[valid_cnt - 1]
                            else:
                                frac = idx - <double>low
                                pcts_dict[f'{p*100:.0f}%'] = s_tmp[low] * (1.0 - frac) + s_tmp[high] * frac
                    s_meds.append(pcts_dict)

            finally:
                os_aligned_free(s_tmp)

            if is2d and c_probe > 1:
                stats[name] = {'count': r_probe, 'mean': s_means, 'std': s_stds,
                               'min': s_mins, 'max': s_maxs}
                stats[name].update(s_meds[0])
            else:
                stats[name] = {'count': r_probe, 'mean': s_means[0], 'std': s_stds[0],
                               'min': s_mins[0], 'max': s_maxs[0]}
                stats[name].update(s_meds[0])

        # =========================================================
        # 2. 离散类型处理 (String / Category / Bool)
        # =========================================================
        elif col_obj.dtype in (TYPE_CHAR256, TYPE_UCS2, TYPE_BOOL, TYPE_CATEGORY):
            keys = _col_to_keys(col_obj)
            freq_map = {}
            max_freq = 0

            for valk in keys:
                if valk is not None:
                    freq_val = freq_map.get(valk, 0) + 1
                    freq_map[valk] = freq_val
            top_val = ''
            for valk, freq_val in freq_map.items():
                if freq_val > max_freq:
                    max_freq = freq_val
                    top_val = str(valk)

            stats[name] = {'count': r_probe, 'unique': len(freq_map),
                           'top': top_val, 'freq': max_freq}

        # =========================================================
        # 3. 时间序列类型处理 (Date / DateTime / Time)
        # =========================================================
        elif col_obj.dtype in (TYPE_DATE, TYPE_DATETIME, TYPE_TIME):
            d_mins = []; d_maxs = []

            if col_obj.dtype == TYPE_DATETIME:
                i64p = (<DateTimeColumn>col_obj).data_c
                for jc in range(c_probe):
                    base = jc * r_probe
                    if r_probe == 0:
                        d_mins.append(0); d_maxs.append(0)
                        continue
                    v64_min = i64p[base]
                    v64_max = i64p[base]
                    with nogil:
                        for i in range(1, r_probe):
                            if i64p[base + i] < v64_min: v64_min = i64p[base + i]
                            if i64p[base + i] > v64_max: v64_max = i64p[base + i]
                    d_mins.append(v64_min); d_maxs.append(v64_max)
            else:
                # TYPE_DATE 和 TYPE_TIME 都是 int32_t 底层
                i32p = <int32_t*>((<DateColumn>col_obj).data_c)
                for jc in range(c_probe):
                    base = jc * r_probe
                    if r_probe == 0:
                        d_mins.append(0); d_maxs.append(0)
                        continue
                    v32_min = i32p[base]
                    v32_max = i32p[base]
                    with nogil:
                        for i in range(1, r_probe):
                            if i32p[base + i] < v32_min: v32_min = i32p[base + i]
                            if i32p[base + i] > v32_max: v32_max = i32p[base + i]
                    d_mins.append(v32_min); d_maxs.append(v32_max)

            if is2d and c_probe > 1:
                stats[name] = {'count': r_probe, 'min': d_mins, 'max': d_maxs}
            else:
                stats[name] = {'count': r_probe, 'min': d_mins[0], 'max': d_maxs[0]}

    return stats


cdef _value_counts_impl(FastDataFrame self, str col_name):
    if col_name not in self._columns:
        raise KeyError(f"Column '{col_name}' not found")
    cdef BaseColumn col = self._columns[col_name]

    cdef list keys = _col_to_keys(col)
    cdef Py_ssize_t n = len(keys)
    cdef Py_ssize_t i

    cdef dict counts = {}
    for i in range(n):
        val = keys[i]
        counts[val] = counts.get(val, 0) + 1

    sorted_items = sorted(counts.items(), key=lambda x: -x[1])
    cdef FastDataFrame result = FastDataFrame()
    result.df_length = len(sorted_items)
    result.index = RangeIndex(0, len(sorted_items))
    cdef list key_list = [k for k, v in sorted_items]

    if _col_is_2d(col):
        key_col = _col_from_keys_2d(col, key_list)
    else:
        key_col = _column_from_list_impl(result, col_name, key_list)

    cdef IntColumn cnt_col = IntColumn(len(sorted_items))
    cnt_col.length = len(sorted_items)
    cdef int* cnt_ptr = cnt_col.data_c

    for i in range(len(sorted_items)):
        cnt_ptr[i] = sorted_items[i][1]

    result._columns[col_name] = key_col
    result._columns['count'] = cnt_col

    return result

cdef _nunique_impl(FastDataFrame self, str col_name=None):
    cdef dict result = {}
    cdef str name
    cdef BaseColumn col

    if col_name is not None:
        if col_name not in self._columns:
            raise KeyError(f"Column '{col_name}' not found")
        col = self._columns[col_name]
        return len(set(_col_to_keys(col)))
    for name, col in self._columns.items():
        result[name] = len(set(_col_to_keys(col)))
    return result


# ============================================================
# Series 级方便方法：BaseColumn 全局转发入口（模块级公开 def）
# 由 _base.BaseColumn 以局部 import 方式调用，语义与 df.value_counts /
# df.isnull / df.apply 完全对齐（2D 列走行级键，见 crud._col_to_keys）。
# ============================================================

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
    cdef FastDataFrame result = FastDataFrame()
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
    elif col.dtype == TYPE_CHAR256:
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
    return _column_from_list_impl(FastDataFrame(), '', out)
