# _fastdataframe_agg.pxi — Aggregation (module-level cdef)
# Included at module level in fastdataframe.pyx (after class definition)

cdef _numeric_agg_impl(FastDataFrame self, str method, col_names=None):
    cdef dict result = {}
    cdef str name
    cdef object col
    if col_names is None:
        col_names = list(self.columns.keys())
    for name in col_names:
        if name not in self.columns: continue
        col = self.columns[name]
        if col.dtype == TYPE_INT or col.dtype == TYPE_DOUBLE:
            result[name] = getattr(col, method)()
    return result

cdef _numeric_agg_arg_impl(FastDataFrame self, str method, col_names, arg):
    cdef dict result = {}
    cdef str name
    cdef object col
    if col_names is None:
        col_names = list(self.columns.keys())
    for name in col_names:
        if name not in self.columns: continue
        col = self.columns[name]
        if col.dtype == TYPE_INT or col.dtype == TYPE_DOUBLE:
            result[name] = getattr(col, method)(arg)
    return result

cdef _mode_impl(FastDataFrame self, col_names=None):
    cdef dict result = {}
    cdef str name
    cdef object col
    if col_names is None:
        col_names = list(self.columns.keys())
    for name in col_names:
        if name not in self.columns: continue
        col = self.columns[name]
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
        col_names = list(self.columns.keys())

    for name in col_names:
        if name not in self.columns:
            continue
        col = self.columns[name]
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
        col_names = list(self.columns.keys())

    for name in col_names:
        if name not in self.columns:
            continue
        col = self.columns[name]
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

    for name, col in self.columns.items():
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
    if col_names is None:
        col_names = list(self.columns.keys())
    for name in col_names:
        if name in self.columns:
            result[name] = self.df_length
    return result

cdef _describe_impl(FastDataFrame self):
    cdef dict stats = {}
    cdef str name, top_val
    cdef Py_ssize_t col_len, i, max_freq, freq_val, col_rows
    cdef long long ts_min, ts_max
    cdef dict freq_map
    cdef bint is2d
    cdef Py_ssize_t c_probe, r_probe

    for name, col_obj in self.columns.items():
        col_len = col_obj.length
        is2d, c_probe, r_probe = _col_shape_probe(col_obj)
        col_rows = r_probe

        if col_obj.dtype == TYPE_INT or col_obj.dtype == TYPE_DOUBLE:
            stats[name] = {
                'count': col_rows,
                'mean': col_obj.mean(),
                'std': col_obj.std(),
                'min': col_obj.min(),
                '50%': col_obj.median(),
                'max': col_obj.max(),
            }
        elif col_obj.dtype in (TYPE_CHAR256, TYPE_UCS2, TYPE_BOOL, TYPE_CATEGORY):
            freq_map = {}
            for i in range(col_len):
                val = col_obj[i]
                freq_map[val] = freq_map.get(val, 0) + 1

            top_val = ''
            max_freq = 0
            for val, freq_val in freq_map.items():
                if freq_val > max_freq:
                    max_freq = freq_val
                    top_val = str(val)
            stats[name] = {
                'count': col_rows,
                'unique': len(freq_map),
                'top': top_val,
                'freq': max_freq,
            }
        elif col_obj.dtype in (TYPE_DATE, TYPE_DATETIME, TYPE_TIME):
            if col_len > 0:
                ts_min = col_obj.min()
                ts_max = col_obj.max()
            else:
                ts_min = ts_max = 0
            stats[name] = {
                'count': col_rows,
                'min': ts_min,
                'max': ts_max,
            }
    return stats


cdef list _col_to_keys_tuple(BaseColumn col):
    """返回长度为 rows 的 list，2D 列每项为 hashable 的 tuple 行键，1D 列每项为标量。"""
    return _col_to_keys(col)

cdef _value_counts_impl(FastDataFrame self, str col_name):
    if col_name not in self.columns:
        raise KeyError(f"Column '{col_name}' not found")
    cdef BaseColumn col = self.columns[col_name]

    cdef list keys = _col_to_keys_tuple(col)
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

    result.columns[col_name] = key_col
    result.columns['count'] = cnt_col

    return result

cdef _nunique_impl(FastDataFrame self, str col_name=None):
    cdef dict result = {}
    cdef str name
    cdef BaseColumn col

    if col_name is not None:
        if col_name not in self.columns:
            raise KeyError(f"Column '{col_name}' not found")
        col = self.columns[col_name]
        return len(set(_col_to_keys_tuple(col)))
    for name, col in self.columns.items():
        result[name] = len(set(_col_to_keys_tuple(col)))
    return result
