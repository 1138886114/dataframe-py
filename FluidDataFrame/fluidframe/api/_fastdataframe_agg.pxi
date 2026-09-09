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

cdef _count_impl(FastDataFrame self, col_names=None):
    cdef dict result = {}
    cdef str name
    cdef object col
    if col_names is None:
        col_names = list(self.columns.keys())
    for name in col_names:
        if name not in self.columns: continue
        col = self.columns[name]
        result[name] = col.length
    return result

cdef _describe_impl(FastDataFrame self):
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
