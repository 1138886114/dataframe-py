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

cdef _value_counts_impl(FastDataFrame self, str col_name):
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
    key_col = _column_from_list_impl(result, col_name, [k for k, v in sorted_items])
    cnt_col = IntColumn(len(sorted_items))
    cnt_col.length = len(sorted_items)
    for i in range(len(sorted_items)):
        cnt_col.data_c[i] = sorted_items[i][1]
    result.columns[col_name] = key_col
    result.columns['count'] = cnt_col
    return result

cdef _nunique_impl(FastDataFrame self):
    cdef dict result = {}
    cdef str name
    cdef BaseColumn col
    cdef set unique_vals
    cdef Py_ssize_t i
    for name, col in self.columns.items():
        unique_vals = set()
        for i in range(self.df_length):
            unique_vals.add(col[i])
        result[name] = len(unique_vals)
    return result
