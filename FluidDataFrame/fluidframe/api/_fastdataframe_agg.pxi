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
    cdef int dtype = col.dtype
    cdef Py_ssize_t c = 1
    cdef Py_ssize_t r = col.length
    cdef bint is2d = False

    if dtype == TYPE_INT:
        is2d = (<IntColumn>col).get_is_2d_c(); c = (<IntColumn>col).get_cols_c(); r = (<IntColumn>col).get_rows_c()
    elif dtype == TYPE_DOUBLE:
        is2d = (<DoubleColumn>col).get_is_2d_c(); c = (<DoubleColumn>col).get_cols_c(); r = (<DoubleColumn>col).get_rows_c()
    elif dtype == TYPE_DATE:
        is2d = (<DateColumn>col).get_is_2d_c(); c = (<DateColumn>col).get_cols_c(); r = (<DateColumn>col).get_rows_c()
    elif dtype == TYPE_DATETIME:
        is2d = (<DateTimeColumn>col).get_is_2d_c(); c = (<DateTimeColumn>col).get_cols_c(); r = (<DateTimeColumn>col).get_rows_c()
    elif dtype == TYPE_TIME:
        is2d = (<TimeColumn>col).get_is_2d_c(); c = (<TimeColumn>col).get_cols_c(); r = (<TimeColumn>col).get_rows_c()
    elif dtype == TYPE_CHAR256 or dtype == TYPE_UCS2 or dtype == TYPE_BOOL or dtype == TYPE_CATEGORY:
        is2d = False; c = 1; r = col.length
    else:
        raise TypeError(f"_col_to_list: unsupported column dtype {dtype}")
    if r == 0:
        return []

    cdef Py_ssize_t i, j
    cdef list res = [None] * r
    cdef list temp_row

    if is2d and c > 1:
        for i in range(r):
            temp_row = [None] * c
            for j in range(c):
                temp_row[j] = col[j * r + i]
            res[i] = tuple(temp_row)
    else:
        for i in range(r):
            res[i] = col[i]

    return res

cdef _value_counts_impl(FastDataFrame self, str col_name):
    if col_name not in self.columns:
        raise KeyError(f"Column '{col_name}' not found")
    cdef BaseColumn col = self.columns[col_name]

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

    cdef int dtype = col.dtype
    cdef bint is2d = False
    if dtype == TYPE_INT: is2d = (<IntColumn>col).get_is_2d_c()
    elif dtype == TYPE_DOUBLE: is2d = (<DoubleColumn>col).get_is_2d_c()
    elif dtype == TYPE_DATE: is2d = (<DateColumn>col).get_is_2d_c()
    elif dtype == TYPE_DATETIME: is2d = (<DateTimeColumn>col).get_is_2d_c()
    elif dtype == TYPE_TIME: is2d = (<TimeColumn>col).get_is_2d_c()

    if is2d:
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
        return len(set(_col_to_keys(col)))
    for name, col in self.columns.items():
        result[name] = len(set(_col_to_keys(col)))
    return result
