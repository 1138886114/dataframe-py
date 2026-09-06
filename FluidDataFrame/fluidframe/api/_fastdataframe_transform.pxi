# _fastdataframe_transform.pxi — Transform, sort, filter, arithmetic (module-level cdef)
# Included at module level in fastdataframe.pyx (after class definition)

cdef _argsort_by_impl(FastDataFrame self, str col_name, bint reverse):
    if col_name not in self.columns:
        raise KeyError(f"Column '{col_name}' not found")

    cdef BaseColumn col = self.columns[col_name]
    cdef Py_ssize_t n = self.df_length
    if n == 0:
        return []

    cdef Py_ssize_t* indices_c = <Py_ssize_t*>aligned_malloc(max(n, 1) * sizeof(Py_ssize_t))
    if not indices_c: raise MemoryError()
    cdef Py_ssize_t i
    cdef Py_ssize_t temp

    try:
        for i in range(n): indices_c[i] = i

        if col.dtype == TYPE_INT:
            with nogil:
                _c_argsort_numeric((<IntColumn>col).data_c, indices_c, 0, n - 1)
        elif col.dtype == TYPE_DOUBLE:
            with nogil:
                _c_argsort_numeric((<DoubleColumn>col).data_c, indices_c, 0, n - 1)
        elif col.dtype == TYPE_CHAR256:
            with nogil:
                _c_argsort_char256((<AsciiColumn>col).data_c, (<AsciiColumn>col).offsets_c, indices_c, 0, n - 1)
        elif col.dtype == TYPE_UCS2:
            with nogil:
                _c_argsort_ucs2((<UCS2Column>col).chars_c, (<UCS2Column>col).offsets_c, indices_c, 0, n - 1)
        elif col.dtype == TYPE_DATE:
            with nogil:
                _c_argsort_numeric(<int*>(<DateColumn>col).data_c, indices_c, 0, n - 1)
        elif col.dtype == TYPE_DATETIME:
            with nogil:
                _c_argsort_numeric(<long long*>(<DateTimeColumn>col).data_c, indices_c, 0, n - 1)
        elif col.dtype == TYPE_TIME:
            with nogil:
                _c_argsort_numeric(<int*>(<TimeColumn>col).data_c, indices_c, 0, n - 1)
        else:
            raise TypeError("Sorting not supported for this column type")

        if reverse:
            for i in range(n // 2):
                temp = indices_c[i]
                indices_c[i] = indices_c[n - 1 - i]
                indices_c[n - 1 - i] = temp

        return [indices_c[j] for j in range(n)]
    finally:
        os_aligned_free(indices_c)

cdef _sort_values_impl(FastDataFrame self, str col_name, bint reverse, bint inplace):
    if col_name not in self.columns:
        raise KeyError(f"Column '{col_name}' not found")

    cdef Py_ssize_t n = self.df_length
    if n == 0:
        return self if inplace else _copy_impl(self)

    cdef list py_indices = _argsort_by_impl(self, col_name, reverse)
    cdef Py_ssize_t* idx_c = <Py_ssize_t*>aligned_malloc(max(n, 1) * sizeof(Py_ssize_t))
    if not idx_c: raise MemoryError()
    cdef Py_ssize_t i
    cdef FastDataFrame result_df
    cdef str name
    cdef BaseColumn col, new_col

    try:
        for i in range(n): idx_c[i] = py_indices[i]
        if inplace:
            result_df = self
        else:
            result_df = FastDataFrame()
            result_df.df_length = n

        for name, col in self.columns.items():
            new_col = self._reorder_column(col, idx_c, n, inplace)
            if not inplace:
                result_df.columns[name] = new_col

        if not inplace and self.index is not None:
            result_df.index = self.index._take(idx_c, n)
        elif inplace and self.index is not None:
            self.index = self.index._take(idx_c, n)

        return result_df
    finally:
        os_aligned_free(idx_c)

cdef _diff_impl(FastDataFrame self, str col_name, int periods):
    if col_name not in self.columns:
        raise KeyError(f"Column '{col_name}' not found")
    cdef BaseColumn col = self.columns[col_name]
    return self._diff_column(col, periods)

cdef _apply_impl(FastDataFrame self, object func, str axis):
    if axis == 'columns':
        result = FastDataFrame()
        result.df_length = self.df_length
        result.index = RangeIndex(0, self.df_length)
        for col_name, col_obj in self.columns.items():
            new_vals = []
            for i in range(self.df_length):
                new_vals.append(func(col_obj[i]))
            result.columns[col_name] = _column_from_list_impl(self, col_name, new_vals)
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

cdef _clip_column_impl(FastDataFrame self, str col_name, double lower, double upper, bint inplace):
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

cdef _where_column_impl(FastDataFrame self, str col_name, str cond_col, object other):
    cdef BaseColumn src = self.columns[col_name]
    cdef BaseColumn cond = self.columns[cond_col]
    if src.dtype == TYPE_DOUBLE and cond.dtype == TYPE_INT:
        return (<DoubleColumn>src).where(<IntColumn>cond, <double>other)
    elif src.dtype == TYPE_INT and cond.dtype == TYPE_INT:
        return (<IntColumn>src).where(<IntColumn>cond, <int>other)
    raise TypeError(f"where not supported for column type {self._type_name(src)}")

cdef _mask_column_impl(FastDataFrame self, str col_name, str cond_col, object other):
    cdef BaseColumn src = self.columns[col_name]
    cdef BaseColumn cond = self.columns[cond_col]
    if src.dtype == TYPE_DOUBLE and cond.dtype == TYPE_INT:
        return (<DoubleColumn>src).mask(<IntColumn>cond, <double>other)
    elif src.dtype == TYPE_INT and cond.dtype == TYPE_INT:
        return (<IntColumn>src).mask(<IntColumn>cond, <int>other)
    raise TypeError(f"mask not supported for column type {self._type_name(src)}")

cdef _where_row_impl(FastDataFrame self, IntColumn mask, object fill):
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

cdef _isnull_impl(FastDataFrame self):
    cdef FastDataFrame result = FastDataFrame()
    result.df_length = self.df_length
    cdef str name
    cdef BaseColumn col
    cdef IntColumn new_col
    cdef Py_ssize_t i, n = self.df_length
    cdef double* d_data
    cdef Py_ssize_t start, end, str_len

    for name, col in self.columns.items():
        new_col = IntColumn(max(n, 1))
        new_col.length = n
        if col.dtype == TYPE_DOUBLE:
            d_data = (<DoubleColumn>col).data_c
            with nogil:
                for i in range(n):
                    new_col.data_c[i] = isnan(d_data[i])
        elif col.dtype == TYPE_CHAR256:
            for i in range(n):
                str_len = (<AsciiColumn>col).offsets_c[i + 1] - (<AsciiColumn>col).offsets_c[i]
                new_col.data_c[i] = 1 if str_len == 0 else 0
        elif col.dtype == TYPE_UCS2:
            for i in range(n):
                start = (<UCS2Column>col).offsets_c[i]
                end = (<UCS2Column>col).offsets_c[i + 1]
                new_col.data_c[i] = 1 if end == start else 0
        else:
            memset(new_col.data_c, 0, n * sizeof(int))
        result.columns[name] = new_col
    return result

cdef _fillna_impl(FastDataFrame self, double value, col_names):
    cdef FastDataFrame result = _copy_impl(self)
    cdef str name
    cdef BaseColumn col
    cdef Py_ssize_t i, n = self.df_length
    cdef double* d_data
    if col_names is None:
        col_names = list(self.columns.keys())
    for name in col_names:
        if name not in result.columns: continue
        col = result.columns[name]
        if col.dtype == TYPE_DOUBLE:
            d_data = (<DoubleColumn>col).data_c
            with nogil:
                for i in range(n):
                    if isnan(d_data[i]):
                        d_data[i] = value
    return result

cdef _dropna_impl(FastDataFrame self):
    if self.df_length == 0: return _copy_impl(self)
    cdef Py_ssize_t n = self.df_length
    cdef char* keep_mask = <char*>aligned_malloc(n * sizeof(char))
    if not keep_mask: raise MemoryError()
    memset(keep_mask, 1, n)

    cdef Py_ssize_t i, j, keep_count = 0
    cdef double* d_data
    cdef str name
    cdef BaseColumn col
    cdef Py_ssize_t* keep_idx = NULL
    cdef Py_ssize_t start, end, str_len = 0

    try:
        for name, col in self.columns.items():
            if col.dtype == TYPE_DOUBLE:
                d_data = (<DoubleColumn>col).data_c
                with nogil:
                    for i in range(n):
                        if keep_mask[i] and isnan(d_data[i]):
                            keep_mask[i] = 0
            elif col.dtype == TYPE_CHAR256:
                for i in range(n):
                    if keep_mask[i]:
                        str_len = (<AsciiColumn>col).offsets_c[i + 1] - (<AsciiColumn>col).offsets_c[i]
                        if str_len == 0:
                            keep_mask[i] = 0
            elif col.dtype == TYPE_UCS2:
                for i in range(n):
                    if keep_mask[i]:
                        start = (<UCS2Column>col).offsets_c[i]
                        end = (<UCS2Column>col).offsets_c[i + 1]
                        if start == end:
                            keep_mask[i] = 0

        for i in range(n):
            keep_count += keep_mask[i]

        if keep_count == n:
            return _copy_impl(self)
        if keep_count == 0:
            return FastDataFrame()

        keep_idx = <Py_ssize_t*>aligned_malloc(max(keep_count, 1) * sizeof(Py_ssize_t))
        if not keep_idx: raise MemoryError()

        j = 0
        for i in range(n):
            if keep_mask[i]:
                keep_idx[j] = i
                j += 1

        return self._take(keep_idx, keep_count)
    finally:
        if keep_mask: os_aligned_free(keep_mask)
        if keep_idx: os_aligned_free(keep_idx)

cdef _head_impl(FastDataFrame self, Py_ssize_t n):
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
        for i in range(n): idx_c[i] = i
        return self._take(idx_c, n)
    finally:
        os_aligned_free(idx_c)

cdef _tail_impl(FastDataFrame self, Py_ssize_t n):
    if n < 0: raise ValueError("n must be non-negative")
    n = min(n, self.df_length)
    if n == 0: return FastDataFrame()
    cdef Py_ssize_t start = self.df_length - n
    cdef Py_ssize_t* idx_c = <Py_ssize_t*>aligned_malloc(max(n, 1) * sizeof(Py_ssize_t))
    if not idx_c: raise MemoryError()
    cdef Py_ssize_t i
    try:
        for i in range(n): idx_c[i] = start + i
        return self._take(idx_c, n)
    finally:
        os_aligned_free(idx_c)

cdef _sample_impl(FastDataFrame self, Py_ssize_t n, double frac, bint replace, int seed):
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

cdef _nsmallest_impl(FastDataFrame self, Py_ssize_t n, str col_name):
    cdef FastDataFrame sorted_df = _sort_values_impl(self, col_name, False, False)
    return _head_impl(sorted_df, n)

cdef _nlargest_impl(FastDataFrame self, Py_ssize_t n, str col_name):
    cdef FastDataFrame sorted_df = _sort_values_impl(self, col_name, True, False)
    return _head_impl(sorted_df, n)

cdef _pipe_impl(FastDataFrame self, object func, tuple args, dict kwargs):
    return func(self, *args, **kwargs)

cdef _assign_impl(FastDataFrame self, dict kwargs):
    cdef FastDataFrame result = _copy_impl(self)
    for name, value in kwargs.items():
        result[name] = value
    return result

cdef _select_dtypes_impl(FastDataFrame self, list include_types, list exclude_types):
    cdef FastDataFrame result = FastDataFrame()
    cdef dict dtype_map = {TYPE_INT: 'int', TYPE_DOUBLE: 'float', TYPE_CHAR256: 'char256',
                           TYPE_UCS2: 'ucs2', TYPE_DATE: 'date',
                           TYPE_BOOL: 'bool', TYPE_CATEGORY: 'category',
                           TYPE_DATETIME: 'datetime', TYPE_TIME: 'time'}
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

cdef _astype_impl(FastDataFrame self, dict dtype_dict):
    cdef FastDataFrame result = FastDataFrame()
    cdef str col_name, target
    for col_name, col_obj in self.columns.items():
        if col_name in dtype_dict:
            target = dtype_dict[col_name]
            result.columns[col_name] = col_obj.astype(target)
        else:
            result.columns[col_name] = col_obj.copy() if hasattr(col_obj, 'copy') else col_obj
    result.df_length = self.df_length
    if self.index is not None:
        if self.index.index_type == INDEX_RANGE:
            result.index = RangeIndex((<RangeIndex>self.index).start,
                                     self.index.length,
                                     (<RangeIndex>self.index).step)
        elif self.index.index_type == INDEX_INT:
            result.index = self.index.copy()
        else:
            result.index = self.index.copy()
    return result

cdef _resolve_cols_impl(FastDataFrame self, tuple col_names):
    cdef list cols = []
    cdef str name
    for name in col_names:
        if name not in self.columns:
            raise KeyError(f"Column '{name}' not found")
        cols.append(self.columns[name])
    return cols

cdef _add_impl(FastDataFrame self, tuple col_names):
    if len(col_names) < 2: raise ValueError("Need at least 2 column names")
    cdef list cols = _resolve_cols_impl(self, col_names)
    cdef BaseColumn result = cols[0].add(cols[1])
    cdef Py_ssize_t j
    for j in range(2, len(cols)):
        result = result.add(cols[j])
    return result

cdef _sub_impl(FastDataFrame self, tuple col_names):
    if len(col_names) < 2: raise ValueError("Need at least 2 column names")
    cdef list cols = _resolve_cols_impl(self, col_names)
    cdef BaseColumn result = cols[0].sub(cols[1])
    cdef Py_ssize_t j
    for j in range(2, len(cols)):
        result = result.sub(cols[j])
    return result

cdef _mul_impl(FastDataFrame self, tuple col_names):
    if len(col_names) < 2: raise ValueError("Need at least 2 column names")
    cdef list cols = _resolve_cols_impl(self, col_names)
    cdef BaseColumn result = cols[0].mul(cols[1])
    cdef Py_ssize_t j
    for j in range(2, len(cols)):
        result = result.mul(cols[j])
    return result

cdef _div_impl(FastDataFrame self, tuple col_names):
    if len(col_names) < 2: raise ValueError("Need at least 2 column names")
    cdef list cols = _resolve_cols_impl(self, col_names)
    cdef BaseColumn result = cols[0].div(cols[1])
    cdef Py_ssize_t j
    for j in range(2, len(cols)):
        result = result.div(cols[j])
    return result

cdef _reshape_impl(FastDataFrame self, Py_ssize_t rows, Py_ssize_t cols):
    cdef Py_ssize_t total_elements = self.df_length * len(self.columns)
    if total_elements != rows * cols:
        raise ValueError(f"Cannot reshape {total_elements} elements into ({rows}, {cols})")

    cdef FastDataFrame result = FastDataFrame()
    result.df_length = rows
    result.index = RangeIndex(0, rows)

    cdef list col_names = [chr(65 + i) if i < 26 else f"C{i}" for i in range(cols)]
    cdef list all_vals = []
    cdef str name
    cdef BaseColumn col_obj
    cdef Py_ssize_t i, r_idx, c_idx

    for i in range(self.df_length):
        for name in self.columns:
            col_obj = self.columns[name]
            all_vals.append(col_obj[i])

    cdef DoubleColumn col
    for c_idx in range(cols):
        col = DoubleColumn(max(rows, 1))
        col.length = rows
        for r_idx in range(rows):
            col.data_c[r_idx] = <double>all_vals[r_idx * cols + c_idx]
        result.columns[col_names[c_idx]] = col

    return result

cdef _T_impl(FastDataFrame self):
    cdef Py_ssize_t rows = self.df_length
    cdef Py_ssize_t cols = len(self.columns)
    if rows == 0 or cols == 0:
        return FastDataFrame()

    cdef list col_names = list(self.columns.keys())
    cdef FastDataFrame result = FastDataFrame()
    result.df_length = cols
    result.index = RangeIndex(0, cols)

    cdef Py_ssize_t i, j
    cdef DoubleColumn new_col
    for i in range(rows):
        new_col = DoubleColumn(max(cols, 1))
        new_col.length = cols
        for j in range(cols):
            new_col.data_c[j] = self._get_col_value_as_double(self.columns[col_names[j]], i)
        result.columns[col_names[i] if i < cols else str(i)] = new_col
    return result

cdef _abs_impl(FastDataFrame self):
    cdef FastDataFrame result = FastDataFrame()
    result.df_length = self.df_length
    result.index = self.index
    cdef str name
    cdef BaseColumn col
    cdef Py_ssize_t i, n = self.df_length
    cdef IntColumn int_col
    cdef DoubleColumn dbl_col
    for name, col in self.columns.items():
        if col.dtype == TYPE_INT:
            int_col = IntColumn(max(n, 1))
            int_col.length = n
            for i in range(n):
                int_col.data_c[i] = abs((<IntColumn>col).data_c[i])
            result.columns[name] = int_col
        elif col.dtype == TYPE_DOUBLE:
            dbl_col = DoubleColumn(max(n, 1))
            dbl_col.length = n
            for i in range(n):
                dbl_col.data_c[i] = fabs((<DoubleColumn>col).data_c[i])
            result.columns[name] = dbl_col
        else:
            result.columns[name] = col.copy()
    return result

cdef _round_impl(FastDataFrame self, int decimals):
    cdef FastDataFrame result = FastDataFrame()
    result.df_length = self.df_length
    result.index = self.index
    cdef str name
    cdef BaseColumn col
    cdef Py_ssize_t i, n = self.df_length
    cdef IntColumn int_col
    cdef DoubleColumn dbl_col
    cdef double factor = 1.0
    for i in range(decimals):
        factor *= 10.0
    for name, col in self.columns.items():
        if col.dtype == TYPE_INT:
            int_col = IntColumn(max(n, 1))
            int_col.length = n
            if decimals == 0:
                for i in range(n):
                    int_col.data_c[i] = (<IntColumn>col).data_c[i]
            else:
                for i in range(n):
                    int_col.data_c[i] = <int>((<IntColumn>col).data_c[i])
            result.columns[name] = int_col
        elif col.dtype == TYPE_DOUBLE:
            dbl_col = DoubleColumn(max(n, 1))
            dbl_col.length = n
            if decimals == 0:
                for i in range(n):
                    dbl_col.data_c[i] = <double>round((<DoubleColumn>col).data_c[i])
            else:
                for i in range(n):
                    dbl_col.data_c[i] = round((<DoubleColumn>col).data_c[i] * factor) / factor
            result.columns[name] = dbl_col
        else:
            result.columns[name] = col.copy()
    return result

cdef _all_impl(FastDataFrame self):
    cdef Py_ssize_t n = self.df_length
    if n == 0:
        return True
    cdef str name
    cdef BaseColumn col
    cdef Py_ssize_t i
    for name, col in self.columns.items():
        if col.dtype == TYPE_INT:
            for i in range(n):
                if (<IntColumn>col).data_c[i] == 0:
                    return False
        elif col.dtype == TYPE_DOUBLE:
            for i in range(n):
                if (<DoubleColumn>col).data_c[i] == 0.0:
                    return False
        elif col.dtype == TYPE_BOOL:
            for i in range(n):
                if (<BoolColumn>col).data_c[i] == 0:
                    return False
    return True

cdef _any_impl(FastDataFrame self):
    cdef Py_ssize_t n = self.df_length
    if n == 0:
        return False
    cdef str name
    cdef BaseColumn col
    cdef Py_ssize_t i
    for name, col in self.columns.items():
        if col.dtype == TYPE_INT:
            for i in range(n):
                if (<IntColumn>col).data_c[i] != 0:
                    return True
        elif col.dtype == TYPE_DOUBLE:
            for i in range(n):
                if (<DoubleColumn>col).data_c[i] != 0.0:
                    return True
        elif col.dtype == TYPE_BOOL:
            for i in range(n):
                if (<BoolColumn>col).data_c[i] != 0:
                    return True
    return False

cdef _pct_change_impl(FastDataFrame self, int periods=1):
    cdef FastDataFrame result = FastDataFrame()
    result.df_length = self.df_length
    result.index = self.index
    cdef str name
    cdef BaseColumn col
    cdef Py_ssize_t i, n = self.df_length
    cdef IntColumn int_col
    cdef DoubleColumn dbl_col
    cdef double prev_val, curr_val
    for name, col in self.columns.items():
        if col.dtype == TYPE_INT:
            dbl_col = DoubleColumn(max(n, 1))
            dbl_col.length = n
            for i in range(n):
                if i < periods:
                    dbl_col.data_c[i] = NAN
                else:
                    prev_val = <double>(<IntColumn>col).data_c[i - periods]
                    curr_val = <double>(<IntColumn>col).data_c[i]
                    if prev_val == 0.0:
                        dbl_col.data_c[i] = NAN
                    else:
                        dbl_col.data_c[i] = (curr_val - prev_val) / fabs(prev_val)
            result.columns[name] = dbl_col
        elif col.dtype == TYPE_DOUBLE:
            dbl_col = DoubleColumn(max(n, 1))
            dbl_col.length = n
            for i in range(n):
                if i < periods:
                    dbl_col.data_c[i] = NAN
                else:
                    prev_val = (<DoubleColumn>col).data_c[i - periods]
                    curr_val = (<DoubleColumn>col).data_c[i]
                    if prev_val == 0.0:
                        dbl_col.data_c[i] = NAN
                    else:
                        dbl_col.data_c[i] = (curr_val - prev_val) / fabs(prev_val)
            result.columns[name] = dbl_col
        else:
            result.columns[name] = col.copy()
    return result
