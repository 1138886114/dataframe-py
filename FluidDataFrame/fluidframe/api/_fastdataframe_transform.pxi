# _fastdataframe_transform.pxi — Transform, sort, filter, arithmetic (module-level cdef)
# Included at module level in fastdataframe.pyx (after class definition)

cdef _argsort_by_impl(FastDataFrame self, str col_name, bint reverse):
    if col_name not in self.columns:
        raise KeyError(f"Column '{col_name}' not found")

    cdef BaseColumn col = self.columns[col_name]
    cdef Py_ssize_t n = self.df_length
    if n == 0:
        return []

    # 2D 列：整行向量作排序键（_col_row_key 语义），Python 级字典序
    if _col_is_2d(col):
        return sorted(range(n), key=lambda i: _col_row_key(col, i), reverse=reverse)

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
    result.index = self.index
    cdef str name
    cdef BaseColumn col
    cdef IntColumn new_col
    cdef Py_ssize_t i, total
    cdef bint is2d
    cdef Py_ssize_t c_out

    cdef double* d_ptr
    cdef int* out_ptr
    cdef Py_ssize_t* offset_ptr

    for name, col in self.columns.items():
        is2d, c_out, _ = _col_shape_probe(col)
        total = col.length

        new_col = IntColumn(max(total, 1), cols=<Py_ssize_t>c_out)
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
            # TODO 这个的真实逻辑需要调整
            c_ptr = (<CategoryColumn>col).codes_c
            with nogil:
                for i in range(total):
                    out_ptr[i] = 1 if c_ptr[i] == 0 else 0
        else:
            memset(out_ptr, 0, total * sizeof(int))
        result.columns[name] = new_col
    return result

cdef _fillna_impl(FastDataFrame self, object value, col_names):
    """【语义修复】：参数 value 必须是 object，以支持字符串等各种类型填充"""
    cdef FastDataFrame result = _copy_impl(self)
    cdef str name
    cdef BaseColumn col
    cdef Py_ssize_t i, total
    cdef double* d_ptr
    cdef double d_val

    if col_names is None:
        col_names = list(self.columns.keys())
    for name in col_names:
        if name not in result.columns: continue
        col = result.columns[name]
        total = col.length

        if col.dtype == TYPE_DOUBLE and isinstance(value, (int, float)):
            d_ptr = (<DoubleColumn>col).data_c
            d_val = <double>value
            with nogil:
                for i in range(total):
                    if isnan(d_ptr[i]):
                        d_ptr[i] = d_val

        # 字符串/离散类型的填充 (可以按需扩展)
        elif col.dtype in (TYPE_CHAR256, TYPE_UCS2) and isinstance(value, str):
            # TODO: 审查 AsciiColumn UCS2Column内部是否开放 fill_empty() 接口，若是没有批量填充：""
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
    cdef str name, col
    cdef Py_ssize_t i, total, c_out
    cdef bint is2d
    cdef IntColumn int_col
    cdef DoubleColumn dbl_col

    cdef int* i_in
    cdef int* i_out
    cdef double* d_in
    cdef double* d_out

    for name, col in self.columns.items():
        is2d, c_out, _ = _col_shape_probe(col)
        total = col.length

        if col.dtype == TYPE_INT:
            int_col = IntColumn(max(total, 1), cols=<Py_ssize_t>c_out)
            int_col.length = total
            i_in = (<IntColumn>col).data_c
            i_out = int_col.data_c
            with nogil:
                for i in range(total):
                    i_out[i] = c_abs(i_in[i])
            result.columns[name] = int_col

        elif col.dtype == TYPE_DOUBLE:
            dbl_col = DoubleColumn(max(total, 1), cols=<Py_ssize_t>c_out)
            dbl_col.length = total
            d_in = (<DoubleColumn>col).data_c
            d_out = dbl_col.data_c
            with nogil:
                for i in range(total):
                    d_out[i] = fabs(d_in[i])
            result.columns[name] = dbl_col
        else:
            result.columns[name] = col.copy()
    return result

cdef _round_impl(FastDataFrame self, int decimals):
    cdef FastDataFrame result = FastDataFrame()
    result.df_length = self.df_length
    result.index = self.index
    cdef str name, col
    cdef Py_ssize_t i, total, c_out
    cdef bint is2d
    cdef DoubleColumn dbl_col

    cdef double factor = pow(10.0, <double>decimals)
    cdef double* d_in
    cdef double* d_out

    for name, col in self.columns.items():
        is2d, c_out, _ = _col_shape_probe(col)
        total = col.length

        if col.dtype == TYPE_INT:
            result.columns[name] = col.copy()
        elif col.dtype == TYPE_DOUBLE:
            dbl_col = DoubleColumn(max(total, 1), cols=<Py_ssize_t>c_out)
            dbl_col.length = total
            d_in = (<DoubleColumn>col).data_c
            d_out = dbl_col.data_c

            with nogil:
                if decimals == 0:
                    for i in range(total):
                        d_out[i] = round(d_in[i])
                else:
                    for i in range(total):
                        d_out[i] = round(d_in[i] * factor) / factor
            result.columns[name] = dbl_col
        else:
            result.columns[name] = col.copy()
    return result

cdef _all_impl(FastDataFrame self):
    cdef Py_ssize_t n = self.df_length
    if n == 0: return True
    cdef str name
    cdef BaseColumn col
    cdef Py_ssize_t i, total

    cdef int* i_ptr
    cdef double* d_ptr
    cdef int32_t* bool_ptr

    for name, col in self.columns.items():
        total = col.length
        if col.dtype == TYPE_INT:
            i_ptr = (<IntColumn>col).data_c
            with nogil:
                for i in range(total):
                    if i_ptr[i] == 0: return False
        elif col.dtype == TYPE_DOUBLE:
            d_ptr = (<DoubleColumn>col).data_c
            with nogil:
                for i in range(total):
                    if d_ptr[i] == 0.0 or isnan(d_ptr[i]): return False
        elif col.dtype == TYPE_BOOL:
            bool_ptr = <int32_t*>((<BoolColumn>col).data_c)
            with nogil:
                for i in range(total):
                    if bool_ptr[i] == 0: return False
    return True

cdef _any_impl(FastDataFrame self):
    cdef Py_ssize_t n = self.df_length
    if n == 0: return False
    cdef str name
    cdef BaseColumn col
    cdef Py_ssize_t i, total

    cdef int* i_ptr
    cdef double* d_ptr
    cdef int32_t* bool_ptr

    for name, col in self.columns.items():
        total = col.length
        if col.dtype == TYPE_INT:
            i_ptr = (<IntColumn>col).data_c
            with nogil:
                for i in range(total):
                    if i_ptr[i] != 0: return True
        elif col.dtype == TYPE_DOUBLE:
            d_ptr = (<DoubleColumn>col).data_c
            with nogil:
                for i in range(total):
                    if d_ptr[i] != 0.0 and not isnan(d_ptr[i]): return True
        elif col.dtype == TYPE_BOOL:
            bool_ptr = <int32_t*>((<BoolColumn>col).data_c)
            with nogil:
                for i in range(total):
                    if bool_ptr[i] != 0: return True
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

cdef _replace_impl(FastDataFrame self, object to_replace, object value):
    cdef FastDataFrame result = _copy_impl(self)
    cdef str name
    cdef BaseColumn col
    cdef Py_ssize_t i, total

    cdef int i_target, i_val
    cdef double d_target, d_val
    cdef int* i_ptr
    cdef double* d_ptr
    cdef UCS2Column new_col

    for name, col in result.columns.items():
        total = col.length
        if col.dtype == TYPE_INT and isinstance(to_replace, (int, float)) and isinstance(value, (int, float)):
            i_target = <int>to_replace
            i_val = <int>value
            i_ptr = (<IntColumn>col).data_c
            with nogil:
                for i in range(total):
                    if i_ptr[i] == i_target:
                        i_ptr[i] = i_val

        elif col.dtype == TYPE_DOUBLE and isinstance(to_replace, (int, float)) and isinstance(value, (int, float)):
            d_target = <double>to_replace
            d_val = <double>value
            d_ptr = (<DoubleColumn>col).data_c
            with nogil:
                for i in range(total):
                    if d_target != d_target:
                        if isnan(d_ptr[i]): d_ptr[i] = d_val
                    elif d_ptr[i] == d_target:
                        d_ptr[i] = d_val

        elif col.dtype in (TYPE_CHAR256, TYPE_UCS2) and isinstance(to_replace, str):
            new_col = UCS2Column(max(total, 1))
            for i in range(total):
                if col[i] == to_replace:
                    new_col.append(str(value))
                else:
                    new_col.append(str(col[i]))
            result.columns[name] = new_col
    return result

cdef _ffill_impl(FastDataFrame self):
    cdef FastDataFrame result = _copy_impl(self)
    cdef str name
    cdef BaseColumn col
    cdef Py_ssize_t i, n = self.df_length
    cdef DoubleColumn dbl_col
    cdef double last_val
    for name, col in result.columns.items():
        if col.dtype == TYPE_DOUBLE:
            dbl_col = <DoubleColumn>col
            last_val = NAN
            for i in range(n):
                if not isnan(dbl_col.data_c[i]):
                    last_val = dbl_col.data_c[i]
                elif not isnan(last_val):
                    dbl_col.data_c[i] = last_val
    return result

cdef _bfill_impl(FastDataFrame self):
    cdef FastDataFrame result = _copy_impl(self)
    cdef str name
    cdef BaseColumn col
    cdef Py_ssize_t i, n = self.df_length
    cdef DoubleColumn dbl_col
    cdef double next_val
    for name, col in result.columns.items():
        if col.dtype == TYPE_DOUBLE:
            dbl_col = <DoubleColumn>col
            next_val = NAN
            for i in range(n - 1, -1, -1):
                if not isnan(dbl_col.data_c[i]):
                    next_val = dbl_col.data_c[i]
                elif not isnan(next_val):
                    dbl_col.data_c[i] = next_val
    return result

cdef _pivot_impl(FastDataFrame self, str index, str columns, str values):
    cdef Py_ssize_t n = self.df_length
    cdef BaseColumn idx_col = self.columns[index]
    cdef BaseColumn col_col = self.columns[columns]
    cdef BaseColumn val_col = self.columns[values]

    cdef dict pivot_vals = {}
    cdef dict col_keys = {}
    cdef list unique_cols = []
    cdef list unique_idx = []

    for i in range(n):
        idx_val = idx_col[i]
        col_val = col_col[i]
        val_val = val_col[i]
        if idx_val not in pivot_vals:
            pivot_vals[idx_val] = {}
            unique_idx.append(idx_val)
        pivot_vals[idx_val][col_val] = val_val
        if col_val not in col_keys:
            col_keys[col_val] = len(unique_cols)
            unique_cols.append(col_val)

    cdef FastDataFrame result = FastDataFrame()
    result.df_length = len(unique_idx)
    result.index = RangeIndex(0, len(unique_idx))

    cdef UCS2Column idx_result = UCS2Column(max(len(unique_idx), 1))
    cdef DoubleColumn val_result
    for iv in unique_idx:
        idx_result.append(str(iv))
    result.columns[index] = idx_result

    for cv in unique_cols:
        val_result = DoubleColumn(max(len(unique_idx), 1))
        val_result.length = len(unique_idx)
        for ii, iv in enumerate(unique_idx):
            val_result.data_c[ii] = pivot_vals[iv].get(cv, NAN)
        result.columns[str(cv)] = val_result

    return result

cdef _melt_impl(FastDataFrame self, list id_vars=None, list value_vars=None, str var_name='variable', str value_name='value'):
    cdef Py_ssize_t n = self.df_length
    cdef list all_cols = list(self.columns.keys())
    if id_vars is None:
        id_vars = []
    if value_vars is None:
        value_vars = [c for c in all_cols if c not in id_vars]

    cdef FastDataFrame result = FastDataFrame()
    cdef Py_ssize_t total = n * len(value_vars)
    cdef Py_ssize_t vi, i
    result.df_length = total
    result.index = RangeIndex(0, total)

    cdef UCS2Column new_col
    for iv in id_vars:
        src_col = self.columns[iv]
        if src_col.dtype == TYPE_UCS2 or src_col.dtype == TYPE_CHAR256:
            new_col = UCS2Column(max(total, 1))
            for vi in range(len(value_vars)):
                for i in range(n):
                    new_col.append(str(src_col[i]))
            result.columns[iv] = new_col

    new_var = UCS2Column(max(total, 1))
    for vi in range(len(value_vars)):
        for i in range(n):
            new_var.append(str(value_vars[vi]))
    result.columns[var_name] = new_var

    new_val = DoubleColumn(max(total, 1))
    new_val.length = total
    for vi, vn in enumerate(value_vars):
        src_val = self.columns[vn]
        for i in range(n):
            new_val.data_c[vi * n + i] = <double>src_val[i] if src_val.dtype in (TYPE_INT, TYPE_DOUBLE) else NAN
    result.columns[value_name] = new_val

    return result

cdef _expanding_sum_impl(FastDataFrame self, str col_name):
    cdef BaseColumn col = self.columns[col_name]
    cdef Py_ssize_t n = self.df_length
    cdef DoubleColumn res = DoubleColumn(max(n, 1))
    res.length = n
    cdef double acc = 0.0
    cdef Py_ssize_t i
    for i in range(n):
        acc += <double>col[i]
        res.data_c[i] = acc
    return res

cdef _ewm_mean_impl(FastDataFrame self, str col_name, double span):
    cdef BaseColumn col = self.columns[col_name]
    cdef Py_ssize_t n = self.df_length
    cdef DoubleColumn res = DoubleColumn(max(n, 1))
    res.length = n
    cdef double alpha = 2.0 / (span + 1.0)
    cdef double acc = 0.0
    cdef Py_ssize_t i
    for i in range(n):
        acc = acc * (1.0 - alpha) + <double>col[i] * alpha
        res.data_c[i] = acc
    return res
