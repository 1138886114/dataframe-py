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
    cdef str col_name
    cdef BaseColumn col_obj
    cdef Py_ssize_t c_out, r_out, total, i, j
    cdef list new_vals, result_vals, flat_data, col_names, cols_data
    cdef Py_ssize_t num_cols, n
    cdef dict row
    cdef FastDataFrame result
    cdef bint is2d

    if axis == 'columns':
        result = FastDataFrame()
        result.df_length = self.df_length
        result.index = RangeIndex(0, self.df_length)
        for col_name, col_obj in self.columns.items():
            _probe_shape(col_obj, &is2d, &c_out, &r_out)
            total = col_obj.length
            flat_data = [col_obj[idx] for idx in range(total)]
            new_vals = [func(v) for v in flat_data]

            if is2d and c_out > 1:
                result.columns[col_name] = _column_from_2d_list_impl(new_vals, r_out, c_out)
            else:
                result.columns[col_name] = _column_from_list_impl(result, col_name, new_vals)
        return result

    elif axis == 'index':
        col_names = list(self.columns.keys())
        num_cols = len(col_names)
        cols_data = [_col_to_keys(self.columns[name]) for name in col_names]
        n = self.df_length
        result_vals = [None] * n
        for i in range(n):
            row = {}
            for j in range(num_cols):
                row[col_names[j]] = cols_data[j][i]
            result_vals[i] = func(row)
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
    cdef CategoryColumn cat_col

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
            cat_col = <CategoryColumn>col
            for i in range(total):
                out_ptr[i] = 1 if cat_col._resolve(i) == '' else 0
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
    cdef AsciiColumn ascii_col
    cdef UCS2Column ucs2_col
    cdef str s_val

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

        elif col.dtype == TYPE_CHAR256 and isinstance(value, str):
            ascii_col = <AsciiColumn>col
            s_val = value
            for i in range(total):
                if ascii_col._get_str(i) == '':
                    ascii_col._set_item(i, s_val)
        elif col.dtype == TYPE_UCS2 and isinstance(value, str):
            ucs2_col = <UCS2Column>col
            s_val = value
            for i in range(total):
                if ucs2_col._get_str(i) == '':
                    ucs2_col._set_item(i, s_val)
    return result

cdef _dropna_impl(FastDataFrame self):
    if self.df_length == 0: return _copy_impl(self)
    cdef Py_ssize_t n = self.df_length

    cdef char* keep_mask = <char*>aligned_malloc(n * sizeof(char))
    if not keep_mask: raise MemoryError()
    memset(keep_mask, 1, n)

    cdef Py_ssize_t i, j, k, keep_count = 0
    cdef double* d_data
    cdef str name
    cdef BaseColumn col
    cdef Py_ssize_t* keep_idx = NULL
    cdef Py_ssize_t start, end, str_len
    cdef bint is2d
    cdef Py_ssize_t c_out, r_out
    cdef Py_ssize_t* offsets_ptr

    try:
        for name, col in self.columns.items():
            if col.dtype == TYPE_DOUBLE:
                d_data = (<DoubleColumn>col).data_c
                _probe_shape(col, &is2d, &c_out, &r_out)
                with nogil:
                    if is2d and c_out > 1:
                        for k in range(r_out):
                            if not keep_mask[k]: continue
                            for j in range(c_out):
                                if isnan(d_data[j * r_out + k]):
                                    keep_mask[k] = 0
                                    break
                    else:
                        for i in range(n):
                            if keep_mask[i] and isnan(d_data[i]):
                                keep_mask[i] = 0

            elif col.dtype == TYPE_CHAR256:
                offsets_ptr = (<AsciiColumn>col).offsets_c
                with nogil:
                    for i in range(n):
                        if keep_mask[i]:
                            str_len = offsets_ptr[i + 1] - offsets_ptr[i]
                            if str_len == 0:
                                keep_mask[i] = 0

            elif col.dtype == TYPE_UCS2:
                offsets_ptr = (<UCS2Column>col).offsets_c
                with nogil:
                    for i in range(n):
                        if keep_mask[i]:
                            if offsets_ptr[i] == offsets_ptr[i + 1]:
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
        return FastDataFrame()

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
    cdef Py_ssize_t num_cols = len(self.columns)
    cdef Py_ssize_t total_elements = self.df_length * num_cols
    if total_elements != rows * cols:
        raise ValueError(f"Cannot reshape {total_elements} elements into ({rows}, {cols})")

    cdef FastDataFrame result = FastDataFrame()
    result.df_length = rows
    result.index = RangeIndex(0, rows)
    cdef list col_names = [chr(65 + i) if i < 26 else f"C{i}" for i in range(cols)]

    cdef list all_cols_keys = [_col_to_keys(cell) for cell in self.columns.values()]
    cdef list all_cells = [None] * total_elements
    cdef Py_ssize_t i, j, ptr = 0

    for i in range(self.df_length):
        for j in range(num_cols):
            all_cells[ptr] = all_cols_keys[j][i]
            ptr += 1

    cdef Py_ssize_t c_out = 1
    for cell in all_cells:
        if isinstance(cell, tuple):
            c_out = max(c_out, len(cell))

    cdef DoubleColumn col
    cdef Py_ssize_t r_idx, c_idx, k

    for c_idx in range(cols):
        if c_out > 1:
            col = DoubleColumn(max(rows * c_out, 1), cols=c_out)
            col.length = rows * c_out
            col.num_cols = <int>c_out
            col.is_2d = True
            col.owns_data = True
            for r_idx in range(rows):
                cell = all_cells[r_idx * cols + c_idx]
                if isinstance(cell, tuple):
                    for k in range(c_out):
                        col.data_c[k * rows + r_idx] = <double>cell[k]
                else:
                    col.data_c[r_idx] = <double>cell
                    for k in range(1, c_out): col.data_c[k * rows + r_idx] = NAN
        else:
            col = DoubleColumn(max(rows, 1))
            col.length = rows
            for r_idx in range(rows):
                col.data_c[r_idx] = <double>all_cells[r_idx * cols + c_idx]
        result.columns[col_names[c_idx]] = col

    return result

cdef _T_impl(FastDataFrame self):
    cdef Py_ssize_t rows = self.df_length
    cdef Py_ssize_t cols = len(self.columns)
    if rows == 0 or cols == 0: return FastDataFrame()

    cdef list col_names = list(self.columns.keys())
    cdef FastDataFrame result = FastDataFrame()
    result.df_length = cols
    result.index = RangeIndex(0, cols)

    cdef Py_ssize_t i, j, k, c_out = 1, c_j
    cdef bint is2d_j
    cdef DoubleColumn new_col
    cdef BaseColumn sub

    for j in range(cols):
        _probe_shape(self.columns[col_names[j]], &is2d_j, &c_j, &i)
        if c_j > c_out: c_out = c_j

    cdef double** d_ptrs = <double**>malloc(cols * sizeof(double*))
    cdef int** i_ptrs = <int**>malloc(cols * sizeof(int*))
    cdef Py_ssize_t* c_arr = <Py_ssize_t*>malloc(cols * sizeof(Py_ssize_t))

    for j in range(cols):
        sub = self.columns[col_names[j]]
        _probe_shape(sub, &is2d_j, &c_j, &i)
        c_arr[j] = c_j
        d_ptrs[j] = (<DoubleColumn>sub).data_c if sub.dtype == TYPE_DOUBLE else NULL
        i_ptrs[j] = (<IntColumn>sub).data_c if sub.dtype == TYPE_INT else NULL

    try:
        for i in range(rows):
            if c_out > 1:
                new_col = DoubleColumn(max(cols * c_out, 1), cols=c_out)
                new_col.length = cols * c_out
                new_col.num_cols = <int>c_out
                new_col.is_2d = True
                new_col.owns_data = True

                with nogil:
                    for j in range(cols):
                        if d_ptrs[j] != NULL or i_ptrs[j] != NULL:
                            for k in range(c_out):
                                if k < c_arr[j]:
                                    new_col.data_c[k * cols + j] = d_ptrs[j][k * rows + i] if d_ptrs[j] != NULL else <double>i_ptrs[j][k * rows + i]
                                else:
                                    new_col.data_c[k * cols + j] = NAN
                        else:
                            for k in range(c_out): new_col.data_c[k * cols + j] = NAN
            else:
                new_col = DoubleColumn(max(cols, 1))
                new_col.length = cols
                with nogil:
                    for j in range(cols):
                        if d_ptrs[j] != NULL: new_col.data_c[j] = d_ptrs[j][i]
                        elif i_ptrs[j] != NULL: new_col.data_c[j] = <double>i_ptrs[j][i]
                        else: new_col.data_c[j] = NAN

            result.columns[col_names[i] if i < cols else str(i)] = new_col
    finally:
        free(d_ptrs)
        free(i_ptrs)
        free(c_arr)

    return result

cdef _abs_impl(FastDataFrame self):
    cdef FastDataFrame result = FastDataFrame()
    result.df_length = self.df_length
    result.index = self.index
    cdef str name
    cdef BaseColumn col
    for name, col in self.columns.items():
        if col.dtype == TYPE_INT or col.dtype == TYPE_DOUBLE:
            result.columns[name] = col.abs()
        else:
            result.columns[name] = col.copy()
    return result

cdef _round_impl(FastDataFrame self, int decimals):
    cdef FastDataFrame result = FastDataFrame()
    result.df_length = self.df_length
    result.index = self.index
    cdef str name
    cdef BaseColumn col
    for name, col in self.columns.items():
        if col.dtype == TYPE_DOUBLE:
            result.columns[name] = col.round(decimals)
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
    cdef unsigned char* bool_ptr
    cdef bint bad = False

    for name, col in self.columns.items():
        if bad: break
        total = col.length
        if col.dtype == TYPE_INT:
            i_ptr = (<IntColumn>col).data_c
            with nogil:
                for i in range(total):
                    if i_ptr[i] == 0:
                        bad = True
                        break
        elif col.dtype == TYPE_DOUBLE:
            d_ptr = (<DoubleColumn>col).data_c
            with nogil:
                for i in range(total):
                    if d_ptr[i] == 0.0 or isnan(d_ptr[i]):
                        bad = True
                        break
        elif col.dtype == TYPE_BOOL:
            bool_ptr = (<BoolColumn>col).data_c
            with nogil:
                for i in range(total):
                    if bool_ptr[i] == 0:
                        bad = True
                        break
    return not bad

cdef _any_impl(FastDataFrame self):
    cdef Py_ssize_t n = self.df_length
    if n == 0: return False
    cdef str name
    cdef BaseColumn col
    cdef Py_ssize_t i, total

    cdef int* i_ptr
    cdef double* d_ptr
    cdef unsigned char* bool_ptr
    cdef bint found = False

    for name, col in self.columns.items():
        if found: break
        total = col.length
        if col.dtype == TYPE_INT:
            i_ptr = (<IntColumn>col).data_c
            with nogil:
                for i in range(total):
                    if i_ptr[i] != 0:
                        found = True
                        break
        elif col.dtype == TYPE_DOUBLE:
            d_ptr = (<DoubleColumn>col).data_c
            with nogil:
                for i in range(total):
                    if d_ptr[i] != 0.0 and not isnan(d_ptr[i]):
                        found = True
                        break
        elif col.dtype == TYPE_BOOL:
            bool_ptr = (<BoolColumn>col).data_c
            with nogil:
                for i in range(total):
                    if bool_ptr[i] != 0:
                        found = True
                        break
    return found

cdef _pct_change_impl(FastDataFrame self, int periods=1):
    cdef FastDataFrame result = FastDataFrame()
    result.df_length = self.df_length
    result.index = self.index
    cdef str name
    cdef BaseColumn col
    cdef Py_ssize_t i, k, total, base, p
    cdef bint is2d
    cdef Py_ssize_t c_out, r_out
    cdef DoubleColumn dbl_col
    cdef double prev_val, curr_val
    cdef int* i_ptr
    cdef double* d_ptr
    cdef double* d_out

    for name, col in self.columns.items():
        _probe_shape(col, &is2d, &c_out, &r_out)
        total = col.length
        if col.dtype == TYPE_INT or col.dtype == TYPE_DOUBLE:
            dbl_col = DoubleColumn(max(total, 1), cols=<Py_ssize_t>c_out)
            dbl_col.length = total
            if is2d:
                dbl_col.num_cols = <int>c_out
                dbl_col.is_2d = True
                dbl_col.owns_data = True
            d_out = dbl_col.data_c

            i_ptr = (<IntColumn>col).data_c if col.dtype == TYPE_INT else NULL
            d_ptr = (<DoubleColumn>col).data_c if col.dtype == TYPE_DOUBLE else NULL

            with nogil:
                if periods >= 0:
                    p = periods
                    for k in range(c_out):
                        base = k * r_out
                        for i in range(r_out):
                            if i < p:
                                d_out[base + i] = NAN
                            else:
                                prev_val = <double>i_ptr[base + i - p] if i_ptr != NULL else d_ptr[base + i - p]
                                curr_val = <double>i_ptr[base + i] if i_ptr != NULL else d_ptr[base + i]
                                d_out[base + i] = (curr_val - prev_val) / fabs(prev_val) if prev_val != 0.0 else NAN
                else:
                    p = -periods
                    for k in range(c_out):
                        base = k * r_out
                        for i in range(r_out):
                            if i + p >= r_out:
                                d_out[base + i] = NAN
                            else:
                                prev_val = <double>i_ptr[base + i + p] if i_ptr != NULL else d_ptr[base + i + p]
                                curr_val = <double>i_ptr[base + i] if i_ptr != NULL else d_ptr[base + i]
                                d_out[base + i] = (curr_val - prev_val) / fabs(prev_val) if prev_val != 0.0 else NAN

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
    cdef FastDataFrame result = FastDataFrame()
    result.df_length = self.df_length
    result.index = self.index
    cdef str name
    cdef BaseColumn col
    for name, col in self.columns.items():
        if col.dtype == TYPE_DOUBLE:
            result.columns[name] = col.ffill()
        else:
            result.columns[name] = col.copy()
    return result

cdef _bfill_impl(FastDataFrame self):
    cdef FastDataFrame result = FastDataFrame()
    result.df_length = self.df_length
    result.index = self.index
    cdef str name
    cdef BaseColumn col
    for name, col in self.columns.items():
        if col.dtype == TYPE_DOUBLE:
            result.columns[name] = col.bfill()
        else:
            result.columns[name] = col.copy()
    return result

cdef _pivot_impl(FastDataFrame self, str index, str columns, str values):
    cdef Py_ssize_t n = self.df_length

    cdef list idx_keys = _col_to_keys(self.columns[index])
    cdef list col_keys = _col_to_keys(self.columns[columns])
    cdef list val_keys = _col_to_keys(self.columns[values])

    cdef dict pivot_vals = {}
    cdef dict col_dict = {}
    cdef list unique_cols = []
    cdef list unique_idx = []
    cdef object idx_val, col_val, val_val, got
    cdef Py_ssize_t i, ii, j, u, c_val
    cdef bint is2d_val
    cdef UCS2Column idx_result
    cdef DoubleColumn val_result

    for i in range(n):
        idx_val = idx_keys[i]
        col_val = col_keys[i]
        val_val = val_keys[i]

        if idx_val not in pivot_vals:
            pivot_vals[idx_val] = {}
            unique_idx.append(idx_val)
        pivot_vals[idx_val][col_val] = val_val

        if col_val not in col_dict:
            col_dict[col_val] = len(unique_cols)
            unique_cols.append(col_val)

    cdef FastDataFrame result = FastDataFrame()
    result.df_length = len(unique_idx)
    result.index = RangeIndex(0, len(unique_idx))

    idx_result = UCS2Column(max(len(unique_idx), 1))
    for iv in unique_idx:
        idx_result.append(str(iv))
    result.columns[index] = idx_result

    u = len(unique_idx)
    _probe_shape(self.columns[values], &is2d_val, &c_val, &j)

    if is2d_val and c_val > 1:
        for cv in unique_cols:
            val_result = DoubleColumn(max(u * c_val, 1), cols=c_val)
            val_result.length = u * c_val
            for ii in range(u):
                got = pivot_vals[unique_idx[ii]].get(cv)
                if got is None:
                    for j in range(c_val): val_result.data_c[j * u + ii] = NAN
                else:
                    for j in range(c_val): val_result.data_c[j * u + ii] = <double>got[j]
            result.columns[str(cv)] = val_result
    else:
        for cv in unique_cols:
            val_result = DoubleColumn(max(u, 1))
            val_result.length = u
            for ii in range(u):
                val_result.data_c[ii] = <double>pivot_vals[unique_idx[ii]].get(cv, NAN)
            result.columns[str(cv)] = val_result

    return result

cdef _melt_impl(FastDataFrame self, list id_vars=None, list value_vars=None, str var_name='variable', str value_name='value'):
    cdef Py_ssize_t n = self.df_length
    cdef list all_cols = list(self.columns.keys())
    if id_vars is None: id_vars = []
    if value_vars is None: value_vars = [c for c in all_cols if c not in id_vars]

    cdef FastDataFrame result = FastDataFrame()
    cdef Py_ssize_t total = n * len(value_vars)
    cdef Py_ssize_t vi, i, j, c_out, cc, base
    cdef bint is2d
    cdef UCS2Column new_col
    cdef DoubleColumn new_val
    cdef BaseColumn src_val
    cdef double* d_ptr
    cdef int* i_ptr
    result.df_length = total
    result.index = RangeIndex(0, total)

    for iv in id_vars:
        src_keys = _col_to_keys(self.columns[iv])
        new_col = UCS2Column(max(total, 1))
        for vi in range(len(value_vars)):
            for i in range(n):
                new_col.append(str(src_keys[i]))
        result.columns[iv] = new_col

    new_var = UCS2Column(max(total, 1))
    for vi in range(len(value_vars)):
        for i in range(n):
            new_var.append(str(value_vars[vi]))
    result.columns[var_name] = new_var

    c_out = 1
    for vn in value_vars:
        _probe_shape(self.columns[vn], &is2d, &cc, &i)
        if is2d and cc > c_out: c_out = cc

    new_val = DoubleColumn(max(total * c_out, 1), cols=c_out)
    new_val.length = total * c_out
    if c_out > 1:
        new_val.num_cols = <int>c_out
        new_val.is_2d = True
        new_val.owns_data = True
        for i in range(total * c_out):
            new_val.data_c[i] = NAN

    for vi, vn in enumerate(value_vars):
        src_val = self.columns[vn]
        _probe_shape(src_val, &is2d, &cc, &i)
        if src_val.dtype == TYPE_DOUBLE or src_val.dtype == TYPE_INT:
            d_ptr = (<DoubleColumn>src_val).data_c if src_val.dtype == TYPE_DOUBLE else NULL
            i_ptr = (<IntColumn>src_val).data_c if src_val.dtype == TYPE_INT else NULL

            with nogil:
                if is2d and cc > 1:
                    for j in range(cc):
                        base = j * total + vi * n
                        for i in range(n):
                            new_val.data_c[base + i] = d_ptr[j * n + i] if d_ptr != NULL else <double>i_ptr[j * n + i]
                else:
                    base = vi * n
                    for i in range(n):
                        new_val.data_c[base + i] = d_ptr[i] if d_ptr != NULL else <double>i_ptr[i]
        else:
            for i in range(n):
                new_val.data_c[vi * n + i] = NAN

    result.columns[value_name] = new_val
    return result

cdef _expanding_sum_impl(FastDataFrame self, str col_name):
    cdef BaseColumn col = self.columns[col_name]
    cdef bint is2d
    cdef Py_ssize_t c_out, r_out, total, j, base, i
    _probe_shape(col, &is2d, &c_out, &r_out)
    total = col.length

    cdef DoubleColumn res = DoubleColumn(max(total, 1), cols=c_out)
    res.length = total
    if is2d:
        res.num_cols = <int>c_out
        res.is_2d = True
        res.owns_data = True

    cdef double acc
    cdef double* d_out = res.data_c
    cdef double* d_in
    cdef int* i_in

    if col.dtype == TYPE_DOUBLE:
        d_in = (<DoubleColumn>col).data_c
        with nogil:
            for j in range(c_out):
                base = j * r_out
                acc = 0.0
                for i in range(r_out):
                    if not isnan(d_in[base + i]):
                        acc += d_in[base + i]
                    d_out[base + i] = acc
    elif col.dtype == TYPE_INT:
        i_in = (<IntColumn>col).data_c
        with nogil:
            for j in range(c_out):
                base = j * r_out
                acc = 0.0
                for i in range(r_out):
                    acc += <double>i_in[base + i]
                    d_out[base + i] = acc
    else:
        raise TypeError(f"Expanding sum not supported for {col.dtype}")
    return res

cdef _ewm_mean_impl(FastDataFrame self, str col_name, double span):
    cdef BaseColumn col = self.columns[col_name]
    cdef bint is2d
    cdef Py_ssize_t c_out, r_out, total, j, base, i
    _probe_shape(col, &is2d, &c_out, &r_out)
    total = col.length

    cdef DoubleColumn res = DoubleColumn(max(total, 1), cols=c_out)
    res.length = total
    if is2d:
        res.num_cols = <int>c_out
        res.is_2d = True
        res.owns_data = True

    cdef double alpha = 2.0 / (span + 1.0)
    cdef double acc
    cdef double* d_out = res.data_c
    cdef double* d_in
    cdef int* i_in

    if col.dtype == TYPE_DOUBLE:
        d_in = (<DoubleColumn>col).data_c
        with nogil:
            for j in range(c_out):
                base = j * r_out
                acc = 0.0
                for i in range(r_out):
                    if not isnan(d_in[base + i]):
                        acc = acc * (1.0 - alpha) + d_in[base + i] * alpha
                    d_out[base + i] = acc
    elif col.dtype == TYPE_INT:
        i_in = (<IntColumn>col).data_c
        with nogil:
            for j in range(c_out):
                base = j * r_out
                acc = 0.0
                for i in range(r_out):
                    acc = acc * (1.0 - alpha) + <double>i_in[base + i] * alpha
                    d_out[base + i] = acc
    return res
