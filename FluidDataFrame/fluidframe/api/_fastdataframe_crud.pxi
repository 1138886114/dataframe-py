# _fastdataframe_crud.pxi — CRUD and core data operations (module-level cdef)
# Included at module level in fastdataframe.pyx (after class definition)
# NOTE: _column_from_list_impl is defined here and used by other .pxi files

cdef BaseColumn _column_from_list_impl(FastDataFrame self, str name, list values):
    cdef Py_ssize_t n = len(values)
    cdef Py_ssize_t i
    cdef IntColumn int_col
    cdef DoubleColumn dbl_col
    cdef AsciiColumn c256_col
    cdef UCS2Column ucs2_col
    cdef BoolColumn bool_col
    if n == 0:
        return IntColumn(0)

    cdef object sample = None
    for i in range(n):
        if values[i] is not None:
            sample = values[i]
            break

    if sample is None:
        dbl_col = DoubleColumn(max(n, 1))
        dbl_col.length = n
        for i in range(n): dbl_col.data_c[i] = NAN
        return dbl_col

    if isinstance(sample, bool):
        bool_col = BoolColumn(max(n, 1))
        bool_col.length = n
        for i in range(n): bool_col.data_c[i] = 1 if values[i] else 0
        return bool_col
    elif isinstance(sample, int):
        int_col = IntColumn(max(n, 1))
        int_col.length = n
        for i in range(n): int_col.data_c[i] = <int>(values[i] if values[i] is not None else 0)
        return int_col
    elif isinstance(sample, float):
        dbl_col = DoubleColumn(max(n, 1))
        dbl_col.length = n
        for i in range(n):
            dbl_col.data_c[i] = <double>values[i] if values[i] is not None else NAN
        return dbl_col
    elif isinstance(sample, str):
        try:
            (<str>sample).encode('ascii')
            c256_col = AsciiColumn(max(n, 1))
            for i in range(n):
                c256_col.append(<str>(values[i] if values[i] is not None else ''))
            return c256_col
        except UnicodeEncodeError:
            ucs2_col = UCS2Column(max(n, 1))
            for i in range(n):
                ucs2_col.append(<str>(values[i] if values[i] is not None else ''))
            return ucs2_col
    raise TypeError(f"Unsupported type: {type(sample)}")

cdef _eval_impl(FastDataFrame self, str expr, bint inplace):
    cdef FastDataFrame target_df = self if inplace else self.copy()
    cdef bint is_assignment = False
    cdef str target_col = ""
    cdef str calc_expr = expr

    if "=" in expr:
        parts = expr.split("=", 1)
        if not parts[0].strip().endswith(('>', '<', '!', '=')):
            target_col = parts[0].strip()
            calc_expr = parts[1].strip()
            is_assignment = True

    import re
    dangerous_patterns = [
        r'__\w+__',
        r'\bimport\b',
        r'\bfrom\b',
        r'\bexec\b',
        r'\beval\b',
        r'\bcompile\b',
        r'\bglobals\b',
        r'\blocals\b',
        r'\bgetattr\b',
        r'\bsetattr\b',
        r'\bdelattr\b',
        r'\b__import__\b',
    ]
    for pattern in dangerous_patterns:
        if re.search(pattern, calc_expr, re.IGNORECASE):
            raise ValueError(f"Expression contains blocked pattern: {pattern}")

    cdef dict env = dict(self.columns)
    try:
        res_col = eval(calc_expr, {"__builtins__": {}}, env)
    except Exception as e:
        raise ValueError(f"Failed to evaluate expression '{calc_expr}': {e}")

    if is_assignment:
        target_df.columns[target_col] = res_col
        return target_df
    else:
        return res_col

cdef _query_impl(FastDataFrame self, str expr):
    mask = _eval_impl(self, expr, False)
    if not type(mask).__name__ in ('IntColumn', 'BoolColumn'):
        raise TypeError("Query expression must evaluate to a boolean mask (IntColumn or BoolColumn)")
    return self.loc[mask]

cdef _append_row_impl(FastDataFrame self, dict kwargs):
    for col_name in self.columns:
        if col_name not in kwargs:
            raise ValueError(f"Missing value for column: {col_name}")

    cdef Py_ssize_t original_df_length = self.df_length
    cdef Py_ssize_t original_index_length = self.index.length if self.index is not None else 0

    try:
        for col_name, val in kwargs.items():
            self.columns[col_name].append(val)
        self.df_length += 1

        if self.index is not None:
            if self.index.index_type == INDEX_RANGE:
                self.index.length = self.df_length
            elif self.index.index_type == INDEX_INT:
                (<IntIndex>self.index).append(<int>(self.df_length - 1))
            elif self.index.index_type == INDEX_MULTI:
                raise NotImplementedError("Appending rows to MultiIndex is not supported via append_row yet.")
    except Exception:
        self.df_length = original_df_length
        for col_name, col_obj in self.columns.items():
            while col_obj.length > original_df_length:
                col_obj.pop_back()
        if self.index is not None:
            if self.index.index_type == INDEX_RANGE:
                self.index.length = original_index_length
            elif self.index.index_type == INDEX_INT:
                while (<IntIndex>self.index).length > original_index_length:
                    (<IntIndex>self.index).pop_back()
        raise

cdef _append_impl(FastDataFrame self, object row, bint ignore_index):
    cdef FastDataFrame new_df = self.copy()
    cdef list col_names
    if isinstance(row, dict):
        _append_row_impl(new_df, row)
    elif isinstance(row, list):
        col_names = list(new_df.columns.keys())
        if len(row) != len(col_names):
            raise ValueError(f"Length of row {len(row)} != number of columns {len(col_names)}")
        kwargs = {col_names[i]: row[i] for i in range(len(col_names))}
        _append_row_impl(new_df, kwargs)
    else:
        raise TypeError(f"Cannot append row from {type(row)}")
    if ignore_index:
        new_df.index = RangeIndex(0, new_df.df_length)
    return new_df

cdef _copy_impl(FastDataFrame self):
    cdef Py_ssize_t n, i
    cdef Py_ssize_t* idx_c
    cdef FastDataFrame new_df = FastDataFrame()
    new_df.df_length = self.df_length
    for col_name, col_obj in self.columns.items():
        new_df.columns[col_name] = col_obj.copy()

    if self.index is not None:
        if self.index.index_type == INDEX_RANGE:
            new_df.index = RangeIndex((<RangeIndex>self.index).start,
                                      self.index.length,
                                      (<RangeIndex>self.index).step)
        elif self.index.index_type == INDEX_INT:
            n = self.index.length
            idx_c = <Py_ssize_t*>aligned_malloc(max(n, 1) * sizeof(Py_ssize_t))
            if not idx_c: raise MemoryError()
            try:
                for i in range(n): idx_c[i] = i
                new_df.index = self.index._take(idx_c, n)
            finally:
                os_aligned_free(idx_c)
        else:
            new_df.index = self.index
    return new_df

cdef _pop_front_impl(FastDataFrame self):
    if self.df_length == 0: raise IndexError("pop from empty DataFrame")
    row = {}
    for col_name, col_obj in self.columns.items():
        row[col_name] = col_obj.pop_front()
    self.df_length -= 1
    if self.index is not None and self.index.length > 0:
        if self.index.index_type == INDEX_RANGE:
            (<RangeIndex>self.index).start += (<RangeIndex>self.index).step
        elif self.index.index_type == INDEX_INT:
            (<IntIndex>self.index).pop_back()
        self.index.length -= 1
    return row

cdef _pop_back_impl(FastDataFrame self):
    if self.df_length == 0: raise IndexError("pop from empty DataFrame")
    row = {}
    for col_name, col_obj in self.columns.items():
        row[col_name] = col_obj.pop_back()
    self.df_length -= 1
    if self.index is not None and self.index.length > 0:
        if self.index.index_type == INDEX_INT:
            (<IntIndex>self.index).pop_back()
        self.index.length -= 1
    return row

cdef _rename_impl(FastDataFrame self, dict mapper):
    cdef FastDataFrame result = _copy_impl(self)
    cdef str old_name, new_name
    for old_name, new_name in mapper.items():
        if old_name in result.columns:
            result.columns[new_name] = result.columns.pop(old_name)
    return result

cdef _reset_index_impl(FastDataFrame self):
    cdef IntColumn idx_col = IntColumn(self.df_length)
    cdef Py_ssize_t i
    idx_col.length = self.df_length
    for i in range(self.df_length):
        idx_col.data_c[i] = <int>i
    self.columns['index'] = idx_col
    self.index = RangeIndex(0, self.df_length)
    return self

cdef _reindex_impl(FastDataFrame self, list new_index):
    cdef Py_ssize_t new_len = len(new_index)
    cdef dict old_pos = {}
    cdef Py_ssize_t i, old_idx
    if self.index is not None and self.index.length > 0:
        old_vals = self.index.values()
        for i in range(len(old_vals)):
            old_pos[old_vals[i]] = i

    cdef FastDataFrame result = FastDataFrame()
    result.df_length = new_len
    result.index = RangeIndex(0, new_len)

    for col_name, col_obj in self.columns.items():
        new_col_list = []
        for i in range(new_len):
            if new_index[i] in old_pos:
                old_idx = old_pos[new_index[i]]
                new_col_list.append(col_obj[old_idx])
            else:
                new_col_list.append(None if col_obj.dtype in (TYPE_CHAR256, TYPE_UCS2) else (NAN if col_obj.dtype == TYPE_DOUBLE else 0))
        result.columns[col_name] = _column_from_list_impl(self, col_name, new_col_list)
    return result

cdef _insert_row_impl(FastDataFrame self, Py_ssize_t pos, dict kwargs):
    if pos < 0 or pos > self.df_length:
        raise IndexError(f"Position {pos} out of range [0, {self.df_length}]")

    for col_name in self.columns:
        if col_name not in kwargs:
            raise ValueError(f"Missing value for column '{col_name}'")

    cdef Py_ssize_t orig_len = self.df_length
    cdef Py_ssize_t ucs2_val_len, ucs2_start, _ci, _oi
    cdef UCS2Column ucs2_obj
    cdef str ucs2_val
    cdef list shifted_cols = []
    try:
        for col_name, col_obj in self.columns.items():
            val = kwargs[col_name]
            self._shift_right(col_obj, pos)
            shifted_cols.append(col_name)
            if col_obj.dtype == TYPE_INT:
                (<IntColumn>col_obj).data_c[pos] = <int>val
            elif col_obj.dtype == TYPE_DOUBLE:
                (<DoubleColumn>col_obj).data_c[pos] = <double>val
            elif col_obj.dtype == TYPE_CHAR256:
                bval = (<str>val).encode('utf-8')[:255] if isinstance(val, str) else str(val).encode('utf-8')[:255]
                memset((<AsciiColumn>col_obj).data_c + (pos * 256), 0, 256)
                memcpy((<AsciiColumn>col_obj).data_c + (pos * 256), <char*>bval, len(bval))
            elif col_obj.dtype == TYPE_UCS2:
                ucs2_val = <str>val if isinstance(val, str) else str(val)
                ucs2_obj = <UCS2Column>col_obj
                ucs2_val_len = len(ucs2_val)
                ucs2_start = ucs2_obj.offsets_c[pos]
                if ucs2_val_len > 0:
                    ucs2_obj._reserve_chars(ucs2_obj.char_capacity + ucs2_val_len)
                    if ucs2_obj.char_length - ucs2_start > 0:
                        memmove(ucs2_obj.chars_c + ucs2_start + ucs2_val_len,
                                ucs2_obj.chars_c + ucs2_start,
                                (ucs2_obj.char_length - ucs2_start) * sizeof(Py_UCS2))
                    if PyUnicode_KIND(ucs2_val) == PyUnicode_2BYTE_KIND:
                        memcpy(ucs2_obj.chars_c + ucs2_start,
                               PyUnicode_DATA(ucs2_val),
                               ucs2_val_len * sizeof(Py_UCS2))
                    else:
                        for _ci in range(ucs2_val_len):
                            ucs2_obj.chars_c[ucs2_start + _ci] = <Py_UCS2>PyUnicode_READ_CHAR(ucs2_val, _ci)
                    for _oi in range(pos + 1, ucs2_obj.length + 1):
                        ucs2_obj.offsets_c[_oi] += ucs2_val_len
                    ucs2_obj.char_length += ucs2_val_len
            elif col_obj.dtype == TYPE_DATE:
                (<DateColumn>col_obj).data_c[pos] = <int32_t>val
            elif col_obj.dtype == TYPE_BOOL:
                (<BoolColumn>col_obj).data_c[pos] = 1 if val else 0
            elif col_obj.dtype == TYPE_CATEGORY:
                (<CategoryColumn>col_obj).codes_c[pos] = (<CategoryColumn>col_obj)._get_or_add_code(str(val) if val is not None else '')
            elif col_obj.dtype == TYPE_DATETIME:
                (<DateTimeColumn>col_obj).data_c[pos] = 0
            elif col_obj.dtype == TYPE_TIME:
                (<TimeColumn>col_obj).data_c[pos] = <int>val if isinstance(val, int) else 0
        self.df_length += 1
        if self.index is not None:
            if self.index.index_type == INDEX_RANGE:
                self.index.length = self.df_length
            elif self.index.index_type == INDEX_INT:
                self.index.length = self.df_length
    except Exception:
        self.df_length = orig_len
        for cn in shifted_cols:
            self._shift_left(self.columns[cn], pos)
        raise

cdef _drop_impl(FastDataFrame self, list col_names):
    cdef FastDataFrame result = _copy_impl(self)
    cdef str name
    for name in col_names:
        if name in result.columns:
            del result.columns[name]
    return result

cdef _duplicate_column_impl(FastDataFrame self, str src_name, str new_name):
    if src_name not in self.columns:
        raise KeyError(f"Column '{src_name}' not found")
    if new_name is None:
        new_name = src_name + '_copy'
    self.columns[new_name] = self.columns[src_name].copy()

cdef _duplicate_rows_impl(FastDataFrame self, list indices):
    cdef Py_ssize_t n = self.df_length
    cdef list keep_idx = []
    cdef Py_ssize_t i, j
    cdef dict count_map = {}
    cdef Py_ssize_t total = 0

    if indices is None:
        return _copy_impl(self)

    for idx in indices:
        count_map[idx] = count_map.get(idx, 0) + 1

    for i in range(n):
        if i in count_map:
            for j in range(count_map[i]):
                keep_idx.append(i)
                total += 1
        else:
            keep_idx.append(i)
            total += 1

    cdef Py_ssize_t* idx_c = <Py_ssize_t*>aligned_malloc(max(total, 1) * sizeof(Py_ssize_t))
    if not idx_c: raise MemoryError()
    try:
        for i in range(total):
            idx_c[i] = keep_idx[i]
        return self._take(idx_c, total)
    finally:
        os_aligned_free(idx_c)
