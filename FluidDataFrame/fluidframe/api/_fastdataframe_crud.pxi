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
    cdef int* int_ptr
    cdef double* dbl_ptr
    cdef int64_t val64
    cdef bint all_ascii

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
        dbl_ptr = dbl_col.data_c
        for i in range(n): dbl_ptr[i] = NAN
        return dbl_col

    if isinstance(sample, bool):
        bool_col = BoolColumn(max(n, 1))
        bool_col.length = n
        for i in range(n): bool_col.data_c[i] = 1 if values[i] else 0
        return bool_col

    elif isinstance(sample, int):
        int_col = IntColumn(max(n, 1))
        int_col.length = n
        int_ptr = int_col.data_c
        for i in range(n):
            if values[i] is None:
                int_ptr[i] = 0
            else:
                val64 = <int64_t>values[i]
                if val64 > 2147483647 or val64 < -2147483648:
                    raise OverflowError(f"Value {val64} at index {i} out of range for int32")
                int_ptr[i] = <int>val64
        return int_col

    elif isinstance(sample, float):
        dbl_col = DoubleColumn(max(n, 1))
        dbl_col.length = n
        dbl_ptr = dbl_col.data_c
        for i in range(n):
            dbl_ptr[i] = <double>values[i] if values[i] is not None else NAN
        return dbl_col

    elif isinstance(sample, str):
        all_ascii = True
        for i in range(n):
            if values[i] is not None and not (<str>values[i]).isascii():
                all_ascii = False
                break

        if all_ascii:
            c256_col = AsciiColumn(max(n, 1))
            for i in range(n):
                c256_col.append(<str>(values[i]) if values[i] is not None else '')
            return c256_col
        else:
            ucs2_col = UCS2Column(max(n, 1))
            for i in range(n):
                ucs2_col.append(<str>(values[i]) if values[i] is not None else '')
            return ucs2_col

    raise TypeError(f"Unsupported type: {type(sample)}")

cdef inline void _probe_shape(BaseColumn col, bint* is2d, Py_ssize_t* c, Py_ssize_t* r):
    is2d[0] = col.is_2d
    c[0] = col.num_cols
    r[0] = col._get_rows_c()


cdef BaseColumn _column_from_2d_list_impl(list values, Py_ssize_t rows, Py_ssize_t cols):
    """由 2D 列的平铺列表重建列，修复了 NaN 内存残留垃圾值的问题"""
    cdef Py_ssize_t n = len(values)
    cdef object sample = None
    cdef Py_ssize_t i
    cdef DoubleColumn all_nan
    cdef double* d_ptr

    if n == 0:
        return IntColumn(0)

    for i in range(n):
        if values[i] is not None:
            sample = values[i]
            break

    if sample is None:
        all_nan = DoubleColumn(max(n, 1), cols=cols)
        all_nan.length = n
        d_ptr = all_nan.data_c
        with nogil:
            for i in range(n):
                d_ptr[i] = NAN
        return all_nan

    if isinstance(sample, bool):
        raise TypeError("apply() on 2D column: bool results not supported (2D columns are numeric)")
    elif isinstance(sample, int):
        return IntColumn.from_2d(values, rows, cols)
    return DoubleColumn.from_2d(values, rows, cols)


cdef _col_row_key(BaseColumn col, Py_ssize_t row):
    """行感知取值"""
    cdef bint is2d
    cdef Py_ssize_t c, r, j

    _probe_shape(col, &is2d, &c, &r)

    if not is2d or c <= 1:
        return col[row]

    cdef list vals = [None] * c
    for j in range(c):
        vals[j] = col[j * r + row]
    return tuple(vals)


cdef list _col_to_keys(BaseColumn col):
    """列 → 整行键列表：2D 列每项为子列元组，1D 列每项为标量。"""
    cdef bint is2d
    cdef Py_ssize_t c, r, i, j
    cdef list res, temp_row

    _probe_shape(col, &is2d, &c, &r)
    if r == 0:
        return []

    res = [None] * r
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


cdef inline bint _col_is_2d(BaseColumn col):
    """极限布尔探测：C 指针直接栈上抓取，0 对象分配"""
    cdef bint is2d
    cdef Py_ssize_t c, r
    _probe_shape(col, &is2d, &c, &r)
    return is2d and (c > 1)

cdef tuple _col_shape_probe(BaseColumn col):
    """统一列形状探测：返回 (is2d, cols, rows)。
    is2d 为存储层标志，cols=1 表示 1D（即使 is2d=True 且只有单子列的退化情形）。
    rows = logical 行数（length // cols）。
    """
    cdef bint is2d
    cdef Py_ssize_t c, r
    _probe_shape(col, &is2d, &c, &r)
    return (is2d, c, r)


cdef BaseColumn _col_from_keys_2d(BaseColumn src, list keys):
    """由整行键（_col_row_key 输出，各为定长子列取值）重建保持 cols 维的 2D 列。
    Int/Double 用公开 from_2d 工厂；Date 键为 ymd int，转回 days 存储；
    其余 2D 类型无对应存储，明确 NotImpl。"""
    cdef int dtype = src.dtype
    cdef Py_ssize_t nkeys = len(keys)
    if nkeys == 0:
        return IntColumn(0)
    cdef Py_ssize_t c = 1
    if dtype == TYPE_INT:
        c = (<IntColumn>src).num_cols
    elif dtype == TYPE_DOUBLE:
        c = (<DoubleColumn>src).num_cols
    elif dtype == TYPE_DATE:
        c = (<DateColumn>src).num_cols
    elif dtype == TYPE_DATETIME:
        c = (<DateTimeColumn>src).num_cols
    elif dtype == TYPE_TIME:
        c = (<TimeColumn>src).num_cols
    cdef Py_ssize_t total = nkeys * c
    cdef Py_ssize_t k, j
    cdef object flat = []
    cdef DateColumn dcol
    cdef int32_t* dptr
    cdef int ymd
    # from_2d 输入为列优先（先 col0 全部行，再 col1…），行键需转置
    for j in range(c):
        for k in range(nkeys):
            flat.append(keys[k][j])

    if dtype == TYPE_INT:
        return IntColumn.from_2d(<object>flat, nkeys, c)
    elif dtype == TYPE_DOUBLE:
        return DoubleColumn.from_2d(<object>flat, nkeys, c)
    elif dtype == TYPE_DATE:
        dcol = DateColumn(max(total, 1), cols=c)
        dcol.length = total
        dcol.num_cols = <int>c
        dcol.is_2d = True
        dcol.owns_data = True
        dptr = <int32_t*>dcol.data_c
        for k in range(nkeys):
            for j in range(c):
                ymd = <int>keys[k][j]
                dptr[j * nkeys + k] = _ymd_to_days(ymd // 10000, (ymd // 100) % 100, ymd % 100)
        return dcol
    raise NotImplementedError(f"value_counts on 2D column not supported for dtype {dtype}")

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
    cdef FastDataFrame result

    if self.index is not None and self.index.length > 0:
        old_vals = self.index.values()
        for i in range(len(old_vals)):
            old_pos[old_vals[i]] = i

    # 【性能修复 1】：一次性映射索引，将 O(C * N) 的字典查找暴降至 O(N)
    cdef Py_ssize_t* index_map = <Py_ssize_t*>aligned_malloc(max(new_len, 1) * sizeof(Py_ssize_t))
    if not index_map:
        raise MemoryError()

    try:
        for i in range(new_len):
            index_map[i] = old_pos.get(new_index[i], -1)

        result = FastDataFrame()
        result.df_length = new_len
        result.index = RangeIndex(0, new_len)

        for col_name, col_obj in self.columns.items():
            result.columns[col_name] = _reindex_column_impl(col_obj, index_map, new_len)

        return result
    finally:
        os_aligned_free(index_map)

cdef BaseColumn _reindex_column_impl(BaseColumn col, Py_ssize_t* index_map, Py_ssize_t new_len):
    """按 index_map 重新索引单列。2D 列保持 cols 维，缺失标签用 dtype 默认填充。"""
    cdef int dtype = col.dtype
    cdef Py_ssize_t cols = 1
    cdef Py_ssize_t rows = col.length
    cdef bint is2d = False

    if dtype == TYPE_INT:
        cols = (<IntColumn>col).num_cols; rows = (<IntColumn>col)._get_rows_c(); is2d = (<IntColumn>col).is_2d
    elif dtype == TYPE_DOUBLE:
        cols = (<DoubleColumn>col).num_cols; rows = (<DoubleColumn>col)._get_rows_c(); is2d = (<DoubleColumn>col).is_2d
    elif dtype == TYPE_DATE:
        cols = (<DateColumn>col).num_cols; rows = (<DateColumn>col)._get_rows_c(); is2d = (<DateColumn>col).is_2d
    elif dtype == TYPE_DATETIME:
        cols = (<DateTimeColumn>col).num_cols; rows = (<DateTimeColumn>col)._get_rows_c(); is2d = (<DateTimeColumn>col).is_2d
    elif dtype == TYPE_TIME:
        cols = (<TimeColumn>col).num_cols; rows = (<TimeColumn>col)._get_rows_c(); is2d = (<TimeColumn>col).is_2d

    cdef Py_ssize_t c = cols
    cdef Py_ssize_t i, j, old_idx
    cdef Py_ssize_t total = new_len * c
    cdef bint found
    cdef int* nd
    cdef int* sd
    cdef double* ndd
    cdef double* sdd
    cdef int32_t* nda
    cdef int32_t* sda
    cdef int64_t* ndt
    cdef int64_t* sdt
    cdef int32_t* ntm
    cdef int32_t* stm
    cdef IntColumn ric
    cdef DoubleColumn rdc
    cdef DateColumn rdc2
    cdef DateTimeColumn rdtc
    cdef TimeColumn rtmc

    # 1. 离散非连续对象列 (布尔、类别、字符串)
    if dtype == TYPE_BOOL:
        new_col = BoolColumn(max(new_len, 1))
        for i in range(new_len):
            old_idx = index_map[i]
            if old_idx != -1:
                new_col.append((<BoolColumn>col)[old_idx])
            else:
                new_col.append(False)
        return new_col
    elif dtype == TYPE_CATEGORY:
        new_col = CategoryColumn(max(new_len, 1))
        for i in range(new_len):
            old_idx = index_map[i]
            if old_idx != -1:
                new_col.append((<CategoryColumn>col)[old_idx])
            else:
                new_col.append(None)
        return new_col
    elif dtype == TYPE_CHAR256:
        new_col = AsciiColumn(max(new_len, 1))
        for i in range(new_len):
            old_idx = index_map[i]
            if old_idx != -1:
                new_col.append((<AsciiColumn>col)[old_idx])
            else:
                new_col.append('')
        return new_col
    elif dtype == TYPE_UCS2:
        new_col = UCS2Column(max(new_len, 1))
        for i in range(new_len):
            old_idx = index_map[i]
            if old_idx != -1:
                new_col.append((<UCS2Column>col)[old_idx])
            else:
                new_col.append('')
        return new_col

    if dtype == TYPE_INT:
        ric = IntColumn(max(total, 1), cols=c)
        ric.length = total
        if is2d:
            ric.num_cols = <int>c
            ric.is_2d = True
            ric.owns_data = True
        nd = ric.data_c
        sd = (<IntColumn>col).data_c

        memset(nd, 0, total * sizeof(int))
        with nogil:
            for j in range(c):
                for i in range(new_len):
                    old_idx = index_map[i]
                    if old_idx != -1:
                        nd[j * new_len + i] = sd[j * rows + old_idx]
        return ric

    elif dtype == TYPE_DOUBLE:
        rdc = DoubleColumn(max(total, 1), cols=c)
        rdc.length = total
        if is2d:
            rdc.num_cols = <int>c
            rdc.is_2d = True
            rdc.owns_data = True
        ndd = rdc.data_c
        sdd = (<DoubleColumn>col).data_c

        with nogil:
            for i in range(total): ndd[i] = NAN
            for j in range(c):
                for i in range(new_len):
                    old_idx = index_map[i]
                    if old_idx != -1:
                        ndd[j * new_len + i] = sdd[j * rows + old_idx]
        return rdc

    elif dtype == TYPE_DATE:
        rdc2 = DateColumn(max(total, 1), cols=c)
        rdc2.length = total
        if is2d:
            rdc2.num_cols = <int>c
            rdc2.is_2d = True
            rdc2.owns_data = True
        nda = rdc2.data_c
        sda = (<DateColumn>col).data_c

        memset(nda, 0, total * sizeof(int32_t))
        with nogil:
            for j in range(c):
                for i in range(new_len):
                    old_idx = index_map[i]
                    if old_idx != -1:
                        nda[j * new_len + i] = sda[j * rows + old_idx]
        return rdc2

    elif dtype == TYPE_DATETIME:
        rdtc = DateTimeColumn(max(total, 1), cols=c)
        rdtc.length = total
        if is2d:
            rdtc.num_cols = <int>c
            rdtc.is_2d = True
            rdtc.owns_data = True
        ndt = rdtc.data_c
        sdt = (<DateTimeColumn>col).data_c

        memset(ndt, 0, total * sizeof(int64_t))
        with nogil:
            for j in range(c):
                for i in range(new_len):
                    old_idx = index_map[i]
                    if old_idx != -1:
                        ndt[j * new_len + i] = sdt[j * rows + old_idx]
        return rdtc

    elif dtype == TYPE_TIME:
        rtmc = TimeColumn(max(total, 1), cols=c)
        rtmc.length = total
        if is2d:
            rtmc.num_cols = <int>c
            rtmc.is_2d = True
            rtmc.owns_data = True
        ntm = rtmc.data_c
        stm = (<TimeColumn>col).data_c

        memset(ntm, 0, total * sizeof(int32_t))
        with nogil:
            for j in range(c):
                for i in range(new_len):
                    old_idx = index_map[i]
                    if old_idx != -1:
                        ntm[j * new_len + i] = stm[j * rows + old_idx]
        return rtmc

    raise TypeError(f"Unsupported column dtype: {dtype}")

cdef _insert_row_impl(FastDataFrame self, Py_ssize_t pos, dict kwargs):
    if pos < 0 or pos > self.df_length:
        raise IndexError(f"Position {pos} out of range [0, {self.df_length}]")

    cdef Py_ssize_t orig_len = self.df_length
    cdef Py_ssize_t ucs2_val_len, ucs2_start, _ci, _oi, _j2
    cdef Py_ssize_t _ac_len, _ac_start, _ac_vlen, _ac_span, _ac_tail, _bc_len
    cdef bint is2d
    cdef Py_ssize_t c_out, r_out

    cdef int dtype
    cdef int32_t days
    cdef int64_t tmp_dt
    cdef UCS2Column ucs2_obj
    cdef AsciiColumn cc_obj
    cdef BoolColumn bc_obj
    cdef CategoryColumn catc_obj
    cdef str ucs2_val
    cdef bytes bval

    cdef list shifted_cols = []
    cdef dict row_vals = {}

    # 1. 预先抓取并验证，消除内层哈希消耗
    for col_name in self.columns:
        if col_name not in kwargs:
            raise ValueError(f"Missing value for column '{col_name}'")
        row_vals[col_name] = kwargs[col_name]

    try:
        for col_name, col_obj in self.columns.items():
            val = row_vals[col_name]
            dtype = col_obj.dtype

            # -------------------------------------------------------------
            # 定宽列组：底层使用快速 1D/2D 平移宏
            # -------------------------------------------------------------
            if dtype in (TYPE_INT, TYPE_DOUBLE, TYPE_DATE, TYPE_DATETIME, TYPE_TIME):
                self._shift_right(col_obj, pos)
                shifted_cols.append(col_name) # 此时 length 已经成功 +1，安全入列

                _probe_shape(col_obj, &is2d, &c_out, &r_out) # 获取新分配好的维度

                if dtype == TYPE_INT:
                    if isinstance(val, int) and (val > 2147483647 or val < -2147483648):
                        raise OverflowError(f"Value {val} out of range for int32")
                    if is2d and c_out > 1:
                        for _j2 in range(c_out):
                            (<IntColumn>col_obj).data_c[_j2 * r_out + pos] = <int>val[_j2]
                    else:
                        (<IntColumn>col_obj).data_c[pos] = <int>val

                elif dtype == TYPE_DOUBLE:
                    if is2d and c_out > 1:
                        for _j2 in range(c_out):
                            (<DoubleColumn>col_obj).data_c[_j2 * r_out + pos] = <double>val[_j2]
                    else:
                        (<DoubleColumn>col_obj).data_c[pos] = <double>val

                elif dtype == TYPE_DATE:
                    if isinstance(val, int) and val > 10000000:
                        days = _ymd_to_days(val // 10000, (val // 100) % 100, val % 100)
                    else:
                        days = <int32_t>val
                    if is2d and c_out > 1:
                        for _j2 in range(c_out):
                            (<DateColumn>col_obj).data_c[_j2 * r_out + pos] = days
                    else:
                        (<DateColumn>col_obj).data_c[pos] = days

                elif dtype == TYPE_DATETIME:
                    if isinstance(val, int) and val > 10000000:
                        days = _ymd_to_days(val // 10000, (val // 100) % 100, val % 100)
                        tmp_dt = <int64_t>days * 86400000000LL
                    else:
                        tmp_dt = <int64_t>val * 1000000LL
                    if is2d and c_out > 1:
                        for _j2 in range(c_out):
                            (<DateTimeColumn>col_obj).data_c[_j2 * r_out + pos] = tmp_dt
                    else:
                        (<DateTimeColumn>col_obj).data_c[pos] = tmp_dt

                elif dtype == TYPE_TIME:
                    if not isinstance(val, int):
                        raise TypeError(f"Time column expects int, got {type(val)}")
                    if is2d and c_out > 1:
                        for _j2 in range(c_out):
                            (<TimeColumn>col_obj).data_c[_j2 * r_out + pos] = <int32_t>val
                    else:
                        (<TimeColumn>col_obj).data_c[pos] = <int32_t>val

            # -------------------------------------------------------------
            # 变长列组：ASCII / UCS2 (平移 + 写入 一体化处理)
            # -------------------------------------------------------------
            elif dtype == TYPE_CHAR256:
                cc_obj = <AsciiColumn>col_obj
                if cc_obj.length >= cc_obj.capacity:
                    cc_obj._reserve_rows(max(16, cc_obj.capacity * 2))

                if isinstance(val, bytes): bval = val
                elif isinstance(val, str): bval = val.encode('ascii', errors='ignore')
                else: bval = ('' if val is None else str(val)).encode('ascii', errors='ignore')

                _ac_len = cc_obj.length
                _ac_start = cc_obj.offsets_c[pos]
                _ac_vlen = len(bval)
                _ac_span = _ac_vlen + 1

                if cc_obj.byte_length + _ac_span > cc_obj.byte_capacity:
                    cc_obj._reserve_bytes(max(cc_obj.byte_capacity * 2, cc_obj.byte_length + _ac_span + 4096))

                _ac_tail = cc_obj.byte_length - _ac_start
                if _ac_tail > 0:
                    memmove(cc_obj.data_c + _ac_start + _ac_span,
                            cc_obj.data_c + _ac_start,
                            _ac_tail)
                if _ac_vlen > 0:
                    memcpy(cc_obj.data_c + _ac_start, <char*>bval, _ac_vlen)
                cc_obj.data_c[_ac_start + _ac_vlen] = 0
                cc_obj.byte_length += _ac_span

                if _ac_len > pos:
                    memmove(cc_obj.offsets_c + pos + 2,
                            cc_obj.offsets_c + pos + 1,
                            (_ac_len - pos) * sizeof(Py_ssize_t))
                cc_obj.offsets_c[pos + 1] = _ac_start + _ac_span
                for _oi in range(pos + 2, _ac_len + 2):
                    cc_obj.offsets_c[_oi] += _ac_span

                cc_obj.length = _ac_len + 1
                shifted_cols.append(col_name)

            elif dtype == TYPE_UCS2:
                ucs2_obj = <UCS2Column>col_obj
                if ucs2_obj.length >= ucs2_obj.capacity:
                    ucs2_obj._reserve_rows(max(16, ucs2_obj.capacity * 2))

                ucs2_val = val if isinstance(val, str) else ('' if val is None else str(val))
                ucs2_val_len = len(ucs2_val)
                ucs2_start = ucs2_obj.offsets_c[pos]

                if ucs2_val_len > 0 and ucs2_obj.char_length + ucs2_val_len > ucs2_obj.char_capacity:
                    ucs2_obj._reserve_chars(max(ucs2_obj.char_capacity * 2, ucs2_obj.char_length + ucs2_val_len + 1024))

                if ucs2_val_len > 0:
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
                    ucs2_obj.char_length += ucs2_val_len

                if ucs2_obj.length > pos:
                    memmove(ucs2_obj.offsets_c + pos + 2,
                            ucs2_obj.offsets_c + pos + 1,
                            (ucs2_obj.length - pos) * sizeof(Py_ssize_t))
                ucs2_obj.offsets_c[pos + 1] = ucs2_start + ucs2_val_len
                for _oi in range(pos + 2, ucs2_obj.length + 2):
                    ucs2_obj.offsets_c[_oi] += ucs2_val_len

                ucs2_obj.length += 1
                shifted_cols.append(col_name)

            # -------------------------------------------------------------
            # 固定一维结构组：离散类型
            # -------------------------------------------------------------
            elif dtype == TYPE_BOOL:
                bc_obj = <BoolColumn>col_obj
                _bc_len = bc_obj.length
                if _bc_len >= bc_obj.capacity:
                    bc_obj._reserve(max(16, bc_obj.capacity * 2))
                _c_shift_right(bc_obj.data_c, pos, _bc_len, sizeof(unsigned char))
                bc_obj.data_c[pos] = 1 if val else 0
                bc_obj.length = _bc_len + 1
                shifted_cols.append(col_name)

            elif dtype == TYPE_CATEGORY:
                catc_obj = <CategoryColumn>col_obj
                _bc_len = catc_obj.length
                if _bc_len >= catc_obj.capacity:
                    catc_obj._reserve(max(16, catc_obj.capacity * 2))
                _c_shift_right(catc_obj.codes_c, pos, _bc_len, sizeof(int))

                if val is None:
                    catc_obj.codes_c[pos] = catc_obj._get_or_add_code('')
                else:
                    catc_obj.codes_c[pos] = catc_obj._get_or_add_code(str(val))

                catc_obj.length = _bc_len + 1
                shifted_cols.append(col_name)

        # -------------------------------------------------------------
        # 整体 DataFrame 状态提交
        # -------------------------------------------------------------
        self.df_length += 1
        if self.index is not None:
            if self.index.index_type == INDEX_RANGE or self.index.index_type == INDEX_INT:
                self.index.length = self.df_length

    except Exception:
        # 【完美回滚】：由于 shifted_cols 严格遵循了修改后入列的纪律
        self.df_length = orig_len
        for cn in shifted_cols:
            self._shift_left(self.columns[cn], pos)
        raise

cdef _drop_with_index_impl(FastDataFrame self, list drop_indices, bint inplace):
    cdef Py_ssize_t n_orig = self.df_length
    if n_orig == 0:
        return self if inplace else _copy_impl(self)

    cdef char* drop_mask = <char*>aligned_malloc(n_orig * sizeof(char))
    if not drop_mask: raise MemoryError()
    memset(drop_mask, 0, n_orig)

    cdef Py_ssize_t idx, i, j
    cdef Py_ssize_t keep_count = 0
    cdef Py_ssize_t* keep_idx_c = NULL
    cdef FastDataFrame result_df
    cdef str name
    cdef BaseColumn col, new_col

    try:
        for py_idx in drop_indices:
            idx = <Py_ssize_t>py_idx
            if idx < 0: idx += n_orig
            if idx < 0 or idx >= n_orig:
                raise IndexError(f"Index {py_idx} out of bounds")
            drop_mask[idx] = 1

        for i in range(n_orig):
            if drop_mask[i] == 0:
                keep_count += 1

        if keep_count == n_orig:
            return self if inplace else _copy_impl(self)

        if keep_count == 0:
            if inplace:
                self.df_length = 0
                for col_obj in self.columns.values(): col_obj.length = 0
                return self
            else:
                return FastDataFrame()

        keep_idx_c = <Py_ssize_t*>aligned_malloc(keep_count * sizeof(Py_ssize_t))
        if not keep_idx_c: raise MemoryError()

        j = 0
        for i in range(n_orig):
            if drop_mask[i] == 0:
                keep_idx_c[j] = i
                j += 1

        if inplace:
            result_df = self
            self.df_length = keep_count
        else:
            result_df = FastDataFrame()
            result_df.df_length = keep_count

        for name, col in self.columns.items():
            new_col = self._reorder_column(col, keep_idx_c, keep_count, inplace)
            if not inplace:
                result_df.columns[name] = new_col

        if self.index is not None:
            if inplace:
                self.index = self.index._take(keep_idx_c, keep_count)
            else:
                result_df.index = self.index._take(keep_idx_c, keep_count)

        return result_df
    finally:
        if drop_mask: os_aligned_free(drop_mask)
        if keep_idx_c: os_aligned_free(keep_idx_c)

cdef _drop_index_impl(FastDataFrame self, object labels, bint inplace):
    if self.index is None:
        raise ValueError("DataFrame has no index set")
    cdef list drop_positions = []
    if isinstance(labels, list):
        for label in labels:
            drop_positions.append(self.index.get_loc(label))
    else:
        drop_positions.append(self.index.get_loc(labels))
    return _drop_with_index_impl(self, drop_positions, inplace)

cdef _drop_raw_impl(FastDataFrame self, Py_ssize_t row_idx):
    if row_idx < 0: row_idx += self.df_length
    if row_idx < 0 or row_idx >= self.df_length:
        raise IndexError("Row index out of range")

    cdef str name
    cdef BaseColumn col
    cdef Py_ssize_t rem_len_idx, ii
    cdef IntColumn tmp_col, idx_col
    cdef RangeIndex ri

    for name, col in self.columns.items():
        self._drop_row_column(col, row_idx)

    self.df_length -= 1
    if self.index is not None and self.index.length > 0:
        if self.index.index_type == INDEX_RANGE:
            ri = <RangeIndex>self.index
            if row_idx == 0:
                ri.start += ri.step
            elif row_idx == self.index.length - 1:
                pass
            else:
                # When deleting from middle, convert to IntIndex
                rem_len_idx = self.index.length - row_idx - 1
                tmp_col = IntColumn(self.index.length)
                tmp_col.length = self.index.length
                for ii in range(self.index.length):
                    tmp_col.data_c[ii] = <int>(ri.start + ii * ri.step)
                memmove(tmp_col.data_c + row_idx,
                        tmp_col.data_c + row_idx + 1,
                        rem_len_idx * sizeof(int))
                tmp_col.length = self.index.length - 1
                self.index = IntIndex(tmp_col)
                return
        elif self.index.index_type == INDEX_INT:
            idx_col = (<IntIndex>self.index).data
            rem_len_idx = self.index.length - row_idx - 1
            if rem_len_idx > 0:
                memmove(idx_col.data_c + row_idx,
                        idx_col.data_c + row_idx + 1,
                        rem_len_idx * sizeof(int))
            idx_col.length -= 1
        self.index.length -= 1

cdef _insert_column_impl(FastDataFrame self, str col_name, object col_data, Py_ssize_t pos):
    cdef BaseColumn col
    cdef bint is2d
    cdef Py_ssize_t c_col, r_col

    if isinstance(col_data, BaseColumn):
        col = col_data
    elif isinstance(col_data, list):
        col = _column_from_list_impl(self, col_name, col_data)
    else:
        raise TypeError(f"Cannot create column from {type(col_data)}")

    is2d, c_col, r_col = _col_shape_probe(col)

    if self.df_length == 0 and r_col > 0:
        self.df_length = <uint32_t>r_col
    elif self.df_length > 0 and r_col > self.df_length:
        raise ValueError(f"Shape mismatch: DataFrame has {self.df_length} rows, incoming column has {r_col} logical rows.")

    cdef list names = list(self.columns.keys())
    cdef Py_ssize_t n = len(names)

    if pos < 0 or pos >= n:
        self.columns[col_name] = col
    else:
        new_cols = {}
        for i in range(n):
            if i == pos:
                new_cols[col_name] = col
            new_cols[names[i]] = self.columns[names[i]]
        self.columns = new_cols

cdef _duplicate_column_impl(FastDataFrame self, str src_name, str new_name):
    if src_name not in self.columns:
        raise KeyError(f"Column '{src_name}' not found")
    if new_name is None:
        new_name = src_name + '_dup'
    self.columns[new_name] = self.columns[src_name].copy()

cdef _duplicate_rows_impl(FastDataFrame self, list indices):
    if indices is None:
        return _copy_impl(self)
    cdef Py_ssize_t n = len(indices)
    cdef Py_ssize_t* idx_c = <Py_ssize_t*>aligned_malloc(max(n, 1) * sizeof(Py_ssize_t))
    if not idx_c: raise MemoryError()
    cdef Py_ssize_t i
    try:
        for i in range(n):
            idx_c[i] = indices[i]
        return self._take(idx_c, n)
    finally:
        os_aligned_free(idx_c)
