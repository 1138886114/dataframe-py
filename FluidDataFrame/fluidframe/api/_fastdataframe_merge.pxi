# _fastdataframe_merge.pxi — Merge, join, concat, duplicates (module-level cdef)
# Included at module level in fastdataframe.pyx (after class definition)

cdef _concat_impl(FastDataFrame self, FastDataFrame other):
    if set(self.columns.keys()) != set(other.columns.keys()):
        raise ValueError("Schema mismatch during concat")
    cdef FastDataFrame result = self.copy()
    for col_name, col_obj in result.columns.items():
        col_obj.concat(other.columns[col_name])
    result.df_length = self.df_length + other.df_length
    return result

cdef _merge_impl(FastDataFrame self, FastDataFrame other, list on, str how='inner'):
    cdef list join_cols
    if on is not None:
        join_cols = on if isinstance(on, list) else [on]
        for jc in join_cols:
            if jc not in self.columns:
                raise KeyError(f"Column '{jc}' not found in left DataFrame")
            if jc not in other.columns:
                raise KeyError(f"Column '{jc}' not found in right DataFrame")
    else:
        common = set(self.columns.keys()) & set(other.columns.keys())
        if not common:
            raise ValueError("No common columns to merge on")
        join_cols = sorted(common)

    cdef dict left_map = {}
    cdef dict right_map = {}
    cdef Py_ssize_t i, j

    for i in range(self.df_length):
        key = tuple(self.columns[jc][i] for jc in join_cols)
        if key not in left_map:
            left_map[key] = []
        left_map[key].append(i)

    for j in range(other.df_length):
        key = tuple(other.columns[jc][j] for jc in join_cols)
        if key not in right_map:
            right_map[key] = []
        right_map[key].append(j)

    cdef list left_indices = []
    cdef list right_indices = []

    if how == 'inner':
        for key in left_map:
            if key in right_map:
                for li in left_map[key]:
                    for ri in right_map[key]:
                        left_indices.append(li)
                        right_indices.append(ri)
    elif how == 'left':
        for key in left_map:
            rights = right_map.get(key, [])
            if not rights:
                for li in left_map[key]:
                    left_indices.append(li)
                    right_indices.append(-1)
            else:
                for li in left_map[key]:
                    for ri in rights:
                        left_indices.append(li)
                        right_indices.append(ri)
    elif how == 'right':
        for key in right_map:
            lefts = left_map.get(key, [])
            if not lefts:
                for ri in right_map[key]:
                    left_indices.append(-1)
                    right_indices.append(ri)
            else:
                for li in lefts:
                    for ri in right_map[key]:
                        left_indices.append(li)
                        right_indices.append(ri)
    elif how == 'outer':
        all_keys = set(left_map.keys()) | set(right_map.keys())
        for key in all_keys:
            lefts = left_map.get(key, [])
            rights = right_map.get(key, [])
            if not lefts:
                for ri in rights:
                    left_indices.append(-1)
                    right_indices.append(ri)
            elif not rights:
                for li in lefts:
                    left_indices.append(li)
                    right_indices.append(-1)
            else:
                for li in lefts:
                    for ri in rights:
                        left_indices.append(li)
                        right_indices.append(ri)

    cdef Py_ssize_t result_len = len(left_indices)
    cdef FastDataFrame result = FastDataFrame()
    result.df_length = result_len
    result.index = RangeIndex(0, result_len)

    cdef set join_set = set(join_cols)
    cdef list left_col_names = [n for n in self.columns.keys() if n not in join_set]
    cdef list right_col_names = [n for n in other.columns.keys() if n not in join_set]

    cdef Py_ssize_t* left_idx_c = <Py_ssize_t*>aligned_malloc(max(result_len, 1) * sizeof(Py_ssize_t))
    cdef Py_ssize_t* right_idx_c = <Py_ssize_t*>aligned_malloc(max(result_len, 1) * sizeof(Py_ssize_t))
    if not left_idx_c or not right_idx_c:
        if left_idx_c: os_aligned_free(left_idx_c)
        if right_idx_c: os_aligned_free(right_idx_c)
        raise MemoryError()
    cdef BaseColumn src_col

    try:
        for i in range(result_len):
            left_idx_c[i] = left_indices[i]
            right_idx_c[i] = right_indices[i]

        for cn in join_cols:
            src_col = self.columns[cn]
            if src_col.dtype == TYPE_INT:
                result.columns[cn] = (<IntColumn>src_col).gather_with_sentinel(left_idx_c, result_len, 0)
            elif src_col.dtype == TYPE_DOUBLE:
                result.columns[cn] = (<DoubleColumn>src_col).gather_with_sentinel(left_idx_c, result_len, NAN)
            else:
                result.columns[cn] = self._reorder_column(src_col, left_idx_c, result_len, False)

        for cn in left_col_names:
            src_col = self.columns[cn]
            out_name = cn + '_x' if cn in right_col_names else cn
            if src_col.dtype == TYPE_INT:
                result.columns[out_name] = (<IntColumn>src_col).gather_with_sentinel(left_idx_c, result_len, 0)
            elif src_col.dtype == TYPE_DOUBLE:
                result.columns[out_name] = (<DoubleColumn>src_col).gather_with_sentinel(left_idx_c, result_len, NAN)
            else:
                result.columns[out_name] = self._reorder_column(src_col, left_idx_c, result_len, False)

        for cn in right_col_names:
            src_col = other.columns[cn]
            out_name = cn + '_y' if cn in result.columns or cn in left_col_names else cn
            if src_col.dtype == TYPE_INT:
                result.columns[out_name] = (<IntColumn>src_col).gather_with_sentinel(right_idx_c, result_len, 0)
            elif src_col.dtype == TYPE_DOUBLE:
                result.columns[out_name] = (<DoubleColumn>src_col).gather_with_sentinel(right_idx_c, result_len, NAN)
            else:
                result.columns[out_name] = other._reorder_column(src_col, right_idx_c, result_len, False)

        return result
    finally:
        os_aligned_free(left_idx_c)
        os_aligned_free(right_idx_c)

cdef _col_row_key(BaseColumn col, Py_ssize_t row):
    """返回某列在逻辑行 row 的 Python 键：1D 列返回标量，2D 列返回
    各子列取值构成的 tuple（整行重排语义，与 _take 一致）。"""
    cdef bint is2d = False
    cdef Py_ssize_t c = 1
    cdef Py_ssize_t r = col.length
    if col.dtype == TYPE_INT:
        is2d = (<IntColumn>col).get_is_2d_c(); c = (<IntColumn>col).get_cols_c(); r = (<IntColumn>col).get_rows_c()
    elif col.dtype == TYPE_DOUBLE:
        is2d = (<DoubleColumn>col).get_is_2d_c(); c = (<DoubleColumn>col).get_cols_c(); r = (<DoubleColumn>col).get_rows_c()
    elif col.dtype == TYPE_DATE:
        is2d = (<DateColumn>col).get_is_2d_c(); c = (<DateColumn>col).get_cols_c(); r = (<DateColumn>col).get_rows_c()
    elif col.dtype == TYPE_DATETIME:
        is2d = (<DateTimeColumn>col).get_is_2d_c(); c = (<DateTimeColumn>col).get_cols_c(); r = (<DateTimeColumn>col).get_rows_c()
    elif col.dtype == TYPE_TIME:
        is2d = (<TimeColumn>col).get_is_2d_c(); c = (<TimeColumn>col).get_cols_c(); r = (<TimeColumn>col).get_rows_c()

    if not is2d or c <= 1:
        return col[row]
    cdef Py_ssize_t j
    cdef list vals = [None] * c
    for j in range(c):
        vals[j] = col[j * r + row]
    return tuple(vals)

cdef _duplicated_impl(FastDataFrame self, subset=None, str keep='first'):
    cdef Py_ssize_t n = self.df_length
    cdef IntColumn result = IntColumn(max(n, 1))
    result.length = n
    if n == 0:
        return result

    if subset is None:
        subset = list(self.columns.keys())

    cdef dict seen = {}
    cdef Py_ssize_t i
    cdef tuple row_key
    cdef BaseColumn col

    if keep == 'first':
        for i in range(n):
            row_key = tuple(_col_row_key(self.columns[c], i) for c in subset)
            if row_key in seen:
                result.data_c[i] = 1
            else:
                seen[row_key] = i
                result.data_c[i] = 0
    elif keep == 'last':
        for i in range(n):
            row_key = tuple(_col_row_key(self.columns[c], i) for c in subset)
            if row_key in seen:
                result.data_c[seen[row_key]] = 1
            seen[row_key] = i
        for i in range(n):
            row_key = tuple(_col_row_key(self.columns[c], i) for c in subset)
            if seen[row_key] == i:
                result.data_c[i] = 0
    else:
        for i in range(n):
            result.data_c[i] = 0
    return result

cdef _drop_duplicates_impl(FastDataFrame self, subset=None, str keep='first', bint inplace=False):
    cdef IntColumn mask = _duplicated_impl(self, subset, keep)
    cdef Py_ssize_t n = self.df_length
    cdef Py_ssize_t keep_count = n - mask.count_nonzero()

    if keep_count == n:
        return self if inplace else _copy_impl(self)

    cdef Py_ssize_t* keep_idx = <Py_ssize_t*>aligned_malloc(max(keep_count, 1) * sizeof(Py_ssize_t))
    if not keep_idx: raise MemoryError()
    mask.gather_zero_to(keep_idx)

    cdef FastDataFrame result = self._take(keep_idx, keep_count)
    os_aligned_free(keep_idx)
    if inplace:
        self.columns = result.columns
        self.df_length = result.df_length
        self.index = result.index
        return self
    return result
