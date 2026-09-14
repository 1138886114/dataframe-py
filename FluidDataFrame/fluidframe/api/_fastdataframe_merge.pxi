# _fastdataframe_merge.pxi — Merge, join, concat, duplicates (module-level cdef)
# Included at module level in fastdataframe.pyx (after class definition)

cdef _concat_impl(FastDataFrame self, FastDataFrame other):
    if set(self._columns.keys()) != set(other._columns.keys()):
        raise ValueError("Schema mismatch during concat")
    cdef FastDataFrame result = self.copy()
    for col_name, col_obj in result._columns.items():
        col_obj.concat(other._columns[col_name])
    result.df_length = self.df_length + other.df_length
    result.index = self.index._concat(other.index)
    return result

cdef inline BaseColumn _gather_or_reorder(FastDataFrame df, BaseColumn src_col,
                                          Py_ssize_t* idx_c, Py_ssize_t result_len):
    """
    终极版安全搜集分配器：
    利用明确的维度约束（只有数字/时间可为 2D），彻底删除了无用的分支探测。
    完全摒弃动态 append，采用预分配 length 结合底层 C 宏/CMOV 极速搬运！
    """
    cdef int dtype = src_col.dtype
    cdef Py_ssize_t i, idx
    cdef bint is2d
    cdef Py_ssize_t c_out, r_out
    cdef int null_code
    cdef int* dst_codes
    cdef int* src_codes

    # =========================================================================
    # 第一梯队：原生享有 Gather C 宏 的类型 (自带 2D 识别)
    # =========================================================================
    if dtype == TYPE_INT:
        return (<IntColumn>src_col).gather_with_sentinel(idx_c, result_len, 0)
    elif dtype == TYPE_DOUBLE:
        return (<DoubleColumn>src_col).gather_with_sentinel(idx_c, result_len, NAN)

    # =========================================================================
    # 第二梯队：可能为 2D，利用宏 _col_gather_zero 批量搬运的定宽类型
    # =========================================================================
    elif dtype == TYPE_DATE:
        _probe_shape(src_col, &is2d, &c_out, &r_out)
        res_dt = DateColumn(max(result_len * c_out, 1), cols=c_out)
        res_dt.length = result_len * c_out
        if is2d:
            res_dt.num_cols = <int>c_out
            res_dt.is_2d = True
            res_dt.owns_data = True
        if is2d and c_out > 1:
            _col_gather_2d_zero(res_dt.data_c, (<DateColumn>src_col).data_c, idx_c, result_len, r_out, c_out, sizeof(int32_t))
        else:
            _col_gather_zero(res_dt.data_c, (<DateColumn>src_col).data_c, idx_c, result_len, sizeof(int32_t))
        return res_dt

    elif dtype == TYPE_DATETIME:
        _probe_shape(src_col, &is2d, &c_out, &r_out)
        res_dtn = DateTimeColumn(max(result_len * c_out, 1), cols=c_out)
        res_dtn.length = result_len * c_out
        if is2d:
            res_dtn.num_cols = <int>c_out
            res_dtn.is_2d = True
            res_dtn.owns_data = True
        if is2d and c_out > 1:
            _col_gather_2d_zero(res_dtn.data_c, (<DateTimeColumn>src_col).data_c, idx_c, result_len, r_out, c_out, sizeof(int64_t))
        else:
            _col_gather_zero(res_dtn.data_c, (<DateTimeColumn>src_col).data_c, idx_c, result_len, sizeof(int64_t))
        return res_dtn

    elif dtype == TYPE_TIME:
        _probe_shape(src_col, &is2d, &c_out, &r_out)
        res_tm = TimeColumn(max(result_len * c_out, 1), cols=c_out)
        res_tm.length = result_len * c_out
        if is2d:
            res_tm.num_cols = <int>c_out
            res_tm.is_2d = True
            res_tm.owns_data = True
        if is2d and c_out > 1:
            _col_gather_2d_zero(res_tm.data_c, (<TimeColumn>src_col).data_c, idx_c, result_len, r_out, c_out, sizeof(int32_t))
        else:
            _col_gather_zero(res_tm.data_c, (<TimeColumn>src_col).data_c, idx_c, result_len, sizeof(int32_t))
        return res_tm

    # =========================================================================
    # 第三梯队：永远为 1D 的离散定宽类型 (无需 _probe_shape，直捣黄龙)
    # =========================================================================
    elif dtype == TYPE_BOOL:
        res_bool = BoolColumn(max(result_len, 1))
        res_bool.length = result_len
        # Bool 直接按单字节（unsigned char）调 1D 零分流搬运宏
        _col_gather_zero(res_bool.data_c, (<BoolColumn>src_col).data_c, idx_c, result_len, sizeof(unsigned char))
        return res_bool

    elif dtype == TYPE_CATEGORY:
        res_cat = src_col.copy()
        res_cat._reserve(max(result_len, 1))
        res_cat.length = result_len

        null_code = (<CategoryColumn>res_cat)._get_or_add_code('')
        dst_codes = (<CategoryColumn>res_cat).codes_c
        src_codes = (<CategoryColumn>src_col).codes_c

        with nogil:
            for i in range(result_len):
                idx = idx_c[i]
                dst_codes[i] = src_codes[idx] if idx >= 0 else null_code
        return res_cat

    # =========================================================================
    # 第四梯队：永远为 1D 的变长字符串类型
    # =========================================================================
    elif dtype == TYPE_CHAR256:
        res_c256 = AsciiColumn(max(result_len, 1))
        res_c256._reserve_bytes((<AsciiColumn>src_col).byte_length)
        for i in range(result_len):
            idx = idx_c[i]
            res_c256.append((<AsciiColumn>src_col)[idx] if idx >= 0 else '')
        return res_c256

    elif dtype == TYPE_UCS2:
        res_ucs2 = UCS2Column(max(result_len, 1))
        res_ucs2._reserve_chars((<UCS2Column>src_col).char_length)
        for i in range(result_len):
            idx = idx_c[i]
            res_ucs2.append((<UCS2Column>src_col)[idx] if idx >= 0 else '')
        return res_ucs2

    raise TypeError(f"Unsupported column dtype for merge gather: {dtype}")


cdef void _join_col_fixup(BaseColumn res_col, BaseColumn r_col,
                          Py_ssize_t* l_idx, Py_ssize_t* r_idx, Py_ssize_t n):
    cdef int dtype = res_col.dtype
    cdef bint is2d, ris2d
    cdef Py_ssize_t c, rr, rc, rr2, i, j, ridx
    cdef CategoryColumn cat_res, cat_r

    if n == 0 or r_col.dtype != dtype:
        return

    _probe_shape(r_col, &ris2d, &c, &rr)
    _probe_shape(res_col, &is2d, &rc, &rr2)

    if is2d != ris2d or rc != c:
        return

    cdef int* i_dst
    cdef int* i_src
    cdef double* d_dst
    cdef double* d_src
    cdef int32_t* i32_dst
    cdef int32_t* i32_src
    cdef int64_t* i64_dst
    cdef int64_t* i64_src
    cdef unsigned char* b_dst
    cdef unsigned char* b_src

    # 1. 定宽类型：解开 GIL 锁，纯 C 指针平移
    if dtype == TYPE_INT:
        i_dst = (<IntColumn>res_col).data_c
        i_src = (<IntColumn>r_col).data_c
        with nogil:
            if is2d and c > 1:
                for i in range(n):
                    if l_idx[i] < 0:
                        ridx = r_idx[i]
                        for j in range(c): i_dst[j * n + i] = i_src[j * rr + ridx]
            else:
                for i in range(n):
                    if l_idx[i] < 0: i_dst[i] = i_src[r_idx[i]]

    elif dtype == TYPE_DOUBLE:
        d_dst = (<DoubleColumn>res_col).data_c
        d_src = (<DoubleColumn>r_col).data_c
        with nogil:
            if is2d and c > 1:
                for i in range(n):
                    if l_idx[i] < 0:
                        ridx = r_idx[i]
                        for j in range(c): d_dst[j * n + i] = d_src[j * rr + ridx]
            else:
                for i in range(n):
                    if l_idx[i] < 0: d_dst[i] = d_src[r_idx[i]]

    elif dtype == TYPE_DATE or dtype == TYPE_TIME:
        i32_dst = <int32_t*>((<DateColumn>res_col).data_c)
        i32_src = <int32_t*>((<DateColumn>r_col).data_c)
        with nogil:
            if is2d and c > 1:
                for i in range(n):
                    if l_idx[i] < 0:
                        ridx = r_idx[i]
                        for j in range(c): i32_dst[j * n + i] = i32_src[j * rr + ridx]
            else:
                for i in range(n):
                    if l_idx[i] < 0: i32_dst[i] = i32_src[r_idx[i]]

    elif dtype == TYPE_DATETIME:
        i64_dst = (<DateTimeColumn>res_col).data_c
        i64_src = (<DateTimeColumn>r_col).data_c
        with nogil:
            if is2d and c > 1:
                for i in range(n):
                    if l_idx[i] < 0:
                        ridx = r_idx[i]
                        for j in range(c): i64_dst[j * n + i] = i64_src[j * rr + ridx]
            else:
                for i in range(n):
                    if l_idx[i] < 0: i64_dst[i] = i64_src[r_idx[i]]

    elif dtype == TYPE_BOOL:
        b_dst = (<BoolColumn>res_col).data_c
        b_src = (<BoolColumn>r_col).data_c
        with nogil:
            for i in range(n):
                if l_idx[i] < 0: b_dst[i] = b_src[r_idx[i]]

    # 2. 【核心漏洞修复】：分类列的独立重编码
    elif dtype == TYPE_CATEGORY:
        cat_res = <CategoryColumn>res_col
        cat_r = <CategoryColumn>r_col
        for i in range(n):
            if l_idx[i] < 0:
                cat_res.codes_c[i] = cat_res._get_or_add_code(str(cat_r[r_idx[i]]))

cdef BaseColumn _join_gather_str(BaseColumn l_col, BaseColumn r_col,
                                 Py_ssize_t* l_idx, Py_ssize_t* r_idx, Py_ssize_t n):
    """字符串 join 列双源 gather：完美防止 OOM 和频繁扩容。"""
    cdef Py_ssize_t i
    cdef AsciiColumn jc_a
    cdef UCS2Column jc_u

    if l_col.dtype == TYPE_CHAR256:
        jc_a = AsciiColumn(max(n, 1))
        jc_a._reserve_bytes((<AsciiColumn>l_col).byte_length + (<AsciiColumn>r_col).byte_length)
        for i in range(n):
            if l_idx[i] >= 0:
                jc_a.append((<AsciiColumn>l_col)[l_idx[i]])
            else:
                jc_a.append((<AsciiColumn>r_col)[r_idx[i]])
        return jc_a

    elif l_col.dtype == TYPE_UCS2:
        jc_u = UCS2Column(max(n, 1))
        jc_u._reserve_chars((<UCS2Column>l_col).char_length + (<UCS2Column>r_col).char_length)
        for i in range(n):
            if l_idx[i] >= 0:
                jc_u.append((<UCS2Column>l_col)[l_idx[i]])
            else:
                jc_u.append((<UCS2Column>r_col)[r_idx[i]])
        return jc_u

cdef _merge_impl(FastDataFrame self, FastDataFrame other, list on, str how='inner'):
    cdef list join_cols
    if on is not None:
        join_cols = on if isinstance(on, list) else [on]
        for jc in join_cols:
            if jc not in self._columns: raise KeyError(f"Column '{jc}' not found in left DataFrame")
            if jc not in other._columns: raise KeyError(f"Column '{jc}' not found in right DataFrame")
    else:
        common = set(self._columns.keys()) & set(other._columns.keys())
        if not common: raise ValueError("No common columns to merge on")
        join_cols = sorted(common)

    cdef Py_ssize_t num_join = len(join_cols)
    cdef Py_ssize_t i, j, c

    # =========================================================
    # 批处理提取 Join 键 (O(1) 替代 O(N*C))
    # =========================================================
    cdef list l_cols_data = [_col_to_keys(self._columns[jc]) for jc in join_cols]
    cdef list r_cols_data = [_col_to_keys(other._columns[jc]) for jc in join_cols]

    cdef list left_keys = [None] * self.df_length
    cdef list right_keys = [None] * other.df_length

    if num_join == 1:
        left_keys = l_cols_data[0]
        right_keys = r_cols_data[0]
    else:
        for i in range(self.df_length):
            left_keys[i] = tuple(l_cols_data[c][i] for c in range(num_join))
        for j in range(other.df_length):
            right_keys[j] = tuple(r_cols_data[c][j] for c in range(num_join))

    # =========================================================
    # 高速哈希表构建
    # =========================================================
    cdef dict left_map = {}
    cdef dict right_map = {}
    cdef object key
    cdef list l_list, r_list

    for i in range(self.df_length):
        key = left_keys[i]
        l_list = left_map.get(key)
        if l_list is None:
            left_map[key] = [i]
        else:
            l_list.append(i)

    for j in range(other.df_length):
        key = right_keys[j]
        r_list = right_map.get(key)
        if r_list is None:
            right_map[key] = [j]
        else:
            r_list.append(j)

    # =========================================================
    # 行索引推导 (布尔化简)
    # =========================================================
    cdef list left_indices = []
    cdef list right_indices = []
    cdef bint keep_left = (how in ('left', 'outer'))
    cdef bint keep_right = (how in ('right', 'outer'))
    cdef list all_keys

    if how == 'outer':
        all_keys = list(set(left_map.keys()) | set(right_map.keys()))
    elif how == 'right':
        all_keys = list(right_map.keys())
    else:
        all_keys = list(left_map.keys())

    for key in all_keys:
        lefts = left_map.get(key, [])
        rights = right_map.get(key, [])
        if lefts and rights:
            for li in lefts:
                for ri in rights:
                    left_indices.append(li)
                    right_indices.append(ri)
        elif lefts and keep_left:
            for li in lefts:
                left_indices.append(li)
                right_indices.append(-1)
        elif rights and keep_right:
            for ri in rights:
                left_indices.append(-1)
                right_indices.append(ri)

    # =========================================================
    # 安全指针分配与底层数据拷贝
    # =========================================================
    cdef Py_ssize_t result_len = len(left_indices)
    cdef FastDataFrame result = FastDataFrame()
    result.df_length = result_len
    result.index = RangeIndex(0, result_len)

    cdef set join_set = set(join_cols)
    cdef list left_col_names = [n for n in self._columns.keys() if n not in join_set]
    cdef list right_col_names = [n for n in other._columns.keys() if n not in join_set]

    cdef Py_ssize_t* left_idx_c = <Py_ssize_t*>aligned_malloc(max(result_len, 1) * sizeof(Py_ssize_t))
    cdef Py_ssize_t* right_idx_c = <Py_ssize_t*>aligned_malloc(max(result_len, 1) * sizeof(Py_ssize_t))
    if not left_idx_c or not right_idx_c:
        if left_idx_c: os_aligned_free(left_idx_c)
        if right_idx_c: os_aligned_free(right_idx_c)
        raise MemoryError()

    cdef BaseColumn src_col
    cdef BaseColumn jc_col

    try:
        # 1. 数组赋值 (消灭内层寻址消耗)
        for i in range(result_len):
            left_idx_c[i] = left_indices[i]
            right_idx_c[i] = right_indices[i]

        # 2. 列数据聚合搬运
        for cn in join_cols:
            src_col = self._columns[cn]
            if src_col.dtype == TYPE_CHAR256 or src_col.dtype == TYPE_UCS2:
                jc_col = _join_gather_str(src_col, other._columns[cn], left_idx_c, right_idx_c, result_len)
            else:
                jc_col = _gather_or_reorder(self, src_col, left_idx_c, result_len)
                _join_col_fixup(jc_col, other._columns[cn], left_idx_c, right_idx_c, result_len)
            result._columns[cn] = jc_col

        for cn in left_col_names:
            src_col = self._columns[cn]
            out_name = cn + '_x' if cn in right_col_names else cn
            result._columns[out_name] = _gather_or_reorder(self, src_col, left_idx_c, result_len)

        for cn in right_col_names:
            src_col = other._columns[cn]
            out_name = cn + '_y' if cn in result._columns or cn in left_col_names else cn
            result._columns[out_name] = _gather_or_reorder(other, src_col, right_idx_c, result_len)

        return result
    finally:
        os_aligned_free(left_idx_c)
        os_aligned_free(right_idx_c)

cdef _duplicated_impl(FastDataFrame self, subset=None, str keep='first'):
    cdef Py_ssize_t n = self.df_length
    cdef IntColumn result = IntColumn(max(n, 1))
    result.length = n
    if n == 0:
        return result

    if subset is None:
        subset = list(self._columns.keys())

    cdef dict seen = {}
    cdef Py_ssize_t i
    cdef tuple row_key
    cdef BaseColumn col

    if keep == 'first':
        for i in range(n):
            row_key = tuple(_col_row_key(self._columns[c], i) for c in subset)
            if row_key in seen:
                result.data_c[i] = 1
            else:
                seen[row_key] = i
                result.data_c[i] = 0
    elif keep == 'last':
        for i in range(n):
            row_key = tuple(_col_row_key(self._columns[c], i) for c in subset)
            if row_key in seen:
                result.data_c[seen[row_key]] = 1
            seen[row_key] = i
        for i in range(n):
            row_key = tuple(_col_row_key(self._columns[c], i) for c in subset)
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
        self._columns = result._columns
        self.df_length = result.df_length
        self.index = result.index
        return self
    return result
