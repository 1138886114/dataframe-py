# _fastdataframe_metric.pxi — Correlation, covariance, distance (module-level cdef)
# Included at module level in fastdataframe.pyx (after class definition)

cdef list _numeric_logical_features(FastDataFrame self):
    """展开 2D 数值列为逻辑子特征，返回 [(key, col, sub)]。"""
    cdef list feats = []
    cdef str name
    cdef BaseColumn col
    cdef bint is2d
    cdef Py_ssize_t c, r, k
    for name, col in self.columns.items():
        if col.dtype == TYPE_INT or col.dtype == TYPE_DOUBLE:
            is2d, c, r = _col_shape_probe(col)
            if is2d and c > 1:
                for k in range(c):
                    feats.append((f"{name}_{k}", col, k))
            else:
                feats.append((name, col, 0))
    return feats


# ============================================================
# 安全保护助手：确保取出的必定是一维 DoubleColumn
# ============================================================
cdef inline DoubleColumn _ensure_double_1d(FastDataFrame self, object col_ident):
    """防止 IntColumn 被暴力转为 DoubleColumn 导致 SegFault"""
    cdef str name = col_ident if isinstance(col_ident, str) else None
    cdef BaseColumn col = self.columns[name] if name is not None else <BaseColumn>col_ident

    if col.dtype == TYPE_DOUBLE and not getattr(col, 'is_2d', False):
        return <DoubleColumn>col

    # 需要转换（Int转Double，或者从 2D 子列提取出独立的 1D Double）
    cdef Py_ssize_t n = self.df_length
    cdef DoubleColumn buf = DoubleColumn(max(n, 1))
    buf.length = n
    self._col_to_double_buffer(col, buf.data_c, n, 0)
    return buf


# ============================================================
# Distance Metrics
# ============================================================

cdef _euclidean_dist_withcols_impl(FastDataFrame self, col1, col2):
    cdef DoubleColumn c1 = _ensure_double_1d(self, col1)
    cdef DoubleColumn c2 = _ensure_double_1d(self, col2)
    return c1.euclidean_dist(c2)

cdef _manhattan_dist_withcols_impl(FastDataFrame self, col1, col2):
    cdef DoubleColumn c1 = _ensure_double_1d(self, col1)
    cdef DoubleColumn c2 = _ensure_double_1d(self, col2)
    return c1.manhattan_dist(c2)

cdef _cosine_dist_withcols_impl(FastDataFrame self, col1, col2):
    cdef DoubleColumn c1 = _ensure_double_1d(self, col1)
    cdef DoubleColumn c2 = _ensure_double_1d(self, col2)
    return c1.cosine_dist(c2)

cdef _cosine_similarity_withcols_impl(FastDataFrame self, col1, col2):
    cdef DoubleColumn c1 = _ensure_double_1d(self, col1)
    cdef DoubleColumn c2 = _ensure_double_1d(self, col2)
    return c1.cosine_similarity(c2)

cdef _jaccard_sim_withcols_impl(FastDataFrame self, col1, col2):
    cdef DoubleColumn c1 = _ensure_double_1d(self, col1)
    cdef DoubleColumn c2 = _ensure_double_1d(self, col2)
    return c1.jaccard_sim(c2)

cdef _hamming_dist_withcols_impl(FastDataFrame self, col1, col2):
    cdef DoubleColumn c1 = _ensure_double_1d(self, col1)
    cdef DoubleColumn c2 = _ensure_double_1d(self, col2)
    return c1.hamming_dist(c2)

cdef _corr_impl(FastDataFrame self):
    cdef list feats = _numeric_logical_features(self)
    cdef Py_ssize_t n_feats = len(feats)
    cdef Py_ssize_t n_rows = self.df_length
    if n_feats == 0 or n_rows == 0:
        return FastDataFrame()

    # 仅用一次 malloc 分配 N 个特征的全部连续大内存 (列优先结构)
    cdef double* data_matrix = <double*>aligned_malloc(n_feats * max(n_rows, 1) * sizeof(double))
    if not data_matrix:
        raise MemoryError()

    cdef Py_ssize_t i, j
    cdef double val
    cdef tuple feat

    try:
        # 1. 一次性安全萃取所有特征
        for i in range(n_feats):
            feat = feats[i]
            # 数据步长正好是 n_rows，完全 F-contiguous 连续，对缓存极其友好
            self._col_to_double_buffer(<BaseColumn>feat[1], data_matrix + i * n_rows, n_rows, <Py_ssize_t>feat[2])

        # 申请统一的 2D 连续块作为结果矩阵
        cdef DoubleColumn res_block = DoubleColumn(n_feats * n_feats, cols=n_feats)
        res_block.length = n_feats * n_feats
        cdef double* res_ptr = res_block.data_c

        # 释放 GIL 并在行优先内存地址上并行填充相关性矩阵
        with nogil:
            for i in range(n_feats):
                res_ptr[i * n_feats + i] = 1.0
                for j in range(i + 1, n_feats):
                    val = _vec_pearson_corr(data_matrix + i * n_rows, data_matrix + j * n_rows, n_rows)
                    res_ptr[j * n_feats + i] = val
                    res_ptr[i * n_feats + j] = val

        # 极速装载 View (零拷贝)
        cdef FastDataFrame result = FastDataFrame()
        result.df_length = n_feats
        result.index = RangeIndex(0, n_feats)
        for i in range(n_feats):
            result.columns[feats[i][0]] = DoubleColumn.create_view(res_ptr + i * n_feats, n_feats, res_block)

        if not hasattr(result, '_2d_blocks'):
            result._2d_blocks = []
        result._2d_blocks.append(res_block)

        return result
    finally:
        os_aligned_free(data_matrix)

cdef _cov_impl(FastDataFrame self):
    cdef list feats = _numeric_logical_features(self)
    cdef Py_ssize_t n_feats = len(feats)
    cdef Py_ssize_t n_rows = self.df_length
    if n_feats == 0 or n_rows == 0:
        return FastDataFrame()

    cdef double* data_matrix = <double*>aligned_malloc(n_feats * max(n_rows, 1) * sizeof(double))
    if not data_matrix: raise MemoryError()

    cdef Py_ssize_t i, j
    cdef double val
    cdef tuple feat

    try:
        for i in range(n_feats):
            feat = feats[i]
            self._col_to_double_buffer(<BaseColumn>feat[1], data_matrix + i * n_rows, n_rows, <Py_ssize_t>feat[2])

        cdef DoubleColumn res_block = DoubleColumn(n_feats * n_feats, cols=n_feats)
        res_block.length = n_feats * n_feats
        cdef double* res_ptr = res_block.data_c

        with nogil:
            for i in range(n_feats):
                res_ptr[i * n_feats + i] = _vec_pearson_cov(data_matrix + i * n_rows, data_matrix + i * n_rows, n_rows)
                for j in range(i + 1, n_feats):
                    val = _vec_pearson_cov(data_matrix + i * n_rows, data_matrix + j * n_rows, n_rows)
                    res_ptr[j * n_feats + i] = val
                    res_ptr[i * n_feats + j] = val

        cdef FastDataFrame result = FastDataFrame()
        result.df_length = n_feats
        result.index = RangeIndex(0, n_feats)
        for i in range(n_feats):
            result.columns[feats[i][0]] = DoubleColumn.create_view(res_ptr + i * n_feats, n_feats, res_block)

        if not hasattr(result, '_2d_blocks'):
            result._2d_blocks = []
        result._2d_blocks.append(res_block)

        return result
    finally:
        os_aligned_free(data_matrix)


cdef _pearson_sim_impl(FastDataFrame self):
    """
    数学上，列之间的 Pearson 相似度矩阵 等价于 Pearson 相关系数矩阵。
    直接复用 _corr_impl 的实现，避免了原先实现中 O(N^2) 的内存拷贝和重复分配灾难。
    """
    return _corr_impl(self)


# ============================================================
# Two-Column Compare Metircs
# ============================================================

cdef _corrwith_impl(FastDataFrame self, other):
    cdef dict result = {}
    cdef tuple feat
    cdef Py_ssize_t n = self.df_length
    if n == 0: return result

    cdef DoubleColumn other_buf = DoubleColumn(max(n, 1))
    other_buf.length = n
    self._col_to_double_buffer(other, other_buf.data_c, n, 0)

    cdef DoubleColumn temp = DoubleColumn(max(n, 1))
    temp.length = n

    cdef list feats = _numeric_logical_features(self)
    for feat in feats:
        self._col_to_double_buffer(<BaseColumn>feat[1], temp.data_c, n, <Py_ssize_t>feat[2])
        result[feat[0]] = _vec_pearson_corr(temp.data_c, other_buf.data_c, n)

    return result

cdef _corrcoef_impl(FastDataFrame self, col_a, col_b):
    cdef bint is2d
    cdef Py_ssize_t c_a, r_a, c_b, r_b
    is2d, c_a, r_a = _col_shape_probe(col_a)
    is2d, c_b, r_b = _col_shape_probe(col_b)

    cdef Py_ssize_t n = min(r_a, r_b)
    if n < 2: return NAN

    # 注意这里原代码写得很完美，它安全地使用了 double_buffer 避免了类型强转错误
    cdef DoubleColumn buf_a = DoubleColumn(max(n, 1))
    cdef DoubleColumn buf_b = DoubleColumn(max(n, 1))
    buf_a.length = n
    buf_b.length = n
    self._col_to_double_buffer(col_a, buf_a.data_c, n, 0)
    self._col_to_double_buffer(col_b, buf_b.data_c, n, 0)

    return _vec_pearson_corr(buf_a.data_c, buf_b.data_c, n)

cdef _cov_withcols_impl(FastDataFrame self, col_a, col_b):
    cdef bint is2d
    cdef Py_ssize_t c_a, r_a, c_b, r_b
    is2d, c_a, r_a = _col_shape_probe(col_a)
    is2d, c_b, r_b = _col_shape_probe(col_b)

    cdef Py_ssize_t n = min(r_a, r_b)
    if n < 2: return NAN

    cdef DoubleColumn buf_a = DoubleColumn(max(n, 1))
    cdef DoubleColumn buf_b = DoubleColumn(max(n, 1))
    buf_a.length = n
    buf_b.length = n
    self._col_to_double_buffer(col_a, buf_a.data_c, n, 0)
    self._col_to_double_buffer(col_b, buf_b.data_c, n, 0)

    return _vec_pearson_cov(buf_a.data_c, buf_b.data_c, n)