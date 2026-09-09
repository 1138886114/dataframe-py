# _fastdataframe_metric.pxi — Correlation, covariance, distance (module-level cdef)
# Included at module level in fastdataframe.pyx (after class definition)

cdef list _numeric_logical_features(FastDataFrame self):
    """展开 2D 数值列为逻辑子特征，返回 [(key, col, sub)]。

    1D 列 → (name, col, 0)；2D 列（列优先 data_c[sub*rows + i]）→
    每个子列一个特征 (name_k, col, k)。corr/cov/pearson 依赖此展开。
    """
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

cdef _euclidean_dist_withcols_impl(FastDataFrame self, col1, col2):
    cdef str name1 = col1 if isinstance(col1, str) else None
    cdef str name2 = col2 if isinstance(col2, str) else None
    cdef DoubleColumn c1, c2
    c1 = <DoubleColumn>self.columns[name1] if name1 is not None else <DoubleColumn>col1
    c2 = <DoubleColumn>self.columns[name2] if name2 is not None else <DoubleColumn>col2
    return c1.euclidean_dist(c2)

cdef _manhattan_dist_withcols_impl(FastDataFrame self, col1, col2):
    cdef str name1 = col1 if isinstance(col1, str) else None
    cdef str name2 = col2 if isinstance(col2, str) else None
    cdef DoubleColumn c1, c2
    c1 = <DoubleColumn>self.columns[name1] if name1 is not None else <DoubleColumn>col1
    c2 = <DoubleColumn>self.columns[name2] if name2 is not None else <DoubleColumn>col2
    return c1.manhattan_dist(c2)

cdef _cosine_dist_withcols_impl(FastDataFrame self, col1, col2):
    cdef str name1 = col1 if isinstance(col1, str) else None
    cdef str name2 = col2 if isinstance(col2, str) else None
    cdef DoubleColumn c1, c2
    c1 = <DoubleColumn>self.columns[name1] if name1 is not None else <DoubleColumn>col1
    c2 = <DoubleColumn>self.columns[name2] if name2 is not None else <DoubleColumn>col2
    return c1.cosine_dist(c2)

cdef _cosine_similarity_withcols_impl(FastDataFrame self, col1, col2):
    cdef str name1 = col1 if isinstance(col1, str) else None
    cdef str name2 = col2 if isinstance(col2, str) else None
    cdef DoubleColumn c1, c2
    c1 = <DoubleColumn>self.columns[name1] if name1 is not None else <DoubleColumn>col1
    c2 = <DoubleColumn>self.columns[name2] if name2 is not None else <DoubleColumn>col2
    return c1.cosine_similarity(c2)

cdef _jaccard_sim_withcols_impl(FastDataFrame self, col1, col2):
    cdef str name1 = col1 if isinstance(col1, str) else None
    cdef str name2 = col2 if isinstance(col2, str) else None
    cdef DoubleColumn c1, c2
    c1 = <DoubleColumn>self.columns[name1] if name1 is not None else <DoubleColumn>col1
    c2 = <DoubleColumn>self.columns[name2] if name2 is not None else <DoubleColumn>col2
    return c1.jaccard_sim(c2)

cdef _hamming_dist_withcols_impl(FastDataFrame self, col1, col2):
    cdef str name1 = col1 if isinstance(col1, str) else None
    cdef str name2 = col2 if isinstance(col2, str) else None
    cdef DoubleColumn c1, c2
    c1 = <DoubleColumn>self.columns[name1] if name1 is not None else <DoubleColumn>col1
    c2 = <DoubleColumn>self.columns[name2] if name2 is not None else <DoubleColumn>col2
    return c1.hamming_dist(c2)

cdef _corr_impl(FastDataFrame self):
    cdef list feats = _numeric_logical_features(self)
    cdef Py_ssize_t n_feats = len(feats)
    cdef Py_ssize_t n_rows = self.df_length
    if n_feats == 0 or n_rows == 0:
        return FastDataFrame()

    cdef double** data = <double**>aligned_malloc(max(n_feats, 1) * sizeof(double*))
    if not data: raise MemoryError()
    cdef Py_ssize_t i, j
    for i in range(n_feats): data[i] = NULL

    cdef DoubleColumn c_i, c_j
    cdef double val
    cdef tuple feat
    cdef FastDataFrame result

    try:
        for i in range(n_feats):
            data[i] = <double*>aligned_malloc(max(n_rows, 1) * sizeof(double))
            if not data[i]: raise MemoryError()
            feat = feats[i]
            self._col_to_double_buffer(<BaseColumn>feat[1], data[i], n_rows, <Py_ssize_t>feat[2])

        result = FastDataFrame()
        result.df_length = n_feats
        result.index = RangeIndex(0, n_feats)
        for i in range(n_feats):
            c_i = DoubleColumn(max(n_feats, 1))
            c_i.length = n_feats
            result.columns[feats[i][0]] = c_i

        for i in range(n_feats):
            c_i = <DoubleColumn>result.columns[feats[i][0]]
            c_i.data_c[i] = 1.0
            for j in range(i + 1, n_feats):
                c_j = <DoubleColumn>result.columns[feats[j][0]]
                val = _vec_pearson_corr(data[i], data[j], n_rows)
                c_i.data_c[j] = val
                c_j.data_c[i] = val
        return result
    finally:
        for i in range(n_feats):
            if data[i] != NULL: os_aligned_free(data[i])
        os_aligned_free(data)

cdef _cov_impl(FastDataFrame self):
    cdef list feats = _numeric_logical_features(self)
    cdef Py_ssize_t n_feats = len(feats)
    cdef Py_ssize_t n_rows = self.df_length
    if n_feats == 0 or n_rows == 0:
        return FastDataFrame()

    cdef double** data = <double**>aligned_malloc(max(n_feats, 1) * sizeof(double*))
    if not data: raise MemoryError()
    cdef Py_ssize_t i, j
    for i in range(n_feats): data[i] = NULL

    cdef DoubleColumn c_i, c_j
    cdef double val
    cdef tuple feat
    cdef FastDataFrame result

    try:
        for i in range(n_feats):
            data[i] = <double*>aligned_malloc(max(n_rows, 1) * sizeof(double))
            if not data[i]: raise MemoryError()
            feat = feats[i]
            self._col_to_double_buffer(<BaseColumn>feat[1], data[i], n_rows, <Py_ssize_t>feat[2])

        result = FastDataFrame()
        result.df_length = n_feats
        result.index = RangeIndex(0, n_feats)
        for i in range(n_feats):
            c_i = DoubleColumn(max(n_feats, 1))
            c_i.length = n_feats
            result.columns[feats[i][0]] = c_i

        for i in range(n_feats):
            c_i = <DoubleColumn>result.columns[feats[i][0]]
            c_i.data_c[i] = _vec_pearson_cov(data[i], data[i], n_rows)
            for j in range(i + 1, n_feats):
                c_j = <DoubleColumn>result.columns[feats[j][0]]
                val = _vec_pearson_cov(data[i], data[j], n_rows)
                c_i.data_c[j] = val
                c_j.data_c[i] = val
        return result
    finally:
        for i in range(n_feats):
            if data[i] != NULL: os_aligned_free(data[i])
        os_aligned_free(data)

cdef _corrwith_impl(FastDataFrame self, other):
    cdef dict result = {}
    cdef tuple feat
    cdef Py_ssize_t n = self.df_length
    cdef DoubleColumn temp, other_buf
    if n == 0: return result

    other_buf = DoubleColumn(max(n, 1))
    other_buf.length = n
    self._col_to_double_buffer(other, other_buf.data_c, n, 0)

    cdef list feats = _numeric_logical_features(self)
    for feat in feats:
        temp = DoubleColumn(max(n, 1))
        temp.length = n
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

cdef _pearson_sim_impl(FastDataFrame self):
    cdef list feats = _numeric_logical_features(self)
    cdef Py_ssize_t n_feats = len(feats)
    cdef list num_names = [f[0] for f in feats]
    cdef str name
    cdef BaseColumn col
    cdef Py_ssize_t n_rows, i, j
    cdef double* buf_i = NULL
    cdef double* buf_j = NULL
    cdef DoubleColumn result_col
    cdef FastDataFrame result
    cdef tuple feat

    if n_feats < 2:
        raise ValueError("Need at least 2 numeric columns")
    n_rows = self.df_length

    result = FastDataFrame()
    result.df_length = n_feats
    result.index = RangeIndex(0, n_feats)

    buf_i = <double*>aligned_malloc(max(n_rows, 1) * sizeof(double))
    buf_j = <double*>aligned_malloc(max(n_rows, 1) * sizeof(double))
    if not buf_i or not buf_j:
        if buf_i: os_aligned_free(buf_i)
        if buf_j: os_aligned_free(buf_j)
        raise MemoryError()

    try:
        for i in range(n_feats):
            result_col = DoubleColumn(max(n_feats, 1))
            result_col.length = n_feats
            feat = feats[i]
            self._col_to_double_buffer(<BaseColumn>feat[1], buf_i, n_rows, <Py_ssize_t>feat[2])
            for j in range(n_feats):
                if i == j:
                    result_col.data_c[j] = 1.0
                else:
                    feat = feats[j]
                    self._col_to_double_buffer(<BaseColumn>feat[1], buf_j, n_rows, <Py_ssize_t>feat[2])
                    result_col.data_c[j] = _vec_pearson_corr(buf_i, buf_j, n_rows)
            result.columns[num_names[i]] = result_col
        return result
    finally:
        os_aligned_free(buf_i)
        os_aligned_free(buf_j)
