# _fastdataframe_metric.pxi — Correlation, covariance, distance (module-level cdef)
# Included at module level in fastdataframe.pyx (after class definition)

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
    cdef list num_cols = []
    for name, col in self.columns.items():
        if col.dtype == TYPE_INT or col.dtype == TYPE_DOUBLE:
            num_cols.append(name)

    cdef Py_ssize_t n_cols = len(num_cols)
    cdef Py_ssize_t n_rows = self.df_length
    if n_cols == 0 or n_rows == 0:
        return FastDataFrame()

    cdef double** data = <double**>aligned_malloc(max(n_cols, 1) * sizeof(double*))
    if not data: raise MemoryError()
    cdef Py_ssize_t i, j
    for i in range(n_cols): data[i] = NULL

    cdef DoubleColumn c_i, c_j
    cdef double val
    cdef FastDataFrame result

    try:
        for i in range(n_cols):
            data[i] = <double*>aligned_malloc(max(n_rows, 1) * sizeof(double))
            if not data[i]: raise MemoryError()
            self._col_to_double_buffer(self.columns[num_cols[i]], data[i], n_rows)

        result = FastDataFrame()
        result.df_length = n_cols
        result.index = RangeIndex(0, n_cols)
        for i in range(n_cols):
            c_i = DoubleColumn(max(n_cols, 1))
            c_i.length = n_cols
            result.columns[num_cols[i]] = c_i

        for i in range(n_cols):
            c_i = <DoubleColumn>result.columns[num_cols[i]]
            c_i.data_c[i] = 1.0
            for j in range(i + 1, n_cols):
                c_j = <DoubleColumn>result.columns[num_cols[j]]
                val = _vec_pearson_corr(data[i], data[j], n_rows)
                c_i.data_c[j] = val
                c_j.data_c[i] = val
        return result
    finally:
        for i in range(n_cols):
            if data[i] != NULL: os_aligned_free(data[i])
        os_aligned_free(data)

cdef _cov_impl(FastDataFrame self):
    cdef list num_cols = []
    for name, col in self.columns.items():
        if col.dtype == TYPE_INT or col.dtype == TYPE_DOUBLE:
            num_cols.append(name)

    cdef Py_ssize_t n_cols = len(num_cols)
    cdef Py_ssize_t n_rows = self.df_length
    if n_cols == 0 or n_rows == 0:
        return FastDataFrame()

    cdef double** data = <double**>aligned_malloc(max(n_cols, 1) * sizeof(double*))
    if not data: raise MemoryError()
    cdef Py_ssize_t i, j
    for i in range(n_cols): data[i] = NULL

    cdef DoubleColumn c_i, c_j
    cdef double val
    cdef FastDataFrame result

    try:
        for i in range(n_cols):
            data[i] = <double*>aligned_malloc(max(n_rows, 1) * sizeof(double))
            if not data[i]: raise MemoryError()
            self._col_to_double_buffer(self.columns[num_cols[i]], data[i], n_rows)

        result = FastDataFrame()
        result.df_length = n_cols
        result.index = RangeIndex(0, n_cols)
        for i in range(n_cols):
            c_i = DoubleColumn(max(n_cols, 1))
            c_i.length = n_cols
            result.columns[num_cols[i]] = c_i

        for i in range(n_cols):
            c_i = <DoubleColumn>result.columns[num_cols[i]]
            c_i.data_c[i] = _vec_pearson_cov(data[i], data[i], n_rows)
            for j in range(i + 1, n_cols):
                c_j = <DoubleColumn>result.columns[num_cols[j]]
                val = _vec_pearson_cov(data[i], data[j], n_rows)
                c_i.data_c[j] = val
                c_j.data_c[i] = val
        return result
    finally:
        for i in range(n_cols):
            if data[i] != NULL: os_aligned_free(data[i])
        os_aligned_free(data)

cdef _corrwith_impl(FastDataFrame self, other):
    cdef dict result = {}
    cdef str name
    cdef BaseColumn col
    cdef Py_ssize_t n = self.df_length
    cdef DoubleColumn temp, other_buf
    if n == 0: return result

    other_buf = DoubleColumn(max(n, 1))
    other_buf.length = n
    self._col_to_double_buffer(other, other_buf.data_c, n)

    for name, col in self.columns.items():
        if col.dtype == TYPE_INT or col.dtype == TYPE_DOUBLE:
            temp = DoubleColumn(max(n, 1))
            temp.length = n
            self._col_to_double_buffer(col, temp.data_c, n)
            result[name] = _vec_pearson_corr(temp.data_c, other_buf.data_c, n)
    return result

cdef _corrcoef_impl(FastDataFrame self, col_a, col_b):
    cdef Py_ssize_t n = min(col_a.length, col_b.length)
    if n < 2: return NAN
    cdef DoubleColumn buf_a = DoubleColumn(max(n, 1))
    cdef DoubleColumn buf_b = DoubleColumn(max(n, 1))
    buf_a.length = n
    buf_b.length = n
    self._col_to_double_buffer(col_a, buf_a.data_c, n)
    self._col_to_double_buffer(col_b, buf_b.data_c, n)
    return _vec_pearson_corr(buf_a.data_c, buf_b.data_c, n)

cdef _cov_withcols_impl(FastDataFrame self, col_a, col_b):
    cdef Py_ssize_t n = min(col_a.length, col_b.length)
    if n < 2: return NAN
    cdef DoubleColumn buf_a = DoubleColumn(max(n, 1))
    cdef DoubleColumn buf_b = DoubleColumn(max(n, 1))
    buf_a.length = n
    buf_b.length = n
    self._col_to_double_buffer(col_a, buf_a.data_c, n)
    self._col_to_double_buffer(col_b, buf_b.data_c, n)
    return _vec_pearson_cov(buf_a.data_c, buf_b.data_c, n)

cdef _pearson_sim_impl(FastDataFrame self):
    cdef list num_names = []
    cdef list num_cols = []
    cdef str name
    cdef BaseColumn col
    cdef Py_ssize_t n_cols, n_rows, i, j
    cdef double* buf_i = NULL
    cdef double* buf_j = NULL
    cdef DoubleColumn result_col
    cdef FastDataFrame result

    for name, col in self.columns.items():
        if col.dtype == TYPE_INT or col.dtype == TYPE_DOUBLE:
            num_names.append(name)
            num_cols.append(col)

    n_cols = len(num_names)
    if n_cols < 2:
        raise ValueError("Need at least 2 numeric columns")
    n_rows = self.df_length

    result = FastDataFrame()
    result.df_length = n_cols
    result.index = RangeIndex(0, n_cols)

    buf_i = <double*>aligned_malloc(max(n_rows, 1) * sizeof(double))
    buf_j = <double*>aligned_malloc(max(n_rows, 1) * sizeof(double))
    if not buf_i or not buf_j:
        if buf_i: os_aligned_free(buf_i)
        if buf_j: os_aligned_free(buf_j)
        raise MemoryError()

    try:
        for i in range(n_cols):
            result_col = DoubleColumn(max(n_cols, 1))
            result_col.length = n_cols
            self._col_to_double_buffer(<BaseColumn>num_cols[i], buf_i, n_rows)
            for j in range(n_cols):
                if i == j:
                    result_col.data_c[j] = 1.0
                else:
                    self._col_to_double_buffer(<BaseColumn>num_cols[j], buf_j, n_rows)
                    result_col.data_c[j] = _vec_pearson_corr(buf_i, buf_j, n_rows)
            result.columns[num_names[i]] = result_col
        return result
    finally:
        os_aligned_free(buf_i)
        os_aligned_free(buf_j)
