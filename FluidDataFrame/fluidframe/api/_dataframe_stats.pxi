# _fastdataframe_stats.pxi — rank, cut, qcut operations (module-level cdef)
# Included at module level in fastdataframe.pyx (after class definition)

from libc.stdlib cimport qsort

cdef struct _RankItem:
    Py_ssize_t idx
    double val

cdef int _rank_cmp_asc(const void* a, const void* b) noexcept nogil:
    cdef double va = (<_RankItem*>a).val
    cdef double vb = (<_RankItem*>b).val
    if va < vb: return -1
    if va > vb: return 1
    return 0

cdef int _rank_cmp_desc(const void* a, const void* b) noexcept nogil:
    cdef double va = (<_RankItem*>a).val
    cdef double vb = (<_RankItem*>b).val
    if va > vb: return -1
    if va < vb: return 1
    return 0

cdef _rank_col(NumericType* data, Py_ssize_t n, str method, bint ascending):
    cdef DoubleColumn res = DoubleColumn(max(n, 1))
    res.length = n
    if n == 0:
        return res

    cdef _RankItem* items = <_RankItem*>aligned_malloc(max(n, 1) * sizeof(_RankItem))
    if not items: raise MemoryError()
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t dense_rank_int
    cdef double avg_rank

    try:
        for i in range(n):
            items[i].idx = i
            items[i].val = <double>data[i]

        if ascending:
            qsort(items, n, sizeof(_RankItem), _rank_cmp_asc)
        else:
            qsort(items, n, sizeof(_RankItem), _rank_cmp_desc)

        if method == 'average':
            i = 0
            while i < n:
                j = i
                while j < n and items[j].val == items[i].val:
                    j += 1
                avg_rank = <double>(i + j + 1) / 2.0
                for k in range(i, j):
                    res.data_c[items[k].idx] = avg_rank
                i = j
        elif method == 'min':
            i = 0
            while i < n:
                j = i
                while j < n and items[j].val == items[i].val:
                    j += 1
                for k in range(i, j):
                    res.data_c[items[k].idx] = <double>(i + 1)
                i = j
        elif method == 'max':
            i = 0
            while i < n:
                j = i
                while j < n and items[j].val == items[i].val:
                    j += 1
                for k in range(i, j):
                    res.data_c[items[k].idx] = <double>(j)
                i = j
        elif method == 'first':
            for i in range(n):
                res.data_c[items[i].idx] = <double>(i + 1)
        elif method == 'dense':
            i = 0
            dense_rank_int = 1
            while i < n:
                j = i
                while j < n and items[j].val == items[i].val:
                    j += 1
                for k in range(i, j):
                    res.data_c[items[k].idx] = <double>dense_rank_int
                dense_rank_int += 1
                i = j
        else:
            raise ValueError(f"Unknown ranking method: {method}")
    finally:
        os_aligned_free(items)
    return res

cdef _rank_impl(DataFrame self, str col_name, str method, bint ascending, str na_option):
    if col_name not in self._columns:
        raise KeyError(f"Column '{col_name}' not found")
    cdef BaseColumn col = self._columns[col_name]
    cdef Py_ssize_t n = self.df_length
    if _col_is_2d(col):
        return _rank_row_keys(col, n, method, ascending)
    if col.dtype == TYPE_INT:
        return _rank_col((<IntColumn>col).data_c, n, method, ascending)
    elif col.dtype == TYPE_DOUBLE:
        return _rank_col((<DoubleColumn>col).data_c, n, method, ascending)
    raise TypeError(f"rank not supported for column type {col.dtype}")

cdef _rank_row_keys(BaseColumn col, Py_ssize_t n, str method, bint ascending):
    """2D 列按整行向量（_col_row_key）字典序排名。"""
    cdef DoubleColumn res = DoubleColumn(max(n, 1))
    res.length = n
    if n == 0:
        return res
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t j = 0
    cdef Py_ssize_t k = 0
    cdef Py_ssize_t dense_rank_int = 1
    cdef double avg_rank
    cdef object cur_key
    cdef list keys = [_col_row_key(col, i) for i in range(n)]
    cdef list pairs = sorted([(keys[i], i) for i in range(n)], key=lambda kv: kv[0], reverse=not ascending)

    if method == 'first':
        for k in range(n):
            res.data_c[<Py_ssize_t>pairs[k][1]] = <double>(k + 1)
    else:
        while i < n:
            j = i
            cur_key = pairs[i][0]
            while j < n and pairs[j][0] == cur_key:
                j += 1
            if method == 'average':
                avg_rank = <double>(i + j + 1) / 2.0
                for k in range(i, j):
                    res.data_c[<Py_ssize_t>pairs[k][1]] = avg_rank
            elif method == 'min':
                for k in range(i, j):
                    res.data_c[<Py_ssize_t>pairs[k][1]] = <double>(i + 1)
            elif method == 'max':
                for k in range(i, j):
                    res.data_c[<Py_ssize_t>pairs[k][1]] = <double>(j)
            elif method == 'dense':
                for k in range(i, j):
                    res.data_c[<Py_ssize_t>pairs[k][1]] = <double>dense_rank_int
                dense_rank_int += 1
            else:
                raise ValueError(f"Unknown ranking method: {method}")
            i = j
    return res

cdef _cut_impl(DataFrame self, str col_name, list bins, bint include_lowest, str right):
    if col_name not in self._columns:
        raise KeyError(f"Column '{col_name}' not found")
    cdef BaseColumn col = self._columns[col_name]
    if _col_is_2d(col):
        raise TypeError(f"cut not supported for 2D column '{col_name}' (单元格为整行向量，无标量可切分)")
    cdef Py_ssize_t n = self.df_length
    cdef Py_ssize_t num_bins = len(bins) - 1
    cdef Py_ssize_t i, j
    cdef double val
    cdef bint is_right_closed = (right == 'right')
    cdef list results = []
    cdef double lo_val, hi_val

    for i in range(n):
        if col.dtype == TYPE_INT:
            val = <double>(<IntColumn>col).data_c[i]
        elif col.dtype == TYPE_DOUBLE:
            val = (<DoubleColumn>col).data_c[i]
        else:
            raise TypeError(f"cut not supported for column type {col.dtype}")

        assigned = False
        for j in range(num_bins):
            lo_val = <double>bins[j]
            hi_val = <double>bins[j + 1]
            if is_right_closed:
                if (val > lo_val or (include_lowest and val == lo_val and j == 0)) and val <= hi_val:
                    results.append(f"({lo_val}, {hi_val}]")
                    assigned = True
                    break
            else:
                if val >= lo_val and (val < hi_val or (val == hi_val and j == num_bins - 1)):
                    results.append(f"[{lo_val}, {hi_val})")
                    assigned = True
                    break
        if not assigned:
            results.append('')

    return _column_from_list_impl(self, col_name, results)

cdef _qcut_impl(DataFrame self, str col_name, int q, str labels, double precision):
    if col_name not in self._columns:
        raise KeyError(f"Column '{col_name}' not found")
    cdef BaseColumn col = self._columns[col_name]
    cdef Py_ssize_t n = self.df_length
    if _col_is_2d(col):
        raise TypeError(f"qcut not supported for 2D column '{col_name}' (单元格为整行向量，无标量可分位)")

    cdef DoubleColumn values = DoubleColumn(max(n, 1))
    values.length = n
    cdef Py_ssize_t i
    if col.dtype == TYPE_INT:
        for i in range(n):
            values.data_c[i] = <double>(<IntColumn>col).data_c[i]
    elif col.dtype == TYPE_DOUBLE:
        for i in range(n):
            values.data_c[i] = (<DoubleColumn>col).data_c[i]
    else:
        raise TypeError(f"qcut not supported for column type {col.dtype}")

    cdef DoubleColumn ranked = _rank_col(values.data_c, n, 'first', True)
    cdef list results = []
    cdef double bucket_size = <double>n / <double>q
    cdef int bucket_idx
    cdef double lo, hi

    for i in range(n):
        bucket_idx = <int>((ranked.data_c[i] - 1.0) / bucket_size)
        if bucket_idx >= q:
            bucket_idx = q - 1
        lo = bucket_idx * bucket_size
        hi = (bucket_idx + 1) * bucket_size
        results.append(f"({lo:.{int(precision)}f}, {hi:.{int(precision)}f}]")

    return _column_from_list_impl(self, col_name, results)
