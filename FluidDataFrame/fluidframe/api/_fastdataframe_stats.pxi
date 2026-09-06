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

cdef _rank_double_col(DoubleColumn col, Py_ssize_t n, str method, bint ascending):
    cdef DoubleColumn res = DoubleColumn(max(n, 1))
    res.length = n
    if n == 0:
        return res

    cdef _RankItem* items = <_RankItem*>aligned_malloc(max(n, 1) * sizeof(_RankItem))
    if not items: raise MemoryError()
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t dense_rank_int
    cdef double avg_rank
    cdef double* data = col.data_c

    try:
        for i in range(n):
            items[i].idx = i
            items[i].val = data[i]

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

cdef _rank_int_col(IntColumn col, Py_ssize_t n, str method, bint ascending):
    cdef DoubleColumn res = DoubleColumn(max(n, 1))
    res.length = n
    if n == 0:
        return res

    cdef _RankItem* items = <_RankItem*>aligned_malloc(max(n, 1) * sizeof(_RankItem))
    if not items: raise MemoryError()
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t dense_rank_int
    cdef double avg_rank
    cdef int* data = col.data_c

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

cdef _rank_impl(FastDataFrame self, str col_name, str method, bint ascending, str na_option):
    if col_name not in self.columns:
        raise KeyError(f"Column '{col_name}' not found")
    cdef BaseColumn col = self.columns[col_name]
    cdef Py_ssize_t n = self.df_length
    if col.dtype == TYPE_INT:
        return _rank_int_col(<IntColumn>col, n, method, ascending)
    elif col.dtype == TYPE_DOUBLE:
        return _rank_double_col(<DoubleColumn>col, n, method, ascending)
    raise TypeError(f"rank not supported for column type {col.dtype}")

cdef _cut_impl(FastDataFrame self, str col_name, list bins, bint include_lowest, str right):
    if col_name not in self.columns:
        raise KeyError(f"Column '{col_name}' not found")
    cdef BaseColumn col = self.columns[col_name]
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

cdef _qcut_impl(FastDataFrame self, str col_name, int q, str labels, double precision):
    if col_name not in self.columns:
        raise KeyError(f"Column '{col_name}' not found")
    cdef BaseColumn col = self.columns[col_name]
    cdef Py_ssize_t n = self.df_length

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

    cdef DoubleColumn ranked = _rank_double_col(values, n, 'first', True)
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
