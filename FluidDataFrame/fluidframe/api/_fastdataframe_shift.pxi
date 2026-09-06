# _fastdataframe_shift.pxi — Shift and diff operations (module-level cdef)
# Included at module level in fastdataframe.pyx (after class definition)

cdef _shift_int_column(IntColumn col, Py_ssize_t periods, Py_ssize_t n):
    cdef Py_ssize_t i, p
    cdef IntColumn new_col = IntColumn(max(n, 1))
    cdef int* src = col.data_c
    cdef int* dst = new_col.data_c
    new_col.length = n
    with nogil:
        if periods >= 0:
            for i in range(periods):
                dst[i] = 0
            for i in range(periods, n):
                dst[i] = src[i - periods]
        else:
            p = -periods
            for i in range(n - p, n):
                dst[i] = 0
            for i in range(n - p):
                dst[i] = src[i + p]
    return new_col

cdef _shift_double_column(DoubleColumn col, Py_ssize_t periods, Py_ssize_t n):
    cdef Py_ssize_t i, p
    cdef DoubleColumn new_col = DoubleColumn(max(n, 1))
    cdef double* src = col.data_c
    cdef double* dst = new_col.data_c
    cdef double nan_val = NAN
    new_col.length = n
    with nogil:
        if periods >= 0:
            for i in range(periods):
                dst[i] = nan_val
            for i in range(periods, n):
                dst[i] = src[i - periods]
        else:
            p = -periods
            for i in range(n - p, n):
                dst[i] = nan_val
            for i in range(n - p):
                dst[i] = src[i + p]
    return new_col

cdef _shift_bool_column(BoolColumn col, Py_ssize_t periods, Py_ssize_t n):
    cdef Py_ssize_t i, p
    cdef BoolColumn new_col = BoolColumn(max(n, 1))
    cdef unsigned char* src = col.data_c
    cdef unsigned char* dst = new_col.data_c
    new_col.length = n
    with nogil:
        if periods >= 0:
            for i in range(periods):
                dst[i] = 0
            for i in range(periods, n):
                dst[i] = src[i - periods]
        else:
            p = -periods
            for i in range(n - p, n):
                dst[i] = 0
            for i in range(n - p):
                dst[i] = src[i + p]
    return new_col

cdef _shift_column_impl(BaseColumn col, Py_ssize_t periods, Py_ssize_t n):
    if col.dtype == TYPE_INT:
        return _shift_int_column(<IntColumn>col, periods, n)
    elif col.dtype == TYPE_DOUBLE:
        return _shift_double_column(<DoubleColumn>col, periods, n)
    elif col.dtype == TYPE_BOOL:
        return _shift_bool_column(<BoolColumn>col, periods, n)
    else:
        raise TypeError(f"shift not supported for column type {col.dtype}")

cdef _shift_impl(FastDataFrame self, Py_ssize_t periods, str axis):
    if axis != 'index':
        raise ValueError(f"axis must be 'index', got '{axis}'")
    if periods == 0:
        return _copy_impl(self)

    cdef Py_ssize_t n = self.df_length
    cdef FastDataFrame result = FastDataFrame()
    result.df_length = n
    result.index = self.index

    for col_name, col in self.columns.items():
        result.columns[col_name] = _shift_column_impl(col, periods, n)

    return result

cdef _diff_column_all_impl(FastDataFrame self, int periods):
    cdef FastDataFrame result = FastDataFrame()
    result.df_length = self.df_length
    result.index = self.index
    cdef Py_ssize_t n = self.df_length

    for col_name, col in self.columns.items():
        if col.dtype == TYPE_INT or col.dtype == TYPE_DOUBLE:
            result.columns[col_name] = self._diff_column(col, periods)

    return result
