# _fastdataframe_shift.pxi — Shift and diff operations (module-level cdef)
# Included at module level in fastdataframe.pyx (after class definition)

cdef _shift_int_column(IntColumn col, Py_ssize_t periods, Py_ssize_t n, Py_ssize_t c, Py_ssize_t r):
    cdef IntColumn new_col = IntColumn(max(n, 1), cols=c)
    new_col.length = n
    # 强制锁定新列的 2D 状态，防止发生 1D 降级
    new_col.set_meta_c(<int>c, True, c > 1)

    cdef int* src = col.data_c
    cdef int* dst = new_col.data_c
    cdef Py_ssize_t j, base

    # 绝对防止内存越界引发 Segment Fault
    cdef Py_ssize_t p = periods if periods >= 0 else -periods
    if p > r: p = r

    cdef Py_ssize_t valid_len = r - p

    with nogil:
        if periods >= 0:
            for j in range(c):
                base = j * r
                if p > 0:
                    memset(dst + base, 0, p * sizeof(int))
                if valid_len > 0:
                    memcpy(dst + base + p, src + base, valid_len * sizeof(int))
        else:
            for j in range(c):
                base = j * r
                if valid_len > 0:
                    memcpy(dst + base, src + base + p, valid_len * sizeof(int))
                if p > 0:
                    memset(dst + base + valid_len, 0, p * sizeof(int))
    return new_col


cdef _shift_double_column(DoubleColumn col, Py_ssize_t periods, Py_ssize_t n, Py_ssize_t c, Py_ssize_t r):
    cdef DoubleColumn new_col = DoubleColumn(max(n, 1), cols=c)
    new_col.length = n
    new_col.set_meta_c(<int>c, True, c > 1)

    cdef double* src = col.data_c
    cdef double* dst = new_col.data_c
    cdef Py_ssize_t i, j, base

    cdef Py_ssize_t p = periods if periods >= 0 else -periods
    if p > r: p = r

    cdef Py_ssize_t valid_len = r - p

    with nogil:
        for i in range(n):
            dst[i] = NAN
        if valid_len > 0:
            if periods >= 0:
                for j in range(c):
                    memcpy(dst + j * r + p, src + j * r, valid_len * sizeof(double))
            else:
                for j in range(c):
                    memcpy(dst + j * r, src + j * r + p, valid_len * sizeof(double))

    return new_col


cdef _shift_bool_column(BoolColumn col, Py_ssize_t periods, Py_ssize_t n, Py_ssize_t c, Py_ssize_t r):
    cdef BoolColumn new_col = BoolColumn(max(n, 1), cols=c)
    new_col.length = n
    new_col.set_meta_c(<int>c, True, c > 1)

    cdef unsigned char* src = col.data_c
    cdef unsigned char* dst = new_col.data_c
    cdef Py_ssize_t j, base

    cdef Py_ssize_t p = periods if periods >= 0 else -periods
    if p > r: p = r

    cdef Py_ssize_t valid_len = r - p

    with nogil:
        if periods >= 0:
            for j in range(c):
                base = j * r
                if p > 0:
                    memset(dst + base, 0, p * sizeof(unsigned char))
                if valid_len > 0:
                    memcpy(dst + base + p, src + base, valid_len * sizeof(unsigned char))
        else:
            for j in range(c):
                base = j * r
                if valid_len > 0:
                    memcpy(dst + base, src + base + p, valid_len * sizeof(unsigned char))
                if p > 0:
                    memset(dst + base + valid_len, 0, p * sizeof(unsigned char))

    return new_col


cdef _shift_column_impl(BaseColumn col, Py_ssize_t periods, Py_ssize_t n):
    cdef bint is2d
    cdef Py_ssize_t c_out, r_out

    _probe_shape(col, &is2d, &c_out, &r_out)
    cdef Py_ssize_t total = c_out * r_out

    if col.dtype == TYPE_INT:
        return _shift_int_column(<IntColumn>col, periods, total, c_out, r_out)
    elif col.dtype == TYPE_DOUBLE:
        return _shift_double_column(<DoubleColumn>col, periods, total, c_out, r_out)
    elif col.dtype == TYPE_BOOL:
        return _shift_bool_column(<BoolColumn>col, periods, total, c_out, r_out)
    # TODO：扩展：所有固定字节大小的时间列（Date/DateTime/Time）和 Int 一样，
    # 都可以直接调用 memcpy 和 memset 0 搞定，在此处添加支持
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
