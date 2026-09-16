# ============================================================
#  _databaseframe_helper.pxi —— DataFrame 模块级 cdef/cpdef 辅助函数
#  说明：本文件经 _databaseframe.pyx 末尾 include 注入，正文内全部为
#  模块级 cdef / cpdef 函数（C 工具、列转换工厂、核心实现、io、
#  metric、rolling、统计、groupby、linalg 等）。
#  依赖：python 侧 import / cimport（openpyxl、libc、_columns 等）
#  与 DataFrame 等类型化类均由宿主 pyx 在同文件内提前提供，故编译期
#  符号均可解析（函数为全文件作用域，无需文本序；结构体类型仍在 pyx）。
# ============================================================

# ============================================================
#  Series / DataFrame 桥接助手
#  原则：DataFrame._columns 永远只存原生 BaseColumn，
#  Series 是「随用随建、用完即毁」的临时包装视图。
# ============================================================

cdef object _wrap_series(str name, object res, IndexBase index):
    """若结果为单列，则包成带索引的 Series 视图；否则原样返回。"""
    if isinstance(res, BaseColumn):
        return Series(name, <BaseColumn>res, index)
    return res


cdef Series _agg_series(dict values):
    """把 {col_name: scalar} 聚合结果包成带列名 StringIndex 的 Series（pandas 语义）。"""
    cdef list names = list(values.keys())
    cdef list vals = [values[nm] for nm in names]
    cdef BaseColumn col
    cdef object v
    cdef bint any_float = False
    for v in vals:
        if isinstance(v, float):
            any_float = True
            break
    if any_float:
        vals = [float(v) for v in vals]
    if len(vals) == 0:
        col = IntColumn(0)
    else:
        from fluidframe.core.dataframe import _column_from_list_impl
        col = _column_from_list_impl(None, '', vals)
    return Series('', col, StringIndex(names))


cdef BaseColumn _col_take(BaseColumn col, Py_ssize_t* indices, Py_ssize_t n, bint inplace):
    """与 DataFrame._reorder_column 等价的列级 take（含 2D 列行级重排）。"""
    if col.dtype == TYPE_INT:
        return (<IntColumn>col).take_by_indices(indices, n, inplace)
    elif col.dtype == TYPE_DOUBLE:
        return (<DoubleColumn>col).take_by_indices(indices, n, inplace)
    elif col.dtype == TYPE_CHAR256:
        return (<AsciiColumn>col).take_by_indices(indices, n, inplace)
    elif col.dtype == TYPE_UCS2:
        return (<UCS2Column>col).take_by_indices(indices, n, inplace)
    elif col.dtype == TYPE_DATE:
        return (<DateColumn>col).take_by_indices(indices, n, inplace)
    elif col.dtype == TYPE_BOOL:
        return (<BoolColumn>col).take_by_indices(indices, n, inplace)
    elif col.dtype == TYPE_CATEGORY:
        return (<CategoryColumn>col).take_by_indices(indices, n, inplace)
    elif col.dtype == TYPE_DATETIME:
        return (<DateTimeColumn>col).take_by_indices(indices, n, inplace)
    elif col.dtype == TYPE_TIME:
        return (<TimeColumn>col).take_by_indices(indices, n, inplace)
    raise TypeError(f"Unsupported column dtype: {col.dtype}")


cdef BaseColumn _column_from_scalar(DataFrame df, str name, object val, Py_ssize_t n):
    """标量 → 定长列（长度与 df 对齐），内部走 _column_from_list_impl。"""
    if n <= 0:
        return IntColumn(0)
    return _column_from_list_impl(df, name, [val] * n)


# ============================================================
#  底层 C 级小工具（排序 / 比较 / 快速插入）
# ============================================================
cdef inline int _cstrncmp(const char* a, Py_ssize_t a_len, const char* b, Py_ssize_t b_len) noexcept nogil:
    cdef Py_ssize_t min_len = a_len if a_len < b_len else b_len
    cdef int r = strncmp(a, b, min_len)
    if r != 0:
        return r
    if a_len < b_len:
        return -1
    elif a_len > b_len:
        return 1
    return 0

cdef void _c_argsort_char256(char* data, Py_ssize_t* offsets, Py_ssize_t* indices, Py_ssize_t left, Py_ssize_t right) noexcept nogil:
    if left >= right: return

    cdef Py_ssize_t stack[64]
    cdef Py_ssize_t top = -1
    cdef Py_ssize_t l, r, i, j, temp
    cdef Py_ssize_t pivot_idx, cur_len
    cdef char* pivot_ptr
    cdef Py_ssize_t pivot_len

    top += 1; stack[top] = left
    top += 1; stack[top] = right

    while top >= 0:
        r = stack[top]; top -= 1
        l = stack[top]; top -= 1
        if l >= r:
            continue

        pivot_idx = indices[(l + r) // 2]
        pivot_ptr = data + offsets[pivot_idx]
        pivot_len = offsets[pivot_idx + 1] - offsets[pivot_idx] - 1
        i = l; j = r

        while i <= j:
            while i <= j:
                cur_len = offsets[indices[i] + 1] - offsets[indices[i]] - 1
                if _cstrncmp(data + offsets[indices[i]], cur_len, pivot_ptr, pivot_len) >= 0:
                    break
                i += 1
            while i <= j:
                cur_len = offsets[indices[j] + 1] - offsets[indices[j]] - 1
                if _cstrncmp(data + offsets[indices[j]], cur_len, pivot_ptr, pivot_len) <= 0:
                    break
                j -= 1
            if i <= j:
                temp = indices[i]; indices[i] = indices[j]; indices[j] = temp
                i += 1; j -= 1

        if l < j:
            if j - l < r - i:
                top += 1; stack[top] = l
                top += 1; stack[top] = j
                if i < right:
                    top += 1; stack[top] = i
                    top += 1; stack[top] = r
            else:
                if i < r:
                    top += 1; stack[top] = i
                    top += 1; stack[top] = r
                if l < j:
                    top += 1; stack[top] = l
                    top += 1; stack[top] = j
        elif i < r:
            top += 1; stack[top] = i
            top += 1; stack[top] = r


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _c_argsort_numeric(NumericType* data, Py_ssize_t* indices, Py_ssize_t left, Py_ssize_t right) noexcept nogil:
    if left >= right: return

    cdef Py_ssize_t stack[64]
    cdef Py_ssize_t top = -1
    cdef Py_ssize_t l, r, i, j, temp
    cdef Py_ssize_t mid_idx
    cdef NumericType pivot

    top += 1; stack[top] = left
    top += 1; stack[top] = right

    while top >= 0:
        r = stack[top]; top -= 1
        l = stack[top]; top -= 1
        if l >= r:
            continue

        mid_idx = l + ((r - l) >> 1)
        if data[indices[l]] > data[indices[mid_idx]]:
            temp = indices[l]; indices[l] = indices[mid_idx]; indices[mid_idx] = temp
        if data[indices[l]] > data[indices[r]]:
            temp = indices[l]; indices[l] = indices[r]; indices[r] = temp
        if data[indices[mid_idx]] > data[indices[r]]:
            temp = indices[mid_idx]; indices[mid_idx] = indices[r]; indices[r] = temp

        pivot = data[indices[mid_idx]]
        i = l; j = r

        while i <= j:
            while data[indices[i]] < pivot: i += 1
            while data[indices[j]] > pivot: j -= 1
            if i <= j:
                temp = indices[i]; indices[i] = indices[j]; indices[j] = temp
                i += 1; j -= 1

        if l < j:
            if j - l < r - i:
                top += 1; stack[top] = l
                top += 1; stack[top] = j
                if i < right:
                    top += 1; stack[top] = i
                    top += 1; stack[top] = r
            else:
                if i < r:
                    top += 1; stack[top] = i
                    top += 1; stack[top] = r
                if l < j:
                    top += 1; stack[top] = l
                    top += 1; stack[top] = j
        elif i < r:
            top += 1; stack[top] = i
            top += 1; stack[top] = r


cdef inline int _ucs2_cmp_at(Py_UCS2* chars, Py_ssize_t* offsets, Py_ssize_t i, Py_ssize_t j) noexcept nogil:
    cdef int i_start = offsets[i], i_len = offsets[i+1] - i_start
    cdef int j_start = offsets[j], j_len = offsets[j+1] - j_start
    cdef int k, min_len = i_len if i_len < j_len else j_len
    for k in range(min_len):
        if chars[i_start + k] < chars[j_start + k]: return -1
        if chars[i_start + k] > chars[j_start + k]: return 1
    if i_len < j_len: return -1
    if i_len > j_len: return 1
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _c_argsort_ucs2(Py_UCS2* chars, Py_ssize_t* offsets, Py_ssize_t* indices, Py_ssize_t left, Py_ssize_t right) noexcept nogil:
    if left >= right: return

    cdef Py_ssize_t stack[64]
    cdef Py_ssize_t top = -1
    cdef Py_ssize_t l, r, i, j, temp
    cdef Py_ssize_t pivot_idx

    top += 1; stack[top] = left
    top += 1; stack[top] = right

    while top >= 0:
        r = stack[top]; top -= 1
        l = stack[top]; top -= 1
        if l >= r:
            continue

        pivot_idx = indices[(l + r) // 2]
        i = l; j = r

        while i <= j:
            while _ucs2_cmp_at(chars, offsets, indices[i], pivot_idx) < 0: i += 1
            while _ucs2_cmp_at(chars, offsets, indices[j], pivot_idx) > 0: j -= 1
            if i <= j:
                temp = indices[i]; indices[i] = indices[j]; indices[j] = temp
                i += 1; j -= 1

        if l < j:
            if j - l < r - i:
                top += 1; stack[top] = l
                top += 1; stack[top] = j
                if i < right:
                    top += 1; stack[top] = i
                    top += 1; stack[top] = r
            else:
                if i < r:
                    top += 1; stack[top] = i
                    top += 1; stack[top] = r
                if l < j:
                    top += 1; stack[top] = l
                    top += 1; stack[top] = j
        elif i < r:
            top += 1; stack[top] = i
            top += 1; stack[top] = r



cdef inline void _fast_insert_val(int dtype, void* data_ptr, void* col_obj,
                                  uint32_t row_idx, object val) except *:
    cdef int ival
    cdef int32_t days
    cdef int64_t tmp_dt

    if dtype == TYPE_INT:
        (<int*>data_ptr)[row_idx] = <int>val
    elif dtype == TYPE_DOUBLE:
        (<double*>data_ptr)[row_idx] = <double>val
    elif dtype == TYPE_DATE:
        if isinstance(val, int) and val > 10000000:
            ival = <int>val
            days = _ymd_to_days(ival // 10000, (ival // 100) % 100, ival % 100)
            (<int32_t*>data_ptr)[row_idx] = days
        else:
            (<int32_t*>data_ptr)[row_idx] = <int32_t>val
    elif dtype == TYPE_DATETIME:
        if isinstance(val, int) and val > 10000000:
            ival = <int>val
            days = _ymd_to_days(ival // 10000, (ival // 100) % 100, ival % 100)
            tmp_dt = <int64_t>days
            (<int64_t*>data_ptr)[row_idx] = tmp_dt * 86400000000LL
        else:
            tmp_dt = <int64_t>val
            (<int64_t*>data_ptr)[row_idx] = tmp_dt * 1000000LL
    elif dtype == TYPE_TIME:
        (<int32_t*>data_ptr)[row_idx] = <int32_t>val
    elif dtype == TYPE_CHAR256:
        (<AsciiColumn>col_obj).append_nolen(val)
    elif dtype == TYPE_UCS2:
        (<UCS2Column>col_obj).append_nolen(val)
    elif dtype == TYPE_BOOL:
        (<BoolColumn>col_obj).append_nolen(val)
    elif dtype == TYPE_CATEGORY:
        (<CategoryColumn>col_obj).append_nolen(val)



cdef inline void _probe_shape(BaseColumn col, bint* is2d, Py_ssize_t* c, Py_ssize_t* r):
    is2d[0] = col.is_2d
    c[0] = col.num_cols
    r[0] = col._get_rows_c()


cdef inline void _fast_set_cell(BaseColumn col, int dtype, Py_ssize_t idx, object val) except *:
    cdef bint is2d
    cdef Py_ssize_t c, r, j
    cdef object v
    cdef int32_t days
    cdef int64_t tmp_dt

    _probe_shape(col, &is2d, &c, &r)

    if dtype == TYPE_INT:
        if is2d and c > 1:
            for j in range(c):
                v = val[j]
                (<IntColumn>col).data_c[j * r + idx] = <int>v
        else:
            (<IntColumn>col).data_c[idx] = <int>val

    elif dtype == TYPE_DOUBLE:
        if is2d and c > 1:
            for j in range(c):
                v = val[j]
                (<DoubleColumn>col).data_c[j * r + idx] = <double>v
        else:
            (<DoubleColumn>col).data_c[idx] = <double>val

    elif dtype == TYPE_DATE:
        if is2d and c > 1:
            for j in range(c):
                v = val[j]
                if isinstance(v, int) and v > 10000000:
                    days = _ymd_to_days(v // 10000, (v // 100) % 100, v % 100)
                else:
                    days = <int32_t>v
                (<int32_t*>(<DateColumn>col).data_c)[j * r + idx] = days
        else:
            if isinstance(val, int) and val > 10000000:
                days = _ymd_to_days(val // 10000, (val // 100) % 100, val % 100)
            else:
                days = <int32_t>val
            (<int32_t*>(<DateColumn>col).data_c)[idx] = days

    elif dtype == TYPE_DATETIME:
        if is2d and c > 1:
            for j in range(c):
                v = val[j]
                if isinstance(v, int) and v > 10000000:
                    tmp_dt = <int64_t>_ymd_to_days(v // 10000, (v // 100) % 100, v % 100) * 86400000000LL
                else:
                    tmp_dt = <int64_t>v * 1000000LL
                (<int64_t*>(<DateTimeColumn>col).data_c)[j * r + idx] = tmp_dt
        else:
            if isinstance(val, int) and val > 10000000:
                tmp_dt = <int64_t>_ymd_to_days(val // 10000, (val // 100) % 100, val % 100) * 86400000000LL
            else:
                tmp_dt = <int64_t>val * 1000000LL
            (<int64_t*>(<DateTimeColumn>col).data_c)[idx] = tmp_dt

    elif dtype == TYPE_TIME:
        if is2d and c > 1:
            for j in range(c):
                v = val[j]
                (<int32_t*>(<TimeColumn>col).data_c)[j * r + idx] = <int32_t>v
        else:
            (<int32_t*>(<TimeColumn>col).data_c)[idx] = <int32_t>val

    elif dtype == TYPE_CHAR256:
        (<AsciiColumn>col)._set_item(idx, str(val))

    elif dtype == TYPE_UCS2:
        (<UCS2Column>col)._set_item(idx, str(val))

    elif dtype == TYPE_BOOL:
        (<BoolColumn>col).data_c[idx] = 1 if val else 0

    elif dtype == TYPE_CATEGORY:
        if val is None:
            (<CategoryColumn>col).codes_c[idx] = (<CategoryColumn>col)._get_or_add_code('')
        else:
            (<CategoryColumn>col).codes_c[idx] = (<CategoryColumn>col)._get_or_add_code(str(val))

    else:
        raise TypeError(f"Unsupported dtype {dtype} for setting item")


# ============================================================
#  DataFrame 模块级实现 —— 核心 / CRUD
# ============================================================
cpdef BaseColumn _column_from_list_impl(DataFrame self, str name, list values):
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
    Int/Double 用公开 from_2d 工厂；
    Date 键为 ymd int，DateTime 键为 str（_datetime_to_str 格式），Time 键为 str
    （_time_to_str 格式），各自转回内部存储（days / us / sec）；
    其余 dtype（字符串/布尔/类别）不承载 2D 语义，明确 NotImpl。"""
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
    cdef DateTimeColumn dtcol
    cdef TimeColumn tcol
    cdef int32_t* dptr
    cdef int64_t* dtptr
    cdef int32_t* tptr
    cdef int ymd
    cdef object v
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
    elif dtype == TYPE_DATETIME:
        dtcol = DateTimeColumn(max(total, 1), cols=c)
        dtcol.length = total
        dtcol.num_cols = <int>c
        dtcol.is_2d = True
        dtcol.owns_data = True
        dtptr = <int64_t*>dtcol.data_c
        for k in range(nkeys):
            for j in range(c):
                v = keys[k][j]
                if isinstance(v, str):
                    dtptr[j * nkeys + k] = _parse_datetime_str(<str>v)
                else:
                    dtptr[j * nkeys + k] = <int64_t>v
        return dtcol
    elif dtype == TYPE_TIME:
        tcol = TimeColumn(max(total, 1), cols=c)
        tcol.length = total
        tcol.num_cols = <int>c
        tcol.is_2d = True
        tcol.owns_data = True
        tptr = <int32_t*>tcol.data_c
        for k in range(nkeys):
            for j in range(c):
                v = keys[k][j]
                if isinstance(v, str):
                    tptr[j * nkeys + k] = _parse_time_str(<str>v)
                else:
                    tptr[j * nkeys + k] = <int32_t>v
        return tcol
    raise NotImplementedError(
        f"value_counts on a 2D column is not supported for dtype code {dtype}: "
        "only Int/Double/Date/DateTime/Time carry 2D semantics; "
        "string/bool/category columns are always 1D"
    )

cdef _eval_impl(DataFrame self, str expr, bint inplace):
    cdef DataFrame target_df = self if inplace else self.copy()
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

    cdef dict env = dict(self._columns)
    try:
        res_col = eval(calc_expr, {"__builtins__": {}}, env)
    except Exception as e:
        raise ValueError(f"Failed to evaluate expression '{calc_expr}': {e}")

    if is_assignment:
        target_df._columns[target_col] = res_col
        return target_df
    else:
        return _wrap_series(target_col if target_col else '_result', res_col, self.index)

cdef _query_impl(DataFrame self, str expr):
    mask = _eval_impl(self, expr, False)
    if isinstance(mask, Series):
        mask = (<Series>mask)._col
    if not type(mask).__name__ in ('IntColumn', 'BoolColumn'):
        raise TypeError("Query expression must evaluate to a boolean mask (IntColumn or BoolColumn)")
    return self.loc[mask]

cdef _append_row_impl(DataFrame self, dict kwargs):
    for col_name in self._columns:
        if col_name not in kwargs:
            raise ValueError(f"Missing value for column: {col_name}")

    cdef Py_ssize_t original_df_length = self.df_length
    cdef Py_ssize_t original_index_length = self.index.length if self.index is not None else 0

    try:
        for col_name, val in kwargs.items():
            self._columns[col_name].append(val)
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
        for col_name, col_obj in self._columns.items():
            while col_obj.length > original_df_length:
                col_obj.pop_back()
        if self.index is not None:
            if self.index.index_type == INDEX_RANGE:
                self.index.length = original_index_length
            elif self.index.index_type == INDEX_INT:
                while (<IntIndex>self.index).length > original_index_length:
                    (<IntIndex>self.index).pop_back()
        raise

cdef _append_impl(DataFrame self, object row, bint ignore_index):
    cdef DataFrame new_df = self.copy()
    cdef list col_names
    if isinstance(row, dict):
        _append_row_impl(new_df, row)
    elif isinstance(row, list):
        col_names = list(new_df._columns.keys())
        if len(row) != len(col_names):
            raise ValueError(f"Length of row {len(row)} != number of columns {len(col_names)}")
        kwargs = {col_names[i]: row[i] for i in range(len(col_names))}
        _append_row_impl(new_df, kwargs)
    else:
        raise TypeError(f"Cannot append row from {type(row)}")
    if ignore_index:
        new_df.index = RangeIndex(0, new_df.df_length)
    return new_df

cdef _copy_impl(DataFrame self):
    cdef Py_ssize_t n, i
    cdef Py_ssize_t* idx_c
    cdef DataFrame new_df = DataFrame()
    new_df.df_length = self.df_length
    for col_name, col_obj in self._columns.items():
        new_df._columns[col_name] = col_obj.copy()

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

cdef _pop_front_impl(DataFrame self):
    if self.df_length == 0: raise IndexError("pop from empty DataFrame")
    row = {}
    for col_name, col_obj in self._columns.items():
        row[col_name] = col_obj.pop_front()
    self.df_length -= 1
    if self.index is not None and self.index.length > 0:
        if self.index.index_type == INDEX_RANGE:
            (<RangeIndex>self.index).start += (<RangeIndex>self.index).step
        elif self.index.index_type == INDEX_INT:
            (<IntIndex>self.index).pop_back()
        self.index.length -= 1
    return row

cdef _pop_back_impl(DataFrame self):
    if self.df_length == 0: raise IndexError("pop from empty DataFrame")
    row = {}
    for col_name, col_obj in self._columns.items():
        row[col_name] = col_obj.pop_back()
    self.df_length -= 1
    if self.index is not None and self.index.length > 0:
        if self.index.index_type == INDEX_INT:
            (<IntIndex>self.index).pop_back()
        self.index.length -= 1
    return row

cdef _rename_impl(DataFrame self, dict mapper):
    cdef DataFrame result = _copy_impl(self)
    cdef str old_name, new_name
    for old_name, new_name in mapper.items():
        if old_name in result._columns:
            result._columns[new_name] = result._columns.pop(old_name)
    return result

cdef _reset_index_impl(DataFrame self):
    cdef IntColumn idx_col = IntColumn(self.df_length)
    cdef Py_ssize_t i
    idx_col.length = self.df_length
    for i in range(self.df_length):
        idx_col.data_c[i] = <int>i
    self._columns['index'] = idx_col
    self.index = RangeIndex(0, self.df_length)
    return self

cdef _set_index_impl(DataFrame self, str col_name, bint drop=True):
    """df.set_index：将列提升为索引并（默认）移除。
    全 int 值 → IntIndex；否则 → 单层 MultiIndex（levels 出现序 + codes）。"""
    if col_name not in self._columns:
        raise KeyError(f"Column '{col_name}' not found")
    cdef BaseColumn src = self._columns[col_name]
    cdef list keys = _col_to_keys(src)
    cdef Py_ssize_t i, n = len(keys)
    cdef object v
    cdef bint all_int = True
    cdef IntColumn idx_col
    cdef IntColumn codes
    cdef list levels = []
    cdef dict level_lookup = {}
    for i in range(n):
        v = keys[i]
        if isinstance(v, bool) or not isinstance(v, int):
            all_int = False
            break

    if drop:
        self._columns.pop(col_name)

    if all_int:
        idx_col = IntColumn(max(n, 1))
        idx_col.length = n
        for i in range(n):
            idx_col.data_c[i] = <int>keys[i]
        self.index = IntIndex(idx_col)
        return self

    codes = IntColumn(max(n, 1))
    codes.length = n
    for i in range(n):
        v = keys[i]
        if v not in level_lookup:
            level_lookup[v] = len(levels)
            levels.append(v)
        codes.data_c[i] = <int>level_lookup[v]
    self.index = MultiIndex([levels], [codes], [col_name])
    return self

cdef _get_dummies_impl(DataFrame self, col_names, str prefix, bint drop_first):
    """df.get_dummies：对指定（默认全部字符串/类别）列 Each 唯一值建 0/1 IntColumn。
    2D 列以行级键 one-hot；原列保留。"""
    cdef DataFrame result = _copy_impl(self)
    cdef list selected
    cdef str name
    cdef BaseColumn col
    if col_names is None:
        selected = []
        for name, col in result._columns.items():
            if col.dtype == TYPE_CHAR256 or col.dtype == TYPE_UCS2 or col.dtype == TYPE_CATEGORY:
                selected.append(name)
    else:
        selected = list(col_names)

    cdef str pfx, out_name
    cdef list keys, uniques
    cdef dict seen
    cdef object u
    cdef Py_ssize_t i, nk
    cdef IntColumn dcol

    for name in selected:
        if name not in result._columns:
            raise KeyError(f"Column '{name}' not found")
        col = result._columns[name]
        keys = _col_to_keys(col)
        nk = len(keys)
        uniques = []
        seen = {}
        for v in keys:
            if v not in seen:
                seen[v] = 1
                uniques.append(v)
        if drop_first and uniques:
            uniques = uniques[1:]
        pfx = name if prefix == '' else prefix
        for u in uniques:
            out_name = f"{pfx}_{u}"
            dcol = IntColumn(max(nk, 1))
            dcol.length = nk
            for i in range(nk):
                dcol.data_c[i] = 1 if keys[i] == u else 0
            result._columns[out_name] = dcol
    return result

cdef _reindex_impl(DataFrame self, list new_index):
    cdef Py_ssize_t new_len = len(new_index)
    cdef dict old_pos = {}
    cdef Py_ssize_t i, old_idx
    cdef DataFrame result

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

        result = DataFrame()
        result.df_length = new_len
        result.index = RangeIndex(0, new_len)

        for col_name, col_obj in self._columns.items():
            result._columns[col_name] = _reindex_column_impl(col_obj, index_map, new_len)

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

cdef _insert_row_impl(DataFrame self, Py_ssize_t pos, dict kwargs):
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
    for col_name in self._columns:
        if col_name not in kwargs:
            raise ValueError(f"Missing value for column '{col_name}'")
        row_vals[col_name] = kwargs[col_name]

    try:
        for col_name, col_obj in self._columns.items():
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
            self._shift_left(self._columns[cn], pos)
        raise

cdef _drop_with_index_impl(DataFrame self, list drop_indices, bint inplace):
    cdef Py_ssize_t n_orig = self.df_length
    if n_orig == 0:
        return self if inplace else _copy_impl(self)

    cdef char* drop_mask = <char*>aligned_malloc(n_orig * sizeof(char))
    if not drop_mask: raise MemoryError()
    memset(drop_mask, 0, n_orig)

    cdef Py_ssize_t idx, i, j
    cdef Py_ssize_t keep_count = 0
    cdef Py_ssize_t* keep_idx_c = NULL
    cdef DataFrame result_df
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
                for col_obj in self._columns.values(): col_obj.length = 0
                return self
            else:
                return DataFrame()

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
            result_df = DataFrame()
            result_df.df_length = keep_count

        for name, col in self._columns.items():
            new_col = self._reorder_column(col, keep_idx_c, keep_count, inplace)
            if not inplace:
                result_df._columns[name] = new_col

        if self.index is not None:
            if inplace:
                self.index = self.index._take(keep_idx_c, keep_count)
            else:
                result_df.index = self.index._take(keep_idx_c, keep_count)

        return result_df
    finally:
        if drop_mask: os_aligned_free(drop_mask)
        if keep_idx_c: os_aligned_free(keep_idx_c)

cdef _drop_index_impl(DataFrame self, object labels, bint inplace):
    if self.index is None:
        raise ValueError("DataFrame has no index set")
    cdef list drop_positions = []
    if isinstance(labels, list):
        for label in labels:
            drop_positions.append(self.index.get_loc(label))
    else:
        drop_positions.append(self.index.get_loc(labels))
    return _drop_with_index_impl(self, drop_positions, inplace)

cdef _drop_raw_impl(DataFrame self, Py_ssize_t row_idx):
    if row_idx < 0: row_idx += self.df_length
    if row_idx < 0 or row_idx >= self.df_length:
        raise IndexError("Row index out of range")

    cdef str name
    cdef BaseColumn col
    cdef Py_ssize_t rem_len_idx, ii
    cdef IntColumn tmp_col, idx_col
    cdef RangeIndex ri

    for name, col in self._columns.items():
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

cdef _insert_column_impl(DataFrame self, str col_name, object col_data, Py_ssize_t pos):
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

    cdef list names = list(self._columns.keys())
    cdef Py_ssize_t n = len(names)

    if pos < 0 or pos >= n:
        self._columns[col_name] = col
    else:
        new_cols = {}
        for i in range(n):
            if i == pos:
                new_cols[col_name] = col
            new_cols[names[i]] = self._columns[names[i]]
        self._columns = new_cols

cdef _duplicate_column_impl(DataFrame self, str src_name, str new_name):
    if src_name not in self._columns:
        raise KeyError(f"Column '{src_name}' not found")
    if new_name is None:
        new_name = src_name + '_dup'
    self._columns[new_name] = self._columns[src_name].copy()

cdef _duplicate_rows_impl(DataFrame self, list indices):
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



# ============================================================
#  DataFrame 模块级实现 —— 变换（transform）
# ============================================================

cdef _argsort_by_impl(DataFrame self, str col_name, bint reverse):
    if col_name not in self._columns:
        raise KeyError(f"Column '{col_name}' not found")

    cdef BaseColumn col = self._columns[col_name]
    cdef Py_ssize_t n = self.df_length
    if n == 0:
        return []

    # 2D 列：整行向量作排序键（_col_row_key 语义），Python 级字典序
    if _col_is_2d(col):
        return sorted(range(n), key=lambda i: _col_row_key(col, i), reverse=reverse)

    cdef Py_ssize_t* indices_c = <Py_ssize_t*>aligned_malloc(max(n, 1) * sizeof(Py_ssize_t))
    if not indices_c: raise MemoryError()
    cdef Py_ssize_t i
    cdef Py_ssize_t temp

    try:
        for i in range(n): indices_c[i] = i

        if col.dtype == TYPE_INT:
            with nogil:
                _c_argsort_numeric((<IntColumn>col).data_c, indices_c, 0, n - 1)
        elif col.dtype == TYPE_DOUBLE:
            with nogil:
                _c_argsort_numeric((<DoubleColumn>col).data_c, indices_c, 0, n - 1)
        elif col.dtype == TYPE_CHAR256:
            with nogil:
                _c_argsort_char256((<AsciiColumn>col).data_c, (<AsciiColumn>col).offsets_c, indices_c, 0, n - 1)
        elif col.dtype == TYPE_UCS2:
            with nogil:
                _c_argsort_ucs2((<UCS2Column>col).chars_c, (<UCS2Column>col).offsets_c, indices_c, 0, n - 1)
        elif col.dtype == TYPE_DATE:
            with nogil:
                _c_argsort_numeric(<int*>(<DateColumn>col).data_c, indices_c, 0, n - 1)
        elif col.dtype == TYPE_DATETIME:
            with nogil:
                _c_argsort_numeric(<long long*>(<DateTimeColumn>col).data_c, indices_c, 0, n - 1)
        elif col.dtype == TYPE_TIME:
            with nogil:
                _c_argsort_numeric(<int*>(<TimeColumn>col).data_c, indices_c, 0, n - 1)
        else:
            raise TypeError("Sorting not supported for this column type")

        if reverse:
            for i in range(n // 2):
                temp = indices_c[i]
                indices_c[i] = indices_c[n - 1 - i]
                indices_c[n - 1 - i] = temp

        return [indices_c[j] for j in range(n)]
    finally:
        os_aligned_free(indices_c)

cdef _sort_values_impl(DataFrame self, str col_name, bint reverse, bint inplace):
    if col_name not in self._columns:
        raise KeyError(f"Column '{col_name}' not found")

    cdef Py_ssize_t n = self.df_length
    if n == 0:
        return self if inplace else _copy_impl(self)

    cdef list py_indices = _argsort_by_impl(self, col_name, reverse)
    cdef Py_ssize_t* idx_c = <Py_ssize_t*>aligned_malloc(max(n, 1) * sizeof(Py_ssize_t))
    if not idx_c: raise MemoryError()
    cdef Py_ssize_t i
    cdef DataFrame result_df
    cdef str name
    cdef BaseColumn col, new_col

    try:
        for i in range(n): idx_c[i] = py_indices[i]
        if inplace:
            result_df = self
        else:
            result_df = DataFrame()
            result_df.df_length = n

        for name, col in self._columns.items():
            new_col = self._reorder_column(col, idx_c, n, inplace)
            if not inplace:
                result_df._columns[name] = new_col

        if not inplace and self.index is not None:
            result_df.index = self.index._take(idx_c, n)
        elif inplace and self.index is not None:
            self.index = self.index._take(idx_c, n)

        return result_df
    finally:
        os_aligned_free(idx_c)

cdef _diff_impl(DataFrame self, str col_name, int periods):
    if col_name not in self._columns:
        raise KeyError(f"Column '{col_name}' not found")
    cdef BaseColumn col = self._columns[col_name]
    return self._diff_column(col, periods)

cdef _apply_impl(DataFrame self, object func, str axis, bint raw):
    cdef str col_name
    cdef BaseColumn col_obj
    cdef Py_ssize_t c_out, r_out, total, i, j
    cdef list new_vals, result_vals, flat_data, col_names, cols_data
    cdef Py_ssize_t num_cols, n
    cdef dict row
    cdef DataFrame result
    cdef bint is2d
    import numpy as np

    if axis == 'columns':
        result = DataFrame()
        result.df_length = self.df_length
        result.index = RangeIndex(0, self.df_length)
        for col_name, col_obj in self._columns.items():
            _probe_shape(col_obj, &is2d, &c_out, &r_out)
            total = col_obj.length
            if raw:
                # Convert to numpy array and pass directly to func
                arr = np.array([col_obj[idx] for idx in range(total)])
                new_vals = [func(arr)]
                if not isinstance(new_vals[0], (list, np.ndarray)):
                    raise ValueError("When raw=True, func must return an array-like")
                new_vals = list(new_vals[0]) if isinstance(new_vals[0], (list, np.ndarray)) else [new_vals[0]]
            else:
                flat_data = [col_obj[idx] for idx in range(total)]
                new_vals = [func(v) for v in flat_data]

            if is2d and c_out > 1:
                result._columns[col_name] = _column_from_2d_list_impl(new_vals, r_out, c_out)
            else:
                result._columns[col_name] = _column_from_list_impl(result, col_name, new_vals)
        return result

    elif axis == 'index':
        col_names = list(self._columns.keys())
        num_cols = len(col_names)
        cols_data = [_col_to_keys(self._columns[name]) for name in col_names]
        n = self.df_length
        result_vals = [None] * n
        for i in range(n):
            if raw:
                row_arr = np.array([cols_data[j][i] for j in range(num_cols)])
                result_vals[i] = func(row_arr)
            else:
                row = {}
                for j in range(num_cols):
                    row[col_names[j]] = cols_data[j][i]
                result_vals[i] = func(row)
        return result_vals

    raise ValueError(f"axis must be 'columns' or 'index', got '{axis}'")

cdef _clip_column_impl(DataFrame self, str col_name, object lower, object upper, bint inplace):
    cdef BaseColumn col = self._columns[col_name]
    cdef double lower_val = float('-inf') if lower is None else <double>lower
    cdef double upper_val = float('inf') if upper is None else <double>upper
    if col.dtype == TYPE_INT:
        new_col = (<IntColumn>col).clip(lower_val, upper_val, inplace)
        if not inplace:
            self._columns[col_name] = new_col
    elif col.dtype == TYPE_DOUBLE:
        new_col = (<DoubleColumn>col).clip(lower_val, upper_val, inplace)
        if not inplace:
            self._columns[col_name] = new_col
    else:
        raise TypeError(f"clip not supported for column type {self._type_name(col)}")
    if inplace:
        return self

cdef _clip_all_impl(DataFrame self, object lower, object upper, bint inplace):
    """DataFrame.clip：对所有数值列（Int/Double）统一截断。2D 列由列级 clip 处理。"""
    cdef DataFrame target = self if inplace else _copy_impl(self)
    cdef list names = []
    cdef str name
    cdef BaseColumn col
    for name, col in target._columns.items():
        if col.dtype == TYPE_INT or col.dtype == TYPE_DOUBLE:
            names.append(name)
    for name in names:
        _clip_column_impl(target, name, lower, upper, False)
    if inplace:
        return self
    return target

cdef _where_column_impl(DataFrame self, str col_name, str cond_col, object other):
    cdef BaseColumn src = self._columns[col_name]
    cdef BaseColumn cond = self._columns[cond_col]
    cdef bint src_is_2d = src.is_2d and src.num_cols > 1
    cdef bint cond_is_2d = cond.is_2d and cond.num_cols > 1
    if src_is_2d and cond_is_2d:
        raise ValueError("where does not support 2D condition column; use a 1D condition")
    if src.dtype == TYPE_DOUBLE and cond.dtype == TYPE_INT:
        return (<DoubleColumn>src).where(<IntColumn>cond, <double>other)
    elif src.dtype == TYPE_INT and cond.dtype == TYPE_INT:
        return (<IntColumn>src).where(<IntColumn>cond, <int>other)
    raise TypeError(f"where not supported for column type {self._type_name(src)}")

cdef _mask_column_impl(DataFrame self, str col_name, str cond_col, object other):
    cdef BaseColumn src = self._columns[col_name]
    cdef BaseColumn cond = self._columns[cond_col]
    cdef bint src_is_2d = src.is_2d and src.num_cols > 1
    cdef bint cond_is_2d = cond.is_2d and cond.num_cols > 1
    if src_is_2d and cond_is_2d:
        raise ValueError("mask does not support 2D condition column; use a 1D condition")
    if src.dtype == TYPE_DOUBLE and cond.dtype == TYPE_INT:
        return (<DoubleColumn>src).mask(<IntColumn>cond, <double>other)
    elif src.dtype == TYPE_INT and cond.dtype == TYPE_INT:
        return (<IntColumn>src).mask(<IntColumn>cond, <int>other)
    raise TypeError(f"mask not supported for column type {self._type_name(src)}")

cdef _where_row_impl(DataFrame self, IntColumn mask, object fill):
    cdef Py_ssize_t n = self.df_length
    if mask.length != n:
        raise ValueError(f"Mask length {mask.length} != DataFrame length {n}")
    cdef DataFrame result = DataFrame()
    result.df_length = n
    result.index = self.index
    cdef str name
    cdef BaseColumn col
    for name, col in self._columns.items():
        result._columns[name] = col.where_mask(mask, fill)
    return result

cdef _isnull_impl(DataFrame self):
    cdef DataFrame result = DataFrame()
    result.df_length = self.df_length
    result.index = self.index
    cdef str name
    cdef BaseColumn col
    cdef IntColumn new_col
    cdef Py_ssize_t i, total
    cdef bint is2d
    cdef Py_ssize_t c_out

    cdef double* d_ptr
    cdef int* out_ptr
    cdef Py_ssize_t* offset_ptr
    cdef CategoryColumn cat_col

    for name, col in self._columns.items():
        is2d, c_out, _ = _col_shape_probe(col)
        total = col.length

        new_col = IntColumn(max(total, 1), cols=<Py_ssize_t>c_out)
        new_col.length = total
        out_ptr = new_col.data_c

        if col.dtype == TYPE_DOUBLE:
            d_ptr = (<DoubleColumn>col).data_c
            with nogil:
                for i in range(total):
                    out_ptr[i] = 1 if isnan(d_ptr[i]) else 0
        elif col.dtype == TYPE_CHAR256:
            offset_ptr = (<AsciiColumn>col).offsets_c
            with nogil:
                for i in range(total):
                    out_ptr[i] = 1 if offset_ptr[i + 1] == offset_ptr[i] else 0
        elif col.dtype == TYPE_UCS2:
            offset_ptr = (<UCS2Column>col).offsets_c
            with nogil:
                for i in range(total):
                    out_ptr[i] = 1 if offset_ptr[i + 1] == offset_ptr[i] else 0
        elif col.dtype == TYPE_CATEGORY:
            cat_col = <CategoryColumn>col
            for i in range(total):
                out_ptr[i] = 1 if cat_col._resolve(i) == '' else 0
        else:
            memset(out_ptr, 0, total * sizeof(int))
        result._columns[name] = new_col
    return result

cdef _notna_impl(DataFrame self):
    """df.notna：df.isnull 掩码取反。2D 列保持 cols 维度。"""
    cdef DataFrame result = _isnull_impl(self)
    cdef str name
    cdef BaseColumn col
    cdef IntColumn mask_col
    cdef int* p
    cdef Py_ssize_t i
    for name, col in result._columns.items():
        mask_col = <IntColumn>col
        p = mask_col.data_c
        with nogil:
            for i in range(mask_col.length):
                p[i] = 1 - p[i]
    return result

cdef _fillna_impl(DataFrame self, object value, col_names):
    """【语义修复】：参数 value 必须是 object，以支持字符串等各种类型填充。
    2D 列：数值类型使用 flat 迭代（isnan 逐元素检测，与布局无关）；
    字符串类型按 2D 子列边界逐子列填充。"""
    cdef DataFrame result = _copy_impl(self)
    cdef str name
    cdef BaseColumn col
    cdef Py_ssize_t i, total, rows, cols, j, base
    cdef bint is2d
    cdef double* d_ptr
    cdef double d_val
    cdef AsciiColumn ascii_col
    cdef UCS2Column ucs2_col
    cdef str s_val

    if col_names is None:
        col_names = list(self._columns.keys())
    for name in col_names:
        if name not in result._columns: continue
        col = result._columns[name]
        total = col.length
        is2d, cols, rows = _col_shape_probe(col)

        if col.dtype == TYPE_DOUBLE and isinstance(value, (int, float)):
            d_ptr = (<DoubleColumn>col).data_c
            d_val = <double>value
            with nogil:
                for i in range(total):
                    if isnan(d_ptr[i]):
                        d_ptr[i] = d_val

        elif col.dtype == TYPE_CHAR256 and isinstance(value, str):
            ascii_col = <AsciiColumn>col
            s_val = value
            for i in range(total):
                if ascii_col._get_str(i) == '':
                    ascii_col._set_item(i, s_val)
        elif col.dtype == TYPE_UCS2 and isinstance(value, str):
            ucs2_col = <UCS2Column>col
            s_val = value
            for i in range(total):
                if ucs2_col._get_str(i) == '':
                    ucs2_col._set_item(i, s_val)
    return result

cdef _dropna_impl(DataFrame self):
    if self.df_length == 0: return _copy_impl(self)
    cdef Py_ssize_t n = self.df_length

    cdef char* keep_mask = <char*>aligned_malloc(n * sizeof(char))
    if not keep_mask: raise MemoryError()
    memset(keep_mask, 1, n)

    cdef Py_ssize_t i, j, k, keep_count = 0
    cdef double* d_data
    cdef str name
    cdef BaseColumn col
    cdef Py_ssize_t* keep_idx = NULL
    cdef Py_ssize_t start, end, str_len
    cdef bint is2d
    cdef Py_ssize_t c_out, r_out
    cdef Py_ssize_t* offsets_ptr

    try:
        for name, col in self._columns.items():
            if col.dtype == TYPE_DOUBLE:
                d_data = (<DoubleColumn>col).data_c
                _probe_shape(col, &is2d, &c_out, &r_out)
                with nogil:
                    if is2d and c_out > 1:
                        for k in range(r_out):
                            if not keep_mask[k]: continue
                            for j in range(c_out):
                                if isnan(d_data[j * r_out + k]):
                                    keep_mask[k] = 0
                                    break
                    else:
                        for i in range(n):
                            if keep_mask[i] and isnan(d_data[i]):
                                keep_mask[i] = 0

            elif col.dtype == TYPE_CHAR256:
                offsets_ptr = (<AsciiColumn>col).offsets_c
                with nogil:
                    for i in range(n):
                        if keep_mask[i]:
                            str_len = offsets_ptr[i + 1] - offsets_ptr[i]
                            if str_len == 0:
                                keep_mask[i] = 0

            elif col.dtype == TYPE_UCS2:
                offsets_ptr = (<UCS2Column>col).offsets_c
                with nogil:
                    for i in range(n):
                        if keep_mask[i]:
                            if offsets_ptr[i] == offsets_ptr[i + 1]:
                                keep_mask[i] = 0

        for i in range(n):
            keep_count += keep_mask[i]

        if keep_count == n:
            return _copy_impl(self)
        if keep_count == 0:
            return DataFrame()

        keep_idx = <Py_ssize_t*>aligned_malloc(max(keep_count, 1) * sizeof(Py_ssize_t))
        if not keep_idx: raise MemoryError()

        j = 0
        for i in range(n):
            if keep_mask[i]:
                keep_idx[j] = i
                j += 1

        return self._take(keep_idx, keep_count)
    finally:
        if keep_mask: os_aligned_free(keep_mask)
        if keep_idx: os_aligned_free(keep_idx)

cdef _head_impl(DataFrame self, Py_ssize_t n):
    if n < 0: raise ValueError("n must be non-negative")
    n = min(n, self.df_length)
    if n == 0:
        result = DataFrame()
        for name, col in self._columns.items():
            result._columns[name] = col.__class__()
        return result

    cdef Py_ssize_t* idx_c = <Py_ssize_t*>aligned_malloc(max(n, 1) * sizeof(Py_ssize_t))
    if not idx_c: raise MemoryError()
    cdef Py_ssize_t i
    try:
        for i in range(n): idx_c[i] = i
        return self._take(idx_c, n)
    finally:
        os_aligned_free(idx_c)

cdef _tail_impl(DataFrame self, Py_ssize_t n):
    if n < 0: raise ValueError("n must be non-negative")
    n = min(n, self.df_length)
    if n == 0: return DataFrame()
    cdef Py_ssize_t start = self.df_length - n
    cdef Py_ssize_t* idx_c = <Py_ssize_t*>aligned_malloc(max(n, 1) * sizeof(Py_ssize_t))
    if not idx_c: raise MemoryError()
    cdef Py_ssize_t i
    try:
        for i in range(n): idx_c[i] = start + i
        return self._take(idx_c, n)
    finally:
        os_aligned_free(idx_c)

cdef _sample_impl(DataFrame self, object n, object frac, bint replace, int seed):
    if seed >= 0:
        _xorshift128plus_seed(<unsigned long long>seed)
    else:
        _xorshift128plus_seed(<unsigned long long>time(NULL) ^ <unsigned long long>id(self))
    cdef Py_ssize_t total = self.df_length
    if total == 0:
        return DataFrame()

    cdef Py_ssize_t sample_size
    if n is not None:
        sample_size = <Py_ssize_t>n
    elif frac is not None:
        sample_size = <Py_ssize_t>(total * <double>frac)
        if sample_size == 0: sample_size = 1
    else:
        raise ValueError("Must specify either 'n' or 'frac'")

    if not replace and sample_size > total:
        raise ValueError(f"Sample size {sample_size} > population {total}")

    cdef Py_ssize_t* idx_c = <Py_ssize_t*>aligned_malloc(max(sample_size, 1) * sizeof(Py_ssize_t))
    if not idx_c: raise MemoryError()
    cdef Py_ssize_t i, j, tmp
    cdef Py_ssize_t* shuffle_buf

    if replace:
        for i in range(sample_size):
            idx_c[i] = <Py_ssize_t>(_random_uniform() * total)
            if idx_c[i] >= total:
                idx_c[i] = total - 1
    else:
        shuffle_buf = <Py_ssize_t*>aligned_malloc(total * sizeof(Py_ssize_t))
        if not shuffle_buf:
            os_aligned_free(idx_c)
            raise MemoryError()
        for i in range(total):
            shuffle_buf[i] = i
        for i in range(sample_size):
            j = i + <Py_ssize_t>(_random_uniform() * (total - i))
            if j >= total: j = total - 1
            tmp = shuffle_buf[i]
            shuffle_buf[i] = shuffle_buf[j]
            shuffle_buf[j] = tmp
            idx_c[i] = shuffle_buf[i]
        os_aligned_free(shuffle_buf)

    cdef DataFrame result = self._take(idx_c, sample_size)
    os_aligned_free(idx_c)
    return result

cdef _nsmallest_impl(DataFrame self, Py_ssize_t n, str col_name):
    cdef DataFrame sorted_df = _sort_values_impl(self, col_name, False, False)
    return _head_impl(sorted_df, n)

cdef _nlargest_impl(DataFrame self, Py_ssize_t n, str col_name):
    cdef DataFrame sorted_df = _sort_values_impl(self, col_name, True, False)
    return _head_impl(sorted_df, n)

cdef _pipe_impl(DataFrame self, object func, tuple args, dict kwargs):
    return func(self, *args, **kwargs)

cdef _assign_impl(DataFrame self, dict kwargs):
    cdef DataFrame result = _copy_impl(self)
    for name, value in kwargs.items():
        result[name] = value
    return result

cdef _select_dtypes_impl(DataFrame self, list include_types, list exclude_types):
    cdef DataFrame result = DataFrame()
    cdef dict dtype_map = {TYPE_INT: 'int', TYPE_DOUBLE: 'float', TYPE_CHAR256: 'char256',
                           TYPE_UCS2: 'ucs2', TYPE_DATE: 'date',
                           TYPE_BOOL: 'bool', TYPE_CATEGORY: 'category',
                           TYPE_DATETIME: 'datetime', TYPE_TIME: 'time'}
    for name, col in self._columns.items():
        tn = dtype_map.get(col.dtype, 'unknown')
        if include_types is not None and tn not in include_types:
            continue
        if exclude_types is not None and tn in exclude_types:
            continue
        result._columns[name] = col
    result.df_length = self.df_length
    result.index = self.index
    return result

cdef _astype_impl(DataFrame self, dict dtype_dict):
    cdef DataFrame result = DataFrame()
    cdef str col_name, target
    for col_name, col_obj in self._columns.items():
        if col_name in dtype_dict:
            target = dtype_dict[col_name]
            result._columns[col_name] = col_obj.astype(target)
        else:
            result._columns[col_name] = col_obj.copy() if hasattr(col_obj, 'copy') else col_obj
    result.df_length = self.df_length
    if self.index is not None:
        if self.index.index_type == INDEX_RANGE:
            result.index = RangeIndex((<RangeIndex>self.index).start,
                                     self.index.length,
                                     (<RangeIndex>self.index).step)
        elif self.index.index_type == INDEX_INT:
            result.index = self.index.copy()
        else:
            result.index = self.index.copy()
    return result

cdef _resolve_cols_impl(DataFrame self, tuple col_names):
    cdef list cols = []
    cdef str name
    for name in col_names:
        if name not in self._columns:
            raise KeyError(f"Column '{name}' not found")
        cols.append(self._columns[name])
    return cols

cdef _add_impl(DataFrame self, tuple col_names):
    if len(col_names) < 2: raise ValueError("Need at least 2 column names")
    cdef list cols = _resolve_cols_impl(self, col_names)
    cdef BaseColumn result = cols[0].add(cols[1])
    cdef Py_ssize_t j
    for j in range(2, len(cols)):
        result = result.add(cols[j])
    return result

cdef _sub_impl(DataFrame self, tuple col_names):
    if len(col_names) < 2: raise ValueError("Need at least 2 column names")
    cdef list cols = _resolve_cols_impl(self, col_names)
    cdef BaseColumn result = cols[0].sub(cols[1])
    cdef Py_ssize_t j
    for j in range(2, len(cols)):
        result = result.sub(cols[j])
    return result

cdef _mul_impl(DataFrame self, tuple col_names):
    if len(col_names) < 2: raise ValueError("Need at least 2 column names")
    cdef list cols = _resolve_cols_impl(self, col_names)
    cdef BaseColumn result = cols[0].mul(cols[1])
    cdef Py_ssize_t j
    for j in range(2, len(cols)):
        result = result.mul(cols[j])
    return result

cdef _div_impl(DataFrame self, tuple col_names):
    if len(col_names) < 2: raise ValueError("Need at least 2 column names")
    cdef list cols = _resolve_cols_impl(self, col_names)
    cdef BaseColumn result = cols[0].div(cols[1])
    cdef Py_ssize_t j
    for j in range(2, len(cols)):
        result = result.div(cols[j])
    return result

cdef _reshape_impl(DataFrame self, Py_ssize_t rows, Py_ssize_t cols):
    cdef Py_ssize_t num_cols = len(self._columns)
    cdef Py_ssize_t total_elements = self.df_length * num_cols
    if total_elements != rows * cols:
        raise ValueError(f"Cannot reshape {total_elements} elements into ({rows}, {cols})")

    cdef DataFrame result = DataFrame()
    result.df_length = rows
    result.index = RangeIndex(0, rows)
    cdef list col_names = [chr(65 + i) if i < 26 else f"C{i}" for i in range(cols)]

    cdef list all_cols_keys = [_col_to_keys(cell) for cell in self._columns.values()]
    cdef list all_cells = [None] * total_elements
    cdef Py_ssize_t i, j, ptr = 0

    for i in range(self.df_length):
        for j in range(num_cols):
            all_cells[ptr] = all_cols_keys[j][i]
            ptr += 1

    cdef Py_ssize_t c_out = 1
    for cell in all_cells:
        if isinstance(cell, tuple):
            c_out = max(c_out, len(cell))

    cdef DoubleColumn col
    cdef Py_ssize_t r_idx, c_idx, k

    for c_idx in range(cols):
        if c_out > 1:
            col = DoubleColumn(max(rows * c_out, 1), cols=c_out)
            col.length = rows * c_out
            col.num_cols = <int>c_out
            col.is_2d = True
            col.owns_data = True
            for r_idx in range(rows):
                cell = all_cells[r_idx * cols + c_idx]
                if isinstance(cell, tuple):
                    for k in range(c_out):
                        col.data_c[k * rows + r_idx] = <double>cell[k]
                else:
                    col.data_c[r_idx] = <double>cell
                    for k in range(1, c_out): col.data_c[k * rows + r_idx] = NAN
        else:
            col = DoubleColumn(max(rows, 1))
            col.length = rows
            for r_idx in range(rows):
                col.data_c[r_idx] = <double>all_cells[r_idx * cols + c_idx]
        result._columns[col_names[c_idx]] = col

    return result

cdef _T_impl(DataFrame self):
    cdef Py_ssize_t rows = self.df_length
    cdef Py_ssize_t cols = len(self._columns)
    if rows == 0 or cols == 0: return DataFrame()

    cdef list col_names = list(self._columns.keys())
    cdef DataFrame result = DataFrame()
    result.df_length = cols
    result.index = RangeIndex(0, cols)

    cdef Py_ssize_t i, j, k, c_out = 1, c_j
    cdef bint is2d_j
    cdef DoubleColumn new_col
    cdef BaseColumn sub

    for j in range(cols):
        _probe_shape(self._columns[col_names[j]], &is2d_j, &c_j, &i)
        if c_j > c_out: c_out = c_j

    cdef double** d_ptrs = <double**>malloc(cols * sizeof(double*))
    cdef int** i_ptrs = <int**>malloc(cols * sizeof(int*))
    cdef Py_ssize_t* c_arr = <Py_ssize_t*>malloc(cols * sizeof(Py_ssize_t))

    for j in range(cols):
        sub = self._columns[col_names[j]]
        _probe_shape(sub, &is2d_j, &c_j, &i)
        c_arr[j] = c_j
        d_ptrs[j] = (<DoubleColumn>sub).data_c if sub.dtype == TYPE_DOUBLE else NULL
        i_ptrs[j] = (<IntColumn>sub).data_c if sub.dtype == TYPE_INT else NULL

    try:
        for i in range(rows):
            if c_out > 1:
                new_col = DoubleColumn(max(cols * c_out, 1), cols=c_out)
                new_col.length = cols * c_out
                new_col.num_cols = <int>c_out
                new_col.is_2d = True
                new_col.owns_data = True

                with nogil:
                    for j in range(cols):
                        if d_ptrs[j] != NULL or i_ptrs[j] != NULL:
                            for k in range(c_out):
                                if k < c_arr[j]:
                                    new_col.data_c[k * cols + j] = d_ptrs[j][k * rows + i] if d_ptrs[j] != NULL else <double>i_ptrs[j][k * rows + i]
                                else:
                                    new_col.data_c[k * cols + j] = NAN
                        else:
                            for k in range(c_out): new_col.data_c[k * cols + j] = NAN
            else:
                new_col = DoubleColumn(max(cols, 1))
                new_col.length = cols
                with nogil:
                    for j in range(cols):
                        if d_ptrs[j] != NULL: new_col.data_c[j] = d_ptrs[j][i]
                        elif i_ptrs[j] != NULL: new_col.data_c[j] = <double>i_ptrs[j][i]
                        else: new_col.data_c[j] = NAN

            result._columns[col_names[i] if i < cols else str(i)] = new_col
    finally:
        free(d_ptrs)
        free(i_ptrs)
        free(c_arr)

    return result

cdef _abs_impl(DataFrame self):
    cdef DataFrame result = DataFrame()
    result.df_length = self.df_length
    result.index = self.index
    cdef str name
    cdef BaseColumn col
    for name, col in self._columns.items():
        if col.dtype == TYPE_INT or col.dtype == TYPE_DOUBLE:
            result._columns[name] = col.abs()
        else:
            result._columns[name] = col.copy()
    return result

cdef _round_impl(DataFrame self, int decimals):
    cdef DataFrame result = DataFrame()
    result.df_length = self.df_length
    result.index = self.index
    cdef str name
    cdef BaseColumn col
    for name, col in self._columns.items():
        if col.dtype == TYPE_DOUBLE:
            result._columns[name] = col.round(decimals)
        else:
            result._columns[name] = col.copy()
    return result

cdef _all_impl(DataFrame self):
    cdef Py_ssize_t n = self.df_length
    cdef dict result = {}
    cdef str name
    cdef BaseColumn col
    cdef Py_ssize_t i, total
    cdef bint all_true

    cdef int* i_ptr
    cdef double* d_ptr
    cdef unsigned char* bool_ptr

    for name, col in self._columns.items():
        total = col.length
        all_true = True
        if col.dtype == TYPE_INT:
            i_ptr = (<IntColumn>col).data_c
            for i in range(total):
                if i_ptr[i] == 0:
                    all_true = False
                    break
        elif col.dtype == TYPE_DOUBLE:
            d_ptr = (<DoubleColumn>col).data_c
            for i in range(total):
                if d_ptr[i] == 0.0 or isnan(d_ptr[i]):
                    all_true = False
                    break
        elif col.dtype == TYPE_BOOL:
            bool_ptr = (<BoolColumn>col).data_c
            for i in range(total):
                if bool_ptr[i] == 0:
                    all_true = False
                    break
        result[name] = all_true
    return _agg_series(result)

cdef _any_impl(DataFrame self):
    cdef Py_ssize_t n = self.df_length
    cdef dict result = {}
    cdef str name
    cdef BaseColumn col
    cdef Py_ssize_t i, total
    cdef bint any_true

    cdef int* i_ptr
    cdef double* d_ptr
    cdef unsigned char* bool_ptr

    for name, col in self._columns.items():
        total = col.length
        any_true = False
        if col.dtype == TYPE_INT:
            i_ptr = (<IntColumn>col).data_c
            for i in range(total):
                if i_ptr[i] != 0:
                    any_true = True
                    break
        elif col.dtype == TYPE_DOUBLE:
            d_ptr = (<DoubleColumn>col).data_c
            for i in range(total):
                if d_ptr[i] != 0.0 and not isnan(d_ptr[i]):
                    any_true = True
                    break
        elif col.dtype == TYPE_BOOL:
            bool_ptr = (<BoolColumn>col).data_c
            for i in range(total):
                if bool_ptr[i] != 0:
                    any_true = True
                    break
        result[name] = any_true
    return _agg_series(result)

cdef _pct_change_impl(DataFrame self, int periods=1):
    cdef DataFrame result = DataFrame()
    result.df_length = self.df_length
    result.index = self.index
    cdef str name
    cdef BaseColumn col
    cdef Py_ssize_t i, k, total, base, p
    cdef bint is2d
    cdef Py_ssize_t c_out, r_out
    cdef DoubleColumn dbl_col
    cdef double prev_val, curr_val
    cdef int* i_ptr
    cdef double* d_ptr
    cdef double* d_out

    for name, col in self._columns.items():
        _probe_shape(col, &is2d, &c_out, &r_out)
        total = col.length
        if col.dtype == TYPE_INT or col.dtype == TYPE_DOUBLE:
            dbl_col = DoubleColumn(max(total, 1), cols=<Py_ssize_t>c_out)
            dbl_col.length = total
            if is2d:
                dbl_col.num_cols = <int>c_out
                dbl_col.is_2d = True
                dbl_col.owns_data = True
            d_out = dbl_col.data_c

            i_ptr = (<IntColumn>col).data_c if col.dtype == TYPE_INT else NULL
            d_ptr = (<DoubleColumn>col).data_c if col.dtype == TYPE_DOUBLE else NULL

            with nogil:
                if periods >= 0:
                    p = periods
                    for k in range(c_out):
                        base = k * r_out
                        for i in range(r_out):
                            if i < p:
                                d_out[base + i] = NAN
                            else:
                                prev_val = <double>i_ptr[base + i - p] if i_ptr != NULL else d_ptr[base + i - p]
                                curr_val = <double>i_ptr[base + i] if i_ptr != NULL else d_ptr[base + i]
                                d_out[base + i] = (curr_val - prev_val) / fabs(prev_val) if prev_val != 0.0 else NAN
                else:
                    p = -periods
                    for k in range(c_out):
                        base = k * r_out
                        for i in range(r_out):
                            if i + p >= r_out:
                                d_out[base + i] = NAN
                            else:
                                prev_val = <double>i_ptr[base + i + p] if i_ptr != NULL else d_ptr[base + i + p]
                                curr_val = <double>i_ptr[base + i] if i_ptr != NULL else d_ptr[base + i]
                                d_out[base + i] = (curr_val - prev_val) / fabs(prev_val) if prev_val != 0.0 else NAN

            result._columns[name] = dbl_col
        else:
            result._columns[name] = col.copy()

    return result

cdef _replace_impl(DataFrame self, object to_replace, object value):
    cdef DataFrame result = _copy_impl(self)
    cdef str name
    cdef BaseColumn col
    cdef Py_ssize_t i, total

    cdef int i_target, i_val
    cdef double d_target, d_val
    cdef int* i_ptr
    cdef double* d_ptr
    cdef UCS2Column new_col

    for name, col in result._columns.items():
        total = col.length
        if col.dtype == TYPE_INT and isinstance(to_replace, (int, float)) and isinstance(value, (int, float)):
            i_target = <int>to_replace
            i_val = <int>value
            i_ptr = (<IntColumn>col).data_c
            with nogil:
                for i in range(total):
                    if i_ptr[i] == i_target:
                        i_ptr[i] = i_val

        elif col.dtype == TYPE_DOUBLE and isinstance(to_replace, (int, float)) and isinstance(value, (int, float)):
            d_target = <double>to_replace
            d_val = <double>value
            d_ptr = (<DoubleColumn>col).data_c
            with nogil:
                for i in range(total):
                    if d_target != d_target:
                        if isnan(d_ptr[i]): d_ptr[i] = d_val
                    elif d_ptr[i] == d_target:
                        d_ptr[i] = d_val

        elif col.dtype in (TYPE_CHAR256, TYPE_UCS2) and isinstance(to_replace, str):
            new_col = UCS2Column(max(total, 1))
            for i in range(total):
                if col[i] == to_replace:
                    new_col.append(str(value))
                else:
                    new_col.append(str(col[i]))
            result._columns[name] = new_col
    return result

cdef _ffill_impl(DataFrame self):
    cdef DataFrame result = DataFrame()
    result.df_length = self.df_length
    result.index = self.index
    cdef str name
    cdef BaseColumn col
    for name, col in self._columns.items():
        if col.dtype == TYPE_DOUBLE:
            result._columns[name] = col.ffill()
        else:
            result._columns[name] = col.copy()
    return result

cdef _bfill_impl(DataFrame self):
    cdef DataFrame result = DataFrame()
    result.df_length = self.df_length
    result.index = self.index
    cdef str name
    cdef BaseColumn col
    for name, col in self._columns.items():
        if col.dtype == TYPE_DOUBLE:
            result._columns[name] = col.bfill()
        else:
            result._columns[name] = col.copy()
    return result

cdef _pivot_impl(DataFrame self, str index, str columns, str values):
    cdef Py_ssize_t n = self.df_length

    cdef list idx_keys = _col_to_keys(self._columns[index])
    cdef list col_keys = _col_to_keys(self._columns[columns])
    cdef list val_keys = _col_to_keys(self._columns[values])

    cdef dict pivot_vals = {}
    cdef dict col_dict = {}
    cdef list unique_cols = []
    cdef list unique_idx = []
    cdef object idx_val, col_val, val_val, got
    cdef Py_ssize_t i, ii, j, u, c_val
    cdef bint is2d_val
    cdef UCS2Column idx_result
    cdef DoubleColumn val_result

    for i in range(n):
        idx_val = idx_keys[i]
        col_val = col_keys[i]
        val_val = val_keys[i]

        if idx_val not in pivot_vals:
            pivot_vals[idx_val] = {}
            unique_idx.append(idx_val)
        pivot_vals[idx_val][col_val] = val_val

        if col_val not in col_dict:
            col_dict[col_val] = len(unique_cols)
            unique_cols.append(col_val)

    cdef DataFrame result = DataFrame()
    result.df_length = len(unique_idx)
    result.index = RangeIndex(0, len(unique_idx))

    idx_result = UCS2Column(max(len(unique_idx), 1))
    for iv in unique_idx:
        idx_result.append(str(iv))
    result._columns[index] = idx_result

    u = len(unique_idx)
    _probe_shape(self._columns[values], &is2d_val, &c_val, &j)

    if is2d_val and c_val > 1:
        for cv in unique_cols:
            val_result = DoubleColumn(max(u * c_val, 1), cols=c_val)
            val_result.length = u * c_val
            for ii in range(u):
                got = pivot_vals[unique_idx[ii]].get(cv)
                if got is None:
                    for j in range(c_val): val_result.data_c[j * u + ii] = NAN
                else:
                    for j in range(c_val): val_result.data_c[j * u + ii] = <double>got[j]
            result._columns[str(cv)] = val_result
    else:
        for cv in unique_cols:
            val_result = DoubleColumn(max(u, 1))
            val_result.length = u
            for ii in range(u):
                val_result.data_c[ii] = <double>pivot_vals[unique_idx[ii]].get(cv, NAN)
            result._columns[str(cv)] = val_result

    return result

cdef _melt_impl(DataFrame self, list id_vars=None, list value_vars=None, str var_name='variable', str value_name='value'):
    cdef Py_ssize_t n = self.df_length
    cdef list all_cols = list(self._columns.keys())
    if id_vars is None: id_vars = []
    if value_vars is None: value_vars = [c for c in all_cols if c not in id_vars]

    cdef DataFrame result = DataFrame()
    cdef Py_ssize_t total = n * len(value_vars)
    cdef Py_ssize_t vi, i, j, c_out, cc, base
    cdef bint is2d
    cdef UCS2Column new_col
    cdef DoubleColumn new_val
    cdef BaseColumn src_val
    cdef double* d_ptr
    cdef int* i_ptr
    result.df_length = total
    result.index = RangeIndex(0, total)

    for iv in id_vars:
        src_keys = _col_to_keys(self._columns[iv])
        new_col = UCS2Column(max(total, 1))
        for vi in range(len(value_vars)):
            for i in range(n):
                new_col.append(str(src_keys[i]))
        result._columns[iv] = new_col

    new_var = UCS2Column(max(total, 1))
    for vi in range(len(value_vars)):
        for i in range(n):
            new_var.append(str(value_vars[vi]))
    result._columns[var_name] = new_var

    c_out = 1
    for vn in value_vars:
        _probe_shape(self._columns[vn], &is2d, &cc, &i)
        if is2d and cc > c_out: c_out = cc

    new_val = DoubleColumn(max(total * c_out, 1), cols=c_out)
    new_val.length = total * c_out
    if c_out > 1:
        new_val.num_cols = <int>c_out
        new_val.is_2d = True
        new_val.owns_data = True
        for i in range(total * c_out):
            new_val.data_c[i] = NAN

    for vi, vn in enumerate(value_vars):
        src_val = self._columns[vn]
        _probe_shape(src_val, &is2d, &cc, &i)
        if src_val.dtype == TYPE_DOUBLE or src_val.dtype == TYPE_INT:
            d_ptr = (<DoubleColumn>src_val).data_c if src_val.dtype == TYPE_DOUBLE else NULL
            i_ptr = (<IntColumn>src_val).data_c if src_val.dtype == TYPE_INT else NULL

            with nogil:
                if is2d and cc > 1:
                    for j in range(cc):
                        base = j * total + vi * n
                        for i in range(n):
                            new_val.data_c[base + i] = d_ptr[j * n + i] if d_ptr != NULL else <double>i_ptr[j * n + i]
                else:
                    base = vi * n
                    for i in range(n):
                        new_val.data_c[base + i] = d_ptr[i] if d_ptr != NULL else <double>i_ptr[i]
        else:
            for i in range(n):
                new_val.data_c[vi * n + i] = NAN

    result._columns[value_name] = new_val
    return result

cdef _expanding_sum_impl(DataFrame self, str col_name):
    cdef BaseColumn col = self._columns[col_name]
    cdef bint is2d
    cdef Py_ssize_t c_out, r_out, total, j, base, i
    _probe_shape(col, &is2d, &c_out, &r_out)
    total = col.length

    cdef DoubleColumn res = DoubleColumn(max(total, 1), cols=c_out)
    res.length = total
    if is2d:
        res.num_cols = <int>c_out
        res.is_2d = True
        res.owns_data = True

    cdef double acc
    cdef double* d_out = res.data_c
    cdef double* d_in
    cdef int* i_in

    if col.dtype == TYPE_DOUBLE:
        d_in = (<DoubleColumn>col).data_c
        with nogil:
            for j in range(c_out):
                base = j * r_out
                acc = 0.0
                for i in range(r_out):
                    if not isnan(d_in[base + i]):
                        acc += d_in[base + i]
                    d_out[base + i] = acc
    elif col.dtype == TYPE_INT:
        i_in = (<IntColumn>col).data_c
        with nogil:
            for j in range(c_out):
                base = j * r_out
                acc = 0.0
                for i in range(r_out):
                    acc += <double>i_in[base + i]
                    d_out[base + i] = acc
    else:
        raise TypeError(f"Expanding sum not supported for {col.dtype}")
    return res

cdef _ewm_mean_impl(DataFrame self, str col_name, double span):
    cdef BaseColumn col = self._columns[col_name]
    cdef bint is2d
    cdef Py_ssize_t c_out, r_out, total, j, base, i
    _probe_shape(col, &is2d, &c_out, &r_out)
    total = col.length

    cdef DoubleColumn res = DoubleColumn(max(total, 1), cols=c_out)
    res.length = total
    if is2d:
        res.num_cols = <int>c_out
        res.is_2d = True
        res.owns_data = True

    cdef double alpha = 2.0 / (span + 1.0)
    cdef double acc
    cdef double* d_out = res.data_c
    cdef double* d_in
    cdef int* i_in

    if col.dtype == TYPE_DOUBLE:
        d_in = (<DoubleColumn>col).data_c
        with nogil:
            for j in range(c_out):
                base = j * r_out
                acc = 0.0
                for i in range(r_out):
                    if not isnan(d_in[base + i]):
                        acc = acc * (1.0 - alpha) + d_in[base + i] * alpha
                    d_out[base + i] = acc
    elif col.dtype == TYPE_INT:
        i_in = (<IntColumn>col).data_c
        with nogil:
            for j in range(c_out):
                base = j * r_out
                acc = 0.0
                for i in range(r_out):
                    acc = acc * (1.0 - alpha) + <double>i_in[base + i] * alpha
                    d_out[base + i] = acc
    return res


# ============================================================
#  reshape：explode / stack / unstack
# ============================================================

cdef _explode_impl(DataFrame self, str col_name):
    """DataFrame.explode：将 col_name 中 list/tuple 元素纵向展开为多行，
    其余列按行重复对齐；空 list/tuple 行被丢弃，标量行保持原样。
    源列可以是 BaseColumn（Typed 列只接受标量，通常用于字符串形态），
    也可以直接用 Python list（含 list 元素）赋给 df[col_name]。"""
    cdef Py_ssize_t n = self.df_length
    cdef Py_ssize_t i, j, k
    cdef long total
    cdef str name
    cdef BaseColumn scol
    cdef object src_val

    if col_name not in self._columns:
        raise KeyError(f"Column '{col_name}' not found")
    src = self._columns[col_name]
    scol = src if isinstance(src, BaseColumn) else None
    cdef list col_names = list(self._columns.keys())

    cdef list counts = [0] * n
    total = 0
    for i in range(n):
        src_val = scol[i] if scol is not None else src[i]
        if isinstance(src_val, (list, tuple)):
            counts[i] = len(src_val)
        else:
            counts[i] = 1
        total += counts[i]

    cdef DataFrame result = DataFrame()
    result.df_length = total
    result.index = RangeIndex(0, total)

    # explode 列：逐元素展开
    cdef list explode_vals = [None] * total
    k = 0
    for i in range(n):
        src_val = scol[i] if scol is not None else src[i]
        if isinstance(src_val, (list, tuple)):
            for j in range(len(src_val)):
                explode_vals[k] = src_val[j]
                k += 1
        else:
            explode_vals[k] = src_val
            k += 1
    result._columns[col_name] = _column_from_list_impl(result, col_name, explode_vals)

    # 其余列：按行重复 counts[i] 次
    for name in col_names:
        if name == col_name:
            continue
        other = self._columns[name]
        ocol = other if isinstance(other, BaseColumn) else None
        flat = []
        for i in range(n):
            v = ocol[i] if ocol is not None else other[i]
            for _ in range(counts[i]):
                flat.append(v)
        result._columns[name] = _column_from_list_impl(result, name, flat)
    return result


cdef _stack_impl(DataFrame self):
    """FluidDataFrame.stack：将 2D 列展平为 1D 列，
    index 变为两级 MultiIndex（行号, 子列号，按列主序）。
    1D 列在展平后每个 (row, 子列) 取同一值。
    要求至少存在一个 2D 列；可被 unstack() 逆操作（数值/日期类列）。"""
    cdef Py_ssize_t rows = self.df_length
    cdef Py_ssize_t cols = 0
    cdef Py_ssize_t total, i, j
    cdef bint has2d = False
    cdef str name
    cdef BaseColumn col
    cdef int* p0
    cdef int* p1

    for name, col in self._columns.items():
        _, c, r = _col_shape_probe(col)
        if c > 1:
            has2d = True
            if c > cols:
                cols = c
    if not has2d:
        raise ValueError("stack() requires at least one 2D column")

    total = rows * cols
    cdef IntColumn codes0 = IntColumn(max(total, 1))
    cdef IntColumn codes1 = IntColumn(max(total, 1))
    codes0.length = total
    codes1.length = total
    p0 = codes0.data_c
    p1 = codes1.data_c
    with nogil:
        for i in range(total):
            p0[i] = <int>(i // cols)
            p1[i] = <int>(i % cols)

    cdef DataFrame result = DataFrame()
    result.df_length = total
    result.index = MultiIndex([list(range(rows)), list(range(cols))],
                              [codes0, codes1],
                              [None, None])

    for name, col in self._columns.items():
        cc = (<int>col.num_cols) if (col.is_2d and col.num_cols > 1) else 1
        if cc > 1:
            # 2D 列：列主序天然即展平顺序
            flat = [col[i] for i in range(col.length)]
            if cc < cols:
                # 子列数小于统一 cols 的缺口以 None 填充
                flat = flat + [None] * (rows * (cols - cc))
        else:
            flat = [None] * total
            for j in range(cols):
                base = j * rows
                for i in range(rows):
                    flat[base + i] = col[i]
        result._columns[name] = _column_from_list_impl(result, name, flat)
    return result


cdef _unstack_impl(DataFrame self):
    """FluidDataFrame.unstack：stack() 的逆操作。
    要求 index 为两级 MultiIndex（奇数子列数按列主序），
    将每列 flat buffer 重建为 2D 列（子列数 = level1 长度，行数 = level0 长度）。
    数值 / Date / DateTime / Time 列可逆；字符串 / 布尔列不支持 2D 会抛错。"""
    cdef object idx = self.index
    if not isinstance(idx, MultiIndex) or idx.nlevels != 2:
        raise ValueError("unstack() requires a 2-level MultiIndex index (created by stack())")
    cdef MultiIndex mi = <MultiIndex>idx
    cdef Py_ssize_t rows_out = len(<list>mi.levels[0])
    cdef Py_ssize_t cols_out = len(<list>mi.levels[1])
    cdef Py_ssize_t total = self.df_length
    cdef str name
    cdef BaseColumn col

    cdef DataFrame result = DataFrame()
    result.df_length = rows_out
    result.index = RangeIndex(0, rows_out)

    for name, col in self._columns.items():
        if col.is_2d and col.num_cols > 1:
            raise ValueError("unstack() requires 1D (flat) columns from stack()")
        flat = [col[i] for i in range(total)]
        result._columns[name] = _column_from_2d_list_impl(flat, rows_out, cols_out)
    return result


# ============================================================
#  DataFrame 模块级实现 —— 聚合（agg）
# ============================================================

cdef _numeric_agg_impl(DataFrame self, str method, col_names=None):
    cdef dict result = {}
    cdef str name
    cdef object col
    if col_names is None:
        col_names = list(self._columns.keys())
    for name in col_names:
        if name not in self._columns: continue
        col = self._columns[name]
        if col.dtype == TYPE_INT or col.dtype == TYPE_DOUBLE:
            result[name] = getattr(col, method)()
    return _agg_series(result)

cdef _numeric_agg_arg_impl(DataFrame self, str method, col_names, arg):
    cdef dict result = {}
    cdef str name
    cdef object col
    if col_names is None:
        col_names = list(self._columns.keys())
    for name in col_names:
        if name not in self._columns: continue
        col = self._columns[name]
        if col.dtype == TYPE_INT or col.dtype == TYPE_DOUBLE:
            result[name] = getattr(col, method)(arg)
    return _agg_series(result)

cdef _mode_impl(DataFrame self, col_names=None):
    cdef dict modes = {}
    cdef str name
    cdef object col
    if col_names is None:
        col_names = list(self._columns.keys())
    for name in col_names:
        if name not in self._columns: continue
        col = self._columns[name]
        if col.dtype == TYPE_INT or col.dtype == TYPE_DOUBLE:
            modes[name] = list(col.mode())
    if not modes:
        return DataFrame()
    # 每列独立取众数列表，按最长对齐、不足补 NaN，构造 DataFrame（pandas df.mode() 语义）
    cdef list names = list(modes.keys())
    cdef Py_ssize_t max_len = 0
    cdef list m
    for m in modes.values():
        if len(m) > max_len:
            max_len = len(m)
    cdef list padded
    cdef BaseColumn new_col
    cdef DataFrame result = DataFrame()
    cdef Py_ssize_t i, k
    from fluidframe.core.dataframe import _column_from_list_impl
    for i in range(len(names)):
        m = modes[names[i]]
        if len(m) < max_len:
            padded = [float(v) for v in m] + [float('nan')] * (max_len - len(m))
        else:
            padded = list(m)
        new_col = _column_from_list_impl(None, names[i], padded)
        result._columns[names[i]] = new_col
    result.df_length = max_len
    result.index = RangeIndex(0, max_len)
    return result


cdef _numeric_agg_withcol_impl(DataFrame self, str method, col_names=None):
    cdef dict result = {}
    cdef str name
    cdef object col
    cdef bint is2d
    cdef Py_ssize_t c, r, k, i, offset
    cdef DoubleColumn buf
    cdef int dtype
    cdef int32_t* i_ptr
    cdef double* d_ptr
    cdef double* buf_ptr
    cdef object meth
    cdef list res_list

    if col_names is None:
        col_names = list(self._columns.keys())

    for name in col_names:
        if name not in self._columns:
            continue
        col = self._columns[name]
        dtype = col.dtype

        if dtype == TYPE_INT or dtype == TYPE_DOUBLE:
            is2d, c, r = _col_shape_probe(col)
            if is2d and c > 1:
                res_list = []
                if r > 0:
                    buf = DoubleColumn(r)
                    buf.length = r
                    buf_ptr = buf.data_c
                    meth = getattr(buf, method)

                    if dtype == TYPE_INT:
                        i_ptr = <int32_t*>(<IntColumn>col).data_c
                        for k in range(c):
                            offset = k * r
                            for i in range(r):
                                buf_ptr[i] = <double>i_ptr[offset + i]
                            res_list.append(meth())
                    else:
                        d_ptr = <double*>(<DoubleColumn>col).data_c
                        for k in range(c):
                            offset = k * r
                            for i in range(r):
                                buf_ptr[i] = d_ptr[offset + i]
                            res_list.append(meth())
                else:
                    # 空列边界情况保护：避免 r=0 且 2D 时返回单标量，破坏维度结构
                    for k in range(c):
                        res_list.append(getattr(col, method)())
                result[name] = res_list
            else:
                result[name] = getattr(col, method)()

    return result


cdef _numeric_agg_arg_withcol_impl(DataFrame self, str method, col_names, arg):
    cdef dict result = {}
    cdef str name
    cdef object col
    cdef bint is2d
    cdef Py_ssize_t c, r, k, i, offset
    cdef DoubleColumn buf
    cdef int dtype
    cdef int32_t* i_ptr
    cdef double* d_ptr
    cdef double* buf_ptr
    cdef object meth
    cdef list res_list

    if col_names is None:
        col_names = list(self._columns.keys())

    for name in col_names:
        if name not in self._columns:
            continue
        col = self._columns[name]
        dtype = col.dtype

        if dtype == TYPE_INT or dtype == TYPE_DOUBLE:
            is2d, c, r = _col_shape_probe(col)
            if is2d and c > 1:
                res_list = []
                if r > 0:
                    buf = DoubleColumn(r)
                    buf.length = r
                    buf_ptr = buf.data_c
                    meth = getattr(buf, method)

                    if dtype == TYPE_INT:
                        i_ptr = <int32_t*>(<IntColumn>col).data_c
                        for k in range(c):
                            offset = k * r
                            for i in range(r):
                                buf_ptr[i] = <double>i_ptr[offset + i]
                            res_list.append(meth(arg))
                    else:
                        d_ptr = <double*>(<DoubleColumn>col).data_c
                        for k in range(c):
                            offset = k * r
                            for i in range(r):
                                buf_ptr[i] = d_ptr[offset + i]
                            res_list.append(meth(arg))
                else:
                    for k in range(c):
                        res_list.append(getattr(col, method)(arg))
                result[name] = res_list
            else:
                result[name] = getattr(col, method)(arg)

    return result


cdef _describe_withcol_impl(DataFrame self):
    cdef dict stats = {}
    cdef str name
    cdef object col
    cdef bint is2d
    cdef Py_ssize_t c, r, k, i, offset
    cdef DoubleColumn buf
    cdef int dtype
    cdef int32_t* i_ptr
    cdef double* d_ptr
    cdef double* buf_ptr
    cdef list stats_list

    for name, col in self._columns.items():
        dtype = col.dtype
        if dtype == TYPE_INT or dtype == TYPE_DOUBLE:
            is2d, c, r = _col_shape_probe(col)
            if is2d and c > 1:
                stats_list = []
                if r > 0:
                    buf = DoubleColumn(r)
                    buf.length = r
                    buf_ptr = buf.data_c

                    if dtype == TYPE_INT:
                        i_ptr = <int32_t*>(<IntColumn>col).data_c
                        for k in range(c):
                            offset = k * r
                            for i in range(r):
                                buf_ptr[i] = <double>i_ptr[offset + i]
                            stats_list.append({
                                'count': r,
                                'mean': buf.mean(),
                                'std': buf.std(),
                                'min': buf.min(),
                                '50%': buf.median(),
                                'max': buf.max(),
                            })
                    else:
                        d_ptr = <double*>(<DoubleColumn>col).data_c
                        for k in range(c):
                            offset = k * r
                            for i in range(r):
                                buf_ptr[i] = d_ptr[offset + i]
                            stats_list.append({
                                'count': r,
                                'mean': buf.mean(),
                                'std': buf.std(),
                                'min': buf.min(),
                                '50%': buf.median(),
                                'max': buf.max(),
                            })
                else:
                    for k in range(c):
                        stats_list.append({
                            'count': 0,
                            'mean': col.mean(),
                            'std': col.std(),
                            'min': col.min(),
                            '50%': col.median(),
                            'max': col.max(),
                        })
                stats[name] = stats_list
            else:
                stats[name] = {
                    'count': r,
                    'mean': col.mean(),
                    'std': col.std(),
                    'min': col.min(),
                    '50%': col.median(),
                    'max': col.max(),
                }
    return stats

cdef _count_impl(DataFrame self, col_names=None):
    cdef dict result = {}
    cdef str name
    cdef BaseColumn col
    cdef Py_ssize_t count, i, logical_rows
    cdef int* i_ptr
    cdef double* d_ptr
    cdef unsigned char* bool_ptr
    if col_names is None:
        col_names = list(self._columns.keys())
    for name in col_names:
        if name in self._columns:
            col = self._columns[name]
            # Use logical row count for 2D columns
            if col.is_2d and col.num_cols > 1:
                logical_rows = col._get_rows_c()
            else:
                logical_rows = col.length
            count = 0
            if col.dtype == TYPE_INT:
                # IntColumn has no NaN concept, all values are valid
                count = logical_rows
            elif col.dtype == TYPE_DOUBLE:
                d_ptr = (<DoubleColumn>col).data_c
                for i in range(logical_rows):
                    if not isnan(d_ptr[i]):
                        count += 1
            elif col.dtype == TYPE_BOOL:
                # BoolColumn has no NaN concept, all values are valid
                count = logical_rows
            else:
                count = logical_rows
            result[name] = count
    return _agg_series(result)

cdef inline double _qs_double(double* a, Py_ssize_t n, Py_ssize_t k) noexcept nogil:
    cdef Py_ssize_t l = 0
    cdef Py_ssize_t r = n - 1
    cdef Py_ssize_t i, lt, gt
    cdef double pivot, tmp
    while l <= r:
        pivot = a[l + (r - l) // 2]
        lt = l
        gt = r
        i = l
        while i <= gt:
            if a[i] < pivot:
                tmp = a[lt]; a[lt] = a[i]; a[i] = tmp
                lt += 1; i += 1
            elif a[i] > pivot:
                tmp = a[i]; a[i] = a[gt]; a[gt] = tmp
                gt -= 1
            else:
                i += 1
        if k < lt:
            r = lt - 1
        elif k > gt:
            l = gt + 1
        else:
            return a[k]
    return a[k]


cdef _describe_impl(DataFrame self, object percentiles):
    cdef dict stats = {}
    cdef str name, top_val
    cdef Py_ssize_t col_len, i, jc, valid_cnt, s_mid
    cdef bint is2d
    cdef Py_ssize_t c_probe, r_probe, base

    cdef double s_sum, s_min, s_max, s_mean, s_m2, s_diff, val, s_med
    cdef double* s_tmp = NULL
    cdef double* d_ptr
    cdef int* i_ptr

    cdef int32_t* i32p
    cdef int64_t* i64p
    cdef int32_t v32_min, v32_max
    cdef int64_t v64_min, v64_max

    cdef list s_means, s_stds, s_mins, s_meds, s_maxs, d_mins, d_maxs
    cdef list keys
    cdef dict freq_map
    cdef Py_ssize_t max_freq, freq_val
    
    cdef list pcts
    cdef list pcts_sorted
    cdef dict pcts_dict
    cdef double idx, frac
    cdef Py_ssize_t low, high
    
    # Handle percentiles parameter
    if percentiles is None:
        pcts = [0.25, 0.5, 0.75]
    elif isinstance(percentiles, (list, tuple)):
        pcts = list(percentiles)
    else:
        pcts = [float(percentiles)]
    
    # Validate percentiles
    for p in pcts:
        if p < 0 or p > 1:
            raise ValueError(f"Percentiles must be between 0 and 1, got {p}")
    
    # Sort percentiles
    pcts_sorted = sorted(pcts)

    for name, col_obj in self._columns.items():
        col_len = col_obj.length
        _probe_shape(col_obj, &is2d, &c_probe, &r_probe)

        # =========================================================
        # 1. 数值类型处理 (Int / Double)
        # =========================================================
        if col_obj.dtype == TYPE_INT or col_obj.dtype == TYPE_DOUBLE:
            s_means = []; s_stds = []; s_mins = []; s_meds = []; s_maxs = []

            s_tmp = <double*>aligned_malloc(max(r_probe, 1) * sizeof(double))
            if not s_tmp: raise MemoryError()

            d_ptr = (<DoubleColumn>col_obj).data_c if col_obj.dtype == TYPE_DOUBLE else NULL
            i_ptr = (<IntColumn>col_obj).data_c if col_obj.dtype == TYPE_INT else NULL

            try:
                for jc in range(c_probe):
                    valid_cnt = 0
                    base = jc * r_probe

                    with nogil:
                        for i in range(r_probe):
                            if d_ptr != NULL:
                                val = d_ptr[base + i]
                                if not isnan(val):
                                    s_tmp[valid_cnt] = val
                                    valid_cnt += 1
                            else:
                                s_tmp[valid_cnt] = <double>i_ptr[base + i]
                                valid_cnt += 1

                    if valid_cnt == 0:
                        s_means.append(NAN); s_stds.append(NAN)
                        s_mins.append(NAN); s_maxs.append(NAN); s_meds.append(NAN)
                        continue

                    s_sum = s_tmp[0]
                    s_min = s_tmp[0]
                    s_max = s_tmp[0]

                    with nogil:
                        for i in range(1, valid_cnt):
                            val = s_tmp[i]
                            s_sum += val
                            if val < s_min: s_min = val
                            if val > s_max: s_max = val

                        s_mean = s_sum / <double>valid_cnt

                        if valid_cnt >= 2:
                            s_m2 = 0.0
                            for i in range(valid_cnt):
                                s_diff = s_tmp[i] - s_mean
                                s_m2 += s_diff * s_diff

                    if valid_cnt < 2:
                        s_stds.append(0.0)
                    else:
                        s_stds.append(sqrt(s_m2 / <double>(valid_cnt - 1)))

                    s_means.append(s_mean)
                    s_mins.append(s_min)
                    s_maxs.append(s_max)
                    
                    # Calculate percentiles
                    pcts_dict = {}
                    for p in pcts_sorted:
                        if valid_cnt == 0:
                            pcts_dict[f'{p*100:.0f}%'] = NAN
                        elif valid_cnt == 1:
                            pcts_dict[f'{p*100:.0f}%'] = s_tmp[0]
                        else:
                            idx = p * <double>(valid_cnt - 1)
                            low = <Py_ssize_t>idx
                            high = low + 1
                            if high >= valid_cnt:
                                pcts_dict[f'{p*100:.0f}%'] = s_tmp[valid_cnt - 1]
                            else:
                                frac = idx - <double>low
                                pcts_dict[f'{p*100:.0f}%'] = s_tmp[low] * (1.0 - frac) + s_tmp[high] * frac
                    s_meds.append(pcts_dict)

            finally:
                os_aligned_free(s_tmp)

            if is2d and c_probe > 1:
                stats[name] = {'count': r_probe, 'mean': s_means, 'std': s_stds,
                               'min': s_mins, 'max': s_maxs}
                stats[name].update(s_meds[0])
            else:
                stats[name] = {'count': r_probe, 'mean': s_means[0], 'std': s_stds[0],
                               'min': s_mins[0], 'max': s_maxs[0]}
                stats[name].update(s_meds[0])

        # =========================================================
        # 2. 离散类型处理 (String / Category / Bool)
        # =========================================================
        elif col_obj.dtype in (TYPE_CHAR256, TYPE_UCS2, TYPE_BOOL, TYPE_CATEGORY):
            keys = _col_to_keys(col_obj)
            freq_map = {}
            max_freq = 0

            for valk in keys:
                if valk is not None:
                    freq_val = freq_map.get(valk, 0) + 1
                    freq_map[valk] = freq_val
            top_val = ''
            for valk, freq_val in freq_map.items():
                if freq_val > max_freq:
                    max_freq = freq_val
                    top_val = str(valk)

            stats[name] = {'count': r_probe, 'unique': len(freq_map),
                           'top': top_val, 'freq': max_freq}

        # =========================================================
        # 3. 时间序列类型处理 (Date / DateTime / Time)
        # =========================================================
        elif col_obj.dtype in (TYPE_DATE, TYPE_DATETIME, TYPE_TIME):
            d_mins = []; d_maxs = []

            if col_obj.dtype == TYPE_DATETIME:
                i64p = (<DateTimeColumn>col_obj).data_c
                for jc in range(c_probe):
                    base = jc * r_probe
                    if r_probe == 0:
                        d_mins.append(0); d_maxs.append(0)
                        continue
                    v64_min = i64p[base]
                    v64_max = i64p[base]
                    with nogil:
                        for i in range(1, r_probe):
                            if i64p[base + i] < v64_min: v64_min = i64p[base + i]
                            if i64p[base + i] > v64_max: v64_max = i64p[base + i]
                    d_mins.append(v64_min); d_maxs.append(v64_max)
            else:
                # TYPE_DATE 和 TYPE_TIME 都是 int32_t 底层
                i32p = <int32_t*>((<DateColumn>col_obj).data_c)
                for jc in range(c_probe):
                    base = jc * r_probe
                    if r_probe == 0:
                        d_mins.append(0); d_maxs.append(0)
                        continue
                    v32_min = i32p[base]
                    v32_max = i32p[base]
                    with nogil:
                        for i in range(1, r_probe):
                            if i32p[base + i] < v32_min: v32_min = i32p[base + i]
                            if i32p[base + i] > v32_max: v32_max = i32p[base + i]
                    d_mins.append(v32_min); d_maxs.append(v32_max)

            if is2d and c_probe > 1:
                stats[name] = {'count': r_probe, 'min': d_mins, 'max': d_maxs}
            else:
                stats[name] = {'count': r_probe, 'min': d_mins[0], 'max': d_maxs[0]}

    return stats


cdef _value_counts_impl(DataFrame self, str col_name):
    if col_name not in self._columns:
        raise KeyError(f"Column '{col_name}' not found")
    cdef BaseColumn col = self._columns[col_name]

    cdef list keys = _col_to_keys(col)
    cdef Py_ssize_t n = len(keys)
    cdef Py_ssize_t i

    cdef dict counts = {}
    for i in range(n):
        val = keys[i]
        counts[val] = counts.get(val, 0) + 1

    # 按频次降序（稳定排序），构造 pandas 风格：以值为索引的 count Series
    sorted_items = sorted(counts.items(), key=lambda x: -x[1])
    cdef list key_list = [k for k, v in sorted_items]
    cdef list cnt_list = [v for k, v in sorted_items]

    cdef IntColumn cnt_col = IntColumn(len(sorted_items))
    cnt_col.length = len(sorted_items)
    cdef int* cnt_ptr = cnt_col.data_c
    for i in range(len(sorted_items)):
        cnt_ptr[i] = cnt_list[i]

    return Series('count', cnt_col, StringIndex(key_list))

cdef _nunique_impl(DataFrame self, str col_name=None):
    cdef dict result = {}
    cdef str name
    cdef BaseColumn col

    if col_name is not None:
        if col_name not in self._columns:
            raise KeyError(f"Column '{col_name}' not found")
        col = self._columns[col_name]
        return len(set(_col_to_keys(col)))
    for name, col in self._columns.items():
        result[name] = len(set(_col_to_keys(col)))
    return _agg_series(result)




# ============================================================
#  DataFrame 模块级实现 —— 输入输出（io）
# ============================================================


cdef list _col_to_list(BaseColumn col):
    """
    【核心加速机制】：一次性探测列的 shape（1D 还是 2D），成 Python list 一次性返回。
    """
    cdef bint is2d
    cdef Py_ssize_t c, r
    is2d, c, r = _col_shape_probe(col)

    if r == 0:
        return []

    cdef Py_ssize_t i, j
    cdef list res = [None] * r
    cdef list row_vals

    # 极速批处理 Dump
    if is2d and c > 1:
        for i in range(r):
            row_vals = [None] * c
            for j in range(c):
                row_vals[j] = col[j * r + i]
            res[i] = row_vals
    else:
        for i in range(r):
            res[i] = col[i]
    return res


cdef _to_string_impl(DataFrame self, Py_ssize_t max_rows=80):
    cdef Py_ssize_t n = self.df_length
    cdef list row_indices = []
    cdef Py_ssize_t half

    if n <= max_rows:
        row_indices = list(range(n))
    else:
        half = max_rows // 2
        row_indices = list(range(half)) + [-1] + list(range(n - half, n))

    names, types = self._get_header_strings()
    cdef int num_display_cols = len(names)

    cdef list table_data = []
    cdef Py_ssize_t r
    for r in row_indices:
        if r == -1:
            table_data.append(["..."] * num_display_cols)
        else:
            table_data.append(self._get_row_strings(r))

    cdef list col_widths = [0] * num_display_cols
    cdef int c, max_w, current_w
    for c in range(num_display_cols):
        max_w = self._display_width(names[c])
        current_w = self._display_width(types[c])
        if current_w > max_w: max_w = current_w
        for row in table_data:
            current_w = self._display_width(row[c])
            if current_w > max_w: max_w = current_w
        col_widths[c] = max_w + 2

    cdef list lines = []
    lines.append(f"shape: ({self.df_length}, {len(self._columns)})")

    if num_display_cols > 0:
        def make_border(left, mid, right, fill):
            return left + mid.join([fill * col_widths[k] for k in range(num_display_cols)]) + right

        lines.append(make_border("┌", "┬", "┐", "─"))

        def format_cell(text, width, align_right):
            dw = self._display_width(text)
            padding = width - dw
            return (" " * padding + text) if align_right else (text + " " * padding)

        row_str = "│"
        for c in range(num_display_cols):
            row_str += " " + format_cell(names[c], col_widths[c] - 2, False) + " " + ("│" if c == num_display_cols - 1 else "┆")
        lines.append(row_str)

        row_str = "│"
        for c in range(num_display_cols):
            row_str += " " + format_cell(types[c], col_widths[c] - 2, False) + " " + ("│" if c == num_display_cols - 1 else "┆")
        lines.append(row_str)

        lines.append(make_border("╞", "╪", "╡", "═"))

        for row_idx, row in enumerate(table_data):
            row_str = "│"
            for c in range(num_display_cols):
                val = row[c]
                is_num = val != "..." and types[c] in ("[i64]", "[f64]", "[range]", "[multi]")
                row_str += " " + format_cell(val, col_widths[c] - 2, is_num) + " " + ("│" if c == num_display_cols - 1 else "┆")
            lines.append(row_str)

        lines.append(make_border("└", "┴", "┘", "─"))
    else:
        lines.append("Empty DataFrame")

    return "\n".join(lines)

cdef _to_pickle_impl(DataFrame self, filename):
    import pickle
    cdef dict data = {'df_length': self.df_length, 'columns': {}}
    cdef bint is2d
    cdef Py_ssize_t c, r, i
    cdef Py_ssize_t total_cells
    cdef int dtype
    cdef list values
    cdef int32_t* dptr32
    cdef int64_t* tptr64
    cdef BaseColumn col_obj

    for name, col_obj in self._columns.items():
        is2d, c, r = _col_shape_probe(col_obj)
        dtype = col_obj.dtype
        if is2d and c > 1:
            total_cells = r * c
            if dtype == TYPE_DATE:
                dptr32 = <int32_t*>(<DateColumn>col_obj).data_c
                values = [dptr32[i] for i in range(total_cells)]
            elif dtype == TYPE_DATETIME:
                tptr64 = <int64_t*>(<DateTimeColumn>col_obj).data_c
                values = [tptr64[i] for i in range(total_cells)]
            elif dtype == TYPE_TIME:
                dptr32 = <int32_t*>(<TimeColumn>col_obj).data_c
                values = [dptr32[i] for i in range(total_cells)]
            else:
                values = [col_obj[i] for i in range(total_cells)]
            data['columns'][name] = (self._type_name(col_obj), values, r, c)
        else:
            values = [col_obj[i] for i in range(r)]
            data['columns'][name] = (self._type_name(col_obj), values)

    with open(filename, 'wb') as f:
        pickle.dump(data, f)

cdef _to_csv_impl(DataFrame self, filename, sep=',', str na_rep='', str float_format=None,
                   list columns=None, bint header=True, str line_terminator='\n'):
    """
    导出为 CSV 文件。
    优化点：批量列数据提取，摆脱嵌套循环中的重复类型查询。
    """
    cdef list col_names = columns if columns is not None else list(self._columns.keys())
    cdef Py_ssize_t num_cols = len(col_names)
    cdef Py_ssize_t n = self.df_length
    cdef Py_ssize_t i, j

    cdef size_t buf_pos = 0
    cdef size_t write_res
    cdef char sep_c = <char>ord(sep) if len(sep) == 1 else <char>ord(',')
    cdef char line_buf[65536]
    cdef size_t buf_cap = sizeof(line_buf)
    cdef bytes b_field
    cdef bytes line_term_bytes = line_terminator.encode('utf-8')
    cdef bytes na_rep_bytes = na_rep.encode('utf-8')

    cdef list cols_data = [_col_to_list(self._columns[name]) for name in col_names]

    with open(filename, 'wb') as f:
        if header:
            for j in range(num_cols):
                b_field = col_names[j].encode('utf-8')
                if j > 0:
                    line_buf[buf_pos] = sep_c
                    buf_pos += 1

                write_res = _csv_write_field(line_buf, buf_cap, buf_pos, <const char*>b_field, len(b_field), sep_c)
                if write_res == <size_t>-1:
                    f.write(line_buf[:buf_pos])
                    buf_pos = 0
                    write_res = _csv_write_field(line_buf, buf_cap, buf_pos, <const char*>b_field, len(b_field), sep_c)
                buf_pos = write_res

            for idx in range(len(line_term_bytes)):
                line_buf[buf_pos] = line_term_bytes[idx]
                buf_pos += 1
            f.write(line_buf[:buf_pos])


        for i in range(n):
            buf_pos = 0
            for j in range(num_cols):
                val = cols_data[j][i]
                if val is None or (isinstance(val, float) and val != val):  # Check for NaN
                    b_field = na_rep_bytes
                else:
                    if float_format and isinstance(val, float):
                        b_field = str(float_format % val).encode('utf-8')
                    else:
                        b_field = str(val).encode('utf-8')

                if j > 0:
                    line_buf[buf_pos] = sep_c
                    buf_pos += 1
                write_res = _csv_write_field(line_buf, buf_cap, buf_pos, <const char*>b_field, len(b_field), sep_c)
                if write_res == <size_t>-1:
                    f.write(line_buf[:buf_pos])
                    buf_pos = 0
                    write_res = _csv_write_field(line_buf, buf_cap, buf_pos, <const char*>b_field, len(b_field), sep_c)
                buf_pos = write_res

            for idx in range(len(line_term_bytes)):
                line_buf[buf_pos] = line_term_bytes[idx]
                buf_pos += 1
            f.write(line_buf[:buf_pos])

cdef _to_numpy_impl(DataFrame self):
    """
    导出为二维数组 (List of Lists)。
    """
    cdef Py_ssize_t n = self.df_length
    cdef list col_names = list(self._columns.keys())
    cdef Py_ssize_t num_cols = len(col_names)
    cdef Py_ssize_t i, j

    cdef list cols_data = [_col_to_list(self._columns[name]) for name in col_names]
    cdef list result = [None] * n
    cdef list row

    for i in range(n):
        row = [None] * num_cols
        for j in range(num_cols):
            row[j] = cols_data[j][i]
        result[i] = row
    return result

cdef _info_impl(DataFrame self):
    cdef Py_ssize_t total_cols = len(self._columns)
    cdef Py_ssize_t mem = _memory_usage_impl(self)
    cdef char bytes_buf[16]
    cdef list lines = []
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t col_bytes, col_rows
    cdef bint is2d
    cdef Py_ssize_t c_probe, r_probe
    cdef str type_name, name
    cdef object col

    lines.append("<class 'DataFrame'>")
    lines.append(f"RangeIndex: {self.df_length} entries")
    lines.append(f"Data columns (total {total_cols} columns):")
    lines.append(" #   Column    Non-Null  Dtype     Mem Usage")
    lines.append("---  ------    --------  ----     ---------")

    for name, col in self._columns.items():
        type_name = self._type_name(col)
        col_bytes = col.get_size()
        is2d, c_probe, r_probe = _col_shape_probe(col)

        _c_fmt_bytes(bytes_buf, sizeof(bytes_buf), col_bytes)
        lines.append(f" {idx}   {name:<10}{r_probe:<10}{type_name:<10}{bytes_buf.decode('ascii')}")
        idx += 1

    _c_fmt_bytes(bytes_buf, sizeof(bytes_buf), mem)
    lines.append(f"dtypes: {_dtype_summary_impl(self)}")
    lines.append(f"memory usage: {bytes_buf.decode('ascii')}")
    return '\n'.join(lines)

cdef str _dtype_summary_impl(DataFrame self):
    cdef dict counts = {}
    cdef str name, tname
    for name, col in self._columns.items():
        tname = self._type_name(col)
        counts[tname] = counts.get(tname, 0) + 1
    return ', '.join(f'{v} {k}' for k, v in counts.items())

cdef _memory_usage_impl(DataFrame self):
    cdef Py_ssize_t total = 0
    cdef BaseColumn col
    for col in self._columns.values():
        total += col.get_size()
    return total

cdef _to_dict_impl(DataFrame self):
    """导出为字段字典：{列名: Series}（值列与 self.index 对齐；对标 pandas 列式导出语义）"""
    cdef dict result = {}
    cdef str name
    cdef BaseColumn col

    for name, col in self._columns.items():
        result[name] = Series(name, col, self.index)
    return result

cdef _to_json_impl(DataFrame self, bint orient_records=True):
    """导出为 JSON"""
    import json

    cdef dict columns_dict = {}
    cdef str jname
    cdef BaseColumn jcol
    for jname, jcol in self._columns.items():
        columns_dict[jname] = _col_to_list(jcol)

    if not orient_records:
        return json.dumps(columns_dict)

    # ---------------------------------------------------------
    # Orient Records 模式（行翻转矩阵）
    # ---------------------------------------------------------
    cdef Py_ssize_t n = self.df_length
    cdef list records = [None] * n

    cdef list col_names = list(columns_dict.keys())
    cdef Py_ssize_t num_cols = len(col_names)

    cdef list cols_data = list(columns_dict.values())
    cdef dict row_dict
    cdef Py_ssize_t i, j

    for i in range(n):
        row_dict = {}
        for j in range(num_cols):
            row_dict[col_names[j]] = cols_data[j][i]
        records[i] = row_dict
    return json.dumps(records)


cdef str _html_escape(object val):
    """HTML 特殊字符转义（合并进 to_html 内调用，避免额外 Cython 开销）。"""
    cdef str s = str(val)
    return (s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
             .replace('"', '&quot;').replace("'", '&#39;'))


cdef str _to_html_impl(DataFrame self, Py_ssize_t max_rows=80):
    """导出为 HTML 表格字符串（<table class="dataframe">）。"""
    cdef Py_ssize_t n = self.df_length
    cdef list row_indices = []
    cdef Py_ssize_t half, r

    if n <= max_rows:
        row_indices = list(range(n))
    else:
        half = max_rows // 2
        row_indices = list(range(half)) + [-1] + list(range(n - half, n))

    cdef list names = self._get_header_strings()[0]
    cdef Py_ssize_t num_display_cols = len(names)

    cdef list rows = []
    for r in row_indices:
        rows.append(["..."] * num_display_cols if r == -1 else self._get_row_strings(r))

    cdef list out = ['<table class="dataframe">\n  <thead>', '    <tr>']
    for c in range(num_display_cols):
        out.append(f"      <th>{_html_escape(names[c])}</th>")
    out.append('    </tr>\n    </thead>')
    out.append('  <tbody>')
    for row in rows:
        out.append('    <tr>')
        for c in range(num_display_cols):
            out.append(f"      <td>{_html_escape(row[c])}</td>")
        out.append('    </tr>')
    out.append('  </tbody>')
    out.append('</table>')
    return '\n'.join(out)


cdef _to_excel_impl(DataFrame self, object filename, str sheet_name='Sheet1'):
    """导出为 xlsx 文件（openpyxl 懒加载；未安装时抛 ImportError 提示）。"""
    try:
        import openpyxl
    except ImportError:
        raise ImportError("to_excel() requires 'openpyxl'. Install with: pip install openpyxl")

    cdef list col_names = list(self._columns.keys())
    cdef Py_ssize_t num_cols = len(col_names)
    cdef Py_ssize_t n = self.df_length
    cdef Py_ssize_t i, j

    cdef list cols_data = [_col_to_list(self._columns[name]) for name in col_names]

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet_name
    ws.append([c for c in col_names])
    for i in range(n):
        row = [None] * num_cols
        for j in range(num_cols):
            v = cols_data[j][i]
            row[j] = v if v is not None else ''
        ws.append(row)
    wb.save(filename)
    return filename


# ============================================================
#  DataFrame 模块级实现 —— 合并（merge / join / concat）
# ============================================================

cdef _concat_impl(DataFrame self, DataFrame other):
    if set(self._columns.keys()) != set(other._columns.keys()):
        raise ValueError("Schema mismatch during concat")
    cdef DataFrame result = self.copy()
    for col_name, col_obj in result._columns.items():
        col_obj.concat(other._columns[col_name])
    result.df_length = self.df_length + other.df_length
    result.index = self.index._concat(other.index)
    return result

cdef inline BaseColumn _gather_or_reorder(DataFrame df, BaseColumn src_col,
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

cdef _merge_impl(DataFrame self, DataFrame other, list on, str how='inner'):
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
    cdef DataFrame result = DataFrame()
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

cdef _duplicated_impl(DataFrame self, subset=None, str keep='first'):
    cdef Py_ssize_t n = self.df_length
    cdef BoolColumn result = BoolColumn(max(n, 1))
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

cdef _drop_duplicates_impl(DataFrame self, subset=None, str keep='first', bint inplace=False):
    cdef BoolColumn mask = _duplicated_impl(self, subset, keep)
    cdef Py_ssize_t n = self.df_length
    cdef Py_ssize_t keep_count = n - mask.count_nonzero()

    if keep_count == n:
        return self if inplace else _copy_impl(self)

    cdef Py_ssize_t* keep_idx = <Py_ssize_t*>aligned_malloc(max(keep_count, 1) * sizeof(Py_ssize_t))
    if not keep_idx: raise MemoryError()
    mask.gather_zero_to(keep_idx)

    cdef DataFrame result = self._take(keep_idx, keep_count)
    os_aligned_free(keep_idx)
    if inplace:
        self._columns = result._columns
        self.df_length = result.df_length
        self.index = result.index
        return self
    return result


# ============================================================
#  DataFrame 模块级实现 —— 相关性 / 协方差 / 距离（metric）
# ============================================================

cdef list _numeric_logical_features(DataFrame self):
    """展开 2D 数值列为逻辑子特征，返回 [(key, col, sub)]。"""
    cdef list feats = []
    cdef str name
    cdef BaseColumn col
    cdef bint is2d
    cdef Py_ssize_t c, r, k
    for name, col in self._columns.items():
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
cdef inline DoubleColumn _ensure_double_1d(DataFrame self, object col_ident):
    """防止 IntColumn 被暴力转为 DoubleColumn 导致 SegFault"""
    cdef str name = col_ident if isinstance(col_ident, str) else None
    cdef BaseColumn col
    if name is not None:
        col = self._columns[name]
    elif hasattr(col_ident, '_col'):
        # 处理Series对象
        col = (<object>col_ident)._col
    else:
        col = <BaseColumn>col_ident

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

cdef _metric_dist_withcols_impl(DataFrame self, col1, col2, int op):
    cdef DoubleColumn c1 = _ensure_double_1d(self, col1)
    cdef DoubleColumn c2 = _ensure_double_1d(self, col2)
    if op == PW_EUCLIDEAN:
        return c1.euclidean_dist(c2)
    elif op == PW_MANHATTAN:
        return c1.manhattan_dist(c2)
    elif op == PW_COSINE_DIST:
        return c1.cosine_dist(c2)
    elif op == PW_COSINE_SIM:
        return c1.cosine_similarity(c2)
    elif op == PW_HAMMING:
        return c1.hamming_dist(c2)
    else:
        return c1.jaccard_sim(c2)

cdef DataFrame _corr_cov_impl(DataFrame self, bint is_corr):
    cdef list feats = _numeric_logical_features(self)
    cdef Py_ssize_t n_feats = len(feats)
    cdef Py_ssize_t n_rows = self.df_length
    if n_feats == 0 or n_rows == 0:
        return DataFrame()

    # 仅用一次 malloc 分配 N 个特征的全部连续大内存 (列优先结构)
    cdef double* data_matrix = <double*>aligned_malloc(n_feats * max(n_rows, 1) * sizeof(double))
    if not data_matrix:
        raise MemoryError()

    cdef Py_ssize_t i, j
    cdef double val
    cdef tuple feat
    cdef DoubleColumn res_block
    cdef double* res_ptr = NULL
    cdef DataFrame result

    try:
        # 1. 一次性安全萃取所有特征
        for i in range(n_feats):
            feat = feats[i]
            # 数据步长正好是 n_rows，完全 F-contiguous 连续，对缓存极其友好
            self._col_to_double_buffer(<BaseColumn>feat[1], data_matrix + i * n_rows, n_rows, <Py_ssize_t>feat[2])

        # 申请统一的 2D 连续块作为结果矩阵
        res_block = DoubleColumn(n_feats * n_feats, cols=n_feats)
        res_block.length = n_feats * n_feats
        res_ptr = res_block.data_c

        # 释放 GIL 并在行优先内存地址上并行填充相关性/协方差矩阵
        if is_corr:
            with nogil:
                for i in range(n_feats):
                    res_ptr[i * n_feats + i] = 1.0
                    for j in range(i + 1, n_feats):
                        val = _vec_pearson_corr(data_matrix + i * n_rows, data_matrix + j * n_rows, n_rows)
                        res_ptr[j * n_feats + i] = val
                        res_ptr[i * n_feats + j] = val
        else:
            with nogil:
                for i in range(n_feats):
                    res_ptr[i * n_feats + i] = _vec_pearson_cov(data_matrix + i * n_rows, data_matrix + i * n_rows, n_rows)
                    for j in range(i + 1, n_feats):
                        val = _vec_pearson_cov(data_matrix + i * n_rows, data_matrix + j * n_rows, n_rows)
                        res_ptr[j * n_feats + i] = val
                        res_ptr[i * n_feats + j] = val

        # 极速装载 View (零拷贝)
        result = DataFrame()
        result.df_length = n_feats
        result.index = RangeIndex(0, n_feats)
        for i in range(n_feats):
            result._columns[feats[i][0]] = DoubleColumn.create_view(res_ptr + i * n_feats, n_feats, res_block)

        if not hasattr(result, '_2d_blocks'):
            result._2d_blocks = []
        result._2d_blocks.append(res_block)

        return result
    finally:
        os_aligned_free(data_matrix)

cdef _pearson_sim_impl(DataFrame self):
    """
    数学上，列之间的 Pearson 相似度矩阵 等价于 Pearson 相关系数矩阵。
    直接复用 _corr_cov_impl 的实现，避免了原先实现中 O(N^2) 的内存拷贝和重复分配灾难。
    """
    return _corr_cov_impl(self, True)


# ============================================================
# Two-Column Compare Metircs
# ============================================================

cdef _corrwith_impl(DataFrame self, other):
    cdef dict result = {}
    cdef tuple feat
    cdef Py_ssize_t n = self.df_length
    if n == 0: return result

    if isinstance(other, Series):
        other = (<Series>other)._col

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

cdef double _corrcoef_cov_withcols_impl(DataFrame self, col_a, col_b, bint is_corr):
    cdef bint is2d
    cdef Py_ssize_t c_a, r_a, c_b, r_b
    if isinstance(col_a, Series):
        col_a = (<Series>col_a)._col
    if isinstance(col_b, Series):
        col_b = (<Series>col_b)._col
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

    if is_corr:
        return _vec_pearson_corr(buf_a.data_c, buf_b.data_c, n)
    return _vec_pearson_cov(buf_a.data_c, buf_b.data_c, n)


# ============================================================
#  DataFrame 模块级实现 —— 工厂（from_dict / arange / zeros ...）
# ============================================================

cdef _make_empty_column_impl(int dtype):
    if dtype == TYPE_INT: return IntColumn()
    elif dtype == TYPE_DOUBLE: return DoubleColumn()
    elif dtype == TYPE_CHAR256: return AsciiColumn()
    elif dtype == TYPE_UCS2: return UCS2Column()
    elif dtype == TYPE_DATE: return DateColumn()
    elif dtype == TYPE_BOOL: return BoolColumn()
    elif dtype == TYPE_CATEGORY: return CategoryColumn()
    elif dtype == TYPE_DATETIME: return DateTimeColumn()
    elif dtype == TYPE_TIME: return TimeColumn()
    raise ValueError(f"Unknown dtype: {dtype}")

cdef _populate_column_impl(int dtype, Py_ssize_t n, list values):
    cdef Py_ssize_t i
    if dtype == TYPE_INT:
        int_col = IntColumn(n)
        int_col.length = n
        for i in range(n): int_col.data_c[i] = <int>values[i]
        return int_col
    elif dtype == TYPE_DOUBLE:
        dbl_col = DoubleColumn(n)
        dbl_col.length = n
        for i in range(n): dbl_col.data_c[i] = <double>values[i]
        return dbl_col
    elif dtype == TYPE_CHAR256:
        c256_col = AsciiColumn(n)
        for i in range(n): c256_col.append(<str>values[i])
        return c256_col
    elif dtype == TYPE_UCS2:
        ucs2_col = UCS2Column(n)
        for i in range(n): ucs2_col.append(<str>values[i])
        return ucs2_col
    elif dtype == TYPE_DATE:
        ts_col = DateColumn(n)
        ts_col.length = n
        for i in range(n): ts_col.data_c[i] = <int32_t>values[i]
        return ts_col
    elif dtype == TYPE_BOOL:
        bool_col = BoolColumn(n)
        bool_col.length = n
        for i in range(n): bool_col.data_c[i] = 1 if values[i] else 0
        return bool_col
    elif dtype == TYPE_CATEGORY:
        cat_col = CategoryColumn(n)
        for i in range(n): cat_col.append(<str>values[i])
        return cat_col
    elif dtype == TYPE_DATETIME:
        dtc_col = DateTimeColumn(n)
        for i in range(n): dtc_col.append(<str>values[i])
        return dtc_col
    elif dtype == TYPE_TIME:
        tmc_col = TimeColumn(n)
        for i in range(n): tmc_col.append(<str>values[i])
        return tmc_col
    raise ValueError(f"Unknown dtype: {dtype}")

cdef _rename_columns_impl(DataFrame df, columns):
    if columns is not None:
        col_names = list(df._columns.keys())
        new_names = list(columns)
        if len(new_names) != len(col_names):
            raise ValueError(f"columns length {len(new_names)} != {len(col_names)}")
        new_cols = {}
        for i in range(len(new_names)):
            new_cols[new_names[i]] = df._columns[col_names[i]]
        df._columns = new_cols

cdef _from_dict_impl(dict data, dict schema, object columns):
    cdef dict parsed = _parse_schema(schema) if schema else {}
    cdef DataFrame df = DataFrame()
    cdef str col_name
    cdef list values
    cdef Py_ssize_t n = 0
    cdef Py_ssize_t i

    for col_name, values in data.items():
        if col_name not in parsed:
            raise ValueError(f"Column '{col_name}' not in schema")
        if n == 0:
            n = len(values)
        if len(values) != n:
            raise ValueError(f"Column '{col_name}' length {len(values)} != expected {n}")

    if n == 0:
        return df

    df.df_length = n
    df.index = RangeIndex(0, n)

    for col_name in data:
        df._columns[col_name] = _populate_column_impl(parsed[col_name], n, data[col_name])

    _rename_columns_impl(df, columns)
    return df

cdef _from_list_impl(list rows, dict schema, object columns):
    if not schema:
        raise ValueError("schema is required and must not be empty")
    cdef dict parsed = _parse_schema(schema)
    cdef DataFrame df = DataFrame()
    cdef Py_ssize_t n_rows = len(rows)
    cdef Py_ssize_t n_cols = len(parsed)
    cdef Py_ssize_t i, j
    cdef list col_order = list(parsed.keys())
    cdef str col_name

    if n_rows == 0:
        for col_name in col_order:
            df._columns[col_name] = _make_empty_column_impl(parsed[col_name])
        return df

    df.df_length = n_rows
    df.index = RangeIndex(0, n_rows)

    for j in range(n_cols):
        col_name = col_order[j]
        col_values = [rows[i][j] for i in range(n_rows)]
        df._columns[col_name] = _populate_column_impl(parsed[col_name], n_rows, col_values)

    _rename_columns_impl(df, columns)
    return df

cdef _from_arrays_impl(list arrays, dict schema, object columns):
    if not schema:
        raise ValueError("schema is required and must not be empty")
    cdef dict parsed = _parse_schema(schema)
    cdef DataFrame df = DataFrame()
    cdef Py_ssize_t n_cols = len(arrays)
    cdef Py_ssize_t n_rows = 0
    cdef Py_ssize_t i, j
    cdef list col_order = list(schema.keys())
    cdef str col_name

    if n_cols == 0:
        return df

    n_rows = len(arrays[0])
    for j in range(n_cols):
        if len(arrays[j]) != n_rows:
            raise ValueError(f"Array {j} length {len(arrays[j])} != expected {n_rows}")

    if n_rows == 0:
        for col_name in col_order:
            df._columns[col_name] = _make_empty_column_impl(parsed[col_name])
        return df

    df.df_length = n_rows
    df.index = RangeIndex(0, n_rows)

    for j in range(n_cols):
        col_name = col_order[j]
        df._columns[col_name] = _populate_column_impl(parsed[col_name], n_rows, arrays[j])

    _rename_columns_impl(df, columns)
    return df

cdef _arange_impl(Py_ssize_t n):
    cdef DataFrame df = DataFrame()
    if n <= 0: return df
    df.df_length = n
    df.index = RangeIndex(0, n)
    cdef IntColumn col = IntColumn(max(n, 1))
    col.length = n
    cdef Py_ssize_t i
    with nogil:
        for i in range(n):
            col.data_c[i] = <int>i
    df._columns['0'] = col
    return df

cdef _randn_impl(Py_ssize_t rows, Py_ssize_t cols):
    cdef DataFrame df = DataFrame()
    if rows <= 0 or cols <= 0: return df
    df.df_length = rows
    df.index = RangeIndex(0, rows)
    cdef list col_names = [chr(65 + i) if i < 26 else f"C{i}" for i in range(cols)]
    cdef Py_ssize_t c_idx, r_idx
    cdef DoubleColumn col
    _xorshift128plus_seed(<unsigned long long>time(NULL) ^ 12345ULL)
    for c_idx in range(cols):
        col = DoubleColumn(max(rows, 1))
        col.length = rows
        for r_idx in range(rows):
            col.data_c[r_idx] = _random_gauss()
        df._columns[col_names[c_idx]] = col
    return df

cdef _rand_impl(Py_ssize_t rows, Py_ssize_t cols):
    cdef DataFrame df = DataFrame()
    if rows <= 0 or cols <= 0: return df
    df.df_length = rows
    df.index = RangeIndex(0, rows)
    cdef list col_names = [chr(65 + i) if i < 26 else f"C{i}" for i in range(cols)]
    cdef Py_ssize_t c_idx, r_idx
    cdef DoubleColumn col
    _xorshift128plus_seed(<unsigned long long>time(NULL) ^ 54321ULL)
    for c_idx in range(cols):
        col = DoubleColumn(max(rows, 1))
        col.length = rows
        for r_idx in range(rows):
            col.data_c[r_idx] = _random_uniform()
        df._columns[col_names[c_idx]] = col
    return df

cdef _zeros_impl(Py_ssize_t rows, Py_ssize_t cols):
    cdef DataFrame df = DataFrame()
    if rows <= 0 or cols <= 0: return df
    df.df_length = rows
    df.index = RangeIndex(0, rows)
    cdef list col_names = [chr(65 + i) if i < 26 else f"C{i}" for i in range(cols)]
    cdef Py_ssize_t c_idx
    cdef DoubleColumn col
    for c_idx in range(cols):
        col = DoubleColumn(max(rows, 1))
        col.length = rows
        memset(col.data_c, 0, rows * sizeof(double))
        df._columns[col_names[c_idx]] = col
    return df

cdef _ones_impl(Py_ssize_t rows, Py_ssize_t cols):
    cdef DataFrame df = DataFrame()
    if rows <= 0 or cols <= 0: return df
    df.df_length = rows
    df.index = RangeIndex(0, rows)
    cdef list col_names = [chr(65 + i) if i < 26 else f"C{i}" for i in range(cols)]
    cdef Py_ssize_t c_idx, r_idx
    cdef DoubleColumn col
    for c_idx in range(cols):
        col = DoubleColumn(max(rows, 1))
        col.length = rows
        with nogil:
            for r_idx in range(rows):
                col.data_c[r_idx] = 1.0
        df._columns[col_names[c_idx]] = col
    return df


# ============================================================
#  DataFrame 模块级实现 —— 滚动窗口（rolling / shift / diff）
# ============================================================

cdef _rolling_impl(DataFrame self, Py_ssize_t window, str col_name):
    if window <= 0:
        raise ValueError(f"Window size must be positive, got {window}")
    if col_name not in self._columns:
        raise KeyError(f"Column '{col_name}' not found")
    cdef BaseColumn col = self._columns[col_name]
    cdef int col_dtype = col.dtype
    if col_dtype != TYPE_INT and col_dtype != TYPE_DOUBLE:
        raise TypeError(f"Rolling window requires numeric column, got '{col_name}'")
    return _RollingWrapper(col, window, col_name, self.index)


cdef _shift_int_column(IntColumn col, Py_ssize_t periods, Py_ssize_t n, Py_ssize_t c, Py_ssize_t r):
    cdef IntColumn new_col = IntColumn(max(n, 1), cols=c)
    new_col.length = n
    # 强制锁定新列的 2D 状态，防止发生 1D 降级
    new_col.num_cols = <int>c
    new_col.is_2d = True
    new_col.owns_data = c > 1

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
    new_col.num_cols = <int>c
    new_col.is_2d = True
    new_col.owns_data = c > 1

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
    new_col.num_cols = <int>c
    new_col.is_2d = True
    new_col.owns_data = c > 1

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


cdef void _shift_fixed_zero(void* dst, void* src, Py_ssize_t periods,
                            Py_ssize_t c, Py_ssize_t r, Py_ssize_t elem) noexcept nogil:
    """定宽字节列的 memcpy/memset-0 位移（Date/DateTime/Time 共用，语义同 Int）。"""
    cdef Py_ssize_t p = periods if periods >= 0 else -periods
    if p > r: p = r
    cdef Py_ssize_t valid_len = r - p
    cdef char* d = <char*>dst
    cdef char* s = <char*>src
    cdef Py_ssize_t j, row_bytes = r * elem
    if periods >= 0:
        for j in range(c):
            if p > 0:
                memset(d + j * row_bytes, 0, p * elem)
            if valid_len > 0:
                memcpy(d + j * row_bytes + p * elem, s + j * row_bytes, valid_len * elem)
    else:
        for j in range(c):
            if valid_len > 0:
                memcpy(d + j * row_bytes, s + j * row_bytes + p * elem, valid_len * elem)
            if p > 0:
                memset(d + j * row_bytes + valid_len * elem, 0, p * elem)


cdef _shift_date_column(DateColumn col, Py_ssize_t periods, Py_ssize_t n, Py_ssize_t c, Py_ssize_t r):
    cdef DateColumn new_col = DateColumn(max(n, 1), cols=c)
    new_col.length = n
    new_col.num_cols = <int>c
    new_col.is_2d = True
    new_col.owns_data = c > 1
    _shift_fixed_zero(<void*>new_col.data_c, <void*>col.data_c, periods, c, r, sizeof(int32_t))
    return new_col


cdef _shift_time_column(TimeColumn col, Py_ssize_t periods, Py_ssize_t n, Py_ssize_t c, Py_ssize_t r):
    cdef TimeColumn new_col = TimeColumn(max(n, 1), cols=c)
    new_col.length = n
    new_col.num_cols = <int>c
    new_col.is_2d = True
    new_col.owns_data = c > 1
    _shift_fixed_zero(<void*>new_col.data_c, <void*>col.data_c, periods, c, r, sizeof(int32_t))
    return new_col


cdef _shift_datetime_column(DateTimeColumn col, Py_ssize_t periods, Py_ssize_t n, Py_ssize_t c, Py_ssize_t r):
    cdef DateTimeColumn new_col = DateTimeColumn(max(n, 1), cols=c)
    new_col.length = n
    new_col.num_cols = <int>c
    new_col.is_2d = True
    new_col.owns_data = c > 1
    _shift_fixed_zero(<void*>new_col.data_c, <void*>col.data_c, periods, c, r, sizeof(int64_t))
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
    elif col.dtype == TYPE_DATE:
        return _shift_date_column(<DateColumn>col, periods, total, c_out, r_out)
    elif col.dtype == TYPE_TIME:
        return _shift_time_column(<TimeColumn>col, periods, total, c_out, r_out)
    elif col.dtype == TYPE_DATETIME:
        return _shift_datetime_column(<DateTimeColumn>col, periods, total, c_out, r_out)
    else:
        raise TypeError(f"shift not supported for column type {col.dtype}")

cdef _shift_impl(DataFrame self, Py_ssize_t periods, str axis):
    if axis != 'index':
        raise ValueError(f"axis must be 'index', got '{axis}'")
    if periods == 0:
        return _copy_impl(self)

    cdef Py_ssize_t n = self.df_length
    cdef DataFrame result = DataFrame()
    result.df_length = n
    result.index = self.index

    for col_name, col in self._columns.items():
        result._columns[col_name] = _shift_column_impl(col, periods, n)

    return result

cdef _diff_column_all_impl(DataFrame self, int periods):
    cdef DataFrame result = DataFrame()
    result.df_length = self.df_length
    result.index = self.index
    cdef Py_ssize_t n = self.df_length

    for col_name, col in self._columns.items():
        if col.dtype == TYPE_INT or col.dtype == TYPE_DOUBLE:
            result._columns[col_name] = self._diff_column(col, periods)

    return result


# ============================================================
#  DataFrame 模块级实现 —— 统计（rank / cut / qcut）
# ============================================================




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


# ============================================================
#  DataFrame 模块级实现 —— 分组聚合（groupby）与 _DataFrameGroupByAccessor
# ============================================================




cdef _AggFunc _resolve_agg_func(str func) except? _AGG_SUM:
    if func == 'sum': return _AGG_SUM
    elif func == 'mean': return _AGG_MEAN
    elif func == 'min': return _AGG_MIN
    elif func == 'max': return _AGG_MAX
    elif func == 'var': return _AGG_VAR
    elif func == 'std': return _AGG_STD
    else: raise ValueError(f"Unknown aggregation function: {func}")

# ============================================================
# Pure C hash table (open addressing) replacing unordered_map
# ============================================================



cdef _GroupHashTable* _ght_new(Py_ssize_t initial_capacity) nogil:
    cdef _GroupHashTable* ht = <_GroupHashTable*>aligned_malloc(sizeof(_GroupHashTable))
    if not ht: return NULL
    ht.capacity = initial_capacity
    ht.size = 0
    ht.entries = <_GroupHashEntry*>aligned_malloc(initial_capacity * sizeof(_GroupHashEntry))
    if not ht.entries:
        os_aligned_free(ht)
        return NULL
    cdef Py_ssize_t i
    for i in range(initial_capacity):
        ht.entries[i].occupied = 0
    return ht

cdef void _ght_free(_GroupHashTable* ht) nogil:
    if ht != NULL:
        if ht.entries != NULL: os_aligned_free(ht.entries)
        os_aligned_free(ht)

cdef int _ght_get_or_insert(_GroupHashTable* ht, uint64_t h) nogil:
    if ht.size >= ht.capacity:
        return -1
    cdef Py_ssize_t idx = <Py_ssize_t>(h % <uint64_t>ht.capacity)
    cdef Py_ssize_t probe_count = 0

    while probe_count < ht.capacity:
        if not ht.entries[idx].occupied:
            ht.entries[idx].hash = h
            ht.entries[idx].group_id = ht.size
            ht.entries[idx].occupied = 1
            ht.size += 1
            return ht.size - 1
        if ht.entries[idx].hash == h:
            return ht.entries[idx].group_id
        idx = (idx + 1) % ht.capacity
        probe_count += 1
    return -1

# ============================================================
# Pure C dynamic arrays replacing vector
# ============================================================



cdef _GroupRows* _group_rows_new() nogil:
    cdef _GroupRows* gr = <_GroupRows*>aligned_malloc(sizeof(_GroupRows))
    if not gr: return NULL
    gr.count = 0
    gr.rows = NULL
    gr.row_sizes = NULL
    gr.row_capacities = NULL
    return gr

cdef void _group_rows_free(_GroupRows* gr) nogil:
    if gr == NULL: return
    cdef Py_ssize_t i
    if gr.rows != NULL:
        for i in range(gr.count):
            if gr.rows[i] != NULL: os_aligned_free(gr.rows[i])
        os_aligned_free(gr.rows)
    if gr.row_sizes != NULL: os_aligned_free(gr.row_sizes)
    if gr.row_capacities != NULL: os_aligned_free(gr.row_capacities)
    os_aligned_free(gr)

cdef int _group_rows_add_group(_GroupRows* gr) nogil:
    """Add a new group. Returns group_id."""
    cdef Py_ssize_t new_count = gr.count + 1
    cdef Py_ssize_t* new_sizes
    cdef Py_ssize_t* new_caps
    cdef Py_ssize_t** new_rows

    new_sizes = <Py_ssize_t*>aligned_realloc(gr.row_sizes, max(gr.count, 1) * sizeof(Py_ssize_t), new_count * sizeof(Py_ssize_t))
    if not new_sizes: return -1

    new_caps = <Py_ssize_t*>aligned_realloc(gr.row_capacities, max(gr.count, 1) * sizeof(Py_ssize_t), new_count * sizeof(Py_ssize_t))
    if not new_caps:
        os_aligned_free(new_sizes)
        return -1

    new_rows = <Py_ssize_t**>aligned_realloc(gr.rows, max(gr.count, 1) * sizeof(Py_ssize_t*), new_count * sizeof(Py_ssize_t*))
    if not new_rows:
        os_aligned_free(new_sizes)
        os_aligned_free(new_caps)
        return -1

    gr.row_sizes = new_sizes
    gr.row_capacities = new_caps
    gr.rows = new_rows

    gr.row_sizes[gr.count] = 0
    gr.row_capacities[gr.count] = 4
    gr.rows[gr.count] = <Py_ssize_t*>aligned_malloc(4 * sizeof(Py_ssize_t))
    if not gr.rows[gr.count]: return -1
    gr.count = new_count
    return gr.count - 1

cdef int _group_rows_append(_GroupRows* gr, int group_id, Py_ssize_t row_idx) nogil:
    """Append a row index to a group."""
    cdef Py_ssize_t new_cap
    cdef Py_ssize_t* new_ptr
    if gr.row_sizes[group_id] >= gr.row_capacities[group_id]:
        new_cap = gr.row_capacities[group_id] * 2
        new_ptr = <Py_ssize_t*>aligned_realloc(gr.rows[group_id], new_cap * sizeof(Py_ssize_t), gr.row_capacities[group_id] * sizeof(Py_ssize_t))
        if not new_ptr: return -1
        gr.rows[group_id] = new_ptr
        gr.row_capacities[group_id] = new_cap
    gr.rows[group_id][gr.row_sizes[group_id]] = row_idx
    gr.row_sizes[group_id] += 1
    return 0

# ============================================================
# FNV-1a hashing (same as original)
# ============================================================

cdef uint64_t _gb_fnv1a_bytes(char* data, Py_ssize_t length) noexcept nogil:
    cdef uint64_t h = 14695981039346656037ULL
    cdef Py_ssize_t i
    for i in range(length):
        h ^= <uint64_t><unsigned char>data[i]
        h *= 1099511628211ULL
    return h

cdef uint64_t _gb_fnv1a_ucs2(Py_UCS2* data, Py_ssize_t length) noexcept nogil:
    cdef uint64_t h = 14695981039346656037ULL
    cdef Py_ssize_t i
    for i in range(length):
        h ^= <uint64_t>data[i]
        h *= 1099511628211ULL
    return h

cdef uint64_t _gb_mix_hash(uint64_t h, uint64_t val) noexcept nogil:
    h ^= val
    h *= 1099511628211ULL
    return h

# ============================================================
# Column pointer extraction for fast hashing/comparison
# ============================================================



cdef _ByColPtrs _extract_by_col_ptrs(DataFrame df, list by_cols):
    cdef _ByColPtrs ptrs
    cdef int n = len(by_cols)
    ptrs.count = n
    ptrs.data = <void**>aligned_malloc(max(n, 1) * sizeof(void*))
    ptrs.offsets = <void**>aligned_malloc(max(n, 1) * sizeof(void*))
    ptrs.types = <int*>aligned_malloc(max(n, 1) * sizeof(int))
    ptrs.cols = <int*>aligned_malloc(max(n, 1) * sizeof(int))
    ptrs.strides = <Py_ssize_t*>aligned_malloc(max(n, 1) * sizeof(Py_ssize_t))
    if not ptrs.data or not ptrs.offsets or not ptrs.types or not ptrs.cols or not ptrs.strides:
        if ptrs.data: os_aligned_free(ptrs.data)
        if ptrs.offsets: os_aligned_free(ptrs.offsets)
        if ptrs.types: os_aligned_free(ptrs.types)
        if ptrs.cols: os_aligned_free(ptrs.cols)
        if ptrs.strides: os_aligned_free(ptrs.strides)
        raise MemoryError()
    cdef str col_name
    cdef BaseColumn col
    cdef int idx
    cdef bint is2d
    cdef Py_ssize_t c_col, r_col
    for idx in range(n):
        col_name = by_cols[idx]
        col = df._columns[col_name]
        ptrs.types[idx] = col.dtype
        ptrs.offsets[idx] = NULL
        # 2D 列（is_2d 且 cols>1）：按完整行向量参与分组；其余列子列数=1
        is2d, c_col, r_col = _col_shape_probe(col)
        if is2d and c_col > 1 and col.dtype != TYPE_CHAR256 and col.dtype != TYPE_UCS2:
            ptrs.cols[idx] = <int>c_col
            ptrs.strides[idx] = r_col
        else:
            ptrs.cols[idx] = 1
            ptrs.strides[idx] = df.df_length
        if col.dtype == TYPE_INT:
            ptrs.data[idx] = <void*>(<IntColumn>col).data_c
        elif col.dtype == TYPE_DOUBLE:
            ptrs.data[idx] = <void*>(<DoubleColumn>col).data_c
        elif col.dtype == TYPE_CHAR256:
            ptrs.data[idx] = <void*>(<AsciiColumn>col).data_c
            ptrs.offsets[idx] = <void*>(<AsciiColumn>col).offsets_c
        elif col.dtype == TYPE_UCS2:
            ptrs.data[idx] = <void*>(<UCS2Column>col).chars_c
            ptrs.offsets[idx] = <void*>(<UCS2Column>col).offsets_c
        elif col.dtype == TYPE_DATE:
            ptrs.data[idx] = <void*>(<DateColumn>col).data_c
        elif col.dtype == TYPE_BOOL:
            ptrs.data[idx] = <void*>(<BoolColumn>col).data_c
        elif col.dtype == TYPE_CATEGORY:
            ptrs.data[idx] = <void*>(<CategoryColumn>col).codes_c
        elif col.dtype == TYPE_DATETIME:
            ptrs.data[idx] = <void*>(<DateTimeColumn>col).data_c
        elif col.dtype == TYPE_TIME:
            ptrs.data[idx] = <void*>(<TimeColumn>col).data_c
    return ptrs

cdef void _free_by_col_ptrs(_ByColPtrs ptrs) nogil:
    if ptrs.data != NULL: os_aligned_free(ptrs.data)
    if ptrs.offsets != NULL: os_aligned_free(ptrs.offsets)
    if ptrs.types != NULL: os_aligned_free(ptrs.types)
    if ptrs.cols != NULL: os_aligned_free(ptrs.cols)
    if ptrs.strides != NULL: os_aligned_free(ptrs.strides)

# ============================================================
# Fast row hashing and comparison using extracted pointers
# ============================================================

cdef uint64_t _gb_hash_row(_ByColPtrs ptrs, Py_ssize_t row_idx) nogil:
    cdef uint64_t h = 14695981039346656037ULL
    cdef int col_idx, sub, n_sub
    cdef uint64_t val
    cdef int* i_data
    cdef double* d_data
    cdef long long* t_data
    cdef char* c_data
    cdef unsigned char* b_data
    cdef Py_UCS2* u_data
    cdef Py_ssize_t start, end, stride

    for col_idx in range(ptrs.count):
        n_sub = ptrs.cols[col_idx]
        stride = ptrs.strides[col_idx]
        if ptrs.types[col_idx] == TYPE_INT:
            i_data = <int*>ptrs.data[col_idx]
            for sub in range(n_sub):
                h = _gb_mix_hash(h, <uint64_t>i_data[sub * stride + row_idx])
        elif ptrs.types[col_idx] == TYPE_DOUBLE:
            d_data = <double*>ptrs.data[col_idx]
            for sub in range(n_sub):
                val = (<uint64_t*>(&d_data[sub * stride + row_idx]))[0]
                h = _gb_mix_hash(h, val)
        elif ptrs.types[col_idx] == TYPE_DATE:
            t_data = <long long*>ptrs.data[col_idx]
            for sub in range(n_sub):
                h = _gb_mix_hash(h, <uint64_t>t_data[sub * stride + row_idx])
        elif ptrs.types[col_idx] == TYPE_CHAR256:
            c_data = <char*>ptrs.data[col_idx]
            start = (<Py_ssize_t*>ptrs.offsets[col_idx])[row_idx]
            end = (<Py_ssize_t*>ptrs.offsets[col_idx])[row_idx + 1]
            h = _gb_mix_hash(h, _gb_fnv1a_bytes(c_data + start, end - start))
        elif ptrs.types[col_idx] == TYPE_UCS2:
            u_data = <Py_UCS2*>ptrs.data[col_idx]
            start = (<Py_ssize_t*>ptrs.offsets[col_idx])[row_idx]
            end = (<Py_ssize_t*>ptrs.offsets[col_idx])[row_idx + 1]
            h = _gb_mix_hash(h, _gb_fnv1a_ucs2(u_data + start, end - start))
        elif ptrs.types[col_idx] == TYPE_BOOL:
            b_data = <unsigned char*>ptrs.data[col_idx]
            for sub in range(n_sub):
                h = _gb_mix_hash(h, <uint64_t>b_data[sub * stride + row_idx])
        elif ptrs.types[col_idx] == TYPE_CATEGORY:
            i_data = <int*>ptrs.data[col_idx]
            for sub in range(n_sub):
                h = _gb_mix_hash(h, <uint64_t>i_data[sub * stride + row_idx])
        elif ptrs.types[col_idx] == TYPE_DATETIME:
            t_data = <long long*>ptrs.data[col_idx]
            for sub in range(n_sub):
                h = _gb_mix_hash(h, <uint64_t>t_data[sub * stride + row_idx])
        elif ptrs.types[col_idx] == TYPE_TIME:
            i_data = <int*>ptrs.data[col_idx]
            for sub in range(n_sub):
                h = _gb_mix_hash(h, <uint64_t>i_data[sub * stride + row_idx])
        else:
            h = _gb_mix_hash(h, 0)
    return h

cdef bint _gb_rows_equal(_ByColPtrs ptrs, Py_ssize_t row1, Py_ssize_t row2) noexcept nogil:
    cdef int col_idx, sub, n_sub
    cdef int* i_data
    cdef double* d_data
    cdef long long* t_data
    cdef char* c_data
    cdef unsigned char* b_data
    cdef Py_ssize_t start1, end1, start2, end2, k, stride

    for col_idx in range(ptrs.count):
        n_sub = ptrs.cols[col_idx]
        stride = ptrs.strides[col_idx]
        if ptrs.types[col_idx] == TYPE_INT:
            i_data = <int*>ptrs.data[col_idx]
            for sub in range(n_sub):
                if i_data[sub * stride + row1] != i_data[sub * stride + row2]: return False
        elif ptrs.types[col_idx] == TYPE_DOUBLE:
            d_data = <double*>ptrs.data[col_idx]
            for sub in range(n_sub):
                if d_data[sub * stride + row1] != d_data[sub * stride + row2]: return False
        elif ptrs.types[col_idx] == TYPE_DATE:
            t_data = <long long*>ptrs.data[col_idx]
            for sub in range(n_sub):
                if t_data[sub * stride + row1] != t_data[sub * stride + row2]: return False
        elif ptrs.types[col_idx] == TYPE_CHAR256:
            c_data = <char*>ptrs.data[col_idx]
            start1 = (<Py_ssize_t*>ptrs.offsets[col_idx])[row1]
            end1 = (<Py_ssize_t*>ptrs.offsets[col_idx])[row1 + 1]
            start2 = (<Py_ssize_t*>ptrs.offsets[col_idx])[row2]
            end2 = (<Py_ssize_t*>ptrs.offsets[col_idx])[row2 + 1]
            if (end1 - start1) != (end2 - start2): return False
            for k in range(end1 - start1):
                if c_data[start1 + k] != c_data[start2 + k]: return False
        elif ptrs.types[col_idx] == TYPE_UCS2:
            start1 = (<Py_ssize_t*>ptrs.offsets[col_idx])[row1]
            end1 = (<Py_ssize_t*>ptrs.offsets[col_idx])[row1 + 1]
            start2 = (<Py_ssize_t*>ptrs.offsets[col_idx])[row2]
            end2 = (<Py_ssize_t*>ptrs.offsets[col_idx])[row2 + 1]
            if (end1 - start1) != (end2 - start2): return False
            i_data = <int*>ptrs.data[col_idx]
            for k in range(end1 - start1):
                if (<Py_UCS2*>i_data)[start1 + k] != (<Py_UCS2*>i_data)[start2 + k]: return False
        elif ptrs.types[col_idx] == TYPE_BOOL:
            b_data = <unsigned char*>ptrs.data[col_idx]
            for sub in range(n_sub):
                if b_data[sub * stride + row1] != b_data[sub * stride + row2]: return False
        elif ptrs.types[col_idx] == TYPE_CATEGORY:
            i_data = <int*>ptrs.data[col_idx]
            for sub in range(n_sub):
                if i_data[sub * stride + row1] != i_data[sub * stride + row2]: return False
        elif ptrs.types[col_idx] == TYPE_DATETIME:
            t_data = <long long*>ptrs.data[col_idx]
            for sub in range(n_sub):
                if t_data[sub * stride + row1] != t_data[sub * stride + row2]: return False
        elif ptrs.types[col_idx] == TYPE_TIME:
            i_data = <int*>ptrs.data[col_idx]
            for sub in range(n_sub):
                if i_data[sub * stride + row1] != i_data[sub * stride + row2]: return False
    return True

# ============================================================
# Group building
# ============================================================

cdef int _gb_build_groups(DataFrame df, list by_cols,
                            _GroupHashTable* ht, _GroupRows* gr,
                            _ByColPtrs ptrs) except -1:
    cdef Py_ssize_t n = df.df_length
    cdef Py_ssize_t i
    cdef uint64_t h
    cdef int g_id, ret
    cdef Py_ssize_t first_seen_row
    cdef bint found
    cdef int existing_g

    for i in range(n):
        h = _gb_hash_row(ptrs, i)
        g_id = _ght_get_or_insert(ht, h)

        if g_id == -1:
            raise MemoryError("GroupBy hash table capacity exceeded limit")
        if g_id >= gr.count:
            ret = _group_rows_add_group(gr)
            if ret < 0: raise MemoryError("Failed to add group in GroupBy")
        first_seen_row = gr.rows[g_id][0] if gr.row_sizes[g_id] > 0 else -1
        if first_seen_row >= 0 and not _gb_rows_equal(ptrs, i, first_seen_row):
            found = False
            for existing_g in range(gr.count):
                if gr.row_sizes[existing_g] > 0:
                    if _gb_rows_equal(ptrs, i, gr.rows[existing_g][0]):
                        ret = _group_rows_append(gr, existing_g, i)
                        if ret < 0: raise MemoryError("Failed to append row in GroupBy")
                        found = True
                        break
            if not found:
                g_id = gr.count
                ret = _group_rows_add_group(gr)
                if ret < 0: raise MemoryError("Failed to add group in GroupBy")
                ret = _group_rows_append(gr, g_id, i)
                if ret < 0: raise MemoryError("Failed to append row in GroupBy")
        else:
            ret = _group_rows_append(gr, g_id, i)
            if ret < 0: raise MemoryError("Failed to append row in GroupBy")

# ============================================================
# Aggregation helpers
# ============================================================

cdef inline double _gb_agg_int_sub(int* ptr, Py_ssize_t stride, int sub, Py_ssize_t* idxs, Py_ssize_t sz, _AggFunc agg) noexcept nogil:
    cdef double val = 0.0, d, mean = 0.0, M2 = 0.0, delta, delta2
    cdef Py_ssize_t i, base = sub * stride

    if agg == _AGG_SUM:
        for i in range(sz): val += ptr[base + idxs[i]]
        return val
    if agg == _AGG_MEAN:
        if sz == 0: return NAN
        for i in range(sz): val += ptr[base + idxs[i]]
        return val / <double>sz
    if agg == _AGG_MIN:
        if sz == 0: return NAN
        val = <double>ptr[base + idxs[0]]
        for i in range(1, sz):
            d = <double>ptr[base + idxs[i]]
            if d < val: val = d
        return val
    if agg == _AGG_MAX:
        if sz == 0: return NAN
        val = <double>ptr[base + idxs[0]]
        for i in range(1, sz):
            d = <double>ptr[base + idxs[i]]
            if d > val: val = d
        return val
    if agg == _AGG_VAR or agg == _AGG_STD:
        if sz < 2: return NAN
        # 采用 Welford 算法实现单趟内存访问，省掉一次间接指针查询，速度翻倍
        for i in range(sz):
            val = <double>ptr[base + idxs[i]]
            delta = val - mean
            mean += delta / (i + 1)
            delta2 = val - mean
            M2 += delta * delta2

        if agg == _AGG_VAR: return M2 / <double>(sz - 1)
        return sqrt(M2 / <double>(sz - 1))

    return NAN

cdef inline double _gb_agg_double_sub(double* ptr, Py_ssize_t stride, int sub, Py_ssize_t* idxs, Py_ssize_t sz, _AggFunc agg) noexcept nogil:
    cdef double val = 0.0, d, mean = 0.0, M2 = 0.0, delta, delta2
    cdef Py_ssize_t i, base = sub * stride

    if agg == _AGG_SUM:
        for i in range(sz): val += ptr[base + idxs[i]]
        return val

    if agg == _AGG_MEAN:
        if sz == 0: return NAN
        for i in range(sz): val += ptr[base + idxs[i]]
        return val / <double>sz

    # 使用 Welford 处理 VAR/STD
    if agg == _AGG_VAR or agg == _AGG_STD:
        if sz < 2: return NAN
        for i in range(sz):
            val = ptr[base + idxs[i]]
            delta = val - mean
            mean += delta / (i + 1)
            delta2 = val - mean
            M2 += delta * delta2

        if agg == _AGG_VAR: return M2 / <double>(sz - 1)
        return sqrt(M2 / <double>(sz - 1))

    if agg == _AGG_MIN:
        if sz == 0: return NAN
        val = ptr[base + idxs[0]]
        for i in range(1, sz):
            d = ptr[base + idxs[i]]
            if d < val: val = d
        return val

    if agg == _AGG_MAX:
        if sz == 0: return NAN
        val = ptr[base + idxs[0]]
        for i in range(1, sz):
            d = ptr[base + idxs[i]]
            if d > val: val = d
        return val
    return NAN

cdef DoubleColumn _gb_native_agg(DataFrame df, _GroupRows* gr, str col_name, str func):
    cdef BaseColumn orig_col = df._columns[col_name]
    cdef bint is2d
    cdef Py_ssize_t c_col, r_col
    is2d, c_col, r_col = _col_shape_probe(orig_col)

    cdef Py_ssize_t out_n = gr.count
    cdef _AggFunc agg_type = _resolve_agg_func(func)
    cdef int g_id, sub
    cdef Py_ssize_t sz
    cdef DoubleColumn res_col
    cdef double* out_ptr

    if is2d and c_col > 1:
        res_col = DoubleColumn(max(out_n * c_col, 1), cols=c_col)
        res_col.length = out_n * c_col
        if orig_col.dtype == TYPE_DOUBLE:
            d_ptr = (<DoubleColumn>orig_col).data_c
            for sub in range(c_col):
                for g_id in range(out_n):
                    res_col.data_c[sub * out_n + g_id] = _gb_agg_double_sub(
                        d_ptr, r_col, sub, gr.rows[g_id], gr.row_sizes[g_id], agg_type)
        else:
            i_ptr = (<IntColumn>orig_col).data_c
            for sub in range(c_col):
                for g_id in range(out_n):
                    res_col.data_c[sub * out_n + g_id] = _gb_agg_int_sub(
                        i_ptr, r_col, sub, gr.rows[g_id], gr.row_sizes[g_id], agg_type)
        return res_col

    res_col = DoubleColumn(max(out_n, 1))
    res_col.length = out_n
    out_ptr = res_col.data_c

    if orig_col.dtype == TYPE_DOUBLE:
        col_dbl = <DoubleColumn>orig_col
        if agg_type == _AGG_SUM:
            for g_id in range(out_n): out_ptr[g_id] = col_dbl._sum_rows(gr.rows[g_id], gr.row_sizes[g_id])
        elif agg_type == _AGG_MEAN:
            for g_id in range(out_n): out_ptr[g_id] = col_dbl._mean_rows(gr.rows[g_id], gr.row_sizes[g_id])
        elif agg_type == _AGG_MIN:
            for g_id in range(out_n): out_ptr[g_id] = col_dbl._min_rows(gr.rows[g_id], gr.row_sizes[g_id])
        elif agg_type == _AGG_MAX:
            for g_id in range(out_n): out_ptr[g_id] = col_dbl._max_rows(gr.rows[g_id], gr.row_sizes[g_id])
        elif agg_type == _AGG_VAR:
            for g_id in range(out_n): out_ptr[g_id] = col_dbl._var_rows(gr.rows[g_id], gr.row_sizes[g_id])
        elif agg_type == _AGG_STD:
            for g_id in range(out_n): out_ptr[g_id] = col_dbl._std_rows(gr.rows[g_id], gr.row_sizes[g_id])
    elif orig_col.dtype == TYPE_INT:
        col_int = <IntColumn>orig_col
        if agg_type == _AGG_SUM:
            for g_id in range(out_n): out_ptr[g_id] = col_int._sum_rows(gr.rows[g_id], gr.row_sizes[g_id])
        elif agg_type == _AGG_MEAN:
            for g_id in range(out_n): out_ptr[g_id] = col_int._mean_rows(gr.rows[g_id], gr.row_sizes[g_id])
        elif agg_type == _AGG_MIN:
            for g_id in range(out_n): out_ptr[g_id] = col_int._min_rows(gr.rows[g_id], gr.row_sizes[g_id])
        elif agg_type == _AGG_MAX:
            for g_id in range(out_n): out_ptr[g_id] = col_int._max_rows(gr.rows[g_id], gr.row_sizes[g_id])
        elif agg_type == _AGG_VAR:
            for g_id in range(out_n): out_ptr[g_id] = col_int._var_rows(gr.rows[g_id], gr.row_sizes[g_id])
        elif agg_type == _AGG_STD:
            for g_id in range(out_n): out_ptr[g_id] = col_int._std_rows(gr.rows[g_id], gr.row_sizes[g_id])

    return res_col

cdef IntColumn _gb_native_count(_GroupRows* gr):
    cdef IntColumn res_col = IntColumn(max(gr.count, 1))
    res_col.length = gr.count
    cdef int g_id
    for g_id in range(gr.count):
        res_col.data_c[g_id] = <int>gr.row_sizes[g_id]
    return res_col

cdef object _gb_extract_key_column(DataFrame df, _GroupRows* gr, str col_name):
    cdef BaseColumn orig = df._columns[col_name]
    cdef int i
    cdef IntColumn res_int
    cdef DoubleColumn res_dbl
    cdef Py_ssize_t* fs
    cdef DataFrame tmp

    # 使用 shape_probe
    cdef bint is2d
    cdef Py_ssize_t c_col, r_col
    is2d, c_col, r_col = _col_shape_probe(orig)

    # 如果是 2D 类型 或者 是不支持直拷的复杂类型，走安全的 `_take` API
    if (is2d and c_col > 1) or (orig.dtype != TYPE_INT and orig.dtype != TYPE_DOUBLE):
        fs = <Py_ssize_t*>aligned_malloc(max(gr.count, 1) * sizeof(Py_ssize_t))
        if not fs: raise MemoryError()
        for i in range(gr.count):
            fs[i] = gr.rows[i][0]
        tmp = df._take(fs, gr.count)
        os_aligned_free(fs)
        return tmp._columns[col_name]

    # 1D 的极速直拷路径
    if orig.dtype == TYPE_INT:
        res_int = IntColumn(max(gr.count, 1))
        res_int.length = gr.count
        for i in range(gr.count):
            res_int.data_c[i] = (<IntColumn>orig).data_c[gr.rows[i][0]]
        return res_int
    elif orig.dtype == TYPE_DOUBLE:
        res_dbl = DoubleColumn(max(gr.count, 1))
        res_dbl.length = gr.count
        for i in range(gr.count):
            res_dbl.data_c[i] = (<DoubleColumn>orig).data_c[gr.rows[i][0]]
        return res_dbl

cdef IntColumn _gb_native_idx(DataFrame df, _GroupRows* gr, str col_name, bint find_max):
    # 2D 列取主子列（sub=0）比较：idxmax/idxmin 不确定向量级比较，沿用 _get_col_value_as_double 的 sub0 约定
    cdef IntColumn res_col = IntColumn(max(gr.count, 1))
    res_col.length = gr.count
    cdef BaseColumn orig_col = df._columns[col_name]
    cdef int g_id
    cdef Py_ssize_t sz, row_idx, best_row, i
    cdef double best_val, cur_val
    for g_id in range(gr.count):
        sz = gr.row_sizes[g_id]
        if sz == 0:
            res_col.data_c[g_id] = 0
            continue
        best_row = gr.rows[g_id][0]
        best_val = df._get_col_value_as_double(orig_col, best_row)
        for i in range(1, sz):
            row_idx = gr.rows[g_id][i]
            cur_val = df._get_col_value_as_double(orig_col, row_idx)
            if find_max:
                if cur_val > best_val:
                    best_val = cur_val
                    best_row = row_idx
            else:
                if cur_val < best_val:
                    best_val = cur_val
                    best_row = row_idx
        res_col.data_c[g_id] = <int>best_row
    return res_col

cdef object _gb_python_compute(list vals, str func_name):
    """单子列 Python 级聚合计算（维持原 median/quantile 的语义不变）"""
    cdef list sorted_vals
    cdef Py_ssize_t mid
    if func_name == 'sum':
        return sum(vals)
    elif func_name == 'mean':
        return sum(vals) / len(vals) if vals else NAN
    elif func_name == 'min':
        return min(vals) if vals else NAN
    elif func_name == 'max':
        return max(vals) if vals else NAN
    elif func_name == 'count':
        return len(vals)
    elif func_name == 'median':
        if not vals:
            return NAN
        sorted_vals = sorted(vals)
        mid = len(sorted_vals) // 2
        if len(sorted_vals) % 2 == 0:
            return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
        return sorted_vals[mid]
    elif func_name == 'quantile':
        if not vals:
            return NAN
        sorted_vals = sorted(vals)
        mid = len(sorted_vals) // 2
        if len(sorted_vals) % 2 == 0:
            return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
        return sorted_vals[mid]
    else:
        raise ValueError(f"Unsupported python aggregation function: {func_name}")

cdef object _gb_python_agg(DataFrame df, _GroupRows* gr, str col_name, str func_name):
    cdef list results = []
    cdef int g_id, sub
    cdef Py_ssize_t sz, row_idx, i, out_n = gr.count
    cdef BaseColumn orig_col = df._columns[col_name]
    cdef list vals
    cdef bint is2d
    cdef Py_ssize_t c_col, r_col
    cdef DoubleColumn res_col

    is2d, c_col, r_col = _col_shape_probe(orig_col)

    cdef double* d_ptr = NULL
    cdef int* i_ptr = NULL
    if orig_col.dtype == TYPE_DOUBLE:
        d_ptr = (<DoubleColumn>orig_col).data_c
    elif orig_col.dtype == TYPE_INT:
        i_ptr = (<IntColumn>orig_col).data_c

    if is2d and c_col > 1:
        res_col = DoubleColumn(max(out_n * c_col, 1), cols=c_col)
        res_col.length = out_n * c_col
        for sub in range(c_col):
            results = []
            for g_id in range(out_n):
                sz = gr.row_sizes[g_id]
                vals = []
                for i in range(sz):
                    row_idx = gr.rows[g_id][i]
                    if d_ptr != NULL:
                        vals.append(d_ptr[sub * r_col + row_idx])
                    elif i_ptr != NULL:
                        vals.append(<double>i_ptr[sub * r_col + row_idx])
                    else:
                        vals.append(_col_row_key(orig_col, row_idx)[sub])
                results.append(_gb_python_compute(vals, func_name))
            for g_id in range(out_n):
                res_col.data_c[sub * out_n + g_id] = <double>results[g_id] if results[g_id] is not None else NAN
        return res_col

    res_col = DoubleColumn(max(out_n, 1))
    res_col.length = out_n
    for g_id in range(out_n):
        sz = gr.row_sizes[g_id]
        vals = []
        for i in range(sz):
            row_idx = gr.rows[g_id][i]
            vals.append(orig_col[row_idx])
        results.append(_gb_python_compute(vals, func_name))
    for g_id in range(out_n):
        res_col.data_c[g_id] = <double>results[g_id] if results[g_id] is not None else NAN
    return res_col



cdef inline _DataFrameGroupByAccessor _make_groupby_accessor(DataFrame df, list by_cols):
    cdef _DataFrameGroupByAccessor acc = <_DataFrameGroupByAccessor>_DataFrameGroupByAccessor.__new__(_DataFrameGroupByAccessor)
    acc._df = df
    acc._by_cols = by_cols
    acc._ht = NULL
    acc._gr = NULL
    acc._ptrs.data = NULL
    acc._ptrs.offsets = NULL
    acc._ptrs.types = NULL
    acc._ptrs.count = 0
    acc._built = False
    return acc


cdef inline Py_ssize_t _linalg_logical_cols(DataFrame df):
    """矩阵逻辑列数：2D 列（is_2d 且 cols>1）按子列展开各计 1，1D 列计 1。"""
    cdef Py_ssize_t total = 0
    cdef bint is2d
    cdef Py_ssize_t c, r
    for name, col in df._columns.items():
        is2d, c, r = _col_shape_probe(col)
        total += c if (is2d and c > 1) else 1
    return total

cdef bint _linalg_is_contiguous_columns(DataFrame df):
    """安全修复版：严谨校验多列 DataFrame 底层是否属于同一块物理连续内存"""
    cdef list cols_list = list(df._columns.values())
    cdef Py_ssize_t cols = len(cols_list)
    if cols < 2:
        return False

    cdef BaseColumn col = <BaseColumn>cols_list[0]
    if col.dtype != TYPE_DOUBLE:
        return False

    cdef double* prev_ptr = (<DoubleColumn>col).data_c
    cdef Py_ssize_t rows = df.df_length
    cdef double* curr_ptr
    cdef Py_ssize_t i

    for i in range(1, cols):
        col = <BaseColumn>cols_list[i]
        if col.dtype != TYPE_DOUBLE:
            return False
        curr_ptr = (<DoubleColumn>col).data_c
        if curr_ptr != prev_ptr + rows:
            return False
        prev_ptr = curr_ptr

    return True

cdef double* _linalg_to_contiguous_fast(DataFrame df, Py_ssize_t* out_rows, Py_ssize_t* out_cols) except NULL:
    cdef Py_ssize_t rows = df.df_length
    cdef list cols_list = list(df._columns.values())
    cdef Py_ssize_t cols = len(cols_list)

    if cols == 0 or rows == 0:
        out_rows[0] = rows
        out_cols[0] = 0
        return NULL

    cdef Py_ssize_t logical_cols = _linalg_logical_cols(df)
    cdef Py_ssize_t num_elements = rows * logical_cols

    cdef double* data = <double*>aligned_malloc(num_elements * sizeof(double))
    if not data:
        raise MemoryError()

    cdef Py_ssize_t i, j
    cdef BaseColumn col
    if logical_cols != cols:
        os_aligned_free(data)
        return _linalg_to_contiguous(df, out_rows, out_cols)

    try:
        for j in range(cols):
            col = <BaseColumn>cols_list[j]
            if col.dtype != TYPE_DOUBLE:
                os_aligned_free(data)
                return _linalg_to_contiguous(df, out_rows, out_cols)
            with nogil:
                for i in range(rows):
                    data[i * logical_cols + j] = (<DoubleColumn>col).data_c[i]
    except Exception:
        os_aligned_free(data)
        raise

    out_rows[0] = rows
    out_cols[0] = logical_cols
    return data

cdef double* _linalg_to_contiguous(DataFrame df, Py_ssize_t* out_rows, Py_ssize_t* out_cols) except NULL:
    cdef Py_ssize_t rows = df.df_length
    cdef list cols_list = list(df._columns.values())
    cdef list name_list = list(df._columns.keys())
    cdef Py_ssize_t n_cols_named = len(cols_list)
    cdef Py_ssize_t logical_cols = _linalg_logical_cols(df)

    if rows == 0 or logical_cols == 0:
        out_rows[0] = rows
        out_cols[0] = logical_cols
        return NULL

    cdef Py_ssize_t num_elements = rows * logical_cols
    cdef double* data = <double*>aligned_malloc(num_elements * sizeof(double))
    if not data: raise MemoryError()

    cdef Py_ssize_t i, j, sub, out_col = 0
    cdef BaseColumn col
    cdef int col_dtype
    cdef int* int_ptr
    cdef double* double_ptr
    cdef bint is2d
    cdef Py_ssize_t c_col, r_col, n_sub

    try:
        for j in range(n_cols_named):
            col = <BaseColumn>cols_list[j]
            col_dtype = col.dtype
            if col_dtype != TYPE_INT and col_dtype != TYPE_DOUBLE:
                raise TypeError(f"Cannot perform linear algebra on non-numeric column '{name_list[j]}'")

            is2d, c_col, r_col = _col_shape_probe(col)
            n_sub = c_col if (is2d and c_col > 1) else 1

            if col_dtype == TYPE_INT:
                int_ptr = (<IntColumn>col).data_c
                with nogil:
                    for sub in range(n_sub):
                        for i in range(rows):
                            data[i * logical_cols + out_col + sub] = <double>int_ptr[sub * r_col + i]
            else:
                double_ptr = (<DoubleColumn>col).data_c
                with nogil:
                    for sub in range(n_sub):
                        for i in range(rows):
                            data[i * logical_cols + out_col + sub] = double_ptr[sub * r_col + i]
            out_col += n_sub
    except Exception:
        os_aligned_free(data)
        raise

    out_rows[0] = rows
    out_cols[0] = logical_cols
    return data


cdef DataFrame _linalg_from_contiguous(double* data, Py_ssize_t rows, Py_ssize_t cols):
    """
    消除碎列，生成完美的 2D 连续内存块以匹配 DataFrame 核心设计，避免转置惩罚。
    (输入 data 预期为 行优先 C-contiguous)
    """
    cdef DataFrame df = DataFrame()
    df.df_length = rows
    df.index = RangeIndex(0, rows)

    if rows == 0 or cols == 0:
        return df

    cdef Py_ssize_t total_elements = rows * cols

    # 全量数据申请一块 2D Column，符合 F-contiguous(列优先) 布局
    cdef DoubleColumn block2d = DoubleColumn(total_elements, cols=cols)
    block2d.length = total_elements
    cdef double* b_ptr = block2d.data_c

    cdef Py_ssize_t i, j
    # 矩阵转置：将传入的 C-contiguous (行优先) 转为底层框架期待的 F-contiguous (列优先)
    with nogil:
        for i in range(rows):
            for j in range(cols):
                b_ptr[j * rows + i] = data[i * cols + j]

    cdef list col_names = [chr(65 + i) if i < 26 else f"Col_{i}" for i in range(cols)]

    # 构建零拷贝视图 (Zero-copy views)
    for j in range(cols):
        df._columns[col_names[j]] = DoubleColumn.create_view(b_ptr + j * rows, rows, block2d)

    # 保留对 2D 块的强引用，防止内存被 GC
    if not hasattr(df, '_2d_blocks'):
        df._2d_blocks = []
    df._2d_blocks.append(block2d)

    return df

cdef inline double _linalg_clip_val(double v, double min_v, double max_v) nogil:
    if v < min_v: return min_v
    if v > max_v: return max_v
    return v

# ============================================================
#  Core linalg operations
# ============================================================

cdef DataFrame _linalg_dot(DataFrame a, DataFrame b):
    cdef Py_ssize_t a_rows = a.df_length
    cdef Py_ssize_t a_cols = _linalg_logical_cols(a)
    cdef Py_ssize_t b_rows = b.df_length
    cdef Py_ssize_t b_cols = _linalg_logical_cols(b)

    if a_cols != b_rows:
        raise ValueError(f"Shape mismatch: ({a_rows}, {a_cols}) x ({b_rows}, {b_cols})")
    if a_rows == 0 or b_cols == 0 or a_cols == 0:
        return DataFrame()

    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t size_c
    cdef double s
    cdef double* A = NULL
    cdef double* B = NULL
    cdef double* C = NULL

    try:
        A = _linalg_to_contiguous(a, &a_rows, &a_cols)
        B = _linalg_to_contiguous(b, &b_rows, &b_cols)

        size_c = a_rows * b_cols
        C = <double*>aligned_malloc(size_c * sizeof(double))
        if not C: raise MemoryError()
        memset(C, 0, size_c * sizeof(double))

        # O(N^3) 行优先快速乘法（对缓存友好）
        with nogil:
            for i in range(a_rows):
                for k in range(a_cols):
                    s = A[i * a_cols + k]
                    if s != 0:
                        for j in range(b_cols):
                            C[i * b_cols + j] += s * B[k * b_cols + j]

        return _linalg_from_contiguous(C, a_rows, b_cols)
    finally:
        if A != NULL: os_aligned_free(A)
        if B != NULL: os_aligned_free(B)
        if C != NULL: os_aligned_free(C)

cdef DataFrame _linalg_transpose(DataFrame df):
    cdef Py_ssize_t rows, cols, i, j
    cdef Py_ssize_t size
    cdef double* A = NULL
    cdef double* B = NULL

    if df.df_length == 0 or _linalg_logical_cols(df) == 0:
        return DataFrame()
    try:
        A = _linalg_to_contiguous(df, &rows, &cols)
        size = rows * cols
        B = <double*>aligned_malloc(size * sizeof(double))
        if not B: raise MemoryError()

        with nogil:
            for i in range(rows):
                for j in range(cols):
                    B[j * rows + i] = A[i * cols + j]

        return _linalg_from_contiguous(B, cols, rows)
    finally:
        if A != NULL: os_aligned_free(A)
        if B != NULL: os_aligned_free(B)

cdef double _linalg_det(DataFrame df):
    cdef Py_ssize_t n = df.df_length
    if n == 0:
        return 1.0

    if _linalg_logical_cols(df) != n:
        raise ValueError("Matrix must be square")

    cdef Py_ssize_t i, j, k, pivot_row
    cdef double* A = NULL
    cdef double* LU = NULL
    cdef double det_val = 1.0
    cdef double pivot, temp, max_val, abs_val
    cdef int sign = 1

    try:
        A = _linalg_to_contiguous(df, &n, &n)
        LU = <double*>aligned_malloc(n * n * sizeof(double))
        if not LU: raise MemoryError()
        memcpy(LU, A, n * n * sizeof(double))

        with nogil:
            for i in range(n):
                pivot_row = i
                max_val = fabs(LU[i * n + i])
                for k in range(i + 1, n):
                    abs_val = fabs(LU[k * n + i])
                    if abs_val > max_val:
                        max_val = abs_val
                        pivot_row = k
                if max_val < 1e-15:
                    det_val = 0.0
                    break
                if pivot_row != i:
                    for j in range(i, n):
                        temp = LU[i * n + j]
                        LU[i * n + j] = LU[pivot_row * n + j]
                        LU[pivot_row * n + j] = temp
                    sign = -sign
                pivot = LU[i * n + i]
                for k in range(i + 1, n):
                    temp = LU[k * n + i] / pivot
                    for j in range(i + 1, n):
                        LU[k * n + j] -= temp * LU[i * n + j]

            if det_val != 0.0:
                for i in range(n):
                    det_val *= LU[i * n + i]
                det_val *= sign

        return det_val
    finally:
        if A != NULL: os_aligned_free(A)
        if LU != NULL: os_aligned_free(LU)

cdef DataFrame _linalg_inv(DataFrame df):
    cdef Py_ssize_t n = df.df_length
    if n == 0:
        return DataFrame()

    if _linalg_logical_cols(df) != n:
        raise ValueError("Matrix must be square")

    cdef Py_ssize_t rows, cols, i, j, k, pivot_row
    cdef double* A = NULL
    cdef double* aug = NULL
    cdef double* result_data = NULL
    cdef double pivot, temp, max_val, abs_val
    cdef int singular = 0

    try:
        A = _linalg_to_contiguous(df, &rows, &cols)
        aug = <double*>aligned_malloc(n * 2 * n * sizeof(double))
        if not aug: raise MemoryError()
        with nogil:
            for i in range(n):
                for j in range(n):
                    aug[i * 2 * n + j] = A[i * n + j]
                    aug[i * 2 * n + n + j] = 1.0 if i == j else 0.0
            for i in range(n):
                pivot_row = i
                max_val = fabs(aug[i * 2 * n + i])
                for k in range(i + 1, n):
                    abs_val = fabs(aug[k * 2 * n + i])
                    if abs_val > max_val:
                        max_val = abs_val
                        pivot_row = k
                if max_val < 1e-15:
                    singular = 1
                    break

                if pivot_row != i:
                    for j in range(i, 2 * n):
                        temp = aug[i * 2 * n + j]
                        aug[i * 2 * n + j] = aug[pivot_row * 2 * n + j]
                        aug[pivot_row * 2 * n + j] = temp

                pivot = aug[i * 2 * n + i]
                for j in range(i, 2 * n):
                    aug[i * 2 * n + j] /= pivot

                for k in range(n):
                    if k == i: continue
                    temp = aug[k * 2 * n + i]
                    for j in range(i, 2 * n):
                        aug[k * 2 * n + j] -= temp * aug[i * 2 * n + j]

        if singular:
            raise ValueError("Matrix is singular")

        result_data = <double*>aligned_malloc(n * n * sizeof(double))
        if not result_data: raise MemoryError()

        with nogil:
            for i in range(n):
                for j in range(n):
                    result_data[i * n + j] = aug[i * 2 * n + n + j]

        return _linalg_from_contiguous(result_data, n, n)
    finally:
        if A != NULL: os_aligned_free(A)
        if aug != NULL: os_aligned_free(aug)
        if result_data != NULL: os_aligned_free(result_data)

cdef DataFrame _linalg_solve(DataFrame A_df, DataFrame b_df):
    cdef Py_ssize_t n = A_df.df_length
    if n == 0:
        return DataFrame()

    cdef Py_ssize_t rows, cols, i, j, k, pivot_row
    cdef double* a_data = NULL
    cdef double* b_data = NULL
    cdef double* lu = NULL
    cdef double* x = NULL
    cdef double pivot, temp, max_val, abs_val, sum_val

    if _linalg_logical_cols(A_df) != n:
        raise ValueError("Matrix A must be square")
    if n != b_df.df_length:
        raise ValueError("Shape mismatch: A and b must have same number of rows")
    if _linalg_logical_cols(b_df) != 1:
        raise ValueError("b must have exactly one column")

    try:
        a_data = _linalg_to_contiguous(A_df, &rows, &cols)
        b_data = _linalg_to_contiguous(b_df, &rows, &cols)
        lu = <double*>aligned_malloc(n * n * sizeof(double))
        x = <double*>aligned_malloc(n * sizeof(double))
        if not lu or not x: raise MemoryError()
        memcpy(lu, a_data, n * n * sizeof(double))

        with nogil:
            for i in range(n):
                x[i] = b_data[i]
            for i in range(n):
                pivot_row = i
                max_val = fabs(lu[i * n + i])
                for k in range(i + 1, n):
                    abs_val = fabs(lu[k * n + i])
                    if abs_val > max_val:
                        max_val = abs_val
                        pivot_row = k
                if max_val < 1e-15:
                    raise ValueError("Matrix is singular")
                if pivot_row != i:
                    for j in range(i, n):
                        temp = lu[i * n + j]
                        lu[i * n + j] = lu[pivot_row * n + j]
                        lu[pivot_row * n + j] = temp
                    temp = x[i]
                    x[i] = x[pivot_row]
                    x[pivot_row] = temp
                pivot = lu[i * n + i]
                for k in range(i + 1, n):
                    temp = lu[k * n + i] / pivot
                    for j in range(i + 1, n):
                        lu[k * n + j] -= temp * lu[i * n + j]
                    x[k] -= temp * x[i]
            for i in range(n - 1, -1, -1):
                sum_val = x[i]
                for j in range(i + 1, n):
                    sum_val -= lu[i * n + j] * x[j]
                x[i] = sum_val / lu[i * n + i]

        # solve 输出的是一个 1D 数组(列)
        result = DataFrame()
        result.df_length = n
        col = DoubleColumn(n)
        col.length = n
        for i in range(n):
            col.data_c[i] = x[i]
        b_col_name = list(b_df._columns.keys())[0]
        result._columns[b_col_name] = col
        return result
    finally:
        if a_data != NULL: os_aligned_free(a_data)
        if b_data != NULL: os_aligned_free(b_data)
        if lu != NULL: os_aligned_free(lu)
        if x != NULL: os_aligned_free(x)

cdef double _linalg_norm(DataFrame df, str ord_type="frobenius"):
    cdef Py_ssize_t rows, cols, n, i
    if df.df_length == 0:
        return 0.0

    cdef double* A = NULL
    cdef double result = 0.0
    cdef double val
    cdef int mode = 0

    if ord_type in ('frobenius', 'fro', '2'):
        mode = 1
    elif ord_type in ('count', 'zero'):
        mode = 2
    else:
        mode = 3

    try:
        A = _linalg_to_contiguous(df, &rows, &cols)
        n = rows * cols
        with nogil:
            if mode == 1:
                result = 0.0
                for i in range(n):
                    result += A[i] * A[i]
                result = sqrt(result)
            elif mode == 2:
                result = 0.0
                for i in range(n):
                    if fabs(A[i]) > 1e-15:
                        result += 1.0
            else:
                result = 0.0
                for i in range(n):
                    val = fabs(A[i])
                    if val > result:
                        result = val
        return result
    finally:
        if A != NULL: os_aligned_free(A)

# ============================================================
#  Pairwise distance / similarity metrics
# ============================================================

cdef DataFrame _linalg_pairwise(DataFrame df, int op):
    """共享脚手架的 pairwise 距离/相似度矩阵；仅内核与对角线随 op 变化。"""
    cdef Py_ssize_t rows = 0, cols = 0
    if df.df_length == 0: return DataFrame()

    cdef Py_ssize_t i, j
    cdef double val
    cdef double* A = NULL
    cdef double* D = NULL
    cdef double diag_val = 1.0 if (op == PW_COSINE_SIM or op == PW_JACCARD) else 0.0
    try:
        A = _linalg_to_contiguous_fast(df, &rows, &cols)
        D = <double*>aligned_malloc(rows * rows * sizeof(double))
        if not D: raise MemoryError()
        with nogil:
            for i in range(rows):
                D[i * rows + i] = diag_val
                for j in range(i + 1, rows):
                    if op == PW_EUCLIDEAN:
                        val = _vec_euclidean_dist(A + i * cols, A + j * cols, cols)
                    elif op == PW_MANHATTAN:
                        val = _vec_manhattan_dist(A + i * cols, A + j * cols, cols)
                    elif op == PW_COSINE_DIST:
                        val = 1.0 - _vec_cosine_sim(A + i * cols, A + j * cols, cols)
                    elif op == PW_COSINE_SIM:
                        val = _vec_cosine_sim(A + i * cols, A + j * cols, cols)
                    elif op == PW_HAMMING:
                        val = _vec_hamming_dist(A + i * cols, A + j * cols, cols)
                    else:
                        val = _vec_jaccard_sim(A + i * cols, A + j * cols, cols)
                    D[i * rows + j] = val
                    D[j * rows + i] = val
        return _linalg_from_contiguous(D, rows, rows)
    finally:
        if A != NULL: os_aligned_free(A)
        if D != NULL: os_aligned_free(D)


cdef inline _DataFrameLinalgAccessor _make_linalg_accessor(DataFrame df):
    cdef _DataFrameLinalgAccessor acc = <_DataFrameLinalgAccessor>_DataFrameLinalgAccessor.__new__(_DataFrameLinalgAccessor)
    acc._df = df
    return acc


cdef inline _DataFrameMetricAccessor _make_metric_accessor(DataFrame df):
    cdef _DataFrameMetricAccessor acc = <_DataFrameMetricAccessor>_DataFrameMetricAccessor.__new__(_DataFrameMetricAccessor)
    acc._df = df
    return acc


cdef inline BaseColumn _empty_col_for(BaseColumn col):
    if isinstance(col, IntColumn): return IntColumn()
    if isinstance(col, DoubleColumn): return DoubleColumn()
    if isinstance(col, DateColumn): return DateColumn()
    if isinstance(col, TimeColumn): return TimeColumn()
    if isinstance(col, DateTimeColumn): return DateTimeColumn()
    if isinstance(col, AsciiColumn): return AsciiColumn()
    if isinstance(col, UCS2Column): return UCS2Column()
    if isinstance(col, BoolColumn): return BoolColumn()
    if isinstance(col, CategoryColumn): return CategoryColumn()
    return IntColumn()


cdef inline void _set_2d_shape(BaseColumn col, Py_ssize_t c) noexcept:
    col.num_cols = <int>c
    col.is_2d = True
    col.owns_data = True
