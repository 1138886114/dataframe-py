# _fastdataframe_factory.pxi — Static factory methods (module-level cdef)
# Included at module level in fastdataframe.pyx (after class definition)

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

cdef _rename_columns_impl(FastDataFrame df, columns):
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
    cdef FastDataFrame df = FastDataFrame()
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
    cdef FastDataFrame df = FastDataFrame()
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
    cdef FastDataFrame df = FastDataFrame()
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
    cdef FastDataFrame df = FastDataFrame()
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
    cdef FastDataFrame df = FastDataFrame()
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
    cdef FastDataFrame df = FastDataFrame()
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
    cdef FastDataFrame df = FastDataFrame()
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
    cdef FastDataFrame df = FastDataFrame()
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
