# _fastdataframe_standalone.pxi — Standalone utility functions (module-level cdef)
# Included at module level in fastdataframe.pyx (after class definition)
# Migrated from fluidframe.pyx

import pickle as _pickle
import csv as _csv
from fluidframe._schema_types import Char256 as _Char256, Date as _Date, Bool as _Bool, \
    Category as _Category, DateTime as _DateTime, Time as _Time, _parse_schema as _parse_schema


class _SafeUnpickler(_pickle.Unpickler):
    """Safe unpickler that restricts deserialization to built-in types."""
    _ALLOWED_BUILTINS = frozenset({
        'dict', 'list', 'tuple', 'int', 'float', 'str', 'bytes',
        'NoneType', 'bool', 'set',
    })

    def __init__(self, file, allow_raw_pickle=False):
        super().__init__(file)
        self._allow_raw_pickle = allow_raw_pickle

    def find_class(self, module, name):
        if self._allow_raw_pickle:
            return super().find_class(module, name)
        if module == 'builtins' and name in self._ALLOWED_BUILTINS:
            return super().find_class(module, name)
        full = f"{module}.{name}"
        raise _pickle.UnpicklingError(
            f"Deserialization of '{full}' is not allowed. "
            f"Use allow_raw_pickle=True to bypass (unsafe)."
        )


def from_pickle(filename, bint allow_raw_pickle=False):
    """从 pickle 文件恢复 FastDataFrame。
    2D 列条目为 (typ, values, rows, cols)，重建后保持二维维度。"""
    cdef Py_ssize_t rows = 0, cols = 0
    cdef Py_ssize_t total, i
    cdef IntColumn icol
    cdef DoubleColumn fcol
    cdef DateColumn dcol
    cdef DateTimeColumn tcol
    cdef TimeColumn qcol
    cdef int32_t* dptr
    cdef int64_t* tptr
    with open(filename, 'rb') as f:
        data = _SafeUnpickler(f, allow_raw_pickle).load()
    df = FastDataFrame()
    df.df_length = data['df_length']
    for col_name, entry in data['columns'].items():
        if len(entry) == 4:
            typ, values, rows, cols = entry
        else:
            typ, values = entry
        is_2d = cols > 1
        total = len(values)
        if typ == 'int':
            if is_2d:
                col = IntColumn.from_2d(<object>values, rows, cols)
            else:
                col = IntColumn()
                for v in values:
                    col.append(v)
        elif typ == 'float':
            if is_2d:
                col = DoubleColumn.from_2d(<object>values, rows, cols)
            else:
                col = DoubleColumn()
                for v in values:
                    col.append(v)
        elif typ == 'char256':
            col = AsciiColumn()
            for v in values:
                col.append(v)
        elif typ == 'ucs2':
            col = UCS2Column()
            for v in values:
                col.append(v)
        elif typ == 'date':
            if is_2d:
                if total != rows * cols:
                    raise ValueError(f"DateColumn 2D size mismatch: expected {rows * cols}, got {total}")
                dcol = DateColumn(max(total, 1), cols=cols)
                dcol.length = total
                dcol.num_cols = <int>cols
                dcol.is_2d = True
                dcol.owns_data = True
                dptr = <int32_t*>dcol.data_c
                val_list = <list>values if type(values) is list else list(values)
                for i in range(total):
                    dptr[i] = <int32_t>val_list[i]
                col = dcol
            else:
                col = DateColumn()
                for v in values:
                    col.append(v)
        elif typ == 'bool':
            col = BoolColumn()
            for v in values:
                col.append(v)
        elif typ == 'category':
            col = CategoryColumn()
            for v in values:
                col.append(v)
        elif typ == 'datetime':
            if is_2d:
                if total != rows * cols:
                    raise ValueError(f"DateTimeColumn 2D size mismatch: expected {rows * cols}, got {total}")
                tcol = DateTimeColumn(max(total, 1), cols=cols)
                tcol.length = total
                tcol.num_cols = <int>cols
                tcol.is_2d = True
                tcol.owns_data = True
                tptr = <int64_t*>tcol.data_c
                val_list = <list>values if type(values) is list else list(values)
                for i in range(total):
                    tptr[i] = <int64_t>val_list[i]
                col = tcol
            else:
                col = DateTimeColumn()
                for v in values:
                    col.append(v)
        elif typ == 'time':
            if is_2d:
                if total != rows * cols:
                    raise ValueError(f"TimeColumn 2D size mismatch: expected {rows * cols}, got {total}")
                qcol = TimeColumn(max(total, 1), cols=cols)
                qcol.length = total
                qcol.num_cols = <int>cols
                qcol.is_2d = True
                qcol.owns_data = True
                dptr = <int32_t*>qcol.data_c
                val_list = <list>values if type(values) is list else list(values)
                for i in range(total):
                    dptr[i] = <int32_t>val_list[i]
                col = qcol
            else:
                col = TimeColumn()
                for v in values:
                    col.append(v)
        df.columns[col_name] = col
    return df


def read_csv(filename, dict schema, sep=','):
    """读取 CSV 文件，返回 FastDataFrame。必须提供 schema 指定列类型"""
    if not schema:
        raise ValueError("schema is required and must not be empty")
    cdef dict parsed = _parse_schema(schema)
    with open(filename, 'r', encoding='utf-8') as f:
        reader = _csv.reader(f, delimiter=sep)
        headers = next(reader)
        rows = list(reader)

    cdef Py_ssize_t n_rows = len(rows)
    cdef Py_ssize_t n_cols = len(headers)
    cdef Py_ssize_t i, j
    df = FastDataFrame()

    if n_rows == 0:
        for j in range(n_cols):
            name = headers[j]
            dtype = parsed.get(name, str)
            if dtype == TYPE_INT:
                df.columns[name] = IntColumn()
            elif dtype == TYPE_DOUBLE:
                df.columns[name] = DoubleColumn()
            elif dtype == TYPE_CHAR256:
                df.columns[name] = AsciiColumn()
            elif dtype == TYPE_DATE:
                df.columns[name] = DateColumn()
            elif dtype == TYPE_BOOL:
                df.columns[name] = BoolColumn()
            elif dtype == TYPE_CATEGORY:
                df.columns[name] = CategoryColumn()
            elif dtype == TYPE_DATETIME:
                df.columns[name] = DateTimeColumn()
            elif dtype == TYPE_TIME:
                df.columns[name] = TimeColumn()
            else:
                df.columns[name] = UCS2Column()
        return df

    for j, name in enumerate(headers):
        py_type = parsed.get(name, str)
        if py_type == TYPE_INT:
            col = IntColumn(n_rows)
            for i in range(n_rows):
                val = rows[i][j].strip()
                col.append(int(val) if val else 0)
        elif py_type == TYPE_DOUBLE:
            col = DoubleColumn(n_rows)
            for i in range(n_rows):
                val = rows[i][j].strip()
                col.append(float(val) if val else NAN)
        elif py_type == TYPE_CHAR256:
            col = AsciiColumn(n_rows)
            for i in range(n_rows):
                col.append(rows[i][j])
        elif py_type == TYPE_DATE:
            col = DateColumn(n_rows)
            for i in range(n_rows):
                col.append(rows[i][j].strip())
        elif py_type == TYPE_BOOL:
            col = BoolColumn(n_rows)
            for i in range(n_rows):
                val = rows[i][j].strip().lower()
                col.append(val in ('true', '1', 'yes'))
        elif py_type == TYPE_CATEGORY:
            col = CategoryColumn(n_rows)
            for i in range(n_rows):
                col.append(rows[i][j])
        elif py_type == TYPE_DATETIME:
            col = DateTimeColumn(n_rows)
            for i in range(n_rows):
                col.append(rows[i][j].strip())
        elif py_type == TYPE_TIME:
            col = TimeColumn(n_rows)
            for i in range(n_rows):
                col.append(rows[i][j].strip())
        else:
            col = UCS2Column(n_rows)
            for i in range(n_rows):
                col.append(rows[i][j])
        df.columns[name] = col
    df.df_length = n_rows
    return df


def read_tsv(filename, dict schema):
    """读取 TSV 文件（tab 分隔），返回 FastDataFrame"""
    return read_csv(filename, schema=schema, sep='\t')


def read_json(filename, dict schema):
    """读取 JSON 文件（records 格式），返回 FastDataFrame"""
    import json as _json
    if not schema:
        raise ValueError("schema is required and must not be empty")
    cdef dict parsed = _parse_schema(schema)
    with open(filename, 'r', encoding='utf-8') as f:
        records = _json.load(f)
    if not isinstance(records, list):
        raise ValueError("JSON must be a list of records (records orient)")
    cdef Py_ssize_t n_rows = len(records)
    cdef list headers = list(parsed.keys())
    cdef Py_ssize_t n_cols = len(headers)
    df = FastDataFrame()
    cdef Py_ssize_t i, j

    for j, name in enumerate(headers):
        py_type = parsed.get(name, str)
        if py_type == TYPE_INT:
            col = IntColumn(max(n_rows, 1))
            col.length = n_rows
            for i in range(n_rows):
                col.data_c[i] = <int>records[i].get(name, 0)
            df.columns[name] = col
        elif py_type == TYPE_DOUBLE:
            col_d = DoubleColumn(max(n_rows, 1))
            col_d.length = n_rows
            for i in range(n_rows):
                val = records[i].get(name)
                col_d.data_c[i] = <double>val if val is not None else NAN
            df.columns[name] = col_d
        elif py_type == TYPE_CHAR256:
            col_a = AsciiColumn(max(n_rows, 1))
            for i in range(n_rows):
                col_a.append(str(records[i].get(name, '')))
            df.columns[name] = col_a
        elif py_type == TYPE_DATE:
            col_dt = DateColumn(max(n_rows, 1))
            col_dt.length = n_rows
            for i in range(n_rows):
                col_dt.data_c[i] = <int>records[i].get(name, 0)
            df.columns[name] = col_dt
        elif py_type == TYPE_BOOL:
            col_b = BoolColumn(max(n_rows, 1))
            col_b.length = n_rows
            for i in range(n_rows):
                col_b.data_c[i] = 1 if records[i].get(name, False) else 0
            df.columns[name] = col_b
        else:
            col_u = UCS2Column(max(n_rows, 1))
            for i in range(n_rows):
                col_u.append(str(records[i].get(name, '')))
            df.columns[name] = col_u

    df.df_length = n_rows
    df.index = RangeIndex(0, n_rows)
    return df


cdef object _default_val_standalone(BaseColumn col):
    if isinstance(col, IntColumn): return 0
    if isinstance(col, DoubleColumn): return float('nan')
    if isinstance(col, DateColumn): return 0
    return ''


cdef BaseColumn _empty_col_standalone(BaseColumn col):
    if isinstance(col, IntColumn): return IntColumn()
    if isinstance(col, DoubleColumn): return DoubleColumn()
    if isinstance(col, AsciiColumn): return AsciiColumn()
    if isinstance(col, DateColumn): return DateColumn()
    return UCS2Column()


def concat_outer(list dfs):
    """纵向拼接多个 DataFrame（外连接：取所有列的并集），重置索引"""
    if not dfs: return FastDataFrame()
    all_cols = set()
    for df in dfs:
        all_cols.update(df.columns.keys())
    all_cols = list(all_cols)

    result = FastDataFrame()
    col_types = {}
    for name in all_cols:
        for df in dfs:
            if name in df.columns:
                col_types[name] = df.columns[name]
                break

    for name in all_cols:
        result.columns[name] = _empty_col_standalone(col_types[name])

    for df in dfs:
        for i in range(df.df_length):
            for name in all_cols:
                col = result.columns[name]
                if name in df.columns:
                    col.append(df.columns[name][i])
                else:
                    col.append(_default_val_standalone(col_types[name]))
        result.df_length += df.df_length

    result.reset_index()
    return result


def concat_inner(list dfs):
    """纵向拼接多个 DataFrame（内连接：仅取所有列的交集），重置索引"""
    if not dfs: return FastDataFrame()
    common = set(dfs[0].columns.keys())
    for df in dfs[1:]:
        common &= set(df.columns.keys())
    common = list(common)

    result = FastDataFrame()
    for name in common:
        result.columns[name] = _empty_col_standalone(dfs[0].columns[name])

    for df in dfs:
        for i in range(df.df_length):
            for name in common:
                result.columns[name].append(df.columns[name][i])
        result.df_length += df.df_length

    result.reset_index()
    return result


def concat(list dfs, bint ignore_index=False):
    cdef FastDataFrame result = concat_outer(dfs)
    if ignore_index:
        result.index = RangeIndex(0, result.df_length)
    return result


def to_numeric(column, errors='coerce'):
    """Convert column to numeric values. Like pd.to_numeric.
    2D 输入保持同维度输出（is_2d/rows/cols 不变，逐单元转换）。"""
    cdef Py_ssize_t n = column.length
    cdef bint is2d
    cdef Py_ssize_t c = 1, r
    is2d, c, r = _col_shape_probe(column)

    cdef bint is_2d_c = is2d and c > 1
    cdef Py_ssize_t alloc_size = (r * c) if is_2d_c else r
    cdef DoubleColumn res = DoubleColumn(max(alloc_size, 1))
    res.length = alloc_size
    if is_2d_c:
        res.num_cols = <int>c
        res.is_2d = True
        res.owns_data = True

    cdef Py_ssize_t i, j, idx = 0
    cdef bint raise_err = (errors == 'raise')

    if is_2d_c:
        # 2D 扁平缓冲为列主序（先 col0 全部行，再 col1…）：cell = column[j * r + i]
        for j in range(c):
            for i in range(r):
                idx = j * r + i
                s = str(column[idx])
                try:
                    res.data_c[idx] = float(s)
                except (ValueError, TypeError):
                    if raise_err:
                        raise ValueError(f"Could not convert '{s}' to numeric")
                    res.data_c[idx] = NAN
                idx += 1
    else:
        for i in range(r):
            s = str(column[i])
            try:
                res.data_c[i] = float(s)
            except (ValueError, TypeError):
                if raise_err:
                    raise ValueError(f"Could not convert '{s}' to numeric")
                res.data_c[i] = NAN
    return res


def to_datetime(column, str fmt='', errors='coerce'):
    """Convert column to DateTimeColumn. Like pd.to_datetime.
    2D 输入保持同维度输出。"""
    cdef bint is2d
    cdef Py_ssize_t c = 1, r
    is2d, c, r = _col_shape_probe(column)

    cdef bint is_2d_c = is2d and c > 1
    cdef DateTimeColumn res = DateTimeColumn(max(r * c if is_2d_c else r, 1))
    cdef bint raise_err = (errors == 'raise')
    cdef Py_ssize_t i, j

    if is_2d_c:
        # 2D 扁平缓冲为列主序：追加顺序须为 col0 全部行 → col1 全部行…
        for j in range(c):
            for i in range(r):
                s = str(column[j * r + i])
                try:
                    res.append(s)
                except (ValueError, TypeError):
                    if raise_err:
                        raise ValueError(f"Could not convert '{s}' to datetime")
                    res.append(0)
        res.num_cols = <int>c
        res.is_2d = True
        res.owns_data = True
    else:
        for i in range(r):
            s = str(column[i])
            try:
                res.append(s)
            except (ValueError, TypeError):
                if raise_err:
                    raise ValueError(f"Could not convert '{s}' to datetime")
                res.append(0)
    return res


def to_date(column, str fmt='', errors='coerce'):
    """Convert column to DateColumn (YYYYMMDD).
    2D 输入保持同维度输出。"""
    cdef bint is2d
    cdef Py_ssize_t c = 1, r
    is2d, c, r = _col_shape_probe(column)

    cdef bint is_2d_c = is2d and c > 1
    cdef DateColumn res = DateColumn(max(r * c if is_2d_c else r, 1))
    cdef bint raise_err = (errors == 'raise')
    cdef Py_ssize_t i, j

    if is_2d_c:
        # 2D 扁平缓冲为列主序：追加顺序须为 col0 全部行 → col1 全部行…
        for j in range(c):
            for i in range(r):
                s = str(column[j * r + i])
                try:
                    res.append(s)
                except (ValueError, TypeError):
                    if raise_err:
                        raise ValueError(f"Could not convert '{s}' to date")
                    res.append(0)
        res.num_cols = <int>c
        res.is_2d = True
        res.owns_data = True
    else:
        for i in range(r):
            s = str(column[i])
            try:
                res.append(s)
            except (ValueError, TypeError):
                if raise_err:
                    raise ValueError(f"Could not convert '{s}' to date")
                res.append(0)
    return res


def to_time(column, str fmt='', errors='coerce'):
    """Convert column to TimeColumn (seconds since midnight).
    2D 输入保持同维度输出。"""
    cdef bint is2d
    cdef Py_ssize_t c = 1, r
    is2d, c, r = _col_shape_probe(column)

    cdef bint is_2d_c = is2d and c > 1
    cdef TimeColumn res = TimeColumn(max(r * c if is_2d_c else r, 1))
    cdef bint raise_err = (errors == 'raise')
    cdef Py_ssize_t i, j

    if is_2d_c:
        # 2D 扁平缓冲为列主序：追加顺序须为 col0 全部行 → col1 全部行…
        for j in range(c):
            for i in range(r):
                s = str(column[j * r + i])
                try:
                    res.append(s)
                except (ValueError, TypeError):
                    if raise_err:
                        raise ValueError(f"Could not convert '{s}' to time")
                    res.append(0)
        res.num_cols = <int>c
        res.is_2d = True
        res.owns_data = True
    else:
        for i in range(r):
            s = str(column[i])
            try:
                res.append(s)
            except (ValueError, TypeError):
                if raise_err:
                    raise ValueError(f"Could not convert '{s}' to time")
                res.append(0)
    return res


def to_timedelta(values, str unit='D'):
    """Create a column of timedelta-like values (stored as int days). Like pd.to_timedelta."""
    n = len(values)
    res = IntColumn(n)
    res.length = n
    if unit == 'D':
        factor = 1.0
    elif unit == 'h':
        factor = 1.0 / 24.0
    elif unit == 'm':
        factor = 1.0 / 1440.0
    elif unit == 's':
        factor = 1.0 / 86400.0
    else:
        raise ValueError(f"Unsupported unit: '{unit}'")
    for i in range(n):
        res.data_c[i] = <int>(<double>values[i] * factor)
    return res


@cython.boundscheck(False)
@cython.wraparound(False)
def where(IntColumn cond, object true_val, object false_val):
    """numpy.where 等效替代。根据 cond 掩码列选择填充 true_val 或 false_val。"""
    cdef Py_ssize_t n = cond.length
    cdef Py_ssize_t i
    cdef int* cond_ptr = cond.data_c
    cdef double v_t, v_f
    cdef double* res_ptr
    cdef DoubleColumn res_dbl
    cdef IntColumn inv
    cdef int* inv_ptr

    if isinstance(true_val, str) and isinstance(false_val, str):
        res_str = UCS2Column(max(n, 1))
        res_str.length = n
        for i in range(n):
            if cond_ptr[i]:
                res_str._set_item(i, true_val)
            else:
                res_str._set_item(i, false_val)
        return res_str

    elif isinstance(true_val, (int, float)) and isinstance(false_val, (int, float)):
        res_dbl = DoubleColumn(max(n, 1))
        res_dbl.length = n
        v_t = <double>true_val
        v_f = <double>false_val
        res_ptr = res_dbl.data_c
        with nogil:
            for i in range(n):
                res_ptr[i] = v_t if cond_ptr[i] else v_f
        return res_dbl

    elif hasattr(true_val, 'where_mask'):
        return true_val.where_mask(cond, false_val)

    elif hasattr(false_val, 'where_mask'):
        inv = IntColumn(max(n, 1))
        inv.length = n
        inv_ptr = inv.data_c
        with nogil:
            for i in range(n):
                inv_ptr[i] = 0 if cond_ptr[i] else 1
        return false_val.where_mask(inv, true_val)

    raise TypeError(f"Unsupported types for where: {type(true_val)}, {type(false_val)}")
