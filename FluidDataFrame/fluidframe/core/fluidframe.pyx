# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport cython
from libc.math cimport NAN

from fluidframe.core._base cimport BaseColumn, DataType, TYPE_INT, TYPE_DOUBLE, TYPE_CHAR256, TYPE_UCS2, TYPE_TIMESTAMP, TYPE_BOOL, TYPE_CATEGORY
from fluidframe.core.fastdataframe cimport FastDataFrame
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.doublecolumn cimport DoubleColumn
from fluidframe.core.timestampcolumn cimport TimestampColumn

from fluidframe._schema_types import Char256, Timestamp, Bool, Category, _parse_schema

from fluidframe.core.intcolumn import IntColumn
from fluidframe.core.doublecolumn import DoubleColumn
from fluidframe.core.char256column import Char256Column_SoA
from fluidframe.core.ucs2column import UCS2Column_SoA
from fluidframe.core.timestampcolumn import TimestampColumn
from fluidframe.core.boolcolumn import BoolColumn
from fluidframe.core.categorycolumn import CategoryColumn
from fluidframe.core.index import RangeIndex


def from_pickle(filename, bint allow_raw_pickle=False):
    """从 pickle 文件恢复 FastDataFrame。

    Parameters
    ----------
    filename : str
        Path to the pickle file.
    allow_raw_pickle : bool, default False
        If True, allow unrestricted pickle loading (WARNING: can execute
        arbitrary code). If False (default), only allows safe types.
    """
    import pickle

    class _SafeUnpickler(pickle.Unpickler):
        _ALLOWED = {
            'builtins', 'builtins.dict', 'builtins.list', 'builtins.tuple',
            'builtins.int', 'builtins.float', 'builtins.str', 'builtins.bytes',
            'builtins.NoneType', 'builtins.bool', 'builtins.set',
        }

        def find_class(self, module, name):
            if allow_raw_pickle:
                return super().find_class(module, name)
            full = f"{module}.{name}"
            # Allow FluidDataFrame's own internal dict/list/tuple usage
            if module == 'builtins' and name in ('dict', 'list', 'tuple', 'int', 'float', 'str', 'bytes', 'NoneType', 'bool', 'set'):
                return super().find_class(module, name)
            raise pickle.UnpicklingError(
                f"Deserialization of '{full}' is not allowed. "
                f"Use allow_raw_pickle=True to bypass (unsafe)."
            )

    with open(filename, 'rb') as f:
        data = _SafeUnpickler(f).load()
    df = FastDataFrame()
    df.df_length = data['df_length']
    for col_name, (typ, values) in data['columns'].items():
        if typ == 'int':
            col = IntColumn()
            for v in values:
                col.append(v)
        elif typ == 'float':
            col = DoubleColumn()
            for v in values:
                col.append(v)
        elif typ == 'char256':
            col = Char256Column_SoA()
            for v in values:
                col.append(v)
        elif typ == 'ucs2':
            col = UCS2Column_SoA()
            for v in values:
                col.append(v)
        elif typ == 'timestamp':
            col = TimestampColumn()
            for v in values:
                col.append(v)
        df.columns[col_name] = col
    return df


def read_csv(filename, dict schema, sep=','):
    """读取 CSV 文件，返回 FastDataFrame。必须提供 schema 指定列类型"""
    if not schema:
        raise ValueError("schema is required and must not be empty")
    cdef dict parsed = _parse_schema(schema)
    import csv
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=sep)
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
                df.columns[name] = Char256Column_SoA()
            elif dtype == TYPE_TIMESTAMP:
                df.columns[name] = TimestampColumn()
            else:
                df.columns[name] = UCS2Column_SoA()
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
            col = Char256Column_SoA(n_rows)
            for i in range(n_rows):
                col.append(rows[i][j])
        elif py_type == TYPE_TIMESTAMP:
            col = TimestampColumn(n_rows)
            for i in range(n_rows):
                col.append(rows[i][j].strip())
        else:
            col = UCS2Column_SoA(n_rows)
            for i in range(n_rows):
                col.append(rows[i][j])
        df.columns[name] = col
    df.df_length = n_rows
    return df


def read_tsv(filename, dict schema):
    """读取 TSV 文件（tab 分隔），返回 FastDataFrame"""
    return read_csv(filename, schema=schema, sep='\t')


cdef object _default_val(BaseColumn col):
    if isinstance(col, IntColumn): return 0
    if isinstance(col, DoubleColumn): return float('nan')
    if isinstance(col, TimestampColumn): return 0
    return ''


cdef BaseColumn _empty_col(BaseColumn col):
    if isinstance(col, IntColumn): return IntColumn()
    if isinstance(col, DoubleColumn): return DoubleColumn()
    if isinstance(col, Char256Column_SoA): return Char256Column_SoA()
    if isinstance(col, TimestampColumn): return TimestampColumn()
    return UCS2Column_SoA()


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
        result.columns[name] = _empty_col(col_types[name])

    for df in dfs:
        for i in range(df.df_length):
            for name in all_cols:
                col = result.columns[name]
                if name in df.columns:
                    col.append(df.columns[name][i])
                else:
                    col.append(_default_val(col_types[name]))
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
        result.columns[name] = _empty_col(dfs[0].columns[name])

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


cdef list _build_pairs(df1, df2, str left_on, str right_on, str how):
    cdef dict right_lookup = {}
    cdef BaseColumn right_key = df2.columns[right_on]
    cdef Py_ssize_t i, j
    cdef object val

    for i in range(df2.df_length):
        val = right_key[i]
        if val not in right_lookup:
            right_lookup[val] = []
        right_lookup[val].append(i)

    cdef BaseColumn left_key = df1.columns[left_on]
    cdef list pairs = []
    cdef set right_matched = set()
    cdef list matches

    for i in range(df1.df_length):
        val = left_key[i]
        matches = right_lookup.get(val)
        if matches:
            for j in matches:
                pairs.append((i, j))
                right_matched.add(j)
        elif how in ('left', 'outer'):
            pairs.append((i, -1))

    if how in ('right', 'outer'):
        for j in range(df2.df_length):
            if j not in right_matched:
                pairs.append((-1, j))

    return pairs


def merge_inner(dfs, str left_on, str right_on):
    """内连接：仅保留两表共同拥有匹配键的行"""
    df1, df2 = dfs
    cdef list pairs = _build_pairs(df1, df2, left_on, right_on, 'inner')
    return _build_merge_result(df1, df2, left_on, right_on, pairs)


def merge_left(dfs, str left_on, str right_on):
    """左连接：保留左表所有行，右表无匹配则填充默认值"""
    df1, df2 = dfs
    cdef list pairs = _build_pairs(df1, df2, left_on, right_on, 'left')
    return _build_merge_result(df1, df2, left_on, right_on, pairs)


def merge_right(dfs, str left_on, str right_on):
    """右连接：保留右表所有行，左表无匹配则填充默认值"""
    df1, df2 = dfs
    cdef list pairs = _build_pairs(df1, df2, left_on, right_on, 'right')
    return _build_merge_result(df1, df2, left_on, right_on, pairs)


def merge_outer(dfs, str left_on, str right_on):
    """外连接：保留两表所有行，无匹配则填充默认值"""
    df1, df2 = dfs
    cdef list pairs = _build_pairs(df1, df2, left_on, right_on, 'outer')
    return _build_merge_result(df1, df2, left_on, right_on, pairs)


cdef _build_merge_result(df1, df2, str left_on, str right_on, list pairs):
    cdef FastDataFrame result = FastDataFrame()
    cdef Py_ssize_t n = len(pairs)
    if n == 0:
        result.df_length = 0
        for name in df1.columns:
            result.columns[name] = _empty_col(df1.columns[name])
        for name in df2.columns:
            if name == right_on and name == left_on:
                continue
            if name not in result.columns:
                result.columns[name] = _empty_col(df2.columns[name])
        return result

    cdef tuple pair
    cdef Py_ssize_t i, j
    cdef BaseColumn src_col, new_col

    for name in df1.columns:
        src_col = df1.columns[name]
        new_col = _empty_col(src_col)
        for pair in pairs:
            i = pair[0]
            if i >= 0:
                new_col.append(src_col[i])
            else:
                new_col.append(_default_val(src_col))
        result.columns[name] = new_col

    for name in df2.columns:
        if name == right_on and name == left_on:
            continue
        col_name = name
        if col_name in result.columns:
            col_name = name + '_right'
        src_col = df2.columns[name]
        new_col = _empty_col(src_col)
        for pair in pairs:
            j = pair[1]
            if j >= 0:
                new_col.append(src_col[j])
            else:
                new_col.append(_default_val(src_col))
        result.columns[col_name] = new_col

    result.df_length = n
    return result


def to_numeric(column, errors='coerce'):
    """Convert column to numeric values. Like pd.to_numeric.

    Parameters
    ----------
    column : column-like
        Input column (Char256, UCS2, or numeric).
    errors : str, default 'coerce'
        'raise' will throw on invalid values, 'coerce' will set them to 0.

    Returns
    -------
    DoubleColumn with numeric values.
    """
    n = column.length
    res = DoubleColumn(n)
    res.length = n
    for i in range(n):
        s = str(column[i])
        try:
            val = float(s)
            if val == int(val) and '.' not in s:
                res.data_c[i] = <double>int(s)
            else:
                res.data_c[i] = val
        except (ValueError, TypeError):
            if errors == 'raise':
                raise ValueError(f"Could not convert '{s}' to numeric")
            res.data_c[i] = NAN
    return res


def to_datetime(column, str fmt='', errors='coerce'):
    """Convert column to TimestampColumn. Like pd.to_datetime."""
    n = column.length
    res = TimestampColumn(n)
    for i in range(n):
        s = str(column[i])
        try:
            res.append(s)
        except (ValueError, TypeError):
            if errors == 'raise':
                raise ValueError(f"Could not convert '{s}' to datetime")
            res.append(0)
    return res


def to_timedelta(values, str unit='D'):
    """Create a column of timedelta-like values (stored as int days). Like pd.to_timedelta.

    Parameters
    ----------
    values : list or column-like
        Input values (integers).
    unit : str, default 'D'
        Unit of the values: 'D' (days), 'h' (hours), 'm' (minutes), 's' (seconds).

    Returns
    -------
    IntColumn with timedelta values converted to days.
    """
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
