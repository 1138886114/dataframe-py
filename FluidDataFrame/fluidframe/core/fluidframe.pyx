# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport cython
from libc.math cimport NAN

from fluidframe.core._base cimport BaseColumn, DataType, TYPE_INT, TYPE_DOUBLE, TYPE_CHAR256, TYPE_UCS2, TYPE_DATE, TYPE_BOOL, TYPE_CATEGORY, TYPE_DATETIME, TYPE_TIME
from fluidframe.core.fastdataframe cimport FastDataFrame
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.doublecolumn cimport DoubleColumn
from fluidframe.core.datecolumn cimport DateColumn
from fluidframe.core.datetimecolumn cimport DateTimeColumn
from fluidframe.core.timecolumn cimport TimeColumn

import pickle

from fluidframe._schema_types import Char256, Date, Bool, Category, DateTime, Time, _parse_schema

from fluidframe.core.intcolumn import IntColumn
from fluidframe.core.doublecolumn import DoubleColumn
from fluidframe.core.char256column import AsciiColumn
from fluidframe.core.ucs2column import UCS2Column
from fluidframe.core.boolcolumn import BoolColumn
from fluidframe.core.categorycolumn import CategoryColumn
from fluidframe.core.datetimecolumn import DateTimeColumn
from fluidframe.core.timecolumn import TimeColumn
from fluidframe.core.index import RangeIndex


class _SafeUnpickler(pickle.Unpickler):
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
        raise pickle.UnpicklingError(
            f"Deserialization of '{full}' is not allowed. "
            f"Use allow_raw_pickle=True to bypass (unsafe)."
        )


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
    with open(filename, 'rb') as f:
        data = _SafeUnpickler(f, allow_raw_pickle).load()
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
            col = AsciiColumn()
            for v in values:
                col.append(v)
        elif typ == 'ucs2':
            col = UCS2Column()
            for v in values:
                col.append(v)
        elif typ == 'date':
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
            col = DateTimeColumn()
            for v in values:
                col.append(v)
        elif typ == 'time':
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


cdef object _default_val(BaseColumn col):
    if isinstance(col, IntColumn): return 0
    if isinstance(col, DoubleColumn): return float('nan')
    if isinstance(col, DateColumn): return 0
    return ''


cdef BaseColumn _empty_col(BaseColumn col):
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
    """Convert column to DateTimeColumn. Like pd.to_datetime."""
    cdef Py_ssize_t n = column.length
    cdef DateTimeColumn res = DateTimeColumn(n)
    for i in range(n):
        s = str(column[i])
        try:
            res.append(s)
        except (ValueError, TypeError):
            if errors == 'raise':
                raise ValueError(f"Could not convert '{s}' to datetime")
            res.append(0)
    return res


def to_date(column, str fmt='', errors='coerce'):
    """Convert column to DateColumn (YYYYMMDD)."""
    cdef Py_ssize_t n = column.length
    cdef DateColumn res = DateColumn(n)
    for i in range(n):
        s = str(column[i])
        try:
            res.append(s)
        except (ValueError, TypeError):
            if errors == 'raise':
                raise ValueError(f"Could not convert '{s}' to date")
            res.append(0)
    return res


def to_time(column, str fmt='', errors='coerce'):
    """Convert column to TimeColumn (seconds since midnight)."""
    cdef Py_ssize_t n = column.length
    cdef TimeColumn res = TimeColumn(n)
    for i in range(n):
        s = str(column[i])
        try:
            res.append(s)
        except (ValueError, TypeError):
            if errors == 'raise':
                raise ValueError(f"Could not convert '{s}' to time")
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


@cython.boundscheck(False)
@cython.wraparound(False)
def where(IntColumn cond, object true_val, object false_val):
    """
    极致性能的 numpy.where 等效替代。
    根据 cond 掩码列，选择填充 true_val 或 false_val。
    支持 标量/标量、标量/列、列/标量 等复杂广播。
    """
    cdef Py_ssize_t n = cond.length
    cdef Py_ssize_t i
    cdef int* cond_ptr = cond.data_c
    cdef double v_t, v_f
    cdef double* res_ptr
    cdef DoubleColumn res_dbl
    cdef IntColumn inv
    cdef int* inv_ptr

    # --- scalar / scalar ---
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

    # --- column / scalar: true_val.where_mask(cond, false_val) ---
    elif hasattr(true_val, 'where_mask'):
        return true_val.where_mask(cond, false_val)

    # --- scalar / column: false_val.where_mask(~cond, true_val) ---
    elif hasattr(false_val, 'where_mask'):
        inv = IntColumn(max(n, 1))
        inv.length = n
        inv_ptr = inv.data_c
        with nogil:
            for i in range(n):
                inv_ptr[i] = 0 if cond_ptr[i] else 1
        return false_val.where_mask(inv, true_val)

    raise TypeError(f"Unsupported types for where: {type(true_val)}, {type(false_val)}")