# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

cimport cython
from libc.string cimport memcpy, strncpy
from libc.stdint cimport int32_t, int64_t
from fluidframe.core._base cimport aligned_malloc, os_aligned_free, BaseColumn, TYPE_INT, TYPE_DOUBLE, TYPE_CHAR256, TYPE_UCS2, TYPE_DATE, TYPE_DATETIME, TYPE_TIME, TYPE_BOOL, TYPE_CATEGORY
from fluidframe.core.fastdataframe cimport FastDataFrame
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.doublecolumn cimport DoubleColumn
from fluidframe.core.asciicolumn cimport AsciiColumn
from fluidframe.core.ucs2column cimport UCS2Column
from fluidframe.core.datecolumn cimport DateColumn, _ymd_to_days
from fluidframe.core.datetimecolumn cimport DateTimeColumn
from fluidframe.core.timecolumn cimport TimeColumn
from fluidframe.core.boolcolumn cimport BoolColumn
from fluidframe.core.categorycolumn cimport CategoryColumn
from fluidframe.core.index cimport IndexBase


cdef class _ILocIndexer:
    def __cinit__(self, FastDataFrame frame):
        self.frame = frame

    def __getitem__(self, key):
        cdef Py_ssize_t n_orig = self.frame.df_length
        if n_orig == 0:
            raise IndexError("Cannot index empty DataFrame")

        cdef Py_ssize_t* indices_c = NULL
        cdef Py_ssize_t n_new = 0
        cdef Py_ssize_t i, idx
        cdef FastDataFrame result

        # 解析用户输入的类型
        if isinstance(key, int):
            # 获取单行 (df.iloc[5])
            idx = <Py_ssize_t>key
            if idx < 0: idx += n_orig
            if idx < 0 or idx >= n_orig: raise IndexError("Index out of bounds")

            indices_c = <Py_ssize_t*>aligned_malloc(sizeof(Py_ssize_t))
            if not indices_c: raise MemoryError()
            indices_c[0] = idx
            n_new = 1

        elif isinstance(key, slice):
            # 获取切片 (df.iloc[2:10])
            start, stop, step = key.indices(n_orig)
            n_new = len(range(start, stop, step))
            if n_new <= 0:
                indices_c = <Py_ssize_t*>aligned_malloc(sizeof(Py_ssize_t))
                try:
                    return self.frame._take(indices_c, 0)
                finally:
                    os_aligned_free(indices_c)

            indices_c = <Py_ssize_t*>aligned_malloc(n_new * sizeof(Py_ssize_t))
            if not indices_c: raise MemoryError()

            i = 0
            for idx in range(start, stop, step):
                indices_c[i] = idx
                i += 1

        elif isinstance(key, list):
            # 获取列表 (df.iloc[[0, 2, 4]])
            n_new = len(key)
            if n_new == 0:
                indices_c = <Py_ssize_t*>aligned_malloc(sizeof(Py_ssize_t))
                try:
                    return self.frame._take(indices_c, 0)
                finally:
                    os_aligned_free(indices_c)

            indices_c = <Py_ssize_t*>aligned_malloc(n_new * sizeof(Py_ssize_t))
            if not indices_c: raise MemoryError()

            try:
                for i in range(n_new):
                    idx = <Py_ssize_t>key[i]
                    if idx < 0: idx += n_orig
                    if idx < 0 or idx >= n_orig: raise IndexError("Index out of bounds")
                    indices_c[i] = idx
            except Exception:
                os_aligned_free(indices_c)
                raise
        else:
            raise TypeError("iloc requires integer, slice, or list of integers")

        # 安全地调用底层 C 提取方法
        try:
            result = self.frame._take(indices_c, n_new)
            return result
        finally:
            os_aligned_free(indices_c)


cdef inline void _probe_shape(BaseColumn col, bint* is2d, Py_ssize_t* c, Py_ssize_t* r):
    cdef int dtype = col.dtype
    if dtype == TYPE_INT:
        is2d[0] = (<IntColumn>col).get_is_2d_c()
        c[0] = (<IntColumn>col).get_cols_c()
        r[0] = (<IntColumn>col).get_rows_c()
    elif dtype == TYPE_DOUBLE:
        is2d[0] = (<DoubleColumn>col).get_is_2d_c()
        c[0] = (<DoubleColumn>col).get_cols_c()
        r[0] = (<DoubleColumn>col).get_rows_c()
    elif dtype == TYPE_DATE:
        is2d[0] = (<DateColumn>col).get_is_2d_c()
        c[0] = (<DateColumn>col).get_cols_c()
        r[0] = (<DateColumn>col).get_rows_c()
    elif dtype == TYPE_DATETIME:
        is2d[0] = (<DateTimeColumn>col).get_is_2d_c()
        c[0] = (<DateTimeColumn>col).get_cols_c()
        r[0] = (<DateTimeColumn>col).get_rows_c()
    elif dtype == TYPE_TIME:
        is2d[0] = (<TimeColumn>col).get_is_2d_c()
        c[0] = (<TimeColumn>col).get_cols_c()
        r[0] = (<TimeColumn>col).get_rows_c()
    elif dtype == TYPE_CHAR256 or dtype == TYPE_UCS2 or dtype == TYPE_BOOL or dtype == TYPE_CATEGORY:
        is2d[0] = False
        c[0] = 1
        r[0] = col.length
    else:
        raise TypeError(f"unsupported column dtype {dtype}")


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


cdef class _LocIndexer:

    def __cinit__(self, FastDataFrame frame):
        self.frame = frame

    def __setitem__(self, key, object values):
        cdef Py_ssize_t n_orig = self.frame.df_length
        if n_orig == 0:
            raise IndexError("Cannot setitem on empty DataFrame")

        cdef Py_ssize_t idx, i
        cdef list col_names = list(self.frame.columns.keys())
        cdef Py_ssize_t n_cols = len(col_names)
        cdef str col_name
        cdef BaseColumn col

        if isinstance(key, int):
            idx = <Py_ssize_t>key
            if idx < 0: idx += n_orig
            if idx < 0 or idx >= n_orig:
                raise IndexError("Index out of bounds")

            if isinstance(values, list):
                if len(values) != n_cols:
                    raise ValueError(f"Length of values {len(values)} != number of columns {n_cols}")
                for i in range(n_cols):
                    col_name = col_names[i]
                    col = self.frame.columns[col_name]
                    _fast_set_cell(col, col.dtype, idx, values[i])
            elif isinstance(values, dict):
                for col_name, val in values.items():
                    if col_name not in self.frame.columns:
                        raise KeyError(f"Column '{col_name}' not found")
                    col = self.frame.columns[col_name]
                    _fast_set_cell(col, col.dtype, idx, val)
            else:
                raise TypeError(f"Cannot set row from {type(values)}")

        else:
            raise TypeError(f"Unsupported loc key type for setitem: {type(key).__name__}")

    def __getitem__(self, key):
        cdef Py_ssize_t n_orig = self.frame.df_length
        cdef object loc_res
        cdef FastDataFrame row_result
        cdef FastDataFrame result
        cdef list col_names
        cdef str col_name

        # Case 0: tuple (rows, cols) — df.loc[mask, ['name', 'age']]
        if isinstance(key, tuple) and len(key) == 2:
            row_key, col_key = key
            row_result = self._get_rows(row_key)
            if isinstance(col_key, list):
                result = FastDataFrame()
                result.df_length = row_result.df_length
                result.index = row_result.index
                for col_name in col_key:
                    if col_name not in row_result.columns:
                        raise KeyError(f"Column '{col_name}' not found")
                    result.columns[col_name] = row_result.columns[col_name]
                return result
            elif isinstance(col_key, str):
                if col_key not in row_result.columns:
                    raise KeyError(f"Column '{col_key}' not found")
                return row_result.columns[col_key]
            raise TypeError(f"Unsupported column key type: {type(col_key).__name__}")

        return self._get_rows(key)

    cdef FastDataFrame _get_rows(self, key):
        cdef Py_ssize_t n_orig = self.frame.df_length
        cdef object loc_res

        # Callable filter — df.loc[lambda df: df.col > 5]
        if callable(key):
            mask = key(self.frame)
            return self._apply_mask(mask)

        # IntColumn boolean mask — df.loc[mask_col]
        if isinstance(key, IntColumn):
            return self._apply_mask(key)

        # list of booleans — df.loc[[True, False, True]]
        if isinstance(key, list):
            return self._apply_bool_list(key)

        # tuple key — index-based lookup via index.get_loc()
        if self.frame.index is not None:
            loc_res = self.frame.index.get_loc(key)
            return self.frame.iloc[loc_res]

        raise TypeError(f"Unsupported loc key type: {type(key).__name__}")

    cdef FastDataFrame _apply_mask(self, IntColumn mask):
        cdef Py_ssize_t n_orig = self.frame.df_length
        if isinstance(mask, IntColumn):
            if (<IntColumn>mask).length != n_orig:
                raise ValueError(f"Mask length {(<IntColumn>mask).length} != DataFrame length {n_orig}")
        else:
            raise TypeError("Mask must be an IntColumn (0/1 boolean)")

        cdef IntColumn mask_col = <IntColumn>mask
        cdef Py_ssize_t keep_count = 0
        cdef Py_ssize_t i
        for i in range(n_orig):
            if mask_col.data_c[i] != 0:
                keep_count += 1

        if keep_count == 0:
            result = FastDataFrame()
            result.df_length = 0
            for col_name, col_obj in self.frame.columns.items():
                result.columns[col_name] = col_obj.copy()
                result.columns[col_name].length = 0
            return result

        cdef Py_ssize_t* keep_idx = <Py_ssize_t*>aligned_malloc(keep_count * sizeof(Py_ssize_t))
        if not keep_idx: raise MemoryError()
        cdef Py_ssize_t j = 0
        for i in range(n_orig):
            if mask_col.data_c[i] != 0:
                keep_idx[j] = i
                j += 1

        try:
            return self.frame._take(keep_idx, keep_count)
        finally:
            os_aligned_free(keep_idx)

    cdef FastDataFrame _apply_bool_list(self, list mask):
        cdef Py_ssize_t n_orig = self.frame.df_length
        if len(mask) != n_orig:
            raise ValueError(f"Mask length {len(mask)} != DataFrame length {n_orig}")

        cdef Py_ssize_t keep_count = 0
        cdef Py_ssize_t i
        for i in range(n_orig):
            if mask[i]:
                keep_count += 1

        if keep_count == 0:
            result = FastDataFrame()
            result.df_length = 0
            for col_name, col_obj in self.frame.columns.items():
                result.columns[col_name] = col_obj.copy()
                result.columns[col_name].length = 0
            return result

        cdef Py_ssize_t* keep_idx = <Py_ssize_t*>aligned_malloc(keep_count * sizeof(Py_ssize_t))
        if not keep_idx: raise MemoryError()
        cdef Py_ssize_t j = 0
        for i in range(n_orig):
            if mask[i]:
                keep_idx[j] = i
                j += 1

        try:
            return self.frame._take(keep_idx, keep_count)
        finally:
            os_aligned_free(keep_idx)