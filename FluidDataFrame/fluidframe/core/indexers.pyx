# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

cimport cython
from libc.stdlib cimport malloc, free
from fluidframe.core._base cimport aligned_malloc
from fluidframe.core.fastdataframe cimport FastDataFrame
from fluidframe.core.intcolumn cimport IntColumn
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
                    free(indices_c)

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
                    free(indices_c)
            indices_c = <Py_ssize_t*>aligned_malloc(n_new * sizeof(Py_ssize_t))
            if not indices_c: raise MemoryError()

            try:
                for i in range(n_new):
                    idx = <Py_ssize_t>key[i]
                    if idx < 0: idx += n_orig
                    if idx < 0 or idx >= n_orig: raise IndexError("Index out of bounds")
                    indices_c[i] = idx
            except Exception:
                free(indices_c)
                raise
        else:
            raise TypeError("iloc requires integer, slice, or list of integers")

        # 安全地调用底层 C 提取方法
        try:
            result = self.frame._take(indices_c, n_new)
            return result
        finally:
            free(indices_c)


cdef class _LocIndexer:
    def __cinit__(self, FastDataFrame frame):
        self.frame = frame

    def __getitem__(self, key):
        cdef Py_ssize_t n_orig = self.frame.df_length
        cdef object loc_res

        # Case 1: Callable filter — df.loc[lambda df: df.col > 5]
        if callable(key):
            mask = key(self.frame)
            return self._apply_mask(mask)

        # Case 2: IntColumn boolean mask — df.loc[mask_col]
        if isinstance(key, IntColumn):
            return self._apply_mask(key)

        # Case 3: list of booleans — df.loc[[True, False, True]]
        if isinstance(key, list):
            return self._apply_bool_list(key)

        # Case 4: tuple key — index-based lookup via index.get_loc()
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
            free(keep_idx)

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
            free(keep_idx)