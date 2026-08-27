# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

cimport cython
from libc.stdlib cimport malloc, free
from fluidframe.core._base cimport aligned_malloc
from fluidframe.core.fastdataframe cimport FastDataFrame

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

            # TODO: 后续可以返回类似于 Series 的结构，当前返回 1行的 df
            indices_c = <Py_ssize_t*>aligned_malloc(sizeof(Py_ssize_t))
            if not indices_c: raise MemoryError()
            indices_c[0] = idx
            n_new = 1

        elif isinstance(key, slice):
            # 获取切片 (df.iloc[2:10])
            start, stop, step = key.indices(n_orig)
            n_new = len(range(start, stop, step))
            if n_new <= 0:
                result = FastDataFrame()
                for col_name, col_obj in self.frame.columns.items():
                    result.columns[col_name] = col_obj.copy()
                    result.columns[col_name].length = 0
                return result

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
                result = FastDataFrame()
                for col_name, col_obj in self.frame.columns.items():
                    result.columns[col_name] = col_obj.copy()
                    result.columns[col_name].length = 0
                return result
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
        cdef object loc_res = self.frame.index.get_loc(key)

        # get_loc 的返回可能有多种：
        # 1. 整数 (匹配唯一一条)
        # 2. Slice 对象 (匹配一段区间)
        # 3. 数组/列表 (匹配多条散列数据)

        # 将 get_loc 的结果转发给同构的逻辑即可（复用代码，直接模拟 iloc 行为）
        # 这里为了保持骨架清晰，可以用 Python 层的 _iloc 代理，或者抽取公共 _take_by_py_obj 函数
        return self.frame.iloc[loc_res]