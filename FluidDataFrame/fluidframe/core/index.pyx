# distutils: language = c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

cimport cython
from libc.stdlib cimport malloc, free
from fluidframe.core._base cimport aligned_malloc
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.index cimport IndexBase, RangeIndex, MultiIndex
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector


cdef class IndexBase:
    cdef IndexBase _take(self, Py_ssize_t* indices, Py_ssize_t n):
        raise NotImplementedError()

    cdef IndexBase _concat(self, IndexBase other):
        raise NotImplementedError()

    cpdef get_loc(self, object key):
        raise NotImplementedError()

cdef class RangeIndex(IndexBase):
    def __cinit__(self, Py_ssize_t start, Py_ssize_t length, Py_ssize_t step=1):
        self.start = start
        self.length = length
        self.step = step

    cdef IndexBase _take(self, Py_ssize_t* indices, Py_ssize_t n):
        # RangeIndex 被切片后，可能会变成不连续的，直接降级为普通索引(后续可实现 Int64Index)
        # 这里为了演示，假设直接返回一个包含了提取下标的结构
        pass

    cpdef get_loc(self, object key):
        if isinstance(key, int):
            cdef Py_ssize_t idx = <Py_ssize_t>key
            if idx < self.start or idx >= self.start + self.length * self.step:
                raise KeyError(key)
            return (idx - self.start) // self.step
        raise TypeError("RangeIndex only supports integer keys")

cdef class MultiIndex(IndexBase):
    def __cinit__(self, list levels, list codes, list names=None):
        self.levels = levels
        self.codes = codes  # 这是包含多个 IntColumn 的列表
        self.nlevels = len(levels)
        self.names = names if names else [None] * self.nlevels
        if self.nlevels > 0:
            self.length = (<IntColumn>codes[0]).length
        else:
            self.length = 0

    cdef IndexBase _take(self, Py_ssize_t* indices, Py_ssize_t n):
        # 极速生成新的 MultiIndex：直接对 codes（即 IntColumn）进行 _reorder
        cdef list new_codes = []
        cdef IntColumn col, new_col
        cdef Py_ssize_t i

        # 将 C 数组转换为 Python 列表（或直接在 IntColumn 中实现 _take_c）
        # 这里假设 IntColumn 增加了一个 _take_c(indices, n) 的纯 C 接口
        for col in self.codes:
            # new_col = col._take_c(indices, n) # 需要你在 IntColumn 补充此方法
            pass

        return MultiIndex(self.levels, new_codes, self.names)

    cpdef get_loc(self, object key):
        # 核心逻辑：将来引入 C++ unordered_map<tuple, vector<int>> 进行 O(1) 查询
        # 目前骨架可以先用 Python 的方式占位，后续再优化为 C++
        pass