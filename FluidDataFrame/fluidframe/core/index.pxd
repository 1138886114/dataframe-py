from fluidframe.core.intcolumn cimport IntColumn

cdef enum IndexType:
    INDEX_RANGE = 0
    INDEX_INT = 1
    INDEX_MULTI = 2

cdef class IndexBase:
    cdef public Py_ssize_t length
    cdef public IndexType index_type

    cdef IndexBase _take(self, Py_ssize_t* indices, Py_ssize_t n)
    cdef IndexBase _concat(self, IndexBase other)
    cpdef get_loc(self, object key)
    cpdef list values(self)


cdef class IntIndex(IndexBase):
    cdef public IntColumn data
    cpdef list values(self)

cdef class RangeIndex(IndexBase):
    cdef public Py_ssize_t start
    cdef public Py_ssize_t step
    cpdef list values(self)

cdef class MultiIndex(IndexBase):
    cdef public list levels          # 每层的唯一值 (list of tuples/strings)
    cdef public list codes           # [IntColumn, IntColumn...] 重用 IntColumn
    cdef public int nlevels
    cdef public list names           # 每层的名字
    cpdef list values(self)
