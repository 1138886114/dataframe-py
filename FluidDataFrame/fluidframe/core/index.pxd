
cdef class IndexBase:
    cdef public Py_ssize_t length
    cdef public list _values

    cdef IndexBase _take(self, Py_ssize_t* indices, Py_ssize_t n)
    cdef IndexBase _concat(self, IndexBase other)

    cpdef get_loc(self, object key)

cdef class RangeIndex(IndexBase):
    cdef public Py_ssize_t start
    cdef public Py_ssize_t step

cdef class MultiIndex(IndexBase):
    cdef public list levels          # 每层的唯一值 (list of tuples/strings)
    cdef public list codes           # [IntColumn, IntColumn...] 重用 IntColumn
    cdef public int nlevels
    cdef public list names           # 每层的名字