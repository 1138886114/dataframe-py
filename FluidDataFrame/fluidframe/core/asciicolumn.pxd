# cython: language_level=3
cimport cython
from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memcpy, memmove, strstr, strlen, strncmp, strncpy
from fluidframe.core._base cimport BaseColumn
from fluidframe.core.intcolumn cimport IntColumn

cdef bint _c_str_contains(const char* source, const char* target) noexcept nogil

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class AsciiColumn(BaseColumn):
    cdef public Py_ssize_t length
    cdef public Py_ssize_t capacity
    cdef char* data_c                # 变长字符缓冲区
    cdef Py_ssize_t* offsets_c       # 地址偏移数组: offsets[i] = 字符串i在data_c中的起始字节偏移
    cdef public Py_ssize_t byte_capacity  # data_c 的字节容量
    cdef public Py_ssize_t byte_length    # data_c 已使用的字节总数
    cdef void _reserve_rows(self, Py_ssize_t new_cap)
    cdef void _reserve_bytes(self, Py_ssize_t new_byte_cap)
    cdef inline str _get_str(self, Py_ssize_t idx)
    cdef void _set_item(self, Py_ssize_t idx, str val) except *
    cdef AsciiColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=*)

