# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False

cimport cython
from libc.string cimport memcpy, memmove
from libc.math cimport NAN
from libc.stdio cimport snprintf
from fluidframe.core._base cimport BaseColumn, aligned_malloc, aligned_realloc, os_aligned_free, TYPE_BOOL, TYPE_INT, TYPE_DOUBLE, TYPE_CHAR256, TYPE_UCS2, TYPE_TIMESTAMP, TYPE_CATEGORY, TYPE_DATETIME, TYPE_TIME, _resolve_astype_target
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.doublecolumn cimport DoubleColumn
from fluidframe.core.char256column cimport Char256Column_SoA
from fluidframe.core.ucs2column cimport UCS2Column_SoA
from fluidframe.core.categorycolumn cimport CategoryColumn
from fluidframe.core.datetimecolumn cimport DateTimeColumn
from fluidframe.core.timecolumn cimport TimeColumn


@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef class BoolColumn(BaseColumn):
    def __cinit__(self, Py_ssize_t capacity=0):
        self.dtype = TYPE_BOOL
        self.capacity = capacity if capacity > 0 else 0
        self.length = 0
        if self.capacity > 0:
            self.data_c = <unsigned char*>aligned_malloc(self.capacity * sizeof(unsigned char))
            if not self.data_c: raise MemoryError()
        else:
            self.data_c = NULL

    def __dealloc__(self):
        if self.data_c != NULL:
            os_aligned_free(self.data_c)
            self.data_c = NULL

    cdef void _reserve(self, Py_ssize_t new_cap):
        if new_cap <= self.capacity:
            return
        cdef unsigned char* new_data = <unsigned char*>aligned_realloc(
            self.data_c, self.capacity * sizeof(unsigned char), new_cap * sizeof(unsigned char))
        if not new_data: raise MemoryError()
        self.data_c = new_data
        self.capacity = new_cap

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def append(self, object val):
        if self.length >= self.capacity:
            self._reserve(max(16, self.capacity * 2))
        self.data_c[self.length] = 1 if val else 0
        self.length += 1

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def append_bool(self, bint val):
        if self.length >= self.capacity:
            self._reserve(max(16, self.capacity * 2))
        self.data_c[self.length] = 1 if val else 0
        self.length += 1

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def pop_back(self):
        if self.length == 0: raise IndexError("pop from empty column")
        self.length -= 1
        return bool(self.data_c[self.length])

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def pop_front(self):
        if self.length == 0: raise IndexError("pop from empty column")
        cdef unsigned char val = self.data_c[0]
        memmove(self.data_c, self.data_c + 1, (self.length - 1) * sizeof(unsigned char))
        self.length -= 1
        return bool(val)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def concat(self, BoolColumn other):
        cdef Py_ssize_t new_len = self.length + other.length
        self._reserve(new_len)
        memcpy(self.data_c + self.length, other.data_c, other.length * sizeof(unsigned char))
        self.length = new_len

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def copy(self):
        cdef BoolColumn col = BoolColumn(max(self.capacity, 1))
        col.length = self.length
        if self.length > 0:
            memcpy(col.data_c, self.data_c, self.length * sizeof(unsigned char))
        return col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _reorder(self, list indices):
        cdef Py_ssize_t n = len(indices)
        cdef char err_buf[128]
        if n != self.length:
            snprintf(err_buf, sizeof(err_buf), "Length of indices (%zd) must match column length (%zd)", n, self.length)
            raise ValueError(err_buf.decode('ascii'))

        cdef unsigned char* new_data = <unsigned char*>aligned_malloc(max(n, 1) * sizeof(unsigned char))
        if not new_data: raise MemoryError()
        cdef unsigned char* self_data = self.data_c
        cdef Py_ssize_t i, idx
        try:
            for i in range(n):
                idx = <Py_ssize_t>indices[i]
                if idx < 0: idx += self.length
                if idx < 0 or idx >= self.length:
                    snprintf(err_buf, sizeof(err_buf), "Index out of bounds: %zd", idx)
                    raise IndexError(err_buf.decode('ascii'))
                new_data[i] = self_data[idx]
        except Exception:
            os_aligned_free(new_data)
            raise
        os_aligned_free(self.data_c)
        self.data_c = new_data

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __getitem__(self, Py_ssize_t idx):
        if idx < 0: idx += self.length
        if idx < 0 or idx >= self.length: raise IndexError("Index out of bounds")
        return bool(self.data_c[idx])

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __setitem__(self, Py_ssize_t idx, object val):
        if idx < 0: idx += self.length
        if idx < 0 or idx >= self.length: raise IndexError("Index out of bounds")
        self.data_c[idx] = 1 if val else 0

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __len__(self):
        return self.length

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __bool__(self):
        return self.length > 0

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def isin(self, set values):
        cdef BoolColumn res = BoolColumn(max(self.capacity, 1))
        res.length = self.length
        cdef unsigned char* self_ptr = self.data_c
        cdef unsigned char* res_ptr = res.data_c
        cdef Py_ssize_t i
        for i in range(self.length):
            res_ptr[i] = 1 if bool(self_ptr[i]) in values else 0
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def where_mask(self, IntColumn mask, object fill):
        cdef Py_ssize_t n = self.length
        if mask.length != n:
            raise ValueError(f"Mask length {mask.length} != column length {n}")
        cdef BoolColumn col = BoolColumn(max(n, 1))
        col.length = n
        cdef unsigned char fill_val = 1 if fill else 0
        cdef unsigned char* self_ptr = self.data_c
        cdef unsigned char* col_ptr = col.data_c
        cdef int* mask_ptr = mask.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                col_ptr[i] = self_ptr[i] if mask_ptr[i] else fill_val
        return col

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def sum(self):
        cdef Py_ssize_t i, s = 0
        cdef unsigned char* ptr = self.data_c
        with nogil:
            for i in range(self.length):
                s += ptr[i]
        return s

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def mean(self):
        if self.length == 0: return NAN
        return <double>self.sum() / self.length

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __and__(self, BoolColumn other):
        cdef Py_ssize_t n = self.length
        if other.length != n:
            raise ValueError(f"Column length {other.length} != {n}")
        cdef BoolColumn res = BoolColumn(max(n, 1))
        res.length = n
        cdef unsigned char* self_ptr = self.data_c
        cdef unsigned char* other_ptr = other.data_c
        cdef unsigned char* res_ptr = res.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                res_ptr[i] = self_ptr[i] & other_ptr[i]
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __or__(self, BoolColumn other):
        cdef Py_ssize_t n = self.length
        if other.length != n:
            raise ValueError(f"Column length {other.length} != {n}")
        cdef BoolColumn res = BoolColumn(max(n, 1))
        res.length = n
        cdef unsigned char* self_ptr = self.data_c
        cdef unsigned char* other_ptr = other.data_c
        cdef unsigned char* res_ptr = res.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                res_ptr[i] = self_ptr[i] | other_ptr[i]
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __xor__(self, BoolColumn other):
        cdef Py_ssize_t n = self.length
        if other.length != n:
            raise ValueError(f"Column length {other.length} != {n}")
        cdef BoolColumn res = BoolColumn(max(n, 1))
        res.length = n
        cdef unsigned char* self_ptr = self.data_c
        cdef unsigned char* other_ptr = other.data_c
        cdef unsigned char* res_ptr = res.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                res_ptr[i] = self_ptr[i] ^ other_ptr[i]
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __invert__(self):
        cdef BoolColumn res = BoolColumn(max(self.capacity, 1))
        res.length = self.length
        cdef unsigned char* self_ptr = self.data_c
        cdef unsigned char* res_ptr = res.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(self.length):
                res_ptr[i] = 1 - self_ptr[i]
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __str__(self):
        cdef list items = []
        cdef Py_ssize_t i
        for i in range(self.length):
            items.append(str(bool(self.data_c[i])))
        return f"BoolColumn([{', '.join(items)}])"

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __repr__(self):
        return self.__str__()

    cdef BoolColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=False):
        cdef Py_ssize_t i
        cdef unsigned char* old_data = self.data_c
        if not inplace:
            new_col = BoolColumn(max(n, 1))
            new_col.length = n
            with nogil:
                for i in range(n):
                    new_col.data_c[i] = old_data[indices[i]]
            return new_col
        else:
            new_data = <unsigned char*>aligned_malloc(max(max(n, self.capacity), 1) * sizeof(unsigned char))
            if not new_data: raise MemoryError()
            with nogil:
                for i in range(n):
                    new_data[i] = old_data[indices[i]]
            os_aligned_free(self.data_c)
            self.data_c = new_data
            self.capacity = max(n, self.capacity)
            self.length = n
            return self

    cpdef void drop_row(self, Py_ssize_t row_idx, Py_ssize_t rem_len):
        if row_idx < 0:
            row_idx += self.length
        if rem_len > 0:
            memmove(self.data_c + row_idx,
                    self.data_c + row_idx + 1,
                    rem_len * sizeof(unsigned char))
        self.length -= 1

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __eq__(self, other):
        cdef BoolColumn o
        cdef Py_ssize_t i
        if not isinstance(other, BoolColumn):
            return NotImplemented
        o = <BoolColumn>other
        if o.length != self.length:
            return False
        cdef unsigned char* self_ptr = self.data_c
        cdef unsigned char* o_ptr = o.data_c
        for i in range(self.length):
            if self_ptr[i] != o_ptr[i]:
                return False
        return True

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented: return result
        return not result

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def astype(self, object target):
        cdef int target_type = _resolve_astype_target(target)
        cdef Py_ssize_t i, n = self.length
        cdef Py_ssize_t safe_cap = max(n, 1)
        cdef unsigned char* ptr = self.data_c
        cdef IntColumn ic
        cdef DoubleColumn dc
        cdef UCS2Column_SoA uc
        cdef Char256Column_SoA cc
        cdef CategoryColumn catc
        cdef DateTimeColumn dtc
        cdef TimeColumn tmc
        if target_type == TYPE_INT:
            ic = IntColumn(safe_cap)
            ic.length = n
            with nogil:
                for i in range(n):
                    ic.data_c[i] = <int>ptr[i]
            return ic
        elif target_type == TYPE_DOUBLE:
            dc = DoubleColumn(safe_cap)
            dc.length = n
            with nogil:
                for i in range(n):
                    dc.data_c[i] = <double>ptr[i]
            return dc
        elif target_type == TYPE_UCS2:
            uc = UCS2Column_SoA(safe_cap)
            uc.length = n
            for i in range(n):
                uc._set_item(i, 'True' if ptr[i] else 'False')
            return uc
        elif target_type == TYPE_CHAR256:
            cc = Char256Column_SoA(safe_cap)
            for i in range(n):
                cc.append('True' if ptr[i] else 'False')
            return cc
        elif target_type == TYPE_CATEGORY:
            catc = CategoryColumn(safe_cap)
            for i in range(n):
                catc.append('True' if ptr[i] else 'False')
            return catc
        elif target_type == TYPE_DATETIME:
            dtc = DateTimeColumn(safe_cap)
            for i in range(n):
                dtc.append('True' if ptr[i] else 'False')
            return dtc
        elif target_type == TYPE_TIME:
            tmc = TimeColumn(safe_cap)
            for i in range(n):
                tmc.append(<int>ptr[i])
            return tmc
        elif target_type == TYPE_BOOL:
            return self.copy()
        raise TypeError(f"Failed to cast BOOL column to internal type id: {target_type}")