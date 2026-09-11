# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False

cimport cython
from libc.string cimport memcpy, memmove, memset, memcmp
from libc.stdio cimport snprintf
from fluidframe.core._base cimport (
    BaseColumn, aligned_malloc, aligned_realloc, os_aligned_free,
    TYPE_BOOL, TYPE_INT, TYPE_DOUBLE, TYPE_CHAR256, TYPE_UCS2,
    TYPE_DATE, TYPE_CATEGORY, TYPE_DATETIME, TYPE_TIME, _resolve_astype_target,
    _c_shift_left, _col_pop_back, _col_pop_front, _col_concat,
    _col_drop_row, _col_gather, _col_gather_zero
)
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.doublecolumn cimport DoubleColumn
from fluidframe.core.asciicolumn cimport AsciiColumn
from fluidframe.core.ucs2column cimport UCS2Column
from fluidframe.core.categorycolumn cimport CategoryColumn
from fluidframe.core.datetimecolumn cimport DateTimeColumn
from fluidframe.core.timecolumn cimport TimeColumn

cdef tuple BOOL_STRS = ('False', 'True')

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef class BoolColumn(BaseColumn):
    @cython.always_allow_keywords(False)
    def __cinit__(self, Py_ssize_t capacity=0):
        self.dtype = TYPE_BOOL
        self.capacity = capacity if capacity > 0 else 0
        self.length = 0
        self._owns_data = True
        self._base_obj = None
        if self.capacity > 0:
            self.data_c = <unsigned char*>aligned_malloc(self.capacity * sizeof(unsigned char))
            if not self.data_c:
                raise MemoryError()
            memset(self.data_c, 0, self.capacity)
        else:
            self.data_c = NULL

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __dealloc__(self):
        if self._owns_data and self.data_c != NULL:
            os_aligned_free(self.data_c)
            self.data_c = NULL

    @staticmethod
    cdef BoolColumn create_view(unsigned char* ptr, Py_ssize_t length, object base_obj):
        cdef BoolColumn col = BoolColumn.__new__(BoolColumn)
        col.dtype = TYPE_BOOL
        col.capacity = length
        col.length = length
        col.data_c = ptr
        col._owns_data = False
        col._base_obj = base_obj
        return col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def get_size(self):
        return self.capacity * sizeof(unsigned char)

    cdef void _reserve(self, Py_ssize_t new_cap):
        if new_cap <= self.capacity:
            return
        cdef unsigned char* new_data = <unsigned char*>aligned_realloc(
            self.data_c, self.capacity * sizeof(unsigned char), new_cap * sizeof(unsigned char))
        if not new_data:
            raise MemoryError()
        memset(new_data + self.capacity, 0, new_cap - self.capacity)
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

    cdef void append_nolen(self, object val) except *:
        """写入当前行(length)位置的值，但不推进 length（由 add_row 统一提交）。"""
        if self.length >= self.capacity:
            raise BufferError("BoolColumn row capacity (%d) reached." % self.capacity)
        self.data_c[self.length] = 1 if val else 0

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def pop_back(self):
        if not self._owns_data:
            raise RuntimeError("Cannot modify a view/slice column. Copy it first.")
        if self.length == 0:
            raise IndexError("pop from empty column")
        cdef unsigned char* ptr = <unsigned char*>_col_pop_back(
            self.data_c, &self.length, sizeof(unsigned char))
        return bool(ptr[0])

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def pop_front(self):
        if not self._owns_data:
            raise RuntimeError("Cannot modify a view/slice column. Copy it first.")
        if self.length == 0:
            raise IndexError("pop from empty column")
        cdef unsigned char val = self.data_c[0]
        _col_pop_front(self.data_c, &self.length, sizeof(unsigned char))
        return bool(val)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def concat(self, BoolColumn other):
        if not self._owns_data:
            raise RuntimeError("Cannot concat into a view/slice column. Copy it first.")
        self._reserve(self.length + other.length)
        _col_concat(self.data_c, &self.length, other.data_c, other.length,
                     sizeof(unsigned char))

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def copy(self):
        cdef Py_ssize_t cap = max(self.length, 1)
        cdef BoolColumn col = BoolColumn(cap)
        col.length = self.length
        if self.length > 0:
            memcpy(col.data_c, self.data_c, self.length * sizeof(unsigned char))
        return col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _reorder_impl(self, list indices):
        cdef Py_ssize_t n = len(indices)
        cdef char err_buf[128]
        if n != self.length:
            snprintf(err_buf, sizeof(err_buf), "Length of indices (%zd) must match column length (%zd)", n, self.length)
            raise ValueError(err_buf.decode('ascii'))

        if n == 0:
            return

        cdef unsigned char* new_data = <unsigned char*>aligned_malloc(n * sizeof(unsigned char))
        if not new_data:
            raise MemoryError()

        cdef Py_ssize_t i, idx
        cdef Py_ssize_t* c_indices = <Py_ssize_t*>aligned_malloc(n * sizeof(Py_ssize_t))
        if not c_indices:
            os_aligned_free(new_data)
            raise MemoryError()
        try:
            for i in range(n):
                idx = <Py_ssize_t>indices[i]
                if idx < 0:
                    idx += self.length
                if idx < 0 or idx >= self.length:
                    snprintf(err_buf, sizeof(err_buf), "Index out of bounds: %zd", idx)
                    raise IndexError(err_buf.decode('ascii'))
                c_indices[i] = idx
            _col_gather(new_data, self.data_c, c_indices, n, sizeof(unsigned char))
        except Exception:
            os_aligned_free(new_data)
            os_aligned_free(c_indices)
            raise
        os_aligned_free(c_indices)

        if self._owns_data and self.data_c != NULL:
            os_aligned_free(self.data_c)
        self.data_c = new_data
        self.capacity = n
        self._owns_data = True
        self._base_obj = None

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

    # ------------------------------------------------------------
    # 极致优化 isin：从 O(N) Python Set 查找优化为 O(1) 分支 + 纯 C 向量化
    # ------------------------------------------------------------
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def isin(self, set values):
        cdef Py_ssize_t n = self.length
        cdef BoolColumn res = BoolColumn(max(n, 1))
        res.length = n
        if n == 0:
            return res

        cdef bint has_true = True in values or 1 in values
        cdef bint has_false = False in values or 0 in values
        cdef unsigned char* self_ptr = self.data_c
        cdef unsigned char* res_ptr = res.data_c
        cdef Py_ssize_t i

        if has_true and has_false:
            memset(res_ptr, 1, n)
        elif not has_true and not has_false:
            memset(res_ptr, 0, n)
        elif has_true:
            with nogil:
                for i in range(n):
                    res_ptr[i] = 1 if self_ptr[i] != 0 else 0
        else:
            with nogil:
                for i in range(n):
                    res_ptr[i] = 1 if self_ptr[i] == 0 else 0
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def where_mask(self, IntColumn mask, object fill):
        cdef Py_ssize_t n = self.length
        if mask.length != n:
            raise ValueError(f"Mask length {mask.length} != column length {n}")
        cdef BoolColumn col = BoolColumn(max(n, 1))
        col.length = n
        if n == 0:
            return col

        cdef unsigned char fill_val = 1 if fill else 0
        cdef unsigned char* self_ptr = self.data_c
        cdef unsigned char* col_ptr = col.data_c
        cdef int* mask_ptr = mask.data_c
        cdef Py_ssize_t i

        with nogil:
            for i in range(n):
                col_ptr[i] = self_ptr[i] if mask_ptr[i] != 0 else fill_val
        return col

    # ------------------------------------------------------------
    # 逻辑操作：直接纯 C 循环，编译器会自动生成 256/512-bit AVX 指令
    # 彻底杜绝手写 64-bit 指针在非对齐地址上的 SIGBUS 隐患
    # ------------------------------------------------------------
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __and__(self, BoolColumn other):
        cdef Py_ssize_t n = self.length
        cdef char err_buf[128]
        if other.length != n:
            snprintf(err_buf, sizeof(err_buf), "Column length %zd != %zd", other.length, n)
            raise ValueError(err_buf.decode('ascii'))

        cdef BoolColumn res = BoolColumn(max(n, 1))
        res.length = n
        cdef unsigned char* s_ptr = self.data_c
        cdef unsigned char* o_ptr = other.data_c
        cdef unsigned char* r_ptr = res.data_c
        cdef Py_ssize_t i

        with nogil:
            for i in range(n):
                r_ptr[i] = s_ptr[i] & o_ptr[i]
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __or__(self, BoolColumn other):
        cdef Py_ssize_t n = self.length
        cdef char err_buf[128]
        if other.length != n:
            snprintf(err_buf, sizeof(err_buf), "Column length %zd != %zd", other.length, n)
            raise ValueError(err_buf.decode('ascii'))

        cdef BoolColumn res = BoolColumn(max(n, 1))
        res.length = n
        cdef unsigned char* s_ptr = self.data_c
        cdef unsigned char* o_ptr = other.data_c
        cdef unsigned char* r_ptr = res.data_c
        cdef Py_ssize_t i

        with nogil:
            for i in range(n):
                r_ptr[i] = s_ptr[i] | o_ptr[i]
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __xor__(self, BoolColumn other):
        cdef Py_ssize_t n = self.length
        cdef char err_buf[128]
        if other.length != n:
            snprintf(err_buf, sizeof(err_buf), "Column length %zd != %zd", other.length, n)
            raise ValueError(err_buf.decode('ascii'))

        cdef BoolColumn res = BoolColumn(max(n, 1))
        res.length = n
        cdef unsigned char* s_ptr = self.data_c
        cdef unsigned char* o_ptr = other.data_c
        cdef unsigned char* r_ptr = res.data_c
        cdef Py_ssize_t i

        with nogil:
            for i in range(n):
                r_ptr[i] = s_ptr[i] ^ o_ptr[i]
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __invert__(self):
        cdef Py_ssize_t n = self.length
        cdef BoolColumn res = BoolColumn(max(n, 1))
        res.length = n
        cdef unsigned char* s_ptr = self.data_c
        cdef unsigned char* r_ptr = res.data_c
        cdef Py_ssize_t i

        with nogil:
            for i in range(n):
                r_ptr[i] = s_ptr[i] ^ 1
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __str__(self):
        cdef list items = []
        cdef Py_ssize_t i
        for i in range(self.length):
            items.append(BOOL_STRS[1 if self.data_c[i] != 0 else 0])
        return f"BoolColumn([{', '.join(items)}])"

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __repr__(self):
        return self.__str__()

    cdef BoolColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=False):
        cdef BoolColumn new_col
        cdef unsigned char* new_data

        if not inplace:
            new_col = BoolColumn(max(n, 1))
            new_col.length = n
            _col_gather_zero(new_col.data_c, self.data_c, indices, n, sizeof(unsigned char))
            return new_col
        else:
            new_data = <unsigned char*>aligned_malloc(max(n, 1) * sizeof(unsigned char))
            if not new_data:
                raise MemoryError()
            _col_gather_zero(new_data, self.data_c, indices, n, sizeof(unsigned char))
            if self._owns_data and self.data_c != NULL:
                os_aligned_free(self.data_c)
            self.data_c = new_data
            self.capacity = n
            self.length = n
            self._owns_data = True
            self._base_obj = None
            return self

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def drop_row(self, Py_ssize_t row_idx):
        if not self._owns_data:
            raise RuntimeError("Cannot drop row on a view/slice column. Copy it first.")
        if row_idx < 0:
            row_idx += self.length
        if row_idx < 0 or row_idx >= self.length:
            raise IndexError("Index out of bounds")
        _col_drop_row(self.data_c, &self.length, row_idx, sizeof(unsigned char))

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def shift_left(self, Py_ssize_t pos):
        cdef Py_ssize_t col_len = self.length
        _c_shift_left(self.data_c, pos, col_len, sizeof(unsigned char))
        self.length = col_len - 1

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __eq__(self, other):
        if not isinstance(other, BoolColumn):
            return NotImplemented
        cdef BoolColumn o = <BoolColumn>other
        cdef char err_buf[128]
        if o.length != self.length:
            snprintf(err_buf, sizeof(err_buf), "Column length %zd != %zd", o.length, self.length)
            raise ValueError(err_buf.decode('ascii'))
        if self.length == 0:
            return True
        return memcmp(self.data_c, o.data_c, self.length * sizeof(unsigned char)) == 0

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
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
        cdef UCS2Column uc
        cdef AsciiColumn cc
        cdef CategoryColumn catc
        cdef DateTimeColumn dtc
        cdef TimeColumn tmc
        cdef str s_val

        if target_type == TYPE_INT:
            ic = IntColumn(safe_cap)
            ic.length = n
            with nogil:
                for i in range(n):
                    ic.data_c[i] = <int>(ptr[i] != 0)
            return ic
        elif target_type == TYPE_DOUBLE:
            dc = DoubleColumn(safe_cap)
            dc.length = n
            with nogil:
                for i in range(n):
                    dc.data_c[i] = <double>(ptr[i] != 0)
            return dc
        elif target_type == TYPE_UCS2:
            uc = UCS2Column(safe_cap)
            uc.length = n
            for i in range(n):
                s_val = BOOL_STRS[1 if ptr[i] != 0 else 0]
                uc._set_item(i, s_val)
            return uc
        elif target_type == TYPE_CHAR256:
            cc = AsciiColumn(safe_cap)
            for i in range(n):
                s_val = BOOL_STRS[1 if ptr[i] != 0 else 0]
                cc.append(s_val)
            return cc
        elif target_type == TYPE_CATEGORY:
            catc = CategoryColumn(safe_cap)
            for i in range(n):
                s_val = BOOL_STRS[1 if ptr[i] != 0 else 0]
                catc.append(s_val)
            return catc
        elif target_type == TYPE_DATETIME:
            dtc = DateTimeColumn(safe_cap)
            for i in range(n):
                s_val = BOOL_STRS[1 if ptr[i] != 0 else 0]
                dtc.append(s_val)
            return dtc
        elif target_type == TYPE_TIME:
            tmc = TimeColumn(safe_cap)
            for i in range(n):
                tmc.append(<int>(ptr[i] != 0))
            return tmc
        elif target_type == TYPE_BOOL:
            return self.copy()
        raise TypeError(f"Failed to cast BOOL column to internal type id: {target_type}")