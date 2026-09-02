# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False

cimport cython
from libc.string cimport memcpy, memmove, memset, memcmp
from libc.stdlib cimport malloc, free
from libc.stdio cimport snprintf
from fluidframe.core._base cimport (
    BaseColumn, aligned_malloc, aligned_realloc, os_aligned_free,
    TYPE_CATEGORY, TYPE_INT, TYPE_DOUBLE, TYPE_CHAR256, TYPE_UCS2,
    TYPE_BOOL, TYPE_TIMESTAMP, TYPE_DATETIME, TYPE_TIME, _resolve_astype_target
)
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.doublecolumn cimport DoubleColumn
from fluidframe.core.boolcolumn cimport BoolColumn
from fluidframe.core.char256column cimport Char256Column_SoA
from fluidframe.core.ucs2column cimport UCS2Column_SoA
from fluidframe.core.datecolumn cimport DateColumn
from fluidframe.core.datetimecolumn cimport DateTimeColumn
from fluidframe.core.timecolumn cimport TimeColumn


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef class CategoryColumn(BaseColumn):
    def __cinit__(self, Py_ssize_t capacity=0):
        self.dtype = TYPE_CATEGORY
        self.capacity = capacity if capacity > 0 else 0
        self.length = 0
        self.categories = []
        self._cat_to_code = {}
        if self.capacity > 0:
            self.codes_c = <int*>aligned_malloc(self.capacity * sizeof(int))
            if not self.codes_c:
                raise MemoryError()
        else:
            self.codes_c = NULL

    def __dealloc__(self):
        if self.codes_c != NULL:
            os_aligned_free(self.codes_c)
            self.codes_c = NULL

    cpdef Py_ssize_t get_size(self):
        return self.capacity * sizeof(int) + len(self.categories) * sizeof(void*)

    cdef void _reserve(self, Py_ssize_t new_cap):
        if new_cap <= self.capacity:
            return
        cdef int* new_data = <int*>aligned_realloc(
            self.codes_c, self.capacity * sizeof(int), new_cap * sizeof(int))
        if not new_data:
            raise MemoryError()
        self.codes_c = new_data
        self.capacity = new_cap

    cdef int _get_or_add_code(self, str val):
        cdef int code
        cdef object existing = self._cat_to_code.get(val)
        if existing is not None:
            return <int>existing
        code = len(self.categories)
        self.categories.append(val)
        self._cat_to_code[val] = code
        return code

    cdef inline str _resolve(self, Py_ssize_t i):
        cdef int code = self.codes_c[i]
        if code >= 0 and code < len(self.categories):
            return self.categories[code]
        return ''

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def append(self, object val):
        if self.length >= self.capacity:
            self._reserve(max(16, self.capacity * 2))
        cdef str s = str(val) if val is not None else ''
        self.codes_c[self.length] = self._get_or_add_code(s)
        self.length += 1

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def pop_back(self):
        if self.length == 0:
            raise IndexError("pop from empty column")
        self.length -= 1
        return self._resolve(self.length)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def pop_front(self):
        if self.length == 0:
            raise IndexError("pop from empty column")
        cdef str val = self._resolve(0)
        if self.length > 1:
            memmove(self.codes_c, self.codes_c + 1, (self.length - 1) * sizeof(int))
        self.length -= 1
        return val

    # ------------------------------------------------------------
    # concat 优化：预映射 O(K_other) 字典，然后 O(N) 纯 C 写入
    # ------------------------------------------------------------
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def concat(self, CategoryColumn other):
        cdef Py_ssize_t other_len = other.length
        if other_len == 0:
            return

        cdef Py_ssize_t new_len = self.length + other_len
        self._reserve(new_len)

        cdef Py_ssize_t num_other_cats = len(other.categories)
        cdef int* code_map = <int*>aligned_malloc(max(num_other_cats, 1) * sizeof(int))
        if not code_map: raise MemoryError()

        cdef Py_ssize_t j, i
        cdef int* self_codes = self.codes_c + self.length
        cdef int* other_codes = other.codes_c
        try:
            for j in range(num_other_cats):
                code_map[j] = self._get_or_add_code(other.categories[j])

            with nogil:
                for i in range(other_len):
                    self_codes[i] = code_map[other_codes[i]]
            self.length = new_len
        finally:
            os_aligned_free(code_map)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def copy(self):
        cdef CategoryColumn col = CategoryColumn(max(self.length, 1))
        col.length = self.length
        col.categories = list(self.categories)
        col._cat_to_code = dict(self._cat_to_code)
        if self.length > 0:
            memcpy(col.codes_c, self.codes_c, self.length * sizeof(int))
        return col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def _reorder(self, list indices):
        cdef Py_ssize_t n = len(indices)
        cdef char err_buf[128]
        if n != self.length:
            snprintf(err_buf, sizeof(err_buf), "Length of indices (%zd) must match column length (%zd)", n, self.length)
            raise ValueError(err_buf.decode('ascii'))

        if n == 0:
            return

        cdef int* new_data = <int*>aligned_malloc(n * sizeof(int))
        if not new_data: raise MemoryError()
        cdef int* self_codes = self.codes_c
        cdef Py_ssize_t i, idx
        try:
            for i in range(n):
                idx = <Py_ssize_t>indices[i]
                if idx < 0: idx += self.length
                if idx < 0 or idx >= self.length:
                    snprintf(err_buf, sizeof(err_buf), "Index out of bounds: %zd", idx)
                    raise IndexError(err_buf.decode('ascii'))
                new_data[i] = self_codes[idx]
        except Exception:
            os_aligned_free(new_data)
            raise
        os_aligned_free(self.codes_c)
        self.codes_c = new_data
        self.capacity = n

    @property
    def cate(self):
        return _CategoryAccessor(self)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __getitem__(self, Py_ssize_t idx):
        if idx < 0: idx += self.length
        if idx < 0 or idx >= self.length: raise IndexError("Index out of bounds")
        return self._resolve(idx)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __setitem__(self, Py_ssize_t idx, object val):
        if idx < 0: idx += self.length
        if idx < 0 or idx >= self.length: raise IndexError("Index out of bounds")
        cdef str s = str(val) if val is not None else ''
        self.codes_c[idx] = self._get_or_add_code(s)

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __len__(self):
        return self.length

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __bool__(self):
        return self.length > 0

    # ------------------------------------------------------------
    # isin 优化：O(K) 预计算命中标记表，O(N) 纯 C 查表，消除全部内层 Python 查找
    # ------------------------------------------------------------
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def isin(self, set values):
        cdef Py_ssize_t n = self.length
        cdef BoolColumn res = BoolColumn(max(n, 1))
        res.length = n
        if n == 0:
            return res

        cdef Py_ssize_t num_cats = len(self.categories)
        cdef unsigned char* match_table = <unsigned char*>malloc(max(num_cats, 1) * sizeof(unsigned char))
        if not match_table: raise MemoryError()

        cdef Py_ssize_t k, i
        cdef int* codes = self.codes_c
        cdef unsigned char* res_ptr = res.data_c
        try:
            for k in range(num_cats):
                match_table[k] = 1 if self.categories[k] in values else 0

            with nogil:
                for i in range(n):
                    res_ptr[i] = match_table[codes[i]]
        finally:
            free(match_table)
        return res

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def where_mask(self, IntColumn mask, object fill):
        cdef Py_ssize_t n = self.length
        if mask.length != n:
            raise ValueError(f"Mask length {mask.length} != column length {n}")
        cdef CategoryColumn col = CategoryColumn(max(n, 1))
        col.length = n
        if n == 0:
            return col

        cdef str fill_str = str(fill) if fill is not None else ''
        col.categories = list(self.categories)
        col._cat_to_code = dict(self._cat_to_code)
        cdef int fill_code = col._get_or_add_code(fill_str)

        cdef int* mask_ptr = mask.data_c
        cdef int* self_codes = self.codes_c
        cdef int* col_codes = col.codes_c
        cdef Py_ssize_t i

        with nogil:
            for i in range(n):
                col_codes[i] = self_codes[i] if mask_ptr[i] != 0 else fill_code
        return col

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __str__(self):
        cdef list items = []
        cdef Py_ssize_t i
        for i in range(self.length):
            items.append(repr(self._resolve(i)))
        return f"CategoryColumn([{', '.join(items)}])"

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __repr__(self):
        return self.__str__()

    cdef CategoryColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=False):
        cdef Py_ssize_t i
        cdef int* old_codes = self.codes_c
        cdef int* new_codes
        cdef CategoryColumn new_col

        if not inplace:
            new_col = CategoryColumn(max(n, 1))
            new_col.length = n
            new_col.categories = list(self.categories)
            new_col._cat_to_code = dict(self._cat_to_code)
            with nogil:
                for i in range(n):
                    new_col.codes_c[i] = old_codes[indices[i]]
            return new_col
        else:
            new_codes = <int*>aligned_malloc(max(n, 1) * sizeof(int))
            if not new_codes: raise MemoryError()
            with nogil:
                for i in range(n):
                    new_codes[i] = old_codes[indices[i]]
            os_aligned_free(self.codes_c)
            self.codes_c = new_codes
            self.capacity = n
            self.length = n
            return self

    cpdef void drop_row(self, Py_ssize_t row_idx, Py_ssize_t rem_len):
        if row_idx < 0:
            row_idx += self.length
        if rem_len > 0:
            memmove(self.codes_c + row_idx,
                    self.codes_c + row_idx + 1,
                    rem_len * sizeof(int))
        self.length -= 1

    # ------------------------------------------------------------
    # 优化：若 categories 相同直接 memcmp；不同则构建映射比对
    # ------------------------------------------------------------
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __eq__(self, other):
        if not isinstance(other, CategoryColumn):
            return NotImplemented
        cdef CategoryColumn o = <CategoryColumn>other
        if o.length != self.length:
            return False
        if self.length == 0:
            return True

        # 类别列表完全一致时，直接极速内存比对
        if self.categories == o.categories:
            return memcmp(self.codes_c, o.codes_c, self.length * sizeof(int)) == 0

        # 类别列表顺序不同时，先映射 other 的编码
        cdef Py_ssize_t num_o_cats = len(o.categories)
        cdef int* o_to_self = <int*>malloc(max(num_o_cats, 1) * sizeof(int))
        if not o_to_self: raise MemoryError()

        cdef Py_ssize_t j, i
        cdef bint match = True
        cdef int* s_c = self.codes_c
        cdef int* o_c = o.codes_c
        cdef Py_ssize_t n = self.length
        try:
            for j in range(num_o_cats):
                cat = o.categories[j]
                if cat in self._cat_to_code:
                    o_to_self[j] = self._cat_to_code[cat]
                else:
                    o_to_self[j] = -1

            with nogil:
                for i in range(n):
                    if s_c[i] != o_to_self[o_c[i]]:
                        match = False
                        break
        finally:
            free(o_to_self)
        return match

    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented: return result
        return not result

    # ------------------------------------------------------------
    # astype 优化：预先转换 K 个 Category 的值（Memoization），消除 N 次解析
    # ------------------------------------------------------------
    @cython.always_allow_keywords(False)
    @cython.binding(False)
    def astype(self, object target):
        cdef int target_type = _resolve_astype_target(target)
        cdef Py_ssize_t i, k, n = self.length
        cdef Py_ssize_t num_cats = len(self.categories)
        cdef Py_ssize_t safe_cap = max(n, 1)
        cdef int* codes = self.codes_c
        cdef str s

        cdef IntColumn ic
        cdef DoubleColumn dc
        cdef BoolColumn bc
        cdef UCS2Column_SoA uc
        cdef Char256Column_SoA cc
        cdef DateColumn tc
        cdef DateTimeColumn dtc
        cdef TimeColumn tmc

        # 整数预处理转换
        cdef int* int_lookup
        if target_type == TYPE_INT:
            ic = IntColumn(safe_cap)
            ic.length = n
            if n == 0: return ic

            int_lookup = <int*>malloc(max(num_cats, 1) * sizeof(int))
            if not int_lookup: raise MemoryError()
            try:
                for k in range(num_cats):
                    s = self.categories[k]
                    int_lookup[k] = int(float(s)) if s else 0
                with nogil:
                    for i in range(n):
                        ic.data_c[i] = int_lookup[codes[i]]
            finally:
                free(int_lookup)
            return ic

        # 浮点预处理转换
        cdef double* double_lookup
        if target_type == TYPE_DOUBLE:
            dc = DoubleColumn(safe_cap)
            dc.length = n
            if n == 0: return dc

            double_lookup = <double*>malloc(max(num_cats, 1) * sizeof(double))
            if not double_lookup: raise MemoryError()
            try:
                for k in range(num_cats):
                    s = self.categories[k]
                    double_lookup[k] = float(s) if s else 0.0
                with nogil:
                    for i in range(n):
                        dc.data_c[i] = double_lookup[codes[i]]
            finally:
                free(double_lookup)
            return dc

        # 布尔预处理转换
        cdef unsigned char* bool_lookup
        if target_type == TYPE_BOOL:
            bc = BoolColumn(safe_cap)
            bc.length = n
            if n == 0: return bc

            bool_lookup = <unsigned char*>malloc(max(num_cats, 1) * sizeof(unsigned char))
            if not bool_lookup: raise MemoryError()
            try:
                for k in range(num_cats):
                    s = self.categories[k].lower()
                    bool_lookup[k] = 1 if s in ('true', '1', 'yes') else 0
                with nogil:
                    for i in range(n):
                        bc.data_c[i] = bool_lookup[codes[i]]
            finally:
                free(bool_lookup)
            return bc

        # 字符串/复杂类型转换
        if target_type == TYPE_UCS2:
            uc = UCS2Column_SoA(safe_cap)
            uc.length = n
            for i in range(n):
                uc._set_item(i, self._resolve(i))
            return uc
        elif target_type == TYPE_CHAR256:
            cc = Char256Column_SoA(safe_cap)
            for i in range(n):
                cc.append(self._resolve(i))
            return cc
        elif target_type == TYPE_TIMESTAMP:
            tc = DateColumn(safe_cap)
            for i in range(n):
                s = self._resolve(i)
                tc.append(s if s else 0)
            return tc
        elif target_type == TYPE_DATETIME:
            dtc = DateTimeColumn(safe_cap)
            for i in range(n):
                s = self._resolve(i)
                dtc.append(s if s else '')
            return dtc
        elif target_type == TYPE_TIME:
            tmc = TimeColumn(safe_cap)
            for i in range(n):
                s = self._resolve(i)
                tmc.append(s if s else 0)
            return tmc
        elif target_type == TYPE_CATEGORY:
            return self.copy()

        raise TypeError(f"Failed to cast CATEGORY column to internal type id: {target_type}")


cdef class _CategoryAccessor:
    cdef CategoryColumn col

    def __cinit__(self, CategoryColumn col):
        self.col = col

    @property
    def codes(self):
        """返回底层的整数编码 (IntColumn)。使用底层内存拷贝，速度极快。"""
        cdef IntColumn res = IntColumn(max(self.col.length, 1))
        res.length = self.col.length
        if self.col.length > 0:
            with nogil:
                memcpy(res.data_c, self.col.codes_c, self.col.length * sizeof(int))
        return res

    @property
    def categories(self):
        """返回当前所有的不重复类别列表。"""
        return list(self.col.categories)

    @cython.binding(False)
    def rename_categories(self, mapper, bint inplace=False):
        cdef CategoryColumn target = self.col if inplace else self.col.copy()
        cdef list new_cats = list(target.categories)
        cdef int i

        if isinstance(mapper, dict):
            for i in range(len(new_cats)):
                if new_cats[i] in mapper:
                    new_cats[i] = str(mapper[new_cats[i]])
        elif isinstance(mapper, list):
            if len(mapper) != len(new_cats):
                raise ValueError(f"新类别列表的长度 ({len(mapper)}) 必须等于原类别长度 ({len(new_cats)})")
            for i in range(len(mapper)):
                new_cats[i] = str(mapper[i])
        else:
            raise TypeError("mapper 必须是字典或列表")

        if len(new_cats) != len(set(new_cats)):
            raise ValueError("rename_categories 不能产生重复的类别")

        target.categories = new_cats
        target._cat_to_code = {cat: idx for idx, cat in enumerate(new_cats)}

        if not inplace:
            return target

    @cython.binding(False)
    def add_categories(self, list new_categories, bint inplace=False):
        cdef CategoryColumn target = self.col if inplace else self.col.copy()
        cdef str c, c_str

        for c in new_categories:
            c_str = str(c)
            if c_str not in target._cat_to_code:
                target._cat_to_code[c_str] = len(target.categories)
                target.categories.append(c_str)

        if not inplace:
            return target

    @cython.binding(False)
    def remove_categories(self, list removals, bint inplace=False):
        cdef set rem_set = set([str(x) for x in removals])
        cdef list new_cats = [c for c in self.col.categories if c not in rem_set]
        return self.set_categories(new_cats, inplace=inplace)

    @cython.binding(False)
    def set_categories(self, list new_categories, bint inplace=False):
        cdef CategoryColumn target = self.col if inplace else self.col.copy()
        cdef dict final_cat_to_code = {}
        cdef list final_categories = []
        cdef str c, c_str
        cdef int n_new = 0

        for c in new_categories:
            c_str = str(c)
            if c_str not in final_cat_to_code:
                final_cat_to_code[c_str] = n_new
                final_categories.append(c_str)
                n_new += 1

        if '' not in final_cat_to_code:
            final_cat_to_code[''] = n_new
            final_categories.append('')
            n_new += 1

        cdef int missing_code = final_cat_to_code['']
        cdef int old_len = len(target.categories)
        cdef int* code_map = <int*>aligned_malloc(max(old_len, 1) * sizeof(int))
        if not code_map: raise MemoryError()

        cdef int old_code
        cdef Py_ssize_t i
        try:
            for old_code in range(old_len):
                old_cat = target.categories[old_code]
                code_map[old_code] = final_cat_to_code.get(old_cat, missing_code)

            with nogil:
                for i in range(target.length):
                    target.codes_c[i] = code_map[target.codes_c[i]]
        finally:
            os_aligned_free(code_map)

        target.categories = final_categories
        target._cat_to_code = final_cat_to_code

        if not inplace:
            return target