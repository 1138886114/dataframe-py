# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport cython
from libc.string cimport memcpy, memmove
from fluidframe.core._base cimport BaseColumn, aligned_malloc, aligned_realloc, os_aligned_free, TYPE_CATEGORY
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.doublecolumn cimport DoubleColumn
from fluidframe.core.boolcolumn cimport BoolColumn
from fluidframe.core.char256column cimport Char256Column_SoA
from fluidframe.core.ucs2column cimport UCS2Column_SoA
from fluidframe.core.datecolumn cimport DateColumn
from fluidframe.core.datetimecolumn cimport DateTimeColumn
from fluidframe.core.timecolumn cimport TimeColumn


cdef class CategoryColumn(BaseColumn):
    def __cinit__(self, Py_ssize_t capacity=0):
        self.dtype = TYPE_CATEGORY
        self.capacity = capacity if capacity > 0 else 0
        self.length = 0
        self.categories = []
        self._cat_to_code = {}
        if self.capacity > 0:
            self.codes_c = <int*>aligned_malloc(self.capacity * sizeof(int))
            if not self.codes_c: raise MemoryError()
        else:
            self.codes_c = NULL

    def __dealloc__(self):
        if self.codes_c != NULL:
            os_aligned_free(self.codes_c)
            self.codes_c = NULL

    cdef void _reserve(self, Py_ssize_t new_cap):
        if new_cap <= self.capacity:
            return
        cdef int* new_data = <int*>aligned_realloc(
            self.codes_c, self.capacity * sizeof(int), new_cap * sizeof(int))
        if not new_data: raise MemoryError()
        self.codes_c = new_data
        self.capacity = new_cap

    cdef int _get_or_add_code(self, str val):
        cdef int code
        if val not in self._cat_to_code:
            code = len(self.categories)
            self.categories.append(val)
            self._cat_to_code[val] = code
        return self._cat_to_code[val]

    cdef inline str _resolve(self, Py_ssize_t i):
        if self.codes_c[i] < len(self.categories):
            return self.categories[self.codes_c[i]]
        return ''

    def append(self, object val):
        if self.length >= self.capacity:
            self._reserve(max(16, self.capacity * 2))
        cdef str s = str(val) if val is not None else ''
        self.codes_c[self.length] = self._get_or_add_code(s)
        self.length += 1

    def pop_back(self):
        if self.length == 0: raise IndexError("pop from empty column")
        self.length -= 1
        return self.categories[self.codes_c[self.length]]

    def pop_front(self):
        if self.length == 0: raise IndexError("pop from empty column")
        cdef int val_code = self.codes_c[0]
        memmove(self.codes_c, self.codes_c + 1, (self.length - 1) * sizeof(int))
        self.length -= 1
        return self.categories[val_code]

    def concat(self, CategoryColumn other):
        cdef Py_ssize_t new_len = self.length + other.length
        self._reserve(new_len)
        cdef Py_ssize_t i
        cdef int new_code
        for i in range(other.length):
            new_code = self._get_or_add_code(other.categories[other.codes_c[i]])
            self.codes_c[self.length + i] = new_code
        self.length = new_len

    def copy(self):
        cdef CategoryColumn col = CategoryColumn(max(self.capacity, 1))
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
        if n != self.length:
            raise ValueError("Length of indices must match column length")

        cdef int* new_data = <int*>aligned_malloc(max(n, 1) * sizeof(int))
        if not new_data: raise MemoryError()
        cdef Py_ssize_t i, idx
        try:
            for i in range(n):
                idx = <Py_ssize_t>indices[i]
                if idx < 0: idx += self.length
                if idx < 0 or idx >= self.length:
                    raise IndexError("Index out of bounds")
                new_data[i] = self.codes_c[idx]
        except Exception:
            os_aligned_free(new_data)
            raise
        os_aligned_free(self.codes_c)
        self.codes_c = new_data

    @property
    def cate(self):
        return _CategoryAccessor(self)

    def __getitem__(self, Py_ssize_t idx):
        if idx < 0: idx += self.length
        if idx < 0 or idx >= self.length: raise IndexError("Index out of bounds")
        return self.categories[self.codes_c[idx]]

    def __setitem__(self, Py_ssize_t idx, object val):
        if idx < 0: idx += self.length
        if idx < 0 or idx >= self.length: raise IndexError("Index out of bounds")
        cdef str s = str(val) if val is not None else ''
        self.codes_c[idx] = self._get_or_add_code(s)

    def __len__(self):
        return self.length

    def __bool__(self):
        return self.length > 0

    def isin(self, set values):
        cdef BoolColumn res
        res = BoolColumn(max(self.capacity, 1))
        res.length = self.length
        cdef Py_ssize_t i
        cdef str val
        for i in range(self.length):
            val = self.categories[self.codes_c[i]]
            res.data_c[i] = 1 if val in values else 0
        return res

    def where_mask(self, IntColumn mask, object fill):
        cdef Py_ssize_t n = self.length
        if mask.length != n:
            raise ValueError(f"Mask length {mask.length} != column length {n}")
        cdef CategoryColumn col = CategoryColumn(max(n, 1))
        cdef Py_ssize_t i
        cdef str fill_str = str(fill) if fill is not None else ''
        for i in range(n):
            if mask.data_c[i]:
                col.codes_c[i] = col._get_or_add_code(self.categories[self.codes_c[i]])
            else:
                col.codes_c[i] = col._get_or_add_code(fill_str)
        col.length = n
        return col

    def __str__(self):
        cdef list items = []
        cdef Py_ssize_t i
        for i in range(self.length):
            items.append(repr(self.categories[self.codes_c[i]]))
        return f"CategoryColumn([{', '.join(items)}])"

    def __repr__(self):
        return self.__str__()

    cdef CategoryColumn take_by_indices(self, Py_ssize_t* indices, Py_ssize_t n, bint inplace=False):
        cdef Py_ssize_t i
        if not inplace:
            new_col = CategoryColumn(max(n, 1))
            new_col.length = n
            new_col.categories = list(self.categories)
            new_col._cat_to_code = dict(self._cat_to_code)
            for i in range(n):
                new_col.codes_c[i] = self.codes_c[indices[i]]
            return new_col
        else:
            new_codes = <int*>aligned_malloc(max(max(n, self.capacity), 1) * sizeof(int))
            if not new_codes: raise MemoryError()
            for i in range(n):
                new_codes[i] = self.codes_c[indices[i]]
            os_aligned_free(self.codes_c)
            self.codes_c = new_codes
            self.capacity = max(n, self.capacity)
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

    def __eq__(self, other):
        cdef CategoryColumn o
        cdef Py_ssize_t i
        if not isinstance(other, CategoryColumn):
            return NotImplemented
        o = <CategoryColumn>other
        if o.length != self.length:
            return False
        if self.categories != o.categories:
            return False
        for i in range(self.length):
            if self.codes_c[i] != o.codes_c[i]:
                return False
        return True

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented: return result
        return not result

    def astype(self, object target):
        cdef str tname
        if isinstance(target, str):
            tname = target.lower()
        else:
            tname = {int: 'int', float: 'double', bool: 'bool', str: 'ucs2'}.get(target, '')
        cdef Py_ssize_t i, n = self.length
        cdef str s
        cdef IntColumn ic
        cdef DoubleColumn dc
        cdef UCS2Column_SoA uc
        cdef Char256Column_SoA cc
        cdef DateColumn tc
        cdef BoolColumn bc
        cdef DateTimeColumn dtc
        cdef TimeColumn tmc
        if tname in ('int', 'i4'):
            ic = IntColumn(max(n, 1))
            ic.length = n
            for i in range(n):
                s = self._resolve(i)
                ic.data_c[i] = int(float(s)) if s else 0
            return ic
        elif tname in ('double', 'float'):
            dc = DoubleColumn(max(n, 1))
            dc.length = n
            for i in range(n):
                s = self._resolve(i)
                dc.data_c[i] = float(s) if s else 0.0
            return dc
        elif tname == 'bool':
            bc = BoolColumn(max(n, 1))
            bc.length = n
            for i in range(n):
                s = self._resolve(i)
                bc.data_c[i] = 1 if s.lower() in ('true', '1', 'yes') else 0
            return bc
        elif tname in ('str', 'string', 'ucs2'):
            uc = UCS2Column_SoA(max(n, 1))
            uc.length = n
            for i in range(n):
                s = self._resolve(i)
                uc._set_item(i, s)
            return uc
        elif tname in ('char256', 'ascii'):
            cc = Char256Column_SoA(max(n, 1))
            for i in range(n):
                s = self._resolve(i)
                cc.append(s)
            return cc
        elif tname in ('timestamp', 'date'):
            tc = DateColumn(max(n, 1))
            for i in range(n):
                s = self._resolve(i)
                if s:
                    tc.append(s)
                else:
                    tc.append(0)
            return tc
        elif tname == 'datetime':
            dtc = DateTimeColumn(max(n, 1))
            for i in range(n):
                s = self._resolve(i)
                if s:
                    dtc.append(s)
                else:
                    dtc.append('')
            return dtc
        elif tname == 'time':
            tmc = TimeColumn(max(n, 1))
            for i in range(n):
                s = self._resolve(i)
                if s:
                    tmc.append(s)
                else:
                    tmc.append(0)
            return tmc
        raise TypeError(f"Cannot convert category to '{tname}'")


cdef class _CategoryAccessor:
    cdef CategoryColumn col

    def __cinit__(self, CategoryColumn col):
        self.col = col

    @property
    def codes(self):
        """返回底层的整数编码 (IntColumn)。使用底层内存拷贝，速度极快。"""
        cdef IntColumn res = IntColumn(max(self.col.capacity, 1))
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
        """
        重命名现有的类别。这是一个极速的 O(K) 元数据操作。
        mapper 可以是字典 {'old_name': 'new_name'}，也可以是对应长度的列表。
        """
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
        target._cat_to_code = {}
        for i in range(len(new_cats)):
            target._cat_to_code[new_cats[i]] = i

        if not inplace:
            return target

    @cython.binding(False)
    def add_categories(self, list new_categories, bint inplace=False):
        """添加新的类别选项，原有的数据和编码不发生改变。"""
        cdef CategoryColumn target = self.col if inplace else self.col.copy()
        cdef str c

        for c in new_categories:
            c_str = str(c)
            if c_str not in target._cat_to_code:
                target._cat_to_code[c_str] = len(target.categories)
                target.categories.append(c_str)

        if not inplace:
            return target

    @cython.binding(False)
    def remove_categories(self, list removals, bint inplace=False):
        """移除指定的类别。列中原有属于该类别的数据将被安全置空 ('')。"""
        cdef set rem_set = set([str(x) for x in removals])
        cdef list new_cats = [c for c in self.col.categories if c not in rem_set]
        return self.set_categories(new_cats, inplace=inplace)

    @cython.binding(False)
    def set_categories(self, list new_categories, bint inplace=False):
        """
        强行设置列的类别为指定的列表。
        不在新列表中的老类别的数据将被置空 ('')，底层 O(N) 极速映射。
        """
        cdef CategoryColumn target = self.col if inplace else self.col.copy()
        cdef dict final_cat_to_code = {}
        cdef list final_categories = []
        cdef str c, c_str
        cdef int n_new = 0

        # 1. 构建全新的类别结构
        for c in new_categories:
            c_str = str(c)
            if c_str not in final_cat_to_code:
                final_cat_to_code[c_str] = n_new
                final_categories.append(c_str)
                n_new += 1

        # 确保缺失值占位符 '' 存在，防止引发越界
        if '' not in final_cat_to_code:
            final_cat_to_code[''] = n_new
            final_categories.append('')
            n_new += 1

        cdef int missing_code = final_cat_to_code['']

        # 2. 构建安全的旧代码到新代码映射表 (避免嵌套字典查找)
        cdef int old_len = len(target.categories)
        cdef int* code_map = <int*>aligned_malloc(max(old_len, 1) * sizeof(int))
        if not code_map: raise MemoryError()

        cdef int old_code
        for old_code, old_cat in enumerate(target.categories):
            if old_cat in final_cat_to_code:
                code_map[old_code] = final_cat_to_code[old_cat]
            else:
                code_map[old_code] = missing_code

        # 3. O(N) C级别极速映射
        cdef Py_ssize_t i
        with nogil:
            for i in range(target.length):
                target.codes_c[i] = code_map[target.codes_c[i]]

        os_aligned_free(code_map)

        # 4. 更新对象元数据
        target.categories = final_categories
        target._cat_to_code = final_cat_to_code

        if not inplace:
            return target