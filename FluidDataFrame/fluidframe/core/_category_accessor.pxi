# ============================================================
#  Category Accessor for Series
#  pandas-style .cat accessor for categorical columns
# ============================================================
from libc.string cimport memcpy

@cython.final
cdef class _CategoryAccessor:
    """pandas 风格 .cat 访问器。
    由 Series.cat property 懒加载缓存。
    """

    def __cinit__(self, Series series):
        self._series = series

    @property
    def codes(self):
        """返回底层的整数编码 (IntColumn)。使用底层内存拷贝，速度极快。"""
        cdef CategoryColumn col = <CategoryColumn>self._series._col
        cdef IntColumn res = IntColumn(max(col.length, 1))
        res.length = col.length
        if col.length > 0:
            with nogil:
                memcpy(res.data_c, col.codes_c, col.length * sizeof(int))
        return res

    @property
    def categories(self):
        """返回当前所有的不重复类别列表。"""
        cdef CategoryColumn col = <CategoryColumn>self._series._col
        return list(col.categories)

    @cython.binding(False)
    def rename_categories(self, mapper, bint inplace=False):
        cdef CategoryColumn col = <CategoryColumn>self._series._col
        cdef CategoryColumn target = col if inplace else col.copy()
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
        cdef CategoryColumn col = <CategoryColumn>self._series._col
        cdef CategoryColumn target = col if inplace else col.copy()
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
        cdef CategoryColumn col = <CategoryColumn>self._series._col
        cdef list new_cats = [c for c in col.categories if c not in rem_set]
        return self.set_categories(new_cats, inplace=inplace)

    @cython.binding(False)
    def set_categories(self, list new_categories, bint inplace=False):
        cdef CategoryColumn col = <CategoryColumn>self._series._col
        cdef CategoryColumn target = col if inplace else col.copy()
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

        target.categories = final_categories
        target._cat_to_code = final_cat_to_code

        # Re-map existing codes: for each original code, find its
        # category name and look up new code; unknown categories → NaN (-1)
        cdef int i, old_code, n_rows = col.length
        cdef IntColumn res = IntColumn(max(n_rows, 1))
        res.length = n_rows
        cdef str cat_name
        cdef int new_code

        for i in range(n_rows):
            old_code = col.codes_c[i]
            if 0 <= old_code < len(col.categories):
                cat_name = col.categories[old_code]
                new_code = final_cat_to_code.get(cat_name, -1)
            else:
                new_code = -1
            res.data_c[i] = new_code

        if not inplace:
            target.length = n_rows
            return target
        else:
            with nogil:
                memcpy(target.codes_c, res.data_c, n_rows * sizeof(int))

    @cython.binding(False)
    def as_ordered(self, bint inplace=False):
        cdef CategoryColumn col = <CategoryColumn>self._series._col
        cdef CategoryColumn target = col if inplace else col.copy()
        if not inplace:
            return target

    @cython.binding(False)
    def as_unordered(self, bint inplace=False):
        cdef CategoryColumn col = <CategoryColumn>self._series._col
        cdef CategoryColumn target = col if inplace else col.copy()
        if not inplace:
            return target
