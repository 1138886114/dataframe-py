# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

cimport cython
from fluidframe.core._base cimport BaseColumn, TYPE_INT, TYPE_DOUBLE, TYPE_BOOL, TYPE_CHAR256, TYPE_UCS2, TYPE_DATE, TYPE_DATETIME, TYPE_TIME, TYPE_CATEGORY
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.asciicolumn cimport AsciiColumn
from fluidframe.core.ucs2column cimport UCS2Column
from fluidframe.core.categorycolumn cimport CategoryColumn
from fluidframe.core.index cimport IndexBase

include "_string_accessor.pxi"
include "_datetime_accessor.pxi"
include "_category_accessor.pxi"


@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class Series:
    """轻量级 Series 实现，单列代理，知道自己的名字（name），绑定了一个索引（index）。"""

    def __cinit__(self, str name, BaseColumn col, IndexBase index):
        self.name = name
        self._col = col
        self._index = index

    # =================================================================
    #  Properties
    # =================================================================
    
    @property
    def dtype(self):
        """返回底层列类型"""
        return self._col.dtype
    
    @property
    def is_2d(self):
        """返回是否2D布局"""
        return self._col.is_2d
    
    @property
    def length(self):
        """返回Series长度（逻辑行数），向后兼容"""
        if self._col.is_2d and self._col.num_cols > 1:
            return self._col._get_rows_c()
        return self._col.length
    
    @property
    def index(self):
        """返回索引对象"""
        return self._index
    
    @index.setter
    def index(self, value):
        self._index = value
    
    @property
    def values(self):
        """返回底层数据的Python列表"""
        n = len(self)
        return [self._col[i] for i in range(n)]

    # =================================================================
    #  基本信息
    # =================================================================
    
    def __len__(self):
        """返回Series长度（逻辑行数）"""
        if self._col.is_2d and self._col.num_cols > 1:
            return self._col._get_rows_c()
        return self._col.length
    
    def __repr__(self):
        """字符串表示"""
        n = len(self)
        max_display = 30
        lines = []
        lines.append(f"Series(name='{self.name}')")
        lines.append(f"dtype: {self.dtype}")
        lines.append(f"length: {n}")
        
        if n <= max_display:
            for i in range(n):
                idx_val = self._index[i] if self._index else i
                lines.append(f"  {idx_val}: {self._col[i]}")
        else:
            for i in range(10):
                idx_val = self._index[i] if self._index else i
                lines.append(f"  {idx_val}: {self._col[i]}")
            lines.append(f"  ... ({n - 20} rows omitted)")
            for i in range(n - 10, n):
                idx_val = self._index[i] if self._index else i
                lines.append(f"  {idx_val}: {self._col[i]}")
        
        return "\n".join(lines)

    def __str__(self):
        return self.__repr__()

    # =================================================================
    #  标量提取与切片
    # =================================================================
    
    def __getitem__(self, key):
        """支持切片、标量、布尔掩码"""
        from fluidframe.core.intcolumn import IntColumn
        from fluidframe.core.index import IntIndex
        
        # 布尔掩码筛选
        if isinstance(key, Series) and key.dtype == TYPE_BOOL:
            if key._index is not self._index:
                key = key.reindex(self._index)
            new_col = self._col.filter(key._col)
            new_index = self._index.filter(key._col)
            return Series(self.name, new_col, new_index)
        
        # 整数索引
        if isinstance(key, int):
            if key < 0:
                key = len(self) + key
            if key < 0 or key >= len(self):
                raise IndexError("index out of range")
            return self._col[key]
        
        # 切片
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            indices = list(range(start, stop, step))
            n = len(indices)
            indices_arr = IntColumn.from_list(indices)
            new_col = self._col.take_by_indices_c(indices, n, False)
            new_index = self._index.take_by_indices_c(indices, n)
            return Series(self.name, new_col, new_index)
        
        raise TypeError(f"invalid index type: {type(key)}")

    def __setitem__(self, key, value):
        """标量赋值"""
        if isinstance(key, int):
            if key < 0:
                key = len(self) + key
            if key < 0 or key >= len(self):
                raise IndexError("index out of range")
            self._col[key] = value
        else:
            raise TypeError(f"invalid index type: {type(key)}")

    # =================================================================
    #  算术运算
    # =================================================================
    
    def __add__(self, other):
        if isinstance(other, Series):
            if self._index is (<Series>other)._index:
                new_col = self._col + (<Series>other)._col
            else:
                aligned_other = other.reindex(self._index)
                new_col = self._col + aligned_other._col
            return Series(self.name, new_col, self._index)
        else:
            new_col = self._col + other
            return Series(self.name, new_col, self._index)

    def __sub__(self, other):
        if isinstance(other, Series):
            if self._index is (<Series>other)._index:
                new_col = self._col - (<Series>other)._col
            else:
                aligned_other = other.reindex(self._index)
                new_col = self._col - aligned_other._col
            return Series(self.name, new_col, self._index)
        else:
            new_col = self._col - other
            return Series(self.name, new_col, self._index)

    def __mul__(self, other):
        if isinstance(other, Series):
            if self._index is (<Series>other)._index:
                new_col = self._col * (<Series>other)._col
            else:
                aligned_other = other.reindex(self._index)
                new_col = self._col * aligned_other._col
            return Series(self.name, new_col, self._index)
        else:
            new_col = self._col * other
            return Series(self.name, new_col, self._index)

    def __truediv__(self, other):
        if isinstance(other, Series):
            if self._index is (<Series>other)._index:
                new_col = self._col / (<Series>other)._col
            else:
                aligned_other = other.reindex(self._index)
                new_col = self._col / aligned_other._col
            return Series(self.name, new_col, self._index)
        else:
            new_col = self._col / other
            return Series(self.name, new_col, self._index)

    def __floordiv__(self, other):
        if isinstance(other, Series):
            if self._index is (<Series>other)._index:
                new_col = self._col // (<Series>other)._col
            else:
                aligned_other = other.reindex(self._index)
                new_col = self._col // aligned_other._col
            return Series(self.name, new_col, self._index)
        else:
            new_col = self._col // other
            return Series(self.name, new_col, self._index)

    def __mod__(self, other):
        if isinstance(other, Series):
            if self._index is (<Series>other)._index:
                new_col = self._col % (<Series>other)._col
            else:
                aligned_other = other.reindex(self._index)
                new_col = self._col % aligned_other._col
            return Series(self.name, new_col, self._index)
        else:
            new_col = self._col % other
            return Series(self.name, new_col, self._index)

    def __neg__(self):
        return Series(self.name, -self._col, self._index)

    def __pos__(self):
        return self

    def __abs__(self):
        return Series(self.name, self._col.abs(), self._index)

    # =================================================================
    #  比较运算
    # =================================================================
    
    def __eq__(self, other):
        if isinstance(other, Series):
            if self._index is (<Series>other)._index:
                return self._col == (<Series>other)._col
            else:
                aligned_other = other.reindex(self._index)
                return self._col == aligned_other._col
        else:
            return self._col == other

    def __ne__(self, other):
        if isinstance(other, Series):
            if self._index is (<Series>other)._index:
                return self._col != (<Series>other)._col
            else:
                aligned_other = other.reindex(self._index)
                return self._col != aligned_other._col
        else:
            return self._col != other

    def __lt__(self, other):
        if isinstance(other, Series):
            if self._index is (<Series>other)._index:
                return self._col < (<Series>other)._col
            else:
                aligned_other = other.reindex(self._index)
                return self._col < aligned_other._col
        else:
            return self._col < other

    def __le__(self, other):
        if isinstance(other, Series):
            if self._index is (<Series>other)._index:
                return self._col <= (<Series>other)._col
            else:
                aligned_other = other.reindex(self._index)
                return self._col <= aligned_other._col
        else:
            return self._col <= other

    def __gt__(self, other):
        if isinstance(other, Series):
            if self._index is (<Series>other)._index:
                return self._col > (<Series>other)._col
            else:
                aligned_other = other.reindex(self._index)
                return self._col > aligned_other._col
        else:
            return self._col > other

    def __ge__(self, other):
        if isinstance(other, Series):
            if self._index is (<Series>other)._index:
                return self._col >= (<Series>other)._col
            else:
                aligned_other = other.reindex(self._index)
                return self._col >= aligned_other._col
        else:
            return self._col >= other

    # =================================================================
    #  单列统计
    # =================================================================
    
    def sum(self):
        """求和"""
        return self._col.sum()

    def mean(self):
        """均值"""
        return self._col.mean()

    def std(self):
        """标准差"""
        return self._col.std()

    def var(self):
        """方差"""
        return self._col.var()

    def min(self):
        """最小值"""
        return self._col.min()

    def max(self):
        """最大值"""
        return self._col.max()

    def median(self):
        """中位数"""
        return self._col.median()

    def prod(self):
        """连乘积"""
        return self._col.prod()

    def count(self):
        """计数（非空/非NaN）"""
        return self._col.count()

    def nunique(self):
        """唯一值数量"""
        return len(self.unique())

    def unique(self):
        """唯一值"""
        return self._col.unique()

    def value_counts(self):
        """值计数，返回Series（以value为索引，count为值）"""
        from fluidframe.core.intcolumn import IntColumn
        from fluidframe.core.index import IntIndex
        
        counts = {}
        n = len(self)
        for i in range(n):
            val = self._col[i]
            counts[val] = counts.get(val, 0) + 1
        
        # 按值排序
        sorted_vals = sorted(counts.keys())
        sorted_counts = [counts[v] for v in sorted_vals]
        
        # 创建新列和索引
        new_col = IntColumn.from_list(sorted_counts)
        new_index = IntIndex(len(sorted_vals))
        return Series(self.name, new_col, new_index)

    def argmax(self):
        """最大值索引"""
        return self._col.idxmax()

    def argmin(self):
        """最小值索引"""
        return self._col.idxmin()

    def quantile(self, q):
        """分位数"""
        return self._col.quantile(q)

    def skew(self):
        """偏度"""
        return self._col.skew()

    def kurt(self):
        """峰度"""
        return self._col.kurt()

    # =================================================================
    #  单列变换
    # =================================================================
    
    def fillna(self, value):
        """填充NaN/None值"""
        return Series(self.name, self._col.fillna(value), self._index)

    def ffill(self):
        """向前填充"""
        return Series(self.name, self._col.ffill(), self._index)

    def bfill(self):
        """向后填充"""
        return Series(self.name, self._col.bfill(), self._index)

    def astype(self, dtype):
        """类型转换"""
        return Series(self.name, self._col.astype(dtype), self._index)

    def abs(self):
        """绝对值"""
        return Series(self.name, self._col.abs(), self._index)

    def clip(self, lower=None, upper=None):
        """截断值"""
        return Series(self.name, self._col.clip(lower, upper), self._index)

    def round(self, decimals):
        """四舍五入"""
        return Series(self.name, self._col.round(decimals), self._index)

    def diff(self, periods=1):
        """差分"""
        return Series(self.name, self._col.diff(periods), self._index)

    def pct_change(self, periods=1):
        """百分比变化"""
        return Series(self.name, self._col.pct_change(periods), self._index)

    def cumsum(self):
        """累积和"""
        return Series(self.name, self._col.cumsum(), self._index)

    def cumprod(self):
        """累积积"""
        return Series(self.name, self._col.cumprod(), self._index)

    def cummax(self):
        """累积最大值"""
        return Series(self.name, self._col.cummax(), self._index)

    def cummin(self):
        """累积最小值"""
        return Series(self.name, self._col.cummin(), self._index)

    # =================================================================
    #  排序与索引
    # =================================================================
    
    def sort_values(self, ascending=True):
        """按值排序"""
        return self._col.sort_values(ascending)

    def head(self, n=5):
        """前n行"""
        return self._col.head(n)

    def tail(self, n=5):
        """后n行"""
        return self._col.tail(n)

    def nsmallest(self, n):
        """最小的n个值"""
        return self._col.nsmallest(n)

    def nlargest(self, n):
        """最大的n个值"""
        return self._col.nlargest(n)

    def idxmax(self):
        """最大值索引"""
        return self._col.idxmax()

    def idxmin(self):
        """最小值索引"""
        return self._col.idxmin()

    # =================================================================
    #  布尔判断
    # =================================================================
    
    def isna(self):
        """判断是否为NaN/None"""
        return self._col.isna()

    def notna(self):
        """判断是否非NaN/None"""
        return self._col.notna()

    def isin(self, values):
        """判断是否在值列表中"""
        return self._col.isin(values)

    def between(self, low, high):
        """判断是否在区间内"""
        return self._col.between(low, high)

    def all(self):
        """判断是否全部为True"""
        n = len(self)
        if n == 0:
            return True
        for i in range(n):
            if not self._col[i]:
                return False
        return True

    def any(self):
        """判断是否有True"""
        n = len(self)
        for i in range(n):
            if self._col[i]:
                return True
        return False

    # =================================================================
    #  条件替换
    # =================================================================
    
    def where(self, cond, other):
        """条件替换：当cond为True时保留原值，否则替换为other"""
        n = len(self)
        new_vals = []
        for i in range(n):
            if isinstance(cond, Series):
                c = cond[i]
            else:
                c = cond
            if c:
                new_vals.append(self._col[i])
            else:
                new_vals.append(other)
        
        from fluidframe.core.dataframe import _column_from_list_impl
        new_col = _column_from_list_impl(None, self.name, new_vals)
        return Series(self.name, new_col, self._index)

    def mask(self, cond, other):
        """掩码替换：当cond为True时替换为other，否则保留原值"""
        n = len(self)
        new_vals = []
        for i in range(n):
            if isinstance(cond, Series):
                c = cond[i]
            else:
                c = cond
            if c:
                new_vals.append(other)
            else:
                new_vals.append(self._col[i])
        
        from fluidframe.core.dataframe import _column_from_list_impl
        new_col = _column_from_list_impl(None, self.name, new_vals)
        return Series(self.name, new_col, self._index)

    # =================================================================
    #  映射与转换
    # =================================================================
    
    def map(self, func):
        """映射函数"""
        n = len(self)
        new_vals = []
        for i in range(n):
            new_vals.append(func(self._col[i]))
        
        from fluidframe.core.dataframe import _column_from_list_impl
        new_col = _column_from_list_impl(None, self.name, new_vals)
        return Series(self.name, new_col, self._index)

    def apply(self, func):
        """应用函数"""
        return self.map(func)

    def replace(self, to_replace, value):
        """替换值"""
        n = len(self)
        new_vals = []
        for i in range(n):
            val = self._col[i]
            if val == to_replace:
                new_vals.append(value)
            else:
                new_vals.append(val)
        
        from fluidframe.core.dataframe import _column_from_list_impl
        new_col = _column_from_list_impl(None, self.name, new_vals)
        return Series(self.name, new_col, self._index)

    # =================================================================
    #  索引操作
    # =================================================================
    
    def reindex(self, new_index):
        """重新索引"""
        from fluidframe.core.dataframe import _column_from_list_impl
        
        # 创建映射：旧索引值 -> 新索引值的位置
        old_idx_map = {}
        old_len = len(self)
        for i in range(old_len):
            old_idx_map[self._index[i]] = i
        
        new_len = len(new_index)
        new_vals = []
        for i in range(new_len):
            idx_val = new_index[i]
            if idx_val in old_idx_map:
                new_vals.append(self._col[old_idx_map[idx_val]])
            else:
                new_vals.append(None)  # NaN for missing
        
        new_col = _column_from_list_impl(None, self.name, new_vals)
        return Series(self.name, new_col, new_index)

    def reset_index(self):
        """重置索引"""
        from fluidframe.core.index import RangeIndex
        return Series(self.name, self._col, RangeIndex(len(self)))

    # =================================================================
    #  访问器（使用pxi文件中的访问器类）
    # =================================================================
    
    @property
    def str(self):
        """字符串访问器，返回_StringAccessor实例"""
        cdef int dt = self._col.dtype
        if dt != TYPE_CHAR256 and dt != TYPE_UCS2:
            raise AttributeError("Can only use .str accessor with string values!")
        return _StringAccessor(self)

    @property
    def dt(self):
        """日期时间访问器，返回_DateTimeAccessor实例"""
        cdef int dt = self._col.dtype
        if dt not in (TYPE_DATE, TYPE_DATETIME, TYPE_TIME):
            raise AttributeError("Can only use .dt accessor with datetimelike values!")
        return _DateTimeAccessor(self)

    @property
    def cat(self):
        """分类访问器，返回_CategoryAccessor实例"""
        if self._col.dtype != TYPE_CATEGORY:
            raise AttributeError("Can only use .cat accessor with categorical values!")
        return _CategoryAccessor(self)

    # =================================================================
    #  导出
    # =================================================================
    
    def to_list(self):
        """转换为Python列表"""
        return self.values

    def to_dict(self):
        """转换为字典（索引 -> 值）"""
        result = {}
        n = len(self)
        for i in range(n):
            result[self._index[i]] = self._col[i]
        return result

    def to_pandas(self):
        """转换为pandas Series（如果pandas可用）"""
        try:
            import pandas as pd
            return pd.Series(self.values, index=list(self._index), name=self.name)
        except ImportError:
            raise ImportError("pandas is required for to_pandas()")
