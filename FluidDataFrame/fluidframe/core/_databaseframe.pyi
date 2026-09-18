"""FluidDataFrame 核心模块类型桩 — DataFrame / Series / Index 体系。

本文件是 ``fluidframe.core.dataframe``、``fluidframe.core.series``、
``fluidframe.core.index``、``fluidframe.core.indexers`` 等别名命名空间的
真实类型声明。所有方法均以 pandas 风格 docstring 给出：

- **Parameters 参数**: 每个入参的类型与含义
- **Returns 返回值**: 返回类型与含义
- **Examples 示例**: 可直接运行的示例代码（``>>>`` 为交互式提示符）

示例约定：``df`` 默认指一个小型 DataFrame，例如::

    >>> from fluidframe.core.dataframe import DataFrame
    >>> df = DataFrame({'a': int, 'b': float})
    >>> df.append_row(a=1, b=1.5)
"""

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
    overload,
)

import builtins

from fluidframe._schema_types import (
    TYPE_INT,
    TYPE_DOUBLE,
    TYPE_ASCII,
    TYPE_UCS2,
    TYPE_DATE,
    TYPE_BOOL,
    TYPE_CATEGORY,
    TYPE_DATETIME,
    TYPE_TIME,
    Bool,
    Category,
    Ascii,
    Date,
    DateTime,
    Time,
)
from fluidframe.core._columns import (
    BaseColumn,
    IntColumn,
    DoubleColumn,
    AsciiColumn,
    UCS2Column,
    DateColumn,
    BoolColumn,
    CategoryColumn,
    DateTimeColumn,
    TimeColumn,
    NumericType,
)

# ---------------------------------------------------------------------------
# 常用类型别名
# ---------------------------------------------------------------------------

#: DataFrame/Series 的 dtype 声明：可以是 Python 内置类型或本库类型占位符
DType = Union[
    Type[int], Type[float], Type[bool], Type[str],
    Type[Ascii], Type[Date], Type[DateTime], Type[Time],
    Type[Bool], Type[Category], str,
]

#: Schema：列名 -> dtype 的映射
Schema = Dict[str, DType]

#: 数值聚合中的 dtype 返回：非数值列在聚合时可能被跳过
AggResult = Any


# ===========================================================================
#  索引体系 (Index)
# ===========================================================================

class IndexBase:
    """索引基类：所有索引类型的公共接口。

    索引主要为 ``DataFrame``/``Series`` 的行定位提供支持，并提供
    ``values`` / ``get_loc`` / ``__getitem__`` 等通用能力。

    Attributes
    ----------
    length: int
        索引长度（元素个数）。
    index_type: int
        索引类型编码（RANGE/INT/MULTI/STRING）。

    Examples
    --------
    >>> from fluidframe.core.dataframe import RangeIndex
    >>> idx = RangeIndex(0, 5)
    >>> len(idx)
    5
    >>> idx.values
    [0, 1, 2, 3, 4]
    """

    def get_loc(self, key: object) -> Any:
        """返回标量 ``key`` 对应的整数位置。

        Parameters
        ----------
        key: object
            要定位的标签值。

        Returns
        -------
        Any
            整数位置；多级索引下可能返回一组位置。

        Examples
        --------
        >>> idx = RangeIndex(0, 5)
        >>> idx.get_loc(3)
        3
        """

    def values(self) -> List[Any]:
        """返回索引的底层值列表（Pandas ``Index.values`` 的本库约定）。"""

    def __getitem__(self, key: object) -> Any:
        """按位置或键取索引元素。"""

    def __len__(self) -> int:
        return 0


class IntIndex(IndexBase):
    """整数索引：底层为一列 IntColumn，值可重复。

    Attributes
    ----------
    data: IntColumn
        底层整数列。

    Examples
    --------
    >>> from fluidframe.core._databaseframe import IntIndex, IntColumn
    >>> col = IntColumn(4)
    >>> col.append(10); col.append(20); col.append(20)
    >>> idx = IntIndex(col)
    >>> idx.values
    [10, 20, 20]
    """

    def values(self) -> List[int]:
        """返回索引值（int 列表）。"""

    def __getitem__(self, key: object) -> Any:
        pass

    def get_loc(self, key: object) -> Any:
        pass

    def append(self, val: int) -> None:
        """在索引末尾追加一个值。

        Parameters
        ----------
        val: int
            追加的整数标签。

        Examples
        --------
        >>> idx = IntIndex(IntColumn(1))
        >>> idx.append(5)
        """

    def pop_back(self) -> None:
        """删除并返回末尾的元素（此处为原地删除，无返回值）。"""


class RangeIndex(IndexBase):
    """范围索引：``start`` 起始值、``length`` 长度、``step`` 步长。

    与 pandas RangeIndex 语义一致，``__getitem__``/``get_loc`` 均为 O(1)。

    Examples
    --------
    >>> from fluidframe.core.dataframe import RangeIndex
    >>> idx = RangeIndex(2, 5)          # [2, 3, 4]
    >>> idx.values
    [2, 3, 4]
    >>> idx[0]
    2
    """

    def get_loc(self, key: object) -> Any:
        """返回键的整数位置；键必须落在范围内。

        Parameters
        ----------
        key: object
            目标整数标签。

        Returns
        -------
        Any
            位置（int）。键不在范围内时抛出异常。

        Examples
        --------
        >>> RangeIndex(0, 5).get_loc(3)
        3
        """

    def values(self) -> List[int]:
        """返回从 start 到 start+length-1 的整数列表。"""

    def __getitem__(self, key: object) -> Any:
        pass


class MultiIndex(IndexBase):
    """多级索引：由 ``nlevels`` 层 levels 与 codes 组成。

    Attributes
    ----------
    levels: List[Any]
        每层唯一的标签值列表（list of tuples/strings）。
    codes: List[IntColumn]
        每层的编码列（IntColumn 列表）。
    nlevels: int
        层级数。
    names: Optional[List[str]]
        每层名字。
    """

    def __getitem__(self, key: object) -> Any:
        pass

    def get_loc(self, key: object) -> Any:
        pass

    def values(self) -> List[Tuple[Any, ...]]:
        """返回元组形式的完整层级键列表。"""


class StringIndex(IndexBase):
    """字符串标签索引。

    Attributes
    ----------
    labels: List[str]
        字符串标签列表。

    Examples
    --------
    >>> from fluidframe.core._databaseframe import StringIndex
    >>> idx = StringIndex(['a', 'b', 'c'])
    >>> idx.values
    ['a', 'b', 'c']
    """

    def values(self) -> List[str]:
        """返回字符串标签列表。"""

    def __getitem__(self, key: object) -> Any:
        pass

    def get_loc(self, key: object) -> Any:
        pass


# ===========================================================================
#  索引器 (iloc / loc)
# ===========================================================================

class _ILocIndexer:
    """整数位置索引器（``DataFrame.iloc`` / ``Series.iloc``）。

    支持：

    - 整数位置：``df.iloc[2]`` -> 返回一行 (dict/Series)
    - 切片：``df.iloc[0:3]`` -> 返回子 DataFrame
    - 整数列表：``df.iloc[[0, 2]]`` -> 返回多行
    - 二元元组：``df.iloc[rows, cols]`` -> 行/列混合选择

    Examples
    --------
    >>> df = DataFrame({'a': int, 'b': float})
    >>> df.append_row(a=1, b=1.5)
    >>> df.append_row(a=2, b=2.5)
    >>> df.iloc[0]
    {'a': 1, 'b': 1.5}
    >>> df.iloc[0:2]          # 前两行 -> 子 DataFrame
    >>> df.iloc[[0, 1]]
    """

    def __getitem__(self, key: Any) -> Any:
        pass


class _LocIndexer:
    """标签索引器（``DataFrame.loc`` / ``Series.loc``）。

    支持按索引标签选择行、按条件掩码（布尔 Series/IntColumn/列表）过滤，
    以及 ``df.loc[rows, cols]`` 的行列混合选择。可通过 ``__setitem__`` 赋值。

    Examples
    --------
    >>> df = DataFrame({'a': int, 'b': float})
    >>> df.append_row(a=1, b=1.5); df.append_row(a=2, b=2.5)
    >>> df.loc[1]                 # 标签为 1 的行
    >>> df.loc[df['a'] > 1]       # 条件过滤
    """

    def __setitem__(self, key: Any, values: object) -> None:
        pass

    def __getitem__(self, key: Any) -> Any:
        pass


# ===========================================================================
#  滚动窗口 (rolling)
# ===========================================================================

class _RollingWindow:
    """滚动窗口访问器：由 ``DataFrame.rolling()`` 或 ``Series.rolling()`` 返回。

    DataFrame 模式下各聚合回装整表 (每列一个滚动结果列)，Series 模式下
    回装一个同长度 Series。

    Examples
    --------
    >>> df = DataFrame({'a': int, 'b': int})
    >>> df.append_row(a=1, b=2); df.append_row(a=3, b=4)
    >>> df.append_row(a=5, b=6); df.append_row(a=7, b=8)
    >>> df.rolling(2).mean()     # 每列 2 窗口均值
    >>> df.rolling(2, min_periods=1).sum()
    """

    def mean(self) -> Any:
        """滚动均值。窗口内有效元素数需满足 min_periods，否则为 NaN。

        Parameters
        ----------
        无显式参数（窗口大小、min_periods、center 在 ``rolling()`` 调用时指定）。

        Returns
        -------
        DataFrame 或 Series
            与输入同形状的滚动均值结果。

        Examples
        --------
        >>> df = DataFrame({'a': int})
        >>> df.append_row(a=1); df.append_row(a=3); df.append_row(a=5)
        >>> s = df.rolling(2, min_periods=1).mean()
        """

    def sum(self) -> Any:
        """滚动求和。"""

    def min(self) -> Any:
        """滚动最小值。"""

    def max(self) -> Any:
        """滚动最大值。"""

    def std(self, ddof: bool = True) -> Any:
        """滚动标准差。

        Parameters
        ----------
        ddof: bool, 默认 True
            自由度修正系数（分母 n-ddof）。

        Returns
        -------
        DataFrame 或 Series
            滚动标准差结果。

        Examples
        --------
        >>> df.rolling(3).std(ddof=True)
        """

    def var(self, ddof: bool = True) -> Any:
        """滚动方差。

        Parameters
        ----------
        ddof: bool, 默认 True
            自由度修正系数。

        Returns
        -------
        DataFrame 或 Series
            滚动方差结果。
        """

    def median(self) -> Any:
        """滚动中位数。"""

    def quantile(self, q: float = 0.5) -> Any:
        """滚动分位数。

        Parameters
        ----------
        q: float, 默认 0.5
            分位数（0~1）。

        Returns
        -------
        DataFrame 或 Series
            滚动分位数结果。
        """

    def skew(self) -> Any:
        """滚动偏度。"""

    def kurt(self) -> Any:
        """滚峰度 (excess kurtosis)。"""

    def corr(self, other: Any) -> Any:
        """与另一列的滚动相关系数。

        Parameters
        ----------
        other: Series 或列名或列表
            参与相关的另一列。

        Returns
        -------
        DataFrame 或 Series
            滚动相关系数。
        """

    def cov(self, other: Any) -> Any:
        """与另一列的滚动协方差。"""


class _EWMWindow:
    """指数加权移动窗口访问器：由 ``DataFrame.ewm()`` / ``Series.ewm()`` 返回。

    Examples
    --------
    >>> df = DataFrame({'a': int, 'b': int})
    >>> df.append_row(a=1, b=1); df.append_row(a=2, b=4)
    >>> df.ewm(alpha=0.5).mean()     # 指数加权均值
    """

    def mean(self) -> Any:
        """指数加权均值。

        Returns
        -------
        DataFrame 或 Series
            与输入同形状的 ewm 均值。

        Examples
        --------
        >>> df = DataFrame({'a': int})
        >>> df.append_row(a=1); df.append_row(a=3); df.append_row(a=5)
        >>> df.ewm(alpha=0.5).mean()
        """

    def sum(self) -> Any:
        """指数加权求和。"""

    def var(self, bias: bool = False) -> Any:
        """指数加权方差。

        Parameters
        ----------
        bias: bool, 默认 False
            是否为有偏估计（分母 N）。

        Returns
        -------
        DataFrame 或 Series
            ewm 方差结果。
        """

    def std(self, bias: bool = False) -> Any:
        """指数加权标准差。"""

    def cov(self, other: Any) -> Any:
        """与另一列的 ewm 协方差。"""

    def corr(self, other: Any) -> Any:
        """与另一列的 ewm 相关系数。"""


# ===========================================================================
#  字符串访问器 (Series.str)
# ===========================================================================

class _StringAccessor:
    """Series 字符串访问器：``Series.str`` 返回的对象。

    提供对标 pandas ``str`` 访问器的切片、查找、匹配、大小写、填充、
    拆分拼接等操作。详见各个方法的 Examples。

    Examples
    --------
    >>> df = DataFrame({'s': str})
    >>> df.append_row(s='hello'); df.append_row(s='world')
    >>> df['s'].str.upper()
    """

    def contains(self, pat: object) -> Any:
        """判断每个元素是否包含子串/匹配模式。

        Parameters
        ----------
        pat: object
            查找的子串或模式。

        Returns
        -------
        Series
            布尔 Series 或掩码。

        Examples
        --------
        >>> df['s'].str.contains('ell')
        """

    def startswith(self, prefix: Any) -> Any:
        """判断每个元素是否以指定前缀开头。

        Parameters
        ----------
        prefix: str
            前缀。

        Returns
        -------
        Series 或掩码
            布尔结果。

        Examples
        --------
        >>> df['s'].str.startswith('w')
        """

    def endswith(self, suffix: Any) -> Any:
        """判断每个元素是否以指定后缀结尾。"""

    def find(self, sub: str) -> Any:
        """返回子串首次出现的位置（找不到为 -1）。"""

    def rfind(self, sub: str) -> Any:
        """返回子串最后一次出现的位置。"""

    def index(self, sub: str) -> Any:
        """返回子串首次出现的位置（找不到抛异常）。"""

    def rindex(self, sub: str) -> Any:
        """返回子串最后一次出现的位置（找不到抛异常）。"""

    def count(self, sub: str) -> Any:
        """统计子串在每元素中出现的次数。"""

    def match(self, pattern: str) -> Any:
        """正则：从开头匹配（re.match）。"""

    def fullmatch(self, pattern: str) -> Any:
        """正则：完整匹配（re.fullmatch）。"""

    def extract(self, pattern: str, group: int = 1) -> Any:
        """正则提取捕获组。

        Parameters
        ----------
        pattern: str
            正则表达式。
        group: int, 默认 1
            要提取的分组号。

        Returns
        -------
        Series
            提取结果字符串列。

        Examples
        --------
        >>> df['s'].str.extract(r'(\\w+)')
        """

    def extractall(self, pattern: str) -> Any:
        """正则提取所有匹配分组。"""

    def findall(self, pattern: str) -> Any:
        """正则查找所有匹配，返回列表列。"""

    def isalpha(self) -> Any:
        """判断是否全为字母。"""

    def isnumeric(self) -> Any:
        """判断是否全为数字。"""

    def isalnum(self) -> Any:
        """判断是否全为字母数字。"""

    def isdigit(self) -> Any:
        """判断是否全为十进制数字。"""

    def isdecimal(self) -> Any:
        """判断是否全为十进制数字（Unicode）。"""

    def islower(self) -> Any:
        """判断是否全为小写字母。"""

    def isupper(self) -> Any:
        """判断是否全为大写。"""

    def isspace(self) -> Any:
        """判断是否全为空白。"""

    def upper(self) -> Any:
        """转大写。

        Examples
        --------
        >>> df['s'].str.upper()
        """

    def lower(self) -> Any:
        """转小写。"""

    def title(self) -> Any:
        """标题化 (每个单词首字母大写)。"""

    def capitalize(self) -> Any:
        """首字母大写，其余小写。"""

    def swapcase(self) -> Any:
        """大小写互换。"""

    def casefold(self) -> Any:
        """宽松折叠 (去大小写，适合比较)。"""

    def strip(self, chars: Optional[str] = None) -> Any:
        """移除两端指定字符（默认空白）。"""

    def lstrip(self, chars: Optional[str] = None) -> Any:
        """移除左端指定字符。"""

    def rstrip(self, chars: Optional[str] = None) -> Any:
        """移除右端指定字符。"""

    def center(self, width: int, fillchar: str = ' ') -> Any:
        """居中对齐，两侧用 fillchar 填充到 width。"""

    def ljust(self, width: int, fillchar: str = ' ') -> Any:
        """左对齐，右侧填充到 width。"""

    def rjust(self, width: int, fillchar: str = ' ') -> Any:
        """右对齐，左侧填充到 width。"""

    def pad(self, width: int, side: str = 'left', fillchar: str = ' ') -> Any:
        """按 side 对齐并填充到 width。

        Parameters
        ----------
        width: int
            目标宽度。
        side: {'left', 'right', 'both'}, 默认 'left'
            填充方向。
        fillchar: str, 默认 ' '
            填充字符。
        """

    def zfill(self, width: int) -> Any:
        """左侧补零到 width。"""

    def replace(self, old: str, new: str, count: int = -1) -> Any:
        """字符串替换。

        Parameters
        ----------
        old: str
            被替换子串。
        new: str
            替换内容。
        count: int, 默认 -1
            最多替换次数（-1 表示全部）。
        """

    def split(self, sep: Optional[str] = None, maxsplit: int = -1) -> Any:
        """按分隔符分裂为列表列。"""

    def rsplit(self, sep: Optional[str] = None, maxsplit: int = -1) -> Any:
        """从右端按分隔符分裂为列表列。"""

    def partition(self, sep: str) -> Any:
        """按首个分隔符切为 3 段 (head, sep, tail)。"""

    def rpartition(self, sep: str) -> Any:
        """按最后一个分隔符切为 3 段。"""

    def get(self, index: int, default: str = '') -> Any:
        """按位置取拆分结果中的一段。"""

    def slice(self, start: int = 0, stop: int = -1, step: int = 1) -> Any:
        """按 Python 切片语义截取每元素的子串。

        Parameters
        ----------
        start: int, 默认 0
            起点。
        stop: int, 默认 -1
            终点（-1 表示到末尾）。
        step: int, 默认 1
            步长。
        """

    def len(self) -> Any:
        """返回每元素长度。"""

    def __getitem__(self, key: Any) -> Any:
        pass

    def cat(self, other: Any, sep: str = '') -> Any:
        """拼接：与 other (Series/列表) 按行拼接，或自身逐元素拼接。

        Parameters
        ----------
        other: str / Series / 列表
            拼接的对方。
        sep: str, 默认 ''
            分隔符。
        """

    def prefix(self, pre: str) -> Any:
        """为每个元素加前缀。"""

    def suffix(self, suf: str) -> Any:
        """为每个元素加后缀。"""

    def repeat(self, repeats: int) -> Any:
        """重复每个元素 repeats 次。"""


# ===========================================================================
#  日期时间访问器 (Series.dt)
# ===========================================================================

class _DateTimeAccessor:
    """Series 日期/时间访问器：``Series.dt`` 返回的对象。

    提供 year/month/day/weekday/strftime 等日期字段提取。

    Examples
    --------
    >>> df = DataFrame({'d': Date})
    >>> df.append_row(d='2024-03-15')
    >>> df['d'].dt.year
    """

    def weekday(self) -> Any:
        """周几 (0=周一)。

        Examples
        --------
        >>> df['d'].dt.weekday
        """

    def dayofweek(self) -> Any:
        """周几 (0=周一)，与 weekday 等价。"""

    def dayofyear(self) -> Any:
        """年内第几天 (1~366)。"""

    def weekofyear(self) -> Any:
        """ISO 周序号 (1~53)。"""

    def strftime(self, fmt: str) -> Any:
        """按格式字符串格式化每个日期。

        Parameters
        ----------
        fmt: str
            strftime 格式，支持 %Y/%m/%d 等。

        Returns
        -------
        Series
            格式化后的字符串列。

        Examples
        --------
        >>> df['d'].dt.strftime('%Y-%m-%d')
        """

    # -- 属性式字段提取（.dt.year / .dt.month / ...，返回 Series） ----

    @property
    def year(self) -> Series:
        """年份。"""

    @property
    def month(self) -> Series:
        """月份 (1-12)。"""

    @property
    def day(self) -> Series:
        """日 (1-31)。"""

    @property
    def hour(self) -> Series:
        """小时 (0-23)，仅 DateTime/Time 列。"""

    @property
    def minute(self) -> Series:
        """分钟 (0-59)。"""

    @property
    def second(self) -> Series:
        """秒 (0-59)。"""

    @property
    def quarter(self) -> Series:
        """季度 (1-4)。"""

    @property
    def week_of_year(self) -> Series:
        """ISO 周序号。"""

    @property
    def day_of_week(self) -> Series:
        """周几。"""

    @property
    def day_of_year(self) -> Series:
        """年内第几天。"""

    @property
    def days_in_month(self) -> Series:
        """当月天数。"""

    @property
    def is_leap_year(self) -> Series:
        """是否闰年。"""

    @property
    def is_weekend(self) -> Series:
        """是否周末。"""

    @property
    def is_month_start(self) -> Series:
        """是否月初。"""

    @property
    def is_month_end(self) -> Series:
        """是否月末。"""

    @property
    def is_year_start(self) -> Series:
        """是否年初。"""

    @property
    def is_year_end(self) -> Series:
        """是否年末。"""

    @property
    def is_quarter_start(self) -> Series:
        """是否季初。"""

    @property
    def is_quarter_end(self) -> Series:
        """是否季末。"""

    @property
    def day_name(self) -> Series:
        """星期名（如 'Monday'）。"""

    @property
    def month_name(self) -> Series:
        """月份名（如 'January'）。"""

    def __getattr__(self, name: object) -> Any:
        pass


# ===========================================================================
#  分类访问器 (Series.cat)
# ===========================================================================

class _CategoryAccessor:
    """Series 分类访问器：``Series.cat`` 返回的对象。

    Examples
    --------
    >>> df = DataFrame({'g': Category})
    >>> df.append_row(g='low'); df.append_row(g='high')
    >>> df['g'].cat.codes
    """

    def codes(self) -> Any:
        """返回底层编码 (int 列)。

        Examples
        --------
        >>> df['g'].cat.codes
        """

    def categories(self) -> Any:
        """返回类别列表。"""

    def rename_categories(self, mapper: Any, inplace: bool = False) -> Any:
        """重命名类别。

        Parameters
        ----------
        mapper: dict / 列表 / callable
            类别映射。
        inplace: bool, 默认 False
            是否原地修改。
        """

    def add_categories(self, new_categories: List[Any], inplace: bool = False) -> Any:
        """追加新类别。"""

    def remove_categories(self, removals: List[Any], inplace: bool = False) -> Any:
        """移除类别（被移除值变为缺失或保留原值，视实现而定）。"""

    def set_categories(self, new_categories: List[Any], inplace: bool = False) -> Any:
        """整体替换类别集合。"""

    def as_ordered(self, inplace: bool = False) -> Any:
        """设为有序分类（按类别顺序）。"""

    def as_unordered(self, inplace: bool = False) -> Any:
        """设为无序分类。"""


# ===========================================================================
#  Series
# ===========================================================================

class Series:
    """一维带标签数据结构，对标 pandas.Series。

    本库约定：Series 是单列的轻量代理，内部持有 ``BaseColumn`` 与
    一个索引。字符串列在非 ASCII 字符时使用 UCS2 存储（dtype 为
    TYPE_UCS2），纯 ASCII 时使用定长 AsciiColumn。

    Parameters
    ----------
    name: str
        列名。
    col: BaseColumn
        底层列对象。
    index: IndexBase
        索引。

    Examples
    --------
    >>> from fluidframe.core.series import Series
    >>> from fluidframe.core.intcolumn import IntColumn
    >>> from fluidframe.core.dataframe import RangeIndex
    >>> col = IntColumn(3)
    >>> col.append(10); col.append(20); col.append(30)
    >>> s = Series('a', col, RangeIndex(0, 3))
    >>> s.sum()
    60
    """

    def __init__(self, name: str, col: BaseColumn, index: IndexBase) -> None: ...

    # ------------------------------------------------------------------
    #  属性 (Properties)
    # ------------------------------------------------------------------

    @property
    def dtype(self) -> int:
        """返回底层列类型编码 (见 ``fluidframe._schema_types.TYPE_*``)。

        Returns
        -------
        int
            类型编码。

        Examples
        --------
        >>> s.dtype
        0   # TYPE_INT
        """

    @property
    def is_2d(self) -> bool:
        """是否 2D 布局（多子列）。"""

    @property
    def length(self) -> int:
        """元素个数。"""

    @property
    def index(self) -> IndexBase:
        """返回绑定索引；赋值可设置新索引。

        Returns
        -------
        IndexBase
            当前索引。
        """

    @index.setter
    def index(self, value: IndexBase) -> None: ...

    @property
    def values(self) -> List[Any]:
        """返回底层数据的 Python 列表（2D 列返回行键元组，与
        ``value_counts`` 键一致）。

        Returns
        -------
        List[Any]
            元素列表。

        Examples
        --------
        >>> s.values
        [10, 20, 30]
        """

    @property
    def shape(self) -> Tuple[int]:
        """形状，对标 pandas 单列 ``(len,)``。"""

    # ------------------------------------------------------------------
    #  迭代 / 包含
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Any]:
        """逐元素迭代。

        Examples
        --------
        >>> [v * 2 for v in s]
        [20, 40, 60]
        """

    def __contains__(self, val: Any) -> bool:
        """``val in s``：判断元素是否在列中。"""

    def __getattr__(self, name: str) -> Any:
        pass

    def __len__(self) -> int:
        """元素个数。"""

    def __repr__(self) -> str: ...

    def __str__(self) -> str: ...

    def __getitem__(self, key: Any) -> Any:
        """按位置/标签/切片/掩码取元素，返回标量或 Series。

        Examples
        --------
        >>> s[0]
        10
        >>> s[0:2]        # 切片 -> 子 Series
        >>> s[s > 15]     # 掩码过滤
        """

    def __setitem__(self, key: Any, value: Any) -> None:
        """按位置/标签/掩码赋值。

        Parameters
        ----------
        key: int / slice / Series(掩码) / 标签
            赋值位置。
        value: Any
            标量值（或等长序列）。
        """

    # ------------------------------------------------------------------
    #  运算符
    # ------------------------------------------------------------------

    def __add__(self, other: Any) -> "Series": ...
    def __sub__(self, other: Any) -> "Series": ...
    def __mul__(self, other: Any) -> "Series": ...
    def __truediv__(self, other: Any) -> "Series": ...
    def __floordiv__(self, other: Any) -> "Series": ...
    def __mod__(self, other: Any) -> "Series": ...
    def __neg__(self) -> "Series": ...
    def __pos__(self) -> "Series": ...
    def __abs__(self) -> "Series": ...
    def __eq__(self, other: Any) -> "Series": ...  # type: ignore[override]
    def __ne__(self, other: Any) -> "Series": ...  # type: ignore[override]
    def __lt__(self, other: Any) -> "Series": ...  # type: ignore[override]
    def __le__(self, other: Any) -> "Series": ...  # type: ignore[override]
    def __gt__(self, other: Any) -> "Series": ...  # type: ignore[override]
    def __ge__(self, other: Any) -> "Series": ...  # type: ignore[override]
    def __and__(self, other: Any) -> "Series": ...
    def __or__(self, other: Any) -> "Series": ...
    def __xor__(self, other: Any) -> "Series": ...
    def __invert__(self) -> "Series": ...

    # ------------------------------------------------------------------
    #  算术方法
    # ------------------------------------------------------------------

    def add(self, other: Any) -> "Series":
        """逐元素加。

        Parameters
        ----------
        other: 标量 / Series / 列表
            加数。

        Returns
        -------
        Series
            结果列。

        Examples
        --------
        >>> s.add(1)
        """

    def sub(self, other: Any) -> "Series":
        """逐元素减。"""

    def mul(self, other: Any) -> "Series":
        """逐元素乘。"""

    def div(self, other: Any) -> "Series":
        """逐元素除。"""

    # ------------------------------------------------------------------
    #  统计
    # ------------------------------------------------------------------

    def sum(self) -> Any:
        """求和（非数值列可能不可用或按实现跳过）。

        Examples
        --------
        >>> s.sum()
        60
        """

    def mean(self) -> float:
        """均值。非数值列抛异常。

        Examples
        --------
        >>> s.mean()
        20.0
        """

    def std(self) -> float:
        """标准差 (ddof=1)。"""

    def var(self) -> float:
        """方差 (ddof=1)。"""

    def min(self) -> Any:
        """最小值。"""

    def max(self) -> Any:
        """最大值。"""

    def median(self) -> float:
        """中位数。"""

    def prod(self) -> Any:
        """乘积。"""

    def count(self) -> int:
        """非 NaN 元素个数。"""

    def nunique(self) -> int:
        """去重后元素个数。"""

    def unique(self) -> List[Any]:
        """去重值列表（保持首次出现顺序）。

        Returns
        -------
        List[Any]
            唯一值列表。
        """

    def value_counts(self) -> "Series":
        """统计各值出现频次，按频次降序返回 Series。

        Parameters
        ----------
        无参数。

        Returns
        -------
        Series
            索引为原值，值列名为 'count' 的频次 Series。

        Examples
        --------
        >>> df = DataFrame({'a': int})
        >>> df.append_row(a=1); df.append_row(a=2); df.append_row(a=2)
        >>> df['a'].value_counts()
        """

    def argmax(self) -> int:
        """最大值位置。"""

    def argmin(self) -> int:
        """最小值位置。"""

    def quantile(self, q: float) -> float:
        """分位数。

        Parameters
        ----------
        q: float
            0~1 分位数。

        Returns
        -------
        float
            分位值。
        """

    def skew(self) -> float:
        """偏度。"""

    def kurt(self) -> float:
        """峰度。"""

    def mad(self) -> float:
        """平均绝对偏差。"""

    def sem(self) -> float:
        """标准误 (std/sqrt(n))。"""

    def mode(self) -> "Series":
        """众数（可能有多个）。"""

    # ------------------------------------------------------------------
    #  变换
    # ------------------------------------------------------------------

    def fillna(self, value: Any) -> "Series":
        """用标量填充缺失值。

        Parameters
        ----------
        value: Any
            填充值。

        Returns
        -------
        Series
            新 Series。

        Examples
        --------
        >>> s2 = s.fillna(0)
        """

    def ffill(self) -> "Series":
        """前向填充缺失值。"""

    def bfill(self) -> "Series":
        """后向填充缺失值。"""

    def astype(self, dtype: DType) -> "Series":
        """类型转换。

        Parameters
        ----------
        dtype: DType
            目标类型（int/float/str/Ascii/Date/DateTime 等）。

        Returns
        -------
        Series
            新类型的 Series。

        Examples
        --------
        >>> s.astype(float)
        """

    def clip(self, lower: Any = None, upper: Any = None) -> "Series":
        """截断：所有值限制在 [lower, upper]。

        Parameters
        ----------
        lower: 标量, optional
            下界。
        upper: 标量, optional
            上界。
        """

    def round(self, decimals: int) -> "Series":
        """四舍五入。

        Parameters
        ----------
        decimals: int
            小数位数。
        """

    def diff(self, periods: int = 1) -> "Series":
        """逐行差分。

        Parameters
        ----------
        periods: int, 默认 1
            差分阶数。
        """

    def pct_change(self, periods: int = 1) -> "Series":
        """百分比变化 (当前/上一行 - 1)。"""

    def cumsum(self) -> "Series":
        """累计求和。"""

    def cumprod(self) -> "Series":
        """累计乘积。"""

    def cummax(self) -> "Series":
        """累计最大值。"""

    def cummin(self) -> "Series":
        """累计最小值。"""

    def sort_values(self, ascending: bool = True) -> "Series":
        """按值排序，返回新 Series。

        Parameters
        ----------
        ascending: bool, 默认 True
            升序。
        """

    def head(self, n: int = 5) -> "Series":
        """前 n 行。"""

    def tail(self, n: int = 5) -> "Series":
        """后 n 行。"""

    def nsmallest(self, n: int) -> "Series":
        """最小的 n 个值。"""

    def nlargest(self, n: int) -> "Series":
        """最大的 n 个值。"""

    def idxmax(self) -> Any:
        """最大值对应索引标签。"""

    def idxmin(self) -> Any:
        """最小值对应索引标签。"""

    def isna(self) -> "Series":
        """缺失掩码：double=NaN、字符串=空串、类别=''，其余恒 False。

        Returns
        -------
        Series
            布尔 IntColumn 掩码 (0/1)。
        """

    def notna(self) -> "Series":
        """非缺失掩码 (isna 取反)。"""

    def isin(self, values: Iterable[Any]) -> "Series":
        """检查元素是否在给定集合中。

        Parameters
        ----------
        values: iterable
            值集合。
        """

    def between(self, low: Any, high: Any) -> "Series":
        """``low <= x <= high`` 掩码。"""

    def all(self) -> bool:
        """全部为真 (非零/非空)。"""

    def any(self) -> bool:
        """任一为真。"""

    def where(self, cond: Any, other: Any) -> "Series":
        """条件成立保留原值，否则替换为 other。

        Parameters
        ----------
        cond: 同长布尔 Series / IntColumn 掩码
            条件。
        other: 标量或同长序列
            替代值。
        """

    def mask(self, cond: Any, other: Any) -> "Series":
        """where 的反面：条件成立处替换为 other。"""

    def map(self, func: Callable[[Any], Any]) -> "Series":
        """逐元素映射函数。

        Parameters
        ----------
        func: callable
            元素 -> 新元素 的映射函数。

        Returns
        -------
        Series
            结果列。

        Examples
        --------
        >>> s.map(lambda x: x * 2)
        """

    def factorize(self) -> Any:
        """因子化：返回 (codes, uniques)。

        Returns
        -------
        (codes, uniques)
            codes: int 编码列表；uniques: 唯一值列表（首现顺序）。
        """

    def apply(self, func: Callable[[Any], Any]) -> "Series":
        """逐元素应用函数（与 map 类似）。"""

    def replace(self, to_replace: Any, value: Any) -> "Series":
        """值替换。

        Parameters
        ----------
        to_replace: 标量 / 列表 / dict
            被替换的值。
        value: 标量 / 同长序列
            替换值。
        """

    def reindex(self, new_index: Any) -> "Series":
        """按新索引重排（缺失位置填充 NaN/缺失）。"""

    def reset_index(self) -> "Series":
        """重置索引为 RangeIndex。"""

    # ------------------------------------------------------------------
    #  访问器
    # ------------------------------------------------------------------

    @property
    def str(self) -> _StringAccessor:
        """字符串访问器（仅字符串列可用）。

        Examples
        --------
        >>> df['s'].str.len()
        """

    @property
    def dt(self) -> _DateTimeAccessor:
        """日期/时间访问器（仅日期类型列可用）。

        Examples
        --------
        >>> df['d'].dt.year
        """

    @property
    def cat(self) -> _CategoryAccessor:
        """分类访问器（仅类别列可用）。

        Examples
        --------
        >>> df['g'].cat.codes
        """

    # ------------------------------------------------------------------
    #  窗口
    # ------------------------------------------------------------------

    def rolling(self, window: int, min_periods: Optional[int] = None, center: bool = False, win_type: Optional[builtins.str] = None) -> _RollingWindow:
        """滚动窗口。

        Parameters
        ----------
        window: int
            窗口大小。
        min_periods: int, optional
            最少有效元素数。
        center: bool, 默认 False
            是否居中。
        win_type: str, optional
            加权窗口类型（'boxcar'/'triang'）。

        Returns
        -------
        _RollingWindow
            滚动访问器。

        Examples
        --------
        >>> s.rolling(2).mean()
        """

    def expanding(self, min_periods: int = 1, axis: int = 0) -> _RollingWindow:
        """扩展窗口（等价于 window=n 的滚动窗口）。

        Parameters
        ----------
        min_periods: int, 默认 1
            最少有效元素数。
        axis: int, 默认 0
            轴（仅支持 0）。
        """

    def ewm(self, com: Optional[float] = None, span: Optional[float] = None, halflife: Optional[float] = None, alpha: Optional[float] = None, min_periods: int = 0, adjust: bool = True, ignore_na: bool = False) -> _EWMWindow:
        """指数加权窗口。

        Parameters
        ----------
        com: float, optional
            质心。
        span: float, optional
            跨度。
        halflife: float, optional
            半衰期。
        alpha: float, optional
            平滑系数 (直接指定)。
        min_periods: int, 默认 0
            最少有效元素数。
        adjust: bool, 默认 True
            是否调整。
        ignore_na: bool, 默认 False
            是否忽略 NaN。
        """

    # ------------------------------------------------------------------
    #  距离 / 相似度
    # ------------------------------------------------------------------

    def euclidean_dist(self, other: Any) -> Any:
        """与 another 的欧氏距离。"""

    def manhattan_dist(self, other: Any) -> Any:
        """与 another 的曼哈顿距离。"""

    def cosine_similarity(self, other: Any) -> Any:
        """与 another 的余弦相似度。"""

    def cosine_dist(self, other: Any) -> Any:
        """与 another 的余弦距离。"""

    def jaccard_sim(self, other: Any) -> Any:
        """与 another 的 Jaccard 相似度。"""

    def hamming_dist(self, other: Any) -> Any:
        """与 another 的汉明距离。"""

    # ------------------------------------------------------------------
    #  导出
    # ------------------------------------------------------------------

    def to_list(self) -> List[Any]:
        """转为 Python 列表。

        Examples
        --------
        >>> s.to_list()
        [10, 20, 30]
        """

    def tolist(self) -> List[Any]:
        """to_list 的别名。"""

    def to_numpy(self) -> Any:
        """转为 numpy 数组（若已安装）。"""

    def to_dict(self) -> Dict[Any, Any]:
        """转为 dict：{标签: 值} 或 {位置: 值}（视索引而定）。"""

    def to_pandas(self) -> Any:
        """转为 pandas.Series（若已安装 pandas）。"""

    def to_frame(self, name: Optional[builtins.str] = None) -> "DataFrame":
        """转为一列 DataFrame。

        Parameters
        ----------
        name: str, optional
            新列名。
        """

    def shift(self, periods: int = 1, fill_value: Any = None) -> "Series":
        """移位。

        Parameters
        ----------
        periods: int, 默认 1
            移动位数。
        fill_value: 标量, optional
            空位填充值。
        """

    def rank(self, method: builtins.str = 'average', ascending: bool = True, na_option: builtins.str = 'keep') -> "Series":
        """排名。

        Parameters
        ----------
        method: {'average','min','max','first','dense'}, 默认 'average'
            排名方法。
        ascending: bool, 默认 True
            升序。
        na_option: {'keep','top','bottom'}, 默认 'keep'
            缺失值处理。
        """

    def describe(self, percentiles: Optional[Any] = None) -> "DataFrame":
        """描述性统计汇总 (count/mean/std/min/25%/50%/75%/max)。

        Parameters
        ----------
        percentiles: 列表, optional
            自定义分位数。

        Returns
        -------
        DataFrame
            统计汇总表。
        """


# ===========================================================================
#  DataFrame
# ===========================================================================

class DataFrame:
    """二维表格数据结构，对标 pandas.DataFrame。

    本库采用 **列主序** 存储：数值列 (Int/Double) 支持一维与 2D（多子列）
    布局，字符串列区分 ASCII 定长 (AsciiColumn) 与 UTF-8 变长 (UCS2Column)。

    Parameters
    ----------
    schema: Dict[str, DType], optional
        列名到类型的映射。列类型支持 Python 内置类型
        (``int``/``float``/``bool``/``str``) 或类型占位符
        (``Ascii``/``Date``/``DateTime``/``Time``/``Bool``/``Category``)
        以及字符串简称 (``'int'``/``'str'``/``'date'``/``'category'`` 等)。

    Returns
    -------
    DataFrame
        空 DataFrame（未填充行）。

    Examples
    --------
    >>> from fluidframe.core.dataframe import DataFrame
    >>> from fluidframe import Ascii, Date
    >>> df = DataFrame({'id': int, 'score': float, 'name': str, 'tag': Ascii, 'dt': Date})
    >>> df.append_row(id=1, score=90.5, name='Alice', tag='A', dt='2024-01-01')
    >>> df.append_row(id=2, score=85.0, name='Bob', tag='B', dt='2024-01-02')
    """

    def __init__(self, schema: Optional[Dict[str, DType]] = None) -> None:
        """创建 DataFrame。

        Parameters
        ----------
        schema: Optional[Dict[str, DType]], 默认 None
            列名 -> 类型 映射。

        Examples
        --------
        >>> df = DataFrame({'a': int, 'b': float})
        >>> df.append_row(a=1, b=1.5)
        """

    # ------------------------------------------------------------------
    #  内部 (为对齐 __cinit__ 签名)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """行数。

        Returns
        -------
        int
            行数。

        Examples
        --------
        >>> len(df)
        2
        """

    def __eq__(self, other: object) -> bool:
        """逐值比较两个 DataFrame 是否完全相等（含 dtype 与索引）。

        Parameters
        ----------
        other: object
            另一 DataFrame。

        Returns
        -------
        bool
            是否相等。
        """

    def reserve_with_capacity(self, capacity: int) -> None:
        """预分配固定容量以加速后续逐行填充（add_row）。

        Parameters
        ----------
        capacity: int
            目标行容量（严格正）。

        Examples
        --------
        >>> df2 = DataFrame({'a': int})
        >>> df2.reserve_with_capacity(1024)
        """

    def add_row(self, *args: Any, **kwargs: Any) -> None:
        """追加一行。

        支持两种调用方式：

        - 位置参数（保持 schema 列序）: ``df.add_row(1, 1.5)``
        - 关键字参数: ``df.add_row(a=1, b=1.5)``

        Parameters
        ----------
        *args
            与 schema 列序一致的值。
        **kwargs
            列名 -> 值。

        Returns
        -------
        None

        Examples
        --------
        >>> df = DataFrame({'a': int, 'b': float})
        >>> df.add_row(a=10, b=2.5)          # 关键字形式
        >>> df.add_row(20, 5.5)              # 位置形式
        """

    def add_rows_from_list(self, rows: List[Any]) -> None:
        """批量追加多行。

        Parameters
        ----------
        rows: List[dict 或 list]
            每行一个 dict（列名->值）或按列序的 list。

        Examples
        --------
        >>> df = DataFrame({'a': int, 'b': float})
        >>> df.add_rows_from_list([{'a': 1, 'b': 1.5}, {'a': 2, 'b': 2.5}])
        """

    # ------------------------------------------------------------------
    #  索引 / 切片
    # ------------------------------------------------------------------

    @property
    def loc(self) -> _LocIndexer:
        """标签索引器。

        Examples
        --------
        >>> df.loc[0]                  # 标签为 0 的行
        >>> df.loc[0:2]                # 位置切片
        """

    @property
    def iloc(self) -> _ILocIndexer:
        """整数位置索引器。

        Examples
        --------
        >>> df.iloc[0]
        >>> df.iloc[0:2]
        """

    def __getattr__(self, name: str) -> Any:
        pass

    def __dir__(self) -> List[str]:
        pass

    def __getitem__(self, name: object) -> Any:
        """按列名取列（返回 Series），或传入布尔掩码过滤行。

        Parameters
        ----------
        name: str 或 布尔 Series/掩码
            列名或行掩码。

        Returns
        -------
        Series
            列数据；若为掩码则返回过滤后的 DataFrame 的列。

        Examples
        --------
        >>> df['a']
        >>> df['a'] > 1
        """

    def get_series(self, name: str) -> Series:
        """按列名取列（等价于 ``df[name]``，返回 Series）。

        Parameters
        ----------
        name: str
            列名。

        Returns
        -------
        Series
            列代理。

        Examples
        --------
        >>> df.get_series('a')
        """

    def __iter__(self) -> Iterator[str]:
        """迭代列名。

        Examples
        --------
        >>> [c for c in df]
        ['a', 'b']
        """

    def __contains__(self, name: str) -> bool:
        """``name in df``：判断列是否存在。"""

    def __setitem__(self, name: object, value: object) -> None:
        """按列名赋值（新增或覆盖列）。

        Parameters
        ----------
        name: str
            列名。
        value: 列表 / Series / 标量
            等长值或标量广播。
        """

    def __delitem__(self, name: object) -> None:
        """删除列。"""

    # ------------------------------------------------------------------
    #  属性
    # ------------------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, int]:
        """形状 (行, 列)。

        Examples
        --------
        >>> df.shape
        (2, 2)
        """

    @property
    def values(self) -> List[Any]:
        """导出为二维 List of Lists。"""

    @property
    def size(self) -> int:
        """元素总数：行 × 列。"""

    @property
    def ndim(self) -> int:
        """始终为 2（DataFrame 语义）。"""

    def get(self, key: str, default: Any = None) -> Any:
        """按列名取值；不存在返回 default。

        Parameters
        ----------
        key: str
            列名。
        default: Any, optional
            默认值。
        """

    @property
    def columns(self) -> List[str]:
        """列名列表。

        Returns
        -------
        List[str]
            列名。

        Examples
        --------
        >>> df.columns
        ['a', 'b']
        """

    @property
    def col_names(self) -> List[str]:
        """列名列表（与 columns 等价，本库约定）。"""

    @property
    def dtypes(self) -> List[str]:
        """各列类型名列表（如 'int'/'double'/'ascii'/'ucs2'/'date' 等）。"""

    # ------------------------------------------------------------------
    #  分组 / 表达式
    # ------------------------------------------------------------------

    def groupby(self, by_cols: Any) -> _DataFrameGroupByAccessor:
        """按一列或多列分组，返回 GroupBy 访问器。

        Parameters
        ----------
        by_cols: str 或 List[str]
            分组所依据的列名。

        Returns
        -------
        _DataFrameGroupByAccessor
            分组访问器，可调用 sum/mean/count/agg 等。

        Examples
        --------
        >>> df = DataFrame({'k': int, 'v': int})
        >>> df.add_rows_from_list([{'k': 1, 'v': 10}, {'k': 1, 'v': 20}, {'k': 2, 'v': 30}])
        >>> df.groupby('k').sum()
        """

    def eval(self, expr: str, inplace: bool = False) -> Any:
        """支持简单表达式求值（对标 pandas eval 的轻量子集）。

        Parameters
        ----------
        expr: str
            表达式。
        inplace: bool, 默认 False
            是否原地。
        """

    def query(self, expr: str) -> "DataFrame":
        """按表达式过滤行。

        Parameters
        ----------
        expr: str
            布尔表达式。

        Returns
        -------
        DataFrame
            满足条件的行。
        """

    # ------------------------------------------------------------------
    #  行操作
    # ------------------------------------------------------------------

    def append_row(self, **kwargs: Any) -> None:
        """追加一行（add_row 的关键字别名）。

        Parameters
        ----------
        **kwargs
            列名 -> 值。
        """

    def append(self, row: Any, ignore_index: bool = False) -> None:
        """追加一行（dict 或按列序 list）。"""

    def copy(self) -> "DataFrame":
        """深拷贝。

        Returns
        -------
        DataFrame
            副本（数据互不影响）。

        Examples
        --------
        >>> df2 = df.copy()
        >>> df2.add_row(a=99, b=9.9)     # 不影响 df
        >>> len(df)
        2
        """

    def pop_front(self) -> None:
        """删除首行（原地）。"""

    def pop_back(self) -> None:
        """删除末行（原地）。"""

    def rename(self, mapper: Any = None, columns: Any = None, index: Any = None, inplace: bool = False) -> Optional["DataFrame"]:
        """重命名列名/索引。

        Parameters
        ----------
        mapper: dict 或 callable, optional
            重命名映射。
        columns: dict 或 callable, optional
            列名映射。
        index: dict 或 callable, optional
            索引映射。
        inplace: bool, 默认 False
            是否原地。

        Returns
        -------
        Optional[DataFrame]
            非 inplace 时返回新 DataFrame。
        """

    def reset_index(self) -> "DataFrame":
        """重置索引为 RangeIndex。

        Returns
        -------
        DataFrame
            重置后的表。
        """

    def set_index(self, col_name: str, drop: bool = True) -> "DataFrame":
        """将指定列设为索引。

        Parameters
        ----------
        col_name: str
            作索引的列名。
        drop: bool, 默认 True
            是否从数据列中移除。
        """

    def get_dummies(self, col_names: Any = None, prefix: str = '', drop_first: bool = False) -> "DataFrame":
        """分类列独热编码。

        Parameters
        ----------
        col_names: 列表, optional
            参与编码的列。
        prefix: str, 默认 ''
            新列前缀。
        drop_first: bool, 默认 False
            是否丢弃首列（防共线性）。
        """

    def reindex(self, new_index: List[Any]) -> "DataFrame":
        """按新标签重排行。

        Parameters
        ----------
        new_index: List
            新索引标签序列。
        """

    def insert_row(self, pos: int, **kwargs: Any) -> None:
        """在指定位置插入一行。

        Parameters
        ----------
        pos: int
            插入位置。
        **kwargs
            列名 -> 值。
        """

    def drop(self, labels: Any = None, axis: int = 0, columns: Any = None, index: Any = None) -> "DataFrame":
        """删除行或列。

        Parameters
        ----------
        labels: 单值或列表, optional
            标签。
        axis: {0, 'index', 1, 'columns'}, 默认 0
            删除方向。
        columns: optional
            列名便捷参数。
        index: optional
            行标签便捷参数。
        """

    def drop_with_index(self, drop_indices: List[int], inplace: bool = False) -> Optional["DataFrame"]:
        """按行位置批量删除。"""

    def drop_index(self, labels: object, inplace: bool = False) -> Optional["DataFrame"]:
        """按索引标签删除行。"""

    def drop_raw(self, row_idx: int) -> None:
        """按行位置原地删除一行。

        Parameters
        ----------
        row_idx: int
            行位置。
        """

    def insert(self, loc: int, column: str, value: object, allow_duplicates: bool = False) -> None:
        """在指定列位置插入列。

        Parameters
        ----------
        loc: int
            插入位置。
        column: str
            列名。
        value: 标量 / 列表
            列值。
        allow_duplicates: bool, 默认 False
            是否允许重复列名。
        """

    def insert_column(self, col_name: str, col_data: object, pos: int = -1) -> None:
        """插入一列（数据为列对象 / 列表 / 标量）。"""

    def duplicate_column(self, src_name: str, new_name: Optional[str] = None) -> None:
        """复制列。"""

    def duplicate_rows(self, indices: Optional[List[int]] = None) -> "DataFrame":
        """复制指定行（默认全部行翻倍）。

        Parameters
        ----------
        indices: List[int], optional
            要复制（重复）的行位置。
        """

    # ------------------------------------------------------------------
    #  聚合统计
    # ------------------------------------------------------------------

    def mean(self, col_names: Optional[List[str]] = None) -> Any:
        """每列均值（仅数值列）。

        Parameters
        ----------
        col_names: List[str], optional
            限定计算的列。

        Returns
        -------
        单行 DataFrame 或 dict
            各列均值。

        Examples
        --------
        >>> df = DataFrame({'a': int, 'b': int})
        >>> df.add_row(a=1, b=2); df.add_row(a=3, b=4)
        >>> df.mean()
        """

    def sum(self, col_names: Optional[List[str]] = None) -> Any:
        """每列求和。"""

    def min(self, col_names: Optional[List[str]] = None) -> Any:
        """每列最小值。"""

    def max(self, col_names: Optional[List[str]] = None) -> Any:
        """每列最大值。"""

    def std(self, col_names: Optional[List[str]] = None) -> Any:
        """每列标准差 (ddof=1)。"""

    def var(self, col_names: Optional[List[str]] = None) -> Any:
        """每列方差 (ddof=1)。"""

    def median(self, col_names: Optional[List[str]] = None) -> Any:
        """每列中位数。"""

    def quantile(self, q: float = 0.5, col_names: Optional[List[str]] = None) -> Any:
        """每列分位数。

        Parameters
        ----------
        q: float, 默认 0.5
            分位数。
        col_names: List[str], optional
            限定列。
        """

    def mode(self, col_names: Optional[List[str]] = None) -> Any:
        """每列众数。"""

    def prod(self, col_names: Optional[List[str]] = None) -> Any:
        """每列乘积。"""

    def sem(self, col_names: Optional[List[str]] = None) -> Any:
        """每列标准误。"""

    def mad(self, col_names: Optional[List[str]] = None) -> Any:
        """每列平均绝对偏差。"""

    # --- withcol 变体：返回带列名的 DataFrame ---------------------------

    def mean_withcol(self, col_names: Optional[List[str]] = None) -> "DataFrame":
        """每列均值（返回单行 DataFrame，列名保留）。"""

    def sum_withcol(self, col_names: Optional[List[str]] = None) -> "DataFrame":
        """每列求和（返回单行 DataFrame）。"""

    def min_withcol(self, col_names: Optional[List[str]] = None) -> "DataFrame":
        """每列最小值（返回单行 DataFrame）。"""

    def max_withcol(self, col_names: Optional[List[str]] = None) -> "DataFrame":
        """每列最大值（返回单行 DataFrame）。"""

    def std_withcol(self, col_names: Optional[List[str]] = None) -> "DataFrame":
        """每列标准差（返回单行 DataFrame）。"""

    def var_withcol(self, col_names: Optional[List[str]] = None) -> "DataFrame":
        """每列方差（返回单行 DataFrame）。"""

    def median_withcol(self, col_names: Optional[List[str]] = None) -> "DataFrame":
        """每列中位数（返回单行 DataFrame）。"""

    def quantile_withcol(self, q: float = 0.5, col_names: Optional[List[str]] = None) -> "DataFrame":
        """每列分位数（返回单行 DataFrame）。"""

    def mode_withcol(self, col_names: Optional[List[str]] = None) -> "DataFrame":
        """每列众数（返回 DataFrame）。"""

    def prod_withcol(self, col_names: Optional[List[str]] = None) -> "DataFrame":
        """每列乘积（返回单行 DataFrame）。"""

    def sem_withcol(self, col_names: Optional[List[str]] = None) -> "DataFrame":
        """每列标准误（返回单行 DataFrame）。"""

    def mad_withcol(self, col_names: Optional[List[str]] = None) -> "DataFrame":
        """每列平均绝对偏差（返回单行 DataFrame）。"""

    def describe_withcol(self, ) -> "DataFrame":
        """describe 的带列名变体（返回汇总 DataFrame）。"""

    # ------------------------------------------------------------------
    #  更完整的统计
    # ------------------------------------------------------------------

    def count(self, col_names: Optional[List[str]] = None) -> Any:
        """每列非 NaN 计数。"""

    def describe(self, percentiles: Optional[object] = None) -> "DataFrame":
        """描述性统计汇总：count/mean/std/min/25%/50%/75%/max。

        Parameters
        ----------
        percentiles: optional
            自定义分位数列表。

        Returns
        -------
        DataFrame
            统计汇总表。

        Examples
        --------
        >>> df.describe()
        """

    def value_counts(self, col_name: str) -> Series:
        """单列频次统计，按频次降序。

        Parameters
        ----------
        col_name: str
            统计列。

        Returns
        -------
        Series
            频次 Series（原值作索引）。

        Examples
        --------
        >>> df = DataFrame({'tag': str})
        >>> df.add_row(tag='a'); df.add_row(tag='a'); df.add_row(tag='b')
        >>> df.value_counts('tag')
        """

    def nunique(self, col_name: Optional[str] = None) -> Any:
        """去重计数（col_name=None 时返回各列去重计数）。"""

    def cumsum(self) -> "DataFrame":
        """各列累计求和。"""

    def cumprod(self) -> "DataFrame":
        """各列累计乘积。"""

    def cummax(self) -> "DataFrame":
        """各列累计最大值。"""

    def cummin(self) -> "DataFrame":
        """各列累计最小值。"""

    def skew(self) -> "DataFrame":
        """各列偏度。"""

    def kurt(self) -> "DataFrame":
        """各列峰度。"""

    def argmax(self, col_name: str) -> int:
        """指定列最大值位置。"""

    def argmin(self, col_name: str) -> int:
        """指定列最小值位置。"""

    def idxmax(self, col_name: Optional[str] = None) -> Any:
        """指定列最大值对应索引标签。"""

    def idxmin(self, col_name: Optional[str] = None) -> Any:
        """指定列最小值对应索引标签。"""

    def unique(self, col_name: str) -> List[Any]:
        """指定列去重值（保持首现顺序）。

        Parameters
        ----------
        col_name: str
            列名。

        Returns
        -------
        List[Any]
            唯一值列表。
        """

    def factorize(self, col_name: str) -> Any:
        """指定列因子化，返回 (codes, uniques)。"""

    def isin(self, col_name: str, values: Set[Any]) -> "DataFrame":
        """指定列值是否在集合中，返回掩码 DataFrame。

        Parameters
        ----------
        col_name: str
            列名。
        values: Set
            值集合。
        """

    def between(self, col_name: str, low: Any, high: Any) -> "DataFrame":
        """指定列范围掩码 (low <= x <= high)。"""

    def pop(self, col_name: str) -> Series:
        """弹出列并返回（原地删除列）。

        Parameters
        ----------
        col_name: str
            列名。

        Returns
        -------
        Series
            被删除的列。
        """

    @property
    def empty(self) -> bool:
        """是否空表。"""

    def update(self, other: "DataFrame") -> None:
        """用 other 中非缺失值原地更新本表。

        Parameters
        ----------
        other: DataFrame
            更新源。
        """

    # ------------------------------------------------------------------
    #  迭代 / 记录
    # ------------------------------------------------------------------

    def to_records(self, index: bool = True) -> List[Dict[str, Any]]:
        """转为 dict 记录列表。

        Parameters
        ----------
        index: bool, 默认 True
            是否包含索引键。
        """

    def items(self) -> Iterator[Tuple[str, Series]]:
        """迭代 (列名, Series) 对。"""

    def iterrows(self) -> Iterator[Tuple[Any, Dict[str, Any]]]:
        """逐行迭代，返回 (索引标签, 行 dict)。

        Examples
        --------
        >>> for label, row in df.iterrows():
        ...     print(label, row)
        """

    def itertuples(self) -> Iterator[Tuple[Any, ...]]:
        """逐行迭代为命名元组。"""

    # ------------------------------------------------------------------
    #  合并 / 比较
    # ------------------------------------------------------------------

    def join(self, other: "DataFrame", how: str = 'left') -> "DataFrame":
        """按索引连接（左侧对齐）。

        Parameters
        ----------
        other: DataFrame
            右侧表。
        how: {'left','inner','outer'}, 默认 'left'
            连接方式。
        """

    def compare(self, other: "DataFrame", align_axis: int = 1) -> "DataFrame":
        """比较两表差异，返回左右并排的差异表。"""

    def combine_first(self, other: "DataFrame") -> "DataFrame":
        """以自身数据优先，缺失处用 other 补齐。"""

    def interpolate(self, method: str = 'linear') -> "DataFrame":
        """数值列缺失值插值。

        Parameters
        ----------
        method: {'linear', ...}, 默认 'linear'
            插值方法。
        """

    def concat(self, other: "DataFrame") -> "DataFrame":
        """纵向拼接另一表（列需一致）。"""

    def merge(self, right: "DataFrame", on: object = None, how: str = 'inner') -> "DataFrame":
        """基于共同列的 SQL 风格连接。

        Parameters
        ----------
        right: DataFrame
            右表。
        on: str 或列表, optional
            连接键。
        how: {'inner','left','right','outer'}, 默认 'inner'
            连接方式。
        """

    # ------------------------------------------------------------------
    #  去重 / 排序
    # ------------------------------------------------------------------

    def duplicated(self, subset: Optional[List[str]] = None, keep: str = 'first') -> List[bool]:
        """标记重复行。

        Parameters
        ----------
        subset: List[str], optional
            判定列。
        keep: {'first','last', False}, 默认 'first'
            保留策略。
        """

    def drop_duplicates(self, subset: Optional[List[str]] = None, keep: str = 'first', inplace: bool = False) -> Optional["DataFrame"]:
        """删除重复行。

        Parameters
        ----------
        subset: 可选列名列表。
        keep: str 或 False。
        inplace: bool。
        """

    def argsort_by(self, col_name: str, reverse: bool = False) -> List[int]:
        """按列排序返回位置索引列表。

        Parameters
        ----------
        col_name: str
            排序列。
        reverse: bool, 默认 False
            降序。
        """

    def sort_values(self, col_name: str, ascending: bool = True, inplace: bool = False) -> Optional["DataFrame"]:
        """按列值排序。

        Parameters
        ----------
        col_name: str
            排序列。
        ascending: bool, 默认 True
            升序。
        inplace: bool, 默认 False
            原地。
        """

    def sort_index(self, ascending: bool = True, inplace: bool = False) -> Optional["DataFrame"]:
        """按索引排序。"""

    def diff(self, col_name: str, periods: int = 1) -> Any:
        """指定列逐行差分（返回该列结果或新表，视实现）。"""

    # ------------------------------------------------------------------
    #  逐元素应用
    # ------------------------------------------------------------------

    def apply(self, func: Callable[[Any], Any], axis: str = 'columns', raw: bool = False) -> Any:
        """沿轴应用函数。

        Parameters
        ----------
        func: callable
            raw=False 时逐元素调用；raw=True 时对整列数组调用。
        axis: {'columns', 'index'}, 默认 'columns'
            方向（columns=逐列/逐元素，index=逐行）。
        raw: bool, 默认 False
            True 时 func 接收 numpy 数组并须返回 array-like。

        Examples
        --------
        >>> df['score'].apply(lambda x: round(x))   # 元素级
        >>> df.apply(lambda arr: arr * 2, raw=True)  # 整列级
        """

    def applymap(self, func: Callable[[Any], Any]) -> "DataFrame":
        """逐元素应用函数。"""

    def clip_column(self, col_name: str, lower: Any = None, upper: Any = None, inplace: bool = False) -> Optional["DataFrame"]:
        """单列截断。"""

    def clip(self, lower: Any = None, upper: Any = None, inplace: bool = False) -> Optional["DataFrame"]:
        """整表数值截断。

        Parameters
        ----------
        lower: 标量, optional
            下界。
        upper: 标量, optional
            上界。
        inplace: bool, 默认 False
            原地。
        """

    def where_column(self, col_name: str, cond_col: str, other: Any) -> "DataFrame":
        """按条件列对指定列做 where。"""

    def mask_column(self, col_name: str, cond_col: str, other: Any) -> "DataFrame":
        """按条件列对指定列做 mask。"""

    def where_row(self, mask: IntColumn, fill: object = 0) -> "DataFrame":
        """按行掩码过滤，不满足处填 fill。"""

    # ------------------------------------------------------------------
    #  缺失值
    # ------------------------------------------------------------------

    def isnull(self) -> "DataFrame":
        """缺失掩码（与 isna 等价）。

        Examples
        --------
        >>> df.isnull()
        """

    def isna(self) -> "DataFrame":
        """缺失掩码。"""

    def notna(self) -> "DataFrame":
        """非缺失掩码 (isna 取反)。"""

    def fillna(self, value: Any = None, method: Any = None, col_names: Optional[List[str]] = None) -> "DataFrame":
        """填充缺失值。

        Parameters
        ----------
        value: 标量或 dict, optional
            填充值。
        method: {'ffill','bfill'}, optional
            填充方法。
        col_names: List[str], optional
            限定列。
        """

    def dropna(self, axis: int = 0, how: str = 'any', thresh: int = -1, subset: Optional[List[str]] = None) -> "DataFrame":
        """删除含缺失值的行/列。

        Parameters
        ----------
        axis: {0, 'index', 1, 'columns'}, 默认 0
            方向。
        how: {'any','all'}, 默认 'any'
            缺失条件。
        thresh: int, 默认 -1
            最少非缺失数阈值。
        subset: List[str], optional
            检查列。
        """

    # ------------------------------------------------------------------
    #  数学/统计函数
    # ------------------------------------------------------------------

    def abs(self) -> "DataFrame":
        """绝对值。"""

    def round(self, decimals: int = 0) -> "DataFrame":
        """四舍五入。"""

    def all(self) -> bool:
        """所有元素为真。"""

    def any(self) -> bool:
        """任一元素为真。"""

    def pct_change(self, periods: int = 1) -> "DataFrame":
        """百分比变化。"""

    def replace(self, to_replace: Any, value: Any) -> "DataFrame":
        """值替换。"""

    def ffill(self) -> "DataFrame":
        """前向填充。"""

    def bfill(self) -> "DataFrame":
        """后向填充。"""

    def head(self, n: int = 5) -> "DataFrame":
        """前 n 行。

        Examples
        --------
        >>> df.head(2)
        """

    def tail(self, n: int = 5) -> "DataFrame":
        """后 n 行。"""

    def sample(self, n: object = None, frac: object = None, replace: bool = False, seed: int = -1) -> "DataFrame":
        """随机抽样。

        Parameters
        ----------
        n: int, optional
            抽样数量。
        frac: float, optional
            抽样比例。
        replace: bool, 默认 False
            有放回。
        seed: int, 默认 -1
            随机种子。
        """

    def nsmallest(self, n: int, col_name: str) -> "DataFrame":
        """指定列最小的 n 行。"""

    def nlargest(self, n: int, col_name: str) -> "DataFrame":
        """指定列最大的 n 行。"""

    def pipe(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """管道调用：func(self, *args, **kwargs)。"""

    def assign(self, **kwargs: Any) -> "DataFrame":
        """新增列。

        Parameters
        ----------
        **kwargs
            列名 -> 值/Series/函数。

        Examples
        --------
        >>> df.assign(c=lambda d: d['a'] * 2)
        """

    def select_dtypes(self, include_types: Optional[List[str]] = None, exclude_types: Optional[List[str]] = None) -> "DataFrame":
        """按 dtype 选择列。

        Parameters
        ----------
        include_types: List[str], optional
            保留的类型。
        exclude_types: List[str], optional
            排除的类型。
        """

    def astype(self, dtype: Union[builtins.str, Dict[str, DType]]) -> "DataFrame":
        """整表类型转换。

        Parameters
        ----------
        dtype: str 或 Dict[str, DType]
            单个类型字符串将广播到所有列；dict 形式为
            {列名: 类型} 按列转换。

        Examples
        --------
        >>> df.astype(float)
        >>> df.astype({'id': float, 'score': float})
        """

    def add(self, col_names: Tuple[str, ...]) -> "DataFrame":
        """两列相加。

        Parameters
        ----------
        col_names: (col1, col2) 元组
            参与运算的两列。
        """

    def sub(self, col_names: Tuple[str, ...]) -> "DataFrame":
        """两列相减 (col1 - col2)。"""

    def mul(self, col_names: Tuple[str, ...]) -> "DataFrame":
        """两列相乘。"""

    def div(self, col_names: Tuple[str, ...]) -> "DataFrame":
        """两列相除 (col1 / col2)。"""

    # ------------------------------------------------------------------
    #  形状变换
    # ------------------------------------------------------------------

    def reshape(self, rows: int, cols: int) -> "DataFrame":
        """重塑为 R×C 布局（仅数值列）。"""

    def explode(self, col_name: str) -> "DataFrame":
        """将列表列炸开为多行。

        Parameters
        ----------
        col_name: str
            包含列表/元组的列。
        """

    def stack(self) -> "DataFrame":
        """将 2D 数值列堆叠为长表。

        Examples
        --------
        >>> df.stack()
        """

    def unstack(self) -> "DataFrame":
        """stack 的逆操作。"""

    @property
    def T(self) -> "DataFrame":
        """转置。

        Examples
        --------
        >>> df.T
        """

    # ------------------------------------------------------------------
    #  Metric / Linalg 访问器
    # ------------------------------------------------------------------

    @property
    def metric(self) -> _DataFrameMetricAccessor:
        """列对距离/相似度度量访问器。

        Examples
        --------
        >>> df.metric.euclidean('a', 'b')
        """

    def euclidean_dist_withcols(self, col1: str, col2: str) -> Any:
        """两列欧氏距离。"""

    def manhattan_dist_withcols(self, col1: str, col2: str) -> Any:
        """两列曼哈顿距离。"""

    def cosine_dist_withcols(self, col1: str, col2: str) -> Any:
        """两列余弦距离。"""

    def cosine_similarity_withcols(self, col1: str, col2: str) -> Any:
        """两列余弦相似度。"""

    def jaccard_sim_withcols(self, col1: str, col2: str) -> Any:
        """两列 Jaccard 相似度。"""

    def hamming_dist_withcols(self, col1: str, col2: str) -> Any:
        """两列汉明距离。"""

    @property
    def linalg(self) -> _DataFrameLinalgAccessor:
        """线性代数访问器。

        Examples
        --------
        >>> df.linalg.det()
        """

    def dot(self, b: "DataFrame") -> "DataFrame":
        """矩阵乘法（与 matmul 等价）。

        Parameters
        ----------
        b: DataFrame
            右侧矩阵。

        Returns
        -------
        DataFrame
            结果矩阵。
        """

    def matmul(self, b: "DataFrame") -> "DataFrame":
        """矩阵乘法。"""

    def transpose(self) -> "DataFrame":
        """转置。"""

    def det(self) -> float:
        """行列式。"""

    def inv(self) -> "DataFrame":
        """逆矩阵。"""

    def solve(self, b: "DataFrame") -> "DataFrame":
        """求解线性方程组 A·x = b。

        Parameters
        ----------
        b: DataFrame
            右侧向量/矩阵。
        """

    def norm(self, ord_type: str = "frobenius") -> float:
        """矩阵范数。

        Parameters
        ----------
        ord_type: {'frobenius', ...}, 默认 'frobenius'
            范数类型。
        """

    def euclidean_dist_matrix(self) -> "DataFrame":
        """列间欧氏距离矩阵。"""

    def manhattan_dist_matrix(self) -> "DataFrame":
        """列间曼哈顿距离矩阵。"""

    def cosine_dist_matrix(self) -> "DataFrame":
        """列间余弦距离矩阵。"""

    def hamming_dist_matrix(self) -> "DataFrame":
        """列间汉明距离矩阵。"""

    def jaccard_sim_matrix(self) -> "DataFrame":
        """列间 Jaccard 相似度矩阵。"""

    def cosine_similarity_matrix(self) -> "DataFrame":
        """列间余弦相似度矩阵。"""

    def corr(self) -> "DataFrame":
        """列间相关系数矩阵。"""

    def cov(self) -> "DataFrame":
        """列间协方差矩阵。"""

    def corrwith(self, other: Any) -> "DataFrame":
        """与另一表/Series 的列间相关。"""

    def corrcoef(self, col_a: str, col_b: str) -> float:
        """两列相关系数。"""

    def cov_withcols(self, col_a: str, col_b: str) -> float:
        """两列协方差。"""

    def pearson_sim(self) -> "DataFrame":
        """列间 Pearson 相似度矩阵。"""

    # ------------------------------------------------------------------
    #  静态构造器
    # ------------------------------------------------------------------

    @staticmethod
    def from_dict(data: Dict[str, Any], schema: Optional[Dict[str, DType]] = None, columns: Optional[List[str]] = None) -> "DataFrame":
        """从 dict-of-columns 构造 DataFrame。

        Parameters
        ----------
        data: Dict[str, List]
            列名 -> 值列表。
        schema: Dict[str, DType], optional
            类型声明。
        columns: List[str], optional
            列序。

        Examples
        --------
        >>> DataFrame.from_dict({'a': [1, 2], 'b': [1.5, 2.5]}, schema={'a': int, 'b': float})
        """

    @staticmethod
    def from_list(rows: List[Any], schema: Dict[str, DType], columns: Optional[List[str]] = None) -> "DataFrame":
        """从行列表构造 DataFrame。

        Parameters
        ----------
        rows: List[dict 或 list]
            每行数据。
        schema: Dict[str, DType]
            类型声明。
        """

    @staticmethod
    def from_arrays(arrays: Any, schema: Dict[str, DType], columns: Optional[List[str]] = None) -> "DataFrame":
        """从数组/ndarray 列构造 DataFrame。"""

    @staticmethod
    def arange(n: int) -> "DataFrame":
        """生成 n 个整数的单列表 (等价 0..n-1)。"""

    @staticmethod
    def zeros(*args: int) -> "DataFrame":
        """全 1 列表。行为与 np.zeros 类似，height 为行数。

        Examples
        --------
        >>> DataFrame.zeros(3, 2)      # 3 行 2 列
        """

    @staticmethod
    def ones(*args: int) -> "DataFrame":
        """全 1 表。"""

    @staticmethod
    def randn(*args: int) -> "DataFrame":
        """标准正态随机数表。"""

    @staticmethod
    def rand(*args: int) -> "DataFrame":
        """U[0,1) 均匀随机数表。"""

    # ------------------------------------------------------------------
    #  窗口函数
    # ------------------------------------------------------------------

    def rolling(self, window: int, min_periods: Optional[int] = None, center: bool = False, axis: int = 0, win_type: Optional[str] = None) -> _RollingWindow:
        """滚动窗口。

        Parameters
        ----------
        window: int
            窗口大小。
        min_periods: int, optional
            最少有效元素数。
        center: bool, 默认 False
            居中窗口。
        axis: int, 默认 0
            轴。
        win_type: str, optional
            加权类型。

        Returns
        -------
        _RollingWindow
            滚动访问器。

        Examples
        --------
        >>> df.rolling(2).mean()
        """

    def expanding(self, min_periods: int = 1, axis: int = 0) -> _RollingWindow:
        """扩展窗口。

        Parameters
        ----------
        min_periods: int, 默认 1
            最少有效数。
        axis: int, 默认 0
            轴。
        """

    def ewm(self, com: Optional[float] = None, span: Optional[float] = None, halflife: Optional[float] = None, alpha: Optional[float] = None, min_periods: int = 0, adjust: bool = True, ignore_na: bool = False) -> _EWMWindow:
        """指数加权窗口。

        Parameters
        ----------
        com: float, optional
            质心。
        span: float, optional
            跨度。
        halflife: float, optional
            半衰期。
        alpha: float, optional
            平滑系数。
        min_periods: int, 默认 0
            最少有效数。
        adjust: bool, 默认 True
            是否调整。
        ignore_na: bool, 默认 False
            忽略 NaN。
        """

    def shift(self, periods: int = 1, axis: str = 'index') -> "DataFrame":
        """表移位。

        Parameters
        ----------
        periods: int, 默认 1
            移动位数。
        axis: {'index','columns'}, 默认 'index'
            方向。
        """

    def diff_all(self, periods: int = 1) -> "DataFrame":
        """所有数值列同时差分。"""

    def rank(self, col_name: str, method: str = 'average', ascending: bool = True, na_option: str = 'keep') -> "DataFrame":
        """指定列排名。

        Parameters
        ----------
        col_name: str
            排序列。
        method: str, 默认 'average'
            排名方法。
        ascending: bool, 默认 True
            升序。
        na_option: str, 默认 'keep'
            缺失值处理。
        """

    def cut(self, col_name: str, bins: List[Any], include_lowest: bool = False, right: str = 'right') -> "DataFrame":
        """等距分箱（离散化）。

        Parameters
        ----------
        col_name: str
            数值列。
        bins: List[float]
            箱边界。
        include_lowest: bool, 默认 False
            是否含左边界。
        right: {'right','left'}, 默认 'right'
            区间闭合方向。
        """

    def qcut(self, col_name: str, q: int = 4, labels: str = 'True', precision: float = 3) -> "DataFrame":
        """分位数分箱。

        Parameters
        ----------
        col_name: str
            数值列。
        q: int, 默认 4
            分位数数量。
        labels: str, 默认 'True'
            标签策略。
        precision: float, 默认 3
            边界精度。
        """

    def pivot(self, index: str, columns: str, values: str) -> "DataFrame":
        """透视：index/columns 为键，values 为值。

        Parameters
        ----------
        index: str
            行键列。
        columns: str
            列键列。
        values: str
            值列。
        """

    def melt(self, id_vars: Optional[List[str]] = None, value_vars: Optional[List[str]] = None, var_name: str = 'variable', value_name: str = 'value') -> "DataFrame":
        """宽表转长表。

        Parameters
        ----------
        id_vars: List[str], optional
            标识列。
        value_vars: List[str], optional
            值列。
        var_name: str, 默认 'variable'
            变量名列名。
        value_name: str, 默认 'value'
            值列名。
        """

    def expanding_sum(self, col_name: str) -> "DataFrame":
        """指定列扩展（累计）求和。"""

    def ewm_mean(self, col_name: str, span: float) -> "DataFrame":
        """指定列 ewm 均值（按 span 系数）。

        Parameters
        ----------
        col_name: str
            列名。
        span: float
            跨度。
        """

    # ------------------------------------------------------------------
    #  导出 / IO
    # ------------------------------------------------------------------

    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

    def to_string(self) -> str:
        """转为美观的字符串表格。"""

    def print(self, max_rows: int = 80) -> None:
        """终端打印表格。

        Parameters
        ----------
        max_rows: int, 默认 80
            最大显示行数。
        """

    def to_html(self, max_rows: int = 80) -> str:
        """转 HTML 表格字符串。

        Parameters
        ----------
        max_rows: int, 默认 80
            最大行。
        """

    def to_excel(self, filename: object, sheet_name: str = 'Sheet1') -> None:
        """导出 Excel（需 openpyxl）。

        Parameters
        ----------
        filename: str
            路径。
        sheet_name: str, 默认 'Sheet1'
            表名。
        """

    def to_pickle(self, path: str) -> None:
        """序列化到 pickle 文件。"""

    def to_csv(self, path: str, sep: str = ',', na_rep: str = '', float_format: Optional[str] = None, columns: Optional[List[str]] = None, header: bool = True, line_terminator: str = '\n') -> None:
        """导出 CSV。

        Parameters
        ----------
        path: str
            输出路径。
        sep: str, 默认 ','
            分隔符。
        na_rep: str, 默认 ''
            缺失值表示。
        float_format: str, optional
            浮点格式。
        columns: List[str], optional
            导出列。
        header: bool, 默认 True
            是否写表头。
        line_terminator: str, 默认 '\\n'
            行结束符。

        Examples
        --------
        >>> df.to_csv('/tmp/out.csv')
        """

    def to_tsv(self, filename: str) -> None:
        """导出 TSV（tab 分隔）。"""

    def to_dict(self) -> Dict[str, Any]:
        """转为 {列名: values 列表}。"""

    def to_json(self, orient_records: bool = True) -> str:
        """导出 JSON 字符串。

        Parameters
        ----------
        orient_records: bool, 默认 True
            True=记录数组，False=按列 dict。
        """

    def to_numpy(self) -> Any:
        """转为 numpy 二维数组。"""

    def info(self) -> None:
        """打印表结构信息（列名/类型/内存）。"""

    def _dtype_summary(self) -> Any:
        pass

    def memory_usage(self) -> int:
        """内存占用字节数。"""


# ===========================================================================
#  GroupBy 访问器
# ===========================================================================

class _DataFrameGroupByAccessor:
    """分组访问器：由 ``DataFrame.groupby()`` 返回。

    Examples
    --------
    >>> df = DataFrame({'k': int, 'v': int})
    >>> df.add_rows_from_list([{'k': 1, 'v': 10}, {'k': 1, 'v': 20}, {'k': 2, 'v': 30}])
    >>> g = df.groupby('k')
    >>> g.sum()
    """

    def size(self) -> "DataFrame":
        """各组行数。

        Returns
        -------
        DataFrame
            每组行数（''count'' 列或新列）。

        Examples
        --------
        >>> df.groupby('k').size()
        """

    def agg(self, agg_dict: Dict[str, Any]) -> "DataFrame":
        """聚合：agg_dict = {列名: 函数名或 callable}。

        Parameters
        ----------
        agg_dict: Dict[str, str 或 callable]
            列 -> 聚合方式（'sum'/'mean'/'max'... 或函数）。

        Returns
        -------
        DataFrame
            聚合结果。

        Examples
        --------
        >>> df.groupby('k').agg({'v': 'sum'})
        """

    def sum(self, col_name: Optional[str] = None) -> "DataFrame":
        """各组求和。

        Parameters
        ----------
        col_name: str, optional
            限定列。
        """

    def mean(self, col_name: Optional[str] = None) -> "DataFrame":
        """各组均值。"""

    def count(self) -> "DataFrame":
        """各组非 NaN 计数。"""

    def min(self, col_name: Optional[str] = None) -> "DataFrame":
        """各组最小值。"""

    def max(self, col_name: Optional[str] = None) -> "DataFrame":
        """各组最大值。"""

    def median(self, col_name: Optional[str] = None) -> "DataFrame":
        """各组中位数。"""

    def quantile(self, col_name: Optional[str] = None) -> "DataFrame":
        """各组分位数（默认 0.5）。"""

    def var(self, col_name: Optional[str] = None) -> "DataFrame":
        """各组方差。"""

    def std(self, col_name: Optional[str] = None) -> "DataFrame":
        """各组标准差。"""

    def apply(self, func: Callable[[Any], Any]) -> "DataFrame":
        """对每组应用函数。

        Parameters
        ----------
        func: callable
            接收子 DataFrame 返回 DataFrame/Series。
        """

    def idxmax(self, col_name: Optional[str] = None) -> "DataFrame":
        """各组最大值对应行索引。"""

    def idxmin(self, col_name: Optional[str] = None) -> "DataFrame":
        """各组最小值对应行索引。"""

    def transform(self, func: Callable[[Any], Any]) -> "DataFrame":
        """变换：保持组内行数，返回同形状结果。

        Parameters
        ----------
        func: callable
            输入子列/子组，输出同长变换。
        """

    def filter(self, func: Callable[[Any], bool]) -> "DataFrame":
        """按组函数结果过滤行。

        Parameters
        ----------
        func: callable
            输入组，返回 bool 决定是否保留。
        """


# ===========================================================================
#  线性代数访问器 (df.linalg)
# ===========================================================================

class _DataFrameLinalgAccessor:
    """线性代数访问器：由 ``DataFrame.linalg`` 返回。

    Examples
    --------
    >>> m = DataFrame.from_list([[1, 2], [3, 4]], schema={'c0': int, 'c1': int})
    >>> m.linalg.det()
    -2.0
    """

    def dot(self, b: DataFrame) -> DataFrame:
        """矩阵乘法。"""

    def matmul(self, b: DataFrame) -> DataFrame:
        """矩阵乘法（与 dot 等价）。"""

    def transpose(self) -> DataFrame:
        """转置。"""

    def det(self) -> float:
        """行列式。

        Examples
        --------
        >>> m.linalg.det()
        """

    def inv(self) -> DataFrame:
        """逆矩阵。"""

    def solve(self, b: DataFrame) -> DataFrame:
        """解线性方程组。"""

    def norm(self, ord_type: str = "frobenius") -> float:
        """矩阵范数。"""

    def euclidean_dist(self) -> DataFrame:
        """列间欧氏距离矩阵。"""

    def manhattan_dist(self) -> DataFrame:
        """列间曼哈顿距离矩阵。"""

    def cosine_dist(self) -> DataFrame:
        """列间余弦距离矩阵。"""

    def hamming_dist(self) -> DataFrame:
        """列间汉明距离矩阵。"""

    def jaccard_sim(self) -> DataFrame:
        """列间 Jaccard 相似度矩阵。"""

    def cosine_similarity(self) -> DataFrame:
        """列间余弦相似度矩阵。"""


# ===========================================================================
#  Metric 访问器 (df.metric)
# ===========================================================================

class _DataFrameMetricAccessor:
    """度量访问器：由 ``DataFrame.metric`` 返回，两列之间的距离/相似度。

    Examples
    --------
    >>> df = DataFrame({'a': int, 'b': int})
    >>> df.add_row(a=1, b=2); df.add_row(a=3, b=4)
    >>> df.metric.euclidean('a', 'b')
    """

    def euclidean(self, col1: str, col2: str) -> float:
        """两列欧氏距离。"""

    def manhattan(self, col1: str, col2: str) -> float:
        """两列曼哈顿距离。"""

    def cosine_dist(self, col1: str, col2: str) -> float:
        """两列余弦距离。"""

    def cosine_similarity(self, col1: str, col2: str) -> float:
        """两列余弦相似度。"""

    def jaccard(self, col1: str, col2: str) -> float:
        """两列 Jaccard 相似度。"""

    def hamming(self, col1: str, col2: str) -> float:
        """两列汉明距离。"""

    def corr(self) -> DataFrame:
        """列间相关系数矩阵。"""

    def cov(self) -> DataFrame:
        """列间协方差矩阵。"""

    def corrwith(self, other: Any) -> DataFrame:
        """与另一表列间相关。"""

    def corrcoef(self, col_a: str, col_b: str) -> float:
        """两列相关系数。"""

    def cov_withcols(self, col_a: str, col_b: str) -> float:
        """两列协方差。"""

    def pearson(self) -> DataFrame:
        """列间 Pearson 相似度矩阵。"""


# ===========================================================================
#  反序列化辅助
# ===========================================================================

class _SafeUnpickler:
    """受限的 pickle 加载器：仅恢复本库类型，防止任意代码执行。"""

    def __init__(self, file: Any, allow_raw_pickle: bool = False) -> None: ...

    def find_class(self, module: str, name: str) -> Any: ...


# ===========================================================================
#  IO 与转换函数
# ===========================================================================

def column_from_list(df: DataFrame, name: str, values: List[Any]) -> BaseColumn:
    """从值列表构造列（内部使用的 Python 可调用包装）。

    Parameters
    ----------
    df: DataFrame
        所在表。
    name: str
        列名。
    values: List
        值列表。

    Returns
    -------
    BaseColumn
        构造出的列对象。
    """

def from_pickle(filename: str, allow_raw_pickle: bool = False) -> Any:
    """从 pickle 文件加载对象（受限反序列化）。

    Parameters
    ----------
    filename: str
        文件路径。
    allow_raw_pickle: bool, 默认 False
        允许任意 pickle（不安全）。

    Returns
    -------
    反序列化的对象（通常为 DataFrame）。
    """

def read_csv(filename: str, schema: Dict[str, DType], sep: str = ',') -> DataFrame:
    """读取 CSV 文件到 DataFrame。

    Parameters
    ----------
    filename: str
        文件路径。
    schema: Dict[str, DType]
        列名 -> 类型 映射（必须提供，本库不做类型推断）。
    sep: str, 默认 ','
        分隔符。

    Returns
    -------
    DataFrame
        读取的表格。

    Examples
    --------
    >>> from fluidframe.core.dataframe import read_csv
    >>> df = read_csv('data.csv', schema={'id': int, 'name': str})
    """

def read_tsv(filename: str, schema: Dict[str, DType]) -> DataFrame:
    """读取 TSV 文件（tab 分隔）。

    Parameters
    ----------
    filename: str
        文件路径。
    schema: Dict[str, DType]
        类型声明。

    Returns
    -------
    DataFrame
        表格。
    """

def read_json(filename: str, schema: Dict[str, DType]) -> DataFrame:
    """读取 JSON 文件到 DataFrame。

    Parameters
    ----------
    filename: str
        文件路径。
    schema: Dict[str, DType]
        类型声明。

    Returns
    -------
    DataFrame
        表格。
    """

def concat_outer(list_dfs: List[DataFrame]) -> DataFrame:
    """纵向拼接（外连接：全部保留）。

    Parameters
    ----------
    list_dfs: List[DataFrame]
        待拼接的表列表。

    Returns
    -------
    DataFrame
        拼接结果。
    """

def concat_inner(list_dfs: List[DataFrame]) -> DataFrame:
    """纵向拼接（内连接：仅共同列）。

    Parameters
    ----------
    list_dfs: List[DataFrame]
        待拼接列表。
    """

def concat(list_dfs: List[DataFrame], ignore_index: bool = False) -> DataFrame:
    """纵向拼接多张表。

    Parameters
    ----------
    list_dfs: List[DataFrame]
        待拼接列表。
    ignore_index: bool, 默认 False
        是否重置索引。

    Returns
    -------
    DataFrame
        整表。

    Examples
    --------
    >>> from fluidframe.core.dataframe import concat
    >>> df1 = DataFrame({'a': int}); df1.add_row(a=1)
    >>> df2 = DataFrame({'a': int}); df2.add_row(a=2)
    >>> concat([df1, df2])
    """

def to_numeric(column: BaseColumn, errors: str = 'coerce') -> BaseColumn:
    """将列转为数值列。

    Parameters
    ----------
    column: BaseColumn
        输入列（UCS2Column / AsciiColumn 等）。
    errors: {'coerce','raise','ignore'}, 默认 'coerce'
        错误处理。

    Returns
    -------
    DoubleColumn
        转换结果。
    """

def to_datetime(column: BaseColumn, fmt: str = '', errors: str = 'coerce') -> DateTimeColumn:
    """将列解析为日期时间列。

    Parameters
    ----------
    column: BaseColumn
        输入列（字符串列）。
    fmt: str, 默认 ''
        格式（如 '%Y-%m-%d'）。
    errors: str, 默认 'coerce'
        错误处理。

    Returns
    -------
    DateTimeColumn
        结果列。
    """

def to_date(column: BaseColumn, fmt: str = '', errors: str = 'coerce') -> DateColumn:
    """将列解析为日期列。

    Parameters
    ----------
    column: BaseColumn
        输入列（字符串列）。
    fmt: str, 默认 ''
        格式。
    errors: str, 默认 'coerce'
        错误处理。

    Returns
    -------
    DateColumn
        结果列。

    Examples
    --------
    >>> from fluidframe.core.ucs2column import UCS2Column
    >>> col = UCS2Column(2)
    >>> col.append('2024-01-01'); col.append('20240615')
    >>> d = to_date(col)
    >>> d.length
    2
    """

def to_time(column: BaseColumn, fmt: str = '', errors: str = 'coerce') -> TimeColumn:
    """将列解析为时间列。

    Parameters
    ----------
    column: BaseColumn
        输入列。
    fmt: str, 默认 ''
        格式。
    errors: str, 默认 'coerce'
        错误处理。

    Returns
    -------
    TimeColumn
        结果列。
    """

def to_timedelta(values: BaseColumn, unit: str = 'D') -> BaseColumn:
    """将列转为时间差。

    Parameters
    ----------
    values: BaseColumn
        输入列。
    unit: str, 默认 'D'
        单位（'D' 天等）。

    Returns
    -------
    BaseColumn
        时间差列。
    """

def where(cond: IntColumn, true_val: object, false_val: object) -> Any:
    """三目选择：cond 为 1 处取 true_val，否则 false_val。

    Parameters
    ----------
    cond: IntColumn
        0/1 掩码。
    true_val: 标量或列
        真值分支。
    false_val: 标量或列
        假值分支。
    """


# ---------------------------------------------------------------------------
#  模块级私有辅助（_series_*：供运算符/方法使用的 Cython 可调用入口）
# ---------------------------------------------------------------------------

def _series_value_counts(col: BaseColumn, key_name: str = 'value') -> Series:
    """单列频次统计（``Series.value_counts`` 的底层实现）。

    Parameters
    ----------
    col: BaseColumn
        数据列。
    key_name: str, 默认 'value'
        结果频次列名。

    Returns
    -------
    Series
        频次统计 Series。
    """

def _series_isna(col: BaseColumn) -> Series:
    """缺失掩码（``Series.isna`` 底层实现）。

    Parameters
    ----------
    col: BaseColumn
        数据列。

    Returns
    -------
    Series
        布尔标记列。
    """

def _series_notna(col: BaseColumn) -> Series:
    """非缺失掩码（``Series.notna`` 底层实现）。"""

def _series_factorize(col: BaseColumn) -> Tuple[IntColumn, Any]:
    """因子化（``Series.factorize`` 底层实现）。

    Parameters
    ----------
    col: BaseColumn
        数据列。

    Returns
    -------
    (IntColumn, 去重值) 元组
        编码列与唯一值。
    """

def _series_map(col: BaseColumn, func: Callable[[Any], Any]) -> Series:
    """逐元素映射（``Series.map/apply`` 底层实现）。

    Parameters
    ----------
    col: BaseColumn
        数据列。
    func: callable
        元素级函数。

    Returns
    -------
    Series
        映射结果列。
    """