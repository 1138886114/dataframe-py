"""FluidDataFrame 列类型桩 — 9 种底层列 + 基类。

本文件是 ``fluidframe.core.intcolumn``、``fluidframe.core.doublecolumn``、
``fluidframe.core.asciicolumn``、``fluidframe.core.ucs2column``、
``fluidframe.core.datecolumn``、``fluidframe.core.boolcolumn``、
``fluidframe.core.categorycolumn``、``fluidframe.core.datetimecolumn``、
``fluidframe.core.timecolumn`` 等别名命名空间的真实类型声明。

存储约定
--------
- 数值列 (``IntColumn``/``DoubleColumn``)：支持 1D（单列）与 2D（多子列）
  布局，列主序。
- 字符串列：``AsciiColumn`` 为定宽 256 字节字符串，``UCS2Column`` 为变长
  UTF-8 字符串。
- 日期列 (``DateColumn``)：底层为自 1970-01-01 起的天数的 int32。
- 所有列均实现 ``__len__``/``__getitem__``，可按行号取元素。

示例约定
--------
    >>> from fluidframe.core.intcolumn import IntColumn
    >>> from fluidframe.core.doublecolumn import DoubleColumn
    >>> from fluidframe.core.asciicolumn import AsciiColumn
    >>> from fluidframe.core.datecolumn import DateColumn
    >>> col_int = IntColumn.from_list([1, 2, 3])
    >>> col_dbl = DoubleColumn.from_list([1.5, 2.5, 3.5])
    >>> col_str = AsciiColumn.from_list(['a', 'b', 'c'])
"""

from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

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

# ---------------------------------------------------------------------------
#  常用类型别名
# ---------------------------------------------------------------------------

#: 列的 dtype 声明（对齐 fluidframe.core._databaseframe.DType）
DType = Union[
    type,
    str,
]

#: 标量值：列元素可以是 Python 标量、日期/时间对象
Scalar = Union[int, float, bool, str, bytes, Date, DateTime, Time, Ascii]

NumericType = Union[int, float]


# ===========================================================================
#  基类
# ===========================================================================

class BaseColumn:
    """列基类：所有列类型的公共接口。

    Parameters
    ----------
    name: str, optional
        列名。

    Examples
    --------
    >>> col = IntColumn(name='id')
    """

    def __init__(self, name: Optional[str] = None) -> None:
        """创建列。

        Parameters
        ----------
        name: Optional[str], 默认 None
            列名。
        """

    def get_size(self) -> int:
        """元素个数（1D 时为行数，2D 时为 rows*cols）。

        Returns
        -------
        int
            总元素数。
        """

    def __contains__(self, val: object) -> bool:
        """``val in col``：是否包含某值。"""

    def copy(self) -> "BaseColumn":
        """深拷贝。

        Returns
        -------
        BaseColumn
            独立副本。
        """

    def drop_row(self, row_idx: int) -> None:
        """按行号删除元素（原地）。

        Parameters
        ----------
        row_idx: int
            行位置。
        """

    def shift_left(self, pos: int) -> None:
        """删除首部元素并把后续元素整体左移（原地）。

        Parameters
        ----------
        pos: int
            左移量。
        """

    def astype(self, target: object) -> "BaseColumn":
        """类型转换。

        Parameters
        ----------
        target: DType
            目标类型。

        Returns
        -------
        BaseColumn
            转换后的列。

        Examples
        --------
        >>> col_int.astype(float)
        """

    def where_mask(self, mask: "IntColumn", fill: object) -> "BaseColumn":
        """按 0/1 掩码做三目选择：1 保留原值，0 处填 fill。

        Parameters
        ----------
        mask: IntColumn
            0/1 掩码。
        fill: Scalar
            假分支填充值。
        """

    def ffill(self) -> "BaseColumn":
        """前向填充缺失值。

        Returns
        -------
        BaseColumn
            填充后的列。
        """

    def bfill(self) -> "BaseColumn":
        """后向填充缺失值。

        Returns
        -------
        BaseColumn
            填充后的列。
        """

    def abs(self) -> "BaseColumn":
        """逐元素绝对值。"""

    def round(self, decimals: int = 0) -> "BaseColumn":
        """逐元素四舍五入。

        Parameters
        ----------
        decimals: int, 默认 0
            小数位数。
        """


# ===========================================================================
#  整数列 (IntColumn)
# ===========================================================================

class IntColumn(BaseColumn):
    """int32 整数列。

    - 1D：单一序列；2D：多子列（cols×rows 列主序）。
    - 支持完整算术 / 比较 / 聚合 / 滚动 / 指数加权操作。

    Examples
    --------
    >>> from fluidframe.core.intcolumn import IntColumn
    >>> col = IntColumn.from_list([1, 2, 3, 4])
    >>> col + 1
    >>> col.mean()
    2.5
    """

    def cols(self) -> int:
        """子列数（1D 时为 1）。"""

    def rows(self) -> int:
        """行数。"""

    @property
    def _owns_data(self) -> bool:
        """内部：是否拥有底层内存所有权。"""

    @_owns_data.setter
    def _owns_data(self, val: bool) -> None:
        """内部：设置所有权标记。"""

    def get_size(self) -> int:
        """元素总数（rows * cols）。"""

    @classmethod
    def from_2d(cls, values: object, rows: int, cols: int) -> "IntColumn":
        """从二维结构构造 2D 列。

        Parameters
        ----------
        values: 嵌套列表或数组
            数据。
        rows: int
            行数。
        cols: int
            子列数。
        """

    @classmethod
    def from_list(cls, data: List[int]) -> "IntColumn":
        """从 Python 列表构造列。

        Parameters
        ----------
        data: List[int]
            元素。

        Returns
        -------
        IntColumn
            列。

        Examples
        --------
        >>> IntColumn.from_list([1, 2, 3])
        """

    @classmethod
    def _load_from_1d_array(cls, arr: object) -> "IntColumn":
        """内部：从一维数组加载。"""

    @classmethod
    def _load_from_2d_array(cls, arr: object, col_names: List[str]) -> "IntColumn":
        """内部：从二维数组加载。"""

    def append(self, val: int) -> None:
        """追加一个元素。

        Parameters
        ----------
        val: int
            值。
        """

    def copy(self) -> "IntColumn":
        """深拷贝。"""

    def ffill(self) -> "IntColumn":
        """前向填充。"""

    def bfill(self) -> "IntColumn":
        """后向填充。"""

    def pop_back(self) -> None:
        """删除末元素。"""

    def pop_front(self) -> None:
        """删除首元素。"""

    def concat(self, other: "IntColumn") -> None:
        """拼接另一整数列（原地）。"""

    def max(self) -> int:
        """最大值。

        Examples
        --------
        >>> col.max()
        4
        """

    def min(self) -> int:
        """最小值。"""

    def sum(self) -> int:
        """求和。"""

    def mean(self) -> float:
        """均值。"""

    def var(self) -> float:
        """方差 (ddof=1)。"""

    def std(self) -> float:
        """标准差 (ddof=1)。"""

    def idxmax(self) -> int:
        """最大值所在行位置。

        Examples
        --------
        >>> col.idxmax()
        3
        """

    def idxmin(self) -> int:
        """最小值所在行位置。"""

    def median(self) -> float:
        """中位数。"""

    def mad(self) -> float:
        """平均绝对偏差。"""

    def prod(self) -> int:
        """乘积。"""

    def mode(self) -> List[int]:
        """众数列表。"""

    def quantile(self, q: float = 0.5) -> float:
        """分位数。

        Parameters
        ----------
        q: float, 默认 0.5
            分位数（0~1）。
        """

    def cumsum(self) -> "IntColumn":
        """累计求和。"""

    def cummax(self) -> "IntColumn":
        """累计最大值。"""

    def cummin(self) -> "IntColumn":
        """累计最小值。"""

    def cumprod(self) -> "IntColumn":
        """累计乘积。"""

    def diff(self, periods: int = 1) -> "IntColumn":
        """差分。

        Parameters
        ----------
        periods: int, 默认 1
            差分阶数。
        """

    def skew(self) -> float:
        """偏度。"""

    def kurt(self) -> float:
        """峰度。"""

    def sem(self) -> float:
        """标准误。"""

    def count(self) -> int:
        """非 NaN 计数。"""

    def add_val(self, val: int) -> "IntColumn":
        """逐元素加标量。

        Parameters
        ----------
        val: int
            标量。

        Returns
        -------
        IntColumn
            新列。
        """

    def sub_val(self, val: int) -> "IntColumn":
        """逐元素减标量。"""

    def mul_val(self, val: int) -> "IntColumn":
        """逐元素乘标量。"""

    def div_val(self, val: int) -> "IntColumn":
        """逐元素除标量。"""

    def add(self, other: object) -> "IntColumn":
        """框列或标量相加。"""

    def sub(self, other: object) -> "IntColumn":
        """框列或标量相减。"""

    def mul(self, other: object) -> "IntColumn":
        """框列或标量相乘。"""

    def div(self, other: object) -> "IntColumn":
        """框列或标量相除。"""

    def __add__(self, other: object) -> "IntColumn":
        """``col + x``。

        Examples
        --------
        >>> col + 10
        """

    def __radd__(self, other: object) -> "IntColumn":
        """``x + col``。"""

    def __sub__(self, other: object) -> "IntColumn":
        """``col - x``。"""

    def __rsub__(self, other: object) -> "IntColumn":
        """``x - col``。"""

    def __mul__(self, other: object) -> "IntColumn":
        """``col * x``。"""

    def __rmul__(self, other: object) -> "IntColumn":
        """``x * col``。"""

    def __truediv__(self, other: object) -> "DoubleColumn":
        """``col / x``。"""

    def __rtruediv__(self, other: object) -> "DoubleColumn":
        """``x / col``。"""

    def __getitem__(self, idx: int) -> int:
        """按行号取元素。

        Parameters
        ----------
        idx: int
            行位置。

        Returns
        -------
        int
            元素值。

        Examples
        --------
        >>> col[0]
        1
        """

    def unique(self) -> "IntColumn":
        """去重（保持首现顺序）。"""

    def __lt__(self, val: object) -> "BoolColumn":
        """``col < x``：逐元素比较返回布尔列。"""

    def __le__(self, val: object) -> "BoolColumn":
        """``col <= x``。"""

    def __gt__(self, val: object) -> "BoolColumn":
        """``col > x``。"""

    def __ge__(self, val: object) -> "BoolColumn":
        """``col >= x``。"""

    def __eq__(self, val: object) -> "BoolColumn":  # type: ignore[override]
        """``col == x``。"""

    def __ne__(self, val: object) -> "BoolColumn":  # type: ignore[override]
        """``col != x``。"""

    def __and__(self, other: object) -> "IntColumn":
        """按位与（np.bitwise 语义）。"""

    def __or__(self, other: object) -> "IntColumn":
        """按位或。"""

    def __invert__(self) -> "IntColumn":
        """按位取反。"""

    def isin(self, values: Set[int]) -> "BoolColumn":
        """元素是否在集合中。

        Parameters
        ----------
        values: Set[int]
            值集合。

        Returns
        -------
        BoolColumn
            布尔掩码列。
        """

    def between(self, low: int, high: int) -> "BoolColumn":
        """``low <= x <= high``。

        Parameters
        ----------
        low: int
            下界。
        high: int
            上界。
        """

    def __str__(self) -> str:
        """列字符串表示。"""

    def __bool__(self) -> bool:
        """是否非空。"""

    def __len__(self) -> int:
        """行数。"""

    def clip(self, lower: object = None, upper: object = None, inplace: bool = False) -> "IntColumn":
        """截断到 [lower, upper]。

        Parameters
        ----------
        lower: int, optional
            下界。
        upper: int, optional
            上界。
        inplace: bool, 默认 False
            原地。
        """

    def abs(self) -> "IntColumn":
        """绝对值。"""

    def where(self, cond: "IntColumn", other: int, inplace: bool = False) -> "IntColumn":
        """三目选择：cond 为真取原值，否则取 other。

        Parameters
        ----------
        cond: IntColumn
            0/1 掩码。
        other: int
            假分支值。
        inplace: bool, 默认 False
            原地。
        """

    def where_mask(self, mask: "IntColumn", fill: object) -> "IntColumn":
        """掩码替换：1 保留原值，0 处填 fill。"""

    def mask(self, cond: "IntColumn", other: int, inplace: bool = False) -> "IntColumn":
        """cond 为真处填 other（where 的反操作）。"""

    def drop_row(self, row_idx: int) -> None:
        """删除行。"""

    def shift_left(self, pos: int) -> None:
        """左移删除。"""

    def shift_right(self, pos: int) -> "IntColumn":
        """右移（缺失处留空）。"""

    # --- 距离 / 相似度 ------------------------------------------------

    def euclidean_dist(self, other: object) -> float:
        """与另一列欧氏距离。"""

    def manhattan_dist(self, other: object) -> float:
        """与另一列曼哈顿距离。"""

    def cosine_similarity(self, other: object) -> float:
        """与另一列余弦相似度。"""

    def cosine_dist(self, other: object) -> float:
        """与另一列余弦距离。"""

    def jaccard_sim(self, other: object) -> float:
        """与另一列 Jaccard 相似度。"""

    def hamming_dist(self, other: object) -> float:
        """与另一列汉明距离。"""

    def astype(self, target: object) -> BaseColumn:
        """类型转换（支持 -> DoubleColumn 等）。"""

    # --- 滚动窗口 ------------------------------------------------------

    def roll_mean(self, window: int, min_periods: Optional[int] = None, center: bool = False) -> "IntColumn":
        """滚动均值。

        Parameters
        ----------
        window: int
            窗口大小。
        min_periods: int, optional
            最少有效数。
        center: bool, 默认 False
            居中。
        """

    def roll_sum(self, window: int, min_periods: Optional[int] = None, center: bool = False) -> "IntColumn":
        """滚动求和。"""

    def roll_min(self, window: int, min_periods: Optional[int] = None, center: bool = False) -> "IntColumn":
        """滚动最小值。"""

    def roll_max(self, window: int, min_periods: Optional[int] = None, center: bool = False) -> "IntColumn":
        """滚动最大值。"""

    def roll_var(self, window: int, ddof: bool = True, min_periods: Optional[int] = None, center: bool = False) -> "IntColumn":
        """滚动方差。"""

    def roll_std(self, window: int, ddof: bool = True, min_periods: Optional[int] = None, center: bool = False) -> "IntColumn":
        """滚动标准差。"""

    def roll_median(self, window: int, min_periods: Optional[int] = None, center: bool = False) -> "IntColumn":
        """滚动中位数。"""

    def roll_quantile(self, window: int, quantile: float, min_periods: Optional[int] = None, center: bool = False) -> "IntColumn":
        """滚动分位数。"""

    def roll_skew(self, window: int, min_periods: Optional[int] = None, center: bool = False) -> "IntColumn":
        """滚动偏度。"""

    def roll_kurt(self, window: int, min_periods: Optional[int] = None, center: bool = False) -> "IntColumn":
        """滚动峰度。"""

    def roll_corr(self, window: int, other: object, min_periods: Optional[int] = None, center: bool = False) -> "IntColumn":
        """与另一列滚动相关系数。"""

    def roll_cov(self, window: int, other: object, min_periods: Optional[int] = None, center: bool = False) -> "IntColumn":
        """与另一列滚动协方差。"""

    # --- 指数加权 ------------------------------------------------------

    def ewm_mean(self, alpha: float, min_periods: int = 0, adjust: bool = True, ignore_na: bool = False) -> "IntColumn":
        """指数加权均值。

        Parameters
        ----------
        alpha: float
            平滑系数 (0<alpha<=1)。
        min_periods: int, 默认 0
            最少有效数。
        adjust: bool, 默认 True
            是否归一化调整。
        ignore_na: bool, 默认 False
            忽略 NaN。

        Examples
        --------
        >>> col.ewm_mean(alpha=0.5)
        """

    def ewm_sum(self, alpha: float, min_periods: int = 0, adjust: bool = True, ignore_na: bool = False) -> "IntColumn":
        """指数加权求和。"""

    def ewm_var(self, alpha: float, bias: bool = False, min_periods: int = 0, adjust: bool = True, ignore_na: bool = False) -> "IntColumn":
        """指数加权方差。"""

    def ewm_std(self, alpha: float, bias: bool = False, min_periods: int = 0, adjust: bool = True, ignore_na: bool = False) -> "IntColumn":
        """指数加权标准差。"""

    def ewm_cov(self, other: object, alpha: float, bias: bool = False, min_periods: int = 0, adjust: bool = True, ignore_na: bool = False) -> "IntColumn":
        """指数加权协方差。"""

    def ewm_corr(self, other: object, alpha: float, min_periods: int = 0, adjust: bool = True, ignore_na: bool = False) -> "IntColumn":
        """指数加权相关系数。"""

    # --- 加权滚动 ------------------------------------------------------

    def roll_win_mean(self, window: int, weights: object, min_periods: Optional[int] = None, center: bool = False) -> "IntColumn":
        """窗口加权均值。

        Parameters
        ----------
        window: int
            窗口大小。
        weights: 列表
            权重向量。
        min_periods: int, optional
            最少有效数。
        center: bool, 默认 False
            居中。
        """

    def roll_win_sum(self, window: int, weights: object, min_periods: Optional[int] = None, center: bool = False) -> "IntColumn":
        """窗口加权求和。"""

    def roll_win_var(self, window: int, weights: object, ddof: bool = True, min_periods: Optional[int] = None, center: bool = False) -> "IntColumn":
        """窗口加权方差。"""

    def roll_win_std(self, window: int, weights: object, ddof: bool = True, min_periods: Optional[int] = None, center: bool = False) -> "IntColumn":
        """窗口加权标准差。"""


# ===========================================================================
#  浮点列 (DoubleColumn)
# ===========================================================================

class DoubleColumn(BaseColumn):
    """float64 列。

    与 ``IntColumn`` 接口基本一致：算术、比较、聚合、滚动、指数加权均相同，
    故略去重复示例（见 IntColumn）。

    Examples
    --------
    >>> from fluidframe.core.doublecolumn import DoubleColumn
    >>> col = DoubleColumn.from_list([1.5, 2.5, 3.5])
    >>> col.mean()
    2.5
    """

    def cols(self) -> int:
        """子列数。"""

    def rows(self) -> int:
        """行数。"""

    @property
    def _owns_data(self) -> bool:
        """内部：所有权标记。"""

    @_owns_data.setter
    def _owns_data(self, val: bool) -> None: ...

    def get_size(self) -> int:
        """元素总数。"""

    @classmethod
    def from_2d(cls, values: object, rows: int, cols: int) -> "DoubleColumn":
        """从二维结构构造 2D 列。"""

    @classmethod
    def from_list(cls, data: List[float]) -> "DoubleColumn":
        """从 Python 列表构造。

        Examples
        --------
        >>> DoubleColumn.from_list([1.0, 2.0])
        """

    @classmethod
    def _load_from_1d_array(cls, arr: object) -> "DoubleColumn":
        """内部：从一维数组加载。"""

    @classmethod
    def _load_from_2d_array(cls, arr: object, col_names: List[str]) -> "DoubleColumn":
        """内部：从二维数组加载。"""

    def append(self, val: float) -> None:
        """追加一个元素。"""

    def copy(self) -> "DoubleColumn":
        """深拷贝。"""

    def pop_back(self) -> None:
        """删除末元素。"""

    def pop_front(self) -> None:
        """删除首元素。"""

    def concat(self, other: "DoubleColumn") -> None:
        """拼接另一浮点列。"""

    def max(self) -> float:
        """最大值。"""

    def min(self) -> float:
        """最小值。"""

    def sum(self) -> float:
        """求和。"""

    def mean(self) -> float:
        """均值。"""

    def var(self) -> float:
        """方差。"""

    def std(self) -> float:
        """标准差。"""

    def __getitem__(self, idx: int) -> float:
        """按行号取元素。"""

    def idxmax(self) -> int:
        """最大值位置。"""

    def idxmin(self) -> int:
        """最小值位置。"""

    def median(self) -> float:
        """中位数。"""

    def mad(self) -> float:
        """平均绝对偏差。"""

    def prod(self) -> float:
        """乘积。"""

    def mode(self) -> List[float]:
        """众数。"""

    def quantile(self, q: float = 0.5) -> float:
        """分位数。"""

    def cumsum(self) -> "DoubleColumn":
        """累计求和。"""

    def cummax(self) -> "DoubleColumn":
        """累计最大值。"""

    def cummin(self) -> "DoubleColumn":
        """累计最小值。"""

    def cumprod(self) -> "DoubleColumn":
        """累计乘积。"""

    def diff(self, periods: int = 1) -> "DoubleColumn":
        """差分。"""

    def skew(self) -> float:
        """偏度。"""

    def kurt(self) -> float:
        """峰度。"""

    def sem(self) -> float:
        """标准误。"""

    def count(self) -> int:
        """非 NaN 计数。"""

    def add_val(self, val: float) -> "DoubleColumn":
        """逐元素加标量。"""

    def sub_val(self, val: float) -> "DoubleColumn":
        """逐元素减标量。"""

    def mul_val(self, val: float) -> "DoubleColumn":
        """逐元素乘标量。"""

    def div_val(self, val: float) -> "DoubleColumn":
        """逐元素除标量。"""

    def add(self, other: object) -> "DoubleColumn":
        """框列或标量相加。"""

    def sub(self, other: object) -> "DoubleColumn":
        """框列或标量相减。"""

    def mul(self, other: object) -> "DoubleColumn":
        """框列或标量相乘。"""

    def div(self, other: object) -> "DoubleColumn":
        """框列或标量相除。"""

    def __add__(self, other: object) -> "DoubleColumn":
        """``col + x``。"""

    def __radd__(self, other: object) -> "DoubleColumn":
        """``x + col``。"""

    def __sub__(self, other: object) -> "DoubleColumn":
        """``col - x``。"""

    def __rsub__(self, other: object) -> "DoubleColumn":
        """``x - col``。"""

    def __mul__(self, other: object) -> "DoubleColumn":
        """``col * x``。"""

    def __rmul__(self, other: object) -> "DoubleColumn":
        """``x * col``。"""

    def __truediv__(self, other: object) -> "DoubleColumn":
        """``col / x``。"""

    def __rtruediv__(self, other: object) -> "DoubleColumn":
        """``x / col``。"""

    def unique(self) -> "DoubleColumn":
        """去重。"""

    def __lt__(self, val: object) -> "BoolColumn":
        """``col < x``。"""

    def __le__(self, val: object) -> "BoolColumn":
        """``col <= x``。"""

    def __gt__(self, val: object) -> "BoolColumn":
        """``col > x``。"""

    def __ge__(self, val: object) -> "BoolColumn":
        """``col >= x``。"""

    def __eq__(self, val: object) -> "BoolColumn":  # type: ignore[override]
        """``col == x``。"""

    def __ne__(self, val: object) -> "BoolColumn":  # type: ignore[override]
        """``col != x``。"""

    def __and__(self, other: object) -> "DoubleColumn":
        """按位与。"""

    def __or__(self, other: object) -> "DoubleColumn":
        """按位或。"""

    def __invert__(self) -> "DoubleColumn":
        """按位取反。"""

    def isin(self, values: Set[float]) -> "BoolColumn":
        """元素是否在集合中。"""

    def between(self, low: float, high: float) -> "BoolColumn":
        """``low <= x <= high``。"""

    def __str__(self) -> str:
        """字符串表示。"""

    def __bool__(self) -> bool:
        """是否非空。"""

    def __len__(self) -> int:
        """行数。"""

    def clip(self, lower: object = None, upper: object = None, inplace: bool = False) -> "DoubleColumn":
        """截断。"""

    def abs(self) -> "DoubleColumn":
        """绝对值。"""

    def round(self, decimals: int = 0) -> "DoubleColumn":
        """四舍五入。"""

    def ffill(self) -> "DoubleColumn":
        """前向填充。"""

    def bfill(self) -> "DoubleColumn":
        """后向填充。"""

    def where(self, cond: "IntColumn", other: float, inplace: bool = False) -> "DoubleColumn":
        """三目选择。"""

    def where_mask(self, mask: "IntColumn", fill: object) -> "DoubleColumn":
        """掩码替换。"""

    def mask(self, cond: "IntColumn", other: float, inplace: bool = False) -> "DoubleColumn":
        """反三目选择。"""

    def drop_row(self, row_idx: int) -> None:
        """删除行。"""

    def shift_left(self, pos: int) -> None:
        """左移。"""

    def shift_right(self, pos: int) -> "DoubleColumn":
        """右移。"""

    def euclidean_dist(self, other: object) -> float:
        """欧氏距离。"""

    def manhattan_dist(self, other: object) -> float:
        """曼哈顿距离。"""

    def cosine_dist(self, other: object) -> float:
        """余弦距离。"""

    def cosine_similarity(self, other: object) -> float:
        """余弦相似度。"""

    def jaccard_sim(self, other: object) -> float:
        """Jaccard 相似度。"""

    def hamming_dist(self, other: object) -> float:
        """汉明距离。"""

    def astype(self, target: object) -> BaseColumn:
        """类型转换。"""

    def roll_mean(self, window: int, min_periods: Optional[int] = None, center: bool = False) -> "DoubleColumn":
        """滚动均值。"""

    def roll_sum(self, window: int, min_periods: Optional[int] = None, center: bool = False) -> "DoubleColumn":
        """滚动求和。"""

    def roll_min(self, window: int, min_periods: Optional[int] = None, center: bool = False) -> "DoubleColumn":
        """滚动最小值。"""

    def roll_max(self, window: int, min_periods: Optional[int] = None, center: bool = False) -> "DoubleColumn":
        """滚动最大值。"""

    def roll_var(self, window: int, ddof: bool = True, min_periods: Optional[int] = None, center: bool = False) -> "DoubleColumn":
        """滚动方差。"""

    def roll_std(self, window: int, ddof: bool = True, min_periods: Optional[int] = None, center: bool = False) -> "DoubleColumn":
        """滚动标准差。"""

    def roll_quantile(self, window: int, q: float, min_periods: Optional[int] = None, center: bool = False) -> "DoubleColumn":
        """滚动分位数。"""

    def roll_median(self, window: int, min_periods: Optional[int] = None, center: bool = False) -> "DoubleColumn":
        """滚动中位数。"""

    def roll_skew(self, window: int, min_periods: Optional[int] = None, center: bool = False) -> "DoubleColumn":
        """滚动偏度。"""

    def roll_kurt(self, window: int, min_periods: Optional[int] = None, center: bool = False) -> "DoubleColumn":
        """滚动峰度。"""

    def roll_corr(self, window: int, other: object, min_periods: Optional[int] = None, center: bool = False) -> "DoubleColumn":
        """滚动相关系数。"""

    def roll_cov(self, window: int, other: object, min_periods: Optional[int] = None, center: bool = False) -> "DoubleColumn":
        """滚动协方差。"""

    def ewm_mean(self, alpha: float, min_periods: int = 0, adjust: bool = True, ignore_na: bool = False) -> "DoubleColumn":
        """指数加权均值。"""

    def ewm_sum(self, alpha: float, min_periods: int = 0, adjust: bool = True, ignore_na: bool = False) -> "DoubleColumn":
        """指数加权求和。"""

    def ewm_var(self, alpha: float, bias: bool = False, min_periods: int = 0, adjust: bool = True, ignore_na: bool = False) -> "DoubleColumn":
        """指数加权方差。"""

    def ewm_std(self, alpha: float, bias: bool = False, min_periods: int = 0, adjust: bool = True, ignore_na: bool = False) -> "DoubleColumn":
        """指数加权标准差。"""

    def ewm_cov(self, other: object, alpha: float, bias: bool = False, min_periods: int = 0, adjust: bool = True, ignore_na: bool = False) -> "DoubleColumn":
        """指数加权协方差。"""

    def ewm_corr(self, other: object, alpha: float, min_periods: int = 0, adjust: bool = True, ignore_na: bool = False) -> "DoubleColumn":
        """指数加权相关系数。"""

    def roll_win_mean(self, window: int, weights: object, min_periods: Optional[int] = None, center: bool = False) -> "DoubleColumn":
        """窗口加权均值。"""

    def roll_win_sum(self, window: int, weights: object, min_periods: Optional[int] = None, center: bool = False) -> "DoubleColumn":
        """窗口加权求和。"""

    def roll_win_var(self, window: int, weights: object, ddof: bool = True, min_periods: Optional[int] = None, center: bool = False) -> "DoubleColumn":
        """窗口加权方差。"""

    def roll_win_std(self, window: int, weights: object, ddof: bool = True, min_periods: Optional[int] = None, center: bool = False) -> "DoubleColumn":
        """窗口加权标准差。"""


# ===========================================================================
#  ASCII 定长字符串列 (AsciiColumn)
# ===========================================================================

class AsciiColumn(BaseColumn):
    """256 字节定宽 ASCII 字符串列。

    提供完整的字符串方法（对标 pandas 的 ``str`` accessor）：
    大小写、查找、切片、正则、填充对齐、拆分区等。

    Examples
    --------
    >>> from fluidframe.core.asciicolumn import AsciiColumn
    >>> col = AsciiColumn.from_list(['apple', 'banana', 'cherry'])
    >>> col.upper()
    >>> col.startswith('a')
    """

    def get_size(self) -> int:
        """元素总数。"""

    @classmethod
    def from_list(cls, data: List[str]) -> "AsciiColumn":
        """从 Python 字符串列表构造。

        Parameters
        ----------
        data: List[str]
            元素（需为 ASCII 或可编码，超长会截断/报错）。

        Returns
        -------
        AsciiColumn
            列。

        Examples
        --------
        >>> AsciiColumn.from_list(['a', 'b'])
        """

    def append(self, val: str) -> None:
        """追加一个字符串。"""

    def copy(self) -> "AsciiColumn":
        """深拷贝。"""

    def ffill(self) -> "AsciiColumn":
        """前向填充。"""

    def bfill(self) -> "AsciiColumn":
        """后向填充。"""

    def pop_back(self) -> None:
        """删除末元素。"""

    def pop_front(self) -> None:
        """删除首元素。"""

    def concat(self, other: "AsciiColumn") -> None:
        """拼接另一 ASCII 列。"""

    def __getitem__(self, idx: int) -> str:
        """按行号取元素。

        Returns
        -------
        str
            字符串值。

        Examples
        --------
        >>> col[0]
        'apple'
        """

    def contains(self, substring: str) -> "BoolColumn":
        """是否包含子串。

        Parameters
        ----------
        substring: str
            子串。

        Returns
        -------
        BoolColumn
            布尔掩码。

        Examples
        --------
        >>> col.contains('pp')
        """

    def replace(self, old_str: str, new_str: str, count: int = -1) -> "AsciiColumn":
        """替换子串。

        Parameters
        ----------
        old_str: str
            原子串。
        new_str: str
            新子串。
        count: int, 默认 -1
            替换次数（-1 表示全部）。
        """

    def replaces(self, *args: object) -> "AsciiColumn":
        """批量替换（多组 old/new）。"""

    def extract(self, pattern: str, group: int = 1) -> "AsciiColumn":
        """正则提取分组。

        Parameters
        ----------
        pattern: str
            正则表达式。
        group: int, 默认 1
            捕获组编号。
        """

    def extract_twice(self, pattern: str) -> "AsciiColumn":
        """正则提取两次（两列拼接返回字符串列）。"""

    def len(self) -> "IntColumn":
        """各串字符长度。

        Returns
        -------
        IntColumn
            长度列。

        Examples
        --------
        >>> col.len()
        """

    def len_list(self) -> "IntColumn":
        """各串长度（列表语义别名）。"""

    def startswith(self, prefix: object) -> "BoolColumn":
        """是否以指定前缀开头。"""

    def endswith(self, suffix: object) -> "BoolColumn":
        """是否以指定后缀结尾。"""

    def lower(self) -> "AsciiColumn":
        """转小写。"""

    def upper(self) -> "AsciiColumn":
        """转大写。"""

    def isdigit(self) -> "BoolColumn":
        """是否全为数字。"""

    def isalpha(self) -> "BoolColumn":
        """是否全为字母。"""

    def isalnum(self) -> "BoolColumn":
        """是否全为字母数字。"""

    def isspace(self) -> "BoolColumn":
        """是否全为空白。"""

    def islower(self) -> "BoolColumn":
        """是否全小写。"""

    def isupper(self) -> "BoolColumn":
        """是否全大写。"""

    def strip(self) -> "AsciiColumn":
        """去除首尾空白。"""

    def lstrip(self) -> "AsciiColumn":
        """去除首部空白。"""

    def rstrip(self) -> "AsciiColumn":
        """去除尾部空白。"""

    def removeprefix(self, prefix: str) -> "AsciiColumn":
        """去除前缀。"""

    def removesuffix(self, suffix: str) -> "AsciiColumn":
        """去除后缀。"""

    def zfill(self, width: int) -> "AsciiColumn":
        """前导零填充到宽度。"""

    def find(self, substring: str) -> "IntColumn":
        """子串首次出现位置 (-1 表示未找到)。"""

    def rfind(self, substring: str) -> "IntColumn":
        """子串最后出现位置。"""

    def rindex(self, substring: str) -> "IntColumn":
        """子串最后一次出现位置（未找到抛错）。"""

    def count(self, substring: str) -> "IntColumn":
        """子串出现次数。"""

    def match(self, pattern: str) -> "BoolColumn":
        """正则匹配（从开头）。"""

    def fullmatch(self, pattern: str) -> "BoolColumn":
        """正则完全匹配。"""

    def isnumeric(self) -> "BoolColumn":
        """是否全为数字（Unicode 语义）。"""

    def isdecimal(self) -> "BoolColumn":
        """是否全为十进制数字。"""

    def title(self) -> "AsciiColumn":
        """首字母大写。"""

    def capitalize(self) -> "AsciiColumn":
        """句首大写。"""

    def swapcase(self) -> "AsciiColumn":
        """交换大小写。"""

    def casefold(self) -> "AsciiColumn":
        """casefold 折叠。"""

    def center(self, width: int, fillchar: str = ' ') -> "AsciiColumn":
        """居中填充。"""

    def ljust(self, width: int, fillchar: str = ' ') -> "AsciiColumn":
        """左对齐填充。"""

    def rjust(self, width: int, fillchar: str = ' ') -> "AsciiColumn":
        """右对齐填充。"""

    def pad(self, width: int, side: str = 'left', fillchar: str = ' ') -> "AsciiColumn":
        """填充对齐。

        Parameters
        ----------
        width: int
            目标宽度。
        side: {'left','right','both'}, 默认 'left'
            侧向。
        fillchar: str, 默认 ' '
            填充字符。
        """

    def rsplit(self, sep: Optional[str] = None, maxsplit: int = -1) -> Any:
        """从右拆分。

        Parameters
        ----------
        sep: str, 默认 None (空白)
            分隔符。
        maxsplit: int, 默认 -1
            最大拆分次数。

        Returns
        -------
        拆分后的列表列或扩展列。
        """

    def partition(self, sep: str) -> Any:
        """按首现分隔符三分。"""

    def rpartition(self, sep: str) -> Any:
        """按末现分隔符三分。"""

    def get(self, index: int, default: str = '') -> "AsciiColumn":
        """按字符位置取字符。

        Parameters
        ----------
        index: int
            字符位置（可为负）。
        default: str, 默认 ''
            越界默认值。
        """

    def slice(self, start: int = 0, stop: int = -1, step: int = 1) -> "AsciiColumn":
        """字符切片。"""

    def cat(self, other: object, sep: str = '') -> "AsciiColumn":
        """串接两列字符串。

        Parameters
        ----------
        other: 列或列表
            右侧串。
        sep: str, 默认 ''
            分隔符。
        """

    def prefix(self, pre: str) -> "AsciiColumn":
        """加前缀。"""

    def suffix(self, suf: str) -> "AsciiColumn":
        """加后缀。"""

    def repeat(self, repeats: int) -> "AsciiColumn":
        """重复字符串。"""

    def findall(self, pattern: str) -> Any:
        """正则查找所有匹配。"""

    def extractall(self, pattern: str, group: int = 1) -> Any:
        """正则提取所有分组。"""

    def unique(self) -> "AsciiColumn":
        """去重。"""

    def __lt__(self, val: object) -> "BoolColumn":
        """字典序小于。"""

    def __le__(self, val: object) -> "BoolColumn":
        """字典序小于等于。"""

    def __gt__(self, val: object) -> "BoolColumn":
        """字典序大于。"""

    def __ge__(self, val: object) -> "BoolColumn":
        """字典序大于等于。"""

    def __eq__(self, val: object) -> "BoolColumn":  # type: ignore[override]
        """等于。"""

    def __ne__(self, val: object) -> "BoolColumn":  # type: ignore[override]
        """不等于。"""

    def __and__(self, other: object) -> "BoolColumn":
        """按位与（视为掩码）。"""

    def __or__(self, other: object) -> "BoolColumn":
        """按位或（视为掩码）。"""

    def __invert__(self) -> "BoolColumn":
        """按位取反（掩码）。"""

    def isin(self, values: Set[str]) -> "BoolColumn":
        """是否在集合中。"""

    def where_mask(self, mask: "IntColumn", fill: object) -> "AsciiColumn":
        """掩码替换。"""

    def __str__(self) -> str:
        """字符串表示。"""

    def __bool__(self) -> bool:
        """是否非空。"""

    def __len__(self) -> int:
        """行数。"""

    def drop_row(self, row_idx: int) -> None:
        """删除行。"""

    def shift_left(self, pos: int) -> None:
        """左移。"""

    def astype(self, target: object) -> BaseColumn:
        """类型转换（如 DateColumn）。"""

    def str_split(self, sep: Optional[str] = None, maxsplit: int = -1) -> Any:
        """拆分（str accessor 调用入口）。"""

    def str_getitem(self, key: object) -> Any:
        """字符串下标索引。"""

    def str_contains_mask(self, pat: object) -> "BoolColumn":
        """掩码式包含（str accessor 内部）。"""


# ===========================================================================
#  UCS2 变长 UTF-8 字符串列 (UCS2Column)
# ===========================================================================

class UCS2Column(BaseColumn):
    """变长 UTF-8 字符串列。

    与 ``AsciiColumn`` 字符串方法几乎一致，且不经受 256 字节定宽限制，
    适合任意 Unicode 文本。示例与 AsciiColumn 相同，故不重复。

    Examples
    --------
    >>> from fluidframe.core.ucs2column import UCS2Column
    >>> col = UCS2Column.from_list(['中', '文', '测试'])
    >>> col.contains('文')
    """

    def get_size(self) -> int:
        """元素总数。"""

    @classmethod
    def from_list(cls, data: List[str]) -> "UCS2Column":
        """从 Python 字符串列表构造。"""

    def append(self, val: str) -> None:
        """追加字符串（禁止 None）。"""

    def copy(self) -> "UCS2Column":
        """深拷贝。"""

    def ffill(self) -> "UCS2Column":
        """前向填充。"""

    def bfill(self) -> "UCS2Column":
        """后向填充。"""

    def concat(self, other: "UCS2Column") -> None:
        """拼接另一 UCS2 列。"""

    def __getitem__(self, idx: int) -> str:
        """按行号取元素。"""

    def pop_back(self) -> None:
        """删除末元素。"""

    def pop_front(self) -> None:
        """删除首元素。"""

    def len(self) -> "IntColumn":
        """各串字符长度。"""

    def startswith(self, prefix: str) -> "BoolColumn":
        """是否以前缀开头。"""

    def endswith(self, suffix: str) -> "BoolColumn":
        """是否以后缀结尾。"""

    def contains(self, pat: object) -> "BoolColumn":
        """是否包含子串/模式。"""

    def find(self, substring: str) -> "IntColumn":
        """子串首现位置。"""

    def replace(self, old: str, new: str, count: int = -1) -> "UCS2Column":
        """替换子串。"""

    def lower(self) -> "UCS2Column":
        """转小写。"""

    def upper(self) -> "UCS2Column":
        """转大写。"""

    def strip(self) -> "UCS2Column":
        """去除首尾空白。"""

    def lstrip(self) -> "UCS2Column":
        """去除首部空白。"""

    def rstrip(self) -> "UCS2Column":
        """去除尾部空白。"""

    def count(self, substring: str) -> "IntColumn":
        """子串次数。"""

    def isdigit(self) -> "BoolColumn":
        """是否全数字。"""

    def isalpha(self) -> "BoolColumn":
        """是否全字母。"""

    def isalnum(self) -> "BoolColumn":
        """是否全字母数字。"""

    def isspace(self) -> "BoolColumn":
        """是否全空白。"""

    def rfind(self, substring: str) -> "IntColumn":
        """子串末现位置。"""

    def rindex(self, substring: str) -> "IntColumn":
        """子串最后一次位置。"""

    def title(self) -> "UCS2Column":
        """标题化。"""

    def capitalize(self) -> "UCS2Column":
        """句首大写。"""

    def swapcase(self) -> "UCS2Column":
        """大小写交换。"""

    def casefold(self) -> "UCS2Column":
        """casefold。"""

    def islower(self) -> "BoolColumn":
        """是否全小写。"""

    def isupper(self) -> "BoolColumn":
        """是否全大写。"""

    def isnumeric(self) -> "BoolColumn":
        """是否全数字。"""

    def isdecimal(self) -> "BoolColumn":
        """是否全十进制。"""

    def match(self, pattern: str) -> "BoolColumn":
        """正则开头匹配。"""

    def fullmatch(self, pattern: str) -> "BoolColumn":
        """正则全匹配。"""

    def center(self, width: int, fillchar: str = ' ') -> "UCS2Column":
        """居中。"""

    def ljust(self, width: int, fillchar: str = ' ') -> "UCS2Column":
        """左对齐。"""

    def rjust(self, width: int, fillchar: str = ' ') -> "UCS2Column":
        """右对齐。"""

    def pad(self, width: int, side: str = 'left', fillchar: str = ' ') -> "UCS2Column":
        """填充对齐。"""

    def zfill(self, width: int) -> "UCS2Column":
        """前导零。"""

    def rsplit(self, sep: Optional[str] = None, maxsplit: int = -1) -> Any:
        """从右拆分。"""

    def partition(self, sep: str) -> Any:
        """三分。"""

    def rpartition(self, sep: str) -> Any:
        """右侧三分。"""

    def get(self, index: int, default: str = '') -> "UCS2Column":
        """取字符。"""

    def slice(self, start: int = 0, stop: int = -1, step: int = 1) -> "UCS2Column":
        """切片。"""

    def cat(self, other: object, sep: str = '') -> "UCS2Column":
        """串接。"""

    def prefix(self, pre: str) -> "UCS2Column":
        """加前缀。"""

    def suffix(self, suf: str) -> "UCS2Column":
        """加后缀。"""

    def repeat(self, repeats: int) -> "UCS2Column":
        """重复。"""

    def findall(self, pattern: str) -> Any:
        """查找所有匹配。"""

    def extract(self, pattern: str, group: int = 1) -> "UCS2Column":
        """正则提取。"""

    def extractall(self, pattern: str, group: int = 1) -> Any:
        """提取所有。"""

    def unique(self) -> "UCS2Column":
        """去重。"""

    def __lt__(self, val: object) -> "BoolColumn":
        """字典序小于。"""

    def __le__(self, val: object) -> "BoolColumn":
        """字典序小于等于。"""

    def __gt__(self, val: object) -> "BoolColumn":
        """字典序大于。"""

    def __ge__(self, val: object) -> "BoolColumn":
        """字典序大于等于。"""

    def __eq__(self, val: object) -> "BoolColumn":  # type: ignore[override]
        """等于。"""

    def __ne__(self, val: object) -> "BoolColumn":  # type: ignore[override]
        """不等于。"""

    def __and__(self, other: object) -> "BoolColumn":
        """按位与。"""

    def __or__(self, other: object) -> "BoolColumn":
        """按位或。"""

    def __invert__(self) -> "BoolColumn":
        """按位取反。"""

    def isin(self, values: Set[str]) -> "BoolColumn":
        """是否在集合中。"""

    def where_mask(self, mask: "IntColumn", fill: object) -> "UCS2Column":
        """掩码替换。"""

    def __str__(self) -> str:
        """字符串表示。"""

    def __bool__(self) -> bool:
        """是否非空。"""

    def __len__(self) -> int:
        """行数。"""

    def drop_row(self, row_idx: int) -> None:
        """删除行。"""

    def shift_left(self, pos: int) -> None:
        """左移。"""

    def astype(self, target: object) -> BaseColumn:
        """类型转换。"""

    def str_split(self, sep: Optional[str] = None, maxsplit: int = -1) -> Any:
        """拆分。"""

    def str_getitem(self, key: object) -> Any:
        """字符串下标。"""


# ===========================================================================
#  日期列 (DateColumn)
# ===========================================================================

class DateColumn(BaseColumn):
    """日期列，底层为自 1970-01-01 起的天数的 int32。

    Examples
    --------
    >>> from fluidframe.core.datecolumn import DateColumn
    >>> col = DateColumn.from_list(['2024-01-01', '2024-06-15', '2025-12-31'])
    >>> col.year()
    >>> col.strftime('%Y-%m-%d')
    """

    def cols(self) -> int:
        """子列数。"""

    def rows(self) -> int:
        """行数。"""

    @property
    def _owns_data(self) -> bool:
        """内部：所有权标记。"""

    @_owns_data.setter
    def _owns_data(self, val: bool) -> None: ...

    def get_size(self) -> int:
        """元素总数。"""

    @classmethod
    def from_list(cls, data: List[object]) -> "DateColumn":
        """从日期字符串或 ``datetime.date`` 列表构造。

        Parameters
        ----------
        data: List[str 或 date]
            日期值（'YYYY-MM-DD' 或 datetime.date）。

        Returns
        -------
        DateColumn
            列。
        """

    def append(self, val: object) -> None:
        """追加一个日期（str 或 date 对象）。"""

    def copy(self) -> "DateColumn":
        """深拷贝。"""

    def ffill(self) -> "DateColumn":
        """前向填充。"""

    def bfill(self) -> "DateColumn":
        """后向填充。"""

    def pop_back(self) -> None:
        """删除末元素。"""

    def pop_front(self) -> None:
        """删除首元素。"""

    def concat(self, other: "DateColumn") -> None:
        """拼接另一日期列。"""

    def max(self) -> Date:
        """最新日期。"""

    def min(self) -> Date:
        """最早日期。"""

    def __getitem__(self, idx: int) -> Date:
        """按行号取日期。"""

    def unique(self) -> "DateColumn":
        """去重。"""

    # --- 成分提取 ------------------------------------------------------

    def year(self) -> "IntColumn":
        """年份。"""

    def month(self) -> "IntColumn":
        """月份 (1-12)。"""

    def day(self) -> "IntColumn":
        """日 (1-31)。"""

    def quarter(self) -> "IntColumn":
        """季度 (1-4)。"""

    def strftime(self, fmt: str) -> "AsciiColumn":
        """按格式格式化日期。

        Parameters
        ----------
        fmt: str
            格式化指令（支持 %Y/%m/%d 等；%Y 可安全处理超大年份）。

        Returns
        -------
        AsciiColumn
            格式化字符串列。
        """

    # --- 过滤器 --------------------------------------------------------

    def filter_greater(self, date: object) -> "BoolColumn":
        """> date。"""

    def filter_greater_equal(self, date: object) -> "BoolColumn":
        """>= date。"""

    def filter_less(self, date: object) -> "BoolColumn":
        """< date。"""

    def filter_less_equal(self, date: object) -> "BoolColumn":
        """<= date。"""

    def filter_equal(self, date: object) -> "BoolColumn":
        """== date。"""

    def filter_not_equal(self, date: object) -> "BoolColumn":
        """!= date。"""

    def filter_between(self, start: object, end: object) -> "BoolColumn":
        """start <= x <= end。

        Parameters
        ----------
        start: date 或 str
            下界。
        end: date 或 str
            上界。
        """

    def where_mask(self, mask: "IntColumn", fill: object) -> "DateColumn":
        """掩码替换。"""

    # --- 日期算术 ------------------------------------------------------

    def add_days(self, n: int) -> "DateColumn":
        """加 n 天。

        Parameters
        ----------
        n: int
            天数。
        """

    def sub_days(self, n: int) -> "DateColumn":
        """减 n 天。"""

    def add_months(self, n: int) -> "DateColumn":
        """加 n 个月。"""

    def sub_months(self, n: int) -> "DateColumn":
        """减 n 个月。"""

    def add_years(self, n: int) -> "DateColumn":
        """加 n 年。"""

    def sub_years(self, n: int) -> "DateColumn":
        """减 n 年。"""

    # --- 日历属性 ------------------------------------------------------

    def day_of_week(self) -> "IntColumn":
        """星期几 (0=周一 或 0=周日起，见约定)。"""

    def is_weekend(self) -> "BoolColumn":
        """是否周末。"""

    def day_of_year(self) -> "IntColumn":
        """年内第几天 (1-366)。"""

    def days_in_month(self) -> "IntColumn":
        """当月天数。"""

    def is_leap_year(self) -> "BoolColumn":
        """是否闰年。"""

    def is_month_start(self) -> "BoolColumn":
        """是否月初。"""

    def is_month_end(self) -> "BoolColumn":
        """是否月末。"""

    def is_year_start(self) -> "BoolColumn":
        """是否年初。"""

    def is_year_end(self) -> "BoolColumn":
        """是否年末。"""

    def is_quarter_start(self) -> "BoolColumn":
        """是否季初。"""

    def is_quarter_end(self) -> "BoolColumn":
        """是否季末。"""

    def week_of_year(self) -> "IntColumn":
        """ISO 周数。"""

    def day_name(self) -> "AsciiColumn":
        """星期名（如 'Monday'）。"""

    def month_name(self) -> "AsciiColumn":
        """月份名（如 'January'）。"""

    def last_day_of_month(self) -> "DateColumn":
        """当月最后一天。"""

    def iso_calendar(self) -> Any:
        """ISO 日历 (year, week, weekday) 三列。"""

    # --- 差值 ----------------------------------------------------------

    def diff_days(self, date: object) -> "IntColumn":
        """与指定日期相差天数。"""

    def diff_months(self, date: object) -> "IntColumn":
        """与指定日期相差月数。"""

    def diff_years(self, date: object) -> "IntColumn":
        """与指定日期相差年数。"""

    # --- 比较 ----------------------------------------------------------

    def __lt__(self, val: object) -> "BoolColumn":
        """早于。"""

    def __le__(self, val: object) -> "BoolColumn":
        """早于等于。"""

    def __gt__(self, val: object) -> "BoolColumn":
        """晚于。"""

    def __ge__(self, val: object) -> "BoolColumn":
        """晚于等于。"""

    def __eq__(self, val: object) -> "BoolColumn":  # type: ignore[override]
        """等于。"""

    def __ne__(self, val: object) -> "BoolColumn":  # type: ignore[override]
        """不等于。"""

    def __str__(self) -> str:
        """字符串表示。"""

    def __bool__(self) -> bool:
        """是否非空。"""

    def __len__(self) -> int:
        """行数。"""

    def drop_row(self, row_idx: int) -> None:
        """删除行。"""

    def shift_left(self, pos: int) -> None:
        """左移。"""

    def shift_right(self, pos: int) -> "DateColumn":
        """右移。"""

    def astype(self, target: object) -> BaseColumn:
        """类型转换（如 AsciiColumn / DateTimeColumn）。"""


# ===========================================================================
#  布尔列 (BoolColumn)
# ===========================================================================

class BoolColumn(BaseColumn):
    """布尔列（底层 0/1）。

    Examples
    --------
    >>> from fluidframe.core.boolcolumn import BoolColumn
    >>> col = BoolColumn.from_list([True, False, True])
    >>> col & (col != True)
    """

    def get_size(self) -> int:
        """元素总数。"""

    @classmethod
    def from_list(cls, data: List[object]) -> "BoolColumn":
        """从布尔列表构造。"""

    def append(self, val: object) -> None:
        """追加元素（bool 或 0/1）。"""

    def _append_bool(self, val: bool) -> None:
        """内部：追加原生 bool。"""

    def pop_back(self) -> None:
        """删除末元素。"""

    def pop_front(self) -> None:
        """删除首元素。"""

    def concat(self, other: "BoolColumn") -> None:
        """拼接另一布尔列。"""

    def copy(self) -> "BoolColumn":
        """深拷贝。"""

    def ffill(self) -> "BoolColumn":
        """前向填充。"""

    def bfill(self) -> "BoolColumn":
        """后向填充。"""

    def __getitem__(self, idx: int) -> bool:
        """按行号取元素。"""

    def __setitem__(self, idx: int, val: object) -> None:
        """按行号赋值。"""

    def __len__(self) -> int:
        """行数。"""

    def __bool__(self) -> bool:
        """是否非空。"""

    def isin(self, values: Set[object]) -> "BoolColumn":
        """是否在集合中。"""

    def where_mask(self, mask: "IntColumn", fill: object) -> "BoolColumn":
        """掩码替换。"""

    def __and__(self, other: "BoolColumn") -> "BoolColumn":
        """逻辑与。"""

    def __or__(self, other: "BoolColumn") -> "BoolColumn":
        """逻辑或。"""

    def __xor__(self, other: "BoolColumn") -> "BoolColumn":
        """逻辑异或。"""

    def __invert__(self) -> "BoolColumn":
        """逻辑非。

        Examples
        --------
        >>> ~col
        """

    def __str__(self) -> str:
        """字符串表示。"""

    def __repr__(self) -> str:
        """repr。"""

    def drop_row(self, row_idx: int) -> None:
        """删除行。"""

    def shift_left(self, pos: int) -> None:
        """左移。"""

    def __eq__(self, other: object) -> "BoolColumn":  # type: ignore[override]
        """等于。"""

    def __ne__(self, other: object) -> "BoolColumn":  # type: ignore[override]
        """不等于。"""

    def astype(self, target: object) -> BaseColumn:
        """类型转换（如 IntColumn）。"""


# ===========================================================================
#  分类列 (CategoryColumn)
# ===========================================================================

class CategoryColumn(BaseColumn):
    """分类列：底层维护唯一值字典 + 编码，节省内存。

    Examples
    --------
    >>> from fluidframe.core.categorycolumn import CategoryColumn
    >>> col = CategoryColumn.from_list(['low', 'high', 'low', 'med'])
    >>> col == 'low'
    """

    def get_size(self) -> int:
        """元素总数。"""

    @classmethod
    def from_list(cls, data: List[object]) -> "CategoryColumn":
        """从值列表构造（自动收集分类）。"""

    def append(self, val: object) -> None:
        """追加元素。"""

    def pop_back(self) -> None:
        """删除末元素。"""

    def pop_front(self) -> None:
        """删除首元素。"""

    def concat(self, other: "CategoryColumn") -> None:
        """拼接另一分类列。"""

    def copy(self) -> "CategoryColumn":
        """深拷贝。"""

    def ffill(self) -> "CategoryColumn":
        """前向填充。"""

    def bfill(self) -> "CategoryColumn":
        """后向填充。"""

    def __getitem__(self, idx: int) -> object:
        """按行号取分类值。"""

    def __setitem__(self, idx: int, val: object) -> None:
        """按行号赋值。"""

    def __len__(self) -> int:
        """行数。"""

    def __bool__(self) -> bool:
        """是否非空。"""

    def isin(self, values: Set[object]) -> "BoolColumn":
        """是否在集合中。"""

    def where_mask(self, mask: "IntColumn", fill: object) -> "CategoryColumn":
        """掩码替换。"""

    def __str__(self) -> str:
        """字符串表示。"""

    def __repr__(self) -> str:
        """repr。"""

    def drop_row(self, row_idx: int) -> None:
        """删除行。"""

    def shift_left(self, pos: int) -> None:
        """左移。"""

    def __eq__(self, other: object) -> "BoolColumn":  # type: ignore[override]
        """等于（生成布尔掩码）。"""

    def __ne__(self, other: object) -> "BoolColumn":  # type: ignore[override]
        """不等于。"""

    def astype(self, target: object) -> BaseColumn:
        """类型转换。"""


# ===========================================================================
#  日期时间列 (DateTimeColumn)
# ===========================================================================

class DateTimeColumn(BaseColumn):
    """日期时间列（微秒精度）。

    Examples
    --------
    >>> from fluidframe.core.datetimecolumn import DateTimeColumn
    >>> col = DateTimeColumn.from_list(['2024-01-01 10:00:00', '2024-01-02 12:30:00'])
    >>> col.hour()
    """

    def cols(self) -> int:
        """子列数。"""

    def rows(self) -> int:
        """行数。"""

    @property
    def _owns_data(self) -> bool:
        """内部：所有权标记。"""

    @_owns_data.setter
    def _owns_data(self, val: bool) -> None: ...

    def get_size(self) -> int:
        """元素总数。"""

    @classmethod
    def from_list(cls, data: List[object]) -> "DateTimeColumn":
        """从时间字符串或 ``datetime.datetime`` 列表构造。"""

    def append(self, val: object) -> None:
        """追加一个时间。"""

    def copy(self) -> "DateTimeColumn":
        """深拷贝。"""

    def ffill(self) -> "DateTimeColumn":
        """前向填充。"""

    def bfill(self) -> "DateTimeColumn":
        """后向填充。"""

    def pop_back(self) -> None:
        """删除末元素。"""

    def pop_front(self) -> None:
        """删除首元素。"""

    def concat(self, other: "DateTimeColumn") -> None:
        """拼接另一日期时间列。"""

    def max(self) -> DateTime:
        """最新时间。"""

    def min(self) -> DateTime:
        """最早时间。"""

    def __getitem__(self, idx: int) -> DateTime:
        """按行号取时间。"""

    def unique(self) -> "DateTimeColumn":
        """去重。"""

    def year(self) -> "IntColumn":
        """年份。"""

    def month(self) -> "IntColumn":
        """月份。"""

    def day(self) -> "IntColumn":
        """日。"""

    def hour(self) -> "IntColumn":
        """小时 (0-23)。"""

    def minute(self) -> "IntColumn":
        """分钟 (0-59)。"""

    def second(self) -> "IntColumn":
        """秒 (0-59)。"""

    def where_mask(self, mask: "IntColumn", fill: object) -> "DateTimeColumn":
        """掩码替换。"""

    def isin(self, values: Set[object]) -> "BoolColumn":
        """是否在集合中。"""

    def drop_row(self, row_idx: int) -> None:
        """删除行。"""

    def shift_left(self, pos: int) -> None:
        """左移。"""

    def shift_right(self, pos: int) -> "DateTimeColumn":
        """右移。"""

    def __lt__(self, val: object) -> "BoolColumn":
        """早于。"""

    def __le__(self, val: object) -> "BoolColumn":
        """早于等于。"""

    def __gt__(self, val: object) -> "BoolColumn":
        """晚于。"""

    def __ge__(self, val: object) -> "BoolColumn":
        """晚于等于。"""

    def __eq__(self, val: object) -> "BoolColumn":  # type: ignore[override]
        """等于。"""

    def __ne__(self, val: object) -> "BoolColumn":  # type: ignore[override]
        """不等于。"""

    def __str__(self) -> str:
        """字符串表示。"""

    def __bool__(self) -> bool:
        """是否非空。"""

    def __len__(self) -> int:
        """行数。"""

    def astype(self, target: object) -> BaseColumn:
        """类型转换。"""


# ===========================================================================
#  时间列 (TimeColumn)
# ===========================================================================

class TimeColumn(BaseColumn):
    """时间列（仅时分秒，微秒精度）。

    Examples
    --------
    >>> from fluidframe.core.timecolumn import TimeColumn
    >>> col = TimeColumn.from_list(['10:00:00', '12:30:00'])
    >>> col.hour()
    """

    def get_size(self) -> int:
        """元素总数。"""

    def cols(self) -> int:
        """子列数。"""

    def rows(self) -> int:
        """行数。"""

    @property
    def _owns_data(self) -> bool:
        """内部：所有权标记。"""

    @_owns_data.setter
    def _owns_data(self, val: bool) -> None: ...

    @classmethod
    def from_list(cls, data: List[object]) -> "TimeColumn":
        """从时间字符串或 ``datetime.time`` 列表构造。"""

    def append(self, val: object) -> None:
        """追加一个时间。"""

    def copy(self) -> "TimeColumn":
        """深拷贝。"""

    def ffill(self) -> "TimeColumn":
        """前向填充。"""

    def bfill(self) -> "TimeColumn":
        """后向填充。"""

    def pop_back(self) -> None:
        """删除末元素。"""

    def pop_front(self) -> None:
        """删除首元素。"""

    def concat(self, other: "TimeColumn") -> None:
        """拼接另一时间列。"""

    def max(self) -> Time:
        """最晚时间。"""

    def min(self) -> Time:
        """最早时间。"""

    def __getitem__(self, idx: int) -> Time:
        """按行号取时间。"""

    def unique(self) -> "TimeColumn":
        """去重。"""

    def hour(self) -> "IntColumn":
        """小时。"""

    def minute(self) -> "IntColumn":
        """分钟。"""

    def second(self) -> "IntColumn":
        """秒。"""

    def where_mask(self, mask: "IntColumn", fill: object) -> "TimeColumn":
        """掩码替换。"""

    def isin(self, values: object) -> "BoolColumn":
        """是否在集合中。"""

    def drop_row(self, row_idx: int) -> None:
        """删除行。"""

    def shift_left(self, pos: int) -> None:
        """左移。"""

    def shift_right(self, pos: int) -> "TimeColumn":
        """右移。"""

    def __lt__(self, val: object) -> "BoolColumn":
        """早于。"""

    def __le__(self, val: object) -> "BoolColumn":
        """早于等于。"""

    def __gt__(self, val: object) -> "BoolColumn":
        """晚于。"""

    def __ge__(self, val: object) -> "BoolColumn":
        """晚于等于。"""

    def __eq__(self, val: object) -> "BoolColumn":  # type: ignore[override]
        """等于。"""

    def __ne__(self, val: object) -> "BoolColumn":  # type: ignore[override]
        """不等于。"""

    def __str__(self) -> str:
        """字符串表示。"""

    def __bool__(self) -> bool:
        """是否非空。"""

    def __len__(self) -> int:
        """行数。"""

    def astype(self, target: object) -> BaseColumn:
        """类型转换。"""