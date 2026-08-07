# type: ignore
# =============================================================================
# fixedndframe.pyi — 类型桩 (Type Stub)
#
# 数据模型 (OCR 版面分析行数据):
#   idx      : 行号
#   x0, y0   : 左上角坐标
#   x1, y1   : 右下角坐标
#   width    : 宽 (x1 - x0)
#   height   : 高 (y1 - y0)
#   page     : 页码
#   flags    : 标记位
#   text     : 文本内容
#   text_sep : 分隔文本（可选）
#
# 存储说明:
#   text / text_sep 内部以 UCS2 定长数组存储，所有入口均已完成 UCS2 规范化:
#   超过 BMP (U+FFFF) 的字符 (如 Emoji) 在写入时被降级清洗为 '?'。
#   因此所有字符串匹配/替换操作均以 '?' 为准，不含辅助平面字符。
#
# 构建方法:
#   cd FixedDataFrame
#   python3 -c "from setuptools import setup; from Cython.Build import cythonize; \
#   setup(ext_modules=cythonize('fixedndframe.pyx', language_level=3))" build_ext --build-lib=build
# =============================================================================
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

# 一行数据的字典形态
Row = Dict[str, Any]


# =============================================================================
# NDframe — 核心表格容器（与 SolidDataFrame.ndframe 对外的 API 完全对齐）
#
# 用法示例:
#   import fixedndframe as fnd
#
#   # 方式一：逐行 append
#   df = fnd.NDframe(16)
#   df.append(0, 10, 20, 100, 60, 90, 40, 1, 0, "Hello 世界", "sep")
#   df.append(1, 5, 25, 80, 55, 75, 30, 1, 0, "价格 99.9")
#
#   # 方式二：from_dict 批量构造
#   df2 = fnd.NDframe.from_dict({
#       "x0": [10, 5], "y0": [20, 25], "x1": [100, 80], "y1": [60, 55],
#       "width": [90, 75], "height": [40, 30], "text": ["Hello 世界", "价格 99.9"],
#   })
#
#   # 访问
#   df.to_list()          # -> [{'idx':0, 'x0':10, ..., 'text':'Hello 世界', ...}, ...]
#   df[0]                 # -> 单行 dict
#   df[0:2]               # -> 前两行构成的新 NDframe
#   len(df)               # -> 行数
#   df.text.values()      # -> ['Hello 世界', '价格 99.9']
#   df.x0.values()        # -> [10, 5]
# =============================================================================
class NDframe:
    def __init__(self, capacity: int = 16) -> None: ...

    def __len__(self) -> int: ...
    def __bool__(self) -> bool: ...
    def __iter__(self) -> Iterator[Row]: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def _repr_html_(self) -> str:
        """Jupyter / IPython 富文本 HTML 表格（预览最多 60 行）"""

    # ----- 属性 -----
    @property
    def empty(self) -> bool:
        """是否为空表。示例: df.empty -> False"""
    @property
    def data(self) -> List[Row]:
        """等价于 to_list()，返回全部行字典列表"""
    @property
    def text(self) -> "StringAccessor":
        """text 字段字符串访问器。示例: df.text.contains(r'\\d+')"""
    @property
    def text_sep(self) -> "StringAccessor":
        """text_sep 字段字符串访问器。示例: df.text_sep.join('|')"""
    @property
    def x0(self) -> "NumericAccessor":
        """x0 字段数值访问器。示例: df.x0.max()"""
    @property
    def y0(self) -> "NumericAccessor": ...
    @property
    def x1(self) -> "NumericAccessor": ...
    @property
    def y1(self) -> "NumericAccessor": ...
    @property
    def width(self) -> "NumericAccessor":
        """width 字段数值访问器。示例: df.width.mean()"""
    @property
    def height(self) -> "NumericAccessor":
        """height 字段数值访问器（mode 走专用桶排序快通道）"""
    @property
    def idx(self) -> "NumericAccessor": ...
    @property
    def page(self) -> "NumericAccessor": ...
    @property
    def flags(self) -> "NumericAccessor": ...

    # ----- 索引 / 视图 -----
    def __getitem__(self, key: Union[int, slice]) -> Union[Row, "NDframe"]:
        """df[0] 返回单行 dict；df[1:5] 返回连续切片新 NDframe（不支持 step）"""
    def head(self, n: int = 5) -> "NDframe":
        """返回前 n 行。示例: df.head(3).to_list()"""
    def tail(self, n: int = 5) -> "NDframe":
        """返回后 n 行"""
    def bbox(self, n: int = 0) -> Tuple[int, int, int, int]:
        """返回第 n 行边界框 (x0, y0, x1, y1)，支持负索引"""
    def copy(self) -> "NDframe":
        """深拷贝，返回完全独立的新 NDframe"""

    # ----- 构造 / 输出 -----
    def append(
        self,
        idx: int,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        width: int,
        height: int,
        page: int,
        flags: int,
        text: str,
        text_sep: str = "",
    ) -> None:
        """就地追加一行。text/text_sep 为 None 时按空串处理，入口完成 UCS2 规范化"""
    def to_list(self) -> List[Row]:
        """返回所有行字典列表"""
    @classmethod
    def from_dict(cls, data: Dict[str, Iterable[Any]]) -> "NDframe":
        """从字典批量构造。必填: x0,y0,x1,y1,width,height,text；
        可选: page, flags, text_sep。所有数组长度必须一致"""
    def to_string(self, max_rows: int = 50, name: Optional[str] = None) -> str:
        """构造对齐的 ASCII 表格字符串，超长自动折叠（中间显示 ...）"""
    def print(self, max_rows: int = 50, name: Optional[str] = None) -> None:
        """打印 to_string() 结果"""

    # ----- 删除 / 合并 -----
    def pop_back(self) -> None:
        """就地删除最后一行；空表不操作"""
    @staticmethod
    def concat(df1: "NDframe", df2: "NDframe") -> "NDframe":
        """拼接两个 NDframe，返回新表。
        示例: merged = fnd.NDframe.concat(df_a, df_b)"""
    def df_from_idx(self, indices: Iterable[int]) -> "NDframe":
        """按索引列表重建新表（支持无序/重复）。
        示例: df.df_from_idx([3, 1, 1]).text.values()"""
    def drop_by_idx(self, indices: Iterable[int]) -> "NDframe":
        """就地删除指定索引行，返回原 df"""
    def df_drop(self, index: int) -> "NDframe":
        """返回删除单行后的新 NDframe"""
    def df_drops(self, indices: Iterable[int]) -> "NDframe":
        """返回删除多个索引后的新 NDframe"""
    def drop(self, index: int) -> "NDframe":
        """就地删除单行，返回原 df"""
    def drop_in_bbox(self, x0: int, y0: int, x1: int, y1: int) -> "NDframe":
        """就地删除完全落在 (x0,y0,x1,y1) 内的行（严格大于/小于），返回原 df"""
    def merge_line(
        self, width_threshold: int, height_threshold: int, sep: str = ""
    ) -> "NDframe":
        """按相邻条件合并行（水平拼接文本），返回新表。
        示例: df.merge_line(5, 10, '').text.values()"""

    # ----- 坐标过滤（返回新表）-----
    def filter_in_bbox(self, x0: int, y0: int, x1: int, y1: int) -> "NDframe":
        """返回严格落在框内的行: x0>bx0 且 y0>by0 且 x1<bx1 且 y1<by1"""
    def filter_in_x0x1(self, x0: int, x1: int) -> "NDframe":
        """返回 x 方向在 (x0,x1) 内的行"""
    def filter_in_y0y1(self, y0: int, y1: int) -> "NDframe":
        """返回 y 方向在 (y0,y1) 内的行"""
    def filter_greater_point(self, x0: int, y0: int) -> "NDframe":
        """返回左上角坐标均大于阈值的行: x0>vx 且 y0>vy"""
    def filter_less_point(self, x1: int, y1: int) -> "NDframe":
        """返回右下角坐标均小于阈值的行: x1<vx 且 y1<vy"""

    # ----- 坐标索引提取（返回索引列表）-----
    def idx_in_bbox(self, x0: int, y0: int, x1: int, y1: int) -> List[int]:
        """返回落在框内的行索引列表。示例: df.idx_in_bbox(0, 0, 200, 100)"""
    def idx_in_x0x1(self, x0: int, x1: int) -> List[int]:
        """返回 x 方向在区间内的行索引列表"""
    def idx_in_y0y1(self, y0: int, y1: int) -> List[int]:
        """返回 y 方向在区间内的行索引列表"""

    # ----- 排序 -----
    def sort_by_x0(self, descending: bool = False) -> "NDframe":
        """按 x0 就地排序，返回原 df。示例: df.sort_by_x0().text.values()"""
    def sort_by_y0(self, descending: bool = False) -> "NDframe": ...
    def sort_by_x1(self, descending: bool = False) -> "NDframe": ...
    def sort_by_y1(self, descending: bool = False) -> "NDframe": ...
    def sort_by_width(self, descending: bool = False) -> "NDframe": ...
    def sort_by_height(self, descending: bool = False) -> "NDframe": ...
    def sort_by_idx(self, descending: bool = False) -> "NDframe": ...
    def sort_by_page(self, descending: bool = False) -> "NDframe": ...
    def sort_by_flags(self, descending: bool = False) -> "NDframe": ...
    def sort_by_text(self, descending: bool = False) -> "NDframe":
        """按 text 字段 UCS2 码元字典序就地排序（稳定）。
        示例: df.sort_by_text().text.values()"""
    def sort_by_text_sep(self, descending: bool = False) -> "NDframe":
        """按 text_sep 字段 UCS2 码元字典序就地排序（稳定）"""
    def sort_by(self, field: str = "x0", descending: bool = False) -> "NDframe":
        """按字段名字符串就地排序（支持 text/text_sep 字典序）。
        示例: df.sort_by('height', True)"""

    # ----- 就地整列运算 -----
    def reset_index(self) -> "NDframe":
        """idx 重置为 0..n-1，返回原 df"""
    def add_y0y1(self, offset: int) -> "NDframe":
        """y0/y1 同步平移 offset。示例: df.add_y0y1(-5)"""
    def apply_if_greater_add(
        self, cond_field: str, target_field: str, threshold: int, offset: int
    ) -> "NDframe":
        """若 cond_field > threshold，则 target_field += offset。示例: df.apply_if_greater_add('x0','x0',50,-10)"""
    def apply_if_less_add(
        self, cond_field: str, target_field: str, threshold: int, offset: int
    ) -> "NDframe":
        """若 cond_field < threshold，则 target_field += offset"""
    def apply_if_greater_add_y(
        self, cond_field: str, threshold: int, offset: int
    ) -> "NDframe":
        """若 cond_field > threshold，则 y0/y1 同步 += offset"""
    def apply_if_less_add_y(
        self, cond_field: str, threshold: int, offset: int
    ) -> "NDframe":
        """若 cond_field < threshold，则 y0/y1 同步 += offset"""

    # ----- 几何统计 -----
    def centroid(self) -> Tuple[float, float]:
        """所有矩形中心点平均坐标（质心）；空表返回 (0.0, 0.0)"""
    def centroid_line(self, n: int) -> Tuple[float, float]:
        """第 n 行矩形中心点，支持负索引。示例: df.centroid_line(-1)"""


# =============================================================================
# StringAccessor — 字符串字段访问器 (df.text.xxx / df.text_sep.xxx)
#
# 分组速览:
#   过滤返回新表 : contains / contains_substr / contains_in_bbox / startswith /
#                 endswith / is_* / is_chinese / is_numeric_punct /
#                 strlen_* / is_in / not_in / is_equal / regex_strlen_*
#   就地变换     : strip / lstrip / rstrip / capitalize / swapcase /
#                 keep_chinese / keep_numeric_punct / replace / regex_replace
#   正则提取     : regex_extract
#   索引提取     : idx_contains / idx_contains_in_bbox / idx_strlen_* /
#                 idx_regex_strlen_*
#   其他         : values / len / count / join / slice / contains_any /
#                 drop_regex / replace_to / regex_replace_to
#
# 用法示例:
#   df.text.contains(r'^价格')        # 正则过滤，返回新 NDframe
#   df.text.contains_substr('价格')   # 纯 C 子串匹配（更快）
#   df.text.startswith('第')          # 前缀过滤
#   df.text.strlen_greater(5)         # 长度 > 5 的行
#   df.text.is_chinese()              # 全中文字符的行
#   df.text.keep_numeric_punct()      # 只保留数字和 . , (就地)
#   df.text.replace('价格', '金额')     # 就地替换
#   df.text.regex_extract(r'(\\d+)')   # 提取第一个捕获组，返回新表
#   df.text.join('|')                 # 用 | 拼接全部文本
#   df.text.idx_contains(r'金额')      # 返回匹配行的索引列表
# =============================================================================
class StringAccessor:
    @property
    def field_name(self) -> str:
        """字段名，'text' 或 'text_sep'"""

    def __iter__(self) -> Iterator[str]:
        """逐行迭代字符串值。示例: [s for s in df.text]"""
    def __repr__(self) -> str:
        """预览前 5 行: text['...', ...] (n=行数)"""

    # ----- 基础工具 -----
    def values(self, index: Optional[int] = None) -> Union[List[str], str]:
        """不传参返回全部值列表；传索引返回单行值（支持负数）。
        示例: df.text.values(-1)"""
    def len(self) -> List[int]:
        """每行字符串长度列表。示例: df.text.len() -> [9, 10, 7, 0]"""
    def count(self, sub: str) -> int:
        """所有行中 sub 出现总次数（纯 C 滑动窗口匹配）。示例: df.text.count('99')"""
    def slice(self, start=None, stop=None, step=None) -> List[str]:
        """对每行字符串做切片。示例: df.text.slice(0, 5) -> 每行前 5 字符"""
    def join(self, separator: str = "") -> str:
        """用 separator 拼接全部文本。示例: df.text.join('|')"""
    def contains_any(self, texts: Iterable[str]) -> bool:
        """是否存在某行包含给定集合中任意一个子串（子串匹配）"""
    def is_equal(self, s: str) -> "NDframe":
        """返回字段值恰好等于 s 的行（纯 C memcmp 比较）"""

    # ----- 过滤：正则 / 子串 / 前后缀 -----
    def contains(self, pattern: Union[str, Any]) -> "NDframe":
        """正则过滤（自动编译），返回新 NDframe。示例: df.text.contains(r'\\d{2,}')"""
    def contains_substr(self, s: str) -> "NDframe":
        """纯 C 子串匹配过滤，返回新 NDframe（无正则开销）"""
    def contains_in_bbox(
        self, pattern: Union[str, Any], x0: int, y0: int, x1: int, y1: int
    ) -> "NDframe":
        """坐标框内正则过滤，返回新 NDframe"""
    def startswith(self, prefix: str) -> "NDframe":
        """前缀过滤。示例: df.text.startswith('第')"""
    def endswith(self, suffix: str) -> "NDframe":
        """后缀过滤"""
    def drop_regex(self, pattern: Union[str, Any]) -> "NDframe":
        """就地删除匹配正则的行，返回原 df"""

    # ----- 过滤：字符串属性判定 (is_*) -----
    def isalnum(self) -> "NDframe": ...
    def isalpha(self) -> "NDframe": ...
    def isdigit(self) -> "NDframe": ...
    def isspace(self) -> "NDframe": ...
    def islower(self) -> "NDframe": ...
    def isupper(self) -> "NDframe": ...
    def istitle(self) -> "NDframe": ...
    def isnumeric(self) -> "NDframe": ...
    def isdecimal(self) -> "NDframe": ...
    def is_chinese(self) -> "NDframe":
        """全为中文字符的行。示例: df.text.is_chinese().to_list()"""
    def is_numeric_punct(self) -> "NDframe":
        """全为数字或 [.,] 的行"""
    def is_in(self, texts: Iterable[str]) -> "NDframe":
        """字段值在给定集合中的行。示例: df.text.is_in(['是', '否'])"""
    def not_in(self, texts: Iterable[str]) -> "NDframe":
        """字段值不在给定集合中的行"""

    # ----- 过滤：字符串长度 -----
    def strlen_greater(self, n: int) -> "NDframe":
        """长度 > n。示例: df.text.strlen_greater(10)"""
    def strlen_less(self, n: int) -> "NDframe":
        """长度 < n"""
    def strlen_equal(self, n: int) -> "NDframe":
        """长度 == n"""
    def regex_strlen_greater(self, pattern: Union[str, Any], n: int) -> "NDframe":
        """正则匹配 && 长度 > n"""
    def regex_strlen_less(self, pattern: Union[str, Any], n: int) -> "NDframe":
        """正则匹配 && 长度 < n"""
    def regex_strlen_equal(self, pattern: Union[str, Any], n: int) -> "NDframe":
        """正则匹配 && 长度 == n"""

    # ----- 就地变换 -----
    def strip(self, chars: Optional[str] = None) -> "NDframe":
        """去掉首尾空白（或 chars 中字符），就地，返回原 df。示例: df.text.strip()"""
    def lstrip(self, chars: Optional[str] = None) -> "NDframe":
        """去掉左侧空白/字符"""
    def rstrip(self, chars: Optional[str] = None) -> "NDframe":
        """去掉右侧空白/字符"""
    def capitalize(self) -> "NDframe":
        """首字母大写其余小写"""
    def swapcase(self) -> "NDframe":
        """大小写互换"""
    def keep_chinese(self) -> "NDframe":
        """只保留中文字符（就地）。示例: df.text.keep_chinese().values()"""
    def keep_numeric_punct(self) -> "NDframe":
        """只保留数字和 . ,（就地）"""
    def replace(
        self,
        old_str: str,
        new_str: str,
        old_str2: str = "",
        new_str2: str = "",
    ) -> "NDframe":
        """就地替换（支持可选第二对），返回原 df 便于链式调用。
        示例: df.text.replace(' ', '_').replace('，', ',')"""
    def regex_replace(self, pattern: Union[str, Any], rewrite: str) -> "NDframe":
        """就地正则替换。示例: df.text.regex_replace(r'\\d+', '0')"""
    def regex_extract(self, pattern: Union[str, Any]) -> "NDframe":
        """提取第一个捕获组写入 text 字段，返回新 NDframe。
        示例: df.text.regex_extract(r'([A-Za-z]+)')"""

    # ----- 跨字段替换 -----
    def replace_to(
        self, old_str: str, new_str: str, target_field: str = "text"
    ) -> "NDframe":
        """本字段替换后写入 target_field（就地）。
        示例: df.text_sep.replace_to("\\n", " ", "text")"""
    def regex_replace_to(
        self, pattern: Union[str, Any], rewrite: str, target_field: str = "text"
    ) -> "NDframe":
        """正则替换后写入 target_field（就地）"""

    # ----- 索引提取 (idx_*) -----
    def idx_contains(self, pattern: Union[str, Any]) -> List[int]:
        """返回匹配正则的行索引列表。示例: df.text.idx_contains(r'价格') -> [1, 5]"""
    def idx_contains_in_bbox(
        self, pattern: Union[str, Any], x0: int, y0: int, x1: int, y1: int
    ) -> List[int]:
        """坐标框内匹配正则的行索引列表"""
    def idx_strlen_greater(self, n: int) -> List[int]:
        """长度 > n 的行索引"""
    def idx_strlen_less(self, n: int) -> List[int]:
        """长度 < n 的行索引"""
    def idx_strlen_equal(self, n: int) -> List[int]:
        """长度 == n 的行索引"""
    def idx_regex_strlen_greater(self, pattern: Union[str, Any], n: int) -> List[int]:
        """正则匹配 && 长度 > n 的行索引"""
    def idx_regex_strlen_less(self, pattern: Union[str, Any], n: int) -> List[int]:
        """正则匹配 && 长度 < n 的行索引"""
    def idx_regex_strlen_equal(self, pattern: Union[str, Any], n: int) -> List[int]:
        """正则匹配 && 长度 == n 的行索引"""


# =============================================================================
# NumericAccessor — 数值字段访问器 (df.x0.xxx / df.height.xxx)
#
# 分组速览:
#   统计: max / min / mean / mode / var / std / diff
#   过滤: greater / less / between         (返回新表)
#   就地: add / sub / sort
#   索引: idx_greater / idx_less / idx_equal / idx_between
#
# 用法示例:
#   df.x0.max()                     # 最大值
#   df.width.mean()                 # 均值
#   df.height.mode()                # (众数值, 出现次数)，height 走桶计数快通道
#   df.y0.diff()                    # 一阶差分 [None, d1, d2, ...]
#   df.x0.greater(50)               # x0 > 50 的行（新表）
#   df.height.between(10, 30)       # 10 < height < 30 的行（新表）
#   df.width.add(2)                 # 就地 width += 2
#   df.height.sort(descending=True) # 就地按 height 降序排序
#   df.x0.idx_greater(50)           # 满足条件的行索引列表
# =============================================================================
class NumericAccessor:
    @property
    def field_name(self) -> str:
        """字段名，如 'x0'、'height'"""

    def __iter__(self) -> Iterator[int]:
        """逐行迭代字段值。示例: [v for v in df.x0]"""
    def __repr__(self) -> str:
        """预览前 5 个值: <Numeric x0: [10, 20, ...]>"""

    # ----- 统计 -----
    def values(self, index: Optional[int] = None) -> Union[List[int], int]:
        """不传参返回全部值列表；传索引返回单行值（支持负数）。示例: df.height.values(-1)"""
    def max(self) -> int:
        """最大值。示例: df.x0.max()"""
    def min(self) -> int:
        """最小值"""
    def mean(self) -> float:
        """均值。示例: df.width.mean()"""
    def mode(self) -> Tuple[int, int]:
        """(众数值, 出现次数)；多个众数时取最小。示例: df.height.mode() -> (12, 30)"""
    def var(self, ddof: int = 1) -> float:
        """方差；默认样本方差 (ddof=1)，ddof=0 为总体方差；样本数 <= ddof 时返回 0.0"""
    def std(self, ddof: int = 1) -> float:
        """标准差，语义同 var()"""
    def diff(self) -> List[Optional[int]]:
        """一阶差分；首元素恒为 None。示例: df.y0.diff() -> [None, 5, -3]"""

    # ----- 过滤（返回新表）-----
    def greater(self, value: int) -> "NDframe":
        """字段值 > value 的行。示例: df.x0.greater(100)"""
    def less(self, value: int) -> "NDframe":
        """字段值 < value 的行"""
    def between(self, lo: int, hi: int) -> "NDframe":
        """字段值在 (lo, hi) 开区间内的行"""

    # ----- 就地运算 -----
    def add(self, value: int) -> "NDframe":
        """就地每个值 +value，返回原 df。示例: df.width.add(5)"""
    def sub(self, value: int) -> "NDframe":
        """就地每个值 -value（不低于 0），返回原 df"""
    def sort(self, descending: bool = False) -> "NDframe":
        """就地按该字段排序，返回原 df。示例: df.y1.sort(True)"""

    # ----- 索引提取 -----
    def idx_greater(self, value: int) -> List[int]:
        """字段值 > value 的行索引。示例: df.x0.idx_greater(50) -> [3, 7]"""
    def idx_less(self, value: int) -> List[int]:
        """字段值 < value 的行索引"""
    def idx_equal(self, value: int) -> List[int]:
        """字段值 == value 的行索引"""
    def idx_between(self, lo: int, hi: int) -> List[int]:
        """字段值在 (lo, hi) 内的行索引"""


# =============================================================================
# UCS2Column_SoA — 内部存储引擎（UCS2 字符池 + 偏移量 SoA）
# 仅供 NDframe 内部使用；此类不对外暴露任何 Python 方法。
# 暴露给 Python 的仅有无参/有参构造函数，字符串增删改查全部经 NDframe/访问器完成。
# =============================================================================
class UCS2Column_SoA:
    def __init__(self, init_capacity: int = 64) -> None: ...
