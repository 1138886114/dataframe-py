from typing import Optional, List, Dict, Any, Union, Iterator
from re import Pattern


class NDframe:
    """高性能 OCR/布局 NDframe，存储文本块及其坐标信息

    使用示例::

        df = NDframe()
        df.append(0, 10, 20, 100, 50, 90, 30, 1, 0, "hello", "world")
    """

    def __init__(self, capacity: int = 8) -> None:
        """初始化 NDframe，预分配 capacity 行空间

        示例::

            df = NDframe(32)
        """

    # ── 属性访问器 ──
    @property
    def text(self) -> StringAccessor:
        """字符串字段访问器 (text 列)

        示例::

            df.text.contains(r"\\d+")
        """

    @property
    def text_sep(self) -> StringAccessor:
        """字符串字段访问器 (text_sep 列)

        示例::

            df.text_sep.replace("\\n", " ")
        """

    @property
    def x0(self) -> NumericAccessor:
        """左上角 x 坐标访问器

        示例::

            df.x0.max()
        """

    @property
    def y0(self) -> NumericAccessor:
        """左上角 y 坐标访问器"""

    @property
    def x1(self) -> NumericAccessor:
        """右下角 x 坐标访问器"""

    @property
    def y1(self) -> NumericAccessor:
        """右下角 y 坐标访问器"""

    @property
    def width(self) -> NumericAccessor:
        """宽度字段访问器"""

    @property
    def height(self) -> NumericAccessor:
        """高度字段访问器"""

    @property
    def idx(self) -> NumericAccessor:
        """索引字段访问器"""

    @property
    def page(self) -> NumericAccessor:
        """页码字段访问器"""

    @property
    def flags(self) -> NumericAccessor:
        """标志位字段访问器"""

    @property
    def empty(self) -> bool:
        """NDframe 是否为空

        示例::

            if not df.empty:
                print(df.head())
        """

    @property
    def data(self) -> Optional[List[Dict[str, Any]]]:
        """返回所有行的 dict 列表，供调试器/IDE 查看数据

        示例::

            for row in df.data:
                print(row["text"])
        """

    # ── 魔术方法 ──
    def __getitem__(self, key: Union[int, slice]) -> Union[Dict[str, Any], 'NDframe']:
        """支持 df[3] 返回行字典，df[1:5] 返回切片 NDframe

        示例::

            row = df[0]        # 第一行字典
            sub = df[2:5]      # 第 2-4 行新 NDframe
        """

    def __len__(self) -> int:
        """返回行数

        示例::

            n = len(df)
        """

    def len(self) -> int:
        """返回行数（与 len(df) 等价）

        示例::

            n = df.len()
        """

    def __bool__(self) -> bool:
        """非空为 True

        示例::

            if df:
                print("non-empty")
        """

    def __repr__(self) -> str:
        """调试器友好：NDframe(N rows)"""

    def __str__(self) -> str:
        """<NDframe: N rows>"""

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """逐行迭代，每次 yield 一行 dict

        示例::

            for row in df:
                print(row["text"], row["x0"])
        """

    def _repr_html_(self) -> str:
        """IPython / Jupyter / PyCharm SciView 富文本输出"""

    # ── 数据操作方法 ──
    def append(self, idx: int, x0: int, y0: int, x1: int, y1: int,
               width: int, height: int, page: int, flags: int,
               text: str, text_sep: str = "") -> None:
        """追加一行数据

        示例::

            df.append(0, 10, 20, 100, 50, 90, 30, 1, 0, "hello")
        """

    def to_list(self) -> Optional[List[Dict[str, Any]]]:
        """转换为 dict 列表，空 NDframe 返回 None

        示例::

            rows = df.to_list()
        """

    def to_string(self, max_rows: int = 80) -> str:
        """构造对齐的 ASCII 表格字符串，超过 max_rows 时自动折叠

        示例::

            print(df.to_string())
        """

    def print(self, name: Optional[str] = None) -> Optional[str]:
        """打印 NDframe 的字符串表示

        示例::

            df.print()
            df.print("OCR")
        """

    def head(self, n: int = 10) -> 'NDframe':
        """返回前 n 行的新 NDframe

        示例::

            df.head(10)
        """

    def tail(self, n: int = 10) -> 'NDframe':
        """返回后 n 行的新 NDframe

        示例::

            df.tail(3)
        """

    def copy(self) -> 'NDframe':
        """深拷贝，返回完全独立的新 NDframe

        示例::

            df2 = df.copy()
        """

    def pop_back(self) -> None:
        """删除最后一行（就地），若为空则不执行

        示例::

            df.pop_back()
        """

    def bbox(self, n: int = 0) -> tuple:
        """返回第 n 行的边界框 (x0, y0, x1, y1)

        示例::

            x0, y0, x1, y1 = df.bbox(2)
        """

    def centroid(self) -> tuple:
        """计算所有矩形中心点的平均值（质心），返回 (x_center, y_center)

        示例::

            cx, cy = df.centroid()
        """

    def centroid_line(self, n: int) -> tuple:
        """返回第 n 行矩形的中心点坐标 (x_center, y_center)，支持负索引

        示例::

            cx, cy = df.centroid_line(-1)
        """

    # ── 静态/类方法 ──
    @staticmethod
    def concat(df1: 'NDframe', df2: 'NDframe') -> 'NDframe':
        """拼接两个 NDframe，返回新 NDframe

        示例::

            result = NDframe.concat(df_a, df_b)
        """

    @classmethod
    def from_dict(cls, data: Dict[str, List[Any]]) -> 'NDframe':
        """从字典创建 NDframe

        示例::

            df = NDframe.from_dict({"text": ["A", "B"], "x0": [10, 20]})
        """

    # ── 索引查询 (返回行号列表) ──
    def idx_in_bbox(self, x0: int, y0: int, x1: int, y1: int) -> List[int]:
        """返回在边界框内的行索引列表

        示例::

            ids = df.idx_in_bbox(50, 50, 500, 800)
        """

    def idx_in_x0x1(self, x0: int, x1: int) -> List[int]:
        """返回 x 方向在区间内的行索引列表

        示例::

            ids = df.idx_in_x0x1(100, 400)
        """

    def idx_in_y0y1(self, y0: int, y1: int) -> List[int]:
        """返回 y 方向在区间内的行索引列表

        示例::

            ids = df.idx_in_y0y1(100, 400)
        """

    # ── 索引 → NDframe / 就地删除 ──
    def df_from_idx(self, indices: List[int]) -> 'NDframe':
        """根据索引列表提取新 NDframe

        示例::

            sub = df.df_from_idx([0, 2, 4])
        """

    def drop_by_idx(self, indices: List[int]) -> 'NDframe':
        """就地删除指定索引的行，返回原 df

        示例::

            df.drop_by_idx([1, 3])
        """

    # ── 空间过滤  ──
    def filter_in_bbox(self, x0: int, y0: int, x1: int, y1: int) -> 'NDframe':
        """返回在边界框内的行构成的新 NDframe

        示例::

            sub = df.filter_in_bbox(0, 0, 500, 800)
        """

    def filter_in_x0x1(self, x0: int, x1: int) -> 'NDframe':
        """返回 x 方向在区间内的行构成的新 NDframe"""

    def filter_in_y0y1(self, y0: int, y1: int) -> 'NDframe':
        """返回 y 方向在区间内的行构成的新 NDframe"""

    def filter_greater_point(self, x0: int, y0: int) -> 'NDframe':
        """返回左上角坐标均大于阈值的行构成的新 NDframe

        示例::

            sub = df.filter_greater_point(100, 100)
        """

    def filter_less_point(self, x1: int, y1: int) -> 'NDframe':
        """返回右下角坐标均小于阈值的行构成的新 NDframe

        示例::

            sub = df.filter_less_point(500, 800)
        """

    # ── 排序 (原地操作，返回 self) ──
    def sort_by_x0(self, descending: bool = False) -> 'NDframe':
        """按 x0 排序（就地）

        示例::

            df.sort_by_x0()
        """

    def sort_by_y0(self, descending: bool = False) -> 'NDframe': ...
    def sort_by_x1(self, descending: bool = False) -> 'NDframe': ...
    def sort_by_y1(self, descending: bool = False) -> 'NDframe': ...
    def sort_by_width(self, descending: bool = False) -> 'NDframe': ...
    def sort_by_height(self, descending: bool = False) -> 'NDframe': ...
    def sort_by_idx(self, descending: bool = False) -> 'NDframe': ...
    def sort_by_page(self, descending: bool = False) -> 'NDframe': ...
    def sort_by_flags(self, descending: bool = False) -> 'NDframe': ...

    def sort_by(self, field: str = "x0", descending: bool = False) -> 'NDframe':
        """按字段名字符串排序（就地）

        示例::

            df.sort_by("height", descending=True)
        """

    # ── 索引 / 变换 (原地操作，返回 self) ──
    def reset_index(self) -> 'NDframe':
        """重置 idx 为 0 递增（就地）

        示例::

            df.reset_index()
        """

    def add_y0y1(self, offset: int) -> 'NDframe':
        """y0 和 y1 同步增加 offset（就地）

        示例::

            df.add_y0y1(10)
        """

    # ── 条件运算 (原地操作，返回 self) ──
    def apply_if_greater_add(self, cond_field: str, target_field: str, threshold: int, offset: int) -> 'NDframe':
        """如果 cond_field > threshold，则 target_field += offset（就地）

        示例::

            df.apply_if_greater_add("height", "y0", 50, 5)
        """

    def apply_if_less_add(self, cond_field: str, target_field: str, threshold: int, offset: int) -> 'NDframe':
        """如果 cond_field < threshold，则 target_field += offset（就地）"""

    def apply_if_greater_add_y(self, cond_field: str, threshold: int, offset: int) -> 'NDframe':
        """如果 cond_field > threshold，则 y0/y1 += offset（就地）

        示例::

            df.apply_if_greater_add_y("y0", 100, 5)
        """

    def apply_if_less_add_y(self, cond_field: str, threshold: int, offset: int) -> 'NDframe':
        """如果 cond_field < threshold，则 y0/y1 += offset（就地）"""

    # ── 删除 ──
    def drop(self, index: int) -> 'NDframe':
        """就地删除单行，返回 self

        示例::

            df.drop(2)
        """

    def drop_in_bbox(self, x0: int, y0: int, x1: int, y1: int) -> 'NDframe':
        """就地删除边界框内的行，返回 self

        示例::

            df.drop_in_bbox(0, 0, 200, 300)
        """

    def df_drop(self, index: int) -> 'NDframe':
        """返回不含 index 行的新 NDframe

        示例::

            df2 = df.df_drop(1)
        """

    def df_drops(self, indices: List[int]) -> 'NDframe':
        """返回不含指定索引行的新 NDframe

        示例::

            df2 = df.df_drops([0, 2])
        """

    # ── 合并 ──
    def merge_line(self, width_threshold: int, height_threshold: int, sep: str = "") -> 'NDframe':
        """合并相邻行，返回新 NDframe

        示例::

            merged = df.merge_line(10, 5, " ")
        """

    # ── 子串坐标 ──
    def get_txt_coords(self, a: str) -> List[tuple]:
        """返回 text 字段中包含子串 a 的精确坐标列表

        示例::

            coords = df.get_txt_coords("发票")
        """

    def get_sep_coords(self, a: str) -> List[tuple]:
        """返回 text_sep 字段中包含子串 a 的精确坐标列表

        示例::

            coords = df.get_sep_coords(" ")
        """


class StringAccessor:
    """字符串字段访问器：提供类似 pandas 的 df.text.xxx API"""

    @property
    def field_name(self) -> str:
        """返回字段名称 (text 或 text_sep)

        示例::

            name = df.text.field_name  # "text"
        """

    def __repr__(self) -> str:
        """调试器友好的显示：field_name[values...] (n=N)"""

    def __iter__(self) -> Iterator[str]:
        """逐行迭代该字段的字符串值

        示例::

            for s in df.text:
                print(s)
        """

    def slice(self, start=None, stop=None, step=None) -> List[str]:
        """字符串切片操作，返回每行切片结果列表

        示例::

            df.text.slice(0, 3)  # 每行前 3 字符
        """

    def count(self, sub: str) -> int:
        """统计子串在所有行中的出现总次数

        示例::

            n = df.text.count("abc")
        """

    def startswith(self, prefix: str) -> NDframe:
        """返回以 prefix 开头的行构成的新 NDframe

        示例::

            sub = df.text.startswith("http")
        """

    def endswith(self, suffix: str) -> NDframe:
        """返回以 suffix 结尾的行构成的新 NDframe

        示例::

            sub = df.text.endswith(".pdf")
        """

    def strip(self, chars: Optional[str] = None) -> NDframe:
        """就地 strip，返回原 NDframe

        示例::

            df.text.strip()
        """

    def lstrip(self, chars: Optional[str] = None) -> NDframe:
        """就地 lstrip，返回原 NDframe"""

    def rstrip(self, chars: Optional[str] = None) -> NDframe:
        """就地 rstrip，返回原 NDframe"""

    def capitalize(self) -> NDframe:
        """就地 capitalize，返回原 NDframe

        示例::

            df.text.capitalize()
        """

    def swapcase(self) -> NDframe:
        """就地 swapcase，返回原 NDframe"""

    def isalnum(self) -> NDframe:
        """返回文本仅含字母数字的行构成的新 NDframe (str.isalnum)

        示例::

            sub = df.text.isalnum()
        """

    def isalpha(self) -> NDframe:
        """返回文本仅含字母的行构成的新 NDframe (str.isalpha)

        示例::

            sub = df.text.isalpha()
        """

    def isdigit(self) -> NDframe:
        """返回文本仅含数字的行构成的新 NDframe (str.isdigit)

        示例::

            sub = df.text.isdigit()
        """

    def isspace(self) -> NDframe:
        """返回文本仅含空白字符的行构成的新 NDframe (str.isspace)

        示例::

            sub = df.text.isspace()
        """

    def islower(self) -> NDframe:
        """返回文本全小写的行构成的新 NDframe (str.islower)

        示例::

            sub = df.text.islower()
        """

    def isupper(self) -> NDframe:
        """返回文本全大写的行构成的新 NDframe (str.isupper)

        示例::

            sub = df.text.isupper()
        """

    def istitle(self) -> NDframe:
        """返回文本为标题大小写的行构成的新 NDframe (str.istitle)

        示例::

            sub = df.text.istitle()
        """

    def isnumeric(self) -> NDframe:
        """返回文本仅含数字字符的行构成的新 NDframe (str.isnumeric)
        与 isdigit 区别：支持分数、罗马数字等 Unicode 数字

        示例::

            sub = df.text.isnumeric()
        """

    def isdecimal(self) -> NDframe:
        """返回文本仅含十进制数字的行构成的新 NDframe (str.isdecimal)

        示例::

            sub = df.text.isdecimal()
        """

    def is_chinese(self) -> NDframe:
        """返回文本仅含中文的行构成的新 NDframe
        支持基本汉字、扩展A/B、兼容汉字区

        示例::

            sub = df.text.is_chinese()
        """

    def is_numeric_punct(self) -> NDframe:
        """返回文本仅含数字 [0-9] 和标点 [.,] 的行构成的新 NDframe

        示例::

            sub = df.text.is_numeric_punct()
        """

    def keep_chinese(self) -> NDframe:
        """就地仅保留中文字符，返回原 NDframe

        示例::

            df.text.keep_chinese()
        """

    def keep_numeric_punct(self) -> NDframe:
        """就地仅保留数字和标点字符，返回原 NDframe

        示例::

            df.text.keep_numeric_punct()
        """

    def contains(self, pattern: Union[str, Pattern[str]]) -> NDframe:
        """正则匹配过滤，返回匹配行构成的新 NDframe
        支持字符串自动编译为正则

        示例::

            sub = df.text.contains(r"\\d{3,}")
        """

    def contains_in_bbox(self, pattern: Union[str, Pattern[str]], x0: int, y0: int, x1: int, y1: int) -> NDframe:
        """在指定坐标框内进行正则过滤，返回新 NDframe

        示例::

            sub = df.text.contains_in_bbox(r"hello", 0, 0, 500, 800)
        """

    def replace(self, old_str: str, new_str: str, old_str2: str = "", new_str2: str = "") -> NDframe:
        """就地字符串替换，返回原 NDframe 支持链式调用。可选第二对替换 (old_str2 → new_str2)

        示例::

            df.text.replace("\\n", " ").replace("\\t", " ")
            df.text.replace("foo", "bar", "old", "new")
        """

    def regex_replace(self, pattern: Union[str, Pattern[str]], rewrite: str) -> NDframe:
        """就地正则替换，返回原 NDframe 支持链式调用

        示例::

            df.text.regex_replace(r"\\d+", "0")
        """

    def regex_extract(self, pattern: Union[str, Pattern[str]]) -> NDframe:
        """正则提取第一个捕获组，返回新 NDframe，结果写入当前字段

        示例::

            sub = df.text.regex_extract(r"(\\d{4})")
        """

    def join(self, separator: str = "") -> str:
        """将当前字段所有值用分隔符拼接为一个字符串

        示例::

            text = df.text.join("\\n")
        """

    def strlen_greater(self, n: int) -> NDframe:
        """返回字符串长度 > n 的行构成的新 NDframe

        示例::

            sub = df.text.strlen_greater(5)
        """

    def strlen_less(self, n: int) -> NDframe:
        """返回字符串长度 < n 的行构成的新 NDframe"""

    def strlen_equal(self, n: int) -> NDframe:
        """返回字符串长度 == n 的行构成的新 NDframe

        示例::

            sub = df.text.strlen_equal(10)
        """

    def values(self, index: Optional[int] = None) -> Union[List[str], str]:
        """返回当前字段所有值构成的列表，或指定行的值
        - 无参：返回列表
        - 传参：返回指定行字符串（支持负索引）

        示例::

            all_text = df.text.values()
            first = df.text.values(0)
            last = df.text.values(-1)
        """

    def len(self) -> List[int]:
        """返回每行字符串长度的列表

        示例::

            lengths = df.text.len()
        """

    def is_in(self, texts: List[str]) -> NDframe:
        """返回字段值在给定集合中的行构成的新 NDframe

        示例::

            sub = df.text.is_in(["yes", "no"])
        """

    def not_in(self, texts: List[str]) -> NDframe:
        """返回字段值不在给定集合中的行构成的新 NDframe

        示例::

            sub = df.text.not_in(["ignore", "skip"])
        """

    def contains_any(self, texts: List[str]) -> bool:
        """检查字段值是否包含给定字符串集合中的任意一个（子串匹配）

        示例::

            if df.text.contains_any(["error", "fail"]):
                print("has issues")
        """

    def regex_strlen_greater(self, pattern: Union[str, Pattern[str]], n: int) -> NDframe:
        """正则匹配 && 字符串长度 > n → 新 NDframe

        示例::

            sub = df.text.regex_strlen_greater(r"\\d+", 3)
        """

    def regex_strlen_less(self, pattern: Union[str, Pattern[str]], n: int) -> NDframe:
        """正则匹配 && 字符串长度 < n → 新 NDframe"""

    def regex_strlen_equal(self, pattern: Union[str, Pattern[str]], n: int) -> NDframe:
        """正则匹配 && 字符串长度 == n → 新 NDframe

        示例::

            sub = df.text.regex_strlen_equal(r"\\w+", 5)
        """

    def replace_to(self, old_str: str, new_str: str, target_field: str = "text") -> NDframe:
        """将本字段的 old_str 替换为 new_str，写入 target_field（就地）

        target_field: "text" 或 "text_sep"

        示例::

            df.text_sep.replace_to("\\n", " ", "text")
        """

    def regex_replace_to(self, pattern: Union[str, Pattern[str]], rewrite: str, target_field: str = "text") -> NDframe:
        """本字段正则替换后写入 target_field（就地）

        示例::

            df.text.regex_replace_to(r"\\d+", "0", "text_sep")
        """

    # ── 就地正则删除与索引查询 ──
    def drop_regex(self, pattern: Union[str, Pattern[str]]) -> NDframe:
        """就地删除匹配该正则的行，返回原 DF

        示例::

            df.text.drop_regex(r"^$")
        """

    def idx_contains(self, pattern: Union[str, Pattern[str]]) -> List[int]:
        """返回匹配该正则的行索引列表

        示例::

            ids = df.text.idx_contains(r"\\d{3}")
        """

    def idx_contains_in_bbox(self, pattern: Union[str, Pattern[str]], x0: int, y0: int, x1: int, y1: int) -> List[int]:
        """返回匹配该正则且在坐标框内的行索引列表

        示例::

            ids = df.text.idx_contains_in_bbox(r"hello", 0, 0, 500, 800)
        """

    def idx_strlen_greater(self, n: int) -> List[int]:
        """返回字符串长度 > n 的行索引列表

        示例::

            ids = df.text.idx_strlen_greater(10)
        """

    def idx_strlen_less(self, n: int) -> List[int]:
        """返回字符串长度 < n 的行索引列表"""

    def idx_strlen_equal(self, n: int) -> List[int]:
        """返回字符串长度 == n 的行索引列表

        示例::

            ids = df.text.idx_strlen_equal(0)
        """

    def idx_regex_strlen_greater(self, pattern: Union[str, Pattern[str]], n: int) -> List[int]:
        """正则匹配 && 字符串长度 > n → 行索引列表

        示例::

            ids = df.text.idx_regex_strlen_greater(r"\\w+", 5)
        """

    def idx_regex_strlen_less(self, pattern: Union[str, Pattern[str]], n: int) -> List[int]:
        """正则匹配 && 字符串长度 < n → 行索引列表"""

    def idx_regex_strlen_equal(self, pattern: Union[str, Pattern[str]], n: int) -> List[int]:
        """正则匹配 && 字符串长度 == n → 行索引列表

        示例::

            ids = df.text.idx_regex_strlen_equal(r"\\d", 1)
        """


class NumericAccessor:
    """数值字段访问器：提供 df.x0.max() 等 API"""

    @property
    def field_name(self) -> str:
        """返回字段名称 (x0, y0, ...)

        示例::

            name = df.x0.field_name  # "x0"
        """

    def __repr__(self) -> str:
        """调试器友好的显示：<Numeric field_name: [values...]>"""

    def __iter__(self) -> Iterator[int]:
        """逐行迭代该字段的整数值

        示例::

            for v in df.x0:
                print(v)
        """

    def diff(self) -> List[Optional[int]]:
        """返回一阶差分列表，第 0 个元素为 None

        示例::

            diffs = df.y0.diff()  # [None, y0[1]-y0[0], y0[2]-y0[1], ...]
        """

    def var(self, ddof: int = 1) -> float:
        """计算方差，默认 ddof=1 样本方差；ddof=0 总体方差

        示例::

            v = df.height.var()
        """

    def std(self, ddof: int = 1) -> float:
        """计算标准差，默认 ddof=1 样本标准差；ddof=0 总体标准差

        示例::

            s = df.height.std()
        """

    def mode(self) -> tuple:
        """返回众数 (mode_value, count)，若多个众数返回最小值

        示例::

            val, cnt = df.height.mode()
        """

    def max(self) -> int:
        """返回该字段的最大值

        示例::

            m = df.x0.max()
        """

    def min(self) -> int:
        """返回该字段的最小值

        示例::

            m = df.x0.min()
        """

    def mean(self) -> float:
        """返回该字段的平均值

        示例::

            avg = df.height.mean()
        """

    def values(self, index: Optional[int] = None) -> Union[List[int], int]:
        """返回该字段的值列表，或指定行的值（支持负索引）

        示例::

            all_vals = df.x0.values()
            first = df.x0.values(0)
        """

    def greater(self, value: int) -> NDframe:
        """返回该字段值 > value 的行构成的新 NDframe

        示例::

            sub = df.height.greater(50)
        """

    def less(self, value: int) -> NDframe:
        """返回该字段值 < value 的行构成的新 NDframe"""

    def between(self, lo: int, hi: int) -> NDframe:
        """返回该字段值在 (lo, hi) 区间内的行构成的新 NDframe

        示例::

            sub = df.x0.between(100, 500)
        """

    def add(self, value: int) -> NDframe:
        """就地加值操作，每个元素 +value，返回原 NDframe

        示例::

            df.x0.add(10)
        """

    def sub(self, value: int) -> NDframe:
        """就地减值操作，每个元素 -value（不低于 0），返回原 NDframe

        示例::

            df.x0.sub(5)
        """

    def sort(self, descending: bool = False) -> NDframe:
        """按该字段排序（就地），返回原 NDframe 支持链式调用

        示例::

            df.height.sort(descending=True)
        """

    def idx_greater(self, value: int) -> List[int]:
        """返回该字段值 > value 的行索引列表

        示例::

            ids = df.height.idx_greater(50)
        """

    def idx_less(self, value: int) -> List[int]:
        """返回该字段值 < value 的行索引列表"""

    def idx_equal(self, value: int) -> List[int]:
        """返回该字段值 == value 的行索引列表

        示例::

            ids = df.page.idx_equal(1)
        """

    def idx_between(self, lo: int, hi: int) -> List[int]:
        """返回该字段值在 [lo, hi) 之间的行索引列表

        示例::

            ids = df.x0.idx_between(100, 500)
        """
