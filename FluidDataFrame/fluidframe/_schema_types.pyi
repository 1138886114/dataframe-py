"""FluidDataFrame 类型桩 (type stub)。

本文件为 ``fluidframe`` 包的类型声明文件，供 mypy / pyright / Pylance 等
类型检查器与编辑器智能提示使用。所有类、函数均带 pandas 风格 docstring，
包含参数 (Parameters)、返回值 (Returns) 与使用示例 (Examples)。

示例统一约定：先构造一个小型 DataFrame，再展示该方法的典型用法。
"""

from typing import Any, Type

# ---------------------------------------------------------------------------
# 类型占位符 (类型桩专用)：实际运行时为 Cython 扩展类，这里仅声明公开成员。
# ---------------------------------------------------------------------------

class Ascii:
    """用于在 Schema 中标记纯 ASCII 256字节定长字符串类型的占位类。

    Examples
    --------
    >>> from fluidframe import Ascii
    >>> from fluidframe.core.dataframe import DataFrame
    >>> df = DataFrame({'tag': Ascii})
    """

class Date:
    """日期类型，内部按 YYYYMMDD 整数存储。

    Examples
    --------
    >>> from fluidframe import Date
    >>> from fluidframe.core.dataframe import DataFrame
    >>> df = DataFrame({'dt': Date})
    """


class DateTime:
    """日期时间类型，内部按微秒 since epoch 存储。

    Examples
    --------
    >>> from fluidframe.core.dataframe import DataFrame
    >>> df = DataFrame({'ts': 'datetime'})
    """


class Time:
    """时间类型，内部按秒 since midnight 存储。

    Examples
    --------
    >>> df = DataFrame({'t': 'time'})
    """


class Bool:
    """布尔类型，只能为 True/False。

    Examples
    --------
    >>> df = DataFrame({'ok': Bool})
    """


class Category:
    """枚举/分类类型，内部按 int32 codes + 字符串字典存储。

    Examples
    --------
    >>> df = DataFrame({'grade': Category})
    """


# ---------------------------------------------------------------------------
# 类型常量
# ---------------------------------------------------------------------------

TYPE_INT: int
TYPE_DOUBLE: int
TYPE_ASCII: int
TYPE_UCS2: int
TYPE_DATE: int
TYPE_BOOL: int
TYPE_CATEGORY: int
TYPE_DATETIME: int
TYPE_TIME: int


# ---------------------------------------------------------------------------
# 内部工具 (不对外保证稳定的实现细节，仅声明以配合 Cython 模块)
# ---------------------------------------------------------------------------

def _parse_schema(schema_dict: dict) -> dict:
    """解析 schema 字典到内部类型编码。仅供内部使用。"""