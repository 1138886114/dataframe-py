"""FluidDataFrame 顶层 API。

快速上手：

>>> from fluidframe import DataFrame, Series
>>> from fluidframe.core.dataframe import concat, to_numeric, to_datetime, to_date, to_time, to_timedelta
>>> from fluidframe import Ascii, Date

DataFrame / Series 的详细用法见 ``fluidframe.core.dataframe`` 与
``fluidframe.core.series`` 的类型桩中每个方法的 docstring。
"""

from fluidframe.core.dataframe import (
    DataFrame as DataFrame,
    Series as Series,
)
from fluidframe._schema_types import (
    Bool as Bool,
    Category as Category,
    Ascii as Ascii,
    Date as Date,
    DateTime as DateTime,
    Time as Time,
    TYPE_INT as TYPE_INT,
    TYPE_DOUBLE as TYPE_DOUBLE,
    TYPE_ASCII as TYPE_ASCII,
    TYPE_UCS2 as TYPE_UCS2,
    TYPE_DATE as TYPE_DATE,
    TYPE_BOOL as TYPE_BOOL,
    TYPE_CATEGORY as TYPE_CATEGORY,
    TYPE_DATETIME as TYPE_DATETIME,
    TYPE_TIME as TYPE_TIME,
)
from fluidframe.core.dataframe import (
    concat as concat,
    to_numeric as to_numeric,
    to_datetime as to_datetime,
    to_date as to_date,
    to_time as to_time,
    to_timedelta as to_timedelta,
)