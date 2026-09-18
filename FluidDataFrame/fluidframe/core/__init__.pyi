"""FluidDataFrame 核心子包。

``fluidframe.core`` 下的 ``dataframe``/``series``/``index``/``indexers`` 以及
各列类型模块 (``intcolumn`` 等) 在运行时通过 ``sys.modules`` 重定向到
Cython 扩展模块 ``_databaseframe`` 与 ``_columns``。此处类型桩保持同一套
公开命名，方便 `from fluidframe.core.dataframe import DataFrame` 这类导入。
"""

from fluidframe.core._databaseframe import *  # re-export: DataFrame, Series, Index...
from fluidframe.core._columns import *  # re-export: IntColumn, DoubleColumn, ...