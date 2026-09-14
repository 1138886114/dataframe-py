import sys

import fluidframe.core._columns as _columns_mod
import fluidframe.core._databaseframe as _databaseframe_mod

_core = sys.modules[__name__]

for _name in (
    "intcolumn",
    "doublecolumn",
    "asciicolumn",
    "ucs2column",
    "datecolumn",
    "boolcolumn",
    "categorycolumn",
    "datetimecolumn",
    "timecolumn",
):
    sys.modules["fluidframe.core." + _name] = _columns_mod
    setattr(_core, _name, _columns_mod)

for _name in ("dataframe", "series", "index", "indexers"):
    sys.modules["fluidframe.core." + _name] = _databaseframe_mod
    setattr(_core, _name, _databaseframe_mod)