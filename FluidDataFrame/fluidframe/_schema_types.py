# Shared type definitions for the fluidframe package

TYPE_INT = 0
TYPE_FLOAT = 1
TYPE_DOUBLE = 2
TYPE_CHAR256 = 3
TYPE_UCS2 = 4
TYPE_TIMESTAMP = 5
TYPE_BOOL = 6
TYPE_CATEGORY = 7


class Char256:
    """用于在 Schema 中标记纯 ASCII 256字节定长字符串类型的占位类"""
    pass


class Timestamp:
    """时间戳类型，内部按 YYYYMMDD 整数存储；构造时支持 Timestamp('20160101') 或 Timestamp(20160101)"""
    def __init__(self, val=None):
        pass


class Bool:
    """布尔类型，只能为 True/False"""
    pass


class Category:
    """枚举/分类类型，内部按 int32 codes + 字符串字典存储"""
    pass


_STRING_TYPE_MAP = {
    'int': TYPE_INT,
    'float': TYPE_DOUBLE,
    'double': TYPE_DOUBLE,
    'char256': TYPE_CHAR256,
    'ucs2': TYPE_UCS2,
    'str': TYPE_UCS2,
    'string': TYPE_UCS2,
    'timestamp': TYPE_TIMESTAMP,
    'bool': TYPE_BOOL,
    'boolean': TYPE_BOOL,
    'category': TYPE_CATEGORY,
    'cat': TYPE_CATEGORY,
}


def _parse_schema(schema_dict):
    parsed = {}
    for k, v in schema_dict.items():
        if v is int: parsed[k] = TYPE_INT
        elif v is float: parsed[k] = TYPE_DOUBLE
        elif v is Char256: parsed[k] = TYPE_CHAR256
        elif v is str: parsed[k] = TYPE_UCS2
        elif v is Timestamp: parsed[k] = TYPE_TIMESTAMP
        elif v is Bool: parsed[k] = TYPE_BOOL
        elif v is Category: parsed[k] = TYPE_CATEGORY
        elif isinstance(v, str):
            low = v.lower()
            if low in _STRING_TYPE_MAP:
                parsed[k] = _STRING_TYPE_MAP[low]
            else:
                raise TypeError(f"Unknown type string: '{v}'")
        else:
            raise TypeError(f"Unsupported type: {v}")
    return parsed
