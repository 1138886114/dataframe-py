# Shared type definitions for the fluidframe package

TYPE_INT = 0
TYPE_FLOAT = 1
TYPE_DOUBLE = 2
TYPE_CHAR256 = 3
TYPE_UCS2 = 4
TYPE_TIMESTAMP = 5


class Char256:
    """用于在 Schema 中标记纯 ASCII 256字节定长字符串类型的占位类"""
    pass


class Timestamp:
    """时间戳类型，内部按 YYYYMMDD 整数存储；构造时支持 Timestamp('20160101') 或 Timestamp(20160101)"""
    def __init__(self, val=None):
        pass


def _parse_schema(schema_dict):
    parsed = {}
    for k, v in schema_dict.items():
        if v is int: parsed[k] = TYPE_INT
        elif v is float: parsed[k] = TYPE_DOUBLE
        elif v is Char256: parsed[k] = TYPE_CHAR256
        elif v is str: parsed[k] = TYPE_UCS2
        elif v is Timestamp: parsed[k] = TYPE_TIMESTAMP
        else: raise TypeError(f"Unsupported type: {v}")
    return parsed
