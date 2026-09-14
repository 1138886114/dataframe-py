import sys
sys.path.insert(0, '.')
from fluidframe.core.intcolumn import IntColumn
from fluidframe.core.doublecolumn import DoubleColumn
from fluidframe.core.boolcolumn import BoolColumn
from fluidframe.core.asciicolumn import AsciiColumn
from fluidframe.core.ucs2column import UCS2Column
from fluidframe.core.datecolumn import DateColumn as DateColumn
from fluidframe.core.categorycolumn import CategoryColumn
from fluidframe.core.dataframe import DataFrame
from fluidframe.core.dataframe import to_numeric, to_datetime, to_timedelta

PASSED = 0
FAILED = 0

def check(condition, msg):
    global PASSED, FAILED
    if condition:
        PASSED += 1
    else:
        FAILED += 1
        print(f"  FAIL: {msg}")

# ==================== IntColumn.astype ====================
print("=== IntColumn.astype ===")
ic = IntColumn(4)
ic.append(1); ic.append(0); ic.append(42); ic.append(-7)

# int -> double
dc = ic.astype('double')
check(dc.dtype == 2, "int->double dtype")
check(dc[0] == 1.0, "int->double [0]")
check(dc[2] == 42.0, "int->double [2]")

# int -> bool
bc = ic.astype('bool')
check(bc[0] is True, "int->bool [0]")
check(bc[1] is False, "int->bool [1]")
check(bc[2] is True, "int->bool [2]")

# int -> ucs2
uc = ic.astype('ucs2')
check(uc[0] == '1', "int->ucs2 [0]")
check(uc[2] == '42', "int->ucs2 [2]")

# int -> char256
cc = ic.astype('char256')
check(cc[0] == '1', "int->char256 [0]")

# int -> timestamp (small ints treated as epoch days, read back as YYYYMMDD)
tc = ic.astype('timestamp')
check(tc[0] == 19700102, "int->timestamp [0]")  # epoch day 1 = 1970-01-02

# int -> category
catc = ic.astype('category')
check(catc[0] == '1', "int->category [0]")
check(catc[2] == '42', "int->category [2]")

# ==================== DoubleColumn.astype ====================
print("=== DoubleColumn.astype ===")
dc2 = DoubleColumn(4)
dc2.append(1.5); dc2.append(0.0); dc2.append(42.7); dc2.append(-3.14)

# double -> int (truncation)
ic2 = dc2.astype('int')
check(ic2[0] == 1, "double->int [0] trunc")
check(ic2[2] == 42, "double->int [2] trunc")

# double -> bool
bc2 = dc2.astype('bool')
check(bc2[0] is True, "double->bool [0]")
check(bc2[1] is False, "double->bool [1]")

# double -> ucs2
uc2 = dc2.astype('ucs2')
check(uc2[0] == '1.5', "double->ucs2 [0]")

# ==================== BoolColumn.astype ====================
print("=== BoolColumn.astype ===")
bc3 = BoolColumn(4)
bc3.append(True); bc3.append(False); bc3.append(True); bc3.append(False)

# bool -> int
ic3 = bc3.astype('int')
check(ic3[0] == 1, "bool->int [0]")
check(ic3[1] == 0, "bool->int [1]")

# bool -> double
dc3 = bc3.astype('double')
check(dc3[0] == 1.0, "bool->double [0]")
check(dc3[1] == 0.0, "bool->double [1]")

# bool -> ucs2
uc3 = bc3.astype('ucs2')
check(uc3[0] == 'True', "bool->ucs2 [0]")
check(uc3[1] == 'False', "bool->ucs2 [1]")

# ==================== Char256Column.astype ====================
print("=== Char256Column.astype ===")
cc2 = AsciiColumn(4)
cc2.append('123'); cc2.append('45.67'); cc2.append('-7'); cc2.append('20240101')

# char256 -> int
ic4 = cc2.astype('int')
check(ic4[0] == 123, "char256->int [0]")
check(ic4[2] == -7, "char256->int [2]")

# char256 -> double
dc4 = cc2.astype('double')
check(dc4[1] == 45.67, "char256->double [1]")

# char256 -> ucs2
uc4 = cc2.astype('ucs2')
check(uc4[0] == '123', "char256->ucs2 [0]")

# char256 -> timestamp
tc2 = cc2.astype('timestamp')
check(tc2[3] == 20240101, "char256->timestamp [3]")

# ==================== UCS2Column.astype ====================
print("=== UCS2Column.astype ===")
uc5 = UCS2Column(4)
uc5.append('100'); uc5.append('3.14'); uc5.append('-5'); uc5.append('20231225')

# ucs2 -> int
ic5 = uc5.astype('int')
check(ic5[0] == 100, "ucs2->int [0]")
check(ic5[2] == -5, "ucs2->int [2]")

# ucs2 -> double
dc5 = uc5.astype('double')
check(dc5[1] == 3.14, "ucs2->double [1]")

# ucs2 -> timestamp
tc3 = uc5.astype('timestamp')
check(tc3[3] == 20231225, "ucs2->timestamp [3]")

# ==================== DateColumn.astype ====================
print("=== DateColumn.astype ===")
tc4 = DateColumn(4)
tc4.append(20240101); tc4.append(20230615); tc4.append(19991231); tc4.append(20200229)

# timestamp -> int
ic6 = tc4.astype('int')
check(ic6[0] == 20240101, "timestamp->int [0]")

# timestamp -> double
dc6 = tc4.astype('double')
check(dc6[0] == 20240101.0, "timestamp->double [0]")

# timestamp -> ucs2
uc6 = tc4.astype('ucs2')
check(uc6[0] == '2024-01-01', "timestamp->ucs2 [0]")
check(uc6[1] == '2023-06-15', "timestamp->ucs2 [1]")
check(uc6[2] == '1999-12-31', "timestamp->ucs2 [2]")

# timestamp -> category
catc2 = tc4.astype('category')
check(catc2[0] == '2024-01-01', "timestamp->category [0]")

# ==================== CategoryColumn.astype ====================
print("=== CategoryColumn.astype ===")
catc3 = CategoryColumn(4)
catc3.append('10'); catc3.append('20'); catc3.append('10'); catc3.append('30')

# category -> int
ic7 = catc3.astype('int')
check(ic7[0] == 10, "category->int [0]")
check(ic7[1] == 20, "category->int [1]")

# category -> double
dc7 = catc3.astype('double')
check(dc7[0] == 10.0, "category->double [0]")

# category -> ucs2
uc7 = catc3.astype('ucs2')
check(uc7[0] == '10', "category->ucs2 [0]")

# ==================== DataFrame.astype ====================
print("=== DataFrame.astype ===")
df = DataFrame(schema={'A': 'int', 'B': 'str', 'C': 'bool'})
df = df.append({'A': 42, 'B': '3.14', 'C': True})
df = df.append({'A': 10, 'B': '2.71', 'C': False})

df2 = df.astype({'A': 'double', 'B': 'double'})
check(df2['A'].dtype == 2, "df astype A->double")
check(df2['A'][0] == 42.0, "df astype A value")
check(df2['B'].dtype == 2, "df astype B->double")
check(abs(df2['B'][0] - 3.14) < 1e-9, "df astype B value")
check(df2['C'].dtype == 6, "df astype C unchanged")

# ==================== to_numeric ====================
print("=== to_numeric ===")
s = UCS2Column(4)
s.append('123'); s.append('45.67'); s.append('not_a_number'); s.append('-7.5')

res = to_numeric(s, errors='coerce')
check(res[0] == 123.0, "to_numeric int")
check(abs(res[1] - 45.67) < 1e-9, "to_numeric float")
check(res[3] == -7.5, "to_numeric negative")

import math
check(math.isnan(res[2]), "to_numeric coerce NaN")

# errors='raise'
try:
    to_numeric(s, errors='raise')
    FAILED += 1
    print("  FAIL: to_numeric raise should have thrown")
except ValueError:
    PASSED += 1

# ==================== to_datetime ====================
print("=== to_datetime ===")
s2 = UCS2Column(4)
s2.append('2024-01-01'); s2.append('20230615'); s2.append('invalid'); s2.append('2020-02-29')

res2 = to_datetime(s2, errors='coerce')
check(str(res2[0]) == '2024-01-01', "to_datetime YYYY-MM-DD")
check(str(res2[1]) == '2023-06-15', "to_datetime YYYYMMDD")
check(str(res2[3]) == '2020-02-29', "to_datetime leap year")
check(str(res2[2]) == '1970-01-01', "to_datetime coerce epoch")

# ==================== to_timedelta ====================
print("=== to_timedelta ===")
td = to_timedelta([1, 2, 3], unit='D')
check(td[0] == 1, "to_timedelta days")
check(td[1] == 2, "to_timedelta days")

td2 = to_timedelta([24, 48], unit='h')
check(td2[0] == 1, "to_timedelta hours->days")
check(td2[1] == 2, "to_timedelta hours->days")

td3 = to_timedelta([1440], unit='m')
check(td3[0] == 1, "to_timedelta minutes->days")

td4 = to_timedelta([86400], unit='s')
check(td4[0] == 1, "to_timedelta seconds->days")

# ==================== Category in schema ====================
print("=== Category in schema ===")
df3 = DataFrame(schema={'X': 'category'})
df3 = df3.append({'X': 'low'})
df3 = df3.append({'X': 'high'})
df3 = df3.append({'X': 'low'})
check(df3['X'][0] == 'low', "schema category [0]")
check(df3['X'][1] == 'high', "schema category [1]")
check(df3['X'].cat.codes[0] == 0, "schema category codes")
check(len(df3['X'].cat.categories) == 2, "schema category 2 cats")

print(f"\nResults: {PASSED} passed, {FAILED} failed")
if FAILED > 0:
    sys.exit(1)
