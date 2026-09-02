"""Tests for DateTimeColumn and TimeColumn."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fluidframe.core.datetimecolumn import DateTimeColumn
from fluidframe.core.timecolumn import TimeColumn
from fluidframe.core.fastdataframe import FastDataFrame
from fluidframe.core.fluidframe import to_datetime, to_date, to_time, concat
from fluidframe.core.intcolumn import IntColumn
from fluidframe.core.doublecolumn import DoubleColumn
from fluidframe.core.char256column import Char256Column_SoA
from fluidframe.core.ucs2column import UCS2Column_SoA
from fluidframe.core.boolcolumn import BoolColumn
from fluidframe.core.categorycolumn import CategoryColumn
from fluidframe.core.datecolumn import DateColumn

PASSED = 0
FAILED = 0

def check(cond, msg):
    global PASSED, FAILED
    if cond:
        PASSED += 1
    else:
        FAILED += 1
        print(f"  FAIL: {msg}")

# ==================== DateTimeColumn construction ====================
print("=== DateTimeColumn construction ===")
dtc = DateTimeColumn(4)
dtc.append('2024-03-15 10:30:00')
dtc.append('2024-01-01')
dtc.append('2023-12-31 23:59:59')
dtc.append('2020-06-15 14:00:00')
check(dtc.length == 4, "dtc length")
check(str(dtc[0]) == '2024-03-15 10:30:00', "dtc full datetime")
check(str(dtc[1]) == '2024-01-01', "dtc date only")
check(str(dtc[2]) == '2023-12-31 23:59:59', "dtc end of year")
check(str(dtc[3]) == '2020-06-15 14:00:00', "dtc mid year")

# ==================== DateTimeColumn extractors ====================
print("=== DateTimeColumn extractors ===")
dtc_years = dtc.year()
dtc_months = dtc.month()
dtc_days = dtc.day()
dtc_hours = dtc.hour()
dtc_minutes = dtc.minute()
dtc_seconds = dtc.second()
check(dtc_years[0] == 2024, "year 0")
check(dtc_months[0] == 3, "month 0")
check(dtc_days[0] == 15, "day 0")
check(dtc_hours[0] == 10, "hour 0")
check(dtc_minutes[0] == 30, "minute 0")
check(dtc_seconds[0] == 0, "second 0")

check(dtc_years[1] == 2024, "year 1")
check(dtc_hours[1] == 0, "hour 1 (date only)")

check(dtc_years[2] == 2023, "year 2")
check(dtc_hours[2] == 23, "hour 2")
check(dtc_minutes[2] == 59, "minute 2")
check(dtc_seconds[2] == 59, "second 2")

# ==================== DateTimeColumn copy ====================
print("=== DateTimeColumn copy ===")
dtc_copy = dtc.copy()
check(dtc_copy.length == dtc.length, "copy length")
check(str(dtc_copy[0]) == str(dtc[0]), "copy [0]")
check(str(dtc_copy[2]) == str(dtc[2]), "copy [2]")

# ==================== DateTimeColumn append long long ====================
print("=== DateTimeColumn append raw ===")
dtc2 = DateTimeColumn(2)
dtc2.append(0)
dtc2.append(100000000)
check(dtc2.length == 2, "raw append length")
check(str(dtc2[0]) == '1970-01-01', "raw 0 = epoch")

# ==================== DateTimeColumn invalid ====================
print("=== DateTimeColumn invalid ===")
dtc3 = DateTimeColumn(1)
try:
    dtc3.append('invalid')
    check(False, "should raise ValueError for invalid datetime")
except ValueError:
    check(True, "ValueError for invalid datetime")

# ==================== DateTimeColumn append int ====================
print("=== DateTimeColumn append int ===")
dtc4 = DateTimeColumn(1)
dtc4.append(1234567890)
check(dtc4.length == 1, "int append length")

# ==================== TimeColumn construction ====================
print("=== TimeColumn construction ===")
tmc = TimeColumn(4)
tmc.append('10:30:00')
tmc.append('00:00:00')
tmc.append('23:59:59')
tmc.append('12:00')
check(tmc.length == 4, "tmc length")
check(tmc[0] == '10:30:00', "tmc 10:30:00")
check(tmc[1] == '00:00:00', "tmc 00:00:00")
check(tmc[2] == '23:59:59', "tmc 23:59:59")
check(tmc[3] == '12:00:00', "tmc 12:00")

# ==================== TimeColumn extractors ====================
print("=== TimeColumn extractors ===")
tmc_hours = tmc.hour()
tmc_minutes = tmc.minute()
tmc_seconds = tmc.second()
check(tmc_hours[0] == 10, "tmc hour 0")
check(tmc_minutes[0] == 30, "tmc minute 0")
check(tmc_seconds[0] == 0, "tmc second 0")
check(tmc_hours[2] == 23, "tmc hour 2")
check(tmc_minutes[2] == 59, "tmc minute 2")
check(tmc_seconds[2] == 59, "tmc second 2")

# ==================== TimeColumn int append ====================
print("=== TimeColumn int append ===")
tmc2 = TimeColumn(2)
tmc2.append(3600)
tmc2.append(7200)
check(tmc2[0] == '01:00:00', "tmc int 3600")
check(tmc2[1] == '02:00:00', "tmc int 7200")

# ==================== TimeColumn invalid ====================
print("=== TimeColumn invalid ===")
tmc3 = TimeColumn(1)
try:
    tmc3.append('invalid')
    check(False, "should raise ValueError for invalid time")
except ValueError:
    check(True, "ValueError for invalid time")

# ==================== TimeColumn copy ====================
print("=== TimeColumn copy ===")
tmc_copy = tmc.copy()
check(tmc_copy.length == tmc.length, "tmc copy length")
check(tmc_copy[0] == tmc[0], "tmc copy [0]")
check(tmc_copy[2] == tmc[2], "tmc copy [2]")

# ==================== to_datetime ====================
print("=== to_datetime ===")
s = UCS2Column_SoA(3)
s.append('2024-01-01'); s.append('2023-06-15 12:00:00'); s.append('2020-12-31 23:30:00')
res = to_datetime(s)
check(res.length == 3, "to_datetime length")
check(str(res[0]) == '2024-01-01', "to_datetime [0]")
check(str(res[1]) == '2023-06-15 12:00:00', "to_datetime [1]")
check(str(res[2]) == '2020-12-31 23:30:00', "to_datetime [2]")

# ==================== to_time ====================
print("=== to_time ===")
s2 = UCS2Column_SoA(2)
s2.append('10:30:00'); s2.append('23:59:59')
res2 = to_time(s2)
check(res2.length == 2, "to_time length")
check(res2[0] == '10:30:00', "to_time [0]")
check(res2[1] == '23:59:59', "to_time [1]")

# ==================== to_date ====================
print("=== to_date ===")
s3 = UCS2Column_SoA(2)
s3.append('2024-01-01'); s3.append('20230615')
res3 = to_date(s3)
check(res3.length == 2, "to_date length")
check(res3[0] == 20240101, "to_date [0]")
check(res3[1] == 20230615, "to_date [1]")

# ==================== astype: DateTimeColumn -> others ====================
print("=== DateTimeColumn astype ===")
dtc_a = DateTimeColumn(2)
dtc_a.append('2024-03-15 10:30:00')
dtc_a.append('2024-01-01')

ic = dtc_a.astype('int')
check(ic[0] != 0, "dt->int")

dc = dtc_a.astype('double')
check(dc[0] != 0.0, "dt->double")

uc = dtc_a.astype('ucs2')
check(uc[0] == '2024-03-15 10:30:00', "dt->ucs2")

cc = dtc_a.astype('char256')
check(cc[0] == '2024-03-15 10:30:00', "dt->char256")

catc = dtc_a.astype('category')
check(catc[0] == '2024-03-15 10:30:00', "dt->category")

tc = dtc_a.astype('timestamp')
check(tc.length == 2, "dt->timestamp length")

tmc = dtc_a.astype('time')
check(tmc.length == 2, "dt->time length")

dtc_b = dtc_a.astype('datetime')
check(str(dtc_b[0]) == '2024-03-15 10:30:00', "dt->datetime copy")

# ==================== astype: TimeColumn -> others ====================
print("=== TimeColumn astype ===")
tmc_a = TimeColumn(2)
tmc_a.append(3600)
tmc_a.append(7200)

ic2 = tmc_a.astype('int')
check(ic2[0] == 3600, "time->int")

dc2 = tmc_a.astype('double')
check(dc2[0] == 3600.0, "time->double")

uc2 = tmc_a.astype('ucs2')
check(uc2[0] == '01:00:00', "time->ucs2")

cc2 = tmc_a.astype('char256')
check(cc2[0] == '01:00:00', "time->char256")

catc2 = tmc_a.astype('category')
check(catc2[0] == '01:00:00', "time->category")

bc2 = tmc_a.astype('bool')
check(bc2[0] == 1, "time->bool non-zero")

tc2 = tmc_a.astype('timestamp')
check(tc2.length == 2, "time->timestamp length")

# ==================== astype: others -> DateTimeColumn ====================
print("=== astype -> DateTimeColumn ===")
ic3 = IntColumn(2)
ic3.append(100); ic3.append(200)
dtc_c = ic3.astype('datetime')
check(dtc_c.length == 2, "int->datetime length")

dc3 = DoubleColumn(2)
dc3.append(100.0); dc3.append(200.0)
dtc_d = dc3.astype('datetime')
check(dtc_d.length == 2, "double->datetime length")

uc3 = UCS2Column_SoA(1)
uc3.append('2024-06-01 12:00:00')
dtc_e = uc3.astype('datetime')
check(str(dtc_e[0]) == '2024-06-01 12:00:00', "ucs2->datetime")

cc3 = Char256Column_SoA(1)
cc3.append('2024-06-01 12:00:00')
dtc_f = cc3.astype('datetime')
check(str(dtc_f[0]) == '2024-06-01 12:00:00', "char256->datetime")

# ==================== astype: others -> TimeColumn ====================
print("=== astype -> TimeColumn ===")
ic4 = IntColumn(2)
ic4.append(3600); ic4.append(7200)
tmc_b = ic4.astype('time')
check(tmc_b[0] == '01:00:00', "int->time")
check(tmc_b[1] == '02:00:00', "int->time [1]")

dc4 = DoubleColumn(2)
dc4.append(3600.0); dc4.append(7200.0)
tmc_c = dc4.astype('time')
check(tmc_c[0] == '01:00:00', "double->time")

bc3 = BoolColumn(2)
bc3.append(1); bc3.append(0)
tmc_d = bc3.astype('time')
check(tmc_d[0] == '00:00:01', "bool->time")

uc4 = UCS2Column_SoA(1)
uc4.append('01:30:00')
tmc_e = uc4.astype('time')
check(tmc_e[0] == '01:30:00', "ucs2->time")

cc4 = Char256Column_SoA(1)
cc4.append('12:00:00')
tmc_f = cc4.astype('time')
check(tmc_f[0] == '12:00:00', "char256->time")

# ==================== FastDataFrame with DateTime/Time schema ====================
print("=== FastDataFrame schema ===")
df = FastDataFrame(schema={'TS': 'datetime', 'TM': 'time', 'V': 'int'})
check('TS' in df.columns, "schema has TS")
check('TM' in df.columns, "schema has TM")
check('V' in df.columns, "schema has V")
check(df.columns['TS'].dtype == 8, "TS is TYPE_DATETIME")
check(df.columns['TM'].dtype == 9, "TM is TYPE_TIME")

# ==================== FastDataFrame from_dict with datetime/time ====================
print("=== FastDataFrame from_dict ===")
df2 = FastDataFrame.from_dict(
    {'TS': ['2024-01-01', '2024-06-15 12:00:00'],
     'TM': ['10:30', '23:59:00'],
     'V': [1, 2]},
    schema={'TS': 'datetime', 'TM': 'time', 'V': 'int'}
)
check(df2.df_length == 2, "from_dict length")
check(df2['TS'].dtype == 8, "from_dict TS dtype")
check(df2['TM'].dtype == 9, "from_dict TM dtype")
check(str(df2['TS'][0]) == '2024-01-01', "from_dict TS [0]")
check(str(df2['TS'][1]) == '2024-06-15 12:00:00', "from_dict TS [1]")
check(df2['TM'][0] == '10:30:00', "from_dict TM [0]")
check(df2['TM'][1] == '23:59:00', "from_dict TM [1]")
check(df2['V'][0] == 1, "from_dict V [0]")

# ==================== FastDataFrame from_list with datetime/time ====================
print("=== FastDataFrame from_list ===")
df3 = FastDataFrame.from_list(
    [['2024-01-01', '10:30', 1],
     ['2024-06-15 12:00:00', '23:59:00', 2]],
    schema={'TS': 'datetime', 'TM': 'time', 'V': 'int'}
)
check(df3.df_length == 2, "from_list length")
check(str(df3['TS'][0]) == '2024-01-01', "from_list TS [0]")
check(df3['TM'][0] == '10:30:00', "from_list TM [0]")

# ==================== FastDataFrame from_arrays with datetime/time ====================
print("=== FastDataFrame from_arrays ===")
df4 = FastDataFrame.from_arrays(
    [['2024-01-01', '2024-06-15'], ['10:30', '23:59']],
    schema={'TS': 'datetime', 'TM': 'time'}
)
check(df4.df_length == 2, "from_arrays length")
check(str(df4['TS'][0]) == '2024-01-01', "from_arrays TS [0]")
check(df4['TM'][1] == '23:59:00', "from_arrays TM [1]")

# ==================== FastDataFrame __setitem__ with datetime/time ====================
print("=== FastDataFrame __setitem__ ===")
df5 = FastDataFrame(schema={'V': 'int'})
df5['V'] = [1, 2, 3]
dtc_set = DateTimeColumn(3)
dtc_set.append('2024-01-01')
dtc_set.append('2024-06-15')
dtc_set.append('2024-12-31')
df5['TS'] = dtc_set
check(df5['TS'].dtype == 8, "__setitem__ TS dtype")
check(str(df5['TS'][0]) == '2024-01-01', "__setitem__ TS [0]")

tmc_set = TimeColumn(3)
tmc_set.append(3600)
tmc_set.append(7200)
tmc_set.append(86400)
df5['TM'] = tmc_set
check(df5['TM'].dtype == 9, "__setitem__ TM dtype")
check(df5['TM'][0] == '01:00:00', "__setitem__ TM [0]")

# ==================== FastDataFrame dtypes ====================
print("=== FastDataFrame dtypes ===")
dtypes = df2.dtypes
check(dtypes['TS'] == __import__('fluidframe._schema_types', fromlist=['DateTime']).DateTime, "dtypes TS")
check(dtypes['TM'] == __import__('fluidframe._schema_types', fromlist=['Time']).Time, "dtypes TM")

# ==================== FastDataFrame info ====================
print("=== FastDataFrame info ===")
info_str = df2.info()
check('datetime' in info_str.lower() or 'TS' in info_str, "info includes datetime")

# ==================== FastDataFrame describe ====================
print("=== FastDataFrame describe ===")
desc = df2.describe()
check('TS' in desc, "describe includes TS")
check('TM' in desc, "describe includes TM")

# ==================== concat with datetime/time ====================
print("=== concat with datetime/time ===")
df_a = FastDataFrame.from_dict(
    {'TS': ['2024-01-01'], 'TM': ['10:30']},
    schema={'TS': 'datetime', 'TM': 'time'}
)
df_b = FastDataFrame.from_dict(
    {'TS': ['2024-06-15'], 'TM': ['23:59']},
    schema={'TS': 'datetime', 'TM': 'time'}
)
df_cat = concat([df_a, df_b])
check(df_cat.df_length == 2, "concat length")
check(str(df_cat['TS'][0]) == '2024-01-01', "concat TS [0]")
check(str(df_cat['TS'][1]) == '2024-06-15', "concat TS [1]")
check(df_cat['TM'][0] == '10:30:00', "concat TM [0]")
check(df_cat['TM'][1] == '23:59:00', "concat TM [1]")

# ==================== Drop index with datetime/time ====================
print("=== Drop index with datetime/time ===")
df6 = FastDataFrame.from_dict(
    {'TS': ['2024-01-01', '2024-06-15', '2024-12-31'],
     'TM': ['10:30', '12:00', '23:59'],
     'V': [1, 2, 3]},
    schema={'TS': 'datetime', 'TM': 'time', 'V': 'int'}
)
df6_dropped = df6.drop_index(1)
check(df6_dropped.df_length == 2, "drop_index length")
check(str(df6_dropped['TS'][0]) == '2024-01-01', "drop_index TS [0]")
check(str(df6_dropped['TS'][1]) == '2024-12-31', "drop_index TS [1]")

# ==================== min/max on DateTimeColumn ====================
print("=== DateTimeColumn min/max ===")
dtc_minmax = DateTimeColumn(3)
dtc_minmax.append('2024-06-15 12:00:00')
dtc_minmax.append('2024-01-01 00:00:00')
dtc_minmax.append('2024-12-31 23:59:59')
dtc_min = dtc_minmax.min()
dtc_max = dtc_minmax.max()
check(dtc_min < dtc_max, "dtc min < max")

# ==================== min/max on TimeColumn ====================
print("=== TimeColumn min/max ===")
tmc_minmax = TimeColumn(3)
tmc_minmax.append(3600)
tmc_minmax.append(86400)
tmc_minmax.append(0)
tmc_min = tmc_minmax.min()
tmc_max = tmc_minmax.max()
check(tmc_min < tmc_max, "tmc min < max")

print(f"\nResults: {PASSED} passed, {FAILED} failed")
if FAILED > 0:
    sys.exit(1)
