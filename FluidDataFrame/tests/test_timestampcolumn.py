import sys
sys.path.insert(0, '..')
from fluidframe.core.timestampcolumn import TimestampColumn

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f'  FAIL: {name}')

def expect_exception(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        return False
    except exc_type:
        return True
    except Exception:
        return False

print('=== TimestampColumn Tests ===')

col = TimestampColumn()
check('empty length', col.length == 0)
check('empty capacity', col.capacity == 16)

col = TimestampColumn(); col.append(20160101); col.append(20231225)
check('append int length', col.length == 2)
check('append int [0]', col[0] == 20160101)
check('append int [1]', col[1] == 20231225)

col = TimestampColumn(); col.append("20160101"); col.append("20231225")
check('append str length', col.length == 2)
check('append str [0]', col[0] == 20160101)
check('append str [1]', col[1] == 20231225)

col = TimestampColumn(); col.append(20160101); col.append("20231225")
check('append mixed [0]', col[0] == 20160101)
check('append mixed [1]', col[1] == 20231225)

col = TimestampColumn(); col.append(20160101)
check('getitem out of range', expect_exception(IndexError, col.__getitem__, 1))

col = TimestampColumn(); col.append(20160101); col.append(20231225)
val = col.pop_back()
check('pop_back val', val == 20231225)
check('pop_back len', col.length == 1)

col = TimestampColumn()
check('pop_back empty', expect_exception(IndexError, col.pop_back))

col = TimestampColumn()
col.append(20160101); col.append(20231225); col.append(20240615)
val = col.pop_front()
check('pop_front val', val == 20160101)
check('pop_front len', col.length == 2)
check('pop_front [0]', col[0] == 20231225)

col = TimestampColumn()
check('pop_front empty', expect_exception(IndexError, col.pop_front))

col = TimestampColumn(); col.append(20160101)
col2 = col.copy()
check('copy length', col2.length == 1)
check('copy [0]', col2[0] == 20160101)
col2.append(20231225)
check('copy independent', col.length == 1)

col1 = TimestampColumn(); col1.append(20160101)
col2 = TimestampColumn(); col2.append(20231225)
col1.concat(col2)
check('concat length', col1.length == 2)
check('concat [0]', col1[0] == 20160101)
check('concat [1]', col1[1] == 20231225)

col = TimestampColumn()
col.append(20160101); col.append(20231225); col.append(20200615)
check('max', col.max() == 20231225)

col = TimestampColumn()
check('max empty', expect_exception(ValueError, col.max))

col = TimestampColumn()
col.append(20160101); col.append(20231225); col.append(20200615)
check('min', col.min() == 20160101)

col = TimestampColumn()
check('min empty', expect_exception(ValueError, col.min))

col = TimestampColumn(); col.append(20160101); col.append(20231225)
res = col.year()
check('year [0]', res[0] == 2016)
check('year [1]', res[1] == 2023)

col = TimestampColumn(); col.append(20160101); col.append(20231225)
res = col.month()
check('month [0]', res[0] == 1)
check('month [1]', res[1] == 12)

col = TimestampColumn(); col.append(20160101); col.append(20231225)
res = col.day()
check('day [0]', res[0] == 1)
check('day [1]', res[1] == 25)

col = TimestampColumn(); col.append(20160101)
res = col.strftime("%Y-%m-%d")
check('strftime', res[0] == "2016-01-01")

col = TimestampColumn(); col.append(20231225)
res = col.strftime("%d/%m/%Y")
check('strftime custom', res[0] == "25/12/2023")

col = TimestampColumn()
col.append(20160101); col.append(20231225); col.append(20160101)
check('unique', col.unique() == [20160101, 20231225])

col = TimestampColumn()
check('unique empty', col.unique() == [])

col = TimestampColumn()
for i in range(100):
    col.append(20160101 + i)
check('growth length', col.length == 100)
check('growth [99]', col[99] == 20160101 + 99)

col = TimestampColumn()
col.append(20160101); col.append(20231225); col.append(20200615)
col._reorder([2, 0, 1])
check('reorder [0]', col[0] == 20200615)
check('reorder [1]', col[1] == 20160101)
check('reorder [2]', col[2] == 20231225)

col = TimestampColumn(); col.append(99991231)
check('large timestamp', col[0] == 99991231)

# =====================================================================
#  Filter methods
# =====================================================================
print('\n--- Filter methods ---')

col = TimestampColumn()
col.append(20220612); col.append(20230115); col.append(20240229)

r = col.filter_greater(20230101)
check('filter_greater [0]', r[0] == 0)
check('filter_greater [1]', r[1] == 1)
check('filter_greater [2]', r[2] == 1)

r = col.filter_greater('20230101')
check('filter_greater str', r[0] == 0 and r[1] == 1)

r = col.filter_greater_equal(20230115)
check('filter_ge [0]', r[0] == 0)
check('filter_ge [1]', r[1] == 1)
check('filter_ge [2]', r[2] == 1)

r = col.filter_less(20230101)
check('filter_less [0]', r[0] == 1)
check('filter_less [1]', r[1] == 0)
check('filter_less [2]', r[2] == 0)

r = col.filter_less_equal(20220612)
check('filter_le [0]', r[0] == 1)
check('filter_le [1]', r[1] == 0)
check('filter_le [2]', r[2] == 0)

r = col.filter_equal(20220612)
check('filter_equal [0]', r[0] == 1)
check('filter_equal [1]', r[1] == 0)
check('filter_equal [2]', r[2] == 0)

r = col.filter_not_equal(20220612)
check('filter_ne [0]', r[0] == 0)
check('filter_ne [1]', r[1] == 1)
check('filter_ne [2]', r[2] == 1)

r = col.filter_between(20220101, 20230630)
check('filter_between [0]', r[0] == 1)
check('filter_between [1]', r[1] == 1)
check('filter_between [2]', r[2] == 0)

# =====================================================================
#  Date arithmetic: add_days / sub_days
# =====================================================================
print('\n--- add_days / sub_days ---')

col = TimestampColumn()
col.append(20220612); col.append(20240229); col.append(20221231)

r = col.add_days(1)
check('add_days 1 [0]', r[0] == 20220613)
check('add_days 1 [1]', r[1] == 20240301)
check('add_days 1 [2]', r[2] == 20230101)

r = col.sub_days(1)
check('sub_days 1 [0]', r[0] == 20220611)
check('sub_days 1 [1]', r[1] == 20240228)
check('sub_days 1 [2]', r[2] == 20221230)

r = col.add_days(365)
check('add_days 365 [0]', r[0] == 20230612)

r = col.add_days(0)
check('add_days 0', r[0] == col[0] and r[1] == col[1])

# =====================================================================
#  Date arithmetic: add_months / sub_months
# =====================================================================
print('\n--- add_months / sub_months ---')

col = TimestampColumn()
col.append(20220131); col.append(20220612); col.append(20240229)

r = col.add_months(1)
check('add_months 1 [0]', r[0] == 20220228)  # Jan 31 + 1m = Feb 28
check('add_months 1 [1]', r[1] == 20220712)
check('add_months 1 [2]', r[2] == 20240329)  # Feb 29 + 1m = Mar 29

r = col.sub_months(1)
check('sub_months 1 [0]', r[0] == 20211231)
check('sub_months 1 [1]', r[1] == 20220512)

r = col.add_months(12)
check('add_months 12', r[1] == 20230612)

r = col.add_months(0)
check('add_months 0', r[0] == col[0])

# =====================================================================
#  Date arithmetic: add_years / sub_years
# =====================================================================
print('\n--- add_years / sub_years ---')

col = TimestampColumn()
col.append(20200229); col.append(20220612)

r = col.add_years(1)
check('add_years 1 [0]', r[0] == 20210228)  # leap -> non-leap clamp
check('add_years 1 [1]', r[1] == 20230612)

r = col.sub_years(1)
check('sub_years 1 [0]', r[0] == 20190228)
check('sub_years 1 [1]', r[1] == 20210612)

r = col.add_years(4)
check('add_years 4 [0]', r[0] == 20240229)  # back to leap

# =====================================================================
#  Component: day_of_week (ISO 8601: 0=Mon … 6=Sun)
# =====================================================================
print('\n--- day_of_week ---')

col = TimestampColumn()
col.append(20220612)  # Sunday
col.append(20230115)  # Sunday
col.append(20240229)  # Thursday

r = col.day_of_week()
check('dow Sun', r[0] == 6)
check('dow Sun2', r[1] == 6)
check('dow Thu', r[2] == 3)

col = TimestampColumn()
col.append(20230102)  # Monday
r = col.day_of_week()
check('dow Mon', r[0] == 0)

col = TimestampColumn()
col.append(20230106)  # Friday
r = col.day_of_week()
check('dow Fri', r[0] == 4)

# =====================================================================
#  Component: day_of_year
# =====================================================================
print('\n--- day_of_year ---')

col = TimestampColumn()
col.append(20230101); col.append(20231231); col.append(20240229)

r = col.day_of_year()
check('doy Jan1', r[0] == 1)
check('doy Dec31', r[1] == 365)
check('doy Feb29 leap', r[2] == 60)

# =====================================================================
#  Component: days_in_month
# =====================================================================
print('\n--- days_in_month ---')

col = TimestampColumn()
col.append(20230201); col.append(20240201); col.append(20230401)

r = col.days_in_month()
check('dim Feb non-leap', r[0] == 28)
check('dim Feb leap', r[1] == 29)
check('dim Apr', r[2] == 30)

# =====================================================================
#  Component: is_leap_year
# =====================================================================
print('\n--- is_leap_year ---')

col = TimestampColumn()
col.append(20200101); col.append(20230101); col.append(24000101)

r = col.is_leap_year()
check('leap 2020', r[0] == 1)
check('leap 2023', r[1] == 0)
check('leap 2400', r[2] == 1)

# =====================================================================
#  Component: is_weekend
# =====================================================================
print('\n--- is_weekend ---')

col = TimestampColumn()
col.append(20220612)  # Sunday
col.append(20220611)  # Saturday
col.append(20220613)  # Monday

r = col.is_weekend()
check('weekend Sun', r[0] == 1)
check('weekend Sat', r[1] == 1)
check('weekend Mon', r[2] == 0)

# =====================================================================
#  Component: is_month_start / is_month_end
# =====================================================================
print('\n--- is_month_start / is_month_end ---')

col = TimestampColumn()
col.append(20230101); col.append(20230131); col.append(20230115)

r = col.is_month_start()
check('month_start 01', r[0] == 1)
check('month_start 31', r[1] == 0)
check('month_start 15', r[2] == 0)

r = col.is_month_end()
check('month_end 01', r[0] == 0)
check('month_end 31', r[1] == 1)
check('month_end 15', r[2] == 0)

# Feb end
col = TimestampColumn()
col.append(20230228); col.append(20240229)
r = col.is_month_end()
check('month_end Feb non-leap', r[0] == 1)
check('month_end Feb leap', r[1] == 1)

# =====================================================================
#  Component: is_year_start / is_year_end
# =====================================================================
print('\n--- is_year_start / is_year_end ---')

col = TimestampColumn()
col.append(20230101); col.append(20231231); col.append(20230615)

r = col.is_year_start()
check('year_start', r[0] == 1 and r[1] == 0 and r[2] == 0)

r = col.is_year_end()
check('year_end', r[0] == 0 and r[1] == 1 and r[2] == 0)

# =====================================================================
#  diff_days
# =====================================================================
print('\n--- diff_days ---')

col = TimestampColumn()
col.append(20220612); col.append(20230115)

r = col.diff_days(20220101)
check('diff_days [0]', r[0] == 162)  # 20220612 - 20220101 = 162 days
check('diff_days [1]', r[1] == 379)

r = col.diff_days(20220612)
check('diff_days same', r[0] == 0)

r = col.diff_days(20220613)
check('diff_days neg', r[0] == -1)

# =====================================================================
#  Edge cases: month arithmetic boundaries
# =====================================================================
print('\n--- edge cases ---')

col = TimestampColumn()
col.append(20230131)
r = col.add_months(1)
check('Jan31+1m=Feb28', r[0] == 20230228)

col = TimestampColumn()
col.append(20230331)
r = col.add_months(-1)
check('Mar31-1m=Feb28', r[0] == 20230228)

col = TimestampColumn()
col.append(20240229)
r = col.add_months(-1)
check('Feb29 leap -1m=Jan29', r[0] == 20240129)

col = TimestampColumn()
col.append(20240131)
r = col.add_months(12)
check('Jan31+12m=20250131', r[0] == 20250131)

print(f'\nResults: {passed} passed, {failed} failed')
if failed: sys.exit(1)
