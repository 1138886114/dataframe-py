import sys
sys.path.insert(0, '..')
from fluidframe.core.datecolumn import DateColumn as DateColumn

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

def test_empty():
    col = DateColumn()
    check('empty length', col.length == 0)
    check('empty capacity', col.capacity == 16)

def test_append_int():
    col = DateColumn(); col.append(20160101); col.append(20231225)
    check('append int length', col.length == 2)
    check('append int [0]', col[0] == 20160101)
    check('append int [1]', col[1] == 20231225)

def test_append_str():
    col = DateColumn(); col.append("20160101"); col.append("20231225")
    check('append str length', col.length == 2)
    check('append str [0]', col[0] == 20160101)
    check('append str [1]', col[1] == 20231225)

def test_append_mixed():
    col = DateColumn(); col.append(20160101); col.append("20231225")
    check('append mixed [0]', col[0] == 20160101)
    check('append mixed [1]', col[1] == 20231225)

def test_bounds():
    col = DateColumn(); col.append(20160101)
    check('getitem out of range', expect_exception(IndexError, col.__getitem__, 1))

def test_pop_back():
    col = DateColumn(); col.append(20160101); col.append(20231225)
    val = col.pop_back()
    check('pop_back val', val == 20231225)
    check('pop_back len', col.length == 1)

    col = DateColumn()
    check('pop_back empty', expect_exception(IndexError, col.pop_back))

def test_pop_front():
    col = DateColumn()
    col.append(20160101); col.append(20231225); col.append(20240615)
    val = col.pop_front()
    check('pop_front val', val == 20160101)
    check('pop_front len', col.length == 2)
    check('pop_front [0]', col[0] == 20231225)

    col = DateColumn()
    check('pop_front empty', expect_exception(IndexError, col.pop_front))

def test_copy():
    col = DateColumn(); col.append(20160101)
    col2 = col.copy()
    check('copy length', col2.length == 1)
    check('copy [0]', col2[0] == 20160101)
    col2.append(20231225)
    check('copy independent', col.length == 1)

def test_concat():
    col1 = DateColumn(); col1.append(20160101)
    col2 = DateColumn(); col2.append(20231225)
    col1.concat(col2)
    check('concat length', col1.length == 2)
    check('concat [0]', col1[0] == 20160101)
    check('concat [1]', col1[1] == 20231225)

def test_max():
    col = DateColumn()
    col.append(20160101); col.append(20231225); col.append(20200615)
    check('max', col.max() == 20231225)

    col = DateColumn()
    check('max empty', expect_exception(ValueError, col.max))

def test_min():
    col = DateColumn()
    col.append(20160101); col.append(20231225); col.append(20200615)
    check('min', col.min() == 20160101)

    col = DateColumn()
    check('min empty', expect_exception(ValueError, col.min))

def test_year():
    col = DateColumn(); col.append(20160101); col.append(20231225)
    res = col.year()
    check('year [0]', res[0] == 2016)
    check('year [1]', res[1] == 2023)

def test_month():
    col = DateColumn(); col.append(20160101); col.append(20231225)
    res = col.month()
    check('month [0]', res[0] == 1)
    check('month [1]', res[1] == 12)

def test_day():
    col = DateColumn(); col.append(20160101); col.append(20231225)
    res = col.day()
    check('day [0]', res[0] == 1)
    check('day [1]', res[1] == 25)

def test_strftime():
    col = DateColumn(); col.append(20160101)
    res = col.strftime("%Y-%m-%d")
    check('strftime', res[0] == "2016-01-01")

    col = DateColumn(); col.append(20231225)
    res = col.strftime("%d/%m/%Y")
    check('strftime custom', res[0] == "25/12/2023")

def test_unique():
    col = DateColumn()
    col.append(20160101); col.append(20231225); col.append(20160101)
    check('unique', col.unique() == [20160101, 20231225])

    col = DateColumn()
    check('unique empty', col.unique() == [])

def test_growth():
    col = DateColumn()
    for i in range(100):
        col.append(20160101 + i)
    check('growth length', col.length == 100)
    # 20160101 + 99 = 20160200 (invalid YYYYMMDD), stored as epoch days -> 20160131
    check('growth [99]', col[99] == 20160131)

def test_reorder():
    col = DateColumn()
    col.append(20160101); col.append(20231225); col.append(20200615)
    col._reorder([2, 0, 1])
    check('reorder [0]', col[0] == 20200615)
    check('reorder [1]', col[1] == 20160101)
    check('reorder [2]', col[2] == 20231225)

def test_large():
    col = DateColumn(); col.append(99991231)
    check('large timestamp', col[0] == 99991231)

def test_filter_methods():
    col = DateColumn()
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

def test_add_days():
    col = DateColumn()
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

def test_add_months():
    col = DateColumn()
    col.append(20220131); col.append(20220612); col.append(20240229)

    r = col.add_months(1)
    check('add_months 1 [0]', r[0] == 20220228)
    check('add_months 1 [1]', r[1] == 20220712)
    check('add_months 1 [2]', r[2] == 20240329)

    r = col.sub_months(1)
    check('sub_months 1 [0]', r[0] == 20211231)
    check('sub_months 1 [1]', r[1] == 20220512)

    r = col.add_months(12)
    check('add_months 12', r[1] == 20230612)

    r = col.add_months(0)
    check('add_months 0', r[0] == col[0])

def test_add_years():
    col = DateColumn()
    col.append(20200229); col.append(20220612)

    r = col.add_years(1)
    check('add_years 1 [0]', r[0] == 20210228)
    check('add_years 1 [1]', r[1] == 20230612)

    r = col.sub_years(1)
    check('sub_years 1 [0]', r[0] == 20190228)
    check('sub_years 1 [1]', r[1] == 20210612)

    r = col.add_years(4)
    check('add_years 4 [0]', r[0] == 20240229)

def test_day_of_week():
    col = DateColumn()
    col.append(20220612)  # Sunday
    col.append(20230115)  # Sunday
    col.append(20240229)  # Thursday

    r = col.day_of_week()
    check('dow Sun', r[0] == 6)
    check('dow Sun2', r[1] == 6)
    check('dow Thu', r[2] == 3)

    col = DateColumn()
    col.append(20230102)  # Monday
    r = col.day_of_week()
    check('dow Mon', r[0] == 0)

    col = DateColumn()
    col.append(20230106)  # Friday
    r = col.day_of_week()
    check('dow Fri', r[0] == 4)

def test_day_of_year():
    col = DateColumn()
    col.append(20230101); col.append(20231231); col.append(20240229)

    r = col.day_of_year()
    check('doy Jan1', r[0] == 1)
    check('doy Dec31', r[1] == 365)
    check('doy Feb29 leap', r[2] == 60)

def test_days_in_month():
    col = DateColumn()
    col.append(20230201); col.append(20240201); col.append(20230401)

    r = col.days_in_month()
    check('dim Feb non-leap', r[0] == 28)
    check('dim Feb leap', r[1] == 29)
    check('dim Apr', r[2] == 30)

def test_is_leap_year():
    col = DateColumn()
    col.append(20200101); col.append(20230101); col.append(24000101)

    r = col.is_leap_year()
    check('leap 2020', r[0] == 1)
    check('leap 2023', r[1] == 0)
    check('leap 2400', r[2] == 1)

def test_is_weekend():
    col = DateColumn()
    col.append(20220612)  # Sunday
    col.append(20220611)  # Saturday
    col.append(20220613)  # Monday

    r = col.is_weekend()
    check('weekend Sun', r[0] == 1)
    check('weekend Sat', r[1] == 1)
    check('weekend Mon', r[2] == 0)

def test_is_month_start_end():
    col = DateColumn()
    col.append(20230101); col.append(20230131); col.append(20230115)

    r = col.is_month_start()
    check('month_start 01', r[0] == 1)
    check('month_start 31', r[1] == 0)
    check('month_start 15', r[2] == 0)

    r = col.is_month_end()
    check('month_end 01', r[0] == 0)
    check('month_end 31', r[1] == 1)
    check('month_end 15', r[2] == 0)

    col = DateColumn()
    col.append(20230228); col.append(20240229)
    r = col.is_month_end()
    check('month_end Feb non-leap', r[0] == 1)
    check('month_end Feb leap', r[1] == 1)

def test_is_year_start_end():
    col = DateColumn()
    col.append(20230101); col.append(20231231); col.append(20230615)

    r = col.is_year_start()
    check('year_start', r[0] == 1 and r[1] == 0 and r[2] == 0)

    r = col.is_year_end()
    check('year_end', r[0] == 0 and r[1] == 1 and r[2] == 0)

def test_diff_days():
    col = DateColumn()
    col.append(20220612); col.append(20230115)

    r = col.diff_days(20220101)
    check('diff_days [0]', r[0] == 162)
    check('diff_days [1]', r[1] == 379)

    r = col.diff_days(20220612)
    check('diff_days same', r[0] == 0)

    r = col.diff_days(20220613)
    check('diff_days neg', r[0] == -1)

def test_edge_cases():
    col = DateColumn()
    col.append(20230131)
    r = col.add_months(1)
    check('Jan31+1m=Feb28', r[0] == 20230228)

    col = DateColumn()
    col.append(20230331)
    r = col.add_months(-1)
    check('Mar31-1m=Feb28', r[0] == 20230228)

    col = DateColumn()
    col.append(20240229)
    r = col.add_months(-1)
    check('Feb29 leap -1m=Jan29', r[0] == 20240129)

    col = DateColumn()
    col.append(20240131)
    r = col.add_months(12)
    check('Jan31+12m=20250131', r[0] == 20250131)

def test_quarter():
    col = DateColumn()
    for d in [20240115, 20240415, 20240715, 20241015]:
        col.append(d)
    r = col.quarter()
    check('quarter Q1', r[0] == 1)
    check('quarter Q2', r[1] == 2)
    check('quarter Q3', r[2] == 3)
    check('quarter Q4', r[3] == 4)

    col2 = DateColumn(); col2.append(20240331)
    check('quarter Mar31=Q1', col2.quarter()[0] == 1)
    col2.append(20240630)
    check('quarter Jun30=Q2', col2.quarter()[1] == 2)

def test_is_quarter_start_end():
    col = DateColumn()
    for d in [20240101, 20240401, 20240701, 20241001]:
        col.append(d)
    r = col.is_quarter_start()
    check('quarter start 4 dates', r[0] == 1 and r[1] == 1 and r[2] == 1 and r[3] == 1)

    col2 = DateColumn()
    for d in [20240115, 20240331, 20240630, 20240930, 20241231]:
        col2.append(d)
    r2 = col2.is_quarter_end()
    check('quarter end 01/15', r2[0] == 0)
    check('quarter end 03/31', r2[1] == 1)
    check('quarter end 06/30', r2[2] == 1)
    check('quarter end 09/30', r2[3] == 1)
    check('quarter end 12/31', r2[4] == 1)

def test_week_of_year():
    col = DateColumn()
    col.append(20240101)  # Monday, week 1
    col.append(20241231)  # Tuesday, week 1 of 2025
    col.append(20230101)  # Sunday, week 52 of 2022
    r = col.week_of_year()
    check('week 2024-01-01', r[0] == 1)
    check('week 2024-12-31', r[1] == 1)
    check('week 2023-01-01', r[2] == 52)

def test_day_name():
    col = DateColumn()
    col.append(20240101)  # Monday
    col.append(20240102)  # Tuesday
    col.append(20240103)  # Wednesday
    col.append(20240104)  # Thursday
    col.append(20240105)  # Friday
    col.append(20240106)  # Saturday
    col.append(20240107)  # Sunday
    r = col.day_name()
    check('day_name Mon', r[0] == 'Monday')
    check('day_name Tue', r[1] == 'Tuesday')
    check('day_name Wed', r[2] == 'Wednesday')
    check('day_name Thu', r[3] == 'Thursday')
    check('day_name Fri', r[4] == 'Friday')
    check('day_name Sat', r[5] == 'Saturday')
    check('day_name Sun', r[6] == 'Sunday')

def test_month_name():
    col = DateColumn()
    for m in range(1, 13):
        col.append(2024 * 10000 + m * 100 + 1)
    r = col.month_name()
    expected = ['January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December']
    for i in range(12):
        check(f'month_name {expected[i]}', r[i] == expected[i])

def test_last_day_of_month():
    col = DateColumn()
    col.append(20240215)
    col.append(20240101)
    col.append(20230201)
    r = col.last_day_of_month()
    check('last_day Feb2024(leap)', r[0] == 20240229)
    check('last_day Jan2024', r[1] == 20240131)
    check('last_day Feb2023(non-leap)', r[2] == 20230228)

def test_iso_calendar():
    col = DateColumn()
    col.append(20240101)
    col.append(20241231)
    r_y, r_w, r_d = col.iso_calendar()
    check('iso_cal year', r_y[0] == 2024 and r_y[1] == 2025)  # 2024-12-31 is ISO year 2025
    check('iso_cal week', r_w[0] == 1 and r_w[1] == 1)
    check('iso_cal day Mon=1', r_d[0] == 1)
    check('iso_cal day Tue=2', r_d[1] == 2)

def test_diff_months():
    col = DateColumn()
    col.append(20240315)
    col.append(20240101)
    r = col.diff_months(20240101)
    check('diff_months 2mo', r[0] == 2)
    check('diff_months 0mo', r[1] == 0)

    col2 = DateColumn(); col2.append(20230101)
    r2 = col2.diff_months(20240101)
    check('diff_months -12mo', r2[0] == -12)

def test_diff_years():
    col = DateColumn()
    col.append(20260315)
    col.append(20240315)
    col.append(20230601)
    r = col.diff_years(20240101)
    check('diff_years 2yr', r[0] == 2)
    check('diff_years 0yr (same year)', r[1] == 0)
    check('diff_years -1yr', r[2] == -1)

def run_all():
    print('=== DateColumn Tests ===')
    test_empty()
    test_append_int()
    test_append_str()
    test_append_mixed()
    test_bounds()
    test_pop_back()
    test_pop_front()
    test_copy()
    test_concat()
    test_max()
    test_min()
    test_year()
    test_month()
    test_day()
    test_strftime()
    test_unique()
    test_growth()
    test_reorder()
    test_large()
    test_filter_methods()
    test_add_days()
    test_add_months()
    test_add_years()
    test_day_of_week()
    test_day_of_year()
    test_days_in_month()
    test_is_leap_year()
    test_is_weekend()
    test_is_month_start_end()
    test_is_year_start_end()
    test_diff_days()
    test_edge_cases()
    test_quarter()
    test_is_quarter_start_end()
    test_week_of_year()
    test_day_name()
    test_month_name()
    test_last_day_of_month()
    test_iso_calendar()
    test_diff_months()
    test_diff_years()
    print(f'\nResults: {passed} passed, {failed} failed')
    if failed: sys.exit(1)

if __name__ == '__main__':
    run_all()
