import sys
sys.path.insert(0, '.')
from fluidframe.core.datecolumn import DateColumn
from fluidframe.core.intcolumn import IntColumn
from fluidframe.core.fastdataframe import FastDataFrame

passed = 0
failed = 0

def check(name, cond):
    global passed, failed
    if cond:
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


class TestDateColumnConstruction:
    def test_empty(self):
        col = DateColumn()
        check('empty length', col.length == 0)
        check('empty capacity', col.capacity == 16)

    def test_append_int(self):
        col = DateColumn(); col.append(20160101); col.append(20231225)
        check('length', col.length == 2)
        check('val 0', col[0] == 20160101)
        check('val 1', col[1] == 20231225)

    def test_append_str(self):
        col = DateColumn(); col.append("20160101"); col.append("20231225")
        check('length', col.length == 2)
        check('val 0', col[0] == 20160101)

    def test_append_mixed(self):
        col = DateColumn(); col.append(20160101); col.append("20231225")
        check('val 0', col[0] == 20160101)
        check('val 1', col[1] == 20231225)

    def test_bounds(self):
        col = DateColumn(); col.append(20160101)
        check('oob', expect_exception(IndexError, col.__getitem__, 1))


class TestDateColumnOps:
    def test_pop_back(self):
        col = DateColumn(); col.append(20160101); col.append(20231225)
        val = col.pop_back()
        check('val', val == 20231225)
        check('len', col.length == 1)

    def test_pop_front(self):
        col = DateColumn(); col.append(20160101); col.append(20231225); col.append(20240615)
        val = col.pop_front()
        check('val', val == 20160101)
        check('len', col.length == 2)
        check('new [0]', col[0] == 20231225)

    def test_copy(self):
        col = DateColumn(); col.append(20160101)
        col2 = col.copy()
        check('copy len', col2.length == 1)
        col2.append(20231225)
        check('independent', col.length == 1)

    def test_concat(self):
        c1 = DateColumn(); c1.append(20160101)
        c2 = DateColumn(); c2.append(20231225)
        c1.concat(c2)
        check('len', c1.length == 2)
        check('val', c1[0] == 20160101 and c1[1] == 20231225)


class TestDateColumnStats:
    def test_max(self):
        col = DateColumn(); col.append(20160101); col.append(20231225); col.append(20200615)
        check('max', col.max() == 20231225)

    def test_min(self):
        col = DateColumn(); col.append(20160101); col.append(20231225); col.append(20200615)
        check('min', col.min() == 20160101)

    def test_unique(self):
        col = DateColumn(); col.append(20160101); col.append(20231225); col.append(20160101)
        check('unique', col.unique() == [20160101, 20231225])


class TestDateColumnExtractors:
    def test_year(self):
        col = DateColumn(); col.append(20160101); col.append(20231225)
        r = col._year()
        check('year', r[0] == 2016 and r[1] == 2023)

    def test_month(self):
        col = DateColumn(); col.append(20160101); col.append(20231225)
        r = col._month()
        check('month', r[0] == 1 and r[1] == 12)

    def test_day(self):
        col = DateColumn(); col.append(20160101); col.append(20231225)
        r = col._day()
        check('day', r[0] == 1 and r[1] == 25)

    def test_strftime(self):
        col = DateColumn(); col.append(20160101)
        check('strftime', col.strftime("%Y-%m-%d")[0] == "2016-01-01")
        col2 = DateColumn(); col2.append(20231225)
        check('strftime custom', col2.strftime("%d/%m/%Y")[0] == "25/12/2023")


class TestDateColumnArithmetic:
    def test_add_days(self):
        col = DateColumn(); col.append(20220612); col.append(20240229)
        r = col.add_days(1)
        check('+1d', r[0] == 20220613 and r[1] == 20240301)
        r = col.sub_days(1)
        check('-1d', r[0] == 20220611 and r[1] == 20240228)
        r = col.add_days(0)
        check('+0d', r[0] == col[0])

    def test_add_months(self):
        col = DateColumn(); col.append(20220131); col.append(20220612)
        r = col.add_months(1)
        check('+1m Jan31', r[0] == 20220228)
        check('+1m Jun12', r[1] == 20220712)
        r = col.sub_months(1)
        check('-1m', r[0] == 20211231)

    def test_add_years(self):
        col = DateColumn(); col.append(20200229); col.append(20220612)
        r = col.add_years(1)
        check('+1y leap', r[0] == 20210228)
        r = col.add_years(4)
        check('+4y leap', r[0] == 20240229)

    def test_diff_days(self):
        col = DateColumn(); col.append(20220612); col.append(20230115)
        r = col.diff_days(20220101)
        check('diff', r[0] == 162 and r[1] == 379)
        r = col.diff_days(20220612)
        check('diff same', r[0] == 0)

    def test_diff_months(self):
        col = DateColumn(); col.append(20240315); col.append(20240101)
        r = col.diff_months(20240101)
        check('diff months', r[0] == 2 and r[1] == 0)

    def test_diff_years(self):
        col = DateColumn(); col.append(20260315); col.append(20240315)
        r = col.diff_years(20240101)
        check('diff years', r[0] == 2 and r[1] == 0)

    def test_edge_cases(self):
        col = DateColumn(); col.append(20230131)
        r = col.add_months(1)
        check('Jan31+1m', r[0] == 20230228)
        col = DateColumn(); col.append(20240229)
        r = col.add_months(-1)
        check('Feb29-1m', r[0] == 20240129)


class TestDateColumnCalendar:
    def test_day_of_week(self):
        col = DateColumn(); col.append(20220612)  # Sunday
        r = col.day_of_week()
        check('Sun', r[0] == 6)
        col = DateColumn(); col.append(20230102)  # Monday
        r = col.day_of_week()
        check('Mon', r[0] == 0)

    def test_day_of_year(self):
        col = DateColumn(); col.append(20230101); col.append(20231231); col.append(20240229)
        r = col.day_of_year()
        check('Jan1', r[0] == 1)
        check('Dec31', r[1] == 365)
        check('Feb29 leap', r[2] == 60)

    def test_days_in_month(self):
        col = DateColumn(); col.append(20230201); col.append(20240201); col.append(20230401)
        r = col._days_in_month()
        check('Feb non-leap', r[0] == 28)
        check('Feb leap', r[1] == 29)
        check('Apr', r[2] == 30)

    def test_is_leap_year(self):
        col = DateColumn(); col.append(20200101); col.append(20230101); col.append(24000101)
        r = col._is_leap_year()
        check('leap', r[0] == 1 and r[1] == 0 and r[2] == 1)

    def test_is_weekend(self):
        col = DateColumn(); col.append(20220612); col.append(20220611); col.append(20220613)
        r = col._is_weekend()
        check('weekend', r[0] == 1 and r[1] == 1 and r[2] == 0)

    def test_month_start_end(self):
        col = DateColumn(); col.append(20230101); col.append(20230131); col.append(20230115)
        check('month_start', col._is_month_start()[0] == 1)
        check('month_end', col._is_month_end()[1] == 1)

    def test_year_start_end(self):
        col = DateColumn(); col.append(20230101); col.append(20231231)
        check('year_start', col._is_year_start()[0] == 1)
        check('year_end', col._is_year_end()[1] == 1)

    def test_quarter(self):
        col = DateColumn()
        for d in [20240115, 20240415, 20240715, 20241015]: col.append(d)
        r = col._quarter()
        check('quarters', [r[i] for i in range(4)] == [1, 2, 3, 4])

    def test_quarter_start_end(self):
        col = DateColumn()
        for d in [20240101, 20240401, 20240701, 20241001]: col.append(d)
        check('q_start', all(col._is_quarter_start()[i] == 1 for i in range(4)))
        col2 = DateColumn()
        for d in [20240331, 20240630, 20240930, 20241231]: col2.append(d)
        check('q_end', all(col2._is_quarter_end()[i] == 1 for i in range(4)))

    def test_week_of_year(self):
        col = DateColumn(); col.append(20240101); col.append(20241231)
        r = col.week_of_year()
        check('week', r[0] == 1 and r[1] == 1)

    def test_day_name(self):
        col = DateColumn()
        for d in [20240101, 20240102, 20240103, 20240104, 20240105, 20240106, 20240107]:
            col.append(d)
        r = col.day_name()
        expected = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        check('day_names', [r[i] for i in range(7)] == expected)

    def test_month_name(self):
        col = DateColumn()
        for m in range(1, 13): col.append(2024 * 10000 + m * 100 + 1)
        r = col.month_name()
        expected = ['January', 'February', 'March', 'April', 'May', 'June',
                    'July', 'August', 'September', 'October', 'November', 'December']
        check('month_names', [r[i] for i in range(12)] == expected)

    def test_last_day_of_month(self):
        col = DateColumn(); col.append(20240215); col.append(20240101); col.append(20230201)
        r = col.last_day_of_month()
        check('last_day', r[0] == 20240229 and r[1] == 20240131 and r[2] == 20230228)

    def test_iso_calendar(self):
        col = DateColumn(); col.append(20240101)
        y, w, d = col.iso_calendar()
        check('iso year', y[0] == 2024)
        check('iso week', w[0] == 1)

    def test_filter_methods(self):
        col = DateColumn(); col.append(20220612); col.append(20230115); col.append(20240229)
        r = col.filter_greater(20230101)
        check('gt', [r[i] for i in range(3)] == [0, 1, 1])
        r = col.filter_less(20230101)
        check('lt', [r[i] for i in range(3)] == [1, 0, 0])
        r = col.filter_between(20220101, 20230630)
        check('between', [r[i] for i in range(3)] == [1, 1, 0])


class TestDateColumnMisc:
    def test_growth(self):
        col = DateColumn()
        for i in range(100): col.append(20160101 + i)
        check('growth', col.length == 100)

    def test_reorder(self):
        df = FastDataFrame()
        col = DateColumn(); col.append(20160101); col.append(20231225); col.append(20200615)
        df['x'] = col
        df.df_length = 3
        df = df.iloc[[2, 0, 1]]
        c = df['x']
        check('reorder', [c[i] for i in range(3)] == [20200615, 20160101, 20231225])

    def test_large(self):
        col = DateColumn(); col.append(99991231)
        check('large', col[0] == 99991231)


class TestDateColumnOwnership:
    def test_pop_front_raises_on_empty(self):
        col = DateColumn()
        check('pop_front empty', expect_exception(IndexError, col.pop_front))

    def test_pop_back_raises_on_empty(self):
        col = DateColumn()
        check('pop_back empty', expect_exception(IndexError, col.pop_back))

    def test_owning_pop_front_works(self):
        col = DateColumn(); col.append(20160101); col.append(20231225); col.append(20240615)
        col.pop_front()
        check('owning pop_front len', col.length == 2)
        check('owning pop_front val', col[0] == 20231225)

    def test_owning_pop_back_works(self):
        col = DateColumn(); col.append(20160101); col.append(20231225); col.append(20240615)
        col.pop_back()
        check('owning pop_back len', col.length == 2)
        check('owning pop_back val', col[1] == 20231225)


def run_all():
    print('=== DateColumn Tests ===')
    for cls in [TestDateColumnConstruction, TestDateColumnOps, TestDateColumnStats,
                TestDateColumnExtractors, TestDateColumnArithmetic, TestDateColumnCalendar,
                TestDateColumnMisc, TestDateColumnOwnership]:
        for name in dir(cls):
            if name.startswith('test_'):
                getattr(cls(), name)()
    print(f'\nResults: {passed} passed, {failed} failed')
    if failed: sys.exit(1)

if __name__ == '__main__':
    run_all()
