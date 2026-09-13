import sys
sys.path.insert(0, '.')
from fluidframe.core.datetimecolumn import DateTimeColumn
from fluidframe.core.intcolumn import IntColumn
from fluidframe.core.fastdataframe import FastDataFrame
from fluidframe._schema_types import Date

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


class TestDateTimeColumnConstruction:
    def test_empty(self):
        col = DateTimeColumn()
        check('empty', col.length == 0)

    def test_append_str(self):
        col = DateTimeColumn()
        col.append('2023-01-15 10:30:00'); col.append('2023-06-20 14:00:00')
        check('length', col.length == 2)
        check('val 0', col[0] == '2023-01-15 10:30:00')

    def test_append_int(self):
        col = DateTimeColumn()
        col.append(1673788200000000000)  # nanoseconds
        check('int append', col.length == 1)

    def test_invalid(self):
        col = DateTimeColumn()
        check('invalid', expect_exception(ValueError, col.append, 'not-a-date'))


class TestDateTimeColumnExtractors:
    def _make(self):
        col = DateTimeColumn()
        col.append('2023-01-15 10:30:45')
        col.append('2023-06-20 14:00:00')
        return col

    def test_year(self):
        col = self._make()
        r = col.year()
        check('year', r[0] == 2023 and r[1] == 2023)

    def test_month(self):
        col = self._make()
        r = col.month()
        check('month', r[0] == 1 and r[1] == 6)

    def test_day(self):
        col = self._make()
        r = col.day()
        check('day', r[0] == 15 and r[1] == 20)

    def test_hour(self):
        col = self._make()
        r = col.hour()
        check('hour', r[0] == 10 and r[1] == 14)

    def test_minute(self):
        col = self._make()
        r = col.minute()
        check('minute', r[0] == 30 and r[1] == 0)

    def test_second(self):
        col = self._make()
        r = col.second()
        check('second', r[0] == 45 and r[1] == 0)


class TestDateTimeColumnOps:
    def test_copy(self):
        col = DateTimeColumn(); col.append('2023-01-15 10:30:00')
        col2 = col.copy()
        check('copy len', col2.length == 1)
        col2.append('2023-06-20 14:00:00')
        check('independent', col.length == 1)

    def test_min_max(self):
        col = DateTimeColumn()
        col.append('2023-01-15 10:30:00'); col.append('2023-06-20 14:00:00')
        min_val = col.min()
        max_val = col.max()
        check('min < max', min_val < max_val)
        check('min is int', isinstance(min_val, int))


class TestDateTimeInDataFrame:
    def test_schema(self):
        df = FastDataFrame({'dt': 'datetime'})
        df.append_row(dt='2023-01-15 10:30:00')
        check('schema', df.shape == (1, 1))

    def test_from_dict(self):
        df = FastDataFrame.from_dict({'dt': ['2023-01-15 10:30:00', '2023-06-20 14:00:00']}, schema={'dt': 'datetime'})
        check('from_dict len', df.df_length == 2)


class TestDateTimeColumnOps2:
    def test_pop_front(self):
        col = DateTimeColumn()
        col.append('2023-01-15 10:30:00'); col.append('2023-06-20 14:00:00'); col.append('2024-01-01 08:00:00')
        val = col.pop_front()
        check('pop_front len', col.length == 2)

    def test_pop_back(self):
        col = DateTimeColumn()
        col.append('2023-01-15 10:30:00'); col.append('2023-06-20 14:00:00')
        val = col.pop_back()
        check('pop_back len', col.length == 1)

    def test_concat(self):
        c1 = DateTimeColumn(); c1.append('2023-01-15 10:30:00')
        c2 = DateTimeColumn(); c2.append('2023-06-20 14:00:00')
        c1.concat(c2)
        check('concat len', c1.length == 2)

    def test_drop_row(self):
        col = DateTimeColumn()
        col.append('2023-01-15 10:30:00'); col.append('2023-06-20 14:00:00'); col.append('2024-01-01 08:00:00')
        col.drop_row(1)
        check('drop_row len', col.length == 2)

    def test_unique(self):
        col = DateTimeColumn()
        col.append('2023-01-15 10:30:00'); col.append('2023-06-20 14:00:00'); col.append('2023-01-15 10:30:00')
        u = col.unique()
        check('unique len', len(u) == 2)

    def test_isin(self):
        col = DateTimeColumn()
        col.append('2023-01-15 10:30:00'); col.append('2023-06-20 14:00:00')
        r = col.isin({'2023-01-15 10:30:00'})
        check('isin', [r[i] for i in range(2)] == [True, False])

    def test_where_mask(self):
        mask = IntColumn(3); mask.append(1); mask.append(0); mask.append(1)
        col = DateTimeColumn()
        col.append('2023-01-15 10:30:00'); col.append('2023-06-20 14:00:00'); col.append('2024-01-01 08:00:00')
        r = col.where_mask(mask, '2000-01-01 00:00:00')
        check('where_mask masked', r[0] == '2023-01-15 10:30:00')
        check('where_mask fill', '2000' in r[1])

    def test_reorder(self):
        df = FastDataFrame()
        col = DateTimeColumn()
        col.append('2023-01-15 10:30:00'); col.append('2023-06-20 14:00:00'); col.append('2024-01-01 08:00:00')
        df['x'] = col
        df.df_length = 3
        df = df.iloc[[2, 0, 1]]
        c = df['x']
        check('reorder', c[0] == '2024-01-01 08:00:00')


class TestDateTimeColumnOwnership:
    def test_pop_front_raises_on_empty(self):
        col = DateTimeColumn()
        check('pop_front empty', expect_exception(IndexError, col.pop_front))

    def test_pop_back_raises_on_empty(self):
        col = DateTimeColumn()
        check('pop_back empty', expect_exception(IndexError, col.pop_back))

    def test_owning_pop_front_works(self):
        col = DateTimeColumn()
        col.append('2023-01-15 10:30:00'); col.append('2023-06-20 14:00:00'); col.append('2024-01-01 08:00:00')
        col.pop_front()
        check('owning pop_front len', col.length == 2)

    def test_owning_pop_back_works(self):
        col = DateTimeColumn()
        col.append('2023-01-15 10:30:00'); col.append('2023-06-20 14:00:00'); col.append('2024-01-01 08:00:00')
        col.pop_back()
        check('owning pop_back len', col.length == 2)


def run_all():
    print('=== DateTimeColumn Tests ===')
    for cls in [TestDateTimeColumnConstruction, TestDateTimeColumnExtractors,
                TestDateTimeColumnOps, TestDateTimeInDataFrame,
                TestDateTimeColumnOps2, TestDateTimeColumnOwnership]:
        for name in dir(cls):
            if name.startswith('test_'):
                getattr(cls(), name)()
    print(f'\nResults: {passed} passed, {failed} failed')
    if failed: sys.exit(1)

if __name__ == '__main__':
    run_all()
