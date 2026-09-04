import sys
sys.path.insert(0, '.')
from fluidframe.core.datetimecolumn import DateTimeColumn
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


def run_all():
    print('=== DateTimeColumn Tests ===')
    for cls in [TestDateTimeColumnConstruction, TestDateTimeColumnExtractors,
                TestDateTimeColumnOps, TestDateTimeInDataFrame]:
        for name in dir(cls):
            if name.startswith('test_'):
                getattr(cls(), name)()
    print(f'\nResults: {passed} passed, {failed} failed')
    if failed: sys.exit(1)

if __name__ == '__main__':
    run_all()
