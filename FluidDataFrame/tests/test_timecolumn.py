import sys
sys.path.insert(0, '.')
from fluidframe.core.timecolumn import TimeColumn
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


class TestTimeColumnConstruction:
    def test_empty(self):
        col = TimeColumn()
        check('empty', col.length == 0)

    def test_append_str(self):
        col = TimeColumn()
        col.append('10:30:45'); col.append('14:00')
        check('length', col.length == 2)
        check('val 0', col[0] == '10:30:45')
        check('val 1', col[1] == '14:00:00')

    def test_append_int(self):
        col = TimeColumn()
        col.append(37845)  # seconds since midnight
        check('int append', col.length == 1)

    def test_invalid(self):
        col = TimeColumn()
        check('invalid', expect_exception(ValueError, col.append, '25:00:00'))


class TestTimeColumnExtractors:
    def _make(self):
        col = TimeColumn()
        col.append('10:30:45'); col.append('14:00:00')
        return col

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


class TestTimeColumnOps:
    def test_copy(self):
        col = TimeColumn(); col.append('10:30:45')
        col2 = col.copy()
        check('copy len', col2.length == 1)
        col2.append('14:00:00')
        check('independent', col.length == 1)

    def test_min_max(self):
        col = TimeColumn()
        col.append('10:30:45'); col.append('14:00:00')
        min_val = col.min()
        max_val = col.max()
        check('min < max', min_val < max_val)
        check('min is int', isinstance(min_val, int))


class TestTimeInDataFrame:
    def test_schema(self):
        df = FastDataFrame({'t': 'time'})
        df.append_row(t='10:30:45')
        check('schema', df.shape == (1, 1))


def run_all():
    print('=== TimeColumn Tests ===')
    for cls in [TestTimeColumnConstruction, TestTimeColumnExtractors,
                TestTimeColumnOps, TestTimeInDataFrame]:
        for name in dir(cls):
            if name.startswith('test_'):
                getattr(cls(), name)()
    print(f'\nResults: {passed} passed, {failed} failed')
    if failed: sys.exit(1)

if __name__ == '__main__':
    run_all()
