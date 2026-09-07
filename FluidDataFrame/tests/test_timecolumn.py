import sys
sys.path.insert(0, '.')
from fluidframe.core.timecolumn import TimeColumn
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


class TestTimeColumnOps2:
    def test_pop_front(self):
        col = TimeColumn()
        col.append('08:00:00'); col.append('10:30:45'); col.append('14:00:00')
        val = col.pop_front()
        check('pop_front len', col.length == 2)

    def test_pop_back(self):
        col = TimeColumn()
        col.append('08:00:00'); col.append('10:30:45')
        val = col.pop_back()
        check('pop_back len', col.length == 1)

    def test_concat(self):
        c1 = TimeColumn(); c1.append('08:00:00')
        c2 = TimeColumn(); c2.append('10:30:45')
        c1.concat(c2)
        check('concat len', c1.length == 2)

    def test_drop_row(self):
        col = TimeColumn()
        col.append('08:00:00'); col.append('10:30:45'); col.append('14:00:00')
        col.drop_row(1)
        check('drop_row len', col.length == 2)

    def test_unique(self):
        col = TimeColumn()
        col.append('08:00:00'); col.append('10:30:45'); col.append('08:00:00')
        u = col.unique()
        check('unique len', len(u) == 2)

    def test_isin(self):
        col = TimeColumn()
        col.append('08:00:00'); col.append('10:30:45')
        r = col.isin({'08:00:00'})
        check('isin', [r[i] for i in range(2)] == [True, False])

    def test_where_mask(self):
        mask = IntColumn(3); mask.append(1); mask.append(0); mask.append(1)
        col = TimeColumn()
        col.append('08:00:00'); col.append('10:30:45'); col.append('14:00:00')
        r = col.where_mask(mask, '00:00:00')
        check('where_mask', r[1] == '00:00:00')

    def test_reorder(self):
        col = TimeColumn()
        col.append('08:00:00'); col.append('10:30:45'); col.append('14:00:00')
        col._reorder([2, 0, 1])
        check('reorder', col[0] == '14:00:00')


class TestTimeColumnOwnership:
    def test_pop_front_raises_on_empty(self):
        col = TimeColumn()
        check('pop_front empty', expect_exception(IndexError, col.pop_front))

    def test_pop_back_raises_on_empty(self):
        col = TimeColumn()
        check('pop_back empty', expect_exception(IndexError, col.pop_back))

    def test_owning_pop_front_works(self):
        col = TimeColumn()
        col.append('08:00:00'); col.append('10:30:45'); col.append('14:00:00')
        col.pop_front()
        check('owning pop_front len', col.length == 2)

    def test_owning_pop_back_works(self):
        col = TimeColumn()
        col.append('08:00:00'); col.append('10:30:45'); col.append('14:00:00')
        col.pop_back()
        check('owning pop_back len', col.length == 2)


def run_all():
    print('=== TimeColumn Tests ===')
    for cls in [TestTimeColumnConstruction, TestTimeColumnExtractors,
                TestTimeColumnOps, TestTimeInDataFrame,
                TestTimeColumnOps2, TestTimeColumnOwnership]:
        for name in dir(cls):
            if name.startswith('test_'):
                getattr(cls(), name)()
    print(f'\nResults: {passed} passed, {failed} failed')
    if failed: sys.exit(1)

if __name__ == '__main__':
    run_all()
