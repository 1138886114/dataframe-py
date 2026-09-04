import sys
sys.path.insert(0, '.')
from fluidframe.core.fastdataframe import FastDataFrame
from fluidframe.core.index import RangeIndex, MultiIndex
from fluidframe.core.intcolumn import IntColumn

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


class TestRangeIndex:
    def test_creation(self):
        ri = RangeIndex(0, 5)
        check('length', ri.length == 5)
        check('values', ri.values() == [0, 1, 2, 3, 4])

    def test_custom_step(self):
        ri = RangeIndex(10, 3, 2)
        check('custom', ri.values() == [10, 12, 14])

    def test_empty(self):
        ri = RangeIndex(0, 0)
        check('empty', ri.length == 0 and ri.values() == [])

    def test_get_loc(self):
        ri = RangeIndex(0, 5)
        check('get_loc 0', ri.get_loc(0) == 0)
        check('get_loc 3', ri.get_loc(3) == 3)
        check('get_loc missing', expect_exception(KeyError, ri.get_loc, 99))

    def test_take(self):
        df = FastDataFrame({'a': int})
        for v in [30, 10, 50, 20, 40]: df.append_row(a=v)
        check('head idx', df.head(3).index.values() == [0, 1, 2])
        check('tail idx', df.tail(2).index.values() == [3, 4])
        s = df.sort_values('a')
        check('sort idx', s.index.values() == [1, 3, 0, 4, 2])


class TestMultiIndex:
    def _make(self):
        c1 = IntColumn(); c1.append(0); c1.append(1); c1.append(0); c1.append(1)
        c2 = IntColumn(); c2.append(0); c2.append(0); c2.append(1); c2.append(1)
        return MultiIndex([['A', 'B'], ['X', 'Y']], [c1, c2], names=['letter', 'number'])

    def test_creation(self):
        mi = self._make()
        check('length', mi.length == 4)
        check('nlevels', mi.nlevels == 2)
        check('names', mi.names == ['letter', 'number'])

    def test_get_loc_unique(self):
        mi = self._make()
        check('A,X', mi.get_loc(('A', 'X')) == 0)
        check('B,Y', mi.get_loc(('B', 'Y')) == 3)

    def test_get_loc_multi(self):
        c1 = IntColumn(); c1.append(0); c1.append(0); c1.append(0)
        c2 = IntColumn(); c2.append(0); c2.append(1); c2.append(0)
        mi = MultiIndex([['A'], ['X', 'Y']], [c1, c2])
        loc = mi.get_loc(('A', 'X'))
        check('multi', loc == [0, 2])

    def test_get_loc_missing(self):
        mi = self._make()
        check('missing', expect_exception(KeyError, mi.get_loc, ('Z', 'W')))

    def test_get_loc_partial(self):
        mi = self._make()
        loc = mi.get_loc(('A',))
        check('partial', loc == [0, 2])


class TestDataFrameIndex:
    def test_auto_range_index(self):
        df = FastDataFrame({'a': int, 'b': str})
        df.append_row(a=1, b='x'); df.append_row(a=2, b='y')
        check('has index', df.index is not None)
        check('is RangeIndex', isinstance(df.index, RangeIndex))
        check('length', df.index.length == 2)

    def test_head_tail_index(self):
        df = FastDataFrame({'x': int})
        for i in range(10): df.append_row(x=i * 10)
        check('head idx', df.head(3).index.values() == [0, 1, 2])
        check('tail idx', df.tail(3).index.values() == [7, 8, 9])
        check('head 0', df.head(0).index.length == 0)

    def test_dropna_index(self):
        df = FastDataFrame({'a': float})
        df.append_row(a=1.0); df.append_row(a=float('nan')); df.append_row(a=3.0)
        check('dropna idx', df.dropna().index.values() == [0, 2])

    def test_drop_with_index(self):
        df = FastDataFrame({'a': int})
        for i in range(5): df.append_row(a=i)
        check('drop idx', df.drop_with_index([1, 3]).index.values() == [0, 2, 4])

    def test_iloc_index(self):
        df = FastDataFrame({'a': int})
        for i in range(5): df.append_row(a=i * 10)
        check('iloc slice idx', df.iloc[1:4].index.values() == [1, 2, 3])
        check('iloc list idx', df.iloc[[0, 4]].index.values() == [0, 4])
        check('iloc neg idx', df.iloc[-2:].index.values() == [3, 4])


def run_all():
    print('=== Index Tests ===')
    for cls in [TestRangeIndex, TestMultiIndex, TestDataFrameIndex]:
        for name in dir(cls):
            if name.startswith('test_'):
                getattr(cls(), name)()
    print(f'\nResults: {passed} passed, {failed} failed')
    if failed: sys.exit(1)

if __name__ == '__main__':
    run_all()
