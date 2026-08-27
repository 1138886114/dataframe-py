import sys
sys.path.insert(0, '..')
from fluidframe.core.fastdataframe import FastDataFrame
from fluidframe.core.index import RangeIndex, MultiIndex
from fluidframe.core.intcolumn import IntColumn

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
    except Exception as e:
        print(f'  unexpected: {type(e).__name__}: {e}')
        return False

def test_range_index():
    ri = RangeIndex(0, 5)
    check('RangeIndex length', ri.length == 5)
    check('RangeIndex values', ri.values() == [0, 1, 2, 3, 4])

    ri = RangeIndex(10, 3, 2)
    check('RangeIndex custom', ri.values() == [10, 12, 14])

    ri = RangeIndex(0, 0)
    check('RangeIndex empty', ri.length == 0 and ri.values() == [])

def test_range_index_get_loc():
    ri = RangeIndex(0, 5)
    check('get_loc 0', ri.get_loc(0) == 0)
    check('get_loc 3', ri.get_loc(3) == 3)
    check('get_loc missing', expect_exception(KeyError, ri.get_loc, 99))

def test_range_index_take():
    df = FastDataFrame({'a': int})
    df.append_row(a=10)
    df.append_row(a=20)
    df.append_row(a=30)
    df.append_row(a=40)
    df.append_row(a=50)

    h = df.head(3)
    check('head index values', h.index.values() == [0, 1, 2])

    t = df.tail(2)
    check('tail index values', t.index.values() == [3, 4])

    df2 = FastDataFrame({'val': int})
    df2.append_row(val=30)
    df2.append_row(val=10)
    df2.append_row(val=20)
    sorted_df = df2.sort_values('val')
    check('sort index values', sorted_df.index.values() == [1, 2, 0])

def test_multi_index():
    c1 = IntColumn()
    c1.append(0); c1.append(1); c1.append(0); c1.append(1)
    c2 = IntColumn()
    c2.append(0); c2.append(0); c2.append(1); c2.append(1)

    mi = MultiIndex([['A', 'B'], ['X', 'Y']], [c1, c2], names=['letter', 'number'])
    check('MultiIndex length', mi.length == 4)
    check('MultiIndex nlevels', mi.nlevels == 2)
    check('MultiIndex names', mi.names == ['letter', 'number'])

    # get_loc uses actual values, not codes
    # Row 0: codes (0,0) → ('A', 'X')
    loc = mi.get_loc(('A', 'X'))
    check('MultiIndex get_loc unique', loc == 0)

    # Row 3: codes (1,1) → ('B', 'Y')
    loc = mi.get_loc(('B', 'Y'))
    check('MultiIndex get_loc [B,Y]', loc == 3)

    c1b = IntColumn()
    c1b.append(0); c1b.append(0); c1b.append(0)
    c2b = IntColumn()
    c2b.append(0); c2b.append(1); c2b.append(0)
    mi2 = MultiIndex([['A'], ['X', 'Y']], [c1b, c2b])
    loc = mi2.get_loc(('A', 'X'))
    check('MultiIndex get_loc multi', loc == [0, 2])

    check('MultiIndex get_loc missing', expect_exception(KeyError, mi.get_loc, ('Z', 'W')))
    # Partial key lookup is supported (matches all rows with first level = 'A')
    loc = mi.get_loc(('A',))
    check('MultiIndex get_loc partial', loc == [0, 2])

def test_dataframe_index():
    df = FastDataFrame({'a': int, 'b': str})
    df.append_row(a=1, b='x')
    df.append_row(a=2, b='y')
    df.append_row(a=3, b='z')
    check('DataFrame has index', df.index is not None)
    check('DataFrame index type', isinstance(df.index, RangeIndex))
    check('DataFrame index length', df.index.length == 3)

    h = df.head(2)
    check('head index length', h.index.length == 2)
    check('head index values', h.index.values() == [0, 1])

    t = df.tail(2)
    check('tail index length', t.index.length == 2)
    check('tail index values', t.index.values() == [1, 2])

def test_iloc():
    df = FastDataFrame({'a': int, 'b': str})
    for i in range(5):
        df.append_row(a=i, b=f'row{i}')

    r = df.iloc[0]
    check('iloc single row', r.shape == (1, 2))
    check('iloc single a', r['a'][0] == 0)

    r = df.iloc[3]
    check('iloc row 3', r['a'][0] == 3)

    r = df.iloc[-1]
    check('iloc [-1]', r['a'][0] == 4)

    r = df.iloc[1:4]
    check('iloc slice shape', r.shape == (3, 2))
    check('iloc slice [0]', r['a'][0] == 1)
    check('iloc slice [2]', r['a'][2] == 3)

    r = df.iloc[[0, 2, 4]]
    check('iloc list shape', r.shape == (3, 2))
    check('iloc list [0]', r['a'][0] == 0)
    check('iloc list [1]', r['a'][1] == 2)
    check('iloc list [2]', r['a'][2] == 4)

    r = df.iloc[5:10]
    check('iloc empty', r.shape == (0, 2))

    check('iloc oob', expect_exception(IndexError, df.iloc.__getitem__, 10))

def test_loc():
    df = FastDataFrame({'a': int})
    df.append_row(a=10)
    df.append_row(a=20)
    df.append_row(a=30)

    r = df.loc[0]
    check('loc single row', r.shape == (1, 1))
    check('loc value', r['a'][0] == 10)

    r = df.loc[2]
    check('loc row 2', r['a'][0] == 30)

    check('loc missing', expect_exception(KeyError, df.loc.__getitem__, 99))

def test_head_tail_index():
    df = FastDataFrame({'x': int})
    for i in range(10):
        df.append_row(x=i * 10)

    h = df.head(3)
    check('head 3 index', h.index.values() == [0, 1, 2])
    check('head 3 data', h['x'][0] == 0 and h['x'][2] == 20)

    t = df.tail(3)
    check('tail 3 index', t.index.values() == [7, 8, 9])
    check('tail 3 data', t['x'][0] == 70 and t['x'][2] == 90)

    h0 = df.head(0)
    check('head 0 index', h0.index.length == 0)

def test_dropna_index():
    df = FastDataFrame({'a': float})
    df.append_row(a=1.0)
    df.append_row(a=float('nan'))
    df.append_row(a=3.0)
    cleaned = df.dropna()
    check('dropna index', cleaned.index.values() == [0, 2])

def test_drop_with_index_index():
    df = FastDataFrame({'a': int})
    for i in range(5):
        df.append_row(a=i)
    df2 = df.drop_with_index([1, 3])
    check('drop_with_index index', df2.index.values() == [0, 2, 4])

def test_iloc_index():
    df = FastDataFrame({'a': int})
    for i in range(5):
        df.append_row(a=i * 10)

    r = df.iloc[1:4]
    check('iloc index', r.index.values() == [1, 2, 3])

    r = df.iloc[[0, 4]]
    check('iloc list index', r.index.values() == [0, 4])

    r = df.iloc[-2:]
    check('iloc neg slice index', r.index.values() == [3, 4])

def run_all():
    print('=== Index Tests ===')
    test_range_index()
    test_range_index_get_loc()
    test_range_index_take()
    test_multi_index()
    test_dataframe_index()
    test_iloc()
    test_loc()
    test_head_tail_index()
    test_dropna_index()
    test_drop_with_index_index()
    test_iloc_index()
    print(f'\nResults: {passed} passed, {failed} failed')
    if failed: sys.exit(1)

if __name__ == '__main__':
    run_all()
