import math
import os
import sys
import tempfile
sys.path.insert(0, '..')
from fluidframe.core.fastdataframe import FastDataFrame
from fluidframe.core.intcolumn import IntColumn
from fluidframe.core.doublecolumn import DoubleColumn
from fluidframe.core.char256column import Char256Column_SoA
from fluidframe.core.ucs2column import UCS2Column_SoA
from fluidframe.core.timestampcolumn import TimestampColumn
from fluidframe._schema_types import Char256, Timestamp

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

def test_init():
    df = FastDataFrame()
    check('empty shape', df.shape == (0, 0))

    df = FastDataFrame({'a': int})
    check('schema int', isinstance(df.columns['a'], IntColumn))

    df = FastDataFrame({'a': float})
    check('schema float', isinstance(df.columns['a'], DoubleColumn))

    df = FastDataFrame({'a': str})
    check('schema str', isinstance(df.columns['a'], UCS2Column_SoA))

    df = FastDataFrame({'a': Char256})
    check('schema char256', isinstance(df.columns['a'], Char256Column_SoA))

    df = FastDataFrame({'a': Timestamp})
    check('schema timestamp', isinstance(df.columns['a'], TimestampColumn))

    df = FastDataFrame({'id': int, 'name': str, 'score': float})
    check('schema multiple', df.shape == (0, 3))

    check('schema unsupported', expect_exception(TypeError, FastDataFrame, {'a': list}))

def test_append():
    df = FastDataFrame({'id': int, 'name': str})
    df.append_row(id=1, name='Alice')
    df.append_row(id=2, name='Bob')
    check('append shape', df.shape == (2, 2))
    check('append id[0]', df['id'][0] == 1)
    check('append name[1]', df['name'][1] == 'Bob')

    df = FastDataFrame({'i': int, 'f': float, 's': str, 'c': Char256, 't': Timestamp})
    df.append_row(i=1, f=2.5, s='hello', c='world', t=20160101)
    check('append types i', df['i'][0] == 1)
    check('append types f', df['f'][0] == 2.5)
    check('append types s', df['s'][0] == 'hello')
    check('append types c', df['c'][0] == 'world')
    check('append types t', df['t'][0] == 20160101)

    df = FastDataFrame({'val': int})
    for i in range(1000):
        df.append_row(val=i)
    check('append many', df.shape == (1000, 1))
    check('append many [0]', df['val'][0] == 0)
    check('append many [999]', df['val'][999] == 999)

def test_access():
    df = FastDataFrame({'a': int}); df.append_row(a=42)
    check('getitem', isinstance(df['a'], IntColumn))
    check('getitem val', df['a'][0] == 42)
    check('getitem missing', expect_exception(KeyError, df.__getitem__, 'b'))

    df = FastDataFrame({'a': int}); df.append_row(a=42)
    check('getattr', df.a[0] == 42)
    check('getattr missing', expect_exception(AttributeError, getattr, df, 'nonexistent'))

    df = FastDataFrame({'a': int})
    check('repr', 'FastDataFrame' in repr(df) and '0 rows' in repr(df))

def test_copy():
    df = FastDataFrame({'a': int}); df.append_row(a=1)
    df2 = df.copy()
    df2.append_row(a=2)
    check('copy independent orig', df.shape == (1, 1))
    check('copy independent new', df2.shape == (2, 1))

def test_pop():
    df = FastDataFrame({'a': int, 'b': str})
    df.append_row(a=1, b='x'); df.append_row(a=2, b='y')
    row = df.pop_front()
    check('pop_front a', row['a'] == 1)
    check('pop_front b', row['b'] == 'x')
    check('pop_front shape', df.shape == (1, 2))

    df = FastDataFrame({'a': int, 'b': str})
    df.append_row(a=1, b='x'); df.append_row(a=2, b='y')
    row = df.pop_back()
    check('pop_back a', row['a'] == 2)
    check('pop_back b', row['b'] == 'y')
    check('pop_back shape', df.shape == (1, 2))

    df = FastDataFrame({'a': int})
    check('pop_front empty', expect_exception(IndexError, df.pop_front))
    check('pop_back empty', expect_exception(IndexError, df.pop_back))

def test_drop():
    df = FastDataFrame({'a': int, 'b': int, 'c': int})
    df.append_row(a=1, b=2, c=3)
    df.drop('b')
    check('drop shape', df.shape == (1, 2))
    check('drop not in cols', 'b' not in df.columns)

    df = FastDataFrame({'a': int, 'b': int})
    df.append_row(a=1, b=2)
    df.drop_col('a')
    check('drop_col shape', df.shape == (1, 1))

def test_sort():
    df = FastDataFrame({'val': int})
    for v in [5, 1, 3, 2, 4]: df.append_row(val=v)
    s = df.sort_values('val')
    check('sort asc [0]', s['val'][0] == 1)
    check('sort asc [4]', s['val'][4] == 5)

    df = FastDataFrame({'val': int})
    for v in [5, 1, 3, 2, 4]: df.append_row(val=v)
    s = df.sort_values('val', reverse=True)
    check('sort desc [0]', s['val'][0] == 5)
    check('sort desc [4]', s['val'][4] == 1)

    df = FastDataFrame({'val': int})
    for v in [5, 1, 3]: df.append_row(val=v)
    df.sort_values('val', inplace=True)
    check('sort inplace', df['val'][0] == 1)

    df = FastDataFrame({'val': int})
    s = df.sort_values('val')
    check('sort empty', s.shape == (0, 1))

    df = FastDataFrame({'val': float})
    df.append_row(val=3.14); df.append_row(val=1.41); df.append_row(val=2.72)
    s = df.sort_values('val')
    check('sort float [0]', s['val'][0] == 1.41)
    check('sort float [2]', s['val'][2] == 3.14)

    df = FastDataFrame({'val': str})
    df.append_row(val='cherry'); df.append_row(val='apple'); df.append_row(val='banana')
    s = df.sort_values('val')
    check('sort str [0]', s['val'][0] == 'apple')
    check('sort str [2]', s['val'][2] == 'cherry')

    df = FastDataFrame({'val': Char256})
    df.append_row(val='cherry'); df.append_row(val='apple'); df.append_row(val='banana')
    s = df.sort_values('val')
    check('sort char256 [0]', s['val'][0] == 'apple')
    check('sort char256 [2]', s['val'][2] == 'cherry')

    df = FastDataFrame({'val': Timestamp})
    df.append_row(val=20231225); df.append_row(val=20160101); df.append_row(val=20200615)
    s = df.sort_values('val')
    check('sort timestamp [0]', s['val'][0] == 20160101)
    check('sort timestamp [2]', s['val'][2] == 20231225)

    df = FastDataFrame({'val': int}); df.append_row(val=42)
    s = df.sort_values('val')
    check('sort single', s['val'][0] == 42)

    df = FastDataFrame({'val': int})
    for v in [3, 1, 3, 2, 1]: df.append_row(val=v)
    s = df.sort_values('val')
    vals = [s['val'][i] for i in range(5)]
    check('sort dups', vals == [1, 1, 2, 3, 3])

    df = FastDataFrame({'a': int})
    check('sort not found', expect_exception(KeyError, df.sort_values, 'nonexistent'))

def test_head_tail():
    df = FastDataFrame({'a': int})
    for i in range(10): df.append_row(a=i)
    check('head 3', df.head(3).shape == (3, 1))
    check('head large', df.head(100).shape == (10, 1))
    check('head 0', df.head(0).shape == (0, 1))
    check('head neg', expect_exception(ValueError, df.head, -1))

    df = FastDataFrame({'a': int})
    for i in range(10): df.append_row(a=i)
    t = df.tail(3)
    check('tail 3', t.shape == (3, 1))
    check('tail [0]', t['a'][0] == 7)
    check('tail [2]', t['a'][2] == 9)
    check('tail large', df.tail(100).shape == (10, 1))
    check('tail neg', expect_exception(ValueError, df.tail, -1))

def test_shape_names_dtypes():
    df = FastDataFrame({'a': int, 'b': str}); df.append_row(a=1, b='x')
    check('shape', df.shape == (1, 2))
    check('col_names', set(df.col_names) == {'a', 'b'})

    df = FastDataFrame({'i': int, 'f': float, 's': str, 'c': Char256, 't': Timestamp})
    dtypes = df.dtypes
    check('dtypes int', dtypes['i'] is int)
    check('dtypes float', dtypes['f'] is float)
    check('dtypes str', dtypes['s'] is str)
    check('dtypes char256', dtypes['c'] is Char256)
    check('dtypes timestamp', dtypes['t'] is Timestamp)

def test_rename():
    df = FastDataFrame({'a': int, 'b': str}); df.append_row(a=1, b='x')
    df2 = df.rename({'a': 'alpha'})
    check('rename has alpha', 'alpha' in df2.columns)
    check('rename no a', 'a' not in df2.columns)
    check('rename val', df2['alpha'][0] == 1)
    df2 = df.rename({'nonexistent': 'new'})
    check('rename nonexistent', 'a' in df2.columns)

def test_drop_with_index():
    df = FastDataFrame({'a': int})
    for i in range(5): df.append_row(a=i)
    df2 = df.drop_with_index([1, 3])
    check('drop_with_index', df2.shape == (3, 1))
    vals = [df2['a'][i] for i in range(3)]
    check('drop_with_index vals', vals == [0, 2, 4])

    df = FastDataFrame({'a': int})
    for i in range(5): df.append_row(a=i)
    df.drop_with_index([0, 4], inplace=True)
    check('drop_with_index inplace', df.shape == (3, 1))

    df = FastDataFrame({'a': int})
    check('drop_with_index empty', df.drop_with_index([]).shape == (0, 1))

    df = FastDataFrame({'a': int}); df.append_row(a=1)
    check('drop_with_index all', df.drop_with_index([0]).shape == (0, 0))

    df = FastDataFrame({'a': int}); df.append_row(a=1)
    check('drop_with_index oob', expect_exception(IndexError, df.drop_with_index, [5]))

def test_drop_raw():
    df = FastDataFrame({'a': int, 'b': str})
    df.append_row(a=1, b='x'); df.append_row(a=2, b='y'); df.append_row(a=3, b='z')
    df.drop_raw(1)
    check('drop_raw shape', df.shape == (2, 2))
    check('drop_raw [0]', df['a'][0] == 1)
    check('drop_raw [1]', df['a'][1] == 3)

    df = FastDataFrame({'a': int}); df.append_row(a=1); df.append_row(a=2)
    df.drop_raw(0)
    check('drop_raw first', df['a'][0] == 2)

    df = FastDataFrame({'a': int}); df.append_row(a=1); df.append_row(a=2)
    df.drop_raw(1)
    check('drop_raw last', df['a'][0] == 1)

    df = FastDataFrame({'a': int}); df.append_row(a=1)
    check('drop_raw oob', expect_exception(IndexError, df.drop_raw, 5))

def test_diff():
    df = FastDataFrame({'a': int})
    for v in [1, 3, 6, 10]: df.append_row(a=v)
    res = df.diff('a')
    check('diff int type', isinstance(res, IntColumn))
    check('diff int [0]', res[0] == 0)
    check('diff int [1]', res[1] == 2)
    check('diff int [2]', res[2] == 3)
    check('diff int [3]', res[3] == 4)

    df = FastDataFrame({'a': float})
    df.append_row(a=1.0); df.append_row(a=3.0); df.append_row(a=6.0)
    res = df.diff('a')
    check('diff float type', isinstance(res, DoubleColumn))
    check('diff float [0] nan', math.isnan(res[0]))
    check('diff float [1]', res[1] == 2.0)
    check('diff float [2]', res[2] == 3.0)

    df = FastDataFrame({'a': int})
    for v in [1, 2, 4, 7, 11]: df.append_row(a=v)
    res = df.diff('a', periods=2)
    check('diff p2 [0]', res[0] == 0)
    check('diff p2 [1]', res[1] == 0)
    check('diff p2 [2]', res[2] == 3)
    check('diff p2 [3]', res[3] == 5)
    check('diff p2 [4]', res[4] == 7)

    df = FastDataFrame({'a': int}); df.append_row(a=1)
    check('diff neg', expect_exception(ValueError, df.diff, 'a', -1))

    df = FastDataFrame({'a': str}); df.append_row(a='hello')
    check('diff str', expect_exception(TypeError, df.diff, 'a'))

    df = FastDataFrame({'a': int})
    check('diff missing', expect_exception(KeyError, df.diff, 'nonexistent'))

def test_reset_index():
    df = FastDataFrame({'a': int}); df.append_row(a=1); df.append_row(a=2)
    df.reset_index()
    check('reset_index', 'index' in df.columns)
    check('reset_index [0]', df['index'][0] == 0)
    check('reset_index [1]', df['index'][1] == 1)

    df = FastDataFrame({'a': int})
    df.append_row(a=5); df.append_row(a=3); df.append_row(a=1)
    df.reset_index()
    df2 = df.sort_index()
    check('sort_index [0]', df2['a'][0] == 5)
    check('sort_index [2]', df2['a'][2] == 1)

    df = FastDataFrame({'a': int})
    check('sort_index no idx', expect_exception(KeyError, df.sort_index))

def test_unique():
    df = FastDataFrame({'a': int})
    for v in [1, 2, 1, 3, 2]: df.append_row(a=v)
    check('unique int', df.unique('a') == [1, 2, 3])

    df = FastDataFrame({'a': str})
    df.append_row(a='x'); df.append_row(a='y'); df.append_row(a='x')
    check('unique str', df.unique('a') == ['x', 'y'])

def test_argmax_argmin():
    df = FastDataFrame({'a': int})
    for v in [1, 5, 3]: df.append_row(a=v)
    check('argmax', df.argmax('a') == 1)

    df = FastDataFrame({'a': int})
    for v in [5, 1, 3]: df.append_row(a=v)
    check('argmin', df.argmin('a') == 1)

def test_null_handling():
    df = FastDataFrame({'a': float})
    df.append_row(a=1.0); df.append_row(a=float('nan')); df.append_row(a=3.0)
    null_df = df.isnull()
    check('isnull [0]', null_df['a'][0] == 0)
    check('isnull [1]', null_df['a'][1] == 1)
    check('isnull [2]', null_df['a'][2] == 0)

    df = FastDataFrame({'a': float})
    df.append_row(a=1.0); df.append_row(a=float('nan')); df.append_row(a=3.0)
    filled = df.fillna(0.0)
    check('fillna [0]', filled['a'][0] == 1.0)
    check('fillna [1]', filled['a'][1] == 0.0)
    check('fillna [2]', filled['a'][2] == 3.0)

    df = FastDataFrame({'a': float})
    df.append_row(a=1.0); df.append_row(a=float('nan')); df.append_row(a=3.0)
    cleaned = df.dropna()
    check('dropna shape', cleaned.shape == (2, 1))
    check('dropna [0]', cleaned['a'][0] == 1.0)
    check('dropna [1]', cleaned['a'][1] == 3.0)

    df = FastDataFrame({'a': float}); df.append_row(a=1.0); df.append_row(a=2.0)
    check('dropna valid', df.dropna().shape == (2, 1))

    df = FastDataFrame({'a': float})
    df.append_row(a=float('nan')); df.append_row(a=float('nan'))
    check('dropna all nan', df.dropna().shape == (0, 0))

def test_describe():
    df = FastDataFrame({'a': int, 'b': float, 'c': str})
    df.append_row(a=1, b=1.5, c='x')
    df.append_row(a=2, b=2.5, c='y')
    df.append_row(a=3, b=3.5, c='z')
    stats = df.describe()
    check('describe count', stats['a']['count'] == 3)
    check('describe mean', stats['a']['mean'] == 2.0)
    check('describe str count', stats['c']['count'] == 3)

def test_csv():
    df = FastDataFrame({'id': int, 'name': str})
    df.append_row(id=1, name='Alice'); df.append_row(id=2, name='Bob')
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        fname = f.name
    try:
        df.to_csv(fname)
        from fluidframe.core.fluidframe import read_csv
        df2 = read_csv(fname, schema={'id': int, 'name': str})
        check('csv shape', df2.shape == (2, 2))
        check('csv id[0]', df2['id'][0] == 1)
        check('csv name[1]', df2['name'][1] == 'Bob')
    finally:
        os.unlink(fname)

    df = FastDataFrame({'a': int}); df.append_row(a=1)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False, newline='') as f:
        fname = f.name
    try:
        df.to_tsv(fname)
        check('tsv exists', os.path.exists(fname))
    finally:
        os.unlink(fname)

def test_pickle():
    df = FastDataFrame({'id': int, 'name': str, 'score': float})
    df.append_row(id=1, name='Alice', score=95.5)
    df.append_row(id=2, name='Bob', score=87.0)
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        fname = f.name
    try:
        df.to_pickle(fname)
        from fluidframe.core.fluidframe import from_pickle
        df2 = from_pickle(fname)
        check('pkl shape', df2.shape == (2, 3))
        check('pkl id[0]', df2['id'][0] == 1)
        check('pkl name[1]', df2['name'][1] == 'Bob')
        check('pkl score[0]', df2['score'][0] == 95.5)
    finally:
        os.unlink(fname)

def test_concat():
    from fluidframe.core.fluidframe import concat_outer, concat_inner

    check('concat empty', concat_outer([]).shape == (0, 0))

    df = FastDataFrame({'a': int}); df.append_row(a=1)
    check('concat single', concat_outer([df]).df_length == 1)

    df1 = FastDataFrame({'a': int}); df1.append_row(a=1)
    df2 = FastDataFrame({'a': int}); df2.append_row(a=2)
    check('concat two', concat_outer([df1, df2]).df_length == 2)

    df1 = FastDataFrame({'a': int, 'b': int}); df1.append_row(a=1, b=10)
    df2 = FastDataFrame({'a': int, 'c': int}); df2.append_row(a=2, c=20)
    r = concat_inner([df1, df2])
    check('concat inner has a', 'a' in r.columns)
    check('concat inner no b', 'b' not in r.columns)
    check('concat inner no c', 'c' not in r.columns)

def test_merge():
    from fluidframe.core.fluidframe import merge_inner, merge_left, merge_right, merge_outer

    df1 = FastDataFrame({'id': int, 'val1': float})
    df1.append_row(id=1, val1=10.0); df1.append_row(id=2, val1=20.0)
    df2 = FastDataFrame({'id': int, 'val2': float})
    df2.append_row(id=2, val2=200.0); df2.append_row(id=3, val2=300.0)
    check('merge inner', merge_inner([df1, df2], 'id', 'id').shape == (1, 3))
    check('merge left', merge_left([df1, df2], 'id', 'id').shape == (2, 3))
    check('merge right', merge_right([df1, df2], 'id', 'id').shape == (2, 3))
    check('merge outer', merge_outer([df1, df2], 'id', 'id').shape == (3, 3))

    df1 = FastDataFrame({'id': int}); df1.append_row(id=1)
    df2 = FastDataFrame({'id': int}); df2.append_row(id=2)
    check('merge no match', merge_inner([df1, df2], 'id', 'id').shape == (0, 1))

    df1 = FastDataFrame({'id': int, 'val': int}); df1.append_row(id=1, val=10)
    df2 = FastDataFrame({'id': int, 'val': int}); df2.append_row(id=1, val=20)
    m = merge_inner([df1, df2], 'id', 'id')
    check('merge same key has val', 'val' in m.columns)
    check('merge same key has val_right', 'val_right' in m.columns)

def test_read_csv():
    from fluidframe.core.fluidframe import read_csv, read_tsv

    content = "id,name,score\n1,Alice,95.5\n2,Bob,87.0\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(content); fname = f.name
    try:
        df = read_csv(fname)
        check('read_csv auto shape', df.shape == (2, 3))
        check('read_csv auto id[0]', df['id'][0] == 1)
        check('read_csv auto name[1]', df['name'][1] == 'Bob')
    finally:
        os.unlink(fname)

    content = "id,name,score\n1,Alice,95.5\n2,Bob,87.0\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(content); fname = f.name
    try:
        df = read_csv(fname, schema={'id': int, 'name': str, 'score': float})
        check('read_csv schema', df.shape == (2, 3))
    finally:
        os.unlink(fname)

    content = "a,b,c\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(content); fname = f.name
    try:
        df = read_csv(fname)
        check('read_csv empty', df.shape == (0, 3))
    finally:
        os.unlink(fname)

    content = "id\tname\n1\tAlice\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
        f.write(content); fname = f.name
    try:
        df = read_tsv(fname, schema={'id': int, 'name': str})
        check('read_tsv', df.shape == (1, 2))
    finally:
        os.unlink(fname)

def test_loc_filter():
    df = FastDataFrame({'a': int, 'b': str})
    for i in range(5):
        df.append_row(a=i, b=f'row{i}')
    mask = df['a'] < 3
    filtered = df.loc[mask]
    check('loc mask shape', filtered.shape == (3, 2))
    check('loc mask vals', [filtered['a'][i] for i in range(3)] == [0, 1, 2])

    filtered2 = df.loc[[True, False, True, False, True]]
    check('loc bool list shape', filtered2.shape == (3, 2))
    check('loc bool list vals', [filtered2['a'][i] for i in range(3)] == [0, 2, 4])

    filtered3 = df.loc[lambda d: d['a'] > 1]
    check('loc callable shape', filtered3.shape == (3, 2))
    check('loc callable vals', [filtered3['a'][i] for i in range(3)] == [2, 3, 4])

    mask_and = (df['a'] > 1) & (df['a'] < 4)
    filtered4 = df.loc[mask_and]
    check('loc combined shape', filtered4.shape == (2, 2))
    check('loc combined vals', [filtered4['a'][i] for i in range(2)] == [2, 3])

def run_all():
    print('=== FastDataFrame Tests ===')
    test_init()
    test_append()
    test_access()
    test_copy()
    test_pop()
    test_drop()
    test_sort()
    test_head_tail()
    test_shape_names_dtypes()
    test_rename()
    test_drop_with_index()
    test_drop_raw()
    test_diff()
    test_reset_index()
    test_unique()
    test_argmax_argmin()
    test_null_handling()
    test_describe()
    test_csv()
    test_pickle()
    test_concat()
    test_merge()
    test_read_csv()
    test_loc_filter()
    print(f'\nResults: {passed} passed, {failed} failed')
    if failed: sys.exit(1)

if __name__ == '__main__':
    run_all()
