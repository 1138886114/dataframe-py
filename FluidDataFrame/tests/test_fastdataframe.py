import math
import os
import random
import sys
import tempfile
import time
sys.path.insert(0, '..')
from fluidframe.core.fastdataframe import FastDataFrame
from fluidframe.core.intcolumn import IntColumn
from fluidframe.core.doublecolumn import DoubleColumn
from fluidframe.core.char256column import Char256Column_SoA
from fluidframe.core.ucs2column import UCS2Column_SoA
from fluidframe.core.datecolumn import DateColumn as DateColumn
from fluidframe._schema_types import Char256, Date

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


# =====================================================================
#  Helper: build complex DataFrame with all 5 column types
# =====================================================================
def make_complex_df():
    """
    Returns a FastDataFrame with 10 rows and 5 column types:
      id:    int
      score: double
      name:  UCS2 (str)
      tag:   Char256
      dt:    Date
    """
    df = FastDataFrame({'id': int, 'score': float, 'name': str, 'tag': Char256, 'dt': Date})
    rows = [
        (1, 88.5,  'Alice',   'senior',   20230115),
        (2, 92.0,  'Bob',     'mid',      20230620),
        (3, 75.3,  'Charlie', 'senior',   20221201),
        (4, 60.0,  'Diana',   'low',      20240229),
        (5, 95.1,  'Eve',     'senior',   20230115),
        (6, 82.7,  'Frank',   'mid',      20230310),
        (7, 70.0,  'Grace',   'low',      20220615),
        (8, 88.5,  'Heidi',   'senior',   20231225),
        (9, 99.9,  'Ivan',    'mid',      20240101),
        (10, 55.0, 'Judy',    'low',      20230620),
    ]
    for r in rows:
        df.append_row(id=r[0], score=r[1], name=r[2], tag=r[3], dt=r[4])
    return df


# =====================================================================
#  Basic construction & properties
# =====================================================================
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

    df = FastDataFrame({'a': Date})
    check('schema timestamp', isinstance(df.columns['a'], DateColumn))

    df = FastDataFrame({'id': int, 'name': str, 'score': float})
    check('schema multiple', df.shape == (0, 3))

    check('schema unsupported', expect_exception(TypeError, FastDataFrame, {'a': list}))

def test_init_complex():
    df = make_complex_df()
    check('complex shape', df.shape == (10, 5))
    check('complex cols', set(df.col_names) == {'id', 'score', 'name', 'tag', 'dt'})
    check('complex dtypes int', df.dtypes['id'] is int)
    check('complex dtypes float', df.dtypes['score'] is float)
    check('complex dtypes str', df.dtypes['name'] is str)
    check('complex dtypes char256', df.dtypes['tag'] is Char256)
    check('complex dtypes timestamp', df.dtypes['dt'] is Date)


# =====================================================================
#  Append & access
# =====================================================================
def test_append():
    df = FastDataFrame({'id': int, 'name': str})
    df.append_row(id=1, name='Alice')
    df.append_row(id=2, name='Bob')
    check('append shape', df.shape == (2, 2))
    check('append id[0]', df['id'][0] == 1)
    check('append name[1]', df['name'][1] == 'Bob')

    df = FastDataFrame({'i': int, 'f': float, 's': str, 'c': Char256, 't': Date})
    df.append_row(i=1, f=2.5, s='hello', c='world', t=20160101)
    check('append types i', df['i'][0] == 1)
    check('append types f', df['f'][0] == 2.5)
    check('append types s', df['s'][0] == 'hello')
    check('append types c', df['c'][0] == 'world')
    check('append types t', df['t'][0] == 20160101)

def test_append_complex():
    df = make_complex_df()
    check('complex append val id', df['id'][0] == 1)
    check('complex append val score', df['score'][4] == 95.1)
    check('complex append val name', df['name'][2] == 'Charlie')
    check('complex append val tag', df['tag'][6] == 'low')
    check('complex append val dt', df['dt'][9] == 20230620)

    df.append_row(id=11, score=42.0, name='Karl', tag='lead', dt=20250101)
    check('complex append new row', df.shape == (11, 5))
    check('complex append new id', df['id'][10] == 11)
    check('complex append new name', df['name'][10] == 'Karl')

def test_access():
    df = FastDataFrame({'a': int}); df.append_row(a=42)
    check('getitem', isinstance(df['a'], IntColumn))
    check('getitem val', df['a'][0] == 42)
    check('getitem missing', expect_exception(KeyError, df.__getitem__, 'b'))

    df = FastDataFrame({'a': int}); df.append_row(a=42)
    check('getattr', df.a[0] == 42)
    check('getattr missing', expect_exception(AttributeError, getattr, df, 'nonexistent'))

    df = FastDataFrame({'a': int})
    check('repr', 'shape: (0, 1)' in repr(df))


# =====================================================================
#  Copy & pop
# =====================================================================
def test_copy():
    df = make_complex_df()
    df2 = df.copy()
    check('copy shape', df2.shape == (10, 5))
    check('copy val', df2['id'][3] == 4)
    df2.append_row(id=99, score=0.0, name='Z', tag='x', dt=20000101)
    check('copy independent orig', df.shape == (10, 5))
    check('copy independent new', df2.shape == (11, 5))

def test_pop_front():
    df = make_complex_df()
    row = df.pop_front()
    check('pop_front id', row['id'] == 1)
    check('pop_front name', row['name'] == 'Alice')
    check('pop_front shape', df.shape == (9, 5))

def test_pop_back():
    df = make_complex_df()
    row = df.pop_back()
    check('pop_back id', row['id'] == 10)
    check('pop_back name', row['name'] == 'Judy')
    check('pop_back shape', df.shape == (9, 5))

def test_pop_empty():
    df = FastDataFrame({'a': int})
    check('pop_front empty', expect_exception(IndexError, df.pop_front))
    check('pop_back empty', expect_exception(IndexError, df.pop_back))


# =====================================================================
#  Drop & drop_raw
# =====================================================================
def test_drop():
    df = make_complex_df()
    df.drop('tag')
    check('drop shape', df.shape == (10, 4))
    check('drop not in cols', 'tag' not in df.columns)

def test_drop_with_index():
    df = make_complex_df()
    df2 = df.drop_with_index([0, 4, 9])
    check('drop_with_index shape', df2.shape == (7, 5))
    check('drop_with_index ids', [df2['id'][i] for i in range(7)] == [2, 3, 4, 6, 7, 8, 9])

def test_drop_raw():
    df = make_complex_df()
    df.drop_raw(0)
    check('drop_raw first', df.shape == (9, 5))
    check('drop_raw first id', df['id'][0] == 2)

    df = make_complex_df()
    df.drop_raw(9)
    check('drop_raw last', df.shape == (9, 5))
    check('drop_raw last name', df['name'][8] == 'Ivan')

    df = make_complex_df()
    df.drop_raw(5)
    check('drop_raw mid', df['id'][5] == 7)


# =====================================================================
#  Sort
# =====================================================================
def test_sort_int():
    df = make_complex_df()
    s = df.sort_values('id', reverse=True)
    check('sort id desc', [s['id'][i] for i in range(10)] == [10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

def test_sort_float():
    df = make_complex_df()
    s = df.sort_values('score')
    check('sort score asc', s['score'][0] == 55.0)
    check('sort score last', s['score'][9] == 99.9)

def test_sort_str():
    df = make_complex_df()
    s = df.sort_values('name')
    check('sort name asc', s['name'][0] == 'Alice')
    check('sort name last', s['name'][9] == 'Judy')

def test_sort_char256():
    df = make_complex_df()
    s = df.sort_values('tag')
    check('sort tag asc', s['tag'][0] == 'low')
    check('sort tag last', s['tag'][9] == 'senior')

def test_sort_timestamp():
    df = make_complex_df()
    s = df.sort_values('dt')
    check('sort dt asc', s['dt'][0] == 20220615)
    check('sort dt last', s['dt'][9] == 20240229)

def test_sort_index():
    df = make_complex_df()
    s = df.sort_values('id', reverse=True)
    check('sort index desc', s.index.values() == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0])


# =====================================================================
#  Head / tail
# =====================================================================
def test_head_tail():
    df = make_complex_df()
    h = df.head(3)
    check('head 3 shape', h.shape == (3, 5))
    check('head 3 ids', [h['id'][i] for i in range(3)] == [1, 2, 3])
    check('head 3 index', h.index.values() == [0, 1, 2])

    t = df.tail(3)
    check('tail 3 shape', t.shape == (3, 5))
    check('tail 3 ids', [t['id'][i] for i in range(3)] == [8, 9, 10])
    check('tail 3 index', t.index.values() == [7, 8, 9])

    check('head 0', df.head(0).shape == (0, 5))
    check('head large', df.head(100).shape == (10, 5))
    check('tail neg', expect_exception(ValueError, df.tail, -1))


# =====================================================================
#  Diff
# =====================================================================
def test_diff_int():
    df = FastDataFrame({'a': int})
    for v in [1, 3, 6, 10]: df.append_row(a=v)
    res = df.diff('a')
    check('diff int [0]', res[0] == 0)
    check('diff int [1]', res[1] == 2)
    check('diff int [2]', res[2] == 3)
    check('diff int [3]', res[3] == 4)

def test_diff_float():
    df = FastDataFrame({'a': float})
    df.append_row(a=1.0); df.append_row(a=3.0); df.append_row(a=6.0)
    res = df.diff('a')
    check('diff float [0] nan', math.isnan(res[0]))
    check('diff float [1]', res[1] == 2.0)
    check('diff float [2]', res[2] == 3.0)


# =====================================================================
#  Null handling
# =====================================================================
def test_null_handling():
    df = FastDataFrame({'a': float, 'b': int})
    df.append_row(a=1.0, b=10)
    df.append_row(a=float('nan'), b=20)
    df.append_row(a=3.0, b=30)

    null_df = df.isnull()
    check('isnull a[0]', null_df['a'][0] == 0)
    check('isnull a[1]', null_df['a'][1] == 1)
    check('isnull a[2]', null_df['a'][2] == 0)

    filled = df.fillna(0.0)
    check('fillna a[1]', filled['a'][1] == 0.0)

    cleaned = df.dropna()
    check('dropna shape', cleaned.shape == (2, 2))
    check('dropna ids', [cleaned['b'][i] for i in range(2)] == [10, 30])


# =====================================================================
#  Describe
# =====================================================================
def test_describe():
    df = make_complex_df()
    stats = df.describe()
    check('describe count', stats['id']['count'] == 10)
    check('describe mean', stats['id']['mean'] == 5.5)
    check('describe str count', stats['name']['count'] == 10)


# =====================================================================
#  CSV / Pickle / TSV
# =====================================================================
def test_csv():
    df = make_complex_df()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        fname = f.name
    try:
        df.to_csv(fname)
        from fluidframe.core.fluidframe import read_csv
        df2 = read_csv(fname, schema={'id': int, 'score': float, 'name': str, 'tag': Char256, 'dt': Date})
        check('csv shape', df2.shape == (10, 5))
        check('csv id[0]', df2['id'][0] == 1)
        check('csv name[4]', df2['name'][4] == 'Eve')
    finally:
        os.unlink(fname)

def test_pickle():
    df = make_complex_df()
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        fname = f.name
    try:
        df.to_pickle(fname)
        from fluidframe.core.fluidframe import from_pickle
        df2 = from_pickle(fname)
        check('pkl shape', df2.shape == (10, 5))
        check('pkl id[3]', df2['id'][3] == 4)
        check('pkl score[8]', df2['score'][8] == 99.9)
    finally:
        os.unlink(fname)

def test_tsv():
    df = make_complex_df()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False, newline='') as f:
        fname = f.name
    try:
        df.to_tsv(fname)
        check('tsv exists', os.path.exists(fname))
    finally:
        os.unlink(fname)

def test_read_csv_auto():
    from fluidframe.core.fluidframe import read_csv
    content = "id,score,name,tag,dt\n1,88.5,Alice,senior,20230115\n2,92.0,Bob,mid,20230620\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(content); fname = f.name
    try:
        df = read_csv(fname, schema={'id': int, 'score': float, 'name': str, 'tag': str, 'dt': int})
        check('auto shape', df.shape == (2, 5))
        check('auto id[0]', df['id'][0] == 1)
        check('auto name[1]', df['name'][1] == 'Bob')
    finally:
        os.unlink(fname)


# =====================================================================
#  Concat & Merge
# =====================================================================
def test_concat():
    from fluidframe.core.fluidframe import concat_outer, concat_inner
    check('concat empty', concat_outer([]).shape == (0, 0))

    df1 = make_complex_df().head(5)
    df2 = make_complex_df().tail(5)
    r = concat_outer([df1, df2])
    check('concat two shape', r.shape == (10, 6))
    check('concat two has index col', 'index' in r.columns)

def test_merge():
    from fluidframe.core.fluidframe import merge_inner, merge_left, merge_outer
    df1 = FastDataFrame({'id': int, 'val1': float})
    df1.append_row(id=1, val1=10.0); df1.append_row(id=2, val1=20.0)
    df2 = FastDataFrame({'id': int, 'val2': float})
    df2.append_row(id=2, val2=200.0); df2.append_row(id=3, val2=300.0)
    check('merge inner', merge_inner([df1, df2], 'id', 'id').shape == (1, 3))
    check('merge left', merge_left([df1, df2], 'id', 'id').shape == (2, 3))
    check('merge outer', merge_outer([df1, df2], 'id', 'id').shape == (3, 3))


# =====================================================================
#  Reset index / Sort index
# =====================================================================
def test_reset_index():
    df = make_complex_df()
    df.sort_values('score', inplace=True)
    df.reset_index()
    check('reset_index has col', 'index' in df.columns)
    check('reset_index val', df['index'][0] == 0)
    check('reset_index id', df['id'][0] == 10)


# =====================================================================
#  Unique / Argmax / Argmin
# =====================================================================
def test_unique():
    df = make_complex_df()
    check('unique id', df.unique('id') == list(range(1, 11)))
    check('unique tag', set(df.unique('tag')) == {'senior', 'mid', 'low'})

def test_argmax_argmin():
    df = make_complex_df()
    check('argmax score', df.argmax('score') == 8)
    check('argmin score', df.argmin('score') == 9)


# =====================================================================
#  iloc
# =====================================================================
def test_iloc():
    df = make_complex_df()

    r = df.iloc[0]
    check('iloc single', r.shape == (1, 5))
    check('iloc single id', r['id'][0] == 1)

    r = df.iloc[4]
    check('iloc row 4', r['id'][0] == 5)
    check('iloc row 4 name', r['name'][0] == 'Eve')

    r = df.iloc[-1]
    check('iloc [-1]', r['id'][0] == 10)

    r = df.iloc[2:5]
    check('iloc slice shape', r.shape == (3, 5))
    check('iloc slice ids', [r['id'][i] for i in range(3)] == [3, 4, 5])

    r = df.iloc[[0, 4, 9]]
    check('iloc list ids', [r['id'][i] for i in range(3)] == [1, 5, 10])

    r = df.iloc[10:20]
    check('iloc empty', r.shape == (0, 5))

    check('iloc oob', expect_exception(IndexError, df.iloc.__getitem__, 20))


# =====================================================================
#  loc — index label lookup
# =====================================================================
def test_loc_label():
    df = make_complex_df()
    r = df.loc[0]
    check('loc label', r.shape == (1, 5))
    check('loc label id', r['id'][0] == 1)

    r = df.loc[5]
    check('loc label 5', r['id'][0] == 6)

    check('loc missing', expect_exception(KeyError, df.loc.__getitem__, 99))


# =====================================================================
#  loc — boolean mask filtering (multi-type combinations)
# =====================================================================
def test_loc_int_mask():
    df = make_complex_df()
    mask = df['id'] > 5
    r = df.loc[mask]
    check('int mask shape', r.shape == (5, 5))
    check('int mask ids', [r['id'][i] for i in range(5)] == [6, 7, 8, 9, 10])

def test_loc_float_mask():
    df = make_complex_df()
    mask = df['score'] >= 90.0
    r = df.loc[mask]
    check('float mask shape', r.shape == (3, 5))
    check('float mask names', set(r['name'][i] for i in range(3)) == {'Bob', 'Eve', 'Ivan'})

def test_loc_char256_mask():
    df = make_complex_df()
    mask = df['tag'].len() <= 3
    r = df.loc[mask]
    check('char256 mask shape', r.shape == (6, 5))
    check('char256 mask tags', set(r['tag'][i] for i in range(6)) == {'mid', 'low'})

def test_loc_ucs2_mask():
    df = make_complex_df()
    mask = df['name'].len() <= 4
    r = df.loc[mask]
    check('ucs2 mask shape', r.shape == (4, 5))
    check('ucs2 mask names', set(r['name'][i] for i in range(4)) == {'Bob', 'Eve', 'Ivan', 'Judy'})

def test_loc_timestamp_mask():
    df = make_complex_df()
    mask = df['dt'] >= 20240101
    r = df.loc[mask]
    check('ts mask shape', r.shape == (2, 5))
    check('ts mask ids', [r['id'][i] for i in range(2)] == [4, 9])

def test_loc_and_combined():
    """int AND float AND timestamp — 3 different types"""
    df = make_complex_df()
    mask = (df['id'] > 3) & (df['score'] > 70.0) & (df['dt'] < 20240101)
    r = df.loc[mask]
    # Rows: id=5(95.1,20230115), id=6(82.7,20230310), id=8(88.5,20231225)
    check('and 3-type shape', r.shape == (3, 5))
    check('and 3-type ids', sorted([r['id'][i] for i in range(3)]) == [5, 6, 8])

def test_loc_or_combined():
    """int OR char256 len — 2 different types"""
    df = make_complex_df()
    mask = (df['id'] <= 2) | (df['tag'].len() <= 3)
    r = df.loc[mask]
    # id<=2: rows 0,1; tag.len<=3 (mid/low): rows 1,3,5,6,8,9 → union: rows 0,1,3,5,6,8,9 → 7 rows
    check('or 2-type shape', r.shape[0] >= 5)
    check('or 2-type has id1', r['id'][0] == 1)

def test_loc_not_mask():
    df = make_complex_df()
    mask = ~(df['id'] > 5)
    r = df.loc[mask]
    check('not mask shape', r.shape == (5, 5))
    check('not mask ids', [r['id'][i] for i in range(5)] == [1, 2, 3, 4, 5])

def test_loc_complex_4type():
    """int AND float AND char256 AND timestamp — 4 types"""
    df = make_complex_df()
    mask = (df['id'] >= 3) & (df['score'] < 90.0) & (df['tag'].startswith('sen')) & (df['dt'] <= 20230620)
    r = df.loc[mask]
    # id=3(75.3,senior,20221201), id=8(88.5,senior,20231225 not <=20230620) → only id=3
    check('4-type shape', r.shape == (1, 5))
    check('4-type id', r['id'][0] == 3)

def test_loc_callable():
    df = make_complex_df()
    r = df.loc[lambda d: d['id'] > 7]
    check('callable shape', r.shape == (3, 5))
    check('callable ids', [r['id'][i] for i in range(3)] == [8, 9, 10])

def test_loc_bool_list():
    df = make_complex_df()
    mask_list = [i % 2 == 0 for i in range(10)]
    r = df.loc[mask_list]
    check('bool list shape', r.shape == (5, 5))
    check('bool list ids', [r['id'][i] for i in range(5)] == [1, 3, 5, 7, 9])


# =====================================================================
#  loc — combined with index label
# =====================================================================
def test_loc_mask_preserves_index():
    df = make_complex_df()
    mask = df['id'] > 5
    r = df.loc[mask]
    check('index preserved', r.index.values() == [5, 6, 7, 8, 9])


# =====================================================================
#  Stress test: large random dataset
# =====================================================================
def _random_string(length):
    return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=length))

def _random_date():
    return random.randint(20200101, 20251231)

def test_stress_append():
    N = 10000
    t0 = time.time()
    df = FastDataFrame({'id': int, 'val': float, 'name': str, 'tag': Char256, 'dt': Date})
    for i in range(N):
        df.append_row(id=i, val=random.random() * 100,
                      name=_random_string(random.randint(3, 8)),
                      tag=random.choice(['A', 'B', 'C', 'D']),
                      dt=_random_date())
    elapsed = time.time() - t0
    check('stress append shape', df.shape == (N, 5))
    check('stress append speed', elapsed < 10.0)

def test_stress_sort():
    N = 5000
    df = FastDataFrame({'id': int, 'val': float})
    for i in range(N):
        df.append_row(id=random.randint(0, N), val=random.random())
    t0 = time.time()
    s = df.sort_values('val')
    elapsed = time.time() - t0
    check('stress sort shape', s.shape == (N, 2))
    ok = True
    for i in range(N - 1):
        if s['val'][i] > s['val'][i + 1]:
            ok = False
            break
    check('stress sort order', ok)
    check('stress sort speed', elapsed < 5.0)

def test_stress_loc_filter():
    N = 10000
    df = FastDataFrame({'id': int, 'val': float, 'tag': Char256})
    for i in range(N):
        df.append_row(id=i, val=random.random() * 100,
                      tag=random.choice(['X', 'Y', 'Z']))
    mask = (df['id'] > 1000) & (df['val'] > 50.0)
    t0 = time.time()
    r = df.loc[mask]
    elapsed = time.time() - t0
    check('stress loc shape', r.shape[1] == 3)
    check('stress loc all id>1000', all(r['id'][i] > 1000 for i in range(r.shape[0])))
    check('stress loc all val>50', all(r['val'][i] > 50.0 for i in range(r.shape[0])))
    check('stress loc speed', elapsed < 2.0)

def test_stress_iloc():
    N = 10000
    df = FastDataFrame({'a': int})
    for i in range(N):
        df.append_row(a=i)
    r = df.iloc[5000:6000]
    check('stress iloc shape', r.shape == (1000, 1))
    check('stress iloc val', r['a'][0] == 5000)
    r2 = df.iloc[[0, N // 2, N - 1]]
    check('stress iloc list', r2.shape == (3, 1))
    check('stress iloc list vals', [r2['a'][i] for i in range(3)] == [0, 5000, 9999])

def test_stress_copy():
    N = 10000
    df = FastDataFrame({'a': int, 'b': float})
    for i in range(N):
        df.append_row(a=i, b=random.random())
    t0 = time.time()
    df2 = df.copy()
    elapsed = time.time() - t0
    check('stress copy shape', df2.shape == (N, 2))
    check('stress copy speed', elapsed < 2.0)

def test_stress_drop_with_index():
    N = 5000
    df = FastDataFrame({'a': int})
    for i in range(N):
        df.append_row(a=i)
    drop_idx = list(range(0, N, 2))  # drop every even index
    t0 = time.time()
    r = df.drop_with_index(drop_idx)
    elapsed = time.time() - t0
    check('stress drop shape', r.shape == (N // 2, 1))
    check('stress drop vals', [r['a'][i] for i in range(5)] == [1, 3, 5, 7, 9])
    check('stress drop speed', elapsed < 2.0)

def test_stress_pop_front_back():
    N = 5000
    df = FastDataFrame({'a': int})
    for i in range(N):
        df.append_row(a=i)
    t0 = time.time()
    for _ in range(1000):
        df.pop_front()
    elapsed = time.time() - t0
    check('stress pop_front shape', df.shape == (N - 1000, 1))
    check('stress pop_front val', df['a'][0] == 1000)
    check('stress pop_front speed', elapsed < 2.0)

def test_stress_csv_roundtrip():
    N = 5000
    df = FastDataFrame({'id': int, 'val': float, 'name': str})
    for i in range(N):
        df.append_row(id=i, val=random.random(), name=_random_string(6))
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        fname = f.name
    try:
        t0 = time.time()
        df.to_csv(fname)
        from fluidframe.core.fluidframe import read_csv
        df2 = read_csv(fname, schema={'id': int, 'val': float, 'name': str})
        elapsed = time.time() - t0
        check('stress csv shape', df2.shape == (N, 3))
        check('stress csv id[0]', df2['id'][0] == 0)
        check('stress csv speed', elapsed < 5.0)
    finally:
        os.unlink(fname)

def test_stress_multi_type_filter():
    """Stress test: filter on int + float + char256 + timestamp simultaneously"""
    N = 8000
    df = FastDataFrame({'id': int, 'score': float, 'tag': Char256, 'dt': Date})
    for i in range(N):
        df.append_row(id=i, score=random.random() * 100,
                      tag=random.choice(['alpha', 'beta', 'gamma']),
                      dt=_random_date())
    mask = (df['id'] > 1000) & (df['score'] > 30.0) & (df['tag'].len() <= 4) & (df['dt'] >= 20230101)
    t0 = time.time()
    r = df.loc[mask]
    elapsed = time.time() - t0
    check('stress 4-type shape', r.shape[1] == 4)
    check('stress 4-type id>1000', all(r['id'][i] > 1000 for i in range(r.shape[0])))
    check('stress 4-type score>30', all(r['score'][i] > 30.0 for i in range(r.shape[0])))
    check('stress 4-type tag<=4', all(len(r['tag'][i]) <= 4 for i in range(r.shape[0])))
    check('stress 4-type dt>=20230101', all(r['dt'][i] >= 20230101 for i in range(r.shape[0])))
    check('stress 4-type speed', elapsed < 3.0)


# =====================================================================
#  Run all
# =====================================================================
def run_all():
    print('=== FastDataFrame Tests ===')

    # Construction
    test_init()
    test_init_complex()

    # Append & access
    test_append()
    test_append_complex()
    test_access()

    # Copy & pop
    test_copy()
    test_pop_front()
    test_pop_back()
    test_pop_empty()

    # Drop
    test_drop()
    test_drop_with_index()
    test_drop_raw()

    # Sort
    test_sort_int()
    test_sort_float()
    test_sort_str()
    test_sort_char256()
    test_sort_timestamp()
    test_sort_index()

    # Head / tail
    test_head_tail()

    # Diff
    test_diff_int()
    test_diff_float()

    # Null handling
    test_null_handling()

    # Describe
    test_describe()

    # I/O
    test_csv()
    test_pickle()
    test_tsv()
    test_read_csv_auto()

    # Concat & merge
    test_concat()
    test_merge()

    # Reset index
    test_reset_index()

    # Unique / argmax / argmin
    test_unique()
    test_argmax_argmin()

    # iloc
    test_iloc()

    # loc — label
    test_loc_label()

    # loc — boolean masks (single type)
    test_loc_int_mask()
    test_loc_float_mask()
    test_loc_char256_mask()
    test_loc_ucs2_mask()
    test_loc_timestamp_mask()

    # loc — combined masks (multi-type)
    test_loc_and_combined()
    test_loc_or_combined()
    test_loc_not_mask()
    test_loc_complex_4type()
    test_loc_callable()
    test_loc_bool_list()

    # loc — index preservation
    test_loc_mask_preserves_index()

    # Stress tests
    print('\n--- stress tests ---')
    test_stress_append()
    test_stress_sort()
    test_stress_loc_filter()
    test_stress_iloc()
    test_stress_copy()
    test_stress_drop_with_index()
    test_stress_pop_front_back()
    test_stress_csv_roundtrip()
    test_stress_multi_type_filter()

    print(f'\nResults: {passed} passed, {failed} failed')
    if failed: sys.exit(1)

if __name__ == '__main__':
    run_all()
