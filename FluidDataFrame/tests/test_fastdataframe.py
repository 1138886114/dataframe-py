import math
import os
import random
import sys
import tempfile
import time
sys.path.insert(0, '..')
from fluidframe.core.fastdataframe import FastDataFrame, read_json
from fluidframe.core.intcolumn import IntColumn
from fluidframe.core.doublecolumn import DoubleColumn
from fluidframe.core.asciicolumn import AsciiColumn
from fluidframe.core.ucs2column import UCS2Column
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
    check('schema str', isinstance(df.columns['a'], UCS2Column))

    df = FastDataFrame({'a': Char256})
    check('schema char256', isinstance(df.columns['a'], AsciiColumn))

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
        from fluidframe.core.fastdataframe import read_csv
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
        from fluidframe.core.fastdataframe import from_pickle
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
    from fluidframe.core.fastdataframe import read_csv
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
    from fluidframe.core.fastdataframe import concat_outer, concat_inner
    check('concat empty', concat_outer([]).shape == (0, 0))

    df1 = make_complex_df().head(5)
    df2 = make_complex_df().tail(5)
    r = concat_outer([df1, df2])
    check('concat two shape', r.shape == (10, 6))
    check('concat two has index col', 'index' in r.columns)

def test_merge():
    df1 = FastDataFrame({'id': int, 'val1': float})
    df1.append_row(id=1, val1=10.0); df1.append_row(id=2, val1=20.0)
    df2 = FastDataFrame({'id': int, 'val2': float})
    df2.append_row(id=2, val2=200.0); df2.append_row(id=3, val2=300.0)
    check('merge inner', df1.merge(df2, on='id', how='inner').shape == (1, 3))
    check('merge left', df1.merge(df2, on='id', how='left').shape == (2, 3))
    check('merge outer', df1.merge(df2, on='id', how='outer').shape == (3, 3))


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
        from fluidframe.core.fastdataframe import read_csv
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
#  Construction: from_dict, from_list, from_arrays
# =====================================================================
def test_from_dict():
    df = FastDataFrame.from_dict({'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0]}, schema={'a': int, 'b': float})
    check('from_dict shape', df.shape == (3, 2))
    check('from_dict a[1]', df['a'][1] == 2)
    check('from_dict b[2]', df['b'][2] == 6.0)

    df2 = FastDataFrame.from_dict({'x': [10]}, schema={'x': int}, columns=['val'])
    check('from_dict rename', 'val' in df2.columns)

    check('from_dict empty', FastDataFrame.from_dict({}, schema={}).shape == (0, 0))
    check('from_dict mismatch', expect_exception(ValueError, FastDataFrame.from_dict,
          {'a': [1, 2], 'b': [3]}, schema={'a': int, 'b': int}))

def test_from_list():
    df = FastDataFrame.from_list([[1, 'a'], [2, 'b'], [3, 'c']], schema={'id': int, 'ch': str})
    check('from_list shape', df.shape == (3, 2))
    check('from_list id[0]', df['id'][0] == 1)
    check('from_list ch[2]', df['ch'][2] == 'c')

    df2 = FastDataFrame.from_list([[10, 20]], schema={'x': int, 'y': int}, columns=['a', 'b'])
    check('from_list rename', set(df2.columns.keys()) == {'a', 'b'})

def test_from_arrays():
    df = FastDataFrame.from_arrays([[1, 2, 3], [10.0, 20.0, 30.0]], schema={'id': int, 'val': float})
    check('from_arrays shape', df.shape == (3, 2))
    check('from_arrays id[2]', df['id'][2] == 3)
    check('from_arrays val[0]', df['val'][0] == 10.0)

    df2 = FastDataFrame.from_arrays([[1]], schema={'x': int}, columns=['col'])
    check('from_arrays rename', 'col' in df2.columns)

def test_factory_methods():
    df = FastDataFrame.arange(7)
    check('arange shape', df.shape == (7, 1))
    check('arange vals', [df['0'][i] for i in range(7)] == [0, 1, 2, 3, 4, 5, 6])

    df = FastDataFrame.zeros(3, 2)
    check('zeros', all(df.columns[c][i] == 0.0 for c in df.columns for i in range(3)))

    df = FastDataFrame.ones(2, 3)
    check('ones', all(df.columns[c][i] == 1.0 for c in df.columns for i in range(2)))

    df = FastDataFrame.randn(100, 2)
    check('randn shape', df.shape == (100, 2))

    df = FastDataFrame.rand(50, 2)
    check('rand shape', df.shape == (50, 2))
    check('rand range', all(0 <= df.columns[c][i] < 1 for c in df.columns for i in range(50)))


# =====================================================================
#  Column mutations: rename, insert_column, duplicate_column, assign, __setitem__
# =====================================================================
def test_rename():
    df = make_complex_df().head(3)
    df2 = df.rename({'id': 'ID', 'name': 'NAME'})
    check('rename cols', set(df2.columns.keys()) == {'ID', 'score', 'NAME', 'tag', 'dt'})
    check('rename val', df2['ID'][0] == 1)
    check('rename original unchanged', 'id' in df.columns)

def test_insert_column():
    df = FastDataFrame({'a': int})
    df.append_row(a=1); df.append_row(a=2)
    df.insert_column('b', [10, 20], pos=0)
    check('insert pos 0', list(df.columns.keys()) == ['b', 'a'])
    check('insert val', df['b'][1] == 20)

    df.insert_column('c', [100], pos=1)
    check('insert pos 1', list(df.columns.keys()) == ['b', 'c', 'a'])

def test_duplicate_column():
    df = make_complex_df().head(3)
    df.duplicate_column('id', 'id_copy')
    check('dup cols', 'id_copy' in df.columns)
    check('dup val', df['id_copy'][0] == 1)
    check('dup len', df.shape == (3, 6))

    df.duplicate_column('name')
    check('dup default name', 'name_dup' in df.columns)

def test_assign():
    df = FastDataFrame({'a': int, 'b': int})
    df.append_row(a=1, b=2); df.append_row(a=3, b=4)
    df2 = df.assign(c=[10, 20])
    check('assign new col', 'c' in df2.columns)
    check('assign val', df2['c'][0] == 10)
    check('assign original unchanged', 'c' not in df.columns)

    df3 = df.assign(a=[100, 200])
    check('assign replace', df3['a'][0] == 100)

def test_setitem():
    df = FastDataFrame({'a': int})
    df.append_row(a=1)
    df['b'] = [10]
    check('setitem list', df['b'][0] == 10)
    check('setitem cols', set(df.columns.keys()) == {'a', 'b'})

    df['a'] = [99]
    check('setitem replace', df['a'][0] == 99)


# =====================================================================
#  Slicing: sample, nsmallest, nlargest, duplicated, drop_duplicates, select_dtypes, reindex
# =====================================================================
def test_sample():
    df = make_complex_df()
    s = df.sample(3, seed=42)
    check('sample shape', s.shape == (3, 5))

    s2 = df.sample(frac=0.5, seed=42)
    check('sample frac', s2.shape[0] == 5)

    s3 = df.sample(5, replace=True, seed=42)
    check('sample replace', s3.shape == (5, 5))

def test_nsmallest_nlargest():
    df = make_complex_df()
    n = df.nsmallest(3, 'score')
    check('nsmallest shape', n.shape == (3, 5))
    check('nsmallest order', n['score'][0] <= n['score'][1] <= n['score'][2])
    check('nsmallest val', n['score'][0] == 55.0)

    l = df.nlargest(3, 'score')
    check('nlargest shape', l.shape == (3, 5))
    check('nlargest order', l['score'][0] >= l['score'][1] >= l['score'][2])
    check('nlargest val', l['score'][0] == 99.9)

def test_duplicated():
    df = FastDataFrame({'a': int, 'b': str})
    df.append_row(a=1, b='x'); df.append_row(a=2, b='y'); df.append_row(a=1, b='x')
    dup = df.duplicated()
    check('dup mask', [dup[i] for i in range(3)] == [0, 0, 1])

    dup2 = df.duplicated(keep='last')
    check('dup last', [dup2[i] for i in range(3)] == [1, 0, 0])

    dup3 = df.duplicated(subset=['a'])
    check('dup subset', [dup3[i] for i in range(3)] == [0, 0, 1])

def test_drop_duplicates():
    df = FastDataFrame({'a': int, 'b': str})
    df.append_row(a=1, b='x'); df.append_row(a=2, b='y'); df.append_row(a=1, b='x'); df.append_row(a=3, b='z')
    r = df.drop_duplicates()
    check('drop_dup shape', r.shape == (3, 2))
    check('drop_dup vals', [r['a'][i] for i in range(3)] == [1, 2, 3])

    df2 = FastDataFrame({'a': int, 'b': str})
    df2.append_row(a=1, b='x'); df2.append_row(a=1, b='y'); df2.append_row(a=1, b='z')
    r2 = df2.drop_duplicates(subset=['a'], keep='last')
    check('drop_dup keep last', r2.shape == (1, 2))
    check('drop_dup last val', r2['b'][0] == 'z')

def test_select_dtypes():
    df = make_complex_df()
    int_only = df.select_dtypes(include_types=['int'])
    check('select int', set(int_only.columns.keys()) == {'id'})
    float_only = df.select_dtypes(include_types=['float'])
    check('select float', set(float_only.columns.keys()) == {'score'})
    no_int = df.select_dtypes(exclude_types=['int'])
    check('exclude int', 'id' not in no_int.columns)

def test_reindex():
    df = FastDataFrame({'a': int, 'b': str})
    df.append_row(a=1, b='x'); df.append_row(a=2, b='y')
    r = df.reindex([0, 1, 2, 3])
    check('reindex shape', r.shape == (4, 2))
    check('reindex val', r['a'][2] == 0)
    check('reindex val2', r['b'][3] == '')


# =====================================================================
#  Aggregation: mean, median, std, var, max, min, sum, count, quantile, skew, kurt
# =====================================================================
def test_aggregation():
    df = FastDataFrame({'a': int, 'b': float})
    for i in range(10):
        df.append_row(a=i, b=i * 1.5)

    m = df.mean()
    check('mean a', m['a'] == 4.5)
    check('mean b', m['b'] == 6.75)

    med = df.median()
    check('median a', med['a'] == 4.5)

    s = df.sum()
    check('sum a', s['a'] == 45)
    check('sum b', s['b'] == 67.5)

    mx = df.max()
    check('max a', mx['a'] == 9)
    check('max b', mx['b'] == 13.5)

    mn = df.min()
    check('min a', mn['a'] == 0)
    check('min b', mn['b'] == 0.0)

    cnt = df.count()
    check('count a', cnt['a'] == 10)

    sd = df.std()
    check('std a', sd['a'] > 0)

    v = df.var()
    check('var a', v['a'] > 0)

    q = df.quantile(0.5)
    check('quantile 0.5', q['a'] == 4.5)

    sk = df.skew()
    check('skew a', isinstance(sk['a'], float))

    ku = df.kurt()
    check('kurt a', isinstance(ku['a'], float))

def test_aggregation_complex():
    df = make_complex_df()
    m = df.mean()
    check('complex mean id', m['id'] == 5.5)
    mx = df.max()
    check('complex max score', mx['score'] == 99.9)
    mn = df.min()
    check('complex min id', mn['id'] == 1)


# =====================================================================
#  Cumulative: cumsum, cumprod, cummax, cummin
# =====================================================================
def test_cumulative():
    df = FastDataFrame({'a': int, 'b': float})
    for i in range(1, 6):
        df.append_row(a=i, b=float(i))

    cs = df.cumsum()
    check('cumsum a[0]', cs['a'][0] == 1)
    check('cumsum a[4]', cs['a'][4] == 15)

    cp = df.cumprod()
    check('cumprod a[0]', cp['a'][0] == 1)
    check('cumprod a[4]', cp['a'][4] == 120)

    cmx = df.cummax()
    check('cummax a[2]', cmx['a'][2] == 3)
    df2 = FastDataFrame({'x': int})
    for v in [3, 1, 4, 1, 5]: df2.append_row(x=v)
    cmx2 = df2.cummax()
    check('cummax varying', [cmx2['x'][i] for i in range(5)] == [3, 3, 4, 4, 5])

    cmn = df.cummin()
    check('cummin a[3]', cmn['a'][3] == 1)


# =====================================================================
#  Column arithmetic: add, sub, mul, div
# =====================================================================
def test_col_arithmetic():
    df = FastDataFrame({'a': int, 'b': int})
    df.append_row(a=10, b=3); df.append_row(a=20, b=4)

    r = df.add(('a', 'b'))
    check('add', [r[i] for i in range(2)] == [13, 24])

    r = df.sub(('a', 'b'))
    check('sub', [r[i] for i in range(2)] == [7, 16])

    r = df.mul(('a', 'b'))
    check('mul', [r[i] for i in range(2)] == [30, 80])

    r = df.div(('a', 'b'))
    check('div', [r[i] for i in range(2)] == [10/3, 5.0])


# =====================================================================
#  Statistics: corr, cov, corrcoef, dist functions
# =====================================================================
def test_corr_cov():
    df = FastDataFrame({'a': float, 'b': float})
    for i in range(10):
        df.append_row(a=float(i), b=float(i) * 2)

    c = df.corr()
    check('corr shape', c.shape == (2, 2))
    check('corr self', c['a'][0] == 1.0)
    check('corr perfect', abs(c['a'][1] - 1.0) < 1e-9)

    cv = df.cov()
    check('cov shape', cv.shape == (2, 2))
    check('cov self', cv['a'][0] > 0)

    r = df.corrcoef(df['a'], df['b'])
    check('corrcoef', abs(r - 1.0) < 1e-9)

    cv2 = df.cov_withcols(df['a'], df['b'])
    check('cov_withcols', cv2 > 0)

    r2 = df.corrwith(df['a'])
    check('corrwith', r2['a'] == 1.0)

    ps = df.pearson_sim()
    check('pearson_sim', ps['a'][0] == 1.0)

def test_distance_functions():
    df = FastDataFrame({'a': float, 'b': float})
    for i in range(5):
        df.append_row(a=float(i), b=float(i) + 1.0)

    ed = df.euclidean_dist_withcols(df['a'], df['b'])
    check('euclidean', abs(ed - math.sqrt(5)) < 1e-9)

    md = df.manhattan_dist_withcols(df['a'], df['b'])
    check('manhattan', md == 5.0)

    cs = df.cosine_dist_withcols(df['a'], df['b'])
    check('cosine_dist', 0 <= cs <= 2)

    csim = df.cosine_similarity_withcols(df['a'], df['b'])
    check('cosine_sim', 0.9 < csim <= 1.0)

    js = df.jaccard_sim_withcols(df['a'], df['b'])
    check('jaccard_sim', 0 <= js <= 1)

    hm = df.hamming_dist_withcols(df['a'], df['b'])
    check('hamming', hm >= 0)


# =====================================================================
#  Column ops: value_counts, clip_column, where_column, mask_column, where_row
# =====================================================================
def test_value_counts():
    df = FastDataFrame({'tag': str})
    for t in ['a', 'b', 'a', 'c', 'a', 'b']: df.append_row(tag=t)
    vc = df.value_counts('tag')
    check('value_counts shape', vc.shape[0] == 3)
    check('value_counts top', vc['tag'][0] == 'a')
    check('value_counts count', vc['count'][0] == 3)

def test_clip_column():
    df = FastDataFrame({'a': float})
    for v in [1.0, 5.0, 10.0, 15.0, 20.0]: df.append_row(a=v)
    df.clip_column('a', 5.0, 15.0)
    check('clip min', df['a'][0] == 5.0)
    check('clip mid', df['a'][1] == 5.0)
    check('clip max', df['a'][4] == 15.0)

def test_where_column():
    df = FastDataFrame({'val': int, 'cond': int})
    df.append_row(val=10, cond=1); df.append_row(val=20, cond=0); df.append_row(val=30, cond=1)
    r = df.where_column('val', 'cond', 999)
    check('where', [r[i] for i in range(3)] == [10, 999, 30])

def test_mask_column():
    df = FastDataFrame({'val': int, 'cond': int})
    df.append_row(val=10, cond=1); df.append_row(val=20, cond=0); df.append_row(val=30, cond=1)
    r = df.mask_column('val', 'cond', 999)
    check('mask', [r[i] for i in range(3)] == [999, 20, 999])

def test_where_row():
    df = FastDataFrame({'a': int, 'b': int})
    df.append_row(a=1, b=10); df.append_row(a=2, b=20); df.append_row(a=3, b=30)
    mask = IntColumn()
    mask.append(1); mask.append(0); mask.append(1)
    r = df.where_row(mask, fill=-1)
    check('where_row a', [r['a'][i] for i in range(3)] == [1, -1, 3])
    check('where_row b', [r['b'][i] for i in range(3)] == [10, -1, 30])


# =====================================================================
#  Groupby
# =====================================================================
def test_groupby():
    df = FastDataFrame({'dept': str, 'age': int, 'salary': float})
    rows = [
        ('eng', 25, 80000.0), ('eng', 30, 100000.0), ('eng', 35, 120000.0),
        ('sales', 28, 70000.0), ('sales', 32, 85000.0),
        ('hr', 40, 90000.0), ('hr', 29, 65000.0),
    ]
    for r in rows:
        df.append_row(dept=r[0], age=r[1], salary=r[2])

    g = df.groupby('dept')
    r = g.sum()
    check('groupby sum cols', 'age' in r.columns and 'salary' in r.columns)
    check('groupby sum rows', r.shape[0] == 3)

    r = g.mean()
    check('groupby mean', r.shape[0] == 3)

    r = g.max()
    check('groupby max', r.shape[0] == 3)

    r = g.min()
    check('groupby min', r.shape[0] == 3)

    r = g.count()
    check('groupby count', r.shape[0] == 3)

    r = g.std()
    check('groupby std', r.shape[0] == 3)

def test_groupby_multi_col():
    df = FastDataFrame({'a': str, 'b': str, 'val': int})
    df.append_row(a='x', b='p', val=1)
    df.append_row(a='x', b='p', val=2)
    df.append_row(a='x', b='q', val=3)
    df.append_row(a='y', b='p', val=4)
    r = df.groupby(['a', 'b']).sum()
    check('multi groupby rows', r.shape[0] == 3)


# =====================================================================
#  Merge: inner, left, right, outer
# =====================================================================
def test_merge_inner():
    df1 = FastDataFrame({'id': int, 'v1': float})
    df1.append_row(id=1, v1=10.0); df1.append_row(id=2, v1=20.0); df1.append_row(id=3, v1=30.0)
    df2 = FastDataFrame({'id': int, 'v2': float})
    df2.append_row(id=2, v2=200.0); df2.append_row(id=3, v2=300.0); df2.append_row(id=4, v2=400.0)
    r = df1.merge(df2, on='id', how='inner')
    check('inner shape', r.shape == (2, 3))
    check('inner ids', sorted([r['id'][i] for i in range(2)]) == [2, 3])

def test_merge_left():
    df1 = FastDataFrame({'id': int, 'v1': float})
    df1.append_row(id=1, v1=10.0); df1.append_row(id=2, v1=20.0)
    df2 = FastDataFrame({'id': int, 'v2': float})
    df2.append_row(id=2, v2=200.0); df2.append_row(id=3, v2=300.0)
    r = df1.merge(df2, on='id', how='left')
    check('left shape', r.shape == (2, 3))
    check('left id=1', r['v1'][0] == 10.0)

def test_merge_right():
    df1 = FastDataFrame({'id': int, 'v1': float})
    df1.append_row(id=1, v1=10.0); df1.append_row(id=2, v1=20.0)
    df2 = FastDataFrame({'id': int, 'v2': float})
    df2.append_row(id=2, v2=200.0); df2.append_row(id=3, v2=300.0)
    r = df1.merge(df2, on='id', how='right')
    check('right shape', r.shape == (2, 3))
    check('right has all ids', r.shape[0] == 2)

def test_merge_outer():
    df1 = FastDataFrame({'id': int, 'v1': float})
    df1.append_row(id=1, v1=10.0); df1.append_row(id=2, v1=20.0)
    df2 = FastDataFrame({'id': int, 'v2': float})
    df2.append_row(id=2, v2=200.0); df2.append_row(id=3, v2=300.0)
    r = df1.merge(df2, on='id', how='outer')
    check('outer shape', r.shape == (3, 3))


# =====================================================================
#  Concat: multi-DF, inner, outer
# =====================================================================
def test_concat_multi():
    from fluidframe.core.fastdataframe import concat, concat_inner
    df1 = FastDataFrame({'a': int, 'b': int})
    df1.append_row(a=1, b=10); df1.append_row(a=2, b=20)
    df2 = FastDataFrame({'a': int, 'b': int})
    df2.append_row(a=3, b=30)
    df3 = FastDataFrame({'a': int, 'b': int})
    df3.append_row(a=4, b=40); df3.append_row(a=5, b=50)
    r = concat([df1, df2, df3])
    check('concat 3', r.shape[0] == 5)
    check('concat vals', [r['a'][i] for i in range(5)] == [1, 2, 3, 4, 5])

    r2 = concat([df1, df2, df3], ignore_index=True)
    check('concat ignore_index', r2.shape[0] == 5)

def test_concat_inner():
    from fluidframe.core.fastdataframe import concat_inner
    df1 = FastDataFrame({'a': int, 'b': int})
    df1.append_row(a=1, b=10)
    df2 = FastDataFrame({'a': int, 'c': float})
    df2.append_row(a=2, c=20.0)
    r = concat_inner([df1, df2])
    check('concat_inner has a', 'a' in r.columns)
    check('concat_inner rows', r.shape[0] == 2)

def test_concat_outer():
    from fluidframe.core.fastdataframe import concat_outer
    df1 = FastDataFrame({'a': int})
    df1.append_row(a=1)
    df2 = FastDataFrame({'b': float})
    df2.append_row(b=2.0)
    r = concat_outer([df1, df2])
    check('outer has a and b', 'a' in r.columns and 'b' in r.columns)
    check('outer rows', r.shape[0] == 2)


# =====================================================================
#  eval / query
# =====================================================================
def test_eval():
    df = FastDataFrame({'a': int, 'b': int})
    df.append_row(a=10, b=3); df.append_row(a=20, b=4)
    r = df.eval('a + b')
    check('eval add', [r[i] for i in range(2)] == [13, 24])

    r = df.eval('a * 2')
    check('eval scalar', [r[i] for i in range(2)] == [20, 40])

    df.eval('c = a + b', inplace=True)
    check('eval assign', 'c' in df.columns)
    check('eval assign val', df['c'][0] == 13)

def test_query():
    df = make_complex_df()
    r = df.query('id > 5')
    check('query shape', r.shape == (5, 5))
    check('query ids', [r['id'][i] for i in range(5)] == [6, 7, 8, 9, 10])


# =====================================================================
#  apply / pipe
# =====================================================================
def test_apply():
    df = FastDataFrame({'a': int})
    df.append_row(a=1); df.append_row(a=2); df.append_row(a=3)
    r = df.apply(lambda x: x * 2)
    check('apply', [r['a'][i] for i in range(3)] == [2, 4, 6])

def test_pipe():
    df = make_complex_df()
    def add_col(dataframe, col_name, val):
        dataframe = dataframe.copy()
        dataframe[col_name] = [val] * dataframe.df_length
        return dataframe
    r = df.pipe(add_col, 'new_col', 99)
    check('pipe cols', 'new_col' in r.columns)
    check('pipe val', r['new_col'][0] == 99)


# =====================================================================
#  duplicate_rows / reshape / T / memory_usage / __len__
# =====================================================================
def test_duplicate_rows():
    df = make_complex_df()
    r = df.duplicate_rows([0, 2, 4])
    check('dup_rows shape', r.shape == (3, 5))
    check('dup_rows ids', [r['id'][i] for i in range(3)] == [1, 3, 5])

    r2 = df.duplicate_rows()
    check('dup_rows all', r2.shape == (10, 5))

def test_reshape():
    df = FastDataFrame.arange(12)
    r = df.reshape(3, 4)
    check('reshape shape', r.shape == (3, 4))
    check('reshape A[0]', r['A'][0] == 0.0)
    check('reshape D[2]', r['D'][2] == 11.0)

def test_transpose():
    df = FastDataFrame({'a': int, 'b': int})
    df.append_row(a=1, b=10); df.append_row(a=2, b=20)
    t = df.T
    check('T shape', t.shape == (2, 2))
    check('T cols', set(t.columns.keys()) == {'a', 'b'})
    check('T a[0]', t['a'][0] == 1.0)
    check('T a[1]', t['a'][1] == 10.0)

def test_memory_usage():
    df = FastDataFrame({'a': int, 'b': float})
    for i in range(100): df.append_row(a=i, b=float(i))
    mu = df.memory_usage()
    check('memory_usage positive', mu > 0)

def test_len():
    df = make_complex_df()
    check('len', len(df) == 10)
    df2 = FastDataFrame({'x': int})
    check('len empty', len(df2) == 0)


# =====================================================================
#  Insert row / drop_index / sort_index / argsort_by / nunique
# =====================================================================
def test_insert_row():
    df = FastDataFrame({'a': int, 'b': str})
    df.append_row(a=1, b='x'); df.append_row(a=3, b='z')
    df.insert_row(1, a=2, b='y')
    check('insert_row shape', df.shape == (3, 2))
    check('insert_row val', [df['a'][i] for i in range(3)] == [1, 2, 3])

def test_drop_index():
    df = FastDataFrame({'a': int})
    for i in range(5): df.append_row(a=i)
    df.reset_index()
    r = df.drop_index([0, 2])
    check('drop_index shape', r.shape[0] == 3)
    check('drop_index vals', [r['a'][i] for i in range(3)] == [1, 3, 4])

def test_sort_index():
    df = FastDataFrame({'a': int})
    for v in [50, 10, 30, 20, 40]: df.append_row(a=v)
    df.reset_index()
    df.sort_index(inplace=True)
    check('sort_index', [df['a'][i] for i in range(5)] == [50, 10, 30, 20, 40])

def test_argsort_by():
    df = FastDataFrame({'a': int})
    for v in [30, 10, 20]: df.append_row(a=v)
    idx = df.argsort_by('a')
    check('argsort_by', idx == [1, 2, 0])
    idx_desc = df.argsort_by('a', reverse=True)
    check('argsort_by desc', idx_desc == [0, 2, 1])

def test_nunique():
    df = FastDataFrame({'a': str})
    for v in ['x', 'y', 'x', 'z', 'y', 'x']: df.append_row(a=v)
    check('nunique', df.nunique('a') == 3)


# =====================================================================
#  Display: to_string, info, __str__
# =====================================================================
def test_display():
    df = make_complex_df().head(3)
    s = df.to_string()
    check('to_string has id', 'id' in s)
    check('to_string has Alice', 'Alice' in s)

    s2 = str(df)
    check('str', 'id' in s2)

    info = df.info()
    check('info returns', isinstance(info, str))


# =====================================================================
#  Run all
# =====================================================================
# =====================================================================
#  New API tests: to_dict, to_json, abs, round, all/any, pct_change
# =====================================================================
def test_to_dict():
    df = FastDataFrame({'a': int, 'b': float, 'c': str})
    df.append_row(a=1, b=2.5, c='hello')
    df.append_row(a=3, b=4.0, c='world')
    d = df.to_dict()
    check('to_dict keys', set(d.keys()) == {'a', 'b', 'c'})
    check('to_dict int', d['a'] == [1, 3])
    check('to_dict float', d['b'] == [2.5, 4.0])
    check('to_dict str', d['c'] == ['hello', 'world'])

    # Empty DataFrame
    df2 = FastDataFrame({'a': int})
    check('to_dict empty', df2.to_dict() == {'a': []})

def test_to_json():
    import json
    df = FastDataFrame({'x': int, 'y': float})
    df.append_row(x=10, y=1.5)
    df.append_row(x=20, y=2.5)

    # orient='records' (default)
    j = df.to_json()
    parsed = json.loads(j)
    check('json records type', isinstance(parsed, list))
    check('json records len', len(parsed) == 2)
    check('json records vals', parsed[0]['x'] == 10 and parsed[1]['y'] == 2.5)

    # orient='columns'
    j2 = df.to_json(orient_records=False)
    parsed2 = json.loads(j2)
    check('json columns type', isinstance(parsed2, dict))
    check('json columns vals', parsed2['x'] == [10, 20])

def test_abs():
    df = FastDataFrame({'a': int, 'b': float})
    df.append_row(a=-5, b=-3.14)
    df.append_row(a=10, b=2.71)
    df.append_row(a=-7, b=0.0)
    abs_df = df.abs()
    check('abs int', [abs_df['a'][i] for i in range(abs_df['a'].length)] == [5, 10, 7])
    check('abs float', [abs_df['b'][i] for i in range(abs_df['b'].length)] == [3.14, 2.71, 0.0])

    # Empty
    df2 = FastDataFrame({'x': int})
    check('abs empty', df2.abs().to_dict() == {'x': []})

    # Mixed types (string passthrough)
    df3 = FastDataFrame({'a': str})
    df3.append_row(a='hello')
    abs3 = df3.abs()
    check('abs string passthrough', [abs3['a'][i] for i in range(abs3['a'].length)] == ['hello'])

def test_round():
    df = FastDataFrame({'a': float})
    df.append_row(a=1.23456)
    df.append_row(a=5.67890)
    df.append_row(a=-3.14159)

    r0 = df.round(0)
    check('round 0', [r0['a'][i] for i in range(r0['a'].length)] == [1.0, 6.0, -3.0])

    r2 = df.round(2)
    check('round 2', [r2['a'][i] for i in range(r2['a'].length)] == [1.23, 5.68, -3.14])

    # Int column passthrough
    df2 = FastDataFrame({'b': int})
    df2.append_row(b=42)
    r_int = df2.round(0)
    check('round int passthrough', [r_int['b'][i] for i in range(r_int['b'].length)] == [42])

def test_all_any():
    # all() - all True
    df = FastDataFrame({'x': int})
    df.append_row(x=1)
    df.append_row(x=2)
    check('all true', df.all() == True)

    # all() - one False
    df2 = FastDataFrame({'x': int})
    df2.append_row(x=1)
    df2.append_row(x=0)
    check('all false', df2.all() == False)

    # all() - empty
    df3 = FastDataFrame({'x': int})
    check('all empty', df3.all() == True)

    # any() - has True
    df4 = FastDataFrame({'x': int})
    df4.append_row(x=0)
    df4.append_row(x=5)
    check('any true', df4.any() == True)

    # any() - all False
    df5 = FastDataFrame({'x': int})
    df5.append_row(x=0)
    df5.append_row(x=0)
    check('any false', df5.any() == False)

    # any() - empty
    df6 = FastDataFrame({'x': int})
    check('any empty', df6.any() == False)

    # bool column
    df7 = FastDataFrame({'x': bool})
    df7.append_row(x=True)
    df7.append_row(x=True)
    check('all bool true', df7.all() == True)
    df7.append_row(x=False)
    check('all bool mixed', df7.all() == False)

    # multi-column all/any
    df8 = FastDataFrame({'a': int, 'b': int})
    df8.append_row(a=1, b=2)
    df8.append_row(a=3, b=4)
    check('multi all true', df8.all() == True)
    df8.append_row(a=0, b=1)
    check('multi all false', df8.all() == False)

def test_pct_change():
    df = FastDataFrame({'x': int})
    df.append_row(x=100)
    df.append_row(x=110)
    df.append_row(x=99)
    pct = df.pct_change()
    import math
    check('pct_change nan', math.isnan(pct['x'][0]))
    check('pct_change 10%', abs(pct['x'][1] - 0.1) < 1e-10)
    check('pct_change -10%', abs(pct['x'][2] - (-0.1)) < 1e-10)

    # periods=2
    pct2 = df.pct_change(periods=2)
    check('pct_change periods=2 nan1', math.isnan(pct2['x'][0]))
    check('pct_change periods=2 nan2', math.isnan(pct2['x'][1]))
    check('pct_change periods=2 val', abs(pct2['x'][2] - (-0.01)) < 1e-10)

    # float column
    df2 = FastDataFrame({'y': float})
    df2.append_row(y=10.0)
    df2.append_row(y=20.0)
    pct_f = df2.pct_change()
    check('pct_change float nan', math.isnan(pct_f['y'][0]))
    check('pct_change float 100%', abs(pct_f['y'][1] - 1.0) < 1e-10)

    # zero previous value -> nan
    df3 = FastDataFrame({'z': int})
    df3.append_row(z=0)
    df3.append_row(z=5)
    pct_z = df3.pct_change()
    check('pct_change zero prev', math.isnan(pct_z['z'][1]))


def test_replace():
    df = FastDataFrame({'a': int, 'b': float})
    df.append_row(a=1, b=1.0)
    df.append_row(a=2, b=2.0)
    df.append_row(a=3, b=3.0)
    r = df.replace(2, 99)
    check('replace a', r['a'][1] == 99)
    check('replace b', r['b'][1] == 99.0)
    check('replace other unchanged', r['a'][0] == 1)

    df2 = FastDataFrame({'s': str})
    df2.append_row(s='hello')
    df2.append_row(s='world')
    r2 = df2.replace('world', 'python')
    check('replace str', r2['s'][1] == 'python')


def test_ffill_bfill():
    import math
    df = FastDataFrame({'a': float})
    df.append_row(a=1.0)
    df.append_row(a=float('nan'))
    df.append_row(a=3.0)
    df.append_row(a=float('nan'))

    ff = df.ffill()
    check('ffill first', ff['a'][0] == 1.0)
    check('ffill nan->prev', ff['a'][1] == 1.0)
    check('ffill keeps valid', ff['a'][2] == 3.0)
    check('ffill nan->prev2', ff['a'][3] == 3.0)

    bf = df.bfill()
    check('bfill nan->next', bf['a'][1] == 3.0)
    check('bfill last stays nan', math.isnan(bf['a'][3]))


def test_eq():
    df1 = FastDataFrame({'a': int, 'b': float})
    df1.append_row(a=1, b=1.0)
    df1.append_row(a=2, b=2.0)

    df2 = FastDataFrame({'a': int, 'b': float})
    df2.append_row(a=1, b=1.0)
    df2.append_row(a=2, b=2.0)

    check('eq same', df1 == df2)

    df3 = FastDataFrame({'a': int, 'b': float})
    df3.append_row(a=1, b=1.0)
    df3.append_row(a=2, b=99.0)
    check('eq diff val', not (df1 == df3))

    df4 = FastDataFrame({'a': int})
    df4.append_row(a=1)
    check('eq diff cols', not (df1 == df4))

    df5 = FastDataFrame({'a': int, 'b': float})
    df5.append_row(a=1, b=1.0)
    check('eq diff len', not (df1 == df5))


def test_pivot():
    df = FastDataFrame({'idx': str, 'col': str, 'val': float})
    df.append_row(idx='r1', col='A', val=1.0)
    df.append_row(idx='r1', col='B', val=2.0)
    df.append_row(idx='r2', col='A', val=3.0)
    df.append_row(idx='r2', col='B', val=4.0)

    p = df.pivot(index='idx', columns='col', values='val')
    check('pivot shape', p.shape == (2, 3))
    check('pivot r1_A', p['A'][0] == 1.0)
    check('pivot r2_B', p['B'][1] == 4.0)


def test_melt():
    df = FastDataFrame({'id': str, 'x': float, 'y': float})
    df.append_row(id='r1', x=1.0, y=10.0)
    df.append_row(id='r2', x=2.0, y=20.0)

    m = df.melt(id_vars=['id'], value_vars=['x', 'y'])
    check('melt shape', m.shape == (4, 3))
    check('melt var', m['variable'][0] == 'x')
    check('melt val', m['value'][0] == 1.0)
    check('melt id', m['id'][0] == 'r1')


def test_expanding_ewm():
    df = FastDataFrame({'a': float})
    df.append_row(a=1.0)
    df.append_row(a=2.0)
    df.append_row(a=3.0)

    ex = df.expanding_sum('a')
    check('expanding_sum [0]', ex[0] == 1.0)
    check('expanding_sum [1]', ex[1] == 3.0)
    check('expanding_sum [2]', ex[2] == 6.0)

    ew = df.ewm_mean('a', span=3)
    check('ewm_mean first', ew[0] == 0.5)
    check('ewm_mean is float', isinstance(ew[1], float))


def test_read_json():
    import json, os, tempfile
    from fluidframe.core.fastdataframe import read_json
    data = [
        {'id': 1, 'val': 10.0, 'flag': True},
        {'id': 2, 'val': 20.0, 'flag': False},
    ]
    path = os.path.join(tempfile.gettempdir(), '_ff_test.json')
    with open(path, 'w') as f:
        json.dump(data, f)
    df = read_json(path, schema={'id': int, 'val': float, 'flag': bool})
    os.unlink(path)
    check('read_json shape', df.shape == (2, 3))
    check('read_json id', df['id'][0] == 1)
    check('read_json val', df['val'][1] == 20.0)
    check('read_json flag', df['flag'][0] == 1)


def test_merge_multicol():
    df1 = FastDataFrame({'a': int, 'b': int, 'v1': float})
    df1.append_row(a=1, b=10, v1=1.0)
    df1.append_row(a=2, b=20, v1=2.0)

    df2 = FastDataFrame({'a': int, 'b': int, 'v2': float})
    df2.append_row(a=1, b=10, v2=100.0)
    df2.append_row(a=2, b=20, v2=200.0)

    m = df1.merge(df2, on=['a', 'b'])
    check('merge_multicol shape', m.shape == (2, 4))
    check('merge_multicol v2', m['v2'][0] == 100.0)


def test_merge_col_conflict():
    df1 = FastDataFrame({'id': int, 'val': float})
    df1.append_row(id=1, val=1.0)

    df2 = FastDataFrame({'id': int, 'val': float})
    df2.append_row(id=1, val=99.0)

    m = df1.merge(df2, on='id')
    check('merge_conflict has val_x', 'val_x' in m.columns)
    check('merge_conflict has val_y', 'val_y' in m.columns)
    check('merge_conflict val_x', m['val_x'][0] == 1.0)
    check('merge_conflict val_y', m['val_y'][0] == 99.0)


def run_all():
    print('=== FastDataFrame Tests ===')

    # Construction
    test_init()
    test_init_complex()
    test_from_dict()
    test_from_list()
    test_from_arrays()
    test_factory_methods()

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

    # Column mutations
    test_rename()
    test_insert_column()
    test_duplicate_column()
    test_assign()
    test_setitem()

    # Slicing
    test_sample()
    test_nsmallest_nlargest()
    test_duplicated()
    test_drop_duplicates()
    test_select_dtypes()
    test_reindex()

    # Aggregation
    test_aggregation()
    test_aggregation_complex()

    # Cumulative
    test_cumulative()

    # Column arithmetic
    test_col_arithmetic()

    # Statistics
    test_corr_cov()
    test_distance_functions()

    # Column ops
    test_value_counts()
    test_clip_column()
    test_where_column()
    test_mask_column()
    test_where_row()

    # Groupby
    test_groupby()
    test_groupby_multi_col()

    # Merge detailed
    test_merge_inner()
    test_merge_left()
    test_merge_right()
    test_merge_outer()

    # Concat detailed
    test_concat_multi()
    test_concat_inner()
    test_concat_outer()

    # eval / query
    test_eval()
    test_query()

    # apply / pipe
    test_apply()
    test_pipe()

    # duplicate_rows / reshape / T / memory / len
    test_duplicate_rows()
    test_reshape()
    test_transpose()
    test_memory_usage()
    test_len()

    # Insert row / drop_index / sort_index / argsort / nunique
    test_insert_row()
    test_drop_index()
    test_sort_index()
    test_argsort_by()
    test_nunique()

    # Display
    test_display()

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

    # New API methods
    print('\n--- new API tests ---')
    test_to_dict()
    test_to_json()
    test_abs()
    test_round()
    test_all_any()
    test_pct_change()
    test_replace()
    test_ffill_bfill()
    test_eq()
    test_pivot()
    test_melt()
    test_expanding_ewm()
    test_read_json()
    test_merge_multicol()
    test_merge_col_conflict()

    print(f'\nResults: {passed} passed, {failed} failed')
    if failed: sys.exit(1)

if __name__ == '__main__':
    run_all()
