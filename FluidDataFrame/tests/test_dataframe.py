import math
import os
import random
import sys
import tempfile
import time
sys.path.insert(0, '..')
from fluidframe.core.dataframe import DataFrame, read_json, from_pickle, to_numeric, to_date, to_time, to_datetime
from fluidframe.core.intcolumn import IntColumn
from fluidframe.core.series import Series
from fluidframe.core.doublecolumn import DoubleColumn
from fluidframe.core.asciicolumn import AsciiColumn
from fluidframe.core.ucs2column import UCS2Column
from fluidframe.core.datecolumn import DateColumn as DateColumn
from fluidframe.core.timecolumn import TimeColumn
from fluidframe.core.datetimecolumn import DateTimeColumn
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
    Returns a DataFrame with 10 rows and 5 column types:
      id:    int
      score: double
      name:  UCS2 (str)
      tag:   Char256
      dt:    Date
    """
    df = DataFrame({'id': int, 'score': float, 'name': str, 'tag': Char256, 'dt': Date})
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
    df = DataFrame()
    check('empty shape', df.shape == (0, 0))

    df = DataFrame({'a': int})
    check('schema int', isinstance(df._columns['a'], IntColumn))

    df = DataFrame({'a': float})
    check('schema float', isinstance(df._columns['a'], DoubleColumn))

    df = DataFrame({'a': str})
    check('schema str', isinstance(df._columns['a'], UCS2Column))

    df = DataFrame({'a': Char256})
    check('schema char256', isinstance(df._columns['a'], AsciiColumn))

    df = DataFrame({'a': Date})
    check('schema timestamp', isinstance(df._columns['a'], DateColumn))

    df = DataFrame({'id': int, 'name': str, 'score': float})
    check('schema multiple', df.shape == (0, 3))

    check('schema unsupported', expect_exception(TypeError, DataFrame, {'a': list}))

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
#  Reserve capacity — fixed-capacity fast insertion (add_row)
# =====================================================================
_RESERVE_ROWS = [
    (1,  12, 53, 45, 67, 95.0, 'Alice',   'senior', 20230115, 1),
    (2,  12, 25, 45, 67, 92.3, 'Bob',     'mid',    20230620, 2),
    (3,  12, 53, 45, 67, 90.1, 'Charlie', 'senior', 20221201, 3),
    (4,  12, 25, 45, 67, 88.5, 'Diana',   'low',    20240229, 4),
    (5,  12, 53, 45, 67, 87.0, 'Eve',     'senior', 20230310, 5),
    (6,  12, 25, 45, 67, 85.5, 'Frank',   'mid',    20220615, 1),
    (7,  12, 53, 45, 67, 84.2, 'Grace',   'low',    20231225, 2),
    (8,  12, 25, 45, 67, 83.1, 'Helen',   'senior', 20240101, 3),
    (9,  12, 53, 45, 67, 82.0, 'Ivan',    'mid',    20230115, 4),
    (10, 12, 25, 45, 67, 81.5, 'Judy',    'low',    20230620, 5),
]

def _make_reserved_df():
    df = DataFrame({'id': int, 'x0': int, 'y0': int, 'x1': int, 'y1': int,
                        'score': float, 'name': str, 'tag': Char256, 'dt': Date,
                        'flags': int})
    df.reserve_with_capacity(len(_RESERVE_ROWS) + 16)
    return df

def test_reserve_with_capacity():
    # kwargs push path
    df = _make_reserved_df()
    for r in _RESERVE_ROWS:
        df.add_row(id=r[0], x0=r[1], y0=r[2], x1=r[3], y1=r[4], score=r[5],
                    name=r[6], tag=r[7], dt=r[8], flags=r[9])
    check('reserve shape', df.shape == (10, 10))
    check('reserve length', df.df_length == 10)
    check('reserve id[0]', df['id'][0] == 1)
    check('reserve id[9]', df['id'][9] == 10)
    check('reserve y0 alt', [df['y0'][i] for i in [0, 1, 2]] == [53, 25, 53])
    check('reserve score', abs(df['score'][1] - 92.3) < 1e-9)
    check('reserve name', df['name'][0] == 'Alice')
    check('reserve tag', df['tag'][1] == 'mid')
    check('reserve dt conv', df['dt'][3] == 20240229)
    check('reserve flags', df['flags'][9] == 5)
    for col in ['id', 'x0', 'y0', 'x1', 'y1', 'flags', 'score', 'name', 'tag', 'dt']:
        check(f'reserve col len {col}', df[col].length == 10)
    check('reserve blocks', len(df._2d_blocks) == 3)

    # positional push path (same data)
    df2 = _make_reserved_df()
    for r in _RESERVE_ROWS:
        df2.add_row(*r)
    check('reserve pos shape', df2.shape == (10, 10))
    check('reserve pos id', df2['id'][9] == 10)
    check('reserve pos dt', df2['dt'][4] == 20230310)

    # 已预留容量后不可再次预留
    check('reserve twice error', expect_exception(RuntimeError, df.reserve_with_capacity, 100))
    check('reserve zero error', expect_exception(ValueError, DataFrame({'a': int}).reserve_with_capacity, 0))
    df_nonempty = DataFrame({'a': int})
    df_nonempty.append_row(a=1)
    check('reserve non-empty error',
          expect_exception(RuntimeError, df_nonempty.reserve_with_capacity, 10))

    # 缺失列必须报错（防止列长度错位/脏数据）
    check('reserve missing key error', expect_exception(ValueError, df.add_row, id=99))

    # 达到固定容量后必须拒绝写入
    df3 = DataFrame({'id': int, 'score': float, 'name': str, 'tag': Char256, 'dt': Date})
    df3.reserve_with_capacity(len(_RESERVE_ROWS))
    for r in _RESERVE_ROWS:
        df3.add_row(id=r[0], score=r[5], name=r[6], tag=r[7], dt=r[8])
    check('reserve cap full', df3.df_length == 10)
    check('reserve overflow error', expect_exception(BufferError, df3.add_row, id=11, score=1.0, name='X', tag='x', dt=20250101))

    # 预留后走普通 append_row（定宽列是 view）应拒绝而不是写坏内存
    check('reserve append_row blocked', expect_exception(RuntimeError, df.append_row,
                                                         id=11, x0=99, y0=99, x1=99, y1=99,
                                                         score=1.0, name='Z', tag='z', dt=20250101, flags=9))

def test_reserve_filters():
    df = _make_reserved_df()
    for r in _RESERVE_ROWS:
        df.add_row(id=r[0], x0=r[1], y0=r[2], x1=r[3], y1=r[4], score=r[5],
                    name=r[6], tag=r[7], dt=r[8], flags=r[9])

    mask = df['id'] > 5
    r = df.loc[mask]
    check('reserve filter int shape', r.shape == (5, 10))
    check('reserve filter int ids', [r['id'][i] for i in range(5)] == [6, 7, 8, 9, 10])

    mask = df['score'] >= 90.0
    r = df.loc[mask]
    check('reserve filter float shape', r.shape == (3, 10))
    check('reserve filter float names', set(r['name'][i] for i in range(3)) == {'Alice', 'Bob', 'Charlie'})

    mask = df['dt'] == 20230115
    r = df.loc[mask]
    check('reserve filter dt shape', r.shape == (2, 10))
    check('reserve filter dt ids', sorted(r['id'][i] for i in range(2)) == [1, 9])

    mask = (df['id'] >= 3) & (df['score'] <= 87.0) & (df['dt'] < 20240101)
    r = df.loc[mask]
    check('reserve filter 3-type shape', r.shape == (5, 10))
    check('reserve filter 3-type ids', sorted(r['id'][i] for i in range(5)) == [5, 6, 7, 9, 10])

    mask = (df['tag'].len() <= 3) & (df['flags'] >= 3)
    r = df.loc[mask]
    check('reserve filter char256 shape', r.shape == (3, 10))
    check('reserve filter char256 ids', sorted(r['id'][i] for i in range(3)) == [4, 9, 10])

    mask = df['name'].len() <= 4
    r = df.loc[mask]
    check('reserve filter ucs2 ids', sorted(r['id'][i] for i in range(r.shape[0])) == [2, 5, 9, 10])

    mask = ((df['id'] >= 3) & (df['score'] < 85.0) & (df['tag'].startswith('se'))
            & (df['dt'] >= 20230115))
    r = df.loc[mask]
    check('reserve filter 4-type shape', r.shape == (1, 10))
    check('reserve filter 4-type id', r['id'][0] == 8)


def test_add_row_failure_rollback():
    # 中途失败（flags 位于 name/tag 之后）：UCS2/CHAR256 已写入但未提交 → 必须整体回滚
    df = _make_reserved_df()
    for r in _RESERVE_ROWS:
        df.add_row(id=r[0], x0=r[1], y0=r[2], x1=r[3], y1=r[4], score=r[5],
                   name=r[6], tag=r[7], dt=r[8], flags=r[9])

    check('rollback raises', expect_exception(TypeError, df.add_row,
                                              id=11, x0=12, y0=25, x1=45, y1=67,
                                              score=99.0, name='Zulu', tag='low',
                                              dt=20250101, flags='boom'))
    check('rollback keeps df_length', df.df_length == 10)
    for col in ['id', 'x0', 'y0', 'x1', 'y1', 'flags', 'score', 'name', 'tag', 'dt']:
        check(f'rollback col len {col}', df[col].length == 10)
    check('rollback name intact', df['name'][9] == 'Judy')
    check('rollback tag intact', df['tag'][9] == 'low')

    # 回滚后写入合法行仍成功，且各列长度一致
    df.add_row(id=11, x0=12, y0=25, x1=45, y1=67, score=99.0,
               name='Zulu', tag='low', dt=20250101, flags=9)
    check('rollback re-add ok', df.df_length == 11)
    check('rollback re-add name', df['name'][10] == 'Zulu')
    check('rollback re-add tag', df['tag'][10] == 'low')
    check('rollback all lens equal', all(df[c].length == 11 for c in df.col_names))


# =====================================================================
#  Append & access
# =====================================================================
def test_append():
    df = DataFrame({'id': int, 'name': str})
    df.append_row(id=1, name='Alice')
    df.append_row(id=2, name='Bob')
    check('append shape', df.shape == (2, 2))
    check('append id[0]', df['id'][0] == 1)
    check('append name[1]', df['name'][1] == 'Bob')

    df = DataFrame({'i': int, 'f': float, 's': str, 'c': Char256, 't': Date})
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
    df = DataFrame({'a': int}); df.append_row(a=42)
    check('getitem', isinstance(df['a'], Series))
    check('getitem val', df['a'][0] == 42)
    check('getitem missing', expect_exception(KeyError, df.__getitem__, 'b'))

    df = DataFrame({'a': int}); df.append_row(a=42)
    check('getattr', df.a[0] == 42)
    check('getattr missing', expect_exception(AttributeError, getattr, df, 'nonexistent'))

    df = DataFrame({'a': int})
    check('repr', 'shape: (0, 1)' in repr(df))


# =====================================================================
#  DataFrame ↔ Series 衔接契约：
#  列级方法返回 Series（而不是原始 BaseColumn），Series 可回流成 DataFrame
# =====================================================================
def test_series_return_contract():
    df = DataFrame({'a': int, 'b': int, 's': str})
    for a, b, s in [(1, 10, 'x'), (3, 20, 'y'), (3, 30, 'x'), (4, 40, 'z')]:
        df.append_row(a=a, b=b, s=s)

    # 单列结果 → Series
    for name, res in [
        ('isin',       df.isin('a', {1, 3})),
        ('between',    df.between('a', 2, 4)),
        ('diff',       df.diff('a', 1)),
        ('duplicated', df.duplicated()),
        ('rank',       df.rank('a')),
        ('cut',        df.cut('a', [0, 2, 5])),
        ('exp_sum',    df.expanding_sum('a')),
        ('where_col',  df.where_column('a', 'b', 0)),
        ('mask_col',   df.mask_column('a', 'b', 0)),
        ('add',        df.add(('a', 'b'))),
        ('mul',        df.mul(('a', 'b'))),
        ('roll_mean',  df.rolling(2, 'a').mean()),
        ('eval_col',   df.eval('a * 2')),
    ]:
        check(f'{name} is Series', isinstance(res, Series))

    # 掩码/布尔结果同样为 Series
    check('isin vals', df.isin('a', {1, 3}).tolist() == [1, 1, 1, 0])
    check('between vals', df.between('a', 2, 4).tolist() == [0, 1, 1, 1])
    check('rank vals', df.rank('a').tolist() == [1.0, 2.5, 2.5, 4.0])

    # 累积聚合返回 DataFrame（pandas 语义）
    cs = df.cumsum()
    check('cumsum is DataFrame', isinstance(cs, DataFrame))
    check('cumsum cols', set(cs._columns.keys()) == {'a', 'b'})
    check('cumsum a vals', cs['a'].tolist() == [1, 4, 7, 11])

    # Series.isna / notna / isin / between → Series
    check('s.isna Series', isinstance(df['a'].isna(), Series))
    check('s.isna vals', df['a'].isna().tolist() == [0, 0, 0, 0])
    check('s.isin Series', isinstance(df['a'].isin({1, 3}), Series))
    check('s.between Series', isinstance(df['a'].between(2, 4), Series))

    # iloc 两轴：DataFrame → Series
    check('iloc col Series', isinstance(df.iloc[:, 'a'], Series))
    check('iloc col int Series', isinstance(df.iloc[:, 0], Series))
    check('iloc slice col', df.iloc[1:3, 'a'].tolist() == [3, 3])
    check('iloc neg col', df.iloc[-1, 'a'].tolist() == [4])
    check('iloc oob col', expect_exception(IndexError, df.iloc.__getitem__, (slice(None), 9)))

    # Series → DataFrame 转换
    fr = df['s'].to_frame()
    check('to_frame type', isinstance(fr, DataFrame))
    check('to_frame shape', fr.shape == (4, 1))
    check('to_frame val', fr['s'].tolist() == ['x', 'y', 'x', 'z'])
    fr2 = df['a'].to_frame('renamed')
    check('to_frame renamed', list(fr2._columns.keys()) == ['renamed'])

    # DataFrame 构造器直接接收 Series / list
    dfc = DataFrame({'p': df['a'], 'q': df['b']})
    check('from Series cols', set(dfc._columns.keys()) == {'p', 'q'})
    check('from Series len', dfc.shape == (4, 2))
    check('from Series val', dfc['p'].tolist() == [1, 3, 3, 4])
    dfl = DataFrame({'x': [1, 2, 3], 'y': [10, 20, 30]})
    check('from list cols', dfl.shape == (3, 2))
    check('from list val', dfl['y'].tolist() == [10, 20, 30])

    # factorize 修复：不再 AttributeError
    codes, uniques = df.factorize('s')
    check('factorize codes', codes == [0, 1, 0, 2])
    check('factorize uniques', uniques == ['x', 'y', 'z'])

    # 聚合返回带列名索引的 Series（pandas 语义）；非数值列不参与
    agg = df.sum()
    check('agg is Series', isinstance(agg, Series))
    check('agg label a', agg['a'] == 11)
    check('agg label b', agg['b'] == 100)
    check('agg no str', len(agg) == 2)
    ind_names = list(agg.index.values())
    check('agg labels', ind_names == ['a', 'b'])

    # mode 返回 DataFrame（pandas df.mode() 语义），每列独立模式、不足补 NaN
    md = df.mode()
    check('mode is DataFrame', isinstance(md, DataFrame))
    check('mode a', md['a'][0] == 3)
    check('mode cols', list(md._columns.keys()) == ['a', 'b'])


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
    df = DataFrame({'a': int})
    check('pop_front empty', expect_exception(IndexError, df.pop_front))
    check('pop_back empty', expect_exception(IndexError, df.pop_back))


# =====================================================================
#  Drop & drop_raw
# =====================================================================
def test_drop():
    df = make_complex_df()
    df2 = df.drop('tag')
    check('drop original unchanged', df.shape == (10, 5))
    check('drop new shape', df2.shape == (10, 4))
    check('drop not in cols', 'tag' not in df2.columns)

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
#  __getitem__ list, __iter__, __contains__
# =====================================================================
def test_getitem_list():
    df = make_complex_df()
    sub = df[['id', 'name']]
    check('getitem list shape', sub.shape == (10, 2))
    check('getitem list cols', sub.col_names == ['id', 'name'])
    check('getitem list data id', sub['id'][0] == 1)
    check('getitem list data name', sub['name'][0] == 'Alice')
    check('getitem list orig unchanged', df.shape == (10, 5))

def test_getitem_list_3cols():
    df = make_complex_df()
    sub = df[['score', 'tag', 'dt']]
    check('getitem 3cols shape', sub.shape == (10, 3))
    check('getitem 3cols order', sub.col_names == ['score', 'tag', 'dt'])

def test_getitem_list_error():
    df = make_complex_df()
    check('getitem list missing col', expect_exception(KeyError, df.__getitem__, ['id', 'zzz']))

def test_iter():
    df = make_complex_df()
    names = list(df)
    check('iter', names == ['id', 'score', 'name', 'tag', 'dt'])

def test_contains():
    df = make_complex_df()
    check('contains id', 'id' in df)
    check('contains name', 'name' in df)
    check('contains missing', 'zzz' not in df)


# =====================================================================
#  Sort
# =====================================================================
def test_sort_int():
    df = make_complex_df()
    s = df.sort_values('id', ascending=False)
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
    s = df.sort_values('id', ascending=False)
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
    df = DataFrame({'a': int})
    for v in [1, 3, 6, 10]: df.append_row(a=v)
    res = df.diff('a')
    check('diff int [0]', res[0] == 0)
    check('diff int [1]', res[1] == 2)
    check('diff int [2]', res[2] == 3)
    check('diff int [3]', res[3] == 4)

def test_diff_float():
    df = DataFrame({'a': float})
    df.append_row(a=1.0); df.append_row(a=3.0); df.append_row(a=6.0)
    res = df.diff('a')
    check('diff float [0] nan', math.isnan(res[0]))
    check('diff float [1]', res[1] == 2.0)
    check('diff float [2]', res[2] == 3.0)


# =====================================================================
#  Null handling
# =====================================================================
def test_null_handling():
    df = DataFrame({'a': float, 'b': int})
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

def test_2d_count_describe_info():
    """2D 列：count/describe/info 以逻辑行数为准（physical length = rows*cols）"""
    df = DataFrame()
    df.df_length = 4
    df._columns['a'] = IntColumn.from_2d([1, 2, 3, 4, 10, 20, 30, 40], 4, 2)
    df._columns['d'] = DoubleColumn.from_2d([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0], 4, 2)

    cnt = df.count()
    check('2d count logical rows', cnt['a'] == 4 and cnt['d'] == 4)
    check('2d count matches df_length', cnt['a'] == df.df_length)

    stats = df.describe()
    check('2d describe count', stats['a']['count'] == 4 and stats['d']['count'] == 4)
    # 2D 列统计沿用逐子列语义（与 mean_withcol 一致）：a 子列0=[1..4]、子列1=[10..40]
    check('2d describe mean', stats['a']['mean'] == [2.5, 25.0])
    check('2d describe min', stats['a']['min'] == [1, 10])
    check('2d describe max', stats['a']['max'] == [4, 40])

    info = df.info()
    check('2d info str', isinstance(info, str))
    check('2d info non-null rows', f' {4:<10}' in info)
    check('2d info no physical cells', '8' not in info.split('Data columns')[1].split('dtypes')[0])


def test_2d_agg_withcol():
    """_withcol 聚合族：2D 列逐子列独立聚合（不跨子列混合），1D 列与普通聚合一致"""
    df = DataFrame()
    df.df_length = 4
    df._columns['a2d'] = IntColumn.from_2d([1, 2, 3, 4, 10, 20, 30, 40], 4, 2)
    df._columns['d2d'] = DoubleColumn.from_2d([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0], 4, 2)
    x = DoubleColumn()
    for v in [1.0, 2.0, 3.0, 4.0]:
        x.append(v)
    df._columns['x'] = x

    m = df.mean_withcol()
    check('withcol mean 2d', m['a2d'] == [2.5, 25.0] and m['d2d'] == [2.5, 25.0])
    check('withcol mean 1d', m['x'] == 2.5)
    check('withcol sum', df.sum_withcol()['a2d'] == [10.0, 100.0])
    check('withcol min', df.min_withcol()['a2d'] == [1.0, 10.0])
    check('withcol max', df.max_withcol()['a2d'] == [4.0, 40.0])
    check('withcol median', df.median_withcol()['a2d'] == [2.5, 25.0])
    v = df.var_withcol()['a2d']
    check('withcol var', abs(v[0] - 5.0 / 3.0) < 1e-9 and abs(v[1] - 500.0 / 3.0) < 1e-9)
    s = df.std_withcol()['a2d']
    check('withcol std', abs(s[0] - math.sqrt(5.0 / 3.0)) < 1e-9 and abs(s[1] - math.sqrt(500.0 / 3.0)) < 1e-9)
    check('withcol quantile p25', df.quantile_withcol(0.25)['a2d'] == [1.75, 17.5])
    check('withcol quantile p75', df.quantile_withcol(0.75)['a2d'] == [3.25, 32.5])
    check('withcol mode', df.mode_withcol()['a2d'] == [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])
    check('withcol col_names', set(df.mean_withcol(['a2d']).keys()) == {'a2d'})

    ds = df.describe_withcol()
    check('withcol describe 2d count', ds['a2d'][0]['count'] == 4 and ds['a2d'][1]['count'] == 4)
    check('withcol describe 2d mean', ds['a2d'][0]['mean'] == 2.5 and ds['a2d'][1]['mean'] == 25.0)
    check('withcol describe 2d minmax', ds['a2d'][0]['min'] == 1.0 and ds['a2d'][0]['max'] == 4.0)
    check('withcol describe 2d med', ds['d2d'][0]['50%'] == 2.5 and ds['d2d'][1]['50%'] == 25.0)
    check('withcol describe 2d std', abs(ds['a2d'][0]['std'] - math.sqrt(5.0 / 3.0)) < 1e-9)
    check('withcol describe 1d', ds['x']['mean'] == 2.5 and ds['x']['count'] == 4)


# =====================================================================
#  CSV / Pickle / TSV
# =====================================================================
def test_csv():
    df = make_complex_df()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        fname = f.name
    try:
        df.to_csv(fname)
        from fluidframe.core.dataframe import read_csv
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
        from fluidframe.core.dataframe import from_pickle
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

def test_pickle_2d_roundtrip():
    """2D 列 to_pickle/from_pickle 往返保持 is_2d/rows/cols"""
    df = DataFrame()
    df.df_length = 4
    df._columns['a'] = IntColumn.from_2d([1, 2, 3, 4, 10, 20, 30, 40], 4, 2)
    df._columns['d'] = DoubleColumn.from_2d([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0], 4, 2)
    date_codes = IntColumn.from_2d([
        20230101, 20230102, 20230103, 20230104,
        20240101, 20240102, 20240103, 20240104], 4, 2)
    df._columns['dd'] = to_date(date_codes)
    df._columns['dt'] = to_datetime(date_codes)
    df._columns['s'] = UCS2Column()
    for v in ['x', 'y', 'z', 'w']:
        df._columns['s'].append(v)
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        fname = f.name
    try:
        df.to_pickle(fname)
        df2 = from_pickle(fname)
        check('pkl2d shape', df2.shape == (4, 5))
        check('pkl2d a 2d', df2['a'].is_2d and df2['a'].cols == 2 and df2['a'].rows == 4)
        check('pkl2d a flat', [df2['a'][i] for i in range(8)] == [1, 2, 3, 4, 10, 20, 30, 40])
        check('pkl2d d 2d', df2['d'].is_2d and df2['d'].cols == 2 and df2['d'].rows == 4)
        check('pkl2d d flat', [df2['d'][i] for i in range(8)] == [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
        check('pkl2d dd 2d', df2['dd'].is_2d and df2['dd'].cols == 2 and df2['dd'].rows == 4)
        check('pkl2d dd flat', [df2['dd'][i] for i in range(8)][:4] == [20230101, 20230102, 20230103, 20230104])
        check('pkl2d dt 2d', df2['dt'].is_2d and df2['dt'].cols == 2 and df2['dt'].rows == 4)
        check('pkl2d s 1d', [df2['s'][i] for i in range(4)] == ['x', 'y', 'z', 'w'])
    finally:
        os.unlink(fname)

def test_standalone_convert_2d():
    """standalone to_numeric/to_date/to_time/to_datetime 对 2D 输入保持同维度"""
    a = IntColumn.from_2d([1, 2, 3, 4, 10, 20, 30, 40], 4, 2)
    r = to_numeric(a)
    check('conv to_numeric 2d', r.is_2d and r.cols == 2 and r.rows == 4)
    check('conv to_numeric flat', [r[i] for i in range(8)] == [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])

    d = to_date(a)
    check('conv to_date 2d', d.is_2d and d.cols == 2 and d.rows == 4)
    t = to_time(a)
    check('conv to_time 2d', t.is_2d and t.cols == 2 and t.rows == 4)
    dt = to_datetime(a)
    check('conv to_datetime 2d', dt.is_2d and dt.cols == 2 and dt.rows == 4)

    one = IntColumn()
    for v in [1, 2, 3]:
        one.append(v)
    r1 = to_numeric(one)
    check('conv 1d unchanged', not r1.is_2d and [r1[i] for i in range(3)] == [1.0, 2.0, 3.0])

def test_read_csv_auto():
    from fluidframe.core.dataframe import read_csv
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
    from fluidframe.core.dataframe import concat_outer, concat_inner
    check('concat empty', concat_outer([]).shape == (0, 0))

    df1 = make_complex_df().head(5)
    df2 = make_complex_df().tail(5)
    r = concat_outer([df1, df2])
    check('concat two shape', r.shape == (10, 6))
    check('concat two has index col', 'index' in r.columns)

def test_merge():
    df1 = DataFrame({'id': int, 'val1': float})
    df1.append_row(id=1, val1=10.0); df1.append_row(id=2, val1=20.0)
    df2 = DataFrame({'id': int, 'val2': float})
    df2.append_row(id=2, val2=200.0); df2.append_row(id=3, val2=300.0)
    check('merge inner', df1.merge(df2, on='id', how='inner').shape == (1, 3))
    check('merge left', df1.merge(df2, on='id', how='left').shape == (2, 3))
    check('merge outer', df1.merge(df2, on='id', how='outer').shape == (3, 3))

def test_2d_concat_df():
    df1 = DataFrame()
    df1.df_length = 2
    df1._columns['a'] = IntColumn.from_2d([1, 2, 3, 4], 2, 2)
    df2 = DataFrame()
    df2.df_length = 1
    df2._columns['a'] = IntColumn.from_2d([5, 6], 1, 2)
    r = df1.concat(df2)
    a = r._columns['a']
    check('2d concat df rows', r.df_length == 3)
    check('2d concat df index', r.index.length == 3)
    check('2d concat df cols', a.rows == 3 and a.cols == 2)
    check('2d concat df flat', [a[i] for i in range(a.length)] == [1, 2, 5, 3, 4, 6])

def test_2d_merge_df():
    df1 = DataFrame()
    df1.df_length = 3
    df1._columns['k'] = IntColumn.from_2d([1, 2, 7, 5, 6, 8], 3, 2)
    df2 = DataFrame()
    df2.df_length = 2
    df2._columns['k'] = IntColumn.from_2d([2, 9, 6, 0], 2, 2)
    r = df1.merge(df2, on='k', how='inner')
    k = r._columns['k']
    check('2d merge inner rows', r.df_length == 1)
    check('2d merge k 2d', k.rows == 1 and k.cols == 2)
    check('2d merge k flat', [k[i] for i in range(k.length)] == [2, 6])


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
    df = DataFrame({'id': int, 'val': float, 'name': str, 'tag': Char256, 'dt': Date})
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
    df = DataFrame({'id': int, 'val': float})
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
    df = DataFrame({'id': int, 'val': float, 'tag': Char256})
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
    df = DataFrame({'a': int})
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
    df = DataFrame({'a': int, 'b': float})
    for i in range(N):
        df.append_row(a=i, b=random.random())
    t0 = time.time()
    df2 = df.copy()
    elapsed = time.time() - t0
    check('stress copy shape', df2.shape == (N, 2))
    check('stress copy speed', elapsed < 2.0)

def test_stress_drop_with_index():
    N = 5000
    df = DataFrame({'a': int})
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
    df = DataFrame({'a': int})
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
    df = DataFrame({'id': int, 'val': float, 'name': str})
    for i in range(N):
        df.append_row(id=i, val=random.random(), name=_random_string(6))
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        fname = f.name
    try:
        t0 = time.time()
        df.to_csv(fname)
        from fluidframe.core.dataframe import read_csv
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
    df = DataFrame({'id': int, 'score': float, 'tag': Char256, 'dt': Date})
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
    df = DataFrame.from_dict({'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0]}, schema={'a': int, 'b': float})
    check('from_dict shape', df.shape == (3, 2))
    check('from_dict a[1]', df['a'][1] == 2)
    check('from_dict b[2]', df['b'][2] == 6.0)

    df2 = DataFrame.from_dict({'x': [10]}, schema={'x': int}, columns=['val'])
    check('from_dict rename', 'val' in df2.columns)

    check('from_dict empty', DataFrame.from_dict({}, schema={}).shape == (0, 0))
    check('from_dict mismatch', expect_exception(ValueError, DataFrame.from_dict,
          {'a': [1, 2], 'b': [3]}, schema={'a': int, 'b': int}))

def test_from_list():
    df = DataFrame.from_list([[1, 'a'], [2, 'b'], [3, 'c']], schema={'id': int, 'ch': str})
    check('from_list shape', df.shape == (3, 2))
    check('from_list id[0]', df['id'][0] == 1)
    check('from_list ch[2]', df['ch'][2] == 'c')

    df2 = DataFrame.from_list([[10, 20]], schema={'x': int, 'y': int}, columns=['a', 'b'])
    check('from_list rename', set(df2.columns) == {'a', 'b'})

def test_from_arrays():
    df = DataFrame.from_arrays([[1, 2, 3], [10.0, 20.0, 30.0]], schema={'id': int, 'val': float})
    check('from_arrays shape', df.shape == (3, 2))
    check('from_arrays id[2]', df['id'][2] == 3)
    check('from_arrays val[0]', df['val'][0] == 10.0)

    df2 = DataFrame.from_arrays([[1]], schema={'x': int}, columns=['col'])
    check('from_arrays rename', 'col' in df2.columns)

def test_factory_methods():
    df = DataFrame.arange(7)
    check('arange shape', df.shape == (7, 1))
    check('arange vals', [df['0'][i] for i in range(7)] == [0, 1, 2, 3, 4, 5, 6])

    df = DataFrame.zeros(3, 2)
    check('zeros', all(df._columns[c][i] == 0.0 for c in df.columns for i in range(3)))

    df = DataFrame.ones(2, 3)
    check('ones', all(df._columns[c][i] == 1.0 for c in df.columns for i in range(2)))

    df = DataFrame.randn(100, 2)
    check('randn shape', df.shape == (100, 2))

    df = DataFrame.rand(50, 2)
    check('rand shape', df.shape == (50, 2))
    check('rand range', all(0 <= df._columns[c][i] < 1 for c in df.columns for i in range(50)))


# =====================================================================
#  Column mutations: rename, insert_column, duplicate_column, assign, __setitem__
# =====================================================================
def test_rename():
    df = make_complex_df().head(3)
    df2 = df.rename({'id': 'ID', 'name': 'NAME'})
    check('rename cols', set(df2.columns) == {'ID', 'score', 'NAME', 'tag', 'dt'})
    check('rename val', df2['ID'][0] == 1)
    check('rename original unchanged', 'id' in df.columns)

def test_insert_column():
    df = DataFrame({'a': int})
    df.append_row(a=1); df.append_row(a=2)
    df.insert_column('b', [10, 20], pos=0)
    check('insert pos 0', df.columns == ['b', 'a'])
    check('insert val', df['b'][1] == 20)

    df.insert_column('c', [100], pos=1)
    check('insert pos 1', df.columns == ['b', 'c', 'a'])

def test_insert_column_2d():
    col = DoubleColumn.from_2d([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0], 4, 2)
    df = DataFrame()
    df.insert_column('v2d', col)
    check('insert 2d df_length rows', df.df_length == 4)
    check('insert 2d shape', df.shape == (4, 1))
    check('insert 2d col preserved', df['v2d'].is_2d and df['v2d'].cols == 2 and df['v2d'].rows == 4)
    check('insert 2d count', df.count()['v2d'] == 4)

def test_duplicate_column():
    df = make_complex_df().head(3)
    df.duplicate_column('id', 'id_copy')
    check('dup cols', 'id_copy' in df.columns)
    check('dup val', df['id_copy'][0] == 1)
    check('dup len', df.shape == (3, 6))

    df.duplicate_column('name')
    check('dup default name', 'name_dup' in df.columns)

def test_assign():
    df = DataFrame({'a': int, 'b': int})
    df.append_row(a=1, b=2); df.append_row(a=3, b=4)
    df2 = df.assign(c=[10, 20])
    check('assign new col', 'c' in df2.columns)
    check('assign val', df2['c'][0] == 10)
    check('assign original unchanged', 'c' not in df.columns)

    df3 = df.assign(a=[100, 200])
    check('assign replace', df3['a'][0] == 100)

def test_setitem():
    df = DataFrame({'a': int})
    df.append_row(a=1)
    df['b'] = [10]
    check('setitem list', df['b'][0] == 10)
    check('setitem cols', set(df.columns) == {'a', 'b'})

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
    df = DataFrame({'a': int, 'b': str})
    df.append_row(a=1, b='x'); df.append_row(a=2, b='y'); df.append_row(a=1, b='x')
    dup = df.duplicated()
    check('dup mask', [dup[i] for i in range(3)] == [0, 0, 1])

    dup2 = df.duplicated(keep='last')
    check('dup last', [dup2[i] for i in range(3)] == [1, 0, 0])

    dup3 = df.duplicated(subset=['a'])
    check('dup subset', [dup3[i] for i in range(3)] == [0, 0, 1])

def test_drop_duplicates():
    df = DataFrame({'a': int, 'b': str})
    df.append_row(a=1, b='x'); df.append_row(a=2, b='y'); df.append_row(a=1, b='x'); df.append_row(a=3, b='z')
    r = df.drop_duplicates()
    check('drop_dup shape', r.shape == (3, 2))
    check('drop_dup vals', [r['a'][i] for i in range(3)] == [1, 2, 3])

    df2 = DataFrame({'a': int, 'b': str})
    df2.append_row(a=1, b='x'); df2.append_row(a=1, b='y'); df2.append_row(a=1, b='z')
    r2 = df2.drop_duplicates(subset=['a'], keep='last')
    check('drop_dup keep last', r2.shape == (1, 2))
    check('drop_dup last val', r2['b'][0] == 'z')

def test_select_dtypes():
    df = make_complex_df()
    int_only = df.select_dtypes(include_types=['int'])
    check('select int', set(int_only.columns) == {'id'})
    float_only = df.select_dtypes(include_types=['float'])
    check('select float', set(float_only.columns) == {'score'})
    no_int = df.select_dtypes(exclude_types=['int'])
    check('exclude int', 'id' not in no_int.columns)

def test_reindex():
    df = DataFrame({'a': int, 'b': str})
    df.append_row(a=1, b='x'); df.append_row(a=2, b='y')
    r = df.reindex([0, 1, 2, 3])
    check('reindex shape', r.shape == (4, 2))
    check('reindex val', r['a'][2] == 0)
    check('reindex val2', r['b'][3] == '')


# =====================================================================
#  类别二：行选择 _take 路径对 2D 列（整行重排，保持 cols 维）
# =====================================================================
def _make_2d_frame():
    df = DataFrame({'a': int, 's': str})
    df.append_row(a=1, s='w'); df.append_row(a=2, s='x')
    df.append_row(a=3, s='y'); df.append_row(a=4, s='z')
    df['a2d'] = IntColumn.from_2d([1, 2, 3, 4, 10, 20, 30, 40], 4, 2)
    return df

def test_take_path_2d():
    df = _make_2d_frame()
    # iloc list（走 _take）
    sub = df.iloc[[0, 2]]
    check('take 2d len', len(sub) == 2)
    check('take 2d a2d cols', sub['a2d'].cols == 2 and sub['a2d'].rows == 2)
    check('take 2d a2d flat', [sub['a2d'][i] for i in range(4)] == [1, 3, 10, 30])
    check('take 2d a 1d', [sub['a'][i] for i in range(2)] == [1, 3])
    # head / tail
    h = df.head(3)
    check('take head a2d flat', [h['a2d'][i] for i in range(6)] == [1, 2, 3, 10, 20, 30])
    t = df.tail(2)
    check('take tail a2d flat', [t['a2d'][i] for i in range(4)] == [3, 4, 30, 40])
    # sample
    s = df.sample(2, seed=7)
    check('take sample a2d cols', s['a2d'].cols == 2 and s['a2d'].rows == 2)
    check('take sample a2d flat', [s['a2d'][i] for i in range(4)] == [3, 1, 30, 10])

def test_sort_values_2d():
    df = DataFrame({'a': int})
    df.append_row(a=4); df.append_row(a=2); df.append_row(a=3); df.append_row(a=1)
    df['a2d'] = IntColumn.from_2d([4, 2, 3, 1, 40, 20, 30, 10], 4, 2)
    r = df.sort_values('a')
    check('sort 2d a2d flat', [r['a2d'][i] for i in range(8)] == [1, 2, 3, 4, 10, 20, 30, 40])
    check('sort 2d a2d cols', r['a2d'].cols == 2 and r['a2d'].rows == 4)
    # inplace
    df.sort_values('a', inplace=True)
    check('sort 2d inplace flat', [df['a2d'][i] for i in range(8)] == [1, 2, 3, 4, 10, 20, 30, 40])

def test_reindex_2d():
    df = _make_2d_frame()
    r = df.reindex([3, 0])
    check('reindex 2d len', len(r) == 2)
    check('reindex 2d a2d cols', r['a2d'].cols == 2 and r['a2d'].rows == 2)
    check('reindex 2d a2d flat', [r['a2d'][i] for i in range(4)] == [4, 1, 40, 10])
    # missing label 填充：两子列都填默认值
    r2 = df.reindex([0, 1, 2, 3, 4])
    check('reindex 2d fill len', len(r2) == 5)
    check('reindex 2d fill flat', [r2['a2d'][i] for i in range(10)] == [1, 2, 3, 4, 0, 10, 20, 30, 40, 0])
    check('reindex 2d fill rows', r2['a2d'].rows == 5 and r2['a2d'].cols == 2)

def test_drop_duplicates_2d():
    df = DataFrame({'a': int})
    df.append_row(a=1); df.append_row(a=1); df.append_row(a=2); df.append_row(a=2)
    df['a2d'] = IntColumn.from_2d([1, 1, 2, 2, 10, 10, 20, 20], 4, 2)
    r = df.drop_duplicates()
    check('dup 2d len', len(r) == 2)
    check('dup 2d a2d flat', [r['a2d'][i] for i in range(4)] == [1, 2, 10, 20])
    check('dup 2d a2d rows', r['a2d'].rows == 2 and r['a2d'].cols == 2)

def test_duplicated_2d_rowaware():
    # col0 重复但 col1 区分：旧 flat 读只读 col0 会误判为 dup
    df = DataFrame({'a': int})
    df.append_row(a=0); df.append_row(a=0)
    df['a2d'] = IntColumn.from_2d([1, 1, 10, 20], 2, 2)
    df['b'] = IntColumn(2); df['b'].append(7); df['b'].append(7)
    check('dup row-aware distinct', [df.duplicated(subset=['a2d'])[i] for i in range(2)] == [0, 0])
    r = df.drop_duplicates(subset=['a2d'])
    check('dup row-aware drop len', len(r) == 2)
    check('dup row-aware drop flat', [r['a2d'][i] for i in range(4)] == [1, 1, 10, 20])

    # 整行重复 → 仅第 2 个标记 dup
    df2 = DataFrame({'a': int})
    df2.append_row(a=0); df2.append_row(a=0); df2.append_row(a=0)
    df2['a2d'] = IntColumn.from_2d([1, 2, 1, 10, 20, 10], 3, 2)
    check('dup row-aware dup', [df2.duplicated(subset=['a2d'])[i] for i in range(3)] == [0, 0, 1])
    check('dup row-aware keep-last', [df2.duplicated(subset=['a2d'], keep='last')[i] for i in range(3)] == [1, 0, 0])
    r2 = df2.drop_duplicates(subset=['a2d'])
    check('dup row-aware drop flat', [r2['a2d'][i] for i in range(4)] == [1, 2, 10, 20])

def test_display_export_2d():
    df = _make_2d_frame()
    # to_string：2D 列每行显示整行向量（含子列 ≥1），旧实现只显示子列0
    s = df.to_string()
    check('disp 2d row0', '[1, 10]' in s)
    check('disp 2d row3', '[4, 40]' in s)
    # to_dict：每个逻辑行一个行向量
    d = df.to_dict()
    check('exp 2d dict', d['a2d'] == [[1, 10], [2, 20], [3, 30], [4, 40]])
    # to_numpy
    arr = df.to_numpy()
    check('exp 2d numpy', arr[0][2] == [1, 10] and arr[3][2] == [4, 40])
    # to_json records / columns
    j = df.to_json()
    check('exp 2d json rec row0', '"a2d": [1, 10]' in j)
    check('exp 2d json rec row3', '"a2d": [4, 40]' in j)
    j2 = df.to_json(orient_records=False)
    check('exp 2d json col', '[[1, 10], [2, 20], [3, 30], [4, 40]]' in j2)
    # to_csv
    fn = os.path.join(tempfile.mkdtemp(), 't2d.csv')
    df.to_csv(fn)
    txt = open(fn).read()
    check('exp 2d csv row0', '[1, 10]' in txt)
    check('exp 2d csv row3', '[4, 40]' in txt)
    # __eq__：2D 列必须整行比较（子列 ≥1 差异要区分）
    df2 = _make_2d_frame()
    df2['a2d'] = IntColumn.from_2d([1, 2, 3, 4, 99, 98, 97, 96], 4, 2)
    eq_result = df == df2
    check('exp 2d eq differs', eq_result['a2d'][0] == 0 or eq_result['a2d'][1] == 0)
    eq_result2 = df == _make_2d_frame()
    check('exp 2d eq same', all(eq_result2['a2d'][i] == 1 for i in range(4)))
    # double 2D 含 NaN：显示为 null（-ffast-math 下 dval != dval 失效，需位模式判定）
    dd = DataFrame({'a': int})
    dd.append_row(a=0); dd.append_row(a=0)
    dd['d2d'] = DoubleColumn.from_2d([1.5, 2.5, float('nan'), 4.5], 2, 2)
    ds = dd.to_string()
    check('exp 2d nan null', '[1.5, null]' in ds)


def test_elementwise_2d():
    df = DataFrame({'a': int, 's': str})
    for a, s in [(1, 'w'), (-2, 'x'), (-3, 'y'), (4, 'z')]:
        df.append_row(a=a, s=s)
    df['a2d'] = IntColumn.from_2d([1, -2, -3, 4, -10, 20, 30, -40], 4, 2)
    df['d2d'] = DoubleColumn.from_2d([1.5, -2.4, -3.6, 4.2, -10.7, 20.1, 30.9, -40.5], 4, 2)

    # abs：保形 2D，全 flat 逐元素
    o = df.abs()
    c = o['a2d']
    check('elem2d abs int shape', c.cols == 2 and c.rows == 4 and c.length == 4)
    check('elem2d abs int flat', [c[i] for i in range(8)] == [1, 2, 3, 4, 10, 20, 30, 40])
    c = o['d2d']
    check('elem2d abs dbl flat', [c[i] for i in range(8)] == [1.5, 2.4, 3.6, 4.2, 10.7, 20.1, 30.9, 40.5])

    # round：保形 2D
    o = df.round()
    c = o['d2d']
    check('elem2d round shape', c.cols == 2 and c.rows == 4)
    check('elem2d round flat', [c[i] for i in range(8)] == [2, -2, -4, 4, -11, 20, 31, -40])

    # replace：数字遍历全 flat（subcol1 的 20 被替换为 -1）
    o = df.replace(20, -1)
    c = o['a2d']
    check('elem2d replace shape', c.cols == 2)
    check('elem2d replace subcol1', [c[i] for i in range(8)] == [1, -2, -3, 4, -10, -1, 30, -40])

    # isnull：2D 输出保形（subcol1 的 NaN 也要标记）
    d2 = DataFrame({'x': float})
    for v in [1.0, 2.0, 3.0, 4.0]:
        d2.append_row(x=v)
    d2['d2d'] = DoubleColumn.from_2d([0.0, float('nan'), 3.0, 4.0, 5.0, 6.0, float('nan'), 8.0], 4, 2)
    o = d2.isnull()
    c = o['d2d']
    check('elem2d isnull shape', c.cols == 2 and c.rows == 4)
    check('elem2d isnull flat', [c[i] for i in range(8)] == [0, 1, 0, 0, 0, 0, 1, 0])

    # fillna：填满全 flat（subcol1 的 NaN 也要填）
    o = d2.fillna(9.0)
    c = o['d2d']
    check('elem2d fillna shape', c.cols == 2 and c.rows == 4)
    check('elem2d fillna subcol1', [c[i] for i in range(8)] == [0.0, 9.0, 3.0, 4.0, 5.0, 6.0, 9.0, 8.0])

    # all/any：扫描全部子列（subcol1 的 0 需参与判定）
    d3 = DataFrame({'a': int})
    d3.append_row(a=1); d3.append_row(a=2)
    d3['a2d'] = IntColumn.from_2d([1, 2, 5, 0], 2, 2)
    d3_all = d3.all()
    check('elem2d all false', d3_all['a'] == True and d3_all['a2d'] == False)
    d3_any = d3.any()
    check('elem2d any true', d3_any['a'] == True and d3_any['a2d'] == True)
    d4 = DataFrame({'a': int})
    d4.append_row(a=1); d4.append_row(a=2)
    d4['a2d'] = IntColumn.from_2d([1, 2, 5, 6], 2, 2)
    d4_all = d4.all()
    check('elem2d all true', d4_all['a'] == True and d4_all['a2d'] == True)


def test_series_row_2d():
    # 基帧：a2d 子列0=[1,2,3,4] 子列1=[10,20,30,40]；d2d 子列0=[1,NAN,3,NAN] 子列1=[10,20,NAN,40]
    df = DataFrame({'a': int, 's': str})
    for a, s in [(1, 'w'), (2, 'x'), (3, 'y'), (4, 'z')]:
        df.append_row(a=a, s=s)
    df['a2d'] = IntColumn.from_2d([1, 2, 3, 4, 10, 20, 30, 40], 4, 2)
    df['d2d'] = DoubleColumn.from_2d([1.0, float('nan'), 3.0, float('nan'), 10.0, 20.0, float('nan'), 40.0], 4, 2)

    # pct_change：逐子列独立序列，输出保形 2D
    pc = df.pct_change()
    c = pc['a2d']
    check('sr2d pct shape', c.cols == 2 and c.rows == 4 and c.length == 4)
    check('sr2d pct nan0', math.isnan(c[0]) and math.isnan(c[4]))
    check('sr2d pct sub0 r1', abs(c[1] - 1.0) < 1e-9)
    check('sr2d pct sub1 r1', abs(c[5] - 1.0) < 1e-9)
    check('sr2d pct sub1 r2', abs(c[6] - 0.5) < 1e-9)  # (30-20)/20
    c0 = pc['a'][0]
    check('sr2d pct 1d untouched', c0 is not None)  # int col 也给 Double 序列（此处仅保形检查）

    # ffill/bfill：逐子列独立（NaN 不跨子列）
    ff = df.ffill()
    c = ff['d2d']
    check('sr2d ffill sub0', [c[i] for i in range(4)] == [1.0, 1.0, 3.0, 3.0])
    check('sr2d ffill sub1', [c[i] for i in range(4, 8)] == [10.0, 20.0, 20.0, 40.0])
    bf = df.bfill()
    c = bf['d2d']
    check('sr2d bfill sub0', [c[i] for i in range(4)] == [1.0, 3.0, 3.0, float('nan')] or
                             ([c[i] for i in range(3)] == [1.0, 3.0, 3.0] and math.isnan(c[3])))
    check('sr2d bfill sub1', [c[i] for i in range(4, 8)] == [10.0, 20.0, 40.0, 40.0])

    # shift：逐子列独立位移（仅数值列，避免 str 报错）
    dsh = DataFrame({'a': int})
    dsh.append_row(a=1); dsh.append_row(a=2); dsh.append_row(a=3); dsh.append_row(a=4)
    dsh['a2d'] = IntColumn.from_2d([1, 2, 3, 4, 10, 20, 30, 40], 4, 2)
    sh = dsh.shift(1)
    c = sh['a2d']
    check('sr2d shift shape', c.cols == 2 and c.rows == 4)
    check('sr2d shift +1', [c[i] for i in range(8)] == [0, 1, 2, 3, 0, 10, 20, 30])
    sh2 = dsh.shift(-1)
    c = sh2['a2d']
    check('sr2d shift -1', [c[i] for i in range(8)] == [2, 3, 4, 0, 20, 30, 40, 0])

    # dropna：任一子列该行 NaN 即整行丢弃
    dq = DataFrame({'v': float})
    for v in [1.0, 2.0, 3.0, 4.0]:
        dq.append_row(v=v)
    dq['x2d'] = DoubleColumn.from_2d([1.0, float('nan'), 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 4, 2)
    r = dq.dropna()
    check('sr2d dropna rows', r.shape[0] == 3)
    c = r['x2d']
    check('sr2d dropna 2d', c.cols == 2 and c.rows == 3)
    check('sr2d dropna flat', [c[i] for i in range(6)] == [1.0, 3.0, 4.0, 5.0, 7.0, 8.0])

    # pivot：values 为 2D 时单元为向量 → 输出透视列保形 2D
    dfp = DataFrame({'idx': int, 'col': str})
    for t in [(1, 'A'), (1, 'B'), (2, 'A'), (2, 'B')]:
        dfp.append_row(idx=t[0], col=t[1])
    dfp['val2d'] = DoubleColumn.from_2d([1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0], 4, 2)
    p = dfp.pivot(index='idx', columns='col', values='val2d')
    check('sr2d pivot shape', p.shape == (2, 3))
    ca = p['A']
    check('sr2d pivot A 2d', ca.cols == 2 and ca.rows == 2)
    check('sr2d pivot A flat', [ca[i] for i in range(4)] == [1.0, 5.0, 2.0, 6.0])
    cb = p['B']
    check('sr2d pivot B flat', [cb[i] for i in range(4)] == [3.0, 7.0, 4.0, 8.0])

    # melt：values 为 2D → value 输出列保形 2D
    dfm = DataFrame({'id': str})
    dfm.append_row(id='r1'); dfm.append_row(id='r2')
    dfm['x2d'] = DoubleColumn.from_2d([1.0, 2.0, 10.0, 20.0], 2, 2)
    m = dfm.melt(id_vars=['id'], value_vars=['x2d'])
    check('sr2d melt shape', m.shape[0] == 2)
    check('sr2d melt var', m['variable'][0] == 'x2d')
    check('sr2d melt id', m['id'][0] == 'r1')
    check('sr2d melt value 2d', m['value'].cols == 2 and m['value'].rows == 2)
    check('sr2d melt value flat', [m['value'][i] for i in range(4)] == [1.0, 2.0, 10.0, 20.0])

    # expanding_sum / ewm_mean：逐子列独立
    ex = df.expanding_sum('a2d')
    check('sr2d exp shape', ex.cols == 2 and ex.rows == 4)
    check('sr2d exp flat', [ex[i] for i in range(8)] == [1.0, 3.0, 6.0, 10.0, 10.0, 30.0, 60.0, 100.0])
    ew = df.ewm_mean('a2d', span=3)
    check('sr2d ewm shape', ew.cols == 2 and ew.rows == 4)
    check('sr2d ewm sub0', abs(ew[0] - 0.5) < 1e-9 and abs(ew[1] - 1.25) < 1e-9)
    check('sr2d ewm sub1', abs(ew[4] - 5.0) < 1e-9 and abs(ew[5] - 12.5) < 1e-9)

    # apply axis='columns'：func 逐子列标量（n*c 个），输出保形
    o = df.apply(lambda v: v * 2)
    c = o['a2d']
    check('sr2d apply cols 2d', c.cols == 2 and c.rows == 4)
    check('sr2d apply cols flat', [c[i] for i in range(8)] == [2, 4, 6, 8, 20, 40, 60, 80])
    check('sr2d apply 1d len', o['a'].length == 4)
    check('sr2d apply str 1d', o['s'].length == 4)

    # 纯数值帧：apply 返回浮点 → 2D 输出转 Double
    dn = DataFrame({'x': int})
    dn.append_row(x=1); dn.append_row(x=2); dn.append_row(x=3); dn.append_row(x=4)
    dn['x2d'] = IntColumn.from_2d([1, 2, 3, 4, 10, 20, 30, 40], 4, 2)
    of = dn.apply(lambda v: v + 0.5)
    cf = of['x2d']
    check('sr2d apply float2d', cf.cols == 2 and cf.rows == 4)
    check('sr2d apply float flat', [cf[i] for i in range(8)] == [1.5, 2.5, 3.5, 4.5, 10.5, 20.5, 30.5, 40.5])

    # apply axis='index'：2D 单元为整行向量
    rows = df.apply(lambda r: (r['a2d'][0], r['a2d'][1], r['s']), axis='index')
    check('sr2d apply idx r0', rows[0] == (1, 10, 'w'))
    check('sr2d apply idx r3', rows[3] == (4, 40, 'z'))

    # T：cell 保持行向量转置（保形 2D）
    dt = DataFrame({'a': int})
    dt.append_row(a=5); dt.append_row(a=6)
    dt['p2d'] = DoubleColumn.from_2d([1.0, 2.0, 10.0, 20.0], 2, 2)
    dt['q2d'] = DoubleColumn.from_2d([100.0, 200.0, 300.0, 400.0], 2, 2)
    t = dt.T
    check('sr2d T len', t.df_length == 3)
    ta = t['a']
    check('sr2d T a 2d', ta.cols == 2 and ta.rows == 3)
    check('sr2d T a flat', [ta[i] for i in range(6)] == [5.0, 1.0, 100.0, float('nan'), 10.0, 300.0] or
                           ([ta[0], ta[1], ta[2], ta[4], ta[5]] == [5.0, 1.0, 100.0, 10.0, 300.0] and math.isnan(ta[3])))
    tp = t['p2d']
    check('sr2d T p2d flat', [tp[i] for i in range(6)] == [6.0, 2.0, 200.0, float('nan'), 20.0, 400.0] or
                            ([tp[0], tp[1], tp[2], tp[4], tp[5]] == [6.0, 2.0, 200.0, 20.0, 400.0] and math.isnan(tp[3])))

    # reshape：元素 = 整行单元（子列向量），输出保形
    dr = DataFrame({'x': int})
    dr.append_row(x=1); dr.append_row(x=2)
    dr['r1'] = IntColumn.from_2d([1, 2, 10, 20], 2, 2)
    dr['r2'] = IntColumn.from_2d([3, 4, 30, 40], 2, 2)
    res = dr.reshape(2, 3)
    check('sr2d reshape shape', res.shape == (2, 3))
    check('sr2d reshape A', [res['A'][i] for i in range(2)] == [1.0, 2.0])
    check('sr2d reshape B', [res['B'][i] for i in range(4)] == [1.0, 2.0, 10.0, 20.0])
    check('sr2d reshape C', [res['C'][i] for i in range(4)] == [3.0, 4.0, 30.0, 40.0])


def test_shift_time_cols():
    # DataFrame.shift 对定宽时间列（Date/DateTime/Time）支持 memcpy/memset-0（TODO 完成）
    dd = DataFrame({'d': 'date'})
    dd.append_row(d='2024-02-01'); dd.append_row(d='2024-03-01'); dd.append_row(d='2024-04-01')
    sd = dd.shift(1)
    check('tcol date shift+1', [sd['d'][i] for i in range(3)] == [19700101, 20240201, 20240301])
    sd2 = dd.shift(-1)
    check('tcol date shift-1', [sd2['d'][i] for i in range(3)] == [20240301, 20240401, 19700101])

    dt = DataFrame({'t': 'time'})
    dt.append_row(t='01:00:00'); dt.append_row(t='03:00:00'); dt.append_row(t='05:00:00')
    st = dt.shift(1)
    check('tcol time shift+1', [st['t'][i] for i in range(3)] == ['00:00:00', '01:00:00', '03:00:00'])

    dn = DataFrame({'dt': 'datetime'})
    dn.append_row(dt='2023-01-15 10:30:00'); dn.append_row(dt='2023-06-20 14:00:00')
    zero_col = DateTimeColumn(); zero_col.append(0)
    zero_str = zero_col[0]
    sdt = dn.shift(1)
    check('tcol datetime shift+1', sdt['dt'][0] == zero_str and sdt['dt'][1] == '2023-01-15 10:30:00')


def test_agg_stats_2d():
    # 2D 列子列0 有重复：行 (1,10)(1,20)(2,30)(2,30)
    df = DataFrame({'a': int, 's': str})
    for a, s in [(1, 'w'), (1, 'x'), (2, 'y'), (2, 'z')]:
        df.append_row(a=a, s=s)
    df['a2d'] = IntColumn.from_2d([1, 1, 2, 2, 10, 20, 30, 30], 4, 2)

    # nunique：整行去重 = 3（旧实现 flat 读子列0 只得 2）
    check('agg2d nunique', df.nunique('a2d') == 3)

    # value_counts：行 (2,30) 计 2，(1,10)/(1,20) 各 1；结果 key 列保持 2D
    vc = df.value_counts('a2d')
    check('agg2d vc total', len(vc) == 3)
    check('agg2d vc cols', vc['a2d'].cols == 2 and vc['a2d'].rows == 3)
    check('agg2d vc flat', [vc['a2d'][i] for i in range(6)] == [2, 1, 1, 30, 10, 20])
    check('agg2d vc counts', [vc['count'][i] for i in range(3)] == [2, 1, 1])

    # argsort_by：整行字典序 [0,1,2,3]（旧实现 flat 排子列0 → [1,0,3,2]）
    check('agg2d argsort', df.argsort_by('a2d') == [0, 1, 2, 3])

    # rank：average → 行(1,*)并列 rank1/2，行(2,*)并列 3.5
    r = df.rank('a2d')
    check('agg2d rank', [r[i] for i in range(4)] == [1.0, 2.0, 3.5, 3.5])

    # cut / qcut：2D 列无标量可分箱 → 明确 TypeError
    check('agg2d cut guard', expect_exception(TypeError, df.cut, 'a2d', [0, 1, 2]))
    check('agg2d qcut guard', expect_exception(TypeError, df.qcut, 'a2d'))


# =====================================================================
#  Aggregation: mean, median, std, var, max, min, sum, count, quantile, skew, kurt
# =====================================================================
def test_aggregation():
    df = DataFrame({'a': int, 'b': float})
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
    df = DataFrame({'a': int, 'b': float})
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
    df2 = DataFrame({'x': int})
    for v in [3, 1, 4, 1, 5]: df2.append_row(x=v)
    cmx2 = df2.cummax()
    check('cummax varying', [cmx2['x'][i] for i in range(5)] == [3, 3, 4, 4, 5])

    cmn = df.cummin()
    check('cummin a[3]', cmn['a'][3] == 1)


# =====================================================================
#  Column arithmetic: add, sub, mul, div
# =====================================================================
def test_col_arithmetic():
    df = DataFrame({'a': int, 'b': int})
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
    df = DataFrame({'a': float, 'b': float})
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

def _make_2d_corr_df():
    # c 为 2D 列（cols=2，列优先）：子列 c_0=10a，c_1=3a；e 与 a 相关系数为 -0.2
    df = DataFrame()
    df.df_length = 4
    df._columns['a'] = DoubleColumn.from_2d([1, 2, 3, 4], 4, 1)
    df._columns['b'] = DoubleColumn.from_2d([2, 4, 6, 8], 4, 1)
    df._columns['c'] = DoubleColumn.from_2d([10, 20, 30, 40, 3, 6, 9, 12], 4, 2)
    df._columns['e'] = DoubleColumn.from_2d([4, 1, 2, 3], 4, 1)
    return df

def test_corr_cov_2d():
    # 2D 列按子列展开：c → {c_0, c_1}
    df = _make_2d_corr_df()

    c = df.corr()
    check('corr2d shape', c.shape == (5, 5))
    check('corr2d feats', set(c.columns) == {'a', 'b', 'c_0', 'c_1', 'e'})
    check('corr2d diag', c['a'][0] == 1.0 and c['c_1'][3] == 1.0)
    check('corr2d a-b', c['b'][0] == 1.0)
    check('corr2d a-c0', c['c_0'][0] == 1.0)
    check('corr2d a-c1', c['c_1'][0] == 1.0)
    check('corr2d a-e', abs(c['e'][0] + 0.2) < 1e-12)

    cv = df.cov()
    check('cov2d shape', cv.shape == (5, 5))
    check('cov2d feats', set(cv.columns) == {'a', 'b', 'c_0', 'c_1', 'e'})
    check('cov2d var', cv['a'][0] > 0)
    check('cov2d sym', cv['a'][1] == cv['b'][0])
    check('cov2d a-c1', abs(cv['c_1'][0] - 5.0) < 1e-9)

    r = df.corrwith(df['a'])
    check('corrwith2d keys', set(r.keys()) == {'a', 'b', 'c_0', 'c_1', 'e'})
    check('corrwith2d a', r['a'] == 1.0)
    check('corrwith2d c1', r['c_1'] == 1.0)
    check('corrwith2d e', abs(r['e'] + 0.2) < 1e-12)

    # 列-列标量函数：2D 列取主子列（sub=0）
    r2 = df.corrcoef(df['a'], df['c'])
    check('corrcoef2d primary', abs(r2 - 1.0) < 1e-9)
    cv2 = df.cov_withcols(df['a'], df['c'])
    check('cov_withcols2d', cv2 > 0)

    ps = df.pearson_sim()
    check('pearson2d shape', ps.shape == (5, 5))
    check('pearson2d feats', set(ps.columns) == {'a', 'b', 'c_0', 'c_1', 'e'})
    check('pearson2d diag', ps['a'][0] == 1.0)
    check('pearson2d a-c1', ps['c_1'][0] == 1.0)
    check('pearson2d a-e', abs(ps['e'][0] + 0.2) < 1e-12)

def test_distance_functions():
    df = DataFrame({'a': float, 'b': float})
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
    df = DataFrame({'tag': str})
    for t in ['a', 'b', 'a', 'c', 'a', 'b']: df.append_row(tag=t)
    vc = df.value_counts('tag')
    check('value_counts shape', vc.shape[0] == 3)
    check('value_counts top', vc['tag'][0] == 'a')
    check('value_counts count', vc['count'][0] == 3)

def test_clip_column():
    df = DataFrame({'a': float})
    for v in [1.0, 5.0, 10.0, 15.0, 20.0]: df.append_row(a=v)
    df.clip_column('a', 5.0, 15.0)
    check('clip min', df['a'][0] == 5.0)
    check('clip mid', df['a'][1] == 5.0)
    check('clip max', df['a'][4] == 15.0)

def test_where_column():
    df = DataFrame({'val': int, 'cond': int})
    df.append_row(val=10, cond=1); df.append_row(val=20, cond=0); df.append_row(val=30, cond=1)
    r = df.where_column('val', 'cond', 999)
    check('where', [r[i] for i in range(3)] == [10, 999, 30])

def test_mask_column():
    df = DataFrame({'val': int, 'cond': int})
    df.append_row(val=10, cond=1); df.append_row(val=20, cond=0); df.append_row(val=30, cond=1)
    r = df.mask_column('val', 'cond', 999)
    check('mask', [r[i] for i in range(3)] == [999, 20, 999])

def test_where_row():
    df = DataFrame({'a': int, 'b': int})
    df.append_row(a=1, b=10); df.append_row(a=2, b=20); df.append_row(a=3, b=30)
    mask = IntColumn()
    mask.append(1); mask.append(0); mask.append(1)
    r = df.where_row(mask, fill=-1)
    check('where_row a', [r['a'][i] for i in range(3)] == [1, -1, 3])
    check('where_row b', [r['b'][i] for i in range(3)] == [10, -1, 30])

def test_where_mask_row_2d():
    """2D 列：where/mask 掩码按逻辑行判定（每行一个 cond），where_row 按行掩码，结果保持源维度"""
    df = DataFrame()
    df.df_length = 4
    df._columns['v'] = DoubleColumn.from_2d([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0], 4, 2)
    df._columns['iv'] = IntColumn.from_2d([1, 2, 3, 4, 10, 20, 30, 40], 4, 2)
    cond = IntColumn()
    for c in [1, 0, 1, 0]: cond.append(c)
    df._columns['cond'] = cond

    r = df.where_column('v', 'cond', 99.0)
    check('2d where dims', r.rows == 4 and r.is_2d and r.cols == 2)
    check('2d where flat', [r[i] for i in range(8)] == [1.0, 99.0, 3.0, 99.0, 10.0, 99.0, 30.0, 99.0])

    ri = df.where_column('iv', 'cond', 77)
    check('2d where int', [ri[i] for i in range(8)] == [1, 77, 3, 77, 10, 77, 30, 77])

    m = df.mask_column('v', 'cond', 5.0)
    check('2d mask flat', [m[i] for i in range(8)] == [5.0, 2.0, 5.0, 4.0, 5.0, 20.0, 5.0, 40.0])

    mask = IntColumn()
    for c in [1, 0, 1, 0]: mask.append(c)
    rr = df.where_row(mask, fill=-1)
    check('2d where_row shape', rr.shape == (4, 3))
    check('2d where_row double', [rr['v'][i] for i in range(8)] == [1.0, -1.0, 3.0, -1.0, 10.0, -1.0, 30.0, -1.0])
    check('2d where_row int', [rr['iv'][i] for i in range(8)] == [1, -1, 3, -1, 10, -1, 30, -1])

    bad = IntColumn.from_2d([1]*8, 4, 2)
    df._columns['badc'] = bad
    raised = False
    try:
        df.where_column('v', 'badc', 0.0)
    except ValueError:
        raised = True
    check('2d cond length uses rows', raised)


# =====================================================================
#  Groupby
# =====================================================================
def test_groupby():
    df = DataFrame({'dept': str, 'age': int, 'salary': float})
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
    df = DataFrame({'a': str, 'b': str, 'val': int})
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
    df1 = DataFrame({'id': int, 'v1': float})
    df1.append_row(id=1, v1=10.0); df1.append_row(id=2, v1=20.0); df1.append_row(id=3, v1=30.0)
    df2 = DataFrame({'id': int, 'v2': float})
    df2.append_row(id=2, v2=200.0); df2.append_row(id=3, v2=300.0); df2.append_row(id=4, v2=400.0)
    r = df1.merge(df2, on='id', how='inner')
    check('inner shape', r.shape == (2, 3))
    check('inner ids', sorted([r['id'][i] for i in range(2)]) == [2, 3])

def test_merge_left():
    df1 = DataFrame({'id': int, 'v1': float})
    df1.append_row(id=1, v1=10.0); df1.append_row(id=2, v1=20.0)
    df2 = DataFrame({'id': int, 'v2': float})
    df2.append_row(id=2, v2=200.0); df2.append_row(id=3, v2=300.0)
    r = df1.merge(df2, on='id', how='left')
    check('left shape', r.shape == (2, 3))
    check('left id=1', r['v1'][0] == 10.0)

def test_merge_right():
    df1 = DataFrame({'id': int, 'v1': float})
    df1.append_row(id=1, v1=10.0); df1.append_row(id=2, v1=20.0)
    df2 = DataFrame({'id': int, 'v2': float})
    df2.append_row(id=2, v2=200.0); df2.append_row(id=3, v2=300.0)
    r = df1.merge(df2, on='id', how='right')
    check('right shape', r.shape == (2, 3))
    check('right has all ids', r.shape[0] == 2)

def test_merge_outer():
    df1 = DataFrame({'id': int, 'v1': float})
    df1.append_row(id=1, v1=10.0); df1.append_row(id=2, v1=20.0)
    df2 = DataFrame({'id': int, 'v2': float})
    df2.append_row(id=2, v2=200.0); df2.append_row(id=3, v2=300.0)
    r = df1.merge(df2, on='id', how='outer')
    check('outer shape', r.shape == (3, 3))


# =====================================================================
#  Concat: multi-DF, inner, outer
# =====================================================================
def test_concat_multi():
    from fluidframe.core.dataframe import concat, concat_inner
    df1 = DataFrame({'a': int, 'b': int})
    df1.append_row(a=1, b=10); df1.append_row(a=2, b=20)
    df2 = DataFrame({'a': int, 'b': int})
    df2.append_row(a=3, b=30)
    df3 = DataFrame({'a': int, 'b': int})
    df3.append_row(a=4, b=40); df3.append_row(a=5, b=50)
    r = concat([df1, df2, df3])
    check('concat 3', r.shape[0] == 5)
    check('concat vals', [r['a'][i] for i in range(5)] == [1, 2, 3, 4, 5])

    r2 = concat([df1, df2, df3], ignore_index=True)
    check('concat ignore_index', r2.shape[0] == 5)

def test_concat_inner():
    from fluidframe.core.dataframe import concat_inner
    df1 = DataFrame({'a': int, 'b': int})
    df1.append_row(a=1, b=10)
    df2 = DataFrame({'a': int, 'c': float})
    df2.append_row(a=2, c=20.0)
    r = concat_inner([df1, df2])
    check('concat_inner has a', 'a' in r.columns)
    check('concat_inner rows', r.shape[0] == 2)

def test_concat_outer():
    from fluidframe.core.dataframe import concat_outer
    df1 = DataFrame({'a': int})
    df1.append_row(a=1)
    df2 = DataFrame({'b': float})
    df2.append_row(b=2.0)
    r = concat_outer([df1, df2])
    check('outer has a and b', 'a' in r.columns and 'b' in r.columns)
    check('outer rows', r.shape[0] == 2)


# =====================================================================
#  eval / query
# =====================================================================
def test_eval():
    df = DataFrame({'a': int, 'b': int})
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
    df = DataFrame({'a': int})
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
    df = DataFrame.arange(12)
    r = df.reshape(3, 4)
    check('reshape shape', r.shape == (3, 4))
    check('reshape A[0]', r['A'][0] == 0.0)
    check('reshape D[2]', r['D'][2] == 11.0)

def test_transpose():
    df = DataFrame({'a': int, 'b': int})
    df.append_row(a=1, b=10); df.append_row(a=2, b=20)
    t = df.T
    check('T shape', t.shape == (2, 2))
    check('T cols', set(t.columns) == {'a', 'b'})
    check('T a[0]', t['a'][0] == 1.0)
    check('T a[1]', t['a'][1] == 10.0)

def test_memory_usage():
    df = DataFrame({'a': int, 'b': float})
    for i in range(100): df.append_row(a=i, b=float(i))
    mu = df.memory_usage()
    check('memory_usage positive', mu > 0)

def test_len():
    df = make_complex_df()
    check('len', len(df) == 10)
    df2 = DataFrame({'x': int})
    check('len empty', len(df2) == 0)


# =====================================================================
#  Insert row / drop_index / sort_index / argsort_by / nunique
# =====================================================================
def test_insert_row():
    df = DataFrame({'a': int, 'b': str})
    df.append_row(a=1, b='x'); df.append_row(a=3, b='z')
    df.insert_row(1, a=2, b='yy')
    check('insert_row shape', df.shape == (3, 2))
    check('insert_row val', [df['a'][i] for i in range(3)] == [1, 2, 3])
    check('insert_row str', [df['b'][i] for i in range(3)] == ['x', 'yy', 'z'])

def test_drop_index():
    df = DataFrame({'a': int})
    for i in range(5): df.append_row(a=i)
    df.reset_index()
    r = df.drop_index([0, 2])
    check('drop_index shape', r.shape[0] == 3)
    check('drop_index vals', [r['a'][i] for i in range(3)] == [1, 3, 4])

def test_sort_index():
    df = DataFrame({'a': int})
    for v in [50, 10, 30, 20, 40]: df.append_row(a=v)
    df.reset_index()
    df.sort_index(inplace=True)
    check('sort_index', [df['a'][i] for i in range(5)] == [50, 10, 30, 20, 40])

def test_argsort_by():
    df = DataFrame({'a': int})
    for v in [30, 10, 20]: df.append_row(a=v)
    idx = df.argsort_by('a')
    check('argsort_by', idx == [1, 2, 0])
    idx_desc = df.argsort_by('a', reverse=True)
    check('argsort_by desc', idx_desc == [0, 2, 1])

def test_nunique():
    df = DataFrame({'a': str})
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
    df = DataFrame({'a': int, 'b': float, 'c': str})
    df.append_row(a=1, b=2.5, c='hello')
    df.append_row(a=3, b=4.0, c='world')
    d = df.to_dict()
    check('to_dict keys', set(d.keys()) == {'a', 'b', 'c'})
    check('to_dict int', d['a'] == [1, 3])
    check('to_dict float', d['b'] == [2.5, 4.0])
    check('to_dict str', d['c'] == ['hello', 'world'])

    # Empty DataFrame
    df2 = DataFrame({'a': int})
    check('to_dict empty', df2.to_dict() == {'a': []})

def test_to_json():
    import json
    df = DataFrame({'x': int, 'y': float})
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
    df = DataFrame({'a': int, 'b': float})
    df.append_row(a=-5, b=-3.14)
    df.append_row(a=10, b=2.71)
    df.append_row(a=-7, b=0.0)
    abs_df = df.abs()
    check('abs int', [abs_df['a'][i] for i in range(abs_df['a'].length)] == [5, 10, 7])
    check('abs float', [abs_df['b'][i] for i in range(abs_df['b'].length)] == [3.14, 2.71, 0.0])

    # Empty
    df2 = DataFrame({'x': int})
    check('abs empty', df2.abs().to_dict() == {'x': []})

    # Mixed types (string passthrough)
    df3 = DataFrame({'a': str})
    df3.append_row(a='hello')
    abs3 = df3.abs()
    check('abs string passthrough', [abs3['a'][i] for i in range(abs3['a'].length)] == ['hello'])

def test_round():
    df = DataFrame({'a': float})
    df.append_row(a=1.23456)
    df.append_row(a=5.67890)
    df.append_row(a=-3.14159)

    r0 = df.round(0)
    check('round 0', [r0['a'][i] for i in range(r0['a'].length)] == [1.0, 6.0, -3.0])

    r2 = df.round(2)
    check('round 2', [r2['a'][i] for i in range(r2['a'].length)] == [1.23, 5.68, -3.14])

    # Int column passthrough
    df2 = DataFrame({'b': int})
    df2.append_row(b=42)
    r_int = df2.round(0)
    check('round int passthrough', [r_int['b'][i] for i in range(r_int['b'].length)] == [42])

def test_all_any():
    # all() - all True
    df = DataFrame({'x': int})
    df.append_row(x=1)
    df.append_row(x=2)
    check('all true', df.all()['x'] is True)

    # all() - one False
    df2 = DataFrame({'x': int})
    df2.append_row(x=1)
    df2.append_row(x=0)
    check('all false', df2.all()['x'] is False)

    # all() - empty
    df3 = DataFrame({'x': int})
    check('all empty', df3.all()['x'] is True)

    # any() - has True
    df4 = DataFrame({'x': int})
    df4.append_row(x=0)
    df4.append_row(x=5)
    check('any true', df4.any()['x'] is True)

    # any() - all False
    df5 = DataFrame({'x': int})
    df5.append_row(x=0)
    df5.append_row(x=0)
    check('any false', df5.any()['x'] is False)

    # any() - empty
    df6 = DataFrame({'x': int})
    check('any empty', df6.any()['x'] is False)

    # bool column
    df7 = DataFrame({'x': bool})
    df7.append_row(x=True)
    df7.append_row(x=True)
    check('all bool true', df7.all()['x'] is True)
    df7.append_row(x=False)
    check('all bool mixed', df7.all()['x'] is False)

    # multi-column all/any
    df8 = DataFrame({'a': int, 'b': int})
    df8.append_row(a=1, b=2)
    df8.append_row(a=3, b=4)
    check('multi all true', df8.all()['a'] is True and df8.all()['b'] is True)
    df8.append_row(a=0, b=1)
    check('multi all false', df8.all()['a'] is False and df8.all()['b'] is True)

def test_pct_change():
    df = DataFrame({'x': int})
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
    df2 = DataFrame({'y': float})
    df2.append_row(y=10.0)
    df2.append_row(y=20.0)
    pct_f = df2.pct_change()
    check('pct_change float nan', math.isnan(pct_f['y'][0]))
    check('pct_change float 100%', abs(pct_f['y'][1] - 1.0) < 1e-10)

    # zero previous value -> nan
    df3 = DataFrame({'z': int})
    df3.append_row(z=0)
    df3.append_row(z=5)
    pct_z = df3.pct_change()
    check('pct_change zero prev', math.isnan(pct_z['z'][1]))


def test_replace():
    df = DataFrame({'a': int, 'b': float})
    df.append_row(a=1, b=1.0)
    df.append_row(a=2, b=2.0)
    df.append_row(a=3, b=3.0)
    r = df.replace(2, 99)
    check('replace a', r['a'][1] == 99)
    check('replace b', r['b'][1] == 99.0)
    check('replace other unchanged', r['a'][0] == 1)

    df2 = DataFrame({'s': str})
    df2.append_row(s='hello')
    df2.append_row(s='world')
    r2 = df2.replace('world', 'python')
    check('replace str', r2['s'][1] == 'python')


def test_ffill_bfill():
    import math
    df = DataFrame({'a': float})
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
    df1 = DataFrame({'a': int, 'b': float})
    df1.append_row(a=1, b=1.0)
    df1.append_row(a=2, b=2.0)

    df2 = DataFrame({'a': int, 'b': float})
    df2.append_row(a=1, b=1.0)
    df2.append_row(a=2, b=2.0)

    eq_result = df1 == df2
    check('eq same is DataFrame', isinstance(eq_result, DataFrame))
    check('eq same shape', eq_result.shape == (2, 2))
    check('eq same all true', all(eq_result['a'][i] == 1 for i in range(2)))
    check('eq same all true b', all(eq_result['b'][i] == 1 for i in range(2)))

    df3 = DataFrame({'a': int, 'b': float})
    df3.append_row(a=1, b=1.0)
    df3.append_row(a=2, b=99.0)
    eq_result2 = df1 == df3
    check('eq diff val a', eq_result2['a'][0] == 1 and eq_result2['a'][1] == 1)
    check('eq diff val b', eq_result2['b'][0] == 1 and eq_result2['b'][1] == 0)

    df4 = DataFrame({'a': int})
    df4.append_row(a=1)
    eq_diff_cols = df1 == df4
    check('eq diff cols returns False', eq_diff_cols is False)

    df5 = DataFrame({'a': int, 'b': float})
    df5.append_row(a=1, b=1.0)
    eq_diff_len = df1 == df5
    check('eq diff len returns False', eq_diff_len is False)


def test_pivot():
    df = DataFrame({'idx': str, 'col': str, 'val': float})
    df.append_row(idx='r1', col='A', val=1.0)
    df.append_row(idx='r1', col='B', val=2.0)
    df.append_row(idx='r2', col='A', val=3.0)
    df.append_row(idx='r2', col='B', val=4.0)

    p = df.pivot(index='idx', columns='col', values='val')
    check('pivot shape', p.shape == (2, 3))
    check('pivot r1_A', p['A'][0] == 1.0)
    check('pivot r2_B', p['B'][1] == 4.0)


def test_melt():
    df = DataFrame({'id': str, 'x': float, 'y': float})
    df.append_row(id='r1', x=1.0, y=10.0)
    df.append_row(id='r2', x=2.0, y=20.0)

    m = df.melt(id_vars=['id'], value_vars=['x', 'y'])
    check('melt shape', m.shape == (4, 3))
    check('melt var', m['variable'][0] == 'x')
    check('melt val', m['value'][0] == 1.0)
    check('melt id', m['id'][0] == 'r1')


def test_expanding_ewm():
    df = DataFrame({'a': float})
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
    from fluidframe.core.dataframe import read_json
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
    df1 = DataFrame({'a': int, 'b': int, 'v1': float})
    df1.append_row(a=1, b=10, v1=1.0)
    df1.append_row(a=2, b=20, v1=2.0)

    df2 = DataFrame({'a': int, 'b': int, 'v2': float})
    df2.append_row(a=1, b=10, v2=100.0)
    df2.append_row(a=2, b=20, v2=200.0)

    m = df1.merge(df2, on=['a', 'b'])
    check('merge_multicol shape', m.shape == (2, 4))
    check('merge_multicol v2', m['v2'][0] == 100.0)


def test_merge_col_conflict():
    df1 = DataFrame({'id': int, 'val': float})
    df1.append_row(id=1, val=1.0)

    df2 = DataFrame({'id': int, 'val': float})
    df2.append_row(id=1, val=99.0)

    m = df1.merge(df2, on='id')
    check('merge_conflict has val_x', 'val_x' in m.columns)
    check('merge_conflict has val_y', 'val_y' in m.columns)
    check('merge_conflict val_x', m['val_x'][0] == 1.0)
    check('merge_conflict val_y', m['val_y'][0] == 99.0)


def run_all():
    print('=== DataFrame Tests ===')

    # Construction
    test_init()
    test_init_complex()
    test_from_dict()
    test_from_list()
    test_from_arrays()
    test_factory_methods()

    # Reserve capacity — fixed-capacity fast insertion
    test_reserve_with_capacity()
    test_reserve_filters()
    test_add_row_failure_rollback()

    # Append & access
    test_append()
    test_append_complex()
    test_access()
    test_series_return_contract()

    # Copy & pop
    test_copy()
    test_pop_front()
    test_pop_back()
    test_pop_empty()

    # Drop
    test_drop()
    test_drop_with_index()
    test_drop_raw()

    # __getitem__ list, __iter__, __contains__
    test_getitem_list()
    test_getitem_list_3cols()
    test_getitem_list_error()
    test_iter()
    test_contains()

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
    test_2d_count_describe_info()
    test_2d_agg_withcol()

    # I/O
    test_csv()
    test_pickle()
    test_pickle_2d_roundtrip()
    test_standalone_convert_2d()
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
    test_insert_column_2d()
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
    test_take_path_2d()
    test_sort_values_2d()
    test_reindex_2d()
    test_drop_duplicates_2d()
    test_duplicated_2d_rowaware()
    test_display_export_2d()
    test_elementwise_2d()
    test_series_row_2d()
    test_shift_time_cols()

    # Regression: 类别五摘要统计 2D
    test_agg_stats_2d()

    # Aggregation
    test_aggregation()
    test_aggregation_complex()

    # Cumulative
    test_cumulative()

    # Column arithmetic
    test_col_arithmetic()

    # Statistics
    test_corr_cov()
    test_corr_cov_2d()
    test_distance_functions()

    # Column ops
    test_value_counts()
    test_clip_column()
    test_where_column()
    test_mask_column()
    test_where_row()
    test_where_mask_row_2d()

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
