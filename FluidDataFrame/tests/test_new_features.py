import sys
sys.path.insert(0, '.')

from fluidframe.core.fastdataframe import FastDataFrame
from fluidframe.core.intcolumn import IntColumn
from fluidframe.core.doublecolumn import DoubleColumn
from fluidframe.core.char256column import Char256Column_SoA

passed = 0
failed = 0

def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed += 1
        print(f'  FAIL: {name}')

# ============================================================
#  __setitem__
# ============================================================
def test_setitem_list():
    print('--- test_setitem_list ---')
    df = FastDataFrame({'a': int, 'b': float})
    df.append_row(a=1, b=1.0)
    df.append_row(a=2, b=2.0)
    df['c'] = [10, 20]
    check('cols', set(df.columns.keys()) == {'a', 'b', 'c'})
    check('c[0]', df['c'][0] == 10)
    check('c[1]', df['c'][1] == 20)

def test_setitem_scalar_int():
    print('--- test_setitem_scalar_int ---')
    df = FastDataFrame({'a': int})
    df.append_row(a=1)
    df.append_row(a=2)
    df['b'] = 99
    check('b[0]', df['b'][0] == 99)
    check('b[1]', df['b'][1] == 99)

def test_setitem_scalar_float():
    print('--- test_setitem_scalar_float ---')
    df = FastDataFrame({'a': int})
    df.append_row(a=1)
    df['b'] = 3.14
    check('b[0]', df['b'][0] == 3.14)

def test_setitem_column_obj():
    print('--- test_setitem_column_obj ---')
    df = FastDataFrame({'a': int})
    df.append_row(a=1)
    col = IntColumn(1)
    col.append(42)
    df['b'] = col
    check('b[0]', df['b'][0] == 42)

def test_setitem_length_mismatch():
    print('--- test_setitem_length_mismatch ---')
    df = FastDataFrame({'a': int})
    df.append_row(a=1)
    ok = False
    try:
        df['b'] = [1, 2, 3]
    except ValueError:
        ok = True
    check('raises ValueError', ok)

# ============================================================
#  insert_column
# ============================================================
def test_insert_column_at_pos():
    print('--- test_insert_column_at_pos ---')
    df = FastDataFrame({'a': int, 'c': int})
    df.append_row(a=1, c=3)
    df.insert_column('b', [2], pos=1)
    check('order', list(df.columns.keys()) == ['a', 'b', 'c'])
    check('b[0]', df['b'][0] == 2)

def test_insert_column_end():
    print('--- test_insert_column_end ---')
    df = FastDataFrame({'a': int})
    df.append_row(a=1)
    df.insert_column('b', [10])
    check('order', list(df.columns.keys()) == ['a', 'b'])
    check('b[0]', df['b'][0] == 10)

# ============================================================
#  insert_row
# ============================================================
def test_insert_row_beginning():
    print('--- test_insert_row_beginning ---')
    df = FastDataFrame({'a': int, 'b': float})
    df.append_row(a=2, b=2.0)
    df.append_row(a=3, b=3.0)
    df.insert_row(0, a=1, b=1.0)
    check('length', df.df_length == 3)
    check('a[0]', df['a'][0] == 1)
    check('a[1]', df['a'][1] == 2)
    check('a[2]', df['a'][2] == 3)

def test_insert_row_middle():
    print('--- test_insert_row_middle ---')
    df = FastDataFrame({'a': int})
    df.append_row(a=1)
    df.append_row(a=3)
    df.insert_row(1, a=2)
    check('a[0]', df['a'][0] == 1)
    check('a[1]', df['a'][1] == 2)
    check('a[2]', df['a'][2] == 3)

def test_insert_row_end():
    print('--- test_insert_row_end ---')
    df = FastDataFrame({'a': int})
    df.append_row(a=1)
    df.insert_row(1, a=2)
    check('a[1]', df['a'][1] == 2)
    check('length', df.df_length == 2)

# ============================================================
#  duplicate_column
# ============================================================
def test_duplicate_column():
    print('--- test_duplicate_column ---')
    df = FastDataFrame({'a': int})
    df.append_row(a=10)
    df.duplicate_column('a', 'a_copy')
    check('cols', 'a_copy' in df.columns)
    check('val', df['a_copy'][0] == 10)

def test_duplicate_column_auto_name():
    print('--- test_duplicate_column_auto_name ---')
    df = FastDataFrame({'a': int})
    df.append_row(a=10)
    df.duplicate_column('a')
    check('auto name', 'a_dup' in df.columns)

# ============================================================
#  duplicate_rows
# ============================================================
def test_duplicate_rows():
    print('--- test_duplicate_rows ---')
    df = FastDataFrame({'a': int})
    df.append_row(a=1)
    df.append_row(a=2)
    dup = df.duplicate_rows([0, 0, 1])
    check('length', dup.df_length == 3)
    check('val', [dup['a'][i] for i in range(3)] == [1, 1, 2])

def test_duplicate_rows_all():
    print('--- test_duplicate_rows_all ---')
    df = FastDataFrame({'a': int})
    df.append_row(a=1)
    dup = df.duplicate_rows()
    check('is copy', dup['a'][0] == 1)

# ============================================================
#  T (transpose)
# ============================================================
def test_transpose():
    print('--- test_transpose ---')
    df = FastDataFrame({'a': int, 'b': int})
    df.append_row(a=1, b=3)
    df.append_row(a=2, b=4)
    t = df.T
    check('rows', t.df_length == 2)
    check('cols', len(t.columns) == 2)
    check('t[a][0]', t['a'][0] == 1.0)
    check('t[a][1]', t['a'][1] == 3.0)
    check('t[b][0]', t['b'][0] == 2.0)
    check('t[b][1]', t['b'][1] == 4.0)

# ============================================================
#  reindex
# ============================================================
def test_reindex():
    print('--- test_reindex ---')
    df = FastDataFrame({'a': int, 'b': float})
    df.append_row(a=10, b=1.0)
    df.append_row(a=20, b=2.0)
    df.append_row(a=30, b=3.0)
    ri = df.reindex([0, 2])
    check('length', ri.df_length == 2)
    check('a[0]', ri['a'][0] == 10)
    check('a[1]', ri['a'][1] == 30)

# ============================================================
#  duplicated / drop_duplicates
# ============================================================
def test_duplicated():
    print('--- test_duplicated ---')
    df = FastDataFrame({'a': int, 'b': int})
    df.append_row(a=1, b=10)
    df.append_row(a=1, b=20)
    df.append_row(a=2, b=10)
    dup = df.duplicated()
    check('dup[0]', dup[0] == 0)
    check('dup[1]', dup[1] == 0)
    check('dup[2]', dup[2] == 0)

def test_duplicated_with_dupes():
    print('--- test_duplicated_with_dupes ---')
    df = FastDataFrame({'a': int})
    df.append_row(a=1)
    df.append_row(a=1)
    df.append_row(a=2)
    dup = df.duplicated()
    check('dup[0]', dup[0] == 0)
    check('dup[1]', dup[1] == 1)
    check('dup[2]', dup[2] == 0)

def test_drop_duplicates():
    print('--- test_drop_duplicates ---')
    df = FastDataFrame({'a': int})
    df.append_row(a=1)
    df.append_row(a=1)
    df.append_row(a=2)
    result = df.drop_duplicates()
    check('length', result.df_length == 2)
    check('val', [result['a'][i] for i in range(2)] == [1, 2])

def test_drop_duplicates_subset():
    print('--- test_drop_duplicates_subset ---')
    df = FastDataFrame({'a': int, 'b': int})
    df.append_row(a=1, b=10)
    df.append_row(a=1, b=20)
    df.append_row(a=2, b=10)
    result = df.drop_duplicates(subset=['a'])
    check('length', result.df_length == 2)

# ============================================================
#  merge
# ============================================================
def test_merge_inner():
    print('--- test_merge_inner ---')
    left = FastDataFrame({'key': int, 'val_l': int})
    left.append_row(key=1, val_l=10)
    left.append_row(key=2, val_l=20)
    left.append_row(key=3, val_l=30)
    right = FastDataFrame({'key': int, 'val_r': int})
    right.append_row(key=2, val_r=200)
    right.append_row(key=3, val_r=300)
    right.append_row(key=4, val_r=400)
    result = left.merge(right, on='key', how='inner')
    check('length', result.df_length == 2)
    check('cols', set(result.columns.keys()) == {'key', 'val_l', 'val_r'})

def test_merge_left():
    print('--- test_merge_left ---')
    left = FastDataFrame({'key': int, 'val_l': int})
    left.append_row(key=1, val_l=10)
    left.append_row(key=2, val_l=20)
    right = FastDataFrame({'key': int, 'val_r': int})
    right.append_row(key=2, val_r=200)
    result = left.merge(right, on='key', how='left')
    check('length', result.df_length == 2)

def test_merge_outer():
    print('--- test_merge_outer ---')
    left = FastDataFrame({'key': int, 'val_l': int})
    left.append_row(key=1, val_l=10)
    left.append_row(key=2, val_l=20)
    right = FastDataFrame({'key': int, 'val_r': int})
    right.append_row(key=2, val_r=200)
    right.append_row(key=3, val_r=300)
    result = left.merge(right, on='key', how='outer')
    check('length', result.df_length == 3)

# ============================================================
#  sample
# ============================================================
def test_sample_n():
    print('--- test_sample_n ---')
    df = FastDataFrame.arange(100)
    s = df.sample(n=10, seed=42)
    check('length', s.df_length == 10)

def test_sample_frac():
    print('--- test_sample_frac ---')
    df = FastDataFrame.arange(100)
    s = df.sample(frac=0.1, seed=42)
    check('length', s.df_length == 10)

def test_sample_replace():
    print('--- test_sample_replace ---')
    df = FastDataFrame.arange(5)
    s = df.sample(n=10, replace=True, seed=42)
    check('length', s.df_length == 10)

# ============================================================
#  nsmallest / nlargest
# ============================================================
def test_nsmallest():
    print('--- test_nsmallest ---')
    df = FastDataFrame({'a': int})
    for v in [5, 3, 1, 4, 2]:
        df.append_row(a=v)
    result = df.nsmallest(3, 'a')
    check('length', result.df_length == 3)
    check('val', [result['a'][i] for i in range(3)] == [1, 2, 3])

def test_nlargest():
    print('--- test_nlargest ---')
    df = FastDataFrame({'a': int})
    for v in [5, 3, 1, 4, 2]:
        df.append_row(a=v)
    result = df.nlargest(3, 'a')
    check('length', result.df_length == 3)
    check('val', [result['a'][i] for i in range(3)] == [5, 4, 3])

# ============================================================
#  pipe / assign
# ============================================================
def test_pipe():
    print('--- test_pipe ---')
    df = FastDataFrame({'a': int})
    df.append_row(a=1)
    result = df.pipe(lambda d: d.head(1))
    check('length', result.df_length == 1)

def test_assign():
    print('--- test_assign ---')
    df = FastDataFrame({'a': int})
    df.append_row(a=1)
    df.append_row(a=2)
    result = df.assign(b=[10, 20])
    check('cols', 'b' in result.columns)
    check('b[0]', result['b'][0] == 10)
    check('b[1]', result['b'][1] == 20)
    check('original unchanged', 'b' not in df.columns)

# ============================================================
#  corr / cov
# ============================================================
def test_corr():
    print('--- test_corr ---')
    df = FastDataFrame({'a': int, 'b': int})
    df.append_row(a=1, b=2)
    df.append_row(a=2, b=4)
    df.append_row(a=3, b=6)
    c = df.corr()
    cols = list(c.columns.keys())
    ai = cols.index('a')
    bi = cols.index('b')
    check('self corr', c['a'][ai] == 1.0)
    check('perfect corr', c['a'][bi] == 1.0)

def test_cov():
    print('--- test_cov ---')
    df = FastDataFrame({'a': int, 'b': int})
    df.append_row(a=1, b=2)
    df.append_row(a=2, b=4)
    df.append_row(a=3, b=6)
    c = df.cov()
    cols = list(c.columns.keys())
    ai = cols.index('a')
    check('self cov > 0', c['a'][ai] > 0)

# ============================================================
#  value_counts / nunique
# ============================================================
def test_value_counts():
    print('--- test_value_counts ---')
    df = FastDataFrame({'a': int})
    for v in [1, 2, 1, 3, 1]:
        df.append_row(a=v)
    vc = df.value_counts('a')
    check('length', vc.df_length == 3)
    check('count col', 'count' in vc.columns)

def test_nunique():
    print('--- test_nunique ---')
    df = FastDataFrame({'a': int})
    for v in [1, 2, 1, 3, 1]:
        df.append_row(a=v)
    check('nunique', df.nunique('a') == 3)

# ============================================================
#  clip / where / mask
# ============================================================
def test_clip():
    print('--- test_clip ---')
    df = FastDataFrame({'a': int})
    for v in [1, 5, 10, 15, 20]:
        df.append_row(a=v)
    df.clip_column('a', 5, 15)
    check('clipped', [df['a'][i] for i in range(5)] == [5, 5, 10, 15, 15])

def test_where():
    print('--- test_where ---')
    df = FastDataFrame({'a': int, 'b': int})
    df.append_row(a=1, b=1)
    df.append_row(a=2, b=0)
    df.append_row(a=3, b=1)
    result = df.where_column('a', 'b', 0)
    check('where', [result[i] for i in range(3)] == [1, 0, 3])

def test_mask():
    print('--- test_mask ---')
    df = FastDataFrame({'a': int, 'b': int})
    df.append_row(a=1, b=1)
    df.append_row(a=2, b=0)
    df.append_row(a=3, b=1)
    result = df.mask_column('a', 'b', 0)
    check('mask', [result[i] for i in range(3)] == [0, 2, 0])

# ============================================================
#  sort convenience aliases
# ============================================================
def test_sort_values_by():
    print('--- test_sort_values_by ---')
    df = FastDataFrame({'a': int})
    for v in [3, 1, 2]:
        df.append_row(a=v)
    result = df.sort_values('a')
    check('sorted', [result['a'][i] for i in range(3)] == [1, 2, 3])

def test_sort_values_inplace():
    print('--- test_sort_values_inplace ---')
    df = FastDataFrame({'a': int})
    for v in [3, 1, 2]:
        df.append_row(a=v)
    df.sort_values('a', inplace=True)
    check('sorted', [df['a'][i] for i in range(3)] == [1, 2, 3])

# ============================================================
#  info / memory_usage
# ============================================================
def test_info():
    print('--- test_info ---')
    df = FastDataFrame({'a': int, 'b': float})
    df.append_row(a=1, b=1.0)
    info = df.info()
    check('has rows', '1 entries' in info)
    check('has cols', '2 columns' in info)

def test_memory_usage():
    print('--- test_memory_usage ---')
    df = FastDataFrame({'a': int})
    df.append_row(a=1)
    mu = df.memory_usage()
    check('positive', mu > 0)

# ============================================================
#  apply
# ============================================================
def test_apply_columns():
    print('--- test_apply_columns ---')
    df = FastDataFrame({'a': int})
    df.append_row(a=1)
    df.append_row(a=2)
    result = df.apply(lambda col: col)
    check('length', result.df_length == 2)

def test_apply_index():
    print('--- test_apply_index ---')
    df = FastDataFrame({'a': int, 'b': int})
    df.append_row(a=1, b=10)
    result = df.apply(lambda row: row['a'] + row['b'], axis='index')
    check('sum', result[0] == 11)

# ============================================================
#  select_dtypes
# ============================================================
def test_select_dtypes():
    print('--- test_select_dtypes ---')
    df = FastDataFrame({'a': int, 'b': float, 'c': int})
    df.append_row(a=1, b=1.0, c=3)
    result = df.select_dtypes(include_types=['int'])
    check('cols', set(result.columns.keys()) == {'a', 'c'})

# ============================================================
#  P0/P1: df[mask], isin, between, loc[rows, cols]
# ============================================================
def test_df_boolean_indexing():
    print('--- test_df_boolean_indexing ---')
    df = FastDataFrame({'id': int, 'score': float})
    for i in range(10):
        df.append_row(id=i, score=50.0 + i * 5)
    mask = df['id'] > 5
    result = df[mask]
    check('df[mask] shape', result.shape == (4, 2))
    check('df[mask] first', result['id'][0] == 6)
    check('df[mask] last', result['id'][3] == 9)

def test_df_multi_condition():
    print('--- test_df_multi_condition ---')
    df = FastDataFrame({'id': int, 'score': float})
    for i in range(10):
        df.append_row(id=i, score=50.0 + i * 5)
    mask = (df['id'] > 3) & (df['score'] < 80.0)
    result = df[mask]
    check('multi cond shape', result.shape == (2, 2))
    check('multi cond val', result['id'][0] == 4)

def test_isin_int():
    print('--- test_isin_int ---')
    df = FastDataFrame({'a': int})
    for i in range(5):
        df.append_row(a=i)
    mask = df['a'].isin({1, 3})
    result = df[mask]
    check('isin int shape', result.shape == (2, 1))
    check('isin int val', result['a'][0] == 1)

def test_isin_double():
    print('--- test_isin_double ---')
    df = FastDataFrame({'a': float})
    df.append_row(a=1.0)
    df.append_row(a=2.5)
    df.append_row(a=3.0)
    mask = df['a'].isin({1.0, 3.0})
    result = df[mask]
    check('isin double shape', result.shape == (2, 1))
    check('isin double val', result['a'][0] == 1.0)

def test_isin_string():
    print('--- test_isin_string ---')
    df = FastDataFrame({'a': str})
    df.append_row(a='alice')
    df.append_row(a='bob')
    df.append_row(a='carol')
    mask = df['a'].isin({'alice', 'carol'})
    result = df[mask]
    check('isin str shape', result.shape == (2, 1))
    check('isin str val', result['a'][0] == 'alice')

def test_between_int():
    print('--- test_between_int ---')
    df = FastDataFrame({'a': int})
    for i in range(10):
        df.append_row(a=i)
    mask = df['a'].between(3, 7)
    result = df[mask]
    check('between int shape', result.shape == (5, 1))
    check('between int first', result['a'][0] == 3)
    check('between int last', result['a'][4] == 7)

def test_between_double():
    print('--- test_between_double ---')
    df = FastDataFrame({'a': float})
    df.append_row(a=1.0)
    df.append_row(a=2.5)
    df.append_row(a=3.0)
    df.append_row(a=4.5)
    mask = df['a'].between(2.0, 4.0)
    result = df[mask]
    check('between double shape', result.shape == (2, 1))
    check('between double val', result['a'][0] == 2.5)

def test_loc_rows_cols():
    print('--- test_loc_rows_cols ---')
    df = FastDataFrame({'name': str, 'age': int, 'score': float})
    df.append_row(name='alice', age=25, score=90.0)
    df.append_row(name='bob', age=30, score=85.0)
    df.append_row(name='carol', age=35, score=95.0)
    mask = df['age'] > 25
    result = df.loc[mask, ['name', 'score']]
    check('loc rows+cols shape', result.shape == (2, 2))
    check('loc rows+cols has name', 'name' in result.columns)
    check('loc rows+cols has score', 'score' in result.columns)
    check('loc rows+cols no age', 'age' not in result.columns)
    check('loc rows+cols val', result['name'][0] == 'bob')

def test_loc_single_col():
    print('--- test_loc_single_col ---')
    df = FastDataFrame({'name': str, 'age': int})
    df.append_row(name='alice', age=25)
    df.append_row(name='bob', age=30)
    mask = df['age'] > 25
    col = df.loc[mask, 'age']
    check('loc single col type', col.length == 1)
    check('loc single col val', col[0] == 30)

def test_combined_query():
    print('--- test_combined_query ---')
    df = FastDataFrame({'id': int, 'score': float, 'name': str})
    for i in range(10):
        df.append_row(id=i, score=50.0 + i * 5, name='alice' if i % 2 == 0 else 'bob')
    mask = (df['id'].between(2, 7)) & (df['name'].isin({'alice'}))
    result = df[mask]
    check('combined shape', result.shape[1] == 3)
    check('combined all alice', all(result['name'][i] == 'alice' for i in range(result.shape[0])))

# ============================================================
#  where_row, where_index, where_all, groupby idxmax/idxmin
# ============================================================
def test_where_row_int():
    print('--- test_where_row_int ---')
    df = FastDataFrame({'a': int, 'b': int})
    df.append_row(a=1, b=10)
    df.append_row(a=2, b=20)
    df.append_row(a=3, b=30)
    mask = df['a'] > 1
    result = df.where_row(mask, fill=0)
    check('where_row shape', result.shape == (3, 2))
    check('where_row match', result['b'][0] == 0)
    check('where_row keep', result['b'][1] == 20)
    check('where_row keep2', result['b'][2] == 30)

def test_where_row_float():
    print('--- test_where_row_float ---')
    df = FastDataFrame({'a': int, 'b': float})
    df.append_row(a=1, b=1.5)
    df.append_row(a=2, b=2.5)
    mask = df['a'] > 1
    result = df.where_row(mask, fill=-1.0)
    check('where_row float', result['b'][0] == -1.0)
    check('where_row float keep', result['b'][1] == 2.5)

def test_where_index():
    print('--- test_where_index ---')
    df = FastDataFrame({'a': int, 'b': int})
    df.append_row(a=1, b=10)
    df.append_row(a=2, b=20)
    df.append_row(a=3, b=30)
    mask = df['a'] > 1
    result = df.loc[mask]
    check('where_index shape', result.shape == (2, 2))
    check('where_index val', result['a'][0] == 2)

def test_where_all():
    print('--- test_where_all ---')
    df = FastDataFrame({'a': int, 'b': float})
    df.append_row(a=1, b=10.0)
    df.append_row(a=2, b=20.0)
    df.append_row(a=3, b=30.0)
    mask = df['a'] > 1
    result = df.where_row(mask, fill=0)
    check('where_all shape', result.shape == (3, 2))
    check('where_all fill a', result['a'][0] == 0)
    check('where_all fill b', result['b'][0] == 0.0)
    check('where_all keep', result['a'][1] == 2)

def test_groupby_idxmax():
    print('--- test_groupby_idxmax ---')
    df = FastDataFrame({'dept': str, 'age': int})
    df.append_row(dept='IT', age=25)
    df.append_row(dept='IT', age=30)
    df.append_row(dept='HR', age=28)
    df.append_row(dept='HR', age=35)
    result = df.groupby('dept').idxmax('age')
    check('idxmax shape', result.shape == (2, 2))
    # idxmax returns row index of max value per group
    ages = {result['dept'][i]: result['age'][i] for i in range(2)}
    check('idxmax IT=1', ages['IT'] == 1)
    check('idxmax HR=3', ages['HR'] == 3)

def test_groupby_idxmin():
    print('--- test_groupby_idxmin ---')
    df = FastDataFrame({'dept': str, 'age': int})
    df.append_row(dept='IT', age=25)
    df.append_row(dept='IT', age=30)
    df.append_row(dept='HR', age=28)
    df.append_row(dept='HR', age=35)
    result = df.groupby('dept').idxmin('age')
    check('idxmin shape', result.shape == (2, 2))
    ages = {result['dept'][i]: result['age'][i] for i in range(2)}
    check('idxmin IT=0', ages['IT'] == 0)
    check('idxmin HR=2', ages['HR'] == 2)

def test_groupby_idxmax_all():
    print('--- test_groupby_idxmax_all ---')
    df = FastDataFrame({'dept': str, 'age': int, 'salary': float})
    df.append_row(dept='IT', age=25, salary=50000.0)
    df.append_row(dept='IT', age=30, salary=60000.0)
    df.append_row(dept='HR', age=28, salary=55000.0)
    result = df.groupby('dept').idxmax()
    check('idxmax all cols', 'age' in result.columns and 'salary' in result.columns)

# ============================================================
#  Column arithmetic (add_cols / sub_cols / mul_cols / div_cols)
# ============================================================
def test_add_cols_int():
    print('--- test_add_cols_int ---')
    df = FastDataFrame({'A': int, 'B': int})
    df.append_row(A=1, B=10)
    df.append_row(A=2, B=20)
    df.append_row(A=3, B=30)
    result = df.add(('A', 'B'))
    check('dtype', result.dtype == 0)
    check('len', result.length == 3)
    check('val0', result[0] == 11)
    check('val1', result[1] == 22)
    check('val2', result[2] == 33)
    df['C'] = result
    check('assign', df['C'][1] == 22)

def test_sub_cols_int():
    print('--- test_sub_cols_int ---')
    df = FastDataFrame({'A': int, 'B': int})
    df.append_row(A=10, B=3)
    df.append_row(A=20, B=7)
    result = df.sub(('A', 'B'))
    check('dtype', result.dtype == 0)
    check('val0', result[0] == 7)
    check('val1', result[1] == 13)

def test_mul_cols_int():
    print('--- test_mul_cols_int ---')
    df = FastDataFrame({'A': int, 'B': int})
    df.append_row(A=3, B=4)
    df.append_row(A=5, B=6)
    result = df.mul(('A', 'B'))
    check('dtype', result.dtype == 0)
    check('val0', result[0] == 12)
    check('val1', result[1] == 30)

def test_div_cols_int():
    print('--- test_div_cols_int ---')
    df = FastDataFrame({'A': int, 'B': int})
    df.append_row(A=10, B=2)
    df.append_row(A=20, B=4)
    result = df.div(('A', 'B'))
    check('dtype', result.dtype == 2)
    check('val0', result[0] == 5.0)
    check('val1', result[1] == 5.0)

def test_add_cols_mixed():
    print('--- test_add_cols_mixed ---')
    df = FastDataFrame({'A': int, 'B': float})
    df.append_row(A=1, B=2.5)
    df.append_row(A=2, B=3.5)
    result = df.add(('A', 'B'))
    check('dtype', result.dtype == 2)
    check('val0', result[0] == 3.5)
    check('val1', result[1] == 5.5)

def test_add_cols_three():
    print('--- test_add_cols_three ---')
    df = FastDataFrame({'A': int, 'B': int, 'C': int})
    df.append_row(A=1, B=10, C=100)
    df.append_row(A=2, B=20, C=200)
    result = df.add(('A', 'B', 'C'))
    check('val0', result[0] == 111)
    check('val1', result[1] == 222)

# ============================================================
#  _LocIndexer.__setitem__
# ============================================================
def test_loc_setitem_int():
    print('--- test_loc_setitem_int ---')
    df = FastDataFrame({'A': int, 'B': int})
    df.append_row(A=1, B=10)
    df.append_row(A=2, B=20)
    df.append_row(A=3, B=30)
    df.loc[1] = [99, 88]
    check('A', df['A'][1] == 99)
    check('B', df['B'][1] == 88)
    check('unchanged_A', df['A'][0] == 1)
    check('unchanged_B', df['B'][2] == 30)

def test_loc_setitem_dict():
    print('--- test_loc_setitem_dict ---')
    df = FastDataFrame({'A': int, 'B': int})
    df.append_row(A=1, B=10)
    df.append_row(A=2, B=20)
    df.loc[0] = {'A': 77}
    check('A', df['A'][0] == 77)
    check('B_unchanged', df['B'][0] == 10)

# ============================================================
#  append
# ============================================================
def test_append_dict():
    print('--- test_append_dict ---')
    df = FastDataFrame({'A': int, 'B': int})
    df.append_row(A=1, B=10)
    df2 = df.append({'A': 2, 'B': 20})
    check('len', df2.df_length == 2)
    check('orig_len', df.df_length == 1)
    check('val0_A', df2['A'][0] == 1)
    check('val1_A', df2['A'][1] == 2)
    check('val1_B', df2['B'][1] == 20)

def test_append_list():
    print('--- test_append_list ---')
    df = FastDataFrame({'A': int, 'B': int})
    df.append_row(A=1, B=10)
    df2 = df.append([2, 20])
    check('len', df2.df_length == 2)
    check('val1_A', df2['A'][1] == 2)
    check('val1_B', df2['B'][1] == 20)

def test_append_ignore_index():
    print('--- test_append_ignore_index ---')
    df = FastDataFrame({'A': int, 'B': int})
    df.append_row(A=1, B=10)
    df.append_row(A=2, B=20)
    df2 = df.append({'A': 3, 'B': 30}, ignore_index=True)
    idx_vals = df2.index.values()
    check('idx0', idx_vals[0] == 0)
    check('idx1', idx_vals[1] == 1)
    check('idx2', idx_vals[2] == 2)

# ============================================================
#  concat
# ============================================================
def test_concat_basic():
    print('--- test_concat_basic ---')
    from fluidframe.core.fluidframe import concat
    df1 = FastDataFrame({'A': int, 'B': int})
    df1.append_row(A=1, B=10)
    df1.append_row(A=2, B=20)
    df2 = FastDataFrame({'A': int, 'B': int})
    df2.append_row(A=3, B=30)
    result = concat([df1, df2])
    check('len', result.df_length == 3)
    check('val0', result['A'][0] == 1)
    check('val2', result['A'][2] == 3)
    check('orig1_len', df1.df_length == 2)
    check('orig2_len', df2.df_length == 1)

def test_concat_ignore_index():
    print('--- test_concat_ignore_index ---')
    from fluidframe.core.fluidframe import concat
    df1 = FastDataFrame({'A': int, 'B': int})
    df1.append_row(A=1, B=10)
    df2 = FastDataFrame({'A': int, 'B': int})
    df2.append_row(A=2, B=20)
    result = concat([df1, df2], ignore_index=True)
    check('len', result.df_length == 2)
    idx_vals = result.index.values()
    check('idx0', idx_vals[0] == 0)
    check('idx1', idx_vals[1] == 1)

def run_all():
    test_setitem_list()
    test_setitem_scalar_int()
    test_setitem_scalar_float()
    test_setitem_column_obj()
    test_setitem_length_mismatch()
    test_insert_column_at_pos()
    test_insert_column_end()
    test_insert_row_beginning()
    test_insert_row_middle()
    test_insert_row_end()
    test_duplicate_column()
    test_duplicate_column_auto_name()
    test_duplicate_rows()
    test_duplicate_rows_all()
    test_transpose()
    test_reindex()
    test_duplicated()
    test_duplicated_with_dupes()
    test_drop_duplicates()
    test_drop_duplicates_subset()
    test_merge_inner()
    test_merge_left()
    test_merge_outer()
    test_sample_n()
    test_sample_frac()
    test_sample_replace()
    test_nsmallest()
    test_nlargest()
    test_pipe()
    test_assign()
    test_corr()
    test_cov()
    test_value_counts()
    test_nunique()
    test_clip()
    test_where()
    test_mask()
    test_sort_values_by()
    test_sort_values_inplace()
    test_info()
    test_memory_usage()
    test_apply_columns()
    test_apply_index()
    test_select_dtypes()
    test_df_boolean_indexing()
    test_df_multi_condition()
    test_isin_int()
    test_isin_double()
    test_isin_string()
    test_between_int()
    test_between_double()
    test_loc_rows_cols()
    test_loc_single_col()
    test_combined_query()
    test_where_row_int()
    test_where_row_float()
    test_where_index()
    test_where_all()
    test_groupby_idxmax()
    test_groupby_idxmin()
    test_groupby_idxmax_all()
    test_add_cols_int()
    test_sub_cols_int()
    test_mul_cols_int()
    test_div_cols_int()
    test_add_cols_mixed()
    test_add_cols_three()
    test_loc_setitem_int()
    test_loc_setitem_dict()
    test_append_dict()
    test_append_list()
    test_append_ignore_index()
    test_concat_basic()
    test_concat_ignore_index()
    print()
    print(f'Results: {passed} passed, {failed} failed')
    if failed > 0:
        sys.exit(1)

if __name__ == '__main__':
    run_all()
