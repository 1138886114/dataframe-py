import sys
sys.path.insert(0, '.')

from fluidframe.core.dataframe import DataFrame
from fluidframe.core.intcolumn import IntColumn
from fluidframe.core.doublecolumn import DoubleColumn

passed = 0
failed = 0

def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed += 1
        print(f'  FAIL: {name}')

def sub_2d(col, sub, i):
    """读取 2D 列 col 第 i 行行键元组的第 sub 个子列（A2 行元组语义：col[i][sub]）"""
    return col[i][sub]

def sample_var(vals):
    m = sum(vals) / len(vals)
    return sum((x - m) * (x - m) for x in vals) / (len(vals) - 1)

def make_df():
    df = DataFrame({'name': str, 'amount': float, 'qty': int})
    df.append_row(name='Alice', amount=10.0, qty=1)
    df.append_row(name='Bob', amount=20.0, qty=2)
    df.append_row(name='Alice', amount=30.0, qty=3)
    df.append_row(name='Bob', amount=40.0, qty=4)
    df.append_row(name='Alice', amount=50.0, qty=5)
    return df

def test_groupby_basic():
    print('--- test_groupby_basic ---')
    df = make_df()
    gb = df.groupby('name')
    check('size', gb.size() == 2)
    check('repr', '2 groups' in repr(gb))

def test_groupby_sum():
    print('--- test_groupby_sum ---')
    df = make_df()
    res = df.groupby('name').sum('amount')
    check('length', res.df_length == 2)
    amounts = [res['amount'][i] for i in range(2)]
    names = [res['name'][i] for i in range(2)]
    check('alice_sum', amounts[names.index('Alice')] == 90.0)
    check('bob_sum', amounts[names.index('Bob')] == 60.0)

def test_groupby_mean():
    print('--- test_groupby_mean ---')
    df = make_df()
    res = df.groupby('name').mean('amount')
    amounts = [res['amount'][i] for i in range(2)]
    names = [res['name'][i] for i in range(2)]
    check('alice_mean', amounts[names.index('Alice')] == 30.0)
    check('bob_mean', amounts[names.index('Bob')] == 30.0)

def test_groupby_count():
    print('--- test_groupby_count ---')
    df = make_df()
    res = df.groupby('name').count()
    counts = [res['__count__'][i] for i in range(2)]
    names = [res['name'][i] for i in range(2)]
    check('alice_count', counts[names.index('Alice')] == 3)
    check('bob_count', counts[names.index('Bob')] == 2)

def test_groupby_min_max():
    print('--- test_groupby_min_max ---')
    df = make_df()
    res_min = df.groupby('name').min('amount')
    res_max = df.groupby('name').max('amount')
    names_min = [res_min['name'][i] for i in range(2)]
    amounts_min = [res_min['amount'][i] for i in range(2)]
    amounts_max = [res_max['amount'][i] for i in range(2)]
    check('alice_min', amounts_min[names_min.index('Alice')] == 10.0)
    check('bob_min', amounts_min[names_min.index('Bob')] == 20.0)
    check('alice_max', amounts_max[names_min.index('Alice')] == 50.0)
    check('bob_max', amounts_max[names_min.index('Bob')] == 40.0)

def test_groupby_agg():
    print('--- test_groupby_agg ---')
    df = make_df()
    res = df.groupby('name').agg({'amount': 'sum', 'qty': 'mean'})
    check('length', res.df_length == 2)
    names = [res['name'][i] for i in range(2)]
    amounts = [res['amount'][i] for i in range(2)]
    qtys = [res['qty'][i] for i in range(2)]
    check('alice_sum', amounts[names.index('Alice')] == 90.0)
    check('bob_sum', amounts[names.index('Bob')] == 60.0)
    check('alice_mean_qty', qtys[names.index('Alice')] == 3.0)
    check('bob_mean_qty', qtys[names.index('Bob')] == 3.0)

def test_groupby_auto_agg():
    print('--- test_groupby_auto_agg ---')
    df = make_df()
    res = df.groupby('name').sum()
    check('length', res.df_length == 2)
    names = [res['name'][i] for i in range(2)]
    amounts = [res['amount'][i] for i in range(2)]
    qtys = [res['qty'][i] for i in range(2)]
    check('amount_sum', amounts[names.index('Alice')] == 90.0)
    check('qty_sum', qtys[names.index('Alice')] == 9)

def test_groupby_multi_col():
    print('--- test_groupby_multi_col ---')
    df = DataFrame({'name': str, 'dept': str, 'salary': float})
    df.append_row(name='Alice', dept='eng', salary=100)
    df.append_row(name='Bob', dept='eng', salary=200)
    df.append_row(name='Alice', dept='hr', salary=150)
    df.append_row(name='Bob', dept='hr', salary=250)
    df.append_row(name='Alice', dept='eng', salary=300)

    res = df.groupby(['name', 'dept']).sum('salary')
    check('length', res.df_length == 4)
    keys = [(res['name'][i], res['dept'][i]) for i in range(4)]
    vals = [res['salary'][i] for i in range(4)]
    idx = keys.index(('Alice', 'eng'))
    check('alice_eng_sum', vals[idx] == 400.0)
    idx2 = keys.index(('Bob', 'hr'))
    check('bob_hr_sum', vals[idx2] == 250.0)

def test_groupby_apply():
    print('--- test_groupby_apply ---')
    df = make_df()
    def count_rows(sub_df):
        res = DataFrame({'name': str, 'total': int})
        res.append_row(name=sub_df['name'][0], total=sub_df['name'].length)
        return res
    res = df.groupby('name').apply(count_rows)
    check('length', res.df_length == 2)
    names = [res['name'][i] for i in range(2)]
    totals = [res['total'][i] for i in range(2)]
    check('alice_total', totals[names.index('Alice')] == 3)
    check('bob_total', totals[names.index('Bob')] == 2)

def test_groupby_empty():
    print('--- test_groupby_empty ---')
    df = DataFrame({'name': str, 'amount': float})
    gb = df.groupby('name')
    check('empty_size', gb.size() == 0)
    res = gb.sum('amount')
    check('empty_result', res.df_length == 0)

def test_groupby_chaining():
    print('--- test_groupby_chaining ---')
    df = make_df()
    res = df.groupby('name').sum('amount')
    check('chain_length', res.df_length == 2)
    amounts = [res['amount'][i] for i in range(2)]
    check('chain_sum', sum(amounts) == 150.0)

def test_groupby_int_column():
    print('--- test_groupby_int_column ---')
    df = DataFrame({'cat': int, 'val': float})
    df.append_row(cat=1, val=10.0)
    df.append_row(cat=2, val=20.0)
    df.append_row(cat=1, val=30.0)
    df.append_row(cat=2, val=40.0)
    res = df.groupby('cat').sum('val')
    check('length', res.df_length == 2)
    cats = [res['cat'][i] for i in range(2)]
    vals = [res['val'][i] for i in range(2)]
    check('cat1_sum', vals[cats.index(1)] == 40.0)
    check('cat2_sum', vals[cats.index(2)] == 60.0)

def test_groupby_int_key():
    print('--- test_groupby_int_key ---')
    df = DataFrame({'id': int, 'score': float})
    df.append_row(id=10, score=1.0)
    df.append_row(id=20, score=2.0)
    df.append_row(id=10, score=3.0)
    res = df.groupby('id').sum('score')
    ids = [res['id'][i] for i in range(2)]
    scores = [res['score'][i] for i in range(2)]
    check('id10_sum', scores[ids.index(10)] == 4.0)
    check('id20_sum', scores[ids.index(20)] == 2.0)

def test_groupby_var_std():
    print('--- test_groupby_var_std ---')
    df = DataFrame({'name': str, 'val': float})
    df.append_row(name='A', val=1.0)
    df.append_row(name='A', val=3.0)
    df.append_row(name='A', val=5.0)
    df.append_row(name='B', val=2.0)
    df.append_row(name='B', val=4.0)

    res_var = df.groupby('name').var('val')
    res_std = df.groupby('name').std('val')
    names = [res_var['name'][i] for i in range(2)]
    a_idx = names.index('A')
    b_idx = names.index('B')
    # A: [1,3,5] -> var=4.0, std=2.0
    check('A_var', abs(res_var['val'][a_idx] - 4.0) < 1e-10)
    check('A_std', abs(res_std['val'][a_idx] - 2.0) < 1e-10)
    # B: [2,4] -> var=2.0, std=sqrt(2)
    check('B_var', abs(res_var['val'][b_idx] - 2.0) < 1e-10)

def test_groupby_int_column_var_std():
    print('--- test_groupby_int_column_var_std ---')
    df = DataFrame({'cat': int, 'val': int})
    df.append_row(cat=1, val=10)
    df.append_row(cat=1, val=20)
    df.append_row(cat=2, val=5)
    res = df.groupby('cat').std('val')
    cats = [res['cat'][i] for i in range(2)]
    # cat=1: [10,20] -> std=sqrt(50)~7.07
    check('cat1_std', abs(res['val'][cats.index(1)] - 7.071067811865) < 1e-6)

def test_groupby_median():
    print('--- test_groupby_median ---')
    df = DataFrame({'name': str, 'val': float})
    df.append_row(name='A', val=1.0)
    df.append_row(name='A', val=3.0)
    df.append_row(name='A', val=5.0)
    df.append_row(name='B', val=2.0)
    df.append_row(name='B', val=4.0)
    res = df.groupby('name').median('val')
    names = [res['name'][i] for i in range(2)]
    # A: [1,3,5] -> median=3.0
    check('A_median', abs(res['val'][names.index('A')] - 3.0) < 1e-10)
    # B: [2,4] -> median=3.0
    check('B_median', abs(res['val'][names.index('B')] - 3.0) < 1e-10)

def test_groupby_quantile():
    print('--- test_groupby_quantile ---')
    df = DataFrame({'name': str, 'val': float})
    df.append_row(name='A', val=1.0)
    df.append_row(name='A', val=2.0)
    df.append_row(name='A', val=3.0)
    df.append_row(name='A', val=4.0)
    res = df.groupby('name').quantile('val')
    names = [res['name'][i] for i in range(1)]
    # A: [1,2,3,4] -> quantile(median)=2.5
    check('A_quantile', abs(res['val'][names.index('A')] - 2.5) < 1e-10)

def test_groupby_transform():
    print('--- test_groupby_transform ---')
    df = DataFrame({'name': str, 'val': float})
    df.append_row(name='A', val=10.0)
    df.append_row(name='A', val=20.0)
    df.append_row(name='B', val=30.0)
    df.append_row(name='B', val=40.0)
    def normalize(sub_df):
        result = DataFrame({'name': str, 'val': float})
        vals = [sub_df['val'][i] for i in range(sub_df.df_length)]
        mn = min(vals)
        mx = max(vals)
        rng = mx - mn if mx != mn else 1.0
        for i in range(sub_df.df_length):
            result.append_row(name=sub_df['name'][i], val=(sub_df['val'][i] - mn) / rng)
        return result
    res = df.groupby('name').transform(normalize)
    check('length', res.df_length == 4)
    vals = [res['val'][i] for i in range(4)]
    # A: [10,20] -> normalized [0.0, 1.0]
    check('A_norm_0', abs(vals[0] - 0.0) < 1e-10)
    check('A_norm_1', abs(vals[1] - 1.0) < 1e-10)
    # B: [30,40] -> normalized [0.0, 1.0]
    check('B_norm_0', abs(vals[2] - 0.0) < 1e-10)
    check('B_norm_1', abs(vals[3] - 1.0) < 1e-10)

def test_groupby_filter():
    print('--- test_groupby_filter ---')
    df = DataFrame({'name': str, 'val': float})
    df.append_row(name='A', val=10.0)
    df.append_row(name='A', val=20.0)
    df.append_row(name='B', val=5.0)
    df.append_row(name='B', val=15.0)
    df.append_row(name='C', val=50.0)
    # Keep groups where sum(val) > 25
    res = df.groupby('name').filter(lambda sub: sum([sub['val'][i] for i in range(sub.df_length)]) > 25)
    check('length', res.df_length == 3)
    names = [res['name'][i] for i in range(3)]
    check('has_A', 'A' in names)
    check('has_C', 'C' in names)
    check('no_B', 'B' not in names)

def test_groupby_agg_median():
    print('--- test_groupby_agg_median ---')
    df = DataFrame({'name': str, 'val': float})
    df.append_row(name='A', val=1.0)
    df.append_row(name='A', val=5.0)
    df.append_row(name='B', val=2.0)
    df.append_row(name='B', val=8.0)
    res = df.groupby('name').agg({'val': 'median'})
    names = [res['name'][i] for i in range(2)]
    check('A_median', abs(res['val'][names.index('A')] - 3.0) < 1e-10)
    check('B_median', abs(res['val'][names.index('B')] - 5.0) < 1e-10)

# ============================================================
# 2D 列 groupby 语义（完整行向量为键 / 聚合列按子列降维输出）
# ============================================================

def make_2d_df():
    """df_length=5；key 为 1D（退化 2D 元数据）分组键，val 为 1D 聚合列，
    v 为 3 子列 2D 聚合列（列优先）。
    v 子列 0 = [1,10,100,5,8]；子列 1 = [2,20,200,6,9]；子列 2 = [3,30,300,7,10]
    组 key=1 行索引 {0,1,4}，组 key=2 行索引 {2,3}。"""
    df = DataFrame()
    df.df_length = 5
    df._columns['key'] = IntColumn.from_2d([1, 1, 2, 2, 1], 5, 1)
    df._columns['val'] = IntColumn.from_2d([1, 2, 3, 4, 5], 5, 1)
    df._columns['v'] = DoubleColumn.from_2d(
        [1, 10, 100, 5, 8, 2, 20, 200, 6, 9, 3, 30, 300, 7, 10], 5, 3)
    return df

def make_2d_key_df():
    """2D 分组键 key2（子列0=[1,1,2,2,1]，子列1=[1,2,1,2,1]）。
    行键：r0=(1,1) r1=(1,2) r2=(2,1) r3=(2,2) r4=(1,1) → 4 组。"""
    df = DataFrame()
    df.df_length = 5
    df._columns['key2'] = DoubleColumn.from_2d(
        [1, 1, 2, 2, 1, 1, 2, 1, 2, 1], 5, 2)
    df._columns['val'] = IntColumn.from_2d([10, 20, 30, 40, 10], 5, 1)
    return df

def test_groupby_agg_2d_sum():
    print('--- test_groupby_agg_2d_sum ---')
    df = make_2d_df()
    res = df.groupby('key').agg({'v': 'sum'})
    check('agg_2d_sum df_length', res.df_length == 2)
    check('agg_2d_sum key order', res['key'][0] == 1 and res['key'][1] == 2)
    v = res['v']
    check('agg_2d_sum v is_2d', v.is_2d and v.cols == 3 and v.rows == 2)
    check('agg_2d_sum sub0 g0', sub_2d(v, 0, 0) == 19.0)
    check('agg_2d_sum sub1 g0', sub_2d(v, 1, 0) == 31.0)
    check('agg_2d_sum sub2 g0', sub_2d(v, 2, 0) == 43.0)
    check('agg_2d_sum sub0 g1', sub_2d(v, 0, 1) == 105.0)
    check('agg_2d_sum sub1 g1', sub_2d(v, 1, 1) == 206.0)
    check('agg_2d_sum sub2 g1', sub_2d(v, 2, 1) == 307.0)

def test_groupby_agg_2d_mean():
    print('--- test_groupby_agg_2d_mean ---')
    df = make_2d_df()
    res = df.groupby('key').agg({'v': 'mean'})
    v = res['v']
    check('agg_2d_mean sub0 g0', abs(sub_2d(v, 0, 0) - 19.0 / 3.0) < 1e-12)
    check('agg_2d_mean sub2 g0', abs(sub_2d(v, 2, 0) - 43.0 / 3.0) < 1e-12)
    check('agg_2d_mean sub0 g1', sub_2d(v, 0, 1) == 52.5)
    check('agg_2d_mean sub1 g1', sub_2d(v, 1, 1) == 103.0)
    check('agg_2d_mean sub2 g1', sub_2d(v, 2, 1) == 153.5)

def test_groupby_agg_2d_var_std():
    print('--- test_groupby_agg_2d_var_std ---')
    df = make_2d_df()
    res = df.groupby('key').agg({'v': 'var'})
    v = res['v']
    check('agg_2d_var sub0 g0', abs(sub_2d(v, 0, 0) - sample_var([1, 10, 8])) < 1e-9)
    check('agg_2d_var sub1 g0', abs(sub_2d(v, 1, 0) - sample_var([2, 20, 9])) < 1e-9)
    check('agg_2d_var sub2 g0', abs(sub_2d(v, 2, 0) - sample_var([3, 30, 10])) < 1e-9)
    check('agg_2d_var sub0 g1', abs(sub_2d(v, 0, 1) - sample_var([100, 5])) < 1e-9)
    check('agg_2d_var sub1 g1', abs(sub_2d(v, 1, 1) - sample_var([200, 6])) < 1e-9)
    check('agg_2d_var sub2 g1', abs(sub_2d(v, 2, 1) - sample_var([300, 7])) < 1e-9)
    res2 = df.groupby('key').agg({'v': 'std'})
    v2 = res2['v']
    check('agg_2d_std sub0 g0', abs(sub_2d(v2, 0, 0) - sample_var([1, 10, 8]) ** 0.5) < 1e-9)
    check('agg_2d_std sub0 g1', abs(sub_2d(v2, 0, 1) - sample_var([100, 5]) ** 0.5) < 1e-9)

def test_groupby_agg_2d_min_max():
    print('--- test_groupby_agg_2d_min_max ---')
    df = make_2d_df()
    mi = df.groupby('key').agg({'v': 'min'})
    ma = df.groupby('key').agg({'v': 'max'})
    vm, vx = mi['v'], ma['v']
    check('agg_2d_min sub0 g0', sub_2d(vm, 0, 0) == 1.0)
    check('agg_2d_min sub1 g0', sub_2d(vm, 1, 0) == 2.0)
    check('agg_2d_min sub2 g0', sub_2d(vm, 2, 0) == 3.0)
    check('agg_2d_min sub0 g1', sub_2d(vm, 0, 1) == 5.0)
    check('agg_2d_min sub2 g1', sub_2d(vm, 2, 1) == 7.0)
    check('agg_2d_max sub0 g0', sub_2d(vx, 0, 0) == 10.0)
    check('agg_2d_max sub1 g0', sub_2d(vx, 1, 0) == 20.0)
    check('agg_2d_max sub2 g0', sub_2d(vx, 2, 0) == 30.0)
    check('agg_2d_max sub0 g1', sub_2d(vx, 0, 1) == 100.0)
    check('agg_2d_max sub1 g1', sub_2d(vx, 1, 1) == 200.0)
    check('agg_2d_max sub2 g1', sub_2d(vx, 2, 1) == 300.0)

def test_groupby_agg_2d_median():
    print('--- test_groupby_agg_2d_median ---')
    df = make_2d_df()
    res = df.groupby('key').agg({'v': 'median'})
    v = res['v']
    check('agg_2d_median 2d', v.is_2d and v.cols == 3)
    check('agg_2d_median sub0 g0', sub_2d(v, 0, 0) == 8.0)
    check('agg_2d_median sub1 g0', sub_2d(v, 1, 0) == 9.0)
    check('agg_2d_median sub2 g0', sub_2d(v, 2, 0) == 10.0)
    check('agg_2d_median sub0 g1', sub_2d(v, 0, 1) == 52.5)
    check('agg_2d_median sub1 g1', sub_2d(v, 1, 1) == 103.0)
    check('agg_2d_median sub2 g1', sub_2d(v, 2, 1) == 153.5)

def test_groupby_agg_2d_mixed_cols():
    print('--- test_groupby_agg_2d_mixed_cols ---')
    df = make_2d_df()
    res = df.groupby('key').agg({'v': 'sum', 'val': 'sum'})
    check('agg_2d_mixed df_length', res.df_length == 2)
    check('agg_2d_mixed v 2d', res['v'].is_2d and res['v'].cols == 3)
    check('agg_2d_mixed v sub0 g0', sub_2d(res['v'], 0, 0) == 19.0)
    check('agg_2d_mixed v sub2 g1', sub_2d(res['v'], 2, 1) == 307.0)
    check('agg_2d_mixed val 1d', res['val'][0] == 8.0 and res['val'][1] == 7.0)

def test_groupby_2d_key():
    print('--- test_groupby_2d_key ---')
    df = make_2d_key_df()
    gb = df.groupby('key2')
    check('2d_key size', gb.size() == 4)
    res = gb.agg({'val': 'sum'})
    check('2d_key df_length', res.df_length == 4)
    check('2d_key val sums', [res['val'][i] for i in range(4)] == [20.0, 20.0, 30.0, 40.0])
    k = res['key2']
    check('2d_key col 2d', k.is_2d and k.cols == 2 and k.rows == 4)
    check('2d_key sub0', [sub_2d(k, 0, i) for i in range(4)] == [1.0, 1.0, 2.0, 2.0])
    check('2d_key sub1', [sub_2d(k, 1, i) for i in range(4)] == [1.0, 2.0, 1.0, 2.0])

def test_groupby_2d_idx():
    print('--- test_groupby_2d_idx ---')
    df = make_2d_df()
    res = df.groupby('key').idxmax('v')
    check('2d_idxmax df_length', res.df_length == 2)
    check('2d_idxmax idx', res['v'][0] == 1 and res['v'][1] == 2)
    check('2d_idxmax key', res['key'][0] == 1 and res['key'][1] == 2)
    res2 = df.groupby('key').idxmin('v')
    check('2d_idxmin idx', res2['v'][0] == 0 and res2['v'][1] == 3)

def run_all():
    test_groupby_basic()
    test_groupby_sum()
    test_groupby_mean()
    test_groupby_count()
    test_groupby_min_max()
    test_groupby_agg()
    test_groupby_auto_agg()
    test_groupby_multi_col()
    test_groupby_apply()
    test_groupby_empty()
    test_groupby_chaining()
    test_groupby_int_column()
    test_groupby_int_key()
    test_groupby_var_std()
    test_groupby_int_column_var_std()
    test_groupby_median()
    test_groupby_quantile()
    test_groupby_transform()
    test_groupby_filter()
    test_groupby_agg_median()
    test_groupby_agg_2d_sum()
    test_groupby_agg_2d_mean()
    test_groupby_agg_2d_var_std()
    test_groupby_agg_2d_min_max()
    test_groupby_agg_2d_median()
    test_groupby_agg_2d_mixed_cols()
    test_groupby_2d_key()
    test_groupby_2d_idx()
    print()
    print(f'Results: {passed} passed, {failed} failed')
    if failed > 0:
        sys.exit(1)

if __name__ == '__main__':
    run_all()
