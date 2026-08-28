import sys
sys.path.insert(0, '.')

from fluidframe._schema_types import Char256, Timestamp
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

def make_df():
    df = FastDataFrame({'name': str, 'amount': float, 'qty': int})
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
    df = FastDataFrame({'name': str, 'dept': str, 'salary': float})
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
        res = FastDataFrame({'name': str, 'total': int})
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
    df = FastDataFrame({'name': str, 'amount': float})
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
    df = FastDataFrame({'cat': int, 'val': float})
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
    df = FastDataFrame({'id': int, 'score': float})
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
    df = FastDataFrame({'name': str, 'val': float})
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
    df = FastDataFrame({'cat': int, 'val': int})
    df.append_row(cat=1, val=10)
    df.append_row(cat=1, val=20)
    df.append_row(cat=2, val=5)
    res = df.groupby('cat').std('val')
    cats = [res['cat'][i] for i in range(2)]
    # cat=1: [10,20] -> std=sqrt(50)~7.07
    check('cat1_std', abs(res['val'][cats.index(1)] - 7.071067811865) < 1e-6)

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
    print()
    print(f'Results: {passed} passed, {failed} failed')
    if failed > 0:
        sys.exit(1)

if __name__ == '__main__':
    run_all()
