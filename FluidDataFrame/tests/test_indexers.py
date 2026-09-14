import sys
sys.path.insert(0, '.')
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

def _df5():
    df = FastDataFrame({'a': int, 'b': str})
    for i in range(5):
        df.append_row(a=i, b=f'row{i}')
    return df


class TestILoc:
    def test_single_row(self):
        df = _df5()
        r = df.iloc[0]
        check('shape', r.shape == (1, 2))
        check('val', r['a'][0] == 0)

    def test_negative(self):
        df = _df5()
        r = df.iloc[-1]
        check('neg', r['a'][0] == 4)

    def test_slice(self):
        df = _df5()
        r = df.iloc[1:4]
        check('slice shape', r.shape == (3, 2))
        check('slice val', r['a'][0] == 1 and r['a'][2] == 3)

    def test_list(self):
        df = _df5()
        r = df.iloc[[0, 2, 4]]
        check('list shape', r.shape == (3, 2))
        check('list val', [r['a'][i] for i in range(3)] == [0, 2, 4])

    def test_empty(self):
        df = _df5()
        r = df.iloc[5:10]
        check('empty', r.shape == (0, 2))

    def test_oob(self):
        df = _df5()
        check('oob', expect_exception(IndexError, df.iloc.__getitem__, 10))


class TestLoc:
    def test_label(self):
        df = _df5()
        r = df.loc[0]
        check('shape', r.shape == (1, 2))
        check('val', r['a'][0] == 0)

    def test_missing(self):
        df = _df5()
        check('missing', expect_exception(KeyError, df.loc.__getitem__, 99))

    def test_int_mask(self):
        df = FastDataFrame({'id': int, 'score': float})
        for i in range(10): df.append_row(id=i, score=50.0 + i * 5)
        mask = df['id'] > 5
        r = df.loc[mask]
        check('shape', r.shape == (4, 2))
        check('first', r['id'][0] == 6)

    def test_float_mask(self):
        df = FastDataFrame({'id': int, 'score': float})
        for i in range(10): df.append_row(id=i, score=50.0 + i * 5)
        mask = df['score'] >= 80.0
        r = df.loc[mask]
        check('shape', r.shape[0] == 4)

    def test_and_combined(self):
        df = FastDataFrame({'id': int, 'score': float})
        for i in range(10): df.append_row(id=i, score=50.0 + i * 5)
        mask = (df['id'] > 3) & (df['score'] < 80.0)
        r = df.loc[mask]
        check('and shape', r.shape == (2, 2))
        check('and val', r['id'][0] == 4)

    def test_or_combined(self):
        df = FastDataFrame({'id': int, 'score': float})
        for i in range(10): df.append_row(id=i, score=50.0 + i * 5)
        mask = (df['id'] <= 2) | (df['score'] >= 90.0)
        r = df.loc[mask]
        check('or shape', r.shape[0] >= 3)

    def test_not_mask(self):
        df = FastDataFrame({'id': int, 'score': float})
        for i in range(10): df.append_row(id=i, score=50.0 + i * 5)
        mask = ~(df['id'] > 5)
        r = df.loc[mask]
        check('not shape', r.shape == (6, 2))

    def test_callable(self):
        df = FastDataFrame({'id': int, 'score': float})
        for i in range(10): df.append_row(id=i, score=50.0 + i * 5)
        r = df.loc[lambda d: d['id'] > 7]
        check('callable shape', r.shape == (2, 2))

    def test_bool_list(self):
        df = FastDataFrame({'id': int, 'score': float})
        for i in range(10): df.append_row(id=i, score=50.0 + i * 5)
        mask = [i % 2 == 0 for i in range(10)]
        r = df.loc[mask]
        check('list shape', r.shape == (5, 2))

    def test_preserves_index(self):
        df = FastDataFrame({'id': int, 'score': float})
        for i in range(10): df.append_row(id=i, score=50.0 + i * 5)
        mask = df['id'] > 5
        r = df.loc[mask]
        check('index', r.index.values() == [6, 7, 8, 9])

    def test_rows_cols(self):
        df = FastDataFrame({'name': str, 'age': int, 'score': float})
        df.append_row(name='alice', age=25, score=90.0)
        df.append_row(name='bob', age=30, score=85.0)
        df.append_row(name='charlie', age=35, score=95.0)
        mask = df['age'] > 25
        r = df.loc[mask, ['name', 'score']]
        check('shape', r.shape == (2, 2))
        check('cols', set(r.columns) == {'name', 'score'})

    def test_single_col(self):
        df = FastDataFrame({'name': str, 'age': int})
        df.append_row(name='alice', age=25)
        df.append_row(name='bob', age=30)
        mask = df['age'] > 25
        col = df.loc[mask, 'age']
        check('type', col.length == 1)
        check('val', col[0] == 30)

    def test_setitem_int(self):
        df = FastDataFrame({'A': int, 'B': int})
        df.append_row(A=1, B=10); df.append_row(A=2, B=20); df.append_row(A=3, B=30)
        df.loc[1] = [99, 88]
        check('A', df['A'][1] == 99)
        check('B', df['B'][1] == 88)
        check('unchanged', df['A'][0] == 1)

    def test_setitem_dict(self):
        df = FastDataFrame({'A': int, 'B': int})
        df.append_row(A=1, B=10)
        df.loc[0] = {'A': 77}
        check('A', df['A'][0] == 77)
        check('B unchanged', df['B'][0] == 10)


def run_all():
    print('=== Indexer Tests ===')
    for cls in [TestILoc, TestLoc]:
        for name in dir(cls):
            if name.startswith('test_'):
                getattr(cls(), name)()
    print(f'\nResults: {passed} passed, {failed} failed')
    if failed: sys.exit(1)

if __name__ == '__main__':
    run_all()
