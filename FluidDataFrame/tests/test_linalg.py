import sys
import math
sys.path.insert(0, '.')
from fluidframe.core.fastdataframe import FastDataFrame
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

def near(a, b, eps=1e-9):
    return abs(a - b) < eps

def expect_exception(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        return False
    except exc_type:
        return True
    except Exception:
        return False


class TestFromDict:
    def test_int(self):
        df = FastDataFrame.from_dict({'a': [1, 2, 3], 'b': [4, 5, 6]}, schema={'a': int, 'b': int})
        check('length', df.df_length == 3)
        check('a[0]', df['a'][0] == 1)

    def test_float(self):
        df = FastDataFrame.from_dict({'x': [1.5, 2.5]}, schema={'x': float})
        check('x[0]', df['x'][0] == 1.5)

    def test_mixed(self):
        df = FastDataFrame.from_dict({'id': [1, 2], 'val': [10.0, 20.0]}, schema={'id': int, 'val': float})
        check('length', df.df_length == 2)

    def test_rename(self):
        df = FastDataFrame.from_dict({'a': [1], 'b': [2]}, schema={'a': int, 'b': int}, columns=['X', 'Y'])
        check('renamed', set(df.columns.keys()) == {'X', 'Y'})

    def test_empty(self):
        df = FastDataFrame.from_dict({}, schema={})
        check('empty', df.df_length == 0)

    def test_length_mismatch(self):
        check('raises', expect_exception(ValueError, FastDataFrame.from_dict,
              {'a': [1, 2], 'b': [3]}, schema={'a': int, 'b': int}))


class TestArangeReshape:
    def test_arange(self):
        df = FastDataFrame.arange(5)
        check('length', df.df_length == 5)
        check('values', [df['0'][i] for i in range(5)] == [0, 1, 2, 3, 4])

    def test_arange_empty(self):
        df = FastDataFrame.arange(0)
        check('empty', df.df_length == 0)

    def test_reshape(self):
        df = FastDataFrame.arange(6).reshape(2, 3)
        check('rows', df.df_length == 2)
        check('A', [df['A'][i] for i in range(2)] == [0.0, 3.0])

    def test_reshape_3x4(self):
        df = FastDataFrame.arange(12).reshape(3, 4)
        check('rows', df.df_length == 3)
        check('D[2]=11', df['D'][2] == 11.0)

    def test_reshape_mismatch(self):
        check('raises', expect_exception(ValueError, FastDataFrame.arange(5).reshape, 2, 3))


class TestRandomZerosOnes:
    def test_randn(self):
        df = FastDataFrame.randn(100, 3)
        check('rows', df.df_length == 100)
        for name in df.columns:
            mean = sum(df[name][i] for i in range(100)) / 100
            check(f'{name} mean~0', abs(mean) < 0.5)

    def test_rand(self):
        df = FastDataFrame.rand(200, 2)
        check('rows', df.df_length == 200)
        all_ok = all(0 <= df[name][i] < 1 for name in df.columns for i in range(200))
        check('in [0,1)', all_ok)

    def test_zeros(self):
        df = FastDataFrame.zeros(3, 4)
        check('rows', df.df_length == 3)
        check('zeros', all(df[name][i] == 0.0 for name in df.columns for i in range(3)))

    def test_ones(self):
        df = FastDataFrame.ones(2, 3)
        check('rows', df.df_length == 2)
        check('ones', all(df[name][i] == 1.0 for name in df.columns for i in range(2)))


class TestToNumpy:
    def test_basic(self):
        df = FastDataFrame.from_dict({'a': [1, 2], 'b': [3.0, 4.0]}, schema={'a': int, 'b': float})
        mat = df.to_numpy()
        check('shape', len(mat) == 2 and len(mat[0]) == 2)
        check('val', mat[0][0] == 1 and mat[1][1] == 4.0)

    def test_empty(self):
        df = FastDataFrame()
        check('empty', df.to_numpy() == [])


class TestLinalgDot:
    def test_2x2(self):
        A = FastDataFrame.from_dict({'0': [1, 2], '1': [3, 4]}, schema={'0': int, '1': int})
        B = FastDataFrame.from_dict({'0': [5, 6], '1': [7, 8]}, schema={'0': int, '1': int})
        C = A.dot(B)
        check('C[0][0]', C['A'][0] == 23.0)
        check('C[1][1]', C['B'][1] == 46.0)

    def test_identity(self):
        A = FastDataFrame.from_dict({'0': [1, 2], '1': [3, 4]}, schema={'0': int, '1': int})
        I = FastDataFrame.from_dict({'0': [1, 0], '1': [0, 1]}, schema={'0': int, '1': int})
        C = A.dot(I)
        check('A*I~A', C['A'][0] == 1.0 and C['B'][0] == 3.0)

    def test_shape_mismatch(self):
        A = FastDataFrame.from_dict({'0': [1, 2, 3]}, schema={'0': int})
        B = FastDataFrame.from_dict({'0': [1, 2]}, schema={'0': int})
        check('raises', expect_exception(ValueError, A.dot, B))

    def test_3x3(self):
        A = FastDataFrame.from_dict({'0': [1,0,0], '1': [0,2,0], '2': [0,0,3]}, schema={'0': int, '1': int, '2': int})
        B = FastDataFrame.from_dict({'0': [4,0,0], '1': [0,5,0], '2': [0,0,6]}, schema={'0': int, '1': int, '2': int})
        C = A.dot(B)
        check('diag', C['A'][0] == 4.0 and C['B'][1] == 10.0 and C['C'][2] == 18.0)


class TestLinalgTranspose:
    def test_basic(self):
        A = FastDataFrame.from_dict({'0': [1, 2, 3], '1': [4, 5, 6]}, schema={'0': int, '1': int})
        T = A.transpose()
        check('rows', T.df_length == 2)
        check('T[0][0]', T['A'][0] == 1.0)
        check('T[2][1]', T['C'][1] == 6.0)

    def test_square(self):
        A = FastDataFrame.from_dict({'0': [1, 2], '1': [3, 4]}, schema={'0': int, '1': int})
        T = A.transpose()
        check('T[0][1]', T['A'][1] == 3.0)


class TestLinalgDet:
    def test_2x2(self):
        A = FastDataFrame.from_dict({'0': [1, 2], '1': [3, 4]}, schema={'0': int, '1': int})
        check('det', A.det() == -2.0)

    def test_identity(self):
        I = FastDataFrame.from_dict({'0': [1, 0], '1': [0, 1]}, schema={'0': int, '1': int})
        check('det(I)', I.det() == 1.0)

    def test_singular(self):
        S = FastDataFrame.from_dict({'0': [1, 2], '1': [2, 4]}, schema={'0': int, '1': int})
        check('det(singular)', S.det() == 0.0)

    def test_3x3(self):
        M = FastDataFrame.from_dict({'0': [6,1,1], '1': [4,-2,5], '2': [2,8,7]}, schema={'0': int, '1': int, '2': int})
        check('det(3x3)', M.det() == -306.0)

    def test_not_square(self):
        A = FastDataFrame.from_dict({'0': [1,2], '1': [3,4], '2': [5,6]}, schema={'0': int, '1': int, '2': int})
        check('raises', expect_exception(ValueError, A.det))


class TestLinalgInv:
    def test_2x2(self):
        A = FastDataFrame.from_dict({'0': [1, 2], '1': [3, 4]}, schema={'0': int, '1': int})
        A_inv = A.inv()
        check('inv[0][0]', near(A_inv['A'][0], -2.0))
        check('inv[1][1]', near(A_inv['B'][1], -0.5))

    def test_identity(self):
        I = FastDataFrame.from_dict({'0': [1, 0], '1': [0, 1]}, schema={'0': int, '1': int})
        I_inv = I.inv()
        check('inv(I)', near(I_inv['A'][0], 1.0) and near(I_inv['B'][1], 1.0))

    def test_times_self(self):
        A = FastDataFrame.from_dict({'0': [2, 1], '1': [5, 3]}, schema={'0': int, '1': int})
        prod = A.dot(A.inv())
        check('A*A^-1~I', near(prod['A'][0], 1.0) and near(prod['B'][1], 1.0))

    def test_singular(self):
        A = FastDataFrame.from_dict({'0': [1, 2], '1': [2, 4]}, schema={'0': int, '1': int})
        check('raises', expect_exception(ValueError, A.inv))


class TestLinalgSolve:
    def test_basic(self):
        A = FastDataFrame.from_dict({'0': [1, 3], '1': [2, 4]}, schema={'0': int, '1': int})
        b = FastDataFrame.from_dict({'0': [5, 11]}, schema={'0': int})
        x = A.solve(b)
        check('x[0]', near(x['0'][0], 1.0))
        check('x[1]', near(x['0'][1], 2.0))

    def test_diagonal(self):
        A = FastDataFrame.from_dict({'0': [2, 0], '1': [0, 3]}, schema={'0': int, '1': int})
        b = FastDataFrame.from_dict({'0': [4, 9]}, schema={'0': int})
        x = A.solve(b)
        check('x', near(x['0'][0], 2.0) and near(x['0'][1], 3.0))


class TestLinalgNorm:
    def test_frobenius(self):
        A = FastDataFrame.from_dict({'0': [1, 0], '1': [0, 2]}, schema={'0': int, '1': int})
        check('frobenius', near(A.norm(), math.sqrt(5)))

    def test_max(self):
        A = FastDataFrame.from_dict({'0': [1, -5], '1': [3, 2]}, schema={'0': int, '1': int})
        check('max', A.norm('max') == 5.0)

    def test_count(self):
        A = FastDataFrame.from_dict({'0': [1, 0], '1': [0, 4]}, schema={'0': int, '1': int})
        check('count', A.norm('count') == 2.0)


class TestLinalgIntegration:
    def test_arange_inv_dot(self):
        A = FastDataFrame.arange(4).reshape(2, 2)
        prod = A.dot(A.inv())
        check('A*A^-1~I', near(prod['A'][0], 1.0) and near(prod['B'][1], 1.0))

    def test_3x3_inv_dot(self):
        A = FastDataFrame.from_dict({'0': [1, 2, 3], '1': [0, 1, 4], '2': [5, 6, 0]},
                                    schema={'0': int, '1': int, '2': int})
        prod = A.dot(A.inv())
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                col = chr(65 + j)
                check(f'I[{i}][{j}]', near(prod.columns[col][i], expected, 1e-7))


class TestLinalgMatmul:
    def test_basic(self):
        A = FastDataFrame.from_dict({'0': [1, 2], '1': [3, 4]}, schema={'0': int, '1': int})
        B = FastDataFrame.from_dict({'0': [5, 6], '1': [7, 8]}, schema={'0': int, '1': int})
        C = A.matmul(B)
        check('C[0][0]', C['A'][0] == 23.0)
        check('C[1][1]', C['B'][1] == 46.0)


def _make_2d_matrix(kind='double'):
    # 单个 2D 列（cols=2，列优先 m_ij = data_c[j*rows + i]）表示逻辑矩阵 [[1,2],[3,4]]
    df = FastDataFrame()
    col = (IntColumn if kind == 'int' else DoubleColumn).from_2d([1, 3, 2, 4], 2, 2)
    df.columns['M'] = col
    df.df_length = 2
    return df


class TestLinalg2D:
    def test_det_double(self):
        check('det double 2D', near(_make_2d_matrix().det(), -2.0))

    def test_det_int(self):
        check('det int 2D', near(_make_2d_matrix('int').det(), -2.0))

    def test_det_nonsquare(self):
        df = FastDataFrame()
        df.columns['M'] = DoubleColumn.from_2d([1, 2, 3], 3, 1)
        df.df_length = 3
        check('nonsquare raises', expect_exception(ValueError, df.det))

    def test_inv(self):
        Ai = _make_2d_matrix().inv()
        check('inv[0][0]', near(Ai['A'][0], -2.0))
        check('inv[0][1]', near(Ai['B'][0], 1.0))
        check('inv[1][0]', near(Ai['A'][1], 1.5))
        check('inv[1][1]', near(Ai['B'][1], -0.5))

    def test_dot_2d_2d(self):
        A = _make_2d_matrix()
        C = A.dot(A)
        check('C[0][0]', near(C['A'][0], 7.0))
        check('C[1][0]', near(C['A'][1], 15.0))
        check('C[0][1]', near(C['B'][0], 10.0))
        check('C[1][1]', near(C['B'][1], 22.0))

    def test_dot_2d_1d(self):
        A = _make_2d_matrix()
        B = FastDataFrame.from_dict({'0': [1, 1]}, schema={'0': int})
        C = A.dot(B)
        check('C rows', C.df_length == 2)
        check('C[0]', near(C['A'][0], 3.0))
        check('C[1]', near(C['A'][1], 7.0))

    def test_transpose(self):
        T = _make_2d_matrix().transpose()
        check('T rows', T.df_length == 2)
        check('T[0][0]', near(T['A'][0], 1.0))
        check('T[1][0]', near(T['A'][1], 2.0))
        check('T[0][1]', near(T['B'][0], 3.0))
        check('T[1][1]', near(T['B'][1], 4.0))

    def test_solve(self):
        A = _make_2d_matrix()
        b = FastDataFrame.from_dict({'0': [5, 11]}, schema={'0': int})
        x = A.solve(b)
        check('x[0]', near(x['0'][0], 1.0))
        check('x[1]', near(x['0'][1], 2.0))

    def test_norm(self):
        check('norm 2D', near(_make_2d_matrix().norm(), math.sqrt(30)))

    def test_inv_dot_identity(self):
        A = _make_2d_matrix()
        prod = A.dot(A.inv())
        check('I[0][0]', near(prod['A'][0], 1.0))
        check('I[1][1]', near(prod['B'][1], 1.0))


def run_all():
    print('=== Linalg Tests ===')
    for cls in [TestFromDict, TestArangeReshape, TestRandomZerosOnes, TestToNumpy,
                TestLinalgDot, TestLinalgTranspose, TestLinalgDet, TestLinalgInv,
                TestLinalgSolve, TestLinalgNorm, TestLinalgIntegration, TestLinalgMatmul,
                TestLinalg2D]:
        for name in dir(cls):
            if name.startswith('test_'):
                getattr(cls(), name)()
    print(f'\nResults: {passed} passed, {failed} failed')
    if failed: sys.exit(1)

if __name__ == '__main__':
    run_all()
