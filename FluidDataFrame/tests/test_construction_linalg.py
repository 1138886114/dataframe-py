import sys, math
sys.path.insert(0, '.')

from fluidframe.core.fastdataframe import FastDataFrame
from fluidframe.core import linalg

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

# ============================================================
#  from_dict
# ============================================================
def test_from_dict_int():
    print('--- test_from_dict_int ---')
    df = FastDataFrame.from_dict({'a': [1, 2, 3], 'b': [4, 5, 6]},
                                 schema={'a': int, 'b': int})
    check('length', df.df_length == 3)
    check('cols', set(df.columns.keys()) == {'a', 'b'})
    check('a[0]', df['a'][0] == 1)
    check('a[2]', df['a'][2] == 3)
    check('b[1]', df['b'][1] == 5)

def test_from_dict_float():
    print('--- test_from_dict_float ---')
    df = FastDataFrame.from_dict({'x': [1.5, 2.5], 'y': [3.0, 4.0]},
                                 schema={'x': float, 'y': float})
    check('length', df.df_length == 2)
    check('x[0]', df['x'][0] == 1.5)
    check('y[1]', df['y'][1] == 4.0)

def test_from_dict_mixed():
    print('--- test_from_dict_mixed ---')
    df = FastDataFrame.from_dict({'id': [1, 2], 'val': [10.0, 20.0]},
                                 schema={'id': int, 'val': float})
    check('length', df.df_length == 2)
    check('id type', df['id'][0] == 1)
    check('val type', df['val'][0] == 10.0)

def test_from_dict_rename():
    print('--- test_from_dict_rename ---')
    df = FastDataFrame.from_dict({'a': [1], 'b': [2]},
                                 schema={'a': int, 'b': int},
                                 columns=['X', 'Y'])
    check('renamed', set(df.columns.keys()) == {'X', 'Y'})
    check('X[0]', df['X'][0] == 1)
    check('Y[0]', df['Y'][0] == 2)

def test_from_dict_empty():
    print('--- test_from_dict_empty ---')
    df = FastDataFrame.from_dict({}, schema={})
    check('empty', df.df_length == 0)

def test_from_dict_length_mismatch():
    print('--- test_from_dict_length_mismatch ---')
    ok = False
    try:
        FastDataFrame.from_dict({'a': [1, 2], 'b': [3]},
                                schema={'a': int, 'b': int})
    except ValueError:
        ok = True
    check('raises ValueError', ok)

# ============================================================
#  arange / reshape
# ============================================================
def test_arange():
    print('--- test_arange ---')
    df = FastDataFrame.arange(5)
    check('length', df.df_length == 5)
    vals = [df['0'][i] for i in range(5)]
    check('values', vals == [0, 1, 2, 3, 4])

def test_arange_empty():
    print('--- test_arange_empty ---')
    df = FastDataFrame.arange(0)
    check('empty', df.df_length == 0)

def test_reshape():
    print('--- test_reshape ---')
    df = FastDataFrame.arange(6).reshape(2, 3)
    check('rows', df.df_length == 2)
    check('cols', len(df.columns) == 3)
    check('A', [df['A'][i] for i in range(2)] == [0.0, 3.0])
    check('B', [df['B'][i] for i in range(2)] == [1.0, 4.0])
    check('C', [df['C'][i] for i in range(2)] == [2.0, 5.0])

def test_reshape_3x4():
    print('--- test_reshape_3x4 ---')
    df = FastDataFrame.arange(12).reshape(3, 4)
    check('rows', df.df_length == 3)
    check('cols', len(df.columns) == 4)
    check('A[0]=0', df['A'][0] == 0.0)
    check('D[2]=11', df['D'][2] == 11.0)

def test_reshape_mismatch():
    print('--- test_reshape_mismatch ---')
    ok = False
    try:
        FastDataFrame.arange(5).reshape(2, 3)
    except ValueError:
        ok = True
    check('raises ValueError', ok)

# ============================================================
#  random / zeros / ones
# ============================================================
def test_randn():
    print('--- test_randn ---')
    df = FastDataFrame.randn(100, 3)
    check('rows', df.df_length == 100)
    check('cols', len(df.columns) == 3)
    for name in df.columns:
        vals = [df[name][i] for i in range(100)]
        mean = sum(vals) / 100
        check(f'{name} mean~0', abs(mean) < 0.5)

def test_rand():
    print('--- test_rand ---')
    df = FastDataFrame.rand(200, 2)
    check('rows', df.df_length == 200)
    all_in_range = True
    for name in df.columns:
        for i in range(200):
            v = df[name][i]
            if v < 0 or v >= 1:
                all_in_range = False
                break
    check('all in [0,1)', all_in_range)

def test_zeros():
    print('--- test_zeros ---')
    df = FastDataFrame.zeros(3, 4)
    check('rows', df.df_length == 3)
    check('cols', len(df.columns) == 4)
    for name in df.columns:
        for i in range(3):
            check(f'{name}[{i}]=0', df[name][i] == 0.0)

def test_ones():
    print('--- test_ones ---')
    df = FastDataFrame.ones(2, 3)
    check('rows', df.df_length == 2)
    for name in df.columns:
        for i in range(2):
            check(f'{name}[{i}]=1', df[name][i] == 1.0)

# ============================================================
#  to_numpy
# ============================================================
def test_to_numpy():
    print('--- test_to_numpy ---')
    df = FastDataFrame.from_dict({'a': [1, 2], 'b': [3.0, 4.0]},
                                 schema={'a': int, 'b': float})
    mat = df.to_numpy()
    check('shape', len(mat) == 2 and len(mat[0]) == 2)
    check('val[0][0]', mat[0][0] == 1)
    check('val[1][1]', mat[1][1] == 4.0)

def test_to_numpy_empty():
    print('--- test_to_numpy_empty ---')
    df = FastDataFrame()
    check('empty', df.to_numpy() == [])

# ============================================================
#  linalg: dot  (result columns are A, B, C...)
# ============================================================
def test_dot_2x2():
    print('--- test_dot_2x2 ---')
    A = FastDataFrame.from_dict({'0': [1, 2], '1': [3, 4]}, schema={'0': int, '1': int})
    B = FastDataFrame.from_dict({'0': [5, 6], '1': [7, 8]}, schema={'0': int, '1': int})
    C = linalg.dot(A, B)
    check('C[0][0]', C['A'][0] == 23.0)
    check('C[0][1]', C['B'][0] == 31.0)
    check('C[1][0]', C['A'][1] == 34.0)
    check('C[1][1]', C['B'][1] == 46.0)

def test_dot_identity():
    print('--- test_dot_identity ---')
    A = FastDataFrame.from_dict({'0': [1, 2], '1': [3, 4]}, schema={'0': int, '1': int})
    I = FastDataFrame.from_dict({'0': [1, 0], '1': [0, 1]}, schema={'0': int, '1': int})
    C = linalg.dot(A, I)
    check('A*I[0][0]', C['A'][0] == 1.0)
    check('A*I[0][1]', C['B'][0] == 3.0)
    check('A*I[1][0]', C['A'][1] == 2.0)
    check('A*I[1][1]', C['B'][1] == 4.0)

def test_dot_shape_mismatch():
    print('--- test_dot_shape_mismatch ---')
    A = FastDataFrame.from_dict({'0': [1, 2, 3]}, schema={'0': int})
    B = FastDataFrame.from_dict({'0': [1, 2]}, schema={'0': int})
    ok = False
    try:
        linalg.dot(A, B)
    except ValueError:
        ok = True
    check('raises ValueError', ok)

def test_dot_3x3():
    print('--- test_dot_3x3 ---')
    A = FastDataFrame.from_dict({'0': [1,0,0], '1': [0,2,0], '2': [0,0,3]},
                                schema={'0': int, '1': int, '2': int})
    B = FastDataFrame.from_dict({'0': [4,0,0], '1': [0,5,0], '2': [0,0,6]},
                                schema={'0': int, '1': int, '2': int})
    C = linalg.dot(A, B)
    check('diag[0]', C['A'][0] == 4.0)
    check('diag[1]', C['B'][1] == 10.0)
    check('diag[2]', C['C'][2] == 18.0)
    check('off', C['B'][0] == 0.0)

# ============================================================
#  linalg: transpose  (result columns are 0, 1, 2... for rows)
# ============================================================
def test_transpose():
    print('--- test_transpose ---')
    A = FastDataFrame.from_dict({'0': [1, 2, 3], '1': [4, 5, 6]}, schema={'0': int, '1': int})
    T = linalg.transpose(A)
    check('rows', T.df_length == 2)
    check('cols', len(T.columns) == 3)
    check('T[0][0]', T['A'][0] == 1.0)
    check('T[0][1]', T['A'][1] == 4.0)
    check('T[1][0]', T['B'][0] == 2.0)
    check('T[2][1]', T['C'][1] == 6.0)

def test_transpose_square():
    print('--- test_transpose_square ---')
    A = FastDataFrame.from_dict({'0': [1, 2], '1': [3, 4]}, schema={'0': int, '1': int})
    T = linalg.transpose(A)
    check('T[0][0]', T['A'][0] == 1.0)
    check('T[0][1]', T['A'][1] == 3.0)
    check('T[1][0]', T['B'][0] == 2.0)
    check('T[1][1]', T['B'][1] == 4.0)

# ============================================================
#  linalg: det
# ============================================================
def test_det_2x2():
    print('--- test_det_2x2 ---')
    A = FastDataFrame.from_dict({'0': [1, 2], '1': [3, 4]}, schema={'0': int, '1': int})
    check('det', linalg.det(A) == -2.0)

def test_det_identity():
    print('--- test_det_identity ---')
    I = FastDataFrame.from_dict({'0': [1, 0], '1': [0, 1]}, schema={'0': int, '1': int})
    check('det(I)', linalg.det(I) == 1.0)

def test_det_singular():
    print('--- test_det_singular ---')
    S = FastDataFrame.from_dict({'0': [1, 2], '1': [2, 4]}, schema={'0': int, '1': int})
    check('det(singular)', linalg.det(S) == 0.0)

def test_det_3x3():
    print('--- test_det_3x3 ---')
    M = FastDataFrame.from_dict({'0': [6,1,1], '1': [4,-2,5], '2': [2,8,7]},
                                schema={'0': int, '1': int, '2': int})
    check('det(3x3)', linalg.det(M) == -306.0)

def test_det_not_square():
    print('--- test_det_not_square ---')
    ok = False
    try:
        linalg.det(FastDataFrame.from_dict({'0': [1,2], '1': [3,4], '2': [5,6]},
                                           schema={'0': int, '1': int, '2': int}))
    except ValueError:
        ok = True
    check('raises ValueError', ok)

# ============================================================
#  linalg: inv  (result columns are A, B)
# ============================================================
def test_inv_2x2():
    print('--- test_inv_2x2 ---')
    A = FastDataFrame.from_dict({'0': [1, 2], '1': [3, 4]}, schema={'0': int, '1': int})
    A_inv = linalg.inv(A)
    check('inv[0][0]', near(A_inv['A'][0], -2.0))
    check('inv[0][1]', near(A_inv['B'][0], 1.5))
    check('inv[1][0]', near(A_inv['A'][1], 1.0))
    check('inv[1][1]', near(A_inv['B'][1], -0.5))

def test_inv_identity():
    print('--- test_inv_identity ---')
    I = FastDataFrame.from_dict({'0': [1, 0], '1': [0, 1]}, schema={'0': int, '1': int})
    I_inv = linalg.inv(I)
    check('inv(I)[0][0]', near(I_inv['A'][0], 1.0))
    check('inv(I)[0][1]', near(I_inv['B'][0], 0.0))

def test_inv_times_self():
    print('--- test_inv_times_self ---')
    A = FastDataFrame.from_dict({'0': [2, 1], '1': [5, 3]}, schema={'0': int, '1': int})
    A_inv = linalg.inv(A)
    prod = linalg.dot(A, A_inv)
    check('A*A^-1~I[0][0]', near(prod['A'][0], 1.0))
    check('A*A^-1~I[0][1]', near(prod['B'][0], 0.0, 1e-7))
    check('A*A^-1~I[1][0]', near(prod['A'][1], 0.0, 1e-7))
    check('A*A^-1~I[1][1]', near(prod['B'][1], 1.0))

def test_inv_singular():
    print('--- test_inv_singular ---')
    ok = False
    try:
        linalg.inv(FastDataFrame.from_dict({'0': [1, 2], '1': [2, 4]}, schema={'0': int, '1': int}))
    except ValueError:
        ok = True
    check('raises ValueError', ok)

# ============================================================
#  linalg: solve  (result column is A)
# ============================================================
def test_solve():
    print('--- test_solve ---')
    A = FastDataFrame.from_dict({'0': [1, 3], '1': [2, 4]}, schema={'0': int, '1': int})
    b = FastDataFrame.from_dict({'0': [5, 11]}, schema={'0': int})
    x = linalg.solve(A, b)
    check('x[0]', near(x['0'][0], 1.0))
    check('x[1]', near(x['0'][1], 2.0))

def test_solve_diagonal():
    print('--- test_solve_diagonal ---')
    A = FastDataFrame.from_dict({'0': [2, 0], '1': [0, 3]}, schema={'0': int, '1': int})
    b = FastDataFrame.from_dict({'0': [4, 9]}, schema={'0': int})
    x = linalg.solve(A, b)
    check('x[0]', near(x['0'][0], 2.0))
    check('x[1]', near(x['0'][1], 3.0))

# ============================================================
#  linalg: matmul
# ============================================================
def test_matmul():
    print('--- test_matmul ---')
    A = FastDataFrame.from_dict({'0': [1, 2], '1': [3, 4]}, schema={'0': int, '1': int})
    B = FastDataFrame.from_dict({'0': [5, 6], '1': [7, 8]}, schema={'0': int, '1': int})
    C = linalg.matmul(A, B)
    check('C[0][0]', C['A'][0] == 23.0)
    check('C[1][1]', C['B'][1] == 46.0)

# ============================================================
#  linalg: norm
# ============================================================
def test_norm_frobenius():
    print('--- test_norm_frobenius ---')
    A = FastDataFrame.from_dict({'0': [1, 0], '1': [0, 2]}, schema={'0': int, '1': int})
    check('frobenius', near(linalg.norm(A), math.sqrt(5)))

def test_norm_max():
    print('--- test_norm_max ---')
    A = FastDataFrame.from_dict({'0': [1, -5], '1': [3, 2]}, schema={'0': int, '1': int})
    check('max', linalg.norm(A, 'max') == 5.0)

def test_norm_count():
    print('--- test_norm_count ---')
    A = FastDataFrame.from_dict({'0': [1, 0], '1': [0, 4]}, schema={'0': int, '1': int})
    check('count', linalg.norm(A, 'count') == 2.0)

# ============================================================
#  Integration
# ============================================================
def test_integration_arange_linalg():
    print('--- test_integration_arange_linalg ---')
    A = FastDataFrame.arange(4).reshape(2, 2)
    A_inv = linalg.inv(A)
    prod = linalg.dot(A, A_inv)
    check('A*A^-1~I[0][0]', near(prod['A'][0], 1.0))
    check('A*A^-1~I[1][1]', near(prod['B'][1], 1.0))

def test_integration_from_dict_linalg():
    print('--- test_integration_from_dict_linalg ---')
    A = FastDataFrame.from_dict({'0': [1, 2, 3], '1': [0, 1, 4], '2': [5, 6, 0]},
                                schema={'0': int, '1': int, '2': int})
    det = linalg.det(A)
    check('det != 0', abs(det) > 1e-10)
    A_inv = linalg.inv(A)
    prod = linalg.dot(A, A_inv)
    for i in range(3):
        for j in range(3):
            expected = 1.0 if i == j else 0.0
            col = chr(65 + j)
            check(f'A*A^-1[{i}][{j}]', near(prod.columns[col][i], expected, 1e-7))

def run_all():
    test_from_dict_int()
    test_from_dict_float()
    test_from_dict_mixed()
    test_from_dict_rename()
    test_from_dict_empty()
    test_from_dict_length_mismatch()
    test_arange()
    test_arange_empty()
    test_reshape()
    test_reshape_3x4()
    test_reshape_mismatch()
    test_randn()
    test_rand()
    test_zeros()
    test_ones()
    test_to_numpy()
    test_to_numpy_empty()
    test_dot_2x2()
    test_dot_identity()
    test_dot_shape_mismatch()
    test_dot_3x3()
    test_transpose()
    test_transpose_square()
    test_det_2x2()
    test_det_identity()
    test_det_singular()
    test_det_3x3()
    test_det_not_square()
    test_inv_2x2()
    test_inv_identity()
    test_inv_times_self()
    test_inv_singular()
    test_solve()
    test_solve_diagonal()
    test_matmul()
    test_norm_frobenius()
    test_norm_max()
    test_norm_count()
    test_integration_arange_linalg()
    test_integration_from_dict_linalg()
    print()
    print(f'Results: {passed} passed, {failed} failed')
    if failed > 0:
        sys.exit(1)

if __name__ == '__main__':
    run_all()
