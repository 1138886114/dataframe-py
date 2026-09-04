import math
import sys
sys.path.insert(0, '..')
import numpy as np
from fluidframe.core.intcolumn import IntColumn
from fluidframe.core.doublecolumn import DoubleColumn
from fluidframe.core.fastdataframe import FastDataFrame
from fluidframe.core.index import RangeIndex
from fluidframe.core.linalg import (
    euclidean_dist, manhattan_dist, cosine_dist,
    cosine_similarity, jaccard_sim, hamming_dist,
    norm, dot, transpose,
)

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {name}")

def approx_eq(a, b, tol=1e-9):
    return abs(a - b) < tol

def expect_exception(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        return False
    except exc_type:
        return True
    except Exception:
        return False

# ============================================================
#  DoubleColumn.load_from_1d_array
# ============================================================
def test_double_load_from_1d_array_basic():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    col = DoubleColumn.load_from_1d_array(arr)
    check('1d length', col.length == 5)
    check('1d sum', approx_eq(col.sum(), 15.0))
    check('1d mean', approx_eq(col.mean(), 3.0))
    check('1d min', approx_eq(col.min(), 1.0))
    check('1d max', approx_eq(col.max(), 5.0))

def test_double_load_from_1d_array_zero_copy():
    arr = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    col = DoubleColumn.load_from_1d_array(arr)
    check('1d zero-copy sum before', approx_eq(col.sum(), 60.0))
    arr[0] = 999.0
    check('1d zero-copy sum after', approx_eq(col.sum(), 1049.0))

def test_double_load_from_1d_array_empty():
    arr = np.array([], dtype=np.float64)
    col = DoubleColumn.load_from_1d_array(arr)
    check('1d empty length', col.length == 0)

def test_double_load_from_1d_array_single():
    arr = np.array([42.0], dtype=np.float64)
    col = DoubleColumn.load_from_1d_array(arr)
    check('1d single length', col.length == 1)
    check('1d single value', approx_eq(col[0], 42.0))

def test_double_load_from_1d_array_negative():
    arr = np.array([-1.5, -2.5, 3.5], dtype=np.float64)
    col = DoubleColumn.load_from_1d_array(arr)
    check('1d negative sum', approx_eq(col.sum(), -0.5))

def test_double_load_from_1d_array_nan():
    arr = np.array([1.0, float('nan'), 3.0], dtype=np.float64)
    col = DoubleColumn.load_from_1d_array(arr)
    check('1d nan length', col.length == 3)
    check('1d nan isnan', col[1] != col[1])

# ============================================================
#  DoubleColumn.load_from_2d_array
# ============================================================
def test_double_load_from_2d_array_basic():
    data = np.array([[1.0, 10.0, 100.0],
                     [2.0, 20.0, 200.0],
                     [3.0, 30.0, 300.0]], dtype=np.float64, order='F')
    cols = DoubleColumn.load_from_2d_array(data, ['X', 'Y', 'Z'])
    check('2d keys', sorted(cols.keys()) == ['X', 'Y', 'Z'])
    check('2d X sum', approx_eq(cols['X'].sum(), 6.0))
    check('2d Y sum', approx_eq(cols['Y'].sum(), 60.0))
    check('2d Z sum', approx_eq(cols['Z'].sum(), 600.0))
    check('2d X length', cols['X'].length == 3)
    check('2d Y length', cols['Y'].length == 3)
    check('2d Z length', cols['Z'].length == 3)

def test_double_load_from_2d_array_zero_copy():
    data = np.array([[1.0, 10.0],
                     [2.0, 20.0],
                     [3.0, 30.0]], dtype=np.float64, order='F')
    cols = DoubleColumn.load_from_2d_array(data, ['A', 'B'])
    check('2d zero-copy before', approx_eq(cols['A'].sum(), 6.0))
    data[0, 0] = 999.0
    check('2d zero-copy after', approx_eq(cols['A'].sum(), 1004.0))

def test_double_load_from_2d_array_1row():
    data = np.array([[5.0, 6.0, 7.0]], dtype=np.float64, order='F')
    cols = DoubleColumn.load_from_2d_array(data, ['P', 'Q', 'R'])
    check('2d 1-row length', cols['P'].length == 1)
    check('2d 1-row P', approx_eq(cols['P'][0], 5.0))
    check('2d 1-row Q', approx_eq(cols['Q'][0], 6.0))
    check('2d 1-row R', approx_eq(cols['R'][0], 7.0))

def test_double_load_from_2d_array_1col():
    data = np.array([[1.0], [2.0], [3.0]], dtype=np.float64, order='F')
    cols = DoubleColumn.load_from_2d_array(data, ['Only'])
    check('2d 1-col length', cols['Only'].length == 3)
    check('2d 1-col sum', approx_eq(cols['Only'].sum(), 6.0))

def test_double_load_from_2d_array_large():
    rows, cols_n = 100, 5
    data = np.random.randn(rows, cols_n).astype(np.float64, order='F')
    names = [chr(65 + i) for i in range(cols_n)]
    cols = DoubleColumn.load_from_2d_array(data, names)
    for i, name in enumerate(names):
        expected = data[:, i].sum()
        check(f'2d large {name} sum', approx_eq(cols[name].sum(), expected))

def test_double_load_from_2d_array_negative_vals():
    data = np.array([[-1.0, 4.0],
                     [-2.0, 5.0],
                     [-3.0, 6.0]], dtype=np.float64, order='F')
    cols = DoubleColumn.load_from_2d_array(data, ['Neg', 'Pos'])
    check('2d neg sum', approx_eq(cols['Neg'].sum(), -6.0))
    check('2d pos sum', approx_eq(cols['Pos'].sum(), 15.0))

def test_double_load_from_2d_array_asfortran_conversion():
    data = np.array([[1.0, 10.0],
                     [2.0, 20.0],
                     [3.0, 30.0]], dtype=np.float64)  # C-order
    # 新 API 要求传入 Fortran-contiguous，用户需自行转换
    cols = DoubleColumn.load_from_2d_array(np.asfortranarray(data), ['C', 'D'])
    check('2d asfortran C sum', approx_eq(cols['C'].sum(), 6.0))
    check('2d asfortran D sum', approx_eq(cols['D'].sum(), 60.0))

# ============================================================
#  IntColumn.load_from_1d_array
# ============================================================
def test_int_load_from_1d_array_basic():
    arr = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    col = IntColumn.load_from_1d_array(arr)
    check('int 1d length', col.length == 5)
    check('int 1d sum', col.sum() == 150)

def test_int_load_from_1d_array_zero_copy():
    arr = np.array([100, 200, 300], dtype=np.int32)
    col = IntColumn.load_from_1d_array(arr)
    check('int 1d zero-copy before', col.sum() == 600)
    arr[0] = 999
    check('int 1d zero-copy after', col.sum() == 1499)

def test_int_load_from_1d_array_negative():
    arr = np.array([-1, -2, -3], dtype=np.int32)
    col = IntColumn.load_from_1d_array(arr)
    check('int 1d negative sum', col.sum() == -6)

# ============================================================
#  IntColumn.load_from_2d_array
# ============================================================
def test_int_load_from_2d_array_basic():
    data = np.array([[1, 10, 100],
                     [2, 20, 200],
                     [3, 30, 300]], dtype=np.int32, order='F')
    cols = IntColumn.load_from_2d_array(data, ['A', 'B', 'C'])
    check('int 2d keys', sorted(cols.keys()) == ['A', 'B', 'C'])
    check('int 2d A sum', cols['A'].sum() == 6)
    check('int 2d B sum', cols['B'].sum() == 60)
    check('int 2d C sum', cols['C'].sum() == 600)

def test_int_load_from_2d_array_zero_copy():
    data = np.array([[1, 10],
                     [2, 20],
                     [3, 30]], dtype=np.int32, order='F')
    cols = IntColumn.load_from_2d_array(data, ['X', 'Y'])
    check('int 2d zero-copy before', cols['X'].sum() == 6)
    data[0, 0] = 999
    check('int 2d zero-copy after', cols['X'].sum() == 1004)

def test_int_load_from_2d_array_negative():
    data = np.array([[-1, 5],
                     [-2, 6],
                     [-3, 7]], dtype=np.int32, order='F')
    cols = IntColumn.load_from_2d_array(data, ['Neg', 'Pos'])
    check('int 2d neg sum', cols['Neg'].sum() == -6)
    check('int 2d pos sum', cols['Pos'].sum() == 18)

# ============================================================
#  View column: append/concat should fail
# ============================================================
def test_double_view_rejects_append():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    col = DoubleColumn.load_from_1d_array(arr)
    check('view append RuntimeError',
          expect_exception(RuntimeError, col.append, 4.0))

def test_double_view_rejects_concat():
    arr = np.array([1.0, 2.0], dtype=np.float64)
    col = DoubleColumn.load_from_1d_array(arr)
    other = DoubleColumn()
    other.append(3.0)
    check('view concat RuntimeError',
          expect_exception(RuntimeError, col.concat, other))

def test_int_view_rejects_append():
    arr = np.array([1, 2, 3], dtype=np.int32)
    col = IntColumn.load_from_1d_array(arr)
    check('int view append RuntimeError',
          expect_exception(RuntimeError, col.append, 4))

def test_int_view_rejects_concat():
    arr = np.array([1, 2], dtype=np.int32)
    col = IntColumn.load_from_1d_array(arr)
    other = IntColumn()
    other.append(3)
    check('int view concat RuntimeError',
          expect_exception(RuntimeError, col.concat, other))

# ============================================================
#  View column: copy creates independent owned column
# ============================================================
def test_double_view_copy_independent():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    col = DoubleColumn.load_from_1d_array(arr)
    copied = col.copy()
    arr[0] = 999.0
    check('view copy sum unchanged', approx_eq(copied.sum(), 6.0))
    check('view copy length', copied.length == 3)
    copied.append(4.0)
    check('view copy append works', copied.length == 4)

def test_int_view_copy_independent():
    arr = np.array([10, 20, 30], dtype=np.int32)
    col = IntColumn.load_from_1d_array(arr)
    copied = col.copy()
    arr[0] = 999
    check('int view copy sum unchanged', copied.sum() == 60)
    copied.append(40)
    check('int view copy append works', copied.length == 4)

# ============================================================
#  Column roll_xxx: zero-copy for DoubleColumn
# ============================================================
def test_rolling_double_view_zero_copy():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    col = DoubleColumn.load_from_1d_array(arr)
    result = col.roll_sum(3)
    check('rolling view sum length', result.length == 5)
    check('rolling view sum w3', approx_eq(result[2], 6.0))
    check('rolling view sum w3+1', approx_eq(result[3], 9.0))
    check('rolling view sum w3+2', approx_eq(result[4], 12.0))

def test_rolling_double_view_mean():
    arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)
    col = DoubleColumn.load_from_1d_array(arr)
    result = col.roll_mean(3)
    check('rolling view mean w3', approx_eq(result[2], 20.0))
    check('rolling view mean w3+1', approx_eq(result[3], 30.0))
    check('rolling view mean w3+2', approx_eq(result[4], 40.0))

def test_rolling_double_view_std():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    col = DoubleColumn.load_from_1d_array(arr)
    result = col.roll_std(3, ddof=False)
    expected = math.sqrt(2.0 / 3.0)
    check('rolling view std w3', approx_eq(result[2], expected))
    check('rolling view std w3+1', approx_eq(result[3], expected))
    check('rolling view std w3+2', approx_eq(result[4], expected))

def test_rolling_double_view_min():
    arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=np.float64)
    col = DoubleColumn.load_from_1d_array(arr)
    result = col.roll_min(3)
    check('rolling view min w3', approx_eq(result[2], 1.0))
    check('rolling view min w3+1', approx_eq(result[3], 1.0))
    check('rolling view min w3+2', approx_eq(result[4], 1.0))

def test_rolling_double_view_max():
    arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=np.float64)
    col = DoubleColumn.load_from_1d_array(arr)
    result = col.roll_max(3)
    check('rolling view max w3', approx_eq(result[2], 4.0))
    check('rolling view max w3+1', approx_eq(result[3], 4.0))
    check('rolling view max w3+2', approx_eq(result[4], 5.0))

def test_rolling_int_col_still_works():
    col = IntColumn()
    for v in [1, 2, 3, 4, 5]:
        col.append(v)
    result = col.roll_sum(3)
    check('rolling int sum w3', approx_eq(result[2], 6.0))
    check('rolling int sum w3+2', approx_eq(result[4], 12.0))

# ============================================================
#  Linalg: euclidean_dist with zero-copy columns
# ============================================================
def _build_df_from_2d(data, names):
    """Helper: build FastDataFrame from Fortran-order 2D array via zero-copy"""
    cols = DoubleColumn.load_from_2d_array(data, names)
    df = FastDataFrame()
    df.df_length = data.shape[0]
    df.index = RangeIndex(0, data.shape[0])
    for name in names:
        df.columns[name] = cols[name]
    return df

def test_linalg_euclidean_dist_zero_copy():
    data = np.array([[1.0, 0.0],
                     [0.0, 1.0],
                     [1.0, 1.0]], dtype=np.float64, order='F')
    df = _build_df_from_2d(data, ['X', 'Y'])
    dist_df = euclidean_dist(df)
    check('euc dist rows', dist_df.df_length == 3)
    check('euc dist cols', len(dist_df.columns) == 3)
    names = list(dist_df.columns.keys())
    check('euc dist d00', approx_eq(dist_df[names[0]][0], 0.0))
    check('euc dist d01', approx_eq(dist_df[names[1]][0], math.sqrt(2.0)))
    check('euc dist d02', approx_eq(dist_df[names[2]][0], 1.0))
    check('euc dist symmetric', approx_eq(dist_df[names[0]][1], dist_df[names[1]][0]))

def test_linalg_manhattan_dist_zero_copy():
    data = np.array([[1.0, 0.0],
                     [0.0, 1.0],
                     [1.0, 1.0]], dtype=np.float64, order='F')
    df = _build_df_from_2d(data, ['X', 'Y'])
    dist_df = manhattan_dist(df)
    names = list(dist_df.columns.keys())
    check('manh dist d01', approx_eq(dist_df[names[1]][0], 2.0))
    check('manh dist d02', approx_eq(dist_df[names[2]][0], 1.0))

def test_linalg_cosine_dist_zero_copy():
    data = np.array([[1.0, 0.0],
                     [1.0, 1.0]], dtype=np.float64, order='F')
    df = _build_df_from_2d(data, ['X', 'Y'])
    dist_df = cosine_dist(df)
    names = list(dist_df.columns.keys())
    expected = 1.0 - 1.0 / math.sqrt(2.0)
    check('cos dist d01', approx_eq(dist_df[names[1]][0], expected))

def test_linalg_cosine_similarity_zero_copy():
    data = np.array([[1.0, 0.0],
                     [1.0, 1.0]], dtype=np.float64, order='F')
    df = _build_df_from_2d(data, ['X', 'Y'])
    sim_df = cosine_similarity(df)
    names = list(sim_df.columns.keys())
    check('cos sim d01', approx_eq(sim_df[names[1]][0], 1.0 / math.sqrt(2.0)))
    check('cos sim d00', approx_eq(sim_df[names[0]][0], 1.0))

def test_linalg_hamming_dist_zero_copy():
    data = np.array([[1.0, 2.0, 3.0],
                     [1.0, 5.0, 3.0]], dtype=np.float64, order='F')
    df = _build_df_from_2d(data, ['A', 'B', 'C'])
    dist_df = hamming_dist(df)
    names = list(dist_df.columns.keys())
    check('hamm dist d01', approx_eq(dist_df[names[1]][0], 1.0 / 3.0))

def test_linalg_jaccard_sim_zero_copy():
    data = np.array([[1.0, 0.0, 1.0],
                     [1.0, 1.0, 0.0]], dtype=np.float64, order='F')
    df = _build_df_from_2d(data, ['A', 'B', 'C'])
    sim_df = jaccard_sim(df)
    names = list(sim_df.columns.keys())
    # intersection=1({0}), union=3({0,1,2}), jaccard=1/3
    check('jacc sim d01', approx_eq(sim_df[names[1]][0], 1.0 / 3.0))

def test_linalg_norm_zero_copy():
    data = np.array([[1.0, 2.0],
                     [3.0, 4.0]], dtype=np.float64, order='F')
    df = _build_df_from_2d(data, ['X', 'Y'])
    n = norm(df)
    expected = math.sqrt(1 + 4 + 9 + 16)
    check('norm frobenius', approx_eq(n, expected))

def test_linalg_dot_zero_copy():
    a = np.array([[1.0, 2.0],
                  [3.0, 4.0]], dtype=np.float64, order='F')
    b = np.array([[5.0, 6.0],
                  [7.0, 8.0]], dtype=np.float64, order='F')
    df_a = _build_df_from_2d(a, ['A', 'B'])
    df_b = _build_df_from_2d(b, ['C', 'D'])
    result = dot(df_a, df_b)
    names = list(result.columns.keys())
    check('dot rows', result.df_length == 2)
    check('dot [0][0]', approx_eq(result[names[0]][0], 19.0))   # 1*5+2*7
    check('dot [0][1]', approx_eq(result[names[1]][0], 22.0))   # 1*6+2*8
    check('dot [1][0]', approx_eq(result[names[0]][1], 43.0))   # 3*5+4*7
    check('dot [1][1]', approx_eq(result[names[1]][1], 50.0))   # 3*6+4*8

def test_linalg_transpose_zero_copy():
    data = np.array([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0]], dtype=np.float64, order='F')
    df = _build_df_from_2d(data, ['X', 'Y', 'Z'])
    result = transpose(df)
    names = list(result.columns.keys())
    check('transpose rows', result.df_length == 3)
    check('transpose cols', len(result.columns) == 2)
    # result column 'A' = original row 0 = [1,2,3]
    # result column 'B' = original row 1 = [4,5,6]
    check('transpose A[0]', approx_eq(result[names[0]][0], 1.0))
    check('transpose A[1]', approx_eq(result[names[0]][1], 2.0))
    check('transpose A[2]', approx_eq(result[names[0]][2], 3.0))
    check('transpose B[0]', approx_eq(result[names[1]][0], 4.0))
    check('transpose B[1]', approx_eq(result[names[1]][1], 5.0))
    check('transpose B[2]', approx_eq(result[names[1]][2], 6.0))

# ============================================================
#  Linalg with mixed IntColumn + DoubleColumn (non-contiguous)
# ============================================================
def test_linalg_euclidean_dist_mixed_columns():
    df = FastDataFrame()
    df.df_length = 2
    df.index = RangeIndex(0, 2)
    ic = IntColumn()
    ic.append(1); ic.append(0)
    dc = DoubleColumn()
    dc.append(0.0); dc.append(1.0)
    df.columns['I'] = ic
    df.columns['D'] = dc
    dist_df = euclidean_dist(df)
    names = list(dist_df.columns.keys())
    check('mixed dist rows', dist_df.df_length == 2)
    check('mixed dist d01', approx_eq(dist_df[names[1]][0], math.sqrt(2.0)))

# ============================================================
#  Linalg: non-contiguous DataFrame still works (fallback)
# ============================================================
def test_linalg_euclidean_dist_non_contiguous():
    df = FastDataFrame()
    df.df_length = 3
    df.index = RangeIndex(0, 3)
    for name, vals in [('A', [1.0, 0.0, 1.0]), ('B', [0.0, 1.0, 1.0])]:
        col = DoubleColumn()
        for v in vals:
            col.append(v)
        df.columns[name] = col
    dist_df = euclidean_dist(df)
    names = list(dist_df.columns.keys())
    check('non-contiguous dist', approx_eq(dist_df[names[1]][0], math.sqrt(2.0)))

# ============================================================
#  Edge: very large array
# ============================================================
def test_double_load_from_2d_array_many_rows():
    rows, cols_n = 10000, 3
    data = np.ones((rows, cols_n), dtype=np.float64, order='F')
    data[:, 1] = 2.0
    data[:, 2] = 3.0
    names = ['A', 'B', 'C']
    cols = DoubleColumn.load_from_2d_array(data, names)
    check('large rows A sum', approx_eq(cols['A'].sum(), rows * 1.0))
    check('large rows B sum', approx_eq(cols['B'].sum(), rows * 2.0))
    check('large rows C sum', approx_eq(cols['C'].sum(), rows * 3.0))

# ============================================================
#  Edge: memory lifecycle (base_obj keeps numpy alive)
# ============================================================
def test_double_view_base_obj_keeps_alive():
    import gc
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    col = DoubleColumn.load_from_1d_array(arr)
    del arr
    gc.collect()
    check('view alive after del arr', approx_eq(col.sum(), 6.0))

def test_double_2d_view_base_obj_keeps_alive():
    import gc
    data = np.array([[1.0, 10.0],
                     [2.0, 20.0],
                     [3.0, 30.0]], dtype=np.float64, order='F')
    cols = DoubleColumn.load_from_2d_array(data, ['X', 'Y'])
    del data
    gc.collect()
    check('2d view alive after del data', approx_eq(cols['X'].sum(), 6.0))
    check('2d view Y alive after del data', approx_eq(cols['Y'].sum(), 60.0))


# ============================================================
#  Run all
# ============================================================
if __name__ == '__main__':
    print("=== DoubleColumn.load_from_1d_array ===")
    test_double_load_from_1d_array_basic()
    test_double_load_from_1d_array_zero_copy()
    test_double_load_from_1d_array_empty()
    test_double_load_from_1d_array_single()
    test_double_load_from_1d_array_negative()
    test_double_load_from_1d_array_nan()

    print("=== DoubleColumn.load_from_2d_array ===")
    test_double_load_from_2d_array_basic()
    test_double_load_from_2d_array_zero_copy()
    test_double_load_from_2d_array_1row()
    test_double_load_from_2d_array_1col()
    test_double_load_from_2d_array_large()
    test_double_load_from_2d_array_negative_vals()
    test_double_load_from_2d_array_asfortran_conversion()

    print("=== IntColumn.load_from_1d_array ===")
    test_int_load_from_1d_array_basic()
    test_int_load_from_1d_array_zero_copy()
    test_int_load_from_1d_array_negative()

    print("=== IntColumn.load_from_2d_array ===")
    test_int_load_from_2d_array_basic()
    test_int_load_from_2d_array_zero_copy()
    test_int_load_from_2d_array_negative()

    print("=== View column restrictions ===")
    test_double_view_rejects_append()
    test_double_view_rejects_concat()
    test_int_view_rejects_append()
    test_int_view_rejects_concat()

    print("=== View copy independence ===")
    test_double_view_copy_independent()
    test_int_view_copy_independent()

    print("=== RollingWindow zero-copy ===")
    test_rolling_double_view_zero_copy()
    test_rolling_double_view_mean()
    test_rolling_double_view_std()
    test_rolling_double_view_min()
    test_rolling_double_view_max()
    test_rolling_int_col_still_works()

    print("=== Linalg zero-copy ===")
    test_linalg_euclidean_dist_zero_copy()
    test_linalg_manhattan_dist_zero_copy()
    test_linalg_cosine_dist_zero_copy()
    test_linalg_cosine_similarity_zero_copy()
    test_linalg_hamming_dist_zero_copy()
    test_linalg_jaccard_sim_zero_copy()
    test_linalg_norm_zero_copy()
    test_linalg_dot_zero_copy()
    test_linalg_transpose_zero_copy()

    print("=== Linalg mixed / non-contiguous ===")
    test_linalg_euclidean_dist_mixed_columns()
    test_linalg_euclidean_dist_non_contiguous()

    print("=== Edge cases ===")
    test_double_load_from_2d_array_many_rows()
    test_double_view_base_obj_keeps_alive()
    test_double_2d_view_base_obj_keeps_alive()

    print(f"\nResults: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
