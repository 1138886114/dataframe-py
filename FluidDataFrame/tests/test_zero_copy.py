import math
import sys
sys.path.insert(0, '.')
import numpy as np
from fluidframe.core.intcolumn import IntColumn
from fluidframe.core.doublecolumn import DoubleColumn
from fluidframe.core.fastdataframe import FastDataFrame
from fluidframe.core.index import RangeIndex

passed = 0
failed = 0

def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed += 1
        print(f'  FAIL: {name}')

def near(a, b, tol=1e-9):
    return abs(a - b) < tol

def expect_exception(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        return False
    except exc_type:
        return True
    except Exception:
        return False


class TestDoubleLoad1D:
    def test_basic(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        col = DoubleColumn.load_from_1d_array(arr)
        check('length', col.length == 5)
        check('sum', near(col.sum(), 15.0))

    def test_zero_copy(self):
        arr = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        col = DoubleColumn.load_from_1d_array(arr)
        arr[0] = 999.0
        check('zero_copy', near(col.sum(), 1049.0))

    def test_empty(self):
        arr = np.array([], dtype=np.float64)
        col = DoubleColumn.load_from_1d_array(arr)
        check('empty', col.length == 0)

    def test_single(self):
        arr = np.array([42.0], dtype=np.float64)
        col = DoubleColumn.load_from_1d_array(arr)
        check('single', col.length == 1 and near(col[0], 42.0))

    def test_negative(self):
        arr = np.array([-1.5, -2.5, 3.5], dtype=np.float64)
        col = DoubleColumn.load_from_1d_array(arr)
        check('negative', near(col.sum(), -0.5))

    def test_nan(self):
        arr = np.array([1.0, float('nan'), 3.0], dtype=np.float64)
        col = DoubleColumn.load_from_1d_array(arr)
        check('nan', col.length == 3 and col[1] != col[1])


class TestDoubleLoad2D:
    def test_basic(self):
        data = np.array([[1.0, 10.0, 100.0], [2.0, 20.0, 200.0], [3.0, 30.0, 300.0]], dtype=np.float64, order='F')
        cols = DoubleColumn.load_from_2d_array(data, ['X', 'Y', 'Z'])
        check('keys', sorted(cols.keys()) == ['X', 'Y', 'Z'])
        check('X sum', near(cols['X'].sum(), 6.0))

    def test_zero_copy(self):
        data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float64, order='F')
        cols = DoubleColumn.load_from_2d_array(data, ['A', 'B'])
        data[0, 0] = 999.0
        check('zero_copy', near(cols['A'].sum(), 1004.0))

    def test_1row(self):
        data = np.array([[5.0, 6.0, 7.0]], dtype=np.float64, order='F')
        cols = DoubleColumn.load_from_2d_array(data, ['P', 'Q', 'R'])
        check('1row', cols['P'].length == 1 and near(cols['P'][0], 5.0))

    def test_1col(self):
        data = np.array([[1.0], [2.0], [3.0]], dtype=np.float64, order='F')
        cols = DoubleColumn.load_from_2d_array(data, ['Only'])
        check('1col', cols['Only'].length == 3)

    def test_large(self):
        data = np.random.randn(100, 5).astype(np.float64, order='F')
        names = [chr(65 + i) for i in range(5)]
        cols = DoubleColumn.load_from_2d_array(data, names)
        for i, name in enumerate(names):
            check(f'large {name}', near(cols[name].sum(), data[:, i].sum()))

    def test_negative(self):
        data = np.array([[-1.0, 4.0], [-2.0, 5.0], [-3.0, 6.0]], dtype=np.float64, order='F')
        cols = DoubleColumn.load_from_2d_array(data, ['Neg', 'Pos'])
        check('neg', near(cols['Neg'].sum(), -6.0))

    def test_fortran_conversion(self):
        data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float64)
        cols = DoubleColumn.load_from_2d_array(np.asfortranarray(data), ['C', 'D'])
        check('C sum', near(cols['C'].sum(), 6.0))

    def test_many_rows(self):
        data = np.ones((10000, 3), dtype=np.float64, order='F')
        data[:, 1] = 2.0; data[:, 2] = 3.0
        cols = DoubleColumn.load_from_2d_array(data, ['A', 'B', 'C'])
        check('A sum', near(cols['A'].sum(), 10000.0))
        check('B sum', near(cols['B'].sum(), 20000.0))


class TestIntLoad1D:
    def test_basic(self):
        arr = np.array([10, 20, 30, 40, 50], dtype=np.int32)
        col = IntColumn.load_from_1d_array(arr)
        check('length', col.length == 5)
        check('sum', col.sum() == 150)

    def test_zero_copy(self):
        arr = np.array([100, 200, 300], dtype=np.int32)
        col = IntColumn.load_from_1d_array(arr)
        arr[0] = 999
        check('zero_copy', col.sum() == 1499)

    def test_negative(self):
        arr = np.array([-1, -2, -3], dtype=np.int32)
        col = IntColumn.load_from_1d_array(arr)
        check('negative', col.sum() == -6)


class TestIntLoad2D:
    def test_basic(self):
        data = np.array([[1, 10, 100], [2, 20, 200], [3, 30, 300]], dtype=np.int32, order='F')
        cols = IntColumn.load_from_2d_array(data, ['A', 'B', 'C'])
        check('A sum', cols['A'].sum() == 6)

    def test_zero_copy(self):
        data = np.array([[1, 10], [2, 20], [3, 30]], dtype=np.int32, order='F')
        cols = IntColumn.load_from_2d_array(data, ['X', 'Y'])
        data[0, 0] = 999
        check('zero_copy', cols['X'].sum() == 1004)

    def test_negative(self):
        data = np.array([[-1, 5], [-2, 6], [-3, 7]], dtype=np.int32, order='F')
        cols = IntColumn.load_from_2d_array(data, ['Neg', 'Pos'])
        check('neg', cols['Neg'].sum() == -6)


class TestViewRestrictions:
    def test_double_rejects_append(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        col = DoubleColumn.load_from_1d_array(arr)
        check('append', expect_exception(RuntimeError, col.append, 4.0))

    def test_double_rejects_concat(self):
        arr = np.array([1.0, 2.0], dtype=np.float64)
        col = DoubleColumn.load_from_1d_array(arr)
        other = DoubleColumn(); other.append(3.0)
        check('concat', expect_exception(RuntimeError, col.concat, other))

    def test_int_rejects_append(self):
        arr = np.array([1, 2, 3], dtype=np.int32)
        col = IntColumn.load_from_1d_array(arr)
        check('append', expect_exception(RuntimeError, col.append, 4))

    def test_int_rejects_concat(self):
        arr = np.array([1, 2], dtype=np.int32)
        col = IntColumn.load_from_1d_array(arr)
        other = IntColumn(); other.append(3)
        check('concat', expect_exception(RuntimeError, col.concat, other))


class TestViewCopy:
    def test_double_copy(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        col = DoubleColumn.load_from_1d_array(arr)
        copied = col.copy()
        arr[0] = 999.0
        check('sum unchanged', near(copied.sum(), 6.0))
        copied.append(4.0)
        check('append works', copied.length == 4)

    def test_int_copy(self):
        arr = np.array([10, 20, 30], dtype=np.int32)
        col = IntColumn.load_from_1d_array(arr)
        copied = col.copy()
        arr[0] = 999
        check('sum unchanged', copied.sum() == 60)
        copied.append(40)
        check('append works', copied.length == 4)


class TestRollingZeroCopy:
    def test_double_sum(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        col = DoubleColumn.load_from_1d_array(arr)
        r = col.roll_sum(3)
        check('sum w3', near(r[2], 6.0) and near(r[4], 12.0))

    def test_double_mean(self):
        arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)
        col = DoubleColumn.load_from_1d_array(arr)
        r = col.roll_mean(3)
        check('mean w3', near(r[2], 20.0) and near(r[4], 40.0))

    def test_double_std(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        col = DoubleColumn.load_from_1d_array(arr)
        r = col.roll_std(3, ddof=False)
        check('std w3', near(r[2], math.sqrt(2.0/3.0)))

    def test_double_min_max(self):
        arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=np.float64)
        col = DoubleColumn.load_from_1d_array(arr)
        check('min', near(col.roll_min(3)[2], 1.0))
        check('max', near(col.roll_max(3)[2], 4.0))

    def test_int_still_works(self):
        col = IntColumn()
        for v in [1, 2, 3, 4, 5]: col.append(v)
        r = col.roll_sum(3)
        check('int sum', near(r[2], 6.0) and near(r[4], 12.0))


class TestLinalgZeroCopy:
    def _build_df(self, data, names):
        cols = DoubleColumn.load_from_2d_array(data, names)
        df = FastDataFrame()
        df.df_length = data.shape[0]
        df.index = RangeIndex(0, data.shape[0])
        for name in names: df.columns[name] = cols[name]
        return df

    def test_euclidean(self):
        data = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64, order='F')
        df = self._build_df(data, ['X', 'Y'])
        d = df.linalg.euclidean_dist()
        names = list(d.columns.keys())
        check('d00', near(d[names[0]][0], 0.0))
        check('d01', near(d[names[1]][0], math.sqrt(2.0)))

    def test_manhattan(self):
        data = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64, order='F')
        df = self._build_df(data, ['X', 'Y'])
        d = df.linalg.manhattan_dist()
        names = list(d.columns.keys())
        check('d01', near(d[names[1]][0], 2.0))

    def test_cosine(self):
        data = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float64, order='F')
        df = self._build_df(data, ['X', 'Y'])
        d = df.linalg.cosine_dist()
        names = list(d.columns.keys())
        check('cos', near(d[names[1]][0], 1.0 - 1.0/math.sqrt(2.0)))

    def test_cosine_sim(self):
        data = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float64, order='F')
        df = self._build_df(data, ['X', 'Y'])
        d = df.linalg.cosine_similarity()
        names = list(d.columns.keys())
        check('sim', near(d[names[1]][0], 1.0/math.sqrt(2.0)))

    def test_hamming(self):
        data = np.array([[1.0, 2.0, 3.0], [1.0, 5.0, 3.0]], dtype=np.float64, order='F')
        df = self._build_df(data, ['A', 'B', 'C'])
        d = df.linalg.hamming_dist()
        names = list(d.columns.keys())
        check('hamm', near(d[names[1]][0], 1.0/3.0))

    def test_jaccard(self):
        data = np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]], dtype=np.float64, order='F')
        df = self._build_df(data, ['A', 'B', 'C'])
        d = df.linalg.jaccard_sim()
        names = list(d.columns.keys())
        check('jacc', near(d[names[1]][0], 1.0/3.0))

    def test_norm(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64, order='F')
        df = self._build_df(data, ['X', 'Y'])
        check('norm', near(df.linalg.norm(), math.sqrt(30.0)))

    def test_dot(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64, order='F')
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64, order='F')
        df_a = self._build_df(a, ['A', 'B'])
        df_b = self._build_df(b, ['C', 'D'])
        r = df_a.linalg.dot(df_b)
        names = list(r.columns.keys())
        check('dot', near(r[names[0]][0], 19.0) and near(r[names[1]][1], 50.0))

    def test_transpose(self):
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64, order='F')
        df = self._build_df(data, ['X', 'Y', 'Z'])
        r = df.linalg.transpose()
        names = list(r.columns.keys())
        check('T rows', r.df_length == 3)
        check('T val', near(r[names[0]][0], 1.0) and near(r[names[0]][2], 3.0))


class TestLinalgMixed:
    def test_mixed_columns(self):
        df = FastDataFrame()
        df.df_length = 2; df.index = RangeIndex(0, 2)
        ic = IntColumn(); ic.append(1); ic.append(0)
        dc = DoubleColumn(); dc.append(0.0); dc.append(1.0)
        df.columns['I'] = ic; df.columns['D'] = dc
        d = df.linalg.euclidean_dist()
        names = list(d.columns.keys())
        check('mixed', near(d[names[1]][0], math.sqrt(2.0)))

    def test_non_contiguous(self):
        df = FastDataFrame()
        df.df_length = 3; df.index = RangeIndex(0, 3)
        for name, vals in [('A', [1.0, 0.0, 1.0]), ('B', [0.0, 1.0, 1.0])]:
            col = DoubleColumn()
            for v in vals: col.append(v)
            df.columns[name] = col
        d = df.linalg.euclidean_dist()
        names = list(d.columns.keys())
        check('non-contig', near(d[names[1]][0], math.sqrt(2.0)))


class TestMemoryLifecycle:
    def test_1d_keeps_alive(self):
        import gc
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        col = DoubleColumn.load_from_1d_array(arr)
        del arr; gc.collect()
        check('alive', near(col.sum(), 6.0))

    def test_2d_keeps_alive(self):
        import gc
        data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float64, order='F')
        cols = DoubleColumn.load_from_2d_array(data, ['X', 'Y'])
        del data; gc.collect()
        check('2d alive', near(cols['X'].sum(), 6.0) and near(cols['Y'].sum(), 60.0))


def run_all():
    print('=== Zero Copy Tests ===')
    for cls in [TestDoubleLoad1D, TestDoubleLoad2D, TestIntLoad1D, TestIntLoad2D,
                TestViewRestrictions, TestViewCopy, TestRollingZeroCopy,
                TestLinalgZeroCopy, TestLinalgMixed, TestMemoryLifecycle]:
        for name in dir(cls):
            if name.startswith('test_'):
                getattr(cls(), name)()
    print(f'\nResults: {passed} passed, {failed} failed')
    if failed: sys.exit(1)

if __name__ == '__main__':
    run_all()
