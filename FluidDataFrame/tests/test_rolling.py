import math
import sys
sys.path.insert(0, '.')
from fluidframe.core.intcolumn import IntColumn
from fluidframe.core.doublecolumn import DoubleColumn
from fluidframe.core.dataframe import DataFrame

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

def is_nan(v):
    return v != v

def _make_int(vals):
    col = IntColumn()
    for v in vals: col.append(v)
    return col

def _make_double(vals):
    col = DoubleColumn()
    for v in vals: col.append(float(v))
    return col

def _make_df(columns_data):
    df = DataFrame()
    n = len(columns_data[0][1])
    for name, vals in columns_data:
        df[name] = _make_int(vals)
    df.df_length = n
    return df


class TestRollingMean:
    def test_basic(self):
        df = _make_df([('A', [1,2,3,4,5,6,7,8])])
        m = df.rolling(3, 'A').mean()
        check('len', m.length == 8)
        check('nan0', is_nan(m[0]))
        check('nan1', is_nan(m[1]))
        check('mean2', near(m[2], 2.0))
        check('mean3', near(m[3], 3.0))
        check('mean7', near(m[7], 7.0))

    def test_window1(self):
        df = _make_df([('A', [1,2,3,4,5,6,7,8])])
        m = df.rolling(1, 'A').mean()
        check('w1_0', near(m[0], 1.0))
        check('w1_7', near(m[7], 8.0))

    def test_window_equals_len(self):
        df = _make_df([('A', [1,2,3,4,5,6,7,8])])
        m = df.rolling(8, 'A').mean()
        check('w8_nan', all(is_nan(m[i]) for i in range(7)))
        check('w8_mean', near(m[7], 4.5))


class TestRollingSum:
    def test_basic(self):
        df = _make_df([('A', [1,2,3,4,5,6,7,8])])
        s = df.rolling(3, 'A').sum()
        check('sum2', near(s[2], 6.0))
        check('sum3', near(s[3], 9.0))
        check('sum7', near(s[7], 21.0))

    def test_window1(self):
        df = _make_df([('A', [1,2,3,4,5,6,7,8])])
        s = df.rolling(1, 'A').sum()
        check('w1_sum3', near(s[3], 4.0))


class TestRollingMinMax:
    def test_min(self):
        df = _make_df([('A', [1,2,3,4,5,6,7,8])])
        mn = df.rolling(3, 'A').min()
        check('min2', near(mn[2], 1.0))
        check('min4', near(mn[4], 3.0))
        check('min7', near(mn[7], 6.0))

    def test_max(self):
        df = _make_df([('A', [1,2,3,4,5,6,7,8])])
        mx = df.rolling(3, 'A').max()
        check('max2', near(mx[2], 3.0))
        check('max4', near(mx[4], 5.0))
        check('max7', near(mx[7], 8.0))


class TestRollingVarStd:
    def test_var(self):
        df = _make_df([('A', [1,2,3,4,5,6,7,8])])
        v = df.rolling(3, 'A').var()
        check('var2', near(v[2], 2.0/3.0, 1e-6))
        check('var5', near(v[5], 2.0/3.0, 1e-6))

    def test_std(self):
        df = _make_df([('A', [1,2,3,4,5,6,7,8])])
        s = df.rolling(3, 'A').std()
        check('std2', near(s[2], math.sqrt(2.0/3.0), 1e-6))
        check('std5', near(s[5], math.sqrt(2.0/3.0), 1e-6))


class TestRollingMedianQuantile:
    def test_median(self):
        df = _make_df([('A', [1,2,3,4,5,6,7,8])])
        med = df.rolling(3, 'A').median()
        check('med2', near(med[2], 2.0))
        check('med4', near(med[4], 4.0))
        check('med7', near(med[7], 7.0))

    def test_quantile(self):
        df = _make_df([('A', [1,2,3,4,5,6,7,8])])
        q = df.rolling(3, 'A').quantile(0.25)
        check('q25_2', near(q[2], 1.5, 1e-6))
        check('q25_5', near(q[5], 4.5, 1e-6))


class TestRollingSkewKurt:
    def test_skew(self):
        df = _make_df([('A', [1,2,3,4,5,6,7,8])])
        sk = df.rolling(3, 'A').skew()
        check('skew2', near(sk[2], 0.0, 1e-6))

    def test_kurt(self):
        df = _make_df([('A', list(range(1, 13)))])
        ku = df.rolling(4, 'A').kurt()
        check('kurt_len', ku.length == 12)
        check('kurt_nan', is_nan(ku[0]) and is_nan(ku[1]) and is_nan(ku[2]))
        check('kurt_3', near(ku[3], -1.36, 1e-6))


class TestRollingCorrCov:
    def test_corr(self):
        df = _make_df([('A', [1,2,3,4,5,6]), ('B', [2,4,6,8,10,12])])
        r_a = df.rolling(3, 'A')
        r_b = df.rolling(3, 'B')
        corr = r_a.corr(r_b)
        check('corr2', near(corr[2], 1.0, 1e-6))
        check('corr5', near(corr[5], 1.0, 1e-6))

    def test_corr_neg(self):
        df = _make_df([('A', [1,2,3,4,5,6]), ('B', [2,4,6,8,10,12])])
        df['C'] = _make_int([6,5,4,3,2,1])
        df.df_length = 6
        r_a = df.rolling(3, 'A')
        r_c = df.rolling(3, 'C')
        corr = r_a.corr(r_c)
        check('corr_neg', near(corr[2], -1.0, 1e-6))

    def test_cov(self):
        df = _make_df([('A', [1,2,3,4,5,6]), ('B', [2,4,6,8,10,12])])
        r_a = df.rolling(3, 'A')
        r_b = df.rolling(3, 'B')
        cov = r_a.cov(r_b)
        check('cov2', near(cov[2], 4.0/3.0, 1e-6))


class TestRollingErrors:
    def test_non_numeric(self):
        df_str = DataFrame()
        from fluidframe.core.asciicolumn import AsciiColumn
        cc = AsciiColumn()
        for v in ["a", "b", "c"]: cc.append(v)
        df_str['S'] = cc
        df_str.df_length = 3
        try:
            df_str.rolling(2, 'S')
            check('non_numeric', False)
        except TypeError:
            check('non_numeric', True)

    def test_missing_col(self):
        df = _make_df([('A', [1,2,3])])
        try:
            df.rolling(2, 'Z')
            check('missing_col', False)
        except KeyError:
            check('missing_col', True)

    def test_window_zero(self):
        df = _make_df([('A', [1,2,3])])
        try:
            df.rolling(0, 'A')
            check('window_zero', False)
        except ValueError:
            check('window_zero', True)


def run_all():
    print('=== Rolling Tests ===')
    for cls in [TestRollingMean, TestRollingSum, TestRollingMinMax,
                TestRollingVarStd, TestRollingMedianQuantile,
                TestRollingSkewKurt, TestRollingCorrCov, TestRollingErrors]:
        for name in dir(cls):
            if name.startswith('test_'):
                getattr(cls(), name)()
    print(f'\nResults: {passed} passed, {failed} failed')
    if failed: sys.exit(1)

if __name__ == '__main__':
    run_all()
