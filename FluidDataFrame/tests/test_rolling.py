import math
import sys
sys.path.insert(0, '.')
from fluidframe.core.intcolumn import IntColumn
from fluidframe.core.doublecolumn import DoubleColumn
from fluidframe.core.asciicolumn import AsciiColumn
from fluidframe.core.series import Series
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

def expect_exception(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        return False
    except exc_type:
        return True
    except Exception:
        return False

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
        m = df.rolling(3).mean()['A']
        check('len', m.length == 8)
        check('nan0', is_nan(m[0]))
        check('nan1', is_nan(m[1]))
        check('mean2', near(m[2], 2.0))
        check('mean3', near(m[3], 3.0))
        check('mean7', near(m[7], 7.0))

    def test_window1(self):
        df = _make_df([('A', [1,2,3,4,5,6,7,8])])
        m = df.rolling(1).mean()['A']
        check('w1_0', near(m[0], 1.0))
        check('w1_7', near(m[7], 8.0))

    def test_window_equals_len(self):
        df = _make_df([('A', [1,2,3,4,5,6,7,8])])
        m = df.rolling(8).mean()['A']
        check('w8_nan', all(is_nan(m[i]) for i in range(7)))
        check('w8_mean', near(m[7], 4.5))

    def test_window_gt_len(self):
        df = _make_df([('A', [1,2,3])])
        m = df.rolling(8).mean()['A']
        check('w>len all nan', all(is_nan(m[i]) for i in range(3)))

    def test_whole_table(self):
        df = _make_df([('A', [1,2,3,4]), ('B', [10,20,30,40])])
        m = df.rolling(3).mean()
        check('cols', list(m._columns.keys()) == ['A', 'B'])
        check('A2', near(m['A'][2], 2.0))
        check('B3', near(m['B'][3], 30.0))

    def test_min_periods(self):
        df = _make_df([('A', [1,2,3,4,5,6,7,8])])
        m = df.rolling(4, min_periods=2).mean()['A']
        check('mp_nan0', is_nan(m[0]))
        check('mp1', near(m[1], 1.5))
        check('mp3', near(m[3], 2.5))

    def test_center(self):
        df = _make_df([('A', [1,2,3,4,5,6,7,8])])
        m = df.rolling(3, center=True).mean()['A']
        check('center_nan0', is_nan(m[0]))
        check('center[1]', near(m[1], 2.0))
        check('center[2]', near(m[2], 3.0))
        check('center[6]', near(m[6], 7.0))
        check('center_nan7', is_nan(m[7]))

    def test_series_rolling(self):
        df = _make_df([('A', [1,2,3,4,5,6,7,8])])
        s = df['A']
        m = s.rolling(3).mean()
        check('series_type', isinstance(m, Series))
        check('series[2]', near(m[2], 2.0))


class TestRollingSum:
    def test_basic(self):
        df = _make_df([('A', [1,2,3,4,5,6,7,8])])
        s = df.rolling(3).sum()['A']
        check('sum2', near(s[2], 6.0))
        check('sum3', near(s[3], 9.0))
        check('sum7', near(s[7], 21.0))

    def test_window1(self):
        df = _make_df([('A', [1,2,3,4,5,6,7,8])])
        s = df.rolling(1).sum()['A']
        check('w1_sum3', near(s[3], 4.0))


class TestRollingMinMax:
    def test_min(self):
        df = _make_df([('A', [1,2,3,4,5,6,7,8])])
        mn = df.rolling(3).min()['A']
        check('min_sentinel0', mn[0] == 2147483647)
        check('min2', mn[2] == 1)
        check('min4', mn[4] == 3)
        check('min7', mn[7] == 6)

    def test_max(self):
        df = _make_df([('A', [1,2,3,4,5,6,7,8])])
        mx = df.rolling(3).max()['A']
        check('max_sentinel0', mx[0] == -2147483647 - 1)
        check('max2', mx[2] == 3)
        check('max4', mx[4] == 5)
        check('max7', mx[7] == 8)

    def test_int_sentinel_no_leak(self):
        df = _make_df([('A', [1,2,3,4,5])])
        mn = df.rolling(4, min_periods=2).min()['A']
        mx = df.rolling(4, min_periods=2).max()['A']
        check('min_invalid sentinel', mn[0] == 2147483647)
        check('min_valid real', mn[1] == 1)
        check('max_invalid sentinel', mx[0] == -2147483647 - 1)
        check('max_valid real', mx[1] == 2)

    def test_center_min_max(self):
        df = _make_df([('A', [1,2,3,4,5])])
        mn = df.rolling(3, center=True).min()['A']
        mx = df.rolling(3, center=True).max()['A']
        check('center_min1', mn[1] == 1)
        check('center_max3', mx[3] == 5)


class TestRollingVarStd:
    def test_var_ddof_default(self):
        df = _make_df([('A', [1,2,3,4,5,6,7,8])])
        v = df.rolling(3).var()['A']
        check('var2', near(v[2], 1.0, 1e-9))
        check('var5', near(v[5], 1.0, 1e-9))

    def test_var_population(self):
        df = _make_df([('A', [1,2,3,4,5,6,7,8])])
        v = df.rolling(3).var(ddof=False)['A']
        check('var_pop2', near(v[2], 2.0/3.0, 1e-6))

    def test_std_ddof_default(self):
        df = _make_df([('A', [1,2,3,4,5,6,7,8])])
        s = df.rolling(3).std()['A']
        check('std2', near(s[2], 1.0, 1e-9))

    def test_std_population(self):
        df = _make_df([('A', [1,2,3,4,5,6,7,8])])
        s = df.rolling(3).std(ddof=False)['A']
        check('std_pop2', near(s[2], math.sqrt(2.0/3.0), 1e-6))

    def test_center_var(self):
        df = _make_df([('A', [1,2,3,4])])
        v = df.rolling(3, center=True).var()['A']
        check('center_var1', near(v[1], 1.0, 1e-9))


class TestRollingMedianQuantile:
    def test_median(self):
        df = _make_df([('A', [1,2,3,4,5,6,7,8])])
        med = df.rolling(3).median()['A']
        check('med2', near(med[2], 2.0))
        check('med4', near(med[4], 4.0))
        check('med7', near(med[7], 7.0))

    def test_quantile(self):
        df = _make_df([('A', [1,2,3,4,5,6,7,8])])
        q = df.rolling(3).quantile(0.25)['A']
        check('q25_2', near(q[2], 1.5, 1e-6))
        check('q25_5', near(q[5], 4.5, 1e-6))

    def test_center_median(self):
        df = _make_df([('A', [1,2,3,4,5])])
        med = df.rolling(3, center=True).median()['A']
        check('center_med1', near(med[1], 2.0))
        check('center_med3', near(med[3], 4.0))


class TestRollingSkewKurt:
    def test_skew(self):
        df = _make_df([('A', [1,2,3,4,5,6,7,8])])
        sk = df.rolling(3).skew()['A']
        check('skew2', near(sk[2], 0.0, 1e-6))
        check('skew_nan0', is_nan(sk[0]))

    def test_kurt(self):
        df = _make_df([('A', list(range(1, 13)))])
        ku = df.rolling(4).kurt()['A']
        check('kurt_len', ku.length == 12)
        check('kurt_nan', is_nan(ku[0]) and is_nan(ku[1]) and is_nan(ku[2]))
        check('kurt_3', near(ku[3], -1.36, 1e-6))


class TestRollingCorrCov:
    def test_corr(self):
        df = _make_df([('A', [1,2,3,4,5,6]), ('B', [2,4,6,8,10,12])])
        corr = df.rolling(3).corr(df['B'])
        check('corr2', near(corr['A'][2], 1.0, 1e-6))
        check('corr5', near(corr['A'][5], 1.0, 1e-6))

    def test_corr_neg(self):
        df = _make_df([('A', [1,2,3,4,5,6]), ('B', [2,4,6,8,10,12])])
        df['C'] = _make_int([6,5,4,3,2,1])
        df.df_length = 6
        corr = df.rolling(3).corr(df['C'])
        check('corr_neg2', near(corr['A'][2], -1.0, 1e-6))
        check('corr_neg5', near(corr['A'][5], -1.0, 1e-6))

    def test_corr_rolling_other(self):
        df = _make_df([('A', [1,2,3,4,5,6]), ('B', [2,4,6,8,10,12])])
        corr = df['A'].rolling(3).corr(df['B'].rolling(3))
        check('corr_win2', near(corr[2], 1.0, 1e-6))

    def test_corr_int_double(self):
        df = _make_df([('A', [1,2,3,4,5,6])])
        df['D'] = _make_double([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
        df.df_length = 6
        corr = df.rolling(3).corr(df['D'])
        check('corr_mix2', near(corr['A'][2], 1.0, 1e-6))

    def test_corr_dataframe_other(self):
        df = _make_df([('A', [1,2,3,4,5,6]), ('B', [1,1,1,4,4,4])])
        other = _make_df([('A', [2,4,6,8,10,12]), ('B', [1,1,1,4,4,4])])
        corr = df.rolling(3).corr(other)
        check('corr_dfA2', near(corr['A'][2], 1.0, 1e-6))
        check('corr_dfB2', abs(corr['B'][2]) < 1e-9)

    def test_cov(self):
        df = _make_df([('A', [1,2,3,4,5,6]), ('B', [2,4,6,8,10,12])])
        cov = df.rolling(3).cov(df['B'])
        check('cov2', near(cov['A'][2], 4.0/3.0, 1e-6))

    def test_cov_min_periods(self):
        df = _make_df([('A', [1,2,3,4,5,6]), ('B', [2,4,6,8,10,12])])
        cov = df.rolling(4, min_periods=2).cov(df['B'])
        check('cov_mp_nan0', is_nan(cov['A'][0]))
        check('cov_mp1', near(cov['A'][1], 0.5, 1e-6))


class TestRollingErrors:
    def test_non_numeric(self):
        df_str = DataFrame()
        cc = AsciiColumn()
        for v in ["a", "b", "c"]: cc.append(v)
        df_str['S'] = cc
        df_str.df_length = 3
        check('non_numeric', expect_exception(TypeError, df_str.rolling(2).mean))

    def test_mixed_skips_string(self):
        df = DataFrame()
        cc = AsciiColumn()
        for v in ["a", "b", "c"]: cc.append(v)
        df['S'] = cc
        df['A'] = _make_int([1,2,3])
        df.df_length = 3
        m = df.rolling(2).mean()
        check('mixed_cols', list(m._columns.keys()) == ['A'])
        check('mixed_A1', near(m['A'][1], 1.5))

    def test_window_zero(self):
        df = _make_df([('A', [1,2,3])])
        check('window_zero', expect_exception(ValueError, df.rolling, 0))

    def test_min_periods_gt_window(self):
        df = _make_df([('A', [1,2,3,4])])
        check('mp_gt_window', expect_exception(ValueError, lambda: df.rolling(3, min_periods=4).mean()))

    def test_axis_unsupported(self):
        df = _make_df([('A', [1,2,3])])
        check('axis1', expect_exception(NotImplementedError, df.rolling, 3, None, False, 1))


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