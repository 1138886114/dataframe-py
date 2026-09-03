import math
import sys
sys.path.insert(0, '..')
from fluidframe.core.intcolumn import IntColumn
from fluidframe.core.doublecolumn import DoubleColumn
from fluidframe.core.fastdataframe import FastDataFrame

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f'  FAIL: {name}')

def close(a, b, tol=1e-9):
    if math.isnan(a) and math.isnan(b): return True
    return abs(a - b) < tol

# ==================== IntColumn ====================

def test_int_count():
    col = IntColumn()
    check('int count empty', col.count() == 0)
    col.append(10)
    check('int count 1', col.count() == 1)
    for v in [20, 30, 40]: col.append(v)
    check('int count 4', col.count() == 4)

def test_int_kurtosis():
    col = IntColumn()
    check('int kurtosis empty', math.isnan(col.kurt()))
    for v in [1, 2, 3]: col.append(v)
    check('int kurtosis <4', math.isnan(col.kurt()))
    for v in [4, 5, 6, 7]: col.append(v)
    result = col.kurt()
    check('int kurtosis is number', not math.isnan(result))

def test_int_cumprod():
    col = IntColumn()
    for v in [1, 2, 3, 4]: col.append(v)
    r = col.cumprod()
    check('int cumprod len', r.length == 4)
    check('int cumprod [0]', r[0] == 1)
    check('int cumprod [1]', r[1] == 2)
    check('int cumprod [2]', r[2] == 6)
    check('int cumprod [3]', r[3] == 24)

def test_int_cumprod_empty():
    col = IntColumn()
    r = col.cumprod()
    check('int cumprod empty', r.length == 0)

def test_int_cumprod_single():
    col = IntColumn()
    col.append(5)
    r = col.cumprod()
    check('int cumprod single', r[0] == 5)

def test_int_cumprod_negative():
    col = IntColumn()
    for v in [-1, 2, -3]: col.append(v)
    r = col.cumprod()
    check('int cumprod neg [0]', r[0] == -1)
    check('int cumprod neg [1]', r[1] == -2)
    check('int cumprod neg [2]', r[2] == 6)

def test_int_describe():
    col = IntColumn()
    check('int describe empty', col.describe()['count'] == 0)
    for v in [1, 2, 3, 4, 5]: col.append(v)
    d = col.describe()
    check('int describe count', d['count'] == 5)
    check('int describe mean', close(d['mean'], 3.0))
    check('int describe min', close(d['min'], 1.0))
    check('int describe max', close(d['max'], 5.0))
    check('int describe 50%', close(d['50%'], 3.0))
    check('int describe has std', 'std' in d)
    check('int describe has 25%', '25%' in d)
    check('int describe has 75%', '75%' in d)

# ==================== DoubleColumn ====================

def test_double_count():
    col = DoubleColumn()
    check('double count empty', col.count() == 0)
    col.append(1.5)
    check('double count 1', col.count() == 1)
    for v in [2.5, 3.5]: col.append(v)
    check('double count 3', col.count() == 3)

def test_double_kurtosis():
    col = DoubleColumn()
    check('double kurtosis empty', math.isnan(col.kurt()))
    for v in [1.0, 2.0, 3.0]: col.append(v)
    check('double kurtosis <4', math.isnan(col.kurt()))
    for v in [4.0, 5.0, 6.0, 7.0]: col.append(v)
    result = col.kurt()
    check('double kurtosis is number', not math.isnan(result))
    check('double kurtosis matches kurt', close(result, col.kurt()))

def test_double_cumprod():
    col = DoubleColumn()
    for v in [1.0, 2.0, 3.0, 4.0]: col.append(v)
    r = col.cumprod()
    check('double cumprod len', r.length == 4)
    check('double cumprod [0]', close(r[0], 1.0))
    check('double cumprod [1]', close(r[1], 2.0))
    check('double cumprod [2]', close(r[2], 6.0))
    check('double cumprod [3]', close(r[3], 24.0))

def test_double_cumprod_fractional():
    col = DoubleColumn()
    for v in [0.5, 2.0, 3.0]: col.append(v)
    r = col.cumprod()
    check('double cumprod frac [0]', close(r[0], 0.5))
    check('double cumprod frac [1]', close(r[1], 1.0))
    check('double cumprod frac [2]', close(r[2], 3.0))

def test_double_cumprod_empty():
    col = DoubleColumn()
    r = col.cumprod()
    check('double cumprod empty', r.length == 0)

def test_double_describe():
    col = DoubleColumn()
    check('double describe empty', col.describe()['count'] == 0)
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]: col.append(v)
    d = col.describe()
    check('double describe count', d['count'] == 5)
    check('double describe mean', close(d['mean'], 3.0))
    check('double describe std', close(d['std'], 1.5811388300841898))
    check('double describe min', close(d['min'], 1.0))
    check('double describe max', close(d['max'], 5.0))
    check('double describe 25%', close(d['25%'], 2.0))
    check('double describe 50%', close(d['50%'], 3.0))
    check('double describe 75%', close(d['75%'], 4.0))

# ==================== DataFrame ====================

def _make_numeric_df():
    df = FastDataFrame()
    df['A'] = IntColumn(4)
    df['B'] = DoubleColumn(4)
    for i in range(4):
        df['A'].append(i + 1)
        df['B'].append((i + 1) * 1.5)
    return df

def test_df_count():
    df = _make_numeric_df()
    r = df.count()
    check('df count A', r['A'] == 4)
    check('df count B', r['B'] == 4)

def test_df_mean():
    df = _make_numeric_df()
    r = df.mean()
    check('df mean A', close(r['A'], 2.5))
    check('df mean B', close(r['B'], 3.75))

def test_df_median():
    df = _make_numeric_df()
    r = df.median()
    check('df median A', close(r['A'], 2.5))
    check('df median B', close(r['B'], 3.75))

def test_df_std():
    df = _make_numeric_df()
    r = df.std()
    check('df std A', close(r['A'], 1.2909944487358056))
    check('df std B', close(r['B'], 1.9364916731037085))

def test_df_var():
    df = _make_numeric_df()
    r = df.var()
    check('df var A', close(r['A'], 1.6666666666666667))
    check('df var B', close(r['B'], 3.75))

def test_df_max():
    df = _make_numeric_df()
    r = df.max()
    check('df max A', r['A'] == 4)
    check('df max B', close(r['B'], 6.0))

def test_df_min():
    df = _make_numeric_df()
    r = df.min()
    check('df min A', r['A'] == 1)
    check('df min B', close(r['B'], 1.5))

def test_df_sum():
    df = _make_numeric_df()
    r = df.sum()
    check('df sum A', r['A'] == 10)
    check('df sum B', close(r['B'], 15.0))

def test_df_quantile():
    df = _make_numeric_df()
    r = df.quantile(0.5)
    check('df quantile 0.5 A', close(r['A'], 2.5))
    check('df quantile 0.5 B', close(r['B'], 3.75))

def test_df_skew():
    df = _make_numeric_df()
    r = df.skew()
    check('df skew A has value', not math.isnan(r['A']))
    check('df skew B has value', not math.isnan(r['B']))

def test_df_kurt():
    df = _make_numeric_df()
    r = df.kurt()
    check('df kurt A', not math.isnan(r['A']))
    check('df kurt B', not math.isnan(r['B']))

def test_df_kurtosis():
    df = _make_numeric_df()
    r = df.kurt()
    k = df.kurt()
    check('df kurtosis matches kurt', close(r['A'], k['A']))

def test_df_cumsum():
    df = _make_numeric_df()
    r = df.cumsum()
    check('df cumsum A [0]', r['A'][0] == 1)
    check('df cumsum A [3]', r['A'][3] == 10)
    check('df cumsum B [0]', close(r['B'][0], 1.5))
    check('df cumsum B [3]', close(r['B'][3], 15.0))

def test_df_cumprod():
    df = _make_numeric_df()
    r = df.cumprod()
    check('df cumprod A [0]', r['A'][0] == 1)
    check('df cumprod A [3]', r['A'][3] == 24)
    check('df cumprod B [0]', close(r['B'][0], 1.5))
    check('df cumprod B [3]', close(r['B'][3], 121.5))

def test_df_cummax():
    df = _make_numeric_df()
    r = df.cummax()
    check('df cummax A [0]', r['A'][0] == 1)
    check('df cummax A [3]', r['A'][3] == 4)
    check('df cummax B [3]', close(r['B'][3], 6.0))

def test_df_cummin():
    df = _make_numeric_df()
    r = df.cummin()
    check('df cummin A [0]', r['A'][0] == 1)
    check('df cummin A [3]', r['A'][3] == 1)
    check('df cummin B [0]', close(r['B'][0], 1.5))

def test_df_mode():
    df = FastDataFrame()
    df['A'] = IntColumn(5)
    for v in [1, 1, 2, 3, 3]: df['A'].append(v)
    r = df.mode()
    check('df mode exists', 'A' in r)

def test_df_describe():
    df = _make_numeric_df()
    r = df.describe()
    check('df describe has A', 'A' in r)
    check('df describe has B', 'B' in r)
    check('df describe A count', r['A']['count'] == 4)
    check('df describe B count', r['B']['count'] == 4)

def test_df_mixed_types():
    df = FastDataFrame()
    df['A'] = IntColumn(3)
    df['B'] = DoubleColumn(3)
    df['C'] = IntColumn(3)
    for i in range(3):
        df['A'].append(i + 1)
        df['B'].append(float(i + 1))
        df['C'].append(i + 10)
    r = df.mean()
    check('df mixed mean has A', 'A' in r)
    check('df mixed mean has B', 'B' in r)
    check('df mixed mean has C', 'C' in r)

def run_all():
    print('=== IntColumn New Stats ===')
    test_int_count()
    test_int_kurtosis()
    test_int_cumprod()
    test_int_cumprod_empty()
    test_int_cumprod_single()
    test_int_cumprod_negative()
    test_int_describe()

    print('=== DoubleColumn New Stats ===')
    test_double_count()
    test_double_kurtosis()
    test_double_cumprod()
    test_double_cumprod_fractional()
    test_double_cumprod_empty()
    test_double_describe()

    print('=== DataFrame Stats ===')
    test_df_count()
    test_df_mean()
    test_df_median()
    test_df_std()
    test_df_var()
    test_df_max()
    test_df_min()
    test_df_sum()
    test_df_quantile()
    test_df_skew()
    test_df_kurt()
    test_df_kurtosis()
    test_df_cumsum()
    test_df_cumprod()
    test_df_cummax()
    test_df_cummin()
    test_df_mode()
    test_df_describe()
    test_df_mixed_types()

    print(f'\nResults: {passed} passed, {failed} failed')
    if failed: sys.exit(1)

if __name__ == '__main__':
    run_all()
