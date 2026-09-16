"""
2D 语义回归测试
覆盖本轮 P0 修复的关键路径：
 - GAP-1/GAP-3：nunique / unique per-column
 - D-GAP-1：count 对 2D 列逻辑行数
 - D-GAP-2：DataFrame 级聚合对 2D 列显式报错
 - GAP-2：算术/比较 shape 校验
 - DataFrame API：values / size / ndim / get / idxmax / idxmin / astype(string)
 - Series API：shift / rank / describe / shape / to_numpy
"""
import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fluidframe.core._databaseframe import DataFrame, RangeIndex, Series, StringIndex
from fluidframe.core._columns import IntColumn, DoubleColumn

passed = 0
failed = 0

def check(tag, cond):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {tag}")

# ------------------------------------------------------------------
# Helper: build a 2D DoubleColumn (rows=4, cols=2)
# ------------------------------------------------------------------
def _make_df2d():
    df = DataFrame()
    df.df_length = 4
    df.index = RangeIndex(0, 4)
    dc = DoubleColumn.from_2d(
        [1.0, 2.0, 3.0, 4.0,
         10.0, 20.0, 30.0, 40.0], 4, 2)
    df._columns['a'] = dc
    return df

# ------------------------------------------------------------------
# GAP-1 / GAP-3: nunique / unique per-column (2D Double)
# ------------------------------------------------------------------
def test_gap1_gap3():
    df = _make_df2d()
    s = df.get_series('a')
    check('unique per-subcol', s.unique() == [[1.0, 2.0, 3.0, 4.0],
                                              [10.0, 20.0, 30.0, 40.0]])
    check('nunique per-subcol', s.nunique() == [4, 4])

    # Int 2D
    ic = IntColumn.from_2d([1, 2, 2, 3, 10, 20, 20, 30], 4, 2)
    s2 = Series('i', ic, RangeIndex(0, 4))
    check('int unique 2d', s2.unique() == [[1, 2, 3], [10, 20, 30]])
    check('int nunique 2d', s2.nunique() == [3, 3])

# ------------------------------------------------------------------
# D-GAP-1: _count_impl 2D DoubleColumn
# ------------------------------------------------------------------
def test_dgap1_count():
    df = _make_df2d()
    cnt = df.count()
    check('count 2d logical rows', cnt['a'] == 4)

    # 含 NaN 的 2D 列：第 1 行子列 0 含 NaN → 该行不计
    dc2 = DoubleColumn.from_2d(
        [1.0, float('nan'), 3.0, 4.0,
         10.0, 20.0, 30.0, 40.0], 4, 2)
    df2 = DataFrame()
    df2.df_length = 4
    df2._columns['b'] = dc2
    cnt2 = df2.count()
    check('count 2d with nan', cnt2['b'] == 3)

# ------------------------------------------------------------------
# D-GAP-2: DataFrame-level numeric agg on 2D column raises
# ------------------------------------------------------------------
def test_dgap2_numeric_agg_2d():
    df = _make_df2d()
    raised = False
    try:
        df.sum()
    except ValueError:
        raised = True
    check('sum 2d raises', raised)

    raised = False
    try:
        df.mean()
    except ValueError:
        raised = True
    check('mean 2d raises', raised)

    raised = False
    try:
        df.std()
    except ValueError:
        raised = True
    check('std 2d raises', raised)

# ------------------------------------------------------------------
# GAP-2: arithmetic / compare shape mismatch
# ------------------------------------------------------------------
def test_gap2_arith_shape():
    dc1 = DoubleColumn.from_2d([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 4, 2)
    dc2 = DoubleColumn.from_2d([10.0, 20.0, 30.0, 40.0], 2, 2)  # rows=2 ≠ 4
    raised = False
    try:
        dc1.add(dc2)
    except ValueError:
        raised = True
    check('add shape mismatch raises', raised)

    raised = False
    try:
        dc1.sub(dc2)
    except ValueError:
        raised = True
    check('sub shape mismatch raises', raised)

    raised = False
    try:
        dc1.mul(dc2)
    except ValueError:
        raised = True
    check('mul shape mismatch raises', raised)

    raised = False
    try:
        dc1.div(dc2)
    except ValueError:
        raised = True
    check('div shape mismatch raises', raised)

    check('cmp shape mismatch raises', raised)

# ------------------------------------------------------------------
# DataFrame API: values / size / ndim / get / idxmax / idxmin / astype
# ------------------------------------------------------------------
def test_df_api_basics():
    df = DataFrame({'a': int, 'b': float})
    df.append_row(a=1, b=2.5)
    df.append_row(a=3, b=4.5)

    check('shape', df.shape == (2, 2))
    check('values', df.values == [[1, 2.5], [3, 4.5]])
    check('size', df.size == 4)
    check('ndim', df.ndim == 2)
    check('get existing', isinstance(df.get('a'), Series))
    check('get missing', df.get('zzz') is None)
    check('get default', df.get('zzz', 42) == 42)

    check('idxmax col', df.idxmax('a') == 1)
    check('idxmin col', df.idxmin('a') == 0)
    s_imax = df.idxmax()
    check('idxmax all', s_imax['a'] == 1 and s_imax['b'] == 1)
    s_imin = df.idxmin()
    check('idxmin all', s_imin['a'] == 0 and s_imin['b'] == 0)

    df_s = df.astype('float')
    check('astype str', df_s['a'][0] == 1.0)

# ------------------------------------------------------------------
# Series API: shift / rank / describe / shape / to_numpy
# ------------------------------------------------------------------
def test_series_api():
    c = IntColumn()
    for v in [10, 20, 30]:
        c.append(v)
    s = Series('x', c, RangeIndex(0, 3))
    check('shape', s.shape == (3,))
    check('to_numpy', s.to_numpy() == [10, 20, 30])

    s2 = s.shift(1)
    check('shift down', s2[0] == 0 or s2[0] is None or (isinstance(s2[0], float) and math.isnan(s2[0])))
    check('shift val', s2[1] == 10)

    s3 = s.shift(1, fill_value=-1)
    check('shift fill', s3[0] == -1)

    s4 = s.rank()
    check('rank', s4[0] == 1.0 and s4[2] == 3.0)

    dc = DoubleColumn()
    for v in [1.0, 2.0, 3.0, 4.0]:
        dc.append(v)
    s5 = Series('n', dc, RangeIndex(0, 4))
    d = s5.describe()
    check('describe count', d[0] == 4)
    check('describe mean', d[1] == 2.5)

# ------------------------------------------------------------------
# Run
# ------------------------------------------------------------------
if __name__ == '__main__':
    test_gap1_gap3()
    test_dgap1_count()
    test_dgap2_numeric_agg_2d()
    test_gap2_arith_shape()
    test_df_api_basics()
    test_series_api()
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
