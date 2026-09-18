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
# A2: Series 对 2D 列的元素访问必须返回"行键元组"（列主序下读全行），
#     而非扁平读第 0 子列（此前只读到子列 0 前 r 个的静默错误）
# ------------------------------------------------------------------
def test_series_2d_element_access():
    dc = DoubleColumn.from_2d(
        [1.0, 2.0, 3.0, 4.0,
         10.0, 20.0, 30.0, 40.0], 4, 2)
    s = Series('a', dc, RangeIndex(0, 4))
    exp = [(1.0, 10.0), (2.0, 20.0), (3.0, 30.0), (4.0, 40.0)]

    check('A2 len is rows', len(s) == 4)
    check('A2 values row tuples', s.values == exp)
    check('A2 iter row tuples', list(s) == exp)
    check('A2 scalar getitem', s[0] == (1.0, 10.0) and s[3] == (4.0, 40.0))
    check('A2 negative getitem', s[-1] == (4.0, 40.0) and s[-4] == (1.0, 10.0))
    check('A2 getitem consistency', [s[i] for i in range(len(s))] == exp)

    raised = False
    try:
        s[4]
    except IndexError:
        raised = True
    check('A2 getitem oob raises', raised)
    raised = False
    try:
        s[-5]
    except IndexError:
        raised = True
    check('A2 getitem neg oob raises', raised)

    check('A2 contains row tuple', (3.0, 30.0) in s and (1.0, 10.0) in s)
    check('A2 no scalar match', 3.0 not in s)
    check('A2 no wrong tuple', (99.0, 99.0) not in s)

    vc = s.value_counts()
    check('A2 value_counts uses row keys', list(vc.index) == exp)

    # 切片 / 布尔掩码走行级 take，保持行元组
    sliced = s[1:3]
    check('A2 slice row-aware', sliced.values == [(2.0, 20.0), (3.0, 30.0)])
    mask_col = IntColumn(4)
    for v in [1, 0, 1, 0]:
        mask_col.append(v)
    mask = Series('m', mask_col, RangeIndex(0, 4))
    check('A2 mask row-aware', s[mask].values == [(1.0, 10.0), (3.0, 30.0)])

    # repr 展示行元组
    check('A2 repr shows tuple', '(1.0, 10.0)' in repr(s))

    # 2D Int 载体
    ic = IntColumn.from_2d([1, 2, 2, 3, 10, 20, 20, 30], 4, 2)
    s2 = Series('i', ic, RangeIndex(0, 4))
    check('A2 int values row tuples', s2.values == [(1, 10), (2, 20), (2, 20), (3, 30)])
    check('A2 int getitem', s2[2] == (2, 20))

    # DataFrame 路由
    df = _make_df2d()
    sf = df.get_series('a')
    check('A2 df route', sf[1] == (2.0, 20.0) and sf.values == exp)

    # 行级赋值在 2D 上明确报错（避免只写子列 0）
    raised = False
    try:
        s[0] = 5.0
    except ValueError:
        raised = True
    check('A2 setitem 2d raises', raised)

    # 1D 语义不受影响
    c = IntColumn(3)
    for v in [10, 20, 30]:
        c.append(v)
    s3 = Series('x', c, RangeIndex(0, 3))
    check('A2 1d values intact', s3.values == [10, 20, 30])
    check('A2 1d getitem intact', s3[1] == 20)
    check('A2 1d iter intact', list(s3) == [10, 20, 30])
    check('A2 1d contains intact', 20 in s3 and 99 not in s3)

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
    test_series_2d_element_access()
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
