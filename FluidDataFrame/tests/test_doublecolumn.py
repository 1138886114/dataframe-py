import math
import sys
sys.path.insert(0, '..')
from fluidframe.core.doublecolumn import DoubleColumn
from fluidframe.core.intcolumn import IntColumn
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

def expect_exception(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        return False
    except exc_type:
        return True
    except Exception:
        return False

def test_empty():
    col = DoubleColumn()
    check('empty length', col.length == 0)
    check('empty capacity', col.capacity == 16)

def test_append():
    col = DoubleColumn()
    col.append(1.5); col.append(2.5); col.append(3.5)
    check('append length', col.length == 3)
    check('getitem 0', col[0] == 1.5)
    check('getitem 1', col[1] == 2.5)
    check('getitem 2', col[2] == 3.5)

def test_special_values():
    col = DoubleColumn(); col.append(float('nan'))
    check('append nan', math.isnan(col[0]))

    col = DoubleColumn(); col.append(-0.0)
    check('append neg zero', col[0] == -0.0)

    col = DoubleColumn(); col.append(1e308); col.append(-1e308)
    check('append large', col[0] == 1e308)
    check('append small', col[1] == -1e308)

    col = DoubleColumn(); col.append(0.0)
    check('append zero', col[0] == 0.0)

def test_bounds():
    col = DoubleColumn(); col.append(1.0)
    check('getitem out of range', expect_exception(IndexError, col.__getitem__, 1))

def test_pop_back():
    col = DoubleColumn(); col.append(1.0); col.append(2.0)
    val = col.pop_back()
    check('pop_back val', val == 2.0)
    check('pop_back len', col.length == 1)

    col = DoubleColumn()
    check('pop_back empty', expect_exception(IndexError, col.pop_back))

def test_pop_front():
    col = DoubleColumn(); col.append(1.0); col.append(2.0); col.append(3.0)
    val = col.pop_front()
    check('pop_front val', val == 1.0)
    check('pop_front len', col.length == 2)
    check('pop_front [0]', col[0] == 2.0)

    col = DoubleColumn()
    check('pop_front empty', expect_exception(IndexError, col.pop_front))

def test_copy():
    col = DoubleColumn(); col.append(1.5); col.append(2.5)
    col2 = col.copy()
    check('copy length', col2.length == 2)
    check('copy [0]', col2[0] == 1.5)
    col2.append(3.5)
    check('copy independent', col.length == 2)

def test_concat():
    col1 = DoubleColumn(); col1.append(1.0)
    col2 = DoubleColumn(); col2.append(2.0)
    col1.concat(col2)
    check('concat length', col1.length == 2)
    check('concat [0]', col1[0] == 1.0)
    check('concat [1]', col1[1] == 2.0)

def test_max():
    col = DoubleColumn(); col.append(1.5); col.append(3.5); col.append(2.5)
    check('max', col.max() == 3.5)

    col = DoubleColumn()
    check('max empty', expect_exception(ValueError, col.max))

def test_min():
    col = DoubleColumn(); col.append(1.5); col.append(3.5); col.append(2.5)
    check('min', col.min() == 1.5)

    col = DoubleColumn()
    check('min empty', expect_exception(ValueError, col.min))

def test_mean():
    col = DoubleColumn(); col.append(1.0); col.append(2.0); col.append(3.0)
    check('mean', col.mean() == 2.0)

    col = DoubleColumn()
    check('mean empty', math.isnan(col.mean()))

def test_var():
    col = DoubleColumn(); col.append(1.0); col.append(2.0); col.append(3.0)
    check('var', col.var() == 1.0)

def test_std():
    col = DoubleColumn(); col.append(1.0); col.append(2.0); col.append(3.0)
    check('std', col.std() == 1.0)

def test_sum():
    col = DoubleColumn(); col.append(1.5); col.append(2.5)
    check('sum', col.sum() == 4.0)

def test_count_nonzeros():
    col = DoubleColumn(); col.append(0.0); col.append(1.0); col.append(0.0); col.append(5.0)
    check('count_nonzeros', col.count_nonzeros() == 2)

def test_idxmax():
    col = DoubleColumn(); col.append(1.5); col.append(5.5); col.append(3.5)
    check('idxmax', col.idxmax() == 1)

def test_idxmin():
    col = DoubleColumn(); col.append(5.5); col.append(1.5); col.append(3.5)
    check('idxmin', col.idxmin() == 1)

def test_median():
    col = DoubleColumn(); col.append(3.0); col.append(1.0); col.append(2.0)
    check('median odd', col.median() == 2.0)

    col = DoubleColumn(); col.append(4.0); col.append(1.0); col.append(3.0); col.append(2.0)
    check('median even', col.median() == 2.5)

    col = DoubleColumn()
    check('median empty', math.isnan(col.median()))

def test_mad():
    col = DoubleColumn(); col.append(1.0); col.append(2.0); col.append(3.0)
    check('mad', col.mad() == 2.0 / 3.0)

def test_add():
    col = DoubleColumn(); col.append(1.5)
    res = col.add_val(2.5)
    check('add', res[0] == 4.0)

def test_sub():
    col = DoubleColumn(); col.append(5.5)
    res = col.sub_val(2.5)
    check('sub', res[0] == 3.0)

def test_mul():
    col = DoubleColumn(); col.append(3.0)
    res = col.mul_val(2.0)
    check('mul', res[0] == 6.0)

def test_div():
    col = DoubleColumn(); col.append(10.0)
    res = col.div_val(2.0)
    check('div', res[0] == 5.0)

    col = DoubleColumn(); col.append(10.0)
    res = col.div_val(0.0)
    check('div by zero', math.isnan(res[0]))

def test_unique():
    col = DoubleColumn(); col.append(1.0); col.append(2.0); col.append(1.0)
    check('unique', col.unique() == [1.0, 2.0])

    col = DoubleColumn()
    check('unique empty', col.unique() == [])

def test_growth():
    col = DoubleColumn()
    for i in range(100):
        col.append(float(i))
    check('growth length', col.length == 100)
    check('growth [99]', col[99] == 99.0)

def test_comparison():
    col = DoubleColumn()
    col.append(1.0); col.append(3.0); col.append(5.0)
    m = col < 4.0
    check('lt', [m[i] for i in range(m.length)] == [1, 1, 0])
    m = col >= 3.0
    check('ge', [m[i] for i in range(m.length)] == [0, 1, 1])

def test_bitwise():
    col = DoubleColumn()
    col.append(1.0); col.append(3.0); col.append(5.0)
    m1 = col < 4.0
    m2 = col > 2.0
    m_and = m1 & m2
    check('and', [m_and[i] for i in range(m_and.length)] == [0, 1, 0])

def test_dunder_add():
    a = DoubleColumn(); a.append(1.0); a.append(2.0); a.append(3.0)
    b = DoubleColumn(); b.append(10.0); b.append(20.0); b.append(30.0)
    r = a + b
    check('__add__ col-col', r[0] == 11.0 and r[1] == 22.0 and r[2] == 33.0)
    r = a + 5.0
    check('__add__ scalar', r[0] == 6.0 and r[1] == 7.0 and r[2] == 8.0)
    r = 100.0 + a
    check('__radd__ scalar', r[0] == 101.0 and r[1] == 102.0 and r[2] == 103.0)
    a_d = DoubleColumn(); a_d.append(1.5); a_d.append(2.5)
    b_i = IntColumn(); b_i.append(10); b_i.append(20)
    r = a_d + b_i
    check('__add__ double+int', r[0] == 11.5 and r[1] == 22.5)

def test_dunder_sub():
    a = DoubleColumn(); a.append(10.0); a.append(20.0); a.append(30.0)
    b = DoubleColumn(); b.append(1.0); b.append(2.0); b.append(3.0)
    r = a - b
    check('__sub__ col-col', r[0] == 9.0 and r[1] == 18.0 and r[2] == 27.0)
    r = a - 5.0
    check('__sub__ scalar', r[0] == 5.0 and r[1] == 15.0 and r[2] == 25.0)
    r = 100.0 - a
    check('__rsub__ scalar', r[0] == 90.0 and r[1] == 80.0 and r[2] == 70.0)

def test_dunder_mul():
    a = DoubleColumn(); a.append(2.0); a.append(3.0); a.append(4.0)
    b = DoubleColumn(); b.append(10.0); b.append(20.0); b.append(30.0)
    r = a * b
    check('__mul__ col-col', r[0] == 20.0 and r[1] == 60.0 and r[2] == 120.0)
    r = a * 3.0
    check('__mul__ scalar', r[0] == 6.0 and r[1] == 9.0 and r[2] == 12.0)
    r = 3.0 * a
    check('__rmul__ scalar', r[0] == 6.0 and r[1] == 9.0 and r[2] == 12.0)

def test_dunder_div():
    a = DoubleColumn(); a.append(10.0); a.append(20.0); a.append(30.0)
    b = DoubleColumn(); b.append(2.0); b.append(4.0); b.append(5.0)
    r = a / b
    check('__truediv__ col-col', r[0] == 5.0 and r[1] == 5.0 and r[2] == 6.0)
    r = a / 3.0
    check('__truediv__ scalar', math.isclose(r[0], 10.0/3) and math.isclose(r[1], 20.0/3) and r[2] == 10.0)
    r = 60.0 / a
    check('__rtruediv__ scalar', r[0] == 6.0 and r[1] == 3.0 and r[2] == 2.0)
    r = a / 0.0
    check('__truediv__ div by zero', math.isnan(r[0]))
    r = 0.0 / a
    check('__rtruediv__ zero numerator', r[0] == 0.0)

def test_dunder_notimplemented():
    a = DoubleColumn(); a.append(1.0)
    check('__add__ NotImplemented', type(a).__add__(a, "x") is NotImplemented)
    check('__sub__ NotImplemented', type(a).__sub__(a, "x") is NotImplemented)
    check('__mul__ NotImplemented', type(a).__mul__(a, "x") is NotImplemented)
    check('__truediv__ NotImplemented', type(a).__truediv__(a, "x") is NotImplemented)
    check('__rsub__ NotImplemented', type(a).__rsub__(a, "x") is NotImplemented)
    check('__rtruediv__ NotImplemented', type(a).__rtruediv__(a, "x") is NotImplemented)

# ============================================================
#  Roll 系列函数测试 (1D)
# ============================================================
def _make_1d(vals):
    col = DoubleColumn()
    for v in vals:
        col.append(float(v))
    return col

def _approx(a, b, tol=1e-9):
    return abs(a - b) < tol

def _is_nan(v):
    return v != v

def test_roll_mean_1d():
    col = _make_1d([1, 2, 3, 4, 5, 6, 7, 8])
    r = col.roll_mean(3)
    check('roll_mean_1d length', r.length == 8)
    check('roll_mean_1d nan0', _is_nan(r[0]))
    check('roll_mean_1d nan1', _is_nan(r[1]))
    check('roll_mean_1d [2]', _approx(r[2], 2.0))
    check('roll_mean_1d [3]', _approx(r[3], 3.0))
    check('roll_mean_1d [7]', _approx(r[7], 7.0))

    r1 = col.roll_mean(1)
    check('roll_mean_1d w=1 [0]', _approx(r1[0], 1.0))
    check('roll_mean_1d w=1 [7]', _approx(r1[7], 8.0))

    r8 = col.roll_mean(8)
    check('roll_mean_1d w=len nan', all(_is_nan(r8[i]) for i in range(7)))
    check('roll_mean_1d w=len [7]', _approx(r8[7], 4.5))

def test_roll_sum_1d():
    col = _make_1d([1, 2, 3, 4, 5, 6, 7, 8])
    r = col.roll_sum(3)
    check('roll_sum_1d [2]', _approx(r[2], 6.0))
    check('roll_sum_1d [3]', _approx(r[3], 9.0))
    check('roll_sum_1d [7]', _approx(r[7], 21.0))

    r1 = col.roll_sum(1)
    check('roll_sum_1d w=1 [3]', _approx(r1[3], 4.0))

def test_roll_min_1d():
    col = _make_1d([3, 1, 4, 1, 5, 9, 2, 6])
    r = col.roll_min(3)
    check('roll_min_1d [2]', _approx(r[2], 1.0))
    check('roll_min_1d [3]', _approx(r[3], 1.0))
    check('roll_min_1d [4]', _approx(r[4], 1.0))
    check('roll_min_1d [5]', _approx(r[5], 1.0))
    check('roll_min_1d [7]', _approx(r[7], 2.0))

def test_roll_max_1d():
    col = _make_1d([3, 1, 4, 1, 5, 9, 2, 6])
    r = col.roll_max(3)
    check('roll_max_1d [2]', _approx(r[2], 4.0))
    check('roll_max_1d [3]', _approx(r[3], 4.0))
    check('roll_max_1d [4]', _approx(r[4], 5.0))
    check('roll_max_1d [5]', _approx(r[5], 9.0))
    check('roll_max_1d [7]', _approx(r[7], 9.0))

def test_roll_var_1d():
    col = _make_1d([1, 2, 3, 4, 5, 6, 7, 8])
    r = col.roll_var(4)
    # window [1,2,3,4]: sample var = 5/3 ≈ 1.6667
    check('roll_var_1d ddof=True [3]', _approx(r[3], 5.0/3.0, 1e-6))
    # window [5,6,7,8]: same variance
    check('roll_var_1d ddof=True [7]', _approx(r[7], 5.0/3.0, 1e-6))

    rp = col.roll_var(4, ddof=False)
    check('roll_var_1d ddof=False [3]', _approx(rp[3], 5.0/4.0, 1e-6))

def test_roll_std_1d():
    col = _make_1d([1, 2, 3, 4, 5, 6, 7, 8])
    r = col.roll_std(4)
    check('roll_std_1d [3]', _approx(r[3], math.sqrt(5.0/3.0), 1e-6))
    check('roll_std_1d [7]', _approx(r[7], math.sqrt(5.0/3.0), 1e-6))

def test_roll_median_1d():
    col = _make_1d([3, 1, 4, 1, 5, 9, 2, 6])
    r = col.roll_median(3)
    check('roll_median_1d [2]', _approx(r[2], 3.0))  # sorted [1,3,4]
    check('roll_median_1d [3]', _approx(r[3], 1.0))  # sorted [1,1,4]
    check('roll_median_1d [5]', _approx(r[5], 5.0))  # sorted [1,5,9]
    check('roll_median_1d [7]', _approx(r[7], 6.0))  # sorted [2,6,6] -> 6

def test_roll_quantile_1d():
    col = _make_1d([1, 2, 3, 4, 5, 6, 7, 8])
    r = col.roll_quantile(4, 0.25)
    check('roll_quantile_1d q=0.25 [3]', _approx(r[3], 1.75, 1e-6))

    r = col.roll_quantile(4, 0.5)
    check('roll_quantile_1d q=0.5 [3]', _approx(r[3], 2.5, 1e-6))

    r = col.roll_quantile(4, 0.75)
    check('roll_quantile_1d q=0.75 [3]', _approx(r[3], 3.25, 1e-6))

def test_roll_apply_weights_1d():
    col = _make_1d([1, 2, 3, 4, 5])
    # weights = [1, 1, 1] -> same as roll_sum(3)
    r = col.roll_apply_weights([1.0, 1.0, 1.0])
    check('roll_weights_1d equal', _approx(r[2], 6.0))
    check('roll_weights_1d equal [4]', _approx(r[4], 12.0))

    # weights = [1, 2, 3] -> 1*x[i-2] + 2*x[i-1] + 3*x[i]
    r = col.roll_apply_weights([1.0, 2.0, 3.0])
    # i=2: 1*1 + 2*2 + 3*3 = 1+4+9=14
    check('roll_weights_1d weighted [2]', _approx(r[2], 14.0))
    # i=4: 1*3 + 2*4 + 3*5 = 3+8+15=26
    check('roll_weights_1d weighted [4]', _approx(r[4], 26.0))

def test_roll_custom_1d():
    col = _make_1d([1, 2, 3, 4, 5])
    # custom func: sum of window
    r = col.roll_custom(3, lambda w: w.sum())
    check('roll_custom_1d sum [2]', _approx(r[2], 6.0))
    check('roll_custom_1d sum [4]', _approx(r[4], 12.0))

    # custom func: max - min (range)
    r = col.roll_custom(3, lambda w: w.max() - w.min())
    check('roll_custom_1d range [2]', _approx(r[2], 2.0))
    check('roll_custom_1d range [3]', _approx(r[3], 2.0))

def test_roll_dispatch_1d():
    col = _make_1d([1, 2, 3, 4, 5, 6, 7, 8])
    r = col.roll(3, 'mean')
    check('roll_dispatch_1d mean', _approx(r[2], 2.0))

    r = col.roll(3, 'sum')
    check('roll_dispatch_1d sum', _approx(r[2], 6.0))

    r = col.roll(3, 'min')
    check('roll_dispatch_1d min', _approx(r[2], 1.0))

    r = col.roll(3, 'max')
    check('roll_dispatch_1d max', _approx(r[2], 3.0))

    r = col.roll(4, 'std', ddof=True)
    check('roll_dispatch_1d std', _approx(r[3], math.sqrt(5.0/3.0), 1e-6))

    r = col.roll(4, 'var', ddof=False)
    check('roll_dispatch_1d var', _approx(r[3], 5.0/4.0, 1e-6))

    r = col.roll(3, 'median')
    check('roll_dispatch_1d median', _approx(r[2], 2.0))

    r = col.roll(4, 'quantile', q=0.5)
    check('roll_dispatch_1d quantile', _approx(r[3], 2.5, 1e-6))

def test_roll_error_1d():
    col = _make_1d([1, 2, 3])
    check('roll_error window<1', expect_exception(ValueError, col.roll_mean, 0))
    check('roll_error var w<2', expect_exception(ValueError, col.roll_var, 1))
    check('roll_error bad method', expect_exception(ValueError, col.roll, 2, 'invalid'))

# ============================================================
#  Roll 系列函数测试 (2D 列优先)
# ============================================================
def _make_2d(rows, cols_vals):
    """创建 2D 列优先 DoubleColumn。cols_vals = [[col0 vals], [col1 vals], ...]"""
    c = len(cols_vals)
    total = rows * c
    col = DoubleColumn(max(total, 1), cols=c)
    for col_idx, vals in enumerate(cols_vals):
        for row_idx, v in enumerate(vals):
            col.append(float(v))
    return col

def test_roll_mean_2d():
    # 2D: 4 rows x 2 cols, col-major
    # col0: [1, 2, 3, 4], col1: [10, 20, 30, 40]
    col = _make_2d(4, [[1, 2, 3, 4], [10, 20, 30, 40]])
    check('roll_mean_2d rows', col.rows == 4)
    check('roll_mean_2d cols', col.cols == 2)

    r = col.roll_mean(3)
    # col0: mean([1,2,3])=2.0, mean([2,3,4])=3.0
    check('roll_mean_2d c0 [2]', _approx(r[2], 2.0))
    check('roll_mean_2d c0 [3]', _approx(r[3], 3.0))
    # col1: mean([10,20,30])=20.0, mean([20,30,40])=30.0
    check('roll_mean_2d c1 [6]', _approx(r[6], 20.0))
    check('roll_mean_2d c1 [7]', _approx(r[7], 30.0))

def test_roll_sum_2d():
    col = _make_2d(4, [[1, 2, 3, 4], [10, 20, 30, 40]])
    r = col.roll_sum(3)
    check('roll_sum_2d c0 [2]', _approx(r[2], 6.0))
    check('roll_sum_2d c1 [6]', _approx(r[6], 60.0))

def test_roll_min_2d():
    col = _make_2d(5, [[5, 3, 1, 4, 2], [50, 30, 10, 40, 20]])
    r = col.roll_min(3)
    # col0: min([5,3,1])=1, min([3,1,4])=1, min([1,4,2])=1
    check('roll_min_2d c0 [2]', _approx(r[2], 1.0))
    check('roll_min_2d c0 [3]', _approx(r[3], 1.0))
    check('roll_min_2d c0 [4]', _approx(r[4], 1.0))
    # col1: min([50,30,10])=10, min([30,10,40])=10, min([10,40,20])=10
    check('roll_min_2d c1 [7]', _approx(r[7], 10.0))

def test_roll_max_2d():
    col = _make_2d(5, [[5, 3, 1, 4, 2], [50, 30, 10, 40, 20]])
    r = col.roll_max(3)
    # col0: max([5,3,1])=5, max([3,1,4])=4, max([1,4,2])=4
    check('roll_max_2d c0 [2]', _approx(r[2], 5.0))
    check('roll_max_2d c0 [3]', _approx(r[3], 4.0))
    check('roll_max_2d c0 [4]', _approx(r[4], 4.0))
    # col1: max([50,30,10])=50, max([30,10,40])=40, max([10,40,20])=40
    check('roll_max_2d c1 [7]', _approx(r[7], 50.0))

def test_roll_var_2d():
    col = _make_2d(5, [[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]])
    r = col.roll_var(3)
    # col0: var([1,2,3]) = 1.0, var([2,3,4]) = 1.0
    check('roll_var_2d c0 [2]', _approx(r[2], 1.0, 1e-6))
    check('roll_var_2d c0 [3]', _approx(r[3], 1.0, 1e-6))
    # col1: var([10,20,30]) = 100.0
    check('roll_var_2d c1 [7]', _approx(r[7], 100.0, 1e-6))

def test_roll_std_2d():
    col = _make_2d(5, [[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]])
    r = col.roll_std(3)
    check('roll_std_2d c0 [2]', _approx(r[2], 1.0, 1e-6))
    check('roll_std_2d c1 [7]', _approx(r[7], 10.0, 1e-6))

def test_roll_median_2d():
    col = _make_2d(5, [[3, 1, 4, 1, 5], [30, 10, 40, 10, 50]])
    r = col.roll_median(3)
    # col0: sorted [1,3,4]->3, [1,1,4]->1, [1,4,5]->4
    check('roll_median_2d c0 [2]', _approx(r[2], 3.0))
    check('roll_median_2d c0 [3]', _approx(r[3], 1.0))
    check('roll_median_2d c0 [4]', _approx(r[4], 4.0))
    # col1: sorted [10,30,40]->30, [10,10,40]->10, [10,40,50]->40
    check('roll_median_2d c1 [7]', _approx(r[7], 30.0))

def test_roll_quantile_2d():
    col = _make_2d(5, [[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]])
    r = col.roll_quantile(3, 0.5)
    check('roll_quantile_2d c0 [2]', _approx(r[2], 2.0))
    check('roll_quantile_2d c1 [7]', _approx(r[7], 20.0))

def test_roll_apply_weights_2d():
    col = _make_2d(4, [[1, 2, 3, 4], [10, 20, 30, 40]])
    r = col.roll_apply_weights([1.0, 1.0, 1.0])
    # col0: 1+2+3=6, 2+3+4=9
    check('roll_weights_2d c0 [2]', _approx(r[2], 6.0))
    check('roll_weights_2d c0 [3]', _approx(r[3], 9.0))
    # col1: 10+20+30=60, 20+30+40=90
    check('roll_weights_2d c1 [6]', _approx(r[6], 60.0))
    check('roll_weights_2d c1 [7]', _approx(r[7], 90.0))

def test_roll_custom_2d():
    col = _make_2d(4, [[1, 2, 3, 4], [10, 20, 30, 40]])
    r = col.roll_custom(3, lambda w: w.sum())
    check('roll_custom_2d c0 [2]', _approx(r[2], 6.0))
    check('roll_custom_2d c1 [6]', _approx(r[6], 60.0))

def test_roll_dispatch_2d():
    col = _make_2d(4, [[1, 2, 3, 4], [10, 20, 30, 40]])
    r = col.roll(3, 'mean')
    check('roll_dispatch_2d mean c0', _approx(r[2], 2.0))
    check('roll_dispatch_2d mean c1', _approx(r[6], 20.0))

    r = col.roll(3, 'sum')
    check('roll_dispatch_2d sum c0', _approx(r[2], 6.0))
    check('roll_dispatch_2d sum c1', _approx(r[6], 60.0))

    r = col.roll(3, 'min')
    check('roll_dispatch_2d min c0', _approx(r[2], 1.0))
    check('roll_dispatch_2d min c1', _approx(r[6], 10.0))

    r = col.roll(3, 'max')
    check('roll_dispatch_2d max c0', _approx(r[2], 3.0))
    check('roll_dispatch_2d max c1', _approx(r[6], 30.0))

def test_roll_nan_propagation_1d():
    """测试 NaN 在窗口中的传播行为 (O(1) 滑窗累加器一旦遇到 NaN 无法恢复)"""
    col = _make_1d([1, float('nan'), 3, 4, 5])
    r = col.roll_mean(3)
    # 窗口 [1, nan, 3] -> NaN
    check('roll_nan_prop [2]', _is_nan(r[2]))
    # 窗口 [nan, 3, 4] -> NaN
    check('roll_nan_prop [3]', _is_nan(r[3]))
    # 窗口 [3, 4, 5]: 虽然本窗口无 NaN，但 O(1) 累加器已被污染，仍为 NaN
    check('roll_nan_prop [4]', _is_nan(r[4]))

    r = col.roll_sum(3)
    check('roll_nan_sum [2]', _is_nan(r[2]))
    check('roll_nan_sum [4]', _is_nan(r[4]))

def test_roll_nan_propagation_2d():
    """测试 2D 中 NaN 传播——各列独立"""
    col = _make_2d(4, [[1, float('nan'), 3, 4], [10, 20, 30, 40]])
    r = col.roll_mean(3)
    # col0 含 NaN
    check('roll_nan_2d c0 [2]', _is_nan(r[2]))
    check('roll_nan_2d c0 [3]', _is_nan(r[3]))
    # col1 正常
    check('roll_nan_2d c1 [6]', _approx(r[6], 20.0))

def test_roll_window_larger_than_data_1d():
    """窗口大于数据长度，全部返回 NaN"""
    col = _make_1d([1, 2, 3])
    r = col.roll_mean(5)
    check('roll_win>len_1d', all(_is_nan(r[i]) for i in range(3)))

def test_roll_window_larger_than_data_2d():
    """2D: 窗口大于行数，全部返回 NaN"""
    col = _make_2d(3, [[1, 2, 3], [10, 20, 30]])
    r = col.roll_mean(5)
    check('roll_win>len_2d', all(_is_nan(r[i]) for i in range(6)))

def test_roll_single_element_window_2d():
    """2D: window=1 等于自身"""
    col = _make_2d(4, [[1, 2, 3, 4], [10, 20, 30, 40]])
    r = col.roll_mean(1)
    check('roll_w1_2d c0', all(_approx(r[i], col[i]) for i in range(4)))
    check('roll_w1_2d c1', all(_approx(r[4+i], col[4+i]) for i in range(4)))

# ============================================================
#  构造方法测试: load_from_1d_array / load_from_2d_array / create_view
# ============================================================
def test_load_from_1d_array():
    import array
    arr = array.array('d', [1.0, 2.0, 3.0, 4.0, 5.0])
    col = DoubleColumn.load_from_1d_array(arr)
    check('load_1d length', col.length == 5)
    check('load_1d [0]', col[0] == 1.0)
    check('load_1d [4]', col[4] == 5.0)
    check('load_1d owns_data', col._owns_data == False)
    check('load_1d cols', col.cols == 1)
    check('load_1d is_2d', col.is_2d == False)

    # 修改原数组应反映到列 (零拷贝)
    arr[2] = 99.0
    check('load_1d zero_copy', col[2] == 99.0)
    arr[2] = 3.0  # 恢复

    # roll 应在零拷贝视图上正常工作
    r = col.roll_mean(3)
    check('load_1d roll_mean', _approx(r[2], 2.0))

def test_load_from_2d_array():
    try:
        import numpy as np
    except ImportError:
        print('  SKIP: numpy not available')
        return
    # 4 rows x 2 cols, column-major (Fortran order)
    arr = np.array([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=np.float64, order='F')
    cols_dict = DoubleColumn.load_from_2d_array(arr, ['A', 'B'])
    check('load_2d keys', set(cols_dict.keys()) == {'A', 'B'})

    col_a = cols_dict['A']
    col_b = cols_dict['B']
    check('load_2d A length', col_a.length == 4)
    check('load_2d A [0]', col_a[0] == 1.0)
    check('load_2d A [3]', col_a[3] == 4.0)
    check('load_2d A owns_data', col_a._owns_data == False)
    check('load_2d A cols', col_a.cols == 1)

    check('load_2d B [0]', col_b[0] == 10.0)
    check('load_2d B [3]', col_b[3] == 40.0)

    # 零拷贝: 修改原数组应反映到列
    arr[1, 0] = 99.0
    check('load_2d zero_copy A', col_a[1] == 99.0)
    arr[1, 0] = 2.0  # 恢复

    # roll 在零拷贝视图上工作
    r = col_a.roll_mean(3)
    check('load_2d roll_mean A', _approx(r[2], 2.0))

    r = col_b.roll_sum(2)
    check('load_2d roll_sum B', _approx(r[1], 30.0))

    # 错误: col_names 长度不匹配
    check('load_2d col_mismatch',
          expect_exception(ValueError, DoubleColumn.load_from_2d_array, arr, ['A']))

def test_load_from_2d_array_empty():
    try:
        import numpy as np
    except ImportError:
        return
    arr = np.zeros((0, 0), dtype=np.float64, order='F')
    result = DoubleColumn.load_from_2d_array(arr, [])
    check('load_2d empty', result == {})

def test_create_view_via_roll():
    """通过 load_from_2d_array 间接测试 create_view 在 roll 上的表现"""
    try:
        import numpy as np
    except ImportError:
        return
    arr = np.array([[1, 100], [2, 200], [3, 300], [4, 400], [5, 500]],
                   dtype=np.float64, order='F')
    cols_dict = DoubleColumn.load_from_2d_array(arr, ['X', 'Y'])
    cx = cols_dict['X']
    cy = cols_dict['Y']

    # 每列独立 roll
    rx = cx.roll_mean(3)
    ry = cy.roll_mean(3)
    check('create_view_roll X [2]', _approx(rx[2], 2.0))
    check('create_view_roll Y [2]', _approx(ry[2], 200.0))
    check('create_view_roll X [4]', _approx(rx[4], 4.0))
    check('create_view_roll Y [4]', _approx(ry[4], 400.0))

    # 每列独立 min/max
    rx_min = cx.roll_min(2)
    ry_max = cy.roll_max(2)
    check('create_view_roll min X', _approx(rx_min[2], 2.0))
    check('create_view_roll max Y', _approx(ry_max[3], 400.0))

# ============================================================
#  未覆盖的公共方法测试
# ============================================================
def test_prod():
    col = _make_1d([1.0, 2.0, 3.0])
    check('prod', _approx(col.prod(), 6.0))
    col2 = _make_1d([])
    check('prod empty', _is_nan(col2.prod()))

def test_nunique():
    col = _make_1d([1.0, 2.0, 2.0, 3.0, 3.0, 3.0])
    check('nunique', col.nunique() == 3)

def test_value_counts():
    col = _make_1d([1.0, 2.0, 2.0, 3.0])
    vc = col.value_counts()
    check('value_counts 2.0', vc[2.0] == 2)
    check('value_counts 1.0', vc[1.0] == 1)

def test_mode():
    col = _make_1d([1.0, 2.0, 2.0, 3.0])
    m = col.mode()
    check('mode', 2.0 in m)

def test_cumsum():
    col = _make_1d([1.0, 2.0, 3.0, 4.0])
    r = col.cumsum()
    check('cumsum [0]', r[0] == 1.0)
    check('cumsum [1]', r[1] == 3.0)
    check('cumsum [3]', r[3] == 10.0)

def test_cummax():
    col = _make_1d([3.0, 1.0, 4.0, 1.0, 5.0])
    r = col.cummax()
    check('cummax [0]', r[0] == 3.0)
    check('cummax [1]', r[1] == 3.0)
    check('cummax [2]', r[2] == 4.0)
    check('cummax [4]', r[4] == 5.0)

def test_cummin():
    col = _make_1d([3.0, 1.0, 4.0, 0.0, 5.0])
    r = col.cummin()
    check('cummin [0]', r[0] == 3.0)
    check('cummin [1]', r[1] == 1.0)
    check('cummin [3]', r[3] == 0.0)
    check('cummin [4]', r[4] == 0.0)

def test_cumprod():
    col = _make_1d([1.0, 2.0, 3.0, 4.0])
    r = col.cumprod()
    check('cumprod [0]', r[0] == 1.0)
    check('cumprod [1]', r[1] == 2.0)
    check('cumprod [3]', r[3] == 24.0)

def test_diff():
    col = _make_1d([1.0, 3.0, 6.0, 10.0])
    r = col.diff(1)
    check('diff nan0', _is_nan(r[0]))
    check('diff [1]', _approx(r[1], 2.0))
    check('diff [2]', _approx(r[2], 3.0))
    check('diff [3]', _approx(r[3], 4.0))

    r_neg = col.diff(-1)
    check('diff neg nan3', _is_nan(r_neg[3]))
    check('diff neg [0]', _approx(r_neg[0], -2.0))

def test_pct_change():
    col = _make_1d([10.0, 20.0, 40.0])
    r = col.pct_change(1)
    check('pct_change nan0', _is_nan(r[0]))
    check('pct_change [1]', _approx(r[1], 1.0))  # (20-10)/10
    check('pct_change [2]', _approx(r[2], 1.0))  # (40-20)/20

def test_skew():
    col = _make_1d([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0])
    s = col.skew()
    check('skew positive', s > 0)

def test_kurt():
    col = _make_1d([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0])
    k = col.kurt()
    check('kurt large', k > 0)

def test_sem():
    col = _make_1d([1.0, 2.0, 3.0, 4.0, 5.0])
    s = col.sem()
    check('sem positive', s > 0)
    check('sem finite', math.isfinite(s))

def test_describe():
    col = _make_1d([1.0, 2.0, 3.0, 4.0, 5.0])
    d = col.describe()
    check('describe count', d['count'] == 5)
    check('describe mean', _approx(d['mean'], 3.0))
    check('describe min', d['min'] == 1.0)
    check('describe max', d['max'] == 5.0)

def test_isin():
    col = _make_1d([1.0, 2.0, 3.0, 4.0, 5.0])
    r = col.isin({2.0, 4.0})
    check('isin [0]', r[0] == 0)
    check('isin [1]', r[1] == 1)
    check('isin [2]', r[2] == 0)
    check('isin [3]', r[3] == 1)

def test_between():
    col = _make_1d([1.0, 2.0, 3.0, 4.0, 5.0])
    r = col.between(2.0, 4.0)
    check('between [0]', r[0] == 0)
    check('between [1]', r[1] == 1)
    check('between [2]', r[2] == 1)
    check('between [3]', r[3] == 1)
    check('between [4]', r[4] == 0)

def test_str():
    col = _make_1d([1.0, 2.0, 3.0])
    s = str(col)
    check('str contains 1.0', '1.0' in s)
    check('str contains 3.0', '3.0' in s)

def test_bool():
    check('bool empty', not bool(DoubleColumn()))
    check('bool nonempty', bool(_make_1d([1.0])))

def test_len():
    col = _make_1d([1.0, 2.0, 3.0])
    check('len', len(col) == 3)

def test_clip():
    col = _make_1d([1.0, 5.0, 10.0])
    r = col.clip(3.0, 7.0)
    check('clip [0]', r[0] == 3.0)
    check('clip [1]', r[1] == 5.0)
    check('clip [2]', r[2] == 7.0)

def test_where():
    col = _make_1d([1.0, 2.0, 3.0, 4.0])
    cond = IntColumn()
    for v in [1, 0, 1, 0]:
        cond.append(v)
    r = col.where(cond, 99.0)
    check('where [0]', r[0] == 1.0)
    check('where [1]', r[1] == 99.0)
    check('where [2]', r[2] == 3.0)
    check('where [3]', r[3] == 99.0)

def test_where_mask():
    col = _make_1d([1.0, 2.0, 3.0, 4.0])
    mask = IntColumn()
    for v in [1, 0, 1, 0]:
        mask.append(v)
    r = col.where_mask(mask, 99.0)
    check('where_mask [0]', r[0] == 1.0)
    check('where_mask [1]', r[1] == 99.0)
    check('where_mask [3]', r[3] == 99.0)

def test_mask():
    col = _make_1d([1.0, 2.0, 3.0, 4.0])
    cond = IntColumn()
    for v in [1, 0, 1, 0]:
        cond.append(v)
    r = col.mask(cond, 99.0)
    check('mask [0]', r[0] == 99.0)
    check('mask [1]', r[1] == 2.0)
    check('mask [2]', r[2] == 99.0)
    check('mask [3]', r[3] == 4.0)

def test_drop_row():
    col = _make_1d([1.0, 2.0, 3.0, 4.0])
    col.drop_row(1, 2)
    check('drop_row len', col.length == 3)
    check('drop_row [0]', col[0] == 1.0)
    check('drop_row [1]', col[1] == 3.0)
    check('drop_row [2]', col[2] == 4.0)

def test_distances():
    a = _make_1d([1.0, 2.0, 3.0])
    b = _make_1d([4.0, 5.0, 6.0])
    check('pearson_corr', _approx(a.pearson_corr_with(b), 1.0))
    check('pearson_cov', a.pearson_cov_with(b) > 0)
    check('euclidean', _approx(a.euclidean_dist(b), math.sqrt(27.0)))
    check('manhattan', _approx(a.manhattan_dist(b), 9.0))
    cd = a.cosine_dist(b)
    check('cosine_dist 0<cd<2', 0.0 <= cd <= 2.0)
    cs = a.cosine_similarity(b)
    check('cosine_sim -1<=cs<=1', -1.0 <= cs <= 1.0)

def test_reorder():
    col = _make_1d([10.0, 20.0, 30.0])
    col._reorder([2, 0, 1])
    check('reorder [0]', col[0] == 30.0)
    check('reorder [1]', col[1] == 10.0)
    check('reorder [2]', col[2] == 20.0)

def test_count():
    col = _make_1d([1.0, 2.0, 3.0])
    check('count', col.count() == 3)

# ============================================================
#  df["A"].roll_xxx() 模式测试 (通过 FastDataFrame)
# ============================================================
def _make_df_col(name, vals):
    col = DoubleColumn()
    for v in vals:
        col.append(float(v))
    return col

def test_df_roll_mean():
    df = FastDataFrame()
    df['A'] = _make_df_col('A', [1, 2, 3, 4, 5, 6, 7, 8])
    df.df_length = 8
    r = df['A'].roll_mean(3)
    check('df_roll_mean length', r.length == 8)
    check('df_roll_mean [2]', _approx(r[2], 2.0))
    check('df_roll_mean [7]', _approx(r[7], 7.0))

def test_df_roll_sum():
    df = FastDataFrame()
    df['A'] = _make_df_col('A', [1, 2, 3, 4, 5])
    df.df_length = 5
    r = df['A'].roll_sum(3)
    check('df_roll_sum [2]', _approx(r[2], 6.0))
    check('df_roll_sum [4]', _approx(r[4], 12.0))

def test_df_roll_min():
    df = FastDataFrame()
    df['A'] = _make_df_col('A', [5, 3, 1, 4, 2])
    df.df_length = 5
    r = df['A'].roll_min(3)
    check('df_roll_min [2]', _approx(r[2], 1.0))
    check('df_roll_min [4]', _approx(r[4], 1.0))

def test_df_roll_max():
    df = FastDataFrame()
    df['A'] = _make_df_col('A', [5, 3, 1, 4, 2])
    df.df_length = 5
    r = df['A'].roll_max(3)
    check('df_roll_max [2]', _approx(r[2], 5.0))
    check('df_roll_max [4]', _approx(r[4], 4.0))

def test_df_roll_var():
    df = FastDataFrame()
    df['A'] = _make_df_col('A', [1, 2, 3, 4, 5])
    df.df_length = 5
    r = df['A'].roll_var(3)
    check('df_roll_var [2]', _approx(r[2], 1.0, 1e-6))

def test_df_roll_std():
    df = FastDataFrame()
    df['A'] = _make_df_col('A', [1, 2, 3, 4, 5])
    df.df_length = 5
    r = df['A'].roll_std(3)
    check('df_roll_std [2]', _approx(r[2], 1.0, 1e-6))

def test_df_roll_median():
    df = FastDataFrame()
    df['A'] = _make_df_col('A', [3, 1, 4, 1, 5])
    df.df_length = 5
    r = df['A'].roll_median(3)
    check('df_roll_median [2]', _approx(r[2], 3.0))
    check('df_roll_median [4]', _approx(r[4], 4.0))

def test_df_roll_quantile():
    df = FastDataFrame()
    df['A'] = _make_df_col('A', [1, 2, 3, 4, 5])
    df.df_length = 5
    r = df['A'].roll_quantile(3, 0.5)
    check('df_roll_quantile [2]', _approx(r[2], 2.0))

def test_df_roll_apply_weights():
    df = FastDataFrame()
    df['A'] = _make_df_col('A', [1, 2, 3, 4, 5])
    df.df_length = 5
    r = df['A'].roll_apply_weights([1.0, 1.0, 1.0])
    check('df_roll_weights [2]', _approx(r[2], 6.0))

def test_df_roll_custom():
    df = FastDataFrame()
    df['A'] = _make_df_col('A', [1, 2, 3, 4, 5])
    df.df_length = 5
    r = df['A'].roll_custom(3, lambda w: w.sum())
    check('df_roll_custom [2]', _approx(r[2], 6.0))

def test_df_roll_dispatch():
    df = FastDataFrame()
    df['A'] = _make_df_col('A', [1, 2, 3, 4, 5, 6, 7, 8])
    df.df_length = 8
    r = df['A'].roll(3, 'mean')
    check('df_roll_dispatch mean', _approx(r[2], 2.0))
    r = df['A'].roll(3, 'sum')
    check('df_roll_dispatch sum', _approx(r[2], 6.0))
    r = df['A'].roll(3, 'min')
    check('df_roll_dispatch min', _approx(r[2], 1.0))
    r = df['A'].roll(3, 'max')
    check('df_roll_dispatch max', _approx(r[2], 3.0))
    r = df['A'].roll(3, 'median')
    check('df_roll_dispatch median', _approx(r[2], 2.0))
    r = df['A'].roll(4, 'var', ddof=False)
    check('df_roll_dispatch var', _approx(r[3], 1.25, 1e-6))
    r = df['A'].roll(4, 'quantile', q=0.5)
    check('df_roll_dispatch quantile', _approx(r[3], 2.5, 1e-6))

def test_df_rolling_vs_direct_roll():
    """验证 df.rolling() (RollingWindow) 与 df["A"].roll_xxx() 结果一致"""
    df = FastDataFrame()
    df['A'] = _make_df_col('A', [1, 2, 3, 4, 5, 6, 7, 8])
    df.df_length = 8
    rw = df.rolling(3, 'A')
    direct = df['A']

    # mean
    check('df_rolling==direct mean',
          _approx(rw.mean()[5], direct.roll_mean(3)[5]))
    # sum
    check('df_rolling==direct sum',
          _approx(rw.sum()[5], direct.roll_sum(3)[5]))
    # min
    check('df_rolling==direct min',
          _approx(rw.min()[5], direct.roll_min(3)[5]))
    # max
    check('df_rolling==direct max',
          _approx(rw.max()[5], direct.roll_max(3)[5]))
    # std (RollingWindow uses ddof=False)
    check('df_rolling==direct std',
          _approx(rw.std()[5], direct.roll_std(3, ddof=False)[5], 1e-6))
    # var (RollingWindow uses ddof=False)
    check('df_rolling==direct var',
          _approx(rw.var()[5], direct.roll_var(3, ddof=False)[5], 1e-6))
    # median
    check('df_rolling==direct median',
          _approx(rw.median()[5], direct.roll_median(3)[5]))

def test_df_rolling_corr_cov():
    """通过 df.rolling() 测试 corr/cov"""
    df = FastDataFrame()
    df['A'] = _make_df_col('A', [1, 2, 3, 4, 5, 6])
    df['B'] = _make_df_col('B', [2, 4, 6, 8, 10, 12])
    df.df_length = 6
    ra = df.rolling(3, 'A')
    rb = df.rolling(3, 'B')
    corr = ra.corr(rb)
    check('df_rolling_corr', _approx(corr[5], 1.0, 1e-6))
    cov = ra.cov(rb)
    check('df_rolling_cov', _approx(cov[5], 4.0/3.0, 1e-6))

# ============================================================
#  标量算术: add_val / sub_val / mul_val / div_val
# ============================================================
def test_add_val():
    col = _make_1d([1.0, 2.0, 3.0])
    r = col.add_val(10.0)
    check('add_val [0]', r[0] == 11.0)
    check('add_val [2]', r[2] == 13.0)
    check('add_val unchanged', col[0] == 1.0)

def test_sub_val():
    col = _make_1d([10.0, 20.0, 30.0])
    r = col.sub_val(5.0)
    check('sub_val [0]', r[0] == 5.0)
    check('sub_val [2]', r[2] == 25.0)

def test_mul_val():
    col = _make_1d([2.0, 3.0, 4.0])
    r = col.mul_val(3.0)
    check('mul_val [0]', r[0] == 6.0)
    check('mul_val [2]', r[2] == 12.0)

def test_div_val():
    col = _make_1d([10.0, 20.0, 30.0])
    r = col.div_val(2.0)
    check('div_val [0]', r[0] == 5.0)
    check('div_val [2]', r[2] == 15.0)
    r0 = col.div_val(0.0)
    check('div_val zero', _is_nan(r0[0]))

# ============================================================
#  argmax / argmin (idxmax/idxmin 别名)
# ============================================================
def test_argmax():
    col = _make_1d([1.0, 5.0, 3.0])
    check('argmax', col.argmax() == 1)

def test_argmin():
    col = _make_1d([5.0, 1.0, 3.0])
    check('argmin', col.argmin() == 1)

# ============================================================
#  属性: cols / rows / is_2d / get_size
# ============================================================
def test_cols_rows_is2d():
    col1d = _make_1d([1.0, 2.0, 3.0])
    check('1d cols', col1d.cols == 1)
    check('1d rows', col1d.rows == 3)
    check('1d is_2d', col1d.is_2d == False)

    col2d = _make_2d(4, [[1, 2, 3, 4], [10, 20, 30, 40]])
    check('2d cols', col2d.cols == 2)
    check('2d rows', col2d.rows == 4)
    check('2d is_2d', col2d.is_2d == True)

def test_get_size():
    col = _make_1d([1.0, 2.0, 3.0])
    sz = col.get_size()
    # get_size returns capacity * sizeof(double), initial capacity=16
    check('get_size', sz == 16 * 8)

# ============================================================
#  quantile (标量分位数)
# ============================================================
def test_quantile():
    col = _make_1d([1.0, 2.0, 3.0, 4.0, 5.0])
    check('quantile 0', col.quantile(0.0) == 1.0)
    check('quantile 0.5', _approx(col.quantile(0.5), 3.0))
    check('quantile 1', col.quantile(1.0) == 5.0)
    check('quantile empty', _is_nan(DoubleColumn().quantile(0.5)))

# ============================================================
#  距离函数: jaccard_sim / hamming_dist
# ============================================================
def test_jaccard_sim():
    a = _make_1d([1.0, 0.0, 1.0, 0.0])
    b = _make_1d([1.0, 1.0, 0.0, 0.0])
    # a 非零: {0,2}, b 非零: {0,1}, 交={0}, 并={0,1,2}
    check('jaccard_sim', _approx(a.jaccard_sim(b), 1.0/3.0))

def test_hamming_dist():
    a = _make_1d([1.0, 2.0, 3.0, 4.0])
    b = _make_1d([1.0, 2.0, 5.0, 4.0])
    check('hamming_dist', _approx(a.hamming_dist(b), 0.25))  # 1 diff out of 4

# ============================================================
#  astype (类型转换)
# ============================================================
def test_astype_int():
    col = _make_1d([1.5, 2.7, 3.0])
    ic = col.astype('int')
    check('astype int [0]', ic[0] == 1)
    check('astype int [1]', ic[1] == 2)
    check('astype int [2]', ic[2] == 3)

def test_astype_bool():
    col = _make_1d([0.0, 1.0, 0.0, 5.0])
    bc = col.astype('bool')
    check('astype bool [0]', bc[0] == False)
    check('astype bool [1]', bc[1] == True)
    check('astype bool [3]', bc[3] == True)

def test_astype_copy():
    col = _make_1d([1.0, 2.0, 3.0])
    cp = col.astype('double')
    check('astype copy [0]', cp[0] == 1.0)
    check('astype copy unchanged', col[0] == 1.0)

# ============================================================
#  2D 下的标量算术与比较
# ============================================================
def test_2d_scalar_ops():
    col = _make_2d(3, [[1, 2, 3], [10, 20, 30]])
    r = col.add_val(100.0)
    check('2d add_val c0', r[0] == 101.0)
    check('2d add_val c1', r[3] == 110.0)

    r = col.mul_val(2.0)
    check('2d mul_val c0', r[0] == 2.0)
    check('2d mul_val c1', r[3] == 20.0)

def test_2d_comparison():
    col = _make_2d(3, [[1, 5, 3], [10, 20, 30]])
    m = col > 4.0
    check('2d gt c0 [0]', m[0] == 0)
    check('2d gt c0 [1]', m[1] == 1)
    check('2d gt c1 [3]', m[3] == 1)   # 10 > 4
    check('2d gt c1 [4]', m[4] == 1)   # 20 > 4

def test_2d_cumsum():
    # cumsum 作用于整个扁平数组 (非逐列独立)
    col = _make_2d(3, [[1, 2, 3], [10, 20, 30]])
    r = col.cumsum()
    # flat: [1, 2, 3, 10, 20, 30] -> cumsum: [1, 3, 6, 16, 36, 66]
    check('2d cumsum [0]', r[0] == 1.0)
    check('2d cumsum [2]', r[2] == 6.0)
    check('2d cumsum [3]', r[3] == 16.0)
    check('2d cumsum [5]', r[5] == 66.0)

def test_2d_diff():
    # diff 作用于整个扁平数组
    col = _make_2d(4, [[1, 3, 6, 10], [100, 200, 350, 550]])
    r = col.diff(1)
    # flat: [1, 3, 6, 10, 100, 200, 350, 550]
    check('2d diff nan [0]', _is_nan(r[0]))
    check('2d diff [1]', _approx(r[1], 2.0))   # 3-1
    check('2d diff [3]', _approx(r[3], 4.0))   # 10-6
    check('2d diff [4]', _approx(r[4], 90.0))  # 100-10

def test_2d_clip():
    col = _make_2d(3, [[1, 5, 10], [2, 8, 15]])
    r = col.clip(3.0, 7.0)
    check('2d clip c0 [0]', r[0] == 3.0)
    check('2d clip c0 [1]', r[1] == 5.0)
    check('2d clip c0 [2]', r[2] == 7.0)
    check('2d clip c1 [3]', r[3] == 3.0)
    check('2d clip c1 [4]', r[4] == 7.0)

def test_2d_unique_nunique():
    col = _make_2d(3, [[1, 2, 1], [10, 10, 20]])
    check('2d nunique', col.nunique() == 4)  # {1, 2, 10, 20}
    u = col.unique()
    check('2d unique len', len(u) == 4)

def test_2d_describe():
    col = _make_2d(3, [[1, 2, 3], [10, 20, 30]])
    d = col.describe()
    # flat: [1, 2, 3, 10, 20, 30], mean = 11.0
    check('2d describe count', d['count'] == 6)
    check('2d describe mean', _approx(d['mean'], 11.0))
    check('2d describe min', d['min'] == 1.0)
    check('2d describe max', d['max'] == 30.0)

# ============================================================
#  drop_row 对 2D 列操作
# ============================================================
def test_2d_drop_row():
    col = _make_2d(4, [[1, 2, 3, 4], [10, 20, 30, 40]])
    col.drop_row(1, 2)
    check('2d drop_row len', col.length == 7)  # 8-1=7
    # flat 原: [1,2,3,4,10,20,30,40], drop index 1 -> [1,3,4,10,20,30,40]
    check('2d drop_row [0]', col[0] == 1.0)
    check('2d drop_row [1]', col[1] == 3.0)
    check('2d drop_row [2]', col[2] == 4.0)

# ============================================================
#  pop_back / pop_front 对 2D 列操作
# ============================================================
def test_2d_pop_back():
    col = _make_2d(3, [[1, 2, 3], [10, 20, 30]])
    val = col.pop_back()
    check('2d pop_back val', val == 30.0)
    check('2d pop_back len', col.length == 5)

def test_2d_pop_front():
    col = _make_2d(3, [[1, 2, 3], [10, 20, 30]])
    val = col.pop_front()
    check('2d pop_front val', val == 1.0)
    check('2d pop_front len', col.length == 5)

def run_all():
    print('=== DoubleColumn Tests ===')
    test_empty()
    test_append()
    test_special_values()
    test_bounds()
    test_pop_back()
    test_pop_front()
    test_copy()
    test_concat()
    test_max()
    test_min()
    test_mean()
    test_var()
    test_std()
    test_sum()
    test_count_nonzeros()
    test_idxmax()
    test_idxmin()
    test_median()
    test_mad()
    test_add()
    test_sub()
    test_mul()
    test_div()
    test_unique()
    test_growth()
    test_comparison()
    test_bitwise()
    test_dunder_add()
    test_dunder_sub()
    test_dunder_mul()
    test_dunder_div()
    test_dunder_notimplemented()
    print('\n--- Roll Tests (1D) ---')
    test_roll_mean_1d()
    test_roll_sum_1d()
    test_roll_min_1d()
    test_roll_max_1d()
    test_roll_var_1d()
    test_roll_std_1d()
    test_roll_median_1d()
    test_roll_quantile_1d()
    test_roll_apply_weights_1d()
    test_roll_custom_1d()
    test_roll_dispatch_1d()
    test_roll_error_1d()
    print('\n--- Roll Tests (2D) ---')
    test_roll_mean_2d()
    test_roll_sum_2d()
    test_roll_min_2d()
    test_roll_max_2d()
    test_roll_var_2d()
    test_roll_std_2d()
    test_roll_median_2d()
    test_roll_quantile_2d()
    test_roll_apply_weights_2d()
    test_roll_custom_2d()
    test_roll_dispatch_2d()
    print('\n--- Roll Tests (Edge Cases) ---')
    test_roll_nan_propagation_1d()
    test_roll_nan_propagation_2d()
    test_roll_window_larger_than_data_1d()
    test_roll_window_larger_than_data_2d()
    test_roll_single_element_window_2d()
    print('\n--- Construction Tests ---')
    test_load_from_1d_array()
    test_load_from_2d_array()
    test_load_from_2d_array_empty()
    test_create_view_via_roll()
    print('\n--- Additional Method Tests ---')
    test_prod()
    test_nunique()
    test_value_counts()
    test_mode()
    test_cumsum()
    test_cummax()
    test_cummin()
    test_cumprod()
    test_diff()
    test_pct_change()
    test_skew()
    test_kurt()
    test_sem()
    test_describe()
    test_isin()
    test_between()
    test_str()
    test_bool()
    test_len()
    test_clip()
    test_where()
    test_where_mask()
    test_mask()
    test_drop_row()
    test_distances()
    test_reorder()
    test_count()
    print('\n--- df[A].roll_xxx() Tests ---')
    test_df_roll_mean()
    test_df_roll_sum()
    test_df_roll_min()
    test_df_roll_max()
    test_df_roll_var()
    test_df_roll_std()
    test_df_roll_median()
    test_df_roll_quantile()
    test_df_roll_apply_weights()
    test_df_roll_custom()
    test_df_roll_dispatch()
    test_df_rolling_vs_direct_roll()
    test_df_rolling_corr_cov()
    print('\n--- Scalar Arithmetic Tests ---')
    test_add_val()
    test_sub_val()
    test_mul_val()
    test_div_val()
    print('\n--- Property / Alias Tests ---')
    test_argmax()
    test_argmin()
    test_cols_rows_is2d()
    test_get_size()
    print('\n--- Quantile / Distance / Astype Tests ---')
    test_quantile()
    test_jaccard_sim()
    test_hamming_dist()
    test_astype_int()
    test_astype_bool()
    test_astype_copy()
    print('\n--- 2D Extended Tests ---')
    test_2d_scalar_ops()
    test_2d_comparison()
    test_2d_cumsum()
    test_2d_diff()
    test_2d_clip()
    test_2d_unique_nunique()
    test_2d_describe()
    test_2d_drop_row()
    test_2d_pop_back()
    test_2d_pop_front()
    print(f'\nResults: {passed} passed, {failed} failed')
    if failed: sys.exit(1)

if __name__ == '__main__':
    run_all()
