import math
import sys
sys.path.insert(0, '..')
from fluidframe.core.intcolumn import IntColumn
from fluidframe.core.doublecolumn import DoubleColumn

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
    col = IntColumn()
    check('empty length', col.length == 0)
    check('empty capacity', col.capacity == 16)

def test_append():
    col = IntColumn()
    col.append(10); col.append(20); col.append(30)
    check('append length', col.length == 3)
    check('getitem 0', col[0] == 10)
    check('getitem 1', col[1] == 20)
    check('getitem 2', col[2] == 30)

def test_negative():
    col = IntColumn()
    col.append(-1); col.append(-2147483648)
    check('append negative', col[0] == -1)
    check('append min int', col[1] == -2147483648)

def test_positive():
    col = IntColumn()
    col.append(2147483647); col.append(0)
    check('append max int', col[0] == 2147483647)
    check('append zero', col[1] == 0)

def test_bounds():
    col = IntColumn(); col.append(1)
    check('getitem out of range', expect_exception(IndexError, col.__getitem__, 1))
    check('getitem negative', expect_exception(IndexError, col.__getitem__, -2))

def test_pop_back():
    col = IntColumn(); col.append(1); col.append(2)
    val = col.pop_back()
    check('pop_back val', val == 2)
    check('pop_back len', col.length == 1)

    col2 = IntColumn()
    check('pop_back empty', expect_exception(IndexError, col2.pop_back))

def test_pop_front():
    col = IntColumn(); col.append(1); col.append(2); col.append(3)
    val = col.pop_front()
    check('pop_front val', val == 1)
    check('pop_front len', col.length == 2)
    check('pop_front [0]', col[0] == 2)
    check('pop_front [1]', col[1] == 3)

    col2 = IntColumn()
    check('pop_front empty', expect_exception(IndexError, col2.pop_front))

    col = IntColumn(); col.append(42)
    val = col.pop_front()
    check('pop_front single val', val == 42)
    check('pop_front single len', col.length == 0)

def test_copy():
    col = IntColumn(); col.append(1); col.append(2)
    col2 = col.copy()
    check('copy length', col2.length == 2)
    check('copy [0]', col2[0] == 1)
    check('copy [1]', col2[1] == 2)
    col2.append(3)
    check('copy independent orig', col.length == 2)
    check('copy independent new', col2.length == 3)

def test_concat():
    col1 = IntColumn(); col1.append(1); col1.append(2)
    col2 = IntColumn(); col2.append(3); col2.append(4)
    col1.concat(col2)
    check('concat length', col1.length == 4)
    check('concat [0]', col1[0] == 1)
    check('concat [3]', col1[3] == 4)

def test_max():
    col = IntColumn(); col.append(5); col.append(3); col.append(8); col.append(1)
    check('max', col.max() == 8)

    col = IntColumn(); col.append(42)
    check('max single', col.max() == 42)

    col2 = IntColumn()
    check('max empty', expect_exception(ValueError, col2.max))

def test_min():
    col = IntColumn(); col.append(5); col.append(3); col.append(8); col.append(1)
    check('min', col.min() == 1)

    col = IntColumn()
    check('min empty', expect_exception(ValueError, col.min))

def test_mean():
    col = IntColumn(); col.append(10); col.append(20); col.append(30)
    check('mean', col.mean() == 20.0)

    col = IntColumn()
    check('mean empty', math.isnan(col.mean()))

def test_var():
    col = IntColumn(); col.append(1); col.append(2); col.append(3)
    check('var', col.var() == 1.0)

    col = IntColumn(); col.append(5)
    check('var single', col.var() == 0.0)

def test_std():
    col = IntColumn(); col.append(1); col.append(2); col.append(3)
    check('std', col.std() == 1.0)

def test_sum():
    col = IntColumn(); col.append(1); col.append(2); col.append(3)
    check('sum', col.sum() == 6.0)

    col = IntColumn()
    check('sum empty', col.sum() == 0.0)

def test_count_nonzeros():
    col = IntColumn(); col.append(0); col.append(1); col.append(0); col.append(5)
    check('count_nonzeros', col.count_nonzeros() == 2)

def test_idxmax():
    col = IntColumn(); col.append(1); col.append(5); col.append(3)
    check('idxmax', col.idxmax() == 1)

    col = IntColumn()
    check('idxmax empty', expect_exception(ValueError, col.idxmax))

def test_idxmin():
    col = IntColumn(); col.append(5); col.append(1); col.append(3)
    check('idxmin', col.idxmin() == 1)

    col = IntColumn()
    check('idxmin empty', expect_exception(ValueError, col.idxmin))

def test_argmax():
    col = IntColumn(); col.append(1); col.append(5)
    check('argmax', col.argmax() == 1)

def test_argmin():
    col = IntColumn(); col.append(5); col.append(1)
    check('argmin', col.argmin() == 1)

def test_median():
    col = IntColumn(); col.append(3); col.append(1); col.append(2)
    check('median odd', col.median() == 2)

    col = IntColumn(); col.append(4); col.append(1); col.append(3); col.append(2)
    check('median even', col.median() == 2.5)

    col = IntColumn()
    check('median empty', math.isnan(col.median()))

def test_mad():
    col = IntColumn(); col.append(1); col.append(2); col.append(3)
    check('mad', col.mad() == 2.0 / 3.0)

    col = IntColumn()
    check('mad empty', math.isnan(col.mad()))

def test_add():
    col = IntColumn(); col.append(1); col.append(2)
    res = col.add_val(10)
    check('add [0]', res[0] == 11)
    check('add [1]', res[1] == 12)

def test_sub():
    col = IntColumn(); col.append(10); col.append(20)
    res = col.sub_val(5)
    check('sub [0]', res[0] == 5)
    check('sub [1]', res[1] == 15)

def test_mul():
    col = IntColumn(); col.append(3); col.append(4)
    res = col.mul_val(2)
    check('mul [0]', res[0] == 6)
    check('mul [1]', res[1] == 8)

def test_div():
    col = IntColumn(); col.append(10); col.append(20)
    res = col.div_val(5)
    check('div [0]', res[0] == 2.0)
    check('div [1]', res[1] == 4.0)

    col = IntColumn(); col.append(10)
    res = col.div_val(0)
    check('div by zero', math.isnan(res[0]))

def test_diff():
    col = IntColumn(); col.append(1); col.append(3); col.append(6); col.append(10)
    res = col.diff()
    check('diff [0]', res[0] == 0)
    check('diff [1]', res[1] == 2)
    check('diff [2]', res[2] == 3)
    check('diff [3]', res[3] == 4)

    col = IntColumn()
    res = col.diff()
    check('diff empty', res.length == 0)

def test_unique():
    col = IntColumn(); col.append(1); col.append(2); col.append(1); col.append(3); col.append(2)
    check('unique', col.unique() == [1, 2, 3])

    col = IntColumn()
    check('unique empty', col.unique() == [])

    col = IntColumn(); col.append(42)
    check('unique single', col.unique() == [42])

def test_growth():
    col = IntColumn()
    for i in range(100):
        col.append(i)
    check('growth length', col.length == 100)
    check('growth [99]', col[99] == 99)

def test_reorder():
    col = IntColumn(); col.append(10); col.append(20); col.append(30)
    col._reorder([2, 0, 1])
    check('reorder [0]', col[0] == 30)
    check('reorder [1]', col[1] == 10)
    check('reorder [2]', col[2] == 20)

def test_comparison():
    col = IntColumn()
    col.append(1); col.append(3); col.append(5); col.append(2)
    m = col < 3
    check('lt', [m[i] for i in range(m.length)] == [1, 0, 0, 1])
    m = col >= 3
    check('ge', [m[i] for i in range(m.length)] == [0, 1, 1, 0])
    m = col == 5
    check('eq', [m[i] for i in range(m.length)] == [0, 0, 1, 0])
    m = col != 3
    check('ne', [m[i] for i in range(m.length)] == [1, 0, 1, 1])

def test_bitwise():
    col = IntColumn()
    col.append(1); col.append(3); col.append(5); col.append(2)
    m1 = col < 4     # [1, 1, 0, 1]
    m2 = col > 2     # [0, 1, 1, 0]
    m_and = m1 & m2
    check('and', [m_and[i] for i in range(m_and.length)] == [0, 1, 0, 0])
    m_or = m1 | m2
    check('or', [m_or[i] for i in range(m_or.length)] == [1, 1, 1, 1])
    m_not = ~m1
    check('invert', [m_not[i] for i in range(m_not.length)] == [0, 0, 1, 0])

def test_dunder_add():
    a = IntColumn(); a.append(1); a.append(2); a.append(3)
    b = IntColumn(); b.append(10); b.append(20); b.append(30)
    r = a + b
    check('__add__ col-col', r[0] == 11 and r[1] == 22 and r[2] == 33)
    r = a + 5
    check('__add__ scalar', r[0] == 6 and r[1] == 7 and r[2] == 8)
    r = 100 + a
    check('__radd__ scalar', r[0] == 101 and r[1] == 102 and r[2] == 103)
    a_d = DoubleColumn(); a_d.append(1.5); a_d.append(2.5)
    b_i = IntColumn(); b_i.append(10); b_i.append(20)
    r = a_d + b_i
    check('__radd__ double+int', r[0] == 11.5 and r[1] == 22.5)

def test_dunder_sub():
    a = IntColumn(); a.append(10); a.append(20); a.append(30)
    b = IntColumn(); b.append(1); b.append(2); b.append(3)
    r = a - b
    check('__sub__ col-col', r[0] == 9 and r[1] == 18 and r[2] == 27)
    r = a - 5
    check('__sub__ scalar', r[0] == 5 and r[1] == 15 and r[2] == 25)
    r = 100 - a
    check('__rsub__ scalar', r[0] == 90 and r[1] == 80 and r[2] == 70)

def test_dunder_mul():
    a = IntColumn(); a.append(2); a.append(3); a.append(4)
    b = IntColumn(); b.append(10); b.append(20); b.append(30)
    r = a * b
    check('__mul__ col-col', r[0] == 20 and r[1] == 60 and r[2] == 120)
    r = a * 3
    check('__mul__ scalar', r[0] == 6 and r[1] == 9 and r[2] == 12)
    r = 3 * a
    check('__rmul__ scalar', r[0] == 6 and r[1] == 9 and r[2] == 12)

def test_dunder_div():
    a = IntColumn(); a.append(10); a.append(20); a.append(30)
    b = IntColumn(); b.append(2); b.append(4); b.append(5)
    r = a / b
    check('__truediv__ col-col', r[0] == 5.0 and r[1] == 5.0 and r[2] == 6.0)
    r = a / 3
    check('__truediv__ scalar', abs(r[0] - 10.0/3) < 1e-9 and abs(r[1] - 20.0/3) < 1e-9 and r[2] == 10.0)
    r = 60 / a
    check('__rtruediv__ scalar', r[0] == 6.0 and r[1] == 3.0 and r[2] == 2.0)
    r = a / 0
    check('__truediv__ div by zero', math.isnan(r[0]))
    r = 0 / a
    check('__rtruediv__ zero numerator', r[0] == 0.0)

def test_dunder_notimplemented():
    a = IntColumn(); a.append(1)
    check('__add__ NotImplemented', type(a).__add__(a, "x") is NotImplemented)
    check('__sub__ NotImplemented', type(a).__sub__(a, "x") is NotImplemented)
    check('__mul__ NotImplemented', type(a).__mul__(a, "x") is NotImplemented)
    check('__truediv__ NotImplemented', type(a).__truediv__(a, "x") is NotImplemented)
    check('__rsub__ NotImplemented', type(a).__rsub__(a, "x") is NotImplemented)
    check('__rtruediv__ NotImplemented', type(a).__rtruediv__(a, "x") is NotImplemented)

def test_roll_mean():
    col = IntColumn(10)
    for v in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        col.append(v)
    r = col.roll_mean(3)
    check('roll_mean len', r.length == 10)
    check('roll_mean nan0', math.isnan(r[0]))
    check('roll_mean nan1', math.isnan(r[1]))
    check('roll_mean v2', abs(r[2] - 2.0) < 1e-9)
    check('roll_mean v3', abs(r[3] - 3.0) < 1e-9)
    check('roll_mean v9', abs(r[9] - 9.0) < 1e-9)

def test_roll_sum():
    col = IntColumn(10)
    for v in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        col.append(v)
    r = col.roll_sum(3)
    check('roll_sum nan0', math.isnan(r[0]))
    check('roll_sum v2', abs(r[2] - 6.0) < 1e-9)
    check('roll_sum v3', abs(r[3] - 9.0) < 1e-9)
    check('roll_sum v9', abs(r[9] - 27.0) < 1e-9)

def test_roll_min():
    col = IntColumn(10)
    for v in [5, 3, 1, 4, 2, 8, 6, 7, 9, 0]:
        col.append(v)
    r = col.roll_min(3)
    check('roll_min sentinel0', r[0] == 2147483647)
    check('roll_min sentinel1', r[1] == 2147483647)
    check('roll_min v2', r[2] == 1)
    check('roll_min v3', r[3] == 1)
    check('roll_min v4', r[4] == 1)
    check('roll_min v9', r[9] == 0)

def test_roll_max():
    col = IntColumn(10)
    for v in [5, 3, 1, 4, 2, 8, 6, 7, 9, 0]:
        col.append(v)
    r = col.roll_max(3)
    check('roll_max sentinel0', r[0] == -2147483647 - 1)
    check('roll_max v2', r[2] == 5)
    check('roll_max v3', r[3] == 4)
    check('roll_max v5', r[5] == 8)
    check('roll_max v8', r[8] == 9)

def test_roll_var():
    col = IntColumn(10)
    for v in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        col.append(v)
    r = col.roll_var(3, ddof=False)
    check('roll_var nan0', math.isnan(r[0]))
    check('roll_var v2', abs(r[2] - 2.0/3) < 1e-6)
    r2 = col.roll_var(3, ddof=True)
    check('roll_var ddof1', abs(r2[2] - 1.0) < 1e-6)

def test_roll_std():
    col = IntColumn(10)
    for v in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        col.append(v)
    r = col.roll_std(3, ddof=False)
    check('roll_std nan0', math.isnan(r[0]))
    check('roll_std v2', abs(r[2] - (2.0/3)**0.5) < 1e-6)

def test_roll_median():
    col = IntColumn(10)
    for v in [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]:
        col.append(v)
    r = col.roll_median(3)
    check('roll_median nan0', math.isnan(r[0]))
    check('roll_median v2', abs(r[2] - 3.0) < 1e-9)
    check('roll_median v3', abs(r[3] - 1.0) < 1e-9)
    check('roll_median v4', abs(r[4] - 4.0) < 1e-9)

def test_roll_quantile():
    col = IntColumn(10)
    for v in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        col.append(v)
    r = col.roll_quantile(3, 0.5)
    check('roll_quantile nan0', math.isnan(r[0]))
    check('roll_quantile v2', abs(r[2] - 2.0) < 1e-9)

def test_roll_apply_weights():
    col = IntColumn(10)
    for v in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        col.append(v)
    r = col.roll_apply_weights(3, [1.0, 2.0, 3.0])
    check('roll_weights nan0', math.isnan(r[0]))
    check('roll_weights v2', abs(r[2] - 14.0) < 1e-9)
    check('roll_weights v3', abs(r[3] - 20.0) < 1e-9)

def test_roll_custom():
    col = IntColumn(10)
    for v in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        col.append(v)
    r = col.roll_custom(3, lambda x: max(x))
    check('roll_custom nan0', math.isnan(r[0]))
    check('roll_custom v2', abs(r[2] - 3.0) < 1e-9)
    check('roll_custom v3', abs(r[3] - 4.0) < 1e-9)

def test_roll_dispatch():
    col = IntColumn(10)
    for v in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        col.append(v)
    r = col.roll(3, method='sum')
    check('roll_dispatch sum', abs(r[2] - 6.0) < 1e-9)
    r = col.roll(3, method='mean')
    check('roll_dispatch mean', abs(r[2] - 2.0) < 1e-9)
    r = col.roll(3, method='min')
    check('roll_dispatch min', r[2] == 1)
    r = col.roll(3, method='max')
    check('roll_dispatch max', r[2] == 3)
    r = col.roll(3, method='var', ddof=False)
    check('roll_dispatch var', abs(r[2] - 2.0/3) < 1e-6)
    r = col.roll(3, method='std', ddof=False)
    check('roll_dispatch std', abs(r[2] - (2.0/3)**0.5) < 1e-6)
    r = col.roll(3, method='median')
    check('roll_dispatch median', abs(r[2] - 2.0) < 1e-9)
    r = col.roll(3, method='quantile', quantile=0.5)
    check('roll_dispatch quantile', abs(r[2] - 2.0) < 1e-9)
    r = col.roll(3, method='apply_weights', weights=[1.0, 2.0, 3.0])
    check('roll_dispatch weights', abs(r[2] - 14.0) < 1e-9)
    r = col.roll(3, method='custom', func=lambda x: max(x))
    check('roll_dispatch custom', abs(r[2] - 3.0) < 1e-9)

def test_roll_edge_cases():
    col = IntColumn(5)
    for v in [1, 2, 3, 4, 5]:
        col.append(v)
    r = col.roll_mean(1)
    check('roll_w1 mean', abs(r[0] - 1.0) < 1e-9)
    r = col.roll_mean(5)
    check('roll_w5 mean', abs(r[4] - 3.0) < 1e-9)
    r = col.roll_mean(6)
    check('roll_w>len all nan', all(math.isnan(r[i]) for i in range(5)))
    r = col.roll_mean(0)
    check('roll_w0 all nan', all(math.isnan(r[i]) for i in range(5)))

def test_roll_mean_2d():
    col = _make_2d(4, [[1, 2, 3, 4], [10, 20, 30, 40]])
    check('roll_mean_2d rows', col.rows == 4)
    check('roll_mean_2d cols', col.cols == 2)
    r = col.roll_mean(3)
    check('roll_mean_2d c0 [2]', _approx(r[2], 2.0))
    check('roll_mean_2d c0 [3]', _approx(r[3], 3.0))
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
    check('roll_min_2d c0 [2]', _approx(r[2], 1.0))
    check('roll_min_2d c0 [3]', _approx(r[3], 1.0))
    check('roll_min_2d c0 [4]', _approx(r[4], 1.0))
    check('roll_min_2d c1 [7]', _approx(r[7], 10.0))

def test_roll_max_2d():
    col = _make_2d(5, [[5, 3, 1, 4, 2], [50, 30, 10, 40, 20]])
    r = col.roll_max(3)
    check('roll_max_2d c0 [2]', _approx(r[2], 5.0))
    check('roll_max_2d c0 [3]', _approx(r[3], 4.0))
    check('roll_max_2d c0 [4]', _approx(r[4], 4.0))
    check('roll_max_2d c1 [7]', _approx(r[7], 50.0))

def test_roll_var_2d():
    col = _make_2d(5, [[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]])
    r = col.roll_var(3, ddof=False)
    check('roll_var_2d c0 [2]', _approx(r[2], 2.0/3, 1e-6))
    check('roll_var_2d c1 [7]', _approx(r[7], 200.0/3, 1e-6))

def test_roll_std_2d():
    col = _make_2d(5, [[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]])
    r = col.roll_std(3, ddof=False)
    check('roll_std_2d c0 [2]', _approx(r[2], (2.0/3)**0.5, 1e-6))
    check('roll_std_2d c1 [7]', _approx(r[7], (200.0/3)**0.5, 1e-6))

def test_roll_median_2d():
    col = _make_2d(5, [[3, 1, 4, 1, 5], [30, 10, 40, 10, 50]])
    r = col.roll_median(3)
    check('roll_median_2d c0 [2]', _approx(r[2], 3.0))
    check('roll_median_2d c0 [3]', _approx(r[3], 1.0))
    check('roll_median_2d c0 [4]', _approx(r[4], 4.0))
    check('roll_median_2d c1 [7]', _approx(r[7], 30.0))

def test_roll_quantile_2d():
    col = _make_2d(5, [[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]])
    r = col.roll_quantile(3, 0.5)
    check('roll_quantile_2d c0 [2]', _approx(r[2], 2.0))
    check('roll_quantile_2d c1 [7]', _approx(r[7], 20.0))

def test_roll_apply_weights_2d():
    col = _make_2d(4, [[1, 2, 3, 4], [10, 20, 30, 40]])
    r = col.roll_apply_weights(3, [1.0, 2.0, 3.0])
    check('roll_weights_2d c0 [2]', _approx(r[2], 14.0))
    check('roll_weights_2d c1 [6]', _approx(r[6], 140.0))

def test_roll_custom_2d():
    col = _make_2d(4, [[1, 2, 3, 4], [10, 20, 30, 40]])
    r = col.roll_custom(3, lambda x: max(x))
    check('roll_custom_2d c0 [2]', _approx(r[2], 3.0))
    check('roll_custom_2d c1 [6]', _approx(r[6], 30.0))

def test_roll_dispatch_2d():
    col = _make_2d(4, [[1, 2, 3, 4], [10, 20, 30, 40]])
    r = col.roll(3, method='mean')
    check('roll_dispatch_2d mean c0', _approx(r[2], 2.0))
    check('roll_dispatch_2d mean c1', _approx(r[6], 20.0))
    r = col.roll(3, method='sum')
    check('roll_dispatch_2d sum c0', _approx(r[2], 6.0))
    check('roll_dispatch_2d sum c1', _approx(r[6], 60.0))
    r = col.roll(3, method='min')
    check('roll_dispatch_2d min c0', _approx(r[2], 1.0))
    check('roll_dispatch_2d min c1', _approx(r[6], 10.0))
    r = col.roll(3, method='max')
    check('roll_dispatch_2d max c0', _approx(r[2], 3.0))
    check('roll_dispatch_2d max c1', _approx(r[6], 30.0))

def test_roll_edge_cases_2d():
    col = _make_2d(3, [[1, 2, 3], [10, 20, 30]])
    r = col.roll_mean(1)
    check('roll_w1_2d c0', _approx(r[0], 1.0))
    check('roll_w1_2d c1', _approx(r[3], 10.0))

def test_roll_nan_propagation():
    col = IntColumn(6)
    for v in [1, 2, 3, 4, 5, 6]:
        col.append(v)
    r = col.roll_mean(3)
    check('roll_nan no crash', r.length == 6)
    check('roll_nan nan0', math.isnan(r[0]))
    check('roll_nan nan1', math.isnan(r[1]))

def test_roll_nan_propagation_2d():
    col = _make_2d(4, [[1, 2, 3, 4], [10, 20, 30, 40]])
    r = col.roll_mean(3)
    check('roll_nan_2d no crash', r.length == 8)

def test_roll_window_larger_than_data():
    col = IntColumn(3)
    for v in [1, 2, 3]:
        col.append(v)
    r = col.roll_mean(5)
    check('roll_win>len all nan', all(math.isnan(r[i]) for i in range(3)))

def test_roll_window_larger_than_data_2d():
    col = _make_2d(3, [[1, 2, 3], [10, 20, 30]])
    r = col.roll_mean(5)
    check('roll_win>len_2d all nan', all(math.isnan(r[i]) for i in range(6)))

def test_roll_single_element_window():
    col = IntColumn(5)
    for v in [1, 2, 3, 4, 5]:
        col.append(v)
    r = col.roll_mean(1)
    check('roll_w1 c0', _approx(r[0], 1.0))
    check('roll_w1 c4', _approx(r[4], 5.0))

def test_roll_single_element_window_2d():
    col = _make_2d(4, [[1, 2, 3, 4], [10, 20, 30, 40]])
    r = col.roll_mean(1)
    check('roll_w1_2d c0', all(_approx(r[i], col[i]) for i in range(4)))
    check('roll_w1_2d c1', all(_approx(r[4+i], col[4+i]) for i in range(4)))

def _make_2d(rows, cols_vals):
    c = len(cols_vals)
    total = rows * c
    col = IntColumn(max(total, 1), cols=c)
    for col_idx, vals in enumerate(cols_vals):
        for row_idx, v in enumerate(vals):
            col.append(int(v))
    return col

def _approx(a, b, tol=1e-9):
    if math.isnan(a) and math.isnan(b): return True
    return abs(a - b) < tol

def test_cols_rows_is_2d():
    col1d = IntColumn(5)
    for v in [1, 2, 3, 4, 5]:
        col1d.append(v)
    check('1d cols', col1d.cols == 1)
    check('1d rows', col1d.rows == 5)
    check('1d is_2d', col1d.is_2d == False)

    col2d = _make_2d(4, [[1, 2, 3, 4], [10, 20, 30, 40]])
    check('2d cols', col2d.cols == 2)
    check('2d rows', col2d.rows == 4)
    check('2d is_2d', col2d.is_2d == True)

def test_create_view():
    import numpy as np
    arr = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    col = IntColumn.load_from_1d_array(arr)
    check('view length', col.length == 5)
    check('view [0]', col[0] == 10)
    check('view [4]', col[4] == 50)
    check('view owns_data', col._owns_data == False)
    arr[0] = 99
    check('view zero_copy', col[0] == 99)

def test_load_from_1d_array():
    import numpy as np
    arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    col = IntColumn.load_from_1d_array(arr)
    check('load_1d length', col.length == 5)
    check('load_1d [0]', col[0] == 1)
    check('load_1d [4]', col[4] == 5)
    check('load_1d is_2d', col.is_2d == False)
    check('load_1d cols', col.cols == 1)

def test_load_from_2d_array():
    import numpy as np
    arr = np.array([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=np.int32, order='F')
    cols_dict = IntColumn.load_from_2d_array(arr, ['A', 'B'])
    check('load_2d keys', set(cols_dict.keys()) == {'A', 'B'})
    col_a = cols_dict['A']
    check('load_2d A length', col_a.length == 4)
    check('load_2d A [0]', col_a[0] == 1)
    check('load_2d A [3]', col_a[3] == 4)
    check('load_2d A owns_data', col_a._owns_data == False)
    col_b = cols_dict['B']
    check('load_2d B [0]', col_b[0] == 10)
    check('load_2d B [3]', col_b[3] == 40)
    arr[0, 0] = 99
    check('load_2d zero_copy A', col_a[0] == 99)

def test_2d_scalar_ops():
    col = _make_2d(3, [[1, 2, 3], [10, 20, 30]])
    r = col + 5
    check('2d add [0]', r[0] == 6)
    check('2d add [3]', r[3] == 15)
    r = col * 2
    check('2d mul [0]', r[0] == 2)
    check('2d mul [3]', r[3] == 20)

def test_2d_comparison():
    col = _make_2d(3, [[1, 2, 3], [10, 20, 30]])
    r = col.__lt__(3)
    check('2d lt [0]', r[0] == 1)
    check('2d lt [1]', r[1] == 1)
    check('2d lt [2]', r[2] == 0)

def test_2d_cumsum():
    col = _make_2d(3, [[1, 2, 3], [10, 20, 30]])
    r = col.cumsum()
    check('2d cumsum [0]', r[0] == 1)
    check('2d cumsum [2]', r[2] == 6)
    check('2d cumsum [3]', r[3] == 16)

def test_2d_diff():
    col = _make_2d(3, [[1, 2, 3], [10, 20, 30]])
    r = col.diff()
    check('2d diff [0]', r[0] == 0)
    check('2d diff [1]', r[1] == 1)
    check('2d diff [2]', r[2] == 1)

def test_2d_clip():
    col = _make_2d(3, [[1, 2, 3], [10, 20, 30]])
    r = col.clip(2, 25)
    check('2d clip [0]', r[0] == 2)
    check('2d clip [1]', r[1] == 2)
    check('2d clip [2]', r[2] == 3)
    check('2d clip [3]', r[3] == 10)
    check('2d clip [5]', r[5] == 25)

def test_2d_unique():
    col = _make_2d(3, [[1, 1, 2], [10, 10, 20]])
    u = col.unique()
    check('2d unique len', len(u) == 4)

def test_2d_describe():
    col = _make_2d(3, [[1, 2, 3], [10, 20, 30]])
    d = col.describe()
    check('2d describe type', isinstance(d, dict))

def test_2d_drop_row():
    col = _make_2d(3, [[1, 2, 3], [10, 20, 30]])
    col2 = col.copy()
    col2.drop_row(1)
    check('2d drop_row len', col2.length == 5)

def test_2d_pop_back():
    col = _make_2d(3, [[1, 2, 3], [10, 20, 30]])
    v = col.pop_back()
    check('2d pop_back val', v == 30)
    check('2d pop_back len', col.length == 5)

def test_2d_pop_front():
    col = _make_2d(3, [[1, 2, 3], [10, 20, 30]])
    v = col.pop_front()
    check('2d pop_front val', v == 1)
    check('2d pop_front len', col.length == 5)

def test_2d_append():
    col = _make_2d(3, [[1, 2, 3], [10, 20, 30]])
    col.append(40)
    check('2d append len', col.length == 7)
    check('2d append val', col[6] == 40)

def test_2d_concat():
    col1 = _make_2d(3, [[1, 2, 3], [10, 20, 30]])
    col2 = _make_2d(2, [[4, 5], [40, 50]])
    col1.concat(col2)
    check('2d concat len', col1.length == 10)

def test_add_val():
    col = IntColumn(5)
    for v in [1, 2, 3, 4, 5]:
        col.append(v)
    other = IntColumn(5)
    for v in [10, 10, 10, 10, 10]:
        other.append(v)
    r = col.add(other)
    check('add_val [0]', r[0] == 11)
    check('add_val [4]', r[4] == 15)

def test_sub_val():
    col = IntColumn(5)
    for v in [10, 20, 30, 40, 50]:
        col.append(v)
    other = IntColumn(5)
    for v in [5, 5, 5, 5, 5]:
        other.append(v)
    r = col.sub(other)
    check('sub_val [0]', r[0] == 5)
    check('sub_val [4]', r[4] == 45)

def test_mul_val():
    col = IntColumn(5)
    for v in [1, 2, 3, 4, 5]:
        col.append(v)
    other = IntColumn(5)
    for v in [3, 3, 3, 3, 3]:
        other.append(v)
    r = col.mul(other)
    check('mul_val [0]', r[0] == 3)
    check('mul_val [4]', r[4] == 15)

def test_div_val():
    col = IntColumn(5)
    for v in [10, 20, 30, 40, 50]:
        col.append(v)
    other = IntColumn(5)
    for v in [10, 10, 10, 10, 10]:
        other.append(v)
    r = col.div(other)
    check('div_val [0]', r[0] == 1.0)
    check('div_val [4]', r[4] == 5.0)

def test_prod():
    col = IntColumn(5)
    for v in [1, 2, 3, 4, 5]:
        col.append(v)
    check('prod', col.prod() == 120)

def test_nunique():
    col = IntColumn(6)
    for v in [1, 2, 2, 3, 3, 3]:
        col.append(v)
    check('nunique', col.nunique() == 3)

def test_value_counts():
    col = IntColumn(5)
    for v in [1, 1, 2, 3, 3]:
        col.append(v)
    vc = col.value_counts()
    check('value_counts type', isinstance(vc, dict))
    check('value_counts 1', vc.get(1, 0) == 2)

def test_mode():
    col = IntColumn(5)
    for v in [1, 2, 2, 3, 3]:
        col.append(v)
    m = col.mode()
    check('mode type', isinstance(m, list))

def test_cumsum():
    col = IntColumn(5)
    for v in [1, 2, 3, 4, 5]:
        col.append(v)
    r = col.cumsum()
    check('cumsum [0]', r[0] == 1)
    check('cumsum [4]', r[4] == 15)

def test_cummax():
    col = IntColumn(5)
    for v in [3, 1, 4, 1, 5]:
        col.append(v)
    r = col.cummax()
    check('cummax [0]', r[0] == 3)
    check('cummax [1]', r[1] == 3)
    check('cummax [2]', r[2] == 4)
    check('cummax [4]', r[4] == 5)

def test_cummin():
    col = IntColumn(5)
    for v in [3, 1, 4, 1, 5]:
        col.append(v)
    r = col.cummin()
    check('cummin [0]', r[0] == 3)
    check('cummin [1]', r[1] == 1)
    check('cummin [2]', r[2] == 1)

def test_cumprod():
    col = IntColumn(4)
    for v in [1, 2, 3, 4]:
        col.append(v)
    r = col.cumprod()
    check('cumprod [0]', r[0] == 1)
    check('cumprod [3]', r[3] == 24)

def test_pct_change():
    col = IntColumn(5)
    for v in [10, 20, 30, 40, 50]:
        col.append(v)
    r = col.pct_change()
    check('pct_change [0]', math.isnan(r[0]))
    check('pct_change [1]', abs(r[1] - 1.0) < 1e-9)

def test_skew():
    col = IntColumn(10)
    for v in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        col.append(v)
    s = col.skew()
    check('skew type', isinstance(s, float))

def test_kurt():
    col = IntColumn(10)
    for v in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        col.append(v)
    k = col.kurt()
    check('kurt type', isinstance(k, float))

def test_sem():
    col = IntColumn(10)
    for v in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        col.append(v)
    s = col.sem()
    check('sem type', isinstance(s, float))
    check('sem > 0', s > 0)

def test_describe():
    col = IntColumn(10)
    for v in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        col.append(v)
    d = col.describe()
    check('describe type', isinstance(d, dict))

def test_isin():
    col = IntColumn(5)
    for v in [1, 2, 3, 4, 5]:
        col.append(v)
    r = col.isin({2, 4})
    check('isin [0]', r[0] == False)
    check('isin [1]', r[1] == True)
    check('isin [2]', r[2] == False)
    check('isin [3]', r[3] == True)

def test_between():
    col = IntColumn(5)
    for v in [1, 2, 3, 4, 5]:
        col.append(v)
    r = col.between(2, 4)
    check('between [0]', r[0] == False)
    check('between [1]', r[1] == True)
    check('between [4]', r[4] == False)

def test_clip():
    col = IntColumn(5)
    for v in [1, 2, 3, 4, 5]:
        col.append(v)
    r = col.clip(2, 4)
    check('clip [0]', r[0] == 2)
    check('clip [2]', r[2] == 3)
    check('clip [4]', r[4] == 4)

def test_where():
    col = IntColumn(5)
    cond = IntColumn(5)
    for v in [1, 2, 3, 4, 5]:
        col.append(v)
    for v in [1, 0, 1, 0, 1]:
        cond.append(v)
    r = col.where(cond, -1)
    check('where [0]', r[0] == 1)
    check('where [1]', r[1] == -1)
    check('where [2]', r[2] == 3)

def test_where_mask():
    col = IntColumn(5)
    cond = IntColumn(5)
    for v in [1, 2, 3, 4, 5]:
        col.append(v)
    for v in [1, 0, 1, 0, 1]:
        cond.append(v)
    r = col.where_mask(cond, 0)
    check('where_mask [0]', r[0] == 1)
    check('where_mask [1]', r[1] == 0)

def test_mask():
    col = IntColumn(5)
    cond = IntColumn(5)
    for v in [1, 2, 3, 4, 5]:
        col.append(v)
    for v in [1, 0, 1, 0, 1]:
        cond.append(v)
    r = col.mask(cond, -1)
    check('mask [0]', r[0] == -1)
    check('mask [1]', r[1] == 2)
    check('mask [2]', r[2] == -1)

def test_drop_row():
    col = IntColumn(5)
    for v in [1, 2, 3, 4, 5]:
        col.append(v)
    col.drop_row(2)
    check('drop_row len', col.length == 4)
    check('drop_row [2]', col[2] == 4)

def test_pop_back():
    col = IntColumn(5)
    for v in [1, 2, 3, 4, 5]:
        col.append(v)
    v = col.pop_back()
    check('pop_back val', v == 5)
    check('pop_back len', col.length == 4)

def test_pop_front():
    col = IntColumn(5)
    for v in [1, 2, 3, 4, 5]:
        col.append(v)
    v = col.pop_front()
    check('pop_front val', v == 1)
    check('pop_front len', col.length == 4)
    check('pop_front [0]', col[0] == 2)

def test_astype():
    col = IntColumn(3)
    for v in [1, 2, 3]:
        col.append(v)
    r = col.astype('double')
    check('astype double type', isinstance(r, DoubleColumn))
    check('astype double [0]', r[0] == 1.0)

def test_get_size():
    col = IntColumn(10)
    sz = col.get_size()
    check('get_size > 0', sz > 0)

def test_quantile():
    col = IntColumn(10)
    for v in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        col.append(v)
    q = col.quantile(0.5)
    check('quantile 0.5', abs(q - 5.5) < 1e-9)

def test_distances_int_int():
    a = IntColumn(3)
    for v in [1, 2, 3]:
        a.append(v)
    b = IntColumn(3)
    for v in [4, 5, 6]:
        b.append(v)
    check('pearson_corr int-int', _approx(a.pearson_corr_with(b), 1.0))
    check('pearson_cov int-int', a.pearson_cov_with(b) > 0)
    check('euclidean int-int', _approx(a.euclidean_dist(b), math.sqrt(27.0)))
    check('manhattan int-int', _approx(a.manhattan_dist(b), 9.0))
    cd = a.cosine_dist(b)
    check('cosine_dist int-int', 0.0 <= cd <= 2.0)
    cs = a.cosine_similarity(b)
    check('cosine_sim int-int', -1.0 <= cs <= 1.0)

def test_distances_int_double():
    a = IntColumn(3)
    for v in [1, 2, 3]:
        a.append(v)
    b = DoubleColumn(3)
    for v in [4.0, 5.0, 6.0]:
        b.append(v)
    check('pearson_corr int-dbl', _approx(a.pearson_corr_with(b), 1.0))
    check('pearson_cov int-dbl', a.pearson_cov_with(b) > 0)
    check('euclidean int-dbl', _approx(a.euclidean_dist(b), math.sqrt(27.0)))
    check('manhattan int-dbl', _approx(a.manhattan_dist(b), 9.0))
    cd = a.cosine_dist(b)
    check('cosine_dist int-dbl', 0.0 <= cd <= 2.0)
    cs = a.cosine_similarity(b)
    check('cosine_sim int-dbl', -1.0 <= cs <= 1.0)

def test_jaccard_sim_int():
    a = IntColumn(4)
    for v in [1, 0, 1, 0]:
        a.append(v)
    b = IntColumn(4)
    for v in [1, 1, 0, 0]:
        b.append(v)
    check('jaccard_sim int', _approx(a.jaccard_sim(b), 1.0/3.0))

def test_hamming_dist_int():
    a = IntColumn(4)
    for v in [1, 2, 3, 4]:
        a.append(v)
    b = IntColumn(4)
    for v in [1, 2, 5, 4]:
        b.append(v)
    check('hamming_dist int', _approx(a.hamming_dist(b), 0.25))

def run_all():
    print('=== IntColumn Tests ===')
    test_empty()
    test_append()
    test_negative()
    test_positive()
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
    test_argmax()
    test_argmin()
    test_median()
    test_mad()
    test_add()
    test_sub()
    test_mul()
    test_div()
    test_diff()
    test_unique()
    test_growth()
    test_reorder()
    test_comparison()
    test_bitwise()
    test_dunder_add()
    test_dunder_sub()
    test_dunder_mul()
    test_dunder_div()
    test_dunder_notimplemented()
    test_roll_mean()
    test_roll_sum()
    test_roll_min()
    test_roll_max()
    test_roll_var()
    test_roll_std()
    test_roll_median()
    test_roll_quantile()
    test_roll_apply_weights()
    test_roll_custom()
    test_roll_dispatch()
    test_roll_edge_cases()
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
    test_roll_edge_cases_2d()
    test_roll_nan_propagation()
    test_roll_nan_propagation_2d()
    test_roll_window_larger_than_data()
    test_roll_window_larger_than_data_2d()
    test_roll_single_element_window()
    test_roll_single_element_window_2d()
    test_cols_rows_is_2d()
    test_create_view()
    test_load_from_1d_array()
    test_load_from_2d_array()
    test_2d_scalar_ops()
    test_2d_comparison()
    test_2d_cumsum()
    test_2d_diff()
    test_2d_clip()
    test_2d_unique()
    test_2d_describe()
    test_2d_drop_row()
    test_2d_pop_back()
    test_2d_pop_front()
    test_2d_append()
    test_2d_concat()
    test_add_val()
    test_sub_val()
    test_mul_val()
    test_div_val()
    test_prod()
    test_nunique()
    test_value_counts()
    test_mode()
    test_cumsum()
    test_cummax()
    test_cummin()
    test_cumprod()
    test_pct_change()
    test_skew()
    test_kurt()
    test_sem()
    test_describe()
    test_isin()
    test_between()
    test_clip()
    test_where()
    test_where_mask()
    test_mask()
    test_drop_row()
    test_astype()
    test_get_size()
    test_quantile()
    test_distances_int_int()
    test_distances_int_double()
    test_jaccard_sim_int()
    test_hamming_dist_int()
    print(f'\nResults: {passed} passed, {failed} failed')
    if failed: sys.exit(1)

if __name__ == '__main__':
    run_all()
