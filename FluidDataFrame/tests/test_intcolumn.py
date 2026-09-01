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
    print(f'\nResults: {passed} passed, {failed} failed')
    if failed: sys.exit(1)

if __name__ == '__main__':
    run_all()
