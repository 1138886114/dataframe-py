import math
import sys
sys.path.insert(0, '..')
from fluidframe.core.doublecolumn import DoubleColumn
from fluidframe.core.intcolumn import IntColumn

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
    print(f'\nResults: {passed} passed, {failed} failed')
    if failed: sys.exit(1)

if __name__ == '__main__':
    run_all()
