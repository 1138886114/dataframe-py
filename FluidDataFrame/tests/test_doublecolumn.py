import math
import sys
sys.path.insert(0, '..')
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

print('=== DoubleColumn Tests ===')

col = DoubleColumn()
check('empty length', col.length == 0)
check('empty capacity', col.capacity == 16)

col = DoubleColumn()
col.append(1.5); col.append(2.5); col.append(3.5)
check('append length', col.length == 3)
check('getitem 0', col[0] == 1.5)
check('getitem 1', col[1] == 2.5)
check('getitem 2', col[2] == 3.5)

col = DoubleColumn(); col.append(float('nan'))
check('append nan', math.isnan(col[0]))

col = DoubleColumn(); col.append(-0.0)
check('append neg zero', col[0] == -0.0)

col = DoubleColumn(); col.append(1e308); col.append(-1e308)
check('append large', col[0] == 1e308)
check('append small', col[1] == -1e308)

col = DoubleColumn(); col.append(0.0)
check('append zero', col[0] == 0.0)

col = DoubleColumn(); col.append(1.0)
check('getitem out of range', expect_exception(IndexError, col.__getitem__, 1))

col = DoubleColumn(); col.append(1.0); col.append(2.0)
val = col.pop_back()
check('pop_back val', val == 2.0)
check('pop_back len', col.length == 1)

col = DoubleColumn()
check('pop_back empty', expect_exception(IndexError, col.pop_back))

col = DoubleColumn(); col.append(1.0); col.append(2.0); col.append(3.0)
val = col.pop_front()
check('pop_front val', val == 1.0)
check('pop_front len', col.length == 2)
check('pop_front [0]', col[0] == 2.0)

col = DoubleColumn()
check('pop_front empty', expect_exception(IndexError, col.pop_front))

col = DoubleColumn(); col.append(1.5); col.append(2.5)
col2 = col.copy()
check('copy length', col2.length == 2)
check('copy [0]', col2[0] == 1.5)
col2.append(3.5)
check('copy independent', col.length == 2)

col1 = DoubleColumn(); col1.append(1.0)
col2 = DoubleColumn(); col2.append(2.0)
col1.concat(col2)
check('concat length', col1.length == 2)
check('concat [0]', col1[0] == 1.0)
check('concat [1]', col1[1] == 2.0)

col = DoubleColumn(); col.append(1.5); col.append(3.5); col.append(2.5)
check('max', col.max() == 3.5)

col = DoubleColumn()
check('max empty', expect_exception(ValueError, col.max))

col = DoubleColumn(); col.append(1.5); col.append(3.5); col.append(2.5)
check('min', col.min() == 1.5)

col = DoubleColumn()
check('min empty', expect_exception(ValueError, col.min))

col = DoubleColumn(); col.append(1.0); col.append(2.0); col.append(3.0)
check('mean', col.mean() == 2.0)

col = DoubleColumn()
check('mean empty', math.isnan(col.mean()))

col = DoubleColumn(); col.append(1.0); col.append(2.0); col.append(3.0)
check('var', col.var() == 1.0)

col = DoubleColumn(); col.append(1.0); col.append(2.0); col.append(3.0)
check('std', col.std() == 1.0)

col = DoubleColumn(); col.append(1.5); col.append(2.5)
check('sum', col.sum() == 4.0)

col = DoubleColumn(); col.append(0.0); col.append(1.0); col.append(0.0); col.append(5.0)
check('count_nonzeros', col.count_nonzeros() == 2)

col = DoubleColumn(); col.append(1.5); col.append(5.5); col.append(3.5)
check('idxmax', col.idxmax() == 1)

col = DoubleColumn(); col.append(5.5); col.append(1.5); col.append(3.5)
check('idxmin', col.idxmin() == 1)

col = DoubleColumn(); col.append(3.0); col.append(1.0); col.append(2.0)
check('median odd', col.median() == 2.0)

col = DoubleColumn(); col.append(4.0); col.append(1.0); col.append(3.0); col.append(2.0)
check('median even', col.median() == 2.5)

col = DoubleColumn()
check('median empty', math.isnan(col.median()))

col = DoubleColumn(); col.append(1.0); col.append(2.0); col.append(3.0)
check('mad', col.mad() == 2.0 / 3.0)

col = DoubleColumn(); col.append(1.5)
res = col.add(2.5)
check('add', res[0] == 4.0)

col = DoubleColumn(); col.append(5.5)
res = col.sub(2.5)
check('sub', res[0] == 3.0)

col = DoubleColumn(); col.append(3.0)
res = col.mul(2.0)
check('mul', res[0] == 6.0)

col = DoubleColumn(); col.append(10.0)
res = col.div(2.0)
check('div', res[0] == 5.0)

col = DoubleColumn(); col.append(10.0)
res = col.div(0.0)
check('div by zero', math.isnan(res[0]))

col = DoubleColumn(); col.append(1.0); col.append(2.0); col.append(1.0)
check('unique', col.unique() == [1.0, 2.0])

col = DoubleColumn()
check('unique empty', col.unique() == [])

col = DoubleColumn()
for i in range(100):
    col.append(float(i))
check('growth length', col.length == 100)
check('growth [99]', col[99] == 99.0)

print(f'\nResults: {passed} passed, {failed} failed')
if failed: sys.exit(1)
