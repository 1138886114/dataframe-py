import sys
sys.path.insert(0, '.')
from fluidframe.core.boolcolumn import BoolColumn
from fluidframe.core.intcolumn import IntColumn
from fluidframe.core.fastdataframe import FastDataFrame

passed = 0
failed = 0

def check(name, cond):
    global passed, failed
    if cond:
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


class TestBoolColumnConstruction:
    def test_empty(self):
        bc = BoolColumn(4)
        check('empty length', len(bc) == 0)

    def test_append(self):
        bc = BoolColumn(4)
        bc.append(True); bc.append(False); bc.append(True); bc.append(True)
        check('length', len(bc) == 4)
        check('val 0', bc[0] is True)
        check('val 1', bc[1] is False)

    def test_append_bool(self):
        bc = BoolColumn(4)
        bc.append_bool(False); bc.append_bool(True)
        check('append_bool False', bc[0] is False)
        check('append_bool True', bc[1] is True)


class TestBoolColumnOps:
    def test_pop_back(self):
        bc = BoolColumn(4)
        bc.append(True); bc.append(False); bc.append(True)
        bc.pop_back()
        check('pop_back len', len(bc) == 2)
        check('pop_back last', bc[1] is False)

    def test_concat(self):
        bc1 = BoolColumn(4); bc1.append(True); bc1.append(False)
        bc2 = BoolColumn(4); bc2.append(True); bc2.append(True)
        bc1.concat(bc2)
        check('concat len', len(bc1) == 4)
        check('concat val', bc1[2] is True and bc1[3] is True)

    def test_copy(self):
        bc = BoolColumn(4); bc.append(True); bc.append(False)
        bc2 = bc.copy()
        check('copy len', len(bc2) == 2)
        check('copy val', bc2[0] is True and bc2[1] is False)

    def test_eq_ne(self):
        bc1 = BoolColumn(4); bc1.append(True); bc1.append(False)
        bc2 = BoolColumn(4); bc2.append(True); bc2.append(False)
        bc3 = BoolColumn(4); bc3.append(True); bc3.append(True)
        check('eq', bc1 == bc2)
        check('ne', bc1 != bc3)
        check('ne non-column', bc1 != "x")

    def test_bitwise(self):
        a = BoolColumn(4)
        a.append(True); a.append(False); a.append(True); a.append(False)
        b = BoolColumn(4)
        b.append(True); b.append(True); b.append(False); b.append(False)
        r = a & b
        check('and', [r[i] for i in range(4)] == [True, False, False, False])
        r = a | b
        check('or', [r[i] for i in range(4)] == [True, True, True, False])
        r = a ^ b
        check('xor', [r[i] for i in range(4)] == [False, True, True, False])
        r = ~a
        check('not', [r[i] for i in range(4)] == [False, True, False, True])

    def test_sum_mean(self):
        bc = BoolColumn(4)
        bc.append(True); bc.append(False); bc.append(True); bc.append(True)
        check('sum', bc.sum() == 3)
        check('mean', bc.mean() == 0.75)

    def test_isin(self):
        bc = BoolColumn(4)
        bc.append(True); bc.append(False); bc.append(True); bc.append(False)
        r = bc.isin({True})
        check('isin', [r[i] for i in range(4)] == [True, False, True, False])

    def test_where_mask(self):
        mask = IntColumn(4)
        mask.append(1); mask.append(0); mask.append(1); mask.append(0)
        bc = BoolColumn(4)
        bc.append(True); bc.append(True); bc.append(True); bc.append(True)
        r = bc.where_mask(mask, False)
        check('where_mask', [r[i] for i in range(4)] == [True, False, True, False])

    def test_drop_row(self):
        bc = BoolColumn(4)
        bc.append(True); bc.append(False); bc.append(True); bc.append(False)
        bc.drop_row(1)
        check('drop_row len', len(bc) == 3)
        check('drop_row val', [bc[i] for i in range(3)] == [True, True, False])


class TestBoolColumnInDataFrame:
    def test_df_bool(self):
        df = FastDataFrame({'A': int, 'B': bool})
        df.append_row(A=1, B=True); df.append_row(A=2, B=False); df.append_row(A=3, B=True)
        check('df val', df['B'][0] is True and df['B'][1] is False)
        check('df sum', df['B'].sum() == 2)
        check('df mean', abs(df['B'].mean() - 2/3) < 1e-9)


class TestBoolColumnOwnership:
    def test_eq_length_mismatch(self):
        bc1 = BoolColumn(4); bc1.append(True); bc1.append(False); bc1.append(True)
        bc2 = BoolColumn(4); bc2.append(True); bc2.append(False)
        try:
            _ = (bc1 == bc2)
            check('eq mismatch raises', False)
        except ValueError:
            check('eq mismatch raises', True)

    def test_eq_same(self):
        bc1 = BoolColumn(4); bc1.append(True); bc1.append(False)
        bc2 = BoolColumn(4); bc2.append(True); bc2.append(False)
        check('eq same', bc1 == bc2)

    def test_pop_front_raises_on_empty(self):
        bc = BoolColumn(4)
        check('pop_front empty', expect_exception(IndexError, bc.pop_front))

    def test_pop_back_raises_on_empty(self):
        bc = BoolColumn(4)
        check('pop_back empty', expect_exception(IndexError, bc.pop_back))

    def test_owning_pop_front_works(self):
        bc = BoolColumn(4); bc.append(True); bc.append(False); bc.append(True); bc.append(False)
        bc.pop_front()
        check('owning pop_front len', len(bc) == 3)

    def test_owning_pop_back_works(self):
        bc = BoolColumn(4); bc.append(True); bc.append(False); bc.append(True); bc.append(False)
        bc.pop_back()
        check('owning pop_back len', len(bc) == 3)


def run_all():
    print('=== BoolColumn Tests ===')
    for cls in [TestBoolColumnConstruction, TestBoolColumnOps, TestBoolColumnInDataFrame,
                TestBoolColumnOwnership]:
        for name in dir(cls):
            if name.startswith('test_'):
                getattr(cls(), name)()
    print(f'\nResults: {passed} passed, {failed} failed')
    if failed: sys.exit(1)

if __name__ == '__main__':
    run_all()
