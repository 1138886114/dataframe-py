import sys
sys.path.insert(0, '..')
from fluidframe.core.intcolumn import IntColumn
from fluidframe.core.doublecolumn import DoubleColumn
from fluidframe.core.asciicolumn import AsciiColumn
from fluidframe.core.ucs2column import UCS2Column
from fluidframe.core.datecolumn import DateColumn
from fluidframe.core.boolcolumn import BoolColumn
from fluidframe.core.categorycolumn import CategoryColumn
from fluidframe.core.datetimecolumn import DateTimeColumn
from fluidframe.core.timecolumn import TimeColumn

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

# ============================================================
# shift_left tests
# ============================================================

def test_shift_left_int():
    col = IntColumn(4)
    col.append(1)
    col.append(2)
    col.append(3)
    col.shift_left(1)
    check('shift_left_int len', col.length == 2)
    check('shift_left_int [0]', col[0] == 1)
    check('shift_left_int [1]', col[1] == 3)

def test_shift_left_int_first():
    col = IntColumn(4)
    col.append(10)
    col.append(20)
    col.shift_left(0)
    check('shift_left_int_first len', col.length == 1)
    check('shift_left_int_first [0]', col[0] == 20)

def test_shift_left_int_last():
    col = IntColumn(4)
    col.append(10)
    col.append(20)
    col.shift_left(1)
    check('shift_left_int_last len', col.length == 1)
    check('shift_left_int_last [0]', col[0] == 10)

def test_shift_left_double():
    col = DoubleColumn(4)
    col.append(1.5)
    col.append(2.5)
    col.append(3.5)
    col.shift_left(0)
    check('shift_left_double len', col.length == 2)
    check('shift_left_double [0]', col[0] == 2.5)
    check('shift_left_double [1]', col[1] == 3.5)

def test_shift_left_char256():
    col = AsciiColumn(4)
    col.append('abc')
    col.append('def')
    col.append('ghi')
    col.shift_left(1)
    check('shift_left_char256 len', col.length == 2)
    check('shift_left_char256 [0]', col[0] == 'abc')
    check('shift_left_char256 [1]', col[1] == 'ghi')

def test_shift_left_ucs2():
    col = UCS2Column(4)
    col.append('hello')
    col.append('world')
    col.append('foo')
    col.shift_left(1)
    check('shift_left_ucs2 len', col.length == 2)
    check('shift_left_ucs2 [0]', col[0] == 'hello')
    check('shift_left_ucs2 [1]', col[1] == 'foo')

def test_shift_left_date():
    col = DateColumn(4)
    col.append(20240101)
    col.append(20240201)
    col.append(20240301)
    col.shift_left(0)
    check('shift_left_date len', col.length == 2)
    check('shift_left_date [0]', col[0] == 20240201)
    check('shift_left_date [1]', col[1] == 20240301)

def test_shift_left_bool():
    col = BoolColumn(4)
    col.append(True)
    col.append(False)
    col.append(True)
    col.shift_left(1)
    check('shift_left_bool len', col.length == 2)
    check('shift_left_bool [0]', col[0] == True)
    check('shift_left_bool [1]', col[1] == True)

def test_shift_left_category():
    col = CategoryColumn(4)
    col.append('a')
    col.append('b')
    col.append('c')
    col.shift_left(1)
    check('shift_left_category len', col.length == 2)
    check('shift_left_category [0]', col[0] == 'a')
    check('shift_left_category [1]', col[1] == 'c')

def test_shift_left_datetime():
    col = DateTimeColumn(4)
    col.append(1704067200000000000)
    col.append(1704153600000000000)
    col.append(1704240000000000000)
    col.shift_left(0)
    check('shift_left_datetime len', col.length == 2)
    check('shift_left_datetime [0]', col[0] == col[0])

def test_shift_left_time():
    col = TimeColumn(4)
    col.append('01:00:00')
    col.append('02:00:00')
    col.append('03:00:00')
    col.shift_left(1)
    check('shift_left_time len', col.length == 2)
    check('shift_left_time [0]', col[0] == '01:00:00')
    check('shift_left_time [1]', col[1] == '03:00:00')

# ============================================================
# drop_row tests for missing column types
# ============================================================

def test_drop_row_char256():
    col = AsciiColumn(4)
    col.append('aaa')
    col.append('bbb')
    col.append('ccc')
    col.append('ddd')
    col.drop_row(1)
    check('drop_row_char256 len', col.length == 3)
    check('drop_row_char256 [0]', col[0] == 'aaa')
    check('drop_row_char256 [1]', col[1] == 'ccc')
    check('drop_row_char256 [2]', col[2] == 'ddd')

def test_drop_row_char256_first():
    col = AsciiColumn(4)
    col.append('aaa')
    col.append('bbb')
    col.append('ccc')
    col.drop_row(0)
    check('drop_row_char256_first len', col.length == 2)
    check('drop_row_char256_first [0]', col[0] == 'bbb')
    check('drop_row_char256_first [1]', col[1] == 'ccc')

def test_drop_row_char256_last():
    col = AsciiColumn(4)
    col.append('aaa')
    col.append('bbb')
    col.append('ccc')
    col.drop_row(2)
    check('drop_row_char256_last len', col.length == 2)
    check('drop_row_char256_last [0]', col[0] == 'aaa')
    check('drop_row_char256_last [1]', col[1] == 'bbb')

def test_drop_row_ucs2():
    col = UCS2Column(4)
    col.append('aaa')
    col.append('bbb')
    col.append('ccc')
    col.append('ddd')
    col.drop_row(1)
    check('drop_row_ucs2 len', col.length == 3)
    check('drop_row_ucs2 [0]', col[0] == 'aaa')
    check('drop_row_ucs2 [1]', col[1] == 'ccc')
    check('drop_row_ucs2 [2]', col[2] == 'ddd')

def test_drop_row_ucs2_first():
    col = UCS2Column(4)
    col.append('aaa')
    col.append('bbb')
    col.append('ccc')
    col.drop_row(0)
    check('drop_row_ucs2_first len', col.length == 2)
    check('drop_row_ucs2_first [0]', col[0] == 'bbb')
    check('drop_row_ucs2_first [1]', col[1] == 'ccc')

def test_drop_row_ucs2_last():
    col = UCS2Column(4)
    col.append('aaa')
    col.append('bbb')
    col.append('ccc')
    col.drop_row(2)
    check('drop_row_ucs2_last len', col.length == 2)
    check('drop_row_ucs2_last [0]', col[0] == 'aaa')
    check('drop_row_ucs2_last [1]', col[1] == 'bbb')

def test_drop_row_ucs2_varying_lengths():
    col = UCS2Column(4)
    col.append('a')
    col.append('bb')
    col.append('ccc')
    col.append('dddd')
    col.drop_row(1)
    check('drop_row_ucs2_varying len', col.length == 3)
    check('drop_row_ucs2_varying [0]', col[0] == 'a')
    check('drop_row_ucs2_varying [1]', col[1] == 'ccc')
    check('drop_row_ucs2_varying [2]', col[2] == 'dddd')

def test_drop_row_date():
    col = DateColumn(4)
    col.append(20240101)
    col.append(20240201)
    col.append(20240301)
    col.append(20240401)
    col.drop_row(2)
    check('drop_row_date len', col.length == 3)
    check('drop_row_date [0]', col[0] == 20240101)
    check('drop_row_date [1]', col[1] == 20240201)
    check('drop_row_date [2]', col[2] == 20240401)

def test_drop_row_datetime():
    col = DateTimeColumn(4)
    col.append(1704067200000000000)
    col.append(1704153600000000000)
    col.append(1704240000000000000)
    col.append(1704326400000000000)
    col.drop_row(1)
    check('drop_row_datetime len', col.length == 3)
    check('drop_row_datetime [0]', col[0] == col[0])
    check('drop_row_datetime [1]', col[1] == col[1])

def test_drop_row_time():
    col = TimeColumn(4)
    col.append('01:00:00')
    col.append('02:00:00')
    col.append('03:00:00')
    col.append('04:00:00')
    col.drop_row(0)
    check('drop_row_time len', col.length == 3)
    check('drop_row_time [0]', col[0] == '02:00:00')
    check('drop_row_time [1]', col[1] == '03:00:00')
    check('drop_row_time [2]', col[2] == '04:00:00')

# ============================================================
# argsort_by tests
# ============================================================

def test_argsort_by():
    from fluidframe.core.fastdataframe import FastDataFrame
    df = FastDataFrame({'a': int})
    df.append_row(a=3)
    df.append_row(a=1)
    df.append_row(a=2)
    idx = df.argsort_by('a')
    check('argsort_by len', len(idx) == 3)
    check('argsort_by [0]', idx[0] == 1)
    check('argsort_by [1]', idx[1] == 2)
    check('argsort_by [2]', idx[2] == 0)

def test_argsort_by_descending():
    from fluidframe.core.fastdataframe import FastDataFrame
    df = FastDataFrame({'a': int})
    df.append_row(a=3)
    df.append_row(a=1)
    df.append_row(a=2)
    idx = df.argsort_by('a', reverse=True)
    check('argsort_by_desc len', len(idx) == 3)
    check('argsort_by_desc [0]', idx[0] == 0)
    check('argsort_by_desc [1]', idx[1] == 2)
    check('argsort_by_desc [2]', idx[2] == 1)

def test_argsort_by_single():
    from fluidframe.core.fastdataframe import FastDataFrame
    df = FastDataFrame({'a': int})
    df.append_row(a=42)
    idx = df.argsort_by('a')
    check('argsort_by_single len', len(idx) == 1)
    check('argsort_by_single [0]', idx[0] == 0)

def test_argsort_by_doubles():
    from fluidframe.core.fastdataframe import FastDataFrame
    df = FastDataFrame({'a': float})
    df.append_row(a=1.5)
    df.append_row(a=0.5)
    df.append_row(a=2.5)
    idx = df.argsort_by('a')
    check('argsort_by_doubles [0]', idx[0] == 1)
    check('argsort_by_doubles [1]', idx[1] == 0)
    check('argsort_by_doubles [2]', idx[2] == 2)

def test_argsort_by_duplicates():
    from fluidframe.core.fastdataframe import FastDataFrame
    df = FastDataFrame({'a': int})
    df.append_row(a=2)
    df.append_row(a=1)
    df.append_row(a=2)
    idx = df.argsort_by('a')
    check('argsort_by_dup len', len(idx) == 3)
    check('argsort_by_dup [0]', idx[0] == 1)
    check('argsort_by_dup sorted', df['a'][idx[0]] <= df['a'][idx[1]] <= df['a'][idx[2]])

def test_argsort_by_strings():
    from fluidframe.core.fastdataframe import FastDataFrame
    from fluidframe._schema_types import Char256
    df = FastDataFrame({'a': Char256})
    df.append_row(a='c')
    df.append_row(a='a')
    df.append_row(a='b')
    idx = df.argsort_by('a')
    check('argsort_by_str [0]', idx[0] == 1)
    check('argsort_by_str [1]', idx[1] == 2)
    check('argsort_by_str [2]', idx[2] == 0)

# ============================================================
# Edge cases: empty and single-element
# ============================================================

def test_shift_left_empty_int():
    col = IntColumn(4)
    col.append(1)
    col.shift_left(0)
    check('shift_left_empty_int len', col.length == 0)

def test_drop_row_single_int():
    col = IntColumn(4)
    col.append(42)
    col.drop_row(0)
    check('drop_row_single_int len', col.length == 0)

def test_shift_left_single_ucs2():
    col = UCS2Column(4)
    col.append('hello')
    col.shift_left(0)
    check('shift_left_single_ucs2 len', col.length == 0)

def test_drop_row_single_ucs2():
    col = UCS2Column(4)
    col.append('hello')
    col.drop_row(0)
    check('drop_row_single_ucs2 len', col.length == 0)

# ============================================================
# Run all tests
# ============================================================

def run_all():
    global passed, failed
    passed = 0
    failed = 0

    print('=== shift_left Tests ===')
    test_shift_left_int()
    test_shift_left_int_first()
    test_shift_left_int_last()
    test_shift_left_double()
    test_shift_left_char256()
    test_shift_left_ucs2()
    test_shift_left_date()
    test_shift_left_bool()
    test_shift_left_category()
    test_shift_left_datetime()
    test_shift_left_time()

    print('=== drop_row Tests (Missing Types) ===')
    test_drop_row_char256()
    test_drop_row_char256_first()
    test_drop_row_char256_last()
    test_drop_row_ucs2()
    test_drop_row_ucs2_first()
    test_drop_row_ucs2_last()
    test_drop_row_ucs2_varying_lengths()
    test_drop_row_date()
    test_drop_row_datetime()
    test_drop_row_time()

    print('=== argsort_by Tests ===')
    test_argsort_by()
    test_argsort_by_descending()
    test_argsort_by_single()
    test_argsort_by_doubles()
    test_argsort_by_duplicates()
    test_argsort_by_strings()

    print('=== Edge Cases ===')
    test_shift_left_empty_int()
    test_drop_row_single_int()
    test_shift_left_single_ucs2()
    test_drop_row_single_ucs2()

    print(f'\nResults: {passed} passed, {failed} failed')
    return failed == 0

if __name__ == '__main__':
    success = run_all()
    sys.exit(0 if success else 1)
