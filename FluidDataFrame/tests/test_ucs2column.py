import sys
sys.path.insert(0, '.')
from fluidframe.core.ucs2column import UCS2Column

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

def make(vals):
    c = UCS2Column(max(len(vals), 1))
    for v in vals: c.append(v)
    return c


# =====================================================================
#  Column-level tests (from test_ucs2column.py)
# =====================================================================
def test_empty():
    col = UCS2Column()
    check('empty length', col.length == 0)
    check('empty capacity', col.capacity == 64)

def test_append():
    col = UCS2Column()
    col.append("hello"); col.append("world")
    check('append length', col.length == 2)
    check('getitem 0', col[0] == "hello")
    check('getitem 1', col[1] == "world")

def test_unicode():
    col = UCS2Column(); col.append("")
    check('append empty', col[0] == "")

    col = UCS2Column()
    col.append("你好世界"); col.append("こんにちは")
    check('unicode [0]', col[0] == "你好世界")
    check('unicode [1]', col[1] == "こんにちは")

    col = UCS2Column(); col.append("hello你好")
    check('mixed ascii unicode', col[0] == "hello你好")

def test_bounds():
    col = UCS2Column(); col.append("test")
    check('getitem out of range', expect_exception(IndexError, col.__getitem__, 1))

def test_pop_back():
    col = UCS2Column(); col.append("first"); col.append("second")
    val = col.pop_back()
    check('pop_back val', val == "second")
    check('pop_back len', col.length == 1)

    col = UCS2Column()
    check('pop_back empty', expect_exception(IndexError, col.pop_back))

def test_pop_front():
    col = UCS2Column(); col.append("first"); col.append("second"); col.append("third")
    val = col.pop_front()
    check('pop_front val', val == "first")
    check('pop_front len', col.length == 2)
    check('pop_front [0]', col[0] == "second")

    col = UCS2Column()
    check('pop_front empty', expect_exception(IndexError, col.pop_front))

def test_copy():
    col = UCS2Column(); col.append("hello")
    col2 = col.copy()
    check('copy length', col2.length == 1)
    check('copy [0]', col2[0] == "hello")
    col2.append("world")
    check('copy independent', col.length == 1)

def test_concat():
    col1 = UCS2Column(); col1.append("a")
    col2 = UCS2Column(); col2.append("b")
    col1.concat(col2)
    check('concat length', col1.length == 2)
    check('concat [0]', col1[0] == "a")
    check('concat [1]', col1[1] == "b")

def test_unique():
    col = UCS2Column()
    col.append("a"); col.append("b"); col.append("a")
    check('unique', col.unique() == ["a", "b"])

    col = UCS2Column()
    check('unique empty', col.unique() == [])

def test_growth():
    col = UCS2Column()
    for i in range(200):
        col.append(f"str_{i}")
    check('growth length', col.length == 200)
    check('growth [199]', col[199] == "str_199")

def test_reorder():
    col = UCS2Column(); col.append("a"); col.append("b"); col.append("c")
    col._reorder_impl([2, 0, 1])
    check('reorder [0]', col[0] == "c")
    check('reorder [1]', col[1] == "a")
    check('reorder [2]', col[2] == "b")


# =====================================================================
#  Str accessor tests
# =====================================================================
class TestUCS2Search:
    def test_contains(self):
        col = make(['Hello World', 'Python Programming', 'Data Science', 'hello python'])
        r = col._str.contains('Python')
        check('exact', [r[i] for i in range(4)] == [0, 1, 0, 0])
        r = col._str.contains('python')
        check('case', [r[i] for i in range(4)] == [0, 0, 0, 1])

    def test_startswith_endswith(self):
        col = make(['Hello', 'Python', 'Data', 'hello'])
        check('startswith', [col._str.startswith('Hello')[i] for i in range(4)] == [1, 0, 0, 0])
        check('endswith', [col._str.endswith('on')[i] for i in range(4)] == [0, 1, 0, 0])

    def test_find_rfind(self):
        col = make(['Hello', 'Python', 'Data', 'aabaa'])
        check('find', [col._str.find('o')[i] for i in range(4)] == [4, 4, -1, -1])
        check('rfind', [col._str.rfind('a')[i] for i in range(4)] == [-1, -1, 3, 4])

    def test_count(self):
        col = make(['Hello', 'aabaa', 'aaa', 'xyz'])
        check('count', [col._str.count('a')[i] for i in range(4)] == [0, 4, 3, 0])

    def test_match_fullmatch(self):
        col = make(['Hello', 'Hello World', 'hello', '12345'])
        check('match', [col._str.match('Hello')[i] for i in range(4)] == [1, 1, 0, 0])
        check('fullmatch', [col._str.fullmatch(r'\d+')[i] for i in range(4)] == [0, 0, 0, 1])


class TestUCS2TypeCheck:
    def test_isalpha(self):
        col = make(['abc', 'ABC123', 'Hello', ''])
        check('isalpha', [col._str.isalpha()[i] for i in range(4)] == [1, 0, 1, 0])

    def test_isdigit_isnumeric_isdecimal(self):
        col = make(['12345', 'abc', '12.3', ''])
        check('digit', [col._str.isdigit()[i] for i in range(4)] == [1, 0, 0, 0])
        check('numeric', [col._str.isnumeric()[i] for i in range(4)] == [1, 0, 0, 0])
        check('decimal', [col._str.isdecimal()[i] for i in range(4)] == [1, 0, 0, 0])

    def test_islower_isupper(self):
        col = make(['hello', 'HELLO', 'Hello', ''])
        check('islower', [col._str.islower()[i] for i in range(4)] == [1, 0, 0, 0])
        check('isupper', [col._str.isupper()[i] for i in range(4)] == [0, 1, 0, 0])

    def test_isspace(self):
        col = make([' ', '  ', 'a', ''])
        check('isspace', [col._str.isspace()[i] for i in range(4)] == [1, 1, 0, 0])


class TestUCS2CaseConversion:
    def test_upper_lower(self):
        col = make(['Hello', 'PYTHON'])
        check('upper', [col._str.upper()[i] for i in range(2)] == ['HELLO', 'PYTHON'])
        check('lower', [col._str.lower()[i] for i in range(2)] == ['hello', 'python'])

    def test_title_capitalize(self):
        col = make(['hello world', 'PYTHON'])
        check('title', [col._str.title()[i] for i in range(2)] == ['Hello World', 'Python'])
        check('capitalize', [col._str.capitalize()[i] for i in range(2)] == ['Hello world', 'Python'])

    def test_swapcase(self):
        col = make(['Hello'])
        check('swapcase', col._str.swapcase()[0] == 'hELLO')

    def test_casefold(self):
        col = make(['Hello'])
        check('casefold', col._str.casefold()[0] == 'hello')


class TestUCS2StripPad:
    def test_strip(self):
        col = make(['  hello  ', 'world'])
        check('strip', [col._str.strip()[i] for i in range(2)] == ['hello', 'world'])
        check('lstrip', [col._str.lstrip()[i] for i in range(2)] == ['hello  ', 'world'])
        check('rstrip', [col._str.rstrip()[i] for i in range(2)] == ['  hello', 'world'])

    def test_center_ljust_rjust(self):
        col = make(['ab', 'cd'])
        check('center', [col._str.center(6)[i] for i in range(2)] == ['  ab  ', '  cd  '])
        check('ljust', [col._str.ljust(6)[i] for i in range(2)] == ['ab    ', 'cd    '])
        check('rjust', [col._str.rjust(6)[i] for i in range(2)] == ['    ab', '    cd'])

    def test_pad(self):
        col = make(['ab'])
        check('pad left', col._str.pad(6, 'left', '*')[0] == '****ab')
        check('pad right', col._str.pad(6, 'right', '*')[0] == 'ab****')
        check('pad both', col._str.pad(6, 'both', '*')[0] == '**ab**')

    def test_zfill(self):
        col = make(['42', 'hello'])
        check('zfill', [col._str.zfill(8)[i] for i in range(2)] == ['00000042', '000hello'])


class TestUCS2Split:
    def test_split(self):
        col = make(['a b c', 'd-e-f'])
        check('split space', col._str.split(' ') == [['a', 'b', 'c'], ['d-e-f']])
        check('split dash', col._str.split('-', 1) == [['a b c'], ['d', 'e-f']])

    def test_rsplit(self):
        col = make(['a.b.c', 'd-e-f'])
        check('rsplit', col._str.rsplit('.', 1) == [['a.b', 'c'], ['d-e-f']])

    def test_partition(self):
        col = make(['hello@world.com', 'abc'])
        check('partition', col._str.partition('@') == [('hello', '@', 'world.com'), ('abc', '', '')])

    def test_rpartition(self):
        col = make(['a.b.c', 'abc'])
        check('rpartition', col._str.rpartition('.') == [('a.b', '.', 'c'), ('', '', 'abc')])


class TestUCS2Access:
    def test_getitem(self):
        col = make(['Hello', 'World'])
        check('getitem 0', [col._str[0][i] for i in range(2)] == ['H', 'W'])
        check('getitem -1', [col._str[-1][i] for i in range(2)] == ['o', 'd'])
        check('getitem slice', [col._str[1:3][i] for i in range(2)] == ['el', 'or'])

    def test_slice(self):
        col = make(['Hello', 'World'])
        check('slice 0:3', [col._str.slice(0, 3)[i] for i in range(2)] == ['Hel', 'Wor'])
        check('slice -2:', [col._str.slice(-2)[i] for i in range(2)] == ['lo', 'ld'])

    def test_get(self):
        col = make(['Hello', 'World'])
        check('get(0)', [col._str.get(0)[i] for i in range(2)] == ['H', 'W'])
        check('get(10)', [col._str.get(10, '?')[i] for i in range(2)] == ['?', '?'])

    def test_len(self):
        col = make(['Hello', 'World!'])
        check('len', [col._str.len()[i] for i in range(2)] == [5, 6])


class TestUCS2Concat:
    def test_cat(self):
        col = make(['Hello', 'World'])
        check('cat sep', [col._str.cat(col, sep=' ')[i] for i in range(2)] == ['Hello Hello', 'World World'])
        check('cat no sep', [col._str.cat(col)[i] for i in range(2)] == ['HelloHello', 'WorldWorld'])
        check('cat list', [col._str.cat(['!', '?'], sep='')[i] for i in range(2)] == ['Hello!', 'World?'])

    def test_prefix_suffix(self):
        col = make(['Hello', 'World'])
        check('prefix', [col._str.prefix('>>')[i] for i in range(2)] == ['>>Hello', '>>World'])
        check('suffix', [col._str.suffix('<<')[i] for i in range(2)] == ['Hello<<', 'World<<'])

    def test_repeat(self):
        col = make(['ab', 'cd'])
        check('repeat', [col._str.repeat(3)[i] for i in range(2)] == ['ababab', 'cdcdcd'])


class TestUCS2Regex:
    def test_findall(self):
        col = make(['a1b2', 'c3d4', 'xyz'])
        check('findall', col._str.findall(r'\d+') == [['1', '2'], ['3', '4'], []])

    def test_extract(self):
        col = make(['abc-123', 'def-456', 'xyz'])
        r = col._str.extract(r'(\w+)-(\d+)', 1)
        check('extract', [r[i] for i in range(3)] == ['abc', 'def', ''])


class TestUCS2Chained:
    def test_chained(self):
        col = make(['  Hello World  ', '  Python  '])
        check('strip+upper', [col._str.strip().upper()[i] for i in range(2)] == ['HELLO WORLD', 'PYTHON'])
        check('strip+lower', [col._str.strip().lower()[i] for i in range(2)] == ['hello world', 'python'])


def run_all():
    print('=== UCS2Column Tests ===')
    # Column-level tests
    test_empty()
    test_append()
    test_unicode()
    test_bounds()
    test_pop_back()
    test_pop_front()
    test_copy()
    test_concat()
    test_unique()
    test_growth()
    test_reorder()
    # Str accessor tests
    for cls in [TestUCS2Search, TestUCS2TypeCheck, TestUCS2CaseConversion,
                TestUCS2StripPad, TestUCS2Split, TestUCS2Access,
                TestUCS2Concat, TestUCS2Regex, TestUCS2Chained]:
        for name in dir(cls):
            if name.startswith('test_'):
                getattr(cls(), name)()
    print(f'\nResults: {passed} passed, {failed} failed')
    if failed: sys.exit(1)

if __name__ == '__main__':
    run_all()
