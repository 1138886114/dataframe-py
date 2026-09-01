import math
import sys
sys.path.insert(0, '..')
from fluidframe.core.ucs2column import UCS2Column_SoA
from fluidframe.core.char256column import Char256Column_SoA
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

def make_ucs2(vals):
    c = UCS2Column_SoA(max(len(vals), 1))
    for v in vals: c.append(v)
    return c

def make_char256(vals):
    c = Char256Column_SoA(max(len(vals), 1))
    for v in vals: c.append(v)
    return c

# ==================== Search / Match ====================
def test_contains():
    for mk in [make_ucs2, make_char256]:
        col = mk(['Hello World', 'Python Programming', 'Data Science', 'hello python'])
        r = col.str.contains('Python')
        name = mk.__name__
        check(f'{name} contains exact', [r[i] for i in range(4)] == [0, 1, 0, 0])
        r = col.str.contains('python')
        check(f'{name} contains case', [r[i] for i in range(4)] == [0, 0, 0, 1])

def test_startswith_endswith():
    for mk in [make_ucs2, make_char256]:
        col = mk(['Hello', 'Python', 'Data', 'hello'])
        name = mk.__name__
        r = col.str.startswith('Hello')
        check(f'{name} startswith', [r[i] for i in range(4)] == [1, 0, 0, 0])
        r = col.str.endswith('on')
        check(f'{name} endswith', [r[i] for i in range(4)] == [0, 1, 0, 0])

def test_find_rfind():
    for mk in [make_ucs2, make_char256]:
        col = mk(['Hello', 'Python', 'Data', 'aabaa'])
        name = mk.__name__
        r = col.str.find('o')
        check(f'{name} find', [r[i] for i in range(4)] == [4, 4, -1, -1])
        r = col.str.rfind('a')
        check(f'{name} rfind', [r[i] for i in range(4)] == [-1, -1, 3, 4])

def test_count():
    for mk in [make_ucs2, make_char256]:
        col = mk(['Hello', 'aabaa', 'aaa', 'xyz'])
        name = mk.__name__
        r = col.str.count('a')
        check(f'{name} count', [r[i] for i in range(4)] == [0, 4, 3, 0])

def test_match_fullmatch():
    for mk in [make_ucs2, make_char256]:
        col = mk(['Hello', 'Hello World', 'hello', '12345'])
        name = mk.__name__
        r = col.str.match('Hello')
        check(f'{name} match', [r[i] for i in range(4)] == [1, 1, 0, 0])
        r = col.str.fullmatch(r'\d+')
        check(f'{name} fullmatch', [r[i] for i in range(4)] == [0, 0, 0, 1])

# ==================== Type Check ====================
def test_isalpha():
    for mk in [make_ucs2, make_char256]:
        col = mk(['abc', 'ABC123', 'Hello', ''])
        name = mk.__name__
        r = col.str.isalpha()
        check(f'{name} isalpha', [r[i] for i in range(4)] == [1, 0, 1, 0])

def test_isdigit_isnumeric_isdecimal():
    for mk in [make_ucs2, make_char256]:
        col = mk(['12345', 'abc', '12.3', ''])
        name = mk.__name__
        r = col.str.isdigit()
        check(f'{name} digit', [r[i] for i in range(4)] == [1, 0, 0, 0])
        r = col.str.isnumeric()
        check(f'{name} numeric', [r[i] for i in range(4)] == [1, 0, 0, 0])
        r = col.str.isdecimal()
        check(f'{name} decimal', [r[i] for i in range(4)] == [1, 0, 0, 0])

def test_islower_isupper():
    for mk in [make_ucs2, make_char256]:
        col = mk(['hello', 'HELLO', 'Hello', ''])
        name = mk.__name__
        r = col.str.islower()
        check(f'{name} islower', [r[i] for i in range(4)] == [1, 0, 0, 0])
        r = col.str.isupper()
        check(f'{name} isupper', [r[i] for i in range(4)] == [0, 1, 0, 0])

def test_isspace():
    for mk in [make_ucs2, make_char256]:
        col = mk([' ', '  ', 'a', ''])
        name = mk.__name__
        r = col.str.isspace()
        check(f'{name} isspace', [r[i] for i in range(4)] == [1, 1, 0, 0])

# ==================== Case Conversion ====================
def test_upper_lower():
    for mk in [make_ucs2, make_char256]:
        col = mk(['Hello', 'PYTHON'])
        name = mk.__name__
        r = col.str.upper()
        check(f'{name} upper', [r[i] for i in range(2)] == ['HELLO', 'PYTHON'])
        r = col.str.lower()
        check(f'{name} lower', [r[i] for i in range(2)] == ['hello', 'python'])

def test_title_capitalize():
    for mk in [make_ucs2, make_char256]:
        col = mk(['hello world', 'PYTHON'])
        name = mk.__name__
        r = col.str.title()
        check(f'{name} title', [r[i] for i in range(2)] == ['Hello World', 'Python'])
        r = col.str.capitalize()
        check(f'{name} capitalize', [r[i] for i in range(2)] == ['Hello world', 'Python'])

def test_swapcase():
    for mk in [make_ucs2, make_char256]:
        col = mk(['Hello'])
        name = mk.__name__
        r = col.str.swapcase()
        check(f'{name} swapcase', r[0] == 'hELLO')

def test_casefold():
    for mk in [make_ucs2, make_char256]:
        col = mk(['Hello'])
        name = mk.__name__
        r = col.str.casefold()
        check(f'{name} casefold', r[0] == 'hello')

# ==================== Strip / Pad ====================
def test_strip():
    for mk in [make_ucs2, make_char256]:
        col = mk(['  hello  ', 'world'])
        name = mk.__name__
        r = col.str.strip()
        check(f'{name} strip', [r[i] for i in range(2)] == ['hello', 'world'])
        r = col.str.lstrip()
        check(f'{name} lstrip', [r[i] for i in range(2)] == ['hello  ', 'world'])
        r = col.str.rstrip()
        check(f'{name} rstrip', [r[i] for i in range(2)] == ['  hello', 'world'])

def test_center_ljust_rjust():
    for mk in [make_ucs2, make_char256]:
        col = mk(['ab', 'cd'])
        name = mk.__name__
        r = col.str.center(6)
        check(f'{name} center', [r[i] for i in range(2)] == ['  ab  ', '  cd  '])
        r = col.str.ljust(6)
        check(f'{name} ljust', [r[i] for i in range(2)] == ['ab    ', 'cd    '])
        r = col.str.rjust(6)
        check(f'{name} rjust', [r[i] for i in range(2)] == ['    ab', '    cd'])

def test_pad():
    for mk in [make_ucs2, make_char256]:
        col = mk(['ab'])
        name = mk.__name__
        r = col.str.pad(6, 'left', '*')
        check(f'{name} pad left', r[0] == '****ab')
        r = col.str.pad(6, 'right', '*')
        check(f'{name} pad right', r[0] == 'ab****')
        r = col.str.pad(6, 'both', '*')
        check(f'{name} pad both', r[0] == '**ab**')

def test_zfill():
    for mk in [make_ucs2, make_char256]:
        col = mk(['42', 'hello'])
        name = mk.__name__
        r = col.str.zfill(8)
        check(f'{name} zfill', [r[i] for i in range(2)] == ['00000042', '000hello'])

# ==================== Split ====================
def test_split():
    for mk in [make_ucs2, make_char256]:
        col = mk(['a b c', 'd-e-f'])
        name = mk.__name__
        r = col.str.split(' ')
        check(f'{name} split space', r == [['a', 'b', 'c'], ['d-e-f']])
        r = col.str.split('-', 1)
        check(f'{name} split dash n=1', r == [['a b c'], ['d', 'e-f']])

def test_rsplit():
    for mk in [make_ucs2, make_char256]:
        col = mk(['a.b.c', 'd-e-f'])
        name = mk.__name__
        r = col.str.rsplit('.', 1)
        check(f'{name} rsplit', r == [['a.b', 'c'], ['d-e-f']])

def test_partition():
    for mk in [make_ucs2, make_char256]:
        col = mk(['hello@world.com', 'abc'])
        name = mk.__name__
        r = col.str.partition('@')
        check(f'{name} partition', r == [('hello', '@', 'world.com'), ('abc', '', '')])

def test_rpartition():
    for mk in [make_ucs2, make_char256]:
        col = mk(['a.b.c', 'abc'])
        name = mk.__name__
        r = col.str.rpartition('.')
        check(f'{name} rpartition', r == [('a.b', '.', 'c'), ('', '', 'abc')])

# ==================== Slicing / Access ====================
def test_getitem():
    for mk in [make_ucs2, make_char256]:
        col = mk(['Hello', 'World'])
        name = mk.__name__
        r = col.str[0]
        check(f'{name} getitem 0', [r[i] for i in range(2)] == ['H', 'W'])
        r = col.str[-1]
        check(f'{name} getitem -1', [r[i] for i in range(2)] == ['o', 'd'])
        r = col.str[1:3]
        check(f'{name} getitem slice', [r[i] for i in range(2)] == ['el', 'or'])

def test_slice():
    for mk in [make_ucs2, make_char256]:
        col = mk(['Hello', 'World'])
        name = mk.__name__
        r = col.str.slice(0, 3)
        check(f'{name} slice 0:3', [r[i] for i in range(2)] == ['Hel', 'Wor'])
        r = col.str.slice(-2)
        check(f'{name} slice -2:', [r[i] for i in range(2)] == ['lo', 'ld'])

def test_get():
    for mk in [make_ucs2, make_char256]:
        col = mk(['Hello', 'World'])
        name = mk.__name__
        r = col.str.get(0)
        check(f'{name} get(0)', [r[i] for i in range(2)] == ['H', 'W'])
        r = col.str.get(10, '?')
        check(f'{name} get(10) default', [r[i] for i in range(2)] == ['?', '?'])

def test_len():
    for mk in [make_ucs2, make_char256]:
        col = mk(['Hello', 'World!'])
        name = mk.__name__
        r = col.str.len()
        check(f'{name} len', [r[i] for i in range(2)] == [5, 6])

# ==================== Concat ====================
def test_cat():
    for mk in [make_ucs2, make_char256]:
        col = mk(['Hello', 'World'])
        name = mk.__name__
        r = col.str.cat(col, sep=' ')
        check(f'{name} cat sep', [r[i] for i in range(2)] == ['Hello Hello', 'World World'])
        r = col.str.cat(col)
        check(f'{name} cat no sep', [r[i] for i in range(2)] == ['HelloHello', 'WorldWorld'])
        r = col.str.cat(['!', '?'], sep='')
        check(f'{name} cat list', [r[i] for i in range(2)] == ['Hello!', 'World?'])

def test_prefix_suffix():
    for mk in [make_ucs2, make_char256]:
        col = mk(['Hello', 'World'])
        name = mk.__name__
        r = col.str.prefix('>>')
        check(f'{name} prefix', [r[i] for i in range(2)] == ['>>Hello', '>>World'])
        r = col.str.suffix('<<')
        check(f'{name} suffix', [r[i] for i in range(2)] == ['Hello<<', 'World<<'])

def test_repeat():
    for mk in [make_ucs2, make_char256]:
        col = mk(['ab', 'cd'])
        name = mk.__name__
        r = col.str.repeat(3)
        check(f'{name} repeat', [r[i] for i in range(2)] == ['ababab', 'cdcdcd'])

# ==================== Extract / Findall ====================
def test_findall():
    for mk in [make_ucs2, make_char256]:
        col = mk(['a1b2', 'c3d4', 'xyz'])
        name = mk.__name__
        r = col.str.findall(r'\d+')
        check(f'{name} findall', r == [['1', '2'], ['3', '4'], []])

def test_extract():
    for mk in [make_ucs2, make_char256]:
        col = mk(['abc-123', 'def-456', 'xyz'])
        name = mk.__name__
        r = col.str.extract(r'(\w+)-(\d+)', 1)
        check(f'{name} extract g1', [r[i] for i in range(3)] == ['abc', 'def', ''])

# ==================== Chained Operations ====================
def test_chained():
    for mk in [make_ucs2, make_char256]:
        col = mk(['  Hello World  ', '  Python  '])
        name = mk.__name__
        r = col.str.strip().upper()
        check(f'{name} chain strip+upper', [r[i] for i in range(2)] == ['HELLO WORLD', 'PYTHON'])
        r = col.str.strip().lower()
        check(f'{name} chain strip+lower', [r[i] for i in range(2)] == ['hello world', 'python'])

# ==================== Registration ====================
def run_all():
    print('=== StringAccessor Tests ===')
    test_contains()
    test_startswith_endswith()
    test_find_rfind()
    test_count()
    test_match_fullmatch()
    test_isalpha()
    test_isdigit_isnumeric_isdecimal()
    test_islower_isupper()
    test_isspace()
    test_upper_lower()
    test_title_capitalize()
    test_swapcase()
    test_casefold()
    test_strip()
    test_center_ljust_rjust()
    test_pad()
    test_zfill()
    test_split()
    test_rsplit()
    test_partition()
    test_rpartition()
    test_getitem()
    test_slice()
    test_get()
    test_len()
    test_cat()
    test_prefix_suffix()
    test_repeat()
    test_findall()
    test_extract()
    test_chained()
    print(f'\nResults: {passed} passed, {failed} failed')
    if failed: sys.exit(1)

if __name__ == '__main__':
    run_all()
