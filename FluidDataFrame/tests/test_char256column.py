import sys
sys.path.insert(0, '..')
from fluidframe.core.char256column import Char256Column_SoA

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

print('=== Char256Column Tests ===')

col = Char256Column_SoA()
check('empty length', col.length == 0)
check('empty capacity', col.capacity == 16)

col = Char256Column_SoA()
col.append("hello"); col.append("world")
check('append length', col.length == 2)
check('getitem 0', col[0] == "hello")
check('getitem 1', col[1] == "world")

col = Char256Column_SoA(); col.append("")
check('append empty', col[0] == "")

col = Char256Column_SoA(); col.append("x" * 300)
check('append long truncates', len(col[0]) == 255)

col = Char256Column_SoA(); col.append("abc123!@#")
check('append ascii', col[0] == "abc123!@#")

col = Char256Column_SoA(); col.append("test")
check('getitem out of range', expect_exception(IndexError, col.__getitem__, 1))

col = Char256Column_SoA(); col.append("first"); col.append("second")
val = col.pop_back()
check('pop_back val', val == "second")
check('pop_back len', col.length == 1)

col = Char256Column_SoA()
check('pop_back empty', expect_exception(IndexError, col.pop_back))

col = Char256Column_SoA(); col.append("first"); col.append("second"); col.append("third")
val = col.pop_front()
check('pop_front val', val == "first")
check('pop_front len', col.length == 2)
check('pop_front [0]', col[0] == "second")

col = Char256Column_SoA()
check('pop_front empty', expect_exception(IndexError, col.pop_front))

col = Char256Column_SoA(); col.append("hello")
col2 = col.copy()
check('copy length', col2.length == 1)
check('copy [0]', col2[0] == "hello")
col2.append("world")
check('copy independent', col.length == 1)

col1 = Char256Column_SoA(); col1.append("a")
col2 = Char256Column_SoA(); col2.append("b")
col1.concat(col2)
check('concat length', col1.length == 2)
check('concat [0]', col1[0] == "a")
check('concat [1]', col1[1] == "b")

col = Char256Column_SoA()
col.append("hello world"); col.append("foo bar"); col.append("hello again")
result = col.contains("hello")
check('contains', result == [0, 2])

col = Char256Column_SoA(); col.append("hello world")
col.replace("world", "python")
check('replace', col[0] == "hello python")

col = Char256Column_SoA(); col.append("a b c")
col.replaces("a", "x", "b", "y")
check('replaces', col[0] == "x y c")

col = Char256Column_SoA(); col.append("HELLO")
res = col.lower()
check('lower', res[0] == "hello")

col = Char256Column_SoA(); col.append("hello")
res = col.upper()
check('upper', res[0] == "HELLO")

col = Char256Column_SoA(); col.append("  hello  ")
res = col.strip()
check('strip', res[0] == "hello")

col = Char256Column_SoA(); col.append("  hello  ")
res = col.lstrip()
check('lstrip', res[0] == "hello  ")

col = Char256Column_SoA(); col.append("  hello  ")
res = col.rstrip()
check('rstrip', res[0] == "  hello")

col = Char256Column_SoA(); col.append("hello world")
res = col.find("world")
check('find', res[0] == 6)

col = Char256Column_SoA(); col.append("hello")
res = col.find("xyz")
check('find not found', res[0] == -1)

col = Char256Column_SoA(); col.append("aaa")
res = col.count("a")
check('count', res[0] == 3)

col = Char256Column_SoA(); col.append("hello"); col.append("hi")
res = col.len()
check('len [0]', res[0] == 5)
check('len [1]', res[1] == 2)

col = Char256Column_SoA()
col.append("hello world"); col.append("foo bar")
res = col.startswith("hello")
check('startswith [0]', res[0] == 1)
check('startswith [1]', res[1] == 0)

col = Char256Column_SoA()
col.append("hello world"); col.append("foo bar")
res = col.endswith("world")
check('endswith [0]', res[0] == 1)
check('endswith [1]', res[1] == 0)

col = Char256Column_SoA()
col.append("12345"); col.append("abc"); col.append("12a")
res = col.isdigit()
check('isdigit [0]', res[0] == 1)
check('isdigit [1]', res[1] == 0)
check('isdigit [2]', res[2] == 0)

col = Char256Column_SoA()
col.append("abc"); col.append("123"); col.append("a1")
res = col.isalpha()
check('isalpha [0]', res[0] == 1)
check('isalpha [1]', res[1] == 0)
check('isalpha [2]', res[2] == 0)

col = Char256Column_SoA()
col.append("abc123"); col.append("abc 123")
res = col.isalnum()
check('isalnum [0]', res[0] == 1)
check('isalnum [1]', res[1] == 0)

col = Char256Column_SoA()
col.append("   "); col.append("abc")
res = col.isspace()
check('isspace [0]', res[0] == 1)
check('isspace [1]', res[1] == 0)

col = Char256Column_SoA()
col.append("hello"); col.append("HELLO"); col.append("Hello")
res = col.islower()
check('islower [0]', res[0] == 1)
check('islower [1]', res[1] == 0)
check('islower [2]', res[2] == 0)

col = Char256Column_SoA()
col.append("HELLO"); col.append("hello"); col.append("Hello")
res = col.isupper()
check('isupper [0]', res[0] == 1)
check('isupper [1]', res[1] == 0)
check('isupper [2]', res[2] == 0)

col = Char256Column_SoA(); col.append("hello world")
res = col.removeprefix("hello ")
check('removeprefix', res[0] == "world")

col = Char256Column_SoA(); col.append("hello world")
res = col.removesuffix(" world")
check('removesuffix', res[0] == "hello")

col = Char256Column_SoA(); col.append("42")
res = col.zfill(5)
check('zfill', res[0] == "00042")

col = Char256Column_SoA(); col.append("12345")
res = col.zfill(3)
check('zfill already wide', res[0] == "12345")

col = Char256Column_SoA()
col.append("a"); col.append("b"); col.append("a"); col.append("c")
check('unique', col.unique() == ["a", "b", "c"])

col = Char256Column_SoA()
check('unique empty', col.unique() == [])

col = Char256Column_SoA(); col.append("2024-01-15"); col.append("2023-12-25")
res = col.extract(r"(\d{4})-(\d{2})-(\d{2})")
check('extract', res[0] == "2024")

col = Char256Column_SoA(); col.append("no date here")
res = col.extract(r"(\d{4})")
check('extract no match', res[0] == "")

col = Char256Column_SoA(); col.append("2024-01-15")
res1, res2 = col.extract_twice(r"(\d{4})-(\d{2})-(\d{2})")
check('extract_twice [0]', res1[0] == "2024")
check('extract_twice [1]', res2[0] == "01")

col = Char256Column_SoA()
for i in range(100):
    col.append(f"str_{i}")
check('growth length', col.length == 100)
check('growth [99]', col[99] == "str_99")

col = Char256Column_SoA()
col.append("hello\nworld"); col.append("tab\there")
check('special chars [0]', col[0] == "hello\nworld")
check('special chars [1]', col[1] == "tab\there")

print(f'\nResults: {passed} passed, {failed} failed')
if failed: sys.exit(1)
