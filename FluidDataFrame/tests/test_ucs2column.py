import sys
sys.path.insert(0, '..')
from fluidframe.core.ucs2column import UCS2Column_SoA

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

print('=== UCS2Column Tests ===')

col = UCS2Column_SoA()
check('empty length', col.length == 0)
check('empty capacity', col.capacity == 64)

col = UCS2Column_SoA()
col.append("hello"); col.append("world")
check('append length', col.length == 2)
check('getitem 0', col[0] == "hello")
check('getitem 1', col[1] == "world")

col = UCS2Column_SoA(); col.append("")
check('append empty', col[0] == "")

col = UCS2Column_SoA()
col.append("你好世界"); col.append("こんにちは")
check('unicode [0]', col[0] == "你好世界")
check('unicode [1]', col[1] == "こんにちは")

col = UCS2Column_SoA(); col.append("hello你好")
check('mixed ascii unicode', col[0] == "hello你好")

col = UCS2Column_SoA(); col.append("test")
check('getitem out of range', expect_exception(IndexError, col.__getitem__, 1))

col = UCS2Column_SoA(); col.append("first"); col.append("second")
val = col.pop_back()
check('pop_back val', val == "second")
check('pop_back len', col.length == 1)

col = UCS2Column_SoA()
check('pop_back empty', expect_exception(IndexError, col.pop_back))

col = UCS2Column_SoA(); col.append("first"); col.append("second"); col.append("third")
val = col.pop_front()
check('pop_front val', val == "first")
check('pop_front len', col.length == 2)
check('pop_front [0]', col[0] == "second")

col = UCS2Column_SoA()
check('pop_front empty', expect_exception(IndexError, col.pop_front))

col = UCS2Column_SoA(); col.append("hello")
col2 = col.copy()
check('copy length', col2.length == 1)
check('copy [0]', col2[0] == "hello")
col2.append("world")
check('copy independent', col.length == 1)

col1 = UCS2Column_SoA(); col1.append("a")
col2 = UCS2Column_SoA(); col2.append("b")
col1.concat(col2)
check('concat length', col1.length == 2)
check('concat [0]', col1[0] == "a")
check('concat [1]', col1[1] == "b")

col = UCS2Column_SoA()
col.append("hello world"); col.append("foo bar")
res = col.contains("hello")
check('contains [0]', res[0] == 1)
check('contains [1]', res[1] == 0)

col = UCS2Column_SoA(); col.append("hello world")
res = col.find("world")
check('find', res[0] == 6)

col = UCS2Column_SoA(); col.append("hello")
res = col.find("xyz")
check('find not found', res[0] == -1)

col = UCS2Column_SoA(); col.append("hello world")
res = col.replace("world", "python")
check('replace', res[0] == "hello python")

col = UCS2Column_SoA(); col.append("HELLO")
res = col.lower()
check('lower', res[0] == "hello")

col = UCS2Column_SoA(); col.append("hello")
res = col.upper()
check('upper', res[0] == "HELLO")

col = UCS2Column_SoA(); col.append("  hello  ")
res = col.strip()
check('strip', res[0] == "hello")

col = UCS2Column_SoA(); col.append("  hello  ")
res = col.lstrip()
check('lstrip', res[0] == "hello  ")

col = UCS2Column_SoA(); col.append("  hello  ")
res = col.rstrip()
check('rstrip', res[0] == "  hello")

col = UCS2Column_SoA(); col.append("hello world")
res = col.startswith("hello")
check('startswith', res[0] == 1)

col = UCS2Column_SoA(); col.append("hello world")
res = col.endswith("world")
check('endswith', res[0] == 1)

col = UCS2Column_SoA()
col.append("12345"); col.append("abc")
res = col.isdigit()
check('isdigit [0]', res[0] == 1)
check('isdigit [1]', res[1] == 0)

col = UCS2Column_SoA()
col.append("abc"); col.append("123")
res = col.isalpha()
check('isalpha [0]', res[0] == 1)
check('isalpha [1]', res[1] == 0)

col = UCS2Column_SoA()
col.append("abc123"); col.append("abc 123")
res = col.isalnum()
check('isalnum [0]', res[0] == 1)
check('isalnum [1]', res[1] == 0)

col = UCS2Column_SoA()
col.append("   "); col.append("abc")
res = col.isspace()
check('isspace [0]', res[0] == 1)
check('isspace [1]', res[1] == 0)

col = UCS2Column_SoA(); col.append("hello"); col.append("hi")
res = col.len()
check('len [0]', res[0] == 5)
check('len [1]', res[1] == 2)

col = UCS2Column_SoA()
col.append("a"); col.append("b"); col.append("a")
check('unique', col.unique() == ["a", "b"])

col = UCS2Column_SoA()
check('unique empty', col.unique() == [])

col = UCS2Column_SoA()
for i in range(200):
    col.append(f"str_{i}")
check('growth length', col.length == 200)
check('growth [199]', col[199] == "str_199")

col = UCS2Column_SoA(); col.append("a"); col.append("b"); col.append("c")
col._reorder([2, 0, 1])
check('reorder [0]', col[0] == "c")
check('reorder [1]', col[1] == "a")
check('reorder [2]', col[2] == "b")

print(f'\nResults: {passed} passed, {failed} failed')
if failed: sys.exit(1)
