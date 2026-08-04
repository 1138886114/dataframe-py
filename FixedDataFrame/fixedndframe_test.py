#!/usr/bin/env python3
"""
fixedndframe 完整功能测试套件
数据来源: FixedDataFrame/test_data.txt (OCR 识别 JSON, 单行)
构建: python3 -c "from setuptools import setup; from Cython.Build import cythonize; setup(ext_modules=cythonize('fixedndframe.pyx', language_level=3))" build_ext --build-lib=build
用法: python fixedndframe_test.py
"""
import sys, os, json, re, math, random, time, unicodedata
from typing import List, Tuple, Optional
from collections import Counter

test_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(test_dir, "build")
if os.path.isdir(build_dir):
    sys.path.insert(0, build_dir)

try:
    from fixedndframe import NDframe
except ImportError:
    print("找不到 fixedndframe 模块，请先编译:")
    print("  cd FixedDataFrame && python3 -c \"from setuptools import setup; from Cython.Build import cythonize; setup(ext_modules=cythonize('fixedndframe.pyx', language_level=3))\" build_ext --build-lib=build")
    sys.exit(1)

# ════════════════════════════════════════════════════════════════
# 测试基础设施
# ════════════════════════════════════════════════════════════════
PASS = FAIL = 0

def check(desc: str, actual, expected):
    global PASS, FAIL
    if actual == expected:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {desc} — expected {expected!r}, got {actual!r}")

def section(name: str):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

def subsection(name: str):
    print(f"\n--- {name}")

# ════════════════════════════════════════════════════════════════
# 测试数据加载
# ════════════════════════════════════════════════════════════════
ParsedRow = dict

def _parse_one(i: int, r: dict) -> ParsedRow:
    pos = r["position"]
    left = pos["left"]
    top = pos["top"]
    w = pos["width"]
    h = pos["height"]
    return {
        "idx": i,
        "x0": left,
        "y0": top,
        "x1": left + w,
        "y1": top + h,
        "width": w,
        "height": h,
        "page": 1,
        "flags": 0,
        "text": r.get("words", ""),
        "text_sep": "",
    }


def load_test_data() -> Tuple[NDframe, List[ParsedRow]]:
    """读取 test_data.txt (OCR JSON) 并构建 NDframe，返回 (df, rows)"""
    data_path = os.path.join(test_dir, "test_data.txt")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data["result"]
    n = len(results)
    rows = [_parse_one(i, r) for i, r in enumerate(results)]
    df = NDframe(n)
    for i, r in enumerate(results):
        pos = r["position"]
        left = pos["left"]
        top = pos["top"]
        w = pos["width"]
        h = pos["height"]
        df.append(i, left, top, left + w, top + h, w, h, 1, 0, r.get("words", ""), "")
    return df, rows


def make_small_df() -> NDframe:
    """构造小型标准测试 DF（用于需要精确控制特征值的测试）"""
    df = NDframe()
    df.append(0, 10, 20, 100, 60, 90, 40, 1, 0, "hello world 你好世界", "sep_a")
    df.append(1, 50, 80, 200, 150, 150, 70, 2, 1, "foo bar baz ", "sep_b_connector")
    df.append(2, 30, 40, 80, 90, 50, 50, 1, 0, "test 123 regex 正则测试", "sep_c")
    df.append(3, 5, 5, 30, 30, 25, 25, 3, 2, "short 快 短", "sep_d")
    df.append(4, 60, 100, 180, 200, 120, 100, 1, 0, "another one 这一个", "sep_e")
    return df


# ════════════════════════════════════════════════════════════════
# 测试套件
# ════════════════════════════════════════════════════════════════

def test_basic_properties(df: NDframe, rows: List[ParsedRow]):
    section("1. 基础构造与属性")
    check("len(df)", len(df), len(rows))
    check("df.empty", df.empty, False)
    check("bool(df)", bool(df), True)

    empty_df = NDframe()
    check("empty df.empty", empty_df.empty, True)
    check("bool(empty)", bool(empty_df), False)

    check("df[0]['x0']", df[0]["x0"], rows[0]["x0"])
    check("df[0]['y0']", df[0]["y0"], rows[0]["y0"])
    check("df[0]['x1']", df[0]["x1"], rows[0]["x1"])
    check("df[0]['y1']", df[0]["y1"], rows[0]["y1"])
    check("df[0]['idx']", df[0]["idx"], rows[0]["idx"])
    check("df[0]['width']", df[0]["width"], rows[0]["width"])
    check("df[0]['height']", df[0]["height"], rows[0]["height"])
    check("df[0]['page']", df[0]["page"], rows[0]["page"])
    check("df[0]['flags']", df[0]["flags"], rows[0]["flags"])
    check("df[0]['text']", df[0]["text"], rows[0]["text"])
    check("df[0]['text_sep']", df[0]["text_sep"], rows[0]["text_sep"])

    df_first = df[0]
    check("df[0] dict keys", set(df_first.keys()), {"idx", "x0", "y0", "x1", "y1", "width", "height", "page", "flags", "text", "text_sep"})

    # Negative index
    check("df[-1] idx", df[-1]["idx"], rows[-1]["idx"])
    check("df[-2] text", df[-2]["text"], rows[-2]["text"])

    # Slice
    sliced = df[1:5]
    check("df[1:5] len", len(sliced), 4)
    check("df[1:5][0] text", sliced[0]["text"], rows[1]["text"])
    check("df[1:5][0] x0", sliced[0]["x0"], rows[1]["x0"])

    nd = df[3:3]
    check("df[3:3] empty", len(nd), 0)

    # 越界 / 非法索引
    try:
        df[len(rows)]
        check("df[len] raises IndexError", False, True)
    except IndexError:
        check("df[len] raises IndexError", True, True)

    try:
        df["bad"]
        check("df['bad'] raises TypeError", False, True)
    except TypeError:
        check("df['bad'] raises TypeError", True, True)


def test_to_list_string(df: NDframe, rows: List[ParsedRow]):
    section("2. to_list / to_string / print / data / iter")
    lst = df.to_list()
    check("to_list len", len(lst), len(rows))
    check("to_list[0]['text']", lst[0]["text"], rows[0]["text"])
    check("to_list[0]['x0']", lst[0]["x0"], rows[0]["x0"])

    s = df.to_string()
    check("to_string produces output", len(s) > 100, True)
    check("to_string contains row count", str(len(rows)) in s, True)
    check("to_string contains idx header", "idx" in s, True)

    s2 = df.to_string(max_rows=10, name="TEST")
    check("to_string(max_rows, name)", len(s2) > 0, True)

    rep = repr(df)
    check("__repr__", rep, f"NDframe({len(rows)} rows)")
    check("__str__", str(df), f"<NDframe: {len(rows)} rows>")

    html = df._repr_html_()
    check("_repr_html_ has <table>", "<table" in html, True)

    check("df.data len", len(df.data), len(rows))
    check("df.data[0]['text']", df.data[0]["text"], rows[0]["text"])

    it = list(df)
    check("list(df) len", len(it), len(rows))
    check("list(df)[0]['text']", it[0]["text"], rows[0]["text"])


# ──────────────────────────────────────────────────────────────
# NumericAccessor 测试组
# ──────────────────────────────────────────────────────────────

def test_numeric_aggregation(df: NDframe, rows: List[ParsedRow]):
    section("3. NumericAccessor 聚合")
    fields = ["x0", "y0", "x1", "y1", "width", "height", "page"]
    for f in fields:
        vals = [r[f] for r in rows]
        acc = getattr(df, f)
        check(f"{f}.max", acc.max(), max(vals))
        check(f"{f}.min", acc.min(), min(vals))
        check(f"{f}.mean", round(acc.mean(), 6), round(sum(vals) / len(vals), 6))

    check("idx.max", df.idx.max(), rows[-1]["idx"])
    check("flags.min", df.flags.min(), min(r["flags"] for r in rows))

    # field_name
    check("x0.field_name", df.x0.field_name, "x0")
    check("y0.field_name", df.y0.field_name, "y0")
    check("idx.field_name", df.idx.field_name, "idx")
    check("height.field_name", df.height.field_name, "height")
    check("repr(x0)", repr(df.x0).startswith("<Numeric x0:"), True)
    check("repr(height)", repr(df.height).startswith("<Numeric height:"), True)

    # iter
    check("list(x0)", list(df.x0), df.x0.values())


def test_numeric_filter(df: NDframe, rows: List[ParsedRow]):
    section("4. NumericAccessor 过滤")
    x0_vals = [r["x0"] for r in rows]
    y0_vals = [r["y0"] for r in rows]
    w_vals = [r["width"] for r in rows]

    gt40 = sum(1 for v in x0_vals if v > 40)
    check("x0.greater(40)", len(df.x0.greater(40)), gt40)

    lt40 = sum(1 for v in x0_vals if v < 40)
    check("x0.less(40)", len(df.x0.less(40)), lt40)

    mid_y0 = sum(1 for v in y0_vals if 50 < v < 800)
    check("y0.between(50,800)", len(df.y0.between(50, 800)), mid_y0)

    eq100 = sum(1 for v in w_vals if v == 100)
    check("width.greater(99)", len(df.width.greater(99)), sum(1 for v in w_vals if v > 99))

    # values
    check("x0.values == x0 list", list(df.x0.values()), x0_vals)
    check("x0.values(0)", df.x0.values(0), x0_vals[0])
    check("x0.values(-1)", df.x0.values(-1), x0_vals[-1])
    check("idx.values(5)", df.idx.values(5), rows[5]["idx"])

    # 空 DF 过滤
    empty_df = NDframe()
    check("empty greater", len(empty_df.x0.greater(10)), 0)


def test_numeric_arithmetic(df: NDframe, rows: List[ParsedRow]):
    section("5. NumericAccessor 运算")
    x0_vals = [r["x0"] for r in rows]

    df_a = df.copy()
    df_a.x0.add(5)
    check("x0.add(5)", list(df_a.x0.values()), [v + 5 for v in x0_vals])

    df_s = df.copy()
    df_s.x0.sub(3)
    check("x0.sub(3)", list(df_s.x0.values()), [max(v - 3, 0) for v in x0_vals])

    # diff
    raw_diff = [None] + [x0_vals[i] - x0_vals[i-1] for i in range(1, len(x0_vals))]
    check("x0.diff", df.x0.diff(), raw_diff)

    # var / std (C 均值按 double 累积，与 Python 精确整数均值存在末位 ULP 差，用相对容差)
    mean_x = sum(x0_vals) / len(x0_vals)
    var1 = sum((v - mean_x) ** 2 for v in x0_vals) / (len(x0_vals) - 1)
    var0 = sum((v - mean_x) ** 2 for v in x0_vals) / len(x0_vals)
    check("x0.var(ddof=1)", math.isclose(df.x0.var(1), var1, rel_tol=1e-9), True)
    check("x0.var(ddof=0)", math.isclose(df.x0.var(0), var0, rel_tol=1e-9), True)
    check("x0.std(ddof=1)", math.isclose(df.x0.std(1), math.sqrt(var1), rel_tol=1e-9), True)
    check("x0.std(ddof=0)", math.isclose(df.x0.std(0), math.sqrt(var0), rel_tol=1e-9), True)

    # mode
    c = Counter(x0_vals)
    mode_val, mode_cnt = min((v for v in c.items()), key=lambda x: (-x[1], x[0]))
    check("x0.mode", df.x0.mode(), (mode_val, mode_cnt))

    # empty df stats
    empty_df = NDframe()
    check("empty var", empty_df.x0.var(), 0.0)
    check("empty std", empty_df.x0.std(), 0.0)
    check("empty mean", empty_df.x0.mean(), 0.0)
    check("empty diff", empty_df.x0.diff(), [])
    check("empty mode", empty_df.x0.mode(), (0, 0))


def test_numeric_sort(df: NDframe, rows: List[ParsedRow]):
    section("6. NumericAccessor 排序")
    x0_vals = [r["x0"] for r in rows]
    y0_vals = [r["y0"] for r in rows]
    width_vals = [r["width"] for r in rows]
    x1_vals = [r["x1"] for r in rows]
    height_vals = [r["height"] for r in rows]

    df_a = df.copy()
    df_a.x0.sort()
    check("x0.sort asc", list(df_a.x0.values()), sorted(x0_vals))

    df_b = df.copy()
    df_b.y0.sort(descending=True)
    check("y0.sort desc", list(df_b.y0.values()), sorted(y0_vals, reverse=True))

    df_c = df.copy()
    df_c.width.sort()
    check("width.sort asc", list(df_c.width.values()), sorted(width_vals))

    # sort_by_x0 convenience
    df_d = df.copy()
    df_d.sort_by_x0(descending=True)
    check("sort_by_x0 desc", list(df_d.x0.values()), sorted(x0_vals, reverse=True))

    df_e = df.copy()
    df_e.sort_by_y0()
    check("sort_by_y0 asc", list(df_e.y0.values()), sorted(y0_vals))

    df_f = df.copy()
    df_f.sort_by("x1")
    check("sort_by('x1')", list(df_f.x1.values()), sorted(x1_vals))

    df_g = df.copy()
    df_g.sort_by("height", descending=True)
    check("sort_by('height' desc)", list(df_g.height.values()), sorted(height_vals, reverse=True))

    df_h = df.copy()
    df_h.sort_by_idx(descending=True)
    check("sort_by_idx desc", list(df_h.idx.values()), sorted([r["idx"] for r in rows], reverse=True))

    df_i = df.copy()
    df_i.sort_by_page()
    df_j = df.copy()
    df_j.sort_by_flags()
    df_k = df.copy()
    df_k.sort_by_width()
    df_l = df.copy()
    df_l.sort_by_y1()
    check("sort_by_page/sort_by_flags/sort_by_width/sort_by_y1 len", len(df_i) == len(df_j) == len(df_k) == len(df_l) == len(rows), True)

    # chained sort
    df_m = df.copy()
    df_m.x0.sort().y0.sort()
    check("chained sort len", len(df_m), len(rows))


def test_numeric_idx(df: NDframe, rows: List[ParsedRow]):
    section("7. NumericAccessor idx_* 索引")
    x0_vals = [r["x0"] for r in rows]
    y0_vals = [r["y0"] for r in rows]

    expected_gt = [i for i, v in enumerate(x0_vals) if v > 40]
    expected_lt = [i for i, v in enumerate(x0_vals) if v < 40]
    expected_eq = [i for i, v in enumerate(x0_vals) if v == 30]

    check("idx_greater(40)", df.x0.idx_greater(40), expected_gt)
    check("idx_less(40)", df.x0.idx_less(40), expected_lt)
    check("idx_equal(30)", df.x0.idx_equal(30), expected_eq)

    expected_btw = [i for i, v in enumerate(y0_vals) if 50 < v < 800]
    check("y0.idx_between(50,800)", df.y0.idx_between(50, 800), expected_btw)


# ──────────────────────────────────────────────────────────────
# StringAccessor 测试组
# ──────────────────────────────────────────────────────────────

def test_string_basic(df: NDframe, rows: List[ParsedRow]):
    section("8. StringAccessor 基础")
    texts = [r["text"] for r in rows]

    check("text.field_name", df.text.field_name, "text")
    check("text_sep.field_name", df.text_sep.field_name, "text_sep")
    check("text.join", df.text.join("|"), "|".join(texts))
    check("text.len", df.text.len(), [len(t) for t in texts])
    check("text.values", df.text.values(), texts)
    check("text.values(0)", df.text.values(0), texts[0])
    check("text.values(-1)", df.text.values(-1), texts[-1])

    check("list(text) == text.values()", list(df.text), df.text.values())
    check("repr(text) starts with 'text['", repr(df.text).startswith("text["), True)
    check("repr(text_sep) starts with 'text_sep['", repr(df.text_sep).startswith("text_sep["), True)

    # slice / count
    sliced = df.text.slice(0, 5)
    check("text.slice(0,5)[0]", sliced[0], texts[0][0:5])
    check("text.slice(-3) last", df.text.slice(-3)[-1], texts[-1][-3:])
    total_e = sum(t.count("e") for t in texts)
    check("text.count('e')", df.text.count("e"), total_e)
    check("text.count('xyz')", df.text.count("xyz"), 0)
    check("text.count('')", df.text.count(""), sum(len(t) + 1 for t in texts))


def test_string_transform(df: NDframe, rows: List[ParsedRow]):
    section("9. StringAccessor 变换")
    texts = [r["text"] for r in rows]

    # replace（双替换参数）
    df_r = df.copy()
    df_r.text.replace(" ", "_")
    check("text.replace(' ', '_')",
          df_r.text.join("|"),
          "|".join(t.replace(" ", "_") for t in texts))

    df_r2 = df.copy()
    df_r2.text.replace("a", "X", "e", "Y")
    expected2 = "".join(t.replace("a", "X").replace("e", "Y") for t in texts)
    check("text.replace('a','X','e','Y')", df_r2.text.join("|"), "|".join(
        t.replace("a", "X").replace("e", "Y") for t in texts))

    # strip / rstrip / lstrip
    df_st = df.copy()
    df_st.text.strip()
    check("text.strip[0]", df_st.text.values(0), texts[0].strip())

    df_rst = df.copy()
    df_rst.text.rstrip()
    check("text.rstrip[0]", df_rst.text.values(0), texts[0].rstrip())

    df_lst = df.copy()
    df_lst.text.lstrip()
    check("text.lstrip[0]", df_lst.text.values(0), texts[0].lstrip())

    df_st2 = df.copy()
    df_st2.text.strip("0123456789")
    check("text.strip(chars)[0]", df_st2.text.values(0), texts[0].strip("0123456789"))

    # capitalize
    df_cap = df.copy()
    df_cap.text.capitalize()
    check("text.capitalize[0]", df_cap.text.values(0), texts[0].capitalize())

    # swapcase
    df_swp = df.copy()
    df_swp.text.swapcase()
    check("text.swapcase", df_swp.text.values(0), texts[0].swapcase())


def test_string_is(df: NDframe, rows: List[ParsedRow]):
    section("10. StringAccessor 字符类型判断")
    texts = [r["text"] for r in rows]

    check("text.isalnum", len(df.text.isalnum()), sum(1 for t in texts if t and all(c.isalnum() for c in t)))
    check("text.isalpha", len(df.text.isalpha()), sum(1 for t in texts if t and all(c.isalpha() for c in t)))
    check("text.isdigit", len(df.text.isdigit()), sum(1 for t in texts if t and all(c.isdigit() for c in t)))
    check("text.isspace", len(df.text.isspace()), sum(1 for t in texts if t and all(c.isspace() for c in t)))
    # 实现为逐字符严格判断（数字/空格等非字母字符不参与），与 Python str.islower 的“存在小写字母”语义不同
    check("text.islower", len(df.text.islower()), sum(1 for t in texts if t and all(c.islower() for c in t)))
    check("text.isupper", len(df.text.isupper()), sum(1 for t in texts if t and all(c.isupper() for c in t)))
    # istitle 实现为「每字符 ISLOWER 或标题字母（Unicode 类别 Lt）」（允许全小写）
    check("text.istitle", len(df.text.istitle()), sum(1 for t in texts if t and all(c.islower() or unicodedata.category(c) == 'Lt' for c in t)))
    check("text.isnumeric", len(df.text.isnumeric()), sum(1 for t in texts if t and t.isnumeric()))
    check("text.isdecimal", len(df.text.isdecimal()), sum(1 for t in texts if t and t.isdecimal()))

    check("text.is_chinese", len(df.text.is_chinese()), sum(1 for t in texts if t and all('\u4e00' <= c <= '\u9fff' for c in t)))
    # is_numeric_punct 实现为「每字符是数字或 . 或 ,」
    check("text.is_numeric_punct", len(df.text.is_numeric_punct()), sum(1 for t in texts if t and all(c.isdigit() or c in ".," for c in t)))

    # keep_chinese / keep_numeric_punct
    df_kc = df.copy()
    df_kc.text.keep_chinese()
    expected_kc = ["".join(c for c in t if '\u4e00' <= c <= '\u9fff') for t in texts]
    check("keep_chinese", df_kc.text.values(), expected_kc)

    df_knp = df.copy()
    df_knp.text.keep_numeric_punct()
    # keep_numeric_punct 实现为逐字符保留 [0-9.,]，其余一律剔除
    expected_knp = ["".join(c for c in t if c.isdigit() or c in ".,") for t in texts]
    check("keep_numeric_punct", df_knp.text.values(), expected_knp)


def test_string_starts_ends(df: NDframe, rows: List[ParsedRow]):
    section("11. StringAccessor startswith / endswith")
    texts = [r["text"] for r in rows]

    check("text.startswith('收')", len(df.text.startswith("收")), sum(1 for t in texts if t.startswith("收")))
    check("text.startswith('nonexist')", len(df.text.startswith("nonexist")), 0)
    check("text.endswith('.')", len(df.text.endswith(".")), sum(1 for t in texts if t.endswith(".")))

    # 空前缀/空后缀：非空行全部匹配
    non_empty = sum(1 for t in texts if t)
    check("text.startswith('')", len(df.text.startswith("")), non_empty)
    check("text.endswith('')", len(df.text.endswith("")), non_empty)
    check("text_sep.startswith('')", len(df.text_sep.startswith("")), len(rows))


def test_string_regex(df: NDframe, rows: List[ParsedRow]):
    section("12. StringAccessor 正则")
    texts = [r["text"] for r in rows]

    r_数字 = re.compile(r"\d+")
    check("contains(数字)", len(df.text.contains(r_数字)), sum(1 for t in texts if r_数字.search(t)))

    r_收 = re.compile(r"收")
    check("contains(收)", len(df.text.contains(r_收)), sum(1 for t in texts if r_收.search(t)))

    # 空 pattern 正则路径（与旧语义一致：空串行不匹配）
    check("contains('') regex", len(df.text.contains("")), sum(1 for t in texts if t))

    # 传入字符串也能走正则
    check("contains('Macau')", len(df.text.contains("Macau")), sum(1 for t in texts if re.search("Macau", t)))

    # regex_replace
    df_rr = df.copy()
    df_rr.text.regex_replace(re.compile(r"\d+"), "X")
    check("regex_replace 数字→X[0]", df_rr.text.values(0), re.sub(r"\d+", "X", texts[0]))

    # regex_extract — 返回仅含匹配行的新 NDframe，text 替换为第一个捕获组
    r_extract = re.compile(r"(\d+)")
    df_re = df.copy()
    res = df_re.text.regex_extract(r_extract)
    expected_count = sum(1 for t in texts if r_extract.search(t))
    check("regex_extract len", len(res), expected_count)
    expected_first = r_extract.search(next(t for t in texts if r_extract.search(t))).group(1)
    check("regex_extract first value", res.text.values(0), expected_first)


def test_string_contains_pure(df: NDframe, rows: List[ParsedRow]):
    section("13. 纯 C 字符串匹配 (contains_substr)")
    texts = [r["text"] for r in rows]

    # contains_substr 纯路径：非空子串
    for sub in ["收", "Macau", "123", "nonexistent"]:
        expect = sum(1 for t in texts if sub in t)
        got = len(df.text.contains_substr(sub))
        check(f"contains_substr({sub!r})", got, expect)

    # 与正则路径结果一致性
    for sub in ["收", "Macau", "2"]:
        pure = set(df.text.contains_substr(sub).idx.values())
        reg = set(df.text.contains(sub).idx.values())
        check(f"contains_substr==contains idx set for {sub!r}", pure, reg)

    # contains_in_bbox（正则引擎）
    sub = "收"
    expect = sum(1 for i, r in enumerate(rows) if sub in r["text"] and r["x0"] > 100 and r["y0"] > 0 and r["x1"] < 1500 and r["y1"] < 800)
    got = len(df.text.contains_in_bbox(sub, 100, 0, 1500, 800))
    check("contains_in_bbox(收, 100,0,1500,800)", got, expect)

    # regex_strlen_*（正则引擎）
    lengths = [len(t) for t in texts]
    for n in (5, 10, 15):
        check(f"regex_strlen_greater('e', {n})",
              len(df.text.regex_strlen_greater("e", n)),
              sum(1 for i, l in enumerate(lengths) if l > n and "e" in texts[i]))
        check(f"regex_strlen_less('a', {n})",
              len(df.text.regex_strlen_less("a", n)),
              sum(1 for i, l in enumerate(lengths) if l < n and "a" in texts[i]))
        check(f"regex_strlen_equal('2', {n})",
              len(df.text.regex_strlen_equal("2", n)),
              sum(1 for i, l in enumerate(lengths) if l == n and "2" in texts[i]))

    # idx_* 索引路径
    check("idx_regex_strlen_greater('e', 5)",
          df.text.idx_regex_strlen_greater("e", 5),
          [i for i, l in enumerate(lengths) if l > 5 and "e" in texts[i]])
    check("idx_regex_strlen_equal('2', 15)",
          df.text.idx_regex_strlen_equal("2", 15),
          [i for i, l in enumerate(lengths) if l == 15 and "2" in texts[i]])


def test_string_len_filter(df: NDframe, rows: List[ParsedRow]):
    section("14. StringAccessor 长度过滤")
    texts = [r["text"] for r in rows]
    lengths = [len(t) for t in texts]

    check("strlen_greater(10)", len(df.text.strlen_greater(10)), sum(1 for l in lengths if l > 10))
    check("strlen_less(10)", len(df.text.strlen_less(10)), sum(1 for l in lengths if l < 10))
    check("strlen_equal(15)", len(df.text.strlen_equal(15)), sum(1 for l in lengths if l == 15))

    # regex + length combo
    r_all = re.compile(r".")
    check("regex_strlen_greater(10)", len(df.text.regex_strlen_greater(r_all, 10)), sum(1 for l in lengths if l > 10))
    check("regex_strlen_less(10)", len(df.text.regex_strlen_less(r_all, 10)), sum(1 for l in lengths if l < 10))
    check("regex_strlen_equal(15)", len(df.text.regex_strlen_equal(r_all, 15)), sum(1 for l in lengths if l == 15))

    r_num = re.compile(r"\d+")
    expected = sum(1 for i, t in enumerate(texts) if len(t) > 10 and r_num.search(t))
    check("regex_strlen_greater(10) with 数字", len(df.text.regex_strlen_greater(r_num, 10)), expected)


def test_string_set_ops(df: NDframe, rows: List[ParsedRow]):
    section("15. StringAccessor 集合筛选")
    texts = [r["text"] for r in rows]
    target_set = ["收", "据", "姓名", "nonexistent"]
    check("is_in", len(df.text.is_in(target_set)), sum(1 for t in texts if t in target_set))
    check("not_in", len(df.text.not_in(["收", "据"])), sum(1 for t in texts if t not in ["收", "据"]))

    check("contains_any True", df.text.contains_any(["收据", "nope"]), any(any(sub in t for sub in ["收据", "nope"]) for t in texts))
    check("contains_any False", df.text.contains_any(["ZZZZNONEXIST"]), False)


def test_string_equal_substr(df: NDframe, rows: List[ParsedRow]):
    section("16b. StringAccessor is_equal / contains_substr")
    texts = [r["text"] for r in rows]
    t0 = texts[0]
    last = texts[-1]

    # ── is_equal: 全等过滤 ──
    check("text.is_equal(first) len",
          len(df.text.is_equal(t0)), sum(1 for t in texts if t == t0))
    check("text.is_equal(last) len",
          len(df.text.is_equal(last)), sum(1 for t in texts if t == last))
    check("text.is_equal(nonexistent)", len(df.text.is_equal("ZZZZ_不存在_ZZZZ")), 0)
    check("text.is_equal('')", len(df.text.is_equal("")), sum(1 for t in texts if t == ""))
    check("text.is_equal('') idx",
          sorted(df.text.is_equal("").idx.values()),
          sorted(i for i, t in enumerate(texts) if t == ""))

    # 与 is_in 单元素语义一致
    check("is_equal idx == is_in 单元素 idx",
          set(df.text.is_equal(t0).idx.values()),
          set(df.text.is_in([t0]).idx.values()))

    # 空字段 text_sep 全为 ""
    check("text_sep.is_equal('') len", len(df.text_sep.is_equal("")), len(rows))
    check("text_sep.is_equal('x')", len(df.text_sep.is_equal("x")), 0)

    # 空 DF
    empty_df = NDframe()
    check("empty is_equal", len(empty_df.text.is_equal("a")), 0)

    # ── contains_substr: 子串包含过滤（纯 C 引擎）──
    for sub in ["收", "Macau", "123", "nonexistent"]:
        expect = sum(1 for t in texts if sub in t)
        check(f"contains_substr({sub!r})", len(df.text.contains_substr(sub)), expect)
        check(f"contains_substr==contains {sub!r}",
              set(df.text.contains_substr(sub).idx.values()),
              set(df.text.contains(sub).idx.values()))

    # 空子串：非空行全部匹配
    non_empty = sum(1 for t in texts if t)
    check("contains_substr('')", len(df.text.contains_substr("")), non_empty)

    # 多行命中时保持原行序
    r_m = re.compile(r"Macau")
    macau_idx = [i for i, t in enumerate(texts) if r_m.search(t)]
    if macau_idx:
        check("contains_substr keeps order",
              df.text.contains_substr("Macau").idx.values(), macau_idx)

    # 空 DF
    check("empty contains_substr", len(empty_df.text.contains_substr("a")), 0)


def test_string_idx_ops(df: NDframe, rows: List[ParsedRow]):
    section("16. StringAccessor idx_* 索引")
    texts = [r["text"] for r in rows]
    r_macau = re.compile(r"Macau")

    check("idx_contains(Macau)", df.text.idx_contains(r_macau), [i for i, t in enumerate(texts) if r_macau.search(t)])

    # strlen idx
    lengths = [len(t) for t in texts]
    check("idx_strlen_greater(10)", df.text.idx_strlen_greater(10), [i for i, l in enumerate(lengths) if l > 10])
    check("idx_strlen_less(10)", df.text.idx_strlen_less(10), [i for i, l in enumerate(lengths) if l < 10])
    check("idx_strlen_equal(15)", df.text.idx_strlen_equal(15), [i for i, l in enumerate(lengths) if l == 15])

    # regex + strlen idx combo
    r_all = re.compile(r".")
    check("idx_regex_strlen_greater(10)", df.text.idx_regex_strlen_greater(r_all, 10), [i for i, l in enumerate(lengths) if l > 10])
    check("idx_regex_strlen_less(10)", df.text.idx_regex_strlen_less(r_all, 10), [i for i, l in enumerate(lengths) if l < 10])
    check("idx_regex_strlen_equal(15)", df.text.idx_regex_strlen_equal(r_all, 15), [i for i, l in enumerate(lengths) if l == 15])

    # idx_contains_in_bbox
    r_M = re.compile(r"M")
    expected_bbox = [i for i, t in enumerate(texts) if r_M.search(t) and rows[i]["x0"] > 0 and rows[i]["y0"] > 0 and rows[i]["x1"] < 2000 and rows[i]["y1"] < 2000]
    check("idx_contains_in_bbox M", df.text.idx_contains_in_bbox(r_M, 0, 0, 2000, 2000), expected_bbox)


def test_string_drop_regex(df: NDframe, rows: List[ParsedRow]):
    section("17. StringAccessor drop_regex")
    texts = [r["text"] for r in rows]
    r_hello = re.compile(r"hello")
    r_nope = re.compile(r"ZZZZNONEXIST")

    df_dr = df.copy()
    df_dr.text.drop_regex(r_hello)
    expected_count = sum(1 for t in texts if not r_hello.search(t))
    check("drop_regex(hello) len", len(df_dr), expected_count)

    df_dr2 = df.copy()
    df_dr2.text.drop_regex(r_nope)
    check("drop_regex(no match) len", len(df_dr2), len(rows))


def test_cross_replace(df: NDframe, rows: List[ParsedRow]):
    section("18. 跨字段替换")
    texts = [r["text"] for r in rows]

    # replace_to
    df_cr = df.copy()
    df_cr.text.replace_to(" ", "_", "text_sep")
    expected_seps = [t.replace(" ", "_") for t in texts]
    check("replace_to text→text_sep", [df_cr.text_sep.values(i) for i in range(len(rows))], expected_seps)
    check("replace_to text unchanged", list(df_cr.text.values()), texts)

    # regex_replace_to
    df_cr2 = df.copy()
    df_cr2.text.regex_replace_to(re.compile(r"\d+"), "[X]", "text_sep")
    expected = [re.sub(r"\d+", "[X]", t) for t in texts]
    check("regex_replace_to text→text_sep", list(df_cr2.text_sep.values()), expected)


# ──────────────────────────────────────────────────────────────
# 空间过滤 / 索引
# ──────────────────────────────────────────────────────────────

def test_spatial_filter(df: NDframe, rows: List[ParsedRow]):
    section("19. 坐标过滤 filter_in_*")
    in_bbox = [r for r in rows if r["x0"] > 500 and r["y0"] > 200 and r["x1"] < 1200 and r["y1"] < 1000]
    check("filter_in_bbox(500,200,1200,1000)", len(df.filter_in_bbox(500, 200, 1200, 1000)), len(in_bbox))

    in_x = [r for r in rows if r["x0"] > 500 and r["x1"] < 1200]
    check("filter_in_x0x1(500,1200)", len(df.filter_in_x0x1(500, 1200)), len(in_x))

    in_y = [r for r in rows if r["y0"] > 200 and r["y1"] < 800]
    check("filter_in_y0y1(200,800)", len(df.filter_in_y0y1(200, 800)), len(in_y))

    gp = [r for r in rows if r["x0"] > 500 and r["y0"] > 200]
    check("filter_greater_point(500,200)", len(df.filter_greater_point(500, 200)), len(gp))

    lp = [r for r in rows if r["x1"] < 1200 and r["y1"] < 1000]
    check("filter_less_point(1200,1000)", len(df.filter_less_point(1200, 1000)), len(lp))

    # Negative clamp
    result = df.filter_in_bbox(-1, -5, 2000, 2000)
    check("filter_in_bbox neg clamp runs without error", len(result) > 0, True)

    # Invalid boundary raises
    try:
        df.filter_in_bbox(20, 0, 10, 10)
        check("filter_in_bbox x0>x1 raises", False, True)
    except ValueError:
        check("filter_in_bbox x0>x1 raises", True, True)


def test_spatial_idx(df: NDframe, rows: List[ParsedRow]):
    section("20. 坐标索引 idx_in_*")
    expected = [i for i, r in enumerate(rows) if r["x0"] > 500 and r["y0"] > 200 and r["x1"] < 1200 and r["y1"] < 1000]
    check("idx_in_bbox(500,200,1200,1000)", df.idx_in_bbox(500, 200, 1200, 1000), expected)

    expected_x = [i for i, r in enumerate(rows) if r["x0"] > 500 and r["x1"] < 1200]
    check("idx_in_x0x1(500,1200)", df.idx_in_x0x1(500, 1200), expected_x)

    expected_y = [i for i, r in enumerate(rows) if r["y0"] > 200 and r["y1"] < 800]
    check("idx_in_y0y1(200,800)", df.idx_in_y0y1(200, 800), expected_y)

    try:
        df.idx_in_bbox(20, 0, 10, 10)
        check("idx_in_bbox x0>x1 raises", False, True)
    except ValueError:
        check("idx_in_bbox x0>x1 raises", True, True)


# ──────────────────────────────────────────────────────────────
# 删除 / 合并 / df_from_idx
# ──────────────────────────────────────────────────────────────

def test_drop_operations(df: NDframe, rows: List[ParsedRow]):
    section("21. 删除操作 drop / drop_in_bbox / df_drop / df_drops / drop_by_idx")

    # drop (in-place)
    df_d1 = df.copy()
    df_d1.drop(0)
    check("drop(0) len", len(df_d1), len(rows) - 1)
    check("drop(0) first text", df_d1[0]["text"], rows[1]["text"])

    # df_drop (returns new)
    df_dd = df.copy()
    df_dd_new = df_dd.df_drop(0)
    check("df_drop len", len(df_dd_new), len(rows) - 1)
    check("df_drop original unchanged", len(df_dd), len(rows))

    # df_drops (returns new)
    df_dds = df.copy()
    df_dds_new = df_dds.df_drops([0, 2, 4])
    expected_cnt = len(rows) - 3
    check("df_drops len", len(df_dds_new), expected_cnt)
    check("df_drops[0] text", df_dds_new[0]["text"], rows[1]["text"])

    # drop_by_idx (in-place)
    df_dbi = df.copy()
    df_dbi.drop_by_idx([1, 3, 5])
    expected = [i for i in range(len(rows)) if i not in (1, 3, 5)]
    check("drop_by_idx len", len(df_dbi), len(expected))
    if df_dbi:
        check("drop_by_idx[0] text", df_dbi[0]["text"], rows[expected[0]]["text"])
        check("drop_by_idx[1] text", df_dbi[1]["text"], rows[expected[1]]["text"])

    # drop_by_idx 空集合
    df_dbe = df.copy()
    df_dbe.drop_by_idx([])
    check("drop_by_idx([]) no-op", len(df_dbe), len(rows))

    # df_from_idx
    df_fi = df.df_from_idx([0, 3, 7])
    check("df_from_idx len", len(df_fi), 3)
    check("df_from_idx[0] text", df_fi[0]["text"], rows[0]["text"])
    check("df_from_idx[1] text", df_fi[1]["text"], rows[3]["text"])
    check("df_from_idx[2] text", df_fi[2]["text"], rows[7]["text"])
    check("df_from_idx empty", len(df.df_from_idx([])), 0)
    df_fid = df.df_from_idx([0, 0, 1])
    check("df_from_idx dup", len(df_fid), 3)
    check("df_from_idx dup[1] text", df_fid[1]["text"], rows[0]["text"])

    # pop_back
    df_pb = df.copy()
    df_pb.pop_back()
    check("pop_back len", len(df_pb), len(rows) - 1)
    check("pop_back[-1] text", df_pb[-1]["text"], rows[-2]["text"])
    empty_pb = NDframe()
    empty_pb.pop_back()
    check("pop_back empty no crash", len(empty_pb), 0)


def test_copy_head_tail(df: NDframe, rows: List[ParsedRow]):
    section("22. copy / head / tail")
    check("head(3) len", len(df.head(3)), 3)
    check("tail(3) len", len(df.tail(3)), 3)
    head1 = df.head(1)
    check("head(1)[0] text", head1[0]["text"], rows[0]["text"])
    tail1 = df.tail(1)
    check("tail(1)[0] text", tail1[0]["text"], rows[-1]["text"])
    check("head() default 5", len(df.head()), 5)

    df_copy = df.copy()
    df_copy.x0.add(1000)
    check("copy independent", list(df.x0.values()) != list(df_copy.x0.values()), True)
    check("copy text identical", df_copy.text.values(), [r["text"] for r in rows])


def test_merge_bbox(df: NDframe, rows: List[ParsedRow]):
    section("23. bbox / centroid / centroid_line / merge_line")
    r0 = rows[0]
    check("bbox(0)", df.bbox(0), (r0["x0"], r0["y0"], r0["x1"], r0["y1"]))
    check("bbox(-1)", df.bbox(-1), (rows[-1]["x0"], rows[-1]["y0"], rows[-1]["x1"], rows[-1]["y1"]))

    # centroid
    cx = sum((r["x0"] + r["x1"]) * 0.5 for r in rows) / len(rows)
    cy = sum((r["y0"] + r["y1"]) * 0.5 for r in rows) / len(rows)
    cxc, cyc = df.centroid()
    check("centroid x", abs(cxc - cx) < 1e-10, True)
    check("centroid y", abs(cyc - cy) < 1e-10, True)

    # centroid_line
    check("centroid_line(0)", df.centroid_line(0), ((r0["x0"] + r0["x1"]) * 0.5, (r0["y0"] + r0["y1"]) * 0.5))

    empty_df = NDframe()
    check("centroid empty", empty_df.centroid(), (0.0, 0.0))

    # merge_line (small test)
    df_ml = NDframe()
    df_ml.append(0, 10, 10, 50, 40, 40, 30, 1, 0, "first", "")
    df_ml.append(1, 52, 12, 100, 42, 48, 30, 1, 0, "second", "")
    df_ml.append(2, 200, 200, 300, 250, 100, 50, 1, 0, "far", "")
    merged = df_ml.merge_line(10, 10, " ")
    check("merge_line len", len(merged), 2)
    check("merge_line[0] text", merged[0]["text"], "first second")


def test_misc_operations(df: NDframe, rows: List[ParsedRow]):
    section("24. reset_index / add_y0y1 / concat / from_dict")

    # reset_index
    df_ri = df.copy()
    df_ri.sort_by_x0(descending=True)
    df_ri.reset_index()
    check("reset_index[0] idx", df_ri[0]["idx"], 0)
    check("reset_index[-1] idx", df_ri[-1]["idx"], len(rows) - 1)

    # add_y0y1
    df_ay = df.copy()
    orig_y0 = list(df_ay.y0.values())
    df_ay.add_y0y1(10)
    check("add_y0y1 y0", list(df_ay.y0.values()), [v + 10 for v in orig_y0])
    check("add_y0y1 y1", list(df_ay.y1.values()), [v + 10 for v in [r["y1"] for r in rows]])

    # concat
    df_cat = NDframe.concat(df, df.head(2))
    check("concat len", len(df_cat), len(rows) + 2)
    check("concat[0] text", df_cat[0]["text"], df[0]["text"])
    check("concat[-1] text", df_cat[-1]["text"], df.head(2)[-1]["text"])

    # from_dict
    fd = NDframe.from_dict({
        "text": ["A", "B", "C"],
        "x0": [1, 2, 3],
        "x1": [1, 2, 3],
        "y0": [4, 5, 6],
        "y1": [4, 5, 6],
        "width": [4, 5, 6],
        "height": [4, 5, 6],
    })
    check("from_dict len", len(fd), 3)
    check("from_dict text[0]", fd.text.values(0), "A")
    check("from_dict x0", list(fd.x0.values()), [1, 2, 3])
    check("from_dict idx auto", list(fd.idx.values()), [0, 1, 2])

    fd2 = NDframe.from_dict({
        "text": ["A", "B"],
        "x0": [1, 2], "x1": [1, 2], "y0": [1, 2], "y1": [1, 2],
        "width": [1, 2], "height": [1, 2],
        "page": [7, 8], "flags": [1, 2], "text_sep": ["s1", "s2"],
    })
    check("from_dict page", list(fd2.page.values()), [7, 8])
    check("from_dict text_sep", list(fd2.text_sep.values()), ["s1", "s2"])

    try:
        NDframe.from_dict({"x0": [1]})
        check("from_dict missing key raises", False, True)
    except ValueError:
        check("from_dict missing key raises", True, True)


def test_conditional_ops(df: NDframe, rows: List[ParsedRow]):
    section("25. 条件运算 apply_if_*")

    df_aga = df.copy()
    df_aga.apply_if_greater_add("x0", "width", 800, 10)
    expected_widths = [r["width"] + 10 if r["x0"] > 800 else r["width"] for r in rows]
    check("apply_if_greater_add x0>800 → width+10", list(df_aga.width.values()), expected_widths)

    df_ala = df.copy()
    df_ala.apply_if_less_add("width", "x0", 30, 5)
    expected_x0 = [r["x0"] + 5 if r["width"] < 30 else r["x0"] for r in rows]
    check("apply_if_less_add width<30 → x0+5", list(df_ala.x0.values()), expected_x0)

    df_agay = df.copy()
    df_agay.apply_if_greater_add_y("width", 150, 5)
    expected_y0 = [r["y0"] + 5 if r["width"] > 150 else r["y0"] for r in rows]
    check("apply_if_greater_add_y width>150 → y0+5", list(df_agay.y0.values()), expected_y0)

    df_alay = df.copy()
    df_alay.apply_if_less_add_y("height", 20, 3)
    expected_y0_2 = [r["y0"] + 3 if r["height"] < 20 else r["y0"] for r in rows]
    check("apply_if_less_add_y height<20 → y0+3", list(df_alay.y0.values()), expected_y0_2)


def test_drop_in_bbox(df: NDframe, rows: List[ParsedRow]):
    section("26. drop_in_bbox")
    df_d = df.copy()
    df_d.drop_in_bbox(0, 0, 300, 300)
    expected = [r for r in rows if not (r["x0"] > 0 and r["y0"] > 0 and r["x1"] < 300 and r["y1"] < 300)]
    check("drop_in_bbox(0,0,300,300) len", len(df_d), len(expected))


def test_small_df_suite():
    """小型人工构造 DF 的专项测试（处理精确边界值的验证）"""
    section("27. 小型 DF 专项测试")
    df = make_small_df()
    rows_data = [
        {"idx": 0, "x0": 10, "y0": 20, "x1": 100, "y1": 60, "width": 90, "height": 40, "text": "hello world 你好世界", "text_sep": "sep_a"},
        {"idx": 1, "x0": 50, "y0": 80, "x1": 200, "y1": 150, "width": 150, "height": 70, "text": "foo bar baz ", "text_sep": "sep_b_connector"},
        {"idx": 2, "x0": 30, "y0": 40, "x1": 80, "y1": 90, "width": 50, "height": 50, "text": "test 123 regex 正则测试", "text_sep": "sep_c"},
        {"idx": 3, "x0": 5, "y0": 5, "x1": 30, "y1": 30, "width": 25, "height": 25, "text": "short 快 短", "text_sep": "sep_d"},
        {"idx": 4, "x0": 60, "y0": 100, "x1": 180, "y1": 200, "width": 120, "height": 100, "text": "another one 这一个", "text_sep": "sep_e"},
    ]
    texts = [r["text"] for r in rows_data]

    subsection("构造与属性")
    check("len", len(df), 5)
    check("empty", df.empty, False)
    lst = df.to_list()
    check("to_list len", len(lst), 5)

    subsection("数值聚合")
    check("x0.max", df.x0.max(), 60)
    check("x0.min", df.x0.min(), 5)
    check("y0.max", df.y0.max(), 100)
    check("y0.min", df.y0.min(), 5)
    check("width.max", df.width.max(), 150)
    check("x0.mean", df.x0.mean(), 31.0)
    check("x0.mode", df.x0.mode(), (5, 1))
    check("x0.diff", df.x0.diff(), [None, 40, -20, -25, 55])
    x0_var1 = sum((v - 31) ** 2 for v in [10, 50, 30, 5, 60]) / 4
    check("x0.var", round(df.x0.var(1), 10), round(x0_var1, 10))

    subsection("数值过滤")
    check("x0.greater(40)", len(df.x0.greater(40)), 2)
    check("x0.less(40)", len(df.x0.less(40)), 3)
    check("y0.between(10,90)", len(df.y0.between(10, 90)), 3)
    check("x0.idx_greater(40)", df.x0.idx_greater(40), [1, 4])
    check("x0.idx_equal(30)", df.x0.idx_equal(30), [2])

    subsection("数值运算")
    df_a = df.copy()
    orig = list(df_a.x0.values())
    df_a.x0.add(5)
    check("x0.add(5)", list(df_a.x0.values()), [v + 5 for v in orig])
    df_a.x0.sub(3)
    check("x0.add(5).sub(3)", list(df_a.x0.values()), [max(v + 2, 0) for v in orig])

    subsection("字符串操作")
    check("text.join('|')", df.text.join(" | "), " | ".join(t for t in texts))
    check("text.len", df.text.len(), [len(t) for t in texts])
    check("text.count('e')", df.text.count('e'), 6)

    df_r = df.copy()
    df_r.text.replace(" ", "_")
    check("text.replace", df_r.text.join("|"), "|".join(t.replace(" ", "_") for t in texts))

    # is_equal / contains_substr 精确值
    check("is_equal exact", df.text.is_equal("foo bar baz ").idx.values(), [1])
    check("is_equal miss", len(df.text.is_equal("foo")), 0)
    check("is_equal empty", len(df.text.is_equal("")), 0)
    check("contains_substr 123", df.text.contains_substr("123").idx.values(), [2])
    check("contains_substr 你好", df.text.contains_substr("你好").idx.values(), [0])
    check("contains_substr miss", len(df.text.contains_substr("none")), 0)

    df_st = df.copy()
    df_st.text.strip()
    check("text.strip[1] endswith space", df_st.to_list()[1]["text"].endswith(" "), False)

    df_cap = df.copy()
    df_cap.text.capitalize()
    check("capitalize[0]", df_cap.to_list()[0]["text"][0], "H")
    df_swp = df.copy()
    df_swp.text.swapcase()
    check("swapcase[0]", df_swp.to_list()[0]["text"][:5], "HELLO")

    subsection("字符类型判断")
    df_is = NDframe()
    for t in ["abc123", "hello", "123", "   ", "HELLO", "Hello World"]:
        df_is.append(0, 0, 0, 0, 0, 0, 0, 0, 0, t, "")
    check("isalnum", len(df_is.text.isalnum()), 4)
    check("isalpha", len(df_is.text.isalpha()), 2)
    check("isdigit", len(df_is.text.isdigit()), 1)
    check("isspace", len(df_is.text.isspace()), 1)
    check("islower", len(df_is.text.islower()), 1)  # 逐字符严格: 仅 hello (abc123 含数字)
    check("isupper", len(df_is.text.isupper()), 1)
    check("istitle", len(df_is.text.istitle()), 1)  # 实现为全小写或标题字母：仅 hello
    check("isnumeric", len(df_is.text.isnumeric()), 1)
    check("isdecimal", len(df_is.text.isdecimal()), 1)

    subsection("中文字符操作")
    df_kc = df.copy()
    df_kc.text.keep_chinese()
    check("keep_chinese[0]", df_kc.to_list()[0]["text"], "你好世界")
    check("keep_chinese[1]", df_kc.to_list()[1]["text"], "")
    check("keep_chinese[3]", df_kc.to_list()[3]["text"], "快短")

    df_knp = df.copy()
    df_knp.text.keep_numeric_punct()
    check("keep_numeric_punct[2]", df_knp.to_list()[2]["text"], "123")

    df_iscn = NDframe()
    df_iscn.append(0, 0, 0, 0, 0, 0, 0, 0, 0, "hello世界", "")
    df_iscn.append(1, 0, 0, 0, 0, 0, 0, 0, 0, "你好世界", "")
    check("is_chinese", len(df_iscn.text.is_chinese()), 1)
    check("is_chinese[0] text", df_iscn.text.is_chinese().to_list()[0]["text"], "你好世界")

    subsection("正则 / 纯匹配")
    df_rr = df.copy()
    df_rr.text.regex_replace(re.compile(r"\d+"), "X")
    check("regex_replace d+ -> X", df_rr.to_list()[2]["text"], "test X regex 正则测试")

    check("contains_substr('world')", len(df.text.contains_substr("world")), 1)
    check("contains_substr('123')", len(df.text.contains_substr("123")), 1)
    check("contains_substr('none')", len(df.text.contains_substr("none")), 0)
    check("contains('world')", len(df.text.contains(re.compile(r"world"))), 1)

    subsection("字符串长度过滤")
    check("strlen_greater(10)", len(df.text.strlen_greater(10)), sum(1 for t in texts if len(t) > 10))
    check("strlen_less(10)", len(df.text.strlen_less(10)), sum(1 for t in texts if len(t) < 10))
    check("strlen_equal(15)", len(df.text.strlen_equal(15)), sum(1 for t in texts if len(t) == 15))
    check("idx_strlen_greater(10)", df.text.idx_strlen_greater(10), [i for i, t in enumerate(texts) if len(t) > 10])

    subsection("集合筛选")
    check("is_in", len(df.text.is_in(["hello world 你好世界", "short 快 短"])), 2)
    check("not_in", len(df.text.not_in(["hello world 你好世界"])), 4)
    check("contains_any True", df.text.contains_any(["hello", "nope"]), True)
    check("contains_any False", df.text.contains_any(["nonexistent"]), False)

    subsection("跨字段替换")
    df_cr = df.copy()
    df_cr.text.replace_to(" ", "_", "text_sep")
    check("replace_to text→text_sep[0]", df_cr.to_list()[0]["text_sep"], "hello_world_你好世界")
    check("replace_to text unchanged", list(df_cr.text.values()), texts)

    df_cr2 = df.copy()
    df_cr2.text_sep.replace_to("sep", "REPLACED", "text")
    check("replace_to sep→text[0]", df_cr2.to_list()[0]["text"], "REPLACED_a")

    subsection("排序")
    df_s = df.copy()
    df_s.x0.sort()
    check("x0.sort", list(df_s.x0.values()), [5, 10, 30, 50, 60])
    df_s2 = df.copy()
    df_s2.y0.sort(descending=True)
    check("y0.sort desc", list(df_s2.y0.values()), [100, 80, 40, 20, 5])
    df_s3 = df.copy()
    df_s3.sort_by_y1()
    check("sort_by_y1", list(df_s3.y1.values()), [30, 60, 90, 150, 200])

    subsection("索引 / 切片 / 复制")
    check("head(2)", len(df.head(2)), 2)
    check("tail(2)", len(df.tail(2)), 2)
    check("df[0]['text']", df[0]["text"], "hello world 你好世界")
    check("df[-1]['text']", df[-1]["text"], "another one 这一个")
    check("df[1:3] len", len(df[1:3]), 2)
    check("df[1:3][0] x0", df[1:3][0]["x0"], 50)

    subsection("drop / add_y0y1")
    df_d = df.copy()
    df_d.drop(0)
    check("drop(0)", len(df_d), 4)
    df_ay = df.copy()
    orig = list(df_ay.y0.values())
    df_ay.add_y0y1(10)
    check("add_y0y1", list(df_ay.y0.values()), [v + 10 for v in orig])

    subsection("merge_line")
    df_ml = NDframe()
    df_ml.append(0, 10, 10, 50, 40, 40, 30, 1, 0, "first", "")
    df_ml.append(1, 52, 12, 100, 42, 48, 30, 1, 0, "second", "")
    df_ml.append(2, 200, 200, 300, 250, 100, 50, 1, 0, "far", "")
    merged = df_ml.merge_line(10, 10, " ")
    check("merge_line len", len(merged), 2)
    check("merge_line text", merged[0]["text"], "first second")

    subsection("条件运算")
    df_aga = df.copy()
    df_aga.apply_if_greater_add("x0", "width", 20, 10)
    check("apply_if_greater_add", list(df_aga.width.values()), [90, 160, 60, 25, 130])
    df_ala = df.copy()
    df_ala.apply_if_less_add("x0", "width", 20, 5)
    check("apply_if_less_add", list(df_ala.width.values()), [95, 150, 50, 30, 120])

    subsection("drop_in_bbox / bbox / centroid")
    df_dbi = df.copy()
    df_dbi.drop_in_bbox(0, 0, 30, 30)
    check("drop_in_bbox strict", len(df_dbi), 5)
    check("bbox(0)", df.bbox(0), (10, 20, 100, 60))
    check("centroid_line(0)", df.centroid_line(0), (55.0, 40.0))

    subsection("from_dict / concat / pop_back / drop_by_idx / drop_regex / idx_*")
    fd = NDframe.from_dict({"text": ["A", "B"], "x0": [1, 2], "y0": [1, 2],
                            "x1": [1, 2], "y1": [1, 2], "width": [1, 2], "height": [1, 2]})
    check("from_dict", len(fd), 2)
    check("concat", len(NDframe.concat(df, df.head(2))), 7)

    df_pb = df.copy()
    df_pb.pop_back()
    check("pop_back", len(df_pb), 4)

    df_dbi = df.copy()
    df_dbi.drop_by_idx([1, 3])
    check("drop_by_idx len", len(df_dbi), 3)

    df_dr = df.copy()
    df_dr.text.drop_regex(re.compile(r"hello"))
    check("drop_regex", len(df_dr), 4)

    check("idx_contains hello", df.text.idx_contains(re.compile(r"hello")), [0])
    check("idx_strlen_greater 10", df.text.idx_strlen_greater(10), [0, 1, 2, 4])
    check("idx_contains_in_bbox", df.text.idx_contains_in_bbox(re.compile(r"hello"), 0, 0, 200, 200), [0])


# ════════════════════════════════════════════════════════════════
# 性能压力测试
# ════════════════════════════════════════════════════════════════

def _random_text(length: int) -> str:
    """生成长度约 length 的随机中英混合文本"""
    parts = []
    while len(parts) < length:
        choice = random.random()
        if choice < 0.4:
            wlen = random.randint(2, 12)
            parts.append("".join(random.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!? ") for _ in range(wlen)))
        elif choice < 0.7:
            parts.append(chr(random.randint(0x4E00, 0x9FFF)))
        else:
            parts.append(" ")
    return "".join(parts[:length])


def _random_coord() -> Tuple[int, int, int, int]:
    left = random.randint(0, 2000)
    top = random.randint(0, 2000)
    w = random.randint(5, 300)
    h = random.randint(5, 200)
    return left, top, w, h


def _build_random_df(n: int) -> NDframe:
    df = NDframe(n)
    for i in range(n):
        left, top, w, h = _random_coord()
        page = random.randint(1, 5)
        flags = random.randint(0, 3)
        txt = _random_text(random.randint(1, 30))
        df.append(i, left, top, left + w, top + h, w, h, page, flags, txt, "")
    return df


def test_performance_small():
    section("100. 性能测试 — 小规模 (1000 rows)")
    subsection("构造")
    t0 = time.perf_counter()
    df = _build_random_df(1000)
    dt = time.perf_counter() - t0
    check("build 1000 rows", len(df), 1000)
    print(f"   构建耗时: {dt*1000:.1f} ms")

    subsection("数值聚合")
    for name, acc in [("x0", df.x0), ("y0", df.y0), ("width", df.width), ("height", df.height)]:
        t0 = time.perf_counter()
        _ = acc.max(), acc.min(), acc.mean()
        dt = time.perf_counter() - t0
        print(f"   {name}.max/min/mean: {dt*1000:.2f} ms")

    subsection("数值过滤")
    t0 = time.perf_counter()
    _ = df.x0.greater(500)
    dt = time.perf_counter() - t0
    print(f"   x0.greater(500): {dt*1000:.2f} ms")

    subsection("字符串操作")
    t0 = time.perf_counter()
    _ = df.text.len()
    dt = time.perf_counter() - t0
    print(f"   text.len(): {dt*1000:.2f} ms")

    t0 = time.perf_counter()
    _ = df.text.count("e")
    dt = time.perf_counter() - t0
    print(f"   text.count('e'): {dt*1000:.2f} ms")

    t0 = time.perf_counter()
    _ = df.text.contains(re.compile(r"\d+"))
    dt = time.perf_counter() - t0
    print(f"   text.contains(数字) 正则: {dt*1000:.2f} ms")

    t0 = time.perf_counter()
    _ = df.text.contains_substr("a")
    dt = time.perf_counter() - t0
    print(f"   text.contains_substr('a') 纯C: {dt*1000:.2f} ms")

    subsection("排序")
    t0 = time.perf_counter()
    df_s = df.copy()
    df_s.x0.sort()
    dt = time.perf_counter() - t0
    print(f"   x0.sort: {dt*1000:.2f} ms")

    subsection("复制/转换")
    t0 = time.perf_counter()
    _ = df.copy()
    dt = time.perf_counter() - t0
    print(f"   copy: {dt*1000:.2f} ms")

    t0 = time.perf_counter()
    _ = df.to_list()
    dt = time.perf_counter() - t0
    print(f"   to_list: {dt*1000:.2f} ms")

    t0 = time.perf_counter()
    _ = df.to_string()
    dt = time.perf_counter() - t0
    print(f"   to_string: {dt*1000:.2f} ms")


def test_performance_medium():
    section("101. 性能测试 — 中规模 (10000 rows)")
    subsection("构造")
    t0 = time.perf_counter()
    df = _build_random_df(10000)
    dt = time.perf_counter() - t0
    check("build 10000 rows", len(df), 10000)
    print(f"   构建耗时: {dt*1000:.1f} ms")

    subsection("数值聚合")
    for name in ("x0", "y0", "width", "height"):
        acc = getattr(df, name)
        t0 = time.perf_counter()
        _ = acc.max(), acc.min(), acc.mean()
        dt = time.perf_counter() - t0
        print(f"   {name}.max/min/mean: {dt*1000:.2f} ms")

    subsection("数值过滤")
    for name in ("x0", "y0"):
        acc = getattr(df, name)
        t0 = time.perf_counter()
        _ = acc.greater(500)
        dt = time.perf_counter() - t0
        print(f"   {name}.greater(500): {dt*1000:.2f} ms")

    subsection("字符串扫描")
    t0 = time.perf_counter()
    _ = df.text.len()
    dt = time.perf_counter() - t0
    print(f"   text.len(): {dt*1000:.2f} ms")

    t0 = time.perf_counter()
    _ = df.text.count("e")
    dt = time.perf_counter() - t0
    print(f"   text.count('e'): {dt*1000:.2f} ms")

    subsection("正则/纯匹配过滤")
    t0 = time.perf_counter()
    _ = df.text.contains(re.compile(r"\d+"))
    dt = time.perf_counter() - t0
    print(f"   text.contains(数字) 正则: {dt*1000:.2f} ms")

    t0 = time.perf_counter()
    _ = df.text.contains_substr("a")
    dt = time.perf_counter() - t0
    print(f"   text.contains_substr('a') 纯C: {dt*1000:.2f} ms")

    subsection("排序")
    t0 = time.perf_counter()
    df_s = df.copy()
    df_s.x0.sort()
    dt = time.perf_counter() - t0
    print(f"   x0.sort: {dt*1000:.2f} ms")

    subsection("字符串变换")
    t0 = time.perf_counter()
    df_t = df.copy()
    df_t.text.strip()
    dt = time.perf_counter() - t0
    print(f"   text.strip: {dt*1000:.2f} ms")

    t0 = time.perf_counter()
    df_t2 = df.copy()
    df_t2.text.replace(" ", "_")
    dt = time.perf_counter() - t0
    print(f"   text.replace(' ', '_'): {dt*1000:.2f} ms")

    subsection("删除操作")
    t0 = time.perf_counter()
    df_d = df.copy()
    df_d.drop_by_idx(list(range(0, 10000, 2)))
    dt = time.perf_counter() - t0
    check(f"drop_by_idx half rows", len(df_d), 5000)
    print(f"   drop_by_idx(half): {dt*1000:.1f} ms")

    subsection("空间过滤")
    t0 = time.perf_counter()
    _ = df.filter_in_bbox(100, 100, 1000, 1000)
    dt = time.perf_counter() - t0
    print(f"   filter_in_bbox: {dt*1000:.2f} ms")

    subsection("concat")
    t0 = time.perf_counter()
    _ = NDframe.concat(df, df)
    dt = time.perf_counter() - t0
    print(f"   concat(self, self): {dt*1000:.2f} ms")

    subsection("from_dict")
    d = {
        "text": ["hello"] * 10000,
        "x0": [i for i in range(10000)],
        "x1": [i for i in range(10000)],
        "y0": [i * 2 for i in range(10000)],
        "y1": [i * 2 for i in range(10000)],
        "width": [50] * 10000,
        "height": [30] * 10000,
    }
    t0 = time.perf_counter()
    _ = NDframe.from_dict(d)
    dt = time.perf_counter() - t0
    print(f"   from_dict(10000): {dt*1000:.2f} ms")


def test_performance_large():
    section("102. 性能测试 — 大规模 (100000 rows)")
    subsection("构造")
    t0 = time.perf_counter()
    df = _build_random_df(100000)
    dt = time.perf_counter() - t0
    check("build 100000 rows", len(df), 100000)
    print(f"   构建耗时: {dt*1000:.1f} ms")

    subsection("数值聚合")
    t0 = time.perf_counter()
    _ = df.x0.max(), df.x0.min(), df.x0.mean()
    dt = time.perf_counter() - t0
    print(f"   x0.max/min/mean: {dt*1000:.2f} ms")

    subsection("数值过滤")
    t0 = time.perf_counter()
    _ = df.x0.greater(500)
    dt = time.perf_counter() - t0
    print(f"   x0.greater(500): {dt*1000:.2f} ms")

    subsection("字符串扫描")
    t0 = time.perf_counter()
    _ = df.text.len()
    dt = time.perf_counter() - t0
    print(f"   text.len(): {dt*1000:.2f} ms")

    subsection("排序 (100k)")
    t0 = time.perf_counter()
    df_s = df.copy()
    df_s.x0.sort()
    dt = time.perf_counter() - t0
    print(f"   x0.sort: {dt*1000:.2f} ms")

    subsection("正则/纯匹配过滤 100k")
    t0 = time.perf_counter()
    _ = df.text.contains(re.compile(r"\d+"))
    dt = time.perf_counter() - t0
    print(f"   text.contains(数字) 正则: {dt*1000:.2f} ms")

    t0 = time.perf_counter()
    _ = df.text.contains_substr("a")
    dt = time.perf_counter() - t0
    print(f"   text.contains_substr('a') 纯C: {dt*1000:.2f} ms")

    subsection("字符串替换 100k")
    t0 = time.perf_counter()
    df_t = df.copy()
    df_t.text.replace(" ", "_")
    dt = time.perf_counter() - t0
    print(f"   text.replace(' ', '_'): {dt*1000:.2f} ms")

    subsection("复制 / to_list")
    t0 = time.perf_counter()
    _ = df.copy()
    dt = time.perf_counter() - t0
    print(f"   copy: {dt*1000:.2f} ms")

    t0 = time.perf_counter()
    _ = df.to_list()
    dt = time.perf_counter() - t0
    print(f"   to_list: {dt*1000:.2f} ms")


# ════════════════════════════════════════════════════════════════
# 主函数
# ════════════════════════════════════════════════════════════════

def main():
    global PASS, FAIL

    print("=" * 60)
    print("  fixedndframe 完整功能测试套件")
    print("=" * 60)

    # 加载真实 OCR 数据
    df, rows = load_test_data()
    df.print()
    print("df=", df.text.values())
    print("df=", df.text.len())
    print(f"\n加载测试数据: {len(rows)} 行")

    # ── 功能测试（真实数据）──
    test_basic_properties(df, rows)
    test_to_list_string(df, rows)
    test_numeric_aggregation(df, rows)
    test_numeric_filter(df, rows)
    test_numeric_arithmetic(df, rows)
    test_numeric_sort(df, rows)
    test_numeric_idx(df, rows)
    test_string_basic(df, rows)
    test_string_transform(df, rows)
    test_string_is(df, rows)
    test_string_starts_ends(df, rows)
    test_string_regex(df, rows)
    test_string_contains_pure(df, rows)
    test_string_len_filter(df, rows)
    test_string_set_ops(df, rows)
    test_string_equal_substr(df, rows)
    test_string_idx_ops(df, rows)
    test_string_drop_regex(df, rows)
    test_cross_replace(df, rows)
    test_spatial_filter(df, rows)
    test_spatial_idx(df, rows)
    test_drop_operations(df, rows)
    test_copy_head_tail(df, rows)
    test_merge_bbox(df, rows)
    test_misc_operations(df, rows)
    test_conditional_ops(df, rows)
    test_drop_in_bbox(df, rows)

    # ── 小型人工数据精确测试 ──
    test_small_df_suite()

    # ── 性能压力测试 ──
    test_performance_small()
    test_performance_medium()
    test_performance_large()

    # ── 汇总 ──
    total = PASS + FAIL
    print(f"\n{'=' * 60}")
    print(f"  结果: {PASS}/{total} 通过, {FAIL} 失败")
    print(f"{'=' * 60}")

    if FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
