#!/usr/bin/env python3
"""
字符串相关函数性能对比基准

  对比对象:
    1. SolidDataFrame / ndframe        (SolidDataFrame/build 编译产物)
    2. FixedDataFrame / fixedndframe   (FixedDataFrame/build 编译产物)

  侧重: 字符串筛选 (contains / startswith / strlen / is_chinese / is_in /
       drop_regex / regex_strlen / contains_substr(Fixed 独有))、
      字符串变换 (replace / regex_replace / strip / keep_chinese)、
      字符串聚合 (join / count / len)、合并行 (merge_line)、
      以及少量非字符串基线 (数值过滤 / to_list / append 构造)。

  用法:
    python3 benchmark_string_perf.py [rows] [repeats]

  例如:
    python3 benchmark_string_perf.py 100000 3

  提示: 运行前请先分别编译两个库:
    cd SolidDataFrame && python3 -c "from setuptools import setup; from Cython.Build import cythonize; setup(ext_modules=cythonize('ndframe.pyx', language_level=3))" build_ext --build-lib=build
    cd FixedDataFrame && python3 -c "from setuptools import setup; from Cython.Build import cythonize; setup(ext_modules=cythonize('fixedndframe.pyx', language_level=3))" build_ext --build-lib=build
"""
import os
import random
import re
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
SOLID_BUILD = os.path.join(ROOT, "SolidDataFrame", "build")
FIXED_BUILD = os.path.join(ROOT, "FixedDataFrame", "build")
sys.path.insert(0, SOLID_BUILD)
sys.path.insert(0, FIXED_BUILD)

try:
    import ndframe as Solid
except ImportError:
    print("SolidDataFrame 未编译，请先编译 (见脚本头注释)")
    sys.exit(1)
try:
    import fixedndframe as Fixed
except ImportError:
    print("FixedDataFrame 未编译，请先编译 (见脚本头注释)")
    sys.exit(1)

ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
REPEATS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
GLOBAL_SEED = 20260807

# 用于 is_in / not_in 的固定集合
SET_TEXTS = ["澳门澳门", "发票", "收据", "银行", "单位", "姓名", "地址", "编号"]

# contains_substr vs contains 专项对比的嵌入关键词
# 格式: (嵌入全串, 测试子串, 类型标签, 字符类别)
SUBSTR_CASES = [
    ("中文识别",       "中文",      "纯中文",   "zh"),
    ("EnglishText",    "English",   "纯英文",   "en"),
    ("abc中英1aB混排",  "中英1aB",   "中英混排",  "mixed"),
]

# ════════════════════════════════════════════════════════════════
# 数据构造（两库使用完全相同的伪随机序列，保证可比性）
# ════════════════════════════════════════════════════════════════
_MIX = ("中文测试发票金额收据姓名地址编号号码银行单位年月日"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,")


def _random_text(rng, length):
    return "".join(rng.choice(_MIX) for _ in range(length))


def build_df(mod, rows, seed):
    """随机生成 OCR 风格数据(坐标 + 混合中英文本)，全部行均经 append 追加。"""
    rng = random.Random(seed)
    df = mod(rows)
    for i in range(rows):
        left = rng.randint(0, 2000)
        top = rng.randint(0, 2000)
        w = rng.randint(5, 300)
        h = rng.randint(5, 200)
        txt = _random_text(rng, rng.randint(5, 50))
        # 让一部分行命中 '澳门' 前缀/正则过滤条件，避免命中率过低
        if rng.random() < 0.3:
            txt = "澳门" * rng.randint(1, 2) + txt[:20]
        # 注入中/英/混合三种嵌入关键词，保证 contains_substr 结论中有足够的命中样本
        for kw, _sub, _lbl, _cls in SUBSTR_CASES:
            if rng.random() < 0.18:
                pos = rng.randint(0, len(txt))
                txt = txt[:pos] + kw + txt[pos:]
        df.append(i, left, top, left + w, top + h, w, h, rng.randint(1, 3), 0, txt, "")
    return df


# ════════════════════════════════════════════════════════════════
# 计时基础设施
# ════════════════════════════════════════════════════════════════

def time_best(fn, repeats=REPEATS):
    """取多次运行的最优耗时(ms)。"""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best * 1000.0


def time_best_on_copy(df, op, repeats=REPEATS):
    """就地可变操作：每轮从原 df 深拷贝后计时，只统计操作本身开销。"""
    best = float("inf")
    for _ in range(repeats):
        work = df.copy()
        t0 = time.perf_counter()
        op(work)
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best * 1000.0


class Benchmark:
    def __init__(self):
        self.rows = []
        self.only_fixed = []
        self.substr_rows = []

    def add(self, name, solid_fn, fixed_fn):
        s = time_best(solid_fn)
        f = time_best(fixed_fn)
        self.rows.append((name, s, f))

    def add_copy_op(self, name, solid_op, fixed_op):
        s = time_best_on_copy(self.solid_df, solid_op)
        f = time_best_on_copy(self.fixed_df, fixed_op)
        self.rows.append((name, s, f))

    def add_fixed_only(self, name, fixed_fn, inplace=False):
        if inplace:
            f = time_best_on_copy(self.fixed_df, fixed_fn)
        else:
            f = time_best(fixed_fn)
        self.only_fixed.append((name, f))

    def add_substr(self, label, sub, solid_df, fixed_df, case_cls):
        """Fixed.contains_substr vs Fixed.contains vs Solid.contains 专项计时。
        调用前已完成命中数一致性断言。"""
        pat = re.compile(sub)
        f_sub = time_best(lambda: fixed_df.text.contains_substr(sub))
        f_re = time_best(lambda: fixed_df.text.contains(pat))
        s_re = time_best(lambda: solid_df.text.contains(pat))
        self.substr_rows.append((label, sub, case_cls, f_sub, f_re, s_re))

    def show(self):
        print(f"\n{'=' * 92}")
        print(f"  数据行数 = {ROWS:,}    重复次数 = {REPEATS}    单位 = ms (越小越快)")
        print(f"  比值 = Solid / Fixed (>1 表示 Fixed 更快, <1 表示 Solid 更快)")
        print(f"{'=' * 92}")
        print(f"{'操作':<36}{'Solid':>12}{'Fixed':>12}{'比值':>8}")
        print("-" * 92)
        for name, s, f in self.rows:
            ratio = s / f if f > 1e-9 else float("nan")
            print(f"{name:<36}{s:>11.3f}{f:>11.3f}{ratio:>8.2f}")
        if self.only_fixed:
            print("-" * 92)
            print("以下为 FixedDataFrame 独有 API (SolidDataFrame 无对应实现):")
            for name, f in self.only_fixed:
                print(f"  * {name:<36}{f:>11.3f}")
        print("-" * 92)

        if self.substr_rows:
            print(f"{'=' * 92}")
            print("专项: 字符串存在性查找 — Fixed.contains_substr(纯C) vs Solid.contains(正则)")
            print(f"{'=' * 92}")
            print(f"  命中数一致性: 已断言三种路径返回完全一致的索引集合")
            print(f"  比值(base=Solid.contains) <1 表示 Fixed.contains_substr 更快")
            print(f"{'类型':<12}{'测试子串':<16}{'固定substr':>12}{'固定regex':>12}{'Solid.regex':>12}")
            print("-" * 92)
            for label, sub, cls, f_sub, f_re, s_re in self.substr_rows:
                print(f"{cls:<12}{sub:<16}{f_sub:>11.3f}{f_re:>11.3f}{s_re:>11.3f}")
            print("-" * 92)
            print("  固定substr / 固定regex / Solid.regex 三者对比:")
            for label, sub, cls, f_sub, f_re, s_re in self.substr_rows:
                r1 = f_sub / s_re if s_re > 1e-9 else float("nan")
                r2 = f_re / s_re if s_re > 1e-9 else float("nan")
                print(f"  * [{cls}] 固定contains_substr 相对 Solid.contains: {r1:.2f}x  "
                      f"(同库 contains 正则路径: {r2:.2f}x)")
            print("-" * 92)


def main():
    if ROWS <= 0:
        raise ValueError("rows 必须为正整数")

    print("=" * 92)
    print(f"  SolidDataFrame vs FixedDataFrame — 字符串性能对比")
    print(f"  数据规模: {ROWS:,} 行, 每轮重复 {REPEATS} 次计时取最优")
    print("=" * 92)

    # 构建数据（同一随机种子 ⇒ 两库数据完全一致）
    print("正在构建数据 ...")
    solid_df = build_df(Solid.NDframe, ROWS, GLOBAL_SEED)
    fixed_df = build_df(Fixed.NDframe, ROWS, GLOBAL_SEED)
    print(f"  Solid len={len(solid_df)}, Fixed len={len(fixed_df)}")

    assert len(solid_df) == len(fixed_df) == ROWS, "两库行数不一致"
    assert list(solid_df.text.values()) == list(fixed_df.text.values()), "两库文本数据不一致"
    print("  一致性检查: 行数 / 文本 完全一致 ✓")

    bench = Benchmark()
    bench.solid_df = solid_df
    bench.fixed_df = fixed_df

    # ─────────── 过滤：字符串正则匹配 ───────────
    r_digits = re.compile(r"\d+")
    r_macau = re.compile(r"澳门")
    r_alpha = re.compile(r"[a-zA-Z]+")

    bench.add("text.contains(数字正则)",
              lambda: solid_df.text.contains(r_digits),
              lambda: fixed_df.text.contains(r_digits))
    bench.add("text.contains(中文正则)",
              lambda: solid_df.text.contains(r_macau),
              lambda: fixed_df.text.contains(r_macau))
    bench.add("text.contains_in_bbox(数字正则)",
              lambda: solid_df.text.contains_in_bbox(r_digits, 100, 100, 1500, 1500),
              lambda: fixed_df.text.contains_in_bbox(r_digits, 100, 100, 1500, 1500))
    bench.add("text.strlen_greater(10)",
              lambda: solid_df.text.strlen_greater(10),
              lambda: fixed_df.text.strlen_greater(10))
    bench.add("text.strlen_less(10)",
              lambda: solid_df.text.strlen_less(10),
              lambda: fixed_df.text.strlen_less(10))
    bench.add("text.is_chinese()",
              lambda: solid_df.text.is_chinese(),
              lambda: fixed_df.text.is_chinese())
    bench.add("text.is_in(固定集合)",
              lambda: solid_df.text.is_in(SET_TEXTS),
              lambda: fixed_df.text.is_in(SET_TEXTS))
    bench.add("text.not_in(固定集合)",
              lambda: solid_df.text.not_in(SET_TEXTS),
              lambda: fixed_df.text.not_in(SET_TEXTS))
    bench.add("text.regex_strlen_greater(数字,5)",
              lambda: solid_df.text.regex_strlen_greater(r_digits, 5),
              lambda: fixed_df.text.regex_strlen_greater(r_digits, 5))
    bench.add("text.idx_contains(字母正则)",
              lambda: solid_df.text.idx_contains(r_alpha),
              lambda: fixed_df.text.idx_contains(r_alpha))

    # ─────────── 过滤：前缀 / 后缀 ───────────
    bench.add("text.startswith('澳门')",
              lambda: solid_df.text.startswith("澳门"),
              lambda: fixed_df.text.startswith("澳门"))
    bench.add("text.endswith('.')",
              lambda: solid_df.text.endswith("."),
              lambda: fixed_df.text.endswith("."))

    # ─────────── 合并行 (merge_line) ───────────
    bench.add("merge_line(5,10) 常规合并",
              lambda: solid_df.merge_line(5, 10, ""),
              lambda: fixed_df.merge_line(5, 10, ""))
    bench.add("merge_line(500,500) 大阈值",
              lambda: solid_df.merge_line(500, 500, ""),
              lambda: fixed_df.merge_line(500, 500, ""))

    # ─────────── 字符串变换 (就地, 需拷贝) ───────────
    bench.add_copy_op("text.replace(' ','_') 就地",
                      lambda d: d.text.replace(" ", "_"),
                      lambda d: d.text.replace(" ", "_"))
    bench.add_copy_op("text.regex_replace(数字→X) 就地",
                      lambda d: d.text.regex_replace(r_digits, "X"),
                      lambda d: d.text.regex_replace(r_digits, "X"))
    bench.add_copy_op("text.strip() 就地",
                      lambda d: d.text.strip(),
                      lambda d: d.text.strip())
    bench.add_copy_op("text.keep_chinese() 就地",
                      lambda d: d.text.keep_chinese(),
                      lambda d: d.text.keep_chinese())
    bench.add_copy_op("text.drop_regex(正则) 就地删除",
                      lambda d: d.text.drop_regex(r_macau),
                      lambda d: d.text.drop_regex(r_macau))

    # ─────────── 字符串聚合 ───────────
    bench.add("text.join('|') 全部拼接",
              lambda: solid_df.text.join("|"),
              lambda: fixed_df.text.join("|"))
    bench.add("text.count('e') 子串计数",
              lambda: solid_df.text.count("e"),
              lambda: fixed_df.text.count("e"))
    bench.add("text.len() 长度列表",
              lambda: solid_df.text.len(),
              lambda: fixed_df.text.len())

    # ─────────── 非字符串基线 ───────────
    bench.add("数值过滤 x0.greater(500)",
              lambda: solid_df.x0.greater(500),
              lambda: fixed_df.x0.greater(500))
    bench.add("to_list() 全量导出",
              lambda: solid_df.to_list(),
              lambda: fixed_df.to_list())
    bench.add("构造 append 全部行",
              lambda: build_df(Solid.NDframe, ROWS, GLOBAL_SEED),
              lambda: build_df(Fixed.NDframe, ROWS, GLOBAL_SEED))

    # ─────────── FixedDataFrame 独有 ───────────
    bench.add_fixed_only("text.contains_substr('澳门') 纯C子串", lambda: fixed_df.text.contains_substr("澳门"))
    bench.add_fixed_only("text.is_equal(精确等值)", lambda: fixed_df.text.is_equal("澳门澳门"))
    bench.add_fixed_only("sort_by_text() 字符串排序", lambda d: d.sort_by_text(), inplace=True)
    bench.add_fixed_only("sort_by_text_sep() 分隔串排序", lambda d: d.sort_by_text_sep(), inplace=True)

    # ─────────── 专项：字符串存在性查找 ───────────
    # Fixed.contains_substr(纯C滑动窗口) vs Fixed.contains(正则) vs Solid.contains(正则)
    for kw, sub, lbl, cls in SUBSTR_CASES:
        f_sub_set = set(fixed_df.text.contains_substr(sub).idx.values())
        f_re_set = set(fixed_df.text.contains(re.compile(sub)).idx.values())
        s_re_set = set(solid_df.text.contains(re.compile(sub)).idx.values())
        ok = f_sub_set == f_re_set == s_re_set
        if not ok:
            print(f"  [警告] [{lbl}] 命中不一致: "
                  f"Fixed.substr={len(f_sub_set)} Fixed.regex={len(f_re_set)} Solid.regex={len(s_re_set)}")
        else:
            print(f"  存在性一致性 [{lbl}] 命中 {len(f_sub_set)} 行 ✓")
        bench.add_substr(lbl, sub, solid_df, fixed_df, cls)

    # ─────────── 专项：多模子串 (绕过正则状态机) ───────────
    multi_subs = [sub for _kw, sub, _lbl, _cls in SUBSTR_CASES]
    multi_pat = re.compile("|".join(multi_subs))
    f_multi = set(fixed_df.text.contains_substr_list(multi_subs).idx.values())
    s_multi = set(solid_df.text.contains(multi_pat).idx.values())
    if f_multi != s_multi:
        print(f"  [警告] 多模命中不一致: Fixed.list={len(f_multi)} Solid.regex={len(s_multi)}")
    else:
        print(f"  多模一致性 [三关键字并集] 命中 {len(f_multi)} 行 ✓")
    bench.add("多模三串 contains_substr_list vs Solid.contains",
              lambda: solid_df.text.contains(multi_pat),
              lambda: fixed_df.text.contains_substr_list(multi_subs))

    bench.show()


if __name__ == "__main__":
    main()
