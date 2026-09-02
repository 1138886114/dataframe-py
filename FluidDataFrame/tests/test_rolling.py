import math
import sys
sys.path.insert(0, '..')
from fluidframe.core.intcolumn import IntColumn
from fluidframe.core.doublecolumn import DoubleColumn
from fluidframe.core.fastdataframe import FastDataFrame

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {name}")

def approx_eq(a, b, tol=1e-9):
    return abs(a - b) < tol

def is_nan(v):
    return v != v

# ============================================================
#  Helper: build column from list
# ============================================================
def make_int_col(vals):
    col = IntColumn()
    for v in vals:
        col.append(v)
    return col

def make_double_col(vals):
    col = DoubleColumn()
    for v in vals:
        col.append(float(v))
    return col

def make_df(columns_data):
    df = FastDataFrame()
    n_rows = len(columns_data[0][1])
    for name, vals in columns_data:
        df[name] = make_int_col(vals)
    df.df_length = n_rows
    return df

# ============================================================
#  Test data: [1,2,3,4,5,6,7,8]
# ============================================================
df = make_df([('A', [1,2,3,4,5,6,7,8])])

# ============================================================
#  rolling().mean()
# ============================================================
print("=== rolling mean ===")
r = df.rolling(3, 'A')
m = r.mean()
check("len", m.length == 8)
check("nan_0", is_nan(m[0]))
check("nan_1", is_nan(m[1]))
check("mean_2", approx_eq(m[2], 2.0))   # (1+2+3)/3
check("mean_3", approx_eq(m[3], 3.0))   # (2+3+4)/3
check("mean_7", approx_eq(m[7], 7.0))   # (6+7+8)/3

# ============================================================
#  rolling().sum()
# ============================================================
print("=== rolling sum ===")
s = r.sum()
check("sum_2", approx_eq(s[2], 6.0))
check("sum_3", approx_eq(s[3], 9.0))
check("sum_7", approx_eq(s[7], 21.0))

# ============================================================
#  rolling().min()
# ============================================================
print("=== rolling min ===")
mn = r.min()
check("min_2", approx_eq(mn[2], 1.0))
check("min_4", approx_eq(mn[4], 3.0))
check("min_7", approx_eq(mn[7], 6.0))

# ============================================================
#  rolling().max()
# ============================================================
print("=== rolling max ===")
mx = r.max()
check("max_2", approx_eq(mx[2], 3.0))
check("max_4", approx_eq(mx[4], 5.0))
check("max_7", approx_eq(mx[7], 8.0))

# ============================================================
#  rolling().std()
# ============================================================
print("=== rolling std ===")
std = r.std()
check("std_2", approx_eq(std[2], math.sqrt(2.0/3.0), 1e-6))
check("std_5", approx_eq(std[5], math.sqrt(2.0/3.0), 1e-6))

# ============================================================
#  rolling().var()
# ============================================================
print("=== rolling var ===")
v = r.var()
check("var_2", approx_eq(v[2], 2.0/3.0, 1e-6))
check("var_5", approx_eq(v[5], 2.0/3.0, 1e-6))

# ============================================================
#  rolling().median()
# ============================================================
print("=== rolling median ===")
med = r.median()
check("med_2", approx_eq(med[2], 2.0))
check("med_4", approx_eq(med[4], 4.0))
check("med_7", approx_eq(med[7], 7.0))

# ============================================================
#  rolling().quantile(0.25)
# ============================================================
print("=== rolling quantile ===")
q = r.quantile(0.25)
check("q25_2", approx_eq(q[2], 1.5, 1e-6))
check("q25_5", approx_eq(q[5], 4.5, 1e-6))

# ============================================================
#  rolling().skew()
# ============================================================
print("=== rolling skew ===")
sk = r.skew()
check("skew_2", approx_eq(sk[2], 0.0, 1e-6))

# ============================================================
#  rolling().kurt()
# ============================================================
print("=== rolling kurt ===")
df_k = make_df([('A', list(range(1, 13)))])
rk = df_k.rolling(4, 'A')
ku = rk.kurt()
check("kurt_len", ku.length == 12)
check("kurt_nan", is_nan(ku[0]))
check("kurt_nan2", is_nan(ku[1]))
check("kurt_nan3", is_nan(ku[2]))
# window [1,2,3,4]: symmetric, excess kurtosis of uniform = -1.2
check("kurt_3", approx_eq(ku[3], -1.36, 1e-6))

# ============================================================
#  rolling().corr(other)
# ============================================================
print("=== rolling corr ===")
df2 = make_df([('A', [1,2,3,4,5,6]), ('B', [2,4,6,8,10,12])])
r_a = df2.rolling(3, 'A')
r_b = df2.rolling(3, 'B')
corr = r_a.corr(r_b)
check("corr_len", corr.length == 6)
check("corr_2", approx_eq(corr[2], 1.0, 1e-6))
check("corr_5", approx_eq(corr[5], 1.0, 1e-6))

df2['C'] = make_int_col([6,5,4,3,2,1])
df2.df_length = 6
r_c = df2.rolling(3, 'C')
corr_neg = r_a.corr(r_c)
check("corr_neg", approx_eq(corr_neg[2], -1.0, 1e-6))

# ============================================================
#  rolling().cov(other)
# ============================================================
print("=== rolling cov ===")
cov = r_a.cov(r_b)
check("cov_len", cov.length == 6)
check("cov_2", approx_eq(cov[2], 4.0/3.0, 1e-6))

# ============================================================
#  Edge: window = 1
# ============================================================
print("=== edge: window=1 ===")
r1 = df.rolling(1, 'A')
m1 = r1.mean()
check("w1_mean_0", approx_eq(m1[0], 1.0))
check("w1_mean_7", approx_eq(m1[7], 8.0))
s1 = r1.sum()
check("w1_sum_3", approx_eq(s1[3], 4.0))

# ============================================================
#  Edge: window = len
# ============================================================
print("=== edge: window=len ===")
r8 = df.rolling(8, 'A')
m8 = r8.mean()
check("w8_nan_0_6", all(is_nan(m8[i]) for i in range(7)))
check("w8_mean_7", approx_eq(m8[7], 4.5))

# ============================================================
#  Error: non-numeric column
# ============================================================
print("=== error cases ===")
df_str = FastDataFrame()
from fluidframe.core.char256column import Char256Column_SoA
cc = Char256Column_SoA()
for v in ["a", "b", "c"]:
    cc.append(v)
df_str['S'] = cc
df_str.df_length = 3
try:
    df_str.rolling(2, 'S')
    check("non_numeric_error", False)
except TypeError:
    check("non_numeric_error", True)

try:
    df.rolling(2, 'Z')
    check("missing_col_error", False)
except KeyError:
    check("missing_col_error", True)

try:
    df.rolling(0, 'A')
    check("window_zero_error", False)
except ValueError:
    check("window_zero_error", True)

# ============================================================
#  Summary
# ============================================================
print(f"\nResults: {passed} passed, {failed} failed")
if failed > 0:
    sys.exit(1)
