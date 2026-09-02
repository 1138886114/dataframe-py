import math
import sys
sys.path.insert(0, '..')
from fluidframe.core.intcolumn import IntColumn
from fluidframe.core.doublecolumn import DoubleColumn
from fluidframe.core.fastdataframe import FastDataFrame
from fluidframe.core.linalg import euclidean_dist, manhattan_dist, cosine_dist, cosine_similarity, jaccard_sim, hamming_dist

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

# ============================================================
#  Helper: build FastDataFrame from list of lists
# ============================================================
def make_df(columns_data):
    df = FastDataFrame()
    n_rows = len(columns_data[0][1])
    for name, vals in columns_data:
        col = DoubleColumn()
        for v in vals:
            col.append(float(v))
        df[name] = col
    df.df_length = n_rows
    return df

def make_df_int(columns_data):
    df = FastDataFrame()
    n_rows = len(columns_data[0][1])
    for name, vals in columns_data:
        col = IntColumn()
        for v in vals:
            col.append(v)
        df[name] = col
    df.df_length = n_rows
    return df

def make_int_col(vals):
    col = IntColumn()
    for v in vals:
        col.append(v)
    return col

# Distance matrix: column A=row0, B=row1, C=row2. Row index 0,1,2.
# d_ij = d[chr(65+j)][i]
def dist_val(d, i, j):
    return d[chr(65 + j)][i]

# ============================================================
#  Test data
# ============================================================
df_dist = make_df([('A', [1,4,0]), ('B', [2,5,3])])
df_bin = make_df([('X', [1,0,1]), ('Y', [1,1,0]), ('Z', [0,1,1])])

# ============================================================
#  euclidean_dist
# ============================================================
print("=== euclidean_dist ===")
d = euclidean_dist(df_dist)
check("shape", d.df_length == 3 and len(d.columns) == 3)
check("diag=0", approx_eq(dist_val(d, 0, 0), 0.0))
check("0-1", approx_eq(dist_val(d, 0, 1), math.sqrt(18), 1e-6))
check("0-2", approx_eq(dist_val(d, 0, 2), math.sqrt(2), 1e-6))
check("1-2", approx_eq(dist_val(d, 1, 2), math.sqrt(20), 1e-6))
check("sym", approx_eq(dist_val(d, 0, 1), dist_val(d, 1, 0)))

# ============================================================
#  manhattan_dist
# ============================================================
print("=== manhattan_dist ===")
d = manhattan_dist(df_dist)
check("diag=0", approx_eq(dist_val(d, 0, 0), 0.0))
check("0-1", approx_eq(dist_val(d, 0, 1), 6.0))
check("0-2", approx_eq(dist_val(d, 0, 2), 2.0))
check("1-2", approx_eq(dist_val(d, 1, 2), 6.0))
check("sym", approx_eq(dist_val(d, 0, 1), dist_val(d, 1, 0)))

# ============================================================
#  cosine_dist
# ============================================================
print("=== cosine_dist ===")
d = cosine_dist(df_dist)
check("diag=0", approx_eq(dist_val(d, 0, 0), 0.0))
dot01 = 1*4 + 2*5
n0 = math.sqrt(5)
n1 = math.sqrt(41)
expected = 1.0 - dot01 / (n0 * n1)
check("0-1", approx_eq(dist_val(d, 0, 1), expected, 1e-6))
check("sym", approx_eq(dist_val(d, 0, 1), dist_val(d, 1, 0)))

# ============================================================
#  cosine_similarity
# ============================================================
print("=== cosine_similarity ===")
d = cosine_similarity(df_dist)
check("diag=1", approx_eq(dist_val(d, 0, 0), 1.0))
check("0-1", approx_eq(dist_val(d, 0, 1), dot01 / (n0 * n1), 1e-6))
check("sym", approx_eq(dist_val(d, 0, 1), dist_val(d, 1, 0)))

# ============================================================
#  jaccard_sim
# ============================================================
print("=== jaccard_sim ===")
d = jaccard_sim(df_bin)
# d_bin has columns X(=row0), Y(=row1), Z(=row2), index 0,1,2
# dist_val(d, i, j) = d[chr(88+j)][i] => X=row0, Y=row1, Z=row2
# So dist_val(d, 0, 0) = d['X'][0], dist_val(d, 0, 1) = d['Y'][0], etc.
# But wait - df_bin has 3 rows and 3 columns. After euclidean_dist, result is 3x3.
# Column names: A(=row0), B(=row1), C(=row2). Row index: 0,1,2.
# But df_bin has columns X, Y, Z. So after _from_contiguous, result columns are A, B, C.
# dist_val(d, i, j) = d[chr(65+j)][i]

# Actually, let me just use direct column access:
# d has columns A, B, C (auto-generated). Row index 0, 1, 2.
# d['A'][0] = dist(0,0), d['B'][0] = dist(0,1), d['C'][0] = dist(0,2)
# d['A'][1] = dist(1,0), d['B'][1] = dist(1,1), d['C'][1] = dist(1,2)
d_00 = d['A'][0]
d_01 = d['B'][0]
d_02 = d['C'][0]
d_10 = d['A'][1]
d_11 = d['B'][1]
d_12 = d['C'][1]
d_20 = d['A'][2]
d_21 = d['B'][2]
d_22 = d['C'][2]

check("diag=1", approx_eq(d_00, 1.0))
# X=[1,0,1], Y=[1,1,0]: union=3, intersection=1 => 1/3
check("X-Y", approx_eq(d_01, 1.0/3.0, 1e-6))
# X=[1,0,1], Z=[0,1,1]: union=3, intersection=1 => 1/3
check("X-Z", approx_eq(d_02, 1.0/3.0, 1e-6))
# Y=[1,1,0], Z=[0,1,1]: union=3, intersection=1 => 1/3
check("Y-Z", approx_eq(d_12, 1.0/3.0, 1e-6))
check("sym", approx_eq(d_01, d_10))

# ============================================================
#  hamming_dist
# ============================================================
print("=== hamming_dist ===")
d = hamming_dist(df_bin)
d_00 = d['A'][0]
d_01 = d['B'][0]
d_02 = d['C'][0]
d_10 = d['A'][1]
check("diag=0", approx_eq(d_00, 0.0))
# X=[1,0,1], Y=[1,1,0]: differ at pos 1,2 => 2/3
check("X-Y", approx_eq(d_01, 2.0/3.0, 1e-6))
# X=[1,0,1], Z=[0,1,1]: differ at pos 0,1 => 2/3
check("X-Z", approx_eq(d_02, 2.0/3.0, 1e-6))
check("sym", approx_eq(d_01, d_10))

# ============================================================
#  pearson_sim (on FastDataFrame)
# ============================================================
print("=== pearson_sim ===")
df_ps = make_df_int([('P', [1,2,3,4]), ('Q', [2,4,6,8])])
ps = df_ps.pearson_sim()
check("shape", ps.df_length == 2 and len(ps.columns) == 2)
# ps has columns P, Q with row index 0,1
# ps['P'][0] = corr(P,P) = 1.0
# ps['Q'][0] = corr(P,Q) = 1.0
# ps['P'][1] = corr(Q,P) = 1.0
# ps['Q'][1] = corr(Q,Q) = 1.0
check("diag=1", approx_eq(ps['P'][0], 1.0))
check("P-Q=1", approx_eq(ps['Q'][0], 1.0))

df_ps['R'] = make_int_col([4,3,2,1])
ps = df_ps.pearson_sim()
check("P-R=-1", approx_eq(ps['R'][0], -1.0, 1e-6))

# ============================================================
#  Edge cases
# ============================================================
print("=== Edge cases ===")
df_single = make_df([('A', [5.0])])
d = euclidean_dist(df_single)
check("single_row", d.df_length == 1 and approx_eq(d['A'][0], 0.0))

df_ident = make_df([('A', [3.0, 4.0]), ('B', [3.0, 4.0])])
d = cosine_similarity(df_ident)
check("identical_cosine", approx_eq(d['B'][0], 1.0))

df_zero = make_df([('A', [0.0, 1.0]), ('B', [0.0, 2.0])])
d = cosine_similarity(df_zero)
check("zero_vec_cosine", approx_eq(d['B'][0], 0.0))

# ============================================================
#  Summary
# ============================================================
print(f"\nResults: {passed} passed, {failed} failed")
if failed > 0:
    sys.exit(1)
