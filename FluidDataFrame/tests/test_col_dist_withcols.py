import math
import sys
sys.path.insert(0, '..')
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

def make_col(vals):
    col = DoubleColumn()
    for v in vals:
        col.append(float(v))
    return col

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

# ============================================================
#  Test data
# ============================================================
a = make_col([1.0, 2.0, 3.0])
b = make_col([4.0, 5.0, 6.0])
c = make_col([1.0, 0.0, 1.0])
d = make_col([1.0, 1.0, 0.0])
z = make_col([0.0, 0.0, 0.0])
same = make_col([3.0, 4.0, 5.0])

# ============================================================
#  Column-level: euclidean_dist
# ============================================================
print("=== Column euclidean_dist ===")
check("basic", approx_eq(a.euclidean_dist(b), math.sqrt(27)))
check("same", approx_eq(a.euclidean_dist(a), 0.0))
check("zero_vec", approx_eq(a.euclidean_dist(z), math.sqrt(14)))
check("same_as_b", approx_eq(a.euclidean_dist(same), a.euclidean_dist(same)))

# ============================================================
#  Column-level: manhattan_dist
# ============================================================
print("=== Column manhattan_dist ===")
check("basic", approx_eq(a.manhattan_dist(b), 9.0))
check("same", approx_eq(a.manhattan_dist(a), 0.0))
check("zero_vec", approx_eq(a.manhattan_dist(z), 6.0))

# ============================================================
#  Column-level: cosine_dist
# ============================================================
print("=== Column cosine_dist ===")
dot_ab = 1*4 + 2*5 + 3*6
norm_a = math.sqrt(14)
norm_b = math.sqrt(77)
expected_cos_dist = 1.0 - dot_ab / (norm_a * norm_b)
check("basic", approx_eq(a.cosine_dist(b), expected_cos_dist))
check("same", approx_eq(a.cosine_dist(a), 0.0))
check("zero_vec", approx_eq(a.cosine_dist(z), 1.0))

# ============================================================
#  Column-level: cosine_similarity
# ============================================================
print("=== Column cosine_similarity ===")
check("basic", approx_eq(a.cosine_similarity(b), dot_ab / (norm_a * norm_b)))
check("same", approx_eq(a.cosine_similarity(a), 1.0))
check("zero_vec", approx_eq(a.cosine_similarity(z), 0.0))
check("cosine_dist+sim=1", approx_eq(a.cosine_dist(b) + a.cosine_similarity(b), 1.0))

# ============================================================
#  Column-level: jaccard_sim
# ============================================================
print("=== Column jaccard_sim ===")
# c=[1,0,1], d=[1,1,0]: intersection=1, union=3 => 1/3
check("basic", approx_eq(c.jaccard_sim(d), 1.0/3.0))
# c=[1,0,1], c=[1,0,1]: intersection=2, union=2 => 1.0
check("same", approx_eq(c.jaccard_sim(c), 1.0))
# c=[1,0,1], z=[0,0,0]: intersection=0, union=2 => 0.0
check("zero_vec", approx_eq(c.jaccard_sim(z), 0.0))
# z=[0,0,0], z=[0,0,0]: intersection=0, union=0 => 0.0
check("both_zero", approx_eq(z.jaccard_sim(z), 0.0))

# ============================================================
#  Column-level: hamming_dist
# ============================================================
print("=== Column hamming_dist ===")
# c=[1,0,1], d=[1,1,0]: differ at pos 1,2 => 2/3
check("basic", approx_eq(c.hamming_dist(d), 2.0/3.0))
check("same", approx_eq(c.hamming_dist(c), 0.0))
# c=[1,0,1], z=[0,0,0]: differ at pos 0,2 => 2/3
check("zero_vec", approx_eq(c.hamming_dist(z), 2.0/3.0))

# ============================================================
#  Edge cases: different lengths (uses min)
# ============================================================
print("=== Edge cases: different lengths ===")
short = make_col([1.0, 2.0])
long_ = make_col([10.0, 20.0, 30.0])
# min_len=2: sqrt((1-10)^2 + (2-20)^2) = sqrt(81+324) = sqrt(405)
check("diff_len_eucl", approx_eq(short.euclidean_dist(long_), math.sqrt(405)))
# |1-10| + |2-20| = 9 + 18 = 27
check("diff_len_manh", approx_eq(short.manhattan_dist(long_), 27.0))

# ============================================================
#  Edge cases: single element
# ============================================================
print("=== Edge cases: single element ===")
s1 = make_col([5.0])
s2 = make_col([8.0])
check("single_eucl", approx_eq(s1.euclidean_dist(s2), 3.0))
check("single_manh", approx_eq(s1.manhattan_dist(s2), 3.0))
check("single_hamming", approx_eq(s1.hamming_dist(s2), 1.0))

# ============================================================
#  FastDataFrame euclidean_dist_withcols
# ============================================================
print("=== FastDataFrame euclidean_dist_withcols ===")
df = make_df([('x', [1, 2, 3]), ('y', [4, 5, 6]), ('z', [0, 0, 0])])

check("str_args", approx_eq(df.euclidean_dist_withcols('x', 'y'), math.sqrt(27)))
check("col_objs", approx_eq(df.euclidean_dist_withcols(df['x'], df['y']), math.sqrt(27)))
check("symmetric", approx_eq(df.euclidean_dist_withcols('x', 'y'), df.euclidean_dist_withcols('y', 'x')))
check("self", approx_eq(df.euclidean_dist_withcols('x', 'x'), 0.0))

# ============================================================
#  FastDataFrame manhattan_dist_withcols
# ============================================================
print("=== FastDataFrame manhattan_dist_withcols ===")
check("str_args", approx_eq(df.manhattan_dist_withcols('x', 'y'), 9.0))
check("col_objs", approx_eq(df.manhattan_dist_withcols(df['x'], df['y']), 9.0))
check("self", approx_eq(df.manhattan_dist_withcols('x', 'x'), 0.0))

# ============================================================
#  FastDataFrame cosine_dist_withcols
# ============================================================
print("=== FastDataFrame cosine_dist_withcols ===")
dot_xy = 1*4 + 2*5 + 3*6
nx = math.sqrt(14)
ny = math.sqrt(77)
expected_cd = 1.0 - dot_xy / (nx * ny)
check("str_args", approx_eq(df.cosine_dist_withcols('x', 'y'), expected_cd))
check("col_objs", approx_eq(df.cosine_dist_withcols(df['x'], df['y']), expected_cd))
check("self", approx_eq(df.cosine_dist_withcols('x', 'x'), 0.0))

# ============================================================
#  FastDataFrame cosine_similarity_withcols
# ============================================================
print("=== FastDataFrame cosine_similarity_withcols ===")
expected_cs = dot_xy / (nx * ny)
check("str_args", approx_eq(df.cosine_similarity_withcols('x', 'y'), expected_cs))
check("col_objs", approx_eq(df.cosine_similarity_withcols(df['x'], df['y']), expected_cs))
check("self", approx_eq(df.cosine_similarity_withcols('x', 'x'), 1.0))
check("dist+sim=1", approx_eq(
    df.cosine_dist_withcols('x', 'y') + df.cosine_similarity_withcols('x', 'y'), 1.0))

# ============================================================
#  FastDataFrame jaccard_sim_withcols
# ============================================================
print("=== FastDataFrame jaccard_sim_withcols ===")
check("self", approx_eq(df.jaccard_sim_withcols('x', 'x'), 1.0))

# ============================================================
#  FastDataFrame hamming_dist_withcols
# ============================================================
print("=== FastDataFrame hamming_dist_withcols ===")
check("self", approx_eq(df.hamming_dist_withcols('x', 'x'), 0.0))
# x=[1,2,3], z=[0,0,0]: all differ => 1.0
check("vs_zero", approx_eq(df.hamming_dist_withcols('x', 'z'), 1.0))

# ============================================================
#  Consistency: column-level == withcols
# ============================================================
print("=== Consistency: column-level == withcols ===")
check("eucl_consistent", approx_eq(
    df['x'].euclidean_dist(df['y']),
    df.euclidean_dist_withcols('x', 'y')))
check("manh_consistent", approx_eq(
    df['x'].manhattan_dist(df['y']),
    df.manhattan_dist_withcols('x', 'y')))
check("cosd_consistent", approx_eq(
    df['x'].cosine_dist(df['y']),
    df.cosine_dist_withcols('x', 'y')))
check("coss_consistent", approx_eq(
    df['x'].cosine_similarity(df['y']),
    df.cosine_similarity_withcols('x', 'y')))
check("jacc_consistent", approx_eq(
    df['x'].jaccard_sim(df['y']),
    df.jaccard_sim_withcols('x', 'y')))
check("hamm_consistent", approx_eq(
    df['x'].hamming_dist(df['y']),
    df.hamming_dist_withcols('x', 'y')))

# ============================================================
#  Summary
# ============================================================
print(f"\nResults: {passed} passed, {failed} failed")
if failed > 0:
    sys.exit(1)
