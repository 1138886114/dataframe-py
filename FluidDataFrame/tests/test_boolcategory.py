import sys
sys.path.insert(0, '.')
from fluidframe.core.intcolumn import IntColumn
from fluidframe.core.doublecolumn import DoubleColumn
from fluidframe.core.boolcolumn import BoolColumn
from fluidframe.core.categorycolumn import CategoryColumn
from fluidframe.core.fastdataframe import FastDataFrame

PASSED = 0
FAILED = 0

def check(condition, msg):
    global PASSED, FAILED
    if condition:
        PASSED += 1
    else:
        FAILED += 1
        print(f"  FAIL: {msg}")

# ==================== BoolColumn ====================
print("=== BoolColumn tests ===")

# Construction
bc = BoolColumn(4)
check(len(bc) == 0, "empty BoolColumn")
bc.append(True)
bc.append(False)
bc.append(True)
bc.append(True)
check(len(bc) == 4, "BoolColumn length")
check(bc[0] is True, "BoolColumn[0]")
check(bc[1] is False, "BoolColumn[1]")

# append_bool
bc2 = BoolColumn(4)
bc2.append_bool(False)
bc2.append_bool(True)
check(bc2[0] is False, "append_bool False")
check(bc2[1] is True, "append_bool True")

# pop_back
bc3 = BoolColumn(4)
bc3.append(True)
bc3.append(False)
bc3.append(True)
bc3.pop_back()
check(len(bc3) == 2, "pop_back length")
check(bc3[1] is False, "pop_back last element")

# concat
bc4 = BoolColumn(4)
bc4.append(True)
bc4.append(False)
bc5 = BoolColumn(4)
bc5.append(True)
bc5.append(True)
bc4.concat(bc5)
check(len(bc4) == 4, "concat length")
check(bc4[2] is True, "concat first of other")
check(bc4[3] is True, "concat second of other")

# copy
bc6 = BoolColumn(4)
bc6.append(True)
bc6.append(False)
bc7 = bc6.copy()
check(len(bc7) == 2, "copy length")
check(bc7[0] is True, "copy[0]")
check(bc7[1] is False, "copy[1]")

# __eq__ / __ne__
bc8 = BoolColumn(4)
bc8.append(True)
bc8.append(False)
bc9 = BoolColumn(4)
bc9.append(True)
bc9.append(False)
check(bc8 == bc9, "BoolColumn == True")
bc10 = BoolColumn(4)
bc10.append(True)
bc10.append(True)
check(bc8 != bc10, "BoolColumn != True")
check(bc8 != "not a column", "BoolColumn != non-column")

# __and__, __or__, __xor__
bc11 = BoolColumn(4)
bc11.append(True)
bc11.append(False)
bc11.append(True)
bc11.append(False)
bc12 = BoolColumn(4)
bc12.append(True)
bc12.append(True)
bc12.append(False)
bc12.append(False)
bc_and = bc11 & bc12
check(bc_and[0] is True, "True & True")
check(bc_and[1] is False, "False & True")
check(bc_and[2] is False, "True & False")
check(bc_and[3] is False, "False & False")

bc_or = bc11 | bc12
check(bc_or[0] is True, "True | True")
check(bc_or[1] is True, "False | True")
check(bc_or[2] is True, "True | False")
check(bc_or[3] is False, "False | False")

bc_xor = bc11 ^ bc12
check(bc_xor[0] is False, "True ^ True")
check(bc_xor[1] is True, "False ^ True")
check(bc_xor[2] is True, "True ^ False")
check(bc_xor[3] is False, "False ^ False")

bc_not = ~bc11
check(bc_not[0] is False, "~True")
check(bc_not[1] is True, "~False")

# sum / mean
bc13 = BoolColumn(4)
bc13.append(True)
bc13.append(False)
bc13.append(True)
bc13.append(True)
check(bc13.sum() == 3, "sum")
check(bc13.mean() == 0.75, "mean")

# isin
bc14 = BoolColumn(4)
bc14.append(True)
bc14.append(False)
bc14.append(True)
bc14.append(False)
bc_isin = bc14.isin({True})
check(bc_isin[0] is True, "isin True")
check(bc_isin[1] is False, "isin False")

# where_mask
mask = IntColumn(4)
mask.append(1)
mask.append(0)
mask.append(1)
mask.append(0)
bc15 = BoolColumn(4)
bc15.append(True)
bc15.append(True)
bc15.append(True)
bc15.append(True)
bc_wm = bc15.where_mask(mask, False)
check(bc_wm[0] is True, "where_mask keep")
check(bc_wm[1] is False, "where_mask fill")
check(bc_wm[2] is True, "where_mask keep 2")
check(bc_wm[3] is False, "where_mask fill 2")

# take_by_indices (tested via DataFrame)
bc16 = BoolColumn(4)
bc16.append(True)
bc16.append(False)
bc16.append(True)
bc16.append(False)

# drop_row
bc17 = BoolColumn(4)
bc17.append(True)
bc17.append(False)
bc17.append(True)
bc17.append(False)
bc17.drop_row(1)
check(len(bc17) == 3, "drop_row length")
check(bc17[0] is True, "drop_row [0]")
check(bc17[1] is True, "drop_row [1] (was [2])")
check(bc17[2] is False, "drop_row [2] (was [3])")

# ==================== CategoryColumn ====================
print("=== CategoryColumn tests ===")

# Construction
cc = CategoryColumn(4)
check(len(cc) == 0, "empty CategoryColumn")
cc.append("apple")
cc.append("banana")
cc.append("apple")
cc.append("cherry")
check(len(cc) == 4, "CategoryColumn length")
check(cc[0] == "apple", "CategoryColumn[0]")
check(cc[1] == "banana", "CategoryColumn[1]")
check(cc[2] == "apple", "CategoryColumn[2]")
check(cc[3] == "cherry", "CategoryColumn[3]")

# cate.codes
codes = cc.cate.codes
check(codes[0] == 0, "codes apple=0")
check(codes[1] == 1, "codes banana=1")
check(codes[2] == 0, "codes apple=0 again")
check(codes[3] == 2, "codes cherry=2")

# cate.categories
cats = cc.cate.categories
check(len(cats) == 3, "3 categories")
check(cats[0] == "apple", "category 0")
check(cats[1] == "banana", "category 1")
check(cats[2] == "cherry", "category 2")

# concat
cc2 = CategoryColumn(4)
cc2.append("banana")
cc2.append("date")
cc.concat(cc2)
check(len(cc) == 6, "concat length")
check(cc[4] == "banana", "concat [4]")
check(cc[5] == "date", "concat [5]")

# copy
cc3 = CategoryColumn(4)
cc3.append("x")
cc3.append("y")
cc4 = cc3.copy()
check(len(cc4) == 2, "copy length")
check(cc4[0] == "x", "copy[0]")
check(cc4[1] == "y", "copy[1]")

# __eq__ / __ne__
cc5 = CategoryColumn(4)
cc5.append("a")
cc5.append("b")
cc6 = CategoryColumn(4)
cc6.append("a")
cc6.append("b")
check(cc5 == cc6, "CategoryColumn ==")
cc7 = CategoryColumn(4)
cc7.append("a")
cc7.append("c")
check(cc5 != cc7, "CategoryColumn !=")
check(cc5 != "not a column", "CategoryColumn != non-column")

# isin
cc8 = CategoryColumn(4)
cc8.append("a")
cc8.append("b")
cc8.append("a")
cc8.append("c")
cc_isin = cc8.isin({"a"})
check(cc_isin[0] is True, "isin a")
check(cc_isin[1] is False, "isin b")
check(cc_isin[2] is True, "isin a")
check(cc_isin[3] is False, "isin c")

# where_mask
mask2 = IntColumn(4)
mask2.append(1)
mask2.append(0)
mask2.append(1)
mask2.append(0)
cc9 = CategoryColumn(4)
cc9.append("x")
cc9.append("x")
cc9.append("x")
cc9.append("x")
cc_wm = cc9.where_mask(mask2, "fill")
check(cc_wm[0] == "x", "where_mask keep")
check(cc_wm[1] == "fill", "where_mask fill")
check(cc_wm[2] == "x", "where_mask keep 2")
check(cc_wm[3] == "fill", "where_mask fill 2")

# drop_row
cc10 = CategoryColumn(4)
cc10.append("a")
cc10.append("b")
cc10.append("c")
cc10.append("d")
cc10.drop_row(1)
check(len(cc10) == 3, "drop_row length")
check(cc10[0] == "a", "drop_row [0]")
check(cc10[1] == "c", "drop_row [1]")
check(cc10[2] == "d", "drop_row [2]")

# ==================== DataFrame integration ====================
print("=== DataFrame integration tests ===")

# BoolColumn in DataFrame
df = FastDataFrame(schema={'A': 'int', 'B': 'bool'})
df = df.append({'A': 1, 'B': True})
df = df.append({'A': 2, 'B': False})
df = df.append({'A': 3, 'B': True})
check(df['B'][0] is True, "df bool col [0]")
check(df['B'][1] is False, "df bool col [1]")
check(df['B'][2] is True, "df bool col [2]")
check(df.df_length == 3, "df with bool col length")

# BoolColumn sum/mean
s = df['B'].sum()
check(s == 2, "df bool sum")
m = df['B'].mean()
check(abs(m - 2/3) < 1e-9, "df bool mean")

# CategoryColumn in DataFrame
df2 = FastDataFrame(schema={'C': 'category'})
df2 = df2.append({'C': 'apple'})
df2 = df2.append({'C': 'banana'})
df2 = df2.append({'C': 'apple'})
check(df2['C'][0] == "apple", "df cat col [0]")
check(df2['C'][1] == "banana", "df cat col [1]")
check(df2.df_length == 3, "df with cat col length")

# drop_index
df3 = FastDataFrame(schema={'A': 'int', 'B': 'str'})
df3 = df3.append({'A': 10, 'B': 'a'})
df3 = df3.append({'A': 20, 'B': 'b'})
df3 = df3.append({'A': 30, 'B': 'c'})
df3 = df3.append({'A': 40, 'B': 'd'})
df4 = df3.drop_index([1, 3])
check(df4.df_length == 2, "drop_index length")
check(df4['A'][0] == 10, "drop_index A[0]")
check(df4['A'][1] == 30, "drop_index A[1]")
check(df4['B'][0] == 'a', "drop_index B[0]")
check(df4['B'][1] == 'c', "drop_index B[1]")

# Single label
df5 = df3.drop_index(2)
check(df5.df_length == 3, "drop_index single length")
check(df5['A'][0] == 10, "drop_index single [0]")
check(df5['A'][1] == 20, "drop_index single [1]")
check(df5['A'][2] == 40, "drop_index single [2]")

# BoolColumn append with scalar fill in DataFrame
df6 = FastDataFrame(schema={'X': 'int', 'Z': 'bool'})
df6 = df6.append({'X': 1, 'Z': True})
df6 = df6.append({'X': 2, 'Z': False})
check(df6['Z'][0] is True, "df append bool col")
check(df6['Z'][1] is False, "df append bool col 2")

# ==================== _CategoryAccessor category management ====================
print("=== CategoryAccessor category management ===")

g = CategoryColumn(10)
for v in ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple']:
    g.append(v)

# codes returns a fast IntColumn copy
codes = g.cate.codes
check(isinstance(codes, IntColumn), "cate.codes is IntColumn")
check(list(codes) == [0, 1, 0, 2, 1, 0], "cate.codes values")

# categories returns a fresh list
cats = g.cate.categories
check(cats == ['apple', 'banana', 'cherry'], "cate.categories values")
cats.append('mutated')
check(g.cate.categories == ['apple', 'banana', 'cherry'], "cate.categories is a copy")

# rename_categories with dict (not inplace)
g2 = g.cate.rename_categories({'apple': 'fruit_a', 'banana': 'fruit_b'})
check(g2.cate.categories == ['fruit_a', 'fruit_b', 'cherry'], "rename dict categories")
check(list(g2.cate.codes) == list(g.cate.codes), "rename keeps codes")
check([g2[i] for i in range(len(g2))] == ['fruit_a', 'fruit_b', 'fruit_a', 'cherry', 'fruit_b', 'fruit_a'], "rename dict values")
check(g.cate.categories == ['apple', 'banana', 'cherry'], "rename not inplace preserves original")

# rename_categories with list
g3 = g.cate.rename_categories(['A', 'B', 'C'])
check(g3.cate.categories == ['A', 'B', 'C'], "rename list categories")
check([g3[i] for i in range(len(g3))] == ['A', 'B', 'A', 'C', 'B', 'A'], "rename list values")

# rename_categories duplicate guard
try:
    g.cate.rename_categories({'apple': 'banana'})
    check(False, "rename duplicate raises")
except ValueError:
    check(True, "rename duplicate raises")

# rename_categories list length mismatch
try:
    g.cate.rename_categories(['A', 'B'])
    check(False, "rename wrong-length list raises")
except ValueError:
    check(True, "rename wrong-length list raises")

# rename_categories inplace
g4 = CategoryColumn(6)
for v in ['apple', 'banana']:
    g4.append(v)
r = g4.cate.rename_categories({'apple': 'x'}, inplace=True)
check(r is None, "rename inplace returns None")
check(g4.cate.categories == ['x', 'banana'], "rename inplace mutated")

# add_categories
g5 = g.cate.add_categories(['date', 'elderberry'])
check(g5.cate.categories == ['apple', 'banana', 'cherry', 'date', 'elderberry'], "add_categories list")
g5.append('date')
check(g5[-1] == 'date', "add_categories allows append of new cat")
check(list(g5.cate.codes)[-1] == 3, "add_categories new cat code")
check('date' not in g.cate.categories, "add_categories not inplace preserves original")

# add_categories dedups existing
g6 = g.cate.add_categories(['apple', 'date'])
check(g6.cate.categories == ['apple', 'banana', 'cherry', 'date'], "add_categories dedups existing")

# set_categories to subset maps removed to ''
g7 = g.cate.set_categories(['apple', 'banana'])
check(g7.cate.categories == ['apple', 'banana', ''], "set_categories keeps '' placeholder")
check([g7[i] for i in range(len(g7))] == ['apple', 'banana', 'apple', '', 'banana', 'apple'], "set_categories removed map to ''")
check([g7[i] for i in range(len(g7)) if g7[i] != ''] == ['apple', 'banana', 'apple', 'banana', 'apple'], "set_categories kept cats intact")

# set_categories reorders codes
g8 = g.cate.set_categories(['cherry', 'apple', 'banana'])
check(g8.cate.categories == ['cherry', 'apple', 'banana', ''], "set_categories ordering")
check([g8[i] for i in range(len(g8))] == ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple'], "set_categories reorder values")

# set_categories inplace
g9 = CategoryColumn(4)
for v in ['apple', 'banana', 'apple', 'cherry']:
    g9.append(v)
r = g9.cate.set_categories(['apple'], inplace=True)
check(r is None, "set_categories inplace returns None")
check(g9.cate.categories == ['apple', ''], "set_categories inplace categories")
check([g9[i] for i in range(len(g9))] == ['apple', '', 'apple', ''], "set_categories inplace values")

# remove_categories
g10 = g.cate.remove_categories(['cherry'])
check(g10.cate.categories == ['apple', 'banana', ''], "remove_categories categories")
check([g10[i] for i in range(len(g10))] == ['apple', 'banana', 'apple', '', 'banana', 'apple'], "remove_categories values")

# remove_categories inplace
g11 = CategoryColumn(4)
for v in ['apple', 'banana', 'apple', 'cherry']:
    g11.append(v)
r = g11.cate.remove_categories(['apple'], inplace=True)
check(r is None, "remove_categories inplace returns None")
check(g11.cate.categories == ['banana', 'cherry', ''], "remove_categories inplace categories")
check([g11[i] for i in range(len(g11))] == ['', 'banana', '', 'cherry'], "remove_categories inplace values")

# codes on empty column
g12 = CategoryColumn(0)
empty_codes = g12.cate.codes
check(len(empty_codes) == 0, "cate.codes empty column")
check(g12.cate.categories == [], "cate.categories empty column")

# category management integrated with DataFrame schema
gdf = FastDataFrame(schema={'C': 'category'})
gdf = gdf.append({'C': 'apple'})
gdf = gdf.append({'C': 'banana'})
gdf = gdf.append({'C': 'apple'})
gdfr = gdf['C'].cate.rename_categories({'apple': 'AA'})
check(list(gdfr.cate.codes) == [0, 1, 0], "df cat rename keeps codes")
check(gdfr[0] == 'AA', "df cat rename value")

print(f"\nResults: {PASSED} passed, {FAILED} failed")
if FAILED > 0:
    sys.exit(1)
