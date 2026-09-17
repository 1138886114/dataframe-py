# FluidFrame 库架构设计文档

> 本文档基于对 `fluidframe/` 全部源码（Cython 编译单元 + Python 层）的静态审查编写。
> 代码规模：`_columns.pyx`(10,635 行) + `_databaseframe.pyx`(5,553 行) + `_columns_helper.pxi`(1,277 行) + `_databaseframe_helper.pxi`(6,563 行) + 两个 `.pxd` 接口文件(957 行)。
> 行号均指向当前工作树内的源文件。

---

## 一、概览

### 1.1 库定位

FluidFrame 是一个 **基于 Cython 实现的高性能 pandas 风格 DataFrame 库**。核心设计目标：

- **极致性能**：热路径全部在 C/nogil 环境中执行，64 字节对齐内存分配，为 SIMD 自动向量化铺路。
- **内存效率**：同 dtype 的数值列共享一个连续的 **列优先（column-major）2D 内存块**，列与列之间零拷贝共享存储（`create_view` 窗口）。
- **pandas 兼容 API**：提供 `iloc` / `loc` / `.dt` / `.str` / `.cat` / `groupby` / `rolling` / `metric` / `linalg` 等完整访问器模式。

### 1.2 架构组成

| 组成 | 文件 | 规模 | 职责 |
|---|---|---|---|
| 列存储层 | `core/_columns.pyx` + `.pxd` | 10,635 + 778 | 9 种列类型、底层 C 内核、对齐内存管理 |
| 列帮助层 | `core/_columns_helper.pxi` | 1,277 | 列层 fused 内核、日历/时间解析格式化、对齐内存分配 |
| 表结构层 | `core/_databaseframe.pyx` + `.pxd` | 5,651 + 193 | DataFrame、Series、Index 体系、访问器、GroupBy/Linalg/Metric |
| 表帮助层 | `core/_databaseframe_helper.pxi` | 6,433 | CRUD/聚合/排序/滚动/GroupBy 哈希/线性代数 的实现体 |
| Schema 层 | `_schema_types.py` | 87 | 类型常量、占位类型类、schema 解析 |
| 包入口 | `__init__.py` / `core/__init__.py` | — | 对外 API 面、Cython 模块别名映射 |

编译产物：`setup.py` 将上述 Cython 源编译为 2 个扩展模块：
- `fluidframe.core._columns`（9 种列类型 + 全部列内核）
- `fluidframe.core._databaseframe`（DataFrame 层；在文件末尾 `include "_databaseframe_helper.pxi"`）

编译标志：`-O3 -march=native -ffast-math -flto -funroll-loops -fno-math-errno`。

### 1.3 数据类型体系

9 种数据类型，由 `DataType` 枚举（`_columns.pxd:12-21`）与 `_schema_types.py` 双重定义（Python 侧常量）：

| 枚举值 | 常量 | Python 占位类 / 映射 | C 存储类型 | 物理含义 |
|---|---|---|---|---|
| 0 | TYPE_INT | `int` | `int32_t*` | 原始整数 |
| 2 | TYPE_DOUBLE | `float` | `double*` | IEEE-754 双精度 |
| 3 | TYPE_CHAR256 | `Char256` | `char*` + offset | 变长 ASCII 字节串（定长 256 字节单元） |
| 4 | TYPE_UCS2 | `str` | `Py_UCS2*` + offset | UCS-2 码元串（无 NUL 终止符） |
| 5 | TYPE_DATE | `Date` | `int32_t*` | 距 epoch 的天数（解析时支持 YYYYMMDD） |
| 6 | TYPE_BOOL | `bool` | `unsigned char*` | 0/1 |
| 7 | TYPE_CATEGORY | `Category` | `int*` codes + dict | 字典编码的类别 code |
| 8 | TYPE_DATETIME | `DateTime` | `int64_t*` | 距 epoch 的微秒数 |
| 9 | TYPE_TIME | `Time` | `int32_t*` | 距午夜的秒数（0..86399） |

schema 解析：`_parse_schema()`（`_schema_types.py:63-87`）对列名做 `sys.intern` 驻留（配合 `__getattr__` 的指针比较快路径），把 Python 类型 / 字符串（见 `_STRING_TYPE_MAP` `:45`）统一映射为类型常量。

### 1.4 内存布局核心设计

**两种协同的内存组织**：

1. **列级 1D 布局**：字符串/布尔/类别列每个占用独立缓冲区。
   - 定长类型：`int* data_c` / `double* data_c` / `int32_t* data_c`（Date/Time）/ `int64_t* data_c`（DateTime）。
   - 变长字符串：**双区结构** — 字符区 `data_c`/`chars_c` + 偏移数组 `offsets_c`。AsciiColumn 的字符串以 `\0` 结尾（`offsets[i+1]-offsets[i]-1` = 长度），UCS2Column 无终止符（`offsets[i+1]-offsets[i]` = 长度）。

2. **2D 列优先合并块**：`DataFrame.reserve_with_capacity`（`_databaseframe.pyx:2904`）将同 dtype 的固定宽度数值列（INT/DOUBLE/DATE/DATETIME/TIME）合并进一块连续内存：

```
逻辑视图 (3 rows × 2 cols):
  col_0: [a0, a1, a2], col_1: [b0, b1, b2]

物理内存 (column-major, 列优先):
  [a0, a1, a2, b0, b1, b2]
   ^-- col_0 占据 [0, r)   ^-- col_1 占据 [r, 2r)

扁平索引公式：元素 (row_i, col_j) 的位置 = j * rows + i
```

每一列通过 `create_view(block.data_c + j*capacity, 0, block)`（`_databaseframe.pyx:2939/2950/2961/2972/2983`）获得**零拷贝窗口**——视图持有 `_base_obj` 引用保住底层块存活，`owns_data=False` 禁止改写容量。

### 1.5 DataFrame 的 C 级指针映射（ColMeta）

`_c_cols`（`ColMeta*`，`_databaseframe.pxd:107-111`）是 DataFrame 把 Python 列对象物化为 C 结构的桥梁：

```c
cdef struct ColMeta:
    int dtype;               // 类型枚举
    void* data_ptr;          // 定长类型裸指针（变长类型为 NULL）
    void* col_obj;           // 回指 BaseColumn Python 对象
    Py_ssize_t* len_ptr;     // 指向列对象 .length 字段地址
```

在 `_databaseframe.pyx:3004-3028` 分配/填充。效果：
- `add_row` / `_fast_insert_val`（`_databaseframe_helper.pxi:265-301`）按 `dtype` 做类型分派，直接写裸指针，**零 Python 调用**。
- `len_ptr[0] = df_length` 一条语句同步所有列的长度（`_databaseframe.pyx:3088-3090`）。
- 失败时 `_rollback_pending_writes`（`:3092`）修复变长列长度。
- 表明 `reserve_with_capacity` 完成后设置 `_is_sealed=True` / `_fixed_capacity`，锁定内存，换取直接指针安全。

---

## 二、架构设计

### 2.1 分层结构

```
┌─────────────────────────────────────────────────────────────┐
│  Python 用户 API   (fluidframe/__init__.py)                  │
│   DataFrame  Series  to_numeric to_datetime ... 类型占位类  │
├─────────────────────────────────────────────────────────────┤
│  表逻辑层   _databaseframe.pyx + _databaseframe_helper.pxi  │
│   DataFrame / Series / Index体系 / iloc / loc               │
│   .str/.dt/.cat  .groupby  .rolling  .metric  .linalg      │
│   I/O: read_csv/tsv/json/pickle, concat/merge ...           │
├─────────────────────────────────────────────────────────────┤
│  列存储层   _columns.pyx + _columns.pxd + _columns_helper   │
│   BaseColumn(抽象) + 9 个具体列类                             │
│   对齐内存、1D/2D 双模式、全 C 内核（聚合/滚动/比较/astype）  │
├─────────────────────────────────────────────────────────────┤
│  C 层   aligned_malloc(64B) / nogil 内核 / qsort/bsearch     │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 类关系总图

```
BaseColumn (抽象, _columns.pyx:27)
 ├── IntColumn        (:105)   ── 2D 列优先支持
 ├── DoubleColumn     (:2399)  ── 2D 列优先支持
 ├── AsciiColumn      (:4655)  ── char256 双区存储
 ├── UCS2Column       (:6117)  ── UCS2 双区存储
 ├── DateColumn       (:7406)  ── 2D 列优先支持
 ├── BoolColumn       (:8531)
 ├── CategoryColumn   (:8947)  ── codes + categories dict
 ├── DateTimeColumn   (:9419)  ── 2D 列优先支持
 └── TimeColumn       (:10071) ── 2D 列优先支持

IndexBase (抽象, _databaseframe.pyx:265)
 ├── IntIndex        (:300)  — 包装 IntColumn
 ├── RangeIndex      (:384)  — start/step/length 惰性
 ├── MultiIndex      (:456)  — levels + codes(IntColumn) + 哈希缓存
 └── StringIndex     (:654)  — list[str] 标签

Series (:1741)  ──▶ (name, BaseColumn _col, IndexBase _index) 三视图
   └── 访问器: _StringAccessor(:1055) _DateTimeAccessor(:1551) _CategoryAccessor(:1599)

DataFrame (:2749)
 ├── 持有: _schema / _ordered_cols / _columns dict / _2d_blocks
 │         _c_cols(ColMeta*)、index、3 个访问器缓存
 ├── 索引器: _ILocIndexer(:707)  _LocIndexer(:849)
 └── 访问器: _DataFrameMetricAccessor(:4855)  _DataFrameLinalgAccessor(:4780)
             _DataFrameGroupByAccessor(:4461)  ← 依赖 _GroupHashTable/_GroupRows/_ByColPtrs
```

### 2.3 关键设计模式

1. **接口/实现分离**：cdef 类字段与方法契约在 `.pxd` 声明，`.pyx` 实现，`.pxi` 通过 `include` 文本拼接（Cython 两遍符号解析使后文定义可被前文调用）。
2. **虚拟方法双通道**：`BaseColumn` 提供 C 级虚函数 `_get_rows_c()` / `_get_data_ptr()`（`_columns.pyx:36/42`），允许 DataFrame 绕过 Python 强转直接拿裸指针。
3. **所有权协议**：`owns_data` 标记区分"自有缓冲"与"零拷贝视图"；视图 `create_view` 设 `owns_data=False` 同时持有 `_base_obj`。`_dealloc__` 仅当 `owns_data` 才 `os_aligned_free`；`_reserve` 拒绝视图列改写。
4. **2D 感知的一切**：数值/时间列在 append/pop/concat/drop_row/shift/take/聚合/滚动路径上全部依据 `is_2d` 分支走 `_col_shift_right_2d_*`、`_col_gather_2d_*`、`_col_concat_2d`（`_columns.pxd:167-272`）等行级辅助。
5. **惰性 + 缓存**：`RangeIndex.values` 惰性计算；`MultiIndex._hash_cache/_level_maps` 按需构建；GroupBy/Metric/Linalg 访问器缓存实例（`@cython.final @cython.freelist(16)`）。
6. **GIL 释放热循环**：聚合（Welford 方差）、排序 argsort、滚动、距离矩阵、groupby 哈希分组、linalg 全部在 `nogil` 下运行。

---

## 三、功能模块

### 3.1 列类型模块（`_columns` 扩展）

每个列类都提供一族统一的 API 曲面（以 IntColumn/DoubleColumn 为最完整）：

- **构造/工厂**：`__cinit__`、静态 `create_view`、`from_2d`、`_load_from_1d_array` / `_load_from_2d_array`（numpy memoryview 零拷贝导入，见测试 test_zero_copy.py）。
- **增长/变异**：`append`、`pop_back`、`pop_front`、`concat`、`drop_row`、`shift_left/right`、`_reserve`、`copy`。
- **聚合/统计**：`max/min/sum/mean/var/std/median/mad/prod/count/idxmax/idxmin/skew/kurt/sem/quantile/_describe/_nunique/_value_counts/mode`。
- **元素级**：`abs`、`round`（舍入一半到偶数）、`ffill`/`bfill`、`clip`、`diff`、`_pct_change`、`cumsum/cumprod/cummax/cummin`。
- **比较/掩码**：`_cmp_op` → 6 种比较（返回 0/1 IntColumn，非 BoolColumn）、`isin`（qsort+bsearch）、`between`、`where/where_mask/mask`、`__and__/__or__/__invert__`。
- **选择**：`__getitem__`、`take_by_indices`、`unique`、`gather_with_sentinel`、`gather_zero_to`。
- **算术**：`add/sub/mul/div`（列×列/标量，int 除法提升为 DoubleColumn）、dunder `__add__...__rtruediv__`。
- **类型转换**：`astype(target)` 全套 9 向转换（`_resolve_astype_target` 解析）。
- **距离/相似度**：`euclidean_dist / manhattan_dist / cosine_dist / cosine_similarity / jaccard_sim / hamming_dist`。
- **滚动窗口**：`roll_mean/sum/min/max/var/std/median/quantile/skew/kurt/corr/cov/_roll_apply_weights/_roll_custom/_roll`。
- **对象协议**：`__getitem__/__str__/__bool__/__len__/__contains__`。

### 3.2 表逻辑模块（`_databaseframe` 扩展）

四种职责：

1. **Index 体系**（定位/取行底层）：`IndexBase._take/_concat` 契约；`IntIndex`、`RangeIndex`（O(1) 数学 `get_loc`）、`MultiIndex`（全键 O(1) 哈希、前缀键 C 扫描）、`StringIndex`。
2. **索引器**：`_ILocIndexer.__getitem__`（int/slice/list/tuple(rows,cols) 分派，临时 `Py_ssize_t*` → `frame._take`）；`_LocIndexer`（`__getitem__` + `__setitem__`，标签经 index.get_loc → iloc；布尔掩码双趟计数+收集）。
3. **Series 视图**：三字段视图（name, _col, _index）+ 访问器（.str/.dt/.cat）+ 全套 pandas 风格方法。
4. **DataFrame**：见 3.3。
5. **模块级函数**：`concat_outer/inner/concat`、`to_numeric/to_datetime/to_date/to_time/to_timedelta`、`read_csv/read_tsv/read_json/from_pickle`、`where`。

### 3.3 DataFrame 功能矩阵

| 类别 | 方法 |
|---|---|
| 构造/工厂 | `__init__`、`reserve_with_capacity`、`from_dict/from_list/from_arrays`、`arange/zeros/ones/randn/rand` |
| 行操作 | `add_row`、`add_rows_from_list`、`append`、`insert_row`、`pop_front/back`、`drop`、`drop_with_index`、`drop_index`、`drop_raw`、`head/tail`、`sample`、`nsmallest/nlargest`、`duplicate_rows`、`duplicated`、`drop_duplicates` |
| 列操作 | `__getitem__/__setitem__/__delitem__/__getattr__/__dir__`、`get_series`、`insert/insert_column`、`duplicate_column`、`rename`、`select_dtypes`、`astype`、`get_dummies` |
| 聚合 | `mean/median/sum/min/max/std/var/quantile/mode/prod/sem/mad/count/describe/value_counts/nunique/skew/kurt/argmax/argmin/nunique` + `*_withcol` 系列（追加列不整表返回） |
| 累积 | `cumsum/cumprod/cummax/cummin` |
| 排序/排名 | `argsort_by`、`sort_values`、`sort_index`、`rank`、`cut`、`qcut` |
| 过滤/清洗 | `eval/query`、`where_column/where_row/mask_column`、`isna/notna/isnull`、`fillna`、`dropna`、`replace`、`ffill/bfill`、`clip/clip_column`、`abs/round/pct_change` |
| 变换/重塑 | `apply/applymap/pipe/assign`、`reshape`、`explode`、`stack/unstack`、`T/transpose`、`pivot`、`melt` |
| 行/列级函数 | `add/sub/mul/div`、`diff/diff_all`、`shift`、`rolling(+win_type)`、`expanding`、`ewm`（旧 `expanding_sum`/`ewm_mean` 保留兼容） |
| 二元/连接 | `concat`、`merge` |
| 指标 | `euclidean_dist_withcols / manhattan / cosine_dist / cosine_similarity / jaccard / hamming`、`corr/corrwith/corrcoef/cov/pearson_sim`、`euclidean_dist_matrix...` |
| 线性代数 | `dot/matmul/det/inv/solve/norm` |
| 输出 | `to_string/to_html/to_csv/to_tsv/to_excel/to_pickle/to_json/to_dict/to_numpy/to_list/to_pandas`、`print/info/memory_usage` |
| 索引操作 | `reset_index/set_index/reindex`、`loc/iloc` |

### 3.4 I/O 层

- CSV/TSV：`to_csv/to_tsv` + `read_csv/read_tsv`（支持 schema 指定与自动探测）。
- JSON：`to_json(orient_records)` + `read_json`。
- Pickle：`to_pickle` 先分解为 "schema + 每列列表" 字典再序列化；`from_pickle` 使用 **`_SafeUnpickler`**（`_databaseframe.pyx:4937`，白名单 `find_class`，`allow_raw_pickle=True` 时放开）。2D 列以 `(typ, values, rows, cols)` 四元组存活。
- 没有实现 buffer 协议 / `__reduce__`；零拷贝只存在于列导入（`_load_from_*_array`）与 DataFrame 内部块视图。

---

## 四、模块详情

### 4.1 BaseColumn（抽象基类，`_columns.pyx:27-98`）

只读元数据：`length / capacity / dtype / is_2d / num_cols / owns_data`（cdef readonly）。C 级虚函数 `_get_rows_c()`（逻辑行数 = `length // num_cols`）与 `_get_data_ptr()`（裸指针）。抽象方法 `copy/drop_row/shift_left/abs/round/ffill/bfill`（子类必须实现）。`get_size()` 返回内存字节数；`__contains__` 做成员检测。

### 4.2 IntColumn（`_columns.pyx:105-2398`，@final）

- **布局**：`int* data_c`（64 字节对齐），`object _base_obj`。`is_2d = num_cols>1`。
- **2D 行内核**（`:992-1038`）：`_sum_rows/_mean_rows/_min_rows/_max_rows/_count_rows/_var_rows/_std_rows`，供 FastDataFrame 分组/聚合直取。
- **零拷贝导入**：`_load_from_1d_array`（`int[::1]` C 连续视图）、`_load_from_2d_array`（Fortran 连续 `int[::1,:]` 逐列拆 view）。
- **astype**（`:1891`）9 向转换：如 →DOUBLE nogil 扩宽、→UCS2/CHAR256 走 `snprintf`、→DATE `_ymd_to_days`、→DATETIME 微秒化、→TIME、→CATEGORY。
- 滚动 12 种、距离 6 种、比较 6 种、掩码代数、qsort/bsearch 的 `isin`。

### 4.3 DoubleColumn（`_columns.pyx:2399-4654`，@final）

- **布局**：`double* data_c`（对齐），`_base_obj`，同 2D 双模式。
- `_make_like_nan`（`:2448`）为滚动运算制造同形 NaN 填充列。
- **NaN 语义**：`_c_max/_c_min/_c_sum/_c_prod` 跳过 NaN；`_cmp_op` eq 时双 NaN 视为相等；`take_by_indices` 越界填 NaN 走 `_col_gather_nan`。
- 滚动（`:4262` 起）含 `roll_corr/roll_cov` 增量实现与 `_roll_apply_weights`（FIR/加权平均 nogil 内积）。
- 数值内核全部来自 `_columns_helper.pxi` 的 fused 类型函数（`_n_*`）。

### 4.4 AsciiColumn（`_columns.pyx:4655-6116`，char256）

- **布局**：`char* data_c`（字节竞技场）+ `Py_ssize_t* offsets_c`；**NUL 结尾**。`byte_capacity / byte_length` 双容量字段。`_reserve_rows`/`_reserve_bytes` 分别伸缩两区。
- 双缓冲一致性：`append_nolen/undo_nolen` 支持"未提交写入"（DataFrame add_row 回滚用）。
- **字符串运算**（返回 AsciiColumn 或 IntColumn 掩码）：`contains(startswith/endswith`（C memcmp 级）、`replace/replaces`、`extract/extract_twice`、`len`、`find/rfind/rindex/count`、`match/fullmatch`（正则）、`isdigit/isalpha/isalnum/isspace/islower/isupper/isnumeric/isdecimal`、`upper/lower/title/capitalize/swapcase/casefold`、`strip/lstrip/rstrip/removeprefix/removesuffix`、`center/ljust/rjust/pad/zfill`、`split/rsplit/partition/rpartition`（返回 list）、`get/slice`、`cat/prefix/suffix/repeat`、`findall/extractall`。
- 例外：`str_contains_mask`（`:6094`）走 Python `re.compile` 一次 + 逐行 search，返回 0/1 IntColumn。
- `_cmp_op` 用 `strcmp` 直接在 NUL 结尾缓冲对碰。

### 4.5 UCS2Column（`_columns.pyx:6117-7405`）

- **布局**：`Py_UCS2* chars_c`（**无 NUL**，长度由 offsets 差分求得）+ `offsets_c`；`char_capacity/char_length`。`_get_str` 用 `PyUnicode_FromKindAndData(2,...)` 零拷贝构造 Python str。
- **UCS2 C 内核**（`_columns_helper.pxi:431-463`）：`_ucs2_strstr`（双指针子串）、`_ucs2_startswith/endswith`、`_cmp_ucs2_buffers`（前缀长比较，无 NUL 依赖）。
- 与 AsciiColumn 同名的字符串 API 曲面；需要 Unicode 变换的部分（title/capitalize/casefold/strip 等）逐行解码后 Python 处理。

### 4.6 DateColumn（`_columns.pyx:7406-8530`）

- **布局**：`int32_t* data_c`，语义为 **epoch 天数**；2D 支持。
- 日历元：`_ymd_to_days` / `_days_to_ymd`（`_columns_helper.pxi:525/532`）。
- dt 字段提取：`year/month/day/quarter/day_of_week/day_of_year/days_in_month/is_leap_year/is_weekend/is_month_start/is_month_end/is_year_start/is_year_end/is_quarter_start/is_quarter_end/week_of_year/day_name/month_name/last_day_of_month/iso_calendar` + `strftime`。
- 日期算术：`add_sub_days/months/years`、`diff_days/months/years`。
- 过滤：7 个 `filter_*` 系列 + `where_mask`。

### 4.7 BoolColumn（`_columns.pyx:8531-8946`）

- **布局**：`unsigned char* data_c`（1 字节/元素），`_owns_data` 影子字段 + `_base_obj`；**无 2D**。
- 位运算：`__and__/__or__/__xor__/__invert__`；`isin`、`where_mask`、`take_by_indices`、`count_nonzero`、`gather_zero_to`。

### 4.8 CategoryColumn（`_columns.pyx:8947-9418`，@final）

- **布局**：`int* codes_c` + Python `list categories` + `dict _cat_to_code`；无 2D。
- `_get_or_add_code` 建码；`_resolve(i)` code→str；`.cat` 访问器操作全部落在 `_CategoryAccessor` 上恢复码映射。
- concat 优化：预建 O(K) 码映射表后一次性合并。

### 4.9 DateTimeColumn / TimeColumn（`_columns.pyx:9419-10070 / 10071-10635`）

- **布局**：`int64_t* data_c`（epoch 微秒）/ `int32_t* data_c`（秒）；均支持 2D。
- dt 提取：DateTime 有 `year/month/day/hour/minute/second`；Time 只有 `hour/minute/second`。`_DT_ATTRS` 白名单（`_databaseframe.pyx:49-56`）中的其余字段仅在 DateColumn 上可用，否则 `_DateTimeAccessor.__getattr__` 抛 `AttributeError`。
- `_resolve_dt` / `_resolve_time`：字符串经 `_parse_datetime_str` / `_parse_time_str`，数字直接 cast。
- astype：Time→DateTime 乘 `1_000_000`，DateTime→Date 整数整除 `US_PER_DAY` 等纯算术转换。

### 4.10 共享列内核（`_columns_helper.pxi`）

- **对齐分配**：`aligned_malloc/realloc(alignment=64)`、`os_aligned_free`（`:140-161`）。
- **比较器**：`_cmp_int(76)`、`_cmp_double(174, NaN 全序)`、`_cmp_double_asc(187)`、`_cmp_int64(665)`、`_cmp_int32(1175)`；`_binary_search(83)`。
- **统计内核**：`_n_quickselect(776)` / `_n_median(806)` / `_n_quantile(825)` / `_n_central_moments(846)` / 滚动族 `_resolve_min_periods(893)`·`_roll_win_bounds(907)`·`_n_roll_sum/mean/var/skew/kurt/quantile/min/max/corr/cov(920-1345)`（count-aware，右对齐/居中，`_copy_int_to_double`/`_int_col_to_double` 支持 Int×Double 交叉）/ `_n_cumsum/cumprod/cummax/cummin(1346+)` / `_n_diff(1407)` / `_n_clip(1434)` / `_n_add_val/sub_val/mul_val/div_val(1448+)`。
- **通用向量化**：`_col_to_double`（任意数值/时间列→double 缓冲，`:201`）、`_n_cmp_scalar/_n_cmp_col`（分支无关比较 → 0/1 掩码）、`_n_where_mask`（广播掩码填充）、`_n_isin`（qsort+bsearch）、`_n_unique`。
- **日历/时间**：闰年/月天数表（`:507-518`）、`_ymd_to_days/_days_to_ymd`、`_parse_date_str(546)/_parse_datetime_str(579)/_parse_time_str(729)`、零分配 ISO 格式化 `_format_datetime_c(631)/_format_time_c(705)`。

### 4.11 Index 体系（`_databaseframe.pyx:265-706`）

| 类 | 行号 | 内部表示 | get_loc 复杂度 |
|---|---|---|---|
| IntIndex | 300 | `IntColumn data` | O(n) 线性扫 |
| RangeIndex | 384 | `(start, step, length)` 惰性 | O(1) 数学判定 |
| MultiIndex | 456 | `levels`(list of tuple) + `codes`(list[IntColumn]) + `names` | 全键 O(1)；前缀键 O(n) C 扫 |
| StringIndex | 654 | `labels` list[str] | O(n) 线性扫 |

**MultiIndex 哈希黑科技**（`:467-506`）：把各层 `IntColumn.data_c` 指针取出，逐行把 nlevels 个 4 字节 code 拷贝进临时 `int* buf`，然后直接以 **裸内存字节流 `(<char*>buf)[:bytes_len]` 作为 dict 键** 建 `_hash_cache`（`bytes -> [row_ids]`），零 Python 对象构建开销。查找时同样拼接目标键字节流一次 dict 命中。

`IndexBase._take` / `_concat` 契约：所有 Index 都实现按索引数组取行、与另一 Index 串联（RangeIndex 连续时返回惰性 RangeIndex，否则物化为 IntIndex）。

### 4.12 索引器

**`_ILocIndexer`**（`:707`）：纯位置索引。float 情况已排除（int 分支负号归一 + 越界检查 `:727-735`）；slice 用 `key.indices()` 展开；list 逐元素负号归一；tuple 两元素（rows, cols），列选择接受 str/int；最终统一走 `frame._take(indices_c, n)` 或单列 `Series`。

**`_LocIndexer`**（`:849`）：标签索引。
- `__setitem__`（`:854`）：仅 int 行标签，list/dict 分派逐 cell 写 `_fast_set_cell`。
- `_get_rows`（`:919`）：callable / Series 掩码 / 裸 IntColumn+BoolColumn / bool list / 其它标签五路分派。掩码路径双趟处理（计数 + 收集）后 `frame._take`。
- `_apply_mask`（`:954`）要求掩码 dtype ∈ {INT, BOOL} 且长度一致；`_apply_bool_list`（`:1013`）处理 Python bool list。

### 4.13 Series（`_databaseframe.pyx:1741-2676`，@final）

- **布局**：`(str name, BaseColumn _col, IndexBase _index)` 纯视图——"DataFrame._columns 永远存裸 BaseColumn；Series 是按需构建、用完即可丢弃的临时包装"（`_databaseframe_helper.pxi:11-14`）。
- `__getattr__`（`:1799`）：列方法直通引擎——`getattr(col, name)` 得到方法后闭包转发，结果若为 BaseColumn 则回包 Series（索引兼容时保留原 index，否则新建 RangeIndex）。
- 算术（`:1962-2030`）：列运算自动 index 对齐——`_index is other._index` 时才直接结合，否则先 `other.reindex(self._index)`。
- 统计 22 种、变换 15 种、掩码 8 种、`map/factorize/apply/replace/where/mask`、`reindex/reset_index`、`rolling(window, min_periods=None, center=False, win_type=None)`/`expanding(min_periods=1)`/`ewm(com/span/halflife/alpha, ...)` 返回对应 cdef 访问器（`_databaseframe.pxd` 声明，`pyx` 实现，Series/DataFrame 双模式，委托列 `roll_*`/`ewm_*`）。
- 补齐的 pandas 对齐 API：`shape`、`to_numpy`（别名 `values`）、`shift(periods, fill_value)`（委托 `_shift_column_impl`，Int 空位 0 / 其余 NaN，fill_value 覆盖）、`rank(method/ascending/na_option)`（复用 `_rank_col`/`_rank_row_keys`）、`describe(percentiles)`（数值列 count/mean/std/min/25/50/75/max）。
- 导出：`to_list/tolist/to_dict/to_pandas/to_frame`。
- 访问器属性：`.str`（要求 char256/ucs2）、`.dt`（date/datetime/time）、`.cat`（category），类型不符抛 AttributeError。

### 4.14 访问器

**`_StringAccessor`**（`:1055`，@final）：纯分派器——按住 `_col.dtype` cast 到 AsciiColumn/UCS2Column 转发，结果经 `_wrap` 回包 Series。覆盖搜索 / 类型检查 / 大小写 / strip / replace / split / 切片 / cat 共约 45 个方法。`replace(old, new, count)` 的 `count` 已透传 AsciiColumn/UCS2Column 内核（`str.replace(old,new,count)` 语义，UCS2 为原地修改返回 self）。

**`_DateTimeAccessor`**（`:1551`，@final）：`__getattr__` 动态引擎——名字必须在 `_DT_ATTRS` 白名单内，`getattr(col,name)` 得到零参方法后 `_wrap_series` 包裹；显式别名 `weekday/dayofweek/dayofyear/weekofyear` + `strftime(fmt)`。

**`_CategoryAccessor`**（`:1599`，@final）：`codes`（深拷贝 IntColumn）、`categories`（list）、`rename_categories`、`add_categories`、`remove_categories`、`set_categories`（`cat_To_code_map` 映射表 + **nogil 逐码重映射** `:1708`）、`as_ordered/as_unordered`。

### 4.15 DataFrame（`_databaseframe.pyx:2749-4420`，@final）

- **字段**（`_databaseframe.pxd:143-159`）：`_schema/_schema_keys/_ordered_cols/_2d_blocks/_columns/_fixed_capacity/df_length/_is_sealed/_num_cols/_c_cols/index` + 三个访问器缓存。
- **构造管线**（`__init__:2794` → `reserve_with_capacity:2904`）：按 dtype 分两组（固定宽数值 vs 变长/布尔/类别），为每种数值类型发一块列优先合并块，逐列 `create_view` 窗口，并填充 `_c_cols`；最后封存内存。
- **add_row 极限推流**（`:3034`）：位置/关键字两用，逐列 `_fast_insert_val` 写裸指针，成功后 `df_length+=1` 并同步全部 `len_ptr`；失败 `_rollback_pending_writes`。
- **排序**：C 层 quicksort argsort 内核 `_c_argsort_numeric(pxi:149)/_c_argsort_char256(pxi:89)/_c_argsort_ucs2(pxi:218)`，纯 nogil；2D 列退化为 Python key `_col_row_key`。
- **rolling**：`df.rolling(window, min_periods=None, center=False, axis=0, win_type=None)` 返回 `_RollingWindow`（`_databaseframe.pxd` 声明的 cdef 双模式访问器，实现 `pyx` 于 `_RollingWrapper` 旧址 `:192` 起，`DataFrame.rolling` 入口 `~:4556`）——C 核 `_n_roll_*` 滑动聚合并装回整表 DataFrame（非数值列跳过、无数值列 TypeError、`axis!=0/‘index’→NotImplementedError`）；`min_periods/center` 提前在 `_roll_win_bounds` 计算边界，count-aware 核天然支持部分窗口。
- **expanding**：`df/s.expanding(min_periods=1, axis=0)` = `_RollingWindow(obj, max(r, min_periods), min_periods, False, ...)`（核内 `n=min(i+1,window)` → expanding 计数；`mp>r` 全 NaN 不误报）；≡ pandas `expanding ≡ rolling(window=行数)`，全算子等价已验证。
- **ewm**：`df/s.ewm(com/span/halflife/alpha, min_periods=0, adjust=True, ignore_na=False)` 返回 `_EWMWindow`（`pxd` 声明，`pyx` 实现于 `_RollingWindow` 之后）；参数解析 `_resolve_ewm_alpha`（com/span/halflife→alpha，互斥/缺参错误串与 pandas 一致），委托列级 `ewm_mean/sum/var/std/cov/corr`，C 核 `_n_ewm`/`_n_ewmcov`（`_columns_helper.pxi`，逐列递推、无 times 等间隔假设）；var/std 用 `bias`（对齐 pandas，非 ddof），sum+adjust=False → `NotImplementedError`。
- **win_type（boxcar/triang）**：挂在 `_RollingWindow`（`_win_type/_weights` 字段），仅放行 `mean/sum/std/var`（其余 `AttributeError('Window' object has no attribute ...')`），`min_periods>window→ValueError`；权重纯公式生成（triang 奇窗 `1-|2i-(M-1)|/(M+1)` / 偶窗 `1-|2i-(M-1)|/M`，= scipy `get_window(...,fftbins=False)`）；内核 `_n_roll_wmean_sum`（窗口右对齐，最新值吃末权重）与 `_n_roll_wvar`（pandas West 循环对齐 quirk：绝对位置 `p%win_n`），center 用尾部补 NaN 右对齐再截断等价写法。
- **GroupBy**：见 4.16。
- **Linalg/Metric**：见 4.17/4.18。
- 快速设置器 `_fast_set_cell`（`:310`）、列构造 `_column_from_list_impl`（`:397`）、astype 解析 `_astype_impl`（`:2041`）。
- 补齐的 pandas 对齐 API：`values`（`_to_numpy_impl` 列优先拼装 2D ndarray）、`size`（rows×cols）、`ndim`（=2）、`get(key, default)`、`idxmax/idxmin(col_name=None)`（单列标量或全列 Series，2D 抛错）、`astype` 支持 bare str（如 `'float'`）对全列一次性转换；`_column_from_list_impl` 对 2D 列转 1D 时按列优先重排。

### 4.16 GroupBy 引擎（`_DataFrameGroupByAccessor` :4461 + pxi 5190-5895）

- C 结构：`_GroupHashTable`（开地址线性探测，`_ght_get_or_insert`）、`_GroupRows`（动态 `Py_ssize_t**` 行索引表）、`_ByColPtrs`（void** data + offsets + types + cols + strides，2D 子列 `data_c[sub*stride+row]`，1D stride=length）。
- 分组构建 `_gb_build_groups`（`pxi:5542`）：逐行 `_gb_hash_row`（FNV-1a 混合）→ 哈希槽命中后 `_gb_rows_equal` 逐类型校验，不等则线性扫已有组；O(1) 平均建组。
- 聚合：`pxi` 提供 `_gb_native_agg/_gb_native_count/_gb_native_idx/_gb_python_agg` 四类发动机（原生 6 种 sum/mean/min/max/var/std + count + idxmax/idxmin + Python 回退 median/quantile 等）。
- **C 级聚合分派抽象**（`pyx:4480-4527`）：访问器内共享 `_result_shell`（骨架 DataFrame + 分组键列挂载 `_attach_keys`）、`_target_cols`（非键数值列枚举）、`_dispatch_agg`（函数名 → 四类发动机的单点映射）。`agg/_convenience_agg/sum.../count` 全部走 `_dispatch_agg`；`idxmax/idxmin` 收敛为 `_idx_agg(col_name, find_max)` 同一 C 壳，消除了 5 段逐方法重复的分组键挂载与数值列枚举代码。
- 对外：`agg/apply/transform/filter/size/sum/mean/count/min/max/median/quantile/var/std/idxmax/idxmin`。`@cython.final @cython.freelist(16)`。

### 4.17 Linalg 访问器（`_databaseframe.pyx:4766`）

`dot/matmul/transpose/det/inv/solve/norm(ord_type)` + 两列距离（euclidean/manhattan/cosine/hamming/jaccard/cosine_similarity）。实现核心：`_linalg_to_contiguous_fast`（`pxi:5873`，把数值列专升本成 `double*` 连续阵，2D 感知）、`_linalg_from_contiguous`（`pxi:5969`，由连续缓冲重建 DataFrame）、`_linalg_det/inv/solve/norm`（`pxi:6083/6138/6207/6283`）。

**配对矩阵 C 级分派抽象**（`_linalg_pairwise`，`pxi:6328`）：6 个距离/相似度矩阵（euclidean/manhattan/cosine_dist/cosine_sim/hamming/jaccard）原本是 6 份 ~20 行、仅内核与对角值不同的复制；现由共享枚举 `_PairwiseOp`（`_databaseframe.pxd`）驱动单份脚手架——一次 `aligned_malloc(rows*rows*8)`、一次 `_linalg_to_contiguous_fast`。**循环不变式提升**：`op` 分派只执行一次（抬到行循环之外），每个 op 各有一份无分支的内层填充循环，内核保持 `cdef inline` 以便 `-O3` 自动 SIMD 向量化。对角线值按 op 归并（sim/jaccard=1.0，其余=0.0），全部在 nogil 下写对称三角。

### 4.18 Metric 访问器（`_databaseframe.pyx:4841`）

`euclidean/manhattan/cosine_dist/cosine_similarity/jaccard/hamming/corr/cov/corrwith/corrcoef/cov_withcols/pearson`——全部薄包装 `_*_impl`。列间距离/相关在 DataFrame 上还提供 `*_withcols` 与 `*_dist_matrix` 整表接口。

**Metric 区 C 级分派抽象**（`pxi:4288-4500`）：
- 6 个双列距离 `_*_dist_withcols_impl` 合并为 `_metric_dist_withcols_impl(self, col1, col2, op)`——先统一 `_ensure_double_1d` 安全萃取，再按 `_PairwiseOp` 只切换列核调用。
- `_corr_impl`/`_cov_impl` 合并为 `_corr_cov_impl(self, is_corr)`（`pxi:4304`）：共享一次 malloc 特征矩阵萃取 + `DataFrame()` 视图回装；**`is_corr` 判定已抬到 nogil 循环之外**，corr/cov 各为独立的无分支 `with nogil` 填充块，仅对角线（corr=1.0 / cov=各列自协方差）与核（`_vec_pearson_corr`/`_vec_pearson_cov`）两处分叉；`_pearson_sim_impl` 即 `_corr_cov_impl(True)`。
- `_corrcoef_impl`/`_cov_withcols_impl` 合并为 `_corrcoef_cov_withcols_impl(self, col_a, col_b, is_corr)`（`pxi:4403`），仅核不同，其在 DataFrame 的命名 `cov_withcols` 与 Metric 访问器 `cov_withcols` 保持同名薄壳。

### 4.19 模块级公共函数（`_databaseframe.pyx:4923-5651`）

- `_SafeUnpickler`（`:4923`）：限制 `find_class` 白名单。
- `read_csv/tsv/json`、`from_pickle`。
- `concat_outer/concat_inner/concat`（忽略 index 选项）。
- `to_numeric/to_date/to_datetime/to_time/to_timedelta`（errors 支持 'coerce'）。
- `where(IntColumn cond, true, false)`：纯 C 循环写 double 结果。

---

## 五、与 pandas 的对照要点

| 特性 | pandas | FluidFrame 实现 |
|---|---|---|
| 内存 | BlockManager / Arrow | 列优先 2D 合并块 + 零拷贝视图 |
| 字符串 | object/PyArrow | 双区 char256 / UCS2 专用列类 |
| 日期 | datetime64[ns] | 定制 int32 day 纪元 / int64 微秒纪元 |
| 类别 | Categorical | int codes + dict 字典 |
| 索引 | Index/MultiIndex | 4 种 Index 子类，MultiIndex 位级哈希缓存 |
| 分组 | 哈希 → aggregate | 开地址哈希表 + 行索引数组 + FNV-1a |
| 二进制运算 | 对齐 → 广播 | 视图层 index 对齐（Series 级别） |
| 零拷贝 | copy-on-write / NumPy | `create_view` 窗口 + numpy memoryview 导入 |

## 六、已知边界 / 注意事项

1. **视图列不能改容量**：`create_view` 列 `owns_data=False`，`append/concat/drop_row` 会抛错（测试 test_zero_copy.py 验证）。
2. **`.dt` 能力依列型裁剪**：`_DT_ATTRS` 里的方法只对实现了对应方法的列类型可用（Date 最全；DateTime 仅 y/m/d/h/m/s；Time 仅 h/m/s）。
3. **`_StringAccessor.replace` 的 count 参数**目前未透传给列实现。
4. **二进制 DataFrame 运算缺失**：`merge` 仅 inner + 单键；`*_withcol` 与整表聚合并行两套 API。
5. **封存内存模型**：`reserve_with_capacity` 后无法扩容，超限 `add_row` 抛错。
6. **无真分片缓冲协议**：DataFrame 不导出 memoryview/`__getbuffer__`；跨进程传输靠 pickle 分解（`to_pickle` 先转 dict 形式）。

---

*文档由代码静态审查自动生成，行号基于当前工作树；如后续重构请同步更新。*