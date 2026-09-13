# cython: language_level=3

from libc.string cimport memcpy, memmove, memset
from libc.math cimport sqrt, fabs, NAN
cimport cython
from cpython.unicode cimport PyUnicode_AsUTF8AndSize

# ============================================================
#  Platform-specific aligned allocation (C header injection)
# ============================================================
cdef extern from *:
    """
    #include <stdlib.h>

    #if defined(_MSC_VER) || defined(_WIN32)
    #include <malloc.h>
    static inline void* os_aligned_malloc(size_t size, size_t alignment) {
        return _aligned_malloc(size, alignment);
    }
    static inline void os_aligned_free(void* ptr) {
        _aligned_free(ptr);
    }
    #else
    static inline void* os_aligned_malloc(size_t size, size_t alignment) {
        void* ptr = NULL;
        if (posix_memalign(&ptr, alignment, size) == 0) {
            return ptr;
        }
        return NULL;
    }
    static inline void os_aligned_free(void* ptr) {
        free(ptr);
    }
    #endif
    """
    void* os_aligned_malloc(size_t size, size_t alignment) nogil
    void os_aligned_free(void* ptr) nogil

cdef extern from "<stdlib.h>" nogil:
    int posix_memalign(void **memptr, size_t alignment, size_t size)
    void free(void *ptr)


# ============================================================
#  Aligned memory allocation
# ============================================================
cdef void* aligned_malloc(size_t size, size_t alignment=64) nogil:
    if size == 0:
        return NULL
    cdef size_t ptr_size = sizeof(void*)
    if alignment < ptr_size:
        alignment = ptr_size
    if (alignment & (alignment - 1)) != 0:
        return NULL
    return os_aligned_malloc(size, alignment)


cdef void* aligned_realloc(void* ptr, size_t old_size, size_t new_size, size_t alignment=64) nogil:
    if new_size == 0:
        if ptr != NULL:
            os_aligned_free(ptr)
        return NULL
    if ptr != NULL and new_size == old_size and (<size_t>ptr % alignment) == 0:
        return ptr

    cdef void* new_ptr = aligned_malloc(new_size, alignment)
    if new_ptr != NULL and ptr != NULL:
        if old_size < new_size:
            memcpy(new_ptr, ptr, old_size)
        else:
            memcpy(new_ptr, ptr, new_size)
        os_aligned_free(ptr)
    return new_ptr


# ============================================================
#  Base column class
# ============================================================
cdef class BaseColumn:
    def __cinit__(self):
        self.length = 0
        self.capacity = 0
        self.dtype = -1
        self.is_2d = False
        self.num_cols = 1
        self.owns_data = True

    cdef Py_ssize_t _get_rows_c(self) noexcept nogil:
        """底层无锁获取逻辑行数"""
        if self.is_2d and self.num_cols > 1:
            return self.length // self.num_cols
        return self.length

    cdef void* _get_data_ptr(self) noexcept nogil:
        """
        供子类重写：返回底层 C 数组裸指针。
        """
        return NULL

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def get_size(self):
        return 0

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __contains__(self, object val):
        """元素级成员检测：`val in col`（对标 pandas Series.__contains__）。"""
        cdef Py_ssize_t i
        cdef object getitem = self.__getitem__
        for i in range(self.length):
            if getitem(i) == val:
                return True
        return False

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def copy(self):
        raise NotImplementedError("Subclasses must implement copy()")

    # ------------------------------------------------------------
    # Series 级便利方法：统一转发到懒加载的 SeriesAccessor
    # （2D 语义见 SeriesAccessor 及 fastdataframe 编译单元的 _series_*）
    # ------------------------------------------------------------

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def value_counts(self, str key_name='value'):
        return self.series.value_counts(key_name)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isna(self):
        return self.series.isna()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def notna(self):
        return self.series.notna()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def map(self, object func):
        return self.series.map(func)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def factorize(self):
        return self.series.factorize()

    # ------------------------------------------------------------
    # Series accessor 懒加载（同 .str accessor 模式）
    # ------------------------------------------------------------

    @property
    def series(self):
        if self._series_cache is not None:
            return self._series_cache
        self._series_cache = _make_series_accessor(self)
        return self._series_cache

    # ------------------------------------------------------------
    # .dt accessor 懒加载（仅 date/datetime/time 列有意义）
    # ------------------------------------------------------------

    @property
    def dt(self):
        if self.dtype != TYPE_DATE and self.dtype != TYPE_DATETIME and self.dtype != TYPE_TIME:
            raise AttributeError(
                f"`dt` accessor valid only for date/datetime/time columns, got dtype={self.dtype}")
        if self._dt_cache is not None:
            return self._dt_cache
        self._dt_cache = _make_dt_accessor(self)
        return self._dt_cache

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def drop_row(self, Py_ssize_t row_idx):
        raise NotImplementedError

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def shift_left(self, Py_ssize_t pos):
        raise NotImplementedError

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def abs(self):
        raise NotImplementedError("Subclasses must implement abs()")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def round(self, int decimals=0):
        raise NotImplementedError("Subclasses must implement round()")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def ffill(self):
        raise NotImplementedError("Subclasses must implement ffill()")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def bfill(self):
        raise NotImplementedError("Subclasses must implement bfill()")


# ============================================================
#  Astype helper
# ============================================================
cdef inline bint _fast_match(const char* s, const char* expected, Py_ssize_t length) noexcept nogil:
    cdef Py_ssize_t i
    for i in range(length):
        if (s[i] | 0x20) != expected[i]:
            return False
    return True

cdef int _resolve_astype_target(object target) except -1:
    if target is int: return TYPE_INT
    if target is float: return TYPE_DOUBLE
    if target is bool: return TYPE_BOOL
    if target is str: return TYPE_UCS2

    cdef Py_ssize_t length
    cdef const char* c_str

    if isinstance(target, str):
        c_str = PyUnicode_AsUTF8AndSize(target, &length)
        if c_str == NULL:
            raise TypeError(f"Invalid string type: {target}")

        if length == 2:
            if _fast_match(c_str, b"i4", 2): return TYPE_INT
            if _fast_match(c_str, b"i8", 2): return TYPE_INT
            if _fast_match(c_str, b"f4", 2): return TYPE_DOUBLE
            if _fast_match(c_str, b"f8", 2): return TYPE_DOUBLE
        elif length == 3:
            if _fast_match(c_str, b"int", 3): return TYPE_INT
            if _fast_match(c_str, b"str", 3): return TYPE_UCS2
            if _fast_match(c_str, b"cat", 3): return TYPE_CATEGORY
        elif length == 4:
            if _fast_match(c_str, b"bool", 4): return TYPE_BOOL
            if _fast_match(c_str, b"date", 4): return TYPE_DATE
            if _fast_match(c_str, b"time", 4): return TYPE_TIME
            if _fast_match(c_str, b"ucs2", 4): return TYPE_UCS2
            if _fast_match(c_str, b"cate", 4): return TYPE_CATEGORY
        elif length == 5:
            if _fast_match(c_str, b"float", 5): return TYPE_DOUBLE
            if _fast_match(c_str, b"int32", 5): return TYPE_INT
            if _fast_match(c_str, b"int64", 5): return TYPE_INT
            if _fast_match(c_str, b"ascii", 5): return TYPE_CHAR256
        elif length == 6:
            if _fast_match(c_str, b"double", 6): return TYPE_DOUBLE
            if _fast_match(c_str, b"string", 6): return TYPE_UCS2
        elif length == 7:
            if _fast_match(c_str, b"integer", 7): return TYPE_INT
            if _fast_match(c_str, b"float64", 7): return TYPE_DOUBLE
            if _fast_match(c_str, b"boolean", 7): return TYPE_BOOL
            if _fast_match(c_str, b"unicode", 7): return TYPE_UCS2
            if _fast_match(c_str, b"char256", 7): return TYPE_CHAR256
        elif length == 8:
            if _fast_match(c_str, b"datetime", 8): return TYPE_DATETIME
            if _fast_match(c_str, b"category", 8): return TYPE_CATEGORY
        elif length == 9:
            if _fast_match(c_str, b"timestamp", 9): return TYPE_DATE

    raise TypeError(f"Cannot cast to unsupported target type: {target}")


# ============================================================
#  Series accessor（pandas 风格列级访问器）
# ============================================================
cdef object _series_fdf_mod = None


cdef object _series_module():
    """惰性缓存 fastdataframe 模块引用：实现体在该模块编译单元（全 dtype
    keys/重建/类型推断分发），此处只透传。模块名缓存避免反复 import。"""
    global _series_fdf_mod
    if _series_fdf_mod is None:
        import fluidframe.core.fastdataframe as _m
        _series_fdf_mod = _m
    return _series_fdf_mod


@cython.final
cdef class SeriesAccessor:
    """pandas 风格的列级访问器：df['a'].series.value_counts()/isna()/notna()/map()。
    实例由 BaseColumn.series property 懒加载缓存（同 .str accessor 模式：
    __new__ C 级工厂绕过参数打包）。方法透传到 fastdataframe 编译单元，语义与
    df.value_counts / df.isnull / df.apply 完全对齐（2D 列走行级键或逐 cell）。"""

    def __cinit__(self):
        # 无参 __cinit__：允许工厂函数直接 __new__ 绕过参数打包
        pass

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def value_counts(self, str key_name='value'):
        return _series_module()._series_value_counts(self._col, key_name)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def isna(self):
        return _series_module()._series_isna(self._col)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def notna(self):
        return _series_module()._series_notna(self._col)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def map(self, object func):
        return _series_module()._series_map(self._col, func)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def factorize(self):
        return _series_module()._series_factorize(self._col)


cdef inline SeriesAccessor _make_series_accessor(BaseColumn col):
    cdef SeriesAccessor acc = <SeriesAccessor>SeriesAccessor.__new__(SeriesAccessor)
    acc._col = col
    return acc


# ============================================================
#  date/datetime/time accessor（pandas 风格 .dt）
# ============================================================
# 属性化白名单：列上同名零参方法在 .dt 下以属性返回值列。
# 覆盖 Date / DateTime / Time 三列各自的公开访问器。
cdef list _DT_ATTRS = [
    'year', 'month', 'day', 'hour', 'minute', 'second', 'quarter',
    'week_of_year', 'day_of_week', 'day_of_year', 'days_in_month',
    'is_leap_year', 'is_weekend',
    'is_month_start', 'is_month_end', 'is_year_start', 'is_year_end',
    'is_quarter_start', 'is_quarter_end',
    'day_name', 'month_name',
]


@cython.final
cdef class DateTimeAccessor:
    """pandas 风格 .dt 访问器。
    由 BaseColumn.dt property 懒加载缓存；属性将列上的零参日期访问器
    （year()/month()/...）转为返回值列的形式：
        df['d'].dt.year       -> DateColumn/DateTimeColumn.year()
        df['t'].dt.hour       -> Time/DateTime 的 hour()
    weekday 相关（is_weekend 等）同理。非日期列访问 .dt 抛 AttributeError。"""

    def __cinit__(self):
        pass

    def __getattr__(self, object name):
        cdef object fn
        if name.startswith('_'):
            raise AttributeError(f"'DateTimeAccessor' object has no attribute '{name}'")
        if name not in _DT_ATTRS:
            raise AttributeError(f"'DateTimeAccessor' object has no attribute '{name}'")
        fn = getattr(self._col, name, None)
        if fn is None or not callable(fn):
            raise AttributeError(
                f"`.dt.{name}` not supported for column type {type(self._col).__name__}")
        return fn()

    @cython.binding(False)
    def weekday(self):
        return self._col.day_of_week()

    @cython.binding(False)
    def dayofweek(self):
        return self._col.day_of_week()

    @cython.binding(False)
    def dayofyear(self):
        return self._col.day_of_year()

    @cython.binding(False)
    def weekofyear(self):
        return self._col.week_of_year()

    @cython.binding(False)
    def strftime(self, str fmt):
        return getattr(self._col, 'strftime')(fmt)


cdef inline DateTimeAccessor _make_dt_accessor(BaseColumn col):
    cdef DateTimeAccessor acc = <DateTimeAccessor>DateTimeAccessor.__new__(DateTimeAccessor)
    acc._col = col
    return acc
