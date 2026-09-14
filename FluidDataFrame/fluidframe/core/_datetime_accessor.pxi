# ============================================================
#  DateTime Accessor for Series
#  pandas-style .dt accessor for date/datetime/time columns
# ============================================================

# 属性化白名单：列上同名零参方法在 .dt 下以属性返回值列。
cdef list _DT_ATTRS = [
    'year', 'month', 'day', 'hour', 'minute', 'second', 'quarter',
    'week_of_year', 'day_of_week', 'day_of_year', 'days_in_month',
    'is_leap_year', 'is_weekend',
    'is_month_start', 'is_month_end', 'is_year_start', 'is_year_end',
    'is_quarter_start', 'is_quarter_end',
    'day_name', 'month_name',
]


@cython.final
cdef class _DateTimeAccessor:
    """pandas 风格 .dt 访问器。
    由 Series.dt property 懒加载缓存；属性将列上的零参日期访问器
    （year()/month()/...）转为返回值列的形式。
    """

    def __cinit__(self, Series series):
        self._series = series

    def __getattr__(self, object name):
        cdef object fn
        cdef BaseColumn col = self._series._col
        if name.startswith('_'):
            raise AttributeError(f"'DateTimeAccessor' object has no attribute '{name}'")
        if name not in _DT_ATTRS:
            raise AttributeError(f"'DateTimeAccessor' object has no attribute '{name}'")
        fn = getattr(col, name, None)
        if fn is None or not callable(fn):
            raise AttributeError(
                f"`.dt.{name}` not supported for column type {type(col).__name__}")
        return fn()

    @cython.binding(False)
    def weekday(self):
        return self._series._col.day_of_week()

    @cython.binding(False)
    def dayofweek(self):
        return self._series._col.day_of_week()

    @cython.binding(False)
    def dayofyear(self):
        return self._series._col.day_of_year()

    @cython.binding(False)
    def weekofyear(self):
        return self._series._col.week_of_year()

    @cython.binding(False)
    def strftime(self, str fmt):
        return getattr(self._series._col, 'strftime')(fmt)
