# _fastdataframe_rolling.pxi — Rolling window (module-level cdef)
# Included at module level in fastdataframe.pyx (after class definition)

cdef _rolling_impl(FastDataFrame self, Py_ssize_t window, str col_name):
    if window <= 0:
        raise ValueError(f"Window size must be positive, got {window}")
    if col_name not in self.columns:
        raise KeyError(f"Column '{col_name}' not found")
    cdef BaseColumn col = self.columns[col_name]
    cdef int col_dtype = col.dtype
    if col_dtype != TYPE_INT and col_dtype != TYPE_DOUBLE:
        raise TypeError(f"Rolling window requires numeric column, got '{col_name}'")
    return _RollingWrapper(col, window)
