# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport cython
from libc.string cimport memcpy, memmove, strncmp
from libc.stdlib cimport free
from libc.math cimport NAN
from cpython.unicode cimport Py_UCS2

from fluidframe.core._base cimport BaseColumn, NumericType, DataType
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.doublecolumn cimport DoubleColumn
from fluidframe.core.char256column cimport Char256Column_SoA
from fluidframe.core.ucs2column cimport UCS2Column_SoA
from fluidframe.core.timestampcolumn cimport TimestampColumn

from fluidframe._schema_types import Char256, Timestamp, _parse_schema
from fluidframe._schema_types import TYPE_INT, TYPE_DOUBLE, TYPE_CHAR256, TYPE_UCS2, TYPE_TIMESTAMP

cdef void _c_argsort_char256(char* data, Py_ssize_t* indices, Py_ssize_t left, Py_ssize_t right) noexcept nogil:
    if left >= right: return
    cdef Py_ssize_t i = left, j = right
    cdef Py_ssize_t pivot_idx = indices[(left + right) // 2]
    cdef char* pivot_ptr = data + (pivot_idx * 256)
    cdef Py_ssize_t temp

    while i <= j:
        while strncmp(data + (indices[i] * 256), pivot_ptr, 256) < 0: i += 1
        while strncmp(data + (indices[j] * 256), pivot_ptr, 256) > 0: j -= 1
        if i <= j:
            temp = indices[i]
            indices[i] = indices[j]
            indices[j] = temp
            i += 1
            j -= 1

    _c_argsort_char256(data, indices, left, j)
    _c_argsort_char256(data, indices, i, right)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _c_argsort_numeric(NumericType* data, Py_ssize_t* indices, Py_ssize_t left, Py_ssize_t right) noexcept nogil:
    if left >= right:
        return

    cdef Py_ssize_t i = left, j = right
    cdef Py_ssize_t pivot_idx = indices[(left + right) // 2]
    cdef NumericType pivot = data[pivot_idx]
    cdef Py_ssize_t temp

    while i <= j:
        while data[indices[i]] < pivot: i += 1
        while data[indices[j]] > pivot: j -= 1
        if i <= j:
            temp = indices[i]
            indices[i] = indices[j]
            indices[j] = temp
            i += 1
            j -= 1

    _c_argsort_numeric(data, indices, left, j)
    _c_argsort_numeric(data, indices, i, right)


cdef inline int _ucs2_cmp_at(Py_UCS2* chars, Py_ssize_t* offsets, Py_ssize_t i, Py_ssize_t j) noexcept nogil:
    cdef int i_start = offsets[i], i_len = offsets[i+1] - i_start
    cdef int j_start = offsets[j], j_len = offsets[j+1] - j_start
    cdef int k, min_len = i_len if i_len < j_len else j_len
    for k in range(min_len):
        if chars[i_start + k] < chars[j_start + k]: return -1
        if chars[i_start + k] > chars[j_start + k]: return 1
    if i_len < j_len: return -1
    if i_len > j_len: return 1
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _c_argsort_ucs2(Py_UCS2* chars, Py_ssize_t* offsets, Py_ssize_t* indices, Py_ssize_t left, Py_ssize_t right) noexcept nogil:
    if left >= right: return
    cdef Py_ssize_t i = left, j = right
    cdef Py_ssize_t pivot_idx = indices[(left + right) // 2]
    cdef Py_ssize_t temp

    while i <= j:
        while _ucs2_cmp_at(chars, offsets, indices[i], pivot_idx) < 0: i += 1
        while _ucs2_cmp_at(chars, offsets, indices[j], pivot_idx) > 0: j -= 1
        if i <= j:
            temp = indices[i]
            indices[i] = indices[j]
            indices[j] = temp
            i += 1
            j -= 1

    _c_argsort_ucs2(chars, offsets, indices, left, j)
    _c_argsort_ucs2(chars, offsets, indices, i, right)


@cython.final
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class FastDataFrame:

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __init__(self, dict schema=None):
        self.columns = {}
        self.df_length = 0

        if schema:
            parsed = _parse_schema(schema)
            for col_name, dtype in parsed.items():
                if dtype == TYPE_INT:
                    self.columns[col_name] = IntColumn()
                elif dtype == TYPE_DOUBLE:
                    self.columns[col_name] = DoubleColumn()
                elif dtype == TYPE_CHAR256:
                    self.columns[col_name] = Char256Column_SoA()
                elif dtype == TYPE_UCS2:
                    self.columns[col_name] = UCS2Column_SoA()
                elif dtype == TYPE_TIMESTAMP:
                    self.columns[col_name] = TimestampColumn()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __getattr__(self, name):
        if name in self.columns: return self.columns[name]
        raise AttributeError(f"FastDataFrame has no column '{name}'")

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __getitem__(self, name):
        return self.columns[name]

    @cython.binding(False)
    def append_row(self, **kwargs):
        for col_name, val in kwargs.items():
            self.columns[col_name].append(val)
        self.df_length += 1

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def copy(self):
        new_df = FastDataFrame()
        new_df.df_length = self.df_length
        for col_name, col_obj in self.columns.items():
            new_df.columns[col_name] = col_obj.copy()
        return new_df

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_front(self):
        if self.df_length == 0: raise IndexError("pop from empty DataFrame")
        row = {}
        for col_name, col_obj in self.columns.items():
            row[col_name] = col_obj.pop_front()
        self.df_length -= 1
        return row

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_back(self):
        if self.df_length == 0: raise IndexError("pop from empty DataFrame")
        row = {}
        for col_name, col_obj in self.columns.items():
            row[col_name] = col_obj.pop_back()
        self.df_length -= 1
        return row

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def concat(self, FastDataFrame other):
        for col_name, col_obj in self.columns.items():
            if col_name in other.columns:
                col_obj.concat(other.columns[col_name])
        self.df_length += other.df_length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def drop(self, str col_name):
        if col_name in self.columns:
            del self.columns[col_name]

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def argsort_by(self, str col_name, bint reverse=False):
        if col_name not in self.columns:
            raise KeyError(f"Column '{col_name}' not found")

        cdef BaseColumn col = self.columns[col_name]
        cdef Py_ssize_t n = self.df_length
        if n == 0:
            return []

        cdef Py_ssize_t* indices_c = <Py_ssize_t*>malloc(n * sizeof(Py_ssize_t))
        if not indices_c:
            raise MemoryError()

        cdef Py_ssize_t i
        for i in range(n):
            indices_c[i] = i

        if isinstance(col, IntColumn):
            with nogil:
                _c_argsort_numeric((<IntColumn>col).data_c, indices_c, 0, n - 1)

        elif isinstance(col, DoubleColumn):
            with nogil:
                _c_argsort_numeric((<DoubleColumn>col).data_c, indices_c, 0, n - 1)

        elif isinstance(col, Char256Column_SoA):
            with nogil:
                _c_argsort_char256((<Char256Column_SoA>col).data_c, indices_c, 0, n - 1)

        elif isinstance(col, UCS2Column_SoA):
            with nogil:
                _c_argsort_ucs2((<UCS2Column_SoA>col).chars_c, (<UCS2Column_SoA>col).offsets_c, indices_c, 0, n - 1)

        elif isinstance(col, TimestampColumn):
            with nogil:
                _c_argsort_numeric((<TimestampColumn>col).data_c, indices_c, 0, n - 1)
        else:
            free(indices_c)
            raise TypeError("Sorting not supported for this column type")

        cdef Py_ssize_t temp
        if reverse:
            for i in range(n // 2):
                temp = indices_c[i]
                indices_c[i] = indices_c[n - 1 - i]
                indices_c[n - 1 - i] = temp

        cdef list py_indices = [indices_c[j] for j in range(n)]
        free(indices_c)
        return py_indices


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def sort_values(self, str col_name, bint reverse=False, bint inplace=False):
        if col_name not in self.columns:
            raise KeyError(f"Column '{col_name}' not found")

        cdef Py_ssize_t n = self.df_length
        if n == 0:
            if inplace:
                return self
            return self.copy()

        cdef list py_indices = self.argsort_by(col_name, reverse=reverse)
        cdef Py_ssize_t* idx_c = <Py_ssize_t*>malloc(n * sizeof(Py_ssize_t))
        if not idx_c: raise MemoryError()
        cdef Py_ssize_t i
        for i in range(n):
            idx_c[i] = py_indices[i]

        cdef FastDataFrame result_df
        if inplace:
            result_df = self
        else:
            result_df = FastDataFrame()
            result_df.df_length = n

        cdef str name
        cdef BaseColumn col
        cdef BaseColumn new_col

        for name, col in self.columns.items():
            if isinstance(col, IntColumn):
                new_col = self._reorder_int_column(<IntColumn>col, idx_c, n, inplace)
            elif isinstance(col, DoubleColumn):
                new_col = self._reorder_double_column(<DoubleColumn>col, idx_c, n, inplace)
            elif isinstance(col, Char256Column_SoA):
                new_col = self._reorder_char256_column(<Char256Column_SoA>col, idx_c, n, inplace)
            elif isinstance(col, UCS2Column_SoA):
                new_col = self._reorder_ucs2_column(<UCS2Column_SoA>col, idx_c, n, inplace)
            elif isinstance(col, TimestampColumn):
                new_col = self._reorder_timestamp_column(<TimestampColumn>col, idx_c, n, inplace)
            else:
                raise NotImplementedError()
            if not inplace:
                result_df.columns[name] = new_col

        free(idx_c)
        if not inplace:
            return result_df


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef IntColumn _reorder_int_column(self, IntColumn col, Py_ssize_t* indices, Py_ssize_t n, bint inplace):
        cdef int* old_data = col.data_c
        cdef int* new_data = <int*>malloc(col.capacity * sizeof(int))
        if not new_data: raise MemoryError()

        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                new_data[i] = old_data[indices[i]]

        if inplace:
            free(col.data_c)
            col.data_c = new_data
            return col
        else:
            new_col = IntColumn(col.capacity)
            free(new_col.data_c)
            new_col.data_c = new_data
            new_col.length = n
            return new_col

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef DoubleColumn _reorder_double_column(self, DoubleColumn col, Py_ssize_t* indices, Py_ssize_t n, bint inplace):
        cdef double* old_data = col.data_c
        cdef double* new_data = <double*>malloc(col.capacity * sizeof(double))
        if not new_data: raise MemoryError()

        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                new_data[i] = old_data[indices[i]]

        if inplace:
            free(col.data_c)
            col.data_c = new_data
            return col
        else:
            new_col = DoubleColumn(col.capacity)
            free(new_col.data_c)
            new_col.data_c = new_data
            new_col.length = n
            return new_col

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef Char256Column_SoA _reorder_char256_column(self, Char256Column_SoA col, Py_ssize_t* indices, Py_ssize_t n, bint inplace):
        cdef char* old_data = col.data_c
        cdef char* new_data = <char*>malloc(col.capacity * 256)
        if not new_data: raise MemoryError()

        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                memcpy(new_data + (i * 256), old_data + (indices[i] * 256), 256)

        if inplace:
            free(col.data_c)
            col.data_c = new_data
            return col
        else:
            new_col = Char256Column_SoA(col.capacity)
            free(new_col.data_c)
            new_col.data_c = new_data
            new_col.length = n
            return new_col

    cdef TimestampColumn _reorder_timestamp_column(self, TimestampColumn col, Py_ssize_t* indices, Py_ssize_t n, bint inplace):
        cdef long long* old_data = col.data_c
        cdef long long* new_data = <long long*>malloc(col.capacity * sizeof(long long))
        if not new_data: raise MemoryError()

        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                new_data[i] = old_data[indices[i]]

        if inplace:
            free(col.data_c)
            col.data_c = new_data
            return col
        else:
            new_col = TimestampColumn(col.capacity)
            free(new_col.data_c)
            new_col.data_c = new_data
            new_col.length = n
            return new_col


    @cython.binding(False)
    @cython.always_allow_keywords(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def diff(self, str col_name, int periods=1):
        if col_name not in self.columns:
            raise KeyError(f"Column '{col_name}' not found")

        cdef BaseColumn col = self.columns[col_name]
        if periods < 0:
            raise ValueError("periods must be >= 0")

        if isinstance(col, DoubleColumn):
            return self._diff_double(<DoubleColumn>col, periods)
        elif isinstance(col, IntColumn):
            return self._diff_int(<IntColumn>col, periods)
        else:
            raise TypeError(f"diff is only supported for numeric columns, got {type(col)}")


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef DoubleColumn _diff_double(self, DoubleColumn col, int periods):
        cdef Py_ssize_t n = col.length
        cdef DoubleColumn new_col = DoubleColumn(col.capacity)
        new_col.length = n

        cdef double* old_data = col.data_c
        cdef double* new_data = new_col.data_c
        cdef Py_ssize_t i

        with nogil:
            for i in range(n):
                if i < periods:
                    new_data[i] = NAN
                else:
                    new_data[i] = old_data[i] - old_data[i - periods]
        return new_col

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef IntColumn _diff_int(self, IntColumn col, int periods):
        cdef Py_ssize_t n = col.length
        cdef IntColumn new_col = IntColumn(col.capacity)
        new_col.length = n

        cdef int* old_data = col.data_c
        cdef int* new_data = new_col.data_c
        cdef Py_ssize_t i

        with nogil:
            for i in range(n):
                if i < periods:
                    new_data[i] = 0
                else:
                    new_data[i] = old_data[i] - old_data[i - periods]

        return new_col


    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def reset_index(self):
        cdef IntColumn idx_col = IntColumn(self.df_length)
        cdef Py_ssize_t i
        for i in range(self.df_length):
            idx_col.data_c[i] = <int>i
        idx_col.length = self.df_length
        self.columns['index'] = idx_col

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def __repr__(self):
        return f"<FastDataFrame: {self.df_length} rows, columns: {list(self.columns.keys())}>"


    @cython.binding(False)
    def drop_col(self, str col_name):
        self.drop(col_name)

    @cython.binding(False)
    def sort_index(self, bint ascending=True, bint inplace=False):
        if 'index' not in self.columns:
            raise KeyError("No 'index' column found. Call reset_index() first.")
        return self.sort_values('index', reverse=not ascending, inplace=inplace)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef UCS2Column_SoA _reorder_ucs2_column(self, UCS2Column_SoA col, Py_ssize_t* indices, Py_ssize_t n, bint inplace):
        cdef Py_ssize_t* new_offsets = <Py_ssize_t*>malloc((n + 1) * sizeof(Py_ssize_t))
        cdef Py_UCS2* new_chars = <Py_UCS2*>malloc(col.char_capacity * sizeof(Py_UCS2))
        if not new_offsets or not new_chars: raise MemoryError()

        new_offsets[0] = 0
        cdef Py_ssize_t current_char_len = 0
        cdef Py_ssize_t i, idx, start, end, str_len
        cdef UCS2Column_SoA new_col

        with nogil:
            for i in range(n):
                idx = indices[i]
                start = col.offsets_c[idx]
                end = col.offsets_c[idx + 1]
                str_len = end - start

                if str_len > 0:
                    memcpy(new_chars + current_char_len, col.chars_c + start, str_len * sizeof(Py_UCS2))

                current_char_len += str_len
                new_offsets[i + 1] = current_char_len

        if inplace:
            free(col.offsets_c)
            free(col.chars_c)
            col.offsets_c = new_offsets
            col.chars_c = new_chars
            col.char_length = current_char_len
            col.length = n
            return col
        else:
            new_col = UCS2Column_SoA(col.capacity)
            free(new_col.offsets_c)
            free(new_col.chars_c)
            new_col.offsets_c = new_offsets
            new_col.chars_c = new_chars
            new_col.char_length = current_char_len
            new_col.char_capacity = col.char_capacity
            new_col.length = n
            return new_col

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.binding(False)
    def drop_with_index(self, list drop_indices, bint inplace=False):
        cdef Py_ssize_t n_orig = self.df_length
        if n_orig == 0:
            return self if inplace else self.copy()

        cdef char* drop_mask = <char*>malloc(n_orig * sizeof(char))
        if not drop_mask: raise MemoryError()
        cdef Py_ssize_t i
        for i in range(n_orig):
            drop_mask[i] = 0

        cdef Py_ssize_t idx
        for py_idx in drop_indices:
            idx = <Py_ssize_t>py_idx
            if idx < 0: idx += n_orig
            if idx < 0 or idx >= n_orig:
                free(drop_mask)
                raise IndexError(f"Index {py_idx} out of bounds")
            drop_mask[idx] = 1

        cdef Py_ssize_t keep_count = 0
        for i in range(n_orig):
            if drop_mask[i] == 0:
                keep_count += 1

        if keep_count == n_orig:
            free(drop_mask)
            return self if inplace else self.copy()
        if keep_count == 0:
            free(drop_mask)
            if inplace:
                self.df_length = 0
                for col_obj in self.columns.values(): col_obj.length = 0
                return self
            else:
                return FastDataFrame()

        cdef Py_ssize_t* keep_idx_c = <Py_ssize_t*>malloc(keep_count * sizeof(Py_ssize_t))
        if not keep_idx_c:
            free(drop_mask)
            raise MemoryError()

        cdef Py_ssize_t j = 0
        for i in range(n_orig):
            if drop_mask[i] == 0:
                keep_idx_c[j] = i
                j += 1
        free(drop_mask)

        cdef FastDataFrame result_df
        if inplace:
            result_df = self
            self.df_length = keep_count
        else:
            result_df = FastDataFrame()
            result_df.df_length = keep_count

        cdef str name
        cdef BaseColumn col, new_col
        for name, col in self.columns.items():
            if isinstance(col, IntColumn):
                new_col = self._reorder_int_column(<IntColumn>col, keep_idx_c, keep_count, inplace)
            elif isinstance(col, DoubleColumn):
                new_col = self._reorder_double_column(<DoubleColumn>col, keep_idx_c, keep_count, inplace)
            elif isinstance(col, Char256Column_SoA):
                new_col = self._reorder_char256_column(<Char256Column_SoA>col, keep_idx_c, keep_count, inplace)
            elif isinstance(col, UCS2Column_SoA):
                new_col = self._reorder_ucs2_column(<UCS2Column_SoA>col, keep_idx_c, keep_count, inplace)
            elif isinstance(col, TimestampColumn):
                new_col = self._reorder_timestamp_column(<TimestampColumn>col, keep_idx_c, keep_count, inplace)
            else:
                free(keep_idx_c)
                raise TypeError(f"Drop not supported for column {name}")

            if not inplace:
                result_df.columns[name] = new_col

        free(keep_idx_c)
        if not inplace:
            return result_df

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.binding(False)
    def drop_raw(self, Py_ssize_t row_idx):
        if row_idx < 0 or row_idx >= self.df_length:
            raise IndexError("Row index out of range")

        cdef str name
        cdef BaseColumn col
        cdef Py_ssize_t rem_len = self.df_length - row_idx - 1

        for name, col in self.columns.items():
            if isinstance(col, IntColumn):
                if rem_len > 0:
                    memmove((<IntColumn>col).data_c + row_idx,
                            (<IntColumn>col).data_c + row_idx + 1,
                            rem_len * sizeof(int))
                (<IntColumn>col).length -= 1

            elif isinstance(col, DoubleColumn):
                if rem_len > 0:
                    memmove((<DoubleColumn>col).data_c + row_idx,
                            (<DoubleColumn>col).data_c + row_idx + 1,
                            rem_len * sizeof(double))
                (<DoubleColumn>col).length -= 1

            elif isinstance(col, Char256Column_SoA):
                if rem_len > 0:
                    memmove((<Char256Column_SoA>col).data_c + (row_idx * 256),
                            (<Char256Column_SoA>col).data_c + ((row_idx + 1) * 256),
                            rem_len * 256)
                (<Char256Column_SoA>col).length -= 1

            elif isinstance(col, UCS2Column_SoA):
                self._drop_row_ucs2(<UCS2Column_SoA>col, row_idx, rem_len)

            elif isinstance(col, TimestampColumn):
                if rem_len > 0:
                    memmove((<TimestampColumn>col).data_c + row_idx,
                            (<TimestampColumn>col).data_c + row_idx + 1,
                            rem_len * sizeof(long long))
                (<TimestampColumn>col).length -= 1

        self.df_length -= 1

    cdef void _drop_row_ucs2(self, UCS2Column_SoA col, Py_ssize_t row_idx, Py_ssize_t rem_len):
        cdef Py_ssize_t start_offset = col.offsets_c[row_idx]
        cdef Py_ssize_t end_offset = col.offsets_c[row_idx + 1]
        cdef Py_ssize_t char_len = end_offset - start_offset

        cdef Py_ssize_t rem_chars = col.char_length - end_offset
        if rem_chars > 0:
            memmove(col.chars_c + start_offset,
                    col.chars_c + end_offset,
                    rem_chars * sizeof(Py_UCS2))
        col.char_length -= char_len

        cdef Py_ssize_t i
        for i in range(row_idx, col.length):
            col.offsets_c[i] = col.offsets_c[i + 1] - char_len
        col.length -= 1

    @cython.binding(False)
    def argmax(self, str col_name):
        return self.columns[col_name].idxmax()

    @cython.binding(False)
    def argmin(self, str col_name):
        return self.columns[col_name].idxmin()

    @cython.binding(False)
    def unique(self, str col_name):
        return self.columns[col_name].unique()

    @cython.binding(False)
    def to_pickle(self, filename):
        import pickle
        cdef dict data = {'df_length': self.df_length, 'columns': {}}
        cdef Py_ssize_t col_len
        for name, col_obj in self.columns.items():
            col_len = col_obj.length
            values = [col_obj[i] for i in range(col_len)]
            if isinstance(col_obj, IntColumn):
                data['columns'][name] = ('int', values)
            elif isinstance(col_obj, DoubleColumn):
                data['columns'][name] = ('float', values)
            elif isinstance(col_obj, Char256Column_SoA):
                data['columns'][name] = ('char256', values)
            elif isinstance(col_obj, UCS2Column_SoA):
                data['columns'][name] = ('ucs2', values)
            elif isinstance(col_obj, TimestampColumn):
                data['columns'][name] = ('timestamp', values)
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    @cython.binding(False)
    def to_csv(self, filename, sep=','):
        import csv
        cdef list col_names = list(self.columns.keys())
        cdef Py_ssize_t i
        cdef str name
        cdef BaseColumn col
        cdef list row
        with open(filename, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter=sep)
            writer.writerow(col_names)
            for i in range(self.df_length):
                row = [str(self.columns[name][i]) for name in col_names]
                writer.writerow(row)

    @cython.binding(False)
    def to_tsv(self, filename):
        self.to_csv(filename, sep='\t')

    cdef FastDataFrame _take(self, Py_ssize_t* indices, Py_ssize_t n):
        cdef FastDataFrame result = FastDataFrame()
        result.df_length = n
        cdef str name
        cdef BaseColumn col
        for name, col in self.columns.items():
            if isinstance(col, IntColumn):
                result.columns[name] = self._reorder_int_column(<IntColumn>col, indices, n, False)
            elif isinstance(col, DoubleColumn):
                result.columns[name] = self._reorder_double_column(<DoubleColumn>col, indices, n, False)
            elif isinstance(col, Char256Column_SoA):
                result.columns[name] = self._reorder_char256_column(<Char256Column_SoA>col, indices, n, False)
            elif isinstance(col, UCS2Column_SoA):
                result.columns[name] = self._reorder_ucs2_column(<UCS2Column_SoA>col, indices, n, False)
            elif isinstance(col, TimestampColumn):
                result.columns[name] = self._reorder_timestamp_column(<TimestampColumn>col, indices, n, False)
        return result

    @cython.binding(False)
    def head(self, Py_ssize_t n=5):
        if n < 0: raise ValueError("n must be non-negative")
        n = min(n, self.df_length)
        if n == 0: return FastDataFrame()
        cdef Py_ssize_t* idx_c = <Py_ssize_t*>malloc(n * sizeof(Py_ssize_t))
        if not idx_c: raise MemoryError()
        cdef Py_ssize_t i
        for i in range(n): idx_c[i] = i
        cdef FastDataFrame result = self._take(idx_c, n)
        free(idx_c)
        return result

    @cython.binding(False)
    def tail(self, Py_ssize_t n=5):
        if n < 0: raise ValueError("n must be non-negative")
        n = min(n, self.df_length)
        if n == 0: return FastDataFrame()
        cdef Py_ssize_t start = self.df_length - n
        cdef Py_ssize_t* idx_c = <Py_ssize_t*>malloc(n * sizeof(Py_ssize_t))
        if not idx_c: raise MemoryError()
        cdef Py_ssize_t i
        for i in range(n): idx_c[i] = start + i
        cdef FastDataFrame result = self._take(idx_c, n)
        free(idx_c)
        return result

    @property
    def shape(self):
        return (self.df_length, len(self.columns))

    @property
    def col_names(self):
        return list(self.columns.keys())

    @property
    def dtypes(self):
        cdef dict result = {}
        cdef str name
        cdef BaseColumn col
        for name, col in self.columns.items():
            if isinstance(col, IntColumn):
                result[name] = int
            elif isinstance(col, DoubleColumn):
                result[name] = float
            elif isinstance(col, Char256Column_SoA):
                result[name] = Char256
            elif isinstance(col, UCS2Column_SoA):
                result[name] = str
            elif isinstance(col, TimestampColumn):
                result[name] = Timestamp
        return result

    @cython.binding(False)
    def rename(self, dict mapper):
        cdef FastDataFrame result = self.copy()
        cdef str old_name, new_name
        for old_name, new_name in mapper.items():
            if old_name in result.columns:
                result.columns[new_name] = result.columns.pop(old_name)
        return result

    @cython.binding(False)
    def describe(self):
        cdef dict stats = {}
        cdef str name
        cdef Py_ssize_t col_len
        for name, col_obj in self.columns.items():
            col_len = col_obj.length
            if isinstance(col_obj, (IntColumn, DoubleColumn)):
                stats[name] = {
                    'count': col_len,
                    'mean': col_obj.mean(),
                    'std': col_obj.std(),
                    'min': col_obj.min(),
                    '50%': col_obj.median(),
                    'max': col_obj.max(),
                }
            elif isinstance(col_obj, (Char256Column_SoA, UCS2Column_SoA)):
                stats[name] = {'count': col_len}
        return stats

    @cython.binding(False)
    def isnull(self):
        cdef FastDataFrame result = FastDataFrame()
        result.df_length = self.df_length
        cdef str name
        cdef BaseColumn col
        cdef IntColumn new_col
        cdef Py_ssize_t i
        cdef double* d_data
        for name, col in self.columns.items():
            new_col = IntColumn(self.df_length)
            new_col.length = self.df_length
            if isinstance(col, DoubleColumn):
                d_data = (<DoubleColumn>col).data_c
                with nogil:
                    for i in range(self.df_length):
                        new_col.data_c[i] = 1 if d_data[i] != d_data[i] else 0
            else:
                with nogil:
                    for i in range(self.df_length):
                        new_col.data_c[i] = 0
            result.columns[name] = new_col
        return result

    @cython.binding(False)
    def fillna(self, double value, col_names=None):
        cdef FastDataFrame result = self.copy()
        cdef str name
        cdef BaseColumn col
        cdef Py_ssize_t i
        cdef double* d_data
        if col_names is None:
            col_names = list(self.columns.keys())
        for name in col_names:
            if name not in result.columns: continue
            col = result.columns[name]
            if isinstance(col, DoubleColumn):
                d_data = (<DoubleColumn>col).data_c
                with nogil:
                    for i in range(self.df_length):
                        if d_data[i] != d_data[i]:
                            d_data[i] = value
        return result

    @cython.binding(False)
    def dropna(self):
        if self.df_length == 0: return self.copy()
        cdef Py_ssize_t n = self.df_length
        cdef char* keep_mask = <char*>malloc(n * sizeof(char))
        if not keep_mask: raise MemoryError()
        cdef Py_ssize_t i
        cdef double* d_data
        cdef str name
        cdef BaseColumn col

        for i in range(n): keep_mask[i] = 1

        for name, col in self.columns.items():
            if isinstance(col, DoubleColumn):
                d_data = (<DoubleColumn>col).data_c
                with nogil:
                    for i in range(n):
                        if keep_mask[i] and d_data[i] != d_data[i]:
                            keep_mask[i] = 0

        cdef Py_ssize_t keep_count = 0
        for i in range(n):
            if keep_mask[i]: keep_count += 1

        if keep_count == n:
            free(keep_mask)
            return self.copy()
        if keep_count == 0:
            free(keep_mask)
            return FastDataFrame()

        cdef Py_ssize_t* keep_idx = <Py_ssize_t*>malloc(keep_count * sizeof(Py_ssize_t))
        if not keep_idx: raise MemoryError()
        cdef Py_ssize_t j = 0
        for i in range(n):
            if keep_mask[i]:
                keep_idx[j] = i
                j += 1
        free(keep_mask)
        cdef FastDataFrame result = self._take(keep_idx, keep_count)
        free(keep_idx)
        return result
