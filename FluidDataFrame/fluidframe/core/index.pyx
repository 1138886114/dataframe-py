# distutils: language = c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

cimport cython
from fluidframe.core._base cimport aligned_malloc, os_aligned_free
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.index cimport IndexBase, IntIndex, RangeIndex, MultiIndex
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector


cdef class IndexBase:
    cdef IndexBase _take(self, Py_ssize_t* indices, Py_ssize_t n):
        raise NotImplementedError()

    cdef IndexBase _concat(self, IndexBase other):
        raise NotImplementedError()

    cpdef get_loc(self, object key):
        raise NotImplementedError()

    cpdef list values(self):
        raise NotImplementedError()


cdef class IntIndex(IndexBase):
    def __cinit__(self, IntColumn col):
        self.data = col
        self.length = col.length
        self.index_type = INDEX_INT

    cpdef list values(self):
        cdef list result = []
        cdef Py_ssize_t i
        for i in range(self.length):
            result.append(self.data.data_c[i])
        return result

    cdef IndexBase _take(self, Py_ssize_t* indices, Py_ssize_t n):
        cdef IntColumn new_col = IntColumn(n)
        new_col.length = n
        cdef int* old_data = self.data.data_c
        cdef int* new_data = new_col.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                new_data[i] = old_data[indices[i]]
        return IntIndex(new_col)

    cpdef get_loc(self, object key):
        cdef int target = <int>key
        cdef list matches = []
        cdef Py_ssize_t i
        cdef int* data_ptr = self.data.data_c
        for i in range(self.length):
            if data_ptr[i] == target:
                matches.append(i)
        if not matches: raise KeyError(key)
        if len(matches) == 1: return matches[0]
        return matches

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def append(self, int val):
        self.data.append(val)
        self.length = self.data.length

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def pop_back(self):
        if self.length == 0:
            raise IndexError("pop from empty Index")
        cdef int val = self.data.pop_back()
        self.length = self.data.length
        return val

    cdef IndexBase _concat(self, IndexBase other):
        if not isinstance(other, IntIndex):
            raise TypeError("Cannot concat IntIndex with non-IntIndex")
        cdef IntIndex o = <IntIndex>other
        cdef IntColumn new_col = IntColumn(self.length + o.length)
        new_col.length = self.length + o.length
        cdef Py_ssize_t i
        cdef int* src1 = self.data.data_c
        cdef int* src2 = o.data.data_c
        cdef int* dst = new_col.data_c
        with nogil:
            for i in range(self.length):
                dst[i] = src1[i]
            for i in range(o.length):
                dst[self.length + i] = src2[i]
        return IntIndex(new_col)


cdef class RangeIndex(IndexBase):
    def __cinit__(self, Py_ssize_t start, Py_ssize_t length, Py_ssize_t step=1):
        self.index_type = INDEX_RANGE
        self.start = start
        self.length = length
        self.step = step
        if step == 0: raise ValueError("Step cannot be zero.")

    cdef IndexBase _take(self, Py_ssize_t* indices, Py_ssize_t n):
        cdef IntColumn new_col = IntColumn(n)
        new_col.length = n
        cdef int* out_data = new_col.data_c
        cdef Py_ssize_t i
        with nogil:
            for i in range(n):
                out_data[i] = <int>(self.start + indices[i] * self.step)
        return IntIndex(new_col)

    cpdef get_loc(self, object key):
        if not isinstance(key, int): raise KeyError(key)
        cdef Py_ssize_t val = <Py_ssize_t>key
        cdef Py_ssize_t diff = val - self.start

        # O(1) 数学验证
        if diff % self.step != 0: raise KeyError(key)
        cdef Py_ssize_t idx = diff // self.step
        if idx < 0 or idx >= self.length: raise KeyError(key)
        return idx

    @cython.always_allow_keywords(False)
    cpdef list values(self):
        cdef list result = []
        cdef Py_ssize_t i
        for i in range(self.length):
            result.append(self.start + i * self.step)
        return result

    cdef IndexBase _concat(self, IndexBase other):
        cdef RangeIndex o
        cdef IntColumn new_col
        cdef list v1, v2
        cdef Py_ssize_t i
        cdef int* dst
        if isinstance(other, RangeIndex):
            o = <RangeIndex>other
            if self.step == o.step and self.start + self.length * self.step == o.start:
                return RangeIndex(self.start, self.length + o.length, self.step)
        new_col = IntColumn(self.length + other.length)
        new_col.length = self.length + other.length
        v1 = self.values()
        v2 = other.values()
        dst = new_col.data_c
        for i in range(self.length):
            dst[i] = <int>v1[i]
        for i in range(other.length):
            dst[self.length + i] = <int>v2[i]
        return IntIndex(new_col)

cdef class MultiIndex(IndexBase):
    def __cinit__(self, list levels, list codes, list names=None):
        self.index_type = INDEX_MULTI
        self.levels = levels
        self.codes = codes
        self.nlevels = len(levels)
        self.names = names if names else [None] * self.nlevels
        self.length = (<IntColumn>codes[0]).length if len(codes) > 0 else 0
        self._hash_cache = None

    cdef void _build_hash_cache(self):
        """Build hash cache for O(1) lookups."""
        cdef unordered_map[long long, vector[Py_ssize_t]] hash_map
        cdef Py_ssize_t i, j
        cdef IntColumn col
        cdef long long hash_val
        cdef list row_codes

        for i in range(self.length):
            row_codes = []
            for j in range(self.nlevels):
                col = <IntColumn>self.codes[j]
                row_codes.append(col.data_c[i])
            # Use a simple hash: sum of codes weighted by position
            hash_val = 0
            for j in range(self.nlevels):
                hash_val = hash_val * 1000000 + row_codes[j]
            hash_map[hash_val].push_back(i)

        # Convert to Python dict for caching
        self._hash_cache = {}
        for it in hash_map:
            key = it.first
            indices = list(it.second)
            self._hash_cache[key] = indices

    cdef IndexBase _take(self, Py_ssize_t* indices, Py_ssize_t n):
        cdef list new_codes = []
        cdef IntColumn col, new_col
        cdef Py_ssize_t i
        for col in self.codes:
            new_col = IntColumn(n)
            new_col.length = n
            with nogil:
                for i in range(n):
                    new_col.data_c[i] = col.data_c[indices[i]]
            new_codes.append(new_col)
        return MultiIndex(self.levels, new_codes, self.names)

    cpdef get_loc(self, object key):
        cdef tuple target = key if isinstance(key, tuple) else (key,)
        cdef Py_ssize_t target_len = len(target)
        if target_len > self.nlevels: raise KeyError(key)

        cdef list target_codes = []
        cdef int level_idx
        for level_idx in range(target_len):
            level_values = self.levels[level_idx]
            try:
                code = level_values.index(target[level_idx])
                target_codes.append(code)
            except ValueError:
                raise KeyError(target[level_idx])

        # Partial key: linear scan matching first target_len levels
        cdef IntColumn col
        cdef list matches = []
        cdef Py_ssize_t i, j
        cdef bint match
        if target_len < self.nlevels:
            for i in range(self.length):
                match = True
                for j in range(target_len):
                    col = <IntColumn>self.codes[j]
                    if col.data_c[i] != target_codes[j]:
                        match = False
                        break
                if match:
                    matches.append(i)
            if not matches:
                raise KeyError(key)
            if len(matches) == 1:
                return matches[0]
            return matches

        # Full key: use hash cache for O(1) lookup
        cdef long long hash_val = 0
        for level_idx in range(target_len):
            hash_val = hash_val * 1000000 + target_codes[level_idx]

        if self._hash_cache is None:
            self._build_hash_cache()

        if hash_val in self._hash_cache:
            matches = self._hash_cache[hash_val]
            if not matches:
                raise KeyError(key)
            if len(matches) == 1:
                return matches[0]
            return matches

        raise KeyError(key)

    @cython.always_allow_keywords(False)
    cpdef list values(self):
        cdef list result = []
        cdef Py_ssize_t i, j
        cdef IntColumn col
        for i in range(self.length):
            row = []
            for j in range(self.nlevels):
                col = <IntColumn>self.codes[j]
                row.append(self.levels[j][col.data_c[i]])
            result.append(tuple(row))
        return result

    cdef IndexBase _concat(self, IndexBase other):
        cdef MultiIndex o
        cdef Py_ssize_t total, i, j
        cdef list new_codes
        cdef IntColumn col, new_col, o_col
        cdef int* src1
        cdef int* src2
        cdef int* dst
        if not isinstance(other, MultiIndex):
            raise TypeError("Cannot concat MultiIndex with non-MultiIndex")
        o = <MultiIndex>other
        if self.nlevels != o.nlevels:
            raise ValueError("MultiIndex nlevels mismatch")
        total = self.length + o.length
        new_codes = []
        for j in range(self.nlevels):
            col = <IntColumn>self.codes[j]
            o_col = <IntColumn>o.codes[j]
            new_col = IntColumn(total)
            new_col.length = total
            src1 = col.data_c
            src2 = o_col.data_c
            dst = new_col.data_c
            with nogil:
                for i in range(self.length):
                    dst[i] = src1[i]
                for i in range(o.length):
                    dst[self.length + i] = src2[i]
            new_codes.append(new_col)
        return MultiIndex(self.levels, new_codes, self.names)