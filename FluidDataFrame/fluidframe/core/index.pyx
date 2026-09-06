# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

cimport cython
from fluidframe.core._base cimport aligned_malloc, os_aligned_free
from fluidframe.core.intcolumn cimport IntColumn
from fluidframe.core.index cimport IndexBase, IntIndex, RangeIndex, MultiIndex



cdef class IndexBase:
    cdef IndexBase _take(self, Py_ssize_t* indices, Py_ssize_t n):
        raise NotImplementedError()

    cdef IndexBase _concat(self, IndexBase other):
        raise NotImplementedError()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def get_loc(self, object key):
        raise NotImplementedError()

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def values(self):
        raise NotImplementedError()


cdef class IntIndex(IndexBase):
    def __cinit__(self, IntColumn col):
        self.data = col
        self.length = col.length
        self.index_type = INDEX_INT

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def values(self):
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

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def get_loc(self, object key):
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

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def get_loc(self, object key):
        if not isinstance(key, int): raise KeyError(key)
        cdef Py_ssize_t val = <Py_ssize_t>key
        cdef Py_ssize_t diff = val - self.start

        # O(1) 数学验证
        if diff % self.step != 0: raise KeyError(key)
        cdef Py_ssize_t idx = diff // self.step
        if idx < 0 or idx >= self.length: raise KeyError(key)
        return idx

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def values(self):
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
        self._level_maps = None

    cdef void _build_level_maps(self):
        """延迟构建：将 values 转换为 codes 的 O(1) 反向映射"""
        self._level_maps = []
        for level in self.levels:
            self._level_maps.append({val: idx for idx, val in enumerate(level)})

    cdef void _build_hash_cache(self):
        """黑魔法：直接使用 C 内存字节流作为哈希键，无 Python 对象冗余开销"""
        cdef Py_ssize_t i, j
        cdef int n_levels = self.nlevels

        cdef int** ptrs = <int**>aligned_malloc(n_levels * sizeof(int*))
        cdef int* buf = <int*>aligned_malloc(n_levels * sizeof(int))
        if not ptrs or not buf:
            if ptrs: os_aligned_free(ptrs)
            if buf: os_aligned_free(buf)
            raise MemoryError()

        cdef Py_ssize_t bytes_len = n_levels * sizeof(int)
        cdef bytes b_key

        try:
            for j in range(n_levels):
                ptrs[j] = (<IntColumn>self.codes[j]).data_c

            self._hash_cache = {}

            for i in range(self.length):
                for j in range(n_levels):
                    buf[j] = ptrs[j][i]

                # 将 int 数组内存强制转换为 bytes
                b_key = (<char*>buf)[:bytes_len]

                if b_key not in self._hash_cache:
                    self._hash_cache[b_key] = [i]
                else:
                    (<list>self._hash_cache[b_key]).append(i)
        finally:
            os_aligned_free(ptrs)
            os_aligned_free(buf)

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

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def get_loc(self, object key):
        cdef tuple target = key if isinstance(key, tuple) else (key,)
        cdef Py_ssize_t target_len = len(target)
        if target_len > self.nlevels: raise KeyError(key)

        if self._level_maps is None:
            self._build_level_maps()

        cdef int* target_buf = <int*>aligned_malloc(target_len * sizeof(int))
        if not target_buf: raise MemoryError()

        cdef int level_idx
        cdef object val
        cdef dict level_map
        cdef list matches = []
        cdef Py_ssize_t i, j
        cdef bint match
        cdef int** ptrs
        cdef bytes target_b_key

        try:
            for level_idx in range(target_len):
                level_map = self._level_maps[level_idx]
                val = target[level_idx]
                if val not in level_map:
                    raise KeyError(val)
                target_buf[level_idx] = level_map[val]

            if target_len < self.nlevels:
                ptrs = <int**>aligned_malloc(target_len * sizeof(int*))
                if not ptrs: raise MemoryError()
                try:
                    for j in range(target_len):
                        ptrs[j] = (<IntColumn>self.codes[j]).data_c

                    # C语言级裸奔：内存吞吐极限
                    for i in range(self.length):
                        match = True
                        for j in range(target_len):
                            if ptrs[j][i] != target_buf[j]:
                                match = False
                                break
                        if match:
                            matches.append(i)
                finally:
                    os_aligned_free(ptrs)

                if not matches: raise KeyError(key)
                return matches[0] if len(matches) == 1 else matches

            if self._hash_cache is None:
                self._build_hash_cache()

            target_b_key = (<char*>target_buf)[:target_len * sizeof(int)]
            matches = self._hash_cache.get(target_b_key)

            if matches is None:
                raise KeyError(key)

            return matches[0] if len(matches) == 1 else matches
        finally:
            os_aligned_free(target_buf)

    @cython.binding(False)
    @cython.always_allow_keywords(False)
    def values(self):
        cdef list result = []
        cdef Py_ssize_t i, j
        cdef int code_val
        cdef IntColumn col
        for i in range(self.length):
            row = []
            for j in range(self.nlevels):
                col = <IntColumn>self.codes[j]
                code_val = col.data_c[i]
                if code_val < 0 or code_val >= len(self.levels[j]):
                    row.append(None)
                else:
                    row.append(self.levels[j][code_val])
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