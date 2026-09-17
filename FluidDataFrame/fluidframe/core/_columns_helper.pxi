# ============================================================
#  _columns_helper.pxi
#  列处理模块（_columns.pyx）的模块级独立辅助函数集合。
#  通过 `include "_columns_helper.pxi"` 文本拼入列模块，
#  保持接口签名与 _columns.pxd 中的声明一一对应。
#  覆盖：内存对齐分配、类型转换解析、qsort 比较器、
#        C 层字符串/ucs2 助手、历法算法、日期/时间/日期时间解析等。
# ============================================================


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
#  qsort / 二分查找等纯 C 辅助
# ============================================================
cdef int _cmp_int(const void* a, const void* b) noexcept nogil:
    cdef int va = (<const int*>a)[0]
    cdef int vb = (<const int*>b)[0]
    if va < vb: return -1
    if va > vb: return 1
    return 0

cdef inline bint _binary_search(int* arr, Py_ssize_t n, int target) noexcept nogil:
    cdef Py_ssize_t low = 0, high = n - 1, mid
    cdef int val
    while low <= high:
        mid = (low + high) // 2
        val = arr[mid]
        if val == target:
            return True
        elif val < target:
            low = mid + 1
        else:
            high = mid - 1
    return False


cdef int64_t US_PER_DAY = 86400000000LL


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

#  底层纯 C 数学/度量算子 (零分支与 SIMD 优化)
# ============================================================

cdef int _cmp_double(const void* a, const void* b) noexcept nogil:
    """提供给 qsort 和 bsearch 的严格全序比较函数，安全处理 NaN"""
    cdef double va = (<const double*>a)[0]
    cdef double vb = (<const double*>b)[0]
    cdef bint anan = (va != va)
    cdef bint bnan = (vb != vb)
    if anan and bnan: return 0
    if anan: return -1
    if bnan: return 1
    if va < vb: return -1
    if va > vb: return 1
    return 0

cdef int _cmp_double_asc(const void* a, const void* b) noexcept nogil:
    """升序比较，用于 qsort (NaN 视为最大值，排最后)"""
    cdef double va = (<const double*>a)[0]
    cdef double vb = (<const double*>b)[0]
    cdef bint anan = (va != va)
    cdef bint bnan = (vb != vb)
    if anan and bnan: return 0
    if anan: return 1
    if bnan: return -1
    if va < vb: return -1
    if va > vb: return 1
    return 0


cdef void _col_to_double(object col, double* buf, Py_ssize_t n) except *:
    cdef Py_ssize_t i
    cdef int* int_data
    cdef double* double_data

    if isinstance(col, DoubleColumn):
        double_data = (<DoubleColumn>col).data_c
        if n > 0:
            memcpy(buf, double_data, n * sizeof(double))
    elif isinstance(col, IntColumn):
        int_data = (<IntColumn>col).data_c
        with nogil:
            for i in range(n):
                buf[i] = <double>int_data[i]
    else:
        raise TypeError("Expected IntColumn or DoubleColumn")


cdef inline bint _is_lower(unsigned char c) noexcept nogil:
    return c >= 97 and c <= 122

cdef inline bint _is_upper(unsigned char c) noexcept nogil:
    return c >= 65 and c <= 90

cdef inline bint _is_alpha(unsigned char c) noexcept nogil:
    return _is_lower(c) or _is_upper(c)

cdef inline bint _is_digit(unsigned char c) noexcept nogil:
    return c >= 48 and c <= 57

cdef inline bint _is_alnum(unsigned char c) noexcept nogil:
    return _is_alpha(c) or _is_digit(c)

cdef inline bint _is_space(unsigned char c) noexcept nogil:
    return c == 32 or (c >= 9 and c <= 13)

# ============================================================
#  C helper functions (全部保留，提供原始功能支撑)
# ============================================================

cdef inline bint _c_str_contains(const char* source, const char* target) noexcept nogil:
    return strstr(source, target) != NULL

cdef inline bint _c_startswith(const char* data, const char* pat, size_t pat_len) noexcept nogil:
    if strlen(data) < pat_len:
        return 0
    return strncmp(data, pat, pat_len) == 0

cdef inline bint _c_endswith(const char* data, const char* pat, size_t pat_len) noexcept nogil:
    cdef size_t n = strlen(data)
    if n < pat_len:
        return 0
    return strncmp(data + (n - pat_len), pat, pat_len) == 0

cdef inline Py_ssize_t _c_find(const char *data, const char *pat, size_t pat_len) noexcept nogil:
    cdef const char* match = strstr(data, pat)
    if match != NULL:
        return match - data
    return -1

cdef inline Py_ssize_t _c_rfind(const char *data, const char *pat, size_t pat_len) noexcept nogil:
    cdef Py_ssize_t n = <Py_ssize_t>strlen(data)
    cdef Py_ssize_t p_len = <Py_ssize_t>pat_len
    if p_len == 0:
        return n
    if p_len > n:
        return -1
    cdef Py_ssize_t i
    for i in range(n - p_len, -1, -1):
        if strncmp(data + i, pat, pat_len) == 0:
            return i
    return -1

cdef inline Py_ssize_t _c_count(const char *data, const char *pat, size_t pat_len) noexcept nogil:
    if pat_len == 0:
        return <Py_ssize_t>strlen(data) + 1
    cdef Py_ssize_t cnt = 0
    cdef const char* match = strstr(data, pat)
    while match != NULL:
        cnt += 1
        match = strstr(match + pat_len, pat)
    return cnt


# ============================================================
#  Ascii 逐行分类掩码（共用模式内核，消除 isdigit/isalpha/... 重复循环）
# ============================================================

cdef inline void _c_pred_mask(const char* data, const Py_ssize_t* offs, Py_ssize_t n,
                              int* out, int mode) noexcept nogil:
    """Ascii 逐行分类掩码。mode: 0=digit 1=alpha 2=alnum 3=space。
    mode 分派外提：4 条循环副本只做一次分支，内层逐字符零 mode 判断。"""
    cdef Py_ssize_t i, j, sl
    cdef unsigned char c
    cdef bint valid
    if mode == 0:
        for i in range(n):
            sl = offs[i + 1] - offs[i] - 1
            if sl == 0:
                out[i] = 0
                continue
            valid = 1
            for j in range(sl):
                c = <unsigned char>data[offs[i] + j]
                if not _is_digit(c):  valid = 0; break
            out[i] = valid
    elif mode == 1:
        for i in range(n):
            sl = offs[i + 1] - offs[i] - 1
            if sl == 0:
                out[i] = 0
                continue
            valid = 1
            for j in range(sl):
                c = <unsigned char>data[offs[i] + j]
                if not _is_alpha(c):  valid = 0; break
            out[i] = valid
    elif mode == 2:
        for i in range(n):
            sl = offs[i + 1] - offs[i] - 1
            if sl == 0:
                out[i] = 0
                continue
            valid = 1
            for j in range(sl):
                c = <unsigned char>data[offs[i] + j]
                if not _is_alnum(c):  valid = 0; break
            out[i] = valid
    else:
        for i in range(n):
            sl = offs[i + 1] - offs[i] - 1
            if sl == 0:
                out[i] = 0
                continue
            valid = 1
            for j in range(sl):
                c = <unsigned char>data[offs[i] + j]
                if not _is_space(c):  valid = 0; break
            out[i] = valid

cdef inline void _c_case_mask(const char* data, const Py_ssize_t* offs, Py_ssize_t n,
                              int* out, bint upper) noexcept nogil:
    """Ascii 逐行大小写类型掩码（要求至少含一个字母）。upper=0→islower, 1→isupper。
    upper 分派外提：两条循环副本只做一次分支，内层逐字符零 upper 判断。"""
    cdef Py_ssize_t i, j, sl
    cdef unsigned char c
    cdef bint has_letter, all_ok
    if upper:
        for i in range(n):
            sl = offs[i + 1] - offs[i] - 1
            if sl == 0:
                out[i] = 0
                continue
            has_letter = 0
            all_ok = 1
            for j in range(sl):
                c = <unsigned char>data[offs[i] + j]
                if _is_alpha(c):
                    has_letter = 1
                    if not _is_upper(c):  all_ok = 0; break
            out[i] = 1 if (has_letter and all_ok) else 0
    else:
        for i in range(n):
            sl = offs[i + 1] - offs[i] - 1
            if sl == 0:
                out[i] = 0
                continue
            has_letter = 0
            all_ok = 1
            for j in range(sl):
                c = <unsigned char>data[offs[i] + j]
                if _is_alpha(c):
                    has_letter = 1
                    if not _is_lower(c):  all_ok = 0; break
            out[i] = 1 if (has_letter and all_ok) else 0


# ============================================================
#  布尔逐元素位组合内核（BoolColumn 逻辑运算共用）
# ============================================================

cdef inline void _bool_bitwise(const unsigned char* s, const unsigned char* o,
                               unsigned char* r, Py_ssize_t n, int op) noexcept nogil:
    """op: 0=and 1=or 2=xor。op 分派外提：内层零 op 判断。"""
    cdef Py_ssize_t i
    if op == 0:
        for i in range(n): r[i] = s[i] & o[i]
    elif op == 1:
        for i in range(n): r[i] = s[i] | o[i]
    else:
        for i in range(n): r[i] = s[i] ^ o[i]

cdef inline void _bool_not(const unsigned char* s, unsigned char* r, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    for i in range(n):
        r[i] = s[i] ^ 1


# ============================================================
#  AsciiColumn class definition
# ============================================================

cdef inline void _copy_py_str_to_ucs2(str val, Py_UCS2* dest, Py_ssize_t length) noexcept:
    """
    极速、安全的摄入器：解析底层 Unicode 物理布局并批量直拷
    超限宽字符 (Emoji等) 会在这里被永久替换为 '?' (0x003F)
    """
    cdef int kind = PyUnicode_KIND(val)
    cdef void* data = PyUnicode_DATA(val)
    cdef Py_ssize_t i
    cdef Py_UCS4 ch

    if kind == PyUnicode_1BYTE_KIND:
        # ASCII / Latin1 数据，直接宽化拷贝
        for i in range(length):
            dest[i] = <Py_UCS2>(<Py_UCS1*>data)[i]
    elif kind == PyUnicode_2BYTE_KIND:
        # BMP 基本多文种平面 (包含中文)，安全且完美对齐，内存直拷
        memcpy(dest, data, length * sizeof(Py_UCS2))
    elif kind == PyUnicode_4BYTE_KIND:
        # 包含了 Emoji 的宽字符字符串，清洗拦截
        for i in range(length):
            ch = (<Py_UCS4*>data)[i]
            dest[i] = <Py_UCS2>ch if ch <= 0xFFFF else 0x003F
    else:
        # 安全降级，理论上 Python3.3+ 不会走到这里
        for i in range(length):
            ch = PyUnicode_READ_CHAR(val, i)
            dest[i] = <Py_UCS2>ch if ch <= 0xFFFF else 0x003F

cdef inline Py_UCS2* _ucs2_strstr(Py_UCS2* haystack, int h_len, Py_UCS2* needle, int n_len) noexcept nogil:
    if n_len == 0: return haystack
    if h_len < n_len: return NULL
    cdef int i, j
    cdef bint match
    for i in range(h_len - n_len + 1):
        if haystack[i] == needle[0]:
            match = True
            for j in range(1, n_len):
                if haystack[i+j] != needle[j]:
                    match = False
                    break
            if match: return haystack + i
    return NULL

cdef inline bint _ucs2_startswith(Py_UCS2* s, Py_ssize_t s_len, Py_UCS2* prefix, Py_ssize_t p_len) noexcept nogil:
    if p_len == 0: return True
    if s_len < p_len: return False
    cdef Py_ssize_t i
    for i in range(p_len):
        if s[i] != prefix[i]: return False
    return True

cdef inline bint _ucs2_endswith(Py_UCS2* s, Py_ssize_t s_len, Py_UCS2* suffix, Py_ssize_t suf_len) noexcept nogil:
    if suf_len == 0: return True
    if s_len < suf_len: return False
    cdef Py_ssize_t i
    cdef Py_ssize_t offset = s_len - suf_len
    for i in range(suf_len):
        if s[offset + i] != suffix[i]: return False
    return True

cdef inline int _cmp_ucs2_buffers(Py_UCS2* a, Py_ssize_t len_a, Py_UCS2* b, Py_ssize_t len_b) noexcept nogil:
    cdef Py_ssize_t min_len = len_a if len_a < len_b else len_b
    cdef Py_ssize_t i
    for i in range(min_len):
        if a[i] < b[i]: return -1
        if a[i] > b[i]: return 1
    if len_a < len_b: return -1
    if len_a > len_b: return 1
    return 0


#  静态常量查找表（彻底消除分支预测失败）
# ============================================================

cdef struct NameEntry:
    const char* name
    size_t len

cdef NameEntry[7] _DAY_NAMES = [
    NameEntry(b"Monday", 6),
    NameEntry(b"Tuesday", 7),
    NameEntry(b"Wednesday", 9),
    NameEntry(b"Thursday", 8),
    NameEntry(b"Friday", 6),
    NameEntry(b"Saturday", 8),
    NameEntry(b"Sunday", 6)
]

cdef NameEntry[13] _MONTH_NAMES = [
    NameEntry(b"", 0),
    NameEntry(b"January", 7),
    NameEntry(b"February", 8),
    NameEntry(b"March", 5),
    NameEntry(b"April", 5),
    NameEntry(b"May", 3),
    NameEntry(b"June", 4),
    NameEntry(b"July", 4),
    NameEntry(b"August", 6),
    NameEntry(b"September", 9),
    NameEntry(b"October", 7),
    NameEntry(b"November", 8),
    NameEntry(b"December", 8)
]

cdef int[13] _MONTH_DAYS_NORMAL = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
cdef int[13] _DOY_CUMULATIVE_NORMAL = [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

# ============================================================
#  现代顶级历法算法 (Howard Hinnant Algorithm)
#  专为 1970-01-01 纪元设计，无除法常数表、无循环、零分支
# ============================================================

cdef inline bint _is_leap_year(int y) noexcept nogil:
    return (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)

cdef inline int _days_in_month(int y, int m) noexcept nogil:
    if m == 2:
        return 29 if _is_leap_year(y) else 28
    if m >= 1 and m <= 12:
        return _MONTH_DAYS_NORMAL[m]
    return 30

cdef inline int32_t _ymd_to_days(int y, int m, int d) noexcept nogil:
    """YYYY-MM-DD -> 1970-01-01 起的天数 (Days since Unix Epoch)"""
    cdef int a = (14 - m) // 12
    cdef int y2 = y + 4800 - a
    cdef int m2 = m + 12 * a - 3
    return <int32_t>(d + (153 * m2 + 2) // 5 + 365 * y2 + y2 // 4 - y2 // 100 + y2 // 400 - 32045 - 2440588)

cdef inline void _days_to_ymd(int32_t days, int* y, int* m, int* d) noexcept nogil:
    """1970-01-01 起的天数 -> YYYY-MM-DD"""
    cdef long long daynum = <long long>days + 2440588
    cdef long long a, b, c, d2, e, m2
    a = daynum + 32044
    b = (4 * a + 3) // 146097
    c = a - 146097 * b // 4
    d2 = (4 * c + 3) // 1461
    e = c - 1461 * d2 // 4
    m2 = (5 * e + 2) // 153
    d[0] = <int>(e - (153 * m2 + 2) // 5 + 1)
    m[0] = <int>(m2 + 3 - 12 * (m2 // 10))
    y[0] = <int>(100 * b + d2 - 4800 + m2 // 10)

cdef inline int32_t _parse_date_str(str s) noexcept:
    """极速解析 'YYYYMMDD' 或 'YYYY-MM-DD' 字符串为 Unix 纪元天数"""
    cdef bytes b = s.encode('ascii', errors='ignore')
    cdef const unsigned char* buf = <const unsigned char*>b
    cdef int digits[10]
    cdef int nd = 0, i = 0, n = len(b)
    while i < n and nd < 8:
        if buf[i] >= 48 and buf[i] <= 57:
            digits[nd] = buf[i] - 48
            nd += 1
        i += 1
    if nd < 8:
        return 0
    cdef int y = digits[0] * 1000 + digits[1] * 100 + digits[2] * 10 + digits[3]
    cdef int m = digits[4] * 10 + digits[5]
    cdef int d = digits[6] * 10 + digits[7]
    return _ymd_to_days(y, m, d)


# ============================================================
#  DateColumn 类实现
# ============================================================

cdef tuple BOOL_STRS = ('False', 'True')


#  时间常数定义 (基于 Unix 1970-01-01 00:00:00 纪元)
# ============================================================

cdef int64_t US_PER_SEC = 1000000
cdef int64_t SEC_PER_DAY = 86400


cdef inline int64_t _parse_datetime_str(str s) except -1:
    """解析 'YYYY-MM-DD HH:MM:SS' 或 'YYYY-MM-DD' 为 1970 纪元微秒数"""
    cdef bytes b = s.encode('ascii', errors='ignore')
    cdef const unsigned char* buf = <const unsigned char*>b
    cdef int max_len = len(b)
    cdef int digits[20]
    cdef int ndigits = 0
    cdef int i = 0
    cdef int year = 0, month = 1, day = 1, hour = 0, minute = 0, second = 0
    cdef int32_t days
    cdef int64_t us

    while i < max_len and ndigits < 20:
        if buf[i] >= 48 and buf[i] <= 57:
            digits[ndigits] = buf[i] - 48
            ndigits += 1
        i += 1

    if ndigits < 8:
        raise ValueError(f"Invalid datetime string: '{s}'")

    year = digits[0]*1000 + digits[1]*100 + digits[2]*10 + digits[3]
    month = digits[4]*10 + digits[5]
    day = digits[6]*10 + digits[7]
    if ndigits >= 10:
        hour = digits[8]*10 + digits[9]
    if ndigits >= 12:
        minute = digits[10]*10 + digits[11]
    if ndigits >= 14:
        second = digits[12]*10 + digits[13]

    # 直接基于 1970 纪元天数换算
    days = _ymd_to_days(year, month, day)
    us = <int64_t>days * US_PER_DAY + <int64_t>hour * 3600000000LL + <int64_t>minute * 60000000LL + <int64_t>second * US_PER_SEC
    return us

cdef inline void _datetime_to_ymdhms(int64_t us, int* y, int* m, int* d, int* h, int* mi, int* s) noexcept nogil:
    """1970 纪元微秒数分解为 YMD HMS"""
    cdef int64_t total_seconds = us / US_PER_SEC
    cdef int32_t days = <int32_t>(total_seconds / SEC_PER_DAY)
    cdef int rem = <int>(total_seconds - (<int64_t>days * SEC_PER_DAY))
    if rem < 0:
        rem += SEC_PER_DAY
        days -= 1

    h[0] = rem / 3600
    cdef int rem_m = rem - h[0] * 3600
    mi[0] = rem_m / 60
    s[0] = rem_m - mi[0] * 60

    _days_to_ymd(days, y, m, d)

cdef inline void _format_datetime_c(int64_t us, char* dest) noexcept nogil:
    """纯 C 极速格式化 YYYY-MM-DD HH:MM:SS（零堆内存分配）"""
    cdef int y, m, d, h, mi, s
    _datetime_to_ymdhms(us, &y, &m, &d, &h, &mi, &s)
    dest[0] = <char>(48 + ((y / 1000) % 10))
    dest[1] = <char>(48 + ((y / 100) % 10))
    dest[2] = <char>(48 + ((y / 10) % 10))
    dest[3] = <char>(48 + (y % 10))
    dest[4] = b'-'
    dest[5] = <char>(48 + (m / 10))
    dest[6] = <char>(48 + (m % 10))
    dest[7] = b'-'
    dest[8] = <char>(48 + (d / 10))
    dest[9] = <char>(48 + (d % 10))

    if h == 0 and mi == 0 and s == 0:
        dest[10] = 0
    else:
        dest[10] = b' '
        dest[11] = <char>(48 + (h / 10))
        dest[12] = <char>(48 + (h % 10))
        dest[13] = b':'
        dest[14] = <char>(48 + (mi / 10))
        dest[15] = <char>(48 + (mi % 10))
        dest[16] = b':'
        dest[17] = <char>(48 + (s / 10))
        dest[18] = <char>(48 + (s % 10))
        dest[19] = 0

cdef inline str _datetime_to_str(int64_t us):
    cdef char buf[32]
    _format_datetime_c(us, buf)
    return buf.decode('ascii')

cdef int _cmp_int64(const void* a, const void* b) noexcept nogil:
    cdef int64_t va = (<const int64_t*>a)[0]
    cdef int64_t vb = (<const int64_t*>b)[0]
    if va < vb: return -1
    if va > vb: return 1
    return 0


#  底层纯 C 高性能时间运算、LUT 查表格式化与零拷贝解析
# ============================================================

# 00-99 两位字符查找表（LUT 优化：消除格式化中的所有除法/分支）
cdef extern from *:
    """
    static const char DIGIT_PAIRS[200] =
        "00010203040506070809"
        "10111213141516171819"
        "20212223242526272829"
        "30313233343536373839"
        "40414243444546474849"
        "50515253545556575859"
        "60616263646566676869"
        "70717273747576777879"
        "80818283848586878889"
        "90919293949596979899";
    """
    const char DIGIT_PAIRS[200]

cdef inline void _split_hms(int32_t secs, int32_t* h, int32_t* m, int32_t* s) noexcept nogil:
    """提取时分秒（钳制安全范围：0 ~ 86399）"""
    if secs < 0:
        secs = (secs % 86400 + 86400) % 86400
    elif secs >= 86400:
        secs = secs % 86400

    cdef int32_t tot_m = secs / 60
    s[0] = secs - tot_m * 60
    h[0] = tot_m / 60
    m[0] = tot_m - h[0] * 60

cdef inline void _format_time_c(int32_t secs, char* dest) noexcept nogil:
    """纯 C 查表就地格式化为 'HH:MM:SS'（零除法、零分支、零堆分配）"""
    cdef int32_t h, m, s
    _split_hms(secs, &h, &m, &s)

    cdef int32_t h_idx = h * 2
    cdef int32_t m_idx = m * 2
    cdef int32_t s_idx = s * 2

    dest[0] = DIGIT_PAIRS[h_idx]
    dest[1] = DIGIT_PAIRS[h_idx + 1]
    dest[2] = b':'
    dest[3] = DIGIT_PAIRS[m_idx]
    dest[4] = DIGIT_PAIRS[m_idx + 1]
    dest[5] = b':'
    dest[6] = DIGIT_PAIRS[s_idx]
    dest[7] = DIGIT_PAIRS[s_idx + 1]
    dest[8] = 0

cdef inline str _time_to_str(int32_t secs):
    cdef char buf[16]
    _format_time_c(secs, buf)
    return buf.decode('ascii')

cdef int32_t _parse_time_str(str s) except -1:
    """极速解析 'HH:MM:SS', 'H:M:S', 'HH:MM' 等时间格式为自午夜起的秒数"""
    cdef bytes b = s.encode('ascii')
    cdef const char* buf = <const char*>b
    cdef Py_ssize_t n = len(b)
    cdef int32_t parts[3]
    parts[0] = 0; parts[1] = 0; parts[2] = 0
    cdef int32_t cur_val = 0
    cdef int32_t part_idx = 0
    cdef Py_ssize_t i
    cdef int c

    for i in range(n):
        c = <int>buf[i]
        if c >= 48 and c <= 57:
            cur_val = cur_val * 10 + (c - 48)
        elif c == 58:
            if part_idx >= 2:
                raise ValueError(f"Invalid time format: '{s}'")
            parts[part_idx] = cur_val
            cur_val = 0
            part_idx += 1
        elif c == 32 or c == 9 or c == 13 or c == 10:
            continue
        else:
            raise ValueError(f"Invalid character in time string: '{s}'")

    parts[part_idx] = cur_val

    if part_idx == 0:
        raise ValueError(f"Invalid time string: '{s}'")

    if parts[0] < 0 or parts[0] > 23 or parts[1] < 0 or parts[1] > 59 or parts[2] < 0 or parts[2] > 59:
        raise ValueError(f"Time component out of bounds in '{s}'")

    return parts[0] * 3600 + parts[1] * 60 + parts[2]


# ============================================================
#  Fused numeric_t helpers（IntColumn / DoubleColumn 泛型去重）
#  int / double 两者算法体完全一致，仅指针类型不同。
#  通过 `numeric_t` 融合类型单份实现，两个类各留薄委托。
# ============================================================
ctypedef fused numeric_t:
    int
    double

cdef inline double _n_quickselect(numeric_t* a, Py_ssize_t n, Py_ssize_t k) noexcept nogil:
    """3-way partition 快速选择，返回第 k 小元素（k 基于已排序副本索引）。"""
    cdef Py_ssize_t l = 0, r = n - 1
    cdef Py_ssize_t i, lt, gt
    cdef numeric_t pivot, tmp

    while l <= r:
        pivot = a[l + (r - l) // 2]
        lt = l
        gt = r
        i = l
        while i <= gt:
            if a[i] < pivot:
                tmp = a[lt]; a[lt] = a[i]; a[i] = tmp
                lt += 1
                i += 1
            elif a[i] > pivot:
                tmp = a[i]; a[i] = a[gt]; a[gt] = tmp
                gt -= 1
            else:
                i += 1

        if k < lt:
            r = lt - 1
        elif k > gt:
            l = gt + 1
        else:
            return <double>a[k]
    return <double>a[k]

cdef double _n_median(numeric_t* ptr, Py_ssize_t n) except *:
    """中位数（内部对齐分配副本 + quickselect）。n == 0 时返回 NAN。"""
    if n == 0:
        return NAN
    cdef numeric_t* temp = <numeric_t*>aligned_malloc(n * sizeof(numeric_t))
    if temp == NULL:
        raise MemoryError()
    memcpy(temp, ptr, n * sizeof(numeric_t))
    cdef Py_ssize_t mid = n // 2
    cdef double v1, v2
    try:
        if n % 2 == 0:
            v1 = _n_quickselect(temp, n, mid - 1)
            v2 = _n_quickselect(temp, n, mid)
            return (v1 + v2) * 0.5
        return _n_quickselect(temp, n, mid)
    finally:
        os_aligned_free(temp)

cdef double _n_quantile(numeric_t* ptr, Py_ssize_t n, double q) except *:
    """分位数（线性插值）。n == 0 时返回 NAN。"""
    if n == 0:
        return NAN
    cdef numeric_t* temp = <numeric_t*>aligned_malloc(n * sizeof(numeric_t))
    if temp == NULL:
        raise MemoryError()
    memcpy(temp, ptr, n * sizeof(numeric_t))
    cdef double pos = q * (n - 1)
    cdef Py_ssize_t idx = <Py_ssize_t>pos
    cdef double frac = pos - idx
    cdef double v1, v2, result
    try:
        if idx >= n - 1:
            return _n_quickselect(temp, n, n - 1)
        v1 = _n_quickselect(temp, n, idx)
        v2 = _n_quickselect(temp, n, idx + 1)
        return v1 + frac * (v2 - v1)
    finally:
        os_aligned_free(temp)

cdef inline void _n_central_moments(numeric_t* ptr, Py_ssize_t n,
                                    double* out_mean, double* out_m2,
                                    double* out_m3, double* out_m4) noexcept nogil:
    """Welford 在线递推中心矩（与 IntColumn/DoubleColumn 原实现逐位一致）。"""
    if n == 0:
        out_mean[0] = 0.0; out_m2[0] = 0.0; out_m3[0] = 0.0; out_m4[0] = 0.0
        return
    cdef double mean = 0.0, m2 = 0.0, m3 = 0.0, m4 = 0.0
    cdef double val, delta, delta_n, delta_n2, term1
    cdef Py_ssize_t i, n1
    for i in range(n):
        n1 = i
        val = <double>ptr[i]
        delta = val - mean
        mean += delta / (n1 + 1)
        delta_n = val - mean
        term1 = delta * delta_n * n1
        delta_n2 = delta_n * delta_n
        m4 += term1 * delta_n2 * ((n1 + 1) * (n1 + 1) - 3 * (n1 + 1) + 3) + 6.0 * delta_n2 * m2 - 4.0 * delta_n * m3
        m3 += term1 * delta_n * (n1 - 2) - 3.0 * delta_n * m2
        m2 += term1
    out_mean[0] = mean
    out_m2[0] = fmax(m2, 0.0)
    out_m3[0] = m3
    out_m4[0] = fmax(m4, 0.0)

cdef inline double _n_accum_var(numeric_t* ptr, Py_ssize_t n, double m) noexcept nogil:
    """样本方差：sum((x - m)^2) / (n - 1)，n < 2 时返回 0。"""
    if n < 2:
        return 0.0
    cdef Py_ssize_t i
    cdef double sum_sq = 0.0, diff
    for i in range(n):
        diff = <double>ptr[i] - m
        sum_sq += diff * diff
    return sum_sq / <double>(n - 1)

cdef inline double _n_accum_mad(numeric_t* ptr, Py_ssize_t n, double m) noexcept nogil:
    """平均绝对离差：mean(|x - m|)，n == 0 时返回 NAN。"""
    if n == 0:
        return NAN
    cdef Py_ssize_t i
    cdef double total_dev = 0.0
    for i in range(n):
        total_dev += fabs(<double>ptr[i] - m)
    return total_dev / <double>n

cdef inline Py_ssize_t _resolve_min_periods(Py_ssize_t window, object min_periods):
    """解析滚动窗口最小观测数：None→window；<=0→1；>window→ValueError。"""
    cdef Py_ssize_t mp = window
    if min_periods is not None:
        if isinstance(<object>min_periods, int):
            mp = <Py_ssize_t>min_periods
        else:
            mp = <Py_ssize_t>PyNumber_Index(min_periods)
    if mp < 1:
        mp = 1
    if mp > window:
        raise ValueError("min_periods must be <= window, got %d" % mp)
    return mp

cdef inline void _roll_win_bounds(Py_ssize_t i, Py_ssize_t r, Py_ssize_t window,
                                  bint center, Py_ssize_t* lo, Py_ssize_t* hi) noexcept nogil:
    """窗口边界：右对齐 [max(0,i-w+1), i]；居中 [(i-(w-1)//2) 截断, (i+w//2) 截断]。"""
    if center:
        lo[0] = i - (window - 1) // 2
        if lo[0] < 0: lo[0] = 0
        hi[0] = i + window // 2
        if hi[0] >= r: hi[0] = r - 1
    else:
        lo[0] = i - window + 1
        if lo[0] < 0: lo[0] = 0
        hi[0] = i

cdef inline void _n_roll_sum(numeric_t* src, double* dst, Py_ssize_t c,
                             Py_ssize_t r, Py_ssize_t window,
                             Py_ssize_t min_periods, bint center) noexcept nogil:
    """滚动求和（O(N) 滑窗），输出 double。c/r 为列数/行数（列优先布局）。
    min_periods：至少需要多少个观测才输出；center：居中窗口。"""
    cdef Py_ssize_t col_idx, offset, i, lo, hi, n
    cdef double s
    with nogil:
        if not center:
            for col_idx in range(c):
                offset = col_idx * r
                s = 0.0
                for i in range(r):
                    s += <double>src[offset + i]
                    if i >= window:
                        s -= <double>src[offset + i - window]
                    n = window if i >= window - 1 else i + 1
                    if n >= min_periods:
                        dst[offset + i] = s
        else:
            for col_idx in range(c):
                offset = col_idx * r
                lo = 0
                hi = window // 2
                if hi >= r: hi = r - 1
                s = 0.0
                for i in range(hi + 1):
                    s += <double>src[offset + i]
                for i in range(r):
                    n = hi - lo + 1
                    if n >= min_periods:
                        dst[offset + i] = s
                    if i + window // 2 + 1 <= r - 1:
                        hi += 1
                        s += <double>src[offset + hi]
                    if i + 1 > (window - 1) // 2:
                        s -= <double>src[offset + lo]
                        lo += 1

cdef inline void _n_roll_mean(numeric_t* src, double* dst, Py_ssize_t c,
                              Py_ssize_t r, Py_ssize_t window,
                              Py_ssize_t min_periods, bint center) noexcept nogil:
    """滚动均值（O(N) 滑窗，部分窗口除以实际观测数），输出 double。"""
    cdef Py_ssize_t col_idx, offset, i, lo, hi, n
    cdef double s
    with nogil:
        if not center:
            for col_idx in range(c):
                offset = col_idx * r
                s = 0.0
                for i in range(r):
                    s += <double>src[offset + i]
                    if i >= window:
                        s -= <double>src[offset + i - window]
                    n = window if i >= window - 1 else i + 1
                    if n >= min_periods:
                        dst[offset + i] = s / <double>n
        else:
            for col_idx in range(c):
                offset = col_idx * r
                lo = 0
                hi = window // 2
                if hi >= r: hi = r - 1
                s = 0.0
                for i in range(hi + 1):
                    s += <double>src[offset + i]
                for i in range(r):
                    n = hi - lo + 1
                    if n >= min_periods:
                        dst[offset + i] = s / <double>n
                    if i + window // 2 + 1 <= r - 1:
                        hi += 1
                        s += <double>src[offset + hi]
                    if i + 1 > (window - 1) // 2:
                        s -= <double>src[offset + lo]
                        lo += 1

cdef inline void _n_roll_var(numeric_t* src, double* dst, Py_ssize_t c,
                             Py_ssize_t r, Py_ssize_t window,
                             Py_ssize_t min_periods, bint center,
                             bint ddof) noexcept nogil:
    """滚动方差（O(N) 稳定滑窗），输出 double。计数不足时输出 NAN。"""
    cdef Py_ssize_t col_idx, offset, i, lo, hi, n
    cdef double s, s2, val, var_val, denom
    with nogil:
        if not center:
            for col_idx in range(c):
                offset = col_idx * r
                s = 0.0; s2 = 0.0
                for i in range(r):
                    val = <double>src[offset + i]
                    s += val; s2 += val * val
                    if i >= window:
                        val = <double>src[offset + i - window]
                        s -= val; s2 -= val * val
                    n = window if i >= window - 1 else i + 1
                    if n >= min_periods:
                        var_val = s2 - (s * s) / <double>n
                        if var_val < 0.0: var_val = 0.0
                        denom = <double>(n - ddof)
                        dst[offset + i] = var_val / denom if denom > 0.0 else NAN
        else:
            for col_idx in range(c):
                offset = col_idx * r
                lo = 0
                hi = window // 2
                if hi >= r: hi = r - 1
                s = 0.0; s2 = 0.0
                for i in range(hi + 1):
                    val = <double>src[offset + i]
                    s += val; s2 += val * val
                for i in range(r):
                    n = hi - lo + 1
                    if n >= min_periods:
                        var_val = s2 - (s * s) / <double>n
                        if var_val < 0.0: var_val = 0.0
                        denom = <double>(n - ddof)
                        dst[offset + i] = var_val / denom if denom > 0.0 else NAN
                    if i + window // 2 + 1 <= r - 1:
                        hi += 1
                        val = <double>src[offset + hi]
                        s += val; s2 += val * val
                    if i + 1 > (window - 1) // 2:
                        val = <double>src[offset + lo]
                        s -= val; s2 -= val * val
                        lo += 1

cdef inline void _n_roll_skew(numeric_t* src, double* dst, Py_ssize_t c,
                              Py_ssize_t r, Py_ssize_t window,
                              Py_ssize_t min_periods, bint center) noexcept nogil:
    """滚动偏度（直接窗口重算，O(N*W)），输出 double；观测数 < 3 或不足时 NAN。"""
    cdef Py_ssize_t col_idx, offset, i, j, lo, hi, n
    cdef double s, mean, m2, m3, val, std_dev
    with nogil:
        for col_idx in range(c):
            offset = col_idx * r
            for i in range(r):
                _roll_win_bounds(i, r, window, center, &lo, &hi)
                n = hi - lo + 1
                if n < min_periods or n < 3:
                    continue
                s = 0.0
                for j in range(lo, hi + 1):
                    s += <double>src[offset + j]
                mean = s / <double>n
                m2 = 0.0; m3 = 0.0
                for j in range(lo, hi + 1):
                    val = <double>src[offset + j] - mean
                    m2 += val * val
                    m3 += val * val * val
                if m2 > 1e-14:
                    std_dev = sqrt(m2 / <double>n)
                    dst[offset + i] = (m3 / <double>n) / (std_dev * std_dev * std_dev)
                else:
                    dst[offset + i] = 0.0

cdef inline void _n_roll_kurt(numeric_t* src, double* dst, Py_ssize_t c,
                              Py_ssize_t r, Py_ssize_t window,
                              Py_ssize_t min_periods, bint center) noexcept nogil:
    """滚动峰度（直接窗口重算，O(N*W)），输出 double；观测数 < 4 或不足时 NAN。"""
    cdef Py_ssize_t col_idx, offset, i, j, lo, hi, n
    cdef double s, mean, m2, m4, val, variance
    with nogil:
        for col_idx in range(c):
            offset = col_idx * r
            for i in range(r):
                _roll_win_bounds(i, r, window, center, &lo, &hi)
                n = hi - lo + 1
                if n < min_periods or n < 4:
                    continue
                s = 0.0
                for j in range(lo, hi + 1):
                    s += <double>src[offset + j]
                mean = s / <double>n
                m2 = 0.0; m4 = 0.0
                for j in range(lo, hi + 1):
                    val = <double>src[offset + j] - mean
                    m2 += val * val
                    m4 += val * val * val * val
                if m2 > 1e-14:
                    variance = m2 / <double>n
                    dst[offset + i] = (m4 / <double>n) / (variance * variance) - 3.0
                else:
                    dst[offset + i] = 0.0

cdef inline void _n_roll_quantile(numeric_t* src, double* dst, Py_ssize_t c,
                                  Py_ssize_t r, Py_ssize_t window,
                                  Py_ssize_t min_periods, bint center,
                                  double q) except *:
    """滚动分位数（线性插值，每窗口 qsort），输出 double。"""
    cdef numeric_t* buf = <numeric_t*>aligned_malloc(window * sizeof(numeric_t))
    if buf == NULL:
        raise MemoryError()
    cdef int (*comp)(const void*, const void*) noexcept nogil
    if numeric_t is int:
        comp = _cmp_int
    else:
        comp = _cmp_double_asc
    cdef Py_ssize_t col_idx, offset, i, j, lo, hi, n, i_low
    cdef double pos, frac
    try:
        with nogil:
            for col_idx in range(c):
                offset = col_idx * r
                for i in range(r):
                    _roll_win_bounds(i, r, window, center, &lo, &hi)
                    n = hi - lo + 1
                    if n < min_periods:
                        continue
                    memcpy(buf, src + offset + lo, n * sizeof(numeric_t))
                    qsort(buf, n, sizeof(numeric_t), comp)
                    pos = q * <double>(n - 1)
                    i_low = <Py_ssize_t>pos
                    frac = pos - <double>i_low
                    if i_low >= n - 1:
                        dst[offset + i] = <double>buf[n - 1]
                    else:
                        dst[offset + i] = <double>buf[i_low] + frac * (<double>buf[i_low + 1] - <double>buf[i_low])
    finally:
        os_aligned_free(buf)

cdef inline void _n_roll_min(numeric_t* src, numeric_t* dst, Py_ssize_t c,
                             Py_ssize_t r, Py_ssize_t window,
                             Py_ssize_t min_periods, bint center) noexcept nogil:
    """滚动最小值。int 列无效位填 INT_MAX 哨兵，double 列填 NAN（保持列契约）。
    O(N) 单调队列（右对齐）/ O(N*W) 逐窗重扫（居中，因需预见未来元素）。"""
    cdef Py_ssize_t col_idx, offset, i, j, lo, hi, n, head, tail
    cdef numeric_t fill, best
    if numeric_t is int:
        fill = 2147483647
    else:
        fill = NAN
    cdef Py_ssize_t* q = <Py_ssize_t*>aligned_malloc(r * sizeof(Py_ssize_t))
    if q == NULL: return
    try:
        with nogil:
            if not center:
                for col_idx in range(c):
                    offset = col_idx * r
                    head = 0; tail = 0
                    for i in range(r):
                        lo = i - window + 1
                        if lo < 0: lo = 0
                        while head < tail and q[head] < lo:
                            head += 1
                        while head < tail and src[offset + q[tail - 1]] >= src[offset + i]:
                            tail -= 1
                        q[tail] = i; tail += 1
                        n = i - lo + 1
                        if n >= min_periods:
                            dst[offset + i] = src[offset + q[head]]
                        else:
                            dst[offset + i] = fill
            else:
                for col_idx in range(c):
                    offset = col_idx * r
                    for i in range(r):
                        _roll_win_bounds(i, r, window, 1, &lo, &hi)
                        n = hi - lo + 1
                        if n >= min_periods:
                            best = src[offset + lo]
                            for j in range(lo + 1, hi + 1):
                                if src[offset + j] < best:
                                    best = src[offset + j]
                            dst[offset + i] = best
                        else:
                            dst[offset + i] = fill
    finally:
        os_aligned_free(q)

cdef inline void _n_roll_max(numeric_t* src, numeric_t* dst, Py_ssize_t c,
                             Py_ssize_t r, Py_ssize_t window,
                             Py_ssize_t min_periods, bint center) noexcept nogil:
    """滚动最大值。int 列无效位填 INT_MIN 哨兵，double 列填 NAN。"""
    cdef Py_ssize_t col_idx, offset, i, j, lo, hi, n, head, tail
    cdef numeric_t fill, best
    if numeric_t is int:
        fill = -2147483647 - 1
    else:
        fill = NAN
    cdef Py_ssize_t* q = <Py_ssize_t*>aligned_malloc(r * sizeof(Py_ssize_t))
    if q == NULL: return
    try:
        with nogil:
            if not center:
                for col_idx in range(c):
                    offset = col_idx * r
                    head = 0; tail = 0
                    for i in range(r):
                        lo = i - window + 1
                        if lo < 0: lo = 0
                        while head < tail and q[head] < lo:
                            head += 1
                        while head < tail and src[offset + q[tail - 1]] <= src[offset + i]:
                            tail -= 1
                        q[tail] = i; tail += 1
                        n = i - lo + 1
                        if n >= min_periods:
                            dst[offset + i] = src[offset + q[head]]
                        else:
                            dst[offset + i] = fill
            else:
                for col_idx in range(c):
                    offset = col_idx * r
                    for i in range(r):
                        _roll_win_bounds(i, r, window, 1, &lo, &hi)
                        n = hi - lo + 1
                        if n >= min_periods:
                            best = src[offset + lo]
                            for j in range(lo + 1, hi + 1):
                                if src[offset + j] > best:
                                    best = src[offset + j]
                            dst[offset + i] = best
                        else:
                            dst[offset + i] = fill
    finally:
        os_aligned_free(q)

cdef inline void _copy_int_to_double(int* src, double* dst, Py_ssize_t n) noexcept nogil:
    """int 列裸数据按元素转 double（含哨兵原样转换，语义同旧内联 <double> 强转）。"""
    cdef Py_ssize_t i
    for i in range(n):
        dst[i] = <double>src[i]

cdef inline DoubleColumn _int_col_to_double(IntColumn src_col):
    """把整列 int 数据复制成临时 DoubleColumn（供 Int×Double 双列滚动用）。"""
    cdef Py_ssize_t n = src_col.length
    cdef DoubleColumn res = DoubleColumn(max(n, 1))
    res.length = n
    _copy_int_to_double(src_col.data_c, res.data_c, n)
    return res

cdef inline void _n_roll_corr(numeric_t* x, numeric_t* y, double* dst, Py_ssize_t c,
                              Py_ssize_t r, Py_ssize_t window,
                              Py_ssize_t min_periods, bint center) noexcept nogil:
    """滚动两列相关（O(N) 稳定滑窗），输出 double；计数不足时 NAN。"""
    cdef Py_ssize_t col_idx, offset, i, lo, hi, n
    cdef double sx, sy, sxx, syy, sxy, xv, yv, vx, vy, denom, cov
    with nogil:
        if not center:
            for col_idx in range(c):
                offset = col_idx * r
                sx = 0.0; sy = 0.0; sxx = 0.0; syy = 0.0; sxy = 0.0
                for i in range(r):
                    xv = <double>x[offset + i]; yv = <double>y[offset + i]
                    sx += xv; sy += yv; sxx += xv * xv; syy += yv * yv; sxy += xv * yv
                    if i >= window:
                        xv = <double>x[offset + i - window]; yv = <double>y[offset + i - window]
                        sx -= xv; sy -= yv; sxx -= xv * xv; syy -= yv * yv; sxy -= xv * yv
                    n = window if i >= window - 1 else i + 1
                    if n >= min_periods:
                        vx = fmax(sxx - sx * sx / <double>n, 0.0)
                        vy = fmax(syy - sy * sy / <double>n, 0.0)
                        denom = sqrt(vx * vy)
                        cov = sxy - sx * sy / <double>n
                        dst[offset + i] = cov / denom if denom > 0.0 else 0.0
        else:
            for col_idx in range(c):
                offset = col_idx * r
                lo = 0
                hi = window // 2
                if hi >= r: hi = r - 1
                sx = 0.0; sy = 0.0; sxx = 0.0; syy = 0.0; sxy = 0.0
                for i in range(hi + 1):
                    xv = <double>x[offset + i]; yv = <double>y[offset + i]
                    sx += xv; sy += yv; sxx += xv * xv; syy += yv * yv; sxy += xv * yv
                for i in range(r):
                    n = hi - lo + 1
                    if n >= min_periods:
                        vx = fmax(sxx - sx * sx / <double>n, 0.0)
                        vy = fmax(syy - sy * sy / <double>n, 0.0)
                        denom = sqrt(vx * vy)
                        cov = sxy - sx * sy / <double>n
                        dst[offset + i] = cov / denom if denom > 0.0 else 0.0
                    if i + window // 2 + 1 <= r - 1:
                        hi += 1
                        xv = <double>x[offset + hi]; yv = <double>y[offset + hi]
                        sx += xv; sy += yv; sxx += xv * xv; syy += yv * yv; sxy += xv * yv
                    if i + 1 > (window - 1) // 2:
                        xv = <double>x[offset + lo]; yv = <double>y[offset + lo]
                        sx -= xv; sy -= yv; sxx -= xv * xv; syy -= yv * yv; sxy -= xv * yv
                        lo += 1

cdef inline void _n_roll_cov(numeric_t* x, numeric_t* y, double* dst, Py_ssize_t c,
                             Py_ssize_t r, Py_ssize_t window,
                             Py_ssize_t min_periods, bint center) noexcept nogil:
    """滚动两列协方差（总体，除以窗口观测数），输出 double；计数不足时 NAN。"""
    cdef Py_ssize_t col_idx, offset, i, lo, hi, n
    cdef double sx, sy, sxy, xv, yv
    with nogil:
        if not center:
            for col_idx in range(c):
                offset = col_idx * r
                sx = 0.0; sy = 0.0; sxy = 0.0
                for i in range(r):
                    xv = <double>x[offset + i]; yv = <double>y[offset + i]
                    sx += xv; sy += yv; sxy += xv * yv
                    if i >= window:
                        xv = <double>x[offset + i - window]; yv = <double>y[offset + i - window]
                        sx -= xv; sy -= yv; sxy -= xv * yv
                    n = window if i >= window - 1 else i + 1
                    if n >= min_periods:
                        dst[offset + i] = (sxy - sx * sy / <double>n) / <double>n
        else:
            for col_idx in range(c):
                offset = col_idx * r
                lo = 0
                hi = window // 2
                if hi >= r: hi = r - 1
                sx = 0.0; sy = 0.0; sxy = 0.0
                for i in range(hi + 1):
                    xv = <double>x[offset + i]; yv = <double>y[offset + i]
                    sx += xv; sy += yv; sxy += xv * yv
                for i in range(r):
                    n = hi - lo + 1
                    if n >= min_periods:
                        dst[offset + i] = (sxy - sx * sy / <double>n) / <double>n
                    if i + window // 2 + 1 <= r - 1:
                        hi += 1
                        xv = <double>x[offset + hi]; yv = <double>y[offset + hi]
                        sx += xv; sy += yv; sxy += xv * yv
                    if i + 1 > (window - 1) // 2:
                        xv = <double>x[offset + lo]; yv = <double>y[offset + lo]
                        sx -= xv; sy -= yv; sxy -= xv * yv
                        lo += 1

# ============================================================
#  ewm（指数加权移动）内核：逐列独立，pandas ewm/ewmcov 算法移植
#  alpha 参数由调用侧解析（com/span/halflife/alpha → alpha）。
#  min_periods 默认 0：第一条即输出。无 times 支持（等间隔假设）。
# ============================================================
cdef inline void _n_ewm(numeric_t* src, double* dst, Py_ssize_t c, Py_ssize_t r,
                        double alpha, double com, Py_ssize_t minp,
                        bint adjust, bint ignore_na, bint normalize) noexcept nogil:
    """指数加权移动均值/求和（pandas ewm 递推）。输出 double，计数不足 NAN。
    normalize=True 为 mean；normalize=False 为 sum（需 adjust=True，调用侧校验）。"""
    if r == 0:
        return
    cdef Py_ssize_t col_idx, offset, i
    cdef double old_wt_factor = 1.0 - alpha
    cdef double new_wt = 1.0 if adjust else alpha
    cdef double weighted, old_wt, cur
    cdef Py_ssize_t nobs
    cdef bint is_observation
    with nogil:
        for col_idx in range(c):
            offset = col_idx * r
            weighted = <double>src[offset]
            is_observation = not isnan(weighted)
            nobs = is_observation
            dst[offset] = weighted if nobs >= minp else NAN
            old_wt = 1.0
            for i in range(1, r):
                cur = <double>src[offset + i]
                is_observation = not isnan(cur)
                nobs += is_observation
                if not isnan(weighted):
                    if is_observation or not ignore_na:
                        if normalize:
                            old_wt *= old_wt_factor
                        else:
                            weighted = old_wt_factor * weighted
                        if is_observation:
                            if normalize:
                                if weighted != cur:
                                    if not adjust and com == 1:
                                        new_wt = 1.0 - old_wt
                                    weighted = old_wt * weighted + new_wt * cur
                                    weighted /= (old_wt + new_wt)
                                if adjust:
                                    old_wt += new_wt
                                else:
                                    old_wt = 1.0
                            else:
                                weighted += cur
                elif is_observation:
                    weighted = cur
                dst[offset + i] = weighted if nobs >= minp else NAN

cdef inline void _n_ewmcov(numeric_t* x, numeric_t* y, double* dst, Py_ssize_t c,
                           Py_ssize_t r, double alpha, Py_ssize_t minp,
                           bint adjust, bint ignore_na, bint bias) noexcept nogil:
    """指数加权协方差/方差（pandas ewmcov 递推，Welford + 权重归一因子）。
    var=ewmcov(x,x)；corr=ewmcov(x,y)/sqrt(ewmcov(x,x)*ewmcov(y,y))（bias=True）。"""
    if r == 0:
        return
    cdef Py_ssize_t col_idx, offset, i
    cdef double old_wt_factor = 1.0 - alpha
    cdef double new_wt = 1.0 if adjust else alpha
    cdef double mean_x, mean_y, cov, sum_wt, sum_wt2, old_wt
    cdef double cur_x, cur_y, old_mean_x, old_mean_y
    cdef double numerator, denominator
    cdef Py_ssize_t nobs
    cdef bint is_observation
    with nogil:
        for col_idx in range(c):
            offset = col_idx * r
            mean_x = <double>x[offset]
            mean_y = <double>y[offset]
            is_observation = (not isnan(mean_x)) and (not isnan(mean_y))
            nobs = is_observation
            if not is_observation:
                mean_x = NAN
                mean_y = NAN
            dst[offset] = (0.0 if bias else NAN) if nobs >= minp else NAN
            cov = 0.0
            sum_wt = 1.0
            sum_wt2 = 1.0
            old_wt = 1.0
            for i in range(1, r):
                cur_x = <double>x[offset + i]
                cur_y = <double>y[offset + i]
                is_observation = (not isnan(cur_x)) and (not isnan(cur_y))
                nobs += is_observation
                if not isnan(mean_x):
                    if is_observation or not ignore_na:
                        sum_wt *= old_wt_factor
                        sum_wt2 *= (old_wt_factor * old_wt_factor)
                        old_wt *= old_wt_factor
                        if is_observation:
                            old_mean_x = mean_x
                            old_mean_y = mean_y
                            if mean_x != cur_x:
                                mean_x = (old_wt * old_mean_x + new_wt * cur_x) / (old_wt + new_wt)
                            if mean_y != cur_y:
                                mean_y = (old_wt * old_mean_y + new_wt * cur_y) / (old_wt + new_wt)
                            cov = (old_wt * (cov + (old_mean_x - mean_x) * (old_mean_y - mean_y)) +
                                   new_wt * (cur_x - mean_x) * (cur_y - mean_y)) / (old_wt + new_wt)
                            sum_wt += new_wt
                            sum_wt2 += (new_wt * new_wt)
                            old_wt += new_wt
                            if not adjust:
                                sum_wt /= old_wt
                                sum_wt2 /= (old_wt * old_wt)
                                old_wt = 1.0
                elif is_observation:
                    mean_x = cur_x
                    mean_y = cur_y
                if nobs >= minp:
                    if not bias:
                        numerator = sum_wt * sum_wt
                        denominator = numerator - sum_wt2
                        dst[offset + i] = (numerator / denominator) * cov if denominator > 0.0 else NAN
                    else:
                        dst[offset + i] = cov
                else:
                    dst[offset + i] = NAN

# ============================================================
#  win_type（boxcar/triang）加权滚动内核：仅 mean/sum/var/std
#  mean/sum：窗口右对齐，最新值吃最后一个权重（pandas roll_weighted_sum_mean）；
#  var/std：pandas West 循环对齐（绝对位置 p 权重 wbuf[p % win_n]）。
#  计数只统计非 NaN 值；minp 默认窗口（调用侧已 clamp max(minp,1)）。
# ============================================================
cdef inline void _n_roll_wmean_sum(numeric_t* src, double* dst, Py_ssize_t c, Py_ssize_t r,
                                   double* wbuf, Py_ssize_t win_n, Py_ssize_t minp,
                                   bint center, bint avg) noexcept nogil:
    """加权滚动均值/求和（O(N·W) 逐窗直算）。avg=1 均值（除以非 NaN 权重和），avg=0 求和。
    center：pandas 尾部补 offset 个 NaN 右对齐再截断的等价写法。"""
    cdef Py_ssize_t col_idx, offset, j, k, left, lo, hi, p, nobs
    cdef double s, sw, w, v
    cdef Py_ssize_t pad = (win_n // 2) if center else 0
    with nogil:
        for col_idx in range(c):
            offset = col_idx * r
            for j in range(r):
                k = j + pad
                left = k - win_n + 1
                lo = left if left > 0 else 0
                hi = k
                if hi >= r: hi = r - 1
                s = 0.0; sw = 0.0; nobs = 0
                for p in range(lo, hi + 1):
                    v = <double>src[offset + p]
                    if v == v:
                        w = wbuf[p - left]
                        s += w * v
                        sw += w
                        nobs += 1
                if nobs < minp or sw == 0.0:
                    dst[offset + j] = NAN
                else:
                    dst[offset + j] = s if not avg else s / sw

cdef inline void _n_roll_wvar(numeric_t* src, double* dst, Py_ssize_t c, Py_ssize_t r,
                              double* wbuf, Py_ssize_t win_n, Py_ssize_t minp,
                              bint center, Py_ssize_t ddof) noexcept nogil:
    """加权滚动方差（pandas roll_weighted_var：绝对位置循环权重）。
    结果 = t*win_n/((win_n-ddof)*sum_w)；nobs<=ddof 或 nobs<minp → NAN。"""
    cdef Py_ssize_t col_idx, offset, j, k, lo, hi, p, nobs, pass2
    cdef double sw, swx, mean_w, t, w, v
    cdef Py_ssize_t pad = (win_n // 2) if center else 0
    with nogil:
        for col_idx in range(c):
            offset = col_idx * r
            for j in range(r):
                k = j + pad
                lo = k - win_n + 1
                if lo < 0: lo = 0
                hi = k
                if hi >= r: hi = r - 1
                sw = 0.0; swx = 0.0; nobs = 0
                for p in range(lo, hi + 1):
                    v = <double>src[offset + p]
                    if v == v:
                        w = wbuf[p % win_n]
                        sw += w
                        swx += w * v
                        nobs += 1
                if nobs < minp or nobs <= ddof:
                    dst[offset + j] = NAN
                    continue
                mean_w = swx / sw
                t = 0.0
                for p in range(lo, hi + 1):
                    v = <double>src[offset + p]
                    if v == v:
                        w = wbuf[p % win_n]
                        t += w * (v - mean_w) * (v - mean_w)
                dst[offset + j] = t * <double>win_n / (<double>(win_n - ddof) * sw)

cdef inline void _n_cumsum(numeric_t* src, numeric_t* dst, Py_ssize_t c,
                           Py_ssize_t r) noexcept nogil:
    """逐列独立累计求和（列优先布局）。"""
    cdef numeric_t total
    cdef Py_ssize_t col_idx, offset, i
    with nogil:
        for col_idx in range(c):
            offset = col_idx * r
            if numeric_t is int:
                total = 0
            else:
                total = 0.0
            for i in range(r):
                total += src[offset + i]
                dst[offset + i] = total

cdef inline void _n_cumprod(numeric_t* src, numeric_t* dst, Py_ssize_t c,
                            Py_ssize_t r) noexcept nogil:
    """逐列独立累计乘积（double 累加器，写出时强转回目标类型）。"""
    cdef double total
    cdef Py_ssize_t col_idx, offset, i
    with nogil:
        for col_idx in range(c):
            offset = col_idx * r
            total = 1.0
            for i in range(r):
                total *= <double>src[offset + i]
                dst[offset + i] = <numeric_t>total

cdef inline void _n_cummax(numeric_t* src, numeric_t* dst, Py_ssize_t c,
                           Py_ssize_t r) noexcept nogil:
    """逐列独立累计最大值（列优先布局，r==0 时跳过）。"""
    cdef numeric_t cur_max
    cdef Py_ssize_t col_idx, offset, i
    with nogil:
        for col_idx in range(c):
            offset = col_idx * r
            if r > 0:
                cur_max = src[offset]
                dst[offset] = cur_max
                for i in range(1, r):
                    if src[offset + i] > cur_max:
                        cur_max = src[offset + i]
                    dst[offset + i] = cur_max

cdef inline void _n_cummin(numeric_t* src, numeric_t* dst, Py_ssize_t c,
                           Py_ssize_t r) noexcept nogil:
    """逐列独立累计最小值（列优先布局，r==0 时跳过）。"""
    cdef numeric_t cur_min
    cdef Py_ssize_t col_idx, offset, i
    with nogil:
        for col_idx in range(c):
            offset = col_idx * r
            if r > 0:
                cur_min = src[offset]
                dst[offset] = cur_min
                for i in range(1, r):
                    if src[offset + i] < cur_min:
                        cur_min = src[offset + i]
                    dst[offset + i] = cur_min

cdef inline void _n_diff(numeric_t* src, numeric_t* dst, Py_ssize_t c,
                         Py_ssize_t r, int periods) noexcept nogil:
    """逐列差分：periods>=0 时后值减前值，periods<0 时前值减后值。
    非差分位填 0（int）或 NAN（double），与历史语义一致。"""
    cdef numeric_t fill
    cdef Py_ssize_t col_idx, offset, i
    cdef int p = periods if periods >= 0 else -periods
    with nogil:
        if numeric_t is int:
            fill = 0
        else:
            fill = NAN
        if periods >= 0:
            for col_idx in range(c):
                offset = col_idx * r
                for i in range(min(p, r)):
                    dst[offset + i] = fill
                for i in range(p, r):
                    dst[offset + i] = src[offset + i] - src[offset + i - p]
        else:
            for col_idx in range(c):
                offset = col_idx * r
                for i in range(max(0, r - p), r):
                    dst[offset + i] = fill
                for i in range(r - p):
                    dst[offset + i] = src[offset + i] - src[offset + i + p]

cdef inline void _n_clip(numeric_t* ptr, Py_ssize_t n,
                         numeric_t lower, numeric_t upper) noexcept nogil:
    """就地裁剪到 [lower, upper]，覆盖 NaN 即原样保留的双精度语义。"""
    cdef numeric_t v
    cdef Py_ssize_t i
    with nogil:
        for i in range(n):
            v = ptr[i]
            if v < lower:
                v = lower
            elif v > upper:
                v = upper
            ptr[i] = v

cdef inline void _n_add_val(numeric_t* src, numeric_t* dst, Py_ssize_t n,
                            numeric_t val) noexcept nogil:
    cdef Py_ssize_t i
    with nogil:
        for i in range(n):
            dst[i] = src[i] + val

cdef inline void _n_sub_val(numeric_t* src, numeric_t* dst, Py_ssize_t n,
                            numeric_t val) noexcept nogil:
    cdef Py_ssize_t i
    with nogil:
        for i in range(n):
            dst[i] = src[i] - val

cdef inline void _n_mul_val(numeric_t* src, numeric_t* dst, Py_ssize_t n,
                            numeric_t val) noexcept nogil:
    cdef Py_ssize_t i
    with nogil:
        for i in range(n):
            dst[i] = src[i] * val

cdef inline void _n_div_val(numeric_t* src, double* dst, Py_ssize_t n,
                            double val) noexcept nogil:
    """除法（乘以倒数），输出 double；val==0 时填 NAN。"""
    cdef Py_ssize_t i
    cdef double inv_val
    with nogil:
        if val != 0.0:
            inv_val = 1.0 / val
            for i in range(n):
                dst[i] = <double>src[i] * inv_val
        else:
            for i in range(n):
                dst[i] = NAN

cdef int _cmp_int32(const void* a, const void* b) noexcept nogil:
    cdef int32_t va = (<const int32_t*>a)[0]
    cdef int32_t vb = (<const int32_t*>b)[0]
    return (va > vb) - (va < vb)


# ============================================================
#  DateColumn / DateTimeColumn / TimeColumn 共享的固定宽度整数核
#  (fused over int32_t/int64_t)：三列只保留「2D 簿记 + 值域解析」外壳，
#  纯元素循环统一委托到这里，语义与旧的逐列实现逐字节等价。
# ============================================================
ctypedef fused DateLike:
    int32_t
    int64_t

cdef inline DateLike _n_max(DateLike* ptr, Py_ssize_t n) noexcept nogil:
    """返回 ptr[0..n) 的最大值（调用方保证 n > 0）。"""
    cdef Py_ssize_t i
    cdef DateLike m = ptr[0]
    for i in range(1, n):
        if ptr[i] > m:
            m = ptr[i]
    return m

cdef inline DateLike _n_min(DateLike* ptr, Py_ssize_t n) noexcept nogil:
    """返回 ptr[0..n) 的最小值（调用方保证 n > 0）。"""
    cdef Py_ssize_t i
    cdef DateLike m = ptr[0]
    for i in range(1, n):
        if ptr[i] < m:
            m = ptr[i]
    return m

cdef inline void _n_where_mask(DateLike* dst, DateLike* src, int* mask,
                               DateLike fill, Py_ssize_t n) noexcept nogil:
    """无分支向量化掩码选择：mask != 0 取 src，否则取 fill。"""
    cdef Py_ssize_t i
    for i in range(n):
        dst[i] = src[i] if mask[i] != 0 else fill

cdef inline void _n_cmp_scalar(DateLike* s, int* r, DateLike v,
                               Py_ssize_t n, int op) noexcept nogil:
    """元素级与标量比较，把布尔结果写入 0/1 的 int 掩码。"""
    cdef Py_ssize_t i
    if op == 0:
        for i in range(n): r[i] = s[i] < v
    elif op == 1:
        for i in range(n): r[i] = s[i] <= v
    elif op == 2:
        for i in range(n): r[i] = s[i] > v
    elif op == 3:
        for i in range(n): r[i] = s[i] >= v
    elif op == 4:
        for i in range(n): r[i] = s[i] == v
    else:
        for i in range(n): r[i] = s[i] != v

cdef inline void _n_cmp_col(DateLike* s, DateLike* o, int* r,
                            Py_ssize_t n, int op) noexcept nogil:
    """元素级同类型列比较，把布尔结果写入 0/1 的 int 掩码。"""
    cdef Py_ssize_t i
    if op == 0:
        for i in range(n): r[i] = s[i] < o[i]
    elif op == 1:
        for i in range(n): r[i] = s[i] <= o[i]
    elif op == 2:
        for i in range(n): r[i] = s[i] > o[i]
    elif op == 3:
        for i in range(n): r[i] = s[i] >= o[i]
    elif op == 4:
        for i in range(n): r[i] = s[i] == o[i]
    else:
        for i in range(n): r[i] = s[i] != o[i]

cdef list _n_unique(DateLike* ptr, Py_ssize_t n, set seen, list result) except *:
    """按存储值域去重（DateTime/Time 使用）；Date 因返回 YYYYMMDD 保留自身实现。"""
    cdef Py_ssize_t i
    cdef DateLike val
    for i in range(n):
        val = ptr[i]
        if val not in seen:
            seen.add(val)
            result.append(val)
    return result

cdef inline void _n_isin(DateLike* src, DateLike* keys, Py_ssize_t n,
                         Py_ssize_t m, unsigned char* out) noexcept nogil:
    """齐集检索：就地 qsort keys 后逐元素二分查找，结果写入 0/1 掩码。"""
    cdef Py_ssize_t i
    cdef int (*comp)(const void*, const void*) noexcept nogil
    if DateLike is int32_t:
        comp = _cmp_int32
    else:
        comp = _cmp_int64
    if m > 0:
        qsort(keys, m, sizeof(DateLike), comp)
    for i in range(n):
        out[i] = 1 if bsearch(&src[i], keys, m, sizeof(DateLike), comp) != NULL else 0


# ============================================================
#  TimeColumn 类实现 (全面 int32_t 与 SIMD 向量化优化)
# ============================================================
