#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "========================================="
echo "  FluidDataFrame 清理 + 构建 + 全量测试"
echo "========================================="

# ---------- 清理 ----------
echo ""
echo "[0/3] 清理旧产物..."
rm -rf build/
find fluidframe/core -name "*.so" -delete 2>/dev/null || true
find fluidframe/core -name "*.c" ! -name "_base.c" -delete 2>/dev/null || true
find fluidframe/core -name "*.cpp" -delete 2>/dev/null || true
echo ">>> 清理完成"

# ---------- 构建 ----------
echo ""
echo "[1/3] 构建中..."
python3 setup.py build_ext --inplace 2>&1
BUILD_OK=$?

if [ $BUILD_OK -ne 0 ]; then
    echo ""
    echo ">>> 构建失败，终止测试。"
    exit 1
fi

echo ""
echo ">>> 构建成功"

# ---------- 测试 ----------
echo ""
echo "[2/3] 运行测试..."
echo "-----------------------------------------"

TESTS=(
    test_astype
    test_boolcolumn
    test_categorycolumn
    test_char256_str_accessor
    test_char256column
    test_datecolumn
    test_datetimecolumn
    test_doublecolumn
    test_fastdataframe
    test_fluidframe
    test_groupby
    test_index
    test_indexers
    test_intcolumn
    test_linalg
    test_rolling
    test_shift_and_drop
    test_timecolumn
    test_ucs2_str_accessor
    test_ucs2column
    test_zero_copy
)

TOTAL_PASSED=0
TOTAL_FAILED=0
FAILED_FILES=()

for t in "${TESTS[@]}"; do
    RESULT=$(PYTHONPATH=. python3 "tests/${t}.py" 2>&1 | tail -1)
    PASSED=$(echo "$RESULT" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+')
    FAILED=$(echo "$RESULT" | grep -oE '[0-9]+ failed' | grep -oE '[0-9]+')

    TOTAL_PASSED=$((TOTAL_PASSED + ${PASSED:-0}))
    TOTAL_FAILED=$((TOTAL_FAILED + ${FAILED:-0}))

    if [ "${FAILED:-0}" -eq 0 ]; then
        echo "  ✓ $t: $RESULT"
    else
        echo "  ✗ $t: $RESULT"
        FAILED_FILES+=("$t")
    fi
done

echo ""
echo "========================================="
echo "  [3/3] 总计: ${TOTAL_PASSED} passed, ${TOTAL_FAILED} failed"
echo "========================================="

if [ ${TOTAL_FAILED} -gt 0 ]; then
    echo "  失败文件: ${FAILED_FILES[*]}"
    exit 1
else
    echo "  全部通过！"
    exit 0
fi
