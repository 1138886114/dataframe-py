#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "========================================="
echo "  FluidDataFrame 全量测试"
echo "========================================="

echo ""
echo "[0/2] 运行测试..."
echo "-----------------------------------------"

TESTS=(
    test_astype
    test_boolcolumn
    test_categorycolumn
    test_asciicolumn
    test_datecolumn
    test_datetimecolumn
    test_doublecolumn
    test_dataframe
    test_fluidframe
    test_groupby
    test_index
    test_indexers
    test_intcolumn
    test_linalg
    test_rolling
    test_shift_and_drop
    test_timecolumn
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
echo "  [1/1] 总计: ${TOTAL_PASSED} passed, ${TOTAL_FAILED} failed"
echo "========================================="

if [ ${TOTAL_FAILED} -gt 0 ]; then
    echo "  失败文件: ${FAILED_FILES[*]}"
    exit 1
else
    echo "  全部通过！"
    exit 0
fi
