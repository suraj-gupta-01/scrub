#!/usr/bin/env bash
# =============================================================================
# run_tests.sh — Run the full ASV test suite with clear terminal output
#
# Usage:
#   bash run_tests.sh              # run all tests
#   bash run_tests.sh heading      # run only heading tests
#   bash run_tests.sh recovery     # run only recovery tests
#   bash run_tests.sh sensor_hub   # run only sensor hub tests
#   bash run_tests.sh hardware     # run only hardware tests
#   bash run_tests.sh -v           # run all, extra verbose
# =============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         ASV Navigation System — Unit Test Suite              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Which test file to run (default: all)
TARGET=${1:-""}
VERBOSE=${2:-""}

if [ "$TARGET" = "-v" ]; then
  VERBOSE="-v"
  TARGET=""
fi

if [ -z "$TARGET" ]; then
  TEST_PATH="tests/"
  echo "► Running ALL tests"
else
  TEST_PATH="tests/test_${TARGET}.py"
  echo "► Running tests for: $TARGET"
fi

echo ""

# Run pytest with:
#   -v              : verbose (one line per test)
#   -s              : show print() output so you can see the log messages
#   --timeout=10    : kill any test that hangs after 10 seconds
#   --tb=short      : short traceback on failure (easier to read)
#   -p no:warnings  : hide Python deprecation warnings (keep output clean)

python -m pytest "$TEST_PATH" \
    -v \
    -s \
    --timeout=10 \
    --tb=short \
    -p no:warnings \
    $VERBOSE

EXIT_CODE=$?

echo ""
echo "─────────────────────────────────────────────────────────────"
if [ $EXIT_CODE -eq 0 ]; then
  echo "  ✅  ALL TESTS PASSED"
else
  echo "  ❌  SOME TESTS FAILED  (exit code $EXIT_CODE)"
  echo ""
  echo "  To see more detail on a failure:"
  echo "    pytest tests/ --tb=long -s"
fi
echo "─────────────────────────────────────────────────────────────"
echo ""

exit $EXIT_CODE