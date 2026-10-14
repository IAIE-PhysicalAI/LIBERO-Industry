#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PY="${PYTHON:-python}"
CONVERTER="$SCRIPT_DIR/convert_to_aubo.py"
SOURCE="${1:?Usage: $0 SOURCE_DIR OUTPUT_DIR}"
RAW="${2:?Usage: $0 SOURCE_DIR OUTPUT_DIR}"
RUN_ID="${AUBO_RUN_ID:-aubo-full-raw-$(date +%Y%m%d-%H%M%S)}"
RUN="$REPO/.industry-runs/logs/$RUN_ID"

mkdir -p "$RUN" "$RAW"

record_exit() {
    rc=$?
    printf '%s\n' "$rc" > "$RUN/worker.exitcode"
    date --iso-8601=seconds > "$RUN/finished_at.txt"
}
trap record_exit EXIT

exec 9>"$RAW/full_conversion.lock"
if ! flock -n 9; then
    echo "Another worker owns $RAW/full_conversion.lock" >&2
    exit 75
fi

cd "$REPO"
"$PY" -m py_compile "$CONVERTER" || exit 24

for suite in libero_spatial libero_object libero_goal libero_10; do
    count="$(find "$SOURCE/$suite" -maxdepth 1 -type f -name '*.hdf5' | wc -l)"
    if [[ "$count" != 10 ]]; then
        echo "Expected 10 source HDF5 files in $suite, found $count" >&2
        exit 25
    fi
done

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="$RUN/numba-cache"
export MPLCONFIGDIR="$RUN/matplotlib"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"

printf '%s\n' "$RUN" > "$REPO/.industry-runs/logs/aubo-full-raw-current.txt"
date --iso-8601=seconds > "$RUN/started_at.txt"
overall_rc=0
file_count=0
failed_file_count=0

for suite in libero_spatial libero_object libero_goal libero_10; do
    source_dir="$SOURCE/$suite"
    output_dir="$RAW/$suite"
    mkdir -p "$output_dir"
    printf '[SUITE START] %s %s\n' "$(date --iso-8601=seconds)" "$suite"
    suite_rc=0
    while IFS= read -r -d '' input_hdf5; do
        file_count=$((file_count + 1))
        filename="$(basename "$input_hdf5")"
        output_hdf5="$output_dir/$filename"
        printf '[FILE START] %s %s/%s\n' \
            "$(date --iso-8601=seconds)" "$suite" "$filename"

        # One task owns one fresh Python / MuJoCo process. Both environments
        # close before this child exits, so native state cannot reach the next task.
        "$PY" -u "$CONVERTER" \
            --input_hdf5 "$input_hdf5" \
            --output_hdf5 "$output_hdf5" \
            --seed 0 \
            --resume
        rc=$?
        printf '[FILE END] %s %s/%s rc=%s\n' \
            "$(date --iso-8601=seconds)" "$suite" "$filename" "$rc"
        if [[ $rc -ne 0 ]]; then
            suite_rc=1
            overall_rc=1
            failed_file_count=$((failed_file_count + 1))
        fi
    done < <(find "$source_dir" -maxdepth 1 -type f -name '*.hdf5' -print0 | sort -z)
    printf '%s\n' "$suite_rc" > "$RUN/$suite.exitcode"
    printf '[SUITE END] %s %s rc=%s\n' \
        "$(date --iso-8601=seconds)" "$suite" "$suite_rc"
done

printf '%s\n' "$overall_rc" > "$RUN/conversion.exitcode"
printf '[ALL SUITES END] %s rc=%s files=%s failed_files=%s\n' \
    "$(date --iso-8601=seconds)" "$overall_rc" "$file_count" "$failed_file_count"
exit "$overall_rc"
