#!/usr/bin/env bash
# tools/fetch_inkling2.sh - fetch Inkling 2 Mini's GGUF through the Hugging Face mirror.
#
# huggingface.co is unreachable on this PC (the proxy answers CONNECT with 502), but hf-mirror.com serves the
# same repo.  The mirror's certificate does not chain to the local store, so verification is off (-k /
# --check-certificate=false).  The file is 16.42 GiB; aria2c resumes, so a dropped connection costs nothing and
# rerunning this script picks up where it stopped.
#
# usage: bash tools/fetch_inkling2.sh [QUANT]     (default UD-Q4_K_XL, the quant the plan's llama.cpp runs name)
set -u

QUANT="${1:-UD-Q4_K_XL}"
REPO="unsloth/Inkling-2-Mini-GGUF"
DIR="${INKLING_DIR:-/c/Users/zshua/models/inkling2-ud-q4_k_xl}"
FILE="Inkling-2-Mini-${QUANT}.gguf"
URL="https://hf-mirror.com/${REPO}/resolve/main/${FILE}"
NEED_MIB=$((18 * 1024))          # 16.42 GiB plus a margin, in MiB

mkdir -p "$DIR" || exit 1
cd "$DIR" || exit 1

free_mib=$(df -m . | awk 'NR==2 {print $4}')
if [ "$free_mib" -lt "$NEED_MIB" ]; then
    echo "fetch_inkling2: ${free_mib} MiB free, ${NEED_MIB} MiB needed - free some space first" >&2
    exit 1
fi

# the size the mirror reports is the whole file: the fetch is done once the file matches it
want=$(/c/WINDOWS/system32/curl.exe -sSkL -m 60 -r 0-0 -o /dev/null -D - "$URL" 2>/dev/null |
       tr -d '\r' | grep -i '^content-range' | sed 's/.*\///')
if [ -z "${want:-}" ]; then
    echo "fetch_inkling2: the mirror did not report a size for ${FILE}" >&2
    exit 1
fi
echo "fetch_inkling2: ${FILE} ${want} bytes -> ${DIR} (${free_mib} MiB free)"

for attempt in $(seq 1 40); do
    if command -v aria2c > /dev/null 2>&1; then
        aria2c -c -x8 -s8 -k4M --file-allocation=none --check-certificate=false \
               --max-tries=0 --retry-wait=5 --auto-file-renaming=false --allow-overwrite=true \
               --console-log-level=warn --summary-interval=30 \
               -d "$DIR" -o "$FILE" "$URL"
        rc=$?
    else
        /c/WINDOWS/system32/curl.exe -sSkL -C - --retry 5 --retry-delay 5 --retry-all-errors \
            --connect-timeout 30 --speed-limit 1024 --speed-time 120 -o "$FILE" "$URL"
        rc=$?
    fi
    have=$(stat -c%s "$FILE" 2>/dev/null || echo 0)
    echo "fetch_inkling2: attempt ${attempt} rc=${rc} ${have} of ${want} bytes"
    [ "$have" = "$want" ] && { echo "FETCH_OK ${DIR}/${FILE}"; exit 0; }
done
echo "fetch_inkling2: gave up after 40 attempts" >&2
exit 1
