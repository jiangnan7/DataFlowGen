#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if command -v sbt >/dev/null 2>&1; then
  SBT_CMD=(sbt)
elif [ -f sbt-launch.jar ]; then
  SBT_CMD=(java -jar sbt-launch.jar)
else
  echo "Error: sbt is not installed and sbt-launch.jar is missing." >&2
  exit 1
fi

mapfile -t test_files < <(find src/test/scala/generator -maxdepth 1 -name '*DF_test.scala' | sort)

if [ "${#test_files[@]}" -eq 0 ]; then
  echo "Error: no generator tests found under src/test/scala/generator." >&2
  exit 1
fi

for test_file in "${test_files[@]}"; do
  test_name="$(basename "${test_file}" .scala)"
  echo "Running heteacc.generator.${test_name}"
  "${SBT_CMD[@]}" "testOnly heteacc.generator.${test_name}"
done
