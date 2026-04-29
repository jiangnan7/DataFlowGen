#!/usr/bin/env bash
set -euo pipefail

mkdir -p hardfloat
cd hardfloat

if [ ! -d berkeley-hardfloat/.git ]; then
  git clone https://github.com/ucb-bar/berkeley-hardfloat.git
fi

cd berkeley-hardfloat
HARDFLOAT_COMMIT=70455e53f233a06cb5a342d125e22b7b1505c271
if [ "$(git rev-parse HEAD)" != "${HARDFLOAT_COMMIT}" ]; then
  git checkout --detach "${HARDFLOAT_COMMIT}"
fi

if command -v sbt >/dev/null 2>&1; then
  sbt "publishLocal"
elif [ -f sbt-launch.jar ]; then
  java -jar sbt-launch.jar "publishLocal"
else
  echo "Error: sbt is not installed and sbt-launch.jar is missing." >&2
  exit 1
fi

shopt -s dotglob
for item in *; do
  if [ ! -e "../${item}" ]; then
    mv "${item}" ..
  fi
done
cd ..
rm -rf berkeley-hardfloat
