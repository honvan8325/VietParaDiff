#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

SAMPLE_COUNT=10
SAMPLE_TMP_DIR="$(mktemp -d)"
ALLOWLIST="$SAMPLE_TMP_DIR/allowlist.txt"
TRACKED_IMAGES="$SAMPLE_TMP_DIR/tracked-images.txt"

cleanup() {
  rm -rf "$SAMPLE_TMP_DIR"
}

trap cleanup EXIT

: > "$ALLOWLIST"

for image_dir in data/*/images; do
  if [[ ! -d "$image_dir" ]]; then
    continue
  fi

  find "$image_dir" \
    -maxdepth 1 \
    -type f \
    -print \
    | LC_ALL=C sort \
    | sed -n "1,${SAMPLE_COUNT}p" \
    >> "$ALLOWLIST"
done

LC_ALL=C sort -u -o "$ALLOWLIST" "$ALLOWLIST"
git ls-files "data/*/images/*" \
  | LC_ALL=C sort -u \
  > "$TRACKED_IMAGES"

while IFS= read -r tracked_image; do
  if [[ -z "$tracked_image" ]]; then
    continue
  fi
  if ! grep -Fqx -- "$tracked_image" "$ALLOWLIST"; then
    git rm --cached --ignore-unmatch -- "$tracked_image"
  fi
done < "$TRACKED_IMAGES"

while IFS= read -r sample_image; do
  if [[ -n "$sample_image" ]]; then
    git add -f -- "$sample_image"
  fi
done < "$ALLOWLIST"

sample_total="$(wc -l < "$ALLOWLIST" | tr -d '[:space:]')"
echo "Staged $sample_total deterministic sample image(s)."
