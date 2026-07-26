#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

if ! command -v npx >/dev/null 2>&1; then
  echo "Error: npx không tồn tại. Hãy cài Node.js/npm trước." >&2
  exit 1
fi

echo "Generating llms.txt..."

npx --yes repomix@1.17.0 \
  --style plain \
  --remove-empty-lines \
  --output llms.txt \
  --ignore "**/*.jsonl,llms.txt"

echo "Staging llms.txt..."

git add -- llms.txt
