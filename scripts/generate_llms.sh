#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

if ! command -v npx >/dev/null 2>&1; then
  echo "Error: npx không tồn tại. Hãy cài Node.js/npm trước." >&2
  exit 1
fi

REPOMIX_TMP_DIR="$(mktemp -d)"
REPOMIX_CONFIG="$REPOMIX_TMP_DIR/repomix.config.json"

cleanup() {
  rm -rf "$REPOMIX_TMP_DIR"
}

trap cleanup EXIT

cat > "$REPOMIX_CONFIG" <<'JSON'
{
  "input": {
    "processors": [
      {
        "pattern": "**/*.jsonl",
        "command": "head -n 10 {file}"
      }
    ]
  }
}
JSON

echo "Generating llms.txt..."

npx --yes repomix@1.17.0 \
  --config "$REPOMIX_CONFIG" \
  --style plain \
  --remove-empty-lines \
  --output llms.txt \
  --ignore "llms.txt"

echo "Staging llms.txt..."

git add -- llms.txt

echo "Done."