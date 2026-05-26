#!/bin/sh
set -eu

if [ "${GOOGLE_API_KEY:-}" != "" ] && [ -f "${GOOGLE_API_KEY}" ]; then
  GOOGLE_API_KEY="$(cat "${GOOGLE_API_KEY}")"
  export GOOGLE_API_KEY
fi

exec "$@"
