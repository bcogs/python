#! /bin/sh

ruff="$(python3 -m site --user-base)/bin/ruff"
[ -z "$ruff" ] && exit 1
"$ruff" check . --fix && "$ruff" format .
