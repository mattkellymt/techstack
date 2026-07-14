#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$HOME/github.com/mattkellymt/techstack/fedora"

USAGE=$(cat <<EOF
Usage: stack <command>

Commands:
  install  Install and configure the software stack
  upgrade  Upgrade all tools
EOF
)

case "${1:-}" in
    install)
        exec "$SCRIPT_DIR/install.sh"
        ;;
    upgrade)
        exec "$SCRIPT_DIR/upgrade.sh"
        ;;
    --help|-h|"")
        echo "$USAGE"
        ;;
    *)
        echo "stack: unknown command '$1'" >&2
        echo "$USAGE" >&2
        exit 1
        ;;
esac
