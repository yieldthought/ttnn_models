#!/bin/bash

if [ "$1" == "" ] || [ "$1" == "--help" ]; then
    echo "Usage: $0 AGENT_NAME [n150|n300|t3000]"
    echo "Run a loop taking and working on tasks from https://github.com/users/yieldthought/projects/6."
    echo "AGENT_NAME should be unique to this worker."
    echo ""
    echo "If the system is omitted, we try to infer it from TT_VISIBLE_DEVICES."
    exit 1
fi

agent_name="$1"
system="$2"

detect_system() {
    if [ -n "$system" ]; then
        return
    fi
    if [ -z "$TT_VISIBLE_DEVICES" ]; then
        return
    fi
    visible=$(echo "$TT_VISIBLE_DEVICES" | tr -d " ")
    IFS=',' read -r -a devs <<<"$visible"
    case "${#devs[@]}" in
    1) system="n150" ;;
    2) system="n300" ;;
    8) system="t3000" ;;
    esac
}

detect_system

if [ "$system" != "n150" ] && [ "$system" != "n300" ] && [ "$system" != "t3000" ]; then
    echo "[worker] ERROR: Unknown system '$system'. Pass n150|n300|t3000 or set TT_VISIBLE_DEVICES." >&2
    exit 2
fi

# Always run from the repo root so tasks/* and git commands work even if this
# script is launched from another directory.
repo_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_root" || exit 1

task_file="tasks/run_tests_${system}.yaml"
if [ ! -f "$task_file" ]; then
    echo "[worker] ERROR: Missing task file: $task_file" >&2
    exit 2
fi
echo "[worker] Starting $agent_name for $system using $task_file"

sync_repo() {
    if [ -n "$(git status --porcelain)" ]; then
        echo "[worker] Working tree dirty; stashing before sync"
        git stash push -u -m "worker auto-stash $(date -Is)" >/dev/null || return 1
    fi
    git checkout -q main || return 1
    git pull -q --ff-only origin main || return 1
}

while true; do
    if ! sync_repo; then
        echo "[worker] Repo sync failed; sleeping 60s" >&2
        sleep 60
        continue
    fi
    codexapi task -p https://github.com/users/yieldthought/projects/6 -n "$agent_name" "$task_file"
    sleep 60
done
