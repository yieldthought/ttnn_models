#!/bin/bash

if [ "$1" == "" ] || [ "$1" == "--help" ]; then
    echo "Usage: $0 AGENT_NAME"
    echo "Run a loop taking and working on tasks from https://github.com/users/yieldthought/projects/6. AGENT_NAME should be unique to this worker."
    exit 1
fi

# Always run from the repo root so tasks/* and git commands work even if this
# script is launched from another directory.
repo_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_root" || exit 1

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
    codexapi task -p https://github.com/users/yieldthought/projects/6 -n "$1" tasks/*
    sleep 60
done
