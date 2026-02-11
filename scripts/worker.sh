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

select_task_files() {
    task_files=(tasks/*.yaml)
    if [ ! -f "${task_files[0]}" ]; then
        echo "[worker] ERROR: No task files found (expected tasks/*.yaml)" >&2
        return 1
    fi
}

if ! select_task_files; then
    exit 2
fi
echo "[worker] Starting $agent_name for $system using task files:"
for f in "${task_files[@]}"; do
    echo "[worker] - $f"
done

sync_repo() {
    # Many runner environments have GH_TOKEN but no SSH keys. If origin is an SSH
    # remote, switch to https and let `gh auth setup-git` provide credentials.
    origin_url=$(git remote get-url origin 2>/dev/null || true)
    if [[ "$origin_url" == git@github.com:* ]]; then
        https_url="https://github.com/${origin_url#git@github.com:}"
        echo "[worker] Switching origin remote to https: $https_url"
        git remote set-url origin "$https_url" || return 1
    fi
    if command -v gh >/dev/null 2>&1; then
        gh auth setup-git >/dev/null 2>&1 || true
    fi

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
    # Refresh in case new tasks were added by the latest main pull.
    if ! select_task_files; then
        echo "[worker] Task file selection failed; sleeping 60s" >&2
        sleep 60
        continue
    fi
    only_matching="/${system}/"
    # Ensure codexapi output is flushed promptly to nohup logs.
    PYTHONUNBUFFERED=1 codexapi task  -p https://github.com/users/yieldthought/projects/6 -n "$agent_name" --only-matching "$only_matching" "${task_files[@]}"
    sleep 60
done
