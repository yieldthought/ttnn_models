# Leader Guide

This document is for the "leader" role: triaging the project board, merging PRs, scheduling work, and managing workers/reservations.

Task runners should ignore this file. Task runners should only follow `tasks/*.yaml` for the single issue they are working on.

Leadbook usage:
- Treat `LEADBOOK.md` as the working page for every check-in.
- Capture the process of decisions (signals, turns, next moves), not just outcomes.
- Use archived diaries only for milestone snapshots, not for every check-in.
- Add new entries under "Latest Entries" (newest first).

## Role And Stance

- This machine is a control room, not a lab. Use it to decide, delegate, and unblock.
- A status check is necessary but not sufficient. It orients you; it does not finish the work.
- Use judgment when instructions are incomplete or conflicting. Own the outcome.

Execution model: issues are the work orders and workers execute them. As the lead you advance goals by shaping and moving issues, starting/stopping workers, reviewing results, and closing loops. You should be very agentic and proactive in *running the system* but should not go in and act as a worker/runner yourself; that's what we have the task delegatino system and github project board for. Your focus should be on the smooth prorgession of the whole. As an example of that, SSHing into a worker node is definitely appropriate for observation, diagnosis, and unblocking, but making model code changes and running TT workloads should default to worker tasks by a runner and not be done from you in the lead loop.

## Status Checks

A status check is a short sweep you perform at the start of a check-in to ground yourself.

1. Read `LEADBOOK.md` to pick up the current thread.
2. Scan the project board (`gh-task list`) for Backlog/Ready/In progress/In review/Done.
3. Review In review items: merge if correct, or requeue with guidance (see Issue Hygiene).
4. Ensure runners/reservations line up with Ready work (see Reservations and Device Safety).
5. Confirm the goal-specific state (MODELS.md rows, missing entries, regressions) as needed.

The output of a status check is a decision: what to start, stop, or fix next.

## Where Things Live

- Goal prompts (used for `codexapi lead` loops): `scripts/goal_functional.txt`, `scripts/goal_optimized.txt`, `scripts/goal_current.txt`
- Leadbook working page (created by lead runs): `LEADBOOK.md`
- Archived diaries: `doc/diaries/`
  - Functional release diary (2026-02-09): `doc/diaries/functional_release_diary.md`
- Tasks executed by workers: `tasks/*.yaml`
- Worker loop: `scripts/worker.sh`

## Task Naming And Scope

Prefer tasks named by the *operation* (what changes) rather than the *model family* (what it is).

- Good: `run_tests`, `fix_correctness`, `increase_seq_len`, `bringup_model` (generic bringup, if/when needed)
- Avoid when possible: `bringup_<model-family>` (unless the generic task repeatedly fails without extra guardrails)

Why `bringup_arcee` existed historically:
- At the time it was created, `tasks/` only had `run_tests*` tasks (which assume a working `model.py` already exists).
- The Arcee issues were "bring up missing multi-device ports" (create `model.py`, keep paged cache behavior, then demo/eval).
- The name was convenience, not a requirement. Treat it as an exception, not a pattern.

Current state:
- The bringup task in this repo is still named `tasks/bringup_arcee.yaml` for historical reasons.
- If bringup work expands beyond the Arcee family, consider renaming/adding a generic `tasks/bringup_model.yaml`.

Rule of thumb:
- If a generic task works with a few extra lines in the issue body, keep the task generic and put the model-specific guidance in the issue.
- If multiple attempts keep failing for the same reason, promote those guardrails into the task file.

## Hardware Filtering (Worker Prefilter)

Workers prefilter by issue title with `--only-matching "/<system>/"` (for `<system>` in `n150|n300|t3000`).

Starting workers:
- Prefer passing the system explicitly: `scripts/worker.sh <agent_name> n300`
- `scripts/worker.sh` can infer the system from `TT_VISIBLE_DEVICES`, but that assumes it is already set correctly.

Leader responsibilities:
- Ensure issue titles include a model path containing `/<system>/` (example: `models/<org>/<model>/n300/functional`).
- If you add a new task file, commit it to `main` before moving issues to Ready (workers `git pull` before every loop).

## Device Safety

- Run exactly one TT workload per host. Do not run two workers on the same host.
- Do not share a repo clone between workers: `scripts/worker.sh` stashes on sync (`git stash -u`) and will trample uncommitted logs/edits from other workers.
- Use `tt-smi -ls` to list boards and exit. Avoid `tt-smi -l` (interactive UI).
- If `tt-smi -ls` fails with `unordered_map::at`, assume the host is in a bad state; move the worker to a different host/reservation.

## Reservations (IRD)

- Keep a >4h buffer on active reservations; extend before they get low.
- Release idle reservations early (especially if there are no Ready tickets for that system).
- IRD renumbers remaining reservations after a release; if releasing multiple, re-run `ird list` between releases (or release highest IDs first).
- If IRD cluster health checks flap, `ird list`/`ird release` may work with `--skip-clusters-check`.
- Do not assume all clusters have the same environment. Some hosts may be missing setup scripts or `/proj_sw` content; if basic setup is missing, release and move on.

## Common Runtime Failures (And What To Do)

- `cannot map elf file into memory: No space left on device`:
  - Rerun with `TT_METAL_CACHE=/tmp/tt-metal-cache` and `TT_METAL_RUNTIME_ROOT=/proj_sw/user_dev/moconnor/tt-runtime-root`.
- TT JIT compile errors in `llk_*` headers (e.g. `_llk_unpack_untilize_init_` arg mismatch):
  - Fix the `tt-metal` checkout used on that host:
    - `cd /proj_sw/user_dev/moconnor/tt-metal && git submodule update --init --recursive`
  - Then rerun demo/eval.
- Device open / hugepage pin failures:
  - Usually host-specific. Requeue the issue and move the worker to a different host.
- GitHub transient 500/502 during worker `git pull`:
  - Let the worker retry. Avoid intervention unless the worker is stuck for a long time.

## When Tasks Do Not Succeed Or Things Go Wrong

When a task arrives in In Review in a failed state or when a PR is rejected, it is essential that you ssh into the machine and investigate the ~/.codex/session logs from it and its verifier to understand what actually happened. A failure here means there is something to learn. In the past this has surfaced issues such as:
- A misspecified task that meant the agent was instructed not to commit the model files it had changed.
- A change in codex output meant that codexapi was including thinking summaries in its output which meant the verifier's results no longer parsed as JSON.
- A worker had executed on the wrong host and was trying to attempt t3000 tasks on an n300.
It is very likely that a failed PR means there is something wrong with the tasks, the runners or the system. You must investigate and try to find out what so that you can fix it and avoid future failures.

## Issue Hygiene (gh-task / codexapi Protocol)

`codexapi task` uses the gh-task ownership protocol:

- Adds `owner: <agent_name>` label while in progress.
- Updates a `## Progress` section in the issue body.
- Adds `✓` (success) or `⨉` (failure) marker labels when done and moves to In review.

When requeueing a stuck/failed issue:

- Remove `owner:` label.
- Remove stale `✓` / `⨉`.
- Delete the stale `## Progress` section.
- Move the project item back to Ready.

If an issue cannot be taken due to a stale `Taking: <agent>` comment, delete that comment.

## PR Creation And Auto-Closing Issues

Gotcha: `gh pr create --body "...\nFixes #123"` can end up with literal `\\n` sequences and fail to auto-close the issue.

Preferred:
- Use `gh pr create --body-file ...`
- Put `Fixes #<issue-number>` on its own line in the PR body.

If a PR should not close the issue (work in progress), use `Refs #<issue-number>` instead.

## MODELS.md And Logs

- Always keep `demo.log` and `eval.log` as deliverables (even on failure).
- Expect `MODELS.md` merge conflicts when merging multiple PRs; resolve carefully and keep both rows when appropriate.
- If a worker produced a messy/conflicting PR, it can be faster to replace it with a clean PR containing only the intended changes.
