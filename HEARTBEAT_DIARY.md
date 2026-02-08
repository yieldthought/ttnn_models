# Heartbeat Diary

## 2026-02-09T00:28:33+01:00

- Project `yieldthought/projects/6` status: Done 12, In progress 2, Ready 11, Backlog 3.
- Merged PR `yieldthought/ttnn_models#40` (gemma-3-4b-it n300) and it closed issue #26; MODELS.md now records gemma n300 as `bad` based on gibberish demo output and low eval accuracy.
- Added per-system task labels/files (`run_tests_n150`, `run_tests_n300`, `run_tests_t3000`) and updated `scripts/worker.sh` to run the correct task file per worker (to prevent n300 workers taking t3000 tickets and vice versa).
- Labeled all open tickets with the new per-system labels. Left Backlog tickets in Backlog since their model dirs are missing (AFM-4.5B n300/t3000, Arcee-Spark n300).
- Started three IRD reservations on `tt_aus` and launched workers.
- Runner ID 1 `aus-wh-09:42857` `agent1` (t3000) working on issue #29; log `/tmp/agent1.log`.
- Runner ID 2 `aus-wh-01:42857` `agent2` (n300) working on issue #30; log `/tmp/agent2.log`.
- Runner ID 3 `aus-wh-01:42858` `agent3` (n150) idle (no `run_tests_n150` Ready issues); log `/tmp/agent3.log`.
- Fixed a stuck ticket (#28) that got left in "In progress" with `owner: agent2` when an early/old worker instance was killed; moved it back to Ready and cleared ownership.

## 2026-02-09T00:35:46+01:00

- Project `yieldthought/projects/6` status unchanged: Done 12, In progress 2, Ready 11, Backlog 3. No items in "In review" and no open PRs.
- Runners still healthy with >7h remaining on all reservations (`ird list` shows IDs 1-3 with ~7:45-7:49 left).
- Verified agent processes are active:
- `agent1` (t3000) still running `codexapi task` for issue #29.
- `agent2` (n300) still running `codexapi task` for issue #30.
- `agent3` (n150) idle and polling; no `run_tests_n150` Ready issues.
- Found a bad state on runner ID 2 where an old `codexapi task ... tasks/run_tests.yaml` process was still running alongside the intended `tasks/run_tests_n300.yaml` run; killed the stray `run_tests.yaml` process to restore single-task/device safety.
