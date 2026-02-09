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

## 2026-02-09T01:08:45+01:00

- Project `yieldthought/projects/6` status: Done 13, In progress 2, Ready 10, Backlog 3.
- Merged PR `yieldthought/ttnn_models#42` (ALLaM-7B t3000) and moved issue #29 to Done (issue closed by PR).
- Runners:
- ID 1 `agent1` (t3000) moved on to issue #31 (Llama-3.2-1B t3000); progress not posted yet.
- ID 2 `agent2` (n300) still on issue #30; agent is actively running demo/eval with an explicit n300 mesh graph descriptor and is generating demo/eval logs (plus a small `eval.py` CLI alias tweak for `--max-tokens`).
- ID 3 `agent3` (n150) still idle; keeps creating `worker auto-stash` entries due to periodic dirty working tree (needs cleanup/limits if this runs for days).
- Found a repo correctness gap: `models/arcee-ai/Arcee-Spark/t3000/functional/` has logs but no `model.py` in git even though the logged commands reference it (likely requires a follow-up bringup/fix ticket).

## 2026-02-09T01:29:57+01:00

- Project `yieldthought/projects/6` status unchanged: Done 13, In progress 2 (#30 n300, #31 t3000), Ready 10, Backlog 3. No items in "In review" and no open PRs.
- IRD reservations still healthy with >6h remaining on all three runners (IDs 1-3).
- Agent status:
- `agent1` (t3000) is still working on issue #31; last public progress notes report device discovery failure, but Codex session logs show active investigation in tt-metal fabric/mesh config.
- `agent2` (n300) is still working on issue #30; earlier verifier checks found missing logs/updates, and the agent is now investigating why demo/eval logs appear to disappear between steps.
- `agent3` (n150) remains idle (no `run_tests_n150` Ready issues) and continues periodic auto-stashing due to a dirty tree.
- Learned `tt-smi -l` is "local chips" and launches the interactive UI; use `tt-smi -ls` to list boards and exit (killed a stray `tt-smi -l` I accidentally started during diagnosis).
- Updated all `tasks/run_tests*.yaml` prompts to use `--max_new_tokens` (instead of the incorrect `--max-tokens`) and added short notes about always emitting demo/eval logs, mesh descriptor env vars for n300/t3000, and the TT metal cache/runtime-root gotcha.
- Updated `scripts/worker.sh` to pass all task files matching `tasks/*_<system>.yaml` to `codexapi task` (so new per-system tasks can be added without changing the worker invocation).

## 2026-02-09T01:59:57+01:00

- Project `yieldthought/projects/6` status: Done 14, In progress 1 (#28 n300), Ready 10, Backlog 3.
- Merged PR `yieldthought/ttnn_models#43` (Llama-3.2-1B n300) and moved issue #30 to Done (issue auto-closed by PR). The run produced demo/eval logs showing an n300 mesh graph mapping failure (TT_FATAL), so MODELS.md row was updated to `bad`.
- Fixed a major runner interference issue: I had both n300 and n150 workers on `aus-wh-01` and they shared `/localdev/moconnor/ttnn_models`, causing the idle n150 worker to repeatedly `git stash -u` and silently remove the n300 worker's uncommitted logs/edits. Stopped the n150 worker and released that IRD reservation.
- T3000 runner status: `tt-smi -ls` consistently fails on `aus-wh-09` with `unordered_map::at`, matching the earlier multi-device init failures. Killed the t3000 worker, reset issue #31 back to Ready with stricter instructions (no model.py edits; set TT_VISIBLE_DEVICES from `tt-smi -ls`; set TT_MESH_GRAPH_DESC_PATH), and released the broken IRD reservation.
- Attempted to reserve new t3000 hardware on `tt_aus`; direct wormhole_b0 `--model lb` allocation failed, and a `--team models_team --model glx6u` reservation is still pending in the scheduler.
- Updated all `tasks/run_tests*.yaml` `set_up` sections to require running `tt-smi -ls` and setting `TT_VISIBLE_DEVICES` before demo/eval (IRD containers often expose non-0 PCI Dev IDs like `4,5`).

## 2026-02-09T02:29:57+01:00

- Project `yieldthought/projects/6` status: Done 15, In progress 2 (#31 t3000, #32 n300), Ready 8, Backlog 3.
- Merged PR `yieldthought/ttnn_models#44` (ALLaM-7B-Instruct-preview n300) and moved issue #28 from In review to Done (issue auto-closed by PR). Demo/eval logs show the same n300 mesh mapping failure, so MODELS.md row is `bad`.
- Started a new t3000 runner on a proper Loudbox host in `tt_yyz`:
- IRD ID 2 `wh-lb-43:42857` running `agent4` (t3000), log `/tmp/agent4.log`.
- Verified `tt-smi -ls` works on `wh-lb-43` and shows 4 PCI Dev IDs (0-3) for 4 n300 boards (2 chips each); this is the expected t3k mesh hardware.
- Updated `tasks/run_tests_n300.yaml` and `tasks/run_tests_t3000.yaml` instructions to clarify TT_VISIBLE_DEVICES selection for x2 boards/t3k Loudbox (often `0,1,2,3` on t3000, not `0-7`).
- Started an n150 runner on a dedicated AUS host:
- IRD ID 3 `aus-wh-09:42857` running `agent3` (n150), log `/tmp/agent3.log` (idle, no n150 Ready issues).

## 2026-02-09T03:06:21+01:00

- Project `yieldthought/projects/6` status: Done 17, In progress 2 (#32 n300, #35 t3000), Ready 6, Backlog 3.
- Merged PR `yieldthought/ttnn_models#45` (Phi-3-mini-128k-instruct t3000) and moved issue #33 to Done (issue auto-closed by PR). MODELS.md row is now `bad` with demo/eval logs captured (device open failed due to hugepage pin/memory allocation error).
- Cleaned up the stale n300 runner state:
- Released the broken AUS n300 reservation (two unlinked x1 boards caused mesh mapping failures and the worker was killed).
- Reset issue #32 back to Ready (released `owner: agent2`), then reserved a new AUS n300 x2 board (single PCI Dev ID `0` with n300 L+R) and restarted `agent2` on it.
- `agent2` (n300) is now actively running issue #32 again (Phi n300); log `/tmp/agent2.log` on `aus-wh-01:42857`.
- Re-established an idle n150 runner on a separate AUS host:
- Reserved `aus-wh-09` and started `agent3` (n150); log `/tmp/agent3.log` (still idle, no n150 Ready issues).
- T3000 runner `agent4` on Loudbox `wh-lb-43:42857` is healthy and moved on to issue #35 after completing #33; log `/tmp/agent4.log`.

## 2026-02-09T03:09:53+01:00

- Project `yieldthought/projects/6` status unchanged: Done 17, In progress 2 (#32 n300, #35 t3000), Ready 6, Backlog 3. No items in "In review" and no open PRs.
- IRD reservations still healthy with >7h remaining on all runners (IDs 1-3).
- Runners:
- `agent2` (n300) still actively running issue #32 (Phi n300); process `codexapi task` is alive and working.
- `agent4` (t3000) still actively running issue #35 (Mistral t3000); process `codexapi task` is alive and working.
- `agent3` (n150) idle (no `run_tests_n150` Ready issues).

## 2026-02-09T03:39:53+01:00

- Project `yieldthought/projects/6` status: Done 23, In progress 1 (#36 n300), Ready 1 (#38 n300), Backlog 3 (#22/#23/#24).
- Merged PR `yieldthought/ttnn_models#51` (Mistral-7B-Instruct-v0.3 n300) after bringing the branch up to date with `main` to resolve a `MODELS.md` merge conflict; issue #34 is now Done.
- IRD reservations still healthy with >6h remaining on all three runners (IDs 1-3).
- Runners:
- ID 1 `agent4` (t3000) idle; no `run_tests_t3000` Ready issues.
- ID 2 `agent2` (n300) actively running issue #36 (Falcon3 n300); log `/tmp/agent2.log` on `aus-wh-01:42857`.
- ID 3 `agent3` (n150) idle; no `run_tests_n150` Ready issues.

## 2026-02-09T04:09:53+01:00

- Project `yieldthought/projects/6` status: Done 24, In progress 2 (#23 t3000, #38 n300), Ready 2 (#22, #24).
- Merged PR `yieldthought/ttnn_models#52` (Falcon3 n300); issue #36 is now Done.
- Added new bringup task files and labels for Arcee multi-device bringup:
- `tasks/bringup_arcee_n300.yaml` (label `bringup_arcee_n300`)
- `tasks/bringup_arcee_t3000.yaml` (label `bringup_arcee_t3000`)
- Relabeled and moved Arcee backlog issues to Ready so workers can pick them up:
- #22 AFM-4.5B n300 -> `bringup_arcee_n300`
- #23 AFM-4.5B t3000 -> `bringup_arcee_t3000`
- #24 Arcee-Spark n300 -> `bringup_arcee_n300`
- Reopened #25 (Arcee-Spark t3000) because the repo has logs/MODELS entries but no committed `t3000/functional/model.py`; cleared stale `## Progress`, removed the `✓` label, relabeled to `bringup_arcee_t3000`, and moved it back to Ready.
- IRD reservations still healthy with >6h remaining on all runners (IDs 1-3).
- Runners:
- ID 1 `agent4` (t3000) started issue #23 using the new bringup task; log `/tmp/agent4.log` on `wh-lb-43:42857`.
- ID 2 `agent2` (n300) started issue #38 (Qwen3 n300 run_tests); it should pick #22/#24 next; log `/tmp/agent2.log` on `aus-wh-01:42857`.
- ID 3 `agent3` (n150) idle; no `run_tests_n150` Ready issues.

## 2026-02-09T05:03:34+01:00

- Project `yieldthought/projects/6` status: Done 26, In progress 2 (#22 n300, #25 t3000), Ready 2 (#23 t3000, #56 t3000), Backlog 0, In review 0.
- Merged PR `yieldthought/ttnn_models#53` (Qwen3-0.6B n300 run_tests) after adding missing `Seq len=2048` and it closed issue #38.
- Merged PR `yieldthought/ttnn_models#55` (Arcee-Spark n300 bringup) and it closed issue #24.
- T3000 runner diagnosis/fix:
- Found widespread t3k failures were coming from TT JIT compile errors in `llk_*` headers (e.g. `_llk_unpack_untilize_init_` arg mismatch) on `wh-lb-43`, caused by `tt-metal` submodules not matching the superproject gitlink.
- Fixed `wh-lb-43` by running `git submodule update --init --recursive` in `/proj_sw/user_dev/moconnor/tt-metal` (brought `tt_llk`/`umd` back to the pinned commits).
- Stopped the stuck t3000 worker, reset issue #25 back to Ready, then restarted `agent4` (t3000) on `wh-lb-43`.
- Requeued issue #23:
- Did not merge PR #54 (AFM-4.5B t3000) since demo/eval never ran successfully and the bringup drifted from the n150 paged-cache behavior.
- Replaced the issue body with concrete retry instructions (paged KV cache, no internal cache cap, update existing PR), removed the `✓` label, and moved it back to Ready.
- Task improvements:
- Updated `tasks/bringup_arcee_n300.yaml` and `tasks/bringup_arcee_t3000.yaml` to explicitly require keeping paged attention/KV-cache (matching n150), add a note about the `git submodule update` fix for `llk_*` JIT errors, and handle "PR already exists" when retrying.
- Runner reservations:
- Released an unused galaxy reservation (`wh-glx6u-06`) that had no worker attached.
- Stopped and released the idle n150 runner/reservation (`aus-wh-09`), leaving active n300 + t3000 runners only.
- Scheduled follow-up validation:
- Created issue #56 (`models/Qwen/Qwen3-0.6B/t3000/functional`) labeled `run_tests_t3000` to rerun t3000 demo/eval now that the t3k `tt-metal` submodules are fixed.
