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

## 2026-02-09T11:46:27+01:00

- Merged t3000 run_tests work that was blocked by merge conflicts:
- Recreated and merged PR #64 for issue #31 (Llama-3.2-1B t3000) and closed conflicted PR #63.
- Recreated and merged PR #73 for issue #35 (Mistral t3000) and closed conflicted PR #65.
- Updated local main to include the merges; MODELS.md now has:
- Llama-3.2-1B t3000: 89%/100%, TTFT 155ms, 9.9 t/s/u, seq 131072 (still below Top-1>=90 target).
- Mistral t3000: 100%/100%, TTFT 110ms, 10.6 t/s/u, seq 1024.

- Runners and reservations:
- Extended IRD ID 1 (`aus-wh-01`) timeout back to 8h (it had dropped below 4h).
- Restarted the n300 worker on `aus-wh-01` with the setup environment so PATH/GH tooling is exported correctly.
- Reserved a new AUS wormhole_b0 host (`wh-04`) as IRD ID 3 and started an n150 worker (`agent3`). The repo there was far behind and had untracked files blocking pull; stashed and fast-forwarded to main before starting the worker.

- Current runner state:
- ID 1 `aus-wh-01` -> `agent1` (n300) is working on issue #30 (Llama-3.2-1B n300 run_tests).
- ID 2 `wh-lb-43` -> `agent2` (t3000) is working on issue #37 (Falcon3 t3000 run_tests).
- ID 3 `wh-04` -> `agent3` (n150) is working on issue #66 (Arcee-Spark n150 run_tests).

- Project scheduling:
- Moved #28 (ALLaM n300) back to Ready and clarified TT_VISIBLE_DEVICES guidance in the issue body.
- Created and queued new run_tests issues so workers won't go idle after current work:
- #66 Arcee-Spark n150
- #67 Phi-3-mini n150
- #68 ALLaM t3000
- #69 gemma-3-4b-it n300
- #70 gemma-3-4b-it t3000
- #71 Phi-3-mini t3000
- #72 Arcee-Spark n300

## 2026-02-09T11:59:34+01:00

- Read and refreshed the repo's task runner docs and task files (`tasks/`, `../codexapi/README.md`, `../gh-task/README.md`) to ensure the current worker loop matches expectations.
- Project `yieldthought/projects/6` status: Backlog 0, In review 0, In progress 3 (#28 n300, #37 t3000, #66 n150), Ready 6 (#67-#72). No open PRs at the time of this tick.
- IRD reservations healthy with >6h remaining on all three workers:
- ID 1 `aus-wh-01:42857` -> `agent1` (n300) running `codexapi task` and actively executing ALLaM n300 demo/eval under `TT_VISIBLE_DEVICES=0` and the n300 mesh graph descriptor. `eval.log` shows Top-1 97% / Top-5 100%; `demo.log` shows TTFT 170ms and 8.5 t/s/u (still finalizing issue/PR flow).
- ID 2 `wh-lb-43:42857` -> `agent2` (t3000) running Falcon3 t3000 eval with `TT_VISIBLE_DEVICES=0,1,2,3` and the t3k mesh graph descriptor. `eval.log` shows Top-1 97% / Top-5 100% (awaiting PR).
- ID 3 `wh-04:42857` -> `agent3` (n150) running Arcee-Spark n150 demo; still in kernel build/model load (no results yet).
- Noted remaining release gaps in `MODELS.md` that will likely need follow-up after reruns: Arcee-Spark n150/n300 (<90 Top-1), Phi n150 (<90/<99), and Llama t3000 (Top-1 89%). Also several multi-device rows still use small `Seq len` values (e.g., 256/1024/2048) that may require a dedicated "increase max_seq_len + paged cache" effort depending on DRAM constraints.

## 2026-02-09T12:29:34+01:00

- In review triage/merges:
- Merged PR #80 (Arcee-Spark n150 run_tests rerun) and it closed #66; Top-1 regressed to 86% (below target) with refreshed logs.
- Merged PR #79 (Falcon3 t3000 rerun) and updated MODELS.md to Top-1 97% / Top-5 100% with new logs.
- #79 did not auto-close #37 because the PR body contained literal `\\n` sequences; closed #37 manually and moved it to Done.
- Merged PR #81 (ALLaM t3000 rerun) and it closed #68; MODELS.md now shows Top-1 96% / Top-5 100% (still Seq len 256 due to cache cap).

- Task/automation improvement:
- Merged PR #82 updating all task `on_success` instructions to create PRs using `gh pr create --body-file ...` and include a standalone `Fixes #<issue-number>` line so GitHub reliably auto-closes issues.

- Project `yieldthought/projects/6` status at end of tick: In review 0, In progress 3 (#28 ALLaM n300, #67 Phi n150, #70 gemma t3000), Ready 3 (#69 gemma n300, #71 Phi t3000, #72 Arcee-Spark n300).
- Runners: all IRD reservations healthy with >5h remaining; workers active on separate hosts (no shared /localdev interference).

## 2026-02-09T12:59:34+01:00

- In review cleanup:
- Closed PR #84 (ALLaM n300) since it was an empty placeholder (0 changed files) and the actual MODELS/log updates were already on main (commit aa8752d).
- Moved and closed completed run_tests issues from In review to Done: #28 (ALLaM n300), #67 (Phi n150), #70 (gemma t3000), #69 (gemma n300), #71 (Phi t3000). For #69/#71, added comments linking follow-up fix_correctness tickets.

- Task scheduling improvements:
- Added new generic correctness-fix tasks: `tasks/fix_correctness_n150.yaml`, `tasks/fix_correctness_n300.yaml`, `tasks/fix_correctness_t3000.yaml`.
- Updated the n300/t3000 fix tasks with the known `tt-metal` submodule recovery command for `llk_*` JIT header mismatches (`git submodule update --init --recursive`).

- Queued correctness work to keep runners busy:
- Created issues #86-#92 (fix_correctness) and put them in Ready. Active work started immediately:
- #72 (run_tests n300, owner: agent1) in progress on `aus-wh-01`.
- #86 (Phi n150 fix_correctness, owner: agent3) in progress on `wh-04`.
- #90 (Llama t3000 fix_correctness, owner: agent2) in progress on `wh-lb-43`.

- Runner health:
- IRD reservations still healthy with >5h remaining: ID 1 `aus-wh-01:42857` (n300), ID 2 `wh-lb-43:42857` (t3000), ID 3 `wh-04:42857` (n150).
- Verified each host has an active `codexapi task` process and is currently executing demo runs for its in-progress issue.

## 2026-02-09T13:29:34+01:00

- Project `yieldthought/projects/6` status: In review 0, In progress 3 (#72 Arcee-Spark n300 run_tests, #86 Phi n150 fix_correctness, #90 Llama t3000 fix_correctness), Ready 5 (#87-#92 excluding those in progress), Backlog 0.
- No new PRs opened since the last tick.
- Runners:
- IRD reservations still healthy with >5h remaining on all three workers (IDs 1-3).
- Verified each host has active `codexapi task` + child Python workloads running for the in-progress issues:
  - `aus-wh-01` agent1: running `demo.py` for Arcee-Spark n300.
  - `wh-lb-43` agent2: running `demo.py` for Llama-3.2-1B t3000 correctness work.
  - `wh-04` agent3: running `demo.py` for Phi-3-mini n150 correctness work.
- No intervention required; waiting for the workers to finish and open PRs for review/merge.

## 2026-02-09T13:59:34+01:00

- In review triage/merges:
- Created PR #95 from local branch `pr72_arcee_spark_n300_logs` (Fixes #72) and merged it; #72 is now closed and in Done. MODELS.md Arcee-Spark n300 refreshed to TTFT 329ms / 4.9 t/s/u (Top-1 88%, Top-5 100%).
- Merged PR #94 (Fixes #90) for Llama-3.2-1B t3000 correctness; #90 is now closed and in Done. MODELS.md Llama t3000 is now Top-1 92%, Top-5 100%.

- Runners and reservations:
- Extended IRD timeouts for IDs 1-3 back to 8h to maintain the >4h buffer.
- Verified active workloads:
- ID 3 (`wh-04`, agent3/n150) is running `eval.py` for Arcee-Spark n150 (#87).
- ID 1 (`aus-wh-01`, agent1/n300) is actively editing `models/google/gemma-3-4b-it/n300/functional/model.py` (fixing stale state_dict prefixes) for #89.
- ID 2 (`wh-lb-43`, agent2/t3000) is running `demo.py` for gemma t3000 (#91).

- Project hygiene:
- #88 (Arcee-Spark n300) was double-owned/stale under `owner: agent1` while agent1 was clearly working #89. Cleared ownership and moved #88 back to Ready so it can be picked up cleanly after #89 completes.

- Task scheduling for the "full seq len" release goal:
- Added new tasks `tasks/increase_seq_len_n300.yaml` and `tasks/increase_seq_len_t3000.yaml` (merged via PR #97) so we can systematically raise max_seq_len / KV cache caps while keeping paged attention.
- Created Backlog issues #98-#105 (ALLaM/Mistral/Falcon/Qwen3/Phi n300 + ALLaM/Mistral/Falcon t3000) to raise Seq len toward the n150 baseline (or highest DRAM-fitting value).

- Project `yieldthought/projects/6` status at end of tick: In progress 3 (#87, #89, #91), Ready 2 (#88, #92), In review 0, Backlog 8 (#98-#105).

## 2026-02-09T14:39:20+01:00

- Project `yieldthought/projects/6` status: In progress 3 (#87 Arcee n150, #89 gemma n300, #91 gemma t3000), In review 0.
- Ready queue topped up for the next worker cycles:
- Moved #98 (ALLaM n300 increase_seq_len) and #99 (ALLaM t3000 increase_seq_len) from Backlog to Ready so n300/t3000 workers won't go idle after #88/#92.
- Ready now: #88 (Arcee n300 fix_correctness), #92 (Phi t3000 fix_correctness), #98 (ALLaM n300 increase_seq_len), #99 (ALLaM t3000 increase_seq_len).
- Backlog remaining: #100-#105 (Mistral/Falcon/Qwen3/Phi n300 + Mistral/Falcon t3000 increase_seq_len).

- Runners and reservations:
- IRD reservations still healthy with ~7h45m remaining on all three hosts (IDs 1-3).
- Verified active device work:
- `aus-wh-01` (agent1/n300) running `eval.py` for gemma n300 (issue #89) under TT_VISIBLE_DEVICES=0.
- `wh-04` (agent3/n150) and `wh-lb-43` (agent2/t3000) no longer had demo/eval python processes at the time of checking, suggesting they moved into code edits/LLM steps for #87/#91; no intervention taken.

- No PRs were opened during this heartbeat; waiting for the workers to finish and create PRs for review/merge.

## 2026-02-09T15:09:20+01:00

- In review triage/merges:
- Merged PR #108 (Fixes #91) for `google/gemma-3-4b-it` t3000 correctness; #91 is now closed and in Done.
- Corrected the MODELS.md Seq len entry to match the current KV cache cap (`MAX_CACHE_SEQ_LEN = 256`) before merging.

- Project `yieldthought/projects/6` status: In progress 4 (#87 Arcee n150, #88 Arcee n300, #89 gemma n300, #92 Phi t3000), Ready 2 (#98 ALLaM n300, #99 ALLaM t3000), In review 0, Backlog 8 (#100-#105, #109-#110).

- Runners and reservations:
- IRD reservations still healthy with ~6h53m remaining on IDs 1-3 and ~8h remaining on new ID 4 (`yyzc-wh-05`).
- Started a second n300 worker (agent4 on `yyzc-wh-05`, ID 4) with `tasks/increase_seq_len_n300.yaml` included; agent4 took #88 and can take #98 after.
- `wh-lb-43` (agent2/t3000) is actively running `eval.py` for Phi-3-mini t3000 (#92) at the time of checking.
- `aus-wh-01` (agent1/n300) has local edits in progress on branch `fix_google-gemma-3-4b-it-n300-functional` for #89 (model.py + logs modified; no PR yet).
- `wh-04` (agent3/n150) started #87 (Arcee-Spark n150) and is in early stages (no demo/eval python process at check).

- Follow-up:
- Once agent4 finishes #88, it should be able to take the Ready seq-len ticket #98 and then the n300 seq-len backlog (#100/#102/#104/#105/#110).

## 2026-02-09T15:39:20+01:00

- Project `yieldthought/projects/6` status: In review 0, In progress 4 (#87 Arcee n150, #88 Arcee n300, #89 gemma n300, #92 Phi t3000).
- Expanded the Ready queue so workers don't stall after the current correctness fixes:
- Moved to Ready: #100 (Mistral n300), #101 (Mistral t3000), #102 (Falcon n300), #103 (Falcon t3000), #104 (Qwen3 n300), #105 (Phi n300), #109 (gemma t3000).
- Created #111 (Phi t3000 increase_seq_len) in Backlog to run after #92 lands. Backlog now: #110 (gemma n300 increase_seq_len), #111.

- Runners and reservations:
- IRD reservations still healthy with >6h remaining on IDs 1-3 and >7h remaining on ID 4.
- `wh-04` (agent3/n150): actively running long eval for Arcee-Spark n150 and has local edits staged (model.py/logs/MODELS/bringup doc); issue #87 progress shows verifier success, awaiting commit/PR.
- `yyzc-wh-05` (agent4/n300): actively running long eval for Arcee-Spark n300 (#88).
- `wh-lb-43` (agent2/t3000): running short eval iterations for Phi-3-mini t3000 (#92) while editing model/logs.
- `aus-wh-01` (agent1/n300): editing gemma n300 (#89) with local changes; no PR yet.

## 2026-02-09T17:00:28+01:00

- In review triage/merges:
- Merged PR #112 (Fixes #87) for Arcee-Spark n150 correctness; resolved a MODELS.md conflict first. #87 is now closed.
- Merged PR #114 (Fixes #89) for gemma-3-4b-it n300 correctness; resolved a MODELS.md conflict first. #89 is now closed.
- Merged PR #115 (Fixes #99) for ALLaM t3000 to raise Seq len to 4096 with paged KV + paged attention; includes a small `demo.py` enhancement to accept `--max_seq_len`. #99 is now closed.

- Failed task requeue and hygiene:
- #88 (Arcee-Spark n300 fix_correctness) and #98 (ALLaM n300 increase_seq_len) both failed on `yyzc-wh-05` with device open/hugepage issues. Cleared stale `## Progress`, removed the `⨉` label, added a short host-failure note, and moved both back to Ready.
- #102 (Falcon n300 increase_seq_len) was left with `owner: agent4` after shutting down the broken runner; removed ownership and moved it back to Ready.

- Runner and reservation management:
- Stopped and released the problematic n300 reservation on `yyzc-wh-05` (agent4) after repeated hugepage pinning failures.
- Reserved a replacement n300 host (`aus-wh-10`) and started a new worker (still named `agent4`) there; it immediately took #88.
- Released the idle n150 reservation (`wh-04`) since no n150 work is currently queued.
- Current IRD reservations (all >4h remaining):
- ID 1: `aus-wh-01` (agent1/n300) working #100.
- ID 2: `wh-lb-43` (agent2/t3000) idle/ready to take the next t3000 seq-len tickets.
- ID 3: `aus-wh-10` (agent4/n300) working #88 (note: this was ID 4 before releasing the n150 reservation; IRD renumbered it).

- Project `yieldthought/projects/6` status at end of tick: In review 0, In progress 2 (#88, #100), Ready 9 (#98, #101-#105, #109-#111, #102), Backlog 0.

## 2026-02-09T17:03:28+01:00

- Project `yieldthought/projects/6` status: In review 0, In progress 3 (#88 Arcee n300, #100 Mistral n300, #101 Mistral t3000), Ready 8 (#98, #102-#105, #109-#111), Backlog 0.

- Runners and reservations:
- ID 1 `aus-wh-01` (agent1/n300): codex session active on #100; no long eval python process at check (still in code/LLM stages).
- ID 3 `aus-wh-10` (agent4/n300): actively running `eval.py` long eval for #88 at `--max_seq_len 32768` (high CPU process observed).
- ID 2 `wh-lb-43` (agent2/t3000): codex session active on #101; no long eval python process at check (still in code/LLM stages).
- All reservations still have >5h remaining (IDs 1-2) and >7h remaining on ID 3; no extension needed yet.

- Capacity note:
- Attempted to reserve a second t3000 (`wormhole_b0 --model lb`) to parallelize the remaining t3000 seq-len tickets:
- `tt_aus` allocation failed immediately.
- `tt_yyz` queued and was cancelled to avoid blocking.
- `tt_sjc` reservation succeeded on `wh-lb-81` but the container lacked `~/scripts/ttnn_models_setup.sh` and `/proj_sw/user_dev/moconnor/tt-metal` was empty, so the reservation was released.
- IRD note: the healthy-clusters check briefly returned no responsive clusters; `ird list/release --skip-clusters-check` worked around it.
- No PRs opened during this heartbeat; waiting for workers to finish and produce PRs for review/merge.

## 2026-02-09T18:05:17+01:00

- In review triage/merges:
- Merged PR #118 (Fixes #104) to raise Qwen/Qwen3-0.6B n300 max_seq_len to 40960 with paged KV cache + paged SDPA decode. Issue #104 is now closed and in Done.

- Project `yieldthought/projects/6` status: In review 0, In progress 1 (#88 Arcee-Spark n300 fix_correctness), Ready 7 (#98, #102-#103, #105, #109-#111), Backlog 0.

- Runners and reservations:
- Extended IRD timeouts for IDs 1-2 to restore the >4h buffer (IDs 1-2 now ~10h remaining).
- ID 3 `aus-wh-10` (agent4/n300): active `codexapi task` on #88 and currently running `python demo.py ...` (high CPU).
- ID 1 `aus-wh-01` (agent1/n300): worker loop running but stuck in repo sync retries due to GitHub `git pull` returning 500/502 (no active `codexapi task` process).
- ID 2 `wh-lb-43` (agent2/t3000): worker loop running but stuck in repo sync retries due to GitHub `git pull` returning 500 (no active `codexapi task` process).
- Cleared a stuck `owner: agent2` label on #103 (Falcon t3000 seq-len) with no active t3000 `codexapi task` process, and moved it back to Ready.
- Next: wait for #88 PR to open; once GitHub stabilizes for git pulls, workers should resume and take the remaining seq-len tickets.

## 2026-02-09T18:24:32+01:00

- In review triage/merges:
- #88 (Arcee-Spark n300 fix_correctness) was stuck with a local commit on the runner but no PR (agent4 `codexapi task` appeared hung). Manually pushed/created PRs, then replaced the messy/conflicting PR with a clean one:
  - Closed PR #119 (conflicting/superseded) and deleted its branch.
  - Merged PR #120 (Fixes #88) with only the intended changes: Arcee-Spark n300 Top-1 now 91% (>=90) and logs/docs updated.
- Removed the stale `owner: agent4` label from #88 and added the success marker `✓`. Item is in Done.

- Project `yieldthought/projects/6` status: In review 0, In progress 3 (#103 Falcon t3000, #105 Phi n300, #110 gemma n300), Ready 4 (#98, #102, #109, #111), Backlog 0.

- Runners and reservations:
- Extended IRD timeout for ID 3 (`aus-wh-10`) to the 10h max to restore the >4h buffer (IDs 1-3 now ~9-10h remaining).
- Marked #103 as In progress and re-added `owner: agent2` to reflect that agent2 is actively running demo/eval for it (GitHub status was briefly stale).
- Verified active workloads:
  - `aus-wh-01` agent1: running long eval for Phi n300 (#105) at `--max_seq_len 12288`.
  - `wh-lb-43` agent2: running long eval for Falcon t3000 (#103) at `--max_seq_len 32768`.
  - `aus-wh-10` agent4: active `codexapi task` and owns gemma n300 seq-len ticket (#110).

## 2026-02-09T18:41:34+01:00

- Repo/docs:
- Re-read all `tasks/*.yaml` plus `../codexapi/README.md` and `../gh-task/README.md` to confirm runner/task/ownership behavior.

- In review triage/merges:
- Confirmed PR #121 is merged and issue #103 is closed with `✓` (no open PRs at this time).

- Project `yieldthought/projects/6` status: In review 0, In progress 3 (#105 Phi n300, #110 gemma n300, #109 gemma t3000), Ready 3 (#98 ALLaM n300, #102 Falcon n300, #111 Phi t3000), Backlog 0.

- Queue hygiene:
- Removed stale `owner:` labels from Ready tickets (#98 had `owner: agent1`, #111 had `owner: agent2`) so runners can take them. #111 had also drifted into In progress without active work, so it was moved back to Ready.

- Runners and reservations (all ~9h remaining, no extension needed):
- ID 1 `aus-wh-01` (agent1/n300): `codexapi task` is alive but repeatedly failing repo sync (`git pull` returning GitHub 500/502); currently stalled (no active demo/eval process).
- ID 2 `wh-lb-43` (agent2/t3000): `codexapi task` has moved on to #109 (gemma t3000 seq-len) after completing #103; long run in progress.
- ID 3 `aus-wh-10` (agent4/n300): actively running `python demo.py models/google/gemma-3-4b-it/n300/functional/model.py --max_seq_len 40960` for #110.

- Local repo:
- `git pull --ff-only` to sync `main` to the Falcon t3000 seq-len merge; MODELS.md now reflects Falcon t3000 seq len 32768.
