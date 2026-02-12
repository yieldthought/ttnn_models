## 2026-02-11 23:37:52 +0100
- Project status: 26 items in Ready, 0 in In review, 55 Done. All missing optimized rows have corresponding project issues; no new tickets needed.
- MODELS.md has 27 functional rows and only 2 optimized rows; 25 optimized rows remain to be produced to meet the release goal.
- Started three workers:
- n150: wh-03 (agent1) picked up optimize_model for models/arcee-ai/Arcee-Spark/n150/optimized.
- n300: aus-wh-09 (agent2) picked up optimize_model for models/humain-ai/ALLaM-7B-Instruct-preview/n300/optimized.
- t3000: wh-lb-44 (agent3) picked up optimize_model for models/arcee-ai/Arcee-Spark/t3000/optimized.
- Notes: had to stash a dirty repo on wh-03 before setup; npm ENOTEMPTY warnings during setup did not block worker launch; host key mismatch on wh-lb-44 resolved with ssh-keygen -R.
## 2026-02-12 00:11:21 +0100
- Project status: 54 Done, 3 In progress, 24 Ready. No items in In review.
- In progress tasks: #135 (ALLaM-7B n150 optimized, agent1), #136 (ALLaM-7B n300 optimized, agent2), #131 (Arcee-Spark t3000 optimized, agent3).
- Runners:
  - Released wh-03 (n150) after finding the container missing ~/scripts/ttnn_models_setup.sh and no codexapi in PATH.
  - Reserved aus-wh-08 (n150) and started agent1; log: /tmp/agent1.log.
  - Existing workers still running: aus-wh-09 (agent2, n300) and wh-lb-44 (agent3, t3000).
- Corrected project state after an over-broad codexapi reset: re-added owner labels for #136/#131 and moved them back to In progress.
## 2026-02-12 00:29:27 +0100
- Project status: 54 Done, 3 In progress, 24 Ready. No items in In review.
- In progress tasks remain: #135 (ALLaM-7B n150 optimized, agent1), #136 (ALLaM-7B n300 optimized, agent2), #131 (Arcee-Spark t3000 optimized, agent3).
- Runners healthy with >7h remaining: aus-wh-08 (agent1 n150), aus-wh-09 (agent2 n300), wh-lb-44 (agent3 t3000). All have active codex/codexapi processes; logs show tasks running but no new output yet.
- No new tickets required; continuing to wait on worker progress.
## 2026-02-12 01:01:55 +0100
- Project status: 55 Done, 2 In progress, 24 Ready. No items in In review.
- In progress: #135 (ALLaM-7B n150 optimized, agent1) and #136 (ALLaM-7B n300 optimized, agent2).
- #131 (Arcee-Spark t3000 optimized) is now Done with PR #158 merged; issue is closed. A separate PR #159 (run_eval JSON output mode) is open and mergeable but not tied to a project item.
- Runners healthy with >6h remaining: aus-wh-08 (agent1 n150), aus-wh-09 (agent2 n300), wh-lb-44 (agent3 t3000). Agent3 is running but currently holds no in-progress project item; will recheck if it fails to take the next t3000 task.
## 2026-02-12 01:31:17 +0100
- Project status: 55 Done, 2 In progress, 24 Ready. No items in In review.
- In progress: #135 (ALLaM-7B n150 optimized, agent1) and #136 (ALLaM-7B n300 optimized, agent2).
- Pulled main: Arcee-Spark t3000 optimized artifacts + MODELS update landed; scripts/run_eval.py and test updates landed (from merged work on #131).
- Runners: aus-wh-08 (agent1) and aus-wh-09 (agent2) still running codexapi tasks (~1-2h elapsed) with no new log output yet; wh-lb-44 worker alive and has restarted codexapi for t3000 tasks but has not yet claimed a new project item.
## 2026-02-12 01:59:52 +0100
- Project status: moved #131 (Arcee-Spark t3000 optimized) from In review -> Done, removed ⨉ label and added ✓ after confirming PR #158 merged and issue closed with acceptance evidence.
- Current board: 55 Done, 3 In progress (#135 n150, #136 n300, #132 t3000), 23 Ready.
- Runners: agent1/agent2 still running with no new log output; agent3 has picked up AFM-4.5B t3000 optimized after completing Arcee-Spark t3000.
## 2026-02-12 02:30:28 +0100
- Project status: 57 Done, 2 In progress (#135 n150, #130 n300), 22 Ready. No items in In review.
- Pulled main: AFM-4.5B t3000 optimized and ALLaM n300 optimized artifacts landed (MODELS + logs + model code).
- Runners: agent1 still on ALLaM n150; agent2 moved to Arcee-Spark n300; agent3 running t3000 loop but not holding a project item (AFM-4.5B t3000 already in Done).

## 2026-02-12 02:58

- Project status: Done 57, In progress 2, Ready 22. No In review items.
- In progress: #135 (ALLaM-7B-Instruct-preview n150 optimized, agent1), #130 (Arcee-Spark n300 optimized, agent2). Agent3 still on AFM-4.5B t3000 optimized.
- Runners: aus-wh-08 (n150), aus-wh-09 (n300), wh-lb-44 (t3000) all active with >4h remaining. Tailed logs; no new progress beyond task start lines.
- Actions: checked ird reservations, tailed worker logs, refreshed project board counts, pulled main. No merges or ticket edits.

## 2026-02-12 03:28

- Project status: Done 57, In progress 3, Ready 21. No In review after cleanup.
- In progress: #135 (ALLaM-7B-Instruct-preview n150 optimized, agent1), #130 (Arcee-Spark n300 optimized, agent2), #137 (ALLaM-7B-Instruct-preview t3000 optimized, agent3 newly started).
- Runners: aus-wh-08 (n150), aus-wh-09 (n300), wh-lb-44 (t3000) all active with >4h remaining. Tailed logs; only task start lines visible.
- Actions: moved #132 (AFM-4.5B t3000 optimized) from In review to Done, removed stale ⨉ label; verified MODELS row + logs exist. Refreshed project counts and pulled main.

## 2026-02-12 03:58

- Project status: Done 57, In progress 3, Ready 21. No In review items.
- In progress: #135 (ALLaM-7B-Instruct-preview n150 optimized, agent1), #130 (Arcee-Spark n300 optimized, agent2), #137 (ALLaM-7B-Instruct-preview t3000 optimized, agent3).
- Runners: aus-wh-08 (n150), aus-wh-09 (n300), wh-lb-44 (t3000) still active with >3.5h remaining; tailed logs, no new progress beyond task start lines.
- Actions: checked ird reservations, tailed worker logs, refreshed project counts, pulled main.

## 2026-02-12 03:58

- Project status: Done 57, In progress 3, Ready 21. No In review items.
- In progress: #135 (ALLaM-7B-Instruct-preview n150 optimized, agent1), #130 (Arcee-Spark n300 optimized, agent2), #137 (ALLaM-7B-Instruct-preview t3000 optimized, agent3).
- Runners: aus-wh-08 (n150), aus-wh-09 (n300), wh-lb-44 (t3000) still active with >3.5h remaining; tailed logs, no new progress beyond task start lines.
- Actions: checked ird reservations, tailed worker logs, refreshed project counts, pulled main.
- Extended all three IRD reservations to 6 hours to keep >4h remaining.

## 2026-02-12 04:28

- Project status: Done 57, In progress 3, Ready 21. No In review items.
- In progress: #135 (ALLaM-7B-Instruct-preview n150 optimized, agent1), #130 (Arcee-Spark n300 optimized, agent2), #137 (ALLaM-7B-Instruct-preview t3000 optimized, agent3).
- Runners: all three reservations healthy with >5h remaining after last extension. Tailed logs for aus-wh-08 and wh-lb-44; no new progress lines yet. SSH to aus-wh-09 failed with permission denied, so log tail for agent2 pending.
- Actions: checked ird reservations, tailed logs where possible, refreshed project counts, pulled main.

## 2026-02-12 04:58

- Project status: Done 57, In progress 3, Ready 21. No In review items.
- In progress: #135 (ALLaM-7B-Instruct-preview n150 optimized, agent1), #130 (Arcee-Spark n300 optimized, agent2), #137 (ALLaM-7B-Instruct-preview t3000 optimized, agent3).
- Runners: all three reservations still >5h remaining. Tailed logs for agent1 and agent3 with no new progress lines. SSH to aus-wh-09 (agent2) continues to return permission denied; will retry via ird connect-to or investigate auth next tick if it persists.
- Actions: checked ird reservations, attempted log tails, refreshed project counts, pulled main.

## 2026-02-12 05:28

- Project status: Done 57, In progress 3, Ready 21. No In review items.
- In progress: #135 (ALLaM-7B-Instruct-preview n150 optimized, agent1), #130 (Arcee-Spark n300 optimized, agent2), #137 (ALLaM-7B-Instruct-preview t3000 optimized, agent3).
- Runners: all three reservations healthy with >4.5h remaining. Direct SSH to aus-wh-09 still fails; used `script ... ird connect-to 1` to tail agent2 log successfully (no new progress lines). Agent1/agent3 logs unchanged.
- Actions: checked ird reservations, tailed logs (including via ird connect-to for agent2), refreshed project counts, pulled main.

## 2026-02-12 05:58

- Project status: Done 58, In progress 3, Ready 20. No In review items after merge.
- In progress: #135 (ALLaM-7B-Instruct-preview n150 optimized, agent1), #130 (Arcee-Spark n300 optimized, agent2), #140 (Llama-3.2-1B t3000 optimized, agent3 moved on after finishing ALLaM t3000).
- Merged PR #165 (ALLaM-7B-Instruct-preview t3000 optimized) and moved #137 to Done; MODELS row + logs now on main.
- Runners: all three reservations still >4h remaining. Tailed logs for agent1/agent3 (unchanged); agent2 log tailed via ird connect-to.
- Actions: checked ird reservations, reviewed/merged PR #165, updated project status, pulled main.

## 2026-02-12 06:28

- Project status: Done 58, In progress 3, Ready 20. No In review items.
- In progress: #135 (ALLaM-7B-Instruct-preview n150 optimized, agent1), #130 (Arcee-Spark n300 optimized, agent2), #140 (Llama-3.2-1B t3000 optimized, agent3).
- Runners: all three reservations still >3.5h remaining. Agent1/agent3 logs unchanged; agent2 log tailed via ird connect-to with no new progress lines.
- Actions: checked ird reservations, tailed logs (including via ird connect-to for agent2), refreshed project counts, pulled main.

## 2026-02-12 06:58

- Project status: Done 58, In progress 3, Ready 20. No In review items.
- In progress: #135 (ALLaM-7B-Instruct-preview n150 optimized, agent1), #130 (Arcee-Spark n300 optimized, agent2), #140 (Llama-3.2-1B t3000 optimized, agent3).
- Runners: extended all three reservations back to 6 hours (were ~3h remaining). Logs unchanged; agent2 log tailed via ird connect-to.
- Actions: checked/extended ird reservations, tailed logs, refreshed project counts, pulled main.

## 2026-02-12 07:28

- Project status: Done 59, In progress 3, Ready 19. No In review items.
- In progress: #135 (ALLaM-7B-Instruct-preview n150 optimized, agent1), #130 (Arcee-Spark n300 optimized, agent2), #143 (Mistral-7B-Instruct-v0.3 t3000 optimized, agent3).
- Merged PR #166 (Llama-3.2-1B t3000 optimized) and moved #140 to Done; MODELS row, logs, and model code are now on main. Removed owner label and added ✓.
- Runners: all three reservations still ~5.5h remaining. Agent1/agent3 logs show active tasks; agent2 log tailed via ird connect-to.
- Actions: checked ird reservations, tailed logs, reviewed/merged PR #166, updated project status, pulled main.

## 2026-02-12 07:58

- Project status: Done 59, In progress 3, Ready 19. No In review items.
- In progress: #135 (ALLaM-7B-Instruct-preview n150 optimized, agent1), #130 (Arcee-Spark n300 optimized, agent2), #143 (Mistral-7B-Instruct-v0.3 t3000 optimized, agent3).
- Runners: all three reservations still ~5h remaining. Logs unchanged; agent2 log tailed via ird connect-to; agent3 still on Mistral t3000.
- Actions: checked ird reservations, tailed logs, refreshed project counts, pulled main.

## 2026-02-12 08:28

- Project status: Done 60, In progress 3, Ready 18. No In review items after merge.
- In progress: #135 (ALLaM-7B-Instruct-preview n150 optimized, agent1), #130 (Arcee-Spark n300 optimized, agent2), #146 (Qwen3-0.6B t3000 optimized, agent3).
- Merged PR #167 (Mistral-7B-Instruct-v0.3 t3000 optimized) and #143 closed; MODELS row, logs, and model code now on main.
- Runners: all three reservations ~4.5h remaining. Agent3 log shows new Qwen3-0.6B t3000 task; agent2 log tailed via ird connect-to; agent1 unchanged.
- Actions: checked ird reservations, tailed logs, reviewed/merged PR #167, refreshed project counts, pulled main.

## 2026-02-12 08:58

- Project status: Done 60, In progress 3, Ready 18. No In review items.
- In progress: #135 (ALLaM-7B-Instruct-preview n150 optimized, agent1), #130 (Arcee-Spark n300 optimized, agent2), #146 (Qwen3-0.6B t3000 optimized, agent3).
- Runners: extended all three reservations back to 6 hours (were just under 4h). Agent1/agent3 logs unchanged; agent2 log tailed via ird connect-to.
- Actions: checked/extended ird reservations, tailed logs, refreshed project counts, pulled main.

## 2026-02-12 10:13

- Project status: Done 61, In progress 3, Ready 17. No In review items after merge.
- In progress (per worker logs): #135 (ALLaM-7B-Instruct-preview n150 optimized, agent1), #130 (Arcee-Spark n300 optimized, agent2), #149 (gemma-3-4b-it t3000 optimized, agent3).
- Reviewed PR #168 (Qwen3-0.6B t3000 optimized): MODELS.md adds optimized row (Top-1 98%, Top-5 100%, TTFT 59ms, 61.9 t/s/u, seq len 40960) with demo/eval logs and metrics.json; decode trace confirmed in code. Merged PR #168; issue #146 auto-closed and project item moved to Done.
- Runners: extended all three IRD reservations back to 6 hours (were ~4h40 remaining). Tailed logs on aus-wh-08/wh-lb-44; agent2 log tailed via ird connect-to (direct SSH still not usable).
- Actions: merged PR #168, pulled main, refreshed project counts, extended reservations, tailed worker logs.

## 2026-02-12 10:43

- Project status: Done 61, In progress 3, Ready 17. No In review items.
- Detected dead workers: agent1 (n150) and agent2 (n300) processes were not running on their reservations; only agent3 (t3000) was healthy.
- Recovery:
  - Restarted agent1 on aus-wh-08 (n150): `scripts/worker.sh agent1 n150` is running and `codexapi task` is active.
  - Released broken n300 reservation (aus-wh-09) which lacked `ttnn_models_setup.sh` and `codexapi`.
  - Reserved new n300 host (wh-03, tt_aus, wormhole_b0 --num-pcie-chips 2) and started agent2 there; `scripts/worker.sh agent2 n300` and `codexapi task` are active.
- Cleaned up stale project state: issues #135 (ALLaM n150 optimized) and #130 (Arcee-Spark n300 optimized) had very old `updatedAt` timestamps and no PRs; removed owner labels and moved both back to Ready so active workers can re-claim.
- Runners now: wh-lb-44 (t3000), aus-wh-08 (n150), wh-03 (n300), all with >5h remaining.

## 2026-02-12 11:13

- Project status: Done 61, In progress 3, Ready 17. No In review items.
- In progress: #129 (Arcee-Spark n150 optimized, owner: agent1), #139 (Llama-3.2-1B n300 optimized, owner: agent2), #149 (gemma-3-4b-it t3000 optimized, owner: agent3; PR #169 opened).
- Runners/reservations:
  - wh-lb-44 (t3000), aus-wh-08 (n150), wh-03 (n300); time remaining ~5h, ~5h, ~7.5h respectively.
  - All three `codexapi task` processes are running (agent3 ~1h34m, agent1 ~33m, agent2 ~28m).
- Extended reservations #1 (wh-lb-44) and #2 (aus-wh-08) back to 6 hours remaining to keep >4h headroom.
- Worker progress notes:
  - agent3 is running optimize_model for gemma-3-4b-it t3000; PR #169 exists and is mergeable but issue still In progress.
  - agent1 is actively working on Arcee-Spark n150 optimized; session log shows a demo failure due to tokenizer repo id being set to the model.py path (HFValidationError). It is rerunning demo after verifying the file path exists.
  - agent2 is actively working on Llama-3.2-1B n300 optimized; session log shows it created the n300/optimized directory, copied the t3000 model.py as a starting point, and is running demo/eval with `TTNN_ALLOW_SYSTEM_MESH_FALLBACK=1` (host reports disconnected topology; effective mesh falls back to 1x1). It also added MODEL_BRINGUP.md and metrics.json.
- Added a second n150 worker:
  - Reserved aus-wh-20 (selection ID 4) and launched `scripts/worker.sh agent4 n150`.
  - Setup initially failed `git pull` due to untracked `models/Qwen/Qwen3-0.6B/n150/functional/{demo,eval}.log`; removed them, fast-forwarded to origin/main, then restarted agent4 successfully.
- Local: attempted `git pull --ff-only` but it hung; killed the `git pull`/`git fetch` processes and left the working tree clean.

## 2026-02-12 11:43

- Project status: Done 62, In progress 3, Ready 16.
- Merged PR #169 (gemma-3-4b-it t3000 optimized): issue #149 auto-closed and project item moved to Done; MODELS.md now has t3000 optimized row (Top-1 91%, Top-5 100%, TTFT 78ms, 19.4 t/s/u, seq len 40960) with demo/eval logs, metrics.json, and traced decode in code.
- Closed PR #170 (Llama-3.2-1B n300 optimized): results were collected with `TTNN_ALLOW_SYSTEM_MESH_FALLBACK=1` causing a 1x1 mesh; added watcher note to #139, removed owner label, and moved #139 back to Ready for rerun on real n300 mesh (expected 1x2).
- Runners: wh-lb-44 (t3000 agent3 working on #152 Phi-3-mini-128k-instruct t3000 optimized), aus-wh-08 (n150 agent1 running demo for #129 Arcee-Spark n150 optimized), aus-wh-20 (n150 agent4 running eval for #135 ALLaM n150 optimized), wh-03 (n300 agent2 codexapi still running; now unblocked to pick up another Ready item). All reservations >5h remaining.
- Local: pulled main to merge commit `3954638` after merging PR #169.

## 2026-02-12 12:13

- Project status: Done 62, In progress 4, Ready 15.
- Found #139 incorrectly moved to In review with ✓ label and a new PR #172, but artifacts still used `TTNN_ALLOW_SYSTEM_MESH_FALLBACK=1` and the discovered mesh downgraded to 1x1 (per demo.log).
- Closed PR #172 and reset #139 back to Ready: removed ✓ label, removed stale `## Progress` section, kept watcher note requiring a real n300 mesh run (expected 1x2).
- Open PRs: #171 (ALLaM-7B-Instruct-preview n150 optimized) is open while issue #135 remains In progress.
- Runners/reservations: wh-lb-44 (t3000), aus-wh-08 (n150), wh-03 (n300), aus-wh-20 (n150); all reservations still >5h remaining (no timeout extensions needed this pass).

## 2026-02-12 13:41

- Project status: Done 64, In progress 4, Ready 13. No In review items after cleanup.
- Cleaned up stale In review: took #135 (ALLaM-7B-Instruct-preview n150 optimized), moved its project item to Done, and left a closing comment (issue was already closed; related PR was redundant/closed).
- In progress: #129 (Arcee-Spark n150, owner: agent1), #130 (Arcee-Spark n300, owner: agent2), #133 (AFM-4.5B n150, owner: agent4; PR #174 open), #155 (Falcon3-7B-Instruct t3000, owner: agent3; PR #175 open).
- Runners/reservations:
  - wh-lb-44 (t3000, running `agent3`), aus-wh-08 (n150, `agent1`), wh-03 (n300, `agent2`), aus-wh-20 (n150, `agent4`); all `codexapi task` processes running.
  - Time remaining: ~5:55, ~5:55, ~5:10, ~5:47 (no extensions needed).
- Worker notes:
  - agent3 log shows Phi-3-mini-128k-instruct t3000 optimized passed earlier and it is now running optimize_model for Falcon3-7B-Instruct t3000.
  - agent1/agent2/agent4 logs are quiet due to `--quiet`, but processes are alive and issues remain In progress.
- Local: fast-forward pulled main to `97e594a` (Phi-3-mini-128k-instruct t3000 optimized artifacts merged).

## 2026-02-12 14:03

- Project status: Done 65, In progress 3, Ready 13. No In review items.
- Completed: #133 (AFM-4.5B n150 optimized) is merged and in Done; metrics.json shows acceptance met (Top-1 98%, Top-5 100%, TTFT 57ms, t/s/u 19.6, seq len 65536, traced decode enabled). Removed stale `owner: agent4` label from the closed issue.
- In progress: #129 (Arcee-Spark n150), #130 (Arcee-Spark n300; PR #176 opened), #155 (Falcon3-7B-Instruct t3000; PR #175 open).
- Runners/reservations: all `codexapi task` processes running; time remaining ~5:32 (t3000), ~5:33 (n150), ~4:48 (n300), ~5:25 (n150).
- Local: fast-forward pulled main to `f0ea92a` (AFM-4.5B n150 optimized artifacts merged).

## 2026-02-12 14:33

- Project status: Done 66, In progress 3, Ready 12. No In review items after merges/cleanup.
- Cleaned up project state:
  - #133 (AFM-4.5B n150 optimized) had drifted back into In review; took/moved/released it back to Done.
- Reviewed and merged PR #175 (Falcon3-7B-Instruct t3000 optimized): acceptance met (Top-1 97%, Top-5 100%, TTFT 58ms vs 199ms functional, t/s/u 26.3 vs 7.3 functional, seq len 32768, traced decode present in model.py). Issue #155 closed and project item moved out of In review.
- Open PRs: #176 (Arcee-Spark n300 traced decode bringup) still open; issue #130 remains In progress.
- Runners/reservations:
  - Released t3000 reservation (wh-lb-44 / agent3) since there are no remaining Ready tasks matching `/t3000/`.
  - Remaining reservations: aus-wh-08 (n150), wh-03 (n300), aus-wh-20 (n150). Time remaining ~4:56, ~4:11, ~4:48 respectively (n300 will need extension once <4h).
- Local: fast-forward pulled main to include Falcon3 t3000 optimized artifacts (commit `f3070a1`).

## 2026-02-12 15:40

- Project status: In progress 1 (#129), Ready 13 (including #130 reset), In review 0.
- Reviewed PR #176 (Arcee-Spark n300 optimized): not mergeable because `demo.log`/`eval.log` were collected with `TTNN_ALLOW_SYSTEM_MESH_FALLBACK=1` and the discovered mesh downgrades to 1x1 (single-chip fallback). Left watcher comment on the PR and reset #130 back to Ready (removed ✓ label, added watcher note to rerun on real 2-chip mesh).
- Fixed drift: #138 (Llama-3.2-1B n150 optimized) was already closed but still in In progress and still had `owner: agent4`; moved it to Done and released the owner label.
- Runners/reservations:
  - Released the n300 reservation on wh-03 and reproed that it was an unconnected 2-chip topology (system_health showed all internal-trace links down).
  - Reserved a new n300 host on wh-04 with `wormhole_b0 --model x2`, confirmed internal-trace links UP via system_health, and started a fresh n300 worker (`agent3`).
  - Extended the remaining n150 reservations (aus-wh-08, aus-wh-20) to 6 hours to keep >4h headroom.

## 2026-02-12 15:53

- Reviewed and merged PR #179 (Arcee-Spark n150 optimized): acceptance met (Top-1 91%, Top-5 100%, TTFT 77ms vs 99ms functional, t/s/u 14.5 vs 13.9 functional, seq len 29952, traced decode implemented via `ttnn.begin_trace_capture`/`ttnn.execute_trace`). Issue #129 was already closed; moved its project item to Done and removed stale ownership.
- Project status: In progress 1 (#130), Ready 12, In review 0.
- Runners/reservations: aus-wh-08 (n150 agent1), aus-wh-20 (n150 agent4), wh-04 (n300 agent3). All reservations still have >5h remaining; agent3 is actively running Arcee-Spark n300 demo on a real 2-chip mesh (auto-discovery shows 2 logical/2 physical devices).

## 2026-02-12 16:01

- Restarted the n150 workers to clear idle state and align agent IDs with IRD selection IDs (agent1 on aus-wh-08, agent2 on aus-wh-20). Cleaned up an accidental double-worker start by killing the older worker/codexapi processes on each host.
- Reset #141 (Mistral-7B-Instruct-v0.3 n150 optimized) back to Ready after it was briefly claimed by a now-stopped agent4 (owner label drift).
- In progress now: #130 (n300 agent3), #144 (n150 agent2), #147 (n150 agent1).

## 2026-02-12 16:17

- Fixed project drift: #138 (Llama-3.2-1B n150 optimized) is closed but had drifted to In review; took/moved/released it back to Done.
- Verified Arcee-Spark n300 rerun is using a real 2-chip mesh on wh-04: `demo.log` reports mesh shape 2x1 and no `TTNN_ALLOW_SYSTEM_MESH_FALLBACK` in the command; demo metrics show TTFT 101ms and 16.0 t/s/u at seq len 32768 (functional baseline: 338ms, 5.0 t/s/u).
- Current state: In progress 3 (#130 n300, #144/#147 n150), In review 0, Ready 10. Open PRs: #176 only (waiting for updated eval artifacts).

## 2026-02-12 17:00

- Reviewed PR #176 (Arcee-Spark n300 optimized) after agent3 reran on a healthy discovered 2x1 mesh (no fallback): `metrics.json` shows Top-1 85%, Top-5 100%, TTFT 101ms, decode 16.0 t/s/u, seq len 32768, traced decode enabled.
  - PR branch had drifted and was unintentionally removing other MODELS.md rows; fixed MODELS.md in-PR to match main + add only the new n300 optimized row, then merged.
  - Issue #130 auto-closed by the PR; moved its project item to Done and released ownership.
- #144 (Qwen3-0.6B n150 optimized) had moved to In review but PR #180 was already merged and the issue was closed with ✓; moved the project item to Done and released ownership.
- Runners/reservations:
  - aus-wh-08 (n150 agent1) and aus-wh-20 (n150 agent2) extended to 6h to keep >4h headroom; wh-04 (n300 agent3) still has >6h remaining.
  - `codexapi task` still running on all three hosts; current In progress items are #147 (gemma-3-4b-it n150) and #141 (Mistral-7B n150).
- Local: pulled main to include merged PRs #176 and #180; open PRs now: #181 only.

## 2026-02-12 17:24

- Reviewed PR #181 (gemma-3-4b-it n150 optimized): fixed MODELS.md drift in-PR (it was missing newly merged rows) and merged; issue #147 auto-closed. Removed stale `owner: agent1` label and added ✓; project item already in Done.
- Fixed project drift:
  - #130 (Arcee-Spark n300 optimized) had drifted back into In review; took/moved/released it back to Done.
  - #86 (Phi-3-mini-128k-instruct n150 functional) was closed but had drifted into Ready; moved it to Done so workers don't pick it.
- Runners/reservations:
  - Extended wh-04 (n300) timeout; IRD capped at 10h and now shows ~10h remaining.
  - Current In progress: #139 (Llama-3.2-1B n300 optimized, owner: agent3), #141 (Mistral-7B n150 optimized, owner: agent2). Ready: #142/#145/#148/#150/#151/#153/#154.

## 2026-02-12 17:53

- Project status changes:
  - Drift cleanup: #147 (gemma-3-4b-it n150 optimized) is closed with ✓ but had drifted into In review; moved back to Done.
  - In progress: #139 (agent3, n300), #141 (agent2, n150), #142 (agent4, n300), #150 (agent1, n150).
  - Ready: #145/#148/#151/#153/#154.
- Runners/reservations:
  - Restarted agent1 on aus-wh-08 after it appeared idle; cleaned up an orphaned `codexapi task` process that caused agent1 to temporarily own both #150 and #153, and moved #153 back to Ready.
  - Reserved a second n300 host (wh-03) and started `agent4` to reduce n300 backlog. Initial start failed because the repo had diverged on main (ff-only pull failed); fixed by resetting the remote clone to `origin/main` (saved a backup branch) and relaunched the worker.

## 2026-02-12 18:02

- Reviewed and merged PR #182 (Mistral-7B-Instruct-v0.3 n150 optimized): acceptance met (Top-1 96%, Top-5 100%, TTFT 90ms vs 105ms functional, t/s/u 17.9 vs 16.5 functional, seq len 32768, traced decode enabled by default via `TTNN_USE_DECODE_TRACE=1`). Issue #141 auto-closed; removed stale `owner: agent2` label and added ✓.
- Current project state: In progress #139/#142/#150; Ready #145/#148/#151/#153/#154; no In review items.

## 2026-02-12 18:12

- Unstuck #150 (Phi-3-mini-128k-instruct n150 optimized): removed stale `owner: agent1` (agent1 worker on aus-wh-08 was dead), took/moved/released the issue back to Ready so a runner could pick it up again.
- Restarted agent1 worker on aus-wh-08 and verified `scripts/worker.sh agent1 n150` + `codexapi task` are running; agent1 reclaimed #150 and moved it back to In progress.
- Extended IRD timeouts for aus-wh-08 and aus-wh-20 to 8h to keep >4h headroom.
- Current project state: In progress #139/#142/#150; Ready #145/#148/#151/#153/#154; no In review items.

## 2026-02-12 18:54

- Investigated n300 runner instability for #142 (Mistral-7B n300): PR #183 explicitly reported internal-trace links DOWN and no valid 1x2 mesh on the worker host.
- Reprovision attempt: released `wh-03` and reserved `yyzc-wh-02` for `agent4`, then verified `system_health` again; internal-trace links were also DOWN there, so I shut that reservation down as well to avoid wasting cycles on invalid n300 topology.
- Cleaned stale project ownership after stopping `agent4`:
  - Reset #142 back to Ready (removed `owner: agent4`).
  - #145 and #148 had been briefly taken by `agent4` and drifted to In progress; reset both back to Ready and removed stale owner labels.
- Closed PR #183 as blocked-by-topology (issue remains Ready for rerun on a healthy n300 host).
- Current state: In progress #139/#150/#153; Ready #142/#145/#148/#151/#154; no In review. New PR #184 opened for #139 and pending watcher review once item reaches In review.

## 2026-02-12 19:29

- Reviewed and merged PR #184 (Llama-3.2-1B n300 optimized) after confirming acceptance artifacts on a real n300 mesh: Top-1 91%, Top-5 100%, TTFT 31ms, t/s/u 50.0, seq len 131072, traced decode enabled. Issue #139 auto-closed and project item moved to Done.
- Current project state: In progress #142/#150/#153; Ready #145/#148/#151/#154; In review 0.
- Open PRs: none.
- Runners/reservations:
  - n150: aus-wh-08 (agent1) and aus-wh-20 (agent2) both running `scripts/worker.sh` + `codexapi task` for active n150 work.
  - n300: wh-04 (agent3) running `scripts/worker.sh` + `codexapi task` for active n300 work.
  - No t3000 worker is running (no remaining t3000 Ready/In progress items).
  - Reservation headroom remains above the 4h floor (~6h42, ~6h43, ~7h54), so no extensions were needed this pass.

## 2026-02-12 19:59

- Reviewed and merged PR #185 (Falcon3-7B-Instruct n150 optimized) and PR #186 (Phi-3-mini-128k-instruct n150 optimized); both met release acceptance (quality, traced decode, TTFT/t/s/u improvements, no seq regression) and moved #153/#150 to Done.
- Project now: Done 76, In progress 1 (#142 n300), Ready 4 (#145/#148/#151/#154), In review 0, open PRs 0.
- Runner management:
  - Released idle n150 reservations (aus-wh-20 then aus-wh-08) because no n150 tasks remain.
  - Kept one healthy n300 runner on wh-04 (`agent3`) with >7h remaining.
  - Verified active work via process/session evidence: #142 is still running long eval on a real 1x2 n300 mesh (device open 1x2, eval process active).
- Assessment: release is now entirely n300-bound; remaining work is the five n300 optimized items in/behind #142.

## 2026-02-12 20:24

- Project status remains n300-only: In progress #142 (Mistral-7B-Instruct-v0.3 n300 optimized), Ready #145/#148/#151/#154, In review 0, open PRs 0.
- Runner management:
  - Verified single n300 reservation on wh-04 is healthy with ~6:58 remaining (well above 4h threshold); no timeout extension needed.
  - Confirmed worker process and `codexapi task` are running (`scripts/worker.sh agent3 n300`), and active session files continue updating in `~/.codex/sessions`, indicating ongoing progress on #142.
  - No n150/t3000 tasks remain, so no additional workers were started.
- Assessment: release completion is now fully gated on clearing the remaining n300 queue (5 items including #142).

## 2026-02-12 20:57

- Reviewed and merged PR #187 for #142 (Mistral-7B-Instruct-v0.3 n300 optimized): artifacts show real n300 mesh (`Mesh shape: 1x2`), long eval Top-1 97% / Top-5 100%, TTFT 44ms, decode 24.8 t/s/u, seq len 32768, and traced decode in `model.py`. Issue #142 is now closed and moved to Done.
- Current board state after merge: In progress #145 (Qwen3-0.6B n300 optimized), Ready #148/#151/#154, In review 0, open PRs 0.
- Runner management:
  - Single n300 reservation on wh-04 remains healthy with ~6h26m remaining (>4h threshold); no extension needed this pass.
  - Verified active worker/codex processes for `agent3` and confirmed live progress from session logs (currently running Qwen n300 baseline/demo measurements).
  - No n150/t3000 workers active since no remaining tasks for those types.
- Assessment: release finish remains n300-only with four items left (one active + three ready).

## 2026-02-12 21:24

- Reviewed and merged PR #188 (Qwen3-0.6B n300 optimized): acceptance met with real n300 mesh (`Mesh shape: 1x2`), long eval Top-1 99% / Top-5 100%, TTFT 54ms, decode 55.3 t/s/u, seq len 40960, and traced decode in `model.py`. Issue #145 auto-closed and moved to Done.
- Current project state: In progress #148 (gemma-3-4b-it n300 optimized, owner: agent3), Ready #151/#154, In review 0, open PRs 0.
- Runner management:
  - One n300 reservation on wh-04 remains healthy with ~5:58 remaining (>4h threshold), so no extension/restart required this pass.
  - Verified worker process health (`scripts/worker.sh agent3` + `codexapi task`) and ongoing activity; worker immediately picked up #148 after #145 merged.
  - No n150/t3000 tasks remain, so no additional workers were launched.
- Assessment: release completion now depends on three remaining n300 optimized items (#148 in progress, #151 and #154 ready).
