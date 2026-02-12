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
- Local: attempted `git pull --ff-only` but it hung; killed the `git pull`/`git fetch` processes and left the working tree clean.
