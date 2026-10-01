# Fresh-slate v0.30.0 sync

**Status:** synced to upstream v0.30.0 final; review branch, not yet a build candidate.

## Base and comparison points

- Fresh upstream base: `v0.30.0` (`a8576ccb`) — final release, 2026-09-25.
  Upgraded from `v0.30.0rc1` (`59db6a4285e9086391efc85a6eff6030a6294878`);
  the rc1→v0.30.0 delta is 83 commits / 300 files (bugfixes, CI, model improvements).
- Previous named sync branch: `origin/sync-v0.26.0` (`4592f53a`).
- Current downstream head used for the carry inventory: `origin/main` (`386fb68a`).

## What this mockup reapplies

One Midstream-only commit restores the UBI CUDA/ROCm/CPU Dockerfiles,
`docker-bake.hcl`, the release and fork-sync workflows, and the `midstream/`
control-plane files.  These are build/release infrastructure, not component
runtime carries.

It also retains the `pyproject.toml` package-data rule (`"vllm_omni" =
["**/*"]`) for INFERENG-10371.  Upstream v0.30.0 still packages only a
narrow list of assets, so this remains an owned packaging carry until its
upstream replacement is verified in a built wheel.

`midstream/vllm-version` is updated to `v0.30.0`, but the
wheel index has **no** entry for it.  This branch therefore cannot be treated
as buildable yet: a base-vLLM wheel must be built and its real run ID added.
No placeholder run ID was invented.

## Carries deliberately not reapplied

| Area on current `main` | Fresh-slate disposition | Evidence / next check |
|---|---|---|
| Private Qwen3-Omni Realtime implementation (PR #46 family) | Do not reapply as a patch. | v0.30.0 has its own `/v1/realtime` WebSocket route and `realtime_connection.py`.  Validate the required Qwen3-Omni API contract before declaring behavioral parity. |
| Librosa/Step-Audio2 removal (PR #47) | Dropped. | Upstream commit `cfbf1392` is in the selected tag (upstream #6467). |
| PII redaction in TTS/audio logs (PR #43) | Dropped. | Upstream commit `07ceb332` is in the selected tag (upstream #6329). |
| OmniVoice ASR/instruction/long-form patch series | Do not replay wholesale. | v0.30.0 contains upstream OmniVoice evolution; upstream PR #5784 remains open, so compare the release-required scenarios rather than assume exact parity. |
| Validation hardening, DoS overflow, Qwen3-TTS Base voice-label, and speech-CI patches | Not reapplied. | These require targeted acceptance tests before any one is restored; none is silently retained by this mockup. |
| `fa3-fwd` removal (INFERENG-9470) | **Retained as a valid carry.** | The selected upstream tag still pins `fa3-fwd==0.0.3`; this branch reapplies the proven removal commit and its FA3 fallback/ring-attention compatibility changes. |

## Shape of the fresh-slate delta

Relative to upstream `v0.30.0`, the branch carries the original 14 midstream files
(13 build/release-control-plane files plus the package-data adjustment) and the
`fa3-fwd` removal.  The upstream rc1→v0.30.0 delta (83 commits) has been merged in
cleanly with no conflicts.

## Before turning this into a sync candidate

1. Add the real vLLM v0.30.0 wheel run ID after a successful base-wheel build.
2. Build and inspect the Omni wheel to prove package-data contents (INFERENG-10371).
3. Run the required Realtime, OmniVoice, TTS, and audio validation cases against
   this image; only then reapply a narrowly-owned failed behavior.
4. Build and test the retained INFERENG-9470 `fa3-fwd` removal carry against
   the v0.30.0 release, including its ring-attention compatibility coverage.
