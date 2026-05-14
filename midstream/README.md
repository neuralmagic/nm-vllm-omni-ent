# Midstream

This directory contains midstream-only content for `nm-vllm-omni-ent`. Nothing here exists in the upstream `vllm-project/vllm-omni` repository, so it is safe from upstream rebases and merges.

## Build Pipeline

Omni builds use a three-step pipeline that runs in [nm-cicd](https://github.com/neuralmagic/nm-cicd) (branch: `vllm-omni-build`):

```
Step 1: vLLM wheel       (nm-vllm-ent)         → build-whl.yml
Step 2: vllm-omni wheel  (nm-vllm-omni-ent)    → build-whl.yml  (needs step 1 run ID)
Step 3: Docker image      (nm-vllm-omni-ent)    → build-image.yml (needs step 1+2 run IDs)
```

Each step produces a run ID that feeds into the next step.

### Triggering Builds

Use the **Midstream Build** workflow from the [Actions tab](../../actions/workflows/midstream-build.yml):

1. **vllm-wheel** — builds the base vLLM wheel from nm-vllm-ent. Note the run ID from the summary.
2. **omni-wheel** — builds the vllm-omni wheel. Requires `vllm_run_id` from step 1.
3. **docker-image** — builds the container image. Requires both `vllm_run_id` and `omni_run_id`.

The workflow dispatches to nm-cicd and prints a clickable link to the triggered run in the job summary.

### Tag-Based Triggers

Pushing a tag matching `omni-*` automatically triggers an **omni-wheel** build. The workflow reads `midstream/vllm-version` and looks up the corresponding run ID from `midstream/vllm-wheels.yml` — no manual input needed.

```bash
# Example: tag a commit and push to trigger a build
git tag omni-v0.20.0-rc1
git push origin omni-v0.20.0-rc1
```

The tag name is freeform — use whatever makes sense: `omni-v0.20.0`, `omni-doug-feature-foo`, `omni-ricky-demo-2026-05-15`, etc. The vLLM wheel version is determined by the code at the tagged commit, not the tag name.

## vLLM Version Mapping

Two files control which vLLM wheel gets used:

### `midstream/vllm-version`

A single line declaring the vLLM version this omni code is built against:

```
v0.20.0
```

Update this when rebasing to a new upstream version.

### `midstream/vllm-wheels.yml`

Maps vLLM versions to known-good wheel build run IDs:

```yaml
v0.20.0:
  run_id: "25021945246"
  branch: "main"
  note: "reused from deepseek effort, built 2026-05-10"
```

**When to update:**
- **New vLLM version:** after rebasing upstream, update `vllm-version` and add a new entry to `vllm-wheels.yml` once you've built a wheel for it
- **New wheel for existing version:** update the `run_id` for that version entry

**How the workflow uses it:** when `vllm_run_id` is not provided as input (including all tag-push triggers), the workflow reads `vllm-version`, looks up the run ID from `vllm-wheels.yml`, and uses it automatically. If you provide `vllm_run_id` explicitly, it overrides the mapping.

### Default Runner Labels

| Step | Default Label |
|------|---------------|
| vllm-wheel | `k8s-a100-build-13-0` |
| omni-wheel | `k8s-a100-build-13-0` |
| docker-image | `ibm-wdc-k8s-h100-dind` |

Override via the `build_label` input if runner pools change.

### Manual CLI Alternative

If you have nm-cicd access, you can trigger builds directly:

```bash
# Step 1: vLLM wheel
gh workflow run build-whl.yml --repo neuralmagic/nm-cicd \
  --ref vllm-omni-build \
  -f repo=neuralmagic/nm-vllm-ent \
  -f branch=main \
  -f target_device=cuda \
  -f python=3.12.5 \
  -f build_label=k8s-a100-build-13-0 \
  -f timeout=120 \
  -f partitions_file=neuralmagic/configs/partitions/minimal.yml

# Step 2: vllm-omni wheel (use VLLM_RUN_ID from step 1)
gh workflow run build-whl.yml --repo neuralmagic/nm-cicd \
  --ref vllm-omni-build \
  -f repo=neuralmagic/nm-vllm-omni-ent \
  -f branch=main \
  -f target_device=cuda \
  -f python=3.12.5 \
  -f build_label=k8s-a100-build-13-0 \
  -f timeout=120 \
  -f vllm_run_id=<VLLM_RUN_ID> \
  -f partitions_file=neuralmagic/configs/partitions/minimal.yml

# Step 3: Docker image (use both run IDs)
gh workflow run build-image.yml --repo neuralmagic/nm-cicd \
  --ref vllm-omni-build \
  -f repo=neuralmagic/nm-vllm-omni-ent \
  -f branch=main \
  -f target_device=cuda \
  -f build_label=ibm-wdc-k8s-h100-dind \
  -f run_id=<OMNI_RUN_ID> \
  -f vllm_run_id=<VLLM_RUN_ID>
```

## Workflow Naming Convention

Workflows in `.github/workflows/` are a mix of upstream and midstream:

- **Upstream workflows** (e.g. `build_wheel.yml`, `pre-commit.yml`) — carried forward from `vllm-project/vllm-omni`
- **Midstream workflows** (prefixed `midstream-`) — added by us, never in upstream

See [.github-upstream-policy.md](.github-upstream-policy.md) for rebase guidelines.

## Secret: CICD_OMNI_PAT

The midstream build workflow uses an org-level secret `CICD_OMNI_PAT` — a fine-grained GitHub PAT with Actions (read/write) + Contents (read) permissions on `neuralmagic/nm-cicd`. This allows cross-repo workflow dispatch.
