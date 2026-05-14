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

Pushing a tag matching `omni-*` automatically triggers an **omni-wheel** build. You must provide `vllm_run_id` for this to work — use a manual dispatch instead if you don't have an existing vLLM wheel run.

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
