# Upstream sync and carry replay

This document defines how `nm-vllm-omni-ent` is rebuilt from an immutable
`vllm-project/vllm-omni` release and how intentional downstream differences are
approved, recorded, replayed, validated, and retired.

The governing rule is simple: an old downstream tree is evidence, not input.
Each sync starts from the selected upstream tag and adds back only declared
Midstream infrastructure and approved carries. Never merge the old downstream
branch into the clean sync branch; additive files can survive such a merge
without a conflict or an explicit replay decision.

## Terms and ownership boundaries

- **Upstream baseline**: the selected upstream release tag and its immutable
  commit SHA.
- **Midstream infrastructure**: repository-specific build, packaging,
  release, and integration material that has no upstream equivalent or is
  intentionally owned by Midstream. Examples include the UBI Dockerfiles,
  `midstream/`, and the reviewed `.github/` surface.
- **Carry**: a source, runtime, dependency, packaging, example, or test change
  that causes the release tree or its behavior to differ from the selected
  upstream baseline.
- **Carry owner**: the component engineer or subject-matter expert accountable
  for the carry, its upstream plan, and its focused validation. Midstream
  coordinates the sync and enforces the gate; that does not transfer ownership
  of the implementation.
- **Replay manifest**: the checked-in, machine-readable snapshot of the exact
  infrastructure and carries restored for one sync.

Location does not decide classification. A runtime change under a familiar
path is still a carry, and a new file can be a carry even when Git reports no
conflict. Midstream infrastructure is reviewed separately so expected
repository integration does not hide component carry debt.

## Carry acceptance policy

Every carry pull request must have exactly one INFERENG Jira issue serving as
its ledger record. The downstream PR and Jira issue must link to each other.
A validation task may be linked separately, but it does not replace the carry
issue.

Existing downstream code is not implicitly approved. Before each sync, every
previously active and newly proposed carry is classified against the selected
upstream tree as one of:

- **absorbed upstream**: the selected tag contains the required behavior;
- **partially absorbed**: only a bounded residual delta remains necessary;
- **still required**: the approved carry remains necessary as recorded;
- **replaced**: another bounded downstream or upstream change supersedes it;
- **rejected**: it is not approved for the target sync.

Inspect the selected tree and required behavior, not only commit ancestry or
patch equivalence. Upstream may squash, rewrite, or supersede a change. A carry
whose upstream PR has merged is not retired until the selected tag contains an
acceptable replacement and the required validation proves it.

Net-new feature carries are rejected by default. An exception must satisfy the
release change policy proposed in
[nm-vllm-omni-ent PR #55](https://github.com/neuralmagic/nm-vllm-omni-ent/pull/55):
the Jira record must include an accountable component owner, upstream PR and
merge plan, bounded patch and risk, test plan or automated tests, and explicit
Release Owner approval before merge. Approval does not transfer implementation
or backport ownership to Midstream.

## Jira carry ledger

### Required issue data

The carry issue must record:

1. the downstream repository and pull request;
2. the target upstream sync version;
3. the accountable owner as Jira assignee;
4. additional SMEs in Jira's multi-user **Contributors** field;
5. the reason the selected upstream release is insufficient;
6. the repository-qualified upstream PR or commit, or an explicit upstream
   plan when no contribution exists yet;
7. the bounded patch, affected behavior, and risk;
8. focused tests and release-level validation requirements;
9. approval or rejection evidence when an exception is required; and
10. the condition that will allow the carry to be removed.

The reporter or sync coordinator may be a Midstream maintainer, but the
assignee is the component owner. A carry is not ready for replay while its
owner, upstream plan, validation, or retirement condition is missing.

### Labels

Use these queryable labels:

- `omni-carry`: identifies every carry ledger issue;
- exactly one lifecycle label:
  - `omni-carry-proposed`;
  - `omni-carry-active`;
  - `omni-carry-retired`;
- one label for every sync in which the carry was approved, using
  `omni-sync-vX-Y-Z`, for example `omni-sync-v0-30-0`.

Add the version label only after the carry is approved for that sync. Preserve
older version labels as history. Jira workflow status is not the carry
lifecycle: an implementation issue may be Closed while its delta still needs
to be replayed.

Keep the Jira issue In Progress or Review while its downstream carry PR is
open. Close it after the downstream PR merges and the carry's focused
validation evidence is recorded. The `omni-carry-active` and sync-version
labels remain after closure so future syncs can still discover the obligation.
Release-wide validation belongs to the owning sync issue and does not keep the
carry implementation issue open indefinitely.

To list everything approved for the previous `v0.30.0` sync:

```jql
project = INFERENG
AND component = "INFERENG Midstream"
AND labels = omni-carry
AND labels = omni-sync-v0-30-0
ORDER BY assignee, key
```

To list the current active inventory:

```jql
project = INFERENG
AND component = "INFERENG Midstream"
AND labels = omni-carry
AND labels = omni-carry-active
ORDER BY assignee, key
```

The sync PR must include the exact JQL query used and link its results or list
the returned Jira keys. This makes omissions reviewable.

### Lifecycle updates

- **Proposal**: create the Jira issue, set `omni-carry` and
  `omni-carry-proposed`, and link the downstream PR.
- **Approval for a sync**: record the decision and evidence, replace
  `omni-carry-proposed` with `omni-carry-active`, and add the sync-version
  label.
- **Replay**: record the exact replay identity and validation in both the sync
  manifest and Jira.
- **Downstream merge**: after the carry PR merges and its focused validation is
  recorded, close the Jira issue while preserving `omni-carry-active` and all
  sync-version labels.
- **Retirement**: record the selected upstream tag and validation proving the
  replacement, replace `omni-carry-active` with `omni-carry-retired`, and do
  not add the new sync-version label. A previously closed issue remains closed.
- **Rejection or withdrawal**: record the reason, close the Jira issue, and do
  not add an active or version label.

Update a closed carry issue with replay comments and new sync-version labels;
do not reopen it merely because another sync replays the same delta. Reopen it
only when the accountable owner has new implementation work to perform.

Read the issue back after updating fields or labels. A successful API response
alone is not proof that the ledger is queryable.

## Replay manifest

Each sync commits `midstream/sync-manifests/<upstream-tag>.yaml`. Jira remains
authoritative for mutable approval and ownership; the manifest freezes what a
particular source tree actually contains so the release can be reproduced even
after Jira changes.

The manifest must contain:

```yaml
schema: 1
sync_issue: INFERENG-00000
upstream:
  repository: vllm-project/vllm-omni
  tag: v0.00.0
  commit: 0000000000000000000000000000000000000000
previous_downstream:
  branch: main
  commit: 0000000000000000000000000000000000000000
infrastructure:
  source_commit: 0000000000000000000000000000000000000000
  paths:
    - midstream/
carries:
  - jira: INFERENG-00001
    owner: Example Owner
    downstream_pr: https://github.com/neuralmagic/nm-vllm-omni-ent/pull/000
    replay:
      kind: commit
      commits:
        - 0000000000000000000000000000000000000000
    upstream:
      disposition: still-required
      references:
        - https://github.com/vllm-project/vllm-omni/pull/000
    affected_paths:
      - vllm_omni/example.py
    validation:
      - focused test or immutable CI run
```

Replace the example values; do not commit placeholders. Record exact commit or
patch identities rather than a mutable branch name or PR head. If a carry must
be adapted to the new baseline, the adaptation is reviewed as part of the sync
and its resulting commit is recorded.

An allowed path is not sufficient proof by itself. The final audit must show
that the content or hunks under that path came from the declared
infrastructure restoration or carry replay.

## Sync procedure

### 1. Record immutable inputs

Before editing the tree, record in the owning sync issue and draft manifest:

- upstream repository, selected tag, and resolved SHA;
- current downstream branch and resolved SHA;
- target sync version;
- owning sync Jira issue; and
- the previous active-carry query and proposed new carry issues.

Fetch the selected tag directly and verify its object before constructing the
branch. For example:

```bash
SYNC_TAG=v0.00.0
UPSTREAM_REF="upstream-${SYNC_TAG}"

git fetch https://github.com/vllm-project/vllm-omni.git \
  "refs/tags/${SYNC_TAG}:refs/tags/${UPSTREAM_REF}"
git rev-parse "${UPSTREAM_REF}^{commit}"
```

Record the resolved commit; a tag name alone is not immutable evidence.

### 2. Inventory and reconcile

Produce two inventories:

1. Midstream-owned infrastructure to restore; and
2. runtime, dependency, packaging, example, and test differences that require
   carry disposition.

Query Jira for the prior active inventory and inspect the complete old
downstream tree against the new upstream baseline. The tree comparison is the
candidate superset; it finds forgotten additive files that a remembered commit
list will miss.

For every candidate carry:

1. inspect its Jira issue, downstream PR, upstream references, and owner;
2. compare the selected upstream implementation and behavior;
3. classify it as absorbed, partially absorbed, still required, replaced, or
   rejected;
4. obtain any required exception approval; and
5. update Jira before replaying it.

### 3. Construct the clean branch

Create the candidate directly from the verified upstream commit:

```bash
SYNC_BRANCH="sync-${SYNC_TAG}"
git switch --create "${SYNC_BRANCH}" "${UPSTREAM_REF}^{commit}"
```

Do not merge the old `main` branch. Restore only the exact infrastructure paths
listed in the manifest from the recorded downstream source commit. Replay each
approved carry from its recorded bounded commits or patch, in manifest order.
Do not merge a carry PR branch when it includes stale ancestry; replay only the
approved delta and resolve dependencies explicitly.

After each replay, inspect the staged tree and update the manifest with the
actual resulting identity, paths, and validation requirement.

### 4. Audit the complete tree

Publish both comparisons in the sync PR:

- selected baseline to previous downstream: the candidate superset;
- selected baseline to clean sync: the approved surviving set.

At minimum, capture:

```bash
git diff --name-status "${UPSTREAM_REF}^{commit}..HEAD"
git diff --stat "${UPSTREAM_REF}^{commit}..HEAD"
git diff --check "${UPSTREAM_REF}^{commit}..HEAD"
```

Review `.github/` against
[the Midstream ownership policy](.github-upstream-policy.md), not as an
ordinary upstream carry. Then prove that every other changed path and hunk is
accounted for by one manifest entry. An unexpected delta stops the sync: either
add an approved, owned carry through the Jira process or reconstruct the
candidate without it.

### 5. Validate

Run:

1. each retained carry's focused tests;
2. packaging or artifact inspection required by packaging carries;
3. the required Midstream wheel, image, and accept-sync validation; and
4. the release's selected model-validation scope.

Record exact source SHAs, immutable artifact identities, CI run URLs, results,
and accepted exceptions. A dispatched or still-running job is not validation.

When validation exposes an adaptation problem, fix it on the clean branch,
update the replay identity and manifest, rerun the affected checks, and record
the result in the carry issue. Do not solve the failure by importing additional
history from the old downstream branch.

### 6. Close the replay loop

Before merging the sync PR:

- every manifest carry has one Jira issue, owner, upstream plan, approval, and
  validation evidence;
- every Jira issue approved for this sync appears exactly once in the manifest;
- absorbed, replaced, rejected, and retired carries are not replayed;
- the final tree audit contains no unexplained delta;
- the owning sync issue links the baseline, manifest, carry query, sync PR,
  validation, and unresolved decisions; and
- owners are tagged in Jira for any action that remains theirs.

## Automation boundary

Automation may query Jira, validate PR metadata, compare the tree with the
manifest, or report drift. It must enforce this contract rather than silently
decide ownership, approve a feature exception, or infer carries from history.
The checked-in manifest and human approval evidence remain reviewable inputs;
the automation mechanism may evolve independently.
