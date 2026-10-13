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
- **Versioned carry record**: a small, independently mergeable file under
  `midstream/carries/<upstream-tag>/<jira-key>.yml` that records one carry's
  description, Jira issue, upstream references, compact Git replay recipe,
  validation contract, and retirement condition.
- **Static carry**: a narrowly scoped, Midstream-owned tooling, packaging, or
  release-integration delta under `midstream/carries/static/` that is replayed
  for every sync and does not require a component-owned Jira ledger issue.
- **Sync manifest**: the machine-readable snapshot of the immutable upstream
  and previous-downstream inputs plus the infrastructure replay source for one
  sync. It does not duplicate the carry inventory.

Location does not decide classification. A runtime change under a familiar
path is still a carry, and a new file can be a carry even when Git reports no
conflict. Midstream infrastructure is reviewed separately so expected
repository integration does not hide component carry debt.

## Carry acceptance policy

Every non-static carry pull request must have exactly one INFERENG Jira issue
serving as its ledger record. The Midstream PR and Jira issue must link to each
other after the PR is opened.
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

1. the Midstream repository and pull request, added after the PR is opened;
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

### Issue template

Create the issue as an INFERENG Story in the **INFERENG Midstream** component.
Use the summary `[Carry] <one-line description of the downstream behavior>`.
Set the component owner as assignee, add other accountable SMEs as
**Contributors**, and apply `omni-carry` plus `omni-carry-proposed`. Do not add
an active or sync-version label until approval is recorded.

Keep the description short enough for a component owner to maintain. Copy this
template and replace every prompt; when a field does not apply, state why.

```markdown
## Carry

- **Midstream PR:** <repository-qualified PR URL, or Pending until opened>
- **Target upstream release:** <tag>
- **Owner:** <name; must match assignee>
- **Additional SMEs:** <names; must match Contributors, or None>
- **Why the release is insufficient:** <one or two sentences>

## Upstream plan

- **Upstream reference:** <repository-qualified PR/commit URL, or None yet>
- **Status and plan:** <current disposition and the concrete upstream next step>

## Scope and risk

- **Behavior and bounded delta:** <what changes and the intended limits>
- **Risk:** <main failure or compatibility risks>

## Validation and approval

- **Focused validation:** <tests owned by the carry owner>
- **Release validation:** <required release-level checks>
- **Exception approval:** <Not required, Pending, or approver and evidence URL>

## Retirement condition

<The upstream release and evidence that will allow this carry to be removed.>
```

Link the Jira issue from the Midstream PR and add the PR link to Jira after the
PR is opened. PR creation is not blocked on knowing its eventual URL.
The structured assignee, Contributors, component, and labels remain the
queryable source of truth; repeating owner names in the description keeps the
record readable when it is viewed outside a Jira query.

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

Keep the Jira issue In Progress or Review while its Midstream carry PR is open.
Close it after the Midstream PR merges and the carry's focused
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

- **Proposal**: create the Jira issue and set `omni-carry` plus
  `omni-carry-proposed`; add the Midstream PR link after the PR is opened.
- **Approval for a sync**: record the decision and evidence, replace
  `omni-carry-proposed` with `omni-carry-active`, and add the sync-version
  label.
- **Replay**: record the compact Git replay identity in the versioned carry
  record and record the disposition and validation evidence in Jira.
- **Midstream merge**: after the carry PR merges and its focused validation is
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

## Carry records

Each non-static carry pull request adds one uniquely named
`midstream/carries/<upstream-tag>/<jira-key>.yml` file. The version directory
is the authoritative checked-in inventory for that sync; do not repeat its
members in the sync manifest. Different carry pull requests add different
files, so recording concurrent carries does not itself require their owners to
rebase.

Keep each record short enough to maintain by hand:

```yaml
schema: 1
jira: INFERENG-00001
description: One-line description of the retained Midstream behavior.
upstream:
  references:
    - https://github.com/vllm-project/vllm-omni/pull/000
replay:
  kind: commit
  commits:
    - 0000000000000000000000000000000000000000
validation:
  - focused behavior or artifact check owned by the carry owner
retire_when: >-
  A selected upstream release contains the required behavior and the focused
  validation passes without the downstream delta.
```

The upstream tag is encoded once in the directory name. Jira remains
authoritative for mutable owner, Contributors, approval, lifecycle, and
sync-version labels. Do not duplicate those fields or a Midstream PR URL in the
carry record. The PR can be discovered from Git history and remains linked from
Jira after it is opened.

Keep `replay` compact. Record the commit or bounded commit series that applies
the carry. For an adapted or synthetic merge, record its commit, base commit,
result tree, and source commits. These Git objects shortcut reconstruction but
are not blindly trusted instructions: verify that they resolve and produce the
expected delta or tree against the selected baseline. Do not manually list
affected paths or hunks; Git derives them from the recorded objects. Add an
optional dependency or override only when independent replay is insufficient.

The pull request, Jira issue, and versioned carry record must identify the same
carry. A record may merge before approval, but directory membership does not
make the carry active; Jira's approval evidence and labels remain the gate.

## Static carries

Put an always-replayed, Midstream-owned release invariant in
`midstream/carries/static/<slug>.yml`. Static carries do not require a Jira
issue, but each record must contain a one-line description, compact Git replay
recipe, validation contract, and retirement or review condition. Maintain them
through ordinary code review.

Use this category only for tooling, packaging, build, or release-integration
behavior owned by Midstream and expected in every sync. A component feature,
runtime behavior change, or delta needing an external owner or upstream plan
is a versioned Jira carry instead. Static is an ownership classification, not
a way to bypass the carry approval process.

```yaml
schema: 1
description: One-line description of the Midstream-owned release invariant.
replay:
  kind: commit
  commits:
    - 0000000000000000000000000000000000000000
validation:
  - focused tooling, packaging, or release-integration check
retire_when: >-
  The Midstream release process no longer requires this invariant.
```

## Per-sync replay manifest

Each sync commits `midstream/sync-manifests/<upstream-tag>.yaml`. It records the
immutable sync inputs and infrastructure replay source. It does not contain a
`carries` section: `midstream/carries/static/` plus the selected version
directory define the checked-in carry inventory without a second hand-edited
index.

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
  replay_commit: 0000000000000000000000000000000000000000
```

Replace the example values; do not commit placeholders. Replay each carry as
its own commit or bounded commit series on the clean sync branch. If a carry
must be adapted to the new baseline, review the adaptation as part of the sync
and record its resulting Git objects in that carry's versioned file, not the
stale source-branch ancestry.

Affected paths and hunks are computed from carry-record replay objects with
Git. Automation must detect paths touched by more than one replay and require
an explicit dependency, replay order, or reviewed override. The final audit
still compares the complete tree with the upstream baseline; deriving paths
does not weaken the rule that every non-infrastructure delta must map to one
approved versioned or static carry.

## Sync procedure

### 1. Record immutable inputs

Before editing the tree, record in the owning sync issue and draft manifest:

- upstream repository, selected tag, and resolved SHA;
- current downstream branch and resolved SHA;
- target sync version;
- owning sync Jira issue; and
- the previous active-carry query, proposed new carry issues, and target
  `midstream/carries/<upstream-tag>/` directory.

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

1. inspect its Jira issue, versioned carry record, Midstream PR, upstream
   references, and owner;
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
from the recorded downstream source, preferably as one bounded infrastructure
commit. Replay each approved carry as its own commit or bounded commit series
on the clean branch. Do not merge a carry PR branch when it includes stale
ancestry; replay only the approved delta and resolve dependencies explicitly.

After each replay, inspect the resulting commit and update that carry's
versioned record with its compact Git recipe. Git, not a hand-maintained YAML
list or sync-manifest index, supplies the affected paths and hunks.

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
accounted for by the Git diff of one recorded replay entry. Report overlapping
replays explicitly. An unexpected delta stops the sync: either add an approved,
owned carry through the Jira process or reconstruct the candidate without it.

### 5. Validate

Run:

1. each retained carry's focused tests;
2. packaging or artifact inspection required by packaging carries;
3. the required Midstream wheel, image, and accept-sync validation; and
4. the release's selected model-validation scope.

Record exact source SHAs, immutable artifact identities, CI run URLs, results,
and accepted exceptions. A dispatched or still-running job is not validation.

When validation exposes an adaptation problem, fix it on the clean branch,
update the replay identity in the carry record, rerun the affected checks, and
record the result in the carry issue. Do not solve the failure by importing
additional history from the old downstream branch.

### 6. Close the replay loop

Before merging the sync PR:

- every file in the version directory has one Jira issue, owner, upstream plan,
  approval, and validation evidence;
- every Jira issue approved for this sync appears exactly once in the version
  directory;
- every static record still satisfies the narrow Midstream-owned criteria;
- absorbed, replaced, rejected, and retired carries are not replayed;
- the final tree audit contains no unexplained delta;
- the owning sync issue links the baseline, manifest, carry query, sync PR,
  validation, and unresolved decisions; and
- owners are tagged in Jira for any action that remains theirs.

## Automation boundary

Automation may query Jira, validate PR metadata, compare the tree with the
versioned and static carry records, or report drift. It must enforce this
contract rather than silently decide ownership, approve a feature exception,
or infer carries from history.
The checked-in sync inputs, carry records, and human approval evidence remain
reviewable inputs; the automation mechanism may evolve independently.
