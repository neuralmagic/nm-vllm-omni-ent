# vLLM-Omni Release Operator Guide

This guide tells release participants what to do at each vLLM-Omni release
checkpoint, what evidence is required, and who owns the next action. It is an
operator guide, not a replacement for the RHAII code-freeze or z-stream
policies.

Use this guide for planned fast and stable releases. Apply the separately
approved z-stream process when maintaining an already shipped stable release.

Tracking issue: [INFERENG-10665](https://redhat.atlassian.net/browse/INFERENG-10665)

## Operating principles

1. **Dates come from the release source of truth.** Copy relevant dates into the
   release tracker, link their authoritative source, and name the person who
   confirmed them. A date copied into this guide is an example, not a calendar.
2. **Scope becomes progressively harder to change.** Planning Freeze fixes the
   intended versions, models, capabilities, and owners. The Midstream Tag
   deadline ends ordinary change acceptance. Final RC is for disposition, not
   feature development.
3. **Upstream first.** A functional change must normally be merged in the
   appropriate upstream branch before Midstream accepts it. Any exception must
   identify its approver, reason, forward-port owner, and closure condition.
4. **The component owner owns the change.** The engineer or subject-matter
   expert responsible for the affected functionality owns the upstream fix,
   backport PRs,
   reproduction, and technical evidence. Eligibility for a release does not
   make Midstream the implementation team.
5. **Midstream owns the release path.** Midstream owns intake, policy
   enforcement, sync/tag coordination, build orchestration, release evidence,
   and validation coordination proportional to risk.
6. **Every handoff produces evidence.** Record exact commits, tags, artifact
   digests, build and validation runs, accepted exceptions, and accountable
   owners in Jira.

## Roles and responsibilities

| Role | Responsibilities |
|---|---|
| Release Owner | Confirms schedule and scope, makes go/no-go decisions, approves late-change exceptions, and names a backup approver. |
| Component engineer / subject-matter expert | Owns implementation, upstream contribution, backports, reproduction, tests, risk analysis, and technical support during validation. |
| Midstream | Maintains the release tracker, enforces entry criteria, coordinates syncs and tags, drives Midstream builds, collects evidence, and coordinates downstream handoffs. |
| Validation owner | Defines and executes the agreed model/API/hardware matrix and records results against the exact candidate. |
| AIPCC / productization owner | Builds and publishes downstream wheels and images, records immutable artifact identity, and reports pipeline failures. |
| Product / stakeholder owner | Confirms supported scope and classifies requested models or capabilities as release-gating, best-effort, or out of scope. |

One person may hold more than one role. Every active release tracker must still
name the person holding each role; a team name by itself is not an escalation
path.

## Release checkpoints

### 1. Intake and schedule confirmation

**Stakeholders provide**

- Authoritative Planning Freeze, Midstream Tag / RPMs Due, Final RC, and release
  dates.
- Release type, support level, upstream vLLM and vLLM-Omni targets, and target
  downstream release branch.
- Named Release Owner, component subject-matter experts, validation owner, and
  productization contacts.

**Midstream does**

- Creates or updates the release epic and links the schedule source.
- Records assumptions as assumptions, with an owner and decision date.
- Verifies that repository, branch, and artifact naming are unambiguous.

**Exit criteria**

- Dates, release type, version targets, repositories, branches, owners, and
  escalation contacts are recorded in Jira.

### 2. Planning Freeze

**Stakeholders provide**

- Proposed versions, model list, API/capability scope, hardware matrix, and
  support classification.
- A responsible engineer or subject-matter expert for every item that still
  requires implementation.
- Dependencies on upstream, AIPCC, RHOAI, KServe, or other teams.

**Midstream does**

- Classifies each item as release-gating, best-effort, or out of scope.
- Reconciles the agreed model list with Midstream CI and the shared validation
  inventory.
- Identifies missing ownership or evidence before accepting work into the
  release plan.

**Exit criteria**

- Scope is documented as confirmed or explicitly provisional.
- Every provisional item has an owner and decision deadline.
- Missing implementation and validation work has a linked tracker.

### 3. Upstream-ready and change cutoff

Before the Midstream Tag deadline, the component engineer or subject-matter
expert must provide:

- Upstream PR and merge commit, or an approved exception explaining why
  upstream-first cannot be satisfied yet.
- Reproduction and verification procedure.
- Affected versions, models, APIs, configurations, and accelerators.
- Risk and rollback analysis, including dependency or packaging changes.
- Required backports and their relationship to newer supported versions.

Midstream verifies that the evidence is complete and that the change is in the
agreed scope. Midstream does not silently inherit implementation or backport
ownership when this evidence is missing.

**Exit criteria**

- Ordinary changes are merged upstream and ready for the selected Midstream
  sync or carry.
- Any exception has a named approver, owner, validation plan, and closure
  condition.

### 4. Midstream sync, build, and tag

**Component engineers / subject-matter experts do**

- Prepare and review required backports or carries.
- Resolve conflicts and document deviations from the upstream change.
- Remain available for build and functional failure analysis.

**Midstream does**

- Selects and records the exact source commits.
- Performs the Midstream sync and verifies version/dependency mappings.
- Runs the required wheel, image, and acceptance workflows.
- Creates the Midstream tag only from the reviewed release baseline.

**Required evidence**

- Upstream and Midstream commit SHAs.
- Midstream tag and workflow URLs.
- Wheel/image identity and required acceptance-test results.
- Known failures with owner and disposition.

### 5. Downstream productization and candidate creation

**AIPCC / productization owners do**

- Build the selected wheels and release image from the approved inputs.
- Report pipeline failures and publish the immutable candidate identity.

**Midstream does**

- Verifies versions, package contents, and provenance in the produced image.
- Connects downstream failures to the responsible component owner.
- Records any replacement candidate and why the previous candidate was
  superseded.

**Exit criteria**

- A candidate image exists with immutable digest, component versions, source
  references, and successful productization evidence.

### 6. Candidate validation

The validation owner runs the agreed release-gating matrix against the exact
candidate digest. Select tests according to the changed surface and release
scope, including as applicable:

- Model and API smoke tests.
- Accuracy or functional evaluation.
- Performance regression checks.
- Required accelerator and multi-GPU coverage.
- Packaging, security, and deployment-contract checks.

The component subject-matter expert owns diagnosis and remediation of failures
in their functionality. Midstream coordinates reruns and release disposition.

**Exit criteria**

- Every release-gating result is pass, accepted exception, or explicit no-go.
- Infrastructure failures are distinguished from product failures.
- Results identify the candidate digest and test configuration.

### 7. Final RC, release, and closeout

At Final RC, the Release Owner reviews the evidence and decides whether to
promote, replace, defer, or reject the candidate. After promotion:

- Record released image names and immutable digests.
- Record final component versions and source commits.
- Publish known limitations and accepted exceptions.
- Confirm stakeholder communication and support status.
- Create follow-up work for unresolved durable defects.
- Close release trackers only after publication and evidence are verified.

## Change acceptance guide

### Bug fix versus feature

A **bug fix** restores documented, supported behavior in the shipped or planned
release. A **feature** expands or changes the supported contract, including a
new model, endpoint, API shape, configuration option, accelerator, or previously
unavailable behavior. Diff size and a claim that a change is self-contained do
not change this classification.

### Before the Midstream Tag deadline

Planned changes may enter when they are within agreed scope, have an accountable
subject-matter expert, satisfy upstream-first, and include sufficient build and
validation time.

### After the Midstream Tag deadline

Ordinary changes wait for the next release. A late change is considered only
when it corrects a release-blocking build or validation failure, a qualifying
security issue, or another condition allowed by the approved code-freeze
policy. The exception must be recorded in Jira before merge.

### After a stable release ships

Use the approved z-stream process. The component subject-matter expert owns the
upstream and backport changes. Jira should show whether the fix exists in every
newer supported release before an older z-stream accepts it; unaffected versions
need an explicit disposition.

## Exception record

Every exception must record:

- Target release, repository, and branch.
- Bug, feature, or CVE classification.
- Release impact and available workaround.
- Requester, component subject-matter expert, Release Owner, and approver.
- Upstream PR/commit and forward-port status.
- Backport chain across supported releases.
- Risk, rollback plan, and validation plan.
- Approval decision and expiration or closure condition.

Slack can provide visibility, but the approval and evidence belong in Jira.

## Communication and escalation

| Event | Required communication |
|---|---|
| Schedule or scope confirmed | Update release tracker and notify named stakeholders. |
| Version, model, or capability changes | Record decision, owner, deadline, and CI impact before acceptance. |
| Midstream tag created | Post tag, commits, build runs, and known failures. |
| Downstream candidate created | Post immutable digest, component versions, and productization evidence. |
| Validation failure | Name failure owner, candidate impact, next action, and decision deadline. |
| Late-change request | Record Jira exception and obtain Release Owner approval before merge. |
| Final disposition | Record promotion/no-go decision, released artifacts, limitations, and follow-ups. |

Escalate missing implementation ownership to the responsible component lead,
missing release decisions to the Release Owner, validation capacity to the
validation owner, and productization failures to the downstream owner. Escalate
unresolved cross-team blockers through the release epic rather than allowing
them to remain only in chat.

## Release worksheet

Copy this section into the release epic or a release-specific document.

```text
Release name:
Release type and support status:
Release Owner / backup:
Planning Freeze:
Midstream Tag / RPMs Due:
Final RC:
Release date:
Schedule source and confirmer:

Upstream vLLM version/commit:
Upstream vLLM-Omni version/commit:
Midstream repositories/branches:
Downstream release branch:

Confirmed release-gating scope:
Best-effort scope:
Out-of-scope items:
Provisional items, owners, and decision dates:

Component subject-matter experts:
Validation owner and matrix:
Productization owner:
Escalation contacts:

Midstream tag and build evidence:
Candidate image and digest:
Validation evidence:
Accepted exceptions:
Final disposition:
```

## Working 3.6 GA example

The dates below come from the working example in INFERENG-10665. They remain
provisional until the release owner confirms them against the authoritative
schedule; do not treat this table as the source of truth.

| Milestone | Working date |
|---|---|
| Planning Freeze | September 18, 2026 |
| Midstream Tag / RPMs Due | October 13, 2026 |
| Final RC | November 4, 2026 |
| Release | November 19, 2026 |

[INFERENG-10709](https://redhat.atlassian.net/browse/INFERENG-10709) owns the
3.6 model-list decision and validation-inventory reconciliation.
[INFERENG-10710](https://redhat.atlassian.net/browse/INFERENG-10710) is the
release-planning epic. The release name does not change vLLM-Omni's support
status; Omni is expected to remain Tech Preview in the 3.6 GA release train.

## Related policy work

- [INFERENG-10664](https://redhat.atlassian.net/browse/INFERENG-10664) tracks
  review of the proposed Midstream code-freeze policy in
  [nm-vllm-ent PR #734](https://github.com/neuralmagic/nm-vllm-ent/pull/734).
- [INFERENG-10663](https://redhat.atlassian.net/browse/INFERENG-10663) tracks
  review of the proposed RHAII z-stream process.

These proposals are inputs, not silently adopted policy. Update this guide when
their approved versions establish authoritative exception or backport rules.
