# vLLM-Omni Release Change Policy

This policy protects the release validation window by defining when ordinary
changes stop and how exceptional late changes are approved. It applies to
planned vLLM-Omni releases.

## Release dates

The
[RHAI Release to Component Version Mapping](https://docs.google.com/spreadsheets/d/17GnTfbYP2nE36VF4LXIBVQeXRGh3fUJmqyNUnSFb9bQ/edit?usp=sharing)
spreadsheet is the source of truth for these milestones:

- Planning Freeze
- Midstream Tag
- Final RC
- Release

Do not copy release dates into this policy. Release tracking should link to the
spreadsheet so schedule changes are visible from the authoritative source.

## Change acceptance

| Release stage | Change-acceptance rule |
|---|---|
| Before Planning Freeze | Stakeholders agree on component versions, models, capabilities, validation scope, and owners. |
| Planning Freeze through Midstream Tag | Only changes within the agreed scope are accepted. Functional changes must normally be merged upstream and must have an accountable component owner and enough time for build and validation. |
| After Midstream Tag | Ordinary changes wait for the next release. Only release-blocking build or validation fixes, qualifying security fixes, or another explicitly approved exception may enter. |
| Final RC | The Release Owner promotes, replaces, defers, or rejects the candidate. Final RC is not a feature-development window. |

The component engineer or subject-matter expert owns implementation, upstream
contribution, backports, reproduction, and technical evidence. Midstream owns
intake, enforcement of this policy, build coordination, release evidence, and
validation coordination. The Release Owner makes go/no-go decisions and
approves or rejects late-change exceptions.

## Late-change exceptions

A change proposed after the Midstream Tag deadline must have a Jira exception
before it merges. The exception must record:

- The target release, repository, and branch.
- Why the change cannot wait and the release impact if it is rejected.
- The accountable component owner and requester.
- The upstream PR or commit, or why upstream-first cannot be met.
- The risk and the test plan or automated tests.
- The Release Owner's approval or rejection, recorded in Jira.

The exception, approval, and test information belong in Jira, even when
discussion happens elsewhere. Approval does not transfer implementation or
backport ownership to Midstream.

## Carries

A carry is a Midstream change that deviates from the upstream release baseline.
No net-new feature carries are accepted by default. An exception must have a
Jira issue and Release Owner approval before it merges, regardless of release
stage. Record:

- The reason for the carry and affected release branches.
- The accountable component owner.
- The related upstream PR and merge plan.
- The bounded patch to be carried and its risk.
- The test plan or automated tests.

Carries proposed after the Midstream Tag deadline must also satisfy the
late-change exception requirements above.
