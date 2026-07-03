---
name: agent-handoff
description: Compact the current conversation into a handoff document for another agent to pick up. Use when the conversation grows large, at task completion, or before context compaction. Branches: write-handoff (create a new handoff), read-handoff (load the latest).
argument-hint: "What will the next session be used for?"
disable-model-invocation: false
---

Write a handoff document summarising the current conversation so a fresh agent can continue the work.

## Leading words

- **Handoff** — The document itself, saved to `.handoffs/`. A portable summary of the session's essential state.
- **Compact** — Distill the conversation into the minimum viable context for the next agent. Remove redundancy, keep decisions.
- **Milestone** — A natural point to write a handoff: after a bug is fixed, a feature is added, or a test suite passes. Every task boundary is a milestone.

## Storage

Save to `.handoffs/` in the project root (resolve by walking up from `.agents/skills/agent-handoff/SKILL.md` to find `.handoffs/`). If that directory doesn't exist, fall back to the OS temporary directory.

Use a descriptive filename with a sub-second-granularity timestamp prefix:

- Preferred: `.handoffs/20260703-143022-build-rag-bundle.md`
  (`YYYYMMDD-HHMMSS-descriptive-slug.md`)
- For multiple handoffs in rapid succession, append a letter: `20260703-143022-a.md`, `20260703-143022-b.md`
  or vary the slug sufficiently to disambiguate.

## Content

Include a **suggested skills** section that names which skills the next agent should invoke and what branches/arguments to pass.

Do not duplicate content already captured in other artifacts (PRDs, plans, ADRs, issues, commits, diffs, README). Reference them by path or URL instead. Reserve the handoff for ephemeral context — what changed in the session, what was decided, what comes next.

Redact any sensitive information, such as API keys, passwords, or personally identifiable information.

If the user passed arguments, treat them as a description of what the next session will focus on and tailor the doc accordingly.

## Relationship to Compaction

Some agents have built-in context compaction (the ability to summarise earlier turns and keep working). Handoffs complement compaction, not replace it:

- If the agent **has** compaction, use the handoff as a persistent side-channel — write at key milestones so a future session (or different model) can pick up where this one left off, even if the current session continues.
- If the agent **does not** have compaction, the handoff is the primary handover mechanism. Write one before the conversation grows unwieldy, describing what was done and what the agent should start with next.

When in doubt, write the handoff. An extra one is cheap; a missing one loses the session's state.
