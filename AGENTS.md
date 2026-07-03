# Snowflake Atlas — Agent Guide

This file tells AI agents how to work effectively on this project.

---

## Project in Brief

**Snowflake Atlas** is a local-first AI knowledge layer for Snowflake
documentation exposed via two MCP servers:

- `snowflake-fs` — deterministic, ripgrep-backed filesystem server (verbatim citations).
- `snowflake-rag` — semantic search over a precomputed embedding bundle.

The bundle is built once by the maintainer, distributed via GitHub
Releases, and consumed by end users with zero embedding work.

Language: Python 3.11+ · Package manager: `uv` · Test framework: `pytest`

---

## Skill Catalog

The project ships skills under `.agents/skills/`.  Most are available for
the agent to invoke autonomously; a few are **user-invoked** only and
must not be used unless the user explicitly names them.

| Skill | Invocation | Purpose |
|-------|------------|---------|
| `atlas-creation` | model-invoked | Build the dual MCP server system for any markdown corpus |
| `snowflake-docs-nav` | model-invoked | Navigate Snowflake's `llms.txt` documentation hierarchy |
| `web-browsing` | model-invoked | Search the web and extract page content |
| `code-review` | model-invoked | Review diffs, PRs, and code changes |
| `debugging` | model-invoked | Diagnose test failures, crashes, unexpected behaviour |
| `python-patterns` | model-invoked | Pythonic idioms, type hints, best practices |
| `python-testing` | model-invoked | pytest, TDD, fixtures, mocking, parametrization |
| `readme-writing` | model-invoked | Create, rewrite, or audit README / documentation |
| `agent-handoff` | model-invoked | Compact conversation into a handoff document |
| `skill-compliance` | model-invoked | Audit skill compliance across 6 dimensions |
| `skill-writing` | **user-invoked** | Reference for writing/editing skills. **Do not invoke unless the user explicitly names this skill.** |

### Convention: `disable-model-invocation: true`

Skills with this flag set are **user-invoked** — they are hidden from the
agent's automatic trigger matching.  The agent must **not** invoke them
unless the user explicitly says "use the skill-writing skill" or
equivalent.  As of this writing, `skill-writing` has this flag.

### Scanning for new skills

When you start a session, check `.agents/skills/` for any directories
that aren't listed above.  If you find one, read its `SKILL.md`
frontmatter and update this table.  This keeps the catalog accurate as
the project evolves.

---

## Session Rituals

Handoffs and compliance checks follow two deterministic rituals — one at
session start, one at task completion.  These rituals are **model-agnostic**
(they work whether or not the agent has context compaction) and
**editor-agnostic** (they work in Zed, opencode, Claude Desktop, Copilot,
or any MCP host).

Each ritual is structured as a sequence of steps.  Each step ends with a
**checkpoint**: a one-line summary the agent prints to chat.  The checkpoint
is the completion signal — if you see it, the step is done.  If a step
cannot be completed (tool not available, file not found, audit script
missing), the agent prints a warning and moves on.  Graceful degradation
is better than a ritual that silently skips work.

---

### Startup Ritual

**Completion criterion:** the agent has printed three checkpoints to chat:

1. `Startup — skills scanned: <count> tracked, <n> new`
2. `Startup — handoff loaded: <filename>` (or `no handoff found, starting fresh`)
3. `Startup — health: structure=<PASS|SKIP>, invocation=<PASS|SKIP>`

Perform these steps in order at the start of every session, **reading from
the filesystem**, not from in-context memory.  A compaction summary may
hint at the last handoff, but you must load the actual file from disk to get
the full, un-truncated content.

1. **Scan for new skills.**
   - List `.agents/skills/`.  Compare each directory against the Skill Catalog
     table in this file.
   - For any directory not in the table, read its `SKILL.md` frontmatter
     (`name`, `description`, `disable-model-invocation`) and add a row.
   - Print checkpoint 1.

2. **Load the latest handoff from disk.**
   - List `.handoffs/` sorted by filename.  Read the most recent file.
   - If the filename parses as `YYYYMMDD-HHMMSS-.*\.md`, use the timestamp
     to verify you have the latest.
   - If no handoff exists, start fresh — the project is new or handoffs were
     cleared.
   - Print checkpoint 2.

3. **Quick health check.**
   - If the `skill-compliance` skill is present and its check scripts exist,
     run `audit structure` and `audit invocation` (from the skill root).
   - If the skill or scripts are missing, skip with a note.  The ritual must
     not fail because a tool isn't available.
   - Print checkpoint 3.

---

### Milestone Ritual

**Completion criterion:** the agent has printed two checkpoints to chat:

1. `Milestone — handoff written: <filename>`
2. `Milestone — <check>=<PASS|SKIP> [<check>=<PASS|SKIP> ...]`

Context compaction is a model-level feature — different models (Claude,
GPT, Gemini) and agents (opencode, Claude Code, Copilot) handle it
differently, and the agent cannot reliably detect when compaction is about
to occur.  Therefore, instead of trying to hook into compaction, tie
handoff-writing to **task completion** — an event every agent can detect.

**When to fire:** after the last tool call of a task, before yielding back
to the user.  A "task" is any discrete unit of work: fixing a bug, writing
tests, adding a feature, updating documentation.  If in doubt, fire it — an
extra handoff is cheap, a missing one loses state.

Perform these steps in order:

1. **Write a handoff.**
   - Invoke the `agent-handoff` skill with a one-sentence description of
     what was accomplished.
   - The skill saves a timestamped file to `.handoffs/`.  Wait for the
     file to appear on disk.
   - Verify the file is non-empty (not truncated).  If it is empty or tiny,
     rewrite it — never leave a broken handoff.
   - Print checkpoint 1.

2. **Check affected skills.**
   - Did the task modify any file under `.agents/skills/<name>/`?  If not,
     skip this step and print nothing.
   - For each affected dimension (see table below), run the corresponding
     `audit <check>` script from the `skill-compliance` skill, if available.
   - If `skill-compliance` is missing or the specific check script doesn't
     exist, skip that check with a note.
   - Print checkpoint 2 (one line with all check results).

   | Change | Checks to run |
   |--------|---------------|
   | Added/removed a skill directory | `structure`, `invocation` |
   | Edited `disable-model-invocation` flag | `invocation` |
   | Edited `description` in frontmatter | `description` |
   | Added/removed phases or steps | `completion-criteria` |
   | Edited prose body | `leading-words` |
   | Any change to `agent-handoff/SKILL.md` or AGENTS.md handoff section | `handoff-compliance` |

3. **If the session is ending**, additionally run `audit-all` to produce
   a full compliance snapshot before the final handoff.

---

### Why "file from disk, not from memory"?

Compaction summaries are lossy — they preserve intent, not content.
Reading the handoff file from disk guarantees you have the full,
un-truncated document.  This is also what makes the ritual cross-editor:
no editor injects the handoff automatically; every agent must read the
file itself.

### Why "before yielding, not before compaction"?

Compaction is unpredictable.  Some models compact silently every few
thousand tokens; others never compact.  Relying on a "before compaction"
hook would mean this protocol works for some agents and silently fails
for others.  "Before yielding back to the user" is an observable event
every agent can detect, making it the reliable anchor.

---

## File Map

```
snowflake-atlas/
├── AGENTS.md                  ← This file
├── README.md                  ← User-facing documentation
├── pyproject.toml             ← Dependencies, entry points, tool config
├── .gitignore
│
├── atlas/                     ← Core package (the MCP servers + tooling)
│   ├── embed/                 ← Embedding backends (MLX, ONNX)
│   └── sources/               ← Source adapters (git, web-crawl, local)
│
├── snowflake_docs_nav/        ← Snowflake-specific doc crawler
│   └── crawler.py             ← llms.txt crawler with stealth engine
│
├── scripts/
│   └── publish-bundle.sh      ← Local build + GitHub Release
│
├── tests/                     ← pytest suite
├── data/                      ← Runtime data (gitignored)
└── .agents/
    └── skills/                ← Agent skills (see catalog above)
```

---

## MCP Server Entry Points

| Command | Module | Purpose |
|---------|--------|---------|
| `snowflake-fs` | `atlas.fs_server:main` | Filesystem MCP (ripgrep-backed) |
| `snowflake-rag` | `atlas.rag_server:main` | RAG MCP (vector search) |
| `snowflake-crawl` | `snowflake_docs_nav.crawler:main` | Crawl Snowflake docs |
| `atlas-build` | `atlas.make_bundle:main` | Build RAG bundle from source |
| `atlas-download` | `atlas.download:main` | Download bundle from Releases |
| `atlas-backup` | `atlas.backup:main` | Snapshot current bundle |
| `atlas-restore` | `atlas.restore:main` | Roll back to snapshot |
| `atlas-smoke` | `atlas.smoke_test:main` | End-to-end smoke test |
| `atlas-doctor` | `atlas.doctor:main` | Installation diagnosis |
| `atlas-evaluate` | `atlas.evaluate:main` | RAG quality evaluation |

## Development Workflow

1. **Sync & activate** — `uv sync` (add `--extra mlx` on Apple Silicon).
2. **Lint** — `uv run ruff check .`
3. **Test** — `uv run pytest tests/ -v`
4. **Smoke test** — `uv run atlas-smoke`
5. **Build bundle** — `uv run atlas-build` (maintainers only)
6. **Publish** — `scripts/publish-bundle.sh` (maintainers only)

---

## Design Tenets

- **Dual-server** — FS for verbatim citations, RAG for fuzzy discovery.
  Clients should use both: RAG to find candidates, FS to verify.
- **Portable bundle** — Built once, consumed everywhere.  Embedding
  model choice is recorded in the manifest; the inference backend is
  resolved at run time (MLX → ONNX+CUDA → ONNX+CPU).
- **No cloud dependencies** — Everything runs locally.  No API keys,
  no vector databases, no model training.
- **Stealth-first crawling** — The crawler defaults to polite behaviour
  but supports rotating User-Agents, jittered delays, and optional full
  browser automation for rate-limited sites.
