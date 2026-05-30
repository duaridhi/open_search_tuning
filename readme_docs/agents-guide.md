# Claude Code Multi-Agent Guide

Quick reference for running parallel Claude Code sessions on this repo.

---

## Prerequisites

```bash
claude --version   # need ≥ 2.1.32 for agent teams
which tmux         # needed for split-pane teammate view
```

Install tmux on WSL if missing:
```bash
sudo apt install tmux
```

---

## What's configured in this repo

`.claude/settings.json` sets three things automatically when you open Claude here:

| Setting | Value | Effect |
|---|---|---|
| `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS` | `"1"` | Enables experimental agent teams |
| `teammateMode` | `"tmux"` | Each teammate gets its own tmux pane (falls back to in-process if no tmux session) |
| `worktree.baseRef` | `"head"` | New worktrees branch from your current HEAD, not the remote default |

**Directory restriction:** deny rules in `.claude/settings.json` block reads of `../`, `~/.ssh/`, `~/.aws/`, and `~/.gnupg/`. Agents stay within the project.

**Env files in worktrees:** `.worktreeinclude` copies `.env`, `dev.env`, and `prod.env` into every new worktree automatically.

---

## Approach 1 — Parallel sessions with worktrees (manual)

Run two independent Claude sessions, each in an isolated branch:

```bash
# Terminal 1 (or tmux pane 1)
claude --worktree feature-search-speed

# Terminal 2 (or tmux pane 2)
claude --worktree bugfix-pdf-url
```

Each session gets its own branch under `.claude/worktrees/` and can edit files without touching the other. When you exit, if there are uncommitted changes Claude prompts you to keep or discard the worktree.

Useful commands:
```bash
git worktree list               # see all active worktrees
git worktree remove <path>      # manual cleanup
```

Add `.claude/worktrees/` to `.gitignore` to keep them out of `git status`.

---

## Approach 2 — Agent view dashboard

`claude agents` opens a TUI listing every background session — state, last activity, and any that need your input.

```bash
claude agents                          # all sessions
claude agents --cwd ~/projects/cuad-ai-demo  # scoped to this repo
```

Dispatch a task from the bottom input and press Enter. Each prompt starts a new session. Background sessions run even after you close the terminal (managed by a supervisor process).

Key shortcuts:

| Key | Action |
|---|---|
| `Space` | Peek at output / send reply without attaching |
| `Enter` / `→` | Attach (full session) |
| `←` on empty prompt | Detach back to agent view |
| `Ctrl+X` twice | Stop + delete session |
| `?` | All shortcuts |

Start a session from your current shell and send it to the background:
```bash
claude --bg "investigate why /search latency spikes on long queries"
claude attach <id>    # come back to it later
claude logs <id>      # check output without attaching
claude stop <id>
```

---

## Approach 3 — Experimental agent teams

Agent teams let Claude coordinate multiple sessions with a shared task list and direct messaging between teammates.

**Start a team** — just describe the work:
```
Create an agent team to review the reranker latency hotspot.
Spawn 3 teammates: one profiling the current code, one researching
batch-reranking alternatives, one writing the test harness.
```

Claude spawns teammates, assigns tasks, and (with tmux active) opens each in its own pane.

**Teammate controls:**

| Action | How |
|---|---|
| Cycle teammates | `Shift+Down` |
| Message a teammate directly | Click their pane (split mode) or cycle to them and type |
| Require plan approval before a teammate edits | "Spawn an X teammate, require plan approval" |
| Shut down one teammate | "Ask the X teammate to shut down" |
| Clean up the team | "Clean up the team" (always via the lead) |

**Recommended team size:** 3–5 teammates. Each has its own context window — token cost scales linearly.

**Known limitations (experimental):**
- `/resume` and `/rewind` do not restore in-process teammates
- One team at a time per lead session
- Teammates cannot spawn sub-teams
- Split-pane mode requires tmux (not supported in VS Code integrated terminal)

---

## Approach 4 — Worktree-isolated subagents

The three write-capable agents in this repo (`cuad-ingest`, `search-perf`, `rag-eval`) have `isolation: worktree` in their frontmatter. When spawned as subagents, each automatically gets its own worktree, so parallel eval and ingestion runs never clobber each other.

Spawn them from a session naturally:
```
Use the search-perf agent to try swapping the reranker model.
Use the rag-eval agent to baseline the current search quality.
```

Both run in parallel in separate worktrees. When each finishes with no uncommitted changes, its worktree is removed automatically.

---

## Recommended tmux layout

```bash
# Start a named session
tmux new-session -s cuad

# Split into panes as needed
# Ctrl+B %   vertical split
# Ctrl+B "   horizontal split
# Ctrl+B o   cycle panes

# In one pane: agent view dashboard
claude agents

# In another: your main interactive session
claude
```

When you ask Claude to create an agent team inside a tmux session, teammates appear in new panes automatically (no manual splits needed).

---

## Quick-start: 3-teammate team

```bash
# 1. Open a tmux session
tmux new-session -s cuad-team

# 2. Start Claude
claude

# 3. Kick off a team
```
```
Create an agent team with 3 teammates to work on the search latency problem:
- Teammate 1 (profiler): profile qdrant_search_hf.py highlight_text() and measure
  per-sentence vs batched reranker call timings
- Teammate 2 (implementer): implement batched reranking using the search-perf agent
- Teammate 3 (tester): run the rag-eval agent to baseline quality before changes

Require plan approval from teammate 2 before they edit any files.
```

Teammates appear in split panes. Use `Shift+Down` to cycle if in-process mode.
When done: "Clean up the team."
