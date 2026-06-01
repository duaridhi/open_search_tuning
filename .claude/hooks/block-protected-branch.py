#!/usr/bin/env python3
"""PreToolUse(Bash) guard: block agent pushes/merges to the protected branch.

Enforces "human-only merge to main" (see memory: project-sdlc-workflow).
Denies, by emitting a PreToolUse deny decision on stdout:
  - git push whose destination ref is main/master (bare token or src:dst refspec)
  - bare `git push` while the current branch is main/master
  - git merge while currently on main/master (merging INTO main)
  - compound `git checkout/switch main ... && git merge ...`
Everything else exits 0 with no output (allow).
"""
import json
import shlex
import subprocess
import sys

PROTECTED = {"main", "master"}


def deny(reason: str) -> None:
    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        },
        sys.stdout,
    )
    sys.exit(0)


def is_protected(ref: str) -> bool:
    return ref.removeprefix("refs/heads/") in PROTECTED


def current_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)
    cmd = (data.get("tool_input") or {}).get("command") or ""
    if "git" not in cmd:
        sys.exit(0)

    try:
        toks = shlex.split(cmd)
    except ValueError:
        toks = cmd.split()

    cur = current_branch()
    has_merge = False
    checkout_protected = False
    n = len(toks)

    for i, t in enumerate(toks):
        if t == "push":
            explicit_dest = False
            for a in toks[i + 1:]:
                if a in ("|", "||", "&", "&&", ";"):
                    break
                if a.startswith("-"):
                    continue
                if ":" in a:  # src:dst refspec
                    explicit_dest = True
                    if is_protected(a.split(":", 1)[1]):
                        deny(f"Blocked: 'git push' to protected branch "
                             f"'{a.split(':', 1)[1]}' is human-only. Push your "
                             f"feature branch and open a PR instead.")
                elif is_protected(a):
                    explicit_dest = True
                    deny(f"Blocked: 'git push' to protected branch '{a}' is "
                         f"human-only. Push your feature branch and open a PR "
                         f"instead.")
            if not explicit_dest and cur in PROTECTED:
                deny(f"Blocked: current branch is '{cur}' (protected); a bare "
                     f"'git push' would update main. Switch to a feature "
                     f"branch — merging to main is human-only.")

        elif t == "merge":
            has_merge = True
            if cur in PROTECTED:
                deny(f"Blocked: 'git merge' while on '{cur}' merges into a "
                     f"protected branch. Open a PR and let a human merge to "
                     f"main.")

        elif t in ("checkout", "switch"):
            for a in toks[i + 1:n]:
                if a.startswith("-"):
                    continue
                if is_protected(a):
                    checkout_protected = True
                break

    if has_merge and checkout_protected:
        deny("Blocked: switching to a protected branch and merging is "
             "human-only. Open a PR and let a human merge to main.")

    sys.exit(0)


if __name__ == "__main__":
    main()
