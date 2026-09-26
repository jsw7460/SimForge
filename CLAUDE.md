# SimForge / JaxRLWorld Development Guide

## Language

* Code, commits, and comments: English

## Response Format

* MUST: After modifying files, report the list of modified file paths at the end of the response.

## Code Style

* NEVER: Import inside functions. The only exception is circular imports.
* NEVER: Silent fallback. Crash immediately on failure. No `except: pass` or empty fallbacks.
* NEVER: Defensive coding for statically-known attributes. No `getattr(obj, "attr", default)` or `hasattr` checks when the attribute's presence is guaranteed by the type/class. Access directly and let `AttributeError` surface real bugs.
* MUST: Before removing a top-level import, grep to check whether it is re-exported.

## Architecture Principles

* NEVER: Add functionality not present in the original behavior. Refactoring must preserve behavior.
* MUST: Before removing dead code, grep for all callers exhaustively (including legacy presets).
* When unsure about simulator behavior, explore the `vendor/Genesis/`, `vendor/Newton/`, or `vendor/Mjlab/` directories directly to understand the source before implementing.

## Git / Workflow

* Write descriptive commit messages useful for future reference.
* MonoRepo: `JaxRLWorld/` lives inside `SimForge/` (git tracked); the simulators are git submodules under `vendor/` (`vendor/Genesis/`, `vendor/Newton/`, `vendor/Mjlab/`), pinned by gitlink. Move or rename a submodule only with `git mv`, which also rewrites `.gitmodules` and the submodule's worktree pointer.
* In `.gitignore`, other root-level directories must be root-relative (`/IsaacLab/`, `/third_party/`) to avoid macOS case-insensitivity issues.
* Keep commits focused and atomic — one logical change per commit.
* Reference related issues in commit messages when applicable.
* Do not include AI attribution or co-authorship lines (e.g., "Co-Authored-By: Claude...") in commit messages. Commits should represent human contributions without explicit AI attribution.
* Commit message format:
  * Separate subject from body with a blank line.
  * Subject: imperative mood, capitalized, ~50 chars, no trailing period.
    * Write as a command: "Fix bug" not "Fixed bug" or "Fixes bug".
    * Test: "If applied, this commit will [your subject]".
  * Body: wrap at 72 chars, explain what and why (not how — the diff shows that).
