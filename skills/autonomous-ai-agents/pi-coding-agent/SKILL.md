---
name: pi-coding-agent
description: "Delegate coding to Pi CLI w/ local model (zero API cost)."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Coding-Agent, Pi, Earendil, Local-Model, LM-Studio, Qwen, MLX, Zero-Cost]
    related_skills: [claude-code, codex, opencode, hermes-agent]
---

# Pi Coding Agent — Hermes Orchestration Guide

Delegate coding tasks to [Pi](https://github.com/earendil-works/pi) (Mario Zechner's minimal coding agent, used by OpenClaw and Armin Ronacher). Pi has 4 built-in tools (Read / Write / Edit / Bash), the shortest system prompt in the space, and first-class support for OpenAI-compatible local model servers — making it the **right tool for cost-zero coding work via local models**.

## When to Use

- **Mechanical refactoring** — rename, restructure, add types — when an Opus call would be wasteful
- **Test scaffolding** — generate unit tests from existing code
- **Boilerplate generation** — config files, scaffolding, repetitive patterns
- **Local-model coding sessions** — your MacBook is awake, LM Studio is running, $0 marginal cost
- **Offline coding** — laptop on a plane, no internet, still want an agent
- **High-volume batch coding work** — when token cost matters more than per-task quality

Don't use Pi for:
- **Architecture / design / reasoning-heavy work** — use Mike directly with Opus, or `claude-code` skill
- **Production-critical code** where a model regression would be expensive — use Claude Code with Sonnet/Opus
- **Anything needing tools beyond Read/Write/Edit/Bash** — Pi is intentionally minimal, no MCP, no browser, no native subagents

## Hermes Stack Setup (current as of 2026-05-25)

| Component | Location | Notes |
|---|---|---|
| Pi CLI v0.75.5 | MacBook (`jon-eriks-macbook-pro`) | Installed via `npm install -g @earendil-works/pi-coding-agent` |
| LM Studio | MacBook | Started via `lms server start --port 1234`, model auto-loads on first request or `lms load <id>` |
| Model | Qwen3-Coder-30B-A3B-Instruct MLX 4-bit | 17GB on disk, ~16GB resident, 128K context, 16.4K max output, 4-slot parallel |
| Endpoint | `http://localhost:1234/v1` | OpenAI-compatible. From Beelink: route via `ssh jon-eriks-macbook-pro` (do NOT expose to Tailscale unless needed) |
| Pi config | `~/.pi/agent/models.json` on MacBook | Provider `lmstudio`, single model entry |

### Required `models.json` shape (already deployed)

```json
{
  "providers": {
    "lmstudio": {
      "baseUrl": "http://localhost:1234/v1",
      "api": "openai-completions",
      "apiKey": "lm-studio",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false
      },
      "models": [
        { "id": "qwen/qwen3-coder-30b" }
      ]
    }
  }
}
```

`compat.supportsDeveloperRole=false` is **required** — LM Studio's OpenAI server doesn't understand the `developer` role.

## Two Orchestration Modes

### Mode 1: Print Mode (`-p`) — PREFERRED for Hermes delegation

Non-interactive one-shot. Pi runs the task, returns stdout, exits.

```
terminal(command="ssh jon-eriks-macbook-pro 'export PATH=\"$HOME/.lmstudio/bin:/opt/homebrew/bin:$PATH\"; cd /tmp/work && pi --provider lmstudio --model qwen/qwen3-coder-30b -p \"Add type hints to all functions in utils.py\" --tools read,write,edit,bash'", timeout=300)
```

**Required flags for reliable delegation:**
- `--provider lmstudio` — pin the provider
- `--model qwen/qwen3-coder-30b` — pin the model (otherwise Pi may default elsewhere)
- `-p "<task>"` — print mode
- `--tools read,write,edit,bash` — explicit allowlist (Pi disables grep/find/ls by default; enable per task)

**Always set `workdir`** — Pi operates relative to cwd; without it, Pi works in your home directory.

### Mode 2: Interactive (TTY) — when you want to watch

Only useful when JE is at the keyboard. Hermes-delegated jobs should never need this. If you need to monitor, use `--mode json` for structured output and parse it.

## Pre-flight Checklist (run before delegating)

```bash
# 1. SSH alias resolves (config in ~/.ssh/config on Beelink defaults user=jon-erik)
ssh -o ConnectTimeout=5 jon-eriks-macbook-pro 'whoami'   # → jon-erik

# 2. LM Studio server up
ssh jon-eriks-macbook-pro 'curl -s http://localhost:1234/v1/models' | python3 -m json.tool

# 3. Model loaded (else first call loads it ~12s)
ssh jon-eriks-macbook-pro 'export PATH="$HOME/.lmstudio/bin:$PATH"; lms ps'

# 4. Pi sees the provider
ssh jon-eriks-macbook-pro 'export PATH="$HOME/.lmstudio/bin:/opt/homebrew/bin:$PATH"; pi --list-models'
```

If LM Studio isn't running:
```bash
ssh jon-eriks-macbook-pro 'export PATH="$HOME/.lmstudio/bin:$PATH"; open -ga "LM Studio"; sleep 5; lms server start --port 1234'
```

If the model isn't loaded:
```bash
ssh jon-eriks-macbook-pro 'export PATH="$HOME/.lmstudio/bin:$PATH"; lms load qwen/qwen3-coder-30b'
```

## `lms` CLI Cheatsheet (useful subcommands)

| Command | Purpose |
|---|---|
| `lms bootstrap` | One-time CLI install (adds `lms` to PATH) |
| `lms server start --port 1234` | Start OpenAI-compatible server |
| `lms server status` | Check if server is running |
| `lms ls` | List **all downloaded** models on disk |
| `lms ps` | List **currently loaded** (resident in RAM) models |
| `lms load <id>` | Load a model into memory (~12s for 30B) |
| `lms unload <id>` | Free RAM |
| `lms get <full-id> --mlx -y` | Download from LM Studio Hub. `-y` non-interactive, prefer exact id like `qwen/qwen3-coder-30b` (partial search terms drop you into a TUI picker) |

## CLI Flag Reference (most-used)

| Flag | Purpose |
|---|---|
| `-p, --print` | Non-interactive one-shot mode |
| `--provider <name>` | Provider name from `models.json` (e.g. `lmstudio`) |
| `--model <pattern>` | Model id or pattern (e.g. `qwen/qwen3-coder-30b`) |
| `--tools, -t <list>` | Comma-separated allowlist: `read,write,edit,bash,grep,find,ls` |
| `--no-tools, -nt` | Disable all tools (analysis-only mode) |
| `--no-builtin-tools, -nbt` | Disable Read/Write/Edit/Bash but keep extensions |
| `--system-prompt <text>` | Replace default system prompt |
| `--append-system-prompt <text\|file>` | Add to default system prompt (use this, not replace) |
| `--mode text\|json\|rpc` | Output format. `json` for parseable results |
| `--session <path\|id>` | Use a specific session file |
| `--no-session` | Ephemeral — don't save session |
| `--continue, -c` | Continue most recent session in cwd |
| `--fork <id>` | Branch an existing session (sessions-as-trees) |
| `--skill <path>` | Load Pi-format skill files (different format than Hermes skills) |
| `--no-skills, -ns` | Disable skill discovery |
| `--no-context-files, -nc` | Don't load AGENTS.md / CLAUDE.md |
| `--verbose` | Force verbose startup |
| `--offline` | Disable startup network calls |

## Built-in Tool Names

- `read` — read file contents
- `write` — create/overwrite files
- `edit` — find-and-replace in existing files
- `bash` — execute shell commands
- `grep` — content search (off by default, enable explicitly)
- `find` — file search (off by default)
- `ls` — directory listing (off by default)

## Hermes Delegation Recipes

### Recipe: One-shot refactor

```python
terminal(
    command='ssh jon-eriks-macbook-pro \'export PATH="$HOME/.lmstudio/bin:/opt/homebrew/bin:$PATH"; cd /path/to/repo && pi --provider lmstudio --model qwen/qwen3-coder-30b -p "Refactor auth.py to use dataclasses instead of dicts" --tools read,write,edit,bash --mode json\'',
    timeout=600
)
```

### Recipe: Test generation

```python
terminal(
    command='ssh jon-eriks-macbook-pro \'export PATH="$HOME/.lmstudio/bin:/opt/homebrew/bin:$PATH"; cd /path/to/repo && pi --provider lmstudio --model qwen/qwen3-coder-30b -p "Read src/parser.py and write pytest tests covering happy path, malformed input, and Unicode edge cases. Save to tests/test_parser.py." --tools read,write,bash --mode json\'',
    timeout=600
)
```

### Recipe: Analysis-only (no writes)

```python
terminal(
    command='ssh jon-eriks-macbook-pro \'export PATH="$HOME/.lmstudio/bin:/opt/homebrew/bin:$PATH"; cd /path/to/repo && pi --provider lmstudio --model qwen/qwen3-coder-30b -p "Review src/auth.py for security issues. Do not modify files." --tools read,grep,find,ls\'',
    timeout=300
)
```

### Recipe: Continue a session

```python
terminal(
    command='ssh jon-eriks-macbook-pro \'export PATH="$HOME/.lmstudio/bin:/opt/homebrew/bin:$PATH"; cd /path/to/repo && pi --provider lmstudio --model qwen/qwen3-coder-30b -c -p "Now add type hints to the new functions"\'',
    timeout=300
)
```

## Performance Baselines (M4 Max, 48GB, Qwen3-Coder-30B MLX-4bit)

- **Cold model load:** ~12s (one-time, then resident)
- **Simple task (FizzBuzz write + run):** ~10s end-to-end
- **File refactor (100 LOC, type hints):** ~30-60s
- **Test generation (200 LOC source):** ~60-120s

Speed scales with output token count, not input. Pi's minimal prompt means each turn is fast; the bottleneck is generation.

## When to Pick Pi vs Other Workers

| Task | Use |
|---|---|
| Mechanical refactor, scaffolding, boilerplate | **Pi** (free, fast enough) |
| Test generation | **Pi** if straightforward, **Claude Code** if requires deep understanding |
| Architecture / design / "what should this look like?" | **Mike** directly with Opus |
| Production bug fix in unfamiliar code | **Claude Code** with Sonnet/Opus |
| PR review (security, race conditions) | **Claude Code** with Opus |
| Multi-file feature, agentic | **Claude Code** or **Codex** |
| One-shot file edit with clear spec | **Pi** |
| Reading 50K LOC and synthesizing | **Mike** directly (Pi's minimal prompt struggles with big context unless you target Qwen3-Next-80B) |

## Common Pitfalls

1. **Wrong npm package.** `pi-coding-agent` (unscoped) on npm is a squatter at v0.0.1. The real package is `@earendil-works/pi-coding-agent`. Always install with the scope.

2. **PATH must include `$HOME/.lmstudio/bin`.** The `lms` and `pi` binaries are in different places; SSH non-interactive sessions don't load `.zshrc`. Always prepend `export PATH="$HOME/.lmstudio/bin:/opt/homebrew/bin:$PATH"`.

3. **LM Studio server doesn't survive reboots.** Either ask JE to leave LM Studio open, or use `open -ga "LM Studio"` + `lms server start` at the start of each session. Future TODO: launchd plist.

4. **`compat.supportsDeveloperRole: false` is mandatory for LM Studio.** Without it, Pi sends a `developer` role message LM Studio rejects with 400.

5. **Pi defaults provider to `google`.** If you don't pass `--provider lmstudio` it'll try to call Gemini and fail. Always pin provider AND model.

6. **`--tools` defaults exclude grep/find/ls.** For analysis-only tasks, explicitly enable them: `--tools read,grep,find,ls`.

7. **`rm -rf *` in zsh fails on empty dir** (`zsh:1: no matches found`). When constructing setup commands, use `rm -rf .` or check existence first.

8. **Pi sessions live in `~/.pi/agent/sessions/`** and accumulate. Use `--no-session` for ephemeral one-shots in cron / batch contexts.

9. **Pi's 4-tool minimalism is the point.** Don't try to bolt MCP onto it. If you need MCP, route through Claude Code or Codex instead.

10. **Model not loaded ≠ error.** Pi's first request will trigger LM Studio to load the model (~12s). Set timeout ≥ 60s for the first call after restart.

11. **Brew thinks LM Studio is installed but `.app` is missing → `brew reinstall --cask lm-studio`.** Old install records (pre-0.4.x) survive but the binary gets stranded. Symptom: `which lms` fails, `brew list --cask | grep lm-studio` succeeds. Fix moves `LM Studio.app` to `/Applications/` cleanly.

12. **Never `open -a "LM Studio"` over SSH without `-ga`.** Without `-g` (background) the LM Studio binary launches in the foreground with `--help` parsing, hangs at 110% CPU on the settings-file migration, and the SSH session blocks indefinitely. Always: `open -ga "LM Studio"`. If you find a stuck `LM Studio --help` process, kill it (`pkill -f "LM Studio"`), `touch ~/.lmstudio/settings.json`, then retry with `-ga`.

13. **`lms get <partial-name>` drops to interactive TUI picker.** SSH non-interactive sessions hang waiting for arrow-key input. Always pass the exact id (`qwen/qwen3-coder-30b`, not `qwen` or `qwen2.5-coder`) plus `-y`. If unsure of exact ids, run an interactive `lms get qwen --mlx` once on the Mac directly to browse, then reuse the exact id thereafter.

14. **Long-running `lms get` downloads need `nohup ... &` + `disown`.** Hermes terminal calls timeout at 60s; a 17GB model takes 5-10 min. Pattern: `nohup lms get <id> --mlx -y > /tmp/lms-download.log 2>&1 & disown` then poll the log file. Don't use Hermes `background=true` for the download command itself — the SSH connection multiplexing makes it flaky. Run the daemon-style invocation **on the Mac** via SSH.

15. **Always pin SSH user via `~/.ssh/config`, not Tailscale defaults.** Tailscale DNS resolves `jon-eriks-macbook-pro` but defaults the user to whoever's running the ssh client (`beelink@`), which fails. Add an explicit `Host` block with `User jon-erik`. Same trick for Aurelia. See pre-flight #1.

## Verification Checklist

After delegating to Pi, verify:

- [ ] Exit code 0 from the `ssh ... pi ...` terminal call
- [ ] Files claimed to be created actually exist (`ssh jon-eriks-macbook-pro 'ls -la <path>'`)
- [ ] If code was generated, it runs / lints / type-checks
- [ ] If tests were generated, they actually exercise the target code (not just `assert True`)
- [ ] No files were touched outside the intended `workdir`

Self-reports are not verified facts. Always read back the result.

## Future Work (not yet implemented)

- **Aurelia fallback** — vLLM on RTX 4070 Ti as a second endpoint; LiteLLM router on Beelink for failover
- **launchd plist** — auto-start LM Studio + server on MacBook login
- **Skill bridge** — translate Hermes skill descriptions into Pi `--skill` format so Pi can use Hermes skills
- **Cost telemetry** — track tokens/turns per Pi delegation; log to `vault-2026/agent-memory/patterns/`
