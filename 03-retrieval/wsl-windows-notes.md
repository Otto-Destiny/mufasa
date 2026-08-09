# Windows + WSL command notes

Practical notes for running this project's commands from Windows into the
`Ubuntu-22.04` WSL2 instance. Everything here was hit for real while building
[retrieval-v1](./retrieval-v1/) — written down so nobody has to rediscover it.

Milestone 1 setup itself ([milestone1.md](./milestone1.md#1-set-up-the-machine))
is Ubuntu-22.04 on WSL2. This doc is what came after that: how to actually run
commands into it reliably from a Windows terminal.

---

## TL;DR — the one rule that matters

**Never type a complex command directly after `wsl -d Ubuntu-22.04 --`.**
Quotes, spaces and `$variables` get mangled crossing from a Windows terminal
into WSL. Write the command to a `.sh` file first, then run that file:

```bash
# 1. Write your real command into a script file, e.g. do_thing.sh:
#!/bin/bash
cd ~/mufasa/llama.cpp
./build/bin/llama-cli -m ~/mufasa/models/gemma-3-270m-it-Q8_0.gguf \
  -p "Explain what rice husk ash is used for in concrete." \
  -n 128 -t 4 -c 2048 --no-warmup --single-turn --simple-io

# 2. Run the script file, not an inline command:
MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 -- bash "/mnt/c/path/to/do_thing.sh"
```

Two things make that second line work — both required, explained below:
`MSYS_NO_PATHCONV=1` and passing a **file path**, not an inline command.

A **simple** command with no quotes/spaces/variables (just a path or two) is
fine to run directly, no script file needed:

```bash
MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 -- python3 "/mnt/c/Users/hp/Desktop/Mufasa/03-retrieval/retrieval-v1/build_graph.py"
```

---

## Why this happens

Running `wsl -d Ubuntu-22.04 -- <command>` from a Windows terminal (Git Bash /
PowerShell) sends `<command>` through two layers before Linux ever sees it:
Windows' own command-line handling, then `wsl.exe`'s translation into the
Linux side. Quoting rules do not survive that trip cleanly. Confirmed
failures during this project:

- A `for p in cmake curl; do echo "$p"; done` loop: `$p` came out **empty**
  every time, even though the loop ran the right number of times.
- A prompt string with spaces, `-p "Explain what rice husk ash..."`, arrived
  at the program as **separate arguments** — verified with
  `cat /proc/<pid>/cmdline`, which showed `-p|Explain|what|rice|husk|ash|...`
  instead of one argument. This is what caused a hang once (see the
  `llama-cli` section below).
- A multi-line block with `SRC=...`, `DEST="..."`, `mkdir -p "$DEST"` failed
  with `mkdir: cannot create directory ''` — the quotes were stripped before
  reaching WSL.

None of this is a WSL bug to work around cleverly — it's inherent to piping a
quoted command through two different shells' parsing rules. Don't fight it;
route around it with a script file (below).

---

## The fix, step by step

**1. Write the real command to a `.sh` file** using a normal file-write tool
(not by echoing text through a shell — that reintroduces the same quoting
problem). Any path works; this project used the session scratchpad or the
repo folder directly.

**2. Run it with both of these together:**

```bash
MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 -- bash "/mnt/c/path/to/script.sh"
```

- `MSYS_NO_PATHCONV=1` — Git Bash's MSYS layer sees `/mnt/c/...` and thinks
  it's a POSIX path *on Windows*, and silently rewrites it to something like
  `C:/Program Files/Git/mnt/c/...` — which doesn't exist. This env var turns
  that rewriting off for this one command. Confirmed by reproducing the
  broken path in the error message before adding the flag.
- **A file path, not inline text** — sidesteps the quoting problem entirely,
  because there's nothing left to mis-parse. `bash script.sh` is one clean
  argument.

**3. If the script needs to `cd` into a repo path, use the WSL-mount form**,
not the Windows form:

| Windows path | WSL path |
|---|---|
| `C:\Users\hp\Desktop\Mufasa\...` | `/mnt/c/Users/hp/Desktop/Mufasa/...` |
| (WSL home directory) | `~` or `/home/<wsl-username>/...` |

---

## Running Python scripts (the cleanest option for anything non-trivial)

Once Python is actually running *inside* WSL, every `subprocess` call it
makes to another Linux program (like `llama-server`) is a normal Linux-to-Linux
call — no Windows/WSL boundary involved, no quoting problem at all. This is
why [retrieval-v1](./retrieval-v1/) is plain Python rather than bash: it was
more reliable, not just more convenient.

```bash
MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 -- python3 "/mnt/c/Users/hp/Desktop/Mufasa/03-retrieval/retrieval-v1/build_graph.py"
```

If the script needs a specific working directory (relative imports, etc.),
wrap it in a one-line `.sh` file using the same pattern as above:

```bash
#!/bin/bash
cd /mnt/c/Users/hp/Desktop/Mufasa/03-retrieval/retrieval-v1 && python3 evaluate.py
```

---

## Commands that need a password (`sudo ...`)

**Don't try to run these through an automated tool at all.** There's no
terminal attached for the password prompt, so it just hangs forever waiting
for input nobody can give it. Open the WSL terminal yourself
(`wsl -d Ubuntu-22.04`, or the "Ubuntu-22.04" Start menu shortcut) and run
`sudo apt install ...` etc. by hand.

---

## Long-running commands (builds, downloads)

Compiling `llama.cpp`, downloading models/PDFs, etc. can take minutes.
Run these in the background rather than blocking on them, and check the
output file rather than guessing at progress:

```bash
# object files compiled so far, vs. total source files, is a decent
# progress proxy for a cmake/make build:
find ~/mufasa/llama.cpp/build -name "*.o" | wc -l
```

`ps aux | grep cc1plus` (or whatever the relevant compiler/process is)
confirms something is actually still working, not hung.

---

## `llama.cpp` gotchas

### `llama-cli` hangs when run non-interactively

The current `llama-cli` is an interactive **chat** CLI. After it answers
once, it waits for another line of input. Run it from a script with no real
terminal attached (stdin closed), and it just loops printing empty `>`
prompts forever, spamming the log.

**Fix: always add both of these flags** when running `llama-cli` from a
script:

```bash
--single-turn --simple-io
```

- `--single-turn` — answer once, then exit, instead of looping.
- `--simple-io` — "use basic IO for better compatibility in subprocesses"
  (from `llama-cli --help`) — without it, output can come out empty even
  though the model did generate a reply.

### `llama-cli`'s output is unreliable to parse programmatically

Even with the flags above, `llama-cli`'s terminal UI echoes the prompt back
before answering, and **truncates long echoed lines inconsistently** — real
`(truncated)` markers and 1700+ character lines were found in raw output
during this project. Screen-scraping "just the model's reply" out of that
transcript is not reliable — build info banners and echoed prompt fragments
leaked into what should have been just the answer.

**Fix: use `llama-server` instead, over HTTP, for anything programmatic.**
It returns clean JSON with only the generated text — no transcript to parse.

```bash
# start once, in the background:
./build/bin/llama-server -m ~/mufasa/models/<model>.gguf -t 4 -c 2048 --no-warmup --port 8734

# then call it (this is what generate.py in retrieval-v1 does):
curl -s http://127.0.0.1:8734/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "your prompt here"}], "max_tokens": 128}'
```

Use `/v1/chat/completions` (not the lower-level `/completion`) — it applies
the model's chat template automatically, which matters a lot for
instruction-tuned models like Gemma. `/completion` only prepends `<bos>` and
sends your text raw, which measurably hurt output quality in testing.

Start the server **once** and reuse it across many calls — reloading the
model from disk for every single question is wasteful, and unnecessary
since the server just sits and waits for requests.

---

## Quick reference

| Task | Command |
|---|---|
| Check WSL distros installed | `wsl -l -v` |
| Open a shell in the right distro | `wsl -d Ubuntu-22.04` |
| Run a script file (safe for anything complex) | `MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 -- bash "/mnt/c/path/to/script.sh"` |
| Run a Python script directly (fine — no quoting risk) | `MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 -- python3 "/mnt/c/path/to/script.py"` |
| Check a package is installed | `wsl -d Ubuntu-22.04 -- dpkg -l <pkg>` (look for `ii` not `un`) |
| Anything needing a password | Run by hand in a real WSL terminal — never through a script |
