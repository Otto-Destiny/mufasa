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

## Milestone 1 setup, translated for a Windows terminal

[milestone1.md](./milestone1.md) gives every setup command as if you're
already typing inside a WSL/Ubuntu shell. If you're running from a Windows
terminal instead, here is each of those commands with its actual
Windows-invoked equivalent, task by task. Copy-paste these directly.

### Task 1 — set up the machine

The install commands use `sudo`, which needs a password prompt — per the
section below, **these cannot be run through a script from Windows at all.**
Open a real WSL shell first:

```bash
wsl -d Ubuntu-22.04
```

That drops you into an interactive Ubuntu shell. From there, run
milestone1.md's commands exactly as written:

```bash
sudo apt update
sudo apt install -y build-essential cmake git python3 python3-pip python3-venv \
                    lm-sensors time curl
sudo sensors-detect --auto
```

Once that's done, you can go back to driving things from Windows for
everything else. Checking the Python version, for example, has no
quotes/variables, so it's safe to run directly without a script file:

```bash
MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 -- python3 --version
```

### Task 2 — download the stand-in model

milestone1.md's download commands include quoted Python one-liners and
`hf download` calls — exactly the kind of thing that breaks crossing the
Windows/WSL boundary. Use a script file:

```bash
# download_model.sh
#!/bin/bash
pip install -U "huggingface_hub[cli]"
mkdir -p ~/mufasa/models && cd ~/mufasa/models

# list exact filenames before downloading (repos vary in capitalisation):
python3 -c "from huggingface_hub import list_repo_files; \
print('\n'.join(f for f in list_repo_files('bartowski/Nanbeige_Nanbeige4.2-3B-GGUF') if f.endswith('.gguf')))"

# then pull the exact filenames you found above:
hf download bartowski/Nanbeige_Nanbeige4.2-3B-GGUF <exact-filename>.gguf --local-dir .
hf download Qwen/Qwen3-1.7B-GGUF <exact-filename>.gguf --local-dir .
```

```bash
MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 -- bash "/mnt/c/path/to/download_model.sh"
```

The Gemma 3 270M stand-in is small enough to fetch directly with `wget` —
no variables or nested quotes, so this one's safe to run inline without a
script file:

```bash
MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 -- bash -c 'mkdir -p ~/mufasa/models && cd ~/mufasa/models && wget https://huggingface.co/ggml-org/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q8_0.gguf'
```

### Task 3 — install llama.cpp and run the model

Cloning and building is fine to run directly (no `.sh` file needed) — but
note the **single quotes** around the whole command. The build step takes
several minutes — see "Long-running commands" below for how to check
progress without blocking:

```bash
MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 -- bash -c 'cd ~/mufasa && git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && cmake -B build && cmake --build build --config Release -j$(nproc)'
```

That command contains `$(nproc)`. **Single quotes matter here**: with
double quotes, the *outer* Windows shell would try to expand `$(nproc)`
itself before the command ever reaches WSL — and `nproc` isn't even a
Windows command, so it would fail or expand to nothing. Single quotes stop
the outer shell from touching anything inside, so `$(nproc)` only gets
evaluated once, correctly, by the *inner* Linux shell. Same rule as the
`.sh`-file fix above, just in miniature: **when a command has any `$` in
it, wrap the whole thing in single quotes**, whether it's a whole script
file or one inline `bash -c '...'`.

Running the model, though, has a quoted multi-word prompt — use a script
file:

```bash
# test_model.sh
#!/bin/bash
cd ~/mufasa/llama.cpp
./build/bin/llama-cli \
  -m ~/mufasa/models/<your-model>.gguf \
  -p "Explain what rice husk ash is used for in concrete." \
  -n 128 -t 4 -c 2048 --no-warmup --single-turn --simple-io
```

```bash
MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 -- bash "/mnt/c/path/to/test_model.sh"
```

Note the two extra flags (`--single-turn --simple-io`) added to
milestone1.md's original example — without them this hangs when run from a
script. See "`llama.cpp` gotchas" below for why.

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
| Milestone 1's setup commands, Windows-ready | See "Milestone 1 setup, translated for a Windows terminal" above |
