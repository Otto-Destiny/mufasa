"""
generate(prompt, server) -> str, backed by llama.cpp's server (llama-server)
running the Gemma 3 270M stand-in. This is the seam milestone1.md Task 6
asks for: local llama.cpp is the default and only implementation, and
switching to a hosted API (or later, the real MUFASA model) means changing
what LlamaServer wraps, not the call sites that use generate().

Why llama-server over the HTTP API instead of shelling out to llama-cli:
llama-cli's interactive chat UI echoes the input prompt back with its own
inconsistent line-truncation on long inputs, which makes reliably parsing
"just the model's reply" out of its terminal transcript impractical.
llama-server's /v1/chat/completions returns clean JSON with only the
generated text, and applies the model's chat template properly.

Never ships: the stand-in model is scaffolding to prove the retrieval ->
generation -> validation pipeline holds together, not a submission artifact.
"""
import json
import os
import subprocess
import time
import urllib.request

LLAMA_SERVER = os.path.expanduser("~/mufasa/llama.cpp/build/bin/llama-server")
MODEL = os.path.expanduser("~/mufasa/models/gemma-3-270m-it-Q8_0.gguf")


class LlamaServer:
    """Starts llama-server once; generate() reuses it across many calls
    instead of reloading the model from disk per question."""

    def __init__(self, port=8734, threads=4, ctx=2048):
        self.base_url = f"http://127.0.0.1:{port}"
        self.proc = subprocess.Popen(
            [
                LLAMA_SERVER, "-m", MODEL, "-t", str(threads), "-c", str(ctx),
                "--no-warmup", "--port", str(port),
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._wait_ready()

    def _wait_ready(self, timeout=30):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                urllib.request.urlopen(f"{self.base_url}/health", timeout=1)
                return
            except Exception:
                time.sleep(0.5)
        raise RuntimeError("llama-server did not become ready in time")

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def generate(prompt, server, max_tokens=350, timeout=120):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{server.base_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"].strip()
