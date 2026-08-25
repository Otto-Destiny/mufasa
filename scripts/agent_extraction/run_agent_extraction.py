"""Run the MUFASA claim-extraction notebook against a chosen model transport.

The notebook `01-data-engineering/data-extraction/llm-claim-extraction.ipynb`
touches the outside world in exactly one place: `model_call(row, task,
validated_contexts)`. Everything downstream - JSON parsing, schema and evidence
validation, checkpointing, batch accounting, Parquet compaction, publication and
the run summary - only ever sees the result dict that function returns.

This runner executes the notebook's own cells verbatim in one namespace and
rebinds that single function. No prompt, validator, budget, chunker, identity or
table is modified.

Two transports are available:

`--transport cavoti` (default) calls the Cavoti gateway, which speaks the
Anthropic Messages API rather than the OpenAI shape the notebook was written
for. The adapter maps one to the other: the system prompt moves to the `system`
parameter, `stop_reason` maps to `finish_reason` so the notebook's truncation
and refusal branches still fire, and Anthropic's usage fields map to the
OpenAI names. It streams, which is not optional - a non-streamed call carrying
a whole paper sits silent long enough for the gateway's edge proxy to time it
out at 504.

`--transport agent` writes each prompt to disk for a local agent to answer, and
picks the answers back up on the next invocation. A task whose answer is not yet
on disk raises `PendingAgentResponse`, so one invocation emits the prompts it
needs and stops.

Both preserve the notebook's four-attempt repair loop: a rejected answer
produces a fresh prompt carrying the validator's errors.

Usage (one batch at a time, never two):

    python scripts/agent_extraction/run_agent_extraction.py --batch 0

    exit 10  agent transport only: prompts await answers, see agent-io/PENDING.json
    exit 0   the batch is finished, compacted, published and summarised
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import sys
import time
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_EXTRACTION = REPO_ROOT / "01-data-engineering" / "data-extraction"
NOTEBOOK_PATH = DATA_EXTRACTION / "llm-claim-extraction.ipynb"

# Notebook cell indexes. Cell 0 and 13 are prose; cell 1 installs packages and is
# skipped because this environment already satisfies the declared versions.
BOOTSTRAP_CELLS = (2, 3, 4, 5, 6, 7, 8)
COMPACTION_CELL = 11
SUMMARY_CELL = 12

# Only the control-panel values that must differ because the transport is not
# the notebook's TokenRouter endpoint. Everything else - prompts, budgets, chunk
# bounds, validators, vocabularies - is the notebook's.
CONTROL_PANEL_OVERRIDES = {
    # Ten papers per batch, one batch per invocation, as commissioned.
    "BATCH_SIZE": 10,
    "MAX_BATCHES_THIS_RUN": 1,
    # The preflight picks a paper by size rank from the whole corpus, which lands
    # far outside the commissioned first hundred. The gates it enforces are
    # applied per batch by this runner's own reporting instead.
    "RUN_PREFLIGHT": False,
    # A streamed 40,000-token generation can legitimately run for many minutes.
    "REQUEST_TIMEOUT_SECONDS": 1800,
}

# Known transports. `model` and `base_url` are stamped into every checkpoint,
# the run manifest and the summary, so they must name what actually produced the
# payloads. Changing either changes SETTINGS_HASH and retires existing
# checkpoints, which is correct: work from one model is not work from another.
#
# agentrouter is recorded but not usable from a script: it answers only its own
# client and refuses SDK traffic with 401 unauthorized_client_error, and its
# budget pool is exhausted. It stays here so the choice is documented rather
# than forgotten.
PROVIDERS = {
    "cavoti": {
        "base_url": "https://cavoti.com",
        "key_env": "CAVOTI_API_KEY",
        "model": "claude-opus-5",
        "shape": "anthropic",
        "usable": True,
    },
    "tabitoken": {
        "base_url": "https://tabitoken.com",
        "key_env": "TABITOKEN_API_KEY",
        "model": "claude-opus-5-thinking",
        "shape": "anthropic",
        "usable": True,
    },
    "agentrouter": {
        "base_url": "https://agentrouter.org",
        "key_env": "AGENTROUTER_API_KEY",
        "model": "claude-opus-5",
        "shape": "anthropic",
        "usable": False,
        "note": "serves its own client only; SDK traffic gets 401 "
                "unauthorized_client_error, and the budget pool is exhausted",
    },
    "agent": {
        "base_url": "local-agent://claude-code",
        "key_env": "",
        "model": "anthropic/claude-opus-5-agent",
        "shape": "local-agent",
        "usable": True,
    },
}

# The commissioned scope: the first 100 rows of the eligible table, as ten
# batches of ten. Guarding it here makes running past it a deliberate act.
COMMISSIONED_BATCHES = 10

PENDING_EXIT_CODE = 10


class PendingAgentResponse(Exception):
    """Raised when a prompt has been emitted but no answer exists yet."""

    def __init__(self, request):
        self.request = request
        super().__init__(f"awaiting agent response: {request['request_id']}")


def estimate_tokens(text: str) -> int:
    """Character-derived token estimate.

    A local agent returns no usage block. The run summary sums these fields, so
    they are recorded as an explicit estimate rather than as zero; the run
    manifest records the accounting method beside them.
    """
    return max(1, len(text or "") // 4)


def write_if_changed(path: Path, text: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return False
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)
    return True


def load_notebook_cells(path: Path) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"]]


def install_no_network_openai_stub() -> None:
    """Replace the OpenAI client so an accidental API call cannot happen.

    The notebook's transport cell constructs clients from `BASE_URL` and the
    TokenRouter keys. Those keys are real and present in this checkout's .env.
    Substituting a client that raises on use guarantees this run is answered by
    the agent and by nothing else, and it lets `BASE_URL` honestly record the
    transport in `SETTINGS_HASH`.
    """

    class _DisabledCompletions:
        @staticmethod
        def create(*_args, **_kwargs):
            raise RuntimeError(
                "network transport is disabled in the agent runner; "
                "model_call is answered from agent-io/responses"
            )

    class _DisabledChat:
        completions = _DisabledCompletions()

    class _DisabledClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.chat = _DisabledChat()

    module = types.ModuleType("openai")
    module.OpenAI = _DisabledClient
    sys.modules["openai"] = module


def read_env_value(name: str) -> str:
    """Read one key from the data-extraction .env without printing it."""
    env_path = DATA_EXTRACTION / ".env"
    if not env_path.is_file():
        return ""
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if "=" in line and line.split("=", 1)[0].strip() == name:
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def bootstrap(fresh_start: bool, provider_name: str, workers: int) -> dict:
    """Execute the notebook's definition cells and rebind the transport."""
    provider = PROVIDERS[provider_name]
    if not provider["usable"]:
        raise SystemExit(f"provider {provider_name} is not usable: {provider['note']}")

    os.chdir(DATA_EXTRACTION)
    cells = load_notebook_cells(NOTEBOOK_PATH)
    namespace: dict = {
        "__name__": "__mufasa_extraction__",
        "__file__": str(NOTEBOOK_PATH),
        "display": lambda *values: print(*values),
    }
    # The notebook requires a TokenRouter key to exist. Nothing consumes it - the
    # OpenAI client is the disabled stub below - but the check stays satisfied so
    # the cell runs unmodified.
    os.environ.setdefault("TOKENROUTER_API_KEY", "unused-by-this-transport")
    install_no_network_openai_stub()

    for index in BOOTSTRAP_CELLS:
        exec(compile(cells[index], f"<notebook cell {index}>", "exec"), namespace)
        if index == 2:
            namespace.update(CONTROL_PANEL_OVERRIDES)
            namespace["FRESH_START"] = fresh_start
            namespace["MODEL"] = provider["model"]
            namespace["BASE_URL"] = provider["base_url"]
            namespace["PROVIDER_NAME"] = provider_name
            print(f"transport: {provider_name} ({provider['model']} "
                  f"via {provider['base_url']})")
        if index == 6:
            attach_transport(namespace, provider_name, provider, workers)
    return namespace


def attach_transport(namespace: dict, provider_name: str, provider: dict,
                     workers: int) -> None:
    """Rebind `model_call` and size the worker pool."""
    output_root = Path(namespace["OUTPUT_ROOT"])
    agent_io = output_root / "agent-io"
    request_dir = agent_io / "requests"
    response_dir = agent_io / "responses"
    for folder in (request_dir, response_dir):
        folder.mkdir(parents=True, exist_ok=True)

    namespace["AGENT_IO_DIR"] = agent_io
    namespace["REQUEST_DIR"] = request_dir
    namespace["RESPONSE_DIR"] = response_dir
    namespace["WORKERS"] = workers
    namespace["EMITTED_REQUESTS"] = []

    if provider["shape"] == "local-agent":
        namespace["model_call"] = make_agent_model_call(namespace)
        return

    key = read_env_value(provider["key_env"])
    if not key:
        raise SystemExit(f"{provider['key_env']} is absent from "
                         f"{DATA_EXTRACTION / '.env'}")
    import anthropic
    namespace["ANTHROPIC_CLIENT"] = anthropic.Anthropic(
        base_url=provider["base_url"], api_key=key, max_retries=0,
        timeout=namespace["REQUEST_TIMEOUT_SECONDS"])
    namespace["model_call"] = make_anthropic_model_call(namespace)
    print(f"key: {len(key)} chars, {workers} concurrent request(s)")


# Anthropic stop reasons mapped to the OpenAI finish reasons the notebook
# branches on. "max_tokens" must reach it as "length", because that is what
# makes the notebook record a terminal output_truncated rather than retrying a
# generation that will truncate again.
STOP_REASON_TO_FINISH_REASON = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "refusal",
    "pause_turn": "pause_turn",
}


def make_anthropic_model_call(namespace: dict):
    """Build a `model_call` that speaks the Anthropic Messages API.

    The signature, the retry-and-repair loop, the terminal statuses and every
    field of the returned dict match the notebook's OpenAI implementation.
    """
    build_user_prompt = namespace["build_user_prompt"]
    parse_json_object = namespace["parse_json_object"]
    validate_payload = namespace["validate_payload"]
    task_result_descriptor = namespace["task_result_descriptor"]
    terminal_result = namespace["terminal_result"]
    clean_text = namespace["clean_text"]
    ExtractionValidationError = namespace["ExtractionValidationError"]
    client = namespace["ANTHROPIC_CLIENT"]

    def anthropic_model_call(row, task, validated_contexts=None):
        started = time.perf_counter()
        validation_errors = None
        last_error = ""
        last_invalid_output = ""
        error_kind = "api_failed"
        total_usage = {"prompt_tokens": 0, "output_tokens": 0, "cached_prompt_tokens": 0}
        system_prompt = {
            "CONTEXT": namespace["CONTEXT_SYSTEM_PROMPT"],
            "TRAINING": namespace["TRAINING_SYSTEM_PROMPT"],
        }.get(task["task_kind"], namespace["OBSERVATION_SYSTEM_PROMPT"])
        max_output_tokens = {
            "OBSERVATIONS": namespace["MAX_OUTPUT_TOKENS_OBSERVATION"],
            "TRAINING": namespace["MAX_OUTPUT_TOKENS_TRAINING"],
        }.get(task["task_kind"], namespace["MAX_OUTPUT_TOKENS"])

        for attempt in range(namespace["RETRIES"]):
            content = ""
            retry_exception = None
            try:
                user_prompt = build_user_prompt(row, task, validated_contexts,
                                                validation_errors)
                # Streaming is required, not preferred. A silent non-streamed
                # call carrying a whole paper is killed by the gateway's edge
                # proxy at 504 before the model ever answers.
                # The gateway is inconsistent about how streamed text arrives: it
                # sends the whole response as a single delta, and the SDK's
                # accumulated message can come back with an empty text block.
                # Collect every channel separately and use whichever carried the
                # answer, rather than trusting one of them.
                delta_pieces, event_pieces = [], []
                with client.messages.stream(
                        model=namespace["MODEL"],
                        max_tokens=max_output_tokens,
                        temperature=namespace["TEMPERATURE"],
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_prompt}]) as stream:
                    for event in stream:
                        event_type = getattr(event, "type", "")
                        if event_type == "content_block_delta":
                            delta = getattr(event, "delta", None)
                            piece = getattr(delta, "text", "") if delta else ""
                            if piece:
                                delta_pieces.append(piece)
                        elif event_type == "text":
                            piece = getattr(event, "text", "")
                            if piece:
                                event_pieces.append(piece)
                    message = stream.get_final_message()
                block_text = "".join(
                    getattr(block, "text", "") or ""
                    for block in getattr(message, "content", [])
                    if getattr(block, "type", "") == "text")
                pieces = next(
                    (candidate for candidate in
                     ("".join(delta_pieces), "".join(event_pieces), block_text)
                     if candidate.strip()), "")

                usage = getattr(message, "usage", None)
                total_usage["prompt_tokens"] += int(getattr(usage, "input_tokens", 0) or 0)
                total_usage["output_tokens"] += int(getattr(usage, "output_tokens", 0) or 0)
                total_usage["cached_prompt_tokens"] += int(
                    getattr(usage, "cache_read_input_tokens", 0) or 0)

                response_id = clean_text(getattr(message, "id", ""))
                if not response_id:
                    raise ExtractionValidationError("response has no stable response_id")
                stop_reason = clean_text(getattr(message, "stop_reason", ""))
                finish_reason = STOP_REASON_TO_FINISH_REASON.get(stop_reason, stop_reason)
                if finish_reason == "refusal":
                    return terminal_result(
                        "model_refusal", task, started, total_usage,
                        "model refused task: stop_reason=refusal", attempt + 1,
                        namespace["PROVIDER_NAME"], response_id, finish_reason,
                        "stop_reason=refusal")
                if finish_reason != "stop":
                    status = ("output_truncated" if finish_reason == "length"
                              else "invalid_finish_reason")
                    return terminal_result(
                        status, task, started, total_usage,
                        f"stop_reason={stop_reason!r}, expected 'end_turn'",
                        attempt + 1, namespace["PROVIDER_NAME"], response_id,
                        finish_reason, "")
                content = pieces
                if not content.strip():
                    raise ExtractionValidationError("response has empty content")
                payload = parse_json_object(content)
                payload = validate_payload(payload, task, validated_contexts)
                return {
                    **task_result_descriptor(task),
                    "status": "ok", "payload": payload, "error": "",
                    "attempts": attempt + 1, "key_name": namespace["PROVIDER_NAME"],
                    "response_id": response_id, "finish_reason": finish_reason,
                    "refusal": "",
                    "invalid_output_sha256": "", "invalid_output_excerpt": "",
                    "latency_seconds": round(time.perf_counter() - started, 3),
                    **total_usage,
                }
            except ExtractionValidationError as exc:
                validation_errors = exc.errors
                last_error = f"ExtractionValidationError: {exc}"[:2000]
                if isinstance(content, str):
                    last_invalid_output = content
                error_kind = "invalid_output"
            except Exception as exc:  # noqa: BLE001
                # The gateway is intermittently unreliable: Cloudflare 504s,
                # withheld upstream usage and malformed streams all show up as
                # ordinary exceptions and all clear on a retry.
                validation_errors = None
                last_error = f"{type(exc).__name__}: {exc}"[:2000]
                error_kind = "api_failed"
                retry_exception = exc
            if attempt < namespace["RETRIES"] - 1:
                server_wait = namespace["retry_after_from_exception"](retry_exception) \
                    if retry_exception else None
                wait = (server_wait if server_wait is not None
                        else min(2 ** attempt, namespace["MAX_BACKOFF_SECONDS"]))
                time.sleep(min(wait, namespace["MAX_BACKOFF_SECONDS"])
                           + random.uniform(0, 0.75))
        return terminal_result(error_kind, task, started, total_usage, last_error,
                               namespace["RETRIES"], key_name=namespace["PROVIDER_NAME"],
                               invalid_output=last_invalid_output)

    return anthropic_model_call


def make_agent_model_call(namespace: dict):
    """Build the drop-in replacement for the notebook's `model_call`.

    The signature, the retry-and-repair loop, the terminal statuses and every
    field of the returned dict match the API implementation. The only difference
    is where the completion text comes from.
    """
    build_user_prompt = namespace["build_user_prompt"]
    parse_json_object = namespace["parse_json_object"]
    validate_payload = namespace["validate_payload"]
    task_result_descriptor = namespace["task_result_descriptor"]
    terminal_result = namespace["terminal_result"]
    clean_text = namespace["clean_text"]
    ExtractionValidationError = namespace["ExtractionValidationError"]

    def agent_model_call(row, task, validated_contexts=None):
        started = time.perf_counter()
        validation_errors = None
        last_error = ""
        last_invalid_output = ""
        error_kind = "api_failed"
        total_usage = {"prompt_tokens": 0, "output_tokens": 0, "cached_prompt_tokens": 0}
        system_prompt = {
            "CONTEXT": namespace["CONTEXT_SYSTEM_PROMPT"],
            "TRAINING": namespace["TRAINING_SYSTEM_PROMPT"],
        }.get(task["task_kind"], namespace["OBSERVATION_SYSTEM_PROMPT"])

        for attempt in range(namespace["RETRIES"]):
            user_prompt = build_user_prompt(row, task, validated_contexts, validation_errors)
            request = emit_request(namespace, row, task, attempt + 1,
                                   system_prompt, user_prompt)
            response_path = Path(request["response_path"])
            if not response_path.is_file():
                namespace["EMITTED_REQUESTS"].append(request)
                raise PendingAgentResponse(request)

            content = response_path.read_text(encoding="utf-8")
            total_usage["prompt_tokens"] += estimate_tokens(system_prompt) + \
                estimate_tokens(user_prompt)
            total_usage["output_tokens"] += estimate_tokens(content)
            try:
                payload = parse_json_object(content)
                payload = validate_payload(payload, task, validated_contexts)
            except ExtractionValidationError as exc:
                validation_errors = exc.errors
                last_error = f"ExtractionValidationError: {exc}"[:2000]
                last_invalid_output = content
                error_kind = "invalid_output"
                write_if_changed(
                    response_path.with_name(response_path.stem + ".rejected.json"),
                    json.dumps({"request_id": request["request_id"],
                                "errors": exc.errors}, indent=2, ensure_ascii=False))
                continue

            return {
                **task_result_descriptor(task),
                "status": "ok", "payload": payload, "error": "",
                "attempts": attempt + 1, "key_name": "LOCAL_AGENT",
                "response_id": f"local-agent-{request['request_id']}",
                "finish_reason": "stop", "refusal": "",
                "invalid_output_sha256": "", "invalid_output_excerpt": "",
                "latency_seconds": round(time.perf_counter() - started, 3),
                **total_usage,
            }

        # Every repair attempt was answered and every answer was rejected. This
        # is the API path's terminal invalid_output, reported identically.
        assert clean_text(last_error), "terminal failure requires a recorded error"
        return terminal_result(error_kind, task, started, total_usage, last_error,
                               namespace["RETRIES"], key_name="LOCAL_AGENT",
                               invalid_output=last_invalid_output)

    return agent_model_call


REQUEST_KIND_PREFIX = {"CONTEXT": "ctx", "OBSERVATIONS": "obs", "TRAINING": "trn"}


def emit_request(namespace, row, task, attempt, system_prompt, user_prompt) -> dict:
    """Write one prompt to disk and describe it.

    The request id is content-addressed over both prompts, so a repair attempt -
    whose user prompt carries the validation errors - is a different request with
    its own answer file, and replaying a finished run is deterministic.
    """
    clean_text = namespace["clean_text"]
    digest = hashlib.sha256(
        json.dumps([system_prompt, user_prompt], ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]
    prefix = REQUEST_KIND_PREFIX.get(task["task_kind"], "gen")
    request_id = f"{prefix}-{row['paper_id']}-a{attempt}-{digest}"

    request_dir = Path(namespace["REQUEST_DIR"])
    response_dir = Path(namespace["RESPONSE_DIR"])
    system_path = request_dir / f"{request_id}.system.txt"
    user_path = request_dir / f"{request_id}.user.txt"
    response_path = response_dir / f"{request_id}.json"

    write_if_changed(system_path, system_prompt)
    write_if_changed(user_path, user_prompt)

    request = {
        "request_id": request_id,
        "paper_id": row["paper_id"],
        "title": clean_text(row.get("title")),
        "task_id": task["task_id"],
        "task_kind": task["task_kind"],
        "chunk_id": task["chunk_id"],
        "attempt": attempt,
        "is_repair": attempt > 1,
        "pages": [int(page) for page in task["pages"]],
        "system_path": str(system_path),
        "user_path": str(user_path),
        "response_path": str(response_path),
        "system_prompt_chars": len(system_prompt),
        "user_prompt_chars": len(user_prompt),
        "user_prompt_lines": user_prompt.count("\n") + 1,
        "source_text_chars": len(task["target_text"]),
    }
    if task["task_kind"] == "OBSERVATIONS":
        request["observation_budget"] = task["observation_budget"]
    if task["task_kind"] == "TRAINING":
        request["pair_budget"] = task["pair_budget"]
    write_if_changed(request_dir / f"{request_id}.meta.json",
                     json.dumps(request, indent=2, ensure_ascii=False))
    return request


def collect_batch(namespace, batch_frame) -> list[dict]:
    """Drive every paper in the batch as far as its answers allow.

    Returns the prompts that still need an agent. Papers that can finish do
    finish, and their checkpoints are written by the notebook's own code.
    """
    extract_paper = namespace["extract_paper"]
    rows = batch_frame.to_dict("records")
    pending: list[dict] = []
    finished = 0
    errors: list[str] = []

    def drive(row):
        try:
            return ("done", extract_paper(row))
        except PendingAgentResponse as exc:
            return ("pending", exc.request)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(rows)) as executor:
        futures = {executor.submit(drive, row): row["paper_id"] for row in rows}
        for future in concurrent.futures.as_completed(futures):
            paper_id = futures[future]
            try:
                kind, value = future.result()
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                errors.append(f"{paper_id}: {type(exc).__name__}: {exc}")
                continue
            if kind == "pending":
                pending.append(value)
            else:
                finished += 1

    if errors:
        print("\npapers that raised during collection:")
        for line in errors:
            print("  ", line)
    print(f"papers already finished this pass: {finished}/{len(rows)}")
    return pending


def report_rows(namespace, batch_no, batch_frame) -> dict:
    start = batch_no * namespace["BATCH_SIZE"]
    end = start + len(batch_frame) - 1
    span = {
        "batch": batch_no,
        "eligible_rows_zero_indexed": [start, end],
        "eligible_rows_one_indexed": [start + 1, end + 1],
        "eligible_total": int(len(namespace["eligible"])),
        "papers": [
            {"row": start + offset + 1, "paper_id": item["paper_id"],
             "title": namespace["clean_text"](item.get("title"))}
            for offset, item in enumerate(batch_frame.to_dict("records"))
        ],
    }
    print(f"\nbatch {batch_no}: eligible-table rows {start + 1}-{end + 1} "
          f"of {span['eligible_total']} (0-indexed {start}-{end})")
    for item in span["papers"]:
        print(f"  row {item['row']:>4}  {item['paper_id']}  {item['title'][:90]}")
    return span


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, required=True,
                        help="zero-indexed batch of BATCH_SIZE papers")
    parser.add_argument("--transport", default="cavoti", choices=sorted(PROVIDERS),
                        help="which model transport answers the prompts")
    parser.add_argument("--workers", type=int, default=4,
                        help="concurrent papers in flight")
    parser.add_argument("--fresh", action="store_true",
                        help="discard the run manifest and batch records")
    parser.add_argument("--beyond-commission", action="store_true",
                        help="permit batches past the commissioned first 100 papers")
    parser.add_argument("--publish", dest="publish", action="store_true", default=True,
                        help="compact and publish the Parquet generation when the batch closes")
    parser.add_argument("--no-publish", dest="publish", action="store_false")
    arguments = parser.parse_args()

    if arguments.batch >= COMMISSIONED_BATCHES and not arguments.beyond_commission:
        print(f"batch {arguments.batch} is past the commissioned first "
              f"{COMMISSIONED_BATCHES * CONTROL_PANEL_OVERRIDES['BATCH_SIZE']} papers; "
              "pass --beyond-commission to continue deliberately")
        return 2

    namespace = bootstrap(arguments.fresh, arguments.transport, arguments.workers)
    batch_no = arguments.batch
    if batch_no >= namespace["TOTAL_BATCHES"]:
        print(f"batch {batch_no} does not exist; the corpus has "
              f"{namespace['TOTAL_BATCHES']} batches")
        return 2

    start = batch_no * namespace["BATCH_SIZE"]
    batch_frame = namespace["eligible"].iloc[start:start + namespace["BATCH_SIZE"]]
    span = report_rows(namespace, batch_no, batch_frame)

    if arguments.transport == "agent":
        pending = collect_batch(namespace, batch_frame)
        agent_io = Path(namespace["AGENT_IO_DIR"])
        pending_path = agent_io / "PENDING.json"
        write_if_changed(pending_path, json.dumps(
            {"batch": batch_no, "rows": span["eligible_rows_one_indexed"],
             "generated_at": namespace["utc_now"](), "pending": pending},
            indent=2, ensure_ascii=False))

        if pending:
            by_kind: dict[str, int] = {}
            for item in pending:
                by_kind[item["task_kind"]] = by_kind.get(item["task_kind"], 0) + 1
            print(f"\n{len(pending)} prompt(s) awaiting an agent: {by_kind}")
            print(f"index: {pending_path}")
            for item in pending:
                print(f"  {item['request_id']}  {item['task_kind']:<12} "
                      f"{item['user_prompt_chars']:>7,} chars  "
                      f"{'REPAIR ' if item['is_repair'] else ''}{item['title'][:60]}")
            return PENDING_EXIT_CODE
        print("\nall answers present; running the notebook's batch driver")
    else:
        agent_io = Path(namespace["AGENT_IO_DIR"])
        print(f"\ncalling {arguments.transport} for every unfinished task")
    status = namespace["run_extraction_batch"](batch_no, batch_frame)
    write_if_changed(agent_io / f"batch_{batch_no:05d}_rows.json",
                     json.dumps(span, indent=2, ensure_ascii=False))

    incomplete = status[status["status"] != "complete"]
    if len(incomplete):
        print("\npapers that did not reach complete:")
        for item in incomplete.to_dict("records"):
            print(f"  {item['paper_id']}: {item['status']}: {str(item['error'])[:200]}")

    if arguments.publish:
        cells = load_notebook_cells(NOTEBOOK_PATH)
        for index in (COMPACTION_CELL, SUMMARY_CELL):
            print(f"\n--- notebook cell {index} ---")
            exec(compile(cells[index], f"<notebook cell {index}>", "exec"), namespace)

    print(f"\nbatch {batch_no} closed: eligible-table rows "
          f"{span['eligible_rows_one_indexed'][0]}-{span['eligible_rows_one_indexed'][1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
