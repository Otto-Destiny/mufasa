"""The resource governor.

Small, and it protects the only two failure modes that score zero: exceeding the
7 GB ceiling is disqualification, and thermal throttling costs 10 marks. So one
generation runs at a time across every client, caps are enforced before the
expensive work starts, and cancel actually frees the work rather than
abandoning it to finish in the background.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


class Busy(RuntimeError):
    """A second generation was requested while one was already running."""


class Cancelled(RuntimeError):
    pass


@dataclass
class Job:
    job_id: str
    question: str
    started_at: float
    stage: str = "queued"
    cancelled: threading.Event = field(default_factory=threading.Event)

    @property
    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self.started_at) * 1000)

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "question": self.question,
            "stage": self.stage,
            "elapsed_ms": self.elapsed_ms,
            "cancelled": self.cancelled.is_set(),
        }


class Governor:
    """Single-flight admission control plus the process limits."""

    def __init__(
        self,
        *,
        single_flight: bool = True,
        max_context_tokens: int = 2048,
        max_output_tokens: int = 400,
        threads: int = 4,
    ) -> None:
        self.single_flight = single_flight
        self.max_context_tokens = max_context_tokens
        self.max_output_tokens = max_output_tokens
        self.threads = threads
        self._lock = threading.Lock()
        self._current: Job | None = None

    # -- admission ---------------------------------------------------------

    def acquire(self, question: str) -> Job:
        with self._lock:
            if self.single_flight and self._current is not None:
                raise Busy("An answer is already in progress.")
            job = Job(job_id=uuid.uuid4().hex[:12], question=question, started_at=time.monotonic())
            self._current = job
            return job

    def release(self, job: Job) -> None:
        with self._lock:
            if self._current is not None and self._current.job_id == job.job_id:
                self._current = None

    def cancel(self, job_id: str | None = None) -> bool:
        with self._lock:
            job = self._current
            if job is None:
                return False
            if job_id and job.job_id != job_id:
                return False
            job.cancelled.set()
            return True

    @property
    def current(self) -> Job | None:
        with self._lock:
            return self._current

    @property
    def busy(self) -> bool:
        return self.current is not None

    # -- limits ------------------------------------------------------------

    def checkpoint(self, job: Job, stage: str) -> None:
        """Advance a job's stage, raising if the user cancelled it."""
        if job.cancelled.is_set():
            raise Cancelled(f"cancelled during {stage}")
        job.stage = stage

    def apply_thread_caps(self) -> None:
        """Cap every library that would otherwise use all cores.

        Graph queries and embeddings running flat out while the model decodes
        will overheat the laptop, and a thermal trip is a scored penalty.
        """
        n = str(max(1, self.threads))
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(var, n)

    def as_dict(self) -> dict[str, Any]:
        job = self.current
        return {
            "single_flight": self.single_flight,
            "max_context_tokens": self.max_context_tokens,
            "max_output_tokens": self.max_output_tokens,
            "threads": self.threads,
            "busy": job is not None,
            "current": job.as_dict() if job else None,
        }


def peak_rss_bytes() -> int | None:
    """Peak resident set size of this process, or None where unavailable.

    RSS deliberately, not PSS: the audit records maximum RSS, and PSS divides
    shared pages between processes so it always reads lower. Tuning against PSS
    means discovering the gap on the judging laptop.
    """
    try:
        import resource  # noqa: PLC0415 - POSIX only

        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak_kb) * 1024
    except Exception:  # noqa: BLE001 - Windows dev machines land here
        try:
            import ctypes
            import ctypes.wintypes as wt

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wt.DWORD),
                    ("PageFaultCount", wt.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            ):
                return int(counters.PeakWorkingSetSize)
        except Exception:  # noqa: BLE001
            return None
    return None
