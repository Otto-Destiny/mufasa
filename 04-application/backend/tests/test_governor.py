"""Resource governor: single-flight, cancel, thread caps."""

from __future__ import annotations

import pytest
from mufasa_app.governor import Busy, Cancelled, Governor


def test_single_flight_rejects_second_job() -> None:
    gov = Governor(single_flight=True)
    first = gov.acquire("q1")
    with pytest.raises(Busy):
        gov.acquire("q2")
    gov.release(first)
    second = gov.acquire("q2")
    assert second.job_id != first.job_id
    gov.release(second)


def test_cancel_sets_flag_and_checkpoint_raises() -> None:
    gov = Governor()
    job = gov.acquire("q")
    assert gov.cancel(job.job_id) is True
    with pytest.raises(Cancelled):
        gov.checkpoint(job, "generating")
    gov.release(job)


def test_as_dict_reports_busy_state() -> None:
    gov = Governor(max_output_tokens=200, threads=2)
    assert gov.as_dict()["busy"] is False
    job = gov.acquire("hello")
    assert gov.busy is True
    assert gov.as_dict()["current"]["question"] == "hello"
    gov.release(job)
    assert gov.busy is False
