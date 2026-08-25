"""TTS speakable rewriting — no Piper binary required."""

from __future__ import annotations

from mufasa_app.tts import number_to_speech, speakable, split_sentences


def test_strips_citations_by_default() -> None:
    assert "[E1]" not in speakable("The ash reached 31.2 MPa [E1].")


def test_keeps_spoken_citations_when_asked() -> None:
    out = speakable("Result [E1].", keep_citations=True)
    assert "evidence" in out


def test_numbers_and_units_are_spoken() -> None:
    out = speakable("Conductivity was 39.97 mg/L.")
    assert "point" in out
    assert "milligrams per litre" in out
    assert "39.97" not in out


def test_number_to_speech_handles_thousands() -> None:
    assert "thousand" in number_to_speech("1,200", None)


def test_split_sentences() -> None:
    parts = split_sentences("First claim. Second claim!")
    assert parts == ["First claim.", "Second claim!"]
