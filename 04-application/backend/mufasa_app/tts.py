"""Offline voice output.

The Web Speech API is not an option: on Ubuntu the webview delegates to
speech-dispatcher, which may not be installed or configured on the judging
laptop — exactly the shape of failure this project keeps getting bitten by. So
synthesis is a bundled Piper binary reading a local ONNX voice, swapped by
`.env` the same way models are.

Two things make this better than the streaming assistants it imitates, and both
fall out of decisions already made:

* We do not stream tokens, so by the time audio exists the text has been
  validated. Sentences are known, so playback can highlight the sentence being
  read instead of guessing.
* Scientific notation is normalised before synthesis. A generic engine reads
  "39.97 mg/L" as "thirty nine point nine seven em gee slash ell". For a
  research tool that is unusable.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
         "fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = ("_ _ twenty thirty forty fifty sixty seventy eighty ninety").split()

#: Unit spellings, longest key first so "mg/L" wins over "L".
UNIT_SPEECH: dict[str, str] = {
    "mg/l": "milligrams per litre",
    "µg/l": "micrograms per litre",
    "ug/l": "micrograms per litre",
    "mg/kg": "milligrams per kilogram",
    "µs/cm": "microsiemens per centimetre",
    "us/cm": "microsiemens per centimetre",
    "l/person/day": "litres per person per day",
    "m3/year": "cubic metres per year",
    "km2": "square kilometres",
    "mm/h": "millimetres per hour",
    "ohm-m": "ohm metres",
    "ohm-metre": "ohm metres",
    "ntu": "nephelometric turbidity units",
    "cfu": "colony forming units",
    "mpa": "megapascals",
    "db": "decibels",
    "mm": "millimetres",
    "°c": "degrees Celsius",
    "%": "percent",
}
_UNIT_RE = re.compile(
    "(?<![A-Za-z])(" + "|".join(sorted((re.escape(u) for u in UNIT_SPEECH), key=len, reverse=True))
    + ")(?![A-Za-z])",
    re.IGNORECASE,
)
_TAG_RE = re.compile(r"\s*\[E(\d+)\]")
_NUM_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?![\w])")


def _int_to_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, rest = divmod(n, 10)
        return _TENS[tens] + (f"-{_ONES[rest]}" if rest else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return f"{_ONES[hundreds]} hundred" + (f" and {_int_to_words(rest)}" if rest else "")
    for value, name in ((10**9, "billion"), (10**6, "million"), (1000, "thousand")):
        if n >= value:
            head, rest = divmod(n, value)
            return f"{_int_to_words(head)} {name}" + (f" {_int_to_words(rest)}" if rest else "")
    return str(n)


def number_to_speech(whole: str, frac: str | None) -> str:
    whole_clean = whole.replace(",", "")
    try:
        words = _int_to_words(int(whole_clean))
    except (ValueError, IndexError):
        return whole + (f".{frac}" if frac else "")
    if frac:
        digits = " ".join(_ONES[int(d)] for d in frac)
        return f"{words} point {digits}"
    return words


def speakable(text: str, *, keep_citations: bool = False) -> str:
    """Rewrite a validated answer into something worth listening to."""
    if keep_citations:
        out = _TAG_RE.sub(lambda m: f", evidence {_int_to_words(int(m.group(1)))},", text)
    else:
        out = _TAG_RE.sub("", text)
    out = _NUM_RE.sub(lambda m: number_to_speech(m.group(1), m.group(2)), out)
    out = _UNIT_RE.sub(lambda m: " " + UNIT_SPEECH[m.group(1).lower()], out)
    out = re.sub(r"\s+([,.;:])", r"\1", out)
    return re.sub(r"\s{2,}", " ", out).strip()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip())
    return [p.strip() for p in parts if p.strip()]


@dataclass
class VoiceUnavailable(RuntimeError):
    reason: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.reason


class PiperVoice:
    """Thin wrapper over the bundled Piper binary. Never downloads anything."""

    def __init__(self, binary: Path, model: Path, config: Path | None = None) -> None:
        self.binary, self.model, self.config = Path(binary), Path(model), config

    def available(self) -> bool:
        return self.binary.exists() and self.model.exists()

    def check(self) -> None:
        if not self.binary.exists():
            raise VoiceUnavailable(f"Piper binary not found at {self.binary}")
        if not self.model.exists():
            raise VoiceUnavailable(f"voice model not found at {self.model}")

    def synthesize(self, text: str, *, length_scale: float = 1.0, timeout: float = 120.0) -> bytes:
        """Return one WAV. Raises VoiceUnavailable rather than reaching a network."""
        self.check()
        cmd = [
            str(self.binary),
            "--model", str(self.model),
            "--output_file", "-",
            "--length_scale", f"{length_scale:.2f}",
        ]
        if self.config and Path(self.config).exists():
            cmd += ["--config", str(self.config)]
        try:
            proc = subprocess.run(  # noqa: S603 - fixed local binary, no shell
                cmd,
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise VoiceUnavailable(f"Piper failed: {exc}") from exc
        if proc.returncode != 0 or not proc.stdout:
            raise VoiceUnavailable(
                f"Piper exited {proc.returncode}: {proc.stderr.decode('utf-8', 'replace')[:300]}"
            )
        return proc.stdout
