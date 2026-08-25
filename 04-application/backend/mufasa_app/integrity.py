"""Integrity manifest.

An hour of work that answers "how do we know this is what you tested?" before
it is asked. Backs the System Integrity Check screen: model file, database file
and application code, each hashed, each compared against what was recorded at
release time.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import BACKEND_ROOT, Settings

MANIFEST_NAME = "MANIFEST.sha256"


def sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_tree(root: Path, *, suffixes: tuple[str, ...] = (".py",)) -> str:
    """Order-independent hash of a source tree."""
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in suffixes):
        h.update(str(path.relative_to(root)).replace("\\", "/").encode("utf-8"))
        h.update(sha256_file(path).encode("ascii"))
    return h.hexdigest()


@dataclass
class Component:
    name: str
    kind: str
    path: str
    present: bool
    sha256: str | None
    expected: str | None
    status: str  # verified | unverified | mismatch | missing

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "path": self.path,
            "present": self.present,
            "sha256": self.sha256,
            "expected": self.expected,
            "status": self.status,
        }


def _status(present: bool, actual: str | None, expected: str | None) -> str:
    if not present:
        return "missing"
    if not expected:
        return "unverified"
    return "verified" if actual == expected else "mismatch"


def _component(name: str, kind: str, path: Path, expected: str | None) -> Component:
    present = path.exists() and path.is_file()
    actual = sha256_file(path) if present else None
    return Component(
        name=name,
        kind=kind,
        path=str(path),
        present=present,
        sha256=actual,
        expected=expected or None,
        status=_status(present, actual, expected),
    )


def _release_manifest() -> dict[str, str]:
    path = BACKEND_ROOT / MANIFEST_NAME
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, name = line.partition("  ")
        if digest and name:
            out[name.strip()] = digest.strip()
    return out


def report(settings: Settings) -> dict[str, Any]:
    recorded = _release_manifest()
    components: list[Component] = []

    try:
        model = settings.model_entry()
        components.append(
            _component(model.get("label", model["key"]), "model", Path(model["path"]),
                       model.get("sha256") or recorded.get(model["file"]))
        )
    except KeyError as exc:
        components.append(
            Component(name=str(exc), kind="model", path="", present=False, sha256=None,
                      expected=None, status="missing")
        )

    components.append(
        _component("Evidence database", "database", settings.db_path,
                   recorded.get(settings.db_path.name))
    )

    code_hash = sha256_tree(BACKEND_ROOT / "mufasa_app")
    components.append(
        Component(
            name="Application code",
            kind="code",
            path=str(BACKEND_ROOT / "mufasa_app"),
            present=True,
            sha256=code_hash,
            expected=recorded.get("mufasa_app/") or None,
            status=_status(True, code_hash, recorded.get("mufasa_app/")),
        )
    )

    if settings.mufasa_tts_enabled:
        try:
            voice = settings.voice_entry()
            components.append(
                _component(voice.get("label", voice["key"]), "voice", Path(voice["path"]),
                           voice.get("sha256") or recorded.get(voice["file"]))
            )
        except KeyError:
            pass

    statuses = {c.status for c in components}
    if "mismatch" in statuses:
        overall, headline = "mismatch", "Something changed and should be checked."
    elif "missing" in statuses:
        overall, headline = "incomplete", "One or more pieces needed to answer are missing."
    elif statuses == {"verified"}:
        overall, headline = "verified", "Everything looks ready."
    else:
        overall, headline = "unverified", "Your files are present and ready to use."

    return {
        "overall": overall,
        "headline": headline,
        "components": [c.as_dict() for c in components],
        "manifest_present": bool(recorded),
    }


def write_release_manifest(settings: Settings, extra: dict[str, Path] | None = None) -> Path:
    """Build-plane helper: record the hashes this release should always have."""
    lines = ["# MUFASA release manifest. sha256  name"]
    targets: dict[str, Path] = {settings.db_path.name: settings.db_path}
    try:
        model = settings.model_entry()
        targets[model["file"]] = Path(model["path"])
    except KeyError:
        pass
    targets.update(extra or {})
    for name, path in sorted(targets.items()):
        if path.exists():
            lines.append(f"{sha256_file(path)}  {name}")
    lines.append(f"{sha256_tree(BACKEND_ROOT / 'mufasa_app')}  mufasa_app/")
    out = BACKEND_ROOT / MANIFEST_NAME
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def as_json(settings: Settings) -> str:
    return json.dumps(report(settings), indent=2)
