"""Start the laptop-local service.

    uv run mufasa-serve

Binds to 127.0.0.1 unless MUFASA_SHARE_LAN is on, in which case it also listens
on the private LAN for one paired mobile session.
"""

from __future__ import annotations

import sys

import uvicorn

from .config import get_settings


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    host = "0.0.0.0" if settings.mufasa_share_lan else settings.mufasa_host  # noqa: S104
    if settings.mufasa_share_lan:
        print(
            "WARNING: MUFASA_SHARE_LAN is on. The laptop is listening on the local "
            "network for one paired session. Turn it off to close the listener.",
            file=sys.stderr,
        )
    uvicorn.run(
        "mufasa_app.api:app",
        host=host,
        port=settings.mufasa_port,
        log_level="info",
        access_log=False,
        workers=1,  # single process: peak memory must stay predictable
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
