"""Utilities for controlling TabPFN telemetry behaviour."""

from __future__ import annotations

import os
from typing import Final

_DISABLED_FLAG: Final[str] = "_TABPFN_TELEMETRY_DISABLED"


def _is_truthy(value: str | None) -> bool:
    """Return ``True`` if ``value`` represents an affirmative flag."""

    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _ensure_disabled() -> None:
    """Disable telemetry by patching upstream helpers."""

    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")

    if globals().get(_DISABLED_FLAG):
        return

    globals()[_DISABLED_FLAG] = True

    try:
        from tabpfn_common_utils.telemetry.core import service as telemetry_service
    except Exception:
        telemetry_service = None  # type: ignore[assignment]
    else:
        # Short-circuit capture so the PostHog client is never created.
        telemetry_service.capture_event = lambda *args, **kwargs: None  # type: ignore[attr-defined]

        def _always_disabled(cls) -> bool:  # type: ignore[override]
            return False

        try:
            telemetry_service.ProductTelemetry.telemetry_enabled = classmethod(_always_disabled)  # type: ignore[attr-defined]
        except Exception:
            pass

    try:
        from tabpfn_common_utils.telemetry.core import config as telemetry_config
    except Exception:
        telemetry_config = None  # type: ignore[assignment]
    else:
        telemetry_config.download_config = lambda: {"enabled": False}  # type: ignore[attr-defined]


def configure_telemetry(user_preference: bool | None) -> bool:
    """Determine whether telemetry should be enabled."""

    if user_preference is False:
        _ensure_disabled()
        return False

    if user_preference is True:
        return True

    if _is_truthy(os.environ.get("TABPFN_OFFLINE_MODE")):
        _ensure_disabled()
        return False

    if _is_truthy(os.environ.get("TABPFN_DISABLE_TELEMETRY")):
        _ensure_disabled()
        return False

    return True


__all__ = ["configure_telemetry"]
