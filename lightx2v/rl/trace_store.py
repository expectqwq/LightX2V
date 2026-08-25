"""Short-lived safetensors bundles for image rollout traces."""

from __future__ import annotations

import os
import re
import time
import uuid
from pathlib import Path

from safetensors.torch import save_file

from .sde import SdeTrace

_BUNDLE_RE = re.compile(r"^[a-f0-9]{32}$")


class TraceStore:
    def __init__(self, root: str | os.PathLike[str] = "/tmp/mova_rl_traces", ttl_seconds: int = 3600):
        self.root = Path(root)
        self.ttl_seconds = int(ttl_seconds)
        if self.ttl_seconds <= 0:
            raise ValueError("trace TTL must be positive")
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, bundle_id: str) -> Path:
        if not _BUNDLE_RE.fullmatch(bundle_id):
            raise ValueError("invalid trace bundle id")
        return self.root / f"{bundle_id}.safetensors"

    def put(self, trace: SdeTrace, metadata: dict[str, str] | None = None) -> str:
        self.cleanup_expired()
        bundle_id = uuid.uuid4().hex
        values = {str(key): str(value) for key, value in (metadata or {}).items()}
        values["created_at_unix"] = str(time.time())
        save_file(trace.as_tensors(), str(self._path(bundle_id)), metadata=values)
        return bundle_id

    def get_path(self, bundle_id: str) -> Path:
        self.cleanup_expired()
        path = self._path(bundle_id)
        if not path.is_file():
            raise FileNotFoundError(bundle_id)
        return path

    def delete(self, bundle_id: str) -> bool:
        path = self._path(bundle_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def cleanup_expired(self, now: float | None = None) -> int:
        threshold = (time.time() if now is None else now) - self.ttl_seconds
        removed = 0
        for path in self.root.glob("*.safetensors"):
            if path.stat().st_mtime < threshold:
                path.unlink()
                removed += 1
        return removed
