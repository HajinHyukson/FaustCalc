import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional


class FileCache:
    """Simple JSON file cache with TTL-based reads."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for_key(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def get(self, key: str, ttl_hours: float) -> Optional[Any]:
        cache_path = self._path_for_key(key)
        if not cache_path.exists():
            return None

        max_age = timedelta(hours=ttl_hours)
        modified = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc)
        if datetime.now(timezone.utc) - modified > max_age:
            return None

        try:
            with cache_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def set(self, key: str, value: Any) -> None:
        cache_path = self._path_for_key(key)
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(value, f)
