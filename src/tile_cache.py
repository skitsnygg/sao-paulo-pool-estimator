from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Hashable, Optional

import cv2
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class TileCache:
    """
    Thread-safe shared cache used by the Google XYZ downloader.

    Supports:
    - in-memory LRU cache
    - optional on-disk tile cache
    - backwards-compatible URL fetch cache via `get(url)`
    """

    def __init__(
        self,
        maxsize: int | None = None,
        *,
        max_memory_tiles: int | None = None,
        disk_dir: str | Path | None = None,
        request_timeout: float = 20.0,
        request_retries: int = 4,
        request_backoff: float = 0.5,
    ):
        if max_memory_tiles is None:
            max_memory_tiles = maxsize if maxsize is not None else 5000
        self.max_memory_tiles = max(0, int(max_memory_tiles))
        self.request_timeout = max(0.1, float(request_timeout))

        self._lock = threading.Lock()
        self._memory: OrderedDict[Hashable, bytes] = OrderedDict()
        self._disk_dir = Path(disk_dir).resolve() if disk_dir else None
        if self._disk_dir is not None:
            self._disk_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        retry = Retry(
            total=max(0, int(request_retries)),
            backoff_factor=max(0.0, float(request_backoff)),
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            pool_connections=64,
            pool_maxsize=128,
            max_retries=retry,
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    @property
    def enabled(self) -> bool:
        return self.max_memory_tiles > 0 or self._disk_dir is not None

    def _memory_get(self, key: Hashable) -> Optional[bytes]:
        if self.max_memory_tiles <= 0:
            return None
        with self._lock:
            data = self._memory.pop(key, None)
            if data is not None:
                self._memory[key] = data
            return data

    def _memory_put(self, key: Hashable, data: bytes) -> None:
        if self.max_memory_tiles <= 0:
            return
        with self._lock:
            self._memory.pop(key, None)
            self._memory[key] = data
            while len(self._memory) > self.max_memory_tiles:
                self._memory.popitem(last=False)

    def _tile_key(self, z: int, x: int, y: int) -> tuple[str, int, int, int]:
        return "tile", int(z), int(x), int(y)

    def _url_key(self, url: str) -> tuple[str, str]:
        return "url", str(url)

    def _disk_path(self, z: int, x: int, y: int) -> Path:
        if self._disk_dir is None:
            raise RuntimeError("disk cache disabled")
        return self._disk_dir / f"z{int(z)}" / f"x{int(x)}" / f"y{int(y)}.tile"

    def _disk_get(self, z: int, x: int, y: int) -> Optional[bytes]:
        if self._disk_dir is None:
            return None
        path = self._disk_path(z, x, y)
        if not path.exists():
            return None
        try:
            return path.read_bytes()
        except OSError:
            return None

    def _disk_put(self, z: int, x: int, y: int, data: bytes) -> None:
        if self._disk_dir is None:
            return
        path = self._disk_path(z, x, y)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".{threading.get_ident()}.tmp")
        try:
            with tmp.open("wb") as f:
                f.write(data)
            tmp.replace(path)
        except OSError:
            tmp.unlink(missing_ok=True)

    @staticmethod
    def _normalize_channels(tile: np.ndarray, channels: int) -> Optional[np.ndarray]:
        if tile is None or tile.size == 0:
            return None

        if tile.ndim == 2:
            if channels == 1:
                return tile[:, :, np.newaxis]
            if channels == 3:
                return cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
            if channels == 4:
                return cv2.cvtColor(tile, cv2.COLOR_GRAY2BGRA)
            return None

        if tile.ndim != 3:
            return None

        current = tile.shape[2]
        if current == channels:
            return tile
        if current > channels:
            return tile[:, :, :channels]

        if current == 1 and channels == 3:
            return cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        if current == 1 and channels == 4:
            return cv2.cvtColor(tile, cv2.COLOR_GRAY2BGRA)
        if current == 3 and channels == 4:
            return cv2.cvtColor(tile, cv2.COLOR_BGR2BGRA)
        if current == 4 and channels == 3:
            return cv2.cvtColor(tile, cv2.COLOR_BGRA2BGR)
        return None

    @classmethod
    def decode_tile(cls, data: bytes, channels: int) -> Optional[np.ndarray]:
        if not data:
            return None
        arr = np.frombuffer(data, dtype=np.uint8)
        if arr.size == 0:
            return None
        flag = cv2.IMREAD_UNCHANGED if int(channels) == 4 else cv2.IMREAD_COLOR
        decoded = cv2.imdecode(arr, flag)
        return cls._normalize_channels(decoded, int(channels))

    def get_tile_raw(self, z: int, x: int, y: int) -> Optional[bytes]:
        key = self._tile_key(z, x, y)
        data = self._memory_get(key)
        if data is None:
            data = self._disk_get(z, x, y)
            if data is not None:
                self._memory_put(key, data)
        return data

    def get_decoded(self, z: int, x: int, y: int, channels: int) -> Optional[np.ndarray]:
        data = self.get_tile_raw(z, x, y)
        if data is None:
            return None
        return self.decode_tile(data, channels)

    def put_raw(self, z: int, x: int, y: int, data: bytes) -> None:
        if not data:
            return
        key = self._tile_key(z, x, y)
        self._memory_put(key, data)
        self._disk_put(z, x, y, data)

    # Backwards-compatible URL getter.
    def get(self, url: str) -> bytes:
        key = self._url_key(url)
        data = self._memory_get(key)
        if data is not None:
            return data

        response = self.session.get(str(url), timeout=self.request_timeout)
        response.raise_for_status()
        data = response.content
        self._memory_put(key, data)
        return data
