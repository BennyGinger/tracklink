from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import tracklink.api as api
from tracklink.api import TrackModel


class DummyBackend:
    def __init__(self) -> None:
        self.settings: dict[str, Any] | None = None
        self.track_calls: list[tuple[np.ndarray, np.ndarray]] = []
        self.filter_calls: list[tuple[int, int | None]] = []
        self._tracks = pd.DataFrame({"track_id": [1], "length": [3]})

    def configure(self, user_settings: dict[str, Any]) -> None:
        self.settings = user_settings

    def track(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self.track_calls.append((image, mask))
        return mask + 10

    def filter_by_length(self, min_length: int, max_length: int | None) -> np.ndarray:
        self.filter_calls.append((min_length, max_length))
        return np.full((2, 2), 4, dtype=np.uint16)

    @property
    def tracks_df(self) -> pd.DataFrame:
        return self._tracks


def test_track_model_delegates_configuration_and_tracking(monkeypatch) -> None:
    backend = DummyBackend()
    monkeypatch.setattr(api, "get_backend", lambda name: backend)
    model = TrackModel("trackastra")
    image = np.ones((2, 2), dtype=np.uint16)
    mask = np.full((2, 2), 2, dtype=np.uint16)

    model.configure({"pretrained_model": "general_2d"})
    result = model.track(image, mask)

    assert backend.settings == {"pretrained_model": "general_2d"}
    assert backend.track_calls == [(image, mask)]
    np.testing.assert_array_equal(result, mask + 10)


def test_track_model_delegates_filter_and_table(monkeypatch) -> None:
    backend = DummyBackend()
    monkeypatch.setattr(api, "get_backend", lambda name: backend)
    model = TrackModel()

    filtered = model.filter_by_length(min_length=2, max_length=5)

    assert backend.filter_calls == [(2, 5)]
    np.testing.assert_array_equal(filtered, np.full((2, 2), 4, dtype=np.uint16))
    pd.testing.assert_frame_equal(model.tracks_df, backend.tracks_df)
