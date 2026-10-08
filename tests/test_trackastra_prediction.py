"""Recovery tests need no GPU; equivalence tests exercise real Trackastra helpers."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Self, cast
import weakref

import numpy as np
import networkx as nx
import dask.array as da
import pytest
import torch
from numpy.typing import NDArray
from trackastra.data import WRFeatures, build_windows
from trackastra.model import Trackastra, TrackingTransformer, model_api
from trackastra.model.predict import predict_windows as upstream_predict_windows
from tqdm import tqdm

from tracklink.backends.trackastra import model as adapter
from tracklink.backends.trackastra import prediction
from tracklink.backends.trackastra import features as preparation
from tracklink.backends.trackastra.base import Mode
from tracklink.backends.trackastra.prediction import Predictions
from tracklink.backends.trackastra.features import PredictionWindow

Prepared = tuple[list[WRFeatures], list[PredictionWindow]]


class QuietProgress(tqdm):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs, disable=True)


@pytest.fixture
def prepared() -> Prepared:
    features = []
    for time in range(12):
        # Variable detection counts exercise upstream padding; include an empty frame.
        count = 0 if time == 6 else 1 + time % 3
        features.append(WRFeatures(
            coords=np.full((count, 2), time, dtype=np.float32),
            labels=np.arange(1, count + 1), timepoints=np.full(count, time),
            features=OrderedDict(test=np.full((count, 7), time, dtype=np.float32)),
        ))
    windows = build_windows(features, window_size=4, as_torch=True,
                            progbar_class=QuietProgress)
    return features, cast(list[PredictionWindow], windows)


class FakeTransformer:
    """Expose a device without allocating CUDA tensors."""
    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.config = {"window": 4, "coord_dim": 2}

    def parameters(self) -> Iterator[SimpleNamespace]:
        yield SimpleNamespace(device=torch.device(self.device))

    def to(self, device: str) -> Self:
        self.device = device
        return self

    def eval(self) -> Self:
        return self


def fake_scores(batch: list[PredictionWindow], model: object) -> NDArray[np.float32]:
    size = max(len(window["timepoints"]) for window in batch)
    scores = np.zeros((len(batch), size, size), dtype=np.float32)
    for i, window in enumerate(batch):
        times = window["timepoints"].numpy()
        # Scores depend on absolute window position, making wrong averaging visible.
        scores[i, :len(times), :len(times)] = 0.1 + int(window["t1"]) * 0.02
    return scores


def run_prediction(prepared: Prepared, model: TrackingTransformer | FakeTransformer,
                   batch_size: int = 4) -> Predictions:
    features, windows = prepared
    return prediction.predict_windows(windows, features, cast(TrackingTransformer, model), spatial_dim=2,
                                      batch_size=batch_size, progbar_class=QuietProgress)


def assert_predictions_equal(actual: Predictions, expected: Predictions) -> None:
    assert actual["nodes"] == expected["nodes"]
    actual_weights, expected_weights = dict(actual["weights"]), dict(expected["weights"])
    assert actual_weights.keys() == expected_weights.keys()
    np.testing.assert_allclose(list(actual_weights.values()), list(expected_weights.values()),
                               rtol=1e-5, atol=1e-7)


def test_retry_preserves_completed_windows_and_averaging(
        prepared: Prepared, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(prediction, "predict", fake_scores)
    expected = run_prediction(prepared, FakeTransformer("cpu"), batch_size=1)
    calls = []
    successful = []
    cleanup = []

    def failing_predict(batch: list[PredictionWindow], model: FakeTransformer) -> NDArray[np.float32]:
        start = int(batch[0]["t1"])
        calls.append((start, len(batch), model.device))
        # First four windows succeed. The next position needs batch size 2;
        # the final position needs CPU even with batch size 1.
        if model.device == "cuda" and ((start == 4 and len(batch) > 2) or start == 6):
            raise torch.cuda.OutOfMemoryError("simulated CUDA OOM")
        successful.extend(int(window["t1"]) for window in batch)
        return fake_scores(batch, model)

    monkeypatch.setattr(prediction, "predict", failing_predict)
    monkeypatch.setattr(prediction.torch.cuda, "empty_cache", lambda: cleanup.append(True))
    result = run_prediction(prepared, FakeTransformer())

    assert calls == [(0, 4, "cuda"), (4, 4, "cuda"), (4, 3, "cuda"),
                     (4, 2, "cuda"), (6, 2, "cuda"), (6, 1, "cuda"),
                     (6, 1, "cpu"), (7, 1, "cpu"), (8, 1, "cpu")]
    assert successful == list(range(9))
    assert len(cleanup) == 5
    assert "continuing on CPU" in caplog.text
    assert_predictions_equal(result, expected)


@pytest.mark.parametrize("device,error", [
    ("cuda", RuntimeError("unrelated failure")),
    ("cpu", torch.cuda.OutOfMemoryError("CPU failure must propagate")),
    ("cpu", MemoryError("host RAM exhausted")),
])
def test_unrelated_and_cpu_errors_are_not_retried(
        prepared: Prepared, monkeypatch: pytest.MonkeyPatch,
        device: str, error: Exception) -> None:
    calls = []

    def fail(batch: list[PredictionWindow], model: FakeTransformer) -> NDArray[np.float32]:
        calls.append(True)
        raise error

    monkeypatch.setattr(prediction, "predict", fail)
    with pytest.raises(type(error), match=str(error)):
        run_prediction(prepared, FakeTransformer(device))
    assert calls == [True]


def test_empty_windows_skip_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    features = [WRFeatures(np.zeros((0, 2)), np.array([], dtype=int),
                           np.array([], dtype=int), OrderedDict(test=np.zeros((0, 7))))
                for _ in range(5)]
    windows = build_windows(features, window_size=4, as_torch=True,
                            progbar_class=QuietProgress)

    def unexpected_call(*args: object) -> None:
        pytest.fail("Empty windows must not invoke prediction")

    monkeypatch.setattr(prediction, "predict", unexpected_call)
    assert run_prediction((features, cast(list[PredictionWindow], windows)),
                           FakeTransformer()) == {"nodes": [], "weights": ()}


def test_failed_prediction_references_are_released_before_cache_cleanup(
        prepared: Prepared, monkeypatch: pytest.MonkeyPatch) -> None:
    references = []
    calls = []

    def fail_once(batch: list[PredictionWindow], model: FakeTransformer) -> NDArray[np.float32]:
        calls.append(True)
        if len(calls) == 1:
            temporary = torch.zeros(1)
            references.append(weakref.ref(temporary))
            raise torch.cuda.OutOfMemoryError("temporary retained by traceback")
        return fake_scores(batch, model)

    def cleanup() -> None:
        assert references[0]() is None

    monkeypatch.setattr(prediction, "predict", fail_once)
    monkeypatch.setattr(prediction.torch.cuda, "empty_cache", cleanup)
    run_prediction(prepared, FakeTransformer())


@pytest.fixture
def transformer() -> TrackingTransformer:
    torch.manual_seed(7)
    return TrackingTransformer(coord_dim=2, feat_dim=7, d_model=32, nhead=4,
                               num_encoder_layers=1, num_decoder_layers=1,
                               pos_embed_per_dim=8, feat_embed_per_dim=1,
                               dropout=0, window=4).eval()


@pytest.mark.parametrize("batch_size", [1, 3, 4])
def test_real_predictions_match_upstream(prepared: Prepared, transformer: TrackingTransformer,
                                         batch_size: int) -> None:
    features, windows = prepared
    expected = upstream_predict_windows(cast(list[dict[str, Any]], windows), features, transformer, spatial_dim=2,
                                         batch_size=batch_size, progbar_class=QuietProgress)
    actual = run_prediction(prepared, transformer, batch_size)
    assert_predictions_equal(actual, cast(Predictions, expected))


@pytest.mark.parametrize("feature_type", ["regionprops2", "pretrained_feats", "pretrained_feats_aug"])
def test_prediction_prepares_once_and_updates_device_after_fallback(
        prepared: Prepared, transformer: TrackingTransformer,
        monkeypatch: pytest.MonkeyPatch, feature_type: str) -> None:
    calls = []
    features, windows = prepared
    extractor = SimpleNamespace(force_recompute=False)

    def create_extractor(name: str, shape: tuple[int, int], **kwargs: Any) -> SimpleNamespace:
        calls.append("extractor")
        assert name == "test-model"
        assert shape == (32, 32)
        assert kwargs["device"] == "cuda"
        assert kwargs["batch_size"] == 4
        return extractor

    def extract(**kwargs: Any) -> list[WRFeatures]:
        calls.append("features")
        if feature_type != "regionprops2":
            assert kwargs["feature_extractor"] is extractor
            assert extractor.force_recompute
        return features

    def build(*args: Any, **kwargs: Any) -> list[PredictionWindow]:
        calls.append("windows")
        return windows

    def resume(*args: Any, **kwargs: Any) -> Predictions:
        calls.append("prediction")
        assert kwargs["batch_size"] == 4
        return {"nodes": [], "weights": ()}

    monkeypatch.setattr(preparation, "get_features", extract)
    monkeypatch.setattr(preparation, "build_windows", build)
    monkeypatch.setattr(adapter, "predict_windows", resume)
    monkeypatch.setattr(model_api, "PRETRAINED_FEATS_INSTALLED", True)
    monkeypatch.setattr(model_api, "FeatureExtractor",
                        SimpleNamespace(from_model_name=create_extractor), raising=False)
    model = adapter.ResumableTrackastra(transformer, {"features": feature_type,
                                    "pretrained_feats_model": "test-model",
                                    "pretrained_feats_mode": "test-mode"},
                       device="cpu", batch_size=4)
    model.device = "cuda"  # Simulate the wrapper before CPU fallback.
    model.track(np.zeros((12, 32, 32)), np.zeros((12, 32, 32)),
                progbar_class=QuietProgress)
    expected = ["features", "windows", "prediction"]
    assert calls == (expected if feature_type == "regionprops2" else ["extractor", *expected])
    assert model.device == "cpu"


def test_adapter_does_not_repeat_preparation_during_retries(
        prepared: Prepared, transformer: TrackingTransformer,
        monkeypatch: pytest.MonkeyPatch) -> None:
    preparation_calls: list[bool] = []
    prediction_calls: list[tuple[int, str]] = []

    def prepare(*args: Any, **kwargs: Any) -> Prepared:
        preparation_calls.append(True)
        assert kwargs["normalize_imgs"] is False
        assert kwargs["n_workers"] == 2
        return prepared

    def fail_on_cuda(batch: list[PredictionWindow], model: FakeTransformer) -> NDArray[np.float32]:
        prediction_calls.append((len(batch), model.device))
        if model.device == "cuda":
            raise torch.cuda.OutOfMemoryError("simulated CUDA OOM")
        return fake_scores(batch, model)

    monkeypatch.setattr(adapter, "prepare_features", prepare)
    monkeypatch.setattr(prediction, "predict", fail_on_cuda)
    monkeypatch.setattr(prediction.torch.cuda, "empty_cache", lambda: None)
    model = adapter.ResumableTrackastra(transformer, {"features": "regionprops2"},
                                       device="cpu", batch_size=4)
    model.transformer = cast(TrackingTransformer, FakeTransformer())
    model.device = "cuda"
    actual = model._predict(np.zeros((12, 32, 32)), np.zeros((12, 32, 32)),
                            normalize_imgs=False, n_workers=2, progbar_class=QuietProgress)
    assert preparation_calls == [True]
    assert prediction_calls[:4] == [(4, "cuda"), (3, "cuda"), (2, "cuda"), (1, "cuda")]
    assert model.device == "cpu"
    expected = run_prediction(prepared, FakeTransformer("cpu"), batch_size=1)
    assert_predictions_equal(cast(Predictions, actual), expected)


@pytest.mark.parametrize("normalize_imgs", [False, True])
def test_dask_preparation_matches_upstream(
        transformer: TrackingTransformer, normalize_imgs: bool) -> None:
    masks = np.zeros((5, 32, 32), dtype=np.uint16)
    masks[:, 8:12, 8:12] = 1
    images = masks.astype(np.float32)
    dask_images = da.from_array(images)
    dask_masks = da.from_array(masks)
    upstream = Trackastra(deepcopy(transformer), {"features": "regionprops2"}, device="cpu")
    resumable = adapter.ResumableTrackastra(transformer, {"features": "regionprops2"}, device="cpu")
    expected = upstream._predict(dask_images, dask_masks, normalize_imgs=normalize_imgs,
                                  progbar_class=QuietProgress)
    actual = resumable._predict(dask_images, dask_masks, normalize_imgs=normalize_imgs,
                                progbar_class=QuietProgress)
    assert_predictions_equal(cast(Predictions, actual), cast(Predictions, expected))


@pytest.mark.parametrize("mode", ["greedy", "greedy_nodiv"])
def test_full_tracking_matches_upstream(transformer: TrackingTransformer, mode: Mode) -> None:
    masks = np.zeros((8, 32, 32), dtype=np.uint16)
    for time in range(8):
        masks[time, 5:8, 5 + time:8 + time] = 1
        masks[time, 20:23, 20:23] = 2
    image = masks.astype(np.float32)
    upstream = Trackastra(deepcopy(transformer), {"features": "regionprops2"},
                          device="cpu", batch_size=4)
    resumable = adapter.ResumableTrackastra(transformer, {"features": "regionprops2"},
                                           device="cpu", batch_size=4)
    expected_graph, expected_masks = upstream.track(image, masks, mode=mode,
                                                    progbar_class=QuietProgress)
    graph, tracked_masks = resumable.track(image, masks, mode=mode,
                                                    progbar_class=QuietProgress)
    assert dict(graph.nodes(data=True)) == dict(expected_graph.nodes(data=True))
    assert list(graph.edges(data=True)) == list(expected_graph.edges(data=True))
    np.testing.assert_array_equal(tracked_masks, expected_masks)
