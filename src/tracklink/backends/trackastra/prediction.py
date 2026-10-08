"""Resumable association prediction, adapted from Trackastra 0.5.6.

Feature preparation lives in features.py. This module consumes prepared windows,
then owns batch retries and CPU-side accumulation of successful predictions.
Trackastra still owns each neural prediction. Keep score averaging aligned with
``trackastra.model.predict`` on upgrades.
Adapted portions: Copyright (c) 2024, Benjamin Gallusser, Martin Weigert.
See TRACKASTRA_LICENSE.txt in this directory for the BSD-3-Clause terms.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict, cast

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.sparse import csr_array
from trackastra.data import WRFeatures
from trackastra.model import TrackingTransformer
from trackastra.model.predict import predict
from tqdm import tqdm

from .features import PredictionWindow

logger = logging.getLogger(__name__)


class PredictionNode(TypedDict):
    """
    A single detection, with a global ID for graph construction.
    """
    id: int
    time: int
    label: int
    coords: tuple[float, ...]


class Predictions(TypedDict):
    """
    Stage boundary shared by association prediction and graph linking.
    """
    nodes: list[PredictionNode]
    weights: tuple[tuple[tuple[int, int], float], ...]


def _try_predict(batch: list[PredictionWindow], 
                 model: TrackingTransformer,
                 ) -> NDArray[np.float32] | None:
    """
    Return None only for CUDA OOM; unwind its traceback before cleanup.

    The exception traceback retains prediction tensors while inside the handler.
    Returning a sentinel lets the caller release the cache after those references
    have gone. CPU OOM and all unrelated failures propagate to the caller.
    """
    try:
        # Upstream types windows as plain dicts; our schema describes those dicts.
        return predict(cast(list[dict[str, Any]], batch), model)
    except torch.cuda.OutOfMemoryError:
        if next(model.parameters()).device.type != "cuda":
            raise
        return None


def _accumulate(batch: list[PredictionWindow], 
                scores: NDArray[np.float32],
                node_ids: dict[tuple[int, int], int], 
                sums: csr_array,
                counts: csr_array, 
                edge_threshold: float
                ) -> None:
    """
    Commit a successful batch using Trackastra's default edge semantics.

    Only forward links between adjacent frames are retained (delta_t=1), and all
    windows have equal weight (intra_window_weight=0), matching the upstream API.
    """
    if len(scores) != len(batch):
        raise ValueError("Trackastra returned a different number of window predictions.")
    for window, score in zip(batch, scores, strict=True):
        times = window["timepoints"].numpy()
        labels = window["labels"].numpy()
        dt = times[None, :] - times[:, None]
        score = score[:len(times), :len(times)]
        score[~((dt > 0) & (dt <= 1))] = 0
        ii, jj = np.where(score >= edge_threshold)
        if not len(ii):
            continue
        source = np.array([node_ids[(t, label)] for t, label in zip(times[ii], labels[ii])])
        target = np.array([node_ids[(t, label)] for t, label in zip(times[jj], labels[jj])])
        sums[source, target] += score[ii, jj]
        counts[source, target] += 1


def predict_windows(windows: list[PredictionWindow], 
                    features: list[WRFeatures],
                    model: TrackingTransformer,
                    spatial_dim: int, 
                    batch_size: int,
                    edge_threshold: float = 0.05,
                    progbar_class: type[tqdm] = tqdm
                    ) -> Predictions:
    """
    Predict once per window, decreasing CUDA batches by one on OOM.

    The cursor advances only after scores are accumulated. Failed batches have no
    committed results and are retried at the same cursor. At batch size one, move
    the transformer to CPU and finish there. Completed scores remain in host RAM;
    this is recovery within this call, not an on-disk checkpoint across runs.
    """
    if batch_size < 1:
        raise ValueError("Prediction batch size must be positive.")
    nodes: list[PredictionNode] = []
    node_ids: dict[tuple[int, int], int] = {}
    for feature in features:
        for time, label, coord in zip(feature.timepoints, feature.labels, feature.coords):
            node_ids[(int(time), int(label))] = len(nodes)
            nodes.append(PredictionNode(id=len(nodes), time=int(time), label=int(label),
                                         coords=tuple(float(v) for v in coord[-spatial_dim:])))
    sums = csr_array((len(nodes), len(nodes)), dtype=np.float32)
    counts = csr_array((len(nodes), len(nodes)), dtype=np.float32)
    cursor = 0
    logger.info("Predicting %d windows with batch size %d on %s", len(windows),
                batch_size, next(model.parameters()).device)
    with progbar_class(total=len(windows), desc="Computing associations") as progress:
        while cursor < len(windows):
            batch = windows[cursor:cursor + batch_size]
            if any(len(window["timepoints"]) for window in batch):
                scores = _try_predict(batch, model)
                if scores is None:
                    # _try_predict has returned, releasing its exception traceback.
                    torch.cuda.empty_cache()
                    if batch_size > 1:
                        previous_size = batch_size
                        batch_size -= 1
                        logger.warning("CUDA out of memory at window %d/%d with batch "
                                       "size %d; retrying with %d. Completed windows retained: %d.",
                                       cursor + 1, len(windows), previous_size, batch_size, cursor)
                    else:
                        logger.warning("CUDA out of memory at window %d/%d with batch "
                                       "size 1; continuing on CPU. Completed windows retained: %d.",
                                       cursor + 1, len(windows), cursor)
                        model.to("cpu")
                        torch.cuda.empty_cache()
                    continue
                _accumulate(batch, scores, node_ids, sums, counts, edge_threshold)
                del scores
            cursor += len(batch)
            progress.update(len(batch))

    sums_coo, counts_coo = sums.tocoo(), counts.tocoo()
    if (not np.array_equal(sums_coo.row, counts_coo.row)
            or not np.array_equal(sums_coo.col, counts_coo.col)):
        raise ValueError("Association sums and counts have inconsistent indices.")

    # Overlapping windows can predict the same connection more than once.
    # Give each connection one final weight: its average predicted score.
    weights: list[tuple[tuple[int, int], float]] = []
    for source_id, target_id, total_score, observation_count in zip(sums_coo.row,
                                                                    sums_coo.col,
                                                                    sums_coo.data,
                                                                    counts_coo.data,
                                                                    strict=True,):
        node_pair = (int(source_id), int(target_id))
        average_score = float(total_score / observation_count)
        weights.append((node_pair, average_score))

    return Predictions(nodes=nodes, weights=tuple(weights))
