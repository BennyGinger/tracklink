"""Prepare feature windows once before resumable neural prediction.

Reuse Trackastra's normalization, feature extraction, and window construction.
Preparation follows Trackastra 0.5.6's _predict; it never retries prediction or
solves tracks. Optional pretrained-feature setup remains in upstream track().
Adapted portions: Copyright (c) 2024, Benjamin Gallusser, Martin Weigert.
See TRACKASTRA_LICENSE.txt in this directory for the BSD-3-Clause terms.
"""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict, cast

import dask.array as da
import torch
from numpy.typing import NDArray
from trackastra.data import WRFeatures, build_windows, get_features
from trackastra.model import Trackastra
from trackastra.utils import normalize
from tqdm import tqdm


class PredictionWindow(TypedDict):
    """Tensor window produced by upstream build_windows(as_torch=True)."""

    t1: torch.Tensor
    coords: torch.Tensor
    timepoints: torch.Tensor
    labels: torch.Tensor
    features: torch.Tensor | None
    pretrained_feats: NotRequired[torch.Tensor | None]


def prepare_features(model: Trackastra,
                     images: NDArray[Any] | da.Array,
                     masks: NDArray[Any] | da.Array,
                     normalize_imgs: bool = True,
                     n_workers: int = 0,
                     progbar_class: type[tqdm] = tqdm,
                     ) -> tuple[list[WRFeatures], list[PredictionWindow]]:
    """Normalize, extract detections' features, and construct temporal windows.

    Return CPU-side features/windows for the entire time series. The adapter
    calls this once, outside the prediction retry loop. Preserve upstream's
    normalization and worker options, including its Dask preparation path.
    """
    if normalize_imgs:
        if isinstance(images, da.Array):
            images = images.map_blocks(normalize)
        else:
            images = cast(NDArray[Any], normalize(images))
    # Upstream _predict passes NumPy or Dask, although get_features annotates
    # only NumPy. Its per-frame helper materializes either via np.asarray.
    # It may also return a tuple; expose a consistent list to our prediction stage.
    features: list[WRFeatures] = list(get_features(detections=cast(Any, masks),
                                                   imgs=cast(Any, images),
                                                   features=model.train_args["features"],
                                                   feature_extractor=model.feature_extractor,
                                                   ndim=int(model.transformer.config["coord_dim"]),
                                                   n_workers=n_workers,
                                                   progbar_class=progbar_class,))
    windows = cast(list[PredictionWindow], build_windows(features,
                                                         window_size=int(model.transformer.config["window"]),
                                                         progbar_class=progbar_class,
                                                         as_torch=True,))
    return features, windows
