"""Thin Trackastra adapter separating preparation from resumable prediction.

Only _predict is overridden. Upstream track() retains optional feature-extractor
setup, graph linking, and mask relabelling. Loading/table conversion stay in base.py.
"""

from __future__ import annotations

from typing import Any

import dask.array as da
from numpy.typing import NDArray
from trackastra.model import Trackastra
from tqdm import tqdm

from .features import prepare_features
from .prediction import predict_windows


class ResumableTrackastra(Trackastra):
    """
    Use upstream tracking with batch-local CUDA OOM recovery in prediction.
    """
    def _predict(self,
                 imgs: NDArray[Any] | da.Array,
                 masks: NDArray[Any] | da.Array,
                 edge_threshold: float = 0.05,
                 n_workers: int = 0,
                 normalize_imgs: bool = True,
                 progbar_class: type[tqdm] = tqdm,
                 batch_size: int | None = None,
                 ) -> dict[str, Any]:
        """
        Prepare once; retries operate exclusively on the prepared windows.
        """
        self.transformer.eval()
        features, windows = prepare_features(self, 
                                             imgs, 
                                             masks, 
                                             normalize_imgs=normalize_imgs,
                                             n_workers=n_workers, 
                                             progbar_class=progbar_class,)
        try:
            predictions = predict_windows(windows, 
                                          features, 
                                          self.transformer,
                                          spatial_dim=masks.ndim - 1,
                                          batch_size=self.batch_size if batch_size is None else batch_size,
                                          edge_threshold=edge_threshold,
                                          progbar_class=progbar_class,)
            # Match Trackastra's plain-dict return contract at the override
            # boundary; the prediction stage retains its TypedDict schema.
            return dict(predictions)
        finally:
            # Recovery can move the transformer to CPU, even before a later error.
            self.device = str(next(self.transformer.parameters()).device)
