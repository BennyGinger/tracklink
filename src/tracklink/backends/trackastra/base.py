from typing import Literal, Any
from contextlib import ExitStack
from importlib import import_module
from pathlib import Path
from unittest.mock import patch
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r"urllib3 .* doesn't match a supported version!",
        module=r"requests",
    )
    from trackastra.model import Trackastra
    from trackastra.tracking import (apply_solution_graph_to_masks,
                                     build_graph,
                                     graph_to_ctc,
                                     track_greedy,)
from numpy.typing import NDArray
import pandas as pd
from platformdirs import user_data_dir
from scipy.sparse import SparseEfficiencyWarning
from tqdm import tqdm

PretrainedModel = Literal["ctc", "general_2d", "general_2d_w_SAM2_features"]
Mode = Literal["greedy", "greedy_nodiv", "ilp"]


class _QuietTqdm(tqdm):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


def _quiet_module_tqdm(stack: ExitStack, function: Any) -> None:
    """
    Replace the tqdm symbol in the module defining `function`, when present.
    """
    module = import_module(function.__module__)

    if hasattr(module, "tqdm"):
        stack.enter_context(
            patch.object(module, "tqdm", _QuietTqdm)
        )


def track_astra(img_array: NDArray[Any], 
                mask_array: NDArray[Any], 
                mode: Mode = "greedy_nodiv", 
                pretrained_model: PretrainedModel = "general_2d", 
                max_distance: int = 128
                ) -> tuple[pd.DataFrame, NDArray[Any]]:
    """Perform tracking using the Trackastra model.
    
    Args:
        img_array: The input image array.
        mask_array: The input mask array where each unique value corresponds to a track label.
        mode: The tracking mode to use. Options are "greedy", "greedy_nodiv", and "ilp". Default is "greedy_nodiv".
        pretrained_model: The pretrained model to use. Options are "ctc", "general_2d", and "general_2d_w_SAM2_features". Default is "general_2d".
        max_distance: The maximum distance (in pixels) to consider for linking tracks between frames. Default is 128.
    
    Returns:
        A tuple containing a DataFrame with track information and a mask array with tracked labels.
    """
    with ExitStack() as stack:
        # These functions use their own directly imported tqdm.
        _quiet_module_tqdm(stack, build_graph)
        _quiet_module_tqdm(stack, track_greedy)
        _quiet_module_tqdm(stack, graph_to_ctc)
        _quiet_module_tqdm(stack, apply_solution_graph_to_masks)
    
    # Loading a cached model through ``from_pretrained`` produces an
    # unconditional print in Trackastra. Bypass its downloader when the model
    # is already available, without redirecting stdout used by the pipeline UI.
    model_dir = Path(user_data_dir("trackastra")) / "models" / pretrained_model
    if model_dir.exists():
        model = Trackastra.from_folder(model_dir)
    else:
        model = Trackastra.from_pretrained(pretrained_model)
    
    # Perform tracking
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SparseEfficiencyWarning)
        track_graph, masks_tracked = model.track(img_array,
                                                 mask_array,
                                                 mode=mode,
                                                 max_distance=max_distance,
                                                 progbar_class=_QuietTqdm)

        df_tracks, ctc_masks = graph_to_ctc(track_graph, masks_tracked,)
    
    return df_tracks, ctc_masks
