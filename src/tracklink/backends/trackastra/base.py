from typing import Literal, Any

from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc
from numpy.typing import NDArray
import pandas as pd

PretrainedModel = Literal["ctc", "general_2d", "general_2d_w_SAM2_features"]
Mode = Literal["greedy", "greedy_nodiv", "ilp"]


def track_astra(img_array: NDArray[Any], mask_array: NDArray[Any], mode: Mode = "greedy_nodiv", pretrained_model: PretrainedModel = "general_2d", max_distance: int = 128) -> tuple[pd.DataFrame, NDArray[Any]]:
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
    
    # Initialize the Trackastra model
    model = Trackastra.from_pretrained(pretrained_model)
    
    # Perform tracking
    track_graph, masks_tracked = model.track(img_array, mask_array, mode=mode, max_distance=max_distance)
    
    df_tracks, ctc_masks = graph_to_ctc(track_graph, masks_tracked,)
    
    return df_tracks, ctc_masks