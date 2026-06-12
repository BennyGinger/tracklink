from typing import Any, Optional

from numpy.typing import NDArray
import numpy as np
import pandas as pd

from tracklink.backends.dataframe import LABEL, LENGTH


def filter_len_mask(mask_array: NDArray[Any], tracks_df: pd.DataFrame, min_length: int = 0, max_length: Optional[int] = None) -> NDArray[Any]:
    """
    Filter a mask array based on track lengths specified in a DataFrame.

    Parameters:
    - mask_array: The input mask array where each unique value corresponds to a track label.
    - tracks_df: A DataFrame containing track information, including 'label' and 'length' columns.
    - min_length: Minimum track length to keep (inclusive).
    - max_length: Maximum track length to keep (inclusive). If None, no upper limit is applied.

    Returns:
    - A filtered mask array where tracks that do not meet the length criteria are set to 0.
    """
    
    # Identify labels of tracks that do not meet the length criteria
    if max_length is not None and isinstance(max_length, int):
        labels_to_remove = tracks_df[(tracks_df[LENGTH] < min_length) | (tracks_df[LENGTH] > max_length)][LABEL].tolist()
    else:
        labels_to_remove = tracks_df[tracks_df[LENGTH] < min_length][LABEL].tolist()
    
    # Create a copy of the mask array to modify
    filtered_mask = mask_array.copy()
    
    # Set pixels corresponding to unwanted labels to 0
    filtered_mask[np.isin(filtered_mask, labels_to_remove)] = 0
    
    return filtered_mask