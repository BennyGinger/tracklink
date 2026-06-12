from collections.abc import Mapping

import pandas as pd


# Define constants for standardized column names
LABEL = "label"
FRAME_START = "f_start"
FRAME_END = "f_end"
PARENT = "parent"
LENGTH = "length"


def normalize_tracks_df(tracks_df: pd.DataFrame, col_mapping: Mapping[str, str]) -> pd.DataFrame:
    """
    Normalize the tracks DataFrame by renaming columns according to a provided mapping.

    Parameters:
    - tracks_df: The input DataFrame containing track information.
    - col_mapping: A dictionary mapping existing column names to new standardized column names.

    Returns:
    - A normalized DataFrame with renamed columns.
    """
    # Check if all required columns are present in the DataFrame
    missing_cols = [col for col in col_mapping.keys() if col not in tracks_df.columns]
    if missing_cols:
        raise ValueError(f"The following required columns are missing from the DataFrame: {missing_cols}")
    
    # Rename columns according to the provided mapping
    renamed_df = tracks_df.rename(columns=col_mapping)
    
    # Compute track length
    normalized_df = _compute_track_length(renamed_df)
    
    return normalized_df


def _compute_track_length(tracks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the length of each track and add it as a new column in the DataFrame.

    Parameters:
    - tracks_df: The input DataFrame containing track information, which must include 'FRAME_START' and 'FRAME_END' columns.

    Returns:
    - A DataFrame with an additional 'LENGTH' column representing the duration of each track.
    """
    if FRAME_START not in tracks_df.columns or FRAME_END not in tracks_df.columns:
        raise ValueError(f"The DataFrame must contain '{FRAME_START}' and '{FRAME_END}' columns to compute track length.")
    
    tracks_df[LENGTH] = tracks_df[FRAME_END] - tracks_df[FRAME_START]
    
    return tracks_df