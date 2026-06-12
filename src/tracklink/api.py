from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
from numpy.typing import NDArray

from tracklink.backends import get_backend
from tracklink.backends.protocol import TrackingBackend


class TrackModel:
    """
    Class representing a tracking model. This is a placeholder for the actual implementation.
    """
    def __init__(self, backend: str = "trackastra") -> None:
        self._backend: TrackingBackend = get_backend(backend)
        self.masks_tracked: NDArray[Any] | None = None
    
    
    def configure(self, user_settings: Mapping[str, Any]) -> None:
        """
        Configure the tracking model with user-provided settings.
        """
        self._backend.configure(user_settings)
    
    
    def track(self, img_array: NDArray[Any], mask_array: NDArray[Any]) -> NDArray[Any]:
        """
        Perform tracking on the given image and mask arrays.

        Parameters:
            img_array: The input image array.
            mask_array: The input mask array where each unique value corresponds to a track label. 
        
        Returns:
            A mask array with tracked labels.
        
        Note:
            The tracks DataFrame should have normalized column names according to a standardized mapping and should compute a 'length' column representing the duration of each track in frames.
        """
        self._masks_tracked = self._backend.track(img_array, mask_array)
        return self._masks_tracked
    
    
    def filter_by_length(self, min_length: int = 0, max_length: int | None = None) -> NDArray[Any]:
        """
        Filter the tracked mask array based on track lengths specified in the tracks DataFrame.
        
        Parameters:
            min_length: Minimum track length to keep (inclusive). Default is 0.
            max_length: Maximum track length to keep (inclusive). If None, no upper limit is applied. Default is None.
        
        Returns:
            A filtered mask array where tracks that do not meet the length criteria are set to 0.
        """
        return self._backend.filter_by_length(min_length=min_length, max_length=max_length)
    
    @property
    def tracks_df(self) -> pd.DataFrame:
        """
        DataFrame containing track information with normalized column names and a 'length' column.
        Note: If tracking has not been performed yet, this will return an empty DataFrame.
        """
        return self._backend.tracks_df
    

if __name__ == "__main__":
    from pathlib import Path
    from fits_io import FitsIO
    
    user_settings = {'pretrained_model': "general_2d", 
                     'mode': "greedy_nodiv", 
                     'max_distance': 128}
    
    folder = Path("/media/ben/Analysis/Python/Images/zymosan/zym_chamber_500k_WT_HoxB8_001_s1")
    arrays = FitsIO.from_path(folder / "fits_array.tif")
    BFP_array = arrays.get_channel_array('BFP')
    if isinstance(BFP_array, list):
        BFP_array = BFP_array[0]
    
    masks = FitsIO.from_path(folder / "fits_mask.tif")
    BFP_mask = masks.get_array()
    if isinstance(BFP_mask, list):
        BFP_mask = BFP_mask[0]
    
    tracking = TrackModel(backend="trackastra")
    tracking.configure(user_settings)
    tracked_mask = tracking.track(BFP_array, BFP_mask)
    
    filtered_mask = tracking.filter_by_length(min_length=120)
    
    tracks_df = tracking.tracks_df
    tracks_df.to_csv(folder / "fits_tracks.csv", index=False)
    
    masks.save_array(filtered_mask,
                     axis_order="TYX",
                     channel_labels="BFP",
                     output_name="fits_track.tif",)