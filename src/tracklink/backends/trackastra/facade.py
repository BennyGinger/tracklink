from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd
from numpy.typing import NDArray

from tracklink.backends.dataframe import LABEL, FRAME_START, FRAME_END, PARENT, normalize_tracks_df
from tracklink.backends.protocol import BackendConfig
from tracklink.backends.trackastra.base import PretrainedModel, Mode, track_astra
from tracklink.backends.trackastra.filtration import filter_len_mask


TRACKASTRA_COL_MAPPING = {
    "label": LABEL,
    "t1": FRAME_START,
    "t2": FRAME_END,
    "parent": PARENT,
}


@dataclass(frozen=True)
class TrackAstraConfig(BackendConfig):
    pretrained_model: PretrainedModel = "general_2d"
    mode: Mode = "greedy_nodiv"
    max_distance: int = 128


@dataclass
class TrackAstra:
    _config: TrackAstraConfig = field(init=False)
    _tracks_df: pd.DataFrame = field(init=False)
    masks_tracked: NDArray[Any] = field(init=False)
    
    def configure(self, user_settings: Mapping[str, Any]) -> None:
        self._config = TrackAstraConfig.from_mapping(user_settings)
        
    
    def track(self, img_array: NDArray[Any], mask_array: NDArray[Any]) -> NDArray[Any]:
        """Perform tracking using the Trackastra model and return the tracks DataFrame and tracked mask array.
        
        Parameters:
            img_array: The input image array.
            mask_array: The input mask array where each unique value corresponds to a track label.
            
        Returns:
            A mask array with tracked labels.
        """
        tracks_df, self.masks_tracked = track_astra(
            img_array,
            mask_array,
            mode=self._config.mode,
            pretrained_model=self._config.pretrained_model,
            max_distance=self._config.max_distance,)
        
        # Normalize column names and compute track length
        self._tracks_df = normalize_tracks_df(tracks_df, TRACKASTRA_COL_MAPPING)
        
        return self.masks_tracked
    
    
    def filter_by_length(self, min_length: int = 0, max_length: Optional[int] = None) -> NDArray[Any]:
        """
        Filter the tracked mask array based on track lengths specified in the tracks DataFrame.
        
        Parameters:
            min_length: Minimum track length to keep (inclusive). Default is 0.
            max_length: Maximum track length to keep (inclusive). If None, no upper limit is applied. Default is None.
        
        Returns:
            A filtered mask array where tracks that do not meet the length criteria are set to 0.
        """
        
        filtered_mask = filter_len_mask(self.masks_tracked, 
                                        self._tracks_df, 
                                        min_length=min_length, 
                                        max_length=max_length)
        return filtered_mask
        
    @property
    def tracks_df(self) -> pd.DataFrame:
        """
        DataFrame containing track information with normalized column names and a 'length' column.
        Note: If tracking has not been performed yet, this will return an empty DataFrame.
        """
        if not hasattr(self, "_tracks_df"):
            return pd.DataFrame()
        return self._tracks_df