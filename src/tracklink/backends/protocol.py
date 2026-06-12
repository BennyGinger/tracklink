from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Optional, Protocol, Any, Self

import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class BackendConfig:
    @classmethod
    def from_mapping(cls, settings: Mapping[str, Any]) -> Self:
        valid_names = {field.name for field in fields(cls)}
        kwargs = {
            key: value
            for key, value in settings.items()
            if key in valid_names
        }
        return cls(**kwargs)


class TrackingBackend(Protocol):
    """
    Protocol for tracking backends. Any tracking backend should implement this interface.
    """

    def configure(self, user_settings: Mapping[str, Any]) -> None:
        """Configure the backend with user-provided settings."""
        ...
    
    
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
        ...
    
    
    def filter_by_length(self, min_length: int = 0, max_length: Optional[int] = None) -> NDArray[Any]:
        """
        Filter the tracked mask array based on track lengths specified in the tracks DataFrame.

        Parameters:
            min_length: Minimum track length to keep (inclusive).
            max_length: Maximum track length to keep (inclusive). If None, no upper limit is applied.

        Returns:
            A filtered mask array where tracks that do not meet the length criteria are set to 0.
        """
        ...
        
    @property
    def tracks_df(self) -> pd.DataFrame:
        """
        DataFrame containing track information with normalized column names and a 'length' column.
        """
        ...