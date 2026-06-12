from __future__ import annotations

from .protocol import TrackingBackend


def get_backend(name: str = "trackastra") -> TrackingBackend:
    """
    Return a tracking backend instance by name.
    """
    normalized = name.lower()
    if normalized == "trackastra":
        from .trackastra.facade import TrackAstra
        return TrackAstra()
    
    raise ValueError(f"Unsupported backend '{name}'. Available backends: 'trackastra'.")