# tracklink

`tracklink` links segmented objects through time and returns a label movie in
which each tracked object keeps a stable identity. Its `TrackModel` façade
currently delegates to Trackastra, normalizes the resulting track table, and
can remove tracks outside a requested lifetime range.

```python
from tracklink.api import TrackModel

model = TrackModel(backend="trackastra")
model.configure({"pretrained_model": "general_2d"})
tracked_mask = model.track(image_stack, segmentation_stack)
long_lived_mask = model.filter_by_length(min_length=20)
tracks = model.tracks_df
```

The package works with image and mask arrays. It does not discover experiments
or save outputs, which keeps the tracking backend usable independently of FITS.

## Role in FITS

FITS selects matching image and segmentation channels, supplies backend
settings, merges newly tracked channels with existing results, and writes the
tracked-mask artifact. `tracklink` is responsible only for linking objects and
filtering the resulting tracks.
