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

## GPU memory recovery

Trackastra prediction automatically recovers from CUDA out-of-memory errors.
FITS and other callers continue using the same `TrackModel.track()` API; recovery
is entirely inside tracklink's Trackastra backend.

A **window** contains consecutive frames across the whole field of view. Its
length comes from the model configuration: `general_2d` currently uses four
frames. Windows slide by one frame, so 271 frames produce 268 windows. A **batch**
is a group of windows predicted together. Batches do not repeat windows, although
the windows themselves overlap in frames. Decreasing batch size preserves the
model's temporal context and includes every window and detection.

The backend starts with Trackastra's default batch size (4 on CUDA in the
verified version). Recovery works as follows:

1. Normalize images, extract features, and construct windows once.
2. Predict a batch, accumulate its association scores in host RAM, and only then
   advance the next-window cursor.
3. On CUDA OOM, discard the failed attempt's temporary tensors, release unused
   GPU cache, decrease the batch size by one, and retry at the same cursor:
   **4 → 3 → 2 → 1**. Keep the smaller size for the remainder of this call.
4. If one window still cannot fit, move the transformer to CPU and finish the
   remaining windows one at a time. Previously accumulated scores are retained.
5. Use Trackastra's graph solver and mask relabelling after all predictions finish.

For example, if windows 1–4 finish but predicting windows 5–8 runs out of memory,
the next attempt predicts windows 5–7. Windows 1–4 are not recomputed or counted
again. A failed batch is treated as entirely unfinished, regardless of how much
GPU computation happened before the exception. If only CPU is available, normal
CPU prediction is used directly.

Warnings identify the next window (one-based), failed/new batch size, and number
of completed windows retained. Search a FITS log for `CUDA out of memory` or
`continuing on CPU`; the logger is
`tracklink.backends.trackastra.prediction`.

Recovery covers **association prediction only**. Model loading and optional
pretrained-feature extraction still use upstream behaviour. Other errors,
including host-memory exhaustion and errors during CPU fallback, propagate.
CPU fallback can be much slower and needs sufficient system RAM. Retained
progress lasts for this tracking call; there is no checkpoint for process
termination or restarting FITS. A new channel/call starts with the upstream
default again. Batch changes and CPU execution may introduce small numerical
differences; near a score threshold these can affect a final link.

## Implementation responsibilities and Trackastra upgrades

This adapter was developed and verified against **Trackastra 0.5.6**. The
dependency remains **unpinned** deliberately. It uses internal interfaces, so
an upstream update may require a small adapter change.

| Module | Responsibility |
| --- | --- |
| `api.py` | Public `TrackModel` API and backend delegation. |
| `backends/trackastra/facade.py` | Backend settings, track-table normalization, and lifetime filtering. |
| `backends/trackastra/base.py` | Load the model, invoke tracking, and convert graph/masks to the output table. |
| `backends/trackastra/model.py` | Thin `ResumableTrackastra` adapter: override `_predict` to call preparation once, then resumable prediction. |
| `backends/trackastra/features.py` | Normalize images, extract features, and construct windows using upstream helpers. No prediction retries or linking. |
| `backends/trackastra/prediction.py` | Consume prepared features/windows, predict batches, recover from CUDA OOM, and accumulate association scores. |

Preparation returns typed windows and detection features. Prediction consumes
them and returns a typed `Predictions` dictionary of detection nodes and averaged
association weights. Its retry loop never calls feature preparation. The adapter
then returns those scores to Trackastra's inherited `track()` method, which owns
CPU graph linking and mask relabelling. There is no separate linking module or
custom linking coordinator. Optional pretrained-feature extractor setup also
stays in upstream `track()`.

Upstream still owns feature extraction (`get_features`), window construction
(`build_windows`), neural prediction (`trackastra.model.predict.predict`), graph
construction/solving, and mask relabelling. No installed Trackastra files are
modified and no upstream functions are globally patched. The preparation and
accumulation logic adapted from upstream carries its license in
`backends/trackastra/TRACKASTRA_LICENSE.txt`.

If tracking breaks after updating Trackastra, check these boundaries first:

1. Compare upstream `Trackastra._predict` in `trackastra/model/model_api.py` with
   `features.prepare_features()` and the adapter in `model.py`: normalization,
   preparation arguments, window configuration, and the override's signature.
   Check that upstream `track()` still sets up optional pretrained features,
   calls `_predict`, and consumes its `nodes`/`weights` result before linking.
2. Compare upstream `predict` and `predict_windows` in
   `trackastra/model/predict.py` with our `prediction.py`: window keys, tensor
   shapes, returned batch scores, score thresholding, and averaging. We preserve
   upstream defaults `delta_t=1` and `intra_window_weight=0`; these describe score
   accumulation, independently of graph-linking settings such as `max_distance`.
3. Run the focused tests below. They compare real neural predictions and full
   tracking outputs against upstream, and simulate OOM after successful batches
   to verify resumption, averaging, CPU fallback, and error propagation. An adapter
   test also forces every GPU batch size to fail and verifies that preparation
   runs once across all retries and CPU continuation.
4. After adapting, update the verified version here and in the module docstrings.
   Verify recovery on a real CUDA experiment as well as the CPU tests.

From the FITS workspace root, using its existing environment:

```bash
.venv/bin/python -m pytest -q tracklink/tests
```

For an independent tracklink checkout, run `python -m pytest -q tests` in an
environment containing tracklink, Trackastra, and pytest. The automated recovery
tests simulate CUDA failures and do not require a GPU or download model weights.
