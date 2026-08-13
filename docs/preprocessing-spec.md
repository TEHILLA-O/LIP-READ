# Preprocessing specification

The input contract for the Auto-AVSR visual speech model. These values are not
tunable: they reproduce the preprocessing the published checkpoints were trained
with. Changing one without retraining does not produce an error — it produces a
model that runs normally and returns confident, wrong transcripts.

Derived from `mpc001/auto_avsr` (`preparation/detectors/mediapipe/video_process.py`
and `datamodule/transforms.py`), checked against the checkpoint's own tensor
shapes, and verified against LRS3 ground truth in
`scripts/evaluate_lrs3.py`.

## Summary

| Property | Value |
| --- | --- |
| Frame rate | 25 fps, exactly |
| Colour | Grayscale, BT.601 luma |
| Aligned face | 256 × 256 |
| Mouth crop | 96 × 96 |
| Network input | 88 × 88 (centre crop of the mouth crop) |
| Pixel range | `[0, 1]`, then normalised |
| Normalisation | mean `0.421`, std `0.165` |
| Tensor layout | `(B, T, 1, 88, 88)` float32 |
| Vocabulary | SentencePiece unigram, 5,049 tokens |

## Pipeline

```
video file / camera frames
  -> decode to RGB                        ml/preprocessing/video_io.py
  -> resample to exactly 25 fps           resample_to_fps
  -> scale long side to 640 px            (cost/memory cap, not a model requirement)
  -> detect face + 4 keypoints            ml/preprocessing/face_detector.py
  -> track one speaker across frames      FaceTracker
  -> interpolate undetected frames        ml/preprocessing/landmarks.py
  -> smooth landmarks over +/-6 frames    smooth_landmarks
  -> similarity-warp to the mean face     ml/preprocessing/alignment.py
  -> cut a 96x96 patch at the mouth       cut_patch
  -> centre crop to 88x88                 ml/preprocessing/tensor.py
  -> grayscale, scale, normalise          rois_to_tensor
  -> (B, T, 1, 88, 88)                    add_batch_dimension
```

## Frame rate

The model has no time base other than frame index, so 25 fps is a hard
requirement: feeding 30 fps makes every utterance play 20% slow to the model.

Conversion picks, for each target timestamp, whichever decoded frame is nearest
in time. Target times are computed as `start + n / 25` rather than accumulated,
because repeated addition drifts enough to drop frames over a clip of any
length.

## Face detection and keypoints

MediaPipe's BlazeFace detector, full-range model with the short-range model as a
per-frame fallback. Auto-AVSR aligns on exactly four points, and this is the
required order:

| Index | Point |
| --- | --- |
| 0 | Right eye |
| 1 | Left eye |
| 2 | Nose tip |
| 3 | Mouth centre |

Index 3 doubles as the crop centre. Reordering these silently changes the
alignment and therefore the crop.

> MediaPipe is pinned to 0.10.14. Versions from 0.10.30 removed the
> `mp.solutions` API used here; migrating means rewriting `face_detector.py`
> against the Tasks API and shipping a `.tflite` bundle.

## Alignment

Frames are warped by a **similarity** transform (rotation, uniform scale,
translation — no shear) estimated with `cv2.estimateAffinePartial2D` using
LMEDS. The target is a fixed reference face derived from
`20words_mean_face.npy`, a 68-point mean face in a 256 × 256 frame:

| Reference point | Built from | Position |
| --- | --- | --- |
| Right eye | mean of points 36–41 | (102.07, 94.27) |
| Left eye | mean of points 42–47 | (156.36, 93.58) |
| Nose tip | mean of points 31–35 | (129.00, 135.90) |
| Mouth centre | mean of points 48–67 | (129.31, 157.82) |

Because the mouth lands at y ≈ 157.8, the 96 × 96 crop spans y ∈ [109.8, 205.8]
and x ∈ [81.3, 177.3], comfortably inside the aligned face.

This is what makes the crop invariant to head pose, camera distance and roll, so
the network sees lip motion rather than head motion. `tests/test_pipeline.py`
asserts the aligned mouth position stays within 2 px of standard deviation while
the head translates, scales and rotates.

### Landmark smoothing

Per-frame detector jitter would move the crop, and the 3D-conv front-end cannot
distinguish that from lip motion. Landmarks are averaged over a ±6 frame window
and then re-centred on the current frame's own centroid, so the *shape* is
stabilised while genuine head movement still tracks. The window shrinks at clip
boundaries.

### Missing detections

Interior gaps are linearly interpolated between surrounding detections; leading
and trailing gaps are held at the nearest detection. A clip with no detections
at all raises `NoFaceDetectedError`. A clip where fewer than 30% of frames have
a detection is reported to the user as unreadable rather than transcribed.

### Crops at the frame edge

The reference implementation raises `OverflowError` when the patch would extend
past the aligned face, discarding the whole clip. This project zero-pads
instead, matching the warp's own border colour, so one bad frame cannot destroy
an otherwise good recording.

## Tensor construction

Applied in this order, matching Auto-AVSR's test-time `VideoTransform`:

1. `uint8 (T, 96, 96, 3)` → float, divided by 255
2. Centre crop 96 → 88
3. Grayscale via BT.601 luma weights `(0.2989, 0.587, 0.114)`
4. Normalise: `(x - 0.421) / 0.165`
5. Add batch dimension → `(1, T, 1, 88, 88)`

The luma weights sum to 0.9999, not 1.0. torchvision does not renormalise them
and neither does this code, because the checkpoint's input statistics were
computed with the same slightly-off weights.

## Model tensor flow

```
(1, T, 1, 88, 88)  network input
      -> Conv3dResNet front-end   -> (1, T, 512)
      -> proj_encoder (Linear)    -> (1, T, 768)
      -> Conformer encoder x12    -> (1, T, 768)
      -> CTC head                 -> (1, T, 5049)
      -> Transformer decoder x6   -> tokens
```

The front-end does not downsample time: T frames in, T encoder steps out.

## Checkpoint parameter names

The mirrored checkpoint uses the original espnet layout, which differs from the
`E2E` module in the reference repository. `ml/models/auto_avsr/loader.py`
renames them and refuses to load if any weight is unaccounted for:

| Checkpoint | Model |
| --- | --- |
| `vsr.encoder.frontend.*` | `frontend.*` |
| `vsr.encoder.embed.0.*` | `proj_encoder.*` |
| `vsr.encoder.encoders.*`, `vsr.encoder.after_norm.*` | `encoder.*` |
| `vsr.decoder.*` | `decoder.*` |
| `vsr.ctc.*` | `ctc.*` |

767 tensors, 250.4M parameters, all accounted for.

## Verification

| Check | Command |
| --- | --- |
| Geometry, rates, tensor shapes | `pytest tests/` |
| Watch the actual model input | `python scripts/transcribe.py CLIP --debug` |
| Transcript accuracy vs ground truth | `python scripts/evaluate_lrs3.py` |

The debug export is the fastest way to catch a preprocessing regression: a crop
that has drifted off the mouth is obvious in two seconds of video and nearly
invisible in metrics.
