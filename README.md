# Visual Speech

Reads speech from lip movement in video — uploaded clips or a live webcam — using
a pretrained Auto-AVSR visual speech recognition model.

See [FAILURES.md](./FAILURES.md) for what can go wrong, what broke, how it was fixed, and results.

There is no audio anywhere in this system. Every transcript is produced from a
*sequence* of video frames of a mouth, which is why nothing here will ever tell
you what a single photograph is saying.

> Transcripts are model predictions, not recordings of what was said. Even on the
> benchmark this model was built for, roughly one word in five is wrong. Treat the
> confidence score as a ranking signal, not a probability of being correct.

## What it does

```
video / camera
  -> decode frames         -> 25 fps, exactly
  -> detect + track a face -> MediaPipe, one speaker
  -> align the face        -> similarity warp onto a canonical 256x256 face
  -> crop the mouth        -> 96x96, centre-cropped to 88x88 grayscale
  -> visual speech model   -> Auto-AVSR Conformer encoder, 250M parameters
  -> joint CTC/attention beam search
  -> transcript + confidence + alternatives
```

The exact input contract, and why none of those numbers are adjustable, is in
[`docs/preprocessing-spec.md`](docs/preprocessing-spec.md).

## Accuracy

Measured with `scripts/evaluate_lrs3.py` on LRS3 test utterances:

| Input | Word error rate | What it tells you |
| --- | --- | --- |
| Reference mouth crops, straight to the model | 5.7% | The model, vocabulary and decoder are wired up correctly |
| Full pipeline on composite clips | 25.7% | Our own detection, alignment and cropping preserve what the model needs |
| Published Auto-AVSR on the full LRS3 test set | 20.3% | For reference; both numbers above use five utterances |

The two rows measure different things on purpose. The first bypasses this
project's preprocessing entirely, so a bad number there means the model path is
wrong. The second runs whole video files through the real chain, so the gap
between the rows is the cost of our preprocessing.

## Layout

```
apps/web            Next.js interface: camera, upload, transcript, debug view
services/ml-api     FastAPI service: upload inference, live WebSocket
ml/
  preprocessing     decode, detect, track, align, crop, tensorise
  models            VisualSpeechModel adapters (Auto-AVSR; VALLR/AV-HuBERT stubs)
  decoding          SentencePiece vocabulary and beam search
  inference         the engine that ties preprocessing, model and decoding together
  streaming         rolling buffer, utterance segmentation, inference queue
  evaluation        word and character error rate
scripts             CLI tools: setup, download, transcribe, evaluate, fixtures
tests               pytest suite; preprocessing is testable without a model
references          upstream research checkouts, read but not imported wholesale
```

UI code and model code never mix. The web app knows only the shape of a result;
it has no idea which model produced it.

## Setup

Requires Python 3.10+, Node 20+, and about 1 GB of disk for the checkpoint. A
CUDA GPU is used automatically when present, and the CPU is used when it is not.

```powershell
.\scripts\setup_env.ps1              # add -Cpu for a CPU-only PyTorch build
python scripts\download_checkpoint.py
python scripts\check_env.py
```

`setup_env.ps1` installs PyTorch from the right index first, then the rest of
`requirements.txt`. FFmpeg is not needed: PyAV bundles it.

## Running it

Two processes. The backend loads the model once at start-up, which takes a few
seconds.

```powershell
# terminal 1 - ML backend on http://127.0.0.1:8000
cd services\ml-api
..\..\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000

# terminal 2 - interface on http://localhost:3000
cd apps\web
npm install
npm run dev
```

If the backend runs anywhere other than `http://127.0.0.1:8000`, copy
`apps/web/.env.local.example` to `.env.local` and set `NEXT_PUBLIC_ML_API_URL`,
then add that UI origin to `VSR_API_CORS_ORIGINS` on the backend.

Press **Shift+D**, or use the control at the bottom of the page, for the debug
view: face and mouth boxes, the measured lip outline, buffer depth, frame rates,
inference latency, raw tensor shapes, alternative hypotheses, and a download of
the exact mouth video the model was given.

## Without the interface

```powershell
# transcribe a file, and write the MP4s showing what the model saw
python scripts\transcribe.py data\test_videos\composite\sample_00_none.mp4 --debug

# check preprocessing alone, before any model exists
python scripts\transcribe.py clip.mp4 --preprocess-only --debug --overlay

# measure accuracy
python scripts\evaluate_lrs3.py --source crops
python scripts\evaluate_lrs3.py --source pipeline
```

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Whether the model is loaded, and on which device |
| `GET /v1/models` | Registered adapters and the active one's input contract |
| `POST /v1/transcribe` | Transcribe an uploaded video (`?debug=true` for MP4 exports) |
| `POST /v1/preprocess` | Preprocess only — verify framing without loading a model |
| `GET /v1/debug/{token}/{kind}` | Download a debug MP4 while it still exists |
| `WS /v1/stream` | Live frames in, transcripts out |

Every transcription returns the same object:

```json
{
  "text": "I would love to hear from you",
  "confidence": 0.792,
  "alternatives": [{ "text": "I would love to get it from you", "confidence": 0.653 }],
  "processing_ms": 2195,
  "frames_processed": 34,
  "mouth_detected": true
}
```

When there is no transcript, `text` is empty and `error` explains why — no face,
clip too short, model not loaded. The service never substitutes plausible text
for a failure.

### Live streaming

The browser sends JPEG frames over the WebSocket at the model's frame rate. The
receive loop only decodes, detects and buffers; utterances are handed to a
worker thread. The queue holds two segments, and a third arrival displaces the
oldest rather than extending the backlog — on a live stream a queued segment is
already stale by the time it would be transcribed, and each one pushes the next
result further behind. Drops are reported to the client rather than hidden.

### Deciding when someone is speaking

Running the model continuously is wasteful and produces nonsense between
utterances, so the stream is cut into segments by watching the lips.

The obvious signal — how much the pixels around the mouth change between frames
— does not work. Measured on this project's fixtures, moving your head with your
mouth shut scores *higher* than talking does, because translating a face across
a background changes far more pixels than a jaw does:

| Signal, per frame at 25 fps | Still | Head moving, mouth shut | Speaking |
| --- | --- | --- | --- |
| Frame difference around the mouth | 0.000 | 0.079 | 0.023 |
| Change in lip aperture | 0.0003 | 0.004 | 0.008 - 0.08 |

The second row is what the segmenter uses: the gap between the inner lips, from
a MediaPipe FaceMesh fit, divided by the distance between the eyes. Normalising
by a facial distance makes it a *shape* rather than a size, so it ignores where
the head is, how far away it sits and how it is tilted, and responds only to the
mouth actually moving. Head motion then reads an order of magnitude below
speech instead of above it.

Thresholds are not hand-picked. `scripts/tune_motion_thresholds.py` sweeps them
against the labelled fixtures and reports which settings segment every speech
clip while firing on none of the still or moving-head clips; the defaults sit in
the middle of that plateau. Hysteresis and a hangover period keep an utterance
open through the brief closures inside normal speech, and every utterance is
clamped between `VSR_MIN_UTTERANCE_SECONDS` and `VSR_MAX_UTTERANCE_SECONDS`.

This costs a second MediaPipe graph per frame. Measured end to end, `observe()`
takes about 22 ms per 640x480 frame on CPU — a 45 fps ceiling against a 25 fps
target, so the mesh is affordable without decimation.

Two honest limits. A fast head movement can still open an utterance; it is
abandoned without transcribing when the motion does not persist, so the cost is
a wasted inference rather than an invented transcript. And when the mesh fails
to fit a frame that the face detector handles, segmentation goes blind while the
mouth indicator stays lit — the debug view reports the mesh state separately for
exactly this reason.

## Privacy

Nothing a user sends is kept. Uploads are streamed to a temp directory, deleted
in a `finally` block once inference returns, and any survivors of a crashed
process are swept at start-up and on a timer. Camera frames are held only in a
few-second rolling buffer in memory. Debug MP4s are opt-in per request and expire
with the same sweep.

Persistence is available but must be asked for: `VSR_PERSIST_UPLOADS=1`.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `VSR_MODEL` | `auto_avsr` | Which registered adapter to load |
| `VSR_DEVICE` | `auto` | `auto`, `cuda` or `cpu` |
| `VSR_BEAM_SIZE` | `20` | Beam search width |
| `VSR_MIN_UTTERANCE_SECONDS` | `1.2` | Shortest live segment worth transcribing |
| `VSR_MAX_UTTERANCE_SECONDS` | `4.0` | Longest live segment before it is cut |
| `VSR_MOTION_START_THRESHOLD` | `0.016` | Lip-aperture change that opens an utterance |
| `VSR_MOTION_STOP_THRESHOLD` | `0.008` | Change below which the mouth counts as still |
| `VSR_MAX_QUEUED_JOBS` | `2` | Inference queue depth before dropping |
| `VSR_PERSIST_UPLOADS` | `0` | Keep uploaded media after inference |
| `VSR_ALLOW_DEBUG_EXPORT` | `1` | Permit debug MP4 export |
| `VSR_API_CORS_ORIGINS` | `localhost:3000` | JSON array of allowed UI origins |
| `VSR_API_MAX_UPLOAD_BYTES` | `209715200` | Upload ceiling, enforced while writing |

## Tests

```powershell
.venv\Scripts\python.exe -m pytest        # 158 tests, no checkpoint required
cd apps\web; npx eslint .; npm run build
```

The suite covers frame ordering and frame-rate conversion, face detection and
single-speaker tracking, alignment and mouth cropping, tensor shape and
normalisation, lip measurement, utterance segmentation, the API endpoints,
WebSocket behaviour including a full frames-in-transcript-out pass, the inference
queue's drop policy, and temp-file cleanup. Test clips are generated on demand by
`scripts/make_test_videos.py`, so no video fixtures are committed and no real
person's face is in the repository.

`scripts/make_talking_face_video.py` builds the harder fixture: it grafts an LRS3
mouth sequence back into a full scene through the inverse alignment transform,
producing an ordinary-looking video with real lip motion and a known transcript.
That is what makes an end-to-end accuracy number possible without recording
anyone.

## Swapping the model

Implement `VisualSpeechModel` (`load`, `preprocess`, `infer`, `decode`) and
register it:

```python
register_model("my_model", MyAdapter)
```

Then set `VSR_MODEL=my_model`. Nothing outside `ml/models` needs to change.
`ml/models/custom.py` is a template. `VALLRAdapter` and `AVHubertAdapter` are
interface stubs that raise rather than pretend — AV-HuBERT in particular needs an
old Fairseq environment that the rest of this project deliberately does not
depend on.

## References

Studied, not copied: [mpc001/auto_avsr](https://github.com/mpc001/auto_avsr) for
preprocessing and the model, [MarshallT-99/VALLR](https://github.com/MarshallT-99/VALLR)
for MediaPipe-based face handling and its phoneme decoding design, and
[facebookresearch/av_hubert](https://github.com/facebookresearch/av_hubert) as
research background.
