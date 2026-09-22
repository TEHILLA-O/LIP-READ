# Failure modes, fixes, and results

Honest engineering notes for this project. Nothing here is invented for polish.

## What can go wrong

- **No face / lost mesh / fast head motion.** Impact: empty transcript or wasted inference. Mitigation: utterance segmentation from lip aperture (not frame difference); debug view reports mesh state; failed mesh does not invent speech.
- **Preprocessing drift from the Auto-AVSR contract.** Impact: large WER gap vs reference crops. Mitigation: fixed 25 fps, 96x96→88x88 grayscale crops; numbers not adjustable (`docs/preprocessing-spec.md`).
- **Treating confidence as calibrated probability.** Impact: over-trust. Mitigation: README states confidence is a ranking signal; roughly one word in five wrong even on the model benchmark context.
- **Retention of uploads / debug MP4s.** Impact: privacy risk. Mitigation: uploads deleted after inference by default; debug exports opt-in and expire.

## What went wrong

**No recorded production incident in this repo yet.** Measured engineering gap (not an outage): full pipeline WER on composite clips is **worse** than feeding reference mouth crops straight to the model, i.e. detection/alignment/crop cost is visible in eval.

## How it was resolved

- Model adapter interface so the app is not welded to one research checkpoint.
- Evaluation script `scripts/evaluate_lrs3.py` separates model-path health from preprocessing health.
- Streaming buffer + geometry-based segmentation to reduce head-motion false utterances.

## Results

Measured on LRS3 test utterances (five utterances in the README table):

| Input | WER |
| --- | --- |
| Reference mouth crops | **5.7%** |
| Full pipeline on composite clips | **25.7%** |
| Published Auto-AVSR on full LRS3 (reference) | **20.3%** |

Successful demo: download checkpoint, run ML API + Next web, upload a clip or use webcam, open debug with Shift+D.
