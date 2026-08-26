# Anees — Arabic Handwriting Quality & Similarity Rating
### Unit 7 · Computer Vision · Final Project

A child copies an Arabic letter from a Naskh model. This scores **how accurately
they copied it** — a continuous **0–100 rating**, not OCR. The letter is known in
advance; the question is how well it was written.

```
photo → page detection → perspective warp → ink extraction → ruled-line removal
      → segmentation → 64×64 normalisation → CNN regressor → calibrated rating
```

Everything is in **one class**, `AneesHandwritingCV`, in
[`handwriting_cv.py`](handwriting_cv.py) — dataset, reference bank, pair
construction, all three models, training, evaluation, calibration, photo upload,
rating and feedback.

---

## Files

| File | What it is |
|---|---|
| `handwriting_cv.py` | the whole project as one class |
| `notebooks/Anees_Handwriting_CV.ipynb` | the same project as a Colab run-all notebook |
| `database cv/hijja2.npz` | Hijja2 packed to 5.9 MB instead of 101 MB of CSVs |
| `fonts/Amiri-Regular.ttf` | the Naskh reference face (also auto-downloads) |
| `artifacts/` | the pre-built Amiri reference bank; trained weights and the calibration curves land here too |
| `samples/` | uploaded photos go here |

Total 6.4 MB, so it fits GitHub without Git LFS.

## Dropping this into Anees

`Handwriting Enhancer/` is the precedent — a self-contained folder with its own
`.py` and assets — so this mirrors it:

```
Anees/
  Handwriting CV/                    <- this whole folder, minus the two below
    handwriting_cv.py
    fonts/  artifacts/  samples/
  database cv/hijja2.npz             <- the dataset
  notebooks/Anees_Handwriting_CV.ipynb
```

That layout needs no code changes: `find_dataset()` already looks in
`../database cv/` relative to `handwriting_cv.py`, which lands exactly on
`Anees/database cv/hijja2.npz`. (It falls back to a folder called `datasets/`
too, so dropping the file in beside Anees's other datasets also works.) Keep
`fonts/`, `artifacts/` and `samples/` next to
the `.py` — the paths in it are relative to the file.

## Quick start — Colab (this is where you train)

1. **Runtime → Change runtime type → T4 GPU**
2. Upload `handwriting_cv.py` and `database cv/hijja2.npz` with the folder icon
3. Open `notebooks/Anees_Handwriting_CV.ipynb` → **Run all**

| Preset | What it trains | Roughly, on a T4 |
|---|---|---|
| `QUICK` | production only, 1 member, 20k pairs, 6 epochs | **~5 min** |
| `STANDARD` | scratch + production, 1 member, 90k pairs | **~35 min** |
| `FULL` | all three, 2-member ensemble | **~1.5 hours** |

Run `QUICK` first — it proves the whole path works before you spend an hour on
it. Every model is **checkpointed to `artifacts/` as soon as it finishes**, so
if Colab reclaims the runtime partway through a `FULL` run you lose only the
model that was training.

## Quick start — local

```bash
python3 -m pip install -r requirements.txt
```

```python
from handwriting_cv import AneesHandwritingCV

cv = AneesHandwritingCV().load()        # restore trained weights from artifacts/
result = cv.rate("photo.jpg", "ب")      # the letter the child was asked to write
cv.print_rating(result)
```

Or from the command line:

```bash
python3 handwriting_cv.py train --model all
```

```bash
python3 handwriting_cv.py rate --image photo.jpg --letter ب --save report.png
```

TensorFlow needs Python 3.9–3.12. Without it the class still runs the classical
half — preprocessing, the geometric scorer, the reference bank — and rates
photos with the geometric metric instead of the CNN.

---

## The data

**Hijja2** — 47,434 Arabic characters handwritten by **591 Saudi school children
aged 7–12**, collected in Riyadh, Jan–Apr 2019. 32×32 grayscale, 29 classes
(28 letters + hamza). Split 37,933 train / 9,501 test.

Chosen over AHCD because it is the *same population as the users* — children,
not adults. A model trained on adult handwriting would mark a normal
seven-year-old's letter as bad. Contributors: Najwa Altwaijry, Monera
Al-Megren, Haya Al-Shumisi, Lamya Al-Arwan, Isra Al-Turaiki (KSU).

**Amiri** — the classical Naskh face from Google Fonts, rendered as the
reference "model" the child copies, in all four contextual forms per letter.

## Where the continuous label comes from

Hijja2 records *which* letter a child wrote, never *how well*. Inventing a
neatness number would teach the network our invention, so labels come from three
sources and are **evaluated separately**:

| | Source | Label | Status |
|---|---|---|---|
| **A** | the Amiri model degraded by a measured amount `t` | `1 − t` | **ground truth** |
| **B** | a real child's letter vs its Amiri model | classical geometric score | **proxy** |
| **C** | a real child's letter vs a *different* letter's model | ≈ 0 | **ground truth** |

Strong results on **B** alone would only prove the network memorised the
geometric formula. Results on **A** and **C** show it learned to see distortion
and letter mismatch directly. `evaluate()` prints all three, plus the classical
scorer as the baseline the CNNs have to beat.

## The models

Each pair is stacked into one image — `R` = child stroke, `G` = Amiri model,
`B` = their overlap — so both models stay single-stream and MobileNetV2's
ImageNet weights remain usable. Both sides are skeletonised and re-inflated to a
fixed 2 px, so the network cannot cheat by reading pen thickness.

| | Architecture | From |
|---|---|---|
| **A** `scratch` | CNN from scratch, `Conv→MaxPool` ×4 + dense, 1 sigmoid unit | unit CV_2 |
| **B** `mobilenet` | MobileNetV2, feature extraction → fine-tune the top 60 layers at 10× lower LR | unit CV_3 |
| **C** `production` | residual CNN, 6-channel input, joint-affine augmentation, cosine schedule, 2-seed ensemble | the accuracy answer |

A and B exist so the two architectures from the unit can be compared honestly on
identical data. **C is the one the app ships.** Five things make it the accurate
one:

1. **Six input channels, not three.** Channels 4 and 5 are the decayed distance
   transforms of the model and of the child — exactly what the classical
   Hausdorff term measures. Handing the network that field lets it *see* "this
   stroke is 9 px from where it belongs" instead of re-deriving the idea from
   binary pixels.
2. **Residual blocks with BatchNorm** — depth without the gradient dying.
   Affordable here because the production model trains long enough for BN's
   moving statistics to settle. (`scratch` deliberately has no BatchNorm: on a
   short run it behaves one way while training and another at predict time, and
   the model looks broken when it is not.)
3. **Average *and* max pooling concatenated.** Average pooling reports how wrong
   the letter is overall; max pooling reports the single worst place. A child's
   letter is usually mostly right with one bad stroke, and averaging alone hides
   exactly that.
4. **Label-preserving augmentation.** One small affine transform applied to the
   child and the model *together*. Rotating both by 3° cannot make the copy any
   better or worse, so the label stays valid while the network is pushed to
   measure the two against each other rather than against the canvas.
5. **A 2-seed ensemble plus test-time augmentation.** The runs make different
   mistakes and the mistakes cancel. Pure inference-time accuracy — no
   retraining.

## What it actually scores

Measured on a **small equal budget** — 9,000 pairs, 14 epochs, CPU, no ensemble,
no early stopping reached. These are not the numbers a full run gives; they are
here because a claim without a measurement behind it is worth nothing.

| model | MAE | RMSE | R² | r | within 5 pts | A | B | C |
|---|---|---|---|---|---|---|---|---|
| `geometric` | 11.17 | 16.89 | 0.513 | 0.764 | 46.6% | 17.92 | 0.00 | 20.02 |
| `scratch` | 9.79 | **13.33** | **0.697** | 0.841 | 35.9% | **13.83** | 6.18 | **8.91** |
| `production` | **9.49** | 13.41 | 0.693 | **0.847** | **41.5%** | 14.09 | **4.53** | 10.19 |

MAE and RMSE are in score points (0–100). A / B / C are MAE per label source.

Read this honestly:

* `production` leads on overall MAE, but **only by 3%** — and it is slightly
  *behind* `scratch` on RMSE and on sources A and C. It is not a landslide.
* Where it does win is the slice the app cares about: **real children's
  letters, 4.53 vs 6.18 MAE**, and the share of ratings landing within 5 points,
  **41.5% vs 35.9%**. That is what the two distance-transform channels were
  added for, so the gain shows up where it was predicted to.
* `production` has **three times the parameters**, so 7,650 training pairs
  starve it more than they starve `scratch`. The gap at the full 90k-pair,
  35-epoch, 2-member setting is **untested** — run `FULL` to find out.
* `geometric` has the best within-5-points rate despite the worst MAE. That is
  not a surprise and not a win: source B's label *is* the geometric score, so it
  reproduces 40% of the test set exactly (`B = 0.00`) and is badly wrong on the
  other 60%. It is the baseline, not a contender.

## Calibration — what "82%" is allowed to mean

Raw similarity against typeset Amiri is a harsh scale: real children land in the
0.2–0.5 band, because no seven-year-old writes like a font. Showing a child
"34%" would be both discouraging and meaningless.

So the raw score is mapped through the distribution of *real children's* scores,
and the number the child sees means **"neater than N% of children aged 7–12"** —
which is exactly what 47,434 real samples entitle us to say.

## Feedback

`rate()` returns more than a number. `diagnose()` measures *why* the rating came
out where it did — missing ink, extra strokes, tilt, proportions, placement —
and `feedback()` turns the two or three costliest into plain sentences in Arabic
and English, aimed at a child aged 7–12:

```
letter ب   [################....]  81%   very good / جيد جدًا
  -> neater than 81% of children aged 7-12

  Good! Just one thing to improve:
    - the letter is tilted - keep it upright   (الحرف مائل لليمين — اجعله مستقيمًا)
```

None of that involves the network — it is measured directly off the two
canvases, so the app can name the actual mistake instead of saying "try again".

## Honest limitations

* **No human ever graded a sample.** Source **B**'s labels are a geometric
  formula, so on that slice the network distils a metric rather than learning a
  new judgement of neatness. **The fix is small and concrete: have one teacher
  grade 300–500 real samples 1–5, then fine-tune the last dense layer.**
  Everything else is already built for it — only the label source changes.
* The comparison is **static** — it never sees stroke order, direction or speed.
  A letter drawn backwards but shaped correctly scores well.
* Hijja2 is 32×32; upsampling adds no detail, so fine differences in a curve are
  simply not in the data.
* One rendering of Amiri is the reference. A school teaching a slightly
  different Naskh model would lose points on letterforms that are not mistakes —
  swap the TTF to fix.
* Harakat are excluded from Hijja2, so the model has never seen one.

## Credits

Hijja2 dataset — Altwaijry, Al-Megren, Al-Shumisi, Al-Arwan, Al-Turaiki, King
Saud University. Amiri font — SIL Open Font License.
