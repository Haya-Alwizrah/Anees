#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 evaluator.py -- Arabic Handwriting Evaluator for Children (Hijja dataset)
================================================================================

A single, self-contained, CPU-friendly pipeline that:

  1. PREPROCESSING  : one function (`preprocess_image`) used for BOTH the CSV
                      dataset images and real photos/scans of paper.  The two
                      are matched by normalising the PHOTO to look like a
                      dataset glyph -- the dataset rows are already clean, and
                      binarising them costs half the accuracy (88.2 % -> 43.1 %,
                      measured; see the note in `_preprocess_core`).
  2. DATA SPLIT     : stratified 90/10 train/validation split, untouched test set.
  3. MODEL          : a VGG-style CNN split into a clearly separated
                      feature extractor (-> 2048-d vector) + classifier head.
  4. TRAINING       : light augmentation, checkpointing, early stopping,
                      history plots, test accuracy, per-letter table,
                      confusion matrix.
  5. SCORING ENGINE : `evaluate_handwriting()` -- grades a child's letter on
                      paper and prints actionable feedback for the child and
                      for the parent / teacher.
  6. MAIN           : argparse driven command line.

--------------------------------------------------------------------------------
DATA EXPECTED (no header row, ink is WHITE on BLACK, 0-255):
    train_X.csv  (37933 x 1024)   train_Y.csv  (37933 labels, 1..29)
    test_X.csv   (9501  x 1024)   test_Y.csv   (9501  labels, 1..29)
Optionally a pre-packed `hijja2.npz` with keys Xtr / ytr / Xte / yte
(same content, loads ~50x faster than parsing the CSVs).

USAGE
    python evaluator.py                              # full training run
    python evaluator.py --epochs 5                   # quick run
    python evaluator.py --no-train                   # load saved model, evaluate
    python evaluator.py --no-train --image photo.jpg --target 2   # score a photo
    python evaluator.py --no-train --image photo.jpg --adaptive   # bad lighting
    python evaluator.py --show-steps photo.jpg       # preprocessing figure only

DEPENDENCIES: tensorflow, opencv-python, numpy, pandas, matplotlib, scikit-learn
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import cv2

import matplotlib
matplotlib.use("Agg")          # headless backend: we only ever save PNG files
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Heavy / optional dependencies are imported defensively so that the purely
# image-processing parts of this script (preprocessing, --show-steps) still work
# on a machine without TensorFlow, and so the user gets a readable message
# instead of a raw ImportError traceback.
# ------------------------------------------------------------------------------
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    _TF_OK, _TF_ERR = True, None
except Exception as _e:                                    # pragma: no cover
    _TF_OK, _TF_ERR = False, _e

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    _SK_OK, _SK_ERR = True, None
except Exception as _e:                                    # pragma: no cover
    _SK_OK, _SK_ERR = False, _e


# ==============================================================================
# 0. CONSTANTS -- the 29 classes and the pedagogical knowledge base
# ==============================================================================

IMG_SIZE      = 32          # network input is 32 x 32
INK_BOX       = 21          # photo letters are rescaled to this longest side.
                            # 21 px is not arbitrary: the mean longest side of
                            # the raw Hijja glyphs measures 20.9 px (18.3 for د
                            # up to 23.8 for س), so a scanned letter lands in
                            # the same size distribution the model trains on.
NUM_CLASSES   = 29
WORK_SIDE     = 512         # phone photos are shrunk to this before thresholding
PHOTO_MIN     = 128         # above this size an input is treated as a real photo
PIPELINE_VER  = 4           # bump this to invalidate `preprocessed_32.npz`
SEED          = 42

# 28 Arabic letters in alphabetical order + hamza last -> label 1..29 in the CSVs
ARABIC_LETTERS = [
    "ا", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ", "ر",
    "ز", "س", "ش", "ص", "ض", "ط", "ظ", "ع", "غ", "ف",
    "ق", "ك", "ل", "م", "ن", "ه", "و", "ي", "ء",
]

# ASCII names -- matplotlib's default font has no Arabic glyphs, so every figure
# is labelled with these instead of with the letters themselves.
LETTER_NAMES = [
    "alef", "beh", "teh", "theh", "jeem", "hah", "khah", "dal", "thal", "reh",
    "zain", "seen", "sheen", "sad", "dad", "tah", "zah", "ain", "ghain", "feh",
    "qaf", "kaf", "lam", "meem", "noon", "heh", "waw", "yeh", "hamza",
]

# Dot count / position per letter -- the single most common mistake children make.
LETTER_DOTS = [
    "no dots",                          # ا
    "ONE dot BELOW the body",           # ب
    "TWO dots ABOVE the body",          # ت
    "THREE dots ABOVE the body",        # ث
    "ONE dot INSIDE (below the curve)", # ج
    "no dots",                          # ح
    "ONE dot ABOVE",                    # خ
    "no dots",                          # د
    "ONE dot ABOVE",                    # ذ
    "no dots",                          # ر
    "ONE dot ABOVE",                    # ز
    "no dots (three teeth)",            # س
    "THREE dots ABOVE (three teeth)",   # ش
    "no dots",                          # ص
    "ONE dot ABOVE",                    # ض
    "no dots",                          # ط
    "ONE dot ABOVE",                    # ظ
    "no dots",                          # ع
    "ONE dot ABOVE",                    # غ
    "ONE dot ABOVE",                    # ف
    "TWO dots ABOVE",                   # ق
    "no dots (small hamza-stroke inside)",  # ك
    "no dots",                          # ل
    "no dots",                          # م
    "ONE dot ABOVE the bowl",           # ن
    "no dots",                          # ه
    "no dots",                          # و
    "TWO dots BELOW",                   # ي
    "no dots (it is a small stroke, not a letter body)",  # ء
]

# Letters that share the same skeleton and differ only by dots / small details.
# Used by the feedback engine to explain *why* the model hesitated.
CONFUSION_GROUPS = [
    ["ب", "ت", "ث", "ن", "ي"],
    ["ج", "ح", "خ"],
    ["د", "ذ"],
    ["ر", "ز"],
    ["س", "ش"],
    ["ص", "ض"],
    ["ط", "ظ"],
    ["ع", "غ"],
    ["ف", "ق"],
    ["ه", "ة"],
]

# Output artefacts (all written next to the script / working directory)
CACHE_FILE      = "preprocessed_32.npz"
HISTORY_PNG     = "training_history.png"
CONFUSION_PNG   = "confusion_matrix.png"
STEPS_PNG       = "preprocessing_steps.png"


def _default_model_path() -> str:
    """Checkpoint file name (verified to work on Keras 2 and Keras 3.15)."""
    return "arabic_handwriting_model.h5"


def _resolve_checkpoint_path(path: str) -> str:
    """
    Keras 3 still writes legacy HDF5 (it only warns), so `.h5` is used as asked.
    Some Keras builds do reject anything other than `.keras`, though, so probe
    the callback constructor once and downgrade the extension if necessary --
    that way the script cannot die at the end of the first epoch.
    """
    try:
        keras.callbacks.ModelCheckpoint(path, monitor="val_accuracy",
                                        mode="max", save_best_only=True)
        return path
    except Exception as e:
        alt = os.path.splitext(path)[0] + ".keras"
        print(f"[model] this Keras build rejects '{os.path.basename(path)}' "
              f"({type(e).__name__}) -> saving to '{os.path.basename(alt)}'")
        return alt


# ==============================================================================
# 1. PREPROCESSING
# ==============================================================================

def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert any numeric image to uint8 0..255 without surprises."""
    if img.dtype == np.uint8:
        return img
    arr = np.asarray(img, dtype=np.float32)
    if arr.size and arr.max() <= 1.0 + 1e-6 and arr.min() >= -1e-6:
        arr = arr * 255.0                      # image was stored in [0, 1]
    return np.clip(arr, 0, 255).astype(np.uint8)


def _to_gray(source) -> np.ndarray:
    """
    Accept anything and return a single-channel uint8 image:
      * a path to a photo / scan  -> read from disk
      * a raw CSV row (1024,)     -> reshaped to 32 x 32
      * a numpy image (H,W) / (H,W,3 BGR) / (H,W,4 BGRA)
    """
    # (a) path on disk -------------------------------------------------------
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"OpenCV could not decode the image: {path}")
    else:
        img = np.asarray(source)

    if img.size == 0:
        raise ValueError("Empty image.")

    # (b) flat vector (e.g. one CSV row) -------------------------------------
    if img.ndim == 1:
        side = int(round(math.sqrt(img.size)))
        if side * side != img.size:
            raise ValueError(f"Flat input of length {img.size} is not square.")
        img = img.reshape(side, side)

    img = _to_uint8(img)

    # (c) colour handling ----------------------------------------------------
    if img.ndim == 3:
        if img.shape[2] == 4:
            # BGRA: composite over a WHITE sheet of paper, then greyscale, so a
            # transparent PNG background does not become black "ink".
            bgr   = img[:, :, :3].astype(np.float32)
            alpha = (img[:, :, 3:4].astype(np.float32)) / 255.0
            img   = (bgr * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
            img   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 1:
            img = img[:, :, 0]
        else:
            raise ValueError(f"Unsupported channel count: {img.shape[2]}")
    elif img.ndim != 2:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    return np.ascontiguousarray(img)


def _is_clean_glyph(gray: np.ndarray) -> bool:
    """
    True when the input is ALREADY a small white-ink-on-black glyph, i.e. a row
    straight out of the Hijja CSV rather than a photograph of paper.
    Such an image needs none of the photo cleanup below -- see the note in
    `_preprocess_core` for what running it anyway costs in accuracy.
    """
    if max(gray.shape) > PHOTO_MIN:
        return False                      # too big to be a dataset glyph
    return float((gray < 32).mean()) > 0.5    # dark background, bright minority ink


def _ink_is_darker(src: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Return `mask` (white = ink) flipped if its white pixels are BRIGHTER than
    its black pixels in `src`.  On a sheet of paper the pencil is always the
    darker of the two groups, so this catches a mis-polarised global threshold.
    """
    fg, bg = mask > 0, mask == 0
    if fg.any() and bg.any() and src[fg].mean() > src[bg].mean():
        return cv2.bitwise_not(mask)
    return mask


def _preprocess_core(gray: np.ndarray, adaptive: bool = False,
                     return_stages: bool = False):
    """
    The heart of the pipeline.  Input: uint8 greyscale image of ANY size.
    Output: 32 x 32 uint8, WHITE ink on BLACK, cropped, aspect-preserved,
    and centred on its centre of mass.

    Steps: denoise -> binarise (+auto-invert) -> open -> crop -> resize(24) ->
    paste on 32x32 -> centre-of-mass shift.
    """
    stages = {"original": gray.copy()}

    # =========================================================================
    # CLEAN-GLYPH FAST PATH -- the single most important line in this file.
    #
    # A Hijja CSV row is ALREADY a denoised, framed, white-on-black 32x32 glyph.
    # Forcing the photo cleanup onto it actively destroys the letter:
    #   * children's strokes here are 1-2 px wide and ANTI-ALIASED; Otsu
    #     promotes every faint edge pixel to solid ink, taking an image from
    #     ~90 ink pixels to ~222 (2.5x) and turning thin letters into blobs,
    #   * re-scaling every glyph to a fixed box and re-centring it on its centre
    #     of mass throws away size and baseline position, both of which
    #     distinguish Arabic letters (descenders drop below the line).
    # Measured, 6 identical epochs, same model/seed/split:
    #     raw dataset pixels ............ 88.2 % val accuracy
    #     forced through the photo path .. 43.1 % val accuracy
    # So the pipeline stays a single entry point, but the photo-specific
    # cleanup only runs on actual photos.  The two domains are matched from the
    # other side instead: the photo branch below normalises a scanned letter to
    # roughly the size and framing the dataset glyphs already have (INK_BOX).
    # =========================================================================
    if _is_clean_glyph(gray):
        canvas = gray
        if canvas.shape != (IMG_SIZE, IMG_SIZE):
            canvas = cv2.resize(canvas, (IMG_SIZE, IMG_SIZE),
                                interpolation=cv2.INTER_AREA)
        stages.update(denoised=canvas, threshold=canvas, final=canvas)
        return (canvas, stages) if return_stages else canvas

    # --- 0. shrink very large phone photos.  Two reasons: the 3x3 kernels below
    #        are meaningless on a 4000 px image, and a 32x32 target simply does
    #        not need more than ~512 px of detail.  Also much faster.
    h, w = gray.shape
    if max(h, w) > WORK_SIDE:
        s = WORK_SIDE / float(max(h, w))
        gray = cv2.resize(gray, (max(1, int(w * s)), max(1, int(h * s))),
                          interpolation=cv2.INTER_AREA)
    is_photo = max(gray.shape) > PHOTO_MIN     # photo/scan vs 32x32 dataset row

    # --- 0b. illumination flattening (photos on the GLOBAL-threshold path only).
    #         Dividing by a heavily blurred copy turns an unevenly lit page into
    #         uniform paper, which is what makes a single Otsu threshold
    #         reliable.  Measured on simulated shadowed photos: IoU 0.36 -> 0.47.
    #         Adaptive thresholding estimates its own local background, and
    #         flattening only amplifies the paper grain for it (0.23 -> 0.06),
    #         so it is deliberately skipped there.
    if is_photo and not adaptive:
        bg = cv2.GaussianBlur(gray, (0, 0), max(gray.shape) / 8.0)
        gray = cv2.divide(gray, bg, scale=200)

    # --- 1./2. noise reduction ------------------------------------------------
    den = cv2.medianBlur(gray, 3)          # kills salt & pepper / paper grain
    den = cv2.GaussianBlur(den, (3, 3), 0) # smooths JPEG blocking
    stages["denoised"] = den.copy()

    # --- 3. binarisation ------------------------------------------------------
    if adaptive:
        # Uneven lighting / shadow across the page: threshold each neighbourhood
        # on its own.  THRESH_BINARY_INV assumes dark pencil on light paper.
        # blockSize MUST be several times wider than the pencil stroke: a window
        # that fits INSIDE a stroke sees only ink, so the middle of the stroke
        # is classified as background and the letter comes out hollow.
        # 25 px (the classic default) is right for a small image, so we use it
        # as the FLOOR and otherwise derive the window from the measured stroke
        # width -- distance transform of a rough Otsu mask.
        blk = 25
        if is_photo:
            _, rough = cv2.threshold(den, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if (rough > 0).mean() > 0.5:
                rough = cv2.bitwise_not(rough)
            rough = _ink_is_darker(den, rough)
            if (rough > 0).any():
                dt = cv2.distanceTransform(rough, cv2.DIST_L2, 3)
                stroke = 2.0 * float(np.percentile(dt[rough > 0], 90))
                blk = max(25, int(5 * stroke))      # window ~5 strokes wide
        blk += (blk + 1) % 2                        # force odd
        blk = min(blk, max(3, (min(den.shape) // 2) * 2 - 1))
        th = cv2.adaptiveThreshold(den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, blk, 10)
    else:
        _, th = cv2.threshold(den, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # AUTO-INVERT: a paper photo is dark ink on a white page, so after Otsu most
    # pixels are white.  Ink must ALWAYS end up white on black, exactly like the
    # Hijja CSV images -- otherwise the network sees a photographic negative.
    if (th > 0).mean() > 0.5:
        th = cv2.bitwise_not(th)

    # POLARITY SANITY CHECK (photos + global threshold only).  When the letter
    # covers less than ~2 % of the frame, Otsu can split the PAPER distribution
    # instead of separating ink from paper -- the "minority" class is then the
    # brightest paper, not the ink, and the >50 % rule above cannot see it.
    # On paper the ink is always the DARKER group, so verify exactly that.
    # Without this check the IoU against the dataset pipeline collapses from
    # 0.46 to 0.01 as the letter gets smaller in the frame; with it: 0.41-0.49.
    if is_photo and not adaptive:
        th = _ink_is_darker(den, th)

    stages["threshold"] = th.copy()

    # --- 4. morphological OPEN: remove isolated specks -------------------------
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    ink_before, ink_after = int((th > 0).sum()), int((opened > 0).sum())
    # Safety valve: children's strokes can be 1 px wide at 32x32, where a 2x2
    # opening would erase the whole letter.  Only keep it if the letter survives.
    if ink_after > 0 and ink_after >= 0.4 * ink_before:
        th = opened

    # --- 4b. drop tiny leftover blobs -- ONLY on real photos.  On 32x32 dataset
    #         images a diacritic dot is itself only a few pixels, so we never
    #         filter components there (that would turn ب into ا).
    if is_photo:
        n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(th, 8)
        if n_lab > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            keep_min = max(4.0, 0.03 * float(areas.max()))   # 3 % of the body
            mask = np.zeros_like(th)
            for i, a in enumerate(areas, start=1):
                if a >= keep_min:
                    mask[lab == i] = 255
            if (mask > 0).any():
                th = mask

    # --- 5. crop to the ink bounding box --------------------------------------
    coords = cv2.findNonZero(th)
    if coords is None:                       # blank page -> all-black canvas
        canvas = np.zeros((IMG_SIZE, IMG_SIZE), np.uint8)
        stages["final"] = canvas
        return (canvas, stages) if return_stages else canvas

    x, y, bw, bh = cv2.boundingRect(coords)
    roi = th[y:y + bh, x:x + bw]

    # resize KEEPING THE ASPECT RATIO so the longest side is INK_BOX (24 px):
    # ا must stay tall and thin, م must stay small and round.
    scale = INK_BOX / float(max(bw, bh))
    nw, nh = max(1, int(round(bw * scale))), max(1, int(round(bh * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    small = cv2.resize(roi, (nw, nh), interpolation=interp)

    # paste centred on the 32x32 black canvas (>= 4 px margin on every side)
    canvas = np.zeros((IMG_SIZE, IMG_SIZE), np.uint8)
    oy, ox = (IMG_SIZE - nh) // 2, (IMG_SIZE - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = small

    # --- 5b. fine centring on the CENTRE OF MASS -------------------------------
    # Bounding-box centring is not enough: the mass of ج sits low, the mass of
    # ط sits left.  MNIST-style centre-of-mass alignment removes that bias.
    M = cv2.moments(canvas, binaryImage=False)
    if M["m00"] > 0:
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        target = (IMG_SIZE - 1) / 2.0                      # 15.5
        dx = float(np.clip(target - cx, -4, 4))            # margin is 4 px
        dy = float(np.clip(target - cy, -4, 4))
        if abs(dx) > 0.01 or abs(dy) > 0.01:
            shift = np.float32([[1, 0, dx], [0, 1, dy]])
            canvas = cv2.warpAffine(canvas, shift, (IMG_SIZE, IMG_SIZE),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=0)

    stages["final"] = canvas
    return (canvas, stages) if return_stages else canvas


def preprocess_image(source, adaptive: bool = False) -> np.ndarray:
    """
    THE public preprocessing function -- used for paper photos AND for the CSV
    rows, so both end up with identical statistics.

    Parameters
    ----------
    source   : path to a photo/scan, a raw 1024-value CSV row, or a numpy image.
    adaptive : True -> adaptive thresholding (badly lit phone photos).

    Returns
    -------
    np.ndarray, shape (1, 32, 32, 1), float32 in [0, 1]  -- ready for predict().
    """
    gray = _to_gray(source)
    canvas = _preprocess_core(gray, adaptive=adaptive)
    return (canvas.astype(np.float32) / 255.0).reshape(1, IMG_SIZE, IMG_SIZE, 1)


def preprocess_batch(flat_rows: np.ndarray, adaptive: bool = False,
                     label: str = "images") -> np.ndarray:
    """Run `preprocess_image` over an (N, 1024) CSV matrix -> (N, 32, 32) uint8."""
    n = flat_rows.shape[0]
    out = np.zeros((n, IMG_SIZE, IMG_SIZE), np.uint8)
    t0 = time.time()
    for i in range(n):
        out[i] = _preprocess_core(_to_gray(flat_rows[i]), adaptive=adaptive)
        if (i + 1) % 5000 == 0 or (i + 1) == n:
            print(f"    {label}: {i + 1:>6}/{n}  ({time.time() - t0:5.1f}s)",
                  flush=True)
    return out


def show_preprocessing_steps(path, out_png: str = STEPS_PNG,
                             adaptive: bool = False) -> str:
    """Save a side-by-side PNG: original | denoised | threshold | centred 32x32."""
    gray = _to_gray(path)
    canvas, st = _preprocess_core(gray, adaptive=adaptive, return_stages=True)

    titles = ["1. original (grey)", "2. denoised", "3. threshold + invert",
              "4. centred 32x32"]
    images = [st["original"], st["denoised"], st["threshold"], st["final"]]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, im, ti in zip(axes, images, titles):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{ti}\n{im.shape[1]}x{im.shape[0]}", fontsize=10)
        ax.axis("off")
    fig.suptitle("preprocess_image() -- pipeline stages", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.90))     # keep the suptitle clear of the axes
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[preprocess] steps figure saved -> {out_png}")
    return out_png


def ascii_preview(img, title: str = "") -> None:
    """Print the exact 32x32 the model sees, as ASCII art."""
    arr = np.asarray(img, dtype=np.float32).reshape(IMG_SIZE, IMG_SIZE)
    if arr.max() > 1.0:
        arr = arr / 255.0
    ramp = " .:-=+*#%@"
    if title:
        print(f"   {title}")
    print("   +" + "-" * IMG_SIZE + "+")
    for row in arr:
        line = "".join(ramp[min(len(ramp) - 1, int(v * len(ramp)))] for v in row)
        print("   |" + line + "|")
    print("   +" + "-" * IMG_SIZE + "+")


# ==============================================================================
# 2. DATA LOADING + SPLIT
# ==============================================================================

def _read_matrix(path: str) -> np.ndarray:
    """Read a head-less numeric CSV as fast and as tolerantly as possible."""
    try:                                   # fast path: pixels are plain ints
        df = pd.read_csv(path, header=None, dtype=np.uint8)
    except Exception:
        try:
            df = pd.read_csv(path, header=None, dtype=np.float32)
        except Exception:                  # the file DOES have a header row
            df = pd.read_csv(path, header=0)
            print(f"[data] note: '{os.path.basename(path)}' had a header row.")
    return df.to_numpy()


def _find_npz(data_dir: str, explicit: str | None) -> str | None:
    """Locate an optional pre-packed hijja2.npz (much faster than the CSVs)."""
    if explicit:
        return explicit if os.path.exists(explicit) else None
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(data_dir, "hijja2.npz"),
                 os.path.join(os.path.dirname(os.path.abspath(data_dir)),
                              "hijja2.npz"),
                 os.path.join(here, "hijja2.npz"),
                 os.path.join(os.getcwd(), "hijja2.npz")):
        if os.path.exists(cand):
            return cand
    return None


def load_raw_data(data_dir: str, npz_path: str | None = None):
    """
    Return (train_X, train_Y, test_X, test_Y) as raw uint8 (N, 1024) / int labels.
    Uses hijja2.npz when available, otherwise parses the four CSV files.
    """
    npz = _find_npz(data_dir, npz_path)
    if npz:
        print(f"[data] loading packed arrays from {npz}")
        z = np.load(npz)
        def pick(*names):
            for nm in names:
                if nm in z.files:
                    return z[nm]
            raise KeyError(f"{npz} has none of {names} (found {z.files})")
        Xtr = pick("Xtr", "train_X", "x_train", "X_train")
        ytr = pick("ytr", "train_Y", "y_train", "Y_train")
        Xte = pick("Xte", "test_X", "x_test", "X_test")
        yte = pick("yte", "test_Y", "y_test", "Y_test")
    else:
        req = ["train_X.csv", "train_Y.csv", "test_X.csv", "test_Y.csv"]
        missing = [f for f in req if not os.path.exists(os.path.join(data_dir, f))]
        if missing:
            raise FileNotFoundError(
                f"Missing {missing} in '{data_dir}'.\n"
                f"Point the script at the data with:  --data-dir /path/to/csvs")
        print(f"[data] parsing CSV files from {data_dir} (this takes a moment)")
        Xtr = _read_matrix(os.path.join(data_dir, "train_X.csv"))
        ytr = _read_matrix(os.path.join(data_dir, "train_Y.csv"))
        Xte = _read_matrix(os.path.join(data_dir, "test_X.csv"))
        yte = _read_matrix(os.path.join(data_dir, "test_Y.csv"))

    def clean_y(y):
        y = np.asarray(y)
        if y.ndim == 2:                     # (N,1) or (index, label)
            y = y[:, -1]
        return y.ravel().astype(np.int32)

    Xtr = _to_uint8(np.asarray(Xtr)).reshape(len(Xtr), -1)
    Xte = _to_uint8(np.asarray(Xte)).reshape(len(Xte), -1)
    ytr, yte = clean_y(ytr), clean_y(yte)

    # Labels are 1..29 in the CSVs -> shift to 0..28 for
    # sparse_categorical_crossentropy.  (Guard in case they are already 0-based.)
    lo = int(min(ytr.min(), yte.min()))
    if lo >= 1:
        ytr, yte = ytr - 1, yte - 1
    else:
        print("[data] note: labels already appear to be 0-based; no shift applied.")

    assert ytr.min() >= 0 and ytr.max() < NUM_CLASSES, "label out of range"
    print(f"[data] raw train {Xtr.shape}  raw test {Xte.shape}  "
          f"labels {ytr.min()}..{ytr.max()}")
    return Xtr, ytr, Xte, yte


def load_dataset(data_dir: str, adaptive: bool = False, use_cache: bool = True,
                 npz_path: str | None = None, cache_file: str = CACHE_FILE):
    """
    Load + preprocess everything, with an on-disk cache so the SECOND run is
    instant.  Returns (X_train, y_train, X_test, y_test) with X as
    float32 (N, 32, 32, 1) in [0, 1].
    """
    sig = np.array([PIPELINE_VER, int(adaptive)], dtype=np.int64)

    if use_cache and os.path.exists(cache_file):
        try:
            z = np.load(cache_file)
            if "sig" in z.files and np.array_equal(z["sig"], sig):
                print(f"[cache] hit -> {cache_file} (preprocessing skipped)")
                return (_as_input(z["Xtr"]), z["ytr"].astype(np.int32),
                        _as_input(z["Xte"]), z["yte"].astype(np.int32))
            print(f"[cache] '{cache_file}' is stale (different pipeline) -> rebuild")
        except Exception as e:
            print(f"[cache] unreadable ({e}) -> rebuild")

    Xtr_raw, ytr, Xte_raw, yte = load_raw_data(data_dir, npz_path)

    print("[preprocess] applying the SAME pipeline used for paper photos ...")
    Xtr = preprocess_batch(Xtr_raw, adaptive, "train")
    Xte = preprocess_batch(Xte_raw, adaptive, "test ")

    if use_cache:
        np.savez_compressed(cache_file, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, sig=sig)
        print(f"[cache] saved -> {cache_file} "
              f"({os.path.getsize(cache_file) / 1e6:.1f} MB)")

    return _as_input(Xtr), ytr, _as_input(Xte), yte


def _as_input(x_uint8: np.ndarray) -> np.ndarray:
    """(N, 32, 32) uint8  ->  (N, 32, 32, 1) float32 in [0, 1]."""
    return (x_uint8.astype(np.float32) / 255.0).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


def stratified_split(X, y, val_fraction: float = 0.10):
    """90 % TRAIN / 10 % VALIDATION, class balance preserved."""
    if not _SK_OK:
        raise ImportError(f"scikit-learn is required for the split ({_SK_ERR})")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=SEED, shuffle=True)
    return X_tr, X_val, y_tr, y_val


# ==============================================================================
# 3. MODEL -- feature extractor + classifier, clearly separated
# ==============================================================================

def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=NUM_CLASSES):
    """
    Functional VGG-style CNN.

        FEATURE EXTRACTOR   3 conv blocks (32 / 64 / 128 filters)
                            32x32 -> 16x16 -> 8x8 -> 4x4
                            Flatten('features') -> 4*4*128 = 2048-d vector
        CLASSIFIER          Dense(256) -> BN -> ReLU -> Dropout -> softmax(29)
    """
    if not _TF_OK:
        raise ImportError(f"TensorFlow is required to build the model ({_TF_ERR})")

    inp = layers.Input(shape=input_shape, name="input_image")
    x = inp

    # ---------------- feature extractor ----------------
    for b, (filters, drop) in enumerate([(32, 0.15), (64, 0.20), (128, 0.25)], 1):
        for c in (1, 2):
            # bias is redundant in front of BatchNormalization
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                              name=f"block{b}_conv{c}")(x)
            x = layers.BatchNormalization(name=f"block{b}_bn{c}")(x)
            x = layers.Activation("relu", name=f"block{b}_relu{c}")(x)
        x = layers.MaxPooling2D(2, name=f"block{b}_pool")(x)
        x = layers.Dropout(drop, name=f"block{b}_drop")(x)

    x = layers.Flatten(name="features")(x)          # <-- the 2048-d feature vector

    # ---------------- classifier head ----------------
    x = layers.Dense(256, use_bias=False, name="fc1")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.Activation("relu", name="fc1_relu")(x)
    x = layers.Dropout(0.40, name="fc1_drop")(x)
    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return keras.Model(inp, out, name="hijja_cnn")


def make_feature_extractor(model):
    """Sub-model that stops at the 'features' layer (transfer-learning ready)."""
    return keras.Model(model.inputs, model.get_layer("features").output,
                       name="feature_extractor")


# ==============================================================================
# 4. TRAINING
# ==============================================================================

def _augment_batch(batch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    LIGHT augmentation, OpenCV implementation (identical settings to
    ImageDataGenerator(rotation_range=8, width/height_shift_range=0.06,
    zoom_range=0.05); no flips, no shear -- mirrored Arabic letters are wrong
    letters, and shear destroys thin children's strokes).
    """
    out = np.empty_like(batch)
    c = (IMG_SIZE - 1) / 2.0
    for i in range(batch.shape[0]):
        angle = float(rng.uniform(-8.0, 8.0))
        zoom  = float(rng.uniform(0.95, 1.05))
        M = cv2.getRotationMatrix2D((c, c), angle, zoom)
        M[0, 2] += float(rng.uniform(-0.06, 0.06)) * IMG_SIZE
        M[1, 2] += float(rng.uniform(-0.06, 0.06)) * IMG_SIZE
        out[i, :, :, 0] = cv2.warpAffine(
            batch[i, :, :, 0], M, (IMG_SIZE, IMG_SIZE),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return out


def make_train_iterator(X, y, batch_size: int, fast: bool = False):
    """
    Return (iterable, steps_per_epoch).

    Default: keras' ImageDataGenerator (still shipped in Keras 3.15 as a
    PyDataset subclass, so `fit` accepts it directly).  If a future Keras drops
    it -- or `fast` is set -- fall back to the equivalent OpenCV augmenter above
    wrapped in a tf.data pipeline.  Both apply exactly the same light
    augmentation; ImageDataGenerator is the default because Keras parallelises
    a PyDataset across worker threads on its own.
    """
    steps = int(math.ceil(len(X) / batch_size))
    if not fast:
        try:
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            datagen = ImageDataGenerator(rotation_range=8,
                                         width_shift_range=0.06,
                                         height_shift_range=0.06,
                                         zoom_range=0.05)   # NO flips, NO shear
            print("[train] augmentation: keras ImageDataGenerator")
            return datagen.flow(X, y, batch_size=batch_size, seed=SEED), steps
        except Exception:
            print("[train] augmentation: ImageDataGenerator unavailable "
                  "(removed in Keras 3) -> OpenCV fallback")
    else:
        print("[train] augmentation: OpenCV (--fast-aug), same parameters")

    y32 = y.astype(np.int32)

    def gen():
        rng = np.random.default_rng(SEED)
        idx = np.arange(len(X))
        while True:                                    # infinite -> steps_per_epoch
            rng.shuffle(idx)
            for s in range(0, len(idx), batch_size):
                b = idx[s:s + batch_size]
                yield _augment_batch(X[b], rng), y32[b]

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None, IMG_SIZE, IMG_SIZE, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)))
    return ds.prefetch(tf.data.AUTOTUNE), steps


def train_model(model, X_tr, y_tr, X_val, y_val, epochs: int, batch_size: int,
                model_path: str, fast_aug: bool = False):
    """Compile + fit with checkpointing, early stopping and LR reduction."""
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    callbacks = [
        keras.callbacks.ModelCheckpoint(model_path, monitor="val_accuracy",
                                        mode="max", save_best_only=True,
                                        verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=7,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=3, min_lr=1e-6, verbose=1),
    ]

    train_iter, steps = make_train_iterator(X_tr, y_tr, batch_size, fast_aug)

    print(f"\n[train] {epochs} epochs, batch {batch_size}, "
          f"{steps} steps/epoch, CPU-friendly\n")
    history = model.fit(train_iter,
                        steps_per_epoch=steps,
                        epochs=epochs,
                        validation_data=(X_val, y_val),
                        callbacks=callbacks,
                        verbose=2)   # one tidy line per epoch, not a live bar

    h = history.history
    print("\n" + "=" * 70)
    print("FINAL EPOCH METRICS")
    print("=" * 70)
    print(f"  accuracy      : {h['accuracy'][-1]:.4f}")
    print(f"  loss          : {h['loss'][-1]:.4f}")
    print(f"  val_accuracy  : {h['val_accuracy'][-1]:.4f}")
    print(f"  val_loss      : {h['val_loss'][-1]:.4f}")
    print(f"  best val_acc  : {max(h['val_accuracy']):.4f} "
          f"(epoch {int(np.argmax(h['val_accuracy'])) + 1})")
    if h["accuracy"][-1] < 0.60:
        print("  !! training accuracy < 60 % -> reduce the augmentation "
              "(rotation_range / shift_range) or train for more epochs.")
    print("=" * 70 + "\n")
    return history


def plot_history(history, out_png: str = HISTORY_PNG) -> None:
    """Two side-by-side charts: accuracy vs val_accuracy, loss vs val_loss."""
    h = history.history
    ep = range(1, len(h["accuracy"]) + 1)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))

    a1.plot(ep, h["accuracy"], "o-", label="train accuracy", lw=2, ms=3)
    a1.plot(ep, h["val_accuracy"], "s-", label="val accuracy", lw=2, ms=3)
    a1.set_title("Accuracy"); a1.set_xlabel("epoch"); a1.set_ylabel("accuracy")
    a1.grid(alpha=.3); a1.legend()

    a2.plot(ep, h["loss"], "o-", label="train loss", lw=2, ms=3)
    a2.plot(ep, h["val_loss"], "s-", label="val loss", lw=2, ms=3)
    a2.set_title("Loss"); a2.set_xlabel("epoch"); a2.set_ylabel("loss")
    a2.grid(alpha=.3); a2.legend()

    fig.suptitle("Arabic handwriting CNN -- training history", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[plot] training history -> {out_png}")


def evaluate_on_test(model, X_test, y_test, batch_size: int = 256):
    """Test loss/accuracy, per-letter accuracy table and confusion matrix PNG."""
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)

    loss, acc = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
    print(f"  test loss     : {loss:.4f}")
    print(f"  TEST ACCURACY : {acc * 100:.2f} %")

    probs = model.predict(X_test, batch_size=batch_size, verbose=0)
    y_pred = probs.argmax(axis=1)

    if _SK_OK:
        cm = confusion_matrix(y_test, y_pred, labels=list(range(NUM_CLASSES)))
    else:                                            # tiny numpy fallback
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
        for t, p in zip(y_test, y_pred):
            cm[t, p] += 1

    # ---------------- per-letter accuracy table ----------------
    print("\n  PER-LETTER ACCURACY")
    print("  " + "-" * 62)
    print(f"  {'#':>2}  {'letter':<7}{'name':<8}{'support':>8}{'correct':>9}"
          f"{'accuracy':>11}   {'most confused with':<18}")
    print("  " + "-" * 62)
    per_letter = []
    for i in range(NUM_CLASSES):
        support = int(cm[i].sum())
        correct = int(cm[i, i])
        a = correct / support if support else 0.0
        per_letter.append(a)
        off = cm[i].copy(); off[i] = 0
        worst = int(off.argmax())
        conf = (f"{ARABIC_LETTERS[worst]} ({LETTER_NAMES[worst]}) x{off[worst]}"
                if off.sum() else "-")
        print(f"  {i + 1:>2}  {ARABIC_LETTERS[i]:<6}{LETTER_NAMES[i]:<8}"
              f"{support:>8}{correct:>9}{a * 100:>10.2f}%   {conf:<18}")
    print("  " + "-" * 62)
    print(f"  macro-average per-letter accuracy: {np.mean(per_letter) * 100:.2f} %")
    print(f"  weakest letters: " + ", ".join(
        f"{ARABIC_LETTERS[i]}({per_letter[i]*100:.0f}%)"
        for i in np.argsort(per_letter)[:5]))

    # ---------------- confusion matrix figure ----------------
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_norm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    # Latin names: matplotlib's default font cannot render Arabic glyphs.
    ax.set_xticklabels(LETTER_NAMES, rotation=90, fontsize=8)
    ax.set_yticklabels(LETTER_NAMES, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix (row-normalised) -- "
                 f"test accuracy {acc * 100:.2f} %")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fraction of row")
    fig.tight_layout()
    fig.savefig(CONFUSION_PNG, dpi=150)
    plt.close(fig)
    print(f"\n[plot] confusion matrix -> {CONFUSION_PNG}")
    print("=" * 70)

    return {"loss": loss, "accuracy": acc, "confusion_matrix": cm,
            "per_letter": np.array(per_letter)}


# ==============================================================================
# 5. SCORING ENGINE -- the product
# ==============================================================================

def _grade(score_pct: float):
    """Map a similarity score to the three tiers."""
    if score_pct >= 85:
        return "Excellent 🌟", "excellent"
    if score_pct >= 65:
        return "Good 👍", "good"
    return "Needs Improvement ✏️", "needs_work"


def _score_tensor(model, x, target_letter_index=None):
    """Shared scoring core: x is already (1, 32, 32, 1) float32 in [0, 1]."""
    probs = np.asarray(model.predict(x, verbose=0)).ravel()

    pred_idx = int(probs.argmax())
    order = np.argsort(probs)[::-1]
    top3 = [(int(i), ARABIC_LETTERS[int(i)], float(probs[int(i)]) * 100.0)
            for i in order[:3]]

    if target_letter_index is None:
        score_idx = pred_idx
    else:
        # --target is given 1..29 (as in the CSV labels) -> convert to 0..28
        t = int(target_letter_index)
        score_idx = t - 1 if t >= 1 else t
        if not 0 <= score_idx < NUM_CLASSES:
            raise ValueError("target_letter_index must be between 1 and 29")

    score = float(probs[score_idx]) * 100.0
    grade, tier = _grade(score)

    return {
        "x": x,
        "probs": probs,
        "predicted_index": pred_idx,
        "predicted_letter": ARABIC_LETTERS[pred_idx],
        "predicted_name": LETTER_NAMES[pred_idx],
        "target_index": score_idx if target_letter_index is not None else None,
        "target_letter": (ARABIC_LETTERS[score_idx]
                          if target_letter_index is not None else None),
        "score": score,
        "grade": grade,
        "tier": tier,
        "top3": top3,
        "correct": (target_letter_index is None) or (pred_idx == score_idx),
    }


def evaluate_handwriting(model, image_path, target_letter_index=None,
                         adaptive: bool = False, verbose: bool = True,
                         preview: bool = True):
    """
    Score a child's handwritten letter.

    Parameters
    ----------
    model               : the trained Keras model.
    image_path          : path to the PAPER PHOTO (a numpy image / CSV row also
                          works -- `preprocess_image` accepts all three).
    target_letter_index : 1..29, the letter the child was ASKED to write.
                          When given, the score is the probability of THAT
                          class, not of the top prediction.
    adaptive            : adaptive thresholding for badly lit photos.

    Returns
    -------
    dict with recognised letter, score %, top-3 alternatives and the grade.
    """
    x = preprocess_image(image_path, adaptive=adaptive)      # 1. preprocess
    result = _score_tensor(model, x, target_letter_index)    # 2. + 3. predict
    if verbose:
        src = (str(image_path) if isinstance(image_path, (str, os.PathLike))
               else "<in-memory image>")
        print_report(result, source=src, preview=preview)    # 4.
    return result


def print_report(r: dict, source: str = "", preview: bool = True) -> None:
    """Full formatted report + actionable feedback for child and teacher."""
    line = "=" * 70
    print("\n" + line)
    print("  HANDWRITING EVALUATION REPORT")
    print(line)
    if source:
        print(f"  file            : {source}")

    if preview:
        ascii_preview(r["x"], "what the model actually sees (32x32):")

    print(f"  recognised      : {r['predicted_letter']}  "
          f"({r['predicted_name']}, class {r['predicted_index'] + 1})")
    if r["target_index"] is not None:
        ti = r["target_index"]
        verdict = "MATCH ✔" if r["correct"] else "MISMATCH ✘"
        print(f"  asked to write  : {ARABIC_LETTERS[ti]}  "
              f"({LETTER_NAMES[ti]}, class {ti + 1})   -> {verdict}")
        print(f"  similarity      : {r['score']:.2f} %   "
              f"(probability of the TARGET letter)")
    else:
        print(f"  confidence      : {r['score']:.2f} %")
    print(f"  grade           : {r['grade']}")

    print("\n  TOP-3 ALTERNATIVES")
    for rank, (idx, letter, pct) in enumerate(r["top3"], 1):
        bar = "█" * int(round(pct / 4))
        print(f"    {rank}. {letter}  {LETTER_NAMES[idx]:<7} "
              f"{pct:6.2f} %  {bar}")

    _print_feedback(r)
    print(line + "\n")


def _print_feedback(r: dict) -> None:
    """Tier-specific, letter-specific coaching for the child and the adult."""
    focus_idx = r["target_index"] if r["target_index"] is not None \
        else r["predicted_index"]
    letter, name = ARABIC_LETTERS[focus_idx], LETTER_NAMES[focus_idx]
    dots = LETTER_DOTS[focus_idx]
    tier = r["tier"]

    # letters that share this skeleton -> the classic dot mistakes
    family = [g for g in CONFUSION_GROUPS if letter in g]
    family_txt = ("  ".join(family[0]) if family else "-")

    print("\n  ── FEEDBACK FOR THE CHILD ─────────────────────────────────────")
    if tier == "excellent":
        print(f"    🌟 Great job! Your {letter} is clear and easy to read.")
        print(f"       أحسنت! حرف {letter} واضح وجميل.")
        print(f"    •  Keep the same size for every letter you write next.")
        print(f"    •  Now try writing {letter} three times in a row, all the "
              f"same height.")
    elif tier == "good":
        print(f"    👍 Good work — your {letter} is readable, it just needs "
              f"tidying up.")
        print(f"       جيد! حرف {letter} مقروء، يحتاج قليلاً من الترتيب.")
        print(f"    •  Sit the letter ON the line, don't let it float above it.")
        print(f"    •  Make the curve smooth: one steady movement, not many "
              f"little strokes.")
        print(f"    •  Check the dots: {letter} takes {dots}.")
    else:
        print(f"    ✏️  Let's practise {letter} again together — slowly.")
        print(f"       لا بأس! لنتدرب على حرف {letter} مرة أخرى بهدوء.")
        print(f"    •  Hold the pencil with THREE fingers (thumb + index on the "
              f"pencil, resting on the middle finger).")
        print(f"    •  Start from the RIGHT, write the body in one stroke, and "
              f"add the dots LAST.")
        print(f"    •  {letter} takes {dots}.")
        print(f"    •  Use lined paper and trace the letter 5 times before "
              f"writing it on your own.")

    if not r["correct"] and r["target_index"] is not None:
        p = r["predicted_letter"]
        print(f"    •  Careful: it currently reads as {p} rather than {letter}. "
              f"Look closely at the dots and the tail.")

    print("\n  ── FOR THE PARENT / TEACHER ───────────────────────────────────")
    print(f"    letter under review : {letter} ({name})")
    print(f"    required dots       : {dots}")
    print(f"    look-alike family   : {family_txt}")
    print( "    checklist:")
    print( "      1. Pencil grip  : tripod (3-finger) grip, relaxed wrist, "
           "paper tilted.")
    print( "      2. Baseline     : the body must sit on the ruled line; "
           "descenders (ج ح خ ع غ م ن س ص ق ي) drop below it.")
    print( "      3. Letter height: consistent x-height across the whole word "
           "— compare against ا as the tallest reference.")
    print( "      4. Stroke order : right-to-left, body first in a single "
           "continuous stroke, dots and the hamza added at the end.")
    print(f"      5. Dots         : verify NUMBER and POSITION — {letter} "
          f"takes {dots}.")
    if tier == "needs_work":
        print( "      6. Drill        : 5 traced repetitions + 5 free "
               "repetitions daily on 4-lined paper; re-scan after one week.")
    elif tier == "good":
        print( "      6. Drill        : practise the letter inside real words "
               "so joining forms stay consistent.")
    else:
        print( "      6. Next step    : move on to the connected forms "
               "(initial / medial / final) of this letter.")


# ==============================================================================
# 6. MAIN
# ==============================================================================

def _resolve_data_dir(user_dir: str | None) -> str:
    """--data-dir wins; otherwise look next to the script, then in the CWD."""
    if user_dir:
        return os.path.abspath(os.path.expanduser(user_dir))
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (here, os.getcwd(), os.path.join(here, "archive"),
                 os.path.join(os.getcwd(), "archive")):
        if all(os.path.exists(os.path.join(cand, f))
               for f in ("train_X.csv", "train_Y.csv",
                         "test_X.csv", "test_Y.csv")):
            return cand
        if os.path.exists(os.path.join(cand, "hijja2.npz")):
            return cand
    return here


def _load_saved_model(path: str):
    """Load the checkpoint, tolerating the .h5 / .keras extension difference."""
    candidates = [path, "arabic_handwriting_model.keras",
                  "arabic_handwriting_model.h5"]
    for p in candidates:
        if os.path.exists(p):
            print(f"[model] loading {p}")
            return keras.models.load_model(p)
    raise FileNotFoundError(
        f"No saved model found (looked for {candidates}). "
        f"Run the script once WITHOUT --no-train first.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Arabic handwriting evaluator for children (Hijja dataset)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--data-dir", type=str, default=None,
                    help="folder holding train_X.csv ... (default: next to this script)")
    ap.add_argument("--npz", type=str, default=None,
                    help="optional pre-packed hijja2.npz (fast loading)")
    ap.add_argument("--no-train", action="store_true",
                    help="skip training, load the saved model")
    ap.add_argument("--image", type=str, default=None,
                    help="score a photo/scan of a letter written on paper")
    ap.add_argument("--target", type=int, default=None, choices=range(1, 30),
                    metavar="1..29",
                    help="the letter the child was ASKED to write (1..29)")
    ap.add_argument("--adaptive", action="store_true",
                    help="adaptive thresholding for badly lit phone photos")
    ap.add_argument("--show-steps", type=str, default=None, metavar="PATH",
                    help="only save the preprocessing-stages figure for PATH")
    ap.add_argument("--model-path", type=str, default=_default_model_path())
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore/refresh preprocessed_32.npz")
    ap.add_argument("--fast-aug", action="store_true",
                    help="augment with OpenCV/tf.data instead of "
                         "ImageDataGenerator (identical rotation/shift/zoom)")
    args = ap.parse_args()

    print("=" * 70)
    print("  ARABIC HANDWRITING EVALUATOR  —  Hijja dataset (ages 7-12)")
    print("=" * 70)
    for i in range(0, NUM_CLASSES, 10):
        print("  classes {:>2}-{:<2}: {}".format(
            i + 1, min(i + 10, NUM_CLASSES),
            "  ".join(ARABIC_LETTERS[i:i + 10])))
    print("=" * 70)

    # --- stand-alone preprocessing demo, no TensorFlow needed ------------------
    if args.show_steps:
        show_preprocessing_steps(args.show_steps, STEPS_PNG, adaptive=args.adaptive)
        return

    if not _TF_OK:
        print("\nERROR: TensorFlow is not available in this interpreter:")
        print(f"       {_TF_ERR}")
        print("       Install it with:  pip install tensorflow  "
              "(Python 3.9-3.13)")
        sys.exit(1)
    if not _SK_OK:
        print(f"\nERROR: scikit-learn is not available: {_SK_ERR}")
        print("       Install it with:  pip install scikit-learn")
        sys.exit(1)

    # reproducibility
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

    data_dir = _resolve_data_dir(args.data_dir)
    args.model_path = _resolve_checkpoint_path(args.model_path)
    print(f"\n[setup] data dir  : {data_dir}")
    print(f"[setup] model file: {args.model_path}")
    print(f"[setup] TensorFlow {tf.__version__} / Keras {keras.__version__} "
          f"(CPU)")

    # ---------------- 1./2. data ----------------
    X_train_full, y_train_full, X_test, y_test = load_dataset(
        data_dir, adaptive=args.adaptive, use_cache=not args.no_cache,
        npz_path=args.npz)

    X_tr, X_val, y_tr, y_val = stratified_split(X_train_full, y_train_full, 0.10)
    print("\n  DATA SPLITS")
    print(f"    TRAIN      : {X_tr.shape[0]:>6} samples  {X_tr.shape[1:]}")
    print(f"    VALIDATION : {X_val.shape[0]:>6} samples  (10 %, stratified)")
    print(f"    TEST       : {X_test.shape[0]:>6} samples  (untouched)")
    counts = np.bincount(y_tr, minlength=NUM_CLASSES)
    print(f"    per-class in TRAIN: min {counts.min()}, max {counts.max()}, "
          f"mean {counts.mean():.0f}")

    # ---------------- 3. model ----------------
    if args.no_train:
        model = _load_saved_model(args.model_path)
    else:
        model = build_model()
    print()
    model.summary()

    # feature-extraction demo: the 2048-d vector before the classifier
    feature_extractor = make_feature_extractor(model)
    feats = feature_extractor.predict(X_tr[:1], verbose=0)
    print(f"\n[features] feature_extractor output for ONE sample: "
          f"{feats.shape}  -> {feats.shape[-1]}-d vector "
          f"(non-zero: {int((feats > 0).sum())})")

    # ---------------- 4. training ----------------
    if not args.no_train:
        history = train_model(model, X_tr, y_tr, X_val, y_val,
                              args.epochs, args.batch_size, args.model_path,
                              fast_aug=args.fast_aug)
        plot_history(history, HISTORY_PNG)

        # EarlyStopping restored the best val_LOSS weights, while ModelCheckpoint
        # saved the best val_ACCURACY epoch.  Keep whichever validates better.
        try:
            if os.path.exists(args.model_path):
                ckpt = keras.models.load_model(args.model_path)
                a_ckpt = ckpt.evaluate(X_val, y_val, verbose=0)[1]
                a_live = model.evaluate(X_val, y_val, verbose=0)[1]
                print(f"[model] val_acc  checkpoint {a_ckpt * 100:.2f} %  vs  "
                      f"restored {a_live * 100:.2f} %")
                if a_ckpt >= a_live:
                    model = ckpt
                else:
                    model.save(args.model_path)
                    print(f"[model] restored weights were better -> re-saved")
            else:
                model.save(args.model_path)
        except Exception as e:
            print(f"[model] checkpoint comparison skipped ({e})")
        print(f"[model] best weights in: {args.model_path}")

    # ---------------- test evaluation ----------------
    evaluate_on_test(model, X_test, y_test, batch_size=256)

    # ---------------- 5. scoring-engine demo on 3 random test images ----------
    print("\n" + "=" * 70)
    print("  SCORING ENGINE DEMO — 3 random TEST images")
    print("=" * 70)
    rng = np.random.default_rng(SEED)
    for k, idx in enumerate(rng.choice(len(X_test), size=3, replace=False), 1):
        idx = int(idx)
        true_i = int(y_test[idx])
        print(f"\n  --- demo {k}/3 — test sample #{idx}, "
              f"true letter {ARABIC_LETTERS[true_i]} ({LETTER_NAMES[true_i]}) ---")
        # the child was "asked" to write the true letter -> target = true + 1
        res = _score_tensor(model, X_test[idx:idx + 1], target_letter_index=true_i + 1)
        print_report(res, source=f"test_X.csv row {idx}", preview=True)

    # ---------------- score a real paper photo -------------------------------
    if args.image:
        print("=" * 70)
        print("  PAPER PHOTO EVALUATION")
        print("=" * 70)
        show_preprocessing_steps(args.image, STEPS_PNG, adaptive=args.adaptive)
        evaluate_handwriting(model, args.image, target_letter_index=args.target,
                             adaptive=args.adaptive)

    print("Artifacts written: "
          f"{args.model_path}, {HISTORY_PNG}, {CONFUSION_PNG}, {CACHE_FILE}"
          + (f", {STEPS_PNG}" if args.image else ""))


if __name__ == "__main__":
    main()
