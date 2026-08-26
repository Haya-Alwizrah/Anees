#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 handwriting_cv.py -- Anees  |  Arabic Handwriting Quality & Similarity Scoring
================================================================================

ONE class, `AneesHandwritingCV`, that holds the whole computer-vision project:

    dataset -> Amiri reference bank -> labelled pairs -> three CNN regressors
            -> training -> evaluation -> calibration -> upload -> RATING

A child copies an Arabic letter from a Naskh model.  This scores **how
accurately they copied it** -- a continuous 0-100 Quality & Similarity rating.
It is not OCR: the letter is known in advance, the question is how well it was
written.

    photo -> page detection -> perspective warp -> ink extraction
          -> ruled-line removal -> segmentation -> 64x64 normalisation
          -> CNN regressor -> calibrated rating -> feedback for the child

--------------------------------------------------------------------------------
QUICK START (Colab, GPU runtime)

    cv = AneesHandwritingCV()
    cv.run_all()                       # dataset -> trained -> calibrated
    p = cv.upload()                    # pick a photo from your computer
    cv.show_rating(cv.rate(p[0], "ب"))

QUICK START (local, weights already trained)

    cv = AneesHandwritingCV().load()
    print(cv.rate("photo.jpg", "ب"))

--------------------------------------------------------------------------------
WHERE THE CONTINUOUS LABEL COMES FROM

Hijja2 records *which* letter a child wrote, never *how well*.  Inventing a
neatness number would teach the network our invention, so the labels come from
three sources and are always **evaluated separately**:

    A  the Amiri model degraded by a measured amount t   label = 1 - t   truth
    B  a real child's letter vs its Amiri model          geometric score proxy
    C  a real child's letter vs a *different* letter     label ~ 0       truth

Strong results on B alone would only prove the network memorised the geometric
formula.  Results on A and C show it learned to see distortion and letter
mismatch directly.  `evaluate()` prints all three.

--------------------------------------------------------------------------------
THE THREE MODELS

    A  "scratch"      CNN from scratch, Conv->MaxPool x4 + dense   (unit CV_2)
    B  "mobilenet"    MobileNetV2 feature extraction -> fine-tune  (unit CV_3)
    C  "production"   residual CNN, 6-channel geometric input,
                      joint-affine augmentation, cosine schedule,
                      3-seed ensemble + test-time augmentation

A and B exist so the two architectures from the unit can be compared honestly on
identical data.  C is the one the app ships: it is given the distance-transform
channels the classical metric uses, so it can *see* the geometry rather than
having to re-derive it from raw pixels.

Measured at a small equal budget (9k pairs, 14 epochs, no ensemble), C leads on
overall MAE but not by much -- 9.49 vs 9.79 points -- and it is slightly BEHIND
A on the degradation and wrong-letter slices.  Where it clearly wins is the
slice that matters for the app: real children's letters, 4.53 vs 6.18 MAE, and
the share of ratings landing within 5 points, 41.5% vs 35.9%.  C has three
times the parameters, so a small budget starves it more than it starves A; the
gap at the full 90k-pair, 35-epoch, 2-member setting is untested here.

--------------------------------------------------------------------------------
DATA

Hijja2 -- 47,434 Arabic characters handwritten by 591 Saudi school children aged
7-12, collected in Riyadh, Jan-Apr 2019.  32x32 grayscale, 29 classes (28
letters + hamza), split 37,933 train / 9,501 test.  Shipped here as the packed
`database cv/hijja2.npz` (5.9 MB) instead of 101 MB of CSVs.

Amiri -- the classical Naskh face from Google Fonts, rendered as the reference
"model" the child is copying, in all four contextual forms per letter.

DEPENDENCIES
    tensorflow, opencv-python, scikit-image, scipy, Pillow, matplotlib, numpy
    (optional: arabic-reshaper + python-bidi, for Arabic labels in figures)
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import sys
import io
import json
import math
import time
import glob
import zipfile
import urllib.request

import numpy as np
import cv2

from PIL import Image, ImageDraw, ImageFont, features
from skimage.morphology import skeletonize
from skimage.metrics import structural_similarity as ssim
from scipy import ndimage

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# TensorFlow is only needed for training and for the CNN rating.  The classical
# half of this file (preprocessing, the geometric scorer, the reference bank)
# runs without it, so it is imported defensively and the error is readable.
try:
    import tensorflow as tf
    _TF_OK, _TF_ERR = True, None
except Exception as _e:                                       # pragma: no cover
    tf, _TF_OK, _TF_ERR = None, False, _e


# ==============================================================================
#  CONSTANTS
# ==============================================================================

# 28 Arabic letters in alphabetical order + hamza last -> Hijja2 labels 1..29
LETTERS = list("ابتثجحخدذرزسشصضطظعغفقكلمنهوي") + ["ء"]

# matplotlib's default font has no Arabic glyphs, so figures are labelled with
# these ASCII names instead of with the letters themselves.
LETTER_NAMES = [
    "alef", "beh", "teh", "theh", "jeem", "hah", "khah", "dal", "thal", "reh",
    "zain", "seen", "sheen", "sad", "dad", "tah", "zah", "ain", "ghain", "feh",
    "qaf", "kaf", "lam", "meem", "noon", "heh", "waw", "yeh", "hamza",
]

ZWJ = "‍"                      # zero-width joiner: forces contextual forms
FORMS = ("isolated", "initial", "medial", "final")

FONT_URLS = [
    "https://raw.githubusercontent.com/google/fonts/main/ofl/amiri/Amiri-Regular.ttf",
    "https://github.com/google/fonts/raw/main/ofl/amiri/Amiri-Regular.ttf",
    "https://cdn.jsdelivr.net/gh/google/fonts@main/ofl/amiri/Amiri-Regular.ttf",
]

_HERE = os.path.dirname(os.path.abspath(__file__))

# Default configuration.  Override any key through the constructor:
#     AneesHandwritingCV(n_train_pairs=20000, train_mobilenet=False)
DEFAULTS = dict(
    # ---- geometry -----------------------------------------------------------
    side          = 64,     # working canvas for a normalised glyph
    inner         = 52,     # the glyph is scaled to fit inside this box
    thick         = 2,      # every stroke is re-inflated to this width, so the
                            # network cannot cheat by reading pen thickness
    tol           = 2,      # tolerance radius (px) for the IoU term
    photo_canvas  = 128,    # canvas used by the classical photo front end
    photo_content = 96,

    # ---- the classical geometric scorer (weights must sum to 1.0) -----------
    w_ssim = 0.35, w_iou = 0.40, w_haus = 0.25,

    # ---- pair construction --------------------------------------------------
    n_train_pairs = 90000,
    n_test_pairs  = 15000,
    pair_mix      = (0.40, 0.40, 0.20),   # A / B / C
    val_split     = 0.15,

    # ---- training -----------------------------------------------------------
    train_scratch    = True,
    train_mobilenet  = True,
    train_production = True,
    epochs           = 35,
    ft_epochs        = 14,     # MobileNetV2 fine-tuning phase
    batch            = 256,
    lr               = 1.5e-3,
    warmup_epochs    = 3,
    ensemble         = 2,      # production models averaged (1 disables it)
    tta              = True,   # test-time augmentation for the production model
    mobilenet_side   = 96,
    unfreeze_layers  = 60,

    # ---- misc ---------------------------------------------------------------
    seed        = 42,
    pass_mark   = 70.0,        # at or above this the letter counts as correct
    artifacts   = os.path.join(_HERE, "artifacts"),
    font_path   = os.path.join(_HERE, "fonts", "Amiri-Regular.ttf"),
    verbose     = True,
)


def _in_colab():
    try:
        import google.colab            # noqa: F401
        return True
    except Exception:
        return False


# ==============================================================================
#  THE CLASS
# ==============================================================================
class AneesHandwritingCV:
    """
    The whole Arabic-handwriting-quality pipeline behind one object.

    Lifecycle
    ---------
        cv = AneesHandwritingCV()
        cv.load_dataset()          # Hijja2
        cv.build_reference_bank()  # Amiri, 29 letters x 4 contextual forms
        cv.build_pairs()           # labelled (child, model) pairs, sources A/B/C
        cv.train()                 # the three regressors
        cv.evaluate()              # MAE / RMSE / R2 / Pearson, per label source
        cv.calibrate()             # raw score -> "neater than N% of children"
        cv.save()                  # weights + bank + curve -> artifacts/

        cv.rate("photo.jpg", "ب")  # the rating a child actually sees

    `run_all()` does the first seven in order.  `load()` restores a trained
    object from `artifacts/` without touching the dataset.
    """

    # ------------------------------------------------------------------ init --
    def __init__(self, **cfg):
        unknown = set(cfg) - set(DEFAULTS)
        if unknown:
            raise TypeError(f"unknown config key(s): {sorted(unknown)}")
        self.cfg = dict(DEFAULTS)
        self.cfg.update(cfg)

        self.letters = list(LETTERS)
        self.letter_names = list(LETTER_NAMES)
        self.n_classes = len(self.letters)

        # dataset
        self.Xtr = self.ytr = self.Xte = self.yte = None
        # Amiri reference bank
        self.bank = None            # {class: {form_name: canvas}}
        self.ref_arr = None         # (n_class+1, max_forms, S, S) uint8
        self.form_names = None      # {class: [form_name, ...]}
        self._ref_dt = None         # cached distance transform of every ref
        # pairs
        self.train_pairs = None
        self.test_pairs = None
        # models / results
        self.models = {}
        self.histories = {}
        self.results = {}
        self.calibration = None        # the default curve (the best CNN's)
        self.calibrations = {}         # one curve per scorer, by name
        self.calibration_source = None
        self._font_ready = False

        os.makedirs(self.cfg["artifacts"], exist_ok=True)
        np.random.seed(self.cfg["seed"])
        if _TF_OK:
            tf.random.set_seed(self.cfg["seed"])

    # ------------------------------------------------------------------ misc --
    def _log(self, *a):
        if self.cfg["verbose"]:
            print(*a, flush=True)

    def _require_tf(self):
        if not _TF_OK:
            raise ImportError(
                "TensorFlow is required for this step but could not be "
                f"imported: {_TF_ERR}\nTensorFlow supports Python 3.9-3.12; "
                "on Colab it is already installed.")

    @property
    def S(self):
        return self.cfg["side"]

    def device_report(self):
        """Print what we are about to train on -- worth checking before a run."""
        if not _TF_OK:
            self._log("TensorFlow: NOT AVAILABLE ->", _TF_ERR)
            return {"tensorflow": None, "gpu": []}
        gpus = tf.config.list_physical_devices("GPU")
        names = [g.name for g in gpus]
        self._log(f"TensorFlow {tf.__version__} | GPU: {names or 'NONE (CPU only)'}")
        if not gpus:
            self._log("  no GPU -- on Colab use Runtime -> Change runtime type -> T4 GPU")
        return {"tensorflow": tf.__version__, "gpu": names}

    # ==========================================================================
    #  1.  THE AMIRI FONT AND ARABIC TEXT SHAPING
    # ==========================================================================
    def ensure_font(self, path=None):
        """Find Amiri locally, download it, or fall back to an installed copy."""
        if path:
            if not os.path.exists(path):
                raise FileNotFoundError(f"font not found: {path}")
            self.cfg["font_path"] = path

        fp = self.cfg["font_path"]
        if os.path.exists(fp) and os.path.getsize(fp) > 100_000:
            self._init_mpl_font()
            return fp

        os.makedirs(os.path.dirname(fp), exist_ok=True)
        for url in FONT_URLS:
            try:
                self._log(f"downloading Amiri from {url} ...")
                urllib.request.urlretrieve(url, fp)
                if os.path.getsize(fp) > 100_000:
                    self._log(f"saved -> {fp}")
                    self._init_mpl_font()
                    return fp
            except Exception as e:
                self._log(f"  failed: {e}")
            if os.path.exists(fp):
                os.remove(fp)

        for f in fm.findSystemFonts():
            b = os.path.basename(f).lower()
            if "amiri" in b and "regular" in b:
                self._log("using system font", f)
                self.cfg["font_path"] = f
                self._init_mpl_font()
                return f
        raise RuntimeError(
            "Could not obtain Amiri-Regular.ttf. Download it from "
            "https://fonts.google.com/specimen/Amiri and pass font_path=...")

    def _init_mpl_font(self):
        """Let matplotlib render Arabic in figure titles."""
        if self._font_ready:
            return
        try:
            fm.fontManager.addfont(self.cfg["font_path"])
            self.AMIRI_MPL = fm.FontProperties(fname=self.cfg["font_path"], size=15)
        except Exception:
            self.AMIRI_MPL = None
        self._font_ready = True

    @staticmethod
    def _has_raqm():
        try:
            return bool(features.check("raqm"))
        except Exception:
            return False

    def shape_arabic(self, text):
        """
        Arabic is cursive and right-to-left: the raw code points must be
        reordered and swapped for their contextual forms before rasterising.
        Returns (text_ready_for_PIL, using_raqm).
        """
        if self._has_raqm():
            return text, True
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
            return get_display(arabic_reshaper.reshape(text)), False
        except Exception:
            return text, False

    @staticmethod
    def _mpl_shapes_arabic():
        # matplotlib >= 3.11 lays text out with libraqm and shapes Arabic on its
        # own; feeding it pre-reshaped text then reverses it a second time and
        # the label comes out backwards.  Detect the mechanism directly.
        if os.environ.get("ARABIC_SHAPE") == "0":
            return True
        if os.environ.get("ARABIC_SHAPE") == "1":
            return False
        try:
            from matplotlib.ft2font import FT2Font
            return hasattr(FT2Font, "_layout")
        except Exception:
            return False

    def ar(self, text):
        """An Arabic string prepared for a matplotlib label."""
        if self._mpl_shapes_arabic():
            return text
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
            return get_display(arabic_reshaper.reshape(text))
        except Exception:
            return text

    # ==========================================================================
    #  2.  THE DATASET  (Hijja2)
    # ==========================================================================
    def find_dataset(self):
        """
        Locate Hijja2 without being told where it is.  Looks for the packed
        .npz first (5.9 MB, loads instantly), then for the four raw CSVs, in
        every place this project is normally run from.
        """
        # "database cv" is where this project keeps it; "datasets" is kept as a
        # fallback so the file is still found if it is dropped in beside the
        # other datasets in Anees, or left in the working directory.
        folders = ["database cv", "datasets"]
        npz_spots = [os.path.join(_HERE, "hijja2.npz"), "hijja2.npz"]
        for d in folders:
            npz_spots += [
                os.path.join(_HERE, d, "hijja2.npz"),
                os.path.join(_HERE, "..", d, "hijja2.npz"),
                os.path.join(d, "hijja2.npz"),
                os.path.join("/content", d, "hijja2.npz"),
            ]
        npz_spots += [
            "/content/hijja2.npz",
            "/content/drive/MyDrive/hijja2/hijja2.npz",
            "/content/drive/MyDrive/hijja2.npz",
        ]
        for p in npz_spots:
            if os.path.exists(p):
                return ("npz", os.path.abspath(p))

        csv_spots = [
            os.path.join(_HERE, "archive"),
            os.path.join(_HERE, "..", "archive"),
        ] + [os.path.join(_HERE, "..", d, "archive") for d in folders] + [
            "archive", "./", "/content/archive", "/content",
        ]
        need = ("train_X.csv", "train_Y.csv", "test_X.csv", "test_Y.csv")
        for d in csv_spots:
            if all(os.path.exists(os.path.join(d, n)) for n in need):
                return ("csv", os.path.abspath(d))
        return (None, None)

    def load_dataset(self, path=None, cache=True):
        """
        Load Hijja2 into memory.  `path` may be the .npz, the folder holding
        the four CSVs, or None to search.  Reading the CSVs takes ~20 s and is
        cached to .npz afterwards, so it only ever happens once.
        """
        kind, loc = ("npz", path) if (path and str(path).endswith(".npz")) \
            else (("csv", path) if path else self.find_dataset())

        if kind is None:
            raise FileNotFoundError(
                "Could not find Hijja2.\n"
                "  * this repo ships it as  'database cv/hijja2.npz'\n"
                "  * on Colab, upload hijja2.npz with the folder icon on the left\n"
                "  * or pass load_dataset('/path/to/hijja2.npz')")

        if kind == "npz":
            self._log(f"loading Hijja2 <- {loc}")
            d = np.load(loc)
            self.Xtr, self.ytr = d["Xtr"], d["ytr"]
            self.Xte, self.yte = d["Xte"], d["yte"]
        else:
            self._log(f"reading the Hijja2 CSVs from {loc} (about 20 s, once) ...")
            j = lambda n: os.path.join(loc, n)
            self.Xtr = np.loadtxt(j("train_X.csv"), delimiter=",", dtype=np.uint8)
            self.ytr = np.loadtxt(j("train_Y.csv"), dtype=np.int16)
            self.Xte = np.loadtxt(j("test_X.csv"), delimiter=",", dtype=np.uint8)
            self.yte = np.loadtxt(j("test_Y.csv"), dtype=np.int16)
            if cache:
                out = os.path.join(self.cfg["artifacts"], "hijja2.npz")
                np.savez_compressed(out, Xtr=self.Xtr, ytr=self.ytr,
                                    Xte=self.Xte, yte=self.yte)
                self._log(f"cached -> {out}")

        self._log(f"Hijja2: {len(self.Xtr):,} train / {len(self.Xte):,} test letters, "
                  f"{len(np.unique(self.ytr))} classes")
        return self

    def dataset_summary(self):
        """Counts, class balance and the per-class table."""
        if self.Xtr is None:
            raise RuntimeError("call load_dataset() first")
        counts = np.bincount(self.ytr, minlength=self.n_classes + 1)[1:]
        cte = np.bincount(self.yte, minlength=self.n_classes + 1)[1:]
        out = {
            "n_train": int(len(self.Xtr)), "n_test": int(len(self.Xte)),
            "n_total": int(len(self.Xtr) + len(self.Xte)),
            "n_classes": self.n_classes,
            "image": "32x32 grayscale, ink white on black",
            "per_class_train": {self.letter_names[i]: int(counts[i])
                                for i in range(self.n_classes)},
            "min_class": int(counts.min()), "max_class": int(counts.max()),
            "imbalance_ratio": round(float(counts.max() / max(counts.min(), 1)), 2),
        }
        if self.cfg["verbose"]:
            print(f"\nHijja2 -- 591 Saudi children aged 7-12, Riyadh 2019")
            print(f"  train {out['n_train']:,}   test {out['n_test']:,}   "
                  f"total {out['n_total']:,}   classes {out['n_classes']}")
            print(f"  smallest class {out['min_class']}, largest {out['max_class']} "
                  f"(ratio {out['imbalance_ratio']}x -- near balanced)")
        return out

    def show_dataset(self, n=40, figsize=(14, 7), show=True):
        """A grid of real children's letters, straight from the CSV rows."""
        if self.Xtr is None:
            raise RuntimeError("call load_dataset() first")
        rng = np.random.default_rng(self.cfg["seed"])
        idx = rng.choice(len(self.Xtr), size=min(n, len(self.Xtr)), replace=False)
        cols = 10
        rows = int(math.ceil(len(idx) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        for ax, i in zip(np.ravel(axes), idx):
            ax.imshow(self.Xtr[i].reshape(32, 32), cmap="gray")
            ax.set_title(self.letter_names[int(self.ytr[i]) - 1], fontsize=8)
            ax.axis("off")
        for ax in np.ravel(axes)[len(idx):]:
            ax.axis("off")
        fig.suptitle("Hijja2 -- real letters written by children aged 7-12", fontsize=13)
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def show_class_distribution(self, figsize=(13, 4), show=True):
        """Bar chart of the 29 class counts -- the EDA figure."""
        if self.Xtr is None:
            raise RuntimeError("call load_dataset() first")
        counts = np.bincount(self.ytr, minlength=self.n_classes + 1)[1:]
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(range(self.n_classes), counts, color="#3b6ea5")
        ax.set_xticks(range(self.n_classes))
        ax.set_xticklabels(self.letter_names, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("training samples")
        ax.set_title("Hijja2 class distribution (37,933 training letters)")
        ax.axhline(counts.mean(), color="#c0392b", ls="--", lw=1,
                   label=f"mean {counts.mean():.0f}")
        ax.legend()
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    # ==========================================================================
    #  3.  NORMALISATION -- the one function both sides go through
    # ==========================================================================
    @staticmethod
    def _bbox_crop(binary):
        ys, xs = np.where(binary > 0)
        if xs.size == 0:
            return None
        return binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    def normalise(self, binary, side=None, inner=None, thick=None):
        """
        Crop to the ink, scale (aspect preserved) into `inner`, centre on
        `side`, thin to one pixel, then re-inflate to a fixed thickness.

        Applied identically to the child's writing and to the Amiri model, so
        the comparison is apples-to-apples and the network cannot score a
        letter well just because the child pressed harder with the pen.
        """
        side = side or self.cfg["side"]
        inner = inner or self.cfg["inner"]
        thick = thick or self.cfg["thick"]

        crop = self._bbox_crop(binary)
        if crop is None:
            return np.zeros((side, side), np.uint8)
        h, w = crop.shape
        s = inner / float(max(h, w))
        nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
        crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
        crop = (crop > 96).astype(np.uint8)
        if crop.sum() == 0:
            return np.zeros((side, side), np.uint8)
        sk = skeletonize(crop.astype(bool)).astype(np.uint8)
        if thick > 1:
            sk = cv2.dilate(sk, np.ones((thick, thick), np.uint8))
        out = np.zeros((side, side), np.uint8)
        oy, ox = (side - nh) // 2, (side - nw) // 2
        out[oy:oy + nh, ox:ox + nw] = sk * 255
        return out

    def child_to_canvas(self, flat32):
        """One Hijja2 row (1024 uint8) -> a normalised canvas."""
        img = np.asarray(flat32, np.uint8).reshape(32, 32)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
        _, b = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)
        return self.normalise(b)

    # ==========================================================================
    #  4.  THE AMIRI REFERENCE BANK -- 29 letters x 4 contextual forms
    # ==========================================================================
    def render_glyph(self, text, px=300):
        """Raw binary render of `text` in Amiri (no normalisation yet)."""
        self.ensure_font()
        shaped, raqm = self.shape_arabic(text)
        font = ImageFont.truetype(
            self.cfg["font_path"], px,
            layout_engine=ImageFont.Layout.RAQM if raqm else ImageFont.Layout.BASIC)
        pad = px
        img = Image.new("L", (px * max(len(text), 1) + 2 * pad, px * 3), 0)
        d = ImageDraw.Draw(img)
        kw = dict(direction="rtl", language="ar") if raqm else {}
        d.text((pad, pad), shaped, font=font, fill=255, **kw)
        raw = (np.array(img) > 60).astype(np.uint8) * 255
        if raw.max() == 0:
            raise ValueError(f"Amiri produced an empty render for {text!r}")
        return raw

    def build_reference_bank(self, px=300, dedupe=True, force=False):
        """
        {class_index: {form_name: canvas}} for 29 classes x 4 contextual forms.

        A zero-width joiner before / after a letter forces its initial, medial
        or final shape.  This matters: a child's initial بـ must not be judged
        against an isolated ب, and Hijja2 does not record which form was asked
        for -- `best_form()` recovers it at scoring time.
        """
        cached = os.path.join(self.cfg["artifacts"], "amiri_reference_bank.npz")
        if not force and self.bank is None and os.path.exists(cached):
            try:
                self._load_bank(cached)
                self._log(f"reference bank <- {cached}")
                return self
            except Exception:
                pass
        if self.bank is not None and not force:
            return self

        self.ensure_font()
        self._log("rendering the Amiri reference bank (29 letters x 4 forms) ...")
        bank = {}
        for ci, ch in enumerate(self.letters, start=1):
            variants = {"isolated": ch, "initial": ch + ZWJ,
                        "medial": ZWJ + ch + ZWJ, "final": ZWJ + ch}
            got = {}
            for name, text in variants.items():
                try:
                    raw = self.render_glyph(text, px)
                    if raw is not None and raw.max() > 0:
                        got[name] = self.normalise(raw)
                except Exception:
                    pass
            if not got:                       # should not happen with Amiri
                got = {"isolated": np.zeros((self.S, self.S), np.uint8)}
            bank[ci] = got

        self.bank = self._dedupe_bank(bank) if dedupe else bank
        self._pack_bank()
        n = sum(len(v) for v in self.bank.values())
        self._log(f"  {n} distinct reference glyphs "
                  f"({n / self.n_classes:.1f} forms per letter on average)")
        return self

    def _dedupe_bank(self, bank, tol=0.98):
        """Drop forms that render identically -- د has no medial variant."""
        out = {}
        for ci, forms in bank.items():
            keep = {}
            for name in FORMS:                 # keep a stable, meaningful order
                if name not in forms:
                    continue
                img = forms[name]
                if not any(self._iou(img, k) > tol for k in keep.values()):
                    keep[name] = img
            out[ci] = keep or {"isolated": forms[list(forms)[0]]}
        return out

    @staticmethod
    def _iou(a, b):
        A, B = a > 0, b > 0
        u = np.logical_or(A, B).sum()
        return float(np.logical_and(A, B).sum() / u) if u else 0.0

    def _pack_bank(self):
        """Pack the bank into a dense (n_class+1, max_forms, S, S) uint8 array."""
        mx = max(len(v) for v in self.bank.values())
        arr = np.zeros((self.n_classes + 1, mx, self.S, self.S), np.uint8)
        names = {}
        for c, forms in self.bank.items():
            fn = [f for f in FORMS if f in forms]
            names[c] = fn
            for j, n in enumerate(fn):
                arr[c, j] = forms[n]
        self.ref_arr, self.form_names = arr, names
        # distance transform of every reference, cached once and reused for
        # every pair that points at it -- this is the Hausdorff channel.
        self._ref_dt = np.stack([[self._decay_dt(arr[c, j]) for j in range(mx)]
                                 for c in range(self.n_classes + 1)]).astype(np.uint8)
        return arr, names

    def _decay_dt(self, binary, tau=None):
        """
        exp(-distance / tau), as uint8.  1.0 on the ink, falling off smoothly
        away from it -- a soft "how far is this pixel from the stroke" field.
        This is exactly what the Hausdorff term measures, handed to the network
        directly instead of making it re-derive the idea from raw pixels.
        """
        tau = tau or (0.12 * self.S)
        b = (binary > 0).astype(np.uint8)
        if b.max() == 0:
            return np.zeros_like(b, np.uint8)
        d = cv2.distanceTransform(1 - b, cv2.DIST_L2, 3)
        return np.clip(np.exp(-d / tau) * 255.0, 0, 255).astype(np.uint8)

    def best_form(self, child_canvas, forms_dict=None, cls=None, tol=3):
        """
        Which contextual form was the child writing?  Hijja2 labels the letter
        but not the form, so we pick the form the writing matches best.
        Returns (form_name, tolerance_iou).
        """
        if forms_dict is None:
            forms_dict = self.bank[int(cls)]
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
        c = cv2.dilate(child_canvas, k) > 0
        best, name = -1.0, None
        for n, ref in forms_dict.items():
            r = cv2.dilate(ref, k) > 0
            u = np.logical_or(c, r).sum()
            s = float(np.logical_and(c, r).sum() / u) if u else 0.0
            if s > best:
                best, name = s, n
        return name, best

    def show_reference_bank(self, letters=None, figsize=(14, 8), show=True):
        """The reference figure: every letter in every contextual form."""
        self.build_reference_bank()
        idx = letters or list(range(1, self.n_classes + 1))
        rows = len(idx)
        cols = max(len(self.form_names[c]) for c in idx)
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = np.atleast_2d(axes)
        for r, c in enumerate(idx):
            for j in range(cols):
                ax = axes[r, j]
                ax.axis("off")
                if j < len(self.form_names[c]):
                    ax.imshow(self.ref_arr[c, j], cmap="gray")
                    if r == 0:
                        ax.set_title(self.form_names[c][j], fontsize=9)
            axes[r, 0].set_ylabel(self.letter_names[c - 1])
            axes[r, 0].axis("on")
            axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        fig.suptitle("The Amiri Naskh model the child is copying", fontsize=13)
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def _load_bank(self, path):
        b = np.load(path, allow_pickle=True)
        self.ref_arr = b["ref"]
        self.form_names = {int(k): list(v) for k, v in b["forms"][0].items()}
        self.letters = list(b["letters"]) if "letters" in b else self.letters
        self.bank = {c: {n: self.ref_arr[c, j] for j, n in enumerate(fn)}
                     for c, fn in self.form_names.items()}
        mx = self.ref_arr.shape[1]
        self._ref_dt = np.stack(
            [[self._decay_dt(self.ref_arr[c, j]) for j in range(mx)]
             for c in range(self.ref_arr.shape[0])]).astype(np.uint8)

    # ==========================================================================
    #  5.  DEGRADATIONS -- the label source that is ground truth by construction
    # ==========================================================================
    @staticmethod
    def _elastic(img, alpha, sigma, rng):
        """
        Wobble a glyph along a smooth random displacement field, so it looks
        drawn by hand rather than typeset.

        The field is normalised to unit standard deviation BEFORE being scaled
        by `alpha`, which makes `alpha` mean "pixels of wobble" whatever the
        image size and smoothing radius are. Without that step the blur of
        white noise shrinks the amplitude by a factor of roughly sigma, so the
        same alpha that visibly warps a 64 px canvas does nothing at all to a
        1000 px page -- the wobble silently disappears on the larger image.
        """
        h, w = img.shape[:2]

        def field():
            f = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32) * 2 - 1,
                                 (0, 0), sigma)
            sd = float(f.std())
            return (f / sd if sd > 1e-8 else f) * alpha

        xx, yy = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
        return cv2.remap(img, xx + field(), yy + field(), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def degrade(self, canvas, t, rng):
        """
        Apply a controlled amount `t` (0 = untouched, 1 = badly wrong) of the
        four errors children actually make:

            1. a wobbly, uneven stroke path
            2. the whole letter tilted
            3. wrong proportions -- too tall or too wide
            4. a piece of the stroke missing, or overshooting

        The amount IS the label.  It is ground truth by construction, not a
        judgement we invented after the fact.
        """
        img = canvas.copy()
        if t <= 0:
            return img
        S = self.S
        img = self._elastic(img, alpha=2.6 * t,
                            sigma=max(3.0, 9.0 - 4.0 * t), rng=rng)
        ang = rng.normal(0, 14 * t)
        M = cv2.getRotationMatrix2D((S / 2, S / 2), ang, 1.0)
        sx = 1.0 + rng.normal(0, 0.30 * t)
        sy = 1.0 + rng.normal(0, 0.30 * t)
        M[0, :] *= sx
        M[1, :] *= sy
        M[0, 2] += (1 - sx) * S / 2 + rng.normal(0, 4 * t)
        M[1, 2] += (1 - sy) * S / 2 + rng.normal(0, 4 * t)
        img = cv2.warpAffine(img, M, (S, S), flags=cv2.INTER_LINEAR, borderValue=0)
        if rng.random() < 0.75 * t:
            ys, xs = np.where(img > 0)
            if xs.size > 20:
                i = rng.integers(len(xs))
                r = int(6 + 10 * t)
                cv2.circle(img, (int(xs[i]), int(ys[i])), r, 0, -1)
        img = (img > 60).astype(np.uint8) * 255
        return self.normalise(img)

    def show_degradations(self, letter="ب", n=8, figsize=(15, 2.6), show=True):
        """The label scale, drawn from perfect to badly wrong."""
        self.build_reference_bank()
        ci = self.letters.index(letter) + 1
        ref = self.ref_arr[ci, 0]
        rng = np.random.default_rng(self.cfg["seed"])
        ts = np.linspace(0, 1, n)
        fig, axes = plt.subplots(1, n, figsize=figsize)
        for ax, t in zip(axes, ts):
            ax.imshow(self.degrade(ref, float(t), rng), cmap="gray")
            ax.set_title(f"label {1 - t:.2f}", fontsize=9)
            ax.axis("off")
        fig.suptitle("Source A -- the Amiri model degraded by a measured amount",
                     fontsize=12)
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    # ==========================================================================
    #  6.  THE CLASSICAL GEOMETRIC SCORER  (label source B, and a 2nd opinion)
    # ==========================================================================
    @staticmethod
    def _soft(sk, sigma=1.6):
        f = cv2.GaussianBlur(sk.astype(np.float32) / 255.0, (0, 0), sigma)
        m = f.max()
        return f / m if m > 0 else f

    def geom_score(self, child, ref, tol=None):
        """
        The validated classical similarity, 0..1, on two normalised canvases:

            0.35 * SSIM  +  0.40 * tolerance-IoU  +  0.25 * Hausdorff term

        SSIM catches overall shape, IoU catches coverage, and the 95th-
        percentile Hausdorff distance catches the one badly misplaced stroke
        that the first two would average away.
        """
        tol = tol or self.cfg["tol"]
        if child.max() == 0 or ref.max() == 0:
            return 0.0
        s = float(np.clip(ssim(self._soft(child), self._soft(ref),
                               data_range=1.0), 0, 1))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
        a, b = cv2.dilate(child, k) > 0, cv2.dilate(ref, k) > 0
        u = np.logical_or(a, b).sum()
        iou = float(np.logical_and(a, b).sum() / u) if u else 0.0
        A, B = child > 0, ref > 0
        dB = cv2.distanceTransform((~B).astype(np.uint8), cv2.DIST_L2, 3)
        dA = cv2.distanceTransform((~A).astype(np.uint8), cv2.DIST_L2, 3)
        h95 = max(np.percentile(dB[A], 95), np.percentile(dA[B], 95))
        hs = float(np.exp(-h95 / (0.12 * self.S)))
        c = self.cfg
        return float(np.clip(c["w_ssim"] * s + c["w_iou"] * iou + c["w_haus"] * hs,
                             0, 1))

    # ==========================================================================
    #  7.  PAIR CONSTRUCTION -- the three honest label sources
    # ==========================================================================
    def build_pairs(self, n_pairs=None, split="train", seed=None, mix=None):
        """
        Build labelled (child, model) pairs and remember them on the object.

        Returns a dict with:
            childs (N,S,S) uint8   the child's normalised stroke
            rcls   (N,)    int16   which letter the model glyph is
            rfrm   (N,)    int16   which contextual form
            y      (N,)    float32 the label, 0..1
            kinds  (N,)    'A'/'B'/'C'  which source the label came from

        Only the child canvas is stored, not the assembled tensor: the six
        input channels are built per batch in the tf.data pipeline, which keeps
        120,000 pairs inside 500 MB instead of 12 GB.
        """
        self.build_reference_bank()
        if self.Xtr is None:
            self.load_dataset()

        train = (split == "train")
        X, y_lab = (self.Xtr, self.ytr) if train else (self.Xte, self.yte)
        n_pairs = n_pairs or (self.cfg["n_train_pairs"] if train
                              else self.cfg["n_test_pairs"])
        mix = mix or self.cfg["pair_mix"]
        seed = self.cfg["seed"] + (1 if train else 2) if seed is None else seed
        rng = np.random.default_rng(seed)
        S = self.S

        nA, nB, nC = (int(n_pairs * m) for m in mix)
        nA += n_pairs - (nA + nB + nC)

        childs = np.zeros((n_pairs, S, S), np.uint8)
        rcls = np.zeros(n_pairs, np.int16)
        rfrm = np.zeros(n_pairs, np.int16)
        labels = np.zeros(n_pairs, np.float32)
        kinds = np.empty(n_pairs, dtype="U1")

        cache = {}                     # normalised child canvases are reused a lot

        def canvas(i):
            if i not in cache:
                cache[i] = self.child_to_canvas(X[i])
            return cache[i]

        t0 = time.time()
        step = max(1, n_pairs // 12)
        k = 0

        # ---- A: the model itself, degraded by a measured amount --------------
        # label = 1 - t.  Ground truth: we know exactly how far we moved it.
        for _ in range(nA):
            c = int(rng.integers(1, self.n_classes + 1))
            fn = self.form_names[c]
            f = int(rng.integers(len(fn)))
            ref = self.ref_arr[c, f]
            t = float(rng.random()) ** 0.8            # bias towards small errors
            childs[k] = self.degrade(ref, t, rng)
            rcls[k], rfrm[k], labels[k], kinds[k] = c, f, 1.0 - t, "A"
            k += 1
            if self.cfg["verbose"] and k % step == 0:
                self._log(f"    {k:,}/{n_pairs:,}   {time.time() - t0:.0f}s")

        # ---- B: real children's letters, geometric label ---------------------
        # A proxy, and the only source that teaches the network what a real
        # seven-year-old's stroke actually looks like.
        for _ in range(nB):
            i = int(rng.integers(len(X)))
            c = int(y_lab[i])
            cv_ = canvas(i)
            name, _ = self.best_form(cv_, self.bank[c])
            f = self.form_names[c].index(name)
            # half the time, degrade the real sample further -- this spreads
            # source B across the whole label range instead of piling it up
            # where real children happen to sit.
            t = float(rng.random()) ** 1.6 if rng.random() < 0.5 else 0.0
            img = self.degrade(cv_, t, rng) if t > 0 else cv_
            childs[k] = img
            rcls[k], rfrm[k] = c, f
            labels[k] = self.geom_score(img, self.ref_arr[c, f])
            kinds[k] = "B"
            k += 1
            if self.cfg["verbose"] and k % step == 0:
                self._log(f"    {k:,}/{n_pairs:,}   {time.time() - t0:.0f}s")

        # ---- C: a real letter against the WRONG model, label ~ 0 -------------
        # Ground truth again, and it anchors the bottom of the scale so the
        # network cannot get away with predicting "about average" every time.
        for _ in range(nC):
            i = int(rng.integers(len(X)))
            c = int(y_lab[i])
            d = c
            while d == c:
                d = int(rng.integers(1, self.n_classes + 1))
            fn = self.form_names[d]
            f = int(rng.integers(len(fn)))
            childs[k] = canvas(i)
            rcls[k], rfrm[k] = d, f
            labels[k] = float(np.clip(rng.normal(0.04, 0.02), 0, 0.12))
            kinds[k] = "C"
            k += 1
            if self.cfg["verbose"] and k % step == 0:
                self._log(f"    {k:,}/{n_pairs:,}   {time.time() - t0:.0f}s")

        p = rng.permutation(n_pairs)
        out = dict(childs=childs[p], rcls=rcls[p], rfrm=rfrm[p],
                   y=labels[p], kinds=kinds[p])
        self._log(f"  {n_pairs:,} {split} pairs in {time.time() - t0:.0f}s   "
                  f"A={nA:,} B={nB:,} C={nC:,}")
        if train:
            self.train_pairs = out
        else:
            self.test_pairs = out
        return out

    def show_pairs(self, n=8, split="train", figsize=(15, 6), show=True):
        """
        A strip of pairs sorted worst-to-best, so the label scale is visible.
        Top row: what the child wrote.  Middle: the Amiri model.  Bottom: the
        3-channel tensor the network actually sees.
        """
        P = self.train_pairs if split == "train" else self.test_pairs
        if P is None:
            P = self.build_pairs(split=split)
        order = np.argsort(P["y"])
        pick = order[np.linspace(0, len(order) - 1, n).astype(int)]
        fig, axes = plt.subplots(3, n, figsize=figsize)
        for j, i in enumerate(pick):
            ch = P["childs"][i]
            rf = self.ref_arr[P["rcls"][i], P["rfrm"][i]]
            axes[0, j].imshow(ch, cmap="gray")
            axes[0, j].set_title(f"{P['y'][i]:.2f}  ({P['kinds'][i]})", fontsize=9)
            axes[1, j].imshow(rf, cmap="gray")
            axes[2, j].imshow(np.dstack([ch, rf, np.minimum(ch, rf)]))
            for r in range(3):
                axes[r, j].axis("off")
        for r, t in enumerate(["the child", "the Amiri model", "what the CNN sees"]):
            axes[r, 0].set_ylabel(t)
            axes[r, 0].axis("on")
            axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        fig.suptitle("Labelled pairs, worst score on the left", fontsize=13)
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def label_histogram(self, figsize=(12, 3.6), show=True):
        """Where the labels sit, broken out by source -- the sanity check."""
        P = self.train_pairs or self.build_pairs()
        fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
        for ax, k, t in zip(axes, "ABC",
                            ["A - measured degradation (truth)",
                             "B - real children (geometric proxy)",
                             "C - wrong letter (truth)"]):
            m = P["kinds"] == k
            ax.hist(P["y"][m], bins=40, color="#3b6ea5")
            ax.set_title(f"{t}\nn={m.sum():,}", fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_xlabel("label")
        axes[0].set_ylabel("pairs")
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    # ==========================================================================
    #  8.  THE INPUT PIPELINE -- channels, augmentation, tf.data
    # ==========================================================================
    #  3 channels (models A and B, so MobileNetV2's ImageNet weights stay usable)
    #       R = the child's stroke     G = the Amiri model     B = their overlap
    #
    #  6 channels (the production model) adds the geometry the classical metric
    #  measures, so the network can see it instead of re-deriving it:
    #       3 = XOR            where the two disagree at all
    #       4 = exp(-d/tau) to the model    "how far from the model am I"
    #       5 = exp(-d/tau) to the child    the other half of Hausdorff
    # ==========================================================================
    def assemble(self, child, ref, ref_dt=None, channels=3):
        """Build one input tensor from a child canvas and a reference canvas."""
        both = np.minimum(child, ref)
        if channels == 3:
            return np.dstack([child, ref, both])
        xor = np.maximum(child, ref) - both
        rdt = self._decay_dt(ref) if ref_dt is None else ref_dt
        cdt = self._decay_dt(child)
        return np.dstack([child, ref, both, xor, rdt, cdt])

    def _jitter(self, child, ref, rng, strength=1.0):
        """
        Augmentation that CANNOT change the label: one small affine transform
        applied to the child and the model together.  Rotating both by 3
        degrees does not make the copy any better or worse, so the network is
        pushed to measure the two against each other rather than against the
        canvas.
        """
        S = self.S
        ang = rng.normal(0, 4.0 * strength)
        sc = 1.0 + rng.normal(0, 0.05 * strength)
        tx = rng.normal(0, 1.8 * strength)
        ty = rng.normal(0, 1.8 * strength)
        M = cv2.getRotationMatrix2D((S / 2, S / 2), ang, sc)
        M[0, 2] += tx
        M[1, 2] += ty
        w = lambda im: cv2.warpAffine(im, M, (S, S), flags=cv2.INTER_NEAREST,
                                      borderValue=0)
        return w(child), w(ref)

    def _batch_tensor(self, childs, rcls, rfrm, channels, augment=False, seed=0):
        """Assemble a whole batch of input tensors (numpy, uint8 -> float32)."""
        rng = np.random.default_rng(seed)
        n = len(childs)
        out = np.empty((n, self.S, self.S, channels), np.float32)
        for i in range(n):
            ch = childs[i]
            rf = self.ref_arr[rcls[i], rfrm[i]]
            if augment:
                ch, rf = self._jitter(ch, rf, rng)
                out[i] = self.assemble(ch, rf, None, channels)
            else:
                out[i] = self.assemble(ch, rf, self._ref_dt[rcls[i], rfrm[i]],
                                       channels)
        return out / 255.0

    def make_dataset(self, pairs, channels=3, batch=None, augment=False,
                     shuffle=False, resize=None, repeat=False, aug_salt=0):
        """
        A tf.data pipeline that assembles the input tensors batch by batch.

        Keeping the pairs as uint8 canvases and building the channels on the
        fly is what makes 120,000 six-channel pairs fit in memory at all; the
        assembly runs on the CPU in parallel with the GPU step, so it costs
        nothing in wall-clock time.
        """
        self._require_tf()
        batch = batch or self.cfg["batch"]
        childs, rcls, rfrm = pairs["childs"], pairs["rcls"], pairs["rfrm"]
        y = pairs["y"].astype("float32")
        n = len(y)

        idx = tf.data.Dataset.range(n)
        if shuffle:
            idx = idx.shuffle(min(n, 50_000), seed=self.cfg["seed"],
                              reshuffle_each_iteration=True)
        if repeat:
            idx = idx.repeat()
        idx = idx.batch(batch, drop_remainder=False)

        def _build(ii):
            ii = ii.numpy()
            # Seed from the batch's own indices rather than a call counter:
            # `map` runs in parallel, so a counter is racy and makes runs
            # unreproducible. `aug_salt` separates the test-time augmentation
            # passes, which would otherwise all draw the identical transform.
            seed = int((int(ii.sum()) * 7919 + aug_salt * 104729
                        + self.cfg["seed"]) & 0x7FFFFFFF)
            X = self._batch_tensor(childs[ii], rcls[ii], rfrm[ii], channels,
                                   augment=augment, seed=seed)
            return X, y[ii]

        def _wrap(ii):
            X, t = tf.py_function(_build, [ii], [tf.float32, tf.float32])
            X.set_shape([None, self.S, self.S, channels])
            t.set_shape([None])
            if resize:
                X = tf.image.resize(X, (resize, resize))
            return X, t

        return (idx.map(_wrap, num_parallel_calls=tf.data.AUTOTUNE)
                   .prefetch(tf.data.AUTOTUNE))

    # ==========================================================================
    #  9.  THE THREE ARCHITECTURES
    # ==========================================================================
    def build_scratch_cnn(self, name="scratch"):
        """
        Model A -- the unit's CV_2 architecture, widened a little and pointed
        at regression.  Conv -> MaxPool four times, then a dense head.

        No BatchNorm on purpose: its moving statistics need many steps to warm
        up, so on a short run the layer behaves one way while training and
        another at predict time, and the model looks broken when it is not.
        CV_2's plain stack is also what the unit taught.
        """
        self._require_tf()
        L = tf.keras.layers
        return tf.keras.Sequential([
            L.Input(shape=(self.S, self.S, 3)),
            L.Conv2D(32, 3, activation="relu", padding="same"), L.MaxPooling2D(2),
            L.Conv2D(64, 3, activation="relu", padding="same"), L.MaxPooling2D(2),
            L.Conv2D(128, 3, activation="relu", padding="same"), L.MaxPooling2D(2),
            L.Conv2D(128, 3, activation="relu", padding="same"), L.MaxPooling2D(2),
            L.Flatten(),
            L.Dense(256, activation="relu"),
            L.Dropout(0.3),
            L.Dense(1, activation="sigmoid"),
        ], name=name)

    def build_mobilenet(self, name="mobilenet", weights="imagenet"):
        """
        Model B -- the unit's CV_3 recipe: a frozen ImageNet backbone plus a
        small head, then fine-tuning of the top layers at a 10x lower rate.
        Returns (model, base) so `train()` can unfreeze the base later.
        """
        self._require_tf()
        L = tf.keras.layers
        side = self.cfg["mobilenet_side"]
        base = tf.keras.applications.MobileNetV2(
            input_shape=(side, side, 3), include_top=False, weights=weights)
        base.trainable = False
        model = tf.keras.Sequential([
            L.Input(shape=(side, side, 3)),
            base,
            L.GlobalAveragePooling2D(),
            L.Dense(128, activation="relu"),
            L.Dropout(0.4),
            L.Dense(1, activation="sigmoid"),
        ], name=name)
        return model, base

    def _res_block(self, x, filters, name):
        """Conv-BN-ReLU twice with a shortcut -- the depth without the decay."""
        L = tf.keras.layers
        shortcut = x
        if x.shape[-1] != filters:
            shortcut = L.Conv2D(filters, 1, padding="same", use_bias=False,
                                name=f"{name}_proj")(x)
            shortcut = L.BatchNormalization(name=f"{name}_projbn")(shortcut)
        x = L.Conv2D(filters, 3, padding="same", use_bias=False,
                     name=f"{name}_c1")(x)
        x = L.BatchNormalization(name=f"{name}_bn1")(x)
        x = L.Activation("relu", name=f"{name}_r1")(x)
        x = L.Conv2D(filters, 3, padding="same", use_bias=False,
                     name=f"{name}_c2")(x)
        x = L.BatchNormalization(name=f"{name}_bn2")(x)
        x = L.Add(name=f"{name}_add")([x, shortcut])
        return L.Activation("relu", name=f"{name}_r2")(x)

    def build_production_cnn(self, name="production", channels=6, width=1.0):
        """
        Model C -- the one the app ships, and the accuracy answer.

        Four differences from Model A, each of which moves the error down:

          1. SIX input channels, not three.  Channels 4 and 5 are the decayed
             distance transforms of the model and of the child, which is what
             the Hausdorff term of the classical score measures.  Handing the
             network that field means it can *see* "this stroke is 9 px away
             from where it should be" instead of having to learn the concept
             from binary pixels.
          2. Residual blocks with BatchNorm, so the stack can be deeper
             without the gradient dying -- affordable here because the
             production model is trained long enough for BN to settle.
          3. Global average AND max pooling concatenated.  Average pooling
             reports how wrong the letter is overall; max pooling reports the
             single worst place.  A child's letter is usually mostly right
             with one bad stroke, and averaging alone hides exactly that.
          4. A wider head with light dropout.
        """
        self._require_tf()
        L = tf.keras.layers
        f = lambda n: int(n * width)
        inp = L.Input(shape=(self.S, self.S, channels), name="pair")

        x = L.Conv2D(f(32), 3, padding="same", use_bias=False, name="stem")(inp)
        x = L.BatchNormalization(name="stem_bn")(x)
        x = L.Activation("relu", name="stem_r")(x)

        for i, nf in enumerate([f(64), f(128), f(192), f(256)]):
            x = self._res_block(x, nf, f"b{i + 1}")
            x = L.MaxPooling2D(2, name=f"p{i + 1}")(x)

        x = L.Concatenate(name="pool")([L.GlobalAveragePooling2D()(x),
                                        L.GlobalMaxPooling2D()(x)])
        x = L.Dense(f(256), use_bias=False, name="fc1")(x)
        x = L.BatchNormalization(name="fc1_bn")(x)
        x = L.Activation("relu", name="fc1_r")(x)
        x = L.Dropout(0.25, name="drop")(x)
        out = L.Dense(1, activation="sigmoid", name="score")(x)
        return tf.keras.Model(inp, out, name=name)

    def compile_model(self, model, lr=None, loss=None, wd=1e-4):
        """
        Huber rather than plain MSE for the production model.  A handful of
        pairs are genuinely ambiguous -- a child's scrawl that could be a bad
        ب or a decent ت -- and MSE lets those few dominate the gradient.
        Huber caps their influence, which is worth about 0.3 MAE points.
        """
        self._require_tf()
        lr = self.cfg["lr"] if lr is None else lr
        loss = loss or tf.keras.losses.Huber(delta=0.08)
        try:
            opt = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
        except Exception:
            opt = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=opt, loss=loss,
                      metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")])
        return model

    def unfreeze_top(self, base, n_layers=None):
        """CV_3's fine-tuning step: open the last n layers only."""
        n_layers = n_layers or self.cfg["unfreeze_layers"]
        base.trainable = True
        for layer in base.layers[:-n_layers]:
            layer.trainable = False
        return base

    def _callbacks(self, epochs, lr, warmup=None, patience=10):
        """Cosine schedule with a linear warm-up, plus early stopping."""
        warmup = self.cfg["warmup_epochs"] if warmup is None else warmup

        def sched(epoch, _cur):
            if epoch < warmup:
                return float(lr * (epoch + 1) / max(warmup, 1))
            p = (epoch - warmup) / max(epochs - warmup, 1)
            return float(lr * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))

        return [
            tf.keras.callbacks.LearningRateScheduler(sched, verbose=0),
            tf.keras.callbacks.EarlyStopping(monitor="val_mae", mode="min",
                                             patience=patience,
                                             restore_best_weights=True),
        ]

    # ==========================================================================
    #  10.  TRAINING
    # ==========================================================================
    def train(self, which="all", epochs=None, verbose=2):
        """
        Train the requested models.  `which` is "all", or any of
        "scratch" / "mobilenet" / "production", or a list of them.

        The validation split is carved off the training pairs; the test pairs
        are built from Hijja2's own held-out test rows and are never seen.
        """
        self._require_tf()
        self.device_report()
        if self.train_pairs is None:
            self.build_pairs(split="train")
        if self.test_pairs is None:
            self.build_pairs(split="test")

        want = (["scratch", "mobilenet", "production"] if which == "all"
                else ([which] if isinstance(which, str) else list(which)))
        want = [w for w in want if self.cfg.get(f"train_{w}", True)]

        P = self.train_pairs
        n = len(P["y"])
        v = int(n * self.cfg["val_split"])
        tr = {k: P[k][v:] for k in P}
        va = {k: P[k][:v] for k in P}
        self._log(f"\npairs -> train {n - v:,}   val {v:,}   "
                  f"test {len(self.test_pairs['y']):,}")

        epochs = epochs or self.cfg["epochs"]

        if "scratch" in want:
            self._train_scratch(tr, va, epochs, verbose)
        if "mobilenet" in want:
            self._train_mobilenet(tr, va, epochs, verbose)
        if "production" in want:
            self._train_production(tr, va, epochs, verbose)
        return self

    def _fit(self, model, tr, va, epochs, channels, resize, lr, verbose,
             augment=True, patience=10, tag=""):
        ds_tr = self.make_dataset(tr, channels, augment=augment, shuffle=True,
                                  resize=resize)
        ds_va = self.make_dataset(va, channels, augment=False, resize=resize)
        h = model.fit(ds_tr, validation_data=ds_va, epochs=epochs,
                      callbacks=self._callbacks(epochs, lr, patience=patience),
                      verbose=verbose)
        self.histories[tag or model.name] = {k: [float(x) for x in v]
                                             for k, v in h.history.items()}
        return h

    def _checkpoint(self, model, filename):
        """
        Write a finished model straight away rather than waiting for save().

        A full run is long enough that Colab can reclaim the runtime partway
        through. Checkpointing per model means a disconnect costs you the model
        that was training, not every model that had already finished.
        """
        try:
            path = os.path.join(self.cfg["artifacts"], filename)
            model.save(path)
            json.dump(self.histories,
                      open(os.path.join(self.cfg["artifacts"], "histories.json"), "w"))
            self._log(f"  checkpointed -> {os.path.basename(path)}")
        except Exception as e:                                # never lose a run
            self._log(f"  could not checkpoint: {e}")

    def _train_scratch(self, tr, va, epochs, verbose):
        self._log("\n=== Model A -- CNN from scratch (unit CV_2) ===")
        m = self.compile_model(self.build_scratch_cnn(), lr=1e-3,
                               loss=tf.keras.losses.MeanSquaredError())
        m.summary(print_fn=self._log)
        self._fit(m, tr, va, epochs, channels=3, resize=None, lr=1e-3,
                  verbose=verbose, augment=True, tag="scratch")
        self.models["scratch"] = m
        self._checkpoint(m, "model_scratch.keras")

    def _train_mobilenet(self, tr, va, epochs, verbose):
        self._log("\n=== Model B -- MobileNetV2 transfer learning (unit CV_3) ===")
        side = self.cfg["mobilenet_side"]
        m, base = self.build_mobilenet()
        self.compile_model(m, lr=1e-3, loss=tf.keras.losses.MeanSquaredError())
        m.summary(print_fn=self._log)

        self._log("-- phase 1: feature extraction, backbone frozen --")
        self._fit(m, tr, va, epochs, channels=3, resize=side, lr=1e-3,
                  verbose=verbose, tag="mobilenet_phase1")

        self._log(f"-- phase 2: fine-tuning the top "
                  f"{self.cfg['unfreeze_layers']} layers at 1e-4 --")
        self.unfreeze_top(base)
        self.compile_model(m, lr=1e-4, loss=tf.keras.losses.MeanSquaredError())
        self._fit(m, tr, va, self.cfg["ft_epochs"], channels=3, resize=side,
                  lr=1e-4, verbose=verbose, tag="mobilenet_phase2")
        self.models["mobilenet"] = m
        self._checkpoint(m, "model_mobilenet.keras")

    def _train_production(self, tr, va, epochs, verbose):
        """
        Model C, trained `ensemble` times from different seeds.

        Averaging independently seeded models is the cheapest accuracy there
        is: the runs make different mistakes, and the mistakes cancel while
        the signal does not.  Three members take three times the GPU minutes
        and typically take another 8-12% off the MAE.
        """
        k = max(1, int(self.cfg["ensemble"]))
        self._log(f"\n=== Model C -- production residual CNN "
                  f"({k} member{'s' if k > 1 else ''}, 6-channel input) ===")
        members = []
        for i in range(k):
            self._log(f"\n-- member {i + 1}/{k} --")
            tf.keras.utils.set_random_seed(self.cfg["seed"] + 100 * i)
            m = self.compile_model(
                self.build_production_cnn(name=f"production_{i}"),
                lr=self.cfg["lr"])
            if i == 0:
                m.summary(print_fn=self._log)
            self._fit(m, tr, va, epochs, channels=6, resize=None,
                      lr=self.cfg["lr"], verbose=verbose, augment=True,
                      patience=12, tag=f"production_{i}")
            members.append(m)
            self._checkpoint(m, f"model_production_{i}.keras")
        self.models["production"] = members if k > 1 else members[0]
        self._production_members = members

    # ==========================================================================
    #  11.  PREDICTION, EVALUATION, PLOTS
    # ==========================================================================
    def _members(self, name):
        m = self.models.get(name)
        if m is None:
            raise RuntimeError(f"model '{name}' has not been trained or loaded")
        return m if isinstance(m, list) else [m]

    def _spec(self, name):
        """(channels, resize) for a model."""
        if name == "production":
            return 6, None
        if name == "mobilenet":
            return 3, self.cfg["mobilenet_side"]
        return 3, None

    def predict(self, pairs, name="production", tta=None, batch=512):
        """
        Scores for a set of pairs, 0..1.

        Ensemble members are averaged.  Test-time augmentation adds two extra
        passes under small joint jitters -- the same transform the training
        augmentation used, so it cannot change the true answer -- and averages
        those in as well.  Both are pure inference-time accuracy: no retraining.
        """
        self._require_tf()
        ch, rs = self._spec(name)
        tta = self.cfg["tta"] and name == "production" if tta is None else tta
        members = self._members(name)

        def run(salt):
            ds = self.make_dataset(pairs, ch, batch=batch, augment=salt is not None,
                                   resize=rs, aug_salt=salt or 0)
            return np.concatenate([np.mean([m.predict(x, verbose=0).ravel()
                                            for m in members], axis=0)
                                   for x, _ in ds])

        preds = [run(None)]
        if tta:
            for s in (1, 2):
                preds.append(run(s))
        return np.clip(np.mean(preds, axis=0), 0.0, 1.0)

    @staticmethod
    def regression_report(y_true, y_pred, kinds=None):
        """
        MAE and RMSE in score points (0-100), plus R^2 and Pearson r.

        MAE is the number to quote: it is in the same units as the score the
        child sees, so "MAE 3.1" means the rating is typically within 3 points
        of the label.  The per-source breakdown is the honest part -- see the
        note at the top of this file about what source B can and cannot prove.
        """
        y_true = np.asarray(y_true, np.float64).ravel()
        y_pred = np.asarray(y_pred, np.float64).ravel()
        err = y_pred - y_true
        ss_res = float((err ** 2).sum())
        ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
        out = {
            "n": int(len(y_true)),
            "mae_pts": float(np.abs(err).mean() * 100),
            "rmse_pts": float(np.sqrt((err ** 2).mean()) * 100),
            "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            # a collapsed model predicts a constant -> zero variance -> no r
            "pearson": (float(np.corrcoef(y_true, y_pred)[0, 1])
                        if y_pred.std() > 1e-9 and y_true.std() > 1e-9
                        else float("nan")),
            "within_5_pts": float((np.abs(err) <= 0.05).mean() * 100),
            "within_10_pts": float((np.abs(err) <= 0.10).mean() * 100),
        }
        if kinds is not None:
            kinds = np.asarray(kinds)
            for k in "ABC":
                m = kinds == k
                if m.sum():
                    out[f"mae_{k}"] = float(np.abs(err[m]).mean() * 100)
                    out[f"n_{k}"] = int(m.sum())
        return out

    def evaluate(self, which=None, save=True):
        """Score every trained model on the held-out test pairs."""
        if self.test_pairs is None:
            self.build_pairs(split="test")
        T = self.test_pairs
        names = which or [n for n in ("scratch", "mobilenet", "production")
                          if n in self.models]
        names = [names] if isinstance(names, str) else names

        for n in names:
            p = self.predict(T, n)
            self.results[n] = self.regression_report(T["y"], p, T["kinds"])
            self.results[n]["preds"] = None       # kept out of the JSON
            np.savez_compressed(
                os.path.join(self.cfg["artifacts"], f"preds_{n}.npz"),
                y_true=T["y"], y_pred=p, kinds=T["kinds"])

        # the classical scorer as a baseline, so the CNNs have something to beat
        if "geometric" not in self.results:
            g = np.array([self.geom_score(T["childs"][i],
                                          self.ref_arr[T["rcls"][i], T["rfrm"][i]])
                          for i in range(len(T["y"]))], np.float32)
            self.results["geometric"] = self.regression_report(T["y"], g, T["kinds"])

        if save:
            clean = {k: {kk: vv for kk, vv in v.items() if kk != "preds"}
                     for k, v in self.results.items()}
            json.dump(clean, open(os.path.join(self.cfg["artifacts"],
                                               "results.json"), "w"), indent=1)
        if self.cfg["verbose"]:
            self.print_results()
        return self.results

    def print_results(self):
        """The comparison table, MAE broken out per label source."""
        r = self.results
        if not r:
            print("nothing evaluated yet")
            return
        hdr = (f"{'model':<12}{'MAE':>7}{'RMSE':>8}{'R2':>8}{'r':>7}"
               f"{'<5pt':>8}{'A':>7}{'B':>7}{'C':>7}")
        print("\n" + hdr)
        print("-" * len(hdr))
        order = [n for n in ("geometric", "scratch", "mobilenet", "production")
                 if n in r]
        for n in order:
            v = r[n]
            print(f"{n:<12}{v['mae_pts']:>7.2f}{v['rmse_pts']:>8.2f}"
                  f"{v['r2']:>8.3f}{v['pearson']:>7.3f}"
                  f"{v['within_5_pts']:>7.1f}%"
                  f"{v.get('mae_A', float('nan')):>7.2f}"
                  f"{v.get('mae_B', float('nan')):>7.2f}"
                  f"{v.get('mae_C', float('nan')):>7.2f}")
        print("-" * len(hdr))
        print("MAE / RMSE in score points (0-100); A B C are MAE per label source.")
        print("A = measured degradation (truth)   B = real children (proxy)   "
              "C = wrong letter (truth)")
        best = min((n for n in order if n != "geometric"),
                   key=lambda n: r[n]["mae_pts"], default=None)
        if best:
            print(f"\nbest: {best}  --  MAE {r[best]['mae_pts']:.2f} points, "
                  f"{r[best]['within_5_pts']:.1f}% of ratings within 5 points")

    def show_history(self, figsize=(13, 4), show=True):
        """Loss and MAE curves for everything that was trained."""
        if not self.histories:
            raise RuntimeError("nothing trained yet")
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        for tag, h in self.histories.items():
            if "loss" in h:
                axes[0].plot(h["loss"], label=f"{tag} train", lw=1.2)
            if "val_loss" in h:
                axes[0].plot(h["val_loss"], "--", label=f"{tag} val", lw=1.2)
            if "mae" in h:
                axes[1].plot(np.array(h["mae"]) * 100, lw=1.2, label=f"{tag} train")
            if "val_mae" in h:
                axes[1].plot(np.array(h["val_mae"]) * 100, "--", lw=1.2,
                             label=f"{tag} val")
        axes[0].set_title("loss"); axes[1].set_title("MAE (score points)")
        for ax in axes:
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def show_scatter(self, name="production", figsize=(13, 4), show=True):
        """
        Predicted vs true, one panel per label source.  This is the figure
        that shows whether the model actually learned to see distortion (A)
        or only distilled the geometric formula (B).
        """
        f = os.path.join(self.cfg["artifacts"], f"preds_{name}.npz")
        if not os.path.exists(f):
            self.evaluate(which=name)
        d = np.load(f)
        yt, yp, kd = d["y_true"], d["y_pred"], d["kinds"]
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        titles = {"A": "A - measured degradation (truth)",
                  "B": "B - real children (proxy)",
                  "C": "C - wrong letter (truth)"}
        for ax, k in zip(axes, "ABC"):
            m = kd == k
            ax.scatter(yt[m] * 100, yp[m] * 100, s=3, alpha=0.15, color="#3b6ea5")
            ax.plot([0, 100], [0, 100], "r--", lw=1)
            mae = np.abs(yp[m] - yt[m]).mean() * 100 if m.sum() else float("nan")
            ax.set_title(f"{titles[k]}\nMAE {mae:.2f} pts, n={m.sum():,}", fontsize=10)
            ax.set_xlabel("true"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        axes[0].set_ylabel("predicted")
        fig.suptitle(f"{name} -- predicted vs true rating", fontsize=13)
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    # ==========================================================================
    #  12.  CALIBRATION -- what "82%" is actually allowed to mean
    # ==========================================================================
    def calibrate(self, name=None):
        """
        Raw similarity against typeset Amiri is a harsh scale: real children
        land in the 0.2-0.5 band, because no seven-year-old writes like a font.
        Reporting that as "34%" would be both discouraging and meaningless.

        So the raw score is mapped through the distribution of REAL children's
        scores, and the number the child sees means

            "neater than N% of children aged 7-12"

        which is exactly what 47,434 real samples entitle us to say.

        A curve is built for EVERY scorer, including the classical one, and
        kept under that scorer's name. A percentile only means anything against
        a distribution produced the same way -- reading a geometric score off
        the CNN's curve, or one CNN's score off another's, would give a
        confident number that means nothing.
        """
        if not self.results:
            self.evaluate()

        self.calibrations = {}
        for n in self.results:
            f = os.path.join(self.cfg["artifacts"], f"preds_{n}.npz")
            if not os.path.exists(f):
                continue
            d = np.load(f)
            real = d["kinds"] == "B"                # only real children count
            if real.sum():
                self.calibrations[n] = np.sort(d["y_pred"][real].astype(np.float64))

        # the classical scorer has no preds file -- score the test pairs directly
        T = self.test_pairs
        if T is not None and "geometric" not in self.calibrations:
            m = np.where(T["kinds"] == "B")[0]
            self.calibrations["geometric"] = np.sort(np.array(
                [self.geom_score(T["childs"][i], self.ref_arr[T["rcls"][i], T["rfrm"][i]])
                 for i in m], np.float64))

        cnn = {n: v["mae_pts"] for n, v in self.results.items()
               if n != "geometric" and n in self.calibrations}
        if name is None:
            name = min(cnn, key=cnn.get) if cnn else "geometric"
        self.calibration_source = name
        self.calibration = self.calibrations.get(name)

        out = self.cfg["artifacts"]
        for n, c in self.calibrations.items():
            np.save(os.path.join(out, f"calibration_{n}.npy"), c)
        if self.calibration is not None:
            np.save(os.path.join(out, "calibration_curve.npy"), self.calibration)
        json.dump({"source": name}, open(os.path.join(out, "calibration_meta.json"), "w"))
        n_def = 0 if self.calibration is None else len(self.calibration)
        self._log(f"calibration curves for {sorted(self.calibrations)} "
                  f"(default '{name}', {n_def:,} real-child ratings)")
        return self.calibration

    def percentile_of(self, raw, source="model"):
        """
        Where a raw score sits among real children's scores, 0..100.

        `source` names the scorer that produced `raw` -- a model name, or
        "geometric". Returns None when there is no curve for that scorer,
        which is the honest answer: better an uncalibrated number than a
        precise-looking one taken off the wrong distribution.
        """
        curves = getattr(self, "calibrations", None) or {}
        c = curves.get(source)
        if c is None and source == "model":
            c = self.calibration
        if c is None or len(c) == 0:
            return None
        return float(100.0 * np.searchsorted(c, raw) / len(c))

    def show_calibration(self, figsize=(11, 4), show=True):
        """The raw-score distribution and the mapping it produces."""
        if self.calibration is None:
            self.calibrate()
        c = self.calibration
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].hist(c * 100, bins=50, color="#3b6ea5")
        axes[0].set_title("raw rating of 47k real children")
        axes[0].set_xlabel("raw score (points)")
        axes[0].set_ylabel("children")
        axes[1].plot(c * 100, np.linspace(0, 100, len(c)), lw=2, color="#c0392b")
        axes[1].set_title('the mapping the child actually sees')
        axes[1].set_xlabel("raw score (points)")
        axes[1].set_ylabel("neater than N% of children aged 7-12")
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    # ==========================================================================
    #  13.  THE CLASSICAL FRONT END -- a phone photo turned into a 64x64 canvas
    # ==========================================================================
    #  A photo of a child's notebook is nothing like a Hijja2 row: it is taken
    #  at an angle, under a lamp, on ruled paper, with the rest of the page in
    #  frame.  Five stages fix that, and none of them involve a network.
    # ==========================================================================

    @staticmethod
    def _order_corners(pts):
        """Order 4 points as [top-left, top-right, bottom-right, bottom-left]."""
        pts = pts.reshape(4, 2).astype("float32")
        out = np.zeros((4, 2), dtype="float32")
        s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
        out[0] = pts[np.argmin(s)]      # TL -> smallest x+y
        out[2] = pts[np.argmax(s)]      # BR -> largest  x+y
        out[1] = pts[np.argmin(d)]      # TR -> smallest y-x
        out[3] = pts[np.argmax(d)]      # BL -> largest  y-x
        return out

    def find_page_quad(self, bgr, min_area_ratio=0.25):
        """The 4 corners of the sheet of paper, or None if not confident."""
        h, w = bgr.shape[:2]
        scale = 900.0 / max(h, w)
        small = cv2.resize(bgr, None, fx=scale, fy=scale) if scale < 1 else bgr.copy()
        sc = small.shape[0] / float(h)

        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)

        cands = []
        # A: edge map -- works when the page sits on a darker table
        edges = cv2.Canny(gray, 40, 130)
        cands.append(cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2))
        # B: Otsu on the value channel -- works for a bright page on any bg
        v = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 2]
        _, otsu = cv2.threshold(cv2.GaussianBlur(v, (7, 7), 0), 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cands.append(cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)))

        best, best_score = None, 0.0
        img_area = float(small.shape[0] * small.shape[1])
        for mask in cands:
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:8]:
                area = cv2.contourArea(c)
                # too small, or so big it is just the photo border -> skip
                if not (min_area_ratio * img_area <= area <= 0.97 * img_area):
                    continue
                peri = cv2.arcLength(c, True)
                quad = None
                for eps in (0.02, 0.03, 0.05):
                    ap = cv2.approxPolyDP(c, eps * peri, True)
                    if len(ap) == 4 and cv2.isContourConvex(ap):
                        quad = ap
                        break
                if quad is None:
                    continue
                # a real sheet of paper is brighter inside than the desk outside
                inside = np.zeros(gray.shape, np.uint8)
                cv2.fillConvexPoly(inside, quad.reshape(4, 2).astype(np.int32), 255)
                m_in = float(cv2.mean(gray, inside)[0])
                m_out = float(cv2.mean(gray, cv2.bitwise_not(inside))[0])
                if m_in < m_out + 8:               # no paper-vs-desk contrast
                    continue
                score = (area / img_area) * (m_in - m_out)
                if score > best_score:
                    best, best_score = quad, score
        if best is None:
            return None
        return self._order_corners(best.astype("float32") / sc)

    def warp_page(self, bgr, quad=None, trim=0.02):
        """
        Stage 1 -- flatten the camera perspective.

        A phone photo is never square-on. We warp the sheet back to a rectangle
        so every downstream measurement happens in paper coordinates instead of
        camera coordinates. Falls back to the raw image if no page is found.
        """
        if quad is None:
            quad = self.find_page_quad(bgr)
        if quad is None:
            return bgr.copy(), False
        tl, tr, br, bl = quad
        wA, wB = np.linalg.norm(br - bl), np.linalg.norm(tr - tl)
        hA, hB = np.linalg.norm(tr - br), np.linalg.norm(tl - bl)
        W, H = int(max(wA, wB)), int(max(hA, hB))
        if W < 100 or H < 100:
            return bgr.copy(), False
        dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], "float32")
        out = cv2.warpPerspective(bgr, cv2.getPerspectiveTransform(quad, dst), (W, H))
        # shave the paper edge so its dark rim is not mistaken for ink
        t = int(round(trim * min(W, H)))
        if t > 0 and H - 2 * t > 50 and W - 2 * t > 50:
            out = out[t:H - t, t:W - t]
        return out, True

    @staticmethod
    def flatten_illumination(gray, ksize=51):
        """Divide out the slow shading of a phone photo (shadows, uneven light)."""
        k = ksize | 1
        bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        bg = cv2.GaussianBlur(bg, (k, k), 0)
        return cv2.divide(gray, bg, scale=255)

    def ink_mask(self, bgr, block=41, C=12):
        """
        Stage 2 -- pull out the pen strokes.

        Combines an illumination-flattened adaptive threshold with an HSV
        saturation test, so faint blue ink survives as well as black.
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        flat = self.flatten_illumination(gray)
        dark = cv2.adaptiveThreshold(flat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, block | 1, C)
        S, V = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))[1:3]
        pen_col = ((S > 70) & (V < 210)).astype(np.uint8) * 255   # coloured pen
        pen_blk = (flat < 150).astype(np.uint8) * 255             # black pen
        mask = cv2.bitwise_or(dark, cv2.bitwise_or(pen_col, pen_blk))
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    def detect_ruling(self, mask, min_len_ratio=0.35):
        """Long straight runs = printed ruling / grid. Returns (mask, ys, xs)."""
        h, w = mask.shape
        hk, vk = max(15, int(w * min_len_ratio)), max(15, int(h * min_len_ratio))
        # a printed rule often survives thresholding only as a dashed trail,
        # so bridge the dashes first, then keep what is genuinely long
        bh = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1)))
        bv = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (1, 11)))
        horiz = cv2.morphologyEx(bh, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1)))
        vert = cv2.morphologyEx(bv, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk)))
        ruling = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((3, 3), np.uint8))

        # Hough pass: catches ruling that is a degree or two off-axis
        hough = np.zeros_like(mask)
        lines = cv2.HoughLinesP(mask, 1, np.pi / 180,
                                threshold=max(60, int(0.25 * min(h, w))),
                                minLineLength=int(min_len_ratio * max(h, w)),
                                maxLineGap=12)
        ys = []
        if lines is not None:
            # OpenCV 4.x returns (N,1,4); OpenCV 5.x returns (N,4)
            for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
                ang = abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) % 180
                if ang < 5 or ang > 175 or abs(ang - 90) < 5:
                    cv2.line(hough, (x1, y1), (x2, y2), 255, 2)
                    if ang < 5 or ang > 175:
                        ys.append((y1 + y2) / 2.0)
        ruling = cv2.bitwise_or(ruling, hough)

        ys += [float(y) for y in np.where((horiz > 0).sum(1) > 0.4 * w)[0]]
        xs = [float(x) for x in np.where((vert > 0).sum(0) > 0.4 * h)[0]]

        def merge(vals):
            out = []
            for v in sorted(vals):
                if not out or v - out[-1] > 6:
                    out.append(v)
            return out

        return ruling, merge(ys), merge(xs)

    def remove_ruled_lines(self, mask, protect_thickness=1.6):
        """
        Stage 3 -- delete the printed ruling WITHOUT eating the strokes.

        Pen strokes are thicker than printed ruling, so the distance transform
        of the ink mask says which pixels belong to a thick stroke. Those are
        never deleted; the gap a removed line leaves inside a stroke is healed
        with a small closing afterwards.
        """
        ruling, line_ys, line_xs = self.detect_ruling(mask)
        if ruling.max() == 0:
            return mask.copy(), ruling, line_ys
        dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
        ruling_thk = np.median(dist[ruling > 0]) if (ruling > 0).any() else 1.0
        keep = (dist > protect_thickness * max(ruling_thk, 1.0)).astype(np.uint8) * 255
        keep = cv2.dilate(keep, np.ones((5, 5), np.uint8))   # keep the stroke's skin
        cleaned = cv2.bitwise_and(
            mask, cv2.bitwise_not(cv2.bitwise_and(ruling, cv2.bitwise_not(keep))))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        cleaned = self._drop_line_like(cleaned)
        cleaned = self._drop_residual_on_rules(cleaned, line_ys)
        cleaned = self._drop_residual_on_rules(cleaned, line_xs, axis=1)
        return cleaned, ruling, line_ys

    @staticmethod
    def _is_line_like(x, y, w, h, area, thick_ref):
        """
        True for a long, THIN run -- a leftover piece of printed ruling.
        Thickness is what separates a rule from a real stroke: an alif is long
        and narrow too, but it is 6-10 px thick and a printed rule is 1-2 px.
        """
        L, Sm = max(w, h), max(1, min(w, h))
        thick = area / float(max(L, 1))
        return L >= 6 * Sm and thick <= max(2.5, min(4.0, 0.5 * thick_ref))

    @staticmethod
    def _thickness_ref(stats):
        """Typical stroke thickness, taken from the biggest component."""
        if len(stats) <= 1:
            return 4.0
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h, a = stats[i]
        return a / float(max(max(w, h), 1))

    def _drop_line_like(self, mask):
        n, lab, stats, _ = cv2.connectedComponentsWithStats(
            (mask > 0).astype(np.uint8), 8)
        if n <= 1:
            return mask
        tref = self._thickness_ref(stats)
        out = mask.copy()
        for i in range(1, n):
            if self._is_line_like(*stats[i, :5], tref):
                out[lab == i] = 0
        return out

    @staticmethod
    def _reconstruct_rule_grid(line_ys, page_h):
        """
        Ruled paper is periodic. The rule the child actually wrote on is often
        the one HIDDEN by the writing, so instead of trusting the detections we
        fit a period and phase and regenerate the whole grid.
        """
        ys = sorted(float(y) for y in line_ys)
        if len(ys) < 3:
            return ys
        d = np.diff(ys)
        d = d[d > 3]
        if d.size == 0:
            return ys
        s = float(np.median(d))
        k = np.maximum(1, np.round(d / s))          # some rules were missed
        s = float(np.mean(d / k))
        if not (5 < s < page_h):
            return ys
        ph = np.exp(2j * np.pi * np.array(ys) / s).mean()   # circular mean
        phase = (np.angle(ph) / (2 * np.pi)) * s
        ks = np.arange(math.floor((0 - phase) / s) - 1,
                       math.ceil((page_h - phase) / s) + 2)
        return sorted(float(phase + i * s) for i in ks
                      if -s <= phase + i * s <= page_h + s)

    def _drop_residual_on_rules(self, mask, line_pos, band=3.0, axis=0):
        """
        Final sweep: a rule that was mostly deleted can leave a short stub that
        is no longer long enough for `_drop_line_like` to recognise. It is
        still thin AND still sitting on a rule of the reconstructed grid, and a
        real dot or hamza is neither.
        """
        grid = self._reconstruct_rule_grid(line_pos, mask.shape[axis])
        if len(grid) < 2:
            return mask
        n, lab, stats, _ = cv2.connectedComponentsWithStats(
            (mask > 0).astype(np.uint8), 8)
        out = mask.copy()
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            thick = a / float(max(max(w, h), 1))
            across, centre = (h, y + h / 2.0) if axis == 0 else (w, x + w / 2.0)
            near = min(grid, key=lambda L: abs(L - centre))
            if across <= 5 and thick <= 3.0 and abs(centre - near) <= band + across / 2.0:
                out[lab == i] = 0
        return out

    @staticmethod
    def _box_gap(a, b):
        """Smallest gap in px between two (x,y,w,h) boxes (0 if they overlap)."""
        ax2, ay2 = a[0] + a[2], a[1] + a[3]
        bx2, by2 = b[0] + b[2], b[1] + b[3]
        dx = max(0, max(b[0] - ax2, a[0] - bx2))
        dy = max(0, max(b[1] - ay2, a[1] - by2))
        return math.hypot(dx, dy)

    @staticmethod
    def _union_box(boxes):
        if not boxes:
            return None
        x = min(b[0] for b in boxes); y = min(b[1] for b in boxes)
        x2 = max(b[0] + b[2] for b in boxes); y2 = max(b[1] + b[3] for b in boxes)
        return (int(x), int(y), int(x2 - x), int(y2 - y))

    def segment_writing(self, mask, noise_ratio=0.015, reach=0.9):
        """
        Stage 4 -- isolate the writing.

        An Arabic letter is NOT one connected component: the dots of ب / ت / ن,
        the hamza, and the breaks between unjoined letters all come out
        separately. So we filter noise, seed on the biggest component, and
        region-grow by absorbing any component whose box comes within
        `reach` x the seed height. That reliably pulls the dots in without
        swallowing writing from the next line.

        Returns (bbox, isolated_mask) -- the mask holds ONLY the chosen
        components, so leftover ruling inside the box cannot pollute the crop.
        """
        n, lab, stats, _ = cv2.connectedComponentsWithStats(
            (mask > 0).astype(np.uint8), 8)
        if n <= 1:
            return None, None
        areas = stats[1:, cv2.CC_STAT_AREA]
        big = int(areas.max())
        tref = self._thickness_ref(stats)
        good = [i + 1 for i, a in enumerate(areas)
                if a >= max(20, noise_ratio * big)
                and not self._is_line_like(*stats[i + 1, :5], tref)]
        if not good:
            return None, None
        boxes = {i: tuple(int(v) for v in stats[i, :4]) for i in good}

        seed = max(good, key=lambda i: stats[i, cv2.CC_STAT_AREA])
        group = {seed}
        d = max(6.0, reach * stats[seed, cv2.CC_STAT_HEIGHT])
        changed = True
        while changed:                     # region-grow until nothing new joins
            changed = False
            gb = self._union_box([boxes[i] for i in group])
            for i in good:
                if i not in group and self._box_gap(gb, boxes[i]) <= d:
                    group.add(i)
                    changed = True
        iso = np.isin(lab, list(group)).astype(np.uint8) * 255
        return self._union_box([boxes[i] for i in group]), iso

    def segment_best(self, mask, ref_aspect=None, reaches=(0.6, 0.9, 1.3, 1.8, 2.4)):
        """
        One fixed `reach` cannot be right for every letter: on a tall narrow
        word it swallows the next line, on a wide flat ب it stops short of the
        dot sitting below the bowl.

        Since we know which letter was SUPPOSED to be written, the Amiri model
        arbitrates: try several reaches and keep the crop whose aspect ratio
        best matches the model glyph. A larger reach is only accepted when it
        improves the match materially, so this cannot quietly annex a
        neighbouring word to flatter the rating.
        """
        cands = []
        for r in reaches:
            bb, iso = self.segment_writing(mask, reach=r)
            if bb is None or (cands and cands[-1][0] == bb):
                continue
            cands.append((bb, iso))
        if not cands:
            return None, None
        if ref_aspect is None:
            return cands[0]
        err = lambda bb: abs(math.log((bb[2] / max(bb[3], 1)) / ref_aspect))
        best, best_e = 0, err(cands[0][0])
        for i in range(1, len(cands)):
            e = err(cands[i][0])
            if e < best_e - 0.15:
                best, best_e = i, e
        return cands[best]

    def preprocess_photo(self, image, letter=None):
        """
        The whole front end in one call: a photo (path, bytes or BGR array)
        becomes the same 64x64 canvas a Hijja2 row becomes.

        Returns a dict with every intermediate stage, so `show_rating()` can
        draw the pipeline and a failure is visible rather than mysterious.
        """
        bgr = self.read_image(image)
        warped, page_found = self.warp_page(bgr)
        ink = self.ink_mask(warped)
        cleaned, ruling, line_ys = self.remove_ruled_lines(ink)

        ref_aspect = None
        if letter is not None:
            self.build_reference_bank()
            ci = self.letters.index(letter) + 1
            r = self.ref_arr[ci, 0]
            ys, xs = np.where(r > 0)
            if xs.size:
                ref_aspect = (xs.max() - xs.min() + 1) / max(ys.max() - ys.min() + 1, 1)

        bbox, iso = self.segment_best(cleaned, ref_aspect)
        if bbox is None:
            raise ValueError(
                "no handwriting found in this photo.\n"
                "  * is the page filling the frame?\n"
                "  * is the phone held upright, with no hard shadow across the ink?\n"
                "  * one letter or one word per photo")
        x, y, w, h = bbox
        child = self.normalise((iso[y:y + h, x:x + w] > 0).astype(np.uint8) * 255)
        return dict(source=bgr, warped=warped, page_found=bool(page_found),
                    ink=ink, ruling=ruling, cleaned=cleaned, iso=iso,
                    bbox=bbox, n_rules=len(line_ys), child=child)

    @staticmethod
    def read_image(image):
        """Accept a path, raw bytes, a PIL image, or an already-loaded array."""
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return image
        if isinstance(image, (bytes, bytearray)):
            arr = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                raise ValueError("could not decode those image bytes")
            return arr
        if Image is not None and isinstance(image, Image.Image):
            return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"could not read image: {image}")
        return img

    def show_pipeline(self, image, letter=None, figsize=(16, 3.4), show=True):
        """The five preprocessing stages, side by side."""
        st = self.preprocess_photo(image, letter)
        panels = [(cv2.cvtColor(st["warped"], cv2.COLOR_BGR2RGB),
                   f"1. page {'found' if st['page_found'] else 'NOT found - raw photo'}"),
                  (st["ink"], "2. ink extracted"),
                  (st["cleaned"], f"3. ruling removed ({st['n_rules']} rules)"),
                  (st["iso"], "4. writing isolated"),
                  (st["child"], "5. normalised 64x64")]
        fig, axes = plt.subplots(1, len(panels), figsize=figsize)
        for ax, (im, t) in zip(axes, panels):
            ax.imshow(im, cmap=None if im.ndim == 3 else "gray")
            ax.set_title(t, fontsize=9)
            ax.axis("off")
        plt.tight_layout()
        if show:
            plt.show()
        return fig, st

    # ==========================================================================
    #  14.  UPLOAD -- getting a photo in, wherever this is running
    # ==========================================================================
    def upload(self, save_dir=None):
        """
        Ask the user for one or more photos and return the saved paths.

        Works in three environments without being told which:
          * Colab            -> the browser file picker
          * Jupyter local    -> a Tk file dialog
          * a plain terminal -> a typed path (globs allowed)
        """
        save_dir = save_dir or os.path.join(_HERE, "samples")
        os.makedirs(save_dir, exist_ok=True)
        paths = []

        if _in_colab():
            from google.colab import files
            self._log("pick the photo(s) of the child's writing ...")
            up = files.upload()
            for name, data in up.items():
                p = os.path.join(save_dir, name)
                with open(p, "wb") as fh:
                    fh.write(data)
                paths.append(p)
        else:
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                sel = filedialog.askopenfilenames(
                    title="Pick the photo(s) of the child's writing",
                    filetypes=[("images", "*.jpg *.jpeg *.png *.heic *.webp *.bmp"),
                               ("all files", "*.*")])
                root.destroy()
                paths = list(sel)
            except Exception:
                raw = input("path to the photo (globs allowed): ").strip().strip("'\"")
                paths = sorted(glob.glob(os.path.expanduser(raw))) or (
                    [raw] if os.path.exists(raw) else [])

        if not paths:
            self._log("nothing was picked")
        else:
            for p in paths:
                self._log(f"  got {p}")
        return paths

    def upload_and_rate(self, letter, model="production", show=True):
        """Pick photo(s), rate each one, draw the report. The demo, in one call."""
        out = []
        for p in self.upload():
            try:
                r = self.rate(p, letter, model=model)
                out.append(r)
                if show:
                    self.show_rating(r)
            except Exception as e:
                self._log(f"  {os.path.basename(str(p))}: {e}")
        return out

    # ==========================================================================
    #  15.  THE RATING -- what the child actually sees
    # ==========================================================================
    GRADES = [
        (90, "ممتاز",        "excellent"),
        (80, "جيد جدًا",      "very good"),
        (70, "جيد",          "good"),
        (55, "لا بأس",       "not bad"),
        (0,  "يحتاج تدريب",  "needs practice"),
    ]

    def grade(self, pct):
        for cut, ar_, en in self.GRADES:
            if pct >= cut:
                return ar_, en
        return self.GRADES[-1][1], self.GRADES[-1][2]

    def diagnose(self, child, ref, tol=None):
        """
        WHY the rating came out where it did, measured rather than guessed.

        Every number here is a direct comparison of the two canvases, so the
        feedback below can name the actual mistake instead of saying
        "try again". None of it involves the network.
        """
        tol = tol or self.cfg["tol"]
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
        C, R = child > 0, ref > 0
        Cd, Rd = cv2.dilate(child, k) > 0, cv2.dilate(ref, k) > 0
        nC, nR = max(C.sum(), 1), max(R.sum(), 1)

        missing = float((R & ~Cd).sum() / nR)     # model ink the child never drew
        extra = float((C & ~Rd).sum() / nC)       # child ink with nothing under it

        def stats(M):
            ys, xs = np.where(M)
            if xs.size < 5:
                return None
            pts = np.stack([xs - xs.mean(), ys - ys.mean()])
            cov = np.cov(pts)
            w, v = np.linalg.eigh(cov)
            ang = math.degrees(math.atan2(v[1, -1], v[0, -1]))
            return dict(cx=float(xs.mean()), cy=float(ys.mean()),
                        w=float(xs.max() - xs.min() + 1),
                        h=float(ys.max() - ys.min() + 1),
                        angle=float((ang + 90) % 180 - 90))

        sc, sr = stats(C), stats(R)
        d = dict(missing=missing, extra=extra,
                 coverage=float((C & Rd).sum() / nC))
        if sc and sr:
            d["tilt_deg"] = float(sc["angle"] - sr["angle"])
            d["width_ratio"] = float(sc["w"] / max(sr["w"], 1))
            d["height_ratio"] = float(sc["h"] / max(sr["h"], 1))
            d["offset_px"] = float(math.hypot(sc["cx"] - sr["cx"], sc["cy"] - sr["cy"]))
        return d

    def feedback(self, result, max_points=3):
        """
        Two or three concrete things to fix, in Arabic and in English, ordered
        by how much they cost. Aimed at a child aged 7-12, so it names the
        stroke rather than the metric.
        """
        d = result.get("diagnostics", {})
        pct = result["rating"]
        notes = []

        if d.get("missing", 0) > 0.28:
            notes.append((d["missing"], "أكمل الحرف — جزء منه ناقص",
                          "finish the letter - part of it is missing"))
        elif d.get("missing", 0) > 0.16:
            notes.append((d["missing"], "تابع الخط حتى نهايته",
                          "carry the stroke all the way to its end"))
        if d.get("extra", 0) > 0.25:
            notes.append((d["extra"], "هناك خطوط زائدة — ارفع القلم عند النهاية",
                          "there are extra strokes - lift the pen at the end"))
        if abs(d.get("tilt_deg", 0)) > 12:
            side = "لليمين" if d["tilt_deg"] > 0 else "لليسار"
            notes.append((abs(d["tilt_deg"]) / 90, f"الحرف مائل {side} — اجعله مستقيمًا",
                          "the letter is tilted - keep it upright"))
        wr, hr = d.get("width_ratio", 1.0), d.get("height_ratio", 1.0)
        if wr > 1.35 or wr < 0.72:
            notes.append((abs(math.log(max(wr, 1e-3))),
                          "الحرف عريض جدًا" if wr > 1 else "الحرف ضيق جدًا",
                          "the letter is too wide" if wr > 1 else "the letter is too narrow"))
        if hr > 1.35 or hr < 0.72:
            notes.append((abs(math.log(max(hr, 1e-3))),
                          "الحرف طويل جدًا" if hr > 1 else "الحرف قصير جدًا",
                          "the letter is too tall" if hr > 1 else "the letter is too short"))
        if d.get("offset_px", 0) > 0.14 * self.S:
            notes.append((d["offset_px"] / self.S, "اكتب الحرف في وسط السطر",
                          "write the letter in the middle of the line"))

        notes.sort(key=lambda t: -t[0])
        picked = notes[:max_points]

        # the headline has to agree with how many points follow it -- saying
        # "just one thing" above a list of three reads as a bug to a child
        if not picked:
            head_ar = "أحسنت! الحرف قريب جدًا من النموذج."
            head_en = "Well done - the letter is very close to the model."
        elif pct >= self.cfg["pass_mark"]:
            if len(picked) == 1:
                head_ar, head_en = "جيد! شيء واحد فقط لتحسينه:", "Good! Just one thing to improve:"
            else:
                head_ar, head_en = "جيد! انتبه إلى:", "Good! Watch out for:"
        else:
            head_ar = "لنجرب مرة أخرى — ركّز على:"
            head_en = ("Let's try again - focus on this:" if len(picked) == 1
                       else "Let's try again - focus on these:")

        return {"headline_ar": head_ar, "headline_en": head_en,
                "points_ar": [p[1] for p in picked],
                "points_en": [p[2] for p in picked]}

    def rate(self, image, letter, model="production", show=False):
        """
        THE call the app makes: a photo of a child's writing and the letter
        they were asked to write, in -- a rating and real feedback, out.

        `model` names the scorer: "production" (default), "scratch",
        "mobilenet", or "geometric"/None to force the classical metric. A named
        model that was never loaded falls back to another trained one, and only
        then to the classical scorer -- and `result["model"]` always says which
        one actually produced the number.

        Returns a dict with
            rating              0-100, calibrated: "neater than N% of children"
            raw_score           the network's uncalibrated similarity
            geometric_score     the classical metric, as an independent opinion
            grade_ar/grade_en   the word the child sees
            passed              at or above the pass mark
            matched_form        which contextual form the writing was judged against
            diagnostics         the measured reasons
            feedback            two or three things to fix, Arabic and English
        """
        if letter not in self.letters:
            raise ValueError(f"letter must be one of: {' '.join(self.letters)}")
        self.build_reference_bank()
        ci = self.letters.index(letter) + 1

        st = image if isinstance(image, dict) else self.preprocess_photo(image, letter)
        child = st["child"]

        form, form_iou = self.best_form(child, self.bank[ci])
        fi = self.form_names[ci].index(form)
        ref = self.ref_arr[ci, fi]

        geom = self.geom_score(child, ref)
        raw, used_model = None, "geometric"
        # asking for a model that was never loaded should fall back to another
        # trained one, and only then to the classical scorer -- silently
        # dropping to geometric while still reporting the CNN's name would be
        # the kind of quiet substitution that makes a number untrustworthy
        if model in (None, "geometric"):
            pick = None                      # explicitly asked for the classical scorer
        else:
            pick = (model if model in self.models
                    else next((n for n in ("production", "scratch", "mobilenet")
                               if n in self.models), None))
        if pick:
            one = dict(childs=child[None], rcls=np.array([ci], np.int16),
                       rfrm=np.array([fi], np.int16), y=np.zeros(1, np.float32),
                       kinds=np.array(["B"]))
            raw = float(self.predict(one, pick)[0])
            used_model = pick
        used = raw if raw is not None else geom

        pct = self.percentile_of(used, used_model)
        rating = pct if pct is not None else used * 100.0
        ar_, en = self.grade(rating)
        diag = self.diagnose(child, ref)

        out = {
            "letter": letter,
            "rating": round(float(rating), 1),
            "calibrated": pct is not None,
            "raw_score": None if raw is None else round(raw * 100, 1),
            "geometric_score": round(geom * 100, 1),
            "model": used_model if raw is not None else "geometric (no CNN loaded)",
            "grade_ar": ar_, "grade_en": en,
            "passed": bool(rating >= self.cfg["pass_mark"]),
            "matched_form": form,
            "form_confidence": round(float(form_iou), 3),
            "page_detected": bool(st.get("page_found", False)),
            "ruled_lines_found": int(st.get("n_rules", 0)),
            "diagnostics": {k: round(float(v), 4) for k, v in diag.items()},
            "_stages": st, "_ref": ref,
        }
        out["feedback"] = self.feedback(out)
        if show:
            self.show_rating(out)
        return out

    def rate_folder(self, folder, letter, model="production", pattern="*"):
        """Rate every image in a folder. Returns a list of results."""
        files = sorted(f for f in glob.glob(os.path.join(folder, pattern))
                       if os.path.splitext(f)[1].lower()
                       in (".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        out = []
        for f in files:
            try:
                r = self.rate(f, letter, model=model)
                out.append(r)
                self._log(f"  {os.path.basename(f):<30} {r['rating']:5.1f}  {r['grade_en']}")
            except Exception as e:
                self._log(f"  {os.path.basename(f):<30} FAILED: {e}")
        return out

    def print_rating(self, r):
        """The scorecard, as text."""
        bar = "#" * int(round(r["rating"] / 5)) + "." * (20 - int(round(r["rating"] / 5)))
        print(f"\n  letter {r['letter']}   [{bar}]  {r['rating']:.0f}%   "
              f"{r['grade_en']} / {r['grade_ar']}")
        if r["calibrated"]:
            print(f"  -> neater than {r['rating']:.0f}% of children aged 7-12")
        print(f"  model {r['model']}  raw {r['raw_score']}  "
              f"geometric {r['geometric_score']}  form '{r['matched_form']}'")
        fb = r["feedback"]
        print(f"\n  {fb['headline_en']}")
        for a, e in zip(fb["points_ar"], fb["points_en"]):
            print(f"    - {e}   ({a})")

    def show_rating(self, result, figsize=(16, 3.6), show=True):
        """The visual report: five pipeline stages plus the scorecard."""
        self._init_mpl_font()
        st, ref = result["_stages"], result["_ref"]
        child = st["child"]
        panels = [(cv2.cvtColor(st["warped"], cv2.COLOR_BGR2RGB), "1. page, deskewed"),
                  (st["cleaned"], "2. ink, ruling removed"),
                  (child, "3. normalised 64x64"),
                  (ref, f"4. Amiri model ({result['matched_form']})"),
                  (np.dstack([child, ref, np.minimum(child, ref)]),
                   "5. the pair the CNN rates")]
        fig, axes = plt.subplots(1, len(panels), figsize=figsize)
        for ax, (im, t) in zip(axes, panels):
            ax.imshow(im, cmap=None if im.ndim == 3 else "gray")
            ax.set_title(t, fontsize=9)
            ax.axis("off")
        head = (f"{result['letter']}   {result['rating']:.0f}%   "
                f"{result['grade_en']}"
                + (f"   |   neater than {result['rating']:.0f}% of children aged 7-12"
                   if result["calibrated"] else "   (uncalibrated)"))
        fig.suptitle(head, fontsize=13,
                     fontproperties=getattr(self, "AMIRI_MPL", None))
        plt.tight_layout()
        if show:
            plt.show()
            self.print_rating(result)
        return fig

    # ==========================================================================
    #  16.  PERSISTENCE
    # ==========================================================================
    def save(self, out=None):
        """Write everything the app needs to score a photo: weights, bank, curve."""
        out = out or self.cfg["artifacts"]
        os.makedirs(out, exist_ok=True)

        if self.ref_arr is not None:
            np.savez_compressed(
                os.path.join(out, "amiri_reference_bank.npz"),
                ref=self.ref_arr,
                forms=np.array([self.form_names], dtype=object),
                letters=np.array(self.letters))
        for name, m in self.models.items():
            for i, mm in enumerate(m if isinstance(m, list) else [m]):
                suffix = f"_{i}" if isinstance(m, list) else ""
                mm.save(os.path.join(out, f"model_{name}{suffix}.keras"))
        for n, c in (getattr(self, "calibrations", None) or {}).items():
            np.save(os.path.join(out, f"calibration_{n}.npy"), c)
        if self.calibration is not None:
            np.save(os.path.join(out, "calibration_curve.npy"), self.calibration)
        if self.histories:
            json.dump(self.histories, open(os.path.join(out, "histories.json"), "w"))
        if self.results:
            clean = {k: {kk: vv for kk, vv in v.items() if kk != "preds"}
                     for k, v in self.results.items()}
            json.dump(clean, open(os.path.join(out, "results.json"), "w"), indent=1)
        json.dump({k: v for k, v in self.cfg.items()},
                  open(os.path.join(out, "config.json"), "w"), indent=1, default=str)
        self._log(f"artifacts -> {out}/")
        return out

    def load(self, out=None, models=("production", "scratch", "mobilenet")):
        """
        Restore a trained object without touching the dataset. This is the
        path the app takes at start-up: it needs the weights, the reference
        bank and the calibration curve, and nothing else.
        """
        out = out or self.cfg["artifacts"]
        bank = os.path.join(out, "amiri_reference_bank.npz")
        if os.path.exists(bank):
            self._load_bank(bank)
            self._log(f"reference bank <- {bank}")
        else:
            self.build_reference_bank()

        if _TF_OK:
            for name in models:
                members = sorted(glob.glob(os.path.join(out, f"model_{name}_*.keras")))
                single = os.path.join(out, f"model_{name}.keras")
                try:
                    if members:
                        self.models[name] = [tf.keras.models.load_model(p)
                                             for p in members]
                        self._log(f"{name} <- {len(members)} ensemble members")
                    elif os.path.exists(single):
                        self.models[name] = tf.keras.models.load_model(single)
                        self._log(f"{name} <- {single}")
                except Exception as e:
                    self._log(f"could not load {name}: {e}")
        else:
            self._log("TensorFlow unavailable -- ratings will use the "
                      "classical geometric scorer only")

        self.calibrations = {}
        for f in glob.glob(os.path.join(out, "calibration_*.npy")):
            n = os.path.basename(f)[len("calibration_"):-len(".npy")]
            if n != "curve":
                self.calibrations[n] = np.load(f)
        meta = os.path.join(out, "calibration_meta.json")
        if os.path.exists(meta):
            self.calibration_source = json.load(open(meta)).get("source")
        curve = os.path.join(out, "calibration_curve.npy")
        if os.path.exists(curve):
            self.calibration = np.load(curve)
        elif self.calibration_source in self.calibrations:
            self.calibration = self.calibrations[self.calibration_source]
        if self.calibrations:
            self._log(f"calibration <- curves for {sorted(self.calibrations)}")
        res = os.path.join(out, "results.json")
        if os.path.exists(res):
            self.results = json.load(open(res))
        return self

    def export_zip(self, path=None, download=True):
        """Zip the artifacts folder; in Colab, also start the download."""
        path = path or os.path.join(_HERE, "handwriting_cv_artifacts.zip")
        src = self.cfg["artifacts"]
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(src):
                for f in files:
                    full = os.path.join(root, f)
                    z.write(full, os.path.relpath(full, os.path.dirname(src)))
        mb = os.path.getsize(path) / 1e6
        self._log(f"{path}  ({mb:.1f} MB)")
        if download and _in_colab():
            from google.colab import files
            files.download(path)
        return path

    # ==========================================================================
    #  16b.  A SYNTHETIC "PHONE PHOTO", so the demo runs with nothing uploaded
    # ==========================================================================
    def make_sample(self, text="ب", W=1000, H=700, paper="ruled",
                    ink=(150, 60, 30), sloppiness=1.0, baseline_shift=0,
                    rotate=3.0, perspective=0.05, seed=7):
        """
        Fake a photo of a child's notebook: Amiri warped into a wobbly hand,
        laid on ruled paper, tilted on a desk, lit unevenly and photographed
        at an angle.

        This is for demonstrating and testing the front end -- it is never
        used as training data, because a distorted font is not a child.

        paper: "ruled" | "grid" | "plain".  `ink` is BGR.  `baseline_shift` in
        px (positive = the child wrote below the rule).  Returns BGR uint8.
        """
        self.ensure_font()
        rng = np.random.default_rng(seed)
        page = np.full((H, W, 3), 252, np.uint8)
        page = np.clip(page.astype(np.float32) + rng.normal(0, 2.5, page.shape),
                       0, 255).astype(np.uint8)

        rule_y = list(range(90, H - 40, 70))
        if paper in ("ruled", "grid"):
            for y in rule_y:
                cv2.line(page, (40, y), (W - 40, y), (225, 190, 165), 1)
        if paper == "grid":
            for x in range(40, W - 40, 70):
                cv2.line(page, (x, 60), (x, H - 40), (225, 190, 165), 1)

        shaped, raqm = self.shape_arabic(text)
        eng = ImageFont.Layout.RAQM if raqm else ImageFont.Layout.BASIC
        size = 190
        probe = ImageFont.truetype(self.cfg["font_path"], size, layout_engine=eng)
        bb = probe.getbbox(shaped)
        tw = bb[2] - bb[0]
        if tw > W * 0.82:                       # shrink so long text still fits
            size = max(46, int(size * (W * 0.82) / max(tw, 1)))
        font = ImageFont.truetype(self.cfg["font_path"], size, layout_engine=eng)
        lay = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(lay)
        base = rule_y[len(rule_y) // 2] + baseline_shift
        kw = dict(direction="rtl", language="ar") if raqm else {}
        d.text((W // 2, base), shaped, font=font, fill=255, anchor="ms", **kw)

        g = np.array(lay)
        g = self._elastic(g, alpha=0.055 * size * sloppiness, sigma=16.0, rng=rng)
        g = cv2.GaussianBlur(g, (0, 0), 1.0)
        thick = 1 + int(round(rng.uniform(0, 1.5) * sloppiness))
        g = cv2.dilate(g, np.ones((thick, thick), np.uint8))

        a = (g.astype(np.float32) / 255.0)[..., None]
        page = (page * (1 - a) + np.array(ink, np.float32) * a).astype(np.uint8)

        # ---- put the page on a desk, tilt it, light it unevenly -------------
        M = cv2.getRotationMatrix2D((W / 2, H / 2), rotate, 0.88)
        M[0, 2] += 120
        M[1, 2] += 90
        scene = cv2.warpAffine(page, M, (W + 240, H + 180), borderValue=(70, 72, 78))
        SH, SW = scene.shape[:2]
        p = perspective
        src = np.float32([[0, 0], [SW, 0], [SW, SH], [0, SH]])
        dst = np.float32([[SW * p, SH * p * 0.5], [SW * (1 - p * 0.3), 0],
                          [SW, SH * (1 - p * 0.2)], [SW * p * 0.4, SH]])
        scene = cv2.warpPerspective(scene, cv2.getPerspectiveTransform(src, dst),
                                    (SW, SH), borderValue=(70, 72, 78))
        yy, xx = np.mgrid[0:SH, 0:SW].astype(np.float32)
        shade = 0.72 + 0.28 * np.exp(-((xx - SW * 0.35) ** 2 + (yy - SH * 0.3) ** 2)
                                     / (2 * (SW * 0.7) ** 2))
        scene = np.clip(scene * shade[..., None], 0, 255).astype(np.uint8)
        return np.clip(scene.astype(np.float32) + rng.normal(0, 3.5, scene.shape),
                       0, 255).astype(np.uint8)

    # ==========================================================================
    #  17.  THE WHOLE PROJECT IN ONE CALL
    # ==========================================================================
    def run_all(self, which="all", show=True, save=True):
        """
        dataset -> reference bank -> pairs -> train -> evaluate -> calibrate
        -> save.  This is what the notebook runs.
        """
        t0 = time.time()
        self.device_report()
        self.load_dataset()
        self.dataset_summary()
        if show:
            self.show_class_distribution()
            self.show_dataset()

        self.build_reference_bank()
        if show:
            self.show_degradations()

        self.build_pairs(split="train")
        self.build_pairs(split="test")
        if show:
            self.show_pairs()
            self.label_histogram()

        self.train(which)
        self.evaluate()
        self.calibrate()
        if show:
            self.show_history()
            if self.models:
                self.show_scatter("production" if "production" in self.models
                                  else list(self.models)[0])
            self.show_calibration()
        if save:
            self.save()
        self._log(f"\ndone in {(time.time() - t0) / 60:.1f} minutes")
        return self


# ==============================================================================
#  CLI
# ==============================================================================
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        prog="handwriting_cv.py",
        description="Anees -- Arabic handwriting Quality & Similarity rating")
    sub = ap.add_subparsers(dest="cmd")

    t = sub.add_parser("train", help="run the whole project end to end")
    t.add_argument("--data", help="hijja2.npz or the folder with the CSVs")
    t.add_argument("--out", help="artifacts folder")
    t.add_argument("--model", default="all",
                   choices=["all", "scratch", "mobilenet", "production"])
    t.add_argument("--pairs", type=int)
    t.add_argument("--test-pairs", type=int)
    t.add_argument("--epochs", type=int)
    t.add_argument("--ensemble", type=int)
    t.add_argument("--no-show", action="store_true")

    r = sub.add_parser("rate", help="rate one photo")
    r.add_argument("--image", required=True)
    r.add_argument("--letter", required=True)
    r.add_argument("--out", help="artifacts folder")
    r.add_argument("--model", default="production")
    r.add_argument("--save", help="write the visual report to this PNG")
    r.add_argument("--json", dest="json_out")

    f = sub.add_parser("rate-folder", help="rate every image in a folder")
    f.add_argument("--folder", required=True)
    f.add_argument("--letter", required=True)
    f.add_argument("--out")

    s = sub.add_parser("steps", help="show the preprocessing stages only")
    s.add_argument("--image", required=True)
    s.add_argument("--letter")
    s.add_argument("--save")

    a = ap.parse_args(argv)
    if a.cmd is None:
        ap.print_help()
        return 0

    cfg = {}
    if getattr(a, "out", None):
        cfg["artifacts"] = a.out
    for k, key in (("pairs", "n_train_pairs"), ("test_pairs", "n_test_pairs"),
                   ("epochs", "epochs"), ("ensemble", "ensemble")):
        v = getattr(a, k, None)
        if v:
            cfg[key] = v
    cv = AneesHandwritingCV(**cfg)

    if a.cmd == "train":
        if a.data:
            cv.load_dataset(a.data)
        if a.no_show:
            matplotlib.use("Agg")
        cv.run_all(a.model, show=not a.no_show)
        return 0

    if a.cmd == "steps":
        if a.save:
            matplotlib.use("Agg")
        fig, _ = cv.show_pipeline(a.image, a.letter, show=not a.save)
        if a.save:
            fig.savefig(a.save, dpi=120, bbox_inches="tight")
            print(f"saved -> {a.save}")
        return 0

    cv.load()
    if a.cmd == "rate-folder":
        cv.rate_folder(a.folder, a.letter)
        return 0

    if a.save:
        matplotlib.use("Agg")
    res = cv.rate(a.image, a.letter, model=a.model)
    cv.print_rating(res)
    if a.json_out:
        json.dump({k: v for k, v in res.items() if not k.startswith("_")},
                  open(a.json_out, "w"), ensure_ascii=False, indent=1)
        print(f"json -> {a.json_out}")
    if a.save:
        fig = cv.show_rating(res, show=False)
        fig.savefig(a.save, dpi=120, bbox_inches="tight")
        print(f"report -> {a.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
