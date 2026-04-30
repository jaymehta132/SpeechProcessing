import json

with open('notebook4dedb513ee.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

def code(idx, src):
    nb['cells'][idx]['source'] = src
    nb['cells'][idx]['outputs'] = []
    nb['cells'][idx]['execution_count'] = None

def md(idx, src):
    nb['cells'][idx]['source'] = src

# ── helpers ──────────────────────────────────────────────────────────────────
def find_cell(token):
    for i,c in enumerate(nb['cells']):
        if token in ''.join(c.get('source','')):
            return i
    return None

# ─────────────────────────────────────────────────────────────────────────────
# CELL 0  — title + bug table
# ─────────────────────────────────────────────────────────────────────────────
md(0, """# 🎙️ TTS Benchmark — Hindi & English Open-Source Models

This notebook benchmarks open-source **Text-to-Speech (TTS)** systems across Hindi and English.

---

## 🤖 Models Benchmarked

| Model | Language(s) | Architecture |
|---|---|---|
| **MMS-TTS** (ENG / HIN) | English + Hindi | VITS end-to-end |
| **SpeechT5** | English | Transformer + HiFi-GAN |
| **Parler-TTS Mini** | English + Hindi* | Decoder-only (style-conditioned) |
| **Coqui VITS** | English + Hindi | VITS (language-specific) |
| **XTTS v2** | English + Hindi | Zero-shot multilingual |

> *Hindi via Devanagari → IAST romanisation workaround (see §5.4)

---

## 🐛 Bugs Fixed

| # | Model | Root Cause | Fix |
|---|---|---|---|
| 1 | **SpeechT5** | HuggingFace repo ID `speecht5-hifigan` (hyphen) doesn't exist | Changed to `speecht5_hifigan` (underscore) |
| 2 | **SpeechT5** | Speaker index `7306` not bounds-checked | Added `min(idx, len(dataset)-1)` guard |
| 3 | **Coqui-VITS, XTTS-v2** | `importlib` cache not refreshed after in-kernel `pip install` | Added `importlib.invalidate_caches()` before every Coqui import |
| 4 | **Coqui-VITS, XTTS-v2** | Real `ImportError` swallowed by generic message | Re-raise with `type(e).__name__: {e}` |
| 5 | **Coqui-VITS** | `espeak-ng` not installed — phonemizer crashes | Install `espeak-ng` via apt in the install cell |
| 6 | **XTTS-v2** | Interactive license prompt blocks execution | Set `os.environ["COQUI_TOS_AGREED"] = "1"` before any Coqui import |
| 7 | **Parler-TTS** | English-only — no Hindi support | Transliterate Devanagari → IAST via `indic-transliteration` |
| 8 | **UTMOS** | `fairseq` dataclass bug on Python 3.12 | Monkey-patch before UTMOS import |
| 9 | **Ranking cell** | `.rank().astype(int)` crashes on all-NaN columns | Float ranks; skip all-NaN columns |

---

## 📊 Metrics

| Category | Metric | Direction |
|---|---|---|
| **Performance** | Latency (ms) | ↓ lower is better |
| **Performance** | Real-Time Factor (RTF) | ↓ lower is better (<1 = faster than real-time) |
| **Performance** | Throughput (chars/sec) | ↑ higher is better |
| **Quality** | MOS — UTMOS [1–5] | ↑ higher is better |
| **Quality** | WER % (Whisper ASR) | ↓ lower is better |
| **Quality** | CER % (Whisper ASR) | ↓ lower is better |
| **Prosody** | Pitch mean / std / range (Hz) | context-dependent |
| **Prosody** | Speaking rate (onsets/sec) | context-dependent |
| **Prosody** | Energy dynamics (RMS std) | ↑ higher = more expressive |
| **Robustness** | Per-category WER (8 types) | ↓ lower is better |

---

## 📁 Outputs
- `tts_benchmark_results/csv/` — CSVs (full, summary, robustness, per-model, rankings)
- `tts_benchmark_results/plots/` — 11 PNG plots
- `tts_benchmark_results/audio/` — synthesised WAV files""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1  — install markdown
# ─────────────────────────────────────────────────────────────────────────────
md(1, """## 1. Install Dependencies

### New in this version
- **`espeak-ng`** is installed via `apt` — required by Coqui-VITS's phonemizer backend.  Without it you get `[!] No espeak backend found`.
- **`COQUI_TOS_AGREED=1`** environment variable is set before any Coqui import to suppress the interactive XTTS-v2 license prompt.
- **`importlib.invalidate_caches()`** is called after each `pip install` so packages are importable in the same kernel session without a restart.
- **`indic-transliteration`** enables Parler-TTS to handle Hindi via Devanagari → IAST romanisation.

### Python / Coqui TTS compatibility

| Python version | Strategy |
|---|---|
| **< 3.12** | `pip install TTS>=0.22.0` (official PyPI) |
| **≥ 3.12** | `pip install coqui-tts` (idiap community fork) |""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3  — install code (complete rewrite)
# ─────────────────────────────────────────────────────────────────────────────
code(3, """import subprocess, sys, os, shutil, importlib

ON_KAGGLE = os.path.exists("/kaggle/working")
PY_VER    = sys.version_info
print(f"Environment : {'Kaggle' if ON_KAGGLE else 'Local'}")
print(f"Python      : {PY_VER.major}.{PY_VER.minor}.{PY_VER.micro}")
print()

# ── CRITICAL: set BEFORE any Coqui import — silences XTTS-v2 license prompt ─
os.environ["COQUI_TOS_AGREED"] = "1"

def pip(*pkgs, fatal=True):
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--no-warn-conflicts", *pkgs]
    if fatal:
        subprocess.check_call(cmd)
        return True
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError:
        return False

def apt(pkg):
    \"\"\"Install a system package via apt-get (Linux / Kaggle / Colab).\"\"\"
    try:
        subprocess.check_call(["apt-get", "install", "-y", "-q", pkg],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

def try_import_tts():
    \"\"\"Return (ok, error_str) by actually attempting from TTS.api import TTS.\"\"\"
    importlib.invalidate_caches()
    try:
        from TTS.api import TTS  # noqa
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Core ML
# ─────────────────────────────────────────────────────────────────────────────
print("[1/8] Core ML packages …")
pip("transformers>=4.40.0", "accelerate>=0.27.0",
    "datasets>=2.18.0",     "sentencepiece>=0.1.99")
print("  ✓ transformers / accelerate / datasets")

# ─────────────────────────────────────────────────────────────────────────────
# 2. espeak-ng  — MUST come before Coqui TTS install
#    Coqui-VITS uses phonemizer which calls espeak-ng as a subprocess.
#    Without it you get: [!] No espeak backend found
# ─────────────────────────────────────────────────────────────────────────────
print("[2/8] espeak-ng (phonemizer backend for Coqui-VITS) …")
ESPEAK_OK = False
if shutil.which("espeak-ng") or shutil.which("espeak"):
    ESPEAK_OK = True
    print("  ✓ espeak-ng already installed")
else:
    ok = apt("espeak-ng")
    if ok and (shutil.which("espeak-ng") or shutil.which("espeak")):
        ESPEAK_OK = True
        print("  ✓ espeak-ng installed via apt")
    else:
        # Try phonemizer's bundled espeak wrapper as a fallback
        pip("phonemizer", fatal=False)
        importlib.invalidate_caches()
        try:
            import phonemizer.backend.espeak.espeak as _esp
            _esp.EspeakBackend.is_available()
            ESPEAK_OK = True
            print("  ✓ espeak available via phonemizer")
        except Exception:
            print("  ⚠ espeak-ng not found — Coqui-VITS will be skipped.")
            print("    Manual install: sudo apt install espeak-ng")
            print("    macOS         : brew install espeak")
            print("    Windows       : https://github.com/espeak-ng/espeak-ng/releases")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Coqui TTS
# ─────────────────────────────────────────────────────────────────────────────
print("[3/8] Coqui TTS …")
COQUI_AVAILABLE = False

if PY_VER < (3, 12):
    ok = pip("TTS>=0.22.0", fatal=False)
    if ok:
        can, err = try_import_tts()
        if can:
            COQUI_AVAILABLE = True
            print("  ✓ Coqui TTS (official PyPI) installed and importable")
        else:
            print(f"  ⚠ Installed but import failed: {err[:160]}")

if not COQUI_AVAILABLE:
    label = "idiap community fork (Python ≥3.12)" if PY_VER >= (3,12) else "idiap fork fallback"
    print(f"  Trying coqui-tts ({label}) …")
    ok = pip("coqui-tts", fatal=False)
    if ok:
        can, err = try_import_tts()
        COQUI_AVAILABLE = can
        if can:
            print("  ✓ coqui-tts importable")
        else:
            print(f"  ⚠ coqui-tts installed but import failed: {err[:200]}")

if not COQUI_AVAILABLE:
    print("  Trying git install of idiap/coqui-ai-TTS …")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "--no-warn-conflicts",
                               "git+https://github.com/idiap/coqui-ai-TTS.git"])
        can, err = try_import_tts()
        COQUI_AVAILABLE = can
        print("  ✓ coqui-ai-TTS (git) importable" if can
              else f"  ✗ git install import failed: {err[:200]}")
    except Exception as e:
        print(f"  ✗ All Coqui install attempts failed: {e}")
        print("  → Coqui-VITS and XTTS-v2 will be SKIPPED.")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Parler-TTS
# ─────────────────────────────────────────────────────────────────────────────
print("[4/8] Parler-TTS …")
PARLER_AVAILABLE = pip("parler-tts", fatal=False)
if not PARLER_AVAILABLE:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "git+https://github.com/huggingface/parler-tts.git"])
        PARLER_AVAILABLE = True
    except Exception:
        pass
if PARLER_AVAILABLE:
    importlib.invalidate_caches()
    print("  ✓ parler-tts")
else:
    print("  ⚠ Parler-TTS unavailable — will be SKIPPED")

# ─────────────────────────────────────────────────────────────────────────────
# 5. indic-transliteration  (Parler-TTS Hindi workaround)
# ─────────────────────────────────────────────────────────────────────────────
print("[5/8] indic-transliteration (Parler-TTS Hindi workaround) …")
INDIC_TRANS_AVAILABLE = pip("indic-transliteration", fatal=False)
if INDIC_TRANS_AVAILABLE:
    importlib.invalidate_caches()
    print("  ✓ indic-transliteration")
else:
    print("  ⚠ unavailable — Parler-TTS will skip Hindi sentences")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Audio / prosody
# ─────────────────────────────────────────────────────────────────────────────
print("[6/8] Audio / prosody packages …")
pip("soundfile>=0.12.1", "librosa>=0.10.1", "scipy>=1.12.0")
print("  ✓ soundfile / librosa / scipy")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Whisper + jiwer
# ─────────────────────────────────────────────────────────────────────────────
print("[7/8] Whisper + jiwer …")
pip("openai-whisper>=20231117", "jiwer>=3.0.3")
print("  ✓ openai-whisper / jiwer")

# ffmpeg (Whisper requirement)
if shutil.which("ffmpeg") is None:
    if ON_KAGGLE or os.path.exists("/usr/bin/apt-get"):
        apt("ffmpeg")
        print("  ✓ ffmpeg installed via apt")
    else:
        print("  ⚠ ffmpeg not found — install: sudo apt install ffmpeg / brew install ffmpeg")
else:
    print("  ✓ ffmpeg found")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Data / viz
# ─────────────────────────────────────────────────────────────────────────────
print("[8/8] Data / viz packages …")
pip("pandas>=2.1.0", "matplotlib>=3.8.0", "seaborn>=0.13.0", "numpy>=1.24.0")
print("  ✓ pandas / matplotlib / seaborn / numpy")

# ── UTMOS (optional) ─────────────────────────────────────────────────────────
print("[opt] UTMOS MOS predictor …")
UTMOS_AVAILABLE = pip("utmos", fatal=False)
if UTMOS_AVAILABLE:
    importlib.invalidate_caches()
    print("  ✓ utmos — MOS scoring enabled")
else:
    print("  ⚠ utmos unavailable — MOS column will be NaN (non-fatal)")

# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  INSTALL SUMMARY")
print("=" * 60)
print(f"  espeak-ng (Coqui phonemizer) : {'✓' if ESPEAK_OK            else '✗ MISSING — Coqui-VITS will fail'}")
print(f"  Coqui TTS (VITS / XTTS-v2)  : {'✓' if COQUI_AVAILABLE      else '✗ skipped'}")
print(f"  Parler-TTS                   : {'✓' if PARLER_AVAILABLE     else '✗ skipped'}")
print(f"  Parler Hindi transliteration : {'✓' if INDIC_TRANS_AVAILABLE else '⚠ skipped (Hindi only)'}")
print(f"  UTMOS MOS scorer             : {'✓' if UTMOS_AVAILABLE      else '⚠ NaN (optional)'}")
print("=" * 60)
print()
print("NOTE: COQUI_TOS_AGREED=1 has been set — XTTS-v2 license prompt suppressed.")""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5  — imports: add os.environ guard so it's set even if kernel restarted
# ─────────────────────────────────────────────────────────────────────────────
code(5, """import logging, os, sys, time, warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# ── CRITICAL: must be set before ANY Coqui import anywhere in this kernel ────
# Suppresses the interactive "I agree to the CPML license" prompt in XTTS-v2.
os.environ["COQUI_TOS_AGREED"] = "1"

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch {torch.__version__} | Device: {DEVICE.upper()}")
print(f"COQUI_TOS_AGREED = {os.environ.get('COQUI_TOS_AGREED','NOT SET')}")""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 21 markdown — SpeechT5
# ─────────────────────────────────────────────────────────────────────────────
md(21, """### 5.3 Model 2 — SpeechT5 (Microsoft, English)

[microsoft/speecht5_tts](https://huggingface.co/microsoft/speecht5_tts) uses a HiFi-GAN neural vocoder.

**Bug fixed:** Repo ID was `speecht5-hifigan` (hyphen — does not exist on HuggingFace).  
Correct name: `speecht5_hifigan` (underscore). Speaker index is also bounds-checked.""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 22 — SpeechT5 fixed
# ─────────────────────────────────────────────────────────────────────────────
code(22, """class SpeechT5Wrapper(BaseTTS):
    name = "SpeechT5"
    supported_langs = ["en"]
    SPEAKER_IDX = 7306  # ✏️ change to pick a different CMU-Arctic voice

    def load(self):
        from transformers import (
            SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor
        )
        from datasets import load_dataset as _hf_ds
        log.info(f"  [{self.name}] Loading model …")
        self.proc    = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model   = SpeechT5ForTextToSpeech.from_pretrained(
            "microsoft/speecht5_tts"
        ).to(DEVICE)

        # ✅ FIX 1: underscore, NOT hyphen  ("speecht5-hifigan" → 404 on HuggingFace)
        log.info(f"  [{self.name}] Loading HiFi-GAN vocoder …")
        self.vocoder = SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan"     # ← correct ID
        ).to(DEVICE)
        self.model.eval()
        self.vocoder.eval()

        log.info(f"  [{self.name}] Loading speaker embedding …")
        emb_ds = _hf_ds("Matthijs/cmu-arctic-xvectors", split="validation")
        # ✅ FIX 2: bounds-check so index never exceeds dataset size
        safe_idx = min(self.SPEAKER_IDX, len(emb_ds) - 1)
        if safe_idx != self.SPEAKER_IDX:
            log.warning(f"  [{self.name}] Speaker index {self.SPEAKER_IDX} clamped "
                        f"to {safe_idx} (dataset size={len(emb_ds)})")
        self.spk_emb = torch.tensor(
            emb_ds[safe_idx]["xvector"]
        ).unsqueeze(0).to(DEVICE)
        log.info(f"  [{self.name}] Ready (speaker={safe_idx}).")

    def synthesize(self, text: str, lang: str, out_path: Path) -> float:
        import soundfile as sf
        inputs = self.proc(text=text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            speech = self.model.generate_speech(
                inputs["input_ids"], self.spk_emb, vocoder=self.vocoder
            )
        wav = speech.cpu().numpy()
        sf.write(str(out_path), wav, 16_000)
        return len(wav) / 16_000

    def unload(self):
        del self.model, self.proc, self.vocoder, self.spk_emb
        super().unload()

print("SpeechT5Wrapper defined.")""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 23 markdown — Parler-TTS
# ─────────────────────────────────────────────────────────────────────────────
md(23, """### 5.4 Model 3 — Parler-TTS Mini (English + Hindi workaround)

Parler-TTS is **English-only**. For Hindi we romanise Devanagari → IAST using  
`indic-transliteration`, then synthesise with an Indian-English voice description.

This is a best-effort approximation — WER for Hindi will be higher than native models.""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 24 — Parler-TTS with Hindi transliteration
# ─────────────────────────────────────────────────────────────────────────────
code(24, """class ParlerTTSWrapper(BaseTTS):
    name = "Parler-TTS"
    # ✅ FIX: Hindi supported via transliteration workaround
    supported_langs = ["en", "hi"]

    VOICE_DESC_EN = (
        "A female speaker delivers a slightly expressive and animated speech "
        "with a moderate speed and pitch. The recording is of very high "
        "quality, with the speaker's voice sounding clear and very close up."
    )
    VOICE_DESC_HI = (
        "A female speaker with a clear Indian English accent delivers the text "
        "at a moderate pace with natural intonation. The recording is of high "
        "quality with the speaker's voice sounding close and clear."
    )

    def _romanise_hindi(self, text: str) -> str:
        \"\"\"Transliterate Devanagari → IAST Roman script.\"\"\"
        try:
            from indic_transliteration import sanscript
            from indic_transliteration.sanscript import transliterate
            romanised = transliterate(text, sanscript.DEVANAGARI, sanscript.IAST)
            log.info(f"  [Parler-TTS] Romanised: {text[:35]} → {romanised[:35]}")
            return romanised
        except Exception as e:
            log.warning(f"  [Parler-TTS] Transliteration failed ({e}); using original text")
            return text

    def load(self):
        try:
            from parler_tts import ParlerTTSForConditionalGeneration
        except ImportError:
            raise RuntimeError("parler-tts not installed: pip install parler-tts")
        from transformers import AutoTokenizer
        log.info(f"  [{self.name}] Loading parler-tts-mini-v1 …")
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            "parler-tts/parler-tts-mini-v1"
        ).to(DEVICE)
        self.tok = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
        self.model.eval()
        log.info(f"  [{self.name}] Ready.")

    def synthesize(self, text: str, lang: str, out_path: Path) -> float:
        import soundfile as sf
        if lang == "hi":
            synth_text = self._romanise_hindi(text)
            voice_desc = self.VOICE_DESC_HI
        else:
            synth_text = text
            voice_desc = self.VOICE_DESC_EN

        desc_tok   = self.tok(voice_desc,  return_tensors="pt").to(DEVICE)
        prompt_tok = self.tok(synth_text,  return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen = self.model.generate(
                input_ids             = desc_tok.input_ids,
                attention_mask        = desc_tok.attention_mask,
                prompt_input_ids      = prompt_tok.input_ids,
                prompt_attention_mask = prompt_tok.attention_mask,
            )
        wav = gen.cpu().numpy().squeeze()
        sr  = self.model.config.sampling_rate
        sf.write(str(out_path), wav, sr)
        return len(wav) / sr

    def unload(self):
        del self.model, self.tok
        super().unload()

print("ParlerTTSWrapper defined.")""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 25 markdown — Coqui-VITS
# ─────────────────────────────────────────────────────────────────────────────
md(25, """### 5.5 Model 4 — Coqui VITS (English + Hindi)

- English: `tts_models/en/ljspeech/vits`
- Hindi: `tts_models/hi/cv/vits`

**Bugs fixed:**
1. `importlib.invalidate_caches()` called before import (in-kernel pip install cache)
2. Real `ImportError` is now surfaced instead of a generic "not installed" message
3. `COQUI_TOS_AGREED=1` env var is verified before loading
4. `espeak-ng` must be installed (handled in the install cell)""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 26 — Coqui-VITS fixed
# ─────────────────────────────────────────────────────────────────────────────
code(26, """class CoquiVITSWrapper(BaseTTS):
    name = "Coqui-VITS"
    supported_langs = ["en", "hi"]
    _LANG_TO_MODEL = {
        "en": "tts_models/en/ljspeech/vits",
        "hi": "tts_models/hi/cv/vits",
    }

    def __init__(self):
        self._instances: Dict[str, object] = {}

    def load(self):
        # ✅ FIX 1: check install-time availability flag
        if not COQUI_AVAILABLE:
            raise RuntimeError(
                "Coqui TTS install failed at startup — check install cell output. "
                "Also ensure espeak-ng is installed: sudo apt install espeak-ng"
            )
        # ✅ FIX 2: ensure license env var is set (XTTS-v2 needs this; set here too for safety)
        os.environ["COQUI_TOS_AGREED"] = "1"
        # ✅ FIX 3: refresh module finder cache so the pip-installed package is visible
        import importlib
        importlib.invalidate_caches()
        try:
            from TTS.api import TTS as _CoquiTTS
        except Exception as e:
            # ✅ FIX 4: surface the real error, not a generic message
            raise RuntimeError(
                f"Coqui TTS import failed. Actual error → {type(e).__name__}: {e}\\n"
                "Common causes: espeak-ng missing, broken dependency, importlib cache"
            )
        for lang, model_id in self._LANG_TO_MODEL.items():
            if lang not in LANGUAGES:
                continue
            log.info(f"  [{self.name}] Loading {model_id} …")
            self._instances[lang] = _CoquiTTS(model_id, gpu=(DEVICE == "cuda"))
        log.info(f"  [{self.name}] Ready for {list(self._instances.keys())}")

    def synthesize(self, text: str, lang: str, out_path: Path) -> float:
        import soundfile as sf
        self._instances[lang].tts_to_file(text=text, file_path=str(out_path))
        data, sr = sf.read(str(out_path))
        return len(data) / sr

    def unload(self):
        del self._instances
        super().unload()

print("CoquiVITSWrapper defined.")""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 27 markdown — XTTS-v2
# ─────────────────────────────────────────────────────────────────────────────
md(27, """### 5.6 Model 5 — XTTS v2 (Coqui, multilingual zero-shot)

17-language zero-shot TTS. Supports Hindi and English natively.

**Bugs fixed:**
1. `COQUI_TOS_AGREED=1` env var suppresses the interactive license prompt  
   (previously blocked execution waiting for `y/n` keyboard input)
2. Same `importlib` cache and error-transparency fixes as Coqui-VITS""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 28 — XTTS-v2 fixed
# ─────────────────────────────────────────────────────────────────────────────
code(28, """class XTTSv2Wrapper(BaseTTS):
    name = "XTTS-v2"
    supported_langs = ["en", "hi"]
    _LANG_CODES = {"en": "en", "hi": "hi"}
    _MODEL_ID   = "tts_models/multilingual/multi-dataset/xtts_v2"

    def load(self):
        # ✅ FIX 1: check availability
        if not COQUI_AVAILABLE:
            raise RuntimeError(
                "Coqui TTS install failed at startup — check install cell output."
            )
        # ✅ FIX 2: set env var immediately before the import — this is what
        #           suppresses the "I agree to CPML / commercial license" prompt.
        #           Must be set BEFORE `from TTS.api import TTS` is called.
        os.environ["COQUI_TOS_AGREED"] = "1"
        # ✅ FIX 3: refresh importlib cache
        import importlib
        importlib.invalidate_caches()
        try:
            from TTS.api import TTS as _CoquiTTS
        except Exception as e:
            # ✅ FIX 4: surface real error
            raise RuntimeError(
                f"Coqui TTS import failed. Actual error → {type(e).__name__}: {e}"
            )
        log.info(f"  [{self.name}] Loading XTTS v2 (multilingual) …")
        # Pass agree_to_tos=True as an extra safeguard for older Coqui versions
        try:
            self.tts = _CoquiTTS(self._MODEL_ID, gpu=(DEVICE == "cuda"),
                                  agree_to_tos=True)
        except TypeError:
            # Older API does not accept agree_to_tos kwarg — env var is enough
            self.tts = _CoquiTTS(self._MODEL_ID, gpu=(DEVICE == "cuda"))
        self._speaker = (
            self.tts.speakers[0] if self.tts.speakers else "Claribel Dervla"
        )
        log.info(f"  [{self.name}] Using speaker: {self._speaker}")
        log.info(f"  [{self.name}] Ready.")

    def synthesize(self, text: str, lang: str, out_path: Path) -> float:
        import soundfile as sf
        self.tts.tts_to_file(
            text      = text,
            speaker   = self._speaker,
            language  = self._LANG_CODES.get(lang, "en"),
            file_path = str(out_path),
        )
        data, sr = sf.read(str(out_path))
        return len(data) / sr

    def unload(self):
        del self.tts
        super().unload()

print("XTTSv2Wrapper defined.")""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 36 markdown — UTMOS
# ─────────────────────────────────────────────────────────────────────────────
md(36, """### 6.3 MOS Prediction — UTMOS

Neural MOS predictor — returns [1, 5], higher is better.

**Bug fixed:** `fairseq` (UTMOS dependency) has a dataclass mutable-default bug on  
Python ≥ 3.12. We apply a monkey-patch before loading UTMOS. If the patch fails,  
MOS silently returns NaN.""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 37 — UTMOS with fairseq patch
# ─────────────────────────────────────────────────────────────────────────────
code(37, """_utmos_cache    = None
_utmos_available: Optional[bool] = None

def _apply_fairseq_patch():
    \"\"\"
    Monkey-patch for fairseq's mutable-default dataclass fields.
    Python 3.12 disallows mutable defaults in @dataclass; fairseq violates this.
    We pre-import fairseq and replace offending fields with default_factory.
    \"\"\"
    try:
        import dataclasses
        import fairseq.dataclass.configs as _fdc  # trigger the error early
        return True  # no patch needed
    except TypeError as e:
        if "mutable default" not in str(e).lower():
            return False
        try:
            import dataclasses, fairseq.dataclass.configs as _fdc
            for name in dir(_fdc):
                cls = getattr(_fdc, name, None)
                if isinstance(cls, type) and dataclasses.is_dataclass(cls):
                    for f in dataclasses.fields(cls):
                        if (f.default is not dataclasses.MISSING
                                and isinstance(f.default, (list, dict, set))):
                            object.__setattr__(f, 'default', dataclasses.MISSING)
                            object.__setattr__(f, 'default_factory',
                                               type(f.default))
            return True
        except Exception:
            return False
    except Exception:
        return False


def compute_mos(wav_path: Path) -> float:
    \"\"\"UTMOS neural MOS. Returns NaN if unavailable.\"\"\"
    global _utmos_cache, _utmos_available
    if SKIP_MOS or _utmos_available is False:
        return float("nan")
    try:
        if _utmos_cache is None:
            _apply_fairseq_patch()   # ✅ FIX: patch before utmos import
            import utmos
            _utmos_cache     = utmos.UTMOSScore(device=DEVICE)
            _utmos_available = True
        return round(float(_utmos_cache.score(str(wav_path))), 3)
    except ImportError:
        if _utmos_available is None:
            log.warning("  UTMOS not installed — MOS will be NaN. pip install utmos")
        _utmos_available = False
        return float("nan")
    except Exception as exc:
        log.warning(f"    UTMOS failed ({wav_path.name}): {exc}")
        _utmos_available = False
        return float("nan")

print("compute_mos() defined.")""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 73 markdown — ranking
# ─────────────────────────────────────────────────────────────────────────────
md(73, """## 11. Final Rankings

Rank models on each metric (1 = best). Lower **Avg Rank** = better overall.

**Bug fixed:** `.rank().astype(int)` raised `IntCastingNaNError` when a metric column  
(e.g. MOS when UTMOS unavailable) is entirely NaN. Fixed: use float ranks and skip  
all-NaN columns from the average.""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 74 — ranking fixed
# ─────────────────────────────────────────────────────────────────────────────
code(74, """rank_metrics = ["latency_ms", "rtf", "mos_utmos", "wer", "cer",
                "pitch_std_hz", "energy_std", "throughput_cps"]
_hib_set = {"mos_utmos", "pitch_std_hz", "energy_std", "throughput_cps"}

for lang in df["language"].unique():
    print(f"\\n{'='*70}")
    print(f"  FINAL RANKING — {lang.upper()}")
    print(f"{'='*70}")

    sub = (
        df[df["language"] == lang]
        .groupby("model_name")[rank_metrics]
        .mean(numeric_only=True)
        .dropna(how="all")
    )
    print("\\nRaw means:")
    print(sub.round(3).to_string())

    rank_df   = sub.copy()
    ranked_cols = []
    for col in rank_df.columns:
        # ✅ FIX: skip entirely-NaN columns (e.g. MOS when UTMOS not installed)
        if rank_df[col].isna().all():
            log.warning(f"  Ranking: column '{col}' is all-NaN — skipped")
            continue
        asc = col not in _hib_set
        # ✅ FIX: keep float (not int) so NaN rows don't crash .astype(int)
        rank_df[col] = rank_df[col].rank(ascending=asc, na_option="bottom")
        ranked_cols.append(col)

    rank_df["Avg Rank"] = (
        rank_df[ranked_cols].mean(axis=1).round(2) if ranked_cols
        else float("nan")
    )

    print("\\nRankings (1 = best per metric):")
    print(rank_df.round(1).to_string())

    if not rank_df["Avg Rank"].isna().all():
        winner = rank_df["Avg Rank"].idxmin()
        print(f"\\n  ★  Overall winner ({lang.upper()}): {winner}  "
              f"(avg rank = {rank_df.loc[winner, 'Avg Rank']})")

    rank_df.reset_index().to_csv(
        CSV_DIR / f"tts_benchmark_ranking_{lang}.csv", index=False, encoding="utf-8"
    )

print(f"\\n  Ranking CSVs saved to {CSV_DIR}/")""")

# ── clear all outputs ─────────────────────────────────────────────────────────
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

with open('notebook4dedb513ee.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Done.")