#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  TTS BENCHMARK — Hindi & English Open-Source Models
================================================================================
  Performance Metrics : Latency (ms), Real-Time Factor (RTF), Throughput (CPS)
  Quality Metrics     : MOS (UTMOS neural predictor), Intelligibility (WER/CER
                        via Whisper ASR transcription)
  Prosody Metrics     : Pitch mean / std / range, Speaking rate,
                        Energy dynamics, Pause ratio
  Linguistic Robustness: Per-category WER across 8 linguistic challenge types
  Output              : CSVs (full, summary, robustness) + publication-quality
                        PNG plots (bar, violin, heatmap, radar, scatter, grouped)

  ┌──────────────────────────────────────────────────────────────────┐
  │ Models benchmarked                                               │
  │   English → MMS-TTS-ENG · SpeechT5 · Parler-TTS-Mini · XTTS-v2│
  │   Hindi   → MMS-TTS-HIN · Coqui-VITS-HI · XTTS-v2             │
  └──────────────────────────────────────────────────────────────────┘

  Usage:
    python tts_benchmark.py                          # all models, EN+HI
    python tts_benchmark.py --languages en           # English only
    python tts_benchmark.py --models MMS-TTS SpeechT5
    python tts_benchmark.py --no-plots               # skip visualization

  Pipeline fit: drop-in replacement for ASR + MT benchmarks; use the CSV
  outputs to rank TTS legs for your Hindi↔English S2S pipeline.
================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Standard-library imports (always available)
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import json
import logging
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("tts_benchmark.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Heavy-import guard — fail early with friendly messages
# ─────────────────────────────────────────────────────────────────────────────
_MISSING: List[str] = []

def _require(module: str, pip: str = "") -> object:
    try:
        return __import__(module)
    except ImportError:
        pkg = pip or module
        _MISSING.append(pkg)
        return None

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import scipy.io.wavfile as _wavfile

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    sys.exit("PyTorch is required.  Install: pip install torch")

log.info(f"PyTorch {torch.__version__} | Device: {DEVICE.upper()}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Output directories
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("tts_benchmark_results")
AUDIO_DIR  = OUTPUT_DIR / "audio"
PLOT_DIR   = OUTPUT_DIR / "plots"
CSV_DIR    = OUTPUT_DIR / "csv"
for _d in [OUTPUT_DIR, AUDIO_DIR, PLOT_DIR, CSV_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Global benchmark configuration
# ─────────────────────────────────────────────────────────────────────────────
N_WARMUP_RUNS = 1   # discard: first call is always slow (model JIT / cache)
N_TIMED_RUNS  = 3   # average latency over this many runs

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Test corpus — 8 linguistic challenge categories per language
# ─────────────────────────────────────────────────────────────────────────────
EN_CORPUS: Dict[str, str] = {
    "short":          "Hello, how are you today?",
    "medium":         "The quick brown fox jumps over the lazy dog.",
    "long":           (
        "India is a remarkably diverse country with many languages, cultures, "
        "and traditions that have evolved over thousands of years of rich and "
        "complex history."
    ),
    "numbers":        (
        "Call me at 9876543210 on the 15th of August 2024 at half past three "
        "in the afternoon."
    ),
    "named_entities": (
        "Prime Minister Narendra Modi met President Biden in New Delhi to "
        "discuss bilateral relations and trade agreements."
    ),
    "technical":      (
        "The neural network operates at 3.5 gigahertz with 16 gigabytes of "
        "RAM and a 512-core GPU accelerator."
    ),
    "punctuation":    (
        "Wait — are you serious?  I can't believe it!  Well, that's... "
        "truly unexpected."
    ),
    "abbreviations":  (
        "Dr. Smith from MIT visited NASA's JPL facility in Los Angeles, CA, "
        "last Tuesday afternoon."
    ),
}

HI_CORPUS: Dict[str, str] = {
    "short":          "नमस्ते, आप कैसे हैं?",
    "medium":         "भारत एक विविधताओं से भरा हुआ देश है।",
    "long":           (
        "भारत एक ऐसा महान देश है जहाँ अनेक भाषाएँ, संस्कृतियाँ और "
        "परंपराएँ हजारों वर्षों से निरंतर विकसित होती आई हैं।"
    ),
    "numbers":        (
        "मुझे पाँच किलो चावल, तीन किलो दाल और दो लीटर सरसों का तेल "
        "चाहिए।"
    ),
    "named_entities": (
        "प्रधानमंत्री नरेंद्र मोदी ने नई दिल्ली में राष्ट्रपति भवन में "
        "एक महत्वपूर्ण बैठक आयोजित की।"
    ),
    "technical":      (
        "यह कंप्यूटर 3.5 गीगाहर्ट्ज़ की गति से कार्य करता है और इसमें "
        "सोलह गीगाबाइट की मेमोरी है।"
    ),
    "punctuation":    (
        "रुकिए — क्या आप सच कह रहे हैं?  मुझे बिल्कुल विश्वास नहीं होता!"
    ),
    "mixed_script":   (
        "मेरा फ़ोन नंबर है 9876543210 और ईमेल पता है example@gmail.com।"
    ),
}

CORPORA: Dict[str, Dict[str, str]] = {"en": EN_CORPUS, "hi": HI_CORPUS}

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Result data-class
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TTSResult:
    # Identity
    model_name      : str  = ""
    language        : str  = ""
    category        : str  = ""
    text            : str  = ""
    audio_path      : str  = ""
    # ── Performance ──────────────────────────────────────────────────────────
    latency_ms      : float = float("nan")   # wall-clock synthesis time (ms)
    rtf             : float = float("nan")   # synthesis_time / audio_duration
    throughput_cps  : float = float("nan")   # characters synthesised / second
    audio_duration_s: float = float("nan")   # length of generated audio (s)
    # ── Quality ──────────────────────────────────────────────────────────────
    mos_utmos       : float = float("nan")   # UTMOS MOS prediction [1-5]
    wer             : float = float("nan")   # Whisper WER (%) — lower better
    cer             : float = float("nan")   # Whisper CER (%) — lower better
    # ── Prosody ──────────────────────────────────────────────────────────────
    pitch_mean_hz   : float = float("nan")   # mean fundamental frequency (F0)
    pitch_std_hz    : float = float("nan")   # pitch std dev → naturalness
    pitch_range_hz  : float = float("nan")   # max F0 − min F0
    speaking_rate   : float = float("nan")   # onset events / second (≈ tempo)
    energy_std      : float = float("nan")   # RMS energy std → expressiveness
    pause_ratio     : float = float("nan")   # fraction of frames that are silent
    # ── Meta ─────────────────────────────────────────────────────────────────
    error           : str  = ""

# ─────────────────────────────────────────────────────────────────────────────
# 7.  TTS model wrappers
# ─────────────────────────────────────────────────────────────────────────────

class BaseTTS:
    """
    Abstract base for all TTS backends.
    Subclasses must implement: load(), synthesize(), unload().
    """
    name: str = "base"
    supported_langs: List[str] = []

    def load(self):   raise NotImplementedError
    def synthesize(self, text: str, lang: str, out_path: Path) -> float:
        """Synthesise `text` → WAV at `out_path`. Return audio duration (s)."""
        raise NotImplementedError
    def unload(self):
        torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────
class MMSTTSWrapper(BaseTTS):
    """
    facebook/mms-tts-eng  (English)
    facebook/mms-tts-hin  (Hindi)
    Massively-Multilingual Speech VITS model family from Meta.
    Loaded via HuggingFace `transformers` — no extra packages required.
    """
    name = "MMS-TTS"
    supported_langs = ["en", "hi"]

    _LANG_TO_HF_ID = {
        "en": "facebook/mms-tts-eng",
        "hi": "facebook/mms-tts-hin",
    }

    def __init__(self):
        self._models: Dict[str, Tuple] = {}

    def load(self):
        from transformers import VitsModel, VitsTokenizer
        import soundfile  # noqa — verify available before looping
        for lang, model_id in self._LANG_TO_HF_ID.items():
            log.info(f"  [{self.name}] Loading {model_id} …")
            tok   = VitsTokenizer.from_pretrained(model_id)
            model = VitsModel.from_pretrained(model_id).to(DEVICE)
            model.eval()
            self._models[lang] = (model, tok)
        log.info(f"  [{self.name}] Ready for {list(self._models.keys())}")

    def synthesize(self, text: str, lang: str, out_path: Path) -> float:
        import soundfile as sf
        model, tok = self._models[lang]
        inputs = tok(text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**inputs)
        wav = out.waveform.squeeze().cpu().numpy()
        sr  = model.config.sampling_rate
        sf.write(str(out_path), wav, sr)
        return len(wav) / sr

    def unload(self):
        del self._models
        super().unload()


# ──────────────────────────────────────────────────────────────────────────────
class SpeechT5Wrapper(BaseTTS):
    """
    microsoft/speecht5_tts  (English only)
    Transformer encoder–decoder TTS with HiFi-GAN neural vocoder.
    Speaker embedding taken from CMU-Arctic xvectors (speaker #7306 = SLT,
    female US English) — swap index to change voice.
    """
    name = "SpeechT5"
    supported_langs = ["en"]

    def load(self):
        from transformers import (
            SpeechT5ForTextToSpeech,
            SpeechT5HifiGan,
            SpeechT5Processor,
        )
        from datasets import load_dataset as _hf_ds
        log.info(f"  [{self.name}] Loading model + vocoder …")
        self.proc     = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model    = SpeechT5ForTextToSpeech.from_pretrained(
            "microsoft/speecht5_tts"
        ).to(DEVICE)
        self.vocoder  = SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5-hifigan"
        ).to(DEVICE)
        self.model.eval(); self.vocoder.eval()

        log.info(f"  [{self.name}] Loading speaker embedding …")
        emb_ds = _hf_ds("Matthijs/cmu-arctic-xvectors", split="validation")
        self.spk_emb = torch.tensor(
            emb_ds[7306]["xvector"]
        ).unsqueeze(0).to(DEVICE)
        log.info(f"  [{self.name}] Ready.")

    def synthesize(self, text: str, lang: str, out_path: Path) -> float:
        import soundfile as sf
        # SpeechT5 processes ≤600 tokens; long inputs are silently truncated
        inputs = self.proc(text=text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            speech = self.model.generate_speech(
                inputs["input_ids"], self.spk_emb, vocoder=self.vocoder
            )
        wav = speech.cpu().numpy()
        sr  = 16_000
        sf.write(str(out_path), wav, sr)
        return len(wav) / sr

    def unload(self):
        del self.model, self.proc, self.vocoder, self.spk_emb
        super().unload()


# ──────────────────────────────────────────────────────────────────────────────
class ParlerTTSWrapper(BaseTTS):
    """
    parler-tts/parler-tts-mini-v1  (English only)
    Decoder-only architecture conditioned on a natural-language voice
    description.  Produces highly natural speech with controllable style.
    Install: pip install parler-tts
    """
    name = "Parler-TTS"
    supported_langs = ["en"]

    # Natural-language voice description — edit to change voice style
    VOICE_DESC = (
        "A female speaker delivers a slightly expressive and animated speech "
        "with a moderate speed and pitch. The recording is of very high "
        "quality, with the speaker's voice sounding clear and very close up."
    )

    def load(self):
        try:
            from parler_tts import ParlerTTSForConditionalGeneration
        except ImportError:
            raise RuntimeError(
                "parler-tts not installed.\n  pip install parler-tts"
            )
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
        desc_tok   = self.tok(self.VOICE_DESC, return_tensors="pt").to(DEVICE)
        prompt_tok = self.tok(text,            return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen = self.model.generate(
                input_ids              = desc_tok.input_ids,
                attention_mask         = desc_tok.attention_mask,
                prompt_input_ids       = prompt_tok.input_ids,
                prompt_attention_mask  = prompt_tok.attention_mask,
            )
        wav = gen.cpu().numpy().squeeze()
        sr  = self.model.config.sampling_rate
        sf.write(str(out_path), wav, sr)
        return len(wav) / sr

    def unload(self):
        del self.model, self.tok
        super().unload()


# ──────────────────────────────────────────────────────────────────────────────
class CoquiVITSWrapper(BaseTTS):
    """
    Coqui TTS — language-specific VITS models.
      English : tts_models/en/ljspeech/vits
      Hindi   : tts_models/hi/cv/vits       (CommonVoice dataset)
    Install: pip install TTS
    """
    name = "Coqui-VITS"
    supported_langs = ["en", "hi"]

    _LANG_TO_MODEL = {
        "en": "tts_models/en/ljspeech/vits",
        "hi": "tts_models/hi/cv/vits",
    }

    def __init__(self):
        self._instances: Dict[str, object] = {}

    def load(self):
        try:
            from TTS.api import TTS as _CoquiTTS
        except ImportError:
            raise RuntimeError("Coqui TTS not installed.\n  pip install TTS")
        for lang, model_id in self._LANG_TO_MODEL.items():
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


# ──────────────────────────────────────────────────────────────────────────────
class XTTSv2Wrapper(BaseTTS):
    """
    Coqui XTTS v2 — zero-shot multilingual TTS.
    Uses a built-in studio speaker (no reference audio required).
    Supports Hindi and English natively.
    Install: pip install TTS
    """
    name = "XTTS-v2"
    supported_langs = ["en", "hi"]

    _LANG_CODES = {"en": "en", "hi": "hi"}
    _MODEL_ID   = "tts_models/multilingual/multi-dataset/xtts_v2"

    def load(self):
        try:
            from TTS.api import TTS as _CoquiTTS
        except ImportError:
            raise RuntimeError("Coqui TTS not installed.\n  pip install TTS")
        log.info(f"  [{self.name}] Loading XTTS v2 (multilingual) …")
        self.tts = _CoquiTTS(self._MODEL_ID, gpu=(DEVICE == "cuda"))
        # Pick the first available built-in speaker
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


# ──────────────────────────────────────────────────────────────────────────────
# Model registry (order = benchmark order)
# ──────────────────────────────────────────────────────────────────────────────
ALL_MODELS: List[BaseTTS] = [
    MMSTTSWrapper(),
    SpeechT5Wrapper(),
    ParlerTTSWrapper(),
    CoquiVITSWrapper(),
    XTTSv2Wrapper(),
]

MODEL_REGISTRY: Dict[str, BaseTTS] = {m.name: m for m in ALL_MODELS}

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Metric computation helpers
# ─────────────────────────────────────────────────────────────────────────────

# ── 8a. Prosody (librosa) ────────────────────────────────────────────────────
def compute_prosody(wav_path: Path) -> Dict[str, float]:
    """
    Extract prosodic features using librosa.
      pitch_mean_hz   — mean voiced F0
      pitch_std_hz    — pitch variation (higher → more natural)
      pitch_range_hz  — max−min F0 in voiced frames
      speaking_rate   — onset events / second  (proxy for tempo / syllable rate)
      energy_std      — RMS energy standard deviation (expressiveness proxy)
      pause_ratio     — fraction of frames where RMS < 0.01  (silence proportion)
    """
    nan_dict = {k: float("nan") for k in
                ["pitch_mean_hz", "pitch_std_hz", "pitch_range_hz",
                 "speaking_rate", "energy_std", "pause_ratio"]}
    try:
        import librosa
        y, sr = librosa.load(str(wav_path), sr=None, mono=True)

        # Pitch via probabilistic YIN (robust to noise)
        f0, voiced, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        f0_v = f0[voiced] if voiced is not None else f0[~np.isnan(f0)]
        if len(f0_v) == 0:
            f0_v = np.array([0.0])

        pitch_mean  = float(np.nanmean(f0_v))
        pitch_std   = float(np.nanstd(f0_v))
        pitch_range = float(np.nanmax(f0_v) - np.nanmin(f0_v))

        # Energy (RMS per frame)
        rms = librosa.feature.rms(y=y)[0]
        energy_std = float(np.std(rms))
        pause_ratio = float(np.mean(rms < 0.01))

        # Onset-density as speaking-rate proxy
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
        duration = librosa.get_duration(y=y, sr=sr)
        speaking_rate = float(len(onsets) / max(duration, 1e-9))

        return {
            "pitch_mean_hz" : round(pitch_mean,  2),
            "pitch_std_hz"  : round(pitch_std,   2),
            "pitch_range_hz": round(pitch_range,  2),
            "speaking_rate" : round(speaking_rate, 2),
            "energy_std"    : round(energy_std,   5),
            "pause_ratio"   : round(pause_ratio,  4),
        }
    except Exception as exc:
        log.warning(f"    Prosody extraction failed ({wav_path.name}): {exc}")
        return nan_dict


# ── 8b. Intelligibility — Whisper ASR + WER/CER ─────────────────────────────
_whisper_model_cache = None

def _get_whisper():
    global _whisper_model_cache
    if _whisper_model_cache is None:
        import whisper
        log.info("  [Whisper] Loading base model for intelligibility scoring …")
        _whisper_model_cache = whisper.load_model("base", device=DEVICE)
    return _whisper_model_cache


def compute_intelligibility(
    wav_path: Path, ref_text: str, lang: str
) -> Dict[str, float]:
    """
    Transcribe synthesised audio with Whisper and compute:
      wer  — Word Error Rate  (%) against original input text
      cer  — Character Error Rate (%) against original input text
    Lower is better for both.  A perfect TTS would read back the text
    exactly as it was input, giving WER≈0 and CER≈0.
    """
    nan_dict = {"wer": float("nan"), "cer": float("nan")}
    try:
        from jiwer import cer as jiwer_cer, wer as jiwer_wer
        wmodel = _get_whisper()
        lang_code = "hi" if lang == "hi" else "en"
        result = wmodel.transcribe(str(wav_path), language=lang_code)
        hyp = result["text"].strip()
        ref = ref_text.strip()
        # Clamp at 200% to avoid outliers from hallucinated long outputs
        wer_val = min(jiwer_wer(ref, hyp) * 100, 200.0)
        cer_val = min(jiwer_cer(ref, hyp) * 100, 200.0)
        return {"wer": round(wer_val, 2), "cer": round(cer_val, 2)}
    except ImportError as e:
        log.warning(f"    Intelligibility skipped (missing package: {e})")
        return nan_dict
    except Exception as exc:
        log.warning(f"    Intelligibility failed ({wav_path.name}): {exc}")
        return nan_dict


# ── 8c. MOS prediction — UTMOS ───────────────────────────────────────────────
_utmos_cache = None
_utmos_available: Optional[bool] = None

def compute_mos(wav_path: Path) -> float:
    """
    UTMOS neural MOS predictor (Sarulab, 2022).
    Returns a score in [1, 5] — higher is better.
    Returns NaN gracefully if UTMOS is not installed.
    Install: pip install utmos
    """
    global _utmos_cache, _utmos_available
    if _utmos_available is False:
        return float("nan")
    try:
        if _utmos_cache is None:
            import utmos
            _utmos_cache    = utmos.UTMOSScore(device=DEVICE)
            _utmos_available = True
        score = _utmos_cache.score(str(wav_path))
        return round(float(score), 3)
    except ImportError:
        if _utmos_available is None:
            log.warning(
                "  UTMOS not installed — MOS column will be NaN.\n"
                "  Install with: pip install utmos"
            )
        _utmos_available = False
        return float("nan")
    except Exception as exc:
        log.warning(f"    UTMOS failed ({wav_path.name}): {exc}")
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Core benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

def _timed_synthesis(
    model: BaseTTS,
    text: str,
    lang: str,
    out_path: Path,
) -> Tuple[float, float]:
    """
    Run N_WARMUP_RUNS warm-up iterations then N_TIMED_RUNS measured iterations.
    Returns (mean_wall_time_s, audio_duration_s).
    """
    for _ in range(N_WARMUP_RUNS):
        model.synthesize(text, lang, out_path)

    times, dur = [], 0.0
    for _ in range(N_TIMED_RUNS):
        t0  = time.perf_counter()
        dur = model.synthesize(text, lang, out_path)
        times.append(time.perf_counter() - t0)

    return float(np.mean(times)), dur


def benchmark_one_model(model: BaseTTS, languages: List[str]) -> List[TTSResult]:
    results: List[TTSResult] = []

    for lang in languages:
        if lang not in model.supported_langs:
            log.info(f"  [{model.name}] Language '{lang}' not supported — skipping.")
            continue

        corpus = CORPORA[lang]
        log.info(f"  [{model.name}][{lang.upper()}] Starting {len(corpus)} sentences …")

        for category, text in corpus.items():
            res = TTSResult(
                model_name = model.name,
                language   = lang,
                category   = category,
                text       = text,
            )
            safe_name = f"{model.name}_{lang}_{category}.wav".replace("/", "-")
            out_path  = AUDIO_DIR / safe_name

            try:
                elapsed, audio_dur = _timed_synthesis(model, text, lang, out_path)

                res.audio_path       = str(out_path)
                res.audio_duration_s = round(audio_dur, 3)
                res.latency_ms       = round(elapsed * 1_000, 2)
                res.rtf              = round(elapsed / max(audio_dur, 1e-9), 4)
                res.throughput_cps   = round(len(text) / max(elapsed, 1e-9), 2)

                # Quality
                res.mos_utmos = compute_mos(out_path)
                intel          = compute_intelligibility(out_path, text, lang)
                res.wer        = intel["wer"]
                res.cer        = intel["cer"]

                # Prosody
                prosody            = compute_prosody(out_path)
                res.pitch_mean_hz  = prosody["pitch_mean_hz"]
                res.pitch_std_hz   = prosody["pitch_std_hz"]
                res.pitch_range_hz = prosody["pitch_range_hz"]
                res.speaking_rate  = prosody["speaking_rate"]
                res.energy_std     = prosody["energy_std"]
                res.pause_ratio    = prosody["pause_ratio"]

                log.info(
                    f"    [{category:16s}] latency={res.latency_ms:7.1f}ms  "
                    f"RTF={res.rtf:.3f}  WER={res.wer:.1f}%  "
                    f"MOS={res.mos_utmos:.2f}"
                )

            except Exception as exc:
                log.error(f"    [{category}] FAILED: {exc}")
                res.error = str(exc)

            results.append(res)

    return results


def run_benchmark(
    model_names: Optional[List[str]] = None,
    languages: Optional[List[str]]   = None,
) -> pd.DataFrame:
    languages = languages or ["en", "hi"]

    models = (
        [MODEL_REGISTRY[n] for n in model_names if n in MODEL_REGISTRY]
        if model_names else ALL_MODELS
    )
    if not models:
        log.error("No valid models selected.")
        return pd.DataFrame()

    all_results: List[TTSResult] = []

    for model in models:
        sep = "=" * 66
        log.info(f"\n{sep}")
        log.info(f"  Benchmarking:  {model.name}")
        log.info(sep)
        try:
            model.load()
            results = benchmark_one_model(model, languages)
            all_results.extend(results)
        except Exception as exc:
            log.error(f"  [{model.name}] Fatal error: {exc}")
        finally:
            try:
                model.unload()
            except Exception:
                pass

    if not all_results:
        log.warning("No results collected.")
        return pd.DataFrame()

    df = pd.DataFrame([asdict(r) for r in all_results])

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    full_csv = CSV_DIR / "tts_benchmark_full.csv"
    df.to_csv(full_csv, index=False, encoding="utf-8")
    log.info(f"\nFull results  → {full_csv}")

    _numeric_cols = [
        "latency_ms", "rtf", "throughput_cps", "audio_duration_s",
        "mos_utmos", "wer", "cer",
        "pitch_mean_hz", "pitch_std_hz", "pitch_range_hz",
        "speaking_rate", "energy_std", "pause_ratio",
    ]
    summary_df = (
        df.groupby(["model_name", "language"])[_numeric_cols]
        .mean(numeric_only=True)
        .round(3)
        .reset_index()
    )
    summary_csv = CSV_DIR / "tts_benchmark_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
    log.info(f"Summary       → {summary_csv}")

    robust_csv = CSV_DIR / "tts_benchmark_robustness.csv"
    df[["model_name", "language", "category", "wer", "cer"]].dropna(
        subset=["wer"]
    ).to_csv(robust_csv, index=False, encoding="utf-8")
    log.info(f"Robustness    → {robust_csv}")

    per_model_csv = CSV_DIR / "tts_benchmark_per_model.csv"
    df.groupby(["model_name", "category"])[_numeric_cols].mean(
        numeric_only=True
    ).round(3).reset_index().to_csv(per_model_csv, index=False, encoding="utf-8")
    log.info(f"Per-model     → {per_model_csv}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 10. Visualization helpers
# ─────────────────────────────────────────────────────────────────────────────

_PALETTE   = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860",
              "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
_MODEL_CLR: Dict[str, str] = {}


def _setup_style(models: List[str]):
    global _MODEL_CLR
    _MODEL_CLR = {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(models)}
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor"  : "#f8f9fa",
        "axes.grid"       : True,
        "grid.alpha"      : 0.35,
        "grid.color"      : "#cccccc",
        "font.family"     : "DejaVu Sans",
        "axes.spines.top" : False,
        "axes.spines.right": False,
    })


def _save(fig: plt.Figure, name: str):
    path = PLOT_DIR / f"{name}.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved → {path}")


def _annotate_bars(ax: plt.Axes, bars, vals: List[float], fmt: str = ".2f"):
    for bar, v in zip(bars, vals):
        if not (isinstance(v, float) and np.isnan(v)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{v:{fmt}}",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
            )


# ── 10a. Performance bar charts ───────────────────────────────────────────────
def plot_performance(df: pd.DataFrame):
    log.info("[Plot] Performance metrics …")
    metrics = [
        ("latency_ms",    "Latency (ms)\n↓ lower is better"),
        ("rtf",           "Real-Time Factor\n↓ lower is better  (<1 = faster than real-time)"),
        ("throughput_cps","Throughput (chars/sec)\n↑ higher is better"),
    ]
    for lang in df["language"].unique():
        sub = (
            df[df["language"] == lang]
            .groupby("model_name")[["latency_ms", "rtf", "throughput_cps"]]
            .mean(numeric_only=True)
            .reset_index()
        )
        models = sub["model_name"].tolist()
        colors = [_MODEL_CLR.get(m, "#888") for m in models]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f"Performance Metrics — {lang.upper()}",
            fontsize=14, fontweight="bold",
        )
        for ax, (col, title) in zip(axes, metrics):
            vals = sub[col].tolist()
            bars = ax.bar(range(len(models)), vals, color=colors,
                          edgecolor="white", width=0.6)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=22, ha="right", fontsize=9)
            _annotate_bars(ax, bars, vals)
        plt.tight_layout()
        _save(fig, f"01_performance_{lang}")


# ── 10b. Quality bar charts ───────────────────────────────────────────────────
def plot_quality(df: pd.DataFrame):
    log.info("[Plot] Quality metrics …")
    metrics = [
        ("mos_utmos", "MOS (UTMOS)\n↑ higher is better  [1–5 scale]"),
        ("wer",       "Word Error Rate (%)\n↓ lower is better"),
        ("cer",       "Character Error Rate (%)\n↓ lower is better"),
    ]
    for lang in df["language"].unique():
        sub = (
            df[df["language"] == lang]
            .groupby("model_name")[["mos_utmos", "wer", "cer"]]
            .mean(numeric_only=True)
            .reset_index()
        )
        models = sub["model_name"].tolist()
        colors = [_MODEL_CLR.get(m, "#888") for m in models]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f"Quality Metrics — {lang.upper()}",
            fontsize=14, fontweight="bold",
        )
        for ax, (col, title) in zip(axes, metrics):
            vals = sub[col].tolist()
            bars = ax.bar(range(len(models)), vals, color=colors,
                          edgecolor="white", width=0.6)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=22, ha="right", fontsize=9)
            _annotate_bars(ax, bars, vals)
        plt.tight_layout()
        _save(fig, f"02_quality_{lang}")


# ── 10c. Prosody bar charts ───────────────────────────────────────────────────
def plot_prosody(df: pd.DataFrame):
    log.info("[Plot] Prosody metrics …")
    metrics = [
        ("pitch_mean_hz",  "Mean Pitch (Hz)\nFundamental frequency of voice"),
        ("pitch_std_hz",   "Pitch Std Dev (Hz)\n↑ higher = more varied / natural"),
        ("pitch_range_hz", "Pitch Range (Hz)\nMax−min F0 in voiced frames"),
        ("speaking_rate",  "Speaking Rate (onsets/s)\nProxy for tempo"),
        ("energy_std",     "Energy Dynamics (RMS std)\n↑ higher = more expressive"),
        ("pause_ratio",    "Pause Ratio\nFraction of silent frames"),
    ]
    for lang in df["language"].unique():
        cols = [c for c, _ in metrics]
        sub = (
            df[df["language"] == lang]
            .groupby("model_name")[cols]
            .mean(numeric_only=True)
            .reset_index()
        )
        models = sub["model_name"].tolist()
        colors = [_MODEL_CLR.get(m, "#888") for m in models]
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            f"Prosody Metrics — {lang.upper()}",
            fontsize=14, fontweight="bold",
        )
        for ax, (col, title) in zip(axes.flat, metrics):
            vals = sub[col].tolist()
            bars = ax.bar(range(len(models)), vals, color=colors,
                          edgecolor="white", width=0.6)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=22, ha="right", fontsize=9)
            _annotate_bars(ax, bars, vals)
        plt.tight_layout()
        _save(fig, f"03_prosody_{lang}")


# ── 10d. Linguistic robustness heatmap ───────────────────────────────────────
def plot_robustness_heatmap(df: pd.DataFrame):
    log.info("[Plot] Robustness heatmap …")
    for lang in df["language"].unique():
        sub = df[(df["language"] == lang)].dropna(subset=["wer"])
        if sub.empty:
            continue
        pivot = sub.pivot_table(
            index="model_name", columns="category", values="wer", aggfunc="mean"
        )
        fig, ax = plt.subplots(
            figsize=(max(10, len(pivot.columns) * 1.6), len(pivot) + 2)
        )
        sns.heatmap(
            pivot, annot=True, fmt=".1f", cmap="RdYlGn_r",
            linewidths=0.5, ax=ax,
            cbar_kws={"label": "WER (%) — green=lower=better"},
        )
        ax.set_title(
            f"Linguistic Robustness — WER (%) per Category\nLanguage: {lang.upper()}",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlabel("Sentence Category", fontsize=10)
        ax.set_ylabel("Model",             fontsize=10)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        _save(fig, f"04_robustness_heatmap_{lang}")


# ── 10e. WER grouped bar by category ─────────────────────────────────────────
def plot_wer_by_category(df: pd.DataFrame):
    log.info("[Plot] WER by category …")
    for lang in df["language"].unique():
        sub = df[(df["language"] == lang)].dropna(subset=["wer"])
        if sub.empty:
            continue
        categories = sorted(sub["category"].unique())
        models     = sub["model_name"].unique()
        x          = np.arange(len(categories))
        n          = len(models)
        width      = 0.8 / n

        fig, ax = plt.subplots(figsize=(max(12, len(categories) * 2.2), 6))
        for i, model in enumerate(models):
            vals   = [sub[(sub["model_name"] == model) &
                          (sub["category"]   == cat)]["wer"].mean()
                      for cat in categories]
            offset = (i - n / 2 + 0.5) * width
            bars   = ax.bar(
                x + offset, vals, width=width * 0.9,
                label=model, color=_MODEL_CLR.get(model, "#888"),
                edgecolor="white",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("WER (%) — lower is better", fontsize=10)
        ax.set_title(
            f"Word Error Rate by Linguistic Category — {lang.upper()}",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=9)
        plt.tight_layout()
        _save(fig, f"05_wer_by_category_{lang}")


# ── 10f. RTF vs MOS scatter (speed–quality trade-off) ────────────────────────
def plot_speed_vs_quality(df: pd.DataFrame):
    log.info("[Plot] Speed vs Quality scatter …")
    langs = df["language"].unique()
    fig, axes = plt.subplots(1, len(langs), figsize=(8 * len(langs), 6))
    if len(langs) == 1:
        axes = [axes]

    for ax, lang in zip(axes, langs):
        sub = (
            df[df["language"] == lang]
            .groupby("model_name")[["rtf", "mos_utmos", "wer"]]
            .mean(numeric_only=True)
            .reset_index()
        )
        for _, row in sub.iterrows():
            c = _MODEL_CLR.get(row["model_name"], "#888")
            ax.scatter(row["rtf"], row["mos_utmos"], s=220, color=c, zorder=5)
            ax.annotate(
                row["model_name"], (row["rtf"], row["mos_utmos"]),
                textcoords="offset points", xytext=(8, 5), fontsize=9,
            )
        ax.axhline(3.5, color="gray",  ls="--", alpha=0.5, lw=1,
                   label="MOS=3.5 (good quality)")
        ax.axvline(1.0, color="red",   ls="--", alpha=0.5, lw=1,
                   label="RTF=1.0 (real-time boundary)")
        ax.set_xlabel("Real-Time Factor (RTF) — ↓ faster", fontsize=10)
        ax.set_ylabel("MOS (UTMOS) — ↑ better",            fontsize=10)
        ax.set_title(
            f"Speed–Quality Trade-off — {lang.upper()}",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, "06_speed_vs_quality_scatter")


# ── 10g. Latency violin / distribution ───────────────────────────────────────
def plot_latency_violin(df: pd.DataFrame):
    log.info("[Plot] Latency distribution …")
    for lang in df["language"].unique():
        sub = df[(df["language"] == lang) & df["latency_ms"].notna()]
        if sub.empty:
            continue
        models = sub["model_name"].unique()
        data   = [sub[sub["model_name"] == m]["latency_ms"].values for m in models]
        positions = np.arange(len(models))

        fig, ax = plt.subplots(figsize=(12, 6))
        parts = ax.violinplot(
            data, positions=positions,
            showmeans=True, showmedians=True, showextrema=True,
        )
        for pc, model in zip(parts["bodies"], models):
            pc.set_facecolor(_MODEL_CLR.get(model, "#888"))
            pc.set_alpha(0.7)

        ax.set_xticks(positions)
        ax.set_xticklabels(models, fontsize=9)
        ax.set_xlabel("Model",        fontsize=10)
        ax.set_ylabel("Latency (ms)", fontsize=10)
        ax.set_title(
            f"Latency Distribution across Sentence Types — {lang.upper()}\n"
            "(violin = density; line = median; dot = mean)",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()
        _save(fig, f"07_latency_violin_{lang}")


# ── 10h. Radar chart ─────────────────────────────────────────────────────────
def plot_radar(df: pd.DataFrame):
    log.info("[Plot] Radar charts …")

    # Axes: higher normalised score = always better on the radar
    radar_cfg = [
        ("mos_utmos",    "MOS↑",          True),
        ("wer",          "WER↓",          False),
        ("rtf",          "RTF↓",          False),
        ("pitch_std_hz", "Pitch Var.↑",   True),
        ("throughput_cps","Throughput↑",  True),
        ("energy_std",   "Energy Dyn.↑",  True),
    ]
    cols, labels, hib = zip(*radar_cfg)
    n = len(cols)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    for lang in df["language"].unique():
        sub = (
            df[df["language"] == lang]
            .groupby("model_name")[list(cols)]
            .mean(numeric_only=True)
            .reset_index()
            .dropna()
        )
        if sub.empty:
            continue

        # Normalise each metric to [0, 1]
        norm = sub[list(cols)].copy()
        for col, higher in zip(cols, hib):
            rng = norm[col].max() - norm[col].min()
            norm[col] = ((norm[col] - norm[col].min()) / rng) if rng > 0 else 0.5
            if not higher:
                norm[col] = 1 - norm[col]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

        for (_, row_orig), (_, row_norm) in zip(sub.iterrows(), norm.iterrows()):
            model = row_orig["model_name"]
            vals  = row_norm[list(cols)].tolist() + [row_norm[cols[0]]]
            clr   = _MODEL_CLR.get(model, "#888")
            ax.plot(angles, vals, "o-", linewidth=2, label=model, color=clr)
            ax.fill(angles, vals, alpha=0.07, color=clr)

        ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title(
            f"Model Profile Radar — {lang.upper()}\n"
            "(all axes normalised; outer rim = best)",
            fontsize=12, fontweight="bold", pad=28,
        )
        ax.legend(
            loc="upper right", bbox_to_anchor=(1.4, 1.2), fontsize=9
        )
        plt.tight_layout()
        _save(fig, f"08_radar_{lang}")


# ── 10i. Comprehensive comparison heatmap ────────────────────────────────────
def plot_comprehensive_heatmap(df: pd.DataFrame):
    log.info("[Plot] Comprehensive comparison heatmap …")

    metric_cfg = [
        ("latency_ms",    "Latency (ms)↓",  False),
        ("rtf",           "RTF↓",           False),
        ("throughput_cps","Throughput↑",    True),
        ("mos_utmos",     "MOS↑",           True),
        ("wer",           "WER%↓",          False),
        ("cer",           "CER%↓",          False),
        ("pitch_std_hz",  "Pitch Var.↑",    True),
        ("speaking_rate", "Speak.Rate",     True),
        ("energy_std",    "Energy Dyn.↑",   True),
        ("pause_ratio",   "Pause Ratio↓",   False),
    ]

    for lang in df["language"].unique():
        cols = [c for c, _, _ in metric_cfg]
        sub  = (
            df[df["language"] == lang]
            .groupby("model_name")[cols]
            .mean(numeric_only=True)
            .reset_index()
            .set_index("model_name")
        )
        rename_map = {c: lbl for c, lbl, _ in metric_cfg}
        hib_map    = {lbl: hib for _, lbl, hib in metric_cfg}
        sub.rename(columns=rename_map, inplace=True)

        norm = sub.copy()
        for col in norm.columns:
            rng = norm[col].max() - norm[col].min()
            norm[col] = ((norm[col] - norm[col].min()) / rng) if rng > 0 else 0.5
            if not hib_map.get(col, True):
                norm[col] = 1 - norm[col]

        fig, ax = plt.subplots(
            figsize=(len(sub.columns) * 1.4 + 2, len(sub) + 2)
        )
        sns.heatmap(
            norm, annot=sub.round(2), fmt="g",
            cmap="RdYlGn", linewidths=0.5, ax=ax,
            cbar_kws={"label": "Normalised score (green = better)"},
            vmin=0, vmax=1,
        )
        ax.set_title(
            f"Comprehensive Model Comparison — {lang.upper()}\n"
            "(Cell colour = normalised rank  |  Cell number = raw value)",
            fontsize=12, fontweight="bold",
        )
        ax.set_ylabel("Model", fontsize=10)
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        _save(fig, f"09_comprehensive_heatmap_{lang}")


# ── 10j. Audio duration vs latency scatter ────────────────────────────────────
def plot_duration_vs_latency(df: pd.DataFrame):
    log.info("[Plot] Duration vs Latency scatter …")
    for lang in df["language"].unique():
        sub = df[(df["language"] == lang) &
                 df["audio_duration_s"].notna() &
                 df["latency_ms"].notna()]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 6))
        for model, grp in sub.groupby("model_name"):
            ax.scatter(
                grp["audio_duration_s"], grp["latency_ms"],
                label=model, color=_MODEL_CLR.get(model, "#888"),
                s=70, alpha=0.8,
            )
        # Real-time boundary: latency = 1000 × duration (RTF=1)
        max_dur = sub["audio_duration_s"].max()
        ax.plot([0, max_dur], [0, max_dur * 1_000],
                "r--", lw=1.5, alpha=0.6, label="RTF = 1.0 (real-time)")
        ax.set_xlabel("Audio Duration (s)",  fontsize=10)
        ax.set_ylabel("Synthesis Latency (ms)", fontsize=10)
        ax.set_title(
            f"Audio Duration vs Synthesis Latency — {lang.upper()}\n"
            "Points below the red line are synthesised faster than real-time",
            fontsize=11, fontweight="bold",
        )
        ax.legend(fontsize=9)
        plt.tight_layout()
        _save(fig, f"10_duration_vs_latency_{lang}")


# ── 10k. Pitch distribution box plots ────────────────────────────────────────
def plot_pitch_boxplot(df: pd.DataFrame):
    log.info("[Plot] Pitch distribution box plots …")
    for lang in df["language"].unique():
        sub = df[(df["language"] == lang) & df["pitch_mean_hz"].notna()]
        if sub.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            f"Pitch Distribution — {lang.upper()}",
            fontsize=13, fontweight="bold",
        )
        for ax, (col, title) in zip(axes, [
            ("pitch_mean_hz",  "Mean Pitch (Hz)"),
            ("pitch_std_hz",   "Pitch Std Dev (Hz) — naturalness"),
        ]):
            data   = [sub[sub["model_name"] == m][col].values
                      for m in sub["model_name"].unique()]
            models = list(sub["model_name"].unique())
            bps = ax.boxplot(data, patch_artist=True, labels=models)
            for patch, model in zip(bps["boxes"], models):
                patch.set_facecolor(_MODEL_CLR.get(model, "#888"))
                patch.set_alpha(0.75)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xticklabels(models, rotation=22, ha="right", fontsize=9)
        plt.tight_layout()
        _save(fig, f"11_pitch_boxplot_{lang}")


# ── Master plot runner ────────────────────────────────────────────────────────
def generate_all_plots(df: pd.DataFrame):
    models = df["model_name"].unique().tolist()
    _setup_style(models)

    plot_performance(df)
    plot_quality(df)
    plot_prosody(df)
    plot_robustness_heatmap(df)
    plot_wer_by_category(df)
    plot_speed_vs_quality(df)
    plot_latency_violin(df)
    plot_radar(df)
    plot_comprehensive_heatmap(df)
    plot_duration_vs_latency(df)
    plot_pitch_boxplot(df)

    log.info(f"\nAll plots saved → {PLOT_DIR}/")


# ─────────────────────────────────────────────────────────────────────────────
# 11.  Console summary
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame):
    summary_cols = ["latency_ms", "rtf", "mos_utmos", "wer", "cer",
                    "pitch_std_hz", "energy_std"]
    for lang in df["language"].unique():
        print(f"\n{'='*75}")
        print(f"  BENCHMARK SUMMARY — {lang.upper()}")
        print(f"{'='*75}")
        sub = (
            df[df["language"] == lang]
            .groupby("model_name")[summary_cols]
            .mean(numeric_only=True)
            .round(3)
        )
        print(sub.to_string())

    print(f"\n  ╔══════════════════════════════════════════════════╗")
    print(f"  ║  Metric key: latency↓ rtf↓ mos↑ wer↓ cer↓      ║")
    print(f"  ║  pitch_std↑ energy_std↑ (higher = more natural) ║")
    print(f"  ╠══════════════════════════════════════════════════╣")
    print(f"  ║  CSVs  → {CSV_DIR}")
    print(f"  ║  Plots → {PLOT_DIR}")
    print(f"  ╚══════════════════════════════════════════════════╝")


# ─────────────────────────────────────────────────────────────────────────────
# 12.  Entry-point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="TTS Benchmark — Hindi & English open-source models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        metavar="NAME",
        help=(
            "Model names to benchmark (default: all).  "
            "Choices: " + ", ".join(MODEL_REGISTRY.keys())
        ),
    )
    parser.add_argument(
        "--languages", nargs="+", default=["en", "hi"],
        choices=["en", "hi"],
        help="Languages to evaluate (default: en hi)",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip all visualizations (useful for headless servers)",
    )
    parser.add_argument(
        "--no-mos", action="store_true",
        help="Skip UTMOS MOS prediction (saves time if UTMOS not installed)",
    )
    parser.add_argument(
        "--no-whisper", action="store_true",
        help="Skip Whisper intelligibility scoring (WER/CER will be NaN)",
    )
    args = parser.parse_args()

    # Monkey-patch optional metrics if user opts out
    if args.no_mos:
        global compute_mos
        compute_mos = lambda p: float("nan")  # noqa
        log.info("MOS scoring disabled via --no-mos")

    if args.no_whisper:
        global compute_intelligibility
        compute_intelligibility = lambda p, t, l: {"wer": float("nan"), "cer": float("nan")}  # noqa
        log.info("Whisper intelligibility disabled via --no-whisper")

    log.info("=" * 70)
    log.info("  TTS BENCHMARK — Hindi & English")
    log.info(f"  Device    : {DEVICE.upper()}")
    log.info(f"  Models    : {args.models or list(MODEL_REGISTRY.keys())}")
    log.info(f"  Languages : {args.languages}")
    log.info(f"  Output    : {OUTPUT_DIR}/")
    log.info("=" * 70)

    df = run_benchmark(model_names=args.models, languages=args.languages)

    if df.empty:
        log.error("No results to report.")
        sys.exit(1)

    if not args.no_plots:
        try:
            generate_all_plots(df)
        except Exception as exc:
            log.error(f"Plotting failed: {exc}")

    print_summary(df)
    log.info("\nBenchmark complete!")


if __name__ == "__main__":
    main()
