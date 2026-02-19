#!/usr/bin/env python

from . import _py_qwen3_asr_cpp as native
import importlib.metadata
import logging
import shutil
import sys
from pathlib import Path
from time import time
from collections.abc import Callable
from typing import Any
import numpy as np
import py_qwen3_asr_cpp.utils as utils
import py_qwen3_asr_cpp.constants as constants
import subprocess
import os
import tempfile
import wave
from py_qwen3_asr_cpp.constants import QWEN_SAMPLE_RATE
from dataclasses import dataclass


__author__ = "femelo"
__copyright__ = "Copyright 2026, "
__license__ = "Apache 2.0"
__version__ = importlib.metadata.version("py_qwen3_asr_cpp")

logger = logging.getLogger(__name__)


@dataclass
class Parameters:
    language: str | None = None
    max_tokens: int = 1024
    n_threads: int = 4
    print_timing: bool = False
    print_progress: bool = False


class Qwen3ASRModel:
    """
    A high-level wrapper for the Qwen3 ASR and Forced Aligner engine.
    Handles transcription, audio alignment, and model management.
    """

    def __init__(
        self,
        asr_model: str,
        align_model: str | None = None,
        models_dir: str | None = None,
        language: str | None = None,
        max_tokens: int = 1024,
        n_threads: int = 4,
        print_timing: bool = False,
        print_progress: bool = False,
    ) -> None:
        if Path(asr_model).is_file():
            self.asr_model_path = asr_model
        else:
            self.asr_model_path = utils.download_model(
                asr_model,
                models_dir,
                model_type=utils.ModelType.ASR,
            )

        if align_model:
            if Path(asr_model).is_file():
                self.align_model_path = align_model
            else:
                self.align_model_path = utils.download_model(
                    align_model,
                    models_dir,
                    model_type=utils.ModelType.ALIGNER,
                )
        self.asr = native.Qwen3ASR()
        self.aligner = native.ForcedAligner()
        self._models_dir: str | None = models_dir
        self._language: str | None = language
        self._params: Parameters = Parameters(
            language=language,
            max_tokens=max_tokens,
            n_threads=n_threads,
            print_timing=print_timing,
            print_progress=print_progress,
        )
        self._language: str | None = language
        self._progress_callback: Callable[[int, int], None] | None = None

        # Load models
        self.load_asr_model()
        if align_model:
            self.load_align_model()

    # --- Model Loading ---

    def load_asr_model(self, model_path: str | None = None) -> bool:
        """Loads the GGUF model for transcription."""
        model_path = model_path or self.asr_model_path
        if not Path(model_path).exists():
            raise ValueError(f"Model file could not be found: {model_path}")
        return self.asr.load_model(model_path)

    def load_align_model(self, model_path: str | None = None) -> bool:
        """Loads the GGUF model for forced alignment."""
        model_path = model_path or self.align_model_path
        if not Path(model_path).exists():
            raise ValueError(f"Model file could not be found: {model_path}")
        return self.aligner.load_model(model_path)

    def load_korean_dictionary(self, dict_path: str) -> bool:
        """Loads the dictionary required for Korean word splitting."""
        if Path(dict_path).is_file():
            raise ValueError(f"Model file could not be found: {dict_path}")
        return self.aligner.load_korean_dict(dict_path)

    @staticmethod
    def load_audio(media_file_path: str) -> np.ndarray:
        """
         Helper method to return a `np.array` object from a media file
         If the media file is not a WAV file, it will try to convert it using ffmpeg

        :param media_file_path: Path of the media file
        :return: Numpy array
        """
        convert: bool = False
        if media_file_path.endswith(".wav"):
            samples, sample_rate = native.load_audio_file(media_file_path)
            if sample_rate != QWEN_SAMPLE_RATE:
                convert = True
        else:
            if shutil.which("ffmpeg") is None:
                raise Exception(
                    "FFMPEG is not installed or not in PATH. Please install it, or provide a WAV file or a NumPy array instead!"
                )
            convert = True

        if convert:
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_file_path = temp_file.name
            temp_file.close()
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-i",
                        media_file_path,
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        temp_file_path,
                        "-y",
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                samples, _ = native.load_audio_file(temp_file_path)
            finally:
                os.remove(temp_file_path)
        return samples

    # --- Transcription (ASR) ---

    def transcribe(
        self,
        audio: str | np.ndarray,
        language: str = "",
        max_tokens: int = 1024,
        n_threads: int = 4,
    ) -> native.TranscribeResult:
        """
        Transcribes audio from a file path or a NumPy array.
        """
        params = native.TranscribeParams()
        params.language = language or self._params.language or ""
        params.max_tokens = max_tokens or self._params.max_tokens
        params.n_threads = n_threads or self._params.n_threads
        params.print_timing = self._params.print_timing
        params.print_progress = self._params.print_progress

        if isinstance(audio, str):
            samples = Qwen3ASRModel.load_audio(audio)
        elif isinstance(audio, np.ndarray):
            # Ensure float32 for the C++ backend
            samples = audio.astype(np.float32)
        else:
            raise ValueError("Audio must be a file path (str) or NumPy array.")

        result = self.asr.transcribe_samples(samples, params)
        self._language = result.language  # detected language from ASR result
        return result

    # --- Forced Alignment ---

    def align(
        self,
        audio: str | np.ndarray,
        text: str,
        language: str = "",
        print_timing: bool = True,
    ) -> native.AlignmentResult:
        """
        Aligns a known text transcript to audio to get word-level timestamps.
        """
        if isinstance(audio, str):
            result = self.aligner.align_file(audio, text, language)
        elif isinstance(audio, np.ndarray):
            samples = audio.astype(np.float32)
            result = self.aligner.align_samples(samples, text, language)
        else:
            raise ValueError("Audio must be a file path (str) or NumPy array.")
        if print_timing and self._params.print_timing:
            print("\nTiming:")
            print("  Mel spectrogram: %d ms", result.t_mel_ms)
            print("  Audio encoding:  %d ms", result.t_encode_ms)
            print("  Text decoding:   %d ms", result.t_decode_ms)
            print("  Total:           %d ms", result.t_total_ms)
            print("  Words aligned:   %d", len(result.words))
        return result

    def transcribe_and_align(
        self,
        audio: str | np.ndarray,
        language: str = "",
        max_tokens: int = 1024,
        n_threads: int = 4,
    ) -> tuple[native.TranscribeResult, native.AlignmentResult]:
        asr_result = self.transcribe(
            audio,
            language=language,
            max_tokens=max_tokens,
            n_threads=n_threads,
        )
        align_result = self.align(
            audio,
            asr_result.text,
            language=language,
            print_timing=False,
        )
        if self._params.print_timing:
            print("\nCombined Timing:")
            print("  ASR:           %d ms", asr_result.t_total_ms)
            print("  Alignment:     %d ms", align_result.t_total_ms)
            print(
                "  Total:         %d ms",
                (asr_result.t_total_ms + align_result.t_total_ms),
            )
            print("  Words aligned: %d", len(align_result.words))
        return asr_result, align_result

    # --- Utilities & Callbacks ---

    def get_params(self) -> dict[str, Any]:
        """Returns the current model parameters."""
        return self._params.__dict__

    def set_progress_callback(self, callback: Callable[[int, int], None]):
        """Sets a function to be called during transcription progress."""
        self._progress_callback = callback
        self.asr.set_progress_callback(callback)

    @property
    def last_asr_error(self) -> str:
        return self.asr.get_error()

    @property
    def last_aligner_error(self) -> str:
        return self.aligner.get_error()

    @property
    def detected_language(self) -> str | None:
        return self._language

    def is_ready(self) -> bool:
        """Checks if the ASR model is loaded and ready."""
        return self.asr.is_loaded()
