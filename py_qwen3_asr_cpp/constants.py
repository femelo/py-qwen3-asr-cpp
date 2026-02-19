#!/usr/bin/env python

from pathlib import Path
from platformdirs import user_data_dir


# Constants matching Qwen3-ASR / Whisper preprocessing
QWEN_SAMPLE_RATE = 16000
QWEN_N_FFT = 400
QWEN_HOP_LENGTH = 160
QWEN_N_MELS = 128
QWEN_CHUNK_SIZE = 30  # seconds
QWEN_N_SAMPLES = QWEN_SAMPLE_RATE * QWEN_CHUNK_SIZE  # 480000
QWEN_N_FFT_BINS = 1 + (QWEN_N_FFT / 2)  # 201 positive frequency bins


# example: "https://huggingface.co/OpenVoiceOS/qwen3-asr-0.6b-f16/resolve/main/qwen3-asr-0.6b-f16.gguf"
MODEL_URL_TEMPLATE = "https://huggingface.co/OpenVoiceOS/{model}/resolve/main/{file_name}.gguf"

PACKAGE_NAME = "py_qwen3_asr_cpp"
MODELS_DIR = Path(user_data_dir(PACKAGE_NAME)) / "models"


AVAILABLE_MODELS = {
    "asr": [
        "qwen3-asr-0.6b-f16",
        "qwen3-asr-0.6b-q8-0",
        "qwen3-asr-0.6b-q5-k-m",
        "qwen3-asr-0.6b-q4-k-m",
    ],
    "aligner": [
        "qwen3-forced-aligner-0.6b-f16",
        "qwen3-forced-aligner-0.6b-q8-0",
        "qwen3-forced-aligner-0.6b-q5-k-m",
        "qwen3-forced-aligner-0.6b-q4-k-m",
    ],
}


PARAMS_SCHEMA = {  # as exactly presented in whisper.cpp
    "asr_model": {
        "type": str,
        "description": "ASR model to use, can be a local path or a model name from the available models list",
        "options": None,
        "default": "qwen3-asr-0.6b-f16",
    },
    "align_model": {
        "type": str,
        "description": "Forced aligner model to use, can be a local path or a model name from the available models list. "
        "If not specified, it will be automatically determined based on the ASR model.",
        "options": None,
        "default": None,
    },
    "models_dir": {
        "type": str,
        "description": "Directory to store downloaded models, default to platform-specific user data dir",
        "options": None,
        "default": str(MODELS_DIR),
    },
    "n_threads": {
        "type": int,
        "description": "Number of threads to allocate for the inference"
        "default to min(4, available hardware_concurrency)",
        "options": None,
        "default": 4,
    },
    "max_tokens": {
        "type": int,
        "description": "Maximum tokens to decode, the default is 1024.",
        "options": None,
        "default": 1024,
    },
    "print_timing": {
        "type": bool,
        "description": "Print timing summary",
        "options": None,
        "default": False,
    },
    "print_progress": {
        "type": bool,
        "description": "Print progress information",
        "options": None,
        "default": False,
    },
    "language": {
        "type": str,
        "description": 'for auto-detection, set to None or ""',
        "options": None,
        "default": "",
    },
    # TODO: to be implemented in the future, currently not supported by the underlying C++ library
    # "temperature": {
    #     "type": float,
    #     "description": "initial decoding temperature",
    #     "options": None,
    #     "default": 0.0,
    # },
    # "greedy": {
    #     "type": dict,
    #     "description": "greedy",
    #     "options": None,
    #     "default": {"best_of": -1},
    # },
    # "beam_search": {
    #     "type": dict,
    #     "description": "beam_search",
    #     "options": None,
    #     "default": {"beam_size": -1, "patience": -1.0},
    # },
}

PARAMS_MAPPING = {
    "greedy.best_of": "best_of",
    "beam_search.beam_size": "beam_size",
    "beam_search.patience": "patience",
}
