#!/usr/bin/env python

import argparse
import importlib.metadata
import logging

import sounddevice as sd

import py_qwen3_asr_cpp.constants
from py_qwen3_asr_cpp.model import Qwen3ASRModel


__version__ = importlib.metadata.version("py_qwen3_asr_cpp")

__header__ = f"""
===================================================================
PyQwen3ASR
A simple example of transcribing a recording, based on Qwen3 ASR
Version: {__version__}               
===================================================================
"""


class Recording:
    """
    Recording class

    Example usage
    ```python
    from pywhispercpp.examples.recording import Recording

    myrec = Recording(5)
    myrec.start()
    ```
    """

    def __init__(
        self, duration: int, model: str = "qwen3-asr-0.6b-q4_k_m.gguf", **model_params
    ):
        self.duration = duration
        self.sample_rate = py_qwen3_asr_cpp.constants.QWEN_SAMPLE_RATE
        self.channels = 1
        self.pqw3_model = Qwen3ASRModel(
            asr_model=model, print_timings=True, **model_params
        )

    def start(self):
        logging.info(f"Start recording for {self.duration}s ...")
        recording = sd.rec(
            int(self.duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
        )
        sd.wait()
        logging.info("Duration finished")
        _res = self.pqw3_model.transcribe(recording)


def _main():
    print(__header__)
    parser = argparse.ArgumentParser(description="", allow_abbrev=True)
    # Positional args
    parser.add_argument("duration", type=int, help="Duration in seconds")
    parser.add_argument(
        "-m",
        "--model",
        default="qwen3-asr-0.6b-q4_k_m.gguf",
        type=str,
        help="Qwen3 ASR model, default to %(default)s",
    )

    args = parser.parse_args()

    myrec = Recording(duration=args.duration, model=args.model)
    myrec.start()


if __name__ == "__main__":
    _main()
