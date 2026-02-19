#!/usr/bin/env python

import contextlib
import logging
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

import requests
from tqdm import tqdm

from enum import StrEnum
from py_qwen3_asr_cpp.constants import (
    AVAILABLE_MODELS,
    MODEL_URL_TEMPLATE,
    MODELS_DIR,
)


logger = logging.getLogger(__name__)


class ModelType(StrEnum):
    ASR = "asr"
    ALIGNER = "aligner"


def _get_model_url(model_name: str) -> str:
    """
    Returns the url of the `ggml` model
    :param model_name: name of the model
    :return: URL of the model
    """
    file_name = (
        model_name.replace("q8-0", "q8_0")
        .replace("q5-k-m", "q5_k_m")
        .replace("q4-k-m", "q4_k_m")
    )
    return MODEL_URL_TEMPLATE.format(model=model_name, file_name=file_name)


def download_model(
    model_name: str,
    download_dir: str | None = None,
    chunk_size: int = 1024,
    model_type: ModelType = ModelType.ASR,
) -> str:
    """
    Helper function to download the `ggml` models
    :param model_name: name of the model, one of ::: constants.AVAILABLE_MODELS
    :param download_dir: Where to store the models
    :param chunk_size: size of the download chunk

    :return: Absolute path of the downloaded model
    """
    available_models = AVAILABLE_MODELS[ModelType.ASR]
    if model_name not in available_models:
        logger.error(
            f"Invalid model name `{model_name}`, available models are: {available_models}"
        )
        raise ValueError("Invalid model name")
    if download_dir is None:
        download_dir = str(MODELS_DIR)
        logger.info(
            f"No download directory was provided, models will be downloaded to {download_dir}"
        )

    os.makedirs(download_dir, exist_ok=True)

    url = _get_model_url(model_name=model_name)
    file_path = Path(download_dir) / os.path.basename(url)
    # check if the file is already there
    if file_path.exists():
        logger.info(f"Model {model_name} already exists in {download_dir}")
    else:
        # download it from huggingface
        resp = requests.get(url, stream=True)
        total = int(resp.headers.get("content-length", 0))

        progress_bar = tqdm(
            desc=f"Downloading {model_name} ...",
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        )

        try:
            with open(file_path, "wb") as file, progress_bar:
                for data in resp.iter_content(chunk_size=chunk_size):
                    size = file.write(data)
                    progress_bar.update(size)
            logger.info(f"Model downloaded to {file_path.absolute()}")
        except Exception as e:
            # error download, just remove the file
            os.remove(file_path)
            raise e
    return str(file_path.absolute())


@contextlib.contextmanager
def redirect_stderr(to: bool | TextIO | str | None = False) -> Iterator:
    """
    Redirect stderr to the specified target.

    :param to:
        - None to suppress output (redirect to devnull),
        - sys.stdout to redirect to stdout,
        - A file path (str) to redirect to a file,
        - False to do nothing (no redirection).
    """

    if to is False:
        # do nothing
        yield
        return

    def _resolve_target(target):
        opened_stream = None
        if target is None:
            opened_stream = open(os.devnull, "w")
            return opened_stream, True
        if isinstance(target, str):
            opened_stream = open(target, "w")
            return opened_stream, True
        if hasattr(target, "write"):
            return target, False
        raise ValueError(
            "Invalid `to` parameter; expected None, a filepath string, or a file-like object."
        )

    sys.stderr.flush()
    try:
        original_fd = sys.stderr.fileno()
    except (AttributeError, OSError):
        # Jupyter or non-standard stderr implementations
        original_fd = None

    stream, should_close = _resolve_target(to)

    if original_fd is not None and isinstance(stream, TextIO):
        saved_fd = os.dup(original_fd)
        try:
            os.dup2(stream.fileno(), original_fd)
            yield
        finally:
            os.dup2(saved_fd, original_fd)
            os.close(saved_fd)
            if should_close:
                stream.close()
        return

    # Fallback: Python-level redirect
    if isinstance(stream, TextIO):
        try:
            with contextlib.redirect_stderr(stream):
                yield
        finally:
            if should_close:
                stream.close()
