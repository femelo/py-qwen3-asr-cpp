#!/usr/bin/env python

"""
A simple Command Line Interface to test the package
"""

import argparse
import importlib.metadata
import logging

import py_qwen3_asr_cpp.constants as constants
from py_qwen3_asr_cpp.model import Qwen3ASRModel

__version__ = importlib.metadata.version("py_qwen3_asr_cpp")

__header__ = f"""
PyQwen3ASR
A simple Command Line Interface to test the package
Version: {__version__}               
====================================================
"""

logger = logging.getLogger("py_qwen3_asr_cpp-cli")
logger.setLevel(logging.DEBUG)
logging.basicConfig()


def _get_params(args: argparse.Namespace) -> dict:
    """
    Helper function to get params from argparse as a `dict`
    """
    params = {}
    inv_params_mapping = {v: k for k, v in constants.PARAMS_MAPPING.items()}
    for arg in args.__dict__:
        if arg in constants.PARAMS_SCHEMA and getattr(args, arg) is not None:
            params[arg] = getattr(args, arg)
        elif arg in inv_params_mapping:
            arg_ = inv_params_mapping[arg]
            if "." in arg_:
                arg_, subarg_ = arg_.split(".")
                if arg_ not in params:
                    params[arg_] = {}
                params[arg_][subarg_] = getattr(args, arg)
            else:
                params[arg_] = getattr(args, arg)
        else:
            pass
    return params


def run(args: argparse.Namespace) -> None:
    params = _get_params(args)
    m = Qwen3ASRModel(**params)
    if args.verbose:
        print("\npy-qwen3-asr-cli")
        for key, value in m.get_params().items():
            print(f"  {key}: {value}")
    for file in args.media_file:
        logger.debug(f"Processing file {file} ...")
        try:
            transcr_result = None
            align_result = None
            if args.mode == "transcribe":
                transcr_result = m.transcribe(
                    file,
                )
            elif args.mode == "align":
                align_result = m.align(
                    file,
                    text=args.text,
                )
            elif args.mode == "all":
                transcr_result, align_result = m.transcribe_and_align(
                    file,
                )
            else:
                raise ValueError(f"Unsupported mode: {args.mode}")
        except KeyboardInterrupt:
            logger.info("Transcription manually stopped")
            break
        except Exception as e:
            logger.error(f"Error while processing file {file}: {e}")
        finally:
            if transcr_result:
                logger.info(f"Transcript: {transcr_result.text}")
            if align_result:
                logger.info(f"Alignment: {align_result.words}")


def main() -> None:
    print(__header__)
    parser = argparse.ArgumentParser(description="", allow_abbrev=True)
    # Positional args
    parser.add_argument(
        "media_file",
        type=str,
        nargs="+",
        help="The path of the media file or a list of files separated by space",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["transcribe", "align", "all"],
        default="transcribe",
        help="The mode of operation: transcribe, align, or all (default: transcribe)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Whether to print detailed logs (default: False)",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    # add params from PARAMS_SCHEMA
    for param in constants.PARAMS_SCHEMA:
        param_fields = constants.PARAMS_SCHEMA[param]
        type_ = param_fields["type"]
        descr_ = param_fields["description"]
        default_ = param_fields["default"]
        if type_ is dict:
            for dft_key, dft_val in default_.items():
                map_key = f"{param}.{dft_key}"
                map_val = constants.PARAMS_MAPPING.get(map_key)
                if map_val:
                    parser.add_argument(
                        f"--{map_val.replace('_', '-')}",
                        type=type(dft_val),
                        default=dft_val,
                        help=map_key,
                    )
        elif param in constants.PARAMS_MAPPING:
            mapped_param = constants.PARAMS_MAPPING[param]
            parser.add_argument(
                f"--{mapped_param.replace('_', '-')}",
                type=type_,
                default=default_,
                help=descr_,
            )
        elif type_ is bool:
            parser.add_argument(
                f"--{param.replace('_', '-')}",
                type=lambda v: v.lower() in ("true", "yes", "y", "1"),
                default=default_,
                help=descr_,
            )
        else:
            parser.add_argument(
                f"--{param.replace('_', '-')}",
                type=type_,
                default=default_,
                help=descr_,
            )

    args, _ = parser.parse_known_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    run(args)


if __name__ == "__main__":
    main()
