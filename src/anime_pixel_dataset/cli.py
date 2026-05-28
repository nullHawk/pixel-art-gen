from __future__ import annotations

import argparse
import os
from pathlib import Path

from .converter import ConversionConfig, convert_dataset, export_dataset
from .env import env_bool, load_env_file
from .hub import HubUploadConfig, upload_dataset_to_hub
from .pixelization import PixelizationConfig, PixelizationRunner


DEFAULT_DATASET = "minoruskore/anime-faces-256"
DEFAULT_SPLIT = "train"
DEFAULT_REPO = "vendor/Pixelization"


def env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default))


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return parsed


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-id", default=os.getenv("HF_DATASET_ID", DEFAULT_DATASET), help="Hugging Face dataset id.")
    parser.add_argument("--split", default=os.getenv("HF_DATASET_SPLIT", DEFAULT_SPLIT), help="Dataset split to process.")
    parser.add_argument("--image-column", default="image", help="Column containing images.")
    parser.add_argument("--tags-column", default="tags", help="Column containing tags/captions.")
    parser.add_argument("--rank-column", default="rank", help="Column containing rank metadata.")
    parser.add_argument("--start-index", type=non_negative_int, default=0, help="First dataset row index to process.")
    parser.add_argument("--limit", type=positive_int, default=None, help="Maximum number of rows to process.")
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Use a local Arrow dataset download instead of Hugging Face streaming.",
    )
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"), help="Optional Hugging Face token for gated/private datasets.")


def add_io_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=env_path("PIXEL_OUTPUT_DIR", "data/pixelized/anime-faces-256"),
        help="Directory for generated pixel-art PNGs.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Directory for temporary/source PNGs. Defaults to OUTPUT_DIR/.source.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="JSONL metadata path. Defaults to OUTPUT_DIR/metadata.jsonl.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate outputs that already exist.")
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="Keep exported source images after conversion.",
    )


def add_convert_upload_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--upload-repo-id",
        default=os.getenv("HF_UPLOAD_REPO_ID"),
        help="Upload generated pixel-art images to this Hugging Face dataset repo after conversion.",
    )
    parser.add_argument("--upload-token", default=os.getenv("HF_TOKEN"), help="Hugging Face token for uploading.")
    parser.add_argument(
        "--upload-private",
        action="store_true",
        default=env_bool("HF_UPLOAD_PRIVATE"),
        help="Create/update the Hub dataset as private.",
    )
    parser.add_argument("--upload-revision", default=None, help="Hub branch/revision to upload to.")
    parser.add_argument("--upload-path-in-repo", default=".", help="Destination path inside the Hub dataset repo.")
    parser.add_argument(
        "--upload-stage-dir",
        type=Path,
        default=None,
        help="Local staging directory for the ImageFolder upload layout.",
    )
    parser.add_argument(
        "--upload-large-folder",
        action="store_true",
        help="Use huggingface_hub upload_large_folder for large datasets.",
    )
    parser.add_argument(
        "--upload-num-workers",
        type=positive_int,
        default=None,
        help="Worker count for --upload-large-folder.",
    )


def add_upload_command_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=env_path("PIXEL_OUTPUT_DIR", "data/pixelized/anime-faces-256"),
        help="Directory containing generated pixel-art images.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="JSONL metadata path. Defaults to OUTPUT_DIR/metadata.jsonl.",
    )
    parser.add_argument("--split", default=os.getenv("HF_DATASET_SPLIT", DEFAULT_SPLIT), help="Hub dataset split name.")
    parser.add_argument(
        "--repo-id",
        default=os.getenv("HF_UPLOAD_REPO_ID"),
        required=os.getenv("HF_UPLOAD_REPO_ID") is None,
        help="Hugging Face dataset repo id, e.g. username/repo.",
    )
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"), help="Hugging Face token for uploading.")
    parser.add_argument(
        "--private",
        action="store_true",
        default=env_bool("HF_UPLOAD_PRIVATE"),
        help="Create/update the Hub dataset as private.",
    )
    parser.add_argument("--revision", default=None, help="Hub branch/revision to upload to.")
    parser.add_argument("--path-in-repo", default=".", help="Destination path inside the Hub dataset repo.")
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=None,
        help="Local staging directory for the ImageFolder upload layout.",
    )
    parser.add_argument(
        "--large-folder",
        action="store_true",
        help="Use huggingface_hub upload_large_folder for large datasets.",
    )
    parser.add_argument(
        "--num-workers",
        type=positive_int,
        default=None,
        help="Worker count for --large-folder.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anime-pixel-dataset",
        description="Convert Hugging Face anime image datasets with WuZongWei6/Pixelization.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser("convert", help="Export rows from Hugging Face and pixelize them.")
    add_dataset_args(convert)
    add_io_args(convert)
    convert.add_argument(
        "--pixelization-repo",
        type=Path,
        default=env_path("PIXELIZATION_REPO", DEFAULT_REPO),
        help="Path to a local clone of https://github.com/WuZongWei6/Pixelization.",
    )
    convert.add_argument(
        "--model-name",
        default=os.getenv("PIXELIZATION_MODEL_NAME"),
        required=os.getenv("PIXELIZATION_MODEL_NAME") is None,
        help="Pixelization checkpoint directory name under PIXELIZATION_REPO/checkpoints.",
    )
    convert.add_argument("--cell-size", type=positive_int, default=4, help="Pixelization cell size.")
    convert.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device for Pixelization inference.",
    )
    add_convert_upload_args(convert)

    export = subparsers.add_parser("export", help="Export dataset rows to PNG files without pixelizing.")
    add_dataset_args(export)
    add_io_args(export)

    upload = subparsers.add_parser("upload", help="Upload generated pixel-art images to a Hub dataset repo.")
    add_upload_command_args(upload)

    verify = subparsers.add_parser("verify", help="Validate Pixelization repo, checkpoints, and imports.")
    verify.add_argument(
        "--pixelization-repo",
        type=Path,
        default=env_path("PIXELIZATION_REPO", DEFAULT_REPO),
        help="Path to a local clone of https://github.com/WuZongWei6/Pixelization.",
    )
    verify.add_argument(
        "--model-name",
        default=os.getenv("PIXELIZATION_MODEL_NAME"),
        required=os.getenv("PIXELIZATION_MODEL_NAME") is None,
        help="Pixelization checkpoint directory name under PIXELIZATION_REPO/checkpoints.",
    )
    verify.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device for Pixelization inference.",
    )

    return parser


def make_conversion_config(args: argparse.Namespace) -> ConversionConfig:
    return ConversionConfig(
        dataset_id=args.dataset_id,
        split=args.split,
        image_column=args.image_column,
        tags_column=args.tags_column,
        rank_column=args.rank_column,
        start_index=args.start_index,
        limit=args.limit,
        streaming=not args.no_streaming,
        hf_token=args.hf_token,
        output_dir=args.output_dir,
        source_dir=args.source_dir,
        metadata_path=args.metadata_path,
        overwrite=args.overwrite,
        keep_source=args.keep_source,
    )


def make_upload_config_from_convert(args: argparse.Namespace) -> HubUploadConfig:
    return HubUploadConfig(
        output_dir=args.output_dir,
        repo_id=args.upload_repo_id,
        split=args.split,
        metadata_path=args.metadata_path,
        stage_dir=args.upload_stage_dir,
        token=args.upload_token or args.hf_token,
        private=args.upload_private,
        revision=args.upload_revision,
        path_in_repo=args.upload_path_in_repo,
        large_folder=args.upload_large_folder,
        num_workers=args.upload_num_workers,
    )


def make_upload_config(args: argparse.Namespace) -> HubUploadConfig:
    return HubUploadConfig(
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        split=args.split,
        metadata_path=args.metadata_path,
        stage_dir=args.stage_dir,
        token=args.token,
        private=args.private,
        revision=args.revision,
        path_in_repo=args.path_in_repo,
        large_folder=args.large_folder,
        num_workers=args.num_workers,
    )


def main(argv: list[str] | None = None) -> int:
    load_env_file()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "verify":
        pixelization = PixelizationRunner(
            PixelizationConfig(
                repo_dir=args.pixelization_repo,
                model_name=args.model_name,
                device=args.device,
            )
        )
        pixelization.verify(load_model=False)
        print("Pixelization repo and checkpoint layout look valid.")
        return 0

    if args.command == "export":
        config = make_conversion_config(args)
        summary = export_dataset(config)
        print(summary.format())
        return 0

    if args.command == "upload":
        summary = upload_dataset_to_hub(make_upload_config(args))
        print(summary.format())
        return 0

    if args.command == "convert":
        config = make_conversion_config(args)
        pixelization = PixelizationRunner(
            PixelizationConfig(
                repo_dir=args.pixelization_repo,
                model_name=args.model_name,
                device=args.device,
            )
        )
        summary = convert_dataset(config, pixelization=pixelization, cell_size=args.cell_size)
        print(summary.format())
        if args.upload_repo_id:
            upload_summary = upload_dataset_to_hub(make_upload_config_from_convert(args))
            print(upload_summary.format())
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2
