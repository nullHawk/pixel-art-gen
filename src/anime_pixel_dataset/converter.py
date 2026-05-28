from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageOps
from tqdm import tqdm

from .pixelization import PixelizationRunner


@dataclass(frozen=True)
class ConversionConfig:
    dataset_id: str
    split: str
    image_column: str
    tags_column: str
    rank_column: str
    start_index: int
    limit: int | None
    streaming: bool
    hf_token: str | None
    output_dir: Path
    source_dir: Path | None
    metadata_path: Path | None
    overwrite: bool
    keep_source: bool

    @property
    def resolved_output_dir(self) -> Path:
        return self.output_dir.expanduser().resolve()

    @property
    def resolved_source_dir(self) -> Path:
        if self.source_dir is not None:
            return self.source_dir.expanduser().resolve()
        return self.resolved_output_dir / ".source"

    @property
    def resolved_metadata_path(self) -> Path:
        if self.metadata_path is not None:
            return self.metadata_path.expanduser().resolve()
        return self.resolved_output_dir / "metadata.jsonl"


@dataclass
class ConversionSummary:
    exported: int = 0
    converted: int = 0
    skipped: int = 0
    failed: int = 0
    output_dir: Path | None = None
    metadata_path: Path | None = None

    def format(self) -> str:
        return (
            "Done: "
            f"exported={self.exported}, converted={self.converted}, "
            f"skipped={self.skipped}, failed={self.failed}, "
            f"output_dir={self.output_dir}, metadata={self.metadata_path}"
        )


def load_hf_dataset(config: ConversionConfig) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Missing dependency 'datasets'. Install this project with `pip install -e .`."
        ) from error

    kwargs: dict[str, Any] = {
        "path": config.dataset_id,
        "split": config.split,
        "streaming": config.streaming,
    }
    if config.hf_token:
        kwargs["token"] = config.hf_token

    return load_dataset(**kwargs)


def iter_rows(config: ConversionConfig) -> Iterable[tuple[int, dict[str, Any]]]:
    dataset = load_hf_dataset(config)
    emitted = 0

    for index, row in enumerate(dataset):
        if index < config.start_index:
            continue
        if config.limit is not None and emitted >= config.limit:
            break

        emitted += 1
        yield index, row


def row_to_image(row: dict[str, Any], image_column: str) -> Image.Image:
    if image_column not in row:
        raise KeyError(f"image column '{image_column}' not found in dataset row")

    value = row[image_column]
    image: Image.Image

    if isinstance(value, Image.Image):
        image = value
    elif isinstance(value, dict):
        if value.get("bytes") is not None:
            image = Image.open(BytesIO(value["bytes"]))
        elif value.get("path") is not None:
            image = Image.open(value["path"])
        else:
            raise TypeError(f"unsupported image dict keys: {sorted(value.keys())}")
    elif isinstance(value, (str, Path)):
        image = Image.open(value)
    else:
        raise TypeError(f"unsupported image value type: {type(value).__name__}")

    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def image_name(index: int) -> str:
    return f"{index:08d}.png"


def metadata_for_row(
    *,
    config: ConversionConfig,
    index: int,
    row: dict[str, Any],
    source_path: Path,
    output_path: Path,
    cell_size: int | None,
) -> dict[str, Any]:
    metadata = {
        "dataset_id": config.dataset_id,
        "split": config.split,
        "index": index,
        "source_path": str(source_path),
        "output_path": str(output_path),
    }
    if cell_size is not None:
        metadata["cell_size"] = cell_size
    if config.tags_column in row:
        metadata[config.tags_column] = row[config.tags_column]
    if config.rank_column in row:
        metadata[config.rank_column] = row[config.rank_column]
    return metadata


def append_metadata(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def export_row(
    *,
    config: ConversionConfig,
    index: int,
    row: dict[str, Any],
    source_path: Path,
    output_path: Path,
    cell_size: int | None,
) -> None:
    image = row_to_image(row, config.image_column)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(source_path, format="PNG")


def write_metadata(
    *,
    config: ConversionConfig,
    index: int,
    row: dict[str, Any],
    source_path: Path,
    output_path: Path,
    cell_size: int | None,
) -> None:
    append_metadata(
        config.resolved_metadata_path,
        metadata_for_row(
            config=config,
            index=index,
            row=row,
            source_path=source_path,
            output_path=output_path,
            cell_size=cell_size,
        ),
    )


def export_dataset(config: ConversionConfig) -> ConversionSummary:
    output_dir = config.resolved_output_dir
    source_dir = config.source_dir.expanduser().resolve() if config.source_dir else output_dir
    metadata_path = config.resolved_metadata_path
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    summary = ConversionSummary(output_dir=output_dir, metadata_path=metadata_path)
    progress = tqdm(iter_rows(config), total=config.limit, unit="image")

    for index, row in progress:
        source_path = source_dir / image_name(index)
        output_path = source_path
        if source_path.exists() and not config.overwrite:
            summary.skipped += 1
            continue

        try:
            export_row(
                config=config,
                index=index,
                row=row,
                source_path=source_path,
                output_path=output_path,
                cell_size=None,
            )
            write_metadata(
                config=config,
                index=index,
                row=row,
                source_path=source_path,
                output_path=output_path,
                cell_size=None,
            )
            summary.exported += 1
        except Exception as error:  # noqa: BLE001 - continue the long-running dataset job.
            summary.failed += 1
            progress.write(f"failed export index={index}: {error}")

    return summary


def convert_dataset(
    config: ConversionConfig,
    *,
    pixelization: PixelizationRunner,
    cell_size: int,
) -> ConversionSummary:
    output_dir = config.resolved_output_dir
    source_dir = config.resolved_source_dir
    metadata_path = config.resolved_metadata_path
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    summary = ConversionSummary(output_dir=output_dir, metadata_path=metadata_path)
    pixelization.verify(load_model=False)

    with pixelization.loaded():
        progress = tqdm(iter_rows(config), total=config.limit, unit="image")
        for index, row in progress:
            source_path = source_dir / image_name(index)
            output_path = output_dir / image_name(index)

            if output_path.exists() and not config.overwrite:
                summary.skipped += 1
                continue

            try:
                export_row(
                    config=config,
                    index=index,
                    row=row,
                    source_path=source_path,
                    output_path=output_path,
                    cell_size=cell_size,
                )
                summary.exported += 1
                pixelization.pixelize(source_path, output_path, cell_size=cell_size)
                write_metadata(
                    config=config,
                    index=index,
                    row=row,
                    source_path=source_path,
                    output_path=output_path,
                    cell_size=cell_size,
                )
                summary.converted += 1
                if not config.keep_source:
                    source_path.unlink(missing_ok=True)
            except Exception as error:  # noqa: BLE001 - continue the long-running dataset job.
                summary.failed += 1
                progress.write(f"failed convert index={index}: {error}")

    if not config.keep_source:
        try:
            if source_dir.exists() and not any(source_dir.iterdir()):
                shutil.rmtree(source_dir)
        except OSError:
            pass

    return summary
