from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(frozen=True)
class HubUploadConfig:
    output_dir: Path
    repo_id: str
    split: str = "train"
    metadata_path: Path | None = None
    stage_dir: Path | None = None
    token: str | None = None
    private: bool = False
    revision: str | None = None
    path_in_repo: str = "."
    commit_message: str = "Upload pixelized anime faces dataset"
    large_folder: bool = False
    num_workers: int | None = None

    @property
    def resolved_output_dir(self) -> Path:
        return self.output_dir.expanduser().resolve()

    @property
    def resolved_metadata_path(self) -> Path:
        if self.metadata_path is not None:
            return self.metadata_path.expanduser().resolve()
        return self.resolved_output_dir / "metadata.jsonl"

    @property
    def resolved_stage_dir(self) -> Path:
        if self.stage_dir is not None:
            return self.stage_dir.expanduser().resolve()
        return self.resolved_output_dir / ".hf-upload"


@dataclass
class HubUploadSummary:
    repo_id: str
    staged_dir: Path
    image_count: int
    metadata_count: int
    uploaded: bool = False

    def format(self) -> str:
        return (
            "Hub upload complete: "
            f"repo_id={self.repo_id}, images={self.image_count}, "
            f"metadata_rows={self.metadata_count}, staged_dir={self.staged_dir}"
        )


def load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error

            output_path = record.get("output_path")
            if output_path:
                records[Path(output_path).name] = record
            elif record.get("file_name"):
                records[Path(record["file_name"]).name] = record

    return records


def list_output_images(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        raise FileNotFoundError(f"output directory does not exist: {output_dir}")

    return sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def parse_index_from_name(path: Path) -> int | None:
    try:
        return int(path.stem)
    except ValueError:
        return None


def staged_metadata_row(image_path: Path, record: dict[str, Any] | None) -> dict[str, Any]:
    row: dict[str, Any] = {"file_name": image_path.name}

    if record:
        if "index" in record:
            row["original_index"] = record["index"]
        if "dataset_id" in record:
            row["source_dataset"] = record["dataset_id"]
        if "split" in record:
            row["source_split"] = record["split"]

        for key in ["tags", "rank", "cell_size"]:
            if key in record:
                row[key] = record[key]

        for key, value in record.items():
            if key in {"index", "dataset_id", "split", "source_path", "output_path"}:
                continue
            if key in row:
                continue
            if isinstance(value, str | int | float | bool) or value is None:
                row[key] = value
    else:
        parsed_index = parse_index_from_name(image_path)
        if parsed_index is not None:
            row["original_index"] = parsed_index

    return row


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()

    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_dataset_card(stage_dir: Path, config: HubUploadConfig, image_count: int) -> None:
    card = f"""---
license: other
task_categories:
- image-to-image
tags:
- anime
- pixel-art
- imagefolder
pretty_name: Anime Faces Pixelized
---

# Anime Faces Pixelized

Pixel-art conversion of images generated from `{config.output_dir}`.

This dataset was prepared with `anime-pixel-dataset` and the upstream
`WuZongWei6/Pixelization` implementation. Review the source dataset license
and Pixelization's non-commercial scientific research license before publishing
or using the uploaded result.

## Files

- Split: `{config.split}`
- Images: {image_count}
- Metadata: `{config.split}/metadata.jsonl`

Each metadata row contains `file_name` plus any available source row metadata
such as `original_index`, `tags`, `rank`, and `cell_size`.
"""
    (stage_dir / "README.md").write_text(card, encoding="utf-8")


def prepare_hub_dataset(config: HubUploadConfig) -> HubUploadSummary:
    output_dir = config.resolved_output_dir
    metadata = load_metadata(config.resolved_metadata_path)
    images = list_output_images(output_dir)
    if not images:
        raise FileNotFoundError(f"no image files found in {output_dir}")

    stage_dir = config.resolved_stage_dir
    if stage_dir == output_dir:
        raise ValueError("stage directory must be different from output directory")

    if stage_dir.exists():
        shutil.rmtree(stage_dir)

    split_dir = stage_dir / config.split
    split_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = split_dir / "metadata.jsonl"

    metadata_count = 0
    with metadata_path.open("w", encoding="utf-8") as handle:
        for image_path in tqdm(images, unit="image", desc="staging"):
            staged_path = split_dir / image_path.name
            link_or_copy(image_path, staged_path)
            row = staged_metadata_row(image_path, metadata.get(image_path.name))
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            metadata_count += 1

    write_dataset_card(stage_dir, config, image_count=len(images))

    return HubUploadSummary(
        repo_id=config.repo_id,
        staged_dir=stage_dir,
        image_count=len(images),
        metadata_count=metadata_count,
    )


def upload_dataset_to_hub(config: HubUploadConfig) -> HubUploadSummary:
    summary = prepare_hub_dataset(config)

    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError(
            "Missing dependency 'huggingface_hub'. Install this project with `pip install -e .`."
        ) from error

    api = HfApi(token=config.token)
    api.create_repo(
        repo_id=config.repo_id,
        repo_type="dataset",
        private=config.private,
        exist_ok=True,
    )

    if config.large_folder:
        if config.path_in_repo != ".":
            raise ValueError("--path-in-repo is not supported with --large-folder")
        if not hasattr(api, "upload_large_folder"):
            raise RuntimeError("installed huggingface_hub does not provide upload_large_folder")
        api.upload_large_folder(
            repo_id=config.repo_id,
            folder_path=summary.staged_dir,
            repo_type="dataset",
            revision=config.revision,
            private=config.private,
            num_workers=config.num_workers,
        )
    else:
        api.upload_folder(
            folder_path=summary.staged_dir,
            repo_id=config.repo_id,
            repo_type="dataset",
            path_in_repo=config.path_in_repo,
            revision=config.revision,
            commit_message=config.commit_message,
        )

    summary.uploaded = True
    return summary
