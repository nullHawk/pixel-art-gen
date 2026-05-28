# Anime Pixel Dataset CLI

This project converts the Hugging Face dataset `minoruskore/anime-faces-256` into pixel art using the upstream `WuZongWei6/Pixelization` implementation.

The Pixelization code and checkpoints are not vendored here. Its license is for non-commercial scientific research use, so keep the upstream repo and model setup explicit.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install PyTorch and torchvision for your machine separately. For GPU conversion, install a CUDA-compatible PyTorch build.

Copy the environment template and keep credentials there:

```bash
cp .env.example .env
```

Set `HF_TOKEN` in `.env` instead of passing it on the command line. You can also set `HF_UPLOAD_REPO_ID`, `PIXELIZATION_REPO`, and `PIXELIZATION_MODEL_NAME` there.

Clone Pixelization:

```bash
mkdir -p vendor
git clone https://github.com/WuZongWei6/Pixelization vendor/Pixelization
```

Then download the pretrained Pixelization model files from the upstream README and place them like this:

```text
vendor/Pixelization/
  alias_net.pth
  checkpoints/
    YOUR_MODEL_NAME/
      160_net_G_A.pth
```

`test_pro.py` is the path used by this CLI. If you also use Pixelization's original `test.py` workflow, follow all extra upstream model placement instructions.

## Verify

```bash
anime-pixel-dataset verify \
  --pixelization-repo vendor/Pixelization \
  --model-name YOUR_MODEL_NAME
```

## Convert The Dataset

Start with a small sample:

```bash
anime-pixel-dataset convert \
  --pixelization-repo vendor/Pixelization \
  --model-name YOUR_MODEL_NAME \
  --output-dir data/pixelized/anime-faces-256 \
  --limit 100 \
  --cell-size 4
```

Run the full `train` split:

```bash
anime-pixel-dataset convert \
  --pixelization-repo vendor/Pixelization \
  --model-name YOUR_MODEL_NAME \
  --output-dir data/pixelized/anime-faces-256 \
  --cell-size 4
```

Outputs are named by dataset row index, for example `00000042.png`. Metadata is appended to `metadata.jsonl` with the dataset id, split, row index, source path, output path, tags, rank, and cell size.

Convert and upload the result to a Hugging Face dataset repo:

```bash
anime-pixel-dataset convert \
  --pixelization-repo vendor/Pixelization \
  --model-name YOUR_MODEL_NAME \
  --output-dir data/pixelized/anime-faces-256 \
  --cell-size 4 \
  --upload-repo-id YOUR_USERNAME/anime-faces-256-pixelized \
  --upload-private
```

## Export Only

To only materialize source images from Hugging Face:

```bash
anime-pixel-dataset export \
  --output-dir data/source/anime-faces-256 \
  --limit 100 \
  --keep-source
```

## Upload Existing Output

Upload an already-converted output directory:

```bash
anime-pixel-dataset upload \
  --output-dir data/pixelized/anime-faces-256 \
  --repo-id YOUR_USERNAME/anime-faces-256-pixelized \
  --private
```

For a full dataset, use the resumable large-folder uploader:

```bash
anime-pixel-dataset upload \
  --output-dir data/pixelized/anime-faces-256 \
  --repo-id YOUR_USERNAME/anime-faces-256-pixelized \
  --large-folder \
  --num-workers 8
```

The upload command stages an ImageFolder-style dataset at `OUTPUT_DIR/.hf-upload`:

```text
.hf-upload/
  README.md
  train/
    00000000.png
    00000001.png
    metadata.jsonl
```

The staged `metadata.jsonl` uses `file_name` so Hugging Face Datasets can associate each image with row metadata.

## Useful Options

- `--start-index N` resumes from a specific row.
- `--limit N` caps the number of rows.
- `--overwrite` regenerates existing files.
- `--device cpu` forces CPU inference.
- `--no-streaming` downloads through the normal Hugging Face datasets cache instead of streaming rows.
- `--hf-token TOKEN` passes an access token for gated/private datasets.
- `--upload-repo-id USER/REPO` uploads after `convert`.
- `--large-folder` or `--upload-large-folder` uses Hugging Face Hub's large-folder upload path.

## Notes

The default dataset is `minoruskore/anime-faces-256`, split `train`. The dataset is large, so expect the full conversion to take time and disk space. By default, temporary source PNGs are deleted after each successful conversion; pass `--keep-source` if you want to retain them.

Before publishing, verify that the source dataset license and Pixelization's non-commercial scientific research license allow your intended use and redistribution.
