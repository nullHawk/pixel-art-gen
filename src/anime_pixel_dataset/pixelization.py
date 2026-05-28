from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterator


@dataclass(frozen=True)
class PixelizationConfig:
    repo_dir: Path
    model_name: str
    device: str = "auto"

    @property
    def resolved_repo_dir(self) -> Path:
        return self.repo_dir.expanduser().resolve()


class PixelizationRunner:
    def __init__(self, config: PixelizationConfig) -> None:
        self.config = config
        self._module: ModuleType | None = None
        self._model = None
        self._device: str | None = None

    @property
    def repo_dir(self) -> Path:
        return self.config.resolved_repo_dir

    @property
    def test_pro_path(self) -> Path:
        return self.repo_dir / "test_pro.py"

    @property
    def alias_net_path(self) -> Path:
        return self.repo_dir / "alias_net.pth"

    @property
    def generator_path(self) -> Path:
        return self.repo_dir / "checkpoints" / self.config.model_name / "160_net_G_A.pth"

    def verify(self, *, load_model: bool = False) -> None:
        missing = [
            path
            for path in [self.repo_dir, self.test_pro_path, self.alias_net_path, self.generator_path]
            if not path.exists()
        ]
        if missing:
            formatted = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(
                "Pixelization is not ready. Missing required path(s):\n"
                f"{formatted}\n"
                "Clone https://github.com/WuZongWei6/Pixelization and place the pretrained "
                "checkpoint files according to its README."
            )

        self.import_test_pro()

        if load_model:
            with self.loaded():
                pass

    @contextlib.contextmanager
    def in_repo(self) -> Iterator[None]:
        old_cwd = Path.cwd()
        old_path = list(sys.path)
        sys.path.insert(0, str(self.repo_dir))
        os.chdir(self.repo_dir)
        try:
            yield
        finally:
            os.chdir(old_cwd)
            sys.path[:] = old_path

    def resolve_device(self) -> str:
        if self.config.device != "auto":
            return self.config.device

        try:
            import torch
        except ImportError as error:
            raise RuntimeError(
                "Pixelization requires PyTorch. Install torch/torchvision for your CUDA or CPU environment."
            ) from error

        return "cuda" if torch.cuda.is_available() else "cpu"

    def import_test_pro(self) -> ModuleType:
        if self._module is not None:
            return self._module

        spec = importlib.util.spec_from_file_location("pixelization_test_pro", self.test_pro_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not import {self.test_pro_path}")

        with self.in_repo():
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

        self._module = module
        return module

    def patch_cpu_define_g(self, module: ModuleType) -> None:
        if getattr(module, "_anime_pixel_dataset_cpu_patch", False):
            return

        original_define_g = module.define_G

        def define_g_cpu(*args, **kwargs):
            patched_args = list(args)
            if len(patched_args) >= 9:
                patched_args[8] = []
            else:
                kwargs["gpu_ids"] = []
            net = original_define_g(*patched_args, **kwargs)
            return module.torch.nn.DataParallel(net)

        module.define_G = define_g_cpu
        module._anime_pixel_dataset_cpu_patch = True

    @contextlib.contextmanager
    def loaded(self) -> Iterator["PixelizationRunner"]:
        if self._model is not None:
            yield self
            return

        self.verify(load_model=False)
        device = self.resolve_device()
        module = self.import_test_pro()

        if device == "cpu":
            self.patch_cpu_define_g(module)

        with self.in_repo():
            model = module.Model(self.config.model_name, device=device)
            model.load()

        self._device = device
        self._model = model
        try:
            yield self
        finally:
            self._model = None

    def pixelize(self, input_path: Path, output_path: Path, *, cell_size: int) -> None:
        if self._model is None:
            raise RuntimeError("Pixelization model is not loaded. Use `with runner.loaded():` first.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.in_repo():
            self._model.pixelize(str(input_path.resolve()), str(output_path.resolve()), cell_size)
