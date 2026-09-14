"""Optional SSCD embedding adapter for visual copy grouping."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from PIL import Image

from pixelogue.errors import ExternalInputError


class SscdEmbedder:
    """Run a pinned local SSCD TorchScript model without downloading weights."""

    def __init__(self, model_path: Path, *, device: str = "cpu") -> None:
        """Load a local SSCD model.

        Args:
            model_path: Existing TorchScript weights whose hash is locked by the run.
            device: Explicit Torch device selected by the operator.

        Raises:
            ExternalInputError: If PyTorch or the model file is unavailable.
        """
        try:
            torch = importlib.import_module("torch")
        except ImportError as error:
            raise ExternalInputError(
                "SSCD_DEPENDENCY_MISSING",
                "Install the visual extra with `uv sync --extra visual`",
            ) from error
        if not model_path.is_file():
            raise ExternalInputError("SSCD_MODEL_MISSING", str(model_path))
        self._torch: Any = torch
        self._device = device
        try:
            self._model = torch.jit.load(str(model_path), map_location=device).eval()
        except (OSError, RuntimeError) as error:
            raise ExternalInputError("SSCD_MODEL_INVALID", str(error)) from error

    def embed(self, image: Image.Image) -> tuple[float, ...]:
        """Return a normalized embedding for one canonical RGB image."""
        import numpy as np

        torch = self._torch
        rgb = image.convert("RGB")
        width, height = rgb.size
        edge = min(width, height)
        left = (width - edge) // 2
        top = (height - edge) // 2
        crop = rgb.crop((left, top, left + edge, top + edge)).resize((288, 288))
        array = np.asarray(crop, dtype=np.float32) / 255.0
        mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
        standard_deviation = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
        normalized = (array - mean) / standard_deviation
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            embedding = self._model(tensor).float()
            embedding = torch.nn.functional.normalize(embedding, dim=1)
        return tuple(float(value) for value in embedding[0].cpu().tolist())
