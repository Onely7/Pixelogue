"""Deterministic views that exclude neighboring subjects from local verification."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

from PIL import Image

from pixelogue.task_evidence import ImageRegion


@dataclass(frozen=True)
class FocusedView:
    """Encoded pixels and their exact normalized location in the source view."""

    encoded: bytes
    width: int
    height: int
    source_region: ImageRegion


def focused_view(encoded: bytes, region: ImageRegion) -> FocusedView | None:
    """Crop strictly inside a bound rectangle, returning no view for empty pixels.

    Integer rounding never expands the supplied region. No padding, image
    synthesis, resizing or annotations are introduced into the judge's view.
    """
    with Image.open(io.BytesIO(encoded)) as image:
        width, height = image.size
        left, top = math.ceil(region.left * width), math.ceil(region.top * height)
        right, bottom = math.floor(region.right * width), math.floor(region.bottom * height)
        if left >= right or top >= bottom:
            return None
        cropped = image.crop((left, top, right, bottom))
        buffer = io.BytesIO()
        cropped.save(buffer, format="PNG")
    return FocusedView(
        encoded=buffer.getvalue(),
        width=right - left,
        height=bottom - top,
        source_region=ImageRegion(
            left=left / width, top=top / height, right=right / width, bottom=bottom / height
        ),
    )
