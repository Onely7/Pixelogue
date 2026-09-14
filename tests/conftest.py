from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from pixelogue.contracts import ImageArtifact, ImageView, SourcePurpose


@pytest.fixture
def image_artifact(tmp_path: Path) -> tuple[ImageArtifact, Path]:
    root = tmp_path / "prepared"
    path = root / "images" / "test.png"
    path.parent.mkdir(parents=True)
    Image.new("RGB", (256, 192), "blue").save(path)
    encoded_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    pixel_hash = hashlib.sha256(b"pixel").hexdigest()
    artifact = ImageArtifact(
        image_id=hashlib.sha256(b"image").hexdigest(),
        source_id="source-1",
        purpose=SourcePurpose.TRAINING,
        raw_sha256=encoded_hash,
        canonical_pixel_sha256=pixel_hash,
        source_group_ids=("group-1",),
        visual_group_id="visual-1",
        full_view=ImageView(
            view_id="full:test",
            relative_path="images/test.png",
            encoded_sha256=encoded_hash,
            pixel_sha256=pixel_hash,
            width=256,
            height=192,
            media_type="image/png",
        ),
    )
    return artifact, root
