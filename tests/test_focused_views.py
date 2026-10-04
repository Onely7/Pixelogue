"""Local judge views exclude neighbors and preserve exact source coordinates."""

import io

from PIL import Image

from pixelogue.focused_views import focused_view
from pixelogue.task_evidence import ImageRegion


def pixels() -> bytes:
    image = Image.new("RGB", (60, 50), "red")
    image.paste("blue", (30, 0, 60, 50))
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def test_neighbor_pixels_cannot_reach_the_local_judge() -> None:
    view = focused_view(pixels(), ImageRegion(left=0, top=0, right=0.5, bottom=1))
    assert view is not None
    with Image.open(io.BytesIO(view.encoded)) as crop:
        assert crop.size == (30, 50)
        assert set(crop.getdata()) == {(255, 0, 0)}
    assert view.source_region == ImageRegion(left=0, top=0, right=0.5, bottom=1)


def test_rounding_never_expands_the_public_region_and_is_reproducible() -> None:
    region = ImageRegion(left=0.01, top=0.01, right=0.49, bottom=0.99)
    first = focused_view(pixels(), region)
    assert first == focused_view(pixels(), region)
    assert first is not None
    assert (first.width, first.height) == (28, 48)
    assert first.source_region.left >= region.left
    assert first.source_region.top >= region.top
    assert first.source_region.right <= region.right
    assert first.source_region.bottom <= region.bottom


def test_a_subpixel_region_abstains_instead_of_showing_the_full_image() -> None:
    assert focused_view(pixels(), ImageRegion(left=0, top=0, right=0.001, bottom=0.001)) is None
