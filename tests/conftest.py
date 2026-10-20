import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import synthetic as syn  # noqa: E402
from profile_check.config import load_config  # noqa: E402
from profile_check.pipeline import part_from_image  # noqa: E402
from profile_check.reference import mean_shape  # noqa: E402

# Shape of the "sample image" the variants are made from; deliberately not one of the reference parts.
SAMPLE_SHAPE = dict(length=78.1, inner_width=25.45, base_thickness=12.05, outer_width=46.05)


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def good_images():
    return [syn.render(angle_deg=a, seed=i, **s)
            for i, (s, a) in enumerate(zip(syn.GOOD_PART_SHAPES, syn.GOOD_PART_ANGLES))]


@pytest.fixture(scope="session")
def good_parts(cfg, good_images):
    return [part_from_image(img, syn.PX_PER_MM, cfg, name=f"good{i}") for i, img in enumerate(good_images)]


@pytest.fixture(scope="session")
def reference(cfg, good_parts):
    return mean_shape(good_parts, cfg)


@pytest.fixture(scope="session")
def sample_image():
    return syn.render(angle_deg=0.0, seed=100, **SAMPLE_SHAPE)
