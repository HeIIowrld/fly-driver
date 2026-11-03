"""Golden hashes recorded before geometry caching; protect the receptor input contract."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from flybrain_drive.ac_pixel_retina import NativePixelRetina


@pytest.mark.parametrize('region,expected', [
    ([0, 0, 320, 180], 'bab09f80a7b5a84304ac2c6a60fead2752745fa59c6f250ccc5f09deb9117ba9'),
    ([8, 7, 119, 105], 'c3a06a7ccf36690e16520dd776053c3a353376b11145958f29b462d479265678'),
])
def test_cached_pixel_sampling_preserves_original_full_and_cropped_receptors(region, expected):
    config = json.loads(Path('configs/ac-csp-retina.json').read_text())['native_retina']
    retina = NativePixelRetina(config)
    pixels = np.random.default_rng(862834).integers(0, 256, size=(180, 320, 3), dtype=np.uint8)
    for _ in range(2):
        assert hashlib.sha256(retina._sample(pixels, region).tobytes()).hexdigest() == expected
