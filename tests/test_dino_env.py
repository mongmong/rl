import io

import numpy as np
import pytest
from PIL import Image


playwright = pytest.importorskip("playwright.sync_api", reason="playwright not installed")


def test_preprocess_from_bytes():
    from envs.dino_env import DinoEnv

    env = DinoEnv(headless=True)
    image = Image.fromarray(np.zeros((200, 300, 3), dtype=np.uint8))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    frame = env._preprocess_from_bytes(buf.getvalue())
    assert frame.shape == (env.config.frame_size[1], env.config.frame_size[0])
    assert frame.dtype == np.uint8
