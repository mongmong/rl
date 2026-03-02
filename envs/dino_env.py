import io
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Literal, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from PIL import Image

try:
    from playwright.sync_api import sync_playwright
except Exception:  # pragma: no cover - optional dependency
    sync_playwright = None


RewardMode = Literal["survival", "distance", "hybrid"]


@dataclass
class DinoEnvConfig:
    headless: bool = True
    game_url: str = "https://elgoog.im/t-rex/"
    frame_size: Tuple[int, int] = (84, 84)
    frame_stack: int = 4
    action_repeat: int = 4
    max_episode_seconds: int = 120
    reward_mode: RewardMode = "survival"
    seed: Optional[int] = None


class DinoEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        headless: bool = True,
        game_url: str = "https://elgoog.im/t-rex/",
        frame_size: Tuple[int, int] = (84, 84),
        frame_stack: int = 4,
        action_repeat: int = 4,
        max_episode_seconds: int = 120,
        reward_mode: RewardMode = "survival",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        if sync_playwright is None:
            raise ImportError(
                "playwright is required for DinoEnv. Install with `pip install playwright` "
                "and run `playwright install`.")

        self.config = DinoEnvConfig(
            headless=headless,
            game_url=game_url,
            frame_size=frame_size,
            frame_stack=frame_stack,
            action_repeat=action_repeat,
            max_episode_seconds=max_episode_seconds,
            reward_mode=reward_mode,
            seed=seed,
        )

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.config.frame_stack, self.config.frame_size[1], self.config.frame_size[0]),
            dtype=np.uint8,
        )

        self._np_random, _ = seeding.np_random(self.config.seed)

        self._playwright = None
        self._browser = None
        self._page = None
        self._canvas = None
        self._frame_buffer: Deque[np.ndarray] = deque(maxlen=self.config.frame_stack)

        self._episode_start = 0.0
        self._last_distance = 0.0

    def _ensure_browser(self) -> None:
        if self._browser is not None:
            return
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.config.headless)
        self._page = self._browser.new_page()

    def _load_game(self) -> None:
        assert self._page is not None
        self._page.goto(self.config.game_url, wait_until="domcontentloaded")
        self._page.wait_for_timeout(500)
        self._refresh_canvas()
        if self._canvas is None:
            raise RuntimeError("Could not find game canvas on the page.")

        # Ensure game is ready
        self._page.keyboard.press("Space")
        self._page.wait_for_timeout(200)

    def _refresh_canvas(self) -> None:
        assert self._page is not None
        self._canvas = self._page.query_selector("canvas#gameCanvas")
        if self._canvas is None:
            self._canvas = self._page.query_selector("canvas")

    def _get_js_state(self) -> Tuple[Optional[bool], Optional[float]]:
        assert self._page is not None
        script = """
        () => {
            try {
                const r = window.Runner && window.Runner.instance_;
                if (!r) return { crashed: null, distance: null };
                const crashed = r.crashed === true || r.gameOver === true;
                let distance = null;
                if (r.distanceRan !== undefined) distance = r.distanceRan;
                else if (r.distanceMeter && r.distanceMeter.getActualDistance) {
                    distance = r.distanceMeter.getActualDistance();
                }
                return { crashed, distance };
            } catch (e) {
                return { crashed: null, distance: null };
            }
        }
        """
        state = self._page.evaluate(script)
        crashed = state.get("crashed")
        distance = state.get("distance")
        return crashed, distance

    def _screenshot_frame(self) -> np.ndarray:
        assert self._page is not None
        last_err = None
        for _ in range(3):
            try:
                if self._canvas is None:
                    self._refresh_canvas()
                if self._canvas is None:
                    raise RuntimeError("Canvas handle is unavailable.")
                png_bytes = self._canvas.screenshot()
                return self._preprocess_from_bytes(png_bytes)
            except Exception as err:
                last_err = err
                self._refresh_canvas()
                self._page.wait_for_timeout(50)

        # Fallback to full-page screenshot if canvas capture keeps failing.
        try:
            png_bytes = self._page.screenshot()
            return self._preprocess_from_bytes(png_bytes)
        except Exception as err:
            raise RuntimeError(f"Unable to capture observation frame: {err}") from last_err

    def _preprocess_from_bytes(self, png_bytes: bytes) -> np.ndarray:
        image = Image.open(io.BytesIO(png_bytes))
        image = image.convert("L")
        image = image.resize(self.config.frame_size, Image.BILINEAR)
        frame = np.array(image, dtype=np.uint8)
        return frame

    def _pixel_game_over(self, frame: np.ndarray) -> bool:
        # Heuristic: look for "GAME OVER" text area (top center).
        h, w = frame.shape
        y0 = int(h * 0.2)
        y1 = int(h * 0.35)
        x0 = int(w * 0.25)
        x1 = int(w * 0.75)
        roi = frame[y0:y1, x0:x1]
        mean = float(roi.mean())
        var = float(roi.var())
        # When text appears, variance spikes and mean darkens.
        return mean < 220 and var > 400

    def _get_observation(self) -> np.ndarray:
        frame = self._screenshot_frame()
        self._frame_buffer.append(frame)
        while len(self._frame_buffer) < self.config.frame_stack:
            self._frame_buffer.append(frame)
        stacked = np.stack(list(self._frame_buffer), axis=0)
        return stacked

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self._np_random, _ = seeding.np_random(seed)

        self._ensure_browser()
        self._load_game()

        self._frame_buffer.clear()
        self._episode_start = time.time()
        self._last_distance = 0.0

        obs = None
        reset_err = None
        for _ in range(3):
            try:
                obs = self._get_observation()
                break
            except Exception as err:
                reset_err = err
                self._load_game()
        if obs is None:
            raise RuntimeError(f"Failed to capture initial observation: {reset_err}") from reset_err
        info = {"distance": 0.0}
        return obs, info

    def step(self, action: int):
        assert self._page is not None

        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for _ in range(self.config.action_repeat):
            if action == 1:
                self._page.keyboard.press("Space")
            elif action == 2:
                self._page.keyboard.down("ArrowDown")
                self._page.wait_for_timeout(50)
                self._page.keyboard.up("ArrowDown")

            self._page.wait_for_timeout(50)

            crashed, distance = self._get_js_state()

            reward = 1.0
            if self.config.reward_mode in ("distance", "hybrid") and distance is not None:
                delta = float(distance) - float(self._last_distance)
                if self.config.reward_mode == "distance":
                    reward = max(delta, 0.0)
                else:
                    reward = 1.0 + 0.01 * max(delta, 0.0)
                self._last_distance = float(distance)

            total_reward += reward

            try:
                obs = self._get_observation()
            except Exception as err:
                # End the episode gracefully instead of crashing training.
                terminated = True
                info = {"distance": distance, "obs_error": str(err)}
                if self._frame_buffer:
                    last = self._frame_buffer[-1]
                    obs = np.stack([last] * self.config.frame_stack, axis=0)
                else:
                    obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
                break
            info = {"distance": distance}

            if crashed is True:
                terminated = True
                break
            if crashed is None and self._pixel_game_over(obs[-1]):
                terminated = True
                break

            if time.time() - self._episode_start >= self.config.max_episode_seconds:
                truncated = True
                break

        return obs, total_reward, terminated, truncated, info

    def render(self):
        if self._page is None:
            return None
        return self._get_observation()[-1]

    def close(self):
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None
