import atexit
import io
import time
from collections import deque
from dataclasses import dataclass
from threading import Lock
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


_SHARED_PLAYWRIGHT = None
_SHARED_PLAYWRIGHT_LOCK = Lock()


def _get_shared_playwright():
    if sync_playwright is None:
        raise ImportError(
            "playwright is required for DinoEnv. Install with `pip install playwright` "
            "and run `playwright install`."
        )
    global _SHARED_PLAYWRIGHT
    with _SHARED_PLAYWRIGHT_LOCK:
        if _SHARED_PLAYWRIGHT is None:
            _SHARED_PLAYWRIGHT = sync_playwright().start()
        return _SHARED_PLAYWRIGHT


def _stop_shared_playwright() -> None:
    global _SHARED_PLAYWRIGHT
    with _SHARED_PLAYWRIGHT_LOCK:
        if _SHARED_PLAYWRIGHT is None:
            return
        try:
            _SHARED_PLAYWRIGHT.stop()
        except Exception:
            pass
        _SHARED_PLAYWRIGHT = None


atexit.register(_stop_shared_playwright)


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
        self._canvas_clip = None
        self._last_focus_check = 0.0
        self._frame_buffer: Deque[np.ndarray] = deque(maxlen=self.config.frame_stack)

        self._episode_start = 0.0
        self._last_distance = 0.0

    def _ensure_browser(self) -> None:
        if self._browser is not None:
            return
        self._playwright = _get_shared_playwright()
        launch_args = [
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
        ]
        self._browser = self._playwright.chromium.launch(
            headless=self.config.headless,
            args=launch_args,
        )
        self._page = self._browser.new_page()

    def _teardown_browser(self) -> None:
        if self._browser is not None:
            try:
                self._browser.close()
            except Exception:
                pass
        self._browser = None
        self._page = None
        self._canvas = None
        self._canvas_clip = None

    def _start_episode_state(self) -> None:
        self._frame_buffer.clear()
        self._episode_start = time.time()
        self._last_distance = 0.0
        self._last_focus_check = 0.0

    def _restart_environment(self, max_attempts: int = 2) -> bool:
        last_err = None
        for _ in range(max_attempts):
            try:
                self._teardown_browser()
                self._ensure_browser()
                self._load_game()
                self._start_episode_state()
                return True
            except Exception as err:
                last_err = err
                self._teardown_browser()
        return False

    def _blank_observation(self) -> np.ndarray:
        if self._frame_buffer:
            last = self._frame_buffer[-1]
            return np.stack([last] * self.config.frame_stack, axis=0)
        return np.zeros(self.observation_space.shape, dtype=np.uint8)

    def _load_game(self) -> None:
        assert self._page is not None
        self._page.goto(self.config.game_url, wait_until="domcontentloaded")
        self._page.wait_for_timeout(500)
        self._refresh_canvas()
        if self._canvas is None:
            raise RuntimeError("Could not find game canvas on the page.")

        self._ensure_page_active(force=True)
        # Ensure game is ready
        self._page.keyboard.press("Space")
        self._page.wait_for_timeout(200)
        self._resume_runner_if_needed()

    def _resume_runner_if_needed(self) -> None:
        """
        Best-effort unpause/start guard for headed mode.
        Some pages pause the Runner after focus changes.
        """
        if self._page is None:
            return
        try:
            self._page.evaluate(
                """
                () => {
                    const r = window.Runner && window.Runner.instance_;
                    if (!r) return;
                    if (r.crashed === true || r.gameOver === true) return;
                    if (r.paused === true && typeof r.play === "function") {
                        r.play();
                    }
                    if (r.playing === false && typeof r.play === "function") {
                        r.play();
                    }
                    if (r.activated === false && typeof r.startGame === "function") {
                        r.playingIntro = false;
                        r.startGame();
                    }
                }
                """
            )
        except Exception:
            pass

    def _ensure_page_active(self, force: bool = False) -> None:
        """
        Keep headed evaluation responsive by re-focusing the tab/canvas if needed.
        """
        if self.config.headless:
            return
        assert self._page is not None
        now = time.time()
        if not force and now - self._last_focus_check < 1.0:
            return
        self._last_focus_check = now

        try:
            if not self._page.evaluate("() => document.hasFocus()"):
                self._page.bring_to_front()
                self._refresh_canvas()
                if self._canvas is not None:
                    bbox = self._canvas.bounding_box()
                    if bbox is not None:
                        cx = float(bbox["x"] + bbox["width"] / 2.0)
                        cy = float(bbox["y"] + bbox["height"] / 2.0)
                        self._page.mouse.click(cx, cy)
            self._resume_runner_if_needed()
        except Exception:
            # Best-effort only; training/eval should continue even if focus check fails.
            pass

    def _refresh_canvas(self) -> None:
        assert self._page is not None
        self._canvas = None
        self._canvas_clip = None

        preferred_selectors = [
            "canvas#gameCanvas",
            "canvas#runner-canvas",
            "canvas#runner",
        ]

        for selector in preferred_selectors:
            candidate = self._page.query_selector(selector)
            if candidate is None:
                continue
            bbox = candidate.bounding_box()
            if bbox is None:
                continue
            if bbox["width"] < 120 or bbox["height"] < 20:
                continue
            self._canvas = candidate
            self._canvas_clip = bbox
            return

        # Fallback: choose the largest visible canvas on the page.
        best = None
        best_bbox = None
        best_area = -1.0
        for candidate in self._page.query_selector_all("canvas"):
            bbox = candidate.bounding_box()
            if bbox is None:
                continue
            if bbox["width"] < 120 or bbox["height"] < 20:
                continue
            area = bbox["width"] * bbox["height"]
            if area > best_area:
                best = candidate
                best_bbox = bbox
                best_area = area
        self._canvas = best
        self._canvas_clip = best_bbox

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

        # Fallback to clipped page screenshot using the last known canvas bbox.
        try:
            if self._canvas_clip is not None:
                clip = {
                    "x": float(self._canvas_clip["x"]),
                    "y": float(self._canvas_clip["y"]),
                    "width": float(self._canvas_clip["width"]),
                    "height": float(self._canvas_clip["height"]),
                }
                png_bytes = self._page.screenshot(clip=clip)
            else:
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

        reset_err = None
        for _ in range(3):
            try:
                self._ensure_browser()
                self._load_game()
                self._start_episode_state()
                obs = self._get_observation()
                info = {"distance": 0.0}
                return obs, info
            except Exception as err:
                reset_err = err
                self._teardown_browser()

        raise RuntimeError(f"Failed to reset environment after retries: {reset_err}") from reset_err

    def step(self, action: int):
        if self._page is None:
            restarted = self._restart_environment()
            obs = self._blank_observation()
            info = {"distance": None, "env_restarted": restarted, "obs_error": "Page not initialized"}
            return obs, 0.0, True, False, info

        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        obs = self._blank_observation()
        self._ensure_page_active()

        for _ in range(self.config.action_repeat):
            distance = None
            try:
                self._resume_runner_if_needed()
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

                obs = self._get_observation()
            except Exception as err:
                restarted = self._restart_environment()
                info = {"distance": distance, "obs_error": str(err), "env_restarted": restarted}
                terminated = True
                obs = self._blank_observation()
                break

            total_reward += reward
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
        self._teardown_browser()
