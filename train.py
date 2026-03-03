import argparse
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

import numpy as np
import yaml
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.dino_env import DinoEnv


TIMESTAMP_FMT = "%Y%m%d_%H%M%S"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(headless: bool, reward_mode: str, game_url: str, config: dict):
    def _init():
        return DinoEnv(
            headless=headless,
            auto_focus=True,
            game_url=game_url,
            frame_size=tuple(config["env"]["frame_size"]),
            frame_stack=int(config["env"]["frame_stack"]),
            action_repeat=int(config["env"]["action_repeat"]),
            max_episode_seconds=int(config["env"]["max_episode_seconds"]),
            reward_mode=reward_mode,
        )

    return _init


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def normalize_model_prefix(path_arg: str) -> Path:
    model_prefix = Path(path_arg)
    if model_prefix.suffix == ".zip":
        return model_prefix.with_suffix("")
    return model_prefix


def parse_run_timestamp(run_dir: Path, model_prefix: Path):
    prefix = f"{model_prefix.name}_"
    if not run_dir.name.startswith(prefix):
        return None
    ts = run_dir.name[len(prefix) :]
    try:
        return datetime.strptime(ts, TIMESTAMP_FMT)
    except ValueError:
        return None


def build_timestamped_run_dir(model_prefix: Path) -> Path:
    ts = datetime.now().strftime(TIMESTAMP_FMT)
    return model_prefix.parent / f"{model_prefix.name}_{ts}"


def iter_run_dirs(model_prefix: Path):
    pattern = f"{model_prefix.name}_*"
    for candidate in model_prefix.parent.glob(pattern):
        if not candidate.is_dir():
            continue
        if parse_run_timestamp(candidate, model_prefix) is None:
            continue
        yield candidate


def find_latest_saved_model(model_prefix: Path) -> Path | None:
    latest = None
    latest_mtime = -1.0
    for run_dir in iter_run_dirs(model_prefix):
        model_path = run_dir / "model.zip"
        if not model_path.exists():
            continue
        mtime = model_path.stat().st_mtime
        if mtime > latest_mtime:
            latest_mtime = mtime
            latest = model_path
    return latest


def find_latest_checkpoint(model_prefix: Path) -> Path | None:
    latest = None
    latest_mtime = -1.0
    for run_dir in iter_run_dirs(model_prefix):
        checkpoint_dir = run_dir / "checkpoints"
        if not checkpoint_dir.exists():
            continue
        for ckpt in checkpoint_dir.glob("*.zip"):
            mtime = ckpt.stat().st_mtime
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest = ckpt
    return latest


def latest_checkpoint_in_run(run_dir: Path) -> Path | None:
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return None
    latest = None
    latest_mtime = -1.0
    for ckpt in checkpoint_dir.glob("*.zip"):
        mtime = ckpt.stat().st_mtime
        if mtime > latest_mtime:
            latest_mtime = mtime
            latest = ckpt
    return latest


def sync_saved_models_from_checkpoints(model_prefix: Path, logger: logging.Logger) -> None:
    for run_dir in iter_run_dirs(model_prefix):
        latest_ckpt = latest_checkpoint_in_run(run_dir)
        if latest_ckpt is None:
            continue
        model_path = run_dir / "model.zip"
        needs_update = (not model_path.exists()) or (
            model_path.stat().st_mtime < latest_ckpt.stat().st_mtime
        )
        if needs_update:
            shutil.copy2(latest_ckpt, model_path)
            logger.info(
                "Synced stale saved model from checkpoint: %s <- %s",
                model_path,
                latest_ckpt,
            )


def ensure_saved_models_in_checkpoints(model_prefix: Path, logger: logging.Logger) -> None:
    for run_dir in iter_run_dirs(model_prefix):
        model_path = run_dir / "model.zip"
        if not model_path.exists():
            continue
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        mirror_path = checkpoint_dir / "saved_model_latest.zip"
        needs_update = (not mirror_path.exists()) or (
            mirror_path.stat().st_mtime < model_path.stat().st_mtime
        )
        if needs_update:
            shutil.copy2(model_path, mirror_path)
            logger.info(
                "Mirrored saved model into checkpoints: %s <- %s",
                mirror_path,
                model_path,
            )


def pick_resume_source(model_prefix: Path, logger: logging.Logger) -> Path | None:
    latest_saved = find_latest_saved_model(model_prefix)
    latest_ckpt = find_latest_checkpoint(model_prefix)

    if latest_saved and latest_ckpt:
        if latest_ckpt.stat().st_mtime >= latest_saved.stat().st_mtime:
            logger.info("Resume source selected (checkpoint): %s", latest_ckpt)
            return latest_ckpt
        logger.info("Resume source selected (saved model): %s", latest_saved)
        return latest_saved
    if latest_ckpt:
        logger.info("Resume source selected (checkpoint): %s", latest_ckpt)
        return latest_ckpt
    if latest_saved:
        logger.info("Resume source selected (saved model): %s", latest_saved)
        return latest_saved
    return None


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    return logger


def create_new_model(env, config: dict):
    return PPOWithTrainMetrics(
        "CnnPolicy",
        env,
        learning_rate=config["training"]["learning_rate"],
        n_steps=config["training"]["n_steps"],
        batch_size=config["training"]["batch_size"],
        gamma=config["training"]["gamma"],
        gae_lambda=config["training"]["gae_lambda"],
        ent_coef=config["training"]["ent_coef"],
        clip_range=config["training"]["clip_range"],
        n_epochs=config["training"]["n_epochs"],
        vf_coef=config["training"]["vf_coef"],
        max_grad_norm=config["training"]["max_grad_norm"],
        tensorboard_log=config["training"]["tensorboard_log"],
        verbose=1,
    )


class PPOWithTrainMetrics(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latest_train_metrics = {}
        self.progress_logger: logging.Logger | None = None
        self.progress_total_timesteps_target = 0
        self.progress_start_timesteps = 0

    def train(self) -> None:
        super().train()
        logger_values = getattr(self.logger, "name_to_value", {})
        metric_keys = [
            ("train/policy_gradient_loss", "pg_loss"),
            ("train/value_loss", "v_loss"),
            ("train/entropy_loss", "ent_loss"),
            ("train/approx_kl", "approx_kl"),
            ("train/clip_fraction", "clip_frac"),
            ("train/loss", "total_loss"),
        ]
        metrics = {}
        for key, label in metric_keys:
            value = logger_values.get(key)
            if value is None:
                continue
            try:
                metrics[label] = float(value)
            except (TypeError, ValueError):
                continue
        self.latest_train_metrics = metrics
        if metrics and self.progress_logger is not None:
            progress_prefix = ""
            target = int(getattr(self, "progress_total_timesteps_target", 0) or 0)
            if target > 0:
                start = int(getattr(self, "progress_start_timesteps", 0) or 0)
                current = max(0, int(self.num_timesteps) - start)
                pct = min(100.0, (100.0 * current) / target)
                progress_prefix = (
                    f"{pct:6.2f}% ({current}/{target}) "
                    f"| total_steps {int(self.num_timesteps)} | "
                )
            parts = [f"{name} {value:.4f}" for name, value in metrics.items()]
            self.progress_logger.info("[ppo] %s%s", progress_prefix, " | ".join(parts))


class TrainingProgressCallback(BaseCallback):
    def __init__(
        self,
        total_timesteps: int,
        progress_interval_pct: float = 5.0,
        start_timesteps: int = 0,
        logger: logging.Logger | None = None,
        episode_image_dir: Path | None = None,
    ) -> None:
        super().__init__()
        self.total_timesteps_target = max(1, int(total_timesteps))
        self.start_timesteps = max(0, int(start_timesteps))
        pct = max(0.1, float(progress_interval_pct))
        self.interval_steps = max(1, int(self.total_timesteps_target * (pct / 100.0)))
        self.next_report_step = self.interval_steps
        self.start_time = 0.0
        self.episodes_completed = 0
        self.last_episode_reward = None
        self.last_episode_length = None
        self.action_space_n: int | None = None
        self.n_envs = 1
        self.current_episode_action_counts: list[dict[int, int]] = []
        self.last_episode_action_counts: dict[int, int] | None = None
        self.last_episode_action_total = 0
        self.progress_logger = logger
        self.episode_image_dir = episode_image_dir

    def _log(self, message: str) -> None:
        if self.progress_logger is not None:
            self.progress_logger.info(message)
        else:
            print(message, flush=True)

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        setattr(self.model, "progress_total_timesteps_target", int(self.total_timesteps_target))
        setattr(self.model, "progress_start_timesteps", int(self.start_timesteps))
        action_space = getattr(self.model, "action_space", None)
        n_actions = getattr(action_space, "n", None)
        if isinstance(n_actions, int) and n_actions > 0:
            self.action_space_n = n_actions
        else:
            self.action_space_n = None
        vec_env = getattr(self.model, "env", None)
        self.n_envs = int(getattr(vec_env, "num_envs", 1) or 1)
        self.current_episode_action_counts = [self._new_action_counter() for _ in range(self.n_envs)]
        self.last_episode_action_counts = None
        self.last_episode_action_total = 0
        self._log(
            f"[progress] 0.00% (0/{self.total_timesteps_target}) | episodes {self.episodes_completed}"
        )

    def _on_step(self) -> bool:
        current_total = int(self.num_timesteps)
        actions = self.locals.get("actions", None)
        if actions is None:
            actions = self.locals.get("clipped_actions", None)
        self._update_action_stats(actions)
        infos = self.locals.get("infos", [])
        new_obs = self.locals.get("new_obs")
        for info_idx, info in enumerate(infos):
            if not isinstance(info, dict) or "episode" not in info:
                continue
            ep = info["episode"]
            self.episodes_completed += 1
            if isinstance(ep, dict):
                if "r" in ep:
                    self.last_episode_reward = float(ep["r"])
                if "l" in ep:
                    self.last_episode_length = int(ep["l"])
            self._finalize_episode_action_stats(info_idx)
            self._save_episode_random_image(
                info,
                new_obs,
                self.episodes_completed,
                info_idx,
                current_total,
            )

        current = max(0, current_total - self.start_timesteps)
        if current < self.next_report_step and current < self.total_timesteps_target:
            return True

        elapsed = max(1e-9, time.time() - self.start_time)
        pct = min(100.0, (current / self.total_timesteps_target) * 100.0)
        steps_per_sec = current / elapsed
        remaining_steps = max(0, self.total_timesteps_target - current)
        eta_seconds = remaining_steps / max(1e-9, steps_per_sec)

        metric_parts = []
        ep_rew_mean = None
        ep_len_mean = None

        if self.model.ep_info_buffer:
            rewards = [float(item["r"]) for item in self.model.ep_info_buffer if "r" in item]
            lengths = [float(item["l"]) for item in self.model.ep_info_buffer if "l" in item]
            if rewards:
                ep_rew_mean = mean(rewards)
                metric_parts.append(f"ep_rew_mean {ep_rew_mean:.2f}")
            if lengths:
                ep_len_mean = mean(lengths)
                metric_parts.append(f"ep_len_mean {ep_len_mean:.1f}")

        if self.last_episode_reward is not None:
            metric_parts.append(f"last_ep_rew {self.last_episode_reward:.2f}")
        if self.last_episode_length is not None:
            metric_parts.append(f"last_ep_len {self.last_episode_length}")
        metric_parts.append(self._format_episode_action_stats())

        metrics_str = " | ".join(metric_parts) if metric_parts else "metrics pending"
        self._log(
            f"[progress] {pct:6.2f}% ({current}/{self.total_timesteps_target}) "
            f"| total_steps {current_total} "
            f"| episodes {self.episodes_completed} "
            f"| {steps_per_sec:.2f} steps/s | elapsed {format_seconds(elapsed)} "
            f"| eta {format_seconds(eta_seconds)} "
            f"| {metrics_str}"
        )

        while self.next_report_step <= current:
            self.next_report_step += self.interval_steps
        return True

    def _on_training_end(self) -> None:
        elapsed = time.time() - self.start_time
        self._log(
            f"[progress] 100.00% ({self.total_timesteps_target}/{self.total_timesteps_target}) "
            f"| episodes {self.episodes_completed} "
            f"| elapsed {format_seconds(elapsed)} "
            f"| {self._format_episode_action_stats()}"
        )

    def _update_action_stats(self, actions) -> None:
        if actions is None:
            return
        try:
            arr = np.asarray(actions).reshape(-1)
        except Exception:
            return
        if len(arr) != len(self.current_episode_action_counts):
            self.current_episode_action_counts = [self._new_action_counter() for _ in range(len(arr))]
        for env_idx, value in enumerate(arr):
            try:
                action = int(value)
            except (TypeError, ValueError):
                continue
            counts = self.current_episode_action_counts[env_idx]
            counts[action] = counts.get(action, 0) + 1

    def _new_action_counter(self) -> dict[int, int]:
        if self.action_space_n is not None:
            return {action: 0 for action in range(self.action_space_n)}
        return {}

    def _finalize_episode_action_stats(self, env_idx: int) -> None:
        if env_idx < 0 or env_idx >= len(self.current_episode_action_counts):
            return
        counts = self.current_episode_action_counts[env_idx]
        self.last_episode_action_counts = dict(counts)
        self.last_episode_action_total = int(sum(counts.values()))
        self.current_episode_action_counts[env_idx] = self._new_action_counter()

    def _format_episode_action_stats(self) -> str:
        if not self.last_episode_action_counts or self.last_episode_action_total <= 0:
            return "last_ep_actions n/a"
        total = float(self.last_episode_action_total)
        if self.action_space_n is not None:
            parts = []
            for action in range(self.action_space_n):
                count = self.last_episode_action_counts.get(action, 0)
                pct = (100.0 * count) / total
                parts.append(f"a{action} {pct:.1f}%")
            return f"last_ep_actions ({int(total)}): " + ", ".join(parts)
        parts = []
        for action in sorted(self.last_episode_action_counts):
            count = self.last_episode_action_counts[action]
            pct = (100.0 * count) / total
            parts.append(f"a{action} {pct:.1f}%")
        return f"last_ep_actions ({int(total)}): " + ", ".join(parts)

    def _save_episode_random_image(
        self,
        info: dict,
        new_obs,
        episode_idx: int,
        env_idx: int | None = None,
        total_steps: int | None = None,
    ) -> None:
        if self.episode_image_dir is None:
            return
        try:
            obs_source = info.get("terminal_observation", None)
            if obs_source is None:
                obs_source = new_obs
            if obs_source is None:
                return

            obs_arr = np.asarray(obs_source)
            if obs_arr.ndim == 4:
                # (n_envs, frame_stack, H, W)
                if env_idx is None or env_idx < 0 or env_idx >= obs_arr.shape[0]:
                    env_idx = int(np.random.randint(0, obs_arr.shape[0]))
                frames = obs_arr[env_idx]
                frame_idx = int(np.random.randint(0, frames.shape[0]))
                frame = frames[frame_idx]
            elif obs_arr.ndim == 3:
                # (frame_stack, H, W)
                frame_idx = int(np.random.randint(0, obs_arr.shape[0]))
                frame = obs_arr[frame_idx]
            elif obs_arr.ndim == 2:
                frame = obs_arr
            else:
                return

            frame_u8 = frame.astype(np.uint8, copy=False)
            self.episode_image_dir.mkdir(parents=True, exist_ok=True)
            name_parts = [f"episode_{episode_idx:06d}"]
            if total_steps is not None:
                name_parts.append(f"steps_{int(total_steps):09d}")
            if env_idx is not None and env_idx >= 0:
                name_parts.append(f"env_{int(env_idx)}")
            episode_info = info.get("episode")
            if isinstance(episode_info, dict):
                ep_len = episode_info.get("l")
                ep_rew = episode_info.get("r")
                try:
                    if ep_len is not None:
                        name_parts.append(f"len_{int(ep_len)}")
                except (TypeError, ValueError):
                    pass
                try:
                    if ep_rew is not None:
                        name_parts.append(f"rew_{float(ep_rew):.2f}")
                except (TypeError, ValueError):
                    pass
            image_path = self.episode_image_dir / ("_".join(name_parts) + ".png")
            Image.fromarray(frame_u8, mode="L").save(image_path)
        except Exception as err:
            self._log(f"[warn] Failed to save random episode image: {err}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="dino")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--model_path", default="models/dino_ppo")
    parser.add_argument("--reward_mode", default=None)
    parser.add_argument("--game_url", default=None)
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", default="configs/default_dino.yaml")
    parser.add_argument("--progress_interval_pct", type=float, default=0.1)
    parser.add_argument("--checkpoint_freq", type=int, default=10000)
    parser.add_argument("--ent_coef", type=float, default=None)
    parser.add_argument("--new", action="store_true")
    args = parser.parse_args()

    if args.env != "dino":
        raise ValueError("Only --env dino is supported in this starter.")

    config = load_config(Path(args.config))
    reward_mode = args.reward_mode or config["env"]["reward_mode"]
    game_url = args.game_url or config["env"]["game_url"]
    timesteps = args.timesteps or int(config["training"]["timesteps"])
    headless = not args.show
    if args.show and args.n_envs > 1:
        raise ValueError("--show supports only --n_envs 1 to avoid multiple headed browser windows.")

    os.makedirs("models", exist_ok=True)
    model_prefix = normalize_model_prefix(args.model_path)
    model_prefix.parent.mkdir(parents=True, exist_ok=True)

    run_dir = build_timestamped_run_dir(model_prefix)
    run_dir.mkdir(parents=True, exist_ok=True)

    model_output_path = run_dir / "model.zip"
    checkpoint_freq = int(args.checkpoint_freq)
    checkpoint_callback_freq = max(1, checkpoint_freq // max(1, args.n_envs))
    checkpoint_dir = run_dir / "checkpoints"
    best_dir = run_dir / "best"
    eval_log_dir = run_dir / "eval_logs"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_timestamp = datetime.now().strftime(TIMESTAMP_FMT)
    train_log_path = logs_dir / f"{log_timestamp}_train.log"

    logger = setup_logger(train_log_path)
    logger.info("Model prefix: %s", model_prefix)
    logger.info("Run dir: %s", run_dir)
    logger.info("Model output: %s", model_output_path)
    logger.info("Train log: %s", train_log_path)
    logger.info("Best model dir: %s", best_dir)
    if checkpoint_freq > 0:
        logger.info(
            "Checkpoints: %s (target every %s timesteps, callback every %s steps)",
            checkpoint_dir,
            checkpoint_freq,
            checkpoint_callback_freq,
        )

    env = make_vec_env(
        make_env(headless, reward_mode, game_url, config),
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=DummyVecEnv,
    )
    eval_env = make_vec_env(
        make_env(True, reward_mode, game_url, config),
        n_envs=1,
        seed=args.seed,
        vec_env_cls=DummyVecEnv,
    )

    if not args.new:
        sync_saved_models_from_checkpoints(model_prefix, logger)
        ensure_saved_models_in_checkpoints(model_prefix, logger)

    resume_source = None if args.new else pick_resume_source(model_prefix, logger)
    should_resume = resume_source is not None

    if should_resume:
        logger.info("Resuming from latest artifact: %s", resume_source)
        model = PPOWithTrainMetrics.load(
            str(resume_source),
            env=env,
            tensorboard_log=config["training"]["tensorboard_log"],
            device="auto",
        )
    elif args.new:
        logger.info("--new specified: starting a fresh model")
        model = create_new_model(env, config)
    else:
        logger.info("No prior model or checkpoint found: starting a fresh model")
        model = create_new_model(env, config)

    if args.ent_coef is not None:
        model.ent_coef = float(args.ent_coef)
        logger.info("Overriding entropy coefficient: ent_coef=%s", model.ent_coef)

    model.progress_logger = logger

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(eval_log_dir),
        eval_freq=int(config["training"]["eval_freq"]),
        n_eval_episodes=int(config["training"]["eval_episodes"]),
        deterministic=True,
        render=False,
    )

    callbacks = [eval_callback]
    if checkpoint_freq > 0:
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_callback_freq,
            save_path=str(checkpoint_dir),
            name_prefix="ckpt",
        )
        callbacks.append(checkpoint_callback)

    progress_callback = TrainingProgressCallback(
        total_timesteps=timesteps,
        progress_interval_pct=args.progress_interval_pct,
        start_timesteps=int(model.num_timesteps),
        logger=logger,
        episode_image_dir=logs_dir,
    )
    callbacks.append(progress_callback)
    callback = CallbackList(callbacks)

    model.learn(
        total_timesteps=timesteps,
        callback=callback,
        reset_num_timesteps=not should_resume,
    )

    model.save(str(model_output_path))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_output_path, checkpoint_dir / "saved_model_latest.zip")
    logger.info("Saved model: %s", model_output_path)
    logger.info(
        "Saved model mirrored in checkpoints: %s",
        checkpoint_dir / "saved_model_latest.zip",
    )


if __name__ == "__main__":
    main()
