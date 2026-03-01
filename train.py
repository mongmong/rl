import argparse
import csv
import os
import shutil
import time
from datetime import datetime
from statistics import mean
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
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


def build_timestamped_model_path(model_prefix: Path) -> Path:
    ts = datetime.now().strftime(TIMESTAMP_FMT)
    return model_prefix.with_name(f"{model_prefix.name}_{ts}.zip")


def find_latest_timestamped_model(model_prefix: Path) -> Path | None:
    pattern = f"{model_prefix.name}_*.zip"
    latest_path = None
    latest_dt = None
    for path in model_prefix.parent.glob(pattern):
        suffix = path.stem[len(model_prefix.name) + 1 :]
        try:
            dt = datetime.strptime(suffix, TIMESTAMP_FMT)
        except ValueError:
            continue
        if latest_dt is None or dt > latest_dt:
            latest_dt = dt
            latest_path = path
    return latest_path


def find_latest_saved_model_by_mtime(model_prefix: Path) -> Path | None:
    latest_path = None
    latest_mtime = -1.0
    pattern = f"{model_prefix.name}_*.zip"
    for path in model_prefix.parent.glob(pattern):
        suffix = path.stem[len(model_prefix.name) + 1 :]
        try:
            datetime.strptime(suffix, TIMESTAMP_FMT)
        except ValueError:
            continue
        mtime = path.stat().st_mtime
        if mtime > latest_mtime:
            latest_mtime = mtime
            latest_path = path
    return latest_path


def find_latest_checkpoint_for_prefix(model_prefix: Path) -> Path | None:
    latest_ckpt = None
    latest_mtime = -1.0
    dir_pattern = f"{model_prefix.name}_*_checkpoints"
    for ckpt_dir in model_prefix.parent.glob(dir_pattern):
        if not ckpt_dir.is_dir():
            continue
        for ckpt_file in ckpt_dir.glob("*.zip"):
            mtime = ckpt_file.stat().st_mtime
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_ckpt = ckpt_file
    return latest_ckpt


def sync_saved_models_from_checkpoints(model_prefix: Path) -> list[tuple[Path, Path]]:
    updates = []
    dir_pattern = f"{model_prefix.name}_*_checkpoints"
    for ckpt_dir in model_prefix.parent.glob(dir_pattern):
        if not ckpt_dir.is_dir():
            continue
        latest_ckpt = None
        latest_mtime = -1.0
        for ckpt_file in ckpt_dir.glob("*.zip"):
            mtime = ckpt_file.stat().st_mtime
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_ckpt = ckpt_file
        if latest_ckpt is None:
            continue
        if not ckpt_dir.name.endswith("_checkpoints"):
            continue
        run_stem = ckpt_dir.name[: -len("_checkpoints")]
        saved_model_path = ckpt_dir.parent / f"{run_stem}.zip"
        needs_update = (
            (not saved_model_path.exists())
            or saved_model_path.stat().st_mtime < latest_ckpt.stat().st_mtime
        )
        if needs_update:
            shutil.copy2(latest_ckpt, saved_model_path)
            updates.append((saved_model_path, latest_ckpt))
    return updates


def ensure_saved_models_in_checkpoints(model_prefix: Path) -> list[tuple[Path, Path]]:
    updates = []
    pattern = f"{model_prefix.name}_*.zip"
    for saved_model_path in model_prefix.parent.glob(pattern):
        suffix = saved_model_path.stem[len(model_prefix.name) + 1 :]
        try:
            datetime.strptime(suffix, TIMESTAMP_FMT)
        except ValueError:
            continue
        checkpoint_dir = saved_model_path.parent / f"{saved_model_path.stem}_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        mirrored_path = checkpoint_dir / f"{saved_model_path.stem}_saved_model.zip"
        needs_update = (
            (not mirrored_path.exists())
            or mirrored_path.stat().st_mtime < saved_model_path.stat().st_mtime
        )
        if needs_update:
            shutil.copy2(saved_model_path, mirrored_path)
            updates.append((mirrored_path, saved_model_path))
    return updates


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
        if metrics:
            parts = [f"{name} {value:.4f}" for name, value in metrics.items()]
            print(f"[ppo] {' | '.join(parts)}", flush=True)


class TrainingProgressCallback(BaseCallback):
    def __init__(
        self,
        total_timesteps: int,
        progress_interval_pct: float = 5.0,
        start_timesteps: int = 0,
        metrics_csv_path: str | None = None,
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
        self.metrics_csv_path = metrics_csv_path
        self.metrics_file = None
        self.metrics_writer = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        if self.metrics_csv_path:
            path = Path(self.metrics_csv_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            file_exists = path.exists() and path.stat().st_size > 0
            self.metrics_file = path.open("a", newline="", encoding="utf-8")
            self.metrics_writer = csv.DictWriter(
                self.metrics_file,
                fieldnames=[
                    "timestamp",
                    "run_steps",
                    "total_steps",
                    "target_steps",
                    "percent",
                    "episodes",
                    "steps_per_sec",
                    "elapsed_s",
                    "eta_s",
                    "ep_rew_mean",
                    "ep_len_mean",
                    "last_ep_rew",
                    "last_ep_len",
                    "pg_loss",
                    "v_loss",
                    "ent_loss",
                    "approx_kl",
                    "clip_frac",
                    "total_loss",
                ],
            )
            if not file_exists:
                self.metrics_writer.writeheader()
                self.metrics_file.flush()
        print(
            f"[progress] 0.00% (0/{self.total_timesteps_target}) | episodes 0",
            flush=True,
        )

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if not isinstance(info, dict) or "episode" not in info:
                continue
            ep = info["episode"]
            self.episodes_completed += 1
            if isinstance(ep, dict):
                if "r" in ep:
                    self.last_episode_reward = float(ep["r"])
                if "l" in ep:
                    self.last_episode_length = int(ep["l"])

        current_total = int(self.num_timesteps)
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
        latest_train_metrics = getattr(self.model, "latest_train_metrics", {})
        for key in ["pg_loss", "v_loss", "ent_loss", "approx_kl", "clip_frac", "total_loss"]:
            if key in latest_train_metrics:
                metric_parts.append(f"{key} {latest_train_metrics[key]:.4f}")
        metrics_str = " | ".join(metric_parts) if metric_parts else "metrics pending"
        print(
            f"[progress] {pct:6.2f}% ({current}/{self.total_timesteps_target}) "
            f"| total_steps {current_total} "
            f"| episodes {self.episodes_completed} "
            f"| {steps_per_sec:.2f} steps/s | elapsed {format_seconds(elapsed)} "
            f"| eta {format_seconds(eta_seconds)} "
            f"| {metrics_str}",
            flush=True,
        )
        if self.metrics_writer:
            self.metrics_writer.writerow(
                {
                    "timestamp": int(time.time()),
                    "run_steps": current,
                    "total_steps": current_total,
                    "target_steps": self.total_timesteps_target,
                    "percent": round(pct, 4),
                    "episodes": self.episodes_completed,
                    "steps_per_sec": round(steps_per_sec, 6),
                    "elapsed_s": round(elapsed, 3),
                    "eta_s": round(eta_seconds, 3),
                    "ep_rew_mean": None if ep_rew_mean is None else round(ep_rew_mean, 6),
                    "ep_len_mean": None if ep_len_mean is None else round(ep_len_mean, 6),
                    "last_ep_rew": self.last_episode_reward,
                    "last_ep_len": self.last_episode_length,
                    "pg_loss": latest_train_metrics.get("pg_loss"),
                    "v_loss": latest_train_metrics.get("v_loss"),
                    "ent_loss": latest_train_metrics.get("ent_loss"),
                    "approx_kl": latest_train_metrics.get("approx_kl"),
                    "clip_frac": latest_train_metrics.get("clip_frac"),
                    "total_loss": latest_train_metrics.get("total_loss"),
                }
            )
            self.metrics_file.flush()

        while self.next_report_step <= current:
            self.next_report_step += self.interval_steps
        return True

    def _on_training_end(self) -> None:
        elapsed = time.time() - self.start_time
        print(
            f"[progress] 100.00% ({self.total_timesteps_target}/{self.total_timesteps_target}) "
            f"| episodes {self.episodes_completed} "
            f"| elapsed {format_seconds(elapsed)}",
            flush=True,
        )
        if self.metrics_file:
            self.metrics_file.close()
            self.metrics_file = None
            self.metrics_writer = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="dino")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--model_path", default="models/dino_ppo")
    parser.add_argument("--reward_mode", default=None)
    parser.add_argument("--game_url", default=None)
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", default="configs/default_dino.yaml")
    parser.add_argument("--progress_interval_pct", type=float, default=0.1)
    parser.add_argument("--progress_metrics_suffix", default="_progress_metrics.csv")
    parser.add_argument("--checkpoint_freq", type=int, default=10000)
    parser.add_argument("--new", action="store_true")
    args = parser.parse_args()

    if args.env != "dino":
        raise ValueError("Only --env dino is supported in this starter.")

    config = load_config(Path(args.config))
    reward_mode = args.reward_mode or config["env"]["reward_mode"]
    game_url = args.game_url or config["env"]["game_url"]
    timesteps = args.timesteps or int(config["training"]["timesteps"])

    os.makedirs("models", exist_ok=True)
    model_prefix = normalize_model_prefix(args.model_path)
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    model_output_path = build_timestamped_model_path(model_prefix)
    metrics_csv_path = model_output_path.with_name(
        f"{model_output_path.stem}{args.progress_metrics_suffix}"
    )
    checkpoint_freq = int(args.checkpoint_freq)
    checkpoint_callback_freq = max(1, checkpoint_freq // max(1, args.n_envs))
    checkpoint_dir = model_output_path.parent / f"{model_output_path.stem}_checkpoints"
    best_dir = model_output_path.parent / f"{model_output_path.stem}_best"
    eval_log_dir = model_output_path.parent / f"{model_output_path.stem}_eval_logs"
    print(f"[train] Model prefix: {model_prefix}", flush=True)
    print(f"[train] Run model output: {model_output_path}", flush=True)
    print(f"[train] Progress CSV: {metrics_csv_path}", flush=True)
    print(f"[train] Best model dir: {best_dir}", flush=True)
    if checkpoint_freq > 0:
        print(
            f"[train] Checkpoints: {checkpoint_dir} "
            f"(target every {checkpoint_freq} timesteps, callback every {checkpoint_callback_freq} steps)",
            flush=True,
        )

    env = make_vec_env(
        make_env(args.headless, reward_mode, game_url, config),
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

    sync_updates = sync_saved_models_from_checkpoints(model_prefix)
    for saved_model, source_ckpt in sync_updates:
        print(
            f"[train] Synced stale saved model from checkpoint: {saved_model} <- {source_ckpt}",
            flush=True,
        )
    checkpoint_mirror_updates = ensure_saved_models_in_checkpoints(model_prefix)
    for mirrored_model, saved_model in checkpoint_mirror_updates:
        print(
            f"[train] Mirrored saved model into checkpoints: {mirrored_model} <- {saved_model}",
            flush=True,
        )

    latest_saved_model = None if args.new else find_latest_saved_model_by_mtime(model_prefix)
    latest_checkpoint = None if args.new else find_latest_checkpoint_for_prefix(model_prefix)

    resume_source = None
    if latest_checkpoint and latest_saved_model:
        if latest_checkpoint.stat().st_mtime >= latest_saved_model.stat().st_mtime:
            resume_source = latest_checkpoint
        else:
            resume_source = latest_saved_model
    elif latest_checkpoint:
        resume_source = latest_checkpoint
    elif latest_saved_model:
        resume_source = latest_saved_model

    should_resume = resume_source is not None
    if should_resume:
        print(f"[train] Resuming from latest artifact: {resume_source}", flush=True)
        model = PPOWithTrainMetrics.load(
            str(resume_source),
            env=env,
            tensorboard_log=config["training"]["tensorboard_log"],
            device="auto",
        )
    elif args.new:
        print("[train] --new specified: starting a fresh model.", flush=True)
        model = create_new_model(env, config)
    else:
        print("[train] No prior model or checkpoint found: starting a fresh model.", flush=True)
        model = create_new_model(env, config)

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
            name_prefix=f"{model_output_path.stem}_ckpt",
        )
        callbacks.append(checkpoint_callback)
    progress_callback = TrainingProgressCallback(
        total_timesteps=timesteps,
        progress_interval_pct=args.progress_interval_pct,
        start_timesteps=int(model.num_timesteps),
        metrics_csv_path=str(metrics_csv_path),
    )
    callbacks.append(progress_callback)
    callback = CallbackList(callbacks)

    model.learn(total_timesteps=timesteps, callback=callback, reset_num_timesteps=not should_resume)
    model.save(str(model_output_path))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint_copy = checkpoint_dir / f"{model_output_path.stem}_saved_model.zip"
    shutil.copy2(model_output_path, final_checkpoint_copy)
    print(f"[train] Saved model: {model_output_path}", flush=True)
    print(f"[train] Saved model mirrored in checkpoints: {final_checkpoint_copy}", flush=True)


if __name__ == "__main__":
    main()
