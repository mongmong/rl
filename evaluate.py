import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from envs.dino_env import DinoEnv


TIMESTAMP_FMT = "%Y%m%d_%H%M%S"


def normalize_model_prefix(path_arg: str) -> Path:
    path = Path(path_arg)
    if path.suffix == ".zip":
        return path.with_suffix("")
    return path


def parse_run_timestamp(run_dir: Path, model_prefix: Path):
    prefix = f"{model_prefix.name}_"
    if not run_dir.name.startswith(prefix):
        return None
    ts = run_dir.name[len(prefix) :]
    try:
        return datetime.strptime(ts, TIMESTAMP_FMT)
    except ValueError:
        return None


def resolve_model_path(path_arg: str) -> Path:
    path = Path(path_arg)

    if path.suffix == ".zip":
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        return path

    # Direct run directory path
    if path.is_dir():
        direct = path / "model.zip"
        if direct.exists():
            return direct
        raise FileNotFoundError(f"Run directory does not contain model.zip: {path}")

    # Prefix mode: choose latest timestamped run dir and load model.zip
    prefix = normalize_model_prefix(path_arg)
    latest_run = None
    latest_dt = None
    for candidate in prefix.parent.glob(f"{prefix.name}_*"):
        if not candidate.is_dir():
            continue
        dt = parse_run_timestamp(candidate, prefix)
        if dt is None:
            continue
        model_zip = candidate / "model.zip"
        if not model_zip.exists():
            continue
        if latest_dt is None or dt > latest_dt:
            latest_dt = dt
            latest_run = candidate

    if latest_run is not None:
        return latest_run / "model.zip"

    # Legacy fallback: timestamped zip in parent
    latest_zip = None
    latest_mtime = -1.0
    for candidate in prefix.parent.glob(f"{prefix.name}_*.zip"):
        mtime = candidate.stat().st_mtime
        if mtime > latest_mtime:
            latest_mtime = mtime
            latest_zip = candidate
    if latest_zip is not None:
        return latest_zip

    raise FileNotFoundError(
        f"No model found for path/prefix: {path_arg}. "
        f"Expected either a .zip file, a run dir with model.zip, "
        f"or a prefix with timestamped run directories."
    )


def sample_action_with_temperature(model: PPO, obs, temperature: float) -> int:
    if temperature <= 0:
        raise ValueError("--sample_temperature must be > 0")

    policy = model.policy
    obs_tensor, _ = policy.obs_to_tensor(obs)
    with torch.no_grad():
        dist = policy.get_distribution(obs_tensor).distribution
        logits = dist.logits / float(temperature)
        probs = torch.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

    return int(np.asarray(sampled.cpu().numpy()).reshape(-1)[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/dino_ppo")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--reward_mode", default="survival")
    parser.add_argument("--game_url", default="https://elgoog.im/t-rex/")
    parser.add_argument("--sample_temperature", type=float, default=None)
    args = parser.parse_args()

    env = DinoEnv(
        headless=args.headless,
        reward_mode=args.reward_mode,
        game_url=args.game_url,
    )
    resolved_model_path = resolve_model_path(args.model_path)
    print(f"Using model: {resolved_model_path}")
    model = PPO.load(str(resolved_model_path))

    rewards = []
    lengths = []
    action_counts = Counter()

    for _ in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        steps = 0
        while not done:
            if args.sample_temperature is None:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
            else:
                action = sample_action_with_temperature(
                    model,
                    obs,
                    args.sample_temperature,
                )
            action_counts[action] += 1
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            steps += 1
            done = terminated or truncated
        rewards.append(total)
        lengths.append(steps)

    rewards_arr = np.array(rewards, dtype=np.float32)
    lengths_arr = np.array(lengths, dtype=np.int32)

    print("Evaluation Results")
    print("------------------")
    print(f"Episodes: {args.episodes}")
    if args.sample_temperature is None:
        print("Policy mode: deterministic")
    else:
        print(f"Policy mode: stochastic (temperature={args.sample_temperature})")
    print(f"Mean reward: {rewards_arr.mean():.2f}")
    print(f"Median reward: {np.median(rewards_arr):.2f}")
    print(f"Best episode reward: {rewards_arr.max():.2f}")
    print(f"Mean episode length: {lengths_arr.mean():.1f} steps")
    print("Action distribution:")
    for action, count in sorted(action_counts.items()):
        print(f"  action {action}: {count}")

    env.close()


if __name__ == "__main__":
    main()
