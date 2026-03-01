import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from envs.dino_env import DinoEnv


TIMESTAMP_FMT = "%Y%m%d_%H%M%S"


def normalize_model_prefix(path_arg: str) -> Path:
    path = Path(path_arg)
    if path.suffix == ".zip":
        return path.with_suffix("")
    return path


def resolve_model_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if path.suffix == ".zip":
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        return path

    prefix = normalize_model_prefix(path_arg)
    latest_path = None
    latest_mtime = -1.0
    pattern = f"{prefix.name}_*.zip"
    for candidate in prefix.parent.glob(pattern):
        suffix = candidate.stem[len(prefix.name) + 1 :]
        try:
            datetime.strptime(suffix, TIMESTAMP_FMT)
        except ValueError:
            continue
        mtime = candidate.stat().st_mtime
        if mtime > latest_mtime:
            latest_mtime = mtime
            latest_path = candidate

    if latest_path is None:
        raise FileNotFoundError(
            f"No timestamped model found for prefix: {prefix} (expected {prefix.name}_YYYYMMDD_HHMMSS.zip)"
        )
    return latest_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/dino_ppo")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--reward_mode", default="survival")
    parser.add_argument("--game_url", default="https://elgoog.im/t-rex/")
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
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
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
