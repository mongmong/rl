# RL Chrome Dino

This repo trains a reinforcement learning agent to play the Chrome Dino game in the browser using Playwright and Stable-Baselines3.

## Setup (uv)

1. Create a Python environment and sync dependencies:

```bash
uv venv
uv sync
uv run python -m playwright install chromium
```

## Training

```bash
uv run python train.py --env dino --headless --timesteps 200000
```

Models are saved under `models/`.
`--model_path` is treated as a model prefix, and each run saves a timestamped model file.
For `--model_path models/dino_ppo`, one run outputs:
- `models/dino_ppo_YYYYMMDD_HHMMSS.zip` as the run model
- `models/dino_ppo_YYYYMMDD_HHMMSS_checkpoints/` for periodic checkpoints
- `models/dino_ppo_YYYYMMDD_HHMMSS_best/` for eval-best model (`best_model.zip`)
- `models/dino_ppo_YYYYMMDD_HHMMSS_progress_metrics.csv` for progress metrics CSV

By default, training auto-resumes from the freshest artifact matching the prefix (saved model or checkpoint).
Startup also auto-recovers interrupted runs:
- if a checkpoint is newer than its saved model (or saved model is missing), the saved model is refreshed from the newest checkpoint
- resume source is chosen from the freshest artifact across saved models and checkpoints
- saved models are mirrored into their checkpoint folders
Use `--new` to force a fresh model even if previous runs exist:

```bash
uv run python train.py --env dino --headless --timesteps 50000 --model_path models/dino_ppo --new
```

Control checkpoint frequency (timesteps):

```bash
uv run python train.py --env dino --headless --timesteps 200000 --checkpoint_freq 10000
```

Checkpoint note:
- `--checkpoint_freq` is interpreted in timesteps.
- with `n_envs > 1`, internal callback frequency is adjusted so checkpoint cadence stays close to the requested timestep interval.

## Evaluation

```bash
uv run python evaluate.py --model_path models/dino_ppo --episodes 10 --headless
```

Evaluation model path behavior:
- if `--model_path` ends with `.zip`, that exact file is loaded
- otherwise it is treated as a prefix and the latest timestamped model is loaded

## Configuration

Default settings live in `configs/default_dino.yaml`. You can change frame size, reward mode, game URL, or PPO hyperparameters there or via CLI flags (`--game_url`, `--reward_mode`, etc.).

## Target Game

Default target is `https://elgoog.im/t-rex/` because it is accessible in headless automation. You can point to another Dino URL with `--game_url` or by changing `env.game_url` in the config.
