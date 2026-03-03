# RL Chrome Dino

This repo trains a reinforcement learning agent to play Chrome Dino in a browser using Playwright + Stable-Baselines3 PPO.

## Setup (uv)

```bash
uv venv
uv sync
uv run python -m playwright install chromium
```

## Training

```bash
uv run python train.py --env dino --timesteps 200000 --model_path models/dino_ppo
```

`--model_path` is a prefix. Each run creates a timestamped run folder:

- `models/dino_ppo_YYYYMMDD_HHMMSS/`

Inside that run folder:

- `model.zip` (final saved model for the run)
- `logs/YYYYMMDD_HHMMSS_train.log` (all training logs via Python `logging`)
- `logs/episode_XXXXXX_steps_XXXXXXXXX_...png` (one sampled observation frame per completed episode, with step/episode metadata in filename)
- `checkpoints/` (periodic checkpoints + mirrored saved model)
- `best/` (eval-best `best_model.zip`, if produced)
- `eval_logs/` (EvalCallback logs)

Exploration override (applies to new or resumed models):

```bash
uv run python train.py --env dino --model_path models/dino_ppo --ent_coef 0.02
```

## Resume Behavior

Default (without `--new`):

- auto-discovers latest artifact from prior runs for the same prefix
- resumes from the freshest source among saved models and checkpoints
- repairs stale/missing saved models from newer checkpoints
- mirrors saved models into checkpoint folders

Force fresh training even when prior runs exist:

```bash
uv run python train.py --env dino --timesteps 50000 --model_path models/dino_ppo --new
```

## Checkpoint Frequency

```bash
uv run python train.py --env dino --timesteps 200000 --checkpoint_freq 10000
```

`--checkpoint_freq` is interpreted as timesteps. With `n_envs > 1`, callback frequency is adjusted to keep checkpoint cadence close to requested timestep intervals.

## Cleanup

Dry-run cleanup of run folders missing both `model.zip` and checkpoint zips:

```bash
uv run python scripts/clean_models.py
```

Delete them:

```bash
uv run python scripts/clean_models.py --yes
```

## Evaluation

```bash
uv run python evaluate.py --model_path models/dino_ppo --episodes 10
```

Optional stochastic evaluation with sampling temperature:

```bash
uv run python evaluate.py --model_path models/dino_ppo --episodes 10 --sample_temperature 1.5
```

`train.py` defaults to headless and supports `--show` (headed mode, `--n_envs` must be `1`).
`evaluate.py` defaults to a visible browser and supports `--headless`.

`evaluate.py` model resolution:

- if `--model_path` ends with `.zip`: load exact file
- if `--model_path` is a run directory: load `<run_dir>/model.zip`
- otherwise: treat as prefix and load latest timestamped run directory model

## Configuration

Defaults are in `configs/default_dino.yaml`.

Key env config:

- `game_url`
- `frame_size`
- `frame_stack`
- `action_repeat`
- `max_episode_seconds`
- `reward_mode`

Key training config:

- PPO hyperparameters (`learning_rate`, `n_steps`, `batch_size`, etc.)
- `timesteps`
- `save_freq`, `eval_freq`, `eval_episodes`
- `tensorboard_log`

## Target Game

Default target is `https://elgoog.im/t-rex/` (works in headless automation). Override with `--game_url`.
