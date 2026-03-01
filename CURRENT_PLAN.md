# Reproducible Plan: RL Browser Game Trainer (Chrome Dino)

## Goal
Build and run a reinforcement learning trainer that learns to play a browser Dino game from pixels using Playwright + Stable-Baselines3 PPO, with robust resume/recovery behavior and operational outputs (checkpoints, best models, progress metrics).

## Implementation Snapshot
This plan reflects the generated program currently in this repo:
- Environment: `envs/dino_env.py` (`DinoEnv`, Gymnasium API)
- Trainer: `train.py` (PPO + callbacks + recovery logic)
- Evaluator: `evaluate.py` (loads exact model or latest by prefix)
- Config: `configs/default_dino.yaml`
- Package manager: `uv` (`pyproject.toml`, `uv.lock`)

## Stack and Defaults
- Python + `uv`
- RL algorithm: SB3 PPO with `CnnPolicy`
- Browser automation: Playwright Chromium
- Default game URL: `https://elgoog.im/t-rex/` (headless-accessible)
- Observation: grayscale resized frames, stacked channels
- Action space: discrete `{0: noop, 1: jump, 2: duck}`

## Environment Spec (must match)
Implement `DinoEnv` with:
1. Constructor args:
- `headless: bool`
- `game_url: str`
- `frame_size: tuple[int, int]`
- `frame_stack: int`
- `action_repeat: int`
- `max_episode_seconds: int`
- `reward_mode: Literal["survival", "distance", "hybrid"]`
- `seed: Optional[int]`

2. Spaces:
- `action_space = Discrete(3)`
- `observation_space = Box(0,255,(frame_stack,H,W),uint8)`

3. Reset:
- launch browser if not running
- open `game_url`
- find game canvas (`canvas#gameCanvas`, fallback `canvas`)
- press space to start
- initialize frame stack and counters

4. Step:
- execute selected action
- run `action_repeat` internal mini-steps with Playwright waits
- capture canvas screenshot and preprocess
- compute reward by mode
- detect termination via JS runner state, fallback pixel heuristic
- truncate by max episode wall time
- return `(obs, reward, terminated, truncated, info)`

5. JS state access:
- attempt `window.Runner.instance_` for crash + distance
- if unavailable, use fallback game-over heuristic

## Training Program Spec (`train.py`)
Implement CLI:
- `--env` (only `dino` supported)
- `--timesteps`
- `--headless`
- `--model_path` (PREFIX semantics, not a fixed file)
- `--reward_mode`
- `--game_url`
- `--n_envs`
- `--seed`
- `--config`
- `--progress_interval_pct`
- `--progress_metrics_suffix`
- `--checkpoint_freq`
- `--new`

### Critical output naming semantics
Treat `--model_path` as prefix (e.g., `models/dino_ppo`).
Per run, generate timestamped output stem:
- `PREFIX_YYYYMMDD_HHMMSS.zip` (run model)
- `..._checkpoints/`
- `..._best/` (EvalCallback best)
- `..._eval_logs/`
- `..._progress_metrics.csv`

### Resume/recovery behavior (must match)
1. Default behavior (without `--new`):
- auto-discover freshest artifact among:
  - latest saved timestamped model
  - latest checkpoint zip in any matching checkpoint folder
- resume from whichever is newer by mtime

2. Early-stop recovery:
- at startup, for each run folder, if latest checkpoint is newer than its saved model (or saved model missing), copy checkpoint over saved model path for that run

3. Saved-model/checkpoint consistency:
- mirror saved models into matching checkpoint folders (`..._saved_model.zip` copy)
- after training ends, copy final saved model into the current run checkpoint folder

4. `--new`:
- force fresh model initialization, bypass auto-resume

### PPO model construction
- Use custom subclass `PPOWithTrainMetrics` (extends `PPO`)
- Override `train()` to capture latest optimizer metrics from SB3 logger:
  - `pg_loss`, `v_loss`, `ent_loss`, `approx_kl`, `clip_frac`, `total_loss`
- print these metrics when available

### Callbacks
Use `CallbackList` containing:
1. `EvalCallback`:
- save best model to run-specific `..._best/`
- eval logs to run-specific `..._eval_logs/`

2. `CheckpointCallback`:
- save periodic checkpoints into run-specific `..._checkpoints/`
- interpret `--checkpoint_freq` as timesteps
- convert to callback frequency with `n_envs` adjustment:
  - `save_freq = max(1, checkpoint_freq // n_envs)`

3. `TrainingProgressCallback`:
- print progress: percent, run steps, total steps, episodes, steps/s, elapsed, ETA
- include episode metrics: rolling reward/length + last episode
- include latest PPO metrics from model subclass
- write same metrics to CSV (`..._progress_metrics.csv`) and flush each write

## Evaluation Program Spec (`evaluate.py`)
Implement CLI:
- `--model_path`
- `--episodes`
- `--headless`
- `--reward_mode`
- `--game_url`

Model resolution rules:
- if `--model_path` ends with `.zip`: load exact file
- else: treat as prefix and load latest timestamped model by mtime

Output metrics:
- mean reward
- median reward
- best episode reward
- mean episode length
- action distribution

## Config Contract (`configs/default_dino.yaml`)
Keep two top-level sections:
1. `env`
- `game_url`
- `frame_size`
- `frame_stack`
- `action_repeat`
- `max_episode_seconds`
- `reward_mode`

2. `training`
- PPO hyperparameters (`learning_rate`, `n_steps`, `batch_size`, etc.)
- `timesteps`
- `eval_freq`, `eval_episodes`
- `save_freq` (still present; CLI may override)
- `tensorboard_log`

## Dependency and Runtime Contract
Use `uv` only:
1. `uv venv`
2. `uv sync`
3. `uv run python -m playwright install chromium`
4. run training/eval via `uv run ...`

Include in dependencies:
- `gymnasium`, `numpy`, `Pillow`, `playwright`, `stable-baselines3`, `torch`, `PyYAML`, `tensorboard`, compatible `setuptools`.

## Operational Expectations
1. Training should not fail when a run is interrupted and restarted.
2. Resume should prefer the freshest artifact (checkpoint or model).
3. Progress logs should surface both RL trajectory stats and PPO optimizer stats.
4. Outputs should be timestamped and isolated per run.

## Acceptance Checklist for Another AI
1. `uv run python train.py --help` shows all expected flags (`--new`, `--checkpoint_freq`, progress flags).
2. Starting training prints model prefix, run output path, checkpoint path, and best-model path.
3. Checkpoints appear during training under `..._checkpoints/`.
4. Progress CSV is created and appended during training.
5. Restart without `--new` resumes from freshest artifact.
6. `uv run python evaluate.py --model_path models/dino_ppo` auto-loads latest timestamped model.
7. README documents prefix/timestamp semantics and resume/recovery rules.

## Known Tradeoffs
- Browser-driven pixel RL is throughput-limited (low steps/s relative to simulated envs).
- Cloudflare-protected Dino mirrors may fail in headless mode; default URL should remain headless-accessible.
- SB3 rollout chunking means first optimizer metrics appear only after initial rollout completes.
