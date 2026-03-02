# Student Reproduction Prompt Sequence

Purpose: use these prompts step-by-step to reproduce the program evolution from a minimal RL browser-game trainer to the current robust version.

## Step 1: Build baseline program
1. `PLEASE IMPLEMENT THIS PLAN: ...` (the full RL Chrome Dino plan with `envs/dino_env.py`, `train.py`, `evaluate.py`, config, tests, README)

## Step 2: Environment and dependency management
2. `use uv to manage the packages and dependencies`

## Step 3: Training observability
3. `add training progress to train.py`
4. `OK, add episodes`
5. `yes, print them on screen`
6. `modify the logging format, move the ppo related model metrics to separate line whe model training happens, no need to append to the progress line`

## Step 4: Resume and checkpoint capabilities
7. `add continue training options to pick up latest model to continue training`
8. `save training progress metrics to the same models folder with different suffix name`
9. `save checkpoints during learning including the internal best models`
10. `set default checkpoint_freq`
11. `remove --resume and add --new. change the logic to resume by default unless there is no model found. the model path should only specifies the prefix, add datetime stamp as model suffix. unless --new is specified, continue training on the model with latest timestamp`
12. `automatically discover early stops (models are not saved in such case.), resume from checkpoints instead of saved models. if stale save models discovered, update the saved models with latest checkpionts. make sure the saved models can always be found in check points.`

## Step 5: Logging and output structure refactor
13. `use logging module to output logs to files. restructure the output folder structure to put every related output under the model + timestamp folder`
14. `remove the _progress_metrics output, as the log is good enough. put logs in logs/ foder with log files starts with timestamps`
15. `put logs under the model folder`
16. `put logs under logs sub folder under the model folder`

## Step 6: Headless behavior refinement
17. `set --headless option default`
18. `it is not neccessary to have --headless option. the train.py should by default running at headless mode`
19. `no, when --no-headless is provide, it should create head mode. I want you to use --no-headless to repliace --headless in opposite way`

## Step 7: Robust runtime error handling
20. (From runtime failure report) `playwright._impl._errors.Error: Protocol error (Page.captureScreenshot): Unable to capture screenshot`
21. Follow-up fix request intent: make screenshot capture resilient and avoid training crash.

## Optional maintenance prompts used during refinement
- `review the code and readme.md to discover any discrepancy. add descriptive detail to readme if neccessary`
- `check any redundant logic after last couple changes`

## Suggested classroom usage
- Use one prompt per checkpoint/commit.
- Run a short smoke training after each feature step.
- Keep a changelog mapping each prompt to files modified.
