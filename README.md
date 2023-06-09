# BULL-AI

## Installation
`pip install ta yfinance wandb`
`pip install -U "ray[rllib]" tensorflow torch`
## Training

`python train.py`

### To view the results in tensorboard
`tensorboard --logdir=~/ray_results --bind_all`

Also we have logged the results into `wandb` to view the reports.
## Experiments
`config.json` is for environment and trainer configuration parameters.
## Upcoming
- optimize the observations
- enable high-res trading strategies
-