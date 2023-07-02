
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from typing import Dict
from ray.rllib.policy import Policy
from json import load
with open("./experiments/default/config.json", "r", encoding='utf-8') as f:
    config = load(f)


class TradeCallbacks(DefaultCallbacks):
    """
        trade environment callbacks (similar to Keras callbacks).
        This callback is used for custom metrics and custom postprocessing.
    """
    def on_episode_end(self,
                       *,
                       worker: RolloutWorker,
                       base_env: BaseEnv,
                       policies: Dict[str, Policy],
                       episode: Episode,
                       env_index: int,
                       **kwargs
                       ):
        symbols = episode._last_infos["agent0"]["symbols"]
        episode.custom_metrics["sharpe_ratio"] = episode._last_infos["agent0"]["sharpe_ratio"]
        episode.custom_metrics["portfolio_value"] = episode._last_infos["agent0"]["portfolio_value"]
