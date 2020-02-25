import numpy as np

from .config import Config
from .model_zero import ModelZero

def mcts_learn(model_config_path=None, weight_path=None):

    config = Config()
    config.learn_func = 'mcts_learn'
    qc = config.Qlearn

    total_reward_vec = np.zeros(qc.num_consecutive_iterations)  # 各試行の報酬を格納
    # Qネットワークとメモリ、Actorの生成--------------------------------------------------------
    if model_config_path is None or weight_path is None:
        mainNN_plus = ModelZero(config)     # メインのQネットワーク
        mainNN_minus = ModelZero(config)     # メインのQネットワーク
        mainNN_plus.build()
        mainNN_minus.build()
    else:
        mainNN_plus = ModelZero(config)
        mainNN_minus = ModelZero(config)
        success_load = mainNN_plus.load(model_config_path, weight_path)
        if not success_load:
            raise FileNotFoundError(
                f"{model_config_path} {weight_path}が読み込めませんでした")

        mainNN_minus.load(model_config_path, weight_path)
        config.pre_trained = weight_path

    

    
