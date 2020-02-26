from typing import List
import numpy as np
import pickle

from .config import Config
from .model_zero import ModelZero

def mcts_learn(kifus: List[str], model_config_path=None, weight_path=None):

    config = Config()
    config.learn_func = 'mcts_learn'
    qc = config.Qlearn

    total_reward_vec = np.zeros(qc.num_consecutive_iterations)  # 各試行の報酬を格納
    # Qネットワークとメモリ、Actorの生成--------------------------------------------------------
    if model_config_path is None or weight_path is None:
        mainNN = ModelZero(config)     # メインのQネットワーク
        mainNN.build()

    else:
        mainNN = ModelZero(config)
        success_load = mainNN.load(model_config_path, weight_path)
        if not success_load:
            raise FileNotFoundError(
                f"{model_config_path} {weight_path}が読み込めませんでした")

        config.pre_trained = weight_path

    for path in kifus:
        # with open(path, 'rb') as f:
        #     kifu = pickle.load(f)
        kifu = np.load(path)
        wps = kifu['wp']
        pi_mcts = kifu['pi_mcts']
        board_logs = kifu['board']
        plus_turns = kifu['plus_turn']
        mainNN.replay(wps, pi_mcts, board_logs, plus_turns, len(wps), 1.0)


if __name__ == "__main__":
    mcts_learn(['results/bababax/kifu/2020-02-26-11-42-06.npz'])        
            

    

    
