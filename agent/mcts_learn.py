from typing import List
import pickle
import datetime
import json
import numpy as np

from .config import Config
from .model_zero import ModelZero


def mcts_learn(kifus: List[str], config=None, model_config_path=None, weight_path=None, beta=1.0):

    if config is None:
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
        mainNN.replay(wps, pi_mcts, board_logs, plus_turns, len(wps), beta)

    return save_model(mainNN, config)


def save_model(mainNN, config: Config):
    d = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    model_config_path = f"results/bababax/models/{d}-mainNN.json"
    weight_path = f"results/bababax/models/{d}-mainNN.h5"
    config_path = f"results/bababax/models/{d}-config.json"
    mainNN.save(model_config_path,
                weight_path)
    with open(config_path, 'x') as f:
        json.dump(config._to_dict(), f, indent=4)
    return (model_config_path, weight_path, config_path)



if __name__ == "__main__":
    model_config, weight, _ = mcts_learn(['results/bababax/kifu/2020-02-26-11-42-06.npz'])
