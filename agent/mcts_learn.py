from typing import List
import pickle
import datetime
import json
import numpy as np
import time
import os

from .config import Config
from .model_zero import ModelZero

kifu_folders = []
kifu_folders = ['results/bekasa/2020-08-20-01-22/kifu',
                'results/bekasa/2020-08-20-10-41/kifu']

def mcts_learn(kifus: List[str], config=None, model_config_path=None, weight_path=None, beta=1.0, weight_reduction = 0.1, folder = "results/bababax/models"):

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

    wps = np.empty(shape=(0, ))
    weights = np.empty(shape=(0, ))
    pi_mcts = np.empty(shape=(0, 315))
    board_logs = np.empty(shape=(0, 7, 5))
    plus_turns = np.empty(shape=(0, ))

    begin_time = time.time()
    for path in kifus:
        # with open(path, 'rb') as f:
        #     kifu = pickle.load(f)
        kifu = np.load(path)
        wp = kifu['wp']
        if len(wp) == 0:
            continue
        wps = np.append(wps, wp)
        pi_mcts = np.append(pi_mcts, kifu['pi_mcts'], axis = 0)
        board_logs = np.append(board_logs, kifu['board'], axis = 0)
        plus_turns = np.append(plus_turns, kifu['plus_turn'])
        weights *= (1 - weight_reduction)
        weights = np.append(weights, np.ones(shape=(len(wp), )))
    print(f'preprocess time: {int((time.time() - begin_time) * 1000)}ms')

    if len(wps) > 0:
        pi_mcts = pi_mcts / pi_mcts.sum(axis=1).reshape((len(wps), 1))
        begin_time = time.time()
        mainNN.replay(wps, pi_mcts, board_logs, plus_turns, weights, len(wps), beta)
        print(f'learn time: {int((time.time() - begin_time) * 1000)}ms')

    return save_model(mainNN, config, folder)


def save_model(mainNN, config: Config, folder):
    d = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    model_config_path = folder + f"/{d}-mainNN.json"
    weight_path = folder + f"/{d}-mainNN.h5"
    config_path = folder + f"/{d}-config.json"
    mainNN.save(model_config_path,
                weight_path)
    with open(config_path, 'x') as f:
        json.dump(config._to_dict(), f, indent=4)
    return (model_config_path, weight_path, config_path)


if __name__ == "__main__":
    config = Config(temperature=10000., n_playout=150, c_puct=1.4, ignore_draw=False)

    d = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    basedir = f'results/bekasa/{d}'
    os.makedirs(basedir + '/kifu')
    os.makedirs(basedir + '/models')

    paths = []
    for kifu_folder in kifu_folders:
        paths.extend([kifu_folder + '/' + kifu_file for kifu_file in os.listdir(kifu_folder)])

    model_config, weight, _ = mcts_learn(paths, config, folder=(basedir + '/models'))
