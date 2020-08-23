from typing import *
import time
import datetime
from collections import deque
import hashlib
import json
import os
from logging import getLogger
import numpy as np
from tqdm import tqdm, trange
# noinspection PyPep8Naming
import keras.backend as K
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.losses import mean_squared_error
from keras.optimizers import Adam
from keras.regularizers import l2

from .config import Config, QLearnConfig
from game.game_state import GameState, Winner


logger = getLogger(__name__)


class Memory:
    """Experience ReplayとFixed Target Q-Networkを実現するメモリクラス"""

    def __init__(self, max_size=1000) -> None:
        self.buffer = deque(maxlen=max_size)

    def add(self, experience: Any) -> None:
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> list:
        indices = np.random.choice(
            np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in indices]

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.digest = None

    def build(self) -> None:
        mc = self.config.model
        in_x = x = Input((7, 5, 3))

        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_last", kernel_regularizer=l2(mc.l2_reg))(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation("relu")(x)

        for _ in range(mc.res_layer_num):
            x = self._build_residual_block(x)

        res_out = x
        # for policy output
        x = Conv2D(filters=2, kernel_size=1, data_format="channels_last",
                   kernel_regularizer=l2(mc.l2_reg))(res_out)
        x = BatchNormalization(axis=3)(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        # no output for 'pass'
        out = Dense(315, kernel_regularizer=l2(mc.l2_reg),
                    activation="softmax", name="out")(x)

        # x = Dense(mc.value_fc_size, kernel_regularizer=l2(mc.l2_reg),
        #          activation="relu")(x)
        # value_out = Dense(1, kernel_regularizer=l2(mc.l2_reg),
        #                   activation="tanh", name="value_out")(x)

        self.model = Model(in_x, out, name="slipe_model")
        self.model.compile(loss='mse', optimizer=Adam(lr=mc.learning_rate))
        self.model.summary()

    def _build_residual_block(self, x):
        mc = self.config.model
        in_x = x
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_last", kernel_regularizer=l2(mc.l2_reg))(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_last", kernel_regularizer=l2(mc.l2_reg))(x)
        x = BatchNormalization(axis=3)(x)
        x = Add()([in_x, x])
        x = Activation("relu")(x)
        return x

    # 重みの学習
    def replay(self, memory: Memory, batch_size: int, gamma: float, targetQN: 'QNetwork') -> None:
        inputs = np.zeros((batch_size, 7, 5, 3))
        targets = np.zeros((batch_size, 315))
        mini_batch = memory.sample(batch_size)

        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
            inputs[i] = state_b  # shape=(4, 5, 5)
            target = reward_b  # type: int

            # if not (next_state_b == 0).all():
            # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値関数のQネットワークは分離）
            retmainQs = self.model.predict(next_state_b)
            next_action = np.argmax(retmainQs)  # 最大の報酬を返す行動を選択する
            target = reward_b + gamma * \
                targetQN.model.predict(next_state_b)[0][next_action]

            targets[i] = self.model.predict(state_b)[0][0]   # Qネットワークの出力
            # 教師信号 action_b: int <= 100
            targets[i, action_b] = target
        # epochsは訓練データの反復回数、verbose=0は表示なしの設定
        self.model.fit(inputs, targets, epochs=1, verbose=0)

    @staticmethod
    def fetch_digest(weight_path: str):
        if os.path.exists(weight_path):
            m = hashlib.sha256()
            with open(weight_path, "rb") as f:
                m.update(f.read())
            return m.hexdigest()

    def load(self, config_path: str, weight_path: str) -> bool:
        if os.path.exists(weight_path):  # os.path.exists(config_path) and
            logger.debug(f"loading model from {config_path}")
            with open(config_path, "rt") as f:
                self.model = Model.from_config(json.load(f))
            self.model.load_weights(weight_path)
            self.model.compile(loss='mse',
                               optimizer=Adam(lr=self.config.model.learning_rate))
            self.model.summary()
            self.digest = self.fetch_digest(weight_path)
            logger.debug(f"loaded model digest = {self.digest}")
            return True
        else:
            logger.debug(
                f"model files does not exist at {config_path} and {weight_path}")
            return False

    def save(self, config_path: str, weight_path: str) -> None:
        logger.debug(f"save model to {config_path}")
        with open(config_path, "wt") as f:
            json.dump(self.model.get_config(), f)
        self.model.save_weights(weight_path)
        self.digest = self.fetch_digest(weight_path)
        logger.debug(f"saved model digest {self.digest}")


def take_action_eps_greedy(board: np.ndarray, episode: int, mainQN: QNetwork, gs: GameState) -> Tuple[Winner, int]:
    """ｔ＋１での行動を返す
    boardは入力の型(README参照)で与えること
    returnは勝利判定と打った手"""
    # 徐々に最適行動のみをとる、ε-greedy法
    epsilon = 0.001 + 0.9 / (1.0+episode)

    if epsilon <= np.random.uniform(0, 1):
        retTargetQs = mainQN.model.predict(board)[0]
        s = gs.outputs_to_move_max(retTargetQs)  # 最大の報酬を返す行動を選択する

    else:
        s = gs.random_play()  # ランダムに行動する

    return s


def calc_reward(qc: QLearnConfig, next_board: np.ndarray) -> float:
    return (qc.reward_stone_mine * np.sum(next_board[0, 1])
            + qc.reward_stone_against * np.sum(next_board[0, 0])
            + qc.reward_front_mine * count_front_minus(next_board)
            + qc.reward_front_against * count_front_plus(next_board)
            + qc.reward_check_mine * count_check_minus(next_board)
            + qc.reward_check_against * count_check_plus(next_board))


def count_check_minus(board: np.ndarray) -> int:
    return (board[0, 1, 6, 1] + board[0, 1, 6, 3] + board[0, 1, 5, 2])


def count_check_plus(board: np.ndarray) -> int:
    return (board[0, 0, 0, 1] + board[0, 0, 0, 3] + board[0, 0, 1, 2])


def count_front_minus(board: np.ndarray) -> int:
    return np.sum(board[0, 1, 5:7]) - np.sum(board[0, 1, 2:4])


def count_front_plus(board: np.ndarray) -> int:
    return np.sum(board[0, 1, 0:3]) - np.sum(board[0, 1, 3:4])


def save_model(mainQN: QNetwork, config: Config) -> None:
    d = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    mainQN.save(f"results/001_QLearning/{d}-mainQN.json",
                f"results/001_QLearning/{d}-mainQN.h5")
    with open(f"results/001_QLearning/{d}-config.json", 'x') as f:
        json.dump(config._to_dict(), f, indent=4)


def learn_random(model_config_path=None, weight_path=None) -> None:
    config = Config()
    config.learn_func = 'learn_random'
    qc = config.Qlearn

    total_reward_vec = np.zeros(qc.num_consecutive_iterations)  # 各試行の報酬を格納
    # Qネットワークとメモリ、Actorの生成--------------------------------------------------------
    if model_config_path is None or weight_path is None:
        mainQN = QNetwork(config)     # メインのQネットワーク
        mainQN.build()
        targetQN = QNetwork(config)   # 価値を計算するQネットワーク
        targetQN.build()
    else:
        mainQN = QNetwork(config)
        success_load = mainQN.load(model_config_path, weight_path)
        if not success_load:
            raise FileNotFoundError(
                f"{model_config_path} {weight_path}が読み込めませんでした")
        targetQN = QNetwork(config)
        targetQN.load(model_config_path, weight_path)
        config.pre_trained = weight_path
    memory = Memory(max_size=qc.memory_size)

    for episode in trange(qc.num_episodes):  # 試行数分繰り返す
        gs = GameState()
        state = gs.random_play()  # 1step目は適当な行動をとる
        episode_reward = 0

        # 行動決定と価値計算のQネットワークをおなじにする
        targetQN.model.set_weights(mainQN.model.get_weights())

        for t in range(qc.max_number_of_steps):  # 2手のループ
            board = gs.to_inputs()

            state, action = take_action_eps_greedy(
                board, episode, mainQN, gs)   # 時刻tでの行動を決定する
            # next_state, reward, done, info = env.step(action)   # 行動a_tの実行による、s_{t+1}, _R{t}を計算する

            # verbose ==========
            # if t % 10 == 9:
            #     print(gs)
            # ==================

            if state == Winner.plus:
                reward = qc.reward_lose  # 報酬

            next_board = gs.to_inputs()

            # board = next_board  # 状態更新
            # 1施行終了時の処理
            if state != Winner.not_ended:
                episode_reward += reward  # 合計報酬を更新
                memory.add((board, action, reward, next_board))     # メモリの更新する
                # Qネットワークの重みを学習・更新する replay
                if len(memory) > qc.batch_size:  # and not islearned:
                    mainQN.replay(memory, qc.batch_size, qc.gamma, targetQN)
                if qc.DQN_MODE:
                    # 行動決定と価値計算のQネットワークをおなじにする
                    targetQN.model.set_weights(mainQN.model.get_weights())

                total_reward_vec = np.hstack(
                    (total_reward_vec[1:], episode_reward))  # 報酬を記録
                print('%d/%d: Episode finished after %d time steps / mean %f winner: %s'
                      % (episode+1, qc.num_episodes, t + 1, total_reward_vec.mean(),
                         'plus' if state == Winner.plus else 'minus'))
                break

            state, _ = gs.random_play()

            if state == Winner.minus:
                reward = qc.reward_win
            else:
                reward = calc_reward(qc, next_board)

            episode_reward += reward  # 合計報酬を更新
            memory.add((board, action, reward, next_board))     # メモリの更新する

            # Qネットワークの重みを学習・更新する replay
            if len(memory) > qc.batch_size:  # and not islearned:
                mainQN.replay(memory, qc.batch_size, qc.gamma, targetQN)

            if qc.DQN_MODE:
                # 行動決定と価値計算のQネットワークをおなじにする
                targetQN.model.set_weights(mainQN.model.get_weights())

            # 1施行終了時の処理
            if state != Winner.not_ended:
                total_reward_vec = np.hstack(
                    (total_reward_vec[1:], episode_reward))  # 報酬を記録
                print('%d/%d: Episode finished after %d time steps / mean %f winner: %s'
                      % (episode+1, qc.num_episodes, t + 1, total_reward_vec.mean(),
                         'plus' if state == Winner.plus else 'minus'))
                break

        # 複数施行の平均報酬で終了を判断
        # if total_reward_vec.mean() >= goal_average_reward:
        #     print('Episode %d train agent successfuly!' % episode)
            # islearned = True
        if episode % qc.save_interval == qc.save_interval - 1:
            save_model(mainQN, config)

    # 最後に保存(直前にしていればしない)
    if episode % qc.save_interval != qc.save_interval - 1:
        save_model(mainQN, config)

# 未実装
def learn_self(model_config_path=None, weight_path=None) -> None:
    config = Config()
    config.learn_func = 'learn_self'
    qc = config.Qlearn

    total_reward_vec = np.zeros(qc.num_consecutive_iterations)  # 各試行の報酬を格納
    # Qネットワークとメモリ、Actorの生成--------------------------------------------------------
    if model_config_path is None or weight_path is None:
        mainQN_plus = QNetwork(config)     # メインのQネットワーク
        mainQN_minus = QNetwork(config)     # メインのQネットワーク
        mainQN_plus.build()
        mainQN_minus.build()
        targetQN_plus = QNetwork(config)   # 価値を計算するQネットワーク
        targetQN_minus = QNetwork(config)   # 価値を計算するQネットワーク
        targetQN_plus.build()
        targetQN_minus.build()
    else:
        mainQN_plus = QNetwork(config)
        mainQN_minus = QNetwork(config)
        success_load = mainQN_plus.load(model_config_path, weight_path)
        if not success_load:
            raise FileNotFoundError(
                f"{model_config_path} {weight_path}が読み込めませんでした")

        mainQN_minus.load(model_config_path, weight_path)
        targetQN_plus = QNetwork(config)
        targetQN_minus = QNetwork(config)
        targetQN_plus.load(model_config_path, weight_path)
        targetQN_minus.load(model_config_path, weight_path)
        config.pre_trained = weight_path

    memory = Memory(max_size=qc.memory_size)

    for episode in trange(qc.num_episodes):  # 試行数分繰り返す
        gs = GameState()
        state = gs.random_play()  # 1step目は適当な行動をとる
        episode_reward = 0

        # 行動決定と価値計算のQネットワークをおなじにする
        targetQN.model.set_weights(mainQN.model.get_weights())

        for t in range(qc.max_number_of_steps):  # 2手のループ
            board = gs.to_inputs()

            state, action = take_action_eps_greedy(
                board, episode, mainQN_minus, gs)   # 時刻tでの行動を決定する
            # next_state, reward, done, info = env.step(action)   # 行動a_tの実行による、s_{t+1}, _R{t}を計算する

            # verbose ==========
            # if t % 10 == 9:
            #     print(gs)
            # ==================

            if state == Winner.plus:
                reward = qc.reward_lose  # 報酬

            next_board = gs.to_inputs()

            # board = next_board  # 状態更新
            # 1施行終了時の処理
            if state != Winner.not_ended:
                episode_reward += reward  # 合計報酬を更新
                memory.add((board, action, reward, next_board))     # メモリの更新する
                # Qネットワークの重みを学習・更新する replay
                if len(memory) > qc.batch_size:  # and not islearned:
                    mainQN_minus.replay(
                        memory, qc.batch_size, qc.gamma, targetQN)
                if qc.DQN_MODE:
                    # 行動決定と価値計算のQネットワークをおなじにする
                    targetQN.model.set_weights(mainQN.model.get_weights())

                total_reward_vec = np.hstack(
                    (total_reward_vec[1:], episode_reward))  # 報酬を記録
                print('%d/%d: Episode finished after %d time steps / mean %f winner: %s'
                      % (episode+1, qc.num_episodes, t + 1, total_reward_vec.mean(),
                         'plus' if state == Winner.plus else 'minus'))
                break

            state, _ = gs.random_play()

            if state == Winner.minus:
                reward = qc.reward_win
            else:
                reward = calc_reward(qc, next_board)

            episode_reward += reward  # 合計報酬を更新
            memory.add((board, action, reward, next_board))     # メモリの更新する

            # Qネットワークの重みを学習・更新する replay
            if len(memory) > qc.batch_size:  # and not islearned:
                mainQN.replay(memory, qc.batch_size, qc.gamma, targetQN)

            if qc.DQN_MODE:
                # 行動決定と価値計算のQネットワークをおなじにする
                targetQN.model.set_weights(mainQN.model.get_weights())

            # 1施行終了時の処理
            if state != Winner.not_ended:
                total_reward_vec = np.hstack(
                    (total_reward_vec[1:], episode_reward))  # 報酬を記録
                print('%d/%d: Episode finished after %d time steps / mean %f winner: %s'
                      % (episode+1, qc.num_episodes, t + 1, total_reward_vec.mean(),
                         'plus' if state == Winner.plus else 'minus'))
                break

        # 複数施行の平均報酬で終了を判断
        # if total_reward_vec.mean() >= goal_average_reward:
        #     print('Episode %d train agent successfuly!' % episode)
            # islearned = True
        if episode % qc.save_interval == qc.save_interval - 1:
            d = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            mainQN.save(f"results/001_QLearning/{d}-mainQN.json",
                        f"results/001_QLearning/{d}-mainQN.h5")
            with open(f"results/001_QLearning/{d}-config.json", 'x') as f:
                json.dump(config._to_dict(), f, indent=4)

    # 最後に保存(直前にしていればしない)
    if episode % qc.save_interval != qc.save_interval - 1:
        d = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        mainQN.save(f"results/001_QLearning/{d}-mainQN.json",
                    f"results/001_QLearning/{d}-mainQN.h5")
        with open(f"results/001_QLearning/{d}-config.json", 'x') as f:
            json.dump(config._to_dict(), f, indent=4)


if __name__ == "__main__":
    learn_random()
    # learn_random("results/001_QLearning/2020-02-08-18-42-17-mainQN.json",
    #       "results/001_QLearning/2020-02-08-18-42-17-mainQN.h5")
