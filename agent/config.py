import os
from .configbase import ConfigBase


class Config(ConfigBase):

    def __init__(self, temperature=100.0, n_playout=300, c_puct=1.0):
        self.model = ModelConfig()
        self.Qlearn = QLearnConfig()
        self.mcts = MCTSConfig(temperature, n_playout, c_puct)
        self.pre_trained = None
        self.learn_func = None


class ModelConfig(ConfigBase):

    def __init__(self):
        self.cnn_filter_num = 32
        self.cnn_filter_size = 5
        self.res_layer_num = 2
        self.l2_reg = 1e-4
        self.value_fc_size = 256
        self.learning_rate = 0.01  # もっと小さいほうがいい?


class QLearnConfig(ConfigBase):

    def __init__(self):
        self.DQN_MODE = False  # TrueがDQN、FalseがDDQN

        self.num_episodes = 300  # 総ゲーム数
        self.max_number_of_steps = 25  # 1ゲームの最大手数
        # self.goal_average_reward = 50  # この報酬を超えると学習終了
        self.num_consecutive_iterations = 10  # 学習完了評価の平均計算を行う試行回数
        self.save_interval = 100  # このゲーム数ごとに保存
        self.gamma = 0.99    # 割引係数
        # ---
        # self.hidden_size = 16               # Q-networkの隠れ層のニューロンの数
        self.memory_size = 10000            # バッファーメモリの大きさ
        self.batch_size = 32                # Q-networkを更新するバッチの大記載
        # ---
        self.reward_win = 10000               # 勝利時の報酬
        self.reward_lose = -10000             # 敗北時の報酬
        self.reward_stone_mine = 1          # 自分の石の数1つあたりの報酬
        self.reward_stone_against = -1      # 相手の石の数1つあたりの報酬
        self.reward_front_mine = 0        # 前の方に自石があるときの報酬
        self.reward_front_against = 0     # 前の方に相手の石があるときの報酬
        self.reward_check_mine = 500      # 自分が王手をかけているとき
        self.reward_check_against = -4000  # 自分が王手をかけられているとき
        self.reward_consider_checking = 1000  # 初手のクリア
        # self.reward_auto_play = False # random_play時の報酬(未実装)


class MCTSConfig(ConfigBase):

    def __init__(self, temperature, n_playout, c_puct):
        self.temperature = temperature
        self.n_playout = n_playout
        self.c_puct = c_puct
