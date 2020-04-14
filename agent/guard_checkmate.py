import random
import numpy as np
from tqdm import tqdm, trange
from .config import Config
from .model import QNetwork, Memory, take_action_eps_greedy, calc_reward, save_model
from game.game_state import GameState, Winner


def genetate_crisis(player_plus: bool, n_allies: int, n_enemies: int) -> np.ndarray:
    """自分に王手がかかっているような場面にする
    n_allies: 味方の数(最大)
    n_enemies:敵の数(最大-1)"""
    board = np.zeros((7, 5), dtype=np.int8)
    board[0, 2] = -2
    board[6, 2] = 2
    if player_plus:
        near_my_king = [(6, 1), (6, 3), (5, 2)]
        near_enemy_king = [(0, 1), (0, 3), (1, 2)]
        ally = 1
        enemy = -1
    else:
        near_my_king = [(0, 1), (0, 3), (1, 2)]
        near_enemy_king = [(6, 1), (6, 3), (5, 2)]
        ally = -1
        enemy = 1

    i0, j0 = random.choice(near_my_king)
    board[i0, j0] = enemy  # 王手

    choices = np.arange(35, dtype=int)
    allies_and_enemies = np.random.choice(
        choices, n_allies + n_enemies, replace=False)
    for n, (i, j) in enumerate(map(lambda x: np.unravel_index(x, (7, 5)), allies_and_enemies)):
        if (i, j) in [(i0, j0), (0, 2), (6, 2)]:
            continue
        if n < n_allies:
            if (i, j) in near_enemy_king:
                continue
            board[i, j] = ally
        else:
            board[i, j] = enemy

    return board


def learn_guard_checkmate(model_config_path=None, weight_path=None) -> None:
    config = Config()
    config.learn_func = 'learn_guard_checkmate'
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

        # 行動決定と価値計算のQネットワークをおなじにする
        targetQN.model.set_weights(mainQN.model.get_weights())

        considered_well = False

        while not considered_well:
            gs = GameState()
            state = gs.random_play()  # 1step目は適当な行動をとる
            gs.board = genetate_crisis(False, 6, 5)
            episode_reward = 0
            considered_well = True

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
                    if t == 0:
                        considered_well = False
                        break
                    episode_reward += reward  # 合計報酬を更新
                    # メモリの更新する
                    memory.add((board, action, reward, next_board))
                    # Qネットワークの重みを学習・更新する replay
                    if len(memory) > qc.batch_size:  # and not islearned:
                        mainQN.replay(memory, qc.batch_size,
                                      qc.gamma, targetQN)
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

                if t == 0:
                    reward += qc.reward_consider_checking

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


if __name__ == "__main__":
    learn_guard_checkmate("results/001_QLearning/2020-02-13-19-59-05-mainQN.json",
                          "results/001_QLearning/2020-02-13-19-59-05-mainQN.h5")
