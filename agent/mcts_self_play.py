import datetime
import numpy as np
import pickle

from .config import Config
from game.game_state import GameState, Winner
from uct.mcts import MCTSPlayer


def mcts_self_play(n_games, n_actions=50, model_config_path_plus=None, weight_path_plus=None, model_config_path_minus=None, weight_path_minus=None):

    # action_logs = []
    wp_logs = []
    arr_logs = []
    board_logs = []
    plus_turn_logs = []

    for n in range(n_games):
        gs = GameState()
        player_plus = MCTSPlayer(1)
        player_minus = MCTSPlayer(-1)
        if model_config_path is None or weight_path is None:
            player_plus.initialize_model()
            player_minus.initialize_model()
        else:
            player_plus.load_model(model_config_path_plus,
                                   weight_path_plus)
            player_minus.load_model(model_config_path_minus,
                                    weight_path_minus)

        for _ in range(n_actions):
            player_plus.gs.board = gs.board.copy()
            player_plus.gs.turn = gs.turn
            player_plus.gs.n_turns = gs.n_turns
            best_action, best_wp, arr = player_plus.go()
            # action_logs.append(best_action)
            wp_logs.append(best_wp)
            arr_logs.append(arr)
            board_logs.append(gs.board.copy())
            plus_turn_logs.append(True)
            # if best_action is None:
            #     break
            state = gs.move_with_id(best_action)
            if state != Winner.not_ended:
                break
            player_minus.gs.board = gs.board.copy()
            player_minus.gs.turn = gs.turn
            player_minus.gs.n_turns = gs.n_turns
            best_action, best_wp, arr = player_minus.go()
            # action_logs.append(best_action)
            wp_logs.append(best_wp)
            arr_logs.append(arr)
            board_logs.append(gs.board.copy())
            plus_turn_logs.append(False)
            # if best_action is None:
            #     break
            state = gs.move_with_id(best_action)
            if state != Winner.not_ended:
                break

    print('saving')
    d = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # with open(f'results/bababax/kifu/{d}.pickle', 'wb') as f:
    #     pickle.dump(dict(
    #         action=action_logs, wp=wp_logs, arr=arr_logs, board=board_logs
    #     ), f)
    # np.savezだとメモリ確保に時間がかかるかもしれない
    path = f'results/bababax/kifu/{d}.npz'
    np.savez(path, wp=wp_logs, pi_mcts=arr_logs,
             board=board_logs, plus_turn=plus_turn_logs)
    # print(action_logs)
    print(wp_logs)
    print(arr_logs[0])
    print(board_logs)
    return path


if __name__ == "__main__":
    mcts_self_play(2, 10)
