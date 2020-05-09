import datetime
import numpy as np
import pickle

from agent.config import Config
from game.game_state import GameState, Winner
from uct.mcts import MCTSPlayer
from uct.dfpn import dfpn
from .generate_latter_board import generate_latter_board, random_gen


def mcts_self_play_tsumero(n_games, n_actions=50, model_config_path_plus=None, weight_path_plus=None,
                           model_config_path_minus=None, weight_path_minus=None, temperature=100.0, n_playout=300, c_puct=1.0, ignore_draw=False, all_random=False):

    # action_logs = []
    # wp_logs = []
    winner_or_not = []
    arr_logs = []
    board_logs = []
    plus_turn_logs = []
    player_plus = MCTSPlayer(1, temperature, n_playout, c_puct)
    player_minus = MCTSPlayer(-1, temperature, n_playout, c_puct)
    if model_config_path_plus is None or weight_path_plus is None:
        player_plus.initialize_model()
    else:
        player_plus.load_model(model_config_path_plus,
                               weight_path_plus)
    if model_config_path_minus is None or weight_path_minus is None:
        player_minus.initialize_model()
    else:
        player_minus.load_model(model_config_path_minus,
                                weight_path_minus)

    for n in range(n_games):
        gs = GameState()
        gs.board = generate_latter_board() if not all_random else random_gen()
        print(gs.board)

        _arr_logs, _board_logs, _plus_turn_logs = [], [], []

        for _ in range(n_actions):

            dfpn_r = dfpn(gs)
            if dfpn_r is not None:
                best_action = dfpn_r
                arr = np.zeros(315)
                arr[best_action] = 1.0
            else:
                player_plus.gs.board = gs.board.copy()
                player_plus.gs.turn = gs.turn
                player_plus.gs.n_turns = gs.n_turns
                best_action, st, arr = player_plus.go()
                if best_action is None:
                    state = st
                    break
            # action_logs.append(best_action)
            # wp_logs.append(best_wp)
            _arr_logs.append(arr)
            _board_logs.append(gs.board.copy())
            _plus_turn_logs.append(True)
            # if best_action is None:
            #     break
            state = gs.move_with_id(best_action)
            if state != Winner.not_ended:
                break
            dfpn_r = dfpn(gs)
            if dfpn_r is not None:
                best_action = dfpn_r
                arr = np.zeros(315)
                arr[best_action] = 1.0
            else:
                player_minus.gs.board = gs.board.copy()
                player_minus.gs.turn = gs.turn
                player_minus.gs.n_turns = gs.n_turns
                best_action, st, arr = player_minus.go()
                if best_action is None:
                    state = st
                    break
                # action_logs.append(best_action)
                # wp_logs.append(best_wp)
                _arr_logs.append(arr)
                _board_logs.append(gs.board.copy())
                _plus_turn_logs.append(False)
                # if best_action is None:
                #     break
            state = gs.move_with_id(best_action)
            if state != Winner.not_ended:
                break

        n_turns = len(_arr_logs)
        print(f'n_game: {n}/{n_games}')
        print(n_turns)
        # assert len(_arr_logs) == n_turns

        if state == Winner.plus:
            print('winner: plus')
            winner_or_not += [1., -1.] * (n_turns >> 1) + [1.] * (n_turns & 1)
        elif state == Winner.minus:
            print('winner: minus')
            winner_or_not += [-1., 1.] * (n_turns >> 1) + [-1.] * (n_turns & 1)
        else:
            print('draw')
            print(len(winner_or_not), len(arr_logs))
            if ignore_draw:
                continue
            winner_or_not += [0.] * n_turns

        arr_logs += _arr_logs
        board_logs += _board_logs
        plus_turn_logs += _plus_turn_logs
        print(len(winner_or_not), len(arr_logs))
        # assert len(winner_or_not) == len(arr_logs)

    print('saving')
    d = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # with open(f'results/bababax/kifu/{d}.pickle', 'wb') as f:
    #     pickle.dump(dict(
    #         action=action_logs, wp=wp_logs, arr=arr_logs, board=board_logs
    #     ), f)
    # np.savezだとメモリ確保に時間がかかるかもしれない
    path = f'results/bababax/kifu/{d}.npz'
    np.savez(path, wp=winner_or_not, pi_mcts=arr_logs,
             board=board_logs, plus_turn=plus_turn_logs)
    # print(action_logs)
    # print(winner_or_not)
    # print(arr_logs[0])
    # print(board_logs)
    return path


if __name__ == "__main__":
    mcts_self_play_tsumero(2, 10)
