import datetime
import numpy as np

from .config import Config
from game.game_state import GameState, Winner
from uct.mcts import MCTSPlayer


def mcts_self_play(n_games, n_actions=50, model_config_path=None, weight_path=None):

    player_plus = MCTSPlayer(1)
    player_minus = MCTSPlayer(-1)
    if model_config_path is None or weight_path is None:
        player_plus.initialize_model()
        player_minus.initialize_model()
    else:
        player_plus.load_model(model_config_path,
                            weight_path)
        player_minus.load_model(model_config_path,
                                weight_path)
    # TODO: MCTSPlayerにどちらが先手か認識させる
    action_logs = []
    wps_list = []

    for n in range(n_games):
        gs = GameState()
        player_plus.gs = gs
        player_minus.gs = gs
        wps = []

        for _ in range(n_actions):
            best_action, best_wp = player_plus.go()
            wps.append(best_wp)
            if best_action is None:
                break
            state = gs.move_with_id(best_action)
            if state != Winner.not_ended:
                break
            best_action, best_wp = player_minus.go()
            wps.append(best_wp)
            if best_action is None:
                break
            state = gs.move_with_id(best_action)
            if state != Winner.not_ended:
                break

        action_logs.append(gs.get_action_logs())
        wps_list.append(np.array(wps))

    d = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    np.savez(f'results/bababax/kifu/{d}-kifu.npz', *action_logs)
    np.savez(f'results/bababax/kifu/{d}-eval.npz', *wps_list)
    print(action_logs[0])
    print(wps_list[0])
    

            
if __name__ == "__main__":
    mcts_self_play(2, 10)


