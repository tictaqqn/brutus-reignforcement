from typing import Union
import copy
import numpy as np
from game.game_state import GameState, Winner


def dfpn(gs: GameState) -> Union[None, int]:
    gs = copy.deepcopy(gs)
    state = gs.get_winner()
    if state != Winner.not_ended:
        print('bestmove resign')
        print(gs)
        return None

    turn = gs.turn
    if turn == 1:
        my_side, other_side = Winner.plus, Winner.minus
    else:
        my_side, other_side = Winner.minus, Winner.plus

    # 探索開始時刻の記録
    # begin_time = time.time()
    legal_moves = list(gs.generate_legal_moves())

    if len(legal_moves) == 1:
        return legal_moves[0]

    for a in legal_moves:
        k = np.unravel_index(a, (7, 5, 9))[0]
        if turn == -1:
            k = 6 - k
        if k > 3:
            continue
        state = gs.move_with_id(a)
        if state == my_side:
            return a
        if state == other_side:
            gs.pop()
            continue

        post_legal_moves = list(gs.generate_legal_moves())

        all_win = True
        for a2 in post_legal_moves:
            state = gs.move_with_id(a2)

            if state != my_side:
                all_win = False
                gs.pop()
                break
            gs.pop()

        if all_win:
            return a

        gs.pop()

    return None
