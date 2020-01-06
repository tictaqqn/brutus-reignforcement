from .errors import *
from .game_state import Drc, GameState


def keys_to_drc(key, turn):
    if key == 'q':
        drcs = Drc.B_fl, Drc.W_br
    elif key == 'w':
        drcs = Drc.B_f, Drc.W_b
    elif key == 'e':
        drcs = Drc.B_fr, Drc.W_bl
    elif key == 'a':
        drcs = Drc.B_l, Drc.W_r
    elif key == 'd':
        drcs = Drc.B_r, Drc.W_l
    elif key == 'z':
        drcs = Drc.B_bl, Drc.W_fr
    elif key == 'x':
        drcs = Drc.B_b, Drc.W_f
    elif key == 'c':
        drcs = Drc.B_br, Drc.W_fl
    elif key == 'ww':
        drcs = Drc.B_f2, None
    elif key == 'xx':
        drcs = None, Drc.W_f2

    return drcs[0] if turn == 1 else drcs[1]


def play(gs=None, logs=None):
    if gs is None:
        gs = GameState()
    if logs is None:
        logs = []
    while True:
        print(gs)
        print("移動元の座標とs周りの移動方向を入力してください。例: 1 2 q")
        try:
            inputs = input()
            if inputs == 'exit':
                return gs, logs
            i, j, d = inputs.split()
            drc = keys_to_drc(d, gs.turn)
            i = int(i)
            j = int(j)
        except Exception as e:
            print(e)
            print("もう一度入力してください。")
            continue
        try:
            state = gs.move(i, j, drc)
        except GameError as e:
            print(e)
            print("入力が不正です。もう一度入力してください。")
            continue
        logs.append((i, j, d))
        if state == 1:
            print(gs)
            print("先手勝利")
            return gs, logs
        elif state == -1:
            print(gs)
            print("後手勝利")
            return gs, logs


if __name__ == "__main__":
    play()
