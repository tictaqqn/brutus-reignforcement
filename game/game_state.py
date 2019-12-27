from enum import Enum, auto
import numpy as np
from .errors import *


class Drc(Enum):
    B_f = auto()
    B_f2 = auto()
    B_b = auto()
    B_r = auto()
    B_l = auto()
    B_fr = auto()
    B_fl = auto()
    B_br = auto()
    B_bl = auto()

    W_f = auto()
    W_f2 = auto()
    W_b = auto()
    W_r = auto()
    W_l = auto()
    W_fr = auto()
    W_fl = auto()
    W_br = auto()
    W_bl = auto()


class GameState(object):
    DIRECTIONS = list(map(np.array, ([-1, 1], [0, 1], [1, 1],
                                     [-1, 0], [1, 0],
                                     [-1, -1], [0, -1], [1, -1])))

    def __init__(self):
        self.board = np.array([
            [-1, -1, -2, -1, -1],
            [0, -1, 0, -1, 0],
            [0] * 5,
            [0] * 5,
            [0] * 5,
            [0, 1, 0, 1, 0],
            [1, 1, 2, 1, 1]
        ], dtype=int)
        self.turn = 1  # +が先攻

    def __repr__(self):
        return str(self.board)

    def boundary_check(self, ij) -> bool:
        return 0 <= ij[0] <= 6 and 0 <= ij[1] <= 4

    def move(self, i, j, drc):
        if self.board[i, j] != self.turn:
            raise ChoiceOfMovementError(f"選択したコマが王か色違いか存在しない {i, j}")
        direction = self.directionize(drc)
        nxt = np.array([i, j]) + direction
        if not self.boundary_check(nxt):
            raise ChoiceOfMovementError(f"外側への飛び出し {nxt}")
        if self.board[nxt[0], nxt[1]] != 0:
            raise ChoiceOfMovementError(f"移動先にコマあり {nxt}")
        if drc == Drc.B_f2 or drc == Drc.W_f2:
            between = np.array([i, j]) + direction // 2
            if self.board[between[0], between[1]] == self.turn:
                raise ChoiceOfMovementError(f"間に自コマあり {between}")
        self.board[i, j] = 0
        self.board[nxt[0], nxt[1]] = self.turn
        self.reverse(nxt)
        return self.turn_change()

    def turn_change(self):
        if self.turn == 1:
            if self.board[6, 1] == -1 or self.board[6, 3] == -1 or \
                    self.board[5, 2] == -1:
                return -1  # 後手勝利
            elif (self.board != -1).all():
                return 1  # 先手勝利
        else:
            if self.board[0, 1] == 1 or self.board[0, 3] == 1 or \
                    self.board[0, 2] == 1:
                return 1  # 先手勝利
            elif (self.board != 1).all():
                return -1  # 後手勝利
        self.turn *= -1
        return None

    def directionize(self, drc):
        if drc == Drc.B_f or drc == Drc.W_b:
            return np.array([-1, 0])
        elif drc == Drc.W_f or drc == Drc.B_b:
            return np.array([1, 0])
        elif drc == Drc.B_fr or drc == Drc.W_bl:
            return np.array([-1, 1])
        elif drc == Drc.B_fl or drc == Drc.W_br:
            return np.array([-1, -1])
        elif drc == Drc.B_bl or drc == Drc.W_fr:
            return np.array([1, -1])
        elif drc == Drc.B_br or drc == Drc.W_fl:
            return np.array([-1, -1])
        elif drc == Drc.B_r or drc == Drc.W_l:
            return np.array([0, 1])
        elif drc == Drc.B_l or drc == Drc.W_r:
            return np.array([0, -1])
        elif drc == Drc.B_f2:
            return np.array([-2, 0])
        elif drc == Drc.W_f2:
            return np.array([2, 0])
        else:
            raise GameError("Never reaches here")

    def reverse(self, ij):
        # print(self.DIRECTIONS)
        for dirc in self.DIRECTIONS:
            pos = ij + dirc
            # print(pos)
            while self.boundary_check(pos):
                p = self.board[pos[0], pos[1]]
                if p == 0:  # 空白で終了
                    break
                elif p == self.turn or p == self.turn * 2:  # 自王もok
                    pos -= dirc
                    while not (pos == ij).all():  # ijに戻るまで
                        if self.board[pos[0], pos[1]] != self.turn * -2:  # 相手王でないとき
                            self.board[pos[0], pos[1]] = self.turn  # ひっくり返す
                        pos -= dirc
                    break
                pos += dirc
