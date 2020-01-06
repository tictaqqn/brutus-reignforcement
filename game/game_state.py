from typing import *
from enum import IntEnum, auto
import random
import numpy as np
from .errors import *


class Drc(IntEnum):
    B_f = 0
    B_f2 = 1
    B_b = 2
    B_r = 3
    B_l = 4
    B_fr = 5
    B_fl = 6
    B_br = 7
    B_bl = 8

    W_f = 9
    W_f2 = 10
    W_b = 11
    W_r = 12
    W_l = 13
    W_fr = 14
    W_fl = 15
    W_br = 16
    W_bl = 17


class GameState:
    DIRECTIONS = list(map(np.array, ([-1, 1], [0, 1], [1, 1],
                                     [-1, 0], [1, 0],
                                     [-1, -1], [0, -1], [1, -1])))

    def __init__(self) -> None:
        self.board = np.array([
            [-1, -1, -2, -1, -1],
            [0, -1, 0, -1, 0],
            [0] * 5,
            [0] * 5,
            [0] * 5,
            [0, 1, 0, 1, 0],
            [1, 1, 2, 1, 1]
        ], dtype=np.int8)
        self.turn = 1  # +が先攻

    def __repr__(self):
        return str(self.board)

    def boundary_check(self, ij: Union[Sequence[int], np.ndarray]) -> bool:
        return 0 <= ij[0] <= 6 and 0 <= ij[1] <= 4

    def move(self, i: int, j: int, drc: Drc):
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

    def move_d_vec(self, i: int, j: int, direction: np.array) -> int:
        if direction[0] == 2 * self.turn:
            raise ChoiceOfMovementError(f"後ろ2コマ移動不可{direction}")
        if abs(direction[0]) == 2 and direction[1] != 0:
            raise ChoiceOfMovementError(f"斜め2コマ移動不可{direction}")
        if self.board[i, j] != self.turn:
            raise ChoiceOfMovementError(f"選択したコマが王か色違いか存在しない {i, j}")
        # direction = self.directionize(drc)
        nxt = np.array([i, j]) + direction
        if not self.boundary_check(nxt):
            raise ChoiceOfMovementError(f"外側への飛び出し {nxt}")
        if self.board[nxt[0], nxt[1]] != 0:
            raise ChoiceOfMovementError(f"移動先にコマあり {nxt}")
        if abs(direction[0]) == 2:
            between = np.array([i, j]) + direction // 2
            if self.board[between[0], between[1]] == self.turn:
                raise ChoiceOfMovementError(f"間に自コマあり {between}")
        self.board[i, j] = 0
        self.board[nxt[0], nxt[1]] = self.turn
        self.reverse(nxt)
        return self.turn_change()

    def turn_change(self) -> int:
        if self.turn == 1:
            if self.board[6, 1] == -1 or self.board[6, 3] == -1 or \
                    self.board[5, 2] == -1:
                return -1  # 後手勝利
            elif (self.board != -1).all():
                return 1  # 先手勝利
        else:
            if self.board[0, 1] == 1 or self.board[0, 3] == 1 or \
                    self.board[1, 2] == 1:
                return 1  # 先手勝利
            elif (self.board != 1).all():
                return -1  # 後手勝利
        self.turn *= -1
        return None

    def directionize(self, drc: Drc) -> np.ndarray:
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
            raise ValueError("Never reaches here")

    def reverse(self, ij: np.ndarray) -> None:
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

    def valid_choice(self, i, j, drc) -> bool:
        if self.board[i, j] != self.turn:
            # raise ChoiceOfMovementError(f"選択したコマが王か色違いか存在しない {i, j}")
            return False
        direction = self.directionize(drc)
        nxt = np.array([i, j]) + direction
        if not self.boundary_check(nxt):
            # raise ChoiceOfMovementError(f"外側への飛び出し {nxt}")
            return False
        if self.board[nxt[0], nxt[1]] != 0:
            # raise ChoiceOfMovementError(f"移動先にコマあり {nxt}")
            return False
        if drc == Drc.B_f2 or drc == Drc.W_f2:
            between = np.array([i, j]) + direction // 2
            if self.board[between[0], between[1]] == self.turn:
                # raise ChoiceOfMovementError(f"間に自コマあり {between}")
                return False
        return True

    def random_play(self, decided_pb=1):
        if random.random() < decided_pb:
            if self.prior_checkmate():
                return
        while True:
            i = random.randint(0, 7-1)
            j = random.randint(0, 5-1)
            # if self.board[i, j] != self.turn:
            #     continue
            drc = random.randint(0, 8)
            if self.turn == -1:
                drc += 9
            try:
                self.move(i, j, drc)
            except GameError:
                continue
            else:  # うまくいったとき
                break

            # if self.valid_choice(i, j, drc):
            #     self.move(i, j, drc)
            #     break

    def prior_checkmate(self) -> bool:
        if self.turn == 1:
            near_king = [(0, 1), (0, 3), (1, 2)]
        else:
            near_king = [(6, 1), (6, 3), (5, 2)]
        random.shuffle(near_king)
        for i0, j0 in near_king:
            if self._prior_checkmate_each(i0, j0):
                return True
        return False

    def _prior_checkmate_each(self, i0, j0) -> bool:
        d = np.array([i0, j0])
        ijs = self.near(d)
        for i, j in ijs:
            try:
                self.move_d_vec(i, j, d)
            except GameError:
                pass
            else:
                return True
        return False

    def near(self, ij) -> Iterable[Tuple[int, int]]:
        directions = random.sample(self.DIRECTIONS,
                                   len(self.DIRECTIONS))
        for d in directions:
            p = ij + d
            if self.boundary_check(p) and \
                    self.board[p[0], p[1]] == self.turn:
                yield tuple(p)
