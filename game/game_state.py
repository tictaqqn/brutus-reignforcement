from typing import *
from enum import IntEnum, auto, Enum
import random
import numpy as np
from .errors import ChoiceOfMovementError, GameError


class Drc(IntEnum):
    B_fr = 0
    B_r = 1
    B_br = 2
    B_f = 3
    B_b = 4
    B_fl = 5
    B_l = 6
    B_bl = 7
    f2 = 8

#     W_f = 9
#     W_f2 = 10
#     W_b = 11
#     W_r = 12
#     W_l = 13
#     W_fr = 14
#     W_fl = 15
#     W_br = 16
#     W_bl = 17


class Winner(Enum):
    not_ended = auto()
    plus = auto()
    minus = auto()


DIRECTIONS_LIST = [[-1,  1], [0,  1], [1,  1],
                   [-1,  0],          [1,  0],
                   [-1, -1], [0, -1], [1, -1]]

DIRECTIONS = list(map(np.array, DIRECTIONS_LIST))


class GameState:

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
        self.n_turns = 0  # ターン経過数

    def to_inputs(self, flip=False) -> np.ndarray:
        """強化学習用の入力
        flipで盤面を先手用と後手用の反転"""
        arr = np.empty((1, 2, 7, 5), dtype=bool)
        if not flip:
            board = self.board
        else:
            board = np.flip(self.board * -1, 0)
        arr[0, 0] = board == 1
        arr[0, 1] = board == -1
        return arr

    def __repr__(self) -> str:
        return str(self.board)

    @staticmethod
    def boundary_check(ij: Union[Sequence[int], np.ndarray]) -> bool:
        return 0 <= ij[0] <= 6 and 0 <= ij[1] <= 4

    def move(self, i: int, j: int, drc: Drc) -> Winner:
        """drcへの移動"""
        if self.n_turns == 0 and drc == Drc.f2:
            raise ChoiceOfMovementError(f"先手の初手は2マス移動不可")
        if self.board[i, j] != self.turn:
            raise ChoiceOfMovementError(f"選択したコマが王か色違いか存在しない {i, j}")
        direction = self.directionize(drc)
        nxt = np.array([i, j]) + direction
        if not self.boundary_check(nxt):
            raise ChoiceOfMovementError(f"外側への飛び出し {nxt}")
        if self.board[nxt[0], nxt[1]] != 0:
            raise ChoiceOfMovementError(f"移動先にコマあり {nxt}")
        if drc == Drc.f2:
            between = np.array([i, j]) + direction // 2
            if self.board[between[0], between[1]] == self.turn:
                raise ChoiceOfMovementError(f"間に自コマあり {between}")
        self.board[i, j] = 0
        self.board[nxt[0], nxt[1]] = self.turn
        self.reverse(nxt)
        return self.turn_change()

    def move_d_vec(self, i: int, j: int, direction: np.array) -> Winner:
        """directionのベクトル方向への移動"""
        if self.n_turns == 0 and direction[0] == -2:
            raise ChoiceOfMovementError(f"先手の初手は2マス移動不可")
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

    def turn_change(self) -> Winner:
        if self.turn == 1:
            if self.board[6, 1] == -1 or self.board[6, 3] == -1 or \
                    self.board[5, 2] == -1:
                return Winner.minus  # 後手勝利
            elif (self.board != -1).all():
                return Winner.plus  # 先手勝利
        else:
            if self.board[0, 1] == 1 or self.board[0, 3] == 1 or \
                    self.board[1, 2] == 1:
                return Winner.plus  # 先手勝利
            elif (self.board != 1).all():
                return Winner.minus  # 後手勝利
        self.turn *= -1
        self.n_turns += 1
        return Winner.not_ended

    def directionize(self, drc: Drc) -> np.ndarray:
        if drc == 8:
            return (np.array([-2, 0]) if self.turn == 1
                    else np.array([2, 0]))
        else:
            return DIRECTIONS[drc]

    def reverse(self, ij: np.ndarray) -> None:
        """石を裏返す"""
        for dirc in DIRECTIONS:
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

    def valid_choice(self, i: int, j: int, drc: Drc) -> bool:
        """手が有効かどうかを返す"""
        # if self.board[i, j] != self.turn:
        #     # raise ChoiceOfMovementError(f"選択したコマが王か色違いか存在しない {i, j}")
        #     return False
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

    def random_play(self, decided_pb=1) -> Tuple[Winner, int]:
        """decided_pbの確率で王手を優先的に打つ
        returnは勝利判定と打った手"""
        if random.random() < decided_pb:
            sa = self.prior_checkmate()
            if sa is not None:
                # print('priority')
                return sa
        while True:
            i = random.randint(0, 7-1)
            j = random.randint(0, 5-1)
            # if self.board[i, j] != self.turn:
            #     continue
            drc = random.randint(0, 8)
            # if self.turn == -1:
            #     drc += 9
            try:
                state = self.move(i, j, drc)
            except ChoiceOfMovementError:
                continue
            else:  # うまくいったとき
                return state, self.to_outputs_index(i, j, drc)

            # if self.valid_choice(i, j, drc):
            #     self.move(i, j, drc)
            #     break

    def prior_checkmate(self) -> Optional[Tuple[Winner, int]]:
        """優先的にチェックメイトを狙う"""
        if self.turn == 1:
            near_king = [(0, 1), (0, 3), (1, 2)]
        else:
            near_king = [(6, 1), (6, 3), (5, 2)]
        random.shuffle(near_king)
        for i0, j0 in near_king:
            sa = self._prior_checkmate_each(i0, j0)
            if sa is not None:
                return sa
        return None

    def _prior_checkmate_each(self, i0: int, j0: int) -> Optional[Tuple[Winner, int]]:
        """i0, j0に行けるコマがあれば行かせる"""
        if self.board[i0, j0] != 0:  # i0, j0にそもそもいけない
            return None
        d = np.array([i0, j0])
        ijs = self.near(d)
        for ij in ijs:
            try:
                state = self.move_d_vec(ij[0], ij[1], d - ij)
            except ChoiceOfMovementError:
                pass
            else:
                drc = DIRECTIONS_LIST.index((d-ij).tolist())
                return state, self.to_outputs_index(ij[0], ij[1], drc)
        return None

    def near(self, ij) -> Iterable[Tuple[int, int]]:
        """ijの近くにいるコマをyieldする"""
        directions = random.sample(DIRECTIONS,
                                   len(DIRECTIONS))
        for d in directions:
            p = ij + d
            if self.boundary_check(p) and \
                    self.board[p[0], p[1]] == self.turn:
                yield p

    @staticmethod
    def to_outputs_index(i: int, j: int, drc: Drc) -> int:
        return i * 45 + j * 9 + drc

    def outputs_to_move_max(self, outputs: 'array_like') -> Tuple[Winner, int]:
        """出力から最も高い確率のものに有効手を指す.
        ただしdeepcopyしない場合、outputsに副作用を生じる
        returnは勝利判定と打った手"""
        outputs_ = outputs
        # outputs_ = copy.deepcopy(outputs)
        for _ in range(10):
            argmax = np.argmax(outputs_)
            outputs_[argmax] = -1.0
            try:
                state = self.move(*np.unravel_index(argmax, (7, 5, 9)))
            except ChoiceOfMovementError:
                continue
            else:
                # print(argmax)
                # print(np.unravel_index(argmax, (7, 5, 9)))
                return state, argmax
        return self.random_play(0)

    def outputs_to_move_random(self, outputs: np.ndarray) -> Tuple[Winner, int]:
        """出力からランダムに有効手を指す.
        ただしoutputsは確率分布になっている必要がある(1への規格化が必要).
        returnは勝利判定と打った手"""
        num_choices = min(np.sum(outputs != 0), 10)
        random_choices = np.random.choice(
            315, p=outputs, size=num_choices, replace=False)
        for r in random_choices:
            try:
                state = self.move(*np.unravel_index(r, (7, 5, 9)))
            except ChoiceOfMovementError:
                continue
            else:
                # print(r)
                # print(np.unravel_index(r, (7, 5, 9)))
                return state, r

        return self.random_play(0)

    def board_hash(self) -> int:
        """ハッシュ関数。ZobristのようにXORを用いている"""
        xor_sum = 0
        for i in range(7):
            for j in range(5):
                xor_sum ^= self.board[i, j] * 35 + i * 5 + j
        return abs(xor_sum)

    def generate_legal_moves(self) -> Iterable[int]:
        """有効手をyieldする"""
        for i in range(7):
            for j in range(5):
                if self.board[i, j] != self.turn:
                    continue
                for drc in range(9):
                    if self.valid_choice(i, j, drc):
                        yield self.to_outputs_index(i, j, drc)
