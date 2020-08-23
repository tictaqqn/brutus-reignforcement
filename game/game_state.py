from typing import *
from collections import deque
from enum import IntEnum, auto, Enum
import copy
import random
import numpy as np
from .errors import ChoiceOfMovementError, GameError
from .consts import DEFAULT_RANDOM_ARRAY

try:
    from gmpy2 import popcount as pop_count
    from gmpy2 import bit_scan1 as bit_scan
except ImportError:
    try:
        from gmpy import popcount as pop_count
        from gmpy import scan1 as bit_scan
    except ImportError:
        def pop_count(b):
            return bin(b).count('1')

        def bit_scan(b, n=0):
            string = bin(b)
            l = len(string)
            r = string.rfind('1', 0, l - n)
            if r == -1:
                return -1
            else:
                return l - r - 1


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
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 1, 2, 1, 1]
        ], dtype=np.int8)
        self.turn = 1  # +が先攻
        self.n_turns = 0
        self.logs = deque()  # type: deque[Tuple[int, int]]

    def to_inputs(self, flip=False) -> np.ndarray:
        """強化学習用の入力
        flipで盤面を先手用と後手用の反転"""
        arr = np.empty((1, 7, 5, 3), dtype=bool)
        if not flip:
            board = self.board
        else:
            board = np.flip(self.board * -1, 0)
        arr[0, :, :, 0] = board == 1
        arr[0, :, :, 1] = board == -1
        arr[0, :, :, 2] = np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ], dtype=bool)
        return arr

    def __repr__(self) -> str:
        return f"{self.board}\n\
            turn: {'plus' if self.turn == 1 else 'minus'}\n\
            n_turns: {self.n_turns}"

    def pop(self) -> Optional[int]:
        """一手前へ戻す. 戻した手も返す"""
        try:
            act_id, board = self.logs.pop()
        except IndexError:  # 空の時
            return None
        self.n_turns -= 1
        self.turn *= -1
        self.board = board
        return act_id

    def get_action_logs(self) -> np.ndarray:
        action_logs = list(map(lambda x: x[0], self.logs))
        return np.array(action_logs, dtype=int)

    def get_board_logs(self) -> np.ndarray:
        board_logs = list(map(lambda x: self.board_id(x[1]), self.logs))
        return np.array(board_logs, dtype=int)

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
        before_move_board = self.board.copy()
        self.board[i, j] = 0
        self.board[nxt[0], nxt[1]] = self.turn
        self.reverse(nxt)
        self.logs.append((self.to_outputs_index(i, j, drc),
                          before_move_board))
        return self.turn_change()

    def move_with_id(self, action: int) -> Winner:
        return self.move(*np.unravel_index(action, (7, 5, 9)))

    def move_d_vec(self, i: int, j: int, direction: np.array) -> Winner:
        """directionのベクトル方向への移動"""
        if direction[0] == 2 * self.turn:
            raise ChoiceOfMovementError(f"後ろ2コマ移動不可{direction}")
        if abs(direction[0]) == 2 and direction[1] != 0:
            raise ChoiceOfMovementError(f"斜め2コマ移動不可{direction}")
        try:
            drc = DIRECTIONS_LIST.index(direction.tolist())
        except ValueError:
            drc = Drc.f2
        return self.move(i, j, drc)

    def turn_change(self) -> Winner:
        self.n_turns += 1
        self.turn *= -1  # 勝利判定時にもターン変更するようにした
        return self.get_winner()

    def get_winner(self) -> Winner:
        if self.turn == -1:
            if self.board[6, 1] == -1 or self.board[6, 3] == -1 or \
                    self.board[5, 2] == -1:
                return Winner.minus  # 後手勝利
            else:
                try:
                    next(self.generate_legal_moves())
                except StopIteration:
                    return Winner.plus  # 先手勝利
        else:
            if self.board[0, 1] == 1 or self.board[0, 3] == 1 or \
                    self.board[1, 2] == 1:
                return Winner.plus  # 先手勝利
            else: 
                try:
                    next(self.generate_legal_moves())
                except StopIteration:
                    return Winner.minus  # 先手勝利
        return Winner.not_ended

    def is_game_over(self) -> bool:
        state = self.get_winner()
        if state == Winner.not_ended:
            return False
        return True

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

    def _valid_choice(self, i: int, j: int, drc: Drc) -> bool:
        """手が有効かどうかを返す"""
        # if self.board[i, j] != self.turn:
        #     # raise ChoiceOfMovementError(f"選択したコマが王か色違いか存在しない {i, j}")
        #     return False
        direction = self.directionize(drc)
        nxt = np.array([i, j]) + direction
        if self.n_turns == 0 and drc == Drc.f2:
            # raise ChoiceOfMovementError(f"先手の初手は2マス移動不可")
            return False
        if not self.boundary_check(nxt):
            # raise ChoiceOfMovementError(f"外側への飛び出し {nxt}")
            return False
        if self.board[nxt[0], nxt[1]] != 0:
            # raise ChoiceOfMovementError(f"移動先にコマあり {nxt}")
            return False
        if drc == Drc.f2:
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
    
    @staticmethod
    def flip_turn_outputs_index(index: int) -> int:
        i, j, drc = np.unravel_index(index, (7, 5, 9))
        i = 6 - i
        if drc != Drc.f2:
            d = copy.deepcopy(DIRECTIONS_LIST[drc])
            d[0] = - d[0]
            drc = DIRECTIONS_LIST.index(d)
        return GameState.to_outputs_index(i, j, drc)

    @staticmethod
    def flip_turn_outputs(arr: np.ndarray) -> np.ndarray:
        flipped_arr = np.zeros(315)
        for i in range(315):
            ii = GameState.flip_turn_outputs_index(i)
            flipped_arr[ii] = arr[i]
        return flipped_arr


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

    def board_hash(self, array: list = None) -> int:
        """ハッシュ関数。ZobristのようにXORを用いている"""
        if array is None:
            array = DEFAULT_RANDOM_ARRAY
        i = self.board_id(self.board)
        bit = bit_scan(i)
        zobrist_hash = 0
        while bit != -1 and bit is not None:
            zobrist_hash ^= array[(2269 + bit) % 2286]
            bit = bit_scan(i, bit + 1)

        i = self.n_turns
        while bit != -1 and bit is not None:
            zobrist_hash ^= array[bit % 2286]
            bit = bit_scan(i, bit + 1)

        return zobrist_hash

    @staticmethod
    def board_id(board: np.ndarray) -> int:
        """ボードの状態をintにする"""
        flat_board = board.flatten()
        flat_board[2] = 0
        flat_board[32] = 0
        flat_board += 1  # マイナスをなくす

        b_id = 0
        h = 1
        for x in flat_board:
            b_id += x * h
            h *= 3  # <<= 2
        return b_id

    @staticmethod
    def id_to_board(b_id: int) -> np.ndarray:

        flat_board = np.zeros(35, dtype=np.int8)  # type: np.ndarray
        for i in range(35):
            x, b_id = b_id % 3, b_id // 3
            flat_board[i] = x
        flat_board -= 1
        flat_board[2] = -2
        flat_board[32] = 2
        return flat_board.reshape((7, 5))

    def generate_legal_moves(self) -> Iterable[int]:
        """有効手をyieldする"""
        for i in range(7):
            for j in range(5):
                if self.board[i, j] != self.turn:
                    continue
                for drc in range(9):
                    if self._valid_choice(i, j, drc):
                        yield self.to_outputs_index(i, j, drc)
