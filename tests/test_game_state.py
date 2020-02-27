import unittest, random
import numpy as np
from game.game_state import GameState, Drc, Winner


class TestGameState(unittest.TestCase):

    def setUp(self):
        self.gs = GameState()

    def test_reverse_horizontal(self):
        self.gs.board = np.array([
            [0] * 5,
            [-1, 1, 0, 1, -1],
            [-1, 0, -1, 1, 0],
            [0] * 5,
            [0] * 5,
            [0] * 5,
            [0] * 5
        ])
        self.gs.move(1, 1, Drc.B_b)
        self.assertTrue(
            (self.gs.board == np.array([
                [0] * 5,
                [-1, 0, 0, 1, -1],
                [-1, 1, 1, 1, 0],
                [0] * 5,
                [0] * 5,
                [0] * 5,
                [0] * 5
            ])).all()
        )
        self.gs.move(1, 4, Drc.B_b)
        self.assertTrue(
            (self.gs.board == np.array([
                [0] * 5,
                [-1, 0, 0, 1, 0],
                [-1] * 5,
                [0] * 5,
                [0] * 5,
                [0] * 5,
                [0] * 5
            ])).all()
        )

    def test_reverse_vertical_and_diagonal(self):
        self.gs.board = np.array([
            [0] * 5,
            [0] * 5,
            [1, -1, 1, 0, 0],
            [-1, -1, -1, -1, -1],
            [0, 1, 0, 0, 0],
            [0] * 5,
            [0] * 5
        ])
        state = self.gs.move(4, 1, Drc.B_r)
        self.assertEqual(state, Winner.not_ended)
        self.assertTrue(
            (self.gs.board == np.array([
                [0] * 5,
                [0] * 5,
                [1, -1, 1, 0, 0],
                [-1, 1, 1, -1, -1],
                [0, 0, 1, 0, 0],
                [0] * 5,
                [0] * 5
            ])).all()
        )

    def test_win_of_plus_no_enemies(self):
        self.gs.board = np.array([
            [0] * 5,
            [0] * 5,
            [1, 1, 1, 0, 0],
            [1, -1, -1, 1, 1],
            [0, 1, 0, 0, 0],
            [0] * 5,
            [0] * 5
        ])
        state = self.gs.move(4, 1, Drc.B_r)
        self.assertEqual(state, Winner.plus)

    def test_win_of_minus_checkmate(self):
        self.gs.board = np.array([
            [0] * 5,
            [0] * 5,
            [1, 1, 1, 0, 0],
            [1, -1, -1, 1, 1],
            [0, 1, 0, 0, 0],
            [0] * 5,
            [1, -1, 2, 1, 1]
        ])
        state = self.gs.move(4, 1, Drc.B_r)
        self.assertEqual(state, Winner.minus)

    def test_flip(self):
        self.assertTrue((
            self.gs.to_inputs()
            == self.gs.to_inputs(True)
        ).all())

    def test_outputs_to_move_random(self):
        outputs = np.linspace(0.0, 1.0, 315)
        outputs /= np.sum(outputs)
        self.gs.outputs_to_move_random(outputs)

    def test_board_pop(self):
        for n in range(10):
            for _ in range(n):
                self.gs.random_play(0)
            for _ in range(n):
                self.gs.pop()
            self.assertListEqual(self.gs.board.tolist(),
                                 np.array([
                                     [-1, -1, -2, -1, -1],
                                     [0, -1, 0, -1, 0],
                                     [0] * 5,
                                     [0] * 5,
                                     [0] * 5,
                                     [0, 1, 0, 1, 0],
                                     [1, 1, 2, 1, 1]
                                 ]).tolist())

    def test_board_id(self):
        board = np.array([
            [1, -1, -2, 1, 1],
            [0] * 5,
            [1, 1, 1, 0, 0],
            [1, -1, -1, 1, 1],
            [0, 1, 0, 0, 0],
            [0] * 5,
            [1, -1, 2, 1, 1]
        ], dtype=np.int8)
        board_id = GameState.board_id(board)
        board_2 = GameState.id_to_board(
            board_id)
        self.assertListEqual(board.tolist(),
                             board_2.tolist())

    def test_legal_moves(self):
        np.random.seed(1)
        for _ in range(10):
            for _ in range(100):
                legal_moves = list(self.gs.generate_legal_moves())
                move = np.random.choice(legal_moves)
                state = self.gs.move_with_id(move)
                if state != Winner.not_ended:
                    self.gs = GameState()

    def test_get_action_logs(self):
        actions = []
        random.seed(1)
        for _ in range(5):
            _, ac = self.gs.random_play(0)
            actions.append(ac)
        action_logs = self.gs.get_action_logs()
        self.assertListEqual(action_logs.tolist(),
                             actions)
