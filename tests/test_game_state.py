import unittest
import numpy as np
from game.game_state import GameState, Drc


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
        self.gs.move(1, 4, Drc.W_f)
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
        self.assertIsNone(state)
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
        self.assertEqual(state, 1)

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
        self.assertEqual(state, -1)
