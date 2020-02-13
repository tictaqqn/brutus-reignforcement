import unittest
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
    
