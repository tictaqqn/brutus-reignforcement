import unittest
import numpy as np
from game.game_state import GameState, Drc


class TestGameState(unittest.TestCase):

    def setUp(self):
        self.gs = GameState()

    def test_reverse(self):
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

if __name__ == '__main__':
    unittest.main()