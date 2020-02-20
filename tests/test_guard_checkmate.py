import unittest
import numpy as np
from agent.guard_checkmate import genetate_crisis


class TestGuardCheckmate(unittest.TestCase):

    def setUp(self):
        pass
        # self.gs = GameState()

    def test_generate_crisis(self):

        for _ in range(10):
            board = genetate_crisis(True, 8, 3)
            self.assertEqual(board[0, 2], -2)
            self.assertEqual(board[6, 2], 2)
            self.assertTrue(
                board[5, 2] == -1 or
                board[6, 1] == -1 or
                board[6, 3] == -1
            )
        for _ in range(10):
            board = genetate_crisis(False, 8, 3)
            self.assertEqual(board[0, 2], -2)
            self.assertEqual(board[6, 2], 2)
            self.assertTrue(
                board[1, 2] == 1 or
                board[0, 1] == 1 or
                board[0, 3] == 1
            )
