import sys
import unittest
from game.game_state import GameState, Drc
sys.path.append('../')
from game.errors import ChoiceOfMovementError
import numpy as np


class TestFirstTurmMovement(unittest.TestCase):

    def setUp(self):
        self.gs = GameState()

    def test_first_move_front2(self):
        d = np.array([-2, 0])
        with self.assertRaises(ChoiceOfMovementError):
            self.gs.move_d_vec(6, 0, d)
        with self.assertRaises(ChoiceOfMovementError):
            self.gs.move_d_vec(5, 1, d)
        with self.assertRaises(ChoiceOfMovementError):
            self.gs.move_d_vec(6, 2, d)
        with self.assertRaises(ChoiceOfMovementError):
            self.gs.move_d_vec(5, 3, d)
        with self.assertRaises(ChoiceOfMovementError):
            self.gs.move_d_vec(6, 4, d)

    def test_random_play_first_move(self):
        valid_Drcs = (Drc.B_fr, Drc.B_r, Drc.B_f, Drc.B_fl, Drc.B_l)
            # possible direction for first movement
        for _ in range(1000):
            self.setUp()
            __, outputs_index = self.gs.random_play()
            self.assertTrue(outputs_index % 9 in valid_Drcs)
