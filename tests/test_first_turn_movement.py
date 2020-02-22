import sys
import unittest
from game.game_state import GameState, Drc
sys.path.append('../')
from game.errors import ChoiceOfMovementError


class TestFirstTurmMovement(unittest.TestCase):

    def setUp(self):
        self.gs = GameState()

    def test_first_move_front2(self):
        with self.assertRaises(ChoiceOfMovementError):
            self.gs.move_d_vec(6, 0, [-2, 0])
        with self.assertRaises(ChoiceOfMovementError):
            self.gs.move_d_vec(5, 1, [-2, 0])
        with self.assertRaises(ChoiceOfMovementError):
            self.gs.move_d_vec(6, 2, [-2, 0])
        with self.assertRaises(ChoiceOfMovementError):
            self.gs.move_d_vec(5, 3, [-2, 0])
        with self.assertRaises(ChoiceOfMovementError):
            self.gs.move_d_vec(6, 4, [-2, 0])

    def test_random_play_first_move(self):
        valid_Drcs = (Drc.B_fr, Drc.B_r, Drc.B_f, Drc.B_fl, Drc.B_l)
            # possible direction for first movement
        for _ in range(1000):
            self.setUp()
            __, outputs_index = self.gs.random_play()
            self.assertTrue(outputs_index % 9 in valid_Drcs)
