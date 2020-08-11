import numpy as np
from game.game_state import GameState, Winner


def generate_latter_board(strong=True, flip_pb=0.5):
    board = np.array([
        [-1, 0, -2, 0, -1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        np.random.randint(-1, 2, 5),
        np.random.randint(-1, 2, 5),
        [1, 0, 2, 0, 1]
    ], dtype=np.int8)
    if strong:
        board[6] = np.random.randint(-1, 2, 5)
        board[6, 2] = 2
    if np.random.rand() < flip_pb:
        board = np.flip(board * -1, 0)
    return board

def random_gen():
    board = np.random.randint(-1, 2, (7, 5))
    board[0, 2] = -2
    board[6, 2] = 2
    return board

if __name__ == "__main__":
    print(generate_latter_board())