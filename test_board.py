import numpy as np
from board import load_board, is_valid


def test_is_valid():
    board = np.zeros((9, 9), dtype=int)
    board[0, 0] = 5
    assert not is_valid(board, 0, 1, 5)  # same row
    assert not is_valid(board, 1, 0, 5)  # same column
    assert not is_valid(board, 1, 1, 5)  # same 3x3 box
    assert is_valid(board, 4, 4, 5)  # safe cell
