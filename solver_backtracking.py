import numpy as np

from board import get_first_empty_cell, is_valid


def solve_backtrack(board: np.ndarray):
    empty = get_first_empty_cell(board)
    if not empty:
        return True  # Solved
    row, col = empty

    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row][col] = num

            if solve_backtrack(board):
                return True

            # Backtrack
            board[row][col] = 0

    return False
