import time

import numpy as np

from board import get_first_empty_cell, is_valid, load_board, print_board


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


sudoku_board = load_board("./puzzles/puzzle3.txt")

print_board(sudoku_board)
print("\n\n\n")

start_time = time.time()
if solve_backtrack(sudoku_board):
    print_board(sudoku_board)
else:
    print("No solution exists.")

end_time = time.time()
print(f"Solved in {end_time - start_time:.3f} seconds")
