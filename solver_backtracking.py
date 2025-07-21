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

def solve_backtrack_domains(board: np.ndarray, domains: dict):
    unassigned = [(cell, values) for cell, values in domains.items() if board[cell[0]][cell[1]] == 0]
    if not unassigned:
        return True  

    cell, values = min(unassigned, key=lambda item: len(item[1]))
    row, col = cell

    for value in sorted(values):
        if is_valid(board, row, col, value):
            board[row][col] = value

            new_domains = {k: set(v) for k, v in domains.items()}
            new_domains[cell] = {value}

            if solve_backtrack_domains(board, new_domains):
                return True

            board[row][col] = 0  

    return False
