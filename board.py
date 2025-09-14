import numpy as np
import pandas as pd


# create class to represent steps taken to solve the board
class Step:
    def __init__(self, row: int, col: int, num: int):
        self.row = row
        self.col = col
        self.num = num

    def __repr__(self):
        return f"Step(row={self.row}, col={self.col}, num={self.num})"


def load_board(filename: str) -> np.ndarray:
    with open(filename, "r") as f:
        lines = [list(map(int, line.strip())) for line in f if line.strip()]
    board = np.array(lines, dtype=int)
    if board.shape != (9, 9):
        raise ValueError("Invalid board shape. Expected 9x9")
    return board


def print_board(board: np.ndarray) -> None:
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("-" * 21)
        for j in range(9):
            if j % 3 == 0 and j != 0:
                print("|", end=" ")
            val = board[i, j]
            print(val if val != 0 else ".", end=" ")
        print()


def is_valid(board: np.ndarray, row: int, col: int, num: int) -> bool:
    if num in board[row, :]:
        return False
    if num in board[:, col]:
        return False

    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    if num in board[box_row : box_row + 3, box_col : box_col + 3]:
        return False
    return True


def save_board(board: np.ndarray, filename: str) -> None:
    with open(filename, "w") as f:
        for row in board:
            f.write("".join(str(num) for num in row) + "\n")


def get_empty_cells(board: np.ndarray) -> list[tuple[int, int]]:
    return [(i, j) for i in range(9) for j in range(9) if board[i, j] == 0]


def get_first_empty_cell(board: np.ndarray) -> tuple[int, int] | None:
    for i in range(9):
        for j in range(9):
            if board[i, j] == 0:
                return i, j  # row, col
    return None


def load_boards_csv(filename):
    boards = []
    dataset = pd.read_csv(filename)
    df = pd.DataFrame(dataset)
    boards2 = df["quizzes"].values
    for board in boards2:
        array_9x9 = np.array([int(c) for c in board]).reshape((9, 9))
        boards.append(array_9x9)

    return np.array(boards)

import numpy as np

def is_solution_valid(board: np.ndarray) -> bool:
    """Check if a completed Sudoku board is valid."""
    # Must be 9x9
    if board.shape != (9, 9):
        return False

    # 1) Check no zeros remain
    if np.any(board == 0):
        return False

    # Expected set of digits
    digits = set(range(1, 10))

    # 2) Check rows and columns
    for i in range(9):
        if set(board[i, :]) != digits:  # row
            return False
        if set(board[:, i]) != digits:  # column
            return False

    # 3) Check 3x3 subgrids
    for r in range(0, 9, 3):
        for c in range(0, 9, 3):
            block = board[r:r+3, c:c+3].flatten()
            if set(block) != digits:
                return False

    return True


# board = load_boards_csv("./puzzles/easy.csv")
# print_board(board[0])
# if is_valid(board, 0, 2, 4):
#    print("can be placed")
# else:
#    print("Cannot be placed, number already in the same row/column/box")
