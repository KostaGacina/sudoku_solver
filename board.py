import numpy as np


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


board = load_board("./puzzles/puzzle1.txt")
# print_board(board)
# if is_valid(board, 0, 2, 4):
#    print("can be placed")
# else:
#    print("Cannot be placed, number already in the same row/column/box")
