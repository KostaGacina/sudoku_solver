import numpy as np
import time
import copy
from board import Step, is_valid, load_boards_csv


def get_peers(row: int, col: int) -> set[tuple[int, int]]:
    peers = set()
    for i in range(9):
        peers.add((row, i))
        peers.add((i, col))
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(box_row, box_row + 3):
        for j in range(box_col, box_col + 3):
            peers.add((i, j))
    peers.discard((row, col))
    return peers


def initialize_domains(board: np.ndarray) -> dict[tuple[int, int], set[int]]:
    domains = {}
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                domains[(i, j)] = set(range(1, 10))
            else:
                domains[(i, j)] = set()
    for i in range(9):
        for j in range(9):
            num = board[i][j]
            if num != 0:
                for peer in get_peers(i, j):
                    domains[peer].discard(num)
    return domains


def select_unassigned_cell(domains):
    unassigned = [(len(dom), pos) for pos, dom in domains.items() if dom]
    return min(unassigned)[1] if unassigned else None


def forward_check(domains, row, col, value):
    new_domains = copy.deepcopy(domains)
    new_domains[(row, col)] = set()
    for peer in get_peers(row, col):
        if value in new_domains[peer]:
            new_domains[peer].remove(value)
            if not new_domains[peer]:
                return None
    return new_domains


def get_cnn_suggestions(model, board: np.ndarray):
    puzzle_norm = board.astype(np.float32) / 9.0 - 0.5
    start = time.time()
    pred = model.predict(puzzle_norm.reshape(1, 9, 9, 1), verbose=0)[0]  # shape (9,9,9)
    end = time.time()
    elapsed = end - start
    
    suggestions = {}
    for i in range(9):
        for j in range(9):
            if board[i, j] == 0:
                # sort probabilities descending, convert to digit 1–9
                ranked = np.argsort(-pred[i, j]) + 1
                suggestions[(i, j)] = ranked.tolist()
    return suggestions, elapsed


def cnn_forward_check(
    board: np.ndarray,
    domains: dict[tuple[int, int], set[int]],
    cnn_suggestions: dict[tuple[int, int], list[int]],
    steps,
) -> bool:
    cell = select_unassigned_cell(domains)
    if cell is None:
        return True

    r, c = cell

    # Try CNN suggested values first, filtered by current domain
    for val in cnn_suggestions.get((r, c), []):
        if val in domains[(r, c)] and is_valid(board, r, c, val):
            board[r, c] = val
            steps.append(Step(r, c, val))

            pruned = forward_check(domains, r, c, val)
            if pruned is not None:
                if cnn_forward_check(board, pruned, cnn_suggestions, steps):
                    return True

            board[r, c] = 0
            steps.append(Step(r, c, 0))

    return False

def solve_cnn_forward_check(model, puzzle):
    puzzle = np.array(puzzle, dtype=np.int32)
    domains = initialize_domains(puzzle)
    cnn_suggestions, elapsed = get_cnn_suggestions(model, puzzle)
    steps = []
    success = cnn_forward_check(puzzle, domains, cnn_suggestions, steps)
    return success, puzzle, elapsed

def print_cnn_suggestions(model, puzzle):
    puzzle = np.array(puzzle, dtype=np.int32)
    # domains = initialize_domains(puzzle)
    cnn_suggestions = get_cnn_suggestions(model, puzzle)
    print(cnn_suggestions)
    # success = cnn_forward_check(puzzle, domains, cnn_suggestions, steps)
