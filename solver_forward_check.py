import copy

import numpy as np

from board import Step, is_valid, load_boards_csv


def get_peers(row: int, col: int) -> set[tuple[int, int]]:
    peers = set()

    # Add row and column peers
    for i in range(9):
        peers.add((row, i))  # Row peers
        peers.add((i, col))  # Column peers

    # Add box peers
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(box_row, box_row + 3):
        for j in range(box_col, box_col + 3):
            peers.add((i, j))

    # Remove the cell itself
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
                peers = get_peers(i, j)
                for peer in peers:
                    domains[peer].discard(num)

    return domains


def select_unassigned_cell(domains):
    # Build list of (domain size, (row, col)) for only non‑empty domains
    unassigned = [(len(dom), pos) for pos, dom in domains.items() if dom]
    if not unassigned:
        return None
    # Return the position with minimal remaining values
    return min(unassigned)[1]


def forward_check(domains, row, col, value):
    """
    Simulate assigning `value` to (row,col):
      - Copy the domains dict
      - Clear the assigned cell’s domain
      - Remove `value` from all peer domains
      - If any peer’s domain becomes empty ⇒ return None (failure)
      - Otherwise return the updated domains
    """
    new_domains = copy.deepcopy(domains)
    new_domains[(row, col)] = set()

    for peer in get_peers(row, col):
        if value in new_domains[peer]:
            new_domains[peer].remove(value)
            if not new_domains[peer]:  # domain wiped out
                return None

    return new_domains


def solve_forward_check(
    board: np.ndarray, domains: dict[tuple[int, int], set[int]], steps
) -> bool:
    """
    The recursive search:
      1. Pick an unassigned cell (MRV).
      2. For each value in its domain:
         a) Check quickly if placing it is valid.
         b) Do forward checking to prune neighbors.
         c) If forward_check succeeds, recurse.
      3. If all cells assigned ⇒ success; if no value works ⇒ backtrack.
    """
    # 1) Base case: no domains left ⇒ all cells assigned
    cell = select_unassigned_cell(domains)
    if cell is None:
        return True

    r, c = cell
    for val in sorted(domains[(r, c)]):
        if is_valid(board, r, c, val):
            board[r, c] = val
            steps.append(Step(r, c, val))
            pruned = forward_check(domains, r, c, val)
            if pruned is not None:
                if solve_forward_check(board, pruned, steps):
                    return True
            board[r, c] = 0  # undo
            steps.append(Step(r, c, 0))

    return False


board = load_boards_csv("./puzzles/hard.csv")[0]
domains = initialize_domains(board)
steps = []
if solve_forward_check(board, domains, steps):
    print(steps)
