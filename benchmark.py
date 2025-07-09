import threading
import time

import numpy as np
from colorama import Fore, Style
from tabulate import tabulate

from board import load_boards_csv
from solver_backtracking import solve_backtrack
from solver_forward_check import initialize_domains, solve_forward_check

easy_boards = load_boards_csv("./puzzles/easy.csv")
medium_boards = load_boards_csv("./puzzles/medium.csv")
hard_boards = load_boards_csv("./puzzles/hard.csv")

board_sets = [
    ("easy", easy_boards),
    ("medium", medium_boards),
    ("hard", hard_boards),
]
results = []


def benchmark_backtracking(boards, label=""):
    times = []
    for board in boards:
        start = time.time()
        solve_backtrack(board)
        end = time.time()
        times.append(end - start)
    avg = np.mean(times)
    results.append((label, "Backtracking", f"{avg:.6f}"))


def benchmark_forward_check(boards, label=""):
    times = []
    for board in boards:
        domains = initialize_domains(board)
        start = time.time()
        solve_forward_check(board, domains)
        end = time.time()
        times.append(end - start)
    avg = np.mean(times)
    results.append((label, "Forward Check", f"{avg:.6f}"))


threads = []
for label, boards in board_sets:
    threads.append(
        threading.Thread(target=benchmark_backtracking, args=(boards, label))
    )
    threads.append(
        threading.Thread(target=benchmark_forward_check, args=(boards, label))
    )

for thread in threads:
    thread.start()
for thread in threads:
    thread.join()

# Sort and print results
difficulty_order = {"easy": 0, "medium": 1, "hard": 2}
results.sort(key=lambda x: (difficulty_order[x[0]], x[1]))

print(Fore.CYAN + "\nSUDOKU SOLVER BENCHMARK RESULTS\n" + Style.RESET_ALL)
print(
    tabulate(
        results,
        headers=["Difficulty", "Algorithm", "Avg Time (s)"],
        tablefmt="fancy_grid",
    )
)
