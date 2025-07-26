import threading
import time

import numpy as np
from colorama import Fore, Style
from tabulate import tabulate

from ac3 import AC3, initialize_domains
from ai_solver import solve_ac3_backtrack, solve_ac3_forward
from board import load_boards_csv
from solver_backtracking import solve_backtrack, solve_backtrack_domains
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

all_steps = []


def benchmark_ac3_forward(boards, label=""):
    times = []
    for board in boards:
        steps = []
        start = time.time()
        solve_ac3_forward(board, steps)
        end = time.time()
        times.append(end - start)
    avg = np.mean(times)
    results.append((label, "AC3 + Forward Check", f"{avg:.6f}"))


def benchmark_ac3_backtrack(boards, label=""):
    times = []
    for board in boards:
        steps = []
        start = time.time()
        solve_ac3_backtrack(board, steps)
        end = time.time()
        times.append(end - start)
    avg = np.mean(times)
    results.append((label, "AC3 + Backtracking", f"{avg:.6f}"))


def benchmark_backtracking(boards, label=""):
    times = []
    for board in boards:
        steps = []
        start = time.time()
        solve_backtrack(board, steps)
        end = time.time()
        times.append(end - start)
        all_steps.append(steps)
    avg = np.mean(times)
    results.append((label, "Backtracking", f"{avg:.6f}"))


def benchmark_forward_check(boards, label=""):
    times = []
    for board in boards:
        steps = []
        domains = initialize_domains(board)
        start = time.time()
        solve_forward_check(board, domains, steps)
        end = time.time()
        times.append(end - start)
        all_steps.append(steps)
    avg = np.mean(times)
    results.append((label, "Forward Check", f"{avg:.6f}"))


def get_benchmark_data():
    """Return benchmark results in a structured format"""
    global results

    # Convert results to dictionary format
    benchmark_data = {}
    for difficulty, algorithm, avg_time in results:
        if algorithm not in benchmark_data:
            benchmark_data[algorithm] = {}
        benchmark_data[algorithm][difficulty] = float(avg_time)

    return benchmark_data


threads = []
for label, boards in board_sets:
    threads.append(
        threading.Thread(target=benchmark_backtracking, args=(boards, label))
    )
    threads.append(
        threading.Thread(target=benchmark_forward_check, args=(boards, label))
    )
    threads.append(
        threading.Thread(target=benchmark_ac3_backtrack, args=(boards, label))
    )
    threads.append(threading.Thread(target=benchmark_ac3_forward, args=(boards, label)))

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
