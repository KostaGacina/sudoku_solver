import time
import numpy as np
from colorama import Fore, Style
from tabulate import tabulate

from board import load_boards_csv
from solver_backtracking import solve_backtrack
from solver_forward_check import initialize_domains, solve_forward_check
from ai_solver import solve_ac3_backtrack, solve_ac3_forward

# Load only a few boards for testing
easy_boards = load_boards_csv("./puzzles/easy.csv")[:5]  # Test with only 5 boards
medium_boards = load_boards_csv("./puzzles/medium.csv")[:5]
hard_boards = load_boards_csv("./puzzles/hard.csv")[:5]

results = []

def test_solver(solver_func, boards, solver_name, difficulty):
    print(f"Testing {solver_name} on {difficulty} boards...")
    times = []
    
    for i, board in enumerate(boards):
        try:
            print(f"  Board {i+1}/{len(boards)}...")
            start = time.time()
            
            if solver_name == "Forward Check":
                domains = initialize_domains(board)
                solver_func(board, domains)
            else:
                solver_func(board)
                
            end = time.time()
            times.append(end - start)
            print(f"    Completed in {end - start:.4f}s")
            
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    if times:
        avg = np.mean(times)
        results.append((difficulty, solver_name, f"{avg:.6f}"))
        print(f"  Average time: {avg:.6f}s\n")
    else:
        print(f"  No successful solves\n")

# Test each solver sequentially
solvers = [
    ("Backtracking", solve_backtrack),
    ("Forward Check", solve_forward_check),
    ("AC3 + Backtracking", solve_ac3_backtrack),
    ("AC3 + Forward Check", solve_ac3_forward),
]

for difficulty, boards in [("easy", easy_boards), ("medium", medium_boards), ("hard", hard_boards)]:
    for solver_name, solver_func in solvers:
        test_solver(solver_func, boards, solver_name, difficulty)

print(Fore.CYAN + "\nSUDOKU SOLVER BENCHMARK RESULTS\n" + Style.RESET_ALL)
if results:
    print(tabulate(results, headers=["Difficulty", "Algorithm", "Avg Time (s)"], tablefmt="fancy_grid"))
else:
    print("No results collected")