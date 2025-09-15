from cnn_forward_check import solve_cnn_forward_check, print_cnn_suggestions
import time
from board import load_boards_csv, print_board, is_solution_valid
import numpy as np
from keras import saving
import numpy
import matplotlib.pyplot as plt
import pandas as pd

model_512 = saving.load_model("wetransfer_sudoku_cnn_512_filters-3-keras_2025-09-09_1044\sudoku_cnn_512_filters (3).keras")
model_256 = saving.load_model("wetransfer_sudoku_cnn_512_filters-3-keras_2025-09-09_1044\sudoku_cnn_256_filters (1).keras")

easy_boards = load_boards_csv("./puzzles/easy.csv")
medium_boards = load_boards_csv("./puzzles/medium.csv")
hard_boards = load_boards_csv("./puzzles/hard.csv")

board_sets = [
    ("easy", easy_boards),
    ("medium", medium_boards),
    ("hard", hard_boards),
]

results_256 = dict()
fails = 0
for label, boards in board_sets:
    times = []
    cnn_elapsed_times = []
    for board in boards:
        start = time.perf_counter()
        success, puzzle, cnn_elapsed_time = solve_cnn_forward_check(model_256, board)
        end = time.perf_counter()

        cnn_elapsed_times.append(cnn_elapsed_time)
        times.append(end - start)
        if not success:
            fails += 1
    results_256[label] = (times, cnn_elapsed_times)

print("Average times for 256")
print(f"easy: {np.mean(results_256["easy"][0])}")
print(f"medium: {np.mean(results_256["medium"][0])}")
print(f"hard: {np.mean(results_256["hard"][0])}")

results_512 = dict()
for label, boards in board_sets:
    times = []
    cnn_elapsed_times = []
    for board in boards:
        start = time.perf_counter()
        success, puzzle, cnn_elapsed_time = solve_cnn_forward_check(model_512, board)
        end = time.perf_counter()

        cnn_elapsed_times.append(cnn_elapsed_time)
        times.append(end - start)
        if not success:
            fails += 1
    results_512[label] = (times, cnn_elapsed_times)

print("Average times for 512")
print(f"easy: {np.mean(results_512["easy"][0])}")
print(f"medium: {np.mean(results_512["medium"][0])}")
print(f"hard: {np.mean(results_512["hard"][0])}")

# ----------- GRAPHS -----------

difficulties = ["easy", "medium", "hard"]

# Example for model_256
avg_total_times = [np.mean(results_256[label][0]) for label in difficulties]
avg_cnn_times   = [np.mean(results_256[label][1]) for label in difficulties]

x = np.arange(len(difficulties))  # positions
width = 0.35  # width of bars

fig, ax = plt.subplots()
ax.bar(x - width/2, avg_total_times, width, label="Total time")
ax.bar(x + width/2, avg_cnn_times, width, label="CNN time")

ax.set_xticks(x)
ax.set_xticklabels(difficulties)
ax.set_ylabel("Time (s)")
ax.set_title("Model 256 - Average Solve & CNN Time")
ax.legend()

plt.grid(True)
plt.savefig("./static/times256.png", dpi=300, bbox_inches="tight")  # high-quality PNG
plt.show()

# Example for model_512
avg_total_times = [np.mean(results_512[label][0]) for label in difficulties]
avg_cnn_times   = [np.mean(results_512[label][1]) for label in difficulties]

x = np.arange(len(difficulties))  # positions
width = 0.35  # width of bars

fig, ax = plt.subplots()
ax.bar(x - width/2, avg_total_times, width, label="Total time", color="green")
ax.bar(x + width/2, avg_cnn_times, width, label="CNN time", color="purple")

ax.set_xticks(x)
ax.set_xticklabels(difficulties)
ax.set_ylabel("Time (s)")
ax.set_title("Model 512 - Average Solve & CNN Time")
ax.legend()
plt.grid(True)
plt.savefig("./static/times512.png", dpi=300, bbox_inches="tight")
plt.show()

colors = {"easy": "green", "medium": "orange", "hard": "red"}

def strip_plot(model_name, results):
    plt.figure(figsize=(8, 5))

    for i, label in enumerate(difficulties):
        times = results[label][0] if isinstance(results[label], tuple) else results[label]

        # jitter x positions slightly
        x = np.random.normal(i + 1, 0.04, size=len(times))  
        plt.scatter(x, times, alpha=0.7, color=colors[label], label=label if i == 0 else "")

        # mean line
        mean_val = np.mean(times)
        plt.hlines(mean_val, i + 0.8, i + 1.2, colors="black", linestyles="--", linewidth=2)

    plt.xticks([1, 2, 3], difficulties)
    plt.ylabel("Solve time (s)")
    plt.title(f"{model_name} - Total Solve Times (with means)")
    plt.grid(True)
    plt.savefig(f"./static/strip{model_name}.png", dpi=300, bbox_inches="tight")  # high-quality PNG
    plt.show()

# Plot for both models
strip_plot("Model 256", results_256)
strip_plot("Model 512", results_512)

# PLOT TO COMPARE PURE FORWARD CHECKING AGAINST CNN + FORWARD CHECKING
pure_forward_check_times = [0.029735, 0.148340, 0.444237]
cnn256_times = [np.mean(results_256[label][0]) for label in difficulties]
cnn512_times = [np.mean(results_512[label][0]) for label in difficulties]

# Positions for groups
x = np.arange(len(difficulties))
width = 0.25  # width of each bar

fig, ax = plt.subplots(figsize=(8, 6))

# Bars
rects1 = ax.bar(x - width, pure_forward_check_times, width, label="Pure FC", color="skyblue")
rects2 = ax.bar(x, cnn256_times, width, label="CNN (256) + FC", color="orange")
rects3 = ax.bar(x + width, cnn512_times, width, label="CNN (512) + FC", color="green")

# Labels & styling
ax.set_xticks(x)
ax.set_xticklabels(difficulties)
ax.set_ylabel("Time (s)")
ax.set_title("Comparison of Solve Times: Pure FC vs CNN+FC")
ax.legend()

plt.tight_layout()
plt.grid(True)
plt.savefig("./static/fc_cnn_comparison.png", dpi=300, bbox_inches="tight")
plt.show()