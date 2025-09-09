import numpy as np
def evaluate_solution(predicted_solution, puzzle, true_solution, print_boards=False):
    original_puzzle = np.array(puzzle)
    true_solution = np.array(true_solution)
    
    correct_empty = 0
    empty_total = 0
    empty_total = len([(i,j) for i in range(9) for j in range(9) if original_puzzle[i][j] == 0])
    correct_empty = len([(i,j) for i in range(9) for j in range(9) if (original_puzzle[i][j] == 0 and (predicted_solution[i][j] == true_solution[i][j]))])

    empty_cells_accuracy = correct_empty / empty_total if empty_total > 0 else 0
    all_cells_accuracy = np.sum(predicted_solution == true_solution) / 81

    if print_boards:
        print("\nPredicted:")
        print(predicted_solution)
        print("\nTrue solution:")
        print(true_solution)
        print(f"\nEmpty cells to predict: {empty_total}")
        print(f"Empty cells correct: {correct_empty}/{empty_total}")
        print(f"Empty cells accuracy: {empty_cells_accuracy*100:.1f}%")
        print(f"All cells accuracy: {all_cells_accuracy*100:.1f}%")

    return empty_cells_accuracy, all_cells_accuracy

def single_predict_accuracy(model, puzzle, true_solution, print=False):
    puzzle_norm = np.array(puzzle, dtype=np.float32) / 9.0 - 0.5
    puzzle_input = puzzle_norm.reshape(1, 9, 9, 1)

    pred = model.predict(puzzle_input, verbose=0)
    predicted_solution = np.argmax(pred[0], axis=-1) + 1

    return evaluate_solution(predicted_solution, puzzle, true_solution, print_boards=print)


def iterative_sudoku_solver(model, puzzle, max_iters=50, verbose=False):
    board = np.array(puzzle, dtype=np.int32)
    for it in range(max_iters):
        input_board = board.astype(np.float32) / 9.0 - 0.5
        input_board = input_board.reshape(1,9,9,1)

        pred = model.predict(input_board, verbose=0)[0]  
        pred_digits = np.argmax(pred, axis=-1) + 1       
        pred_conf = np.max(pred, axis=-1)                

        empty_mask = (board == 0)
        if not np.any(empty_mask):
            break 

        max_idx = np.unravel_index(np.argmax(pred_conf * empty_mask), (9,9))
        board[max_idx] = pred_digits[max_idx]

        if verbose:
            print(f"Iteracija {it+1}: popunjeno polje {max_idx} sa {board[max_idx]}")

    return board


def test_iterative_accuracy(model, puzzle, true_solution, verbose=False):
    solved = iterative_sudoku_solver(model, puzzle, verbose=verbose)
    return evaluate_solution(solved, puzzle, true_solution, print_boards=True)