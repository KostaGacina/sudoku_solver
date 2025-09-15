# In[6]:


from solver_backtracking import solve_backtrack
from utils import evaluate_solution


import numpy as np
from keras.models import load_model


import time


class HybridSudokuSolver:
    def __init__(self, model_path, confidence_threshold=0.95):
        self.model = load_model(model_path)
        self.confidence_threshold = confidence_threshold

    def is_valid_move(self, board, row, col, num):
        """Check if a move is valid (doesn't violate Sudoku rules)"""
        if num == 0:
            return True

        # Check row
        if num in board[row, :]:
            return False

        # Check column
        if num in board[:, col]:
            return False

        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        box = board[box_row:box_row+3, box_col:box_col+3]
        if num in box:
            return False

        return True

    def print_board(self, board, title="Board"):
        """Print the sudoku board in a readable format"""
        print(f"\n{title}:")
        print("-" * 25)
        for i in range(9):
            if i % 3 == 0 and i != 0:
                print("-" * 25)
            row_str = ""
            for j in range(9):
                if j % 3 == 0 and j != 0:
                    row_str += "| "
                cell_value = board[i, j]
                row_str += f"{cell_value if cell_value != 0 else '.'} "
            print(row_str)
        print("-" * 25)

    def get_model_predictions(self, board):
        """Get model predictions for the current board state"""
        # Normalize input
        board_norm = board.astype(np.float32) / 9.0 - 0.5
        input_board = board_norm.reshape(1, 9, 9, 1)

        # Get predictions
        pred = self.model.predict(input_board, verbose=0)

        # Get predicted digits and their probabilities
        predicted_digits = np.argmax(pred[0], axis=-1) + 1
        probabilities = np.max(pred[0], axis=-1)

        return predicted_digits, probabilities

    def solve_with_cnn_first(self, puzzle, true_solution):
        """Use CNN for high-confidence predictions, then backtracking for the rest"""
        original_board = np.array(puzzle, dtype=int)
        true_solution = np.array(true_solution, dtype=int)

        print("=" * 60)
        print("HYBRID SUDOKU SOLVER: CNN + BACKTRACKING")
        print("=" * 60)

        # Step 1: Run CNN and get high-confidence predictions
        print("\n1. RUNNING CNN PREDICTIONS...")
        cnn_start_time = time.time()

        predicted_digits, probabilities = self.get_model_predictions(original_board)
        cnn_partial_board = original_board.copy()

        # Only fill cells where CNN has high confidence AND the cell is empty AND the move is valid
        empty_mask = original_board == 0
        high_confidence_mask = probabilities > self.confidence_threshold
        cells_to_fill = empty_mask & high_confidence_mask

        print(f"CNN confidence threshold: {self.confidence_threshold}")
        print(f"High-confidence predictions: {np.sum(cells_to_fill)} cells")

        # Count valid vs invalid high-confidence predictions
        valid_predictions = 0
        invalid_predictions = 0

        for i in range(9):
            for j in range(9):
                if cells_to_fill[i, j]:
                    predicted_value = predicted_digits[i, j]
                    if self.is_valid_move(cnn_partial_board, i, j, predicted_value):
                        cnn_partial_board[i, j] = predicted_value
                        valid_predictions += 1
                    else:
                        # Leave invalid predictions as 0 (don't fill them)
                        invalid_predictions += 1
                        cells_to_fill[i, j] = False  # Remove from filled count

        cnn_time = time.time() - cnn_start_time

        # Calculate REAL accuracy (against true solution, not just validity)
        cnn_filled_cells = np.sum(cells_to_fill)
        cnn_correct_predictions = 0

        for i in range(9):
            for j in range(9):
                if cells_to_fill[i, j]:
                    if cnn_partial_board[i, j] == true_solution[i, j]:
                        cnn_correct_predictions += 1

        cnn_accuracy = cnn_correct_predictions / cnn_filled_cells if cnn_filled_cells > 0 else 0.0

        self.print_board(original_board, "ORIGINAL PUZZLE")
        self.print_board(cnn_partial_board, "AFTER CNN (HIGH-CONFIDENCE + VALID PREDICTIONS)")

        print(f"\nCNN RESULTS:")
        print(f"Time taken: {cnn_time:.4f} seconds")
        print(f"High-confidence cells found: {valid_predictions + invalid_predictions}")
        print(f"  Valid predictions: {valid_predictions}")
        print(f"  Invalid predictions: {invalid_predictions} (not filled)")
        print(f"Cells actually filled by CNN: {cnn_filled_cells}")
        print(f"CNN prediction accuracy: {cnn_accuracy * 100:.2f}%")
        print(f"Remaining empty cells: {np.sum(cnn_partial_board == 0)}")

        # Check if CNN solved it completely
        if np.array_equal(cnn_partial_board, true_solution):
            print("\n🎉 CNN SOLVED THE PUZZLE COMPLETELY!")
            return cnn_partial_board, cnn_time, 0, cnn_accuracy, 1.0

        # Step 2: Use backtracking for remaining empty cells
        remaining_empty = np.sum(cnn_partial_board == 0)
        print(f"\n2. RUNNING BACKTRACKING FOR {remaining_empty} REMAINING CELLS...")
        backtrack_start_time = time.time()

        # Use backtracking to solve remaining cells
        backtrack_board = cnn_partial_board.copy()
        steps = []

        success = solve_backtrack(backtrack_board, steps)
        backtrack_time = time.time() - backtrack_start_time

        if success:
            # Final evaluation
            final_empty_acc, final_all_acc = evaluate_solution(backtrack_board, original_board, true_solution)

            self.print_board(backtrack_board, "FINAL SOLUTION (AFTER BACKTRACKING)")

            print(f"\nBACKTRACKING RESULTS:")
            print(f"Time taken: {backtrack_time:.4f} seconds")
            print(f"Number of backtracking steps: {len(steps)}")
            print(f"Remaining cells solved correctly: {final_empty_acc * 100:.2f}%")
            print(f"Final all cells accuracy: {final_all_acc * 100:.2f}%")

            # Verify final solution
            if np.array_equal(backtrack_board, true_solution):
                print("PUZZLE SOLVED CORRECTLY!")
            else:
                print("Final solution does not match expected result")
                print("Expected solution:")
                self.print_board(true_solution, "EXPECTED SOLUTION")

            return backtrack_board, cnn_time, backtrack_time, cnn_accuracy, final_all_acc

        else:
            print("Backtracking failed to solve the puzzle")
            return None, cnn_time, backtrack_time, cnn_accuracy, 0.0

    def solve_sudoku_hybrid(self, puzzle):
        """Use CNN for high-confidence predictions, then backtracking for the rest"""
        original_board = np.array(puzzle, dtype=int)

        print("=" * 60)
        print("HYBRID SUDOKU SOLVER: CNN + BACKTRACKING")
        print("=" * 60)

        # Step 1: Run CNN and get high-confidence predictions
        print("\n1. RUNNING CNN PREDICTIONS...")
        cnn_start_time = time.time()

        predicted_digits, probabilities = self.get_model_predictions(original_board)
        cnn_partial_board = original_board.copy()

        # Only fill cells where CNN has high confidence AND the cell is empty AND the move is valid
        empty_mask = original_board == 0
        high_confidence_mask = probabilities > self.confidence_threshold
        cells_to_fill = empty_mask & high_confidence_mask

        print(f"CNN confidence threshold: {self.confidence_threshold}")
        print(f"High-confidence predictions: {np.sum(cells_to_fill)} cells")

        # Count valid vs invalid high-confidence predictions
        valid_predictions = 0
        invalid_predictions = 0

        for i in range(9):
            for j in range(9):
                if cells_to_fill[i, j]:
                    predicted_value = predicted_digits[i, j]
                    if self.is_valid_move(cnn_partial_board, i, j, predicted_value):
                        cnn_partial_board[i, j] = predicted_value
                        valid_predictions += 1
                    else:
                        # Leave invalid predictions as 0 (don't fill them)
                        invalid_predictions += 1
                        cells_to_fill[i, j] = False  # Remove from filled count

        cnn_time = time.time() - cnn_start_time

        # Calculate CNN effectiveness
        cnn_filled_cells = np.sum(cells_to_fill)
        
        self.print_board(original_board, "ORIGINAL PUZZLE")
        self.print_board(cnn_partial_board, "AFTER CNN (HIGH-CONFIDENCE + VALID PREDICTIONS)")

        print(f"\nCNN RESULTS:")
        print(f"Time taken: {cnn_time:.4f} seconds")
        print(f"High-confidence cells found: {valid_predictions + invalid_predictions}")
        print(f"  Valid predictions: {valid_predictions}")
        print(f"  Invalid predictions: {invalid_predictions} (not filled)")
        print(f"Cells actually filled by CNN: {cnn_filled_cells}")
        print(f"Remaining empty cells: {np.sum(cnn_partial_board == 0)}")

        # Check if CNN solved it completely
        if np.sum(cnn_partial_board == 0) == 0:
            print("\nCNN SOLVED THE PUZZLE COMPLETELY!")
            if self.is_valid_sudoku_solution(cnn_partial_board):
                print("Solution is valid!")
                return cnn_partial_board, cnn_time, 0, True
            else:
                print("CNN solution is invalid - falling back to backtracking")
                # Reset to original and use pure backtracking
                cnn_partial_board = original_board.copy()

        # Step 2: Use backtracking for remaining empty cells
        remaining_empty = np.sum(cnn_partial_board == 0)
        print(f"\n2. RUNNING BACKTRACKING FOR {remaining_empty} REMAINING CELLS...")
        backtrack_start_time = time.time()

        # Use backtracking to solve remaining cells
        backtrack_board = cnn_partial_board.copy()
        steps = []

        success = solve_backtrack(backtrack_board, steps)
        backtrack_time = time.time() - backtrack_start_time

        if success:
            self.print_board(backtrack_board, "FINAL SOLUTION")

            print(f"\nBACKTRACKING RESULTS:")
            print(f"Time taken: {backtrack_time:.4f} seconds")
            print(f"Number of backtracking steps: {len(steps)}")

            # Verify final solution
            if self.is_valid_sudoku_solution(backtrack_board):
                print("PUZZLE SOLVED CORRECTLY!")
                return backtrack_board, cnn_time, backtrack_time, True
            else:
                print("Final solution is invalid")
                return None, cnn_time, backtrack_time, False

        else:
            print("Backtracking failed to solve the puzzle")
            return None, cnn_time, backtrack_time, False


    def is_valid_sudoku_solution(self, board):
        """Check if the completed Sudoku solution is valid"""
        # Check if all cells are filled
        if np.any(board == 0):
            return False
        
        # Check rows
        for row in board:
            if len(set(row)) != 9 or not set(row) == set(range(1, 10)):
                return False
        
        # Check columns
        for col in range(9):
            column = board[:, col]
            if len(set(column)) != 9 or not set(column) == set(range(1, 10)):
                return False
        
        # Check 3x3 boxes
        for box_row in range(3):
            for box_col in range(3):
                box = board[box_row*3:(box_row+1)*3, box_col*3:(box_col+1)*3]
                box_values = box.flatten()
                if len(set(box_values)) != 9 or not set(box_values) == set(range(1, 10)):
                    return False
        
        return True

