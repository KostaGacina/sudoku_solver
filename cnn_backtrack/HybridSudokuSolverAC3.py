# In[6]:


from cnn_backtrack.hybridarc3 import evaluate_solution, solve_backtrack
from solver_backtracking import solve_backtrack
from utils import evaluate_solution


import numpy as np
from keras.models import load_model


import time


class HybridSudokuSolver:
    def __init__(self, model_path, confidence_threshold=0.95, use_ac3=True):
        self.model = load_model(model_path)
        self.confidence_threshold = confidence_threshold
        self.use_ac3 = use_ac3

    def initialize_domains(self, board):
        """Initialize domains for AC3 based on the current board state"""
        domains = {}
        for i in range(9):
            for j in range(9):
                pos = (i, j)
                if board[i][j] == 0:
                    # Start with all possible values
                    domains[pos] = set(range(1, 10))
                    # Remove values that violate constraints
                    for val in list(domains[pos]):
                        if not self.is_valid_move(board, i, j, val):
                            domains[pos].discard(val)
                else:
                    domains[pos] = {board[i][j]}
        return domains

    def get_sudoku_constraints(self):
        """Generate all constraints for Sudoku puzzle"""
        constraints = set()

        # Row constraints
        for i in range(9):
            for j in range(9):
                for k in range(j + 1, 9):
                    constraints.add(((i, j), (i, k)))

        # Column constraints
        for j in range(9):
            for i in range(9):
                for k in range(i + 1, 9):
                    constraints.add(((i, j), (k, j)))

        # Box constraints
        for box_row in range(3):
            for box_col in range(3):
                cells = []
                for r in range(3):
                    for c in range(3):
                        cells.append((box_row * 3 + r, box_col * 3 + c))

                for i in range(len(cells)):
                    for j in range(i + 1, len(cells)):
                        constraints.add((cells[i], cells[j]))

        return list(constraints)

    def revise(self, domains, Xi, Xj):
        """Revise domain of Xi to be arc-consistent with Xj"""
        revised = False
        for i in list(domains[Xi]):
            found = False
            for j in domains[Xj]:
                if i != j:  # Sudoku constraint: values must be different
                    found = True
                    break
            if not found:
                domains[Xi].remove(i)
                revised = True
        return revised

    def AC3(self, domains, constraints):
        """AC3 algorithm for arc consistency"""
        agenda = []
        for (Xi, Xj) in constraints:
            agenda.append((Xi, Xj))
            agenda.append((Xj, Xi))

        arcs = agenda.copy()

        while len(agenda) > 0:
            (Xi, Xj) = agenda.pop()
            xi_copy = domains[Xi].copy()

            if self.revise(domains, Xi, Xj):
                if len(domains[Xi]) == 0:
                    return False  # No solution exists

            if domains[Xi] != xi_copy:
                # Add all arcs (Xk, Xi) back to agenda
                for cur_arc in arcs:
                    if cur_arc[1] == Xi and cur_arc[0] != Xj:
                        agenda.append(cur_arc)

        return True

    def apply_ac3_preprocessing(self, board):
        """Apply AC3 to reduce domains and possibly fill some cells"""
        print("\n0. APPLYING AC3 PREPROCESSING...")
        ac3_start_time = time.time()

        # Initialize domains based on current board
        domains = self.initialize_domains(board)

        # Get Sudoku constraints
        constraints = self.get_sudoku_constraints()

        # Run AC3
        is_consistent = self.AC3(domains, constraints)

        if not is_consistent:
            print("AC3 detected inconsistency - puzzle may be unsolvable")
            return board, 0, 0

        # Apply singleton domains (cells with only one possible value)
        ac3_board = board.copy()
        cells_filled = 0
        cells_reduced = 0

        for pos, domain in domains.items():
            i, j = pos
            if ac3_board[i][j] == 0:
                if len(domain) == 1:
                    # Only one possible value - fill it
                    ac3_board[i][j] = list(domain)[0]
                    cells_filled += 1
                elif len(domain) < 9:
                    cells_reduced += 1

        ac3_time = time.time() - ac3_start_time

        print(f"AC3 preprocessing completed in {ac3_time:.4f} seconds")
        print(f"Cells filled by AC3: {cells_filled}")
        print(f"Cells with reduced domains: {cells_reduced}")

        # Store domains for potential use in backtracking
        self.ac3_domains = domains

        return ac3_board, cells_filled, ac3_time

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
        """Use AC3, then CNN for high-confidence predictions, then backtracking for the rest"""
        original_board = np.array(puzzle, dtype=int)
        true_solution = np.array(true_solution, dtype=int)

        print("=" * 60)
        print("HYBRID SUDOKU SOLVER: AC3 + CNN + BACKTRACKING")
        print("=" * 60)

        # Step 0: Apply AC3 preprocessing if enabled
        if self.use_ac3:
            ac3_board, ac3_cells_filled, ac3_time = self.apply_ac3_preprocessing(original_board)

            if ac3_cells_filled > 0:
                self.print_board(ac3_board, "AFTER AC3 PREPROCESSING")

                # Check if AC3 solved it completely
                if np.array_equal(ac3_board, true_solution):
                    print("\n🎉 AC3 SOLVED THE PUZZLE COMPLETELY!")
                    return ac3_board, ac3_time, 0, 0, 1.0, 1.0

            # Use AC3-preprocessed board for CNN
            current_board = ac3_board
        else:
            current_board = original_board
            ac3_time = 0
            ac3_cells_filled = 0

        # Step 1: Run CNN and get high-confidence predictions
        print("\n1. RUNNING CNN PREDICTIONS...")
        cnn_start_time = time.time()

        predicted_digits, probabilities = self.get_model_predictions(current_board)
        cnn_partial_board = current_board.copy()

        # Only fill cells where CNN has high confidence AND the cell is empty AND the move is valid
        empty_mask = current_board == 0
        high_confidence_mask = probabilities > self.confidence_threshold
        cells_to_fill = empty_mask & high_confidence_mask

        print(f"CNN confidence threshold: {self.confidence_threshold}")
        print(f"High-confidence predictions: {np.sum(cells_to_fill)} cells")

        # If AC3 was used, also check against reduced domains
        valid_predictions = 0
        invalid_predictions = 0

        for i in range(9):
            for j in range(9):
                if cells_to_fill[i, j]:
                    predicted_value = predicted_digits[i, j]
                    is_valid = self.is_valid_move(cnn_partial_board, i, j, predicted_value)

                    # Additional check against AC3 domains if available
                    if self.use_ac3 and hasattr(self, 'ac3_domains'):
                        if (i, j) in self.ac3_domains:
                            is_valid = is_valid and (predicted_value in self.ac3_domains[(i, j)])

                    if is_valid:
                        cnn_partial_board[i, j] = predicted_value
                        valid_predictions += 1
                    else:
                        invalid_predictions += 1
                        cells_to_fill[i, j] = False

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
            return cnn_partial_board, ac3_time, cnn_time, 0, cnn_accuracy, 1.0

        # Step 2: Use backtracking for remaining empty cells
        remaining_empty = np.sum(cnn_partial_board == 0)
        print(f"\n2. RUNNING BACKTRACKING FOR {remaining_empty} REMAINING CELLS...")
        backtrack_start_time = time.time()

        # Use backtracking to solve remaining cells
        backtrack_board = cnn_partial_board.copy()
        steps = []

        # Use AC3 domains in backtracking if available
        if self.use_ac3 and hasattr(self, 'ac3_domains'):
            success = self.solve_backtrack_with_domains(backtrack_board, steps, self.ac3_domains)
        else:
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

            # Summary
            print("\n" + "=" * 60)
            print("SUMMARY:")
            print(f"AC3 preprocessing: {ac3_time:.4f}s, filled {ac3_cells_filled} cells")
            print(f"CNN predictions: {cnn_time:.4f}s, filled {cnn_filled_cells} cells")
            print(f"Backtracking: {backtrack_time:.4f}s, filled {remaining_empty} cells")
            print(f"Total time: {ac3_time + cnn_time + backtrack_time:.4f}s")

            # Verify final solution
            if np.array_equal(backtrack_board, true_solution):
                print("PUZZLE SOLVED CORRECTLY!")
            else:
                print("Final solution does not match expected result")
                print("Expected solution:")
                self.print_board(true_solution, "EXPECTED SOLUTION")

            return backtrack_board, ac3_time, cnn_time, backtrack_time, cnn_accuracy, final_all_acc

        else:
            print("Backtracking failed to solve the puzzle")
            return None, ac3_time, cnn_time, backtrack_time, cnn_accuracy, 0.0

    def solve_backtrack_with_domains(self, board, steps, domains):
        """Backtracking with AC3 domain information for optimization"""
        # Find empty cell with smallest domain (MRV heuristic)
        min_domain_size = 10
        best_cell = None

        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    if (i, j) in domains:
                        domain_size = len(domains[(i, j)])
                        if domain_size < min_domain_size:
                            min_domain_size = domain_size
                            best_cell = (i, j)

        if best_cell is None:
            return True  # No empty cells, puzzle solved

        row, col = best_cell

        # Try values from the domain
        if best_cell in domains:
            for num in domains[best_cell]:
                if self.is_valid_move(board, row, col, num):
                    board[row][col] = num
                    steps.append((row, col, num))

                    if self.solve_backtrack_with_domains(board, steps, domains):
                        return True

                    board[row][col] = 0
                    steps.append((row, col, 0))

        return False
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
    def solve_sudoku_hybrid(self, puzzle):
        """Use AC3, then CNN for high-confidence predictions, then backtracking for the rest"""
        original_board = np.array(puzzle, dtype=int)

        print("=" * 60)
        print("HYBRID SUDOKU SOLVER: AC3 + CNN + BACKTRACKING")
        print("=" * 60)

        # Step 0: Apply AC3 preprocessing if enabled
        if self.use_ac3:
            ac3_board, ac3_cells_filled, ac3_time = self.apply_ac3_preprocessing(original_board)

            if ac3_cells_filled > 0:
                self.print_board(ac3_board, "AFTER AC3 PREPROCESSING")

                # Check if AC3 solved it completely
                if np.sum(ac3_board == 0) == 0:
                    print("\n🎉 AC3 SOLVED THE PUZZLE COMPLETELY!")
                    if self.is_valid_sudoku_solution(ac3_board):
                        print("Solution is valid!")
                        return ac3_board, ac3_time, 0, 0, True
                    else:
                        print("AC3 solution is invalid - continuing with CNN")

            # Use AC3-preprocessed board for CNN
            current_board = ac3_board
        else:
            current_board = original_board
            ac3_time = 0
            ac3_cells_filled = 0

        # Step 1: Run CNN and get high-confidence predictions
        print("\n1. RUNNING CNN PREDICTIONS...")
        cnn_start_time = time.time()

        predicted_digits, probabilities = self.get_model_predictions(current_board)
        cnn_partial_board = current_board.copy()

        # Only fill cells where CNN has high confidence AND the cell is empty AND the move is valid
        empty_mask = current_board == 0
        high_confidence_mask = probabilities > self.confidence_threshold
        cells_to_fill = empty_mask & high_confidence_mask

        print(f"CNN confidence threshold: {self.confidence_threshold}")
        print(f"High-confidence predictions: {np.sum(cells_to_fill)} cells")

        # If AC3 was used, also check against reduced domains
        valid_predictions = 0
        invalid_predictions = 0

        for i in range(9):
            for j in range(9):
                if cells_to_fill[i, j]:
                    predicted_value = predicted_digits[i, j]
                    is_valid = self.is_valid_move(cnn_partial_board, i, j, predicted_value)

                    # Additional check against AC3 domains if available
                    if self.use_ac3 and hasattr(self, 'ac3_domains'):
                        if (i, j) in self.ac3_domains:
                            is_valid = is_valid and (predicted_value in self.ac3_domains[(i, j)])

                    if is_valid:
                        cnn_partial_board[i, j] = predicted_value
                        valid_predictions += 1
                    else:
                        invalid_predictions += 1
                        cells_to_fill[i, j] = False

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
            print("\n🎉 CNN SOLVED THE PUZZLE COMPLETELY!")
            if self.is_valid_sudoku_solution(cnn_partial_board):
                print("Solution is valid!")
                return cnn_partial_board, ac3_time, cnn_time, 0, True
            else:
                print("CNN solution is invalid - falling back to backtracking")
                # Reset to current_board and use pure backtracking
                cnn_partial_board = current_board.copy()

        # Step 2: Use backtracking for remaining empty cells
        remaining_empty = np.sum(cnn_partial_board == 0)
        print(f"\n2. RUNNING BACKTRACKING FOR {remaining_empty} REMAINING CELLS...")
        backtrack_start_time = time.time()

        # Use backtracking to solve remaining cells
        backtrack_board = cnn_partial_board.copy()
        steps = []

        # Use AC3 domains in backtracking if available
        if self.use_ac3 and hasattr(self, 'ac3_domains'):
            success = self.solve_backtrack_with_domains(backtrack_board, steps, self.ac3_domains)
        else:
            success = solve_backtrack(backtrack_board, steps)

        backtrack_time = time.time() - backtrack_start_time

        if success:

            print(f"\nBACKTRACKING RESULTS:")
            print(f"Time taken: {backtrack_time:.4f} seconds")
            print(f"Number of backtracking steps: {len(steps)}")

            # Summary
            print("\n" + "=" * 60)
            print("SUMMARY:")
            self.print_board(backtrack_board)
            print(f"AC3 preprocessing: {ac3_time:.4f}s, filled {ac3_cells_filled} cells")
            print(f"CNN predictions: {cnn_time:.4f}s, filled {cnn_filled_cells} cells")
            print(f"Backtracking: {backtrack_time:.4f}s, filled {remaining_empty} cells")
            print(f"Total time: {ac3_time + cnn_time + backtrack_time:.4f}s")

            # Verify final solution
            if self.is_valid_sudoku_solution(backtrack_board):
                print("PUZZLE SOLVED CORRECTLY!")
                return backtrack_board, ac3_time, cnn_time, backtrack_time, True
            else:
                print("Final solution is invalid")
                return None, ac3_time, cnn_time, backtrack_time, False

        else:
            print("Backtracking failed to solve the puzzle")
            return None, ac3_time, cnn_time, backtrack_time, False