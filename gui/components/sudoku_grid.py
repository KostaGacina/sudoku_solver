from PyQt5.QtWidgets import QWidget, QGridLayout, QLabel
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont
import random
import time
import copy
import sys
import os
import numpy as np

# Add parent directory to path to import benchmark modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

try:
    from solver_backtracking import solve_backtrack
    from solver_forward_check import initialize_domains, solve_forward_check
    from ai_solver import solve_ac3_backtrack, solve_ac3_forward
except ImportError as e:
    print(f"Import error in sudoku_grid: {e}")

class SolvingThread(QThread):
    solution_ready = pyqtSignal(list, float)  # Change to list for steps
    
    def __init__(self, board, algorithm):
        super().__init__()
        self.board = self.prepare_board(board)
        self.algorithm = algorithm
        self.solution = None
        self.solve_time = 0
    
    def prepare_board(self, board):
        """Convert board to numpy array format for the solving functions"""
        try:
            # If it's already a numpy array, return as is
            if isinstance(board, np.ndarray):
                return board
            
            # Convert list to numpy array
            if isinstance(board, list):
                if len(board) == 9 and isinstance(board[0], list) and len(board[0]) == 9:
                    # 2D list
                    return np.array(board)
                elif len(board) == 81:
                    # Flat list
                    return np.array(board).reshape(9, 9)
            
            print(f"Unknown board format: {type(board)}")
            return None
            
        except Exception as e:
            print(f"Error preparing board: {e}")
            return None
    
    def run(self):
        try:
            if self.board is None:
                print("No valid board to solve")
                return
            
            start_time = time.time()
            
            # Create a copy for solving
            board_copy = self.board.copy()
            solving_steps = []
            
            print(f"Starting to solve with algorithm: {self.algorithm}")
            print(f"Board shape: {board_copy.shape}")
            print(f"Empty cells: {np.sum(board_copy == 0)}")
            
            if self.algorithm == "Backtracking":
                solution = solve_backtrack(board_copy, solving_steps)
            elif self.algorithm == "Forward Check":
                domains = initialize_domains(board_copy)
                solution = solve_forward_check(board_copy, domains, solving_steps)
            elif self.algorithm == "AC3 + Backtracking":
                # Pass solving_steps to capture steps after AC3 preprocessing
                solution = solve_ac3_backtrack(board_copy, solving_steps)
                print(f"AC3 + Backtracking completed with {len(solving_steps)} steps after preprocessing")
            elif self.algorithm == "AC3 + Forward Check":
                print("Attempting AC3 + Forward Checking...")
                try:
                    # Pass solving_steps to capture steps after AC3 preprocessing
                    solution = solve_ac3_forward(board_copy, solving_steps)
                    print(f"AC3 + Forward Checking result: {solution}")
                    print(f"Steps after AC3 preprocessing: {len(solving_steps)}")
                except Exception as e:
                    print(f"Error in AC3 + Forward Checking: {e}")
                    import traceback
                    traceback.print_exc()
                    solution = False
            else:
                solution = solve_backtrack(board_copy, solving_steps)

            end_time = time.time()
            self.solve_time = end_time - start_time
            
            print(f"Algorithm completed. Solution found: {solution}")
            print(f"Steps generated: {len(solving_steps)}")
            
            if solution:
                # Emit the solving steps, not the final board
                self.solution_ready.emit(solving_steps, self.solve_time)
            else:
                print("No solution found!")
                # Emit empty steps so the UI knows solving is complete
                self.solution_ready.emit([], self.solve_time)
                
        except Exception as e:
            print(f"Error in solving thread: {e}")
            import traceback
            traceback.print_exc()

class SudokuGrid(QWidget):
    solving_finished = pyqtSignal()
    
    def __init__(self, difficulty, board=None):
        super().__init__()
        self.difficulty = difficulty
        self.board = board
        self.grid = [[0 for _ in range(9)] for _ in range(9)]
        self.solution = None
        self.cells = []
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate_step)
        self.solving_thread = None
        self.animation_cells = []
        self.animation_index = 0
        self.actual_solve_time = 0
        self.init_ui()
        self.load_puzzle()
        self.solving_steps = []

    def init_ui(self):
        self.setFixedSize(450, 450)
        layout = QGridLayout(self)
        layout.setSpacing(2)
        
        # Create 9x9 grid of cells
        for row in range(9):
            cell_row = []
            for col in range(9):
                cell = QLabel("")
                cell.setFixedSize(48, 48)
                cell.setAlignment(Qt.AlignCenter)
                cell.setFont(QFont("Arial", 16, QFont.Bold))
                
                # Style based on 3x3 box position
                if (row // 3 + col // 3) % 2 == 0:
                    bg_color = "#3a3a3a"
                else:
                    bg_color = "#2a2a2a"
                
                cell.setStyleSheet(f"""
                    QLabel {{
                        background-color: {bg_color};
                        color: white;
                        border: 1px solid #a855f7;
                    }}
                """)
                
                layout.addWidget(cell, row, col)
                cell_row.append(cell)
            self.cells.append(cell_row)
        
        # Add thicker borders for 3x3 boxes
        self.add_box_borders()
    
    def add_box_borders(self):
        # Add visual separation for 3x3 boxes
        for row in range(9):
            for col in range(9):
                cell = self.cells[row][col]
                current_style = cell.styleSheet()
                
                # Add thicker borders at box boundaries
                border_style = "border: 1px solid #a855f7;"
                if row % 3 == 0 and row > 0:
                    border_style += " border-top: 3px solid #a855f7;"
                if col % 3 == 0 and col > 0:
                    border_style += " border-left: 3px solid #a855f7;"
                
                new_style = current_style.replace("border: 1px solid #a855f7;", border_style)
                cell.setStyleSheet(new_style)
    
    def get_cell_value(self, board, row, col):
        """Helper function to get cell value from any board format"""
        try:
            if isinstance(board, np.ndarray):
                return board[row, col]
            elif isinstance(board, list):
                if isinstance(board[0], list):
                    return board[row][col]
                else:
                    # Flat list
                    return board[row * 9 + col]
            return 0
        except:
            return 0
    
    def load_puzzle(self):
        """Load the puzzle into the display grid"""
        if self.board is not None:
            try:
                for row in range(9):
                    for col in range(9):
                        value = self.get_cell_value(self.board, row, col)
                        if value != 0:
                            self.grid[row][col] = int(value)
                            self.cells[row][col].setText(str(int(value)))
                            self.cells[row][col].setStyleSheet(
                                self.cells[row][col].styleSheet() + " color: #10b981; font-weight: bold;"
                            )
            except Exception as e:
                print(f"Error loading puzzle: {e}")
                self.generate_fallback_puzzle()
        else:
            # Generate fallback puzzle
            self.generate_fallback_puzzle()
    
    def generate_fallback_puzzle(self):
        """Generate a simple puzzle if CSV loading fails"""
        clues = {"Easy": 40, "Medium": 30, "Hard": 25}
        num_clues = clues.get(self.difficulty, 30)
        
        filled_positions = set()
        for _ in range(num_clues):
            while True:
                row, col = random.randint(0, 8), random.randint(0, 8)
                if (row, col) not in filled_positions:
                    filled_positions.add((row, col))
                    break
            
            num = random.randint(1, 9)
            self.grid[row][col] = num
            self.cells[row][col].setText(str(num))
            self.cells[row][col].setStyleSheet(
                self.cells[row][col].styleSheet() + " color: #10b981; font-weight: bold;"
            )
    
    def start_solving(self, algorithm):
        """Start the solving process using the actual algorithm"""
        if self.board is not None:
            # Start solving in background thread
            self.solving_thread = SolvingThread(self.board, algorithm)
            self.solving_thread.solution_ready.connect(self.on_solution_ready)
            self.solving_thread.start()
        else:
            print("No board to solve!")
    
    def on_solution_ready(self, solving_steps, solve_time):
        """Called when the solving algorithm completes"""
        self.solving_steps = solving_steps  # Store the steps
        self.actual_solve_time = solve_time
        
        print(f"Received {len(solving_steps)} solving steps")
        
        # Use the solving steps directly for animation
        self.animation_cells = []
        
        # Keep track of which cells were originally empty
        original_empty_cells = set()
        for row in range(9):
            for col in range(9):
                if self.grid[row][col] == 0:
                    original_empty_cells.add((row, col))
        
        print(f"Originally empty cells: {len(original_empty_cells)}")
        
        # Filter steps to only show the "placing" moves (not the backtracking zeros)
        for step in solving_steps:
            if hasattr(step, 'row') and hasattr(step, 'col') and hasattr(step, 'num'):  # Changed from 'value' to 'num'
                # Only animate non-zero values (actual placements, not backtracking)
                if (step.row, step.col) in original_empty_cells:
                    self.animation_cells.append((step.row, step.col, step.num))  # Changed from step.value to step.num
            elif isinstance(step, tuple) and len(step) == 3:
                # Handle if steps are tuples (row, col, value)
                row, col, value = step
                if value != 0 and (row, col) in original_empty_cells:
                    self.animation_cells.append((row, col, value))
        
        print(f"Animation will show {len(self.animation_cells)} cell placements")
        
        self.animation_index = 0
        if self.animation_cells:
            self.animation_timer.start(50)  # Fill one cell every 50ms
        else:
            self.solving_finished.emit()

    def animate_step(self):
        """Animate filling one cell"""
        if self.animation_index < len(self.animation_cells):
            row, col, value = self.animation_cells[self.animation_index]
            
            # Update the display
            self.cells[row][col].setText(str(value))
            self.cells[row][col].setStyleSheet(
                self.cells[row][col].styleSheet() + " color: #a855f7;"
            )
            
            # Update internal grid state
            self.grid[row][col] = value
            
            self.animation_index += 1
        else:
            # Animation complete
            self.animation_timer.stop()
            self.solving_finished.emit()
    
    def pause_solving(self):
        """Pause the animation"""
        self.animation_timer.stop()
        if self.solving_thread and self.solving_thread.isRunning():
            self.solving_thread.terminate()