from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from ..components.sudoku_grid import SudokuGrid
import time
import random
import sys
import os
import numpy as np

# Add parent directory to path to import benchmark modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

try:
    from board import load_boards_csv
    from solver_backtracking import solve_backtrack, solve_backtrack_domains
    from solver_forward_check import initialize_domains, solve_forward_check
    from ai_solver import solve_ac3_backtrack, solve_ac3_forward
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback - will use dummy data if imports fail

class SolverPage(QWidget):
    def __init__(self, main_window, algorithm, difficulty):
        super().__init__()
        self.main_window = main_window
        self.algorithm = algorithm
        self.difficulty = difficulty
        self.is_solving = False
        self.is_solved = False
        self.solve_time = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_time)
        self.board = None
        self.load_puzzle()
        self.init_ui()
    
    def load_puzzle(self):
        """Load a random puzzle from CSV based on difficulty"""
        try:
            if self.difficulty.lower() == "easy":
                boards = load_boards_csv("./puzzles/easy.csv")
            elif self.difficulty.lower() == "medium":
                boards = load_boards_csv("./puzzles/medium.csv")
            elif self.difficulty.lower() == "hard":
                boards = load_boards_csv("./puzzles/hard.csv")
            else:
                boards = load_boards_csv("./puzzles/easy.csv")  # fallback
            
            # Fix: Check if boards is not None and has length > 0
            if boards is not None and len(boards) > 0:
                # Convert numpy array to list if needed
                if isinstance(boards, np.ndarray):
                    boards = boards.tolist()
                
                self.board = random.choice(boards)
                print(f"Loaded {self.difficulty} puzzle with {len(boards)} available")
                
                # Convert board to regular Python list if it's numpy array
                if isinstance(self.board, np.ndarray):
                    self.board = self.board.tolist()
                    
            else:
                print("No boards loaded, using fallback")
                self.board = None
                
        except Exception as e:
            print(f"Error loading puzzle: {e}")
            import traceback
            traceback.print_exc()
            self.board = None
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)
        layout.setSpacing(20)
        
        # Top info row
        info_layout = QHBoxLayout()
        
        # Algorithm and Difficulty info
        algo_label = QLabel(f"Algorithm: {self.algorithm}")
        algo_label.setFont(QFont("Arial", 14, QFont.Bold))
        algo_label.setStyleSheet("color: #a855f7;")
        
        diff_label = QLabel(f"Difficulty: {self.difficulty}")
        diff_label.setFont(QFont("Arial", 14, QFont.Bold))
        diff_label.setStyleSheet("color: #a855f7;")
        
        # Solved time info
        self.time_label = QLabel("Solved in: --")
        self.time_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.time_label.setStyleSheet("color: #a855f7;")
        
        info_layout.addWidget(algo_label)
        info_layout.addStretch()
        info_layout.addWidget(diff_label)
        info_layout.addStretch()
        info_layout.addWidget(self.time_label)
        
        layout.addLayout(info_layout)
        
        # Sudoku grid
        self.sudoku_grid = SudokuGrid(self.difficulty, self.board)
        self.sudoku_grid.solving_finished.connect(self.on_solving_finished)
        layout.addWidget(self.sudoku_grid, alignment=Qt.AlignCenter)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        # Back button
        back_btn = QPushButton("Back to Menu")
        back_btn.setFont(QFont("Arial", 12, QFont.Bold))
        back_btn.setFixedSize(150, 50)
        back_btn.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                color: white;
                border: none;
                border-radius: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #dc2626;
            }
            QPushButton:pressed {
                background-color: #b91c1c;
            }
        """)
        back_btn.clicked.connect(self.main_window.show_main_menu)
        
        self.play_pause_btn = QPushButton("▶")
        self.play_pause_btn.setFont(QFont("Arial", 16, QFont.Bold))
        self.play_pause_btn.setFixedSize(60, 60)
        self.play_pause_btn.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                color: white;
                border: none;
                border-radius: 30px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #059669;
            }
            QPushButton:pressed {
                background-color: #047857;
            }
        """)
        self.play_pause_btn.clicked.connect(self.toggle_solving)
        
        # Next button (initially hidden)
        
        self.next_btn = QPushButton("Next ➡")
        self.next_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.next_btn.setFixedSize(150, 50)
        self.next_btn.setStyleSheet("""
            QPushButton {
                background-color: #f59e0b;
                color: white;
                border: none;
                border-radius: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d97706;
            }
            QPushButton:pressed {
                background-color: #b45309;
            }
        """)
        self.next_btn.clicked.connect(self.show_next_page)
        self.next_btn.setVisible(False)
        
        button_layout.addWidget(back_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.play_pause_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.next_btn)
        
        layout.addLayout(button_layout)
    
    def toggle_solving(self):
        if not self.is_solving and not self.is_solved:
            self.start_solving()
        else:
            self.pause_solving()
    
    def start_solving(self):
        self.is_solving = True
        self.play_pause_btn.setText("⏸")
        self.timer.start(100)  # Update every 100ms
        self.sudoku_grid.start_solving(self.algorithm)
    
    def pause_solving(self):
        self.is_solving = False
        self.play_pause_btn.setText("▶")
        self.timer.stop()
        self.sudoku_grid.pause_solving()
    
    def update_time(self):
        if self.is_solving:
            self.solve_time += 0.1
            self.time_label.setText(f"Solved in: {self.solve_time:.1f}s")
    
    def on_solving_finished(self):
        self.is_solving = False
        self.is_solved = True
        self.timer.stop()
        self.play_pause_btn.setVisible(False)
        self.next_btn.setVisible(True)
        # Get final solve time from the grid
        final_time = self.sudoku_grid.actual_solve_time
        self.time_label.setText(f"Solved in: {final_time:.4f}s")
    
    def show_next_page(self):
        self.main_window.show_analytics_page1(self.algorithm, self.difficulty)