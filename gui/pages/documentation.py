from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

class DocumentationPage(QWidget):
    def __init__(self, main_window, algorithm, difficulty):
        super().__init__()
        self.main_window = main_window
        self.algorithm = algorithm
        self.difficulty = difficulty
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Navigation arrows
        nav_layout = QHBoxLayout()
        
        back_btn = QPushButton("◀")
        back_btn.setFont(QFont("Arial", 16, QFont.Bold))
        back_btn.setFixedSize(50, 50)
        back_btn.setStyleSheet("""
            QPushButton {
                background-color: #a855f7;
                color: white;
                border: none;
                border-radius: 25px;
            }
            QPushButton:hover {
                background-color: #9333ea;
            }
        """)
        back_btn.clicked.connect(lambda: self.main_window.show_analytics_page2(self.algorithm, self.difficulty))
        
        nav_layout.addWidget(back_btn)
        nav_layout.addStretch()
        
        layout.addLayout(nav_layout)
        
        # Documentation content
        content_container = QWidget()
        content_layout = QVBoxLayout(content_container)
        content_layout.setSpacing(20)
        
        # Title
        title = QLabel("Project Documentation")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setStyleSheet("color: #a855f7; margin: 20px;")
        title.setAlignment(Qt.AlignCenter)
        
        # Documentation text
        doc_text = f"""
        <h2 style="color: #a855f7;">Sudoku Solver Project</h2>

        <h3 style="color: #10b981;">Sudoku description</h3>
        <p style="color: white; line-height: 1.6;">Sudoku is a logic-based number puzzle played on a 9x9 grid, divided into nine 3x3 subgrids (also called boxes or regions). The goal is to fill the grid so that: <br/>
        • Each row contains the digits 1 to 9, with no repeats.<br/>
        • Each column contains the digits 1 to 9, with no repeats.<br/>
        • Each 3x3 box contains the digits 1 to 9, with no repeats.<br/>
        A typical Sudoku puzzle starts with some cells already filled. Using logical reasoning and deduction, the player fills in the rest to complete the grid correctly.<br/>
        Sudoku doesn’t require math—just pattern recognition and logical thinking.
        </p>        

        <h3 style="color: #10b981;">Overview</h3>
        <p style="color: white; line-height: 1.6;">
        This project implements and compares two different algorithms and one AI enhancement for solving Sudoku puzzles. 
        Each algorithm has unique characteristics and performance metrics that make them suitable for 
        different scenarios.
        </p>
        
        <h3 style="color: #10b981;">Algorithms Implemented</h3>
        
        <h4 style="color: #a855f7;">1. Backtracking Algorithm</h4>
        <p style="color: white; line-height: 1.6;">
        Backtracking is a general algorithmic technique for solving problems incrementally by trying partial solutions and then abandoning ("backtracking") them if they don’t lead to a valid final solution. It’s often used for constraint satisfaction problems, where you need to explore all possible combinations to find the correct one.<br/>
        Key Idea:<br/>
        • Choose an option.<br/>
        • Recurse (move forward).<br/>
        • If it fails, undo the choice and try the next option.<br/><br/>
        Backtracking in Sudoku works like this:
        <br/>
        • Find the first empty cell.<br/>
        • Try placing digits 1 to 9 in that cell.<br/>
        • For each digit, check if it's valid (not in same row, column, or box).<br/>
        • If valid, place it and recurse to solve the rest.<br/>
        • If the next step fails, undo (backtrack) and try the next digit.<br/>
        • Repeat until the board is solved or all options are exhausted.<br/><br/>
        Why backtracking is suitable:<br/>
            • Sudoku has a clear set of rules (constraints).<br/>
            • It requires exploring combinations, but not necessarily all if invalid options are pruned early.<br/>
        </p>
        
        <h4 style="color: #a855f7;">2. Forward Checking Algorithm</h4>
        <p style="color: white; line-height: 1.6;">
        An enhanced version of backtracking that maintains consistency by checking constraints 
        before making assignments. It eliminates values from the domains of future variables 
        that would conflict with the current assignment, reducing the search space significantly.<br/><br/>
        How Forward Checking Helps:<br/>
        • When you assign a digit to a cell, forward checking removes that digit from the domains of all cells in the same row, column, and box.<br/>
        • If any neighboring cell ends up with an empty domain (no valid choices), backtrack immediately.<br/>
        • This early detection reduces unnecessary recursion and speeds up solving.<br/>
        Benefit:<br/>
        • Prevents wasting time on assignments that will fail later.<br/>
        • Works well with backtracking to prune the search tree efficiently.
        </p>

        <h4 style="color: #a855f7;">3. AC3 (Arc Consistency) Enhancement</h4>
        <p style="color: white; line-height: 1.6;">
        AC-3 (Arc Consistency Algorithm #3) is a constraint propagation technique used in constraint satisfaction problems (CSPs) to reduce variable domains by enforcing arc consistency.
        An arc (𝑋→𝑌) is arc consistent if, for every value in 𝑋's domain, there is at least one valid value in 𝑌's domain that satisfies the constraint between 𝑋 and 𝑌.<br/><br/>
        The algorithm maintains a queue of arcs and iteratively:<br/>
        1. Removes an arc (𝑋→𝑌) from the queue.<br/>
        2. Removes inconsistent values from 𝑋's domain.<br/>
        3. If 𝑋's domain was changed, it re-adds all related arcs (𝑍→𝑋) to the queue.<br/> <br/> 
        How AC-3 Helps: <br/> 
        • Initially, reduce domains of each cell based on known values. <br/> 
        • Apply AC-3 to propagate constraints: <br/> 
        • If a cell can only be one value, remove that value from its neighbors. <br/> 
        • Repeat until no further reduction is possible. <br/> 
        • Combine with backtracking or forward checking for efficient solving. <br/> <br/> 
        Benefit: <br/> 
        • AC-3 can prune many invalid values before deeper search starts. <br/> 
        • Often solves easy/medium puzzles on its own or narrows them down significantly.<br/> 
        </p>
        
        <h3 style="color: #10b981;">Difficulty Levels</h3>
        <p style="color: white; line-height: 1.6;">
        • <strong>Easy:</strong> 40+ given clues, multiple solution paths<br>
        • <strong>Medium:</strong> 30-40 given clues, moderate complexity<br>
        • <strong>Hard:</strong> 25-30 given clues, requires advanced techniques
        </p>
        
        <h3 style="color: #10b981;">Performance Analysis</h3>
        <p style="color: white; line-height: 1.6;">
        The application provides analytics comparing an algorithm's performance across different 
        difficulty levels and also analytics comparing it to other algorithms.
        </p>
        
        <h3 style="color: #10b981;">Technical Implementation</h3>
        <p style="color: white; line-height: 1.6;">
        Built using PyQt5 for the graphical interface, with modular design separating algorithms, 
        GUI components, and utility functions. The architecture allows for easy extension with 
        additional algorithms or analysis features.

        Algorithms are implemented using recursion and backtracking techniques,
        with optimizations like forward checking and AC-3 for efficiency. The GUI provides step-by-step
        solution after solving, with detailed performance metrics displayed after completion.
        </p>
        
        <h3 style="color: #10b981;">Usage</h3>
        <p style="color: white; line-height: 1.6;">
        1. Select an algorithm from the main menu<br>
        2. Choose difficulty level<br>
        3. Watch the solving process step-by-step<br>
        4. Review detailed performance analytics<br>
        5. Compare results across different algorithms and difficulties
        </p>
        
        <h3 style="color: #10b981;">Future Enhancements</h3>
        <p style="color: white; line-height: 1.6;">
        • Custom puzzle input<br>
        • Additional solving algorithms (Dancing Links, Genetic Algorithm)<br>
        • Advanced visualization features<br>
        • Performance benchmarking tools<br>
        • Export functionality for results
        </p>
        """
        
        doc_label = QLabel(doc_text)
        doc_label.setFont(QFont("Arial", 12))
        doc_label.setWordWrap(True)
        doc_label.setAlignment(Qt.AlignLeft)
        doc_label.setMargin(20)
        
        content_layout.addWidget(title)
        content_layout.addWidget(doc_label)
        
        # Scroll area for documentation
        scroll = QScrollArea()
        scroll.setWidget(content_container)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: transparent; border: none;")
        
        layout.addWidget(scroll)
        
        # Next button at bottom
        next_btn = QPushButton("Next →")
        next_btn.setFont(QFont("Arial", 14, QFont.Bold))
        next_btn.setFixedSize(150, 50)
        next_btn.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                color: white;
                border: none;
                border-radius: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #059669;
            }
            QPushButton:pressed {
                background-color: #047857;
            }
        """)
        next_btn.clicked.connect(self.main_window.show_credits_page)
        
        layout.addWidget(next_btn, alignment=Qt.AlignCenter)