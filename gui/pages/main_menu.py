from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

class MainMenuPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.selected_algorithm = None
        self.selected_difficulty = None
        self.selected_enhancement = None
        self.init_ui()
    
    def init_ui(self):
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.setSpacing(30)
        
        # Title
        title = QLabel("Welcome to the ultimate sudoku solver!")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setStyleSheet("color: #a855f7; margin-bottom: 20px;")
        title.setAlignment(Qt.AlignCenter)
        
        # Subtitle
        subtitle = QLabel("Choose an algorithm and level of hardness for solving a sudoku")
        subtitle.setFont(QFont("Arial", 14))
        subtitle.setStyleSheet("color: white; margin-bottom: 40px;")
        subtitle.setAlignment(Qt.AlignCenter)
        
        main_layout.addWidget(title)
        main_layout.addWidget(subtitle)
        
        # Button container
        button_container = QWidget()
        button_layout = QVBoxLayout(button_container)
        button_layout.setSpacing(20)
        
        # Algorithm buttons row
        algo_row = QHBoxLayout()
        algo_row.setSpacing(20)
        
        self.backtracking_btn = self.create_button("Backtracking", lambda: self.select_algorithm("Backtracking"))
        self.forward_checking_btn = self.create_button("Forward\nCheck", lambda: self.select_algorithm("Forward Check"))
        
        algo_row.addWidget(self.backtracking_btn)
        algo_row.addWidget(self.forward_checking_btn)
        
        # Difficulty buttons row
        diff_row = QHBoxLayout()
        diff_row.setSpacing(20)
        
        self.easy_btn = self.create_button("Easy", lambda: self.select_difficulty("Easy"))
        self.medium_btn = self.create_button("Medium", lambda: self.select_difficulty("Medium"))
        self.hard_btn = self.create_button("Hard", lambda: self.select_difficulty("Hard"))
        
        diff_row.addWidget(self.easy_btn)
        diff_row.addWidget(self.medium_btn)
        diff_row.addWidget(self.hard_btn)
        
        # Enhancement label (separate from buttons)
        enhancement_label = QLabel("Enhancement:")
        enhancement_label.setFont(QFont("Arial", 14, QFont.Bold))
        enhancement_label.setStyleSheet("color: white;")
        enhancement_label.setAlignment(Qt.AlignCenter)
        
        # Enhancement buttons row
        enhancement_row = QHBoxLayout()
        enhancement_row.setSpacing(20)
        
        self.ac3_btn = self.create_button("AC3", lambda: self.select_enhancement("AC3"))
        self.no_enhancement_btn = self.create_button("None", lambda: self.select_enhancement("None"))
        
        enhancement_row.addWidget(self.ac3_btn)
        enhancement_row.addWidget(self.no_enhancement_btn)
        
        # Add all layouts to button container
        button_layout.addLayout(algo_row)
        button_layout.addLayout(diff_row)
        button_layout.addWidget(enhancement_label)
        button_layout.addLayout(enhancement_row)
        
        # Start button
        start_btn = self.create_button("Start Solving!", self.start_solving)
        start_btn.setFixedSize(200, 60)
        start_btn.setStyleSheet("""
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
            QPushButton:disabled {
                background-color: #6b7280;
            }
        """)
        start_btn.setEnabled(False)
        self.start_button = start_btn
        
        main_layout.addWidget(button_container)
        main_layout.addWidget(start_btn, alignment=Qt.AlignCenter)
    
    def create_button(self, text, callback):
        button = QPushButton(text)
        button.setFont(QFont("Arial", 12, QFont.Bold))
        button.setFixedSize(150, 60)
        button.setStyleSheet("""
            QPushButton {
                background-color: none;
                color: white;
                border: 2px solid #a855f7;
                border-radius: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #a855f7;
            }
            QPushButton:pressed {
                background-color: #9333ea;
            }
        """)
        button.clicked.connect(callback)
        return button
    
    def select_algorithm(self, algorithm):
        self.selected_algorithm = algorithm
        self.update_button_styles()
        self.check_ready_to_start()
    
    def select_enhancement(self, enhancement):
        self.selected_enhancement = enhancement
        self.update_button_styles()
        self.check_ready_to_start()
    
    def select_difficulty(self, difficulty):
        self.selected_difficulty = difficulty
        self.update_button_styles()
        self.check_ready_to_start()
    
    def update_button_styles(self):
        # Reset all buttons to default style
        buttons = [self.backtracking_btn, self.forward_checking_btn, self.ac3_btn, self.no_enhancement_btn,
                  self.easy_btn, self.medium_btn, self.hard_btn]
        
        for btn in buttons:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: none;
                    color: white;
                    border: 2px solid #a855f7;
                    border-radius: 10px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #a855f7;
                }
                QPushButton:pressed {
                    background-color: #9333ea;
                    border: 2px solid white;
                }
            """)
        
        # Highlight selected buttons
        selected_style = """
            QPushButton {
                background-color: #a855f7;
                color: white;
                border: 2px solid white;
                border-radius: 10px;
                font-weight: bold;
            }
            
        """
        #QPushButton:hover {
        #        background-color: #059669;
        #    }
        
        # Algorithm selection
        if self.selected_algorithm == "Backtracking":
            self.backtracking_btn.setStyleSheet(selected_style)
        elif self.selected_algorithm == "Forward Check":
            self.forward_checking_btn.setStyleSheet(selected_style)
        
        # Enhancement selection
        if self.selected_enhancement == "AC3":
            self.ac3_btn.setStyleSheet(selected_style)
        elif self.selected_enhancement == "None":
            self.no_enhancement_btn.setStyleSheet(selected_style)
        
        # Difficulty selection
        if self.selected_difficulty == "Easy":
            self.easy_btn.setStyleSheet(selected_style)
        elif self.selected_difficulty == "Medium":
            self.medium_btn.setStyleSheet(selected_style)
        elif self.selected_difficulty == "Hard":
            self.hard_btn.setStyleSheet(selected_style)
    
    def check_ready_to_start(self):
        if self.selected_algorithm and self.selected_difficulty and self.selected_enhancement:
            self.start_button.setEnabled(True)
    
    def get_final_algorithm(self):
        """Combine algorithm and enhancement for display"""
        if self.selected_enhancement == "AC3":
            return f"AC3 + {self.selected_algorithm}"
        else:
            return self.selected_algorithm
    
    def start_solving(self):
        final_algorithm = self.get_final_algorithm()
        self.main_window.show_solver_page(final_algorithm, self.selected_difficulty)