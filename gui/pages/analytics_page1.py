from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QTransform
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import sys
import os

# Add parent directory to path to import benchmark modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from benchmark import get_benchmark_data, results  # Import the function and results
except ImportError as e:
    print(f"Could not import benchmark data: {e}")
    # Fallback function
    def get_benchmark_data():
        return {
            'Backtracking': {'easy': 0.045, 'medium': 0.182, 'hard': 0.634},
            'Forward Checking': {'easy': 0.032, 'medium': 0.128, 'hard': 0.445},
            'AC3 + Backtracking': {'easy': 0.028, 'medium': 0.095, 'hard': 0.312},
            'AC3 + Forward Checking': {'easy': 0.024, 'medium': 0.078, 'hard': 0.256}
        }

class AnalyticsPage1(QWidget):
    def __init__(self, main_window, algorithm, difficulty):
        super().__init__()
        self.main_window = main_window
        self.algorithm = algorithm
        self.difficulty = difficulty
        self.init_ui()
    
    def create_difficulty_chart(self):
        """Create a bar chart showing algorithm performance across difficulties"""
        data = get_benchmark_data()  # Call the imported function
        
        if self.algorithm not in data:
            # Return a placeholder if no data available
            placeholder = QLabel("No benchmark data available for this algorithm")
            placeholder.setStyleSheet("color: white; padding: 40px; border: 1px solid #a855f7;")
            placeholder.setAlignment(Qt.AlignCenter)
            return placeholder
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#1a1a1a')  # Dark background
        ax.set_facecolor('#1a1a1a')
        
        # Data for the selected algorithm
        difficulties = ['Easy', 'Medium', 'Hard']
        times = [
            data[self.algorithm]['easy'],
            data[self.algorithm]['medium'],
            data[self.algorithm]['hard']
        ]
        
        # Create bar chart
        bars = ax.bar(difficulties, times, color='#a855f7', alpha=0.8, edgecolor='#9333ea', linewidth=2)
        
        # Customize the chart
        ax.set_xlabel('Difficulty Level', color='white', fontsize=12)
        ax.set_ylabel('Average Time (seconds)', color='white', fontsize=12)
        
        # Style the axes
        ax.tick_params(colors='white', labelsize=11)
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')
        
        # Add value labels on top of bars
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{time_val:.3f}s', ha='center', va='bottom', color='white', fontweight='bold')
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, color='white')
        ax.set_axisbelow(True)
        
        # Highlight the current difficulty
        if self.difficulty.lower() in ['easy', 'medium', 'hard']:
            difficulty_index = ['easy', 'medium', 'hard'].index(self.difficulty.lower())
            bars[difficulty_index].set_color('#10b981')  # Green for current difficulty
            bars[difficulty_index].set_alpha(1.0)
        
        plt.tight_layout()
        
        # Create Qt widget from matplotlib figure
        canvas = FigureCanvas(fig)
        canvas.setMinimumHeight(400)
        return canvas
    
    def create_comparison_chart(self, comparison_algorithm):
        """Create a comparison chart between two algorithms"""
        data = get_benchmark_data()
        
        if self.algorithm not in data or comparison_algorithm not in data:
            placeholder = QLabel("No comparison data available")
            placeholder.setStyleSheet("color: white; padding: 40px; border: 1px solid #a855f7;")
            placeholder.setAlignment(Qt.AlignCenter)
            return placeholder
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#1a1a1a')
        
        difficulties = ['Easy', 'Medium', 'Hard']
        algorithm1_times = [data[self.algorithm][d.lower()] for d in difficulties]
        algorithm2_times = [data[comparison_algorithm][d.lower()] for d in difficulties]
        
        x = np.arange(len(difficulties))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, algorithm1_times, width, label=self.algorithm, 
                      color='#a855f7', alpha=0.8)
        bars2 = ax.bar(x + width/2, algorithm2_times, width, label=comparison_algorithm, 
                      color='#10b981', alpha=0.8)
        
        ax.set_title(f'{self.algorithm} vs {comparison_algorithm}', 
                    color='white', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Difficulty Level', color='white', fontsize=12)
        ax.set_ylabel('Average Time (seconds)', color='white', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(difficulties)
        
        # Style the chart
        ax.tick_params(colors='white', labelsize=11)
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{height:.3f}s', ha='center', va='bottom', color='white', fontsize=9)
        
        ax.legend(facecolor='#2a2a2a', edgecolor='white', labelcolor='white')
        ax.grid(True, alpha=0.3, color='white')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        canvas = FigureCanvas(fig)
        canvas.setMinimumHeight(400)
        return canvas

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Navigation arrows
        nav_layout = QHBoxLayout()
        
        back_btn = QPushButton("⤺")
        back_btn.setFont(QFont("Arial", 20, QFont.Bold))
        back_btn.setFixedSize(50, 50)
        back_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: white;
                border: none;
                border-radius: 25px;
            }
            QPushButton:hover {
                background-color: #a855f7;
            }
        """)
        back_btn.clicked.connect(lambda: self.main_window.show_solver_page(self.algorithm, self.difficulty))
        
        next_btn = QPushButton("⤻")
        next_btn.setFont(QFont("Arial", 20, QFont.Bold))
        next_btn.setFixedSize(50, 50)
        next_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: white;
                border: none;
                border-radius: 25px;
            }
            QPushButton:hover {
                background-color: #a855f7;
            }
        """)
        next_btn.clicked.connect(lambda: self.main_window.show_analytics_page2(self.algorithm, self.difficulty))
        
        nav_layout.addWidget(back_btn)
        nav_layout.addStretch()
        nav_layout.addWidget(next_btn)
        
        layout.addLayout(nav_layout)
        
        # Content container
        content_container = QWidget()
        content_layout = QVBoxLayout(content_container)
        content_layout.setSpacing(30)
        
        # Get other algorithms for comparison
        all_algorithms = [
            'Backtracking', 
            'Forward Check',  # Changed from 'Forward Checking'
            'AC3 + Backtracking', 
            'AC3 + Forward Check'  # Changed from 'AC3 + Forward Checking'
        ]
        
        other_algos = [algo for algo in all_algorithms if algo != self.algorithm]
        
        # Section 1: Algorithm performance across difficulty levels
        title1 = QLabel(f"How algorithm [{self.algorithm}] behaves depending on difficulty level")
        title1.setFont(QFont("Arial", 16, QFont.Bold))
        title1.setStyleSheet("color: #a855f7; margin: 10px;")
        title1.setAlignment(Qt.AlignCenter)
        title1.setWordWrap(True)
        
        content_layout.addWidget(title1)
        content_layout.addWidget(self.create_difficulty_chart())
        
        # Section 2: Algorithm comparison (if other algorithms available)
        if len(other_algos) >= 1:
            if self.algorithm == "Forward Check":
                comp_algo = "Backtracking"
            elif self.algorithm == "AC3 + Forward Check":
                comp_algo = "AC3 + Backtracking"
            elif self.algorithm == "Backtracking":
                comp_algo = "Forward Check"
            else:
                comp_algo = "AC3 + Forward Check"
            title2 = QLabel(f"How algorithm [{self.algorithm}] compares to [{comp_algo}]")
            title2.setFont(QFont("Arial", 16, QFont.Bold))
            title2.setStyleSheet("color: #a855f7; margin: 10px;")
            title2.setAlignment(Qt.AlignCenter)
            title2.setWordWrap(True)
            
            content_layout.addWidget(title2)
            content_layout.addWidget(self.create_comparison_chart(comp_algo))
        
        if len(other_algos) >= 1:
            if self.algorithm == "Forward Check":
                comp_algo = "AC3 + Forward Check"
            elif self.algorithm == "AC3 + Forward Check":
                comp_algo = "Forward Check"
            elif self.algorithm == "Backtracking":
                comp_algo = "AC3 + Backtracking"
            else:
                comp_algo = "Backtracking"
            title2 = QLabel(f"How algorithm [{self.algorithm}] compares to [{comp_algo}]")
            title2.setFont(QFont("Arial", 16, QFont.Bold))
            title2.setStyleSheet("color: #a855f7; margin: 10px;")
            title2.setAlignment(Qt.AlignCenter)
            title2.setWordWrap(True)
            
            content_layout.addWidget(title2)
            content_layout.addWidget(self.create_comparison_chart(comp_algo))
        
        # Scroll area
        scroll = QScrollArea()
        scroll.setWidget(content_container)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: transparent; border: none;")
        
        layout.addWidget(scroll)