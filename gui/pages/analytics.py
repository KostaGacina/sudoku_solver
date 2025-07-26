from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

class AnalyticsPage(QWidget):
    def __init__(self, main_window, algorithm, difficulty):
        super().__init__()
        self.main_window = main_window
        self.algorithm = algorithm
        self.difficulty = difficulty
        self.current_slide = 0
        self.slides = self.create_slides()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        # Navigation arrows
        nav_layout = QHBoxLayout()
        
        self.prev_btn = QPushButton("◀")
        self.prev_btn.setFont(QFont("Arial", 16, QFont.Bold))
        self.prev_btn.setFixedSize(50, 50)
        self.prev_btn.setStyleSheet("""
            QPushButton {
                background-color: #a855f7;
                color: white;
                border: none;
                border-radius: 25px;
            }
            QPushButton:hover {
                background-color: #9333ea;
            }
            QPushButton:disabled {
                background-color: #6b7280;
            }
        """)
        self.prev_btn.clicked.connect(self.prev_slide)
        
        self.next_btn = QPushButton("▶")
        self.next_btn.setFont(QFont("Arial", 16, QFont.Bold))
        self.next_btn.setFixedSize(50, 50)
        self.next_btn.setStyleSheet("""
            QPushButton {
                background-color: #a855f7;
                color: white;
                border: none;
                border-radius: 25px;
            }
            QPushButton:hover {
                background-color: #9333ea;
            }
            QPushButton:disabled {
                background-color: #6b7280;
            }
        """)
        self.next_btn.clicked.connect(self.next_slide)
        
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addStretch()
        nav_layout.addWidget(self.next_btn)
        
        layout.addLayout(nav_layout)
        
        # Slide container
        self.slide_container = QWidget()
        self.slide_layout = QVBoxLayout(self.slide_container)
        self.slide_layout.setAlignment(Qt.AlignCenter)
        
        scroll = QScrollArea()
        scroll.setWidget(self.slide_container)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: transparent; border: none;")
        
        layout.addWidget(scroll)
        
        # Back button
        back_btn = QPushButton("Back to Solver")
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
        back_btn.clicked.connect(lambda: self.main_window.show_solver_page(self.algorithm, self.difficulty))
        
        layout.addWidget(back_btn, alignment=Qt.AlignCenter)
        
        self.update_slide()
    
    def create_slides(self):
        slides = []
        
        # Slide 1: Documentation
        slides.append({
            'title': 'Project Documentation',
            'content': 'Scrollable container for description and documentation'
        })
        
        # Slide 2-4: Algorithm comparisons
        other_algos = ['Backtracking', 'Forward Checking', 'AC3']
        other_algos.remove(self.algorithm)
        
        for i, other_algo in enumerate(other_algos, 1):
            slides.append({
                'title': f'How algorithm [{self.algorithm}] behaves compared to [{other_algo}]',
                'content': 'Line or bar chart goes here'
            })
        
        # Slide 5: Difficulty comparison
        slides.append({
            'title': f'How algorithm [{self.algorithm}] behaves depending on difficulty level',
            'content': 'Line or bar chart goes here',
            'subcontent': [
                f'How algorithm [{self.algorithm}] behaves on the difficulty level you have chosen vs other supported algorithms',
                'Line or bar chart goes here',
                'Line or bar chart goes here',
                'Line or bar chart goes here'
            ]
        })
        
        # Slide 6: Performance analysis
        slides.append({
            'title': f'How algorithm [{self.algorithm}] behaves through 100 examples on same difficulty level',
            'content': 'Line or bar chart goes here'
        })
        
        return slides
    
    def update_slide(self):
        # Clear current slide
        for i in reversed(range(self.slide_layout.count())):
            self.slide_layout.itemAt(i).widget().setParent(None)
        
        slide = self.slides[self.current_slide]
        
        # Title
        title = QLabel(slide['title'])
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #a855f7; margin: 20px;")
        title.setAlignment(Qt.AlignCenter)
        title.setWordWrap(True)
        
        # Content
        content = QLabel(slide['content'])
        content.setFont(QFont("Arial", 14))
        content.setStyleSheet("color: white; margin: 20px; padding: 40px; border: 1px solid #a855f7;")
        content.setAlignment(Qt.AlignCenter)
        content.setMinimumHeight(200)
        
        self.slide_layout.addWidget(title)
        self.slide_layout.addWidget(content)
        
        # Add subcontent if exists
        if 'subcontent' in slide:
            for subcontent_text in slide['subcontent']:
                if 'Line or bar chart' in subcontent_text:
                    subcontent = QLabel(subcontent_text)
                    subcontent.setFont(QFont("Arial", 12))
                    subcontent.setStyleSheet("color: white; margin: 10px; padding: 20px; border: 1px solid #6b7280;")
                    subcontent.setAlignment(Qt.AlignCenter)
                    subcontent.setMinimumHeight(100)
                else:
                    subcontent = QLabel(subcontent_text)
                    subcontent.setFont(QFont("Arial", 14, QFont.Bold))
                    subcontent.setStyleSheet("color: #a855f7; margin: 20px;")
                    subcontent.setAlignment(Qt.AlignCenter)
                    subcontent.setWordWrap(True)
                
                self.slide_layout.addWidget(subcontent)
        
        # Update navigation buttons
        self.prev_btn.setEnabled(self.current_slide > 0)
        self.next_btn.setEnabled(self.current_slide < len(self.slides) - 1)
    
    def prev_slide(self):
        if self.current_slide > 0:
            self.current_slide -= 1
            self.update_slide()
    
    def next_slide(self):
        if self.current_slide < len(self.slides) - 1:
            self.current_slide += 1
            self.update_slide()