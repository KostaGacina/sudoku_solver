from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

class CreditsPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(40)
        
        # Title
        title = QLabel("Thanks for reviewing our project!")
        title.setFont(QFont("Arial", 28, QFont.Bold))
        title.setStyleSheet("color: #a855f7; margin-bottom: 40px;")
        title.setAlignment(Qt.AlignCenter)
        
        # Credits section
        credits_label = QLabel("Created by:")
        credits_label.setFont(QFont("Arial", 18, QFont.Bold))
        credits_label.setStyleSheet("color: #a855f7; margin-bottom: 20px;")
        credits_label.setAlignment(Qt.AlignLeft)
        
        names = [
            "Kosta Gaćina",
            "Aleksa Mrda", 
            "Stefan Crepulja",
            "Bane Božanić"
        ]
        
        names_layout = QVBoxLayout()
        for name in names:
            name_label = QLabel(name)
            name_label.setFont(QFont("Arial", 16))
            name_label.setStyleSheet("color: white; margin: 5px 0;")
            name_label.setAlignment(Qt.AlignLeft)
            names_layout.addWidget(name_label)
        
        # Exit button
        exit_btn = QPushButton("Exit")
        exit_btn.setFont(QFont("Arial", 14, QFont.Bold))
        exit_btn.setFixedSize(120, 50)
        exit_btn.setStyleSheet("""
            QPushButton {
                background-color: #a855f7;
                color: white;
                border: none;
                border-radius: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #9333ea;
            }
            QPushButton:pressed {
                background-color: #7c2d92;
            }
        """)
        exit_btn.clicked.connect(self.main_window.close)
        
        # Back arrow (top-left)
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
        back_btn.clicked.connect(self.main_window.show_main_menu)
        
        # Layout assembly
        top_layout = QHBoxLayout()
        top_layout.addWidget(back_btn)
        top_layout.addStretch()
        
        credits_container = QWidget()
        credits_container_layout = QVBoxLayout(credits_container)
        credits_container_layout.addWidget(credits_label)
        credits_container_layout.addLayout(names_layout)
        
        layout.addLayout(top_layout)
        layout.addWidget(title)
        layout.addWidget(credits_container, alignment=Qt.AlignCenter)
        layout.addWidget(exit_btn, alignment=Qt.AlignCenter)