import sys
from PyQt5.QtWidgets import QMainWindow, QStackedWidget
from .pages.main_menu import MainMenuPage
from .pages.solver import SolverPage
from .pages.analytics_page1 import AnalyticsPage1
from .pages.analytics_page2 import AnalyticsPage2
from .pages.documentation import DocumentationPage
from .pages.credits import CreditsPage

class SudokuSolverApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sudoku Solver")
        self.setGeometry(100, 100, 800, 700)
        self.setStyleSheet("background-color: #2b2b2b;")
        
        # Create stacked widget for different pages
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # Create pages
        self.main_menu_page = MainMenuPage(self)
        self.solver_page = None
        self.analytics_page1 = None
        self.analytics_page2 = None
        self.documentation_page = None
        self.credits_page = None
        
        self.stacked_widget.addWidget(self.main_menu_page)
        
        # Show main menu initially
        self.show_main_menu()
    
    def show_main_menu(self):
        self.stacked_widget.setCurrentWidget(self.main_menu_page)
    
    def show_solver_page(self, algorithm, difficulty):
        if self.solver_page:
            self.stacked_widget.removeWidget(self.solver_page)
        
        self.solver_page = SolverPage(self, algorithm, difficulty)
        self.stacked_widget.addWidget(self.solver_page)
        self.stacked_widget.setCurrentWidget(self.solver_page)
    
    def show_analytics_page1(self, algorithm, difficulty):
        if self.analytics_page1:
            self.stacked_widget.removeWidget(self.analytics_page1)
        
        self.analytics_page1 = AnalyticsPage1(self, algorithm, difficulty)
        self.stacked_widget.addWidget(self.analytics_page1)
        self.stacked_widget.setCurrentWidget(self.analytics_page1)
    
    def show_analytics_page2(self, algorithm, difficulty):
        if self.analytics_page2:
            self.stacked_widget.removeWidget(self.analytics_page2)
        
        self.analytics_page2 = AnalyticsPage2(self, algorithm, difficulty)
        self.stacked_widget.addWidget(self.analytics_page2)
        self.stacked_widget.setCurrentWidget(self.analytics_page2)
    
    def show_documentation_page(self, algorithm, difficulty):
        if self.documentation_page:
            self.stacked_widget.removeWidget(self.documentation_page)
        
        self.documentation_page = DocumentationPage(self, algorithm, difficulty)
        self.stacked_widget.addWidget(self.documentation_page)
        self.stacked_widget.setCurrentWidget(self.documentation_page)
    
    def show_credits_page(self):
        if not self.credits_page:
            self.credits_page = CreditsPage(self)
            self.stacked_widget.addWidget(self.credits_page)
        
        self.stacked_widget.setCurrentWidget(self.credits_page)

def main():
    app = QApplication(sys.argv)
    window = SudokuSolverApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()