import sys
from gui.app import SudokuSolverApp
from PyQt5.QtWidgets import QApplication

def main():
    app = QApplication(sys.argv)
    window = SudokuSolverApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()