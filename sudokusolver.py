import numpy as np
import time
from solver_cnn_forward_check import solve_cnn_forward_check
from solver_forward_check import solve_forward_check, initialize_domains
import pandas as pd
from cnn_backtrack.HybridSudokuSolver import HybridSudokuSolver as hb1
from cnn_backtrack.HybridSudokuSolverAC3 import HybridSudokuSolver as hb2
from keras import saving
from board import Step, print_board, load_boards_csv,is_solution_valid, load_board
from solver_backtracking import solve_backtrack, solve_backtrack_domains
from ac3 import initialize_domains as ac3_domains
import os
class GeneralSudokuSolver:
    def __init__(self, method, model, board, name):
        self.method = method
        self.model = model
        self.name = name
        self.model_name = "CNN 512 Filtera"
        self.board = np.array(board)
    def solve_table(self):
        if(self.name == "CNN + Forward Check"):
            brd = np.copy(self.board)
            res = self.method(self.model, brd)
            print(f"Method: {self.name}\nTime elapsed: {res[2]}\n")
            print_board(res[1])
        elif(self.name == "Forward Check only"):
            brd = np.copy(self.board)
            start = time.time()
            res = self.method(brd, initialize_domains(brd), [])
            end = time.time()
            print(f"Method: {self.name}\nTime elapsed: {round(end - start,2)}s ,\n")
            print_board(brd)
        elif(self.name == "Backtrack only"):
            brd = np.copy(self.board)
            start = time.time()
            self.method(brd, [])
            end = time.time()
            print(f"Method: {self.name}\nTime elapsed: {round(end - start,2)}s ,\n")
            print_board(brd)
        elif(self.name == "Backtrack + AC3"):
            brd = np.copy(self.board)
            start = time.time()
            self.method(brd, ac3_domains(brd), [])
            end = time.time()
            print(f"Method: {self.name}\nTime elapsed: {round(end - start,2)}s ,\n")
            print_board(brd)
        elif(self.name == "CNN + AC3 + Backtrack"):
            return self.method(self.board)
        elif(self.name == "CNN + Backtrack"):
            return self.method(self.board)
        

cwd = os.getcwd()

def LoadCNNModel(model_name):
    model= saving.load_model(os.path.join(cwd,"CNN","model",model_name))
    return model
def LoadSudokuTable():
    while True:
        print("Puzla treba da se nalazi u folderu sudoku_solver/puzzle_solve/")
        print("Primer formata puzle se nalazi u default puzzle_solve/puzla.csv")
        tabla = input("Unesite naziv fajla koji sadrzi sudoku puzlu u CSV formatu:\n")
        try:
            board = load_board(f"puzzle_solve/{tabla}")
            print(board)
            return board
        except Exception as e:
            print(f"Greska prilikom ucitavanja fajla: {e}")
def CNNModelMenu():
    while True:
        print("Opcije:")
        print("1. CNN 256 Filters")
        print("2. CNN 512 Filters")
        choice = input("Odaberite CNN model")
        if choice == '1':
            model = LoadCNNModel("sudoku_cnn_256_filters.keras")
            SolverCNNBacktrack = hb1(os.path.join(cwd,"CNN","model","sudoku_cnn_256_filters.keras"))
            SolverCNNAC3Backtrack = hb2(os.path.join(cwd,"CNN","model","sudoku_cnn_256_filters.keras"))
            return model, SolverCNNAC3Backtrack, SolverCNNBacktrack, "CNN 256 Filtera"
        if choice == '2':
            model = LoadCNNModel("sudoku_cnn_512_filters.keras")
            SolverCNNBacktrack = hb1(os.path.join(cwd,"CNN","model","sudoku_cnn_512_filters.keras"))
            SolverCNNAC3Backtrack = hb2(os.path.join(cwd,"CNN","model","sudoku_cnn_512_filters.keras"))
            return model, SolverCNNAC3Backtrack, SolverCNNBacktrack, "CNN 512 Filtera"
        else:
            print("Odaberite 1 ili 2 !")

def LoadSudokuMethod(SolverCNNBacktrack, SolverCNNAC3Backtrack):

    while True:
        print("Opcije:")
        print("1. Backtrack")
        print("2. AC3 + Backtrack")
        print("3. CNN + Backtrack")
        print("4, CNN + AC3 + Backtrack")
        print("5. Forward Check")
        print("6. CNN + Forward Check")
        choice = input("Odaberite CNN model")
        if choice == '1':
            return solve_backtrack, "Backtrack only"
        elif choice == '2':
            return solve_backtrack_domains, "Backtrack + AC3"
        elif choice == '3':
            return SolverCNNBacktrack.solve_sudoku_hybrid, "CNN + Backtrack"
        elif choice == '4':
            return SolverCNNAC3Backtrack.solve_sudoku_hybrid, "CNN + AC3 + Backtrack"
        elif choice == '5':
            return solve_forward_check, "Forward Check only"
        elif choice == '6':
            return solve_cnn_forward_check, "CNN + Forward Check"
        else:
            print("Odaberite 1-6!")
def Menu():

    solver = GeneralSudokuSolver(None,None,None,None)
    solver.model = LoadCNNModel("sudoku_cnn_512_filters.keras")
    SolverCNNBacktrack = hb1(os.path.join(cwd,"CNN","model","sudoku_cnn_512_filters.keras"))
    SolverCNNAC3Backtrack = hb2(os.path.join(cwd,"CNN","model","sudoku_cnn_512_filters.keras"))
    while True:
        print("Izaberitu jednu od ponuđenih opcija:")
        print("1. Izaberite CNN model za rešavanje sudoku-a")
        print(f"\tTrenutno izbrani model je {solver.model_name}")
        print("2. Unesite svoju tablu") 
        print("3. Odaberi metodu rešavanja")
        print("4. Prikaži trenutnu tablu (nerešenu) !")
        print("5. Izadji iz programa")
        choice = input("Izaberite opciju: ").strip()
        if choice == '1':
            solver.model, SolverCNNBacktrack, SolverCNNAC3Backtrack, solver.model_name = CNNModelMenu()
        elif choice == '2':
            solver.board = LoadSudokuTable()
        elif choice == '3':
            if solver.board != None:
                solver.method, solver.name = LoadSudokuMethod(SolverCNNBacktrack, SolverCNNAC3Backtrack)
                solver.solve_table()
                solver.method = None
                solver.name = None
            else:
                print("Morate prvo učitati sudoku!")
        elif choice == '4':
            if solver.board != None:
                print_board(solver.board)
            else:
                print("Morate prvo učitati sudoku!")
        elif choice == '5':
            return

        else:
            print(" Please enter 1-4")
Menu()
