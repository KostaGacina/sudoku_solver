from board import is_valid, get_empty_cells
from solver_forward_check import get_peers, solve_forward_check
from solver_backtracking import solve_backtrack_domains
from ac3 import AC3, initialize_domains, get_constraints

def ac3(domains): ...
def revise(domains, xi, xj): ...
def select_cell_with_degree(domains, board): ...
def least_constraining_values(cell, domains, board): ...

def solve_ac3_backtrack(board):
    domains = initialize_domains(board)
    constraints = get_constraints()
    if not AC3(domains, constraints):
        return None
    if all(len(vals) == 1 for vals in domains.values()):
        solved_board = [[0 for _ in range(9)] for _ in range(9)]
        for (i, j), val in domains.items():
            solved_board[i][j] = next(iter(val))
        return solved_board

    return solve_backtrack_domains(board, domains)
def solve_ac3_forward(board):
    domains = initialize_domains(board)     
    constraints = get_constraints()         

    AC3(domains, constraints)               

    return solve_forward_check(board, domains)  
def solve_ac3_lcv(board, domains):
    ac3(domains)
    return recursive_ac3_lcv(board, domains)

def recursive_ac3_lcv(board, domains):
    ...
