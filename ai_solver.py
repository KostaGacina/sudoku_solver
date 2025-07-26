from ac3 import AC3, get_constraints, initialize_domains
from board import get_empty_cells, is_valid, Step
from solver_backtracking import solve_backtrack_domains
from solver_forward_check import get_peers, solve_forward_check, initialize_domains as fc_initialize_domains


def ac3(domains): ...
def revise(domains, xi, xj): ...
def select_cell_with_degree(domains, board): ...
def least_constraining_values(cell, domains, board): ...


def solve_ac3_backtrack(board, steps):
    domains = initialize_domains(board)
    constraints = get_constraints()
    
    # Apply AC3 preprocessing (this part is not recorded in steps)
    if not AC3(domains, constraints):
        return False
    
    # Apply AC3 results to board (record these as steps)
    for (i, j), domain in domains.items():
        if len(domain) == 1 and board[i, j] == 0:
            value = next(iter(domain))
            board[i, j] = value
            steps.append(Step(i, j, value))
    
    # Check if AC3 solved it completely
    if all(len(vals) == 1 for vals in domains.values()):
        return True
    
    # Continue with backtracking on remaining unsolved cells
    return solve_backtrack_domains(board, domains, steps)


def solve_ac3_forward(board, steps):
    # First apply AC3 preprocessing (not recorded in steps)
    ac3_domains = initialize_domains(board)
    constraints = get_constraints()
    
    if not AC3(ac3_domains, constraints):
        return False
    
    
    # Apply AC3 results to the board (record these as steps)
    for (i, j), domain in ac3_domains.items():
        if len(domain) == 1 and board[i, j] == 0:
            value = next(iter(domain))
            board[i, j] = value
            steps.append(Step(i, j, value))
    
    # Check if AC3 solved it completely
    if all(len(vals) <= 1 for vals in ac3_domains.values()):
        return True
    
    
    # Use forward checking's initialize_domains on the updated board
    fc_domains = fc_initialize_domains(board)
    
    
    # Debug: Print some domain examples
    empty_cells = [(i, j) for i in range(9) for j in range(9) if board[i, j] == 0]
    for i, (row, col) in enumerate(empty_cells[:3]):  # Show first 3 empty cells
        ac3_domain = ac3_domains.get((row, col), set())
        fc_domain = fc_domains.get((row, col), set())
        
    # Continue with forward checking (steps will be recorded)
    result = solve_forward_check(board, fc_domains, steps)
    return result


def solve_ac3_lcv(board, domains):
    ac3(domains)
    return recursive_ac3_lcv(board, domains)


def recursive_ac3_lcv(board, domains): ...
