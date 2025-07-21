domains = {
    'A': {1, 2},
    'B': {2},
}

constraints = [('A', 'B')]

def revise(domains, Xi, Xj):
    revised = False
    for i in list(domains[Xi]):
        found = False
        for j in domains[Xj]:
            if i != j:
                found = True
                break
        if not found:
            domains[Xi].remove(i)
            revised = True
    return revised
def AC3(domains, constraints): 
    agenda = []
    for (Xi, Xj) in constraints:
        agenda.append((Xi, Xj))
        agenda.append((Xj,Xi))
    arcs = agenda.copy()
    while(len(agenda) > 0):
        (Xi, Xj) = agenda.pop()
        xiCopy = domains[Xi].copy()
        if revise(domains, Xi, Xj):
            if(len(domains[Xi]) == 0):
                return False
        if(domains[Xi] != xiCopy):
            for cur_arc in arcs:
                if cur_arc[1] == Xi:
                    agenda.append(cur_arc)
    return True
def initialize_domains(board):
    domains = {}
    for i in range(9):
        for j in range(9):
            pos = (i,j)
            if board[i][j] == 0:
                domains[pos] = set(range(1,10))
            else:
                domains[pos] = {board[i][j]}
    return domains
def get_constraints():
    constraints = set()
    for i in range(9):
        for j in range(9):
            for k in range(j+1,9):
                constraints.add(((i,j),(i,k)))
        for j in range(9):
            for k in range(j+1,9):
                constraints.add(((j,i),(k,i)))
        box_row = (i // 3) * 3
        box_col = (i % 3) * 3
        cells = [(box_row+r, box_col+c) for r in range(3) for c in range(3)]
        for j in range(len(cells)):
            for k in range(j+1,len(cells)):
                constraints.add((cells[j],cells[k]))
    return list(constraints)

