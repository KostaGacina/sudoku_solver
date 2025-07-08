# board.py Functions

- load_board(filename): Učitava pazl iz tekst fajla (vraća 9x9 ndarray)
- print_board(board): Elegantno printuje sudoku tablu
- is_valid(board, row, col, num): Proverava da li 'num' može biti stavljen na mesto (row,col)
- save_board(board, filename): Čuva tablu u fajl
- get_empty_cells(board): Vraća listu praznih polja
