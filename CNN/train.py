from SudokuCNN import train_data, train_labels, valid_data, valid_labels, SudokuCNN 

res = SudokuCNN.fit(train_data, train_labels, batch_size=64, epochs = 10, validation_data=(valid_data, valid_labels), verbose = 2)
SudokuCNN.evaluate(valid_data, valid_labels, verbose = 3)
SudokuCNN.save("sudoku_cnn.keras")
