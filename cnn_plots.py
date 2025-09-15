# Plot accuracy and loss in separate figures

import matplotlib.pyplot as plt

epochs = list(range(1, 16))
train_acc = [0.5248, 0.6511, 0.6931, 0.7244, 0.7479, 0.7668, 0.7831, 0.7977, 0.8099, 
             0.8193, 0.8278, 0.8335, 0.8385, 0.8425, 0.8471]
val_acc = [0.5894, 0.6621, 0.7051, 0.7321, 0.7519, 0.7691, 0.7831, 0.7944, 0.8020, 
           0.8157, 0.8207, 0.8213, 0.8285, 0.7659, 0.8329]

train_loss = [1.2276, 0.8471, 0.7143, 0.6312, 0.5707, 0.5238, 0.4840, 0.4495, 0.4210, 
              0.3997, 0.3798, 0.3669, 0.3561, 0.3480, 0.3374]
val_loss = [1.0482, 0.8108, 0.6773, 0.6070, 0.5561, 0.5123, 0.4796, 0.4536, 0.4379, 
            0.4036, 0.3925, 0.3959, 0.3751, 0.8066, 0.3642]

# Accuracy plot
plt.figure(figsize=(8,5))
plt.plot(epochs, train_acc, marker='o', label='Training Accuracy')
plt.plot(epochs, val_acc, marker='o', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy 256 filters')
plt.legend()
plt.grid(True)
plt.savefig("./static/cnn_accuracy_plot_256.png", dpi=300, bbox_inches="tight")
plt.show()

# Loss plot
plt.figure(figsize=(8,5))
plt.plot(epochs, train_loss, marker='o', label='Training Loss')
plt.plot(epochs, val_loss, marker='o', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss 256 filters')
plt.legend()
plt.grid(True)
plt.savefig("./static/cnn_loss_plot_256.png", dpi=300, bbox_inches="tight")
plt.show()

# Hardcoded training and validation metrics for 512
epochs = list(range(1, 11))
train_acc = [0.5967, 0.7432, 0.8009, 0.8313, 0.8464, 0.8572, 0.8677, 0.8781, 0.8891, 0.9005]
val_acc = [0.7005, 0.7745, 0.8134, 0.8317, 0.8391, 0.8448, 0.8469, 0.8495, 0.8500, 0.8500]

train_loss = [1.0151, 0.5963, 0.4503, 0.3763, 0.3424, 0.3201, 0.2987, 0.2781, 0.2567, 0.2340]
val_loss = [0.7102, 0.5132, 0.4161, 0.3710, 0.3555, 0.3428, 0.3395, 0.3392, 0.3430, 0.3554]

# Accuracy plot
plt.figure(figsize=(8,5))
plt.plot(epochs, train_acc, marker='o', label='Training Accuracy')
plt.plot(epochs, val_acc, marker='o', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy 512 filters')
plt.legend()
plt.grid(True)
plt.savefig("./static/cnn_accuracy_plot_512.png", dpi=300, bbox_inches="tight")
plt.show()

# Loss plot
plt.figure(figsize=(8,5))
plt.plot(epochs, train_loss, marker='o', label='Training Loss')
plt.plot(epochs, val_loss, marker='o', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss 512 filters')
plt.legend()
plt.grid(True)
plt.savefig("./static/cnn_loss_plot_512.png", dpi=300, bbox_inches="tight")
plt.show()
