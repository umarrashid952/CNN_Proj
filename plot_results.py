import matplotlib.pyplot as plt

epochs = list(range(1, 11))
train_loss = [1.6650, 1.3020, 1.1095, 0.9938, 0.9071, 0.8475, 0.7953, 0.7495, 0.7170, 0.6799]
test_loss = [1.4998, 0.9912, 0.8378, 0.8932, 0.7105, 0.6619, 0.6395, 0.5786, 0.5552, 0.5341]
train_acc = [0.3783, 0.5372, 0.6162, 0.6594, 0.6925, 0.7172, 0.7351, 0.7514, 0.7681, 0.7753]
test_acc = [0.4638, 0.6450, 0.7014, 0.7019, 0.7527, 0.7724, 0.7853, 0.8044, 0.8170, 0.8244]

TRAIN_COLOR = "#2a78d6"
TEST_COLOR = "#eb6834"

fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, train_vals, test_vals, ylabel, title in [
    (ax_loss, train_loss, test_loss, "Loss", "Loss per epoch"),
    (ax_acc, train_acc, test_acc, "Accuracy", "Accuracy per epoch"),
]:
    ax.plot(epochs, train_vals, color=TRAIN_COLOR, linewidth=2, marker="o", markersize=5, label="Train")
    ax.plot(epochs, test_vals, color=TEST_COLOR, linewidth=2, marker="o", markersize=5, label="Test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(epochs)
    ax.grid(True, color="#e1e0d9", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

fig.suptitle("CIFAR-10 CNN training results (10 epochs)")
fig.tight_layout()
fig.savefig("training_results.png", dpi=150)
print("Saved training_results.png")
