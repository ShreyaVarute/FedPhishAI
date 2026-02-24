import json
import matplotlib.pyplot as plt
import os

SAVE_DIR = "models/distilbert_baseline"

# Load history
with open(os.path.join(SAVE_DIR, "history.json"), "r") as f:
    history = json.load(f)

epochs = range(1, len(history["loss"]) + 1)

# Loss graph
plt.figure()
plt.plot(epochs, history["loss"], marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.savefig(os.path.join(SAVE_DIR, "loss.png"))

# Accuracy graph
plt.figure()
plt.plot(epochs, history["accuracy"], marker='o')
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid()
plt.savefig(os.path.join(SAVE_DIR, "accuracy.png"))

# F1 graph
plt.figure()
plt.plot(epochs, history["f1"], marker='o')
plt.title("Validation F1 Score")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.grid()
plt.savefig(os.path.join(SAVE_DIR, "f1.png"))

print("Graphs saved successfully")