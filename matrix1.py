import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import pickle

# 1. Load the trained model
model_path = r"C:\Users\vkous\Pictures\mlkoushik\top.h5"
model = tf.keras.models.load_model(model_path)
print(f"✅ Model loaded from {model_path}")

# 2. Load CIFAR-10 Test Data
cifar10_path = r"C:\Users\vkous\Documents\ML_PROJECT2803\ML_PROJECT\cifar-10-python\cifar-10-batches-py"

def load_cifar10_batch(filename):
    """Loads a single CIFAR-10 batch from a local directory."""
    with open(filename, 'rb') as file:
        batch = pickle.load(file, encoding='bytes')
    data = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
    labels = np.array(batch[b'labels'])
    return data, labels

# Load test batch
x_test, y_test = load_cifar10_batch(os.path.join(cifar10_path, "test_batch"))

# Convert y_test to categorical format (One-hot encoded to class labels)
y_test_labels = np.array(y_test)

# 3. Predict on Test Data
y_pred_probs = model.predict(x_test)  # Get softmax probabilities
y_pred_labels = np.argmax(y_pred_probs, axis=1)  # Convert to class labels

# 4. Compute Confusion Matrix
cm = confusion_matrix(y_test_labels, y_pred_labels)
print("✅ Confusion matrix computed successfully!")

# 5. Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

# 6. Save & Display the Confusion Matrix
plt.savefig("confusion_matrix.png")  # Save the plot
print("📊 Confusion Matrix saved as 'confusion_matrix.png'")

plt.pause(0.1)
