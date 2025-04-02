import tensorflow as tf
import numpy as np
import os
import pickle
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

# ✅ 1. Paths
attacked_images = r"C:\Users\vkous\Downloads\ML_REVIEW\attacked_images"
output_path = r"C:\Users\vkous\Downloads\ML_REVIEW\cifar10_images"
model_path = r"C:\Users\vkous\Downloads\ML_REVIEW\top.h5"

# ✅ 2. CIFAR-10 Labels
cifar10_labels = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ✅ 3. Create Labeled Directories
for label in cifar10_labels:
    os.makedirs(os.path.join(output_path, label), exist_ok=True)

# ✅ 4. Extract and Save Images
def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

for batch in range(1, 6):  # Process CIFAR-10 batches
    batch_path = os.path.join(attacked_images, f"data_batch_{batch}")
    data_dict = unpickle(batch_path)

    images = data_dict[b'data']
    labels = data_dict[b'labels']

    for i in range(len(images)):
        img_array = images[i].reshape(3, 32, 32).transpose(1, 2, 0)  # Convert to (32,32,3)
        img = Image.fromarray(img_array)
        label_name = cifar10_labels[labels[i]]

        img.save(os.path.join(output_path, label_name, f"{batch}_{i}.png"))

print(" CIFAR-10 Images extracted successfully!")

#  5. Load Model
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

model = tf.keras.models.load_model(model_path)
print(" Model loaded successfully!")

#  6. Prepare Image Data Generator
image_size = (32, 32)  # FIXED: Match CIFAR-10's original size
batch_size = 32

datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = datagen.flow_from_directory(
    output_path,  # Path to extracted images
    target_size=image_size,  #  FIXED: Ensure correct image size
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

#  7. Evaluate the Model
loss, accuracy = model.evaluate(test_generator)
print(f" Test Accuracy: {accuracy * 100:.2f}%")
print(f" Test Loss: {loss:.4f}")

#  8. Generate Classification Report
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

class_labels = list(test_generator.class_indices.keys())

report = classification_report(y_true, y_pred, target_names=class_labels)
print("\n Classification Report:\n", report)

#  9. Save Report to File
report_path = r"C:\Users\vkous\Downloads\ML_REVIEW\classification_report.txt"
with open(report_path, "w") as f:
    f.write(report)

print(f" Classification report saved at: {report_path}")
