import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pickle
import os
import cv2

# 1. LOAD CIFAR-10 DATA FROM LOCAL DIRECTORY
cifar10_path = r"C:\Users\vkous\Documents\ML_PROJECT2803\ML_PROJECT\cifar-10-python\cifar-10-batches-py"

def load_cifar10_batch(filename):
    with open(filename, 'rb') as file:
        batch = pickle.load(file, encoding='bytes')
    data = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
    labels = np.array(batch[b'labels'])
    return data, labels

# Load all training batches
x_train, y_train = [], []
for i in range(1, 6):  # 5 training batches
    data, labels = load_cifar10_batch(os.path.join(cifar10_path, f"data_batch_{i}"))
    x_train.append(data)
    y_train.append(labels)

x_train = np.concatenate(x_train)
y_train = np.concatenate(y_train)
y_train = to_categorical(y_train, 10)

# Feature Squeezing
def feature_squeeze(image, bit_depth=8):
    max_value = float(2**bit_depth - 1)
    image = np.round(image * max_value) / max_value
    return image

x_train = np.array([feature_squeeze(img) for img in x_train])

# Randomization as Defense
def randomization(image):
    h, w, _ = image.shape
    image = cv2.resize(image, (h + np.random.randint(-2, 3), w + np.random.randint(-2, 3)))
    image = cv2.resize(image, (h, w))
    return image

x_train = np.array([randomization(img) for img in x_train])

# 2. BUILD THE RESNET50 MODEL WITH PRE-TRAINED IMAGENET WEIGHTS
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3), pooling='avg')
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 3. DATA AUGMENTATION FOR BETTER GENERALIZATION
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

# 4. ADD EARLY STOPPING TO PREVENT OVERFITTING
early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

# 5. TRAIN THE MODEL WITH AUGMENTED DATA
history = model.fit(datagen.flow(x_train, y_train, batch_size=64), 
                    epochs=20, 
                    callbacks=[early_stopping])

# 6. SAVE THE TRAINED MODEL AS .h5 FILE IN THE SPECIFIED DIRECTORY
save_path = r"C:\Users\vkous\Pictures\mlkoushik\top.h5"
model.save(save_path)
print(f"Model saved at: {save_path}")