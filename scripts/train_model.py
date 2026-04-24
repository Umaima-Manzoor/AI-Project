# Suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import time

print("\n" + "="*50)
print("SIGN LANGUAGE MODEL TRAINING")
print("="*50)

# Configuration
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 30
VALIDATION_SPLIT = 0.2

# Classes
# Classes - only use folders that have images
all_classes = [chr(i) for i in range(65, 91)] + [str(i) for i in range(10)]
classes = []
for class_name in all_classes:
    folder = f"dataset/{class_name}"
    if os.path.exists(folder) and len(os.listdir(folder)) > 0:
        classes.append(class_name)

print(f"\nFound {len(classes)} classes with images: {classes}")

if len(classes) == 0:
    print("No images found! Please collect data first.")
    exit()
num_classes = len(classes)
print(f"\nClasses: {num_classes} (A-Z, 0-9)")

# Create mappings
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
idx_to_class = {idx: cls for idx, cls in enumerate(classes)}

# Load data
print("\nLoading dataset...")
images, labels = [], []
total = 0

for class_name in classes:
    folder = f"dataset/{class_name}"
    if not os.path.exists(folder):
        continue
    
    files = os.listdir(folder)
    print(f"  {class_name}: {len(files)} images")
    
    for file in files:
        img = cv2.imread(os.path.join(folder, file))
        if img is None:
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        
        images.append(img)
        labels.append(class_to_idx[class_name])
        total += 1

print(f"\nTotal images: {total}")

if total == 0:
    print("No images found! Please collect data first.")
    exit()

X = np.array(images)
y = np.array(labels)


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=VALIDATION_SPLIT, random_state=42, stratify=y
)
print(f"Training: {len(X_train)} | Testing: {len(X_test)}")

# Build model
model = tf.keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(256, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
os.makedirs('models', exist_ok=True)
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('models/best_model.h5', save_best_only=True)
]

# Train
print("\nTraining...")
start = time.time()
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                   epochs=EPOCHS, batch_size=BATCH_SIZE, 
                   callbacks=callbacks, verbose=1)
train_time = time.time() - start

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

# Save model in BOTH formats
model.save('models/sign_language_final.h5')  # .h5 format
model.save('models/sign_language_final.keras')  # .keras format
print("Model saved as: .h5 and .keras")

# Save class mapping
with open('models/class_mapping.pkl', 'wb') as f:
    pickle.dump({'classes': classes, 'idx_to_class': idx_to_class}, f)
print("Class mapping saved")

# Plot
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('models/training_history.png')
plt.show()

print(f"\nComplete! Time: {train_time:.1f}s | Accuracy: {test_acc*100:.2f}%")