import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'        #basically telling tf to only show errors, not warnings or info messages    1=info, 2=warnings, 3=errors only
import warnings
warnings.filterwarnings('ignore')       #baaki libs and python itself ki warnings ko ignore krne keliye

import tensorflow as tf         #building and training model
from tensorflow.keras import layers         #layers for building the model
import numpy as np
import cv2          #imgae processing
from sklearn.model_selection import train_test_split    #splitting dataset into train and test
import matplotlib.pyplot as plt         #graphs
import pickle               #saving class mappings
import time         #timing the training process

print("-"*50)
print("Sign Language Model Training".center(50))
print("-"*50)

IMG_SIZE = 128          #itna bara k smjh aajaye aur itna chhota k jaldi train hojaaye
BATCH_SIZE = 32         #how many images dekhne k baad model apne weights update karega, ziada hone se training stable hoti hai lekin ziada time lagta hai
EPOCHS = 30             #kitni baar poora dataset dekhega model, ziada hone se better accuracy mil sakti hai lekin overfitting ka risk bhi badhta hai
VALIDATION_SPLIT = 0.2  #20% for testing

all_classes = [chr(i) for i in range(65, 91)] + [str(i) for i in range(10)]
classes = []
for class_name in all_classes:          #srif un folders ko include kro jin mein images hain
    folder = f"dataset/{class_name}"
    if os.path.exists(folder) and len(os.listdir(folder)) > 0:
        classes.append(class_name)

print(f"\nFound {len(classes)} classes with images: {classes}")

if len(classes) == 0:
    print("No images found! Please collect data first.")
    exit()

num_classes = len(classes)
print(f"\nClasses: {num_classes} (A-Z, 0-9)")

class_to_idx = {cls: idx for idx, cls in enumerate(classes)}    #for training
idx_to_class = {idx: cls for idx, cls in enumerate(classes)}    #for testing    -   neural networks no k saath behter kaam krte hain

print("\nLoading dataset...")
images, labels = [], []     #images, indexes
total = 0

for class_name in classes:
    folder = f"dataset/{class_name}"
    if not os.path.exists(folder):
        continue
    
    files = os.listdir(folder)          #gets saari imgs of a class
    print(f"  {class_name}: {len(files)} images")
    
    for file in files:
        img = cv2.imread(os.path.join(folder, file))        #reading the img, if it fails to read for some reason, we skip that img
        if img is None:
            continue
    
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      #BGR to RGB - training keliye tf uses RGB 
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))     # to 128
        img = img.astype(np.float32) / 255.0            #converting to float and normalizing to 0-1 range, better for training
        
        images.append(img)  
        labels.append(class_to_idx[class_name])         #label of each img
        total += 1

print(f"\nTotal images: {total}")

if total == 0:
    print("No images found! Please collect data first.")
    exit()

X = np.array(images)        #tf works fast with np arrays than lists
y = np.array(labels)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VALIDATION_SPLIT, random_state=42, stratify=y)  #keeping same % of each class in train and test
print(f"Training: {len(X_train)} | Testing: {len(X_test)}")

#saari layers aik sequence mein hain
model = tf.keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),   #1st layers, 32 filters (learning 32 different patterns), each pattern is 3x3, only pos no, input shape is 128x128 with 3 color channels, output is 126x126x32 (filter ki wajah se - 32 channels for each pattern)
    layers.MaxPooling2D((2,2)),     #basically img ko mazeed chhota krta hai 2x2 k block mein se max val lekr, output is 63x63x32 (half of width and height, same number of channels)
    
    layers.Conv2D(64, (3,3), activation='relu'),        #dobara se filtering and is baar double time seekhega patterns, output is 61x61x64
    layers.MaxPooling2D((2,2)),     #output is 30x30x64
    
    layers.Conv2D(128, (3,3), activation='relu'),       #output is 28x28x128
    layers.MaxPooling2D((2,2)),     #output is 14x14x128
    
    layers.Conv2D(256, (3,3), activation='relu'),       #output is 12x12x256
    layers.MaxPooling2D((2,2)),     #output is 6x6x256
    
    layers.Flatten(),       #converting 3d output to 1d kionke dense ko 1d mein chahye , output is 9216 (6*6*256)
    layers.Dense(512, activation='relu'),       #512 neurons, output is 512 - each neuron has 512 weights
    layers.Dropout(0.5),            #50% of the neurons ko randomly band krdeta hai taake overfitting na ho
    layers.Dense(256, activation='relu'),       
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')     #output layer, each class ke liye ek neuron, softmax se output probabilities mein convert hoti hai
])

model.compile(      #model ko train krne ke settings
    optimizer=tf.keras.optimizers.Adam(0.001),      #weights ko kaise update krna hai - a=0.001 is the learning rate, ziada hone se jaldi seekhega lekin unstable ho sakta hai, kam hone se stable hoga lekin time lagega
    loss='sparse_categorical_crossentropy',         #model kitna ghalat hai ye measure krta hai - sparse because labels integers hain, 
    metrics=['accuracy']                            #model ne kitna sahi predict kia 
)


os.makedirs('models', exist_ok=True)        #creating folder if doesn't exist
callbacks = [       #callbacks are functions that are called during training at certain points, like end of an epoch, to do something - in this case, early stopping and saving the best model
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),        #5 epochs - after stopping best wights waapis le aao
    tf.keras.callbacks.ModelCheckpoint('models/best_model.h5', save_best_only=True) 
]


print("\nTraining...")
start = time.time()
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),     #images to train, correct ans, testing data
                   epochs=EPOCHS, batch_size=BATCH_SIZE, 
                   callbacks=callbacks, verbose=1)          #verbose=1 means showing progress bar, 0 mtlb no output, 2 mtlb one line per epoch
#history has loss (training loss per epoch), val_loss (testing loss per epoch), accuracy (training accuracy per epoch), val_accuracy (testing accuracy per epoch)

train_time = time.time() - start


test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

model.save('models/sign_language_final.h5') 
model.save('models/sign_language_final.keras')  
print("Model saved as: .h5 and .keras")

with open('models/class_mapping.pkl', 'wb') as f:       #in binary bcz kionke structure maintain hota hai
    pickle.dump({'classes': classes, 'idx_to_class': idx_to_class}, f)      
print("Class mapping saved")

plt.figure(figsize=(10,4))      #creates a new window 10in wide 4in tall
plt.subplot(1,2,1)              #1 row, 2 columns, this is the 1st plot
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.legend()
plt.tight_layout()          #automatically adjust spacing between subplots to prevent overlap
plt.savefig('models/training_history.png')
plt.show()      #pop up

print(f"\nComplete! Time: {train_time:.1f}s | Accuracy: {test_acc*100:.2f}%")
