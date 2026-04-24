import cv2
import numpy as np
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
import os
from collections import deque
import pyttsx3
import threading

print("\n" + "="*50)
print("REAL-TIME SIGN LANGUAGE RECOGNITION")
print("="*50)

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load model
print("\nLoading model...")
try:
    model = tf.keras.models.load_model('models/best_model.h5')
    print("Model loaded successfully!")
except:
    try:
        model = tf.keras.models.load_model('models/sign_language_final.h5')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

# Load class mapping
try:
    import pickle
    with open('models/class_mapping.pkl', 'rb') as f:
        mapping = pickle.load(f)
        classes = mapping['classes']
    print(f"Loaded {len(classes)} classes")
except:
    print("Error loading class mapping")
    exit()

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus for better consistency
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Set manual exposure (may need adjustment based on lighting)

# Initialize hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)



# Initialize text-to-speech
try:
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 150)
    tts_available = True
except:
    tts_available = False

# Recognition settings
IMG_SIZE = 128
prediction_history = deque(maxlen=3)
current_sentence = []
last_prediction = None
prediction_counter = 0
CONFIRMATION_THRESHOLD = 12
confidence_threshold = 0.8

just_cleared = False

def speak_text(text):
    if tts_available and text:
        def speak():
            tts_engine.say(text)
            tts_engine.runAndWait()
        threading.Thread(target=speak, daemon=True).start()

def preprocess_hand(img_hand):
    if img_hand.size == 0:
        return None
    
    h, w = img_hand.shape[:2]
    
    # Create white background (300x300 - matches training)
    img_white = np.ones((300, 300, 3), np.uint8) * 255
    
    # Preserve aspect ratio (no stretching)
    if h > w:
        new_h = 300
        new_w = int(300 * (w / h))
        img_resize = cv2.resize(img_hand, (new_w, new_h))
        x_offset = (300 - new_w) // 2
        img_white[:, x_offset:x_offset + new_w] = img_resize
    else:
        new_w = 300
        new_h = int(300 * (h / w))
        img_resize = cv2.resize(img_hand, (new_w, new_h))
        y_offset = (300 - new_h) // 2
        img_white[y_offset:y_offset + new_h, :] = img_resize
    
    # Resize to 128x128 for model
    img_final = cv2.resize(img_white, (IMG_SIZE, IMG_SIZE))
    
    # Normalize
    img_final = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)
    img_final = img_final.astype(np.float32) / 255.0
    
    return np.expand_dims(img_final, axis=0)


print("\nCONTROLS:")
print("  c - Clear sentence")
print("  Space - Add space")
print("  Backspace - Delete last character")
print("  s - Speak")
print("  ESC - Quit")
print("="*50 + "\n")

while True:
    success, img = cap.read()
    if not success:
        break
    
    # Create display version (mirrored for natural feel)
    display_img = cv2.flip(img, 1)
    
    # BUT process the ORIGINAL (non-mirrored) image for detection
    # This matches how data was saved during collection
    hands, _ = detector.findHands(img, draw=True)  # Use original image
    
    current_prediction = None
    confidence = 0.0
    img_cropped = None
    
    if hands:

        if just_cleared:
            prediction_history.clear()
            last_prediction = None
            prediction_counter = 0
            just_cleared = False
            
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        # Add padding
        padding = 60
        x_start = max(0, x - padding)
        x_end = min(img.shape[1], x + w + padding)
        y_start = max(0, y - padding)
        y_end = min(img.shape[0], y + h + padding)
        
        # Crop from ORIGINAL image
        img_cropped = img[y_start:y_end, x_start:x_end]
        
        
        if img_cropped.size > 0:
            img_processed = preprocess_hand(img_cropped)
            
            if img_processed is not None:
                predictions = model.predict(img_processed, verbose=0)[0]
                pred_idx = np.argmax(predictions)
                confidence = predictions[pred_idx]
                
                if confidence > confidence_threshold:
                    current_prediction = classes[pred_idx]
                    prediction_history.append(current_prediction)
                    
                    # Get most common prediction in history
                    if len(prediction_history) == prediction_history.maxlen:
                        from collections import Counter
                        counter = Counter(prediction_history)
                        smoothed = counter.most_common(1)[0][0]
                        
                        # Draw on display image (convert coordinates)
                        display_x = display_img.shape[1] - x - w
                        cv2.rectangle(display_img, (display_x, y), (display_x + w, y + h), (0, 255, 0), 3)
                        cv2.putText(display_img, f"{smoothed}", (display_x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # Add to sentence if consistent
                        if smoothed == last_prediction:
                            prediction_counter += 1
                        else:
                            last_prediction = smoothed
                            prediction_counter = 1
                            prediction_history.clear()  # Reset history on new prediction
                            prediction_history.append(smoothed)  # Start new history with current prediction
                        
                        if prediction_counter == CONFIRMATION_THRESHOLD:
                            current_sentence.append(smoothed)
                            print(f"Added: {smoothed}")
                            if tts_available:
                                speak_text(smoothed)

                            prediction_counter = 0
                            last_prediction = None  # Critical: Reset so next letter needs fresh detection
                            
    else:
        # NO HAND DETECTED - Clear history to prevent old predictions
        prediction_history.clear()
        last_prediction = None
        prediction_counter = 0
        just_cleared=True


    # Show controls in top-right
    controls_y = 50
    controls_list = [
        "CONTROLS:",
        "c - Clear",
        "Space - Add space",
        "Backspace - Delete",
        "s - Speak",
        "ESC - Quit"
    ]
    
    # Draw semi-transparent background
    overlay = display_img.copy()
    cv2.rectangle(overlay, (display_img.shape[1]-250, 30), (display_img.shape[1]-30, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, display_img, 0.3, 0, display_img)
    
    # Draw control text
    for i, text in enumerate(controls_list):
        color = (255, 255, 255) if i == 0 else (200, 200, 200)
        cv2.putText(display_img, text, (display_img.shape[1]-240, controls_y + i*25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Show current sentence at bottom
    sentence_text = ''.join(current_sentence) if current_sentence else "No gestures yet"
    y_pos = display_img.shape[0] - 50
    cv2.rectangle(display_img, (10, y_pos-30), (display_img.shape[1]-10, y_pos+10), (0, 0, 0), -1)
    cv2.putText(display_img, f"> {sentence_text}", (20, y_pos), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow("Sign Language Recognition", display_img)
    
    # Handle keys
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('c'):
        current_sentence = []
        print("Sentence cleared")
    
    elif key == 32:  # Space
        current_sentence.append(' ')
        print("Added space")
    
    elif key == 8:  # Backspace key (ASCII 8)
        if current_sentence:
            removed = current_sentence.pop()
            print(f"Removed: {removed if removed != ' ' else 'space'}")
        else:
            print("Nothing to remove")
    
    elif key == ord('s'):  # Speak
        if current_sentence and tts_available:
            text = ''.join(current_sentence)
            speak_text(text)
            print(f"Speaking: {text}")
    
    elif key == 27:  # ESC
        print("\n>>> CLOSING...")
        speech_thread_running = False  # Stop speech thread
        # Show closing message
        cv2.putText(display_img, "CLOSING...", (display_img.shape[1]//2-100, display_img.shape[0]//2), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.imshow("Sign Language Recognition", display_img)
        cv2.waitKey(500)
        cv2.destroyAllWindows()
        cap.release()
        exit()

cap.release()
cv2.destroyAllWindows()
print("\nRecognition ended!")