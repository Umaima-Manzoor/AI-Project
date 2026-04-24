import cv2
import os
import time
import math
import numpy as np
from cvzone.HandTrackingModule import HandDetector

print("\n" + "="*60)
print("IMPROVED DATA COLLECTION".center(60))
print("="*60)

# Create folders
if not os.path.exists("dataset"):
    os.makedirs("dataset")

classes = [chr(i) for i in range(65, 91)] + [str(i) for i in range(10)]

for class_name in classes:
    folder_path = f"dataset/{class_name}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {class_name}")

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.05)  # Manual exposure
# cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
# cap.set(cv2.CAP_PROP_CONTRAST, 0.5)


# Initialize hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Image settings
imgSize = 300  # Final image size (square)
offset = 60    # Padding around hand

# Auto-save settings
auto_save = False
save_delay = 0.3
last_save_time = 0
target_per_class = 200

# Collection stats
collection_stats = {}
for class_name in classes:
    folder_path = f"dataset/{class_name}"
    if os.path.exists(folder_path):
        collection_stats[class_name] = len(os.listdir(folder_path))
    else:
        collection_stats[class_name] = 0

print("\nINSTRUCTIONS:")
print("- Screen shows MIRRORED view (like a mirror)")
print("- Images saved at 300x300 with white background (no stretching)")
print("- Press LETTER (A-Z) or NUMBER (0-9) to select class")
print("- Press SPACE to save one image")
print("- Press ENTER to toggle auto-save ON/OFF")
print("- Press ESC to quit\n")

current_class = None
total_saved = 0

while True:
    success, img = cap.read()
    if not success:
        break
    
    # Get original image dimensions
    h_img, w_img, _ = img.shape
    
    # STEP 1: Create a copy for display (MIRRORED - feels natural)
    img_display = cv2.flip(img, 1)
    
    # STEP 2: Use the ORIGINAL image for hand detection
    hands, _ = detector.findHands(img, draw=True)
    
    img_final = None
    hand_detected = False
    
    if hands:
        hand_detected = True
        
        # Process each hand detected (up to 2)
        for hand in hands:
            x, y, w, h = hand['bbox']
            hand_type = hand['type']  # 'Left' or 'Right'
            
            # Convert coordinates to display space
            display_x = w_img - x - w
            
            # Draw on display image (different colors for left/right)
            color = (0, 255, 0) if hand_type == 'Right' else (255, 0, 0)
            cv2.rectangle(img_display, (display_x, y), (display_x + w, y + h), color, 3)
            cv2.putText(img_display, f"{hand_type} HAND", (display_x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # For saving, only use the first hand (or you can modify to save both)
            # This will use the right hand if present, otherwise left
            if 'img_crop' not in locals() or hand_type == 'Right':
                # Crop with padding from ORIGINAL image
                x_start = max(0, x - offset)
                x_end = min(w_img, x + w + offset)
                y_start = max(0, y - offset)
                y_end = min(h_img, y + h + offset)
                
                img_crop = img[y_start:y_end, x_start:x_end]
                
                if img_crop.size > 0:
                    # Create white background image
                    img_white = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                    
                    # Get crop dimensions
                    h_crop, w_crop, _ = img_crop.shape
                    aspect_ratio = h_crop / w_crop
                    
                    # Place crop on white background preserving aspect ratio
                    if aspect_ratio > 1:  # Height > Width
                        k = imgSize / h_crop
                        w_cal = math.ceil(k * w_crop)
                        img_resize = cv2.resize(img_crop, (w_cal, imgSize))
                        w_gap = math.ceil((imgSize - w_cal) / 2)
                        img_white[:, w_gap:w_gap + w_cal] = img_resize
                    else:  # Width >= Height
                        k = imgSize / w_crop
                        h_cal = math.ceil(k * h_crop)
                        img_resize = cv2.resize(img_crop, (imgSize, h_cal))
                        h_gap = math.ceil((imgSize - h_cal) / 2)
                        img_white[h_gap:h_gap + h_cal, :] = img_resize
                    
                    img_final = img_white
                    
                    # Show the final image that will be saved
                    cv2.imshow("Will Save (300x300 - No Stretch)", img_final)
    

    
    # Display info on main screen
    if current_class:
        current_count = collection_stats.get(current_class, 0)
        cv2.putText(img_display, f"CLASS: {current_class}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img_display, f"IMAGES: {current_count}/{target_per_class}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    # Hand status
    if hand_detected:
        cv2.putText(img_display, "HAND DETECTED", (img_display.shape[1] - 200, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(img_display, "NO HAND", (img_display.shape[1] - 200, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Auto-save status
    if auto_save:
        cv2.putText(img_display, "AUTO-SAVE ON", (img_display.shape[1] - 200, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Controls in top-right corner (like recognition script)
    controls_y = 100
    controls_list = [
        "CONTROLS:",
        "A-Z / 0-9 - Select class",
        "SPACE - Save one",
        "ENTER - Toggle autosave",
        "ESC - Quit"
    ]
    
    # Draw semi-transparent background
    overlay = img_display.copy()
    cv2.rectangle(overlay, (img_display.shape[1]-350, 80), (img_display.shape[1]-30, 230), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img_display, 0.3, 0, img_display)
    
    # Draw control text
    for i, text in enumerate(controls_list):
        color = (255, 255, 255) if i == 0 else (200, 200, 200)
        cv2.putText(img_display, text, (img_display.shape[1]-340, controls_y + i*25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    cv2.imshow("Data Collection", img_display)
    
    # Handle keys
    key = cv2.waitKey(1) & 0xFF
    current_time = time.time()
    
    # DEBUG: Print key value
    if key != 255:
        print(f"Key pressed: {key} - {chr(key) if 32 <= key <= 126 else 'non-printable'}")
    
    # FIRST: Check for class selection (ALL letters and numbers)
    # Numbers 0-9 (48-57)
    if key >= 48 and key <= 57:
        current_class = chr(key)
        print(f"\n>>> SELECTED NUMBER: {current_class}")
    
    # Letters A-Z (lowercase 97-122)
    elif key >= 97 and key <= 122:
        current_class = chr(key - 32)  # Convert to uppercase
        print(f"\n>>> SELECTED LETTER: {current_class}")
    
    # Letters A-Z (uppercase 65-90)
    elif key >= 65 and key <= 90:
        current_class = chr(key)
        print(f"\n>>> SELECTED LETTER: {current_class}")
    
    # THEN: Check for other commands
    else:
        # Toggle auto-save (ENTER key - 13)
        if key == 13:
            auto_save = not auto_save
            print(f"\n>>> AUTO-SAVE: {'ON' if auto_save else 'OFF'}")
        
        # MANUAL SAVE (SPACE key - 32)
        elif key == 32:
            if current_class is None:
                print(">>> ERROR: No class selected! Press a letter or number first.")
            elif not hand_detected:
                print(">>> ERROR: No hand detected!")
            elif img_final is None:
                print(">>> ERROR: Could not process hand image!")
            else:
                folder = f"dataset/{current_class}"
                current_count = len(os.listdir(folder))
                
                if current_count < target_per_class:
                    filename = f"{current_count:04d}.jpg"
                    save_path = f"{folder}/{filename}"
                    
                    # Save the final image (already at 300x300 with white background)
                    cv2.imwrite(save_path, img_final)
                    
                    # Update stats
                    collection_stats[current_class] = current_count + 1
                    total_saved += 1
                    print(f">>> SAVED: {current_class}/{filename}")
                else:
                    print(f">>> {current_class} already has {target_per_class} images!")
        
        # AUTO-SAVE (continuous)
        elif auto_save and current_class is not None and hand_detected and img_final is not None:
            if current_time - last_save_time > save_delay:
                folder = f"dataset/{current_class}"
                current_count = len(os.listdir(folder))
                
                if current_count < target_per_class:
                    filename = f"{current_count:04d}.jpg"
                    save_path = f"{folder}/{filename}"
                    
                    # Save the final image
                    cv2.imwrite(save_path, img_final)
                    
                    # Update stats
                    collection_stats[current_class] = current_count + 1
                    total_saved += 1
                    last_save_time = current_time
                    print(f">>> AUTO-SAVED: {current_class}/{filename}")
        
        # Quit (ESC key - 27)
        elif key == 27:
            print("\n>>> CLOSING...")
            # Show closing message
            cv2.putText(img_display, "CLOSING...", (img_display.shape[1]//2-100, img_display.shape[0]//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.imshow("Data Collection", img_display)
            cv2.waitKey(500)  # Show message for 500ms
            cv2.destroyAllWindows()
            cap.release()
            exit()

cap.release()
cv2.destroyAllWindows()

# Show final stats
print(f"\nSession ended. Saved {total_saved} images at 300x300!")

print("\nFINAL COUNTS:")
for class_name in classes:
    folder = f"dataset/{class_name}"
    if os.path.exists(folder):
        count = len(os.listdir(folder))
        print(f"{class_name}: {count} images")