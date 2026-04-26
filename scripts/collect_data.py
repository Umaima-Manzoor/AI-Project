import cv2          # camera use krne keliye and image processing keliye
import os           # classes k folders and files keliye
import time
import math
import numpy as np  # arrays and matrices keliye
from cvzone.HandTrackingModule import HandDetector

print("-"*60)
print("Data Collection for Sign Language Project".center(60))
print("-"*60)

if not os.path.exists("dataset"):   #dataset folder banaane keliye agar pehle se nhi hai
    os.makedirs("dataset")          #makedirs is safer bcz agar yahaan koi parent folder bhi hota to wo bhi ban jaata

classes = [chr(i) for i in range(65, 91)] + [str(i) for i in range(10)]

for class_name in classes:          
    folder_path = f"dataset/{class_name}"       #dataset k ander saari classes ke folders banao agar pehle se nhi hai to
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {class_name}")

cap = cv2.VideoCapture(0)       #opens the default camera (0 is usually the built-in webcam), agar multiple cameras hai to 1, 2, etc. use krte hain, basically hardware se software ko connect krta hai
cap.set(3, 1280)        #this is for resolution
cap.set(4, 720)         #3 mtlb width and 4 mtlb height


detector = HandDetector(detectionCon=0.8, maxHands=1)   #detectionCon is confidence threshold mtlb agar 80% ya ussey ziada sure hoga tabhi usko hand consider krega - maxHands mtlb max kitne hands detect kr skta hai

imgSize = 300       #every image will be 300x300 after processing and cropping and everything and yehi size training keliye use hoga
offset = 60         #cropping k waqt hand k around thora extra space (60 pixels) dene keliye taake koi finger cut off na ho jaye

auto_save = False   #auto-save ka toggle, agar true hoga to jab bhi hand detect hoga aur current class select hogi to automatically save hoga
save_delay = 0.3 
last_save_time = 0  #time when the last image was auto-saved - iske baad se 0.3secs wali cheez check krte hain
target_per_class = 200

collection_stats = {}       #basically for keeping track k har class mein kitni images ho gyi hain and if it has reached the target
for class_name in classes:
    folder_path = f"dataset/{class_name}"
    if os.path.exists(folder_path):
        collection_stats[class_name] = len(os.listdir(folder_path))
    else:
        collection_stats[class_name] = 0

print("\nINSTRUCTIONS:")
print("- Screen shows mirrored view")
print("- Images saved at 300x300 with white background (no stretching)")
print("- Press LETTER (A-Z) or NUMBER (0-9) to select class")
print("- Press SPACE to save one image")
print("- Press ENTER to toggle auto-save ON/OFF")
print("- Press ESC to quit\n")

current_class = None    #this tells image ko kis class mein save krna hai
total_saved = 0         #total count of current session - collection_stats jo overall dekhta hai

while True:
    success, img = cap.read()   #capturing a frame from the webcam, success=True if successfull, img=the captured frame (image)
    if not success:     #agar camera kaam nhi kr rha to exit the loop
        break
    
    h_img, w_img, _ = img.shape     #height (720), width (1280), and number of color channels (3=RGB) of the captured image, channels ki zaroorat nhi hai
    # h_img is used for cropping and resizing, w_img is used for mirroring and display purposes

    img_display = cv2.flip(img, 1)      #mirroring the image for display, taake user ko aisa lage ki wo apne haath ko dekh rha hai, flip function mein 0 means vertical flip, 1 means horizontal flip, -1 means both 
    
    hands, _ = detector.findHands(img, draw=True)   #draw means original image pr bounding box and dots draw krna hai ya nhi, hands is a list of hands detected in it, _ is the image itself being returned with/without the stuff (depends on draw=True/False)
    img_final = None        #wo image jo save hogi, after cropping and resizing
    hand_detected = False 
    
    if hands:       
        hand_detected = True
        hand = hands[0]             #aik hi hand hai to first index pr hi hoga
        x, y, w, h = hand['bbox']   #x,y = top-left corner of bounding box
        hand_type = hand['type']   
        
        if hand_type == 'Right':
            display_x = w_img - x - w       #top left corner for the mirrored one
            cv2.rectangle(img_display, (display_x, y), (display_x + w, y + h), (0, 255, 0), 3)      #green rectangle for right hand, bottom-right corner, 3=thickness of frame
            cv2.putText(img_display, "RIGHT HAND", (display_x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)    #font size, colour, thickness
            
            x_start = max(0, x - offset)            #cropping keliye boundaries with offsets - max taake neg mein na jaaye (outside left ya top), min taake image ke andar hi rahe (outsideright ya bottom)
            x_end = min(w_img, x + w + offset)
            y_start = max(0, y - offset)
            y_end = min(h_img, y + h + offset)
            
            img_crop = img[y_start:y_end, x_start:x_end]        #from actual image - y=rows x=cols
            
            if img_crop.size > 0:       #if cropping was successful to atleast 1 pixel to hoga
                img_white = np.ones((imgSize, imgSize, 3), np.uint8) * 255      #creates a 3d array 300x300 filled with white       agar hum int specify nhi krte to default mein float aayega and then phir prob hogi
                
                h_crop, w_crop, _ = img_crop.shape      #h and w of cropped wali, no need for channels
                aspect_ratio = h_crop / w_crop          # >1 = taller, <1 = wider, 1 = perfect square, aspect ratio se decide hoga ki kaise resize karna hai taake stretch na ho, aur white background mein fit ho jaye
                
                if aspect_ratio > 1:            #resize height to 300 and width uske hisaab se and hand will be beech mein and sides mein white 
                    k = imgSize / h_crop            #scale factor to brign down height ot 300
                    w_cal = math.ceil(k * w_crop)       #agar round down krenge to ho skta hai k thora stretch hojaaye
                    img_resize = cv2.resize(img_crop, (w_cal, imgSize))
                    w_gap = math.ceil((imgSize - w_cal) / 2)
                    img_white[:, w_gap:w_gap + w_cal] = img_resize
                else: 
                    k = imgSize / w_crop
                    h_cal = math.ceil(k * h_crop)
                    img_resize = cv2.resize(img_crop, (imgSize, h_cal))
                    h_gap = math.ceil((imgSize - h_cal) / 2)
                    img_white[h_gap:h_gap + h_cal, :] = img_resize
                
                img_final = img_white
                
                cv2.imshow("Will Save (300x300 - No Stretch)", img_final)
        else:
            display_x = w_img - x - w
            cv2.rectangle(img_display, (display_x, y), (display_x + w, y + h), (0, 0, 255), 3)          #red
            cv2.putText(img_display, "LEFT HAND - NOT SAVED", (display_x, y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    

    
    if current_class:
        current_count = collection_stats.get(current_class, 0)          #agar pehle se nhi hain to 0 - prevents errors if folder doesn't exist or something
        cv2.putText(img_display, f"CLASS: {current_class}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  #20=x 40=y - green
        cv2.putText(img_display, f"IMAGES: {current_count}/{target_per_class}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)  #cyan
    
    if hand_detected:
        cv2.putText(img_display, "HAND DETECTED", (img_display.shape[1] - 200, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)   #top right corer
    else:
        cv2.putText(img_display, "NO HAND", (img_display.shape[1] - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    if auto_save:
        cv2.putText(img_display, "AUTO-SAVE ON", (img_display.shape[1] - 200, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    controls_y = 100        #starting y for control panel
    controls_list = [
        "CONTROLS:",
        "A-Z / 0-9 - Select class",
        "SPACE - Save one",
        "ENTER - Toggle autosave",
        "ESC - Quit"
    ]
    
    overlay = img_display.copy()        #semi-transparent box for control panel
    cv2.rectangle(overlay, (img_display.shape[1]-350, 80), (img_display.shape[1]-30, 230), (0, 0, 0), -1)       #filled black box
    cv2.addWeighted(overlay, 0.7, img_display, 0.3, 0, img_display)     #blending mirrored with overlay (copy with control panel)
    
    for i, text in enumerate(controls_list):
        color = (255, 255, 255) if i == 0 else (200, 200, 200)      #first line white, rest light gray
        cv2.putText(img_display, text, (img_display.shape[1]-340, controls_y + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    cv2.imshow("Data Collection", img_display)
    
    key = cv2.waitKey(1) & 0xFF     #waits for a key press for 1ms, and returns the ASCII code of the key pressed, & 0xFF is used to get the last 8 bits of the key code, since kuch systems mein 16 bit and kuch mein 32 bit use hote hain
    current_time = time.time()
    
    if key != 255:      #no key is pressed
        print(f"Key pressed: {key} - {chr(key) if 32 <= key <= 126 else 'non-printable'}")      #32=space 126=~ wrna non-printable char like shift ya enter
    
    if key >= 48 and key <= 57:
        current_class = chr(key)        #to ascii
        print(f"\n>>> SELECTED NUMBER: {current_class}")
    
    elif key >= 97 and key <= 122:      #lower case
        current_class = chr(key - 32)   #upper case
        print(f"\n>>> SELECTED LETTER: {current_class}")
    
    elif key >= 65 and key <= 90:
        current_class = chr(key)
        print(f"\n>>> SELECTED LETTER: {current_class}")
    
    else:
        if key == 13:       #enter
            auto_save = not auto_save   
            print(f"\n>>> AUTO-SAVE: {'ON' if auto_save else 'OFF'}")
        
        elif key == 32:      #space
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
                    filename = f"{current_count:04d}.jpg"       #0000, 0001
                    save_path = f"{folder}/{filename}"
                    
                    cv2.imwrite(save_path, img_final)
                    
                    collection_stats[current_class] = current_count + 1
                    total_saved += 1
                    print(f">>> SAVED: {current_class}/{filename}")
                else:
                    print(f">>> {current_class} already has {target_per_class} images!")
        
        elif auto_save and current_class is not None and hand_detected and img_final is not None:
            if current_time - last_save_time > save_delay:      #preventing too many saves in a short time
                folder = f"dataset/{current_class}"
                current_count = len(os.listdir(folder))
                
                if current_count < target_per_class:
                    filename = f"{current_count:04d}.jpg"
                    save_path = f"{folder}/{filename}"
                    
                    cv2.imwrite(save_path, img_final)
                    
                    collection_stats[current_class] = current_count + 1
                    total_saved += 1
                    last_save_time = current_time
                    print(f">>> AUTO-SAVED: {current_class}/{filename}")
        
        elif key == 27:
            print("\n>>> CLOSING...")       #esc
            cv2.putText(img_display, "CLOSING...", (img_display.shape[1]//2-100, img_display.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.imshow("Data Collection", img_display)      #writing again to show the closing message - updating the window aik tarah se
            cv2.waitKey(500) 
            cv2.destroyAllWindows()     
            cap.release()
            exit()

cap.release()           #backup closing agar camera aik dam se band hojaye ya something like that
cv2.destroyAllWindows()

print(f"\nSession ended. Saved {total_saved} images at 300x300!")

print("\nFINAL COUNTS:")
for class_name in classes:
    folder = f"dataset/{class_name}"
    if os.path.exists(folder):
        count = len(os.listdir(folder))
        print(f"{class_name}: {count} images")
