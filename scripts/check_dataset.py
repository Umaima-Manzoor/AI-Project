import os

print("-"*25)
print("Dataset Summary:".center(25))
print("-"*25)

total = 0       # Total images ko count krne keliye

# THis is for A-Z and 0-9 classes
# 65-90 are ASCII codes for A-Z, 48-57 are for 0-9
classes = [chr(i) for i in range(65, 91)] + [str(i) for i in range(10)]     # folder names need to be in string isliye str use kia

for class_name in classes:
    folder = f"dataset/{class_name}"    #folder ka path (should be inside dataset)
    if os.path.exists(folder):
        count = len(os.listdir(folder)) #agar folder exist krta hai to uske andar jitne files hai unko count kro, list create hui phir uski length nikaalo
        total += count
        print(f"{class_name}: {count} images")
    else:
        print(f"{class_name}: 0 images (folder missing)")

print("="*50)
print(f"Total: {total} images")

if total < 7200:
    print("\nTIP: Keep collecting! Aim for 200 images per class")
else:
    print("\nGreat! You have enough data to train")
