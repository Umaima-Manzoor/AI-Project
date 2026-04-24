import os
import shutil

print("\n" + "="*50)
print("DELETE ALL DATASET IMAGES")
print("="*50)

# Classes A-Z and 0-9
classes = [chr(i) for i in range(65, 91)] + [str(i) for i in range(10)]

total_deleted = 0

for class_name in classes:
    folder = f"dataset/{class_name}"
    if os.path.exists(folder):
        files = os.listdir(folder)
        count = len(files)
        
        for file in files:
            file_path = os.path.join(folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        total_deleted += count
        print(f"Deleted {count} images from {class_name}")

print("="*50)
print(f"TOTAL DELETED: {total_deleted} images")
print("="*50)