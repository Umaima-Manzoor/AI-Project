import os

print("-"*30)
print("Delete All Images in Dataset".center(30))
print("-"*30)

classes = [chr(i) for i in range(65, 91)] + [str(i) for i in range(10)]     #same thing jo check_dataset mein kia tha

total_deleted = 0

for class_name in classes:
    folder = f"dataset/{class_name}"
    if os.path.exists(folder):
        files = os.listdir(folder)      #list of all images in the folder
        count = len(files)              #counting how many are deleted per class
        
        for file in files:
            file_path = os.path.join(folder, file)      #creates the complete path of img - folder ka path + file name
            if os.path.isfile(file_path):       #agar path file ka hai (not a folder), then delete it
                os.remove(file_path)
        
        total_deleted += count
        print(f"Deleted {count} images from {class_name}")

print("="*50)
print(f"Total Deleted: {total_deleted} images")
