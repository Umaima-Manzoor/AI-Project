import os

print("\n" + "="*50)
print("DATASET SUMMARY")
print("="*50)

total = 0
classes = [chr(i) for i in range(65, 91)] + [str(i) for i in range(10)]

for class_name in classes:
    folder = f"dataset/{class_name}"
    if os.path.exists(folder):
        count = len(os.listdir(folder))
        total += count
        print(f"{class_name}: {count} images")
    else:
        print(f"{class_name}: 0 images (folder missing)")

print("="*50)
print(f"TOTAL: {total} images")
print("="*50)

if total < 2600:  # 36 classes * ~70 images each
    print("\nTIP: Keep collecting! Aim for at least 50-70 images per class")
else:
    print("\n Great! You have enough data to train")
