import os

folder_path = "C:\\Users\\ss790\\Documents\\code\Resnet18-Image-Classification-Model\\test_mini_cat"  # Change this to your folder path

for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
        old_path = os.path.join(folder_path, filename)
        new_filename = f"cat_{filename}"
        new_path = os.path.join(folder_path, new_filename)
        
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} → {new_filename}")