import os

folder = "saved_models"

for filename in os.listdir(folder):
    if "=" in filename:
        old_path = os.path.join(folder, filename)
        new_filename = filename.replace("=", "_")
        new_path = os.path.join(folder, new_filename)

        os.rename(old_path, new_path)
        print(f"{filename} -> {new_filename}")