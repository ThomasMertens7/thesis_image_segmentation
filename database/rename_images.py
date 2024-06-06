from PIL import Image
import os


def rename_images(folder_path, new_folder_path=None):
    i = 1  # Start of incremental factor
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            img_path = os.path.join(folder_path, filename)
            with Image.open(img_path) as img:
                # New filename with incremental factor
                new_filename = folder_path + f'-new-{i}.png'  # Saving as PNG, you can change format if needed
                i += 1

                # If a new folder path is provided, save the image there. Otherwise, replace the old one.
                if new_folder_path:
                    new_image_path = os.path.join(new_folder_path, new_filename)
                else:
                    new_image_path = os.path.join(folder_path, new_filename)

                img.save(new_image_path)
                print(f'Renamed {filename} to {new_filename}.')


# Example usage
folder_path = 'sheep'
new_folder_path = 'sheep-new'  # Optional: Specify a different folder to save the renamed images
rename_images(folder_path, new_folder_path)
