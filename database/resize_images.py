from PIL import Image
import os


def resize_images_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            img_path = os.path.join(folder_path, filename)
            with Image.open(img_path) as img:
                width, height = img.size
                max_dimension = max(width, height)

                # Calculate the scaling factor, ensuring the max dimension is 640 pixels
                scaling_factor = 640 / max_dimension if max_dimension > 640 else 1

                # Only resize if the image is larger than the max dimensions
                if scaling_factor < 1:
                    new_width = int(width * scaling_factor)
                    new_height = int(height * scaling_factor)

                    # Resize the image and save it, replacing the old one
                    resized_img = img.resize((new_width, new_height), Image.ANTIALIAS)
                    resized_img.save(img_path)
                    print(f'Resized {filename} to {new_width}x{new_height} pixels.')

# Example usage
folder_path = 'dog'
resize_images_in_folder(folder_path)