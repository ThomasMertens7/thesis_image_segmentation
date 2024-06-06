from PIL import Image
import os


def convert_png_to_jpeg(source_folder, target_folder=None):
    if not target_folder:
        target_folder = source_folder  # Save JPEGs in the same folder if no target folder is specified

    for filename in os.listdir(source_folder):
        if filename.endswith('.png'):
            img_path = os.path.join(source_folder, filename)
            with Image.open(img_path) as img:
                # Remove the '.png' extension and add '.jpeg'
                base_filename = os.path.splitext(filename)[0]
                new_filename = f'{base_filename}.jpeg'

                # Define the full path for the new image
                new_img_path = os.path.join(target_folder, new_filename)

                # Convert and save the image as JPEG
                rgb_img = img.convert('RGB')  # Convert to RGB in case it's a palette-based image
                rgb_img.save(new_img_path, quality=95)
                print(f'Converted {filename} to {new_filename}.')


# Example usage
source_folder = 'all_imgs_new'
convert_png_to_jpeg(source_folder)
