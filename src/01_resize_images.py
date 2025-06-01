# src/resize_images.py
import os
from PIL import Image
from dask import delayed, compute
from tqdm import tqdm

INPUT_DIR = 'data/raw'
OUTPUT_DIR = 'data/processed'
TARGET_SIZE = (150, 150)

def process_image(class_name, image_name):
    in_path = os.path.join(INPUT_DIR, class_name, image_name)
    out_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, image_name)

    try:
        with Image.open(in_path) as img:
            img = img.convert("RGB")
            img = img.resize(TARGET_SIZE)
            img.save(out_path)
        return f"OK: {image_name}"
    except Exception as e:
        return f"ERROR: {image_name} -> {str(e)}"

def process_all_images():
    tasks = []
    for class_name in os.listdir(INPUT_DIR):
        class_path = os.path.join(INPUT_DIR, class_name)
        for image_name in os.listdir(class_path):
            task = delayed(process_image)(class_name, image_name)
            tasks.append(task)
    results = compute(*tasks)
    for r in results:
        print(r)

if __name__ == '__main__':
    process_all_images()
