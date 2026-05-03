import cv2
import numpy as np
import os

def apply_preprocessing(image_path, size=(224, 224)):
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    resized = cv2.resize(img, size)

    blurred = cv2.GaussianBlur(resized, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    normalized = enhanced.astype('float32') / 255.0
    return normalized

def run_bulk_pipeline():
    
    base_raw = "data/raw"
    base_proc = "data/processed"
    
    subsets = ['train', 'test']
    categories = ['Glioma', 'Meningioma', 'Pituitary', 'No_Tumor']

    print("--- Starting Phase 1: Data Preprocessing ")

    for subset in subsets:
        for cat in categories:
            
            input_folder = os.path.join(base_raw, subset, cat)
            output_folder = os.path.join(base_proc, subset, cat)

            if not os.path.exists(input_folder):
                print(f"Skipping: {input_folder} (Not found)")
                continue

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
                print(f"Created folder: {output_folder}")

            print(f"Processing Category: {subset}/{cat}...")

            for filename in os.listdir(input_folder):
                
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    input_path = os.path.join(input_folder, filename)
                    output_path = os.path.join(output_folder, filename)

                    processed_img = apply_preprocessing(input_path)

                    if processed_img is not None:
                        
                        save_ready = (processed_img * 255).astype(np.uint8)
                        cv2.imwrite(output_path, save_ready)

    print("\nSuccess! Phase 1 complete. All images 'data/processed' are processed and saved.")

if __name__ == "__main__":
    run_bulk_pipeline()