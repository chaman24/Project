from coordinates import get_bay_coordinates
import cv2
import numpy as np
import os

# Define the directories
source_directory = '/home/chaman99/Project/CapturedImages'
destination_directory = '/home/chaman99/Project/ExtractedImages'
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# Get the list of image files in the source directory
image_files = [f for f in os.listdir(source_directory) if f.endswith('.jpg')]

# Process each image file
for image_file in image_files:
    img_path = os.path.join(source_directory, image_file)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Error loading image {image_file}")
        continue

    bay_coordinates = get_bay_coordinates()

    i = 1
    for bay in bay_coordinates:
        pts = np.array(bay, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (0, 255, 255))

        # Start extraction process
        rect = cv2.boundingRect(pts)
        x, y, w, h = rect
        cropped = img[y:y + h, x:x + w].copy()
        
        # Create mask
        pts = pts - pts.min(axis=0)
        mask = np.zeros(cropped.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
        
        # Bitwise operation
        dst = cv2.bitwise_and(cropped, cropped, mask=mask)
        
        # Add black background
        bg = np.zeros_like(cropped, np.uint8)
        dst2 = bg + dst
        
        # Save the extracted images
        extracted_image_path = os.path.join(destination_directory, f"crop{i}_{image_file}")
        cv2.imwrite(extracted_image_path, dst2)
        i += 1

print("Process Complete")
