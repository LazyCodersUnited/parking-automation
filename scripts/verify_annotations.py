import cv2
import os
import random

def verify_annotation(image_path, label_path, classes):
    """
    Reads an image and its YOLO annotation file, draws the bounding box,
    and displays the image for verification.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
        
    h, w, _ = image.shape

    if not os.path.exists(label_path):
        print(f"Warning: Annotation file not found for {os.path.basename(image_path)}.")
        cv2.imshow("Annotation Verification", image)
        print("Press any key to close this window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            class_id, x_center_norm, y_center_norm, width_norm, height_norm = map(float, parts)
            
            x_center = int(x_center_norm * w)
            y_center = int(y_center_norm * h)
            box_width = int(width_norm * w)
            box_height = int(height_norm * h)

            x_min = int(x_center - (box_width / 2))
            y_min = int(y_center - (box_height / 2))
            
            cv2.rectangle(image, (x_min, y_min), (x_min + box_width, y_min + box_height), (0, 255, 0), 2)
            label = classes[int(class_id)]
            cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Annotation Verification", image)
    print("Press any key to close this window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # IMPORTANT: Update this list to match the one in your conversion script
    CLASS_NAMES = ["licence", "licence-plate", "license-plate", "plate", "number-plate"] 
    
    image_dir = os.path.join(project_root, 'data', 'images', 'train')
    label_dir = os.path.join(project_root, 'data', 'labels', 'train')

    try:
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.png')]
    except FileNotFoundError:
        print(f"Error: The directory was not found: {image_dir}")
        print("Please make sure your folder structure is correct.")
        image_files = []

    if not image_files:
        print("No image files found. Exiting.")
    else:
        # --- Main Loop ---
        while True:
            random_image_name = random.choice(image_files)
            image_path = os.path.join(image_dir, random_image_name)
            label_name = os.path.splitext(random_image_name)[0] + '.txt'
            label_path = os.path.join(label_dir, label_name)
            
            print(f"\nVerifying: {random_image_name}")
            verify_annotation(image_path, label_path, CLASS_NAMES)

            # --- User Input to Continue or Stop ---
            user_input = input("Show another image? (y/n): ").lower()

            if user_input == 'y':
                continue  # Continue to the next loop iteration
            elif user_input == 'n':
                print("Stopping verification.")
                break  # Exit the loop
            else:
                print("Invalid input. Stopping verification.")
                break  # Exit the loop for any other input
