import xml.etree.ElementTree as ET
import os

def convert_xml_to_yolo(xml_file_path, classes):
    """
    Parses an XML file (Pascal VOC format) and converts its bounding box
    annotations to the YOLO format.

    Args:
        xml_file_path (str): The full path to the XML annotation file.
        classes (list): A list of class names. The index of the class name
                        in this list will be used as the class ID.

    Returns:
        list: A list of strings, where each string is a YOLO-formatted annotation.
              Returns an empty list if parsing fails or no objects are found.
    """
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        size = root.find('size')
        if size is None:
            print(f"Warning: 'size' tag not found in {xml_file_path}. Skipping.")
            return []
            
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)

        yolo_annotations = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in classes:
                print(f"Warning: Class '{class_name}' not in provided classes list. Skipping.")
                continue
            
            class_id = classes.index(class_name)

            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            # --- YOLO format calculation ---
            # x_center = (xmin + xmax) / 2
            # y_center = (ymin + ymax) / 2
            # box_width = xmax - xmin
            # box_height = ymax - ymin

            # Normalize coordinates (values between 0 and 1)
            x_center_norm = ((xmin + xmax) / 2) / img_width
            y_center_norm = ((ymin + ymax) / 2) / img_height
            width_norm = (xmax - xmin) / img_width
            height_norm = (ymax - ymin) / img_height
            
            yolo_annotations.append(
                f"{class_id} {x_center_norm} {y_center_norm} {width_norm} {height_norm}"
            )
            
        return yolo_annotations

    except Exception as e:
        print(f"Error processing file {xml_file_path}: {e}")
        return []


def process_dataset(annotation_dir, output_dir, classes):
    """
    Processes all XML files in a directory, converts them to YOLO format,
    and saves them in the output directory.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    xml_files = [f for f in os.listdir(annotation_dir) if f.endswith('.xml')]
    print(f"Found {len(xml_files)} XML files in {annotation_dir}.")
    
    converted_count = 0
    for xml_file in xml_files:
        xml_path = os.path.join(annotation_dir, xml_file)
        yolo_data = convert_xml_to_yolo(xml_path, classes)

        if yolo_data:
            # Create the corresponding .txt filename
            base_name = os.path.splitext(xml_file)[0]
            txt_filename = base_name + '.txt'
            txt_path = os.path.join(output_dir, txt_filename)

            # Write the YOLO formatted data to the .txt file
            with open(txt_path, 'w') as f:
                f.write('\n'.join(yolo_data))
            converted_count += 1
            
    print(f"Successfully converted {converted_count}/{len(xml_files)} files.")


if __name__ == '__main__':
    # --- Configuration ---
    # IMPORTANT: Define your class names here. The order matters.
    # If your XML files use a different name for the license plate, change it here.
    CLASS_NAMES = ["licence", "licence-plate", "license-plate", "plate", "number-plate"] 
    
    # Define base paths assuming the script is in the 'scripts' folder
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Input directories (where your XMLs are)
    train_ann_dir = os.path.join(project_root, 'data', 'annotations', 'train')
    valid_ann_dir = os.path.join(project_root, 'data', 'annotations', 'valid')
    
    # Output directories (where the new .txt files will be saved)
    train_labels_out_dir = os.path.join(project_root, 'data', 'labels', 'train')
    valid_labels_out_dir = os.path.join(project_root, 'data', 'labels', 'valid')

    # --- Run the conversion for both training and validation sets ---
    print("="*50)
    print("Converting TRAINING annotations...")
    process_dataset(train_ann_dir, train_labels_out_dir, CLASS_NAMES)
    
    print("\n" + "="*50)
    print("Converting VALIDATION annotations...")
    process_dataset(valid_ann_dir, valid_labels_out_dir, CLASS_NAMES)
    
    print("\n" + "="*50)
    print("Annotation conversion complete!")
    print("New .txt label files are saved in the 'data/labels/' directory.")
    print("="*50)
