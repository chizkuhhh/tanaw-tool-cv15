import os
import cv2
import glob
from ultralytics import YOLO

def yolo_to_voc(bbox, image_size):
    """
    Convert YOLO (x_center, y_center, width, height) normalized format
    to VOC (x_min, y_min, x_max, y_max) pixel coordinates.
    """
    x_center, y_center, w, h = bbox
    img_w, img_h = image_size

    x_center *= img_w
    y_center *= img_h
    w *= img_w
    h *= img_h

    x_min = x_center - w / 2
    y_min = y_center - h / 2
    x_max = x_center + w / 2
    y_max = y_center + h / 2
    return [x_min, y_min, x_max, y_max]

def anonymize(image, regions):
    """
    Blurs the image, given the x1,y1,x2,y2 coordinates using Gaussian Blur.
    """
    for region in regions:
        x1, y1, x2, y2 = region
        x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
        roi = image[y1:y2, x1:x2]
        blurred_roi = cv2.GaussianBlur(roi, (31, 31), 0)
        image[y1:y2, x1:x2] = blurred_roi
    return image

def anonymize_images(input_folder, output_folder, model_path, conf=0.05):
    """
    Anonymize faces and license plates in images
    
    Args:
        input_folder: Directory containing input images
        output_folder: Directory to save anonymized images
        model_path: Path to YOLO model (.pt file)
        conf: Confidence threshold for detections
    
    Returns:
        tuple: (number of processed images, error message if any)
    """
    # Load YOLO model
    try:
        det_model = YOLO(model_path)
    except Exception as e:
        return 0, f"Error loading model: {str(e)}"
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Create temp directories for predictions
    annot_txt_dir = os.path.join(output_folder, "annot_txt")
    os.makedirs(annot_txt_dir, exist_ok=True)
    
    # Run YOLO detection
    try:
        # Clear previous predictions to avoid incrementing folders
        pred_dir = os.path.join(output_folder, "pred")
        if os.path.exists(pred_dir):
            import shutil
            shutil.rmtree(pred_dir)
        
        # Use absolute path for project to ensure correct location
        abs_output_folder = os.path.abspath(output_folder)
        
        results = det_model(
            source=input_folder,
            save=False,
            save_txt=True,
            conf=conf,
            device='cpu',
            project=abs_output_folder,
            name="pred",
            exist_ok=True  # Overwrite if exists
        )
    except Exception as e:
        return 0, f"Error during detection: {str(e)}"
    
    text_dir = os.path.join(output_folder, "pred", "labels")
    abs_text_dir = os.path.abspath(text_dir)
    print(f"Looking for labels at: {abs_text_dir}")
    print(f"Labels directory exists: {os.path.exists(abs_text_dir)}")
    
    # Get image dimensions from first image
    images = sorted(glob.glob(os.path.join(input_folder, "*")))
    if not images:
        return 0, "No images found in input folder"
    
    sample_img = cv2.imread(images[0])
    if sample_img is None:
        return 0, "Could not read sample image"
    height_output_images, width_output_images = sample_img.shape[:2]
    
    # Convert YOLO format to VOC format (only if detections exist)
    if os.path.exists(text_dir):
        print(f"Found labels directory: {text_dir}")
        try:
            label_files = [f for f in os.listdir(text_dir) if f.endswith('.txt')]
            print(f"Found {len(label_files)} label files")
            
            for file in label_files:
                label_path = os.path.join(text_dir, file)
                print(f"Processing label file: {file}")
                
                with open(label_path, 'r') as fin:
                    lines = fin.readlines()
                    print(f"  - {len(lines)} detections in {file}")
                    
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # YOLO format: class_id x_center y_center width height
                            bbox = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                            bbox_voc = yolo_to_voc(bbox, (width_output_images, height_output_images))
                            data_string = " ".join(str(num) for num in bbox_voc)
                            
                            annot_file_path = os.path.join(annot_txt_dir, file)
                            with open(annot_file_path, "a") as f:
                                f.write(data_string + "\n")
                            print(f"  - Wrote VOC bbox: {data_string}")
        except Exception as e:
            print(f"Error converting coordinates: {str(e)}")
            return 0, f"Error converting coordinates: {str(e)}"
    else:
        print(f"No labels directory found at: {text_dir}")
    
    # Apply anonymization (or just copy images if no detections)
    txt_files = [f for f in os.listdir(annot_txt_dir) if f.endswith('.txt')] if os.path.exists(annot_txt_dir) else []
    processed_count = 0
    
    # Get all images from input folder
    image_files = [f for f in os.listdir(input_folder) 
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    # Process all images
    for image_name in image_files:
        try:
            image_path = os.path.join(input_folder, image_name)
            txt_file = os.path.splitext(image_name)[0] + '.txt'
            txt_path = os.path.join(annot_txt_dir, txt_file)
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read image: {image_path}")
                continue
            
            # Apply blur only if annotations exist
            if os.path.exists(txt_path):
                print(f"Found annotations for {image_name}")
                with open(txt_path, 'r') as f:
                    lines = f.readlines()
                
                print(f"  - {len(lines)} bounding boxes to blur")
                bboxes = []
                for line in lines:
                    values = line.strip().split()
                    if len(values) >= 4:
                        x_min, y_min, x_max, y_max = map(float, values[:4])
                        x_min, y_min, x_max, y_max = int(round(x_min)), int(round(y_min)), int(round(x_max)), int(round(y_max))
                        bboxes.append([x_min, y_min, x_max, y_max])
                        print(f"  - Bbox: ({x_min}, {y_min}) to ({x_max}, {y_max})")
                
                if bboxes:
                    image = anonymize(image, bboxes)
                    print(f"  - Applied blur to {len(bboxes)} regions")
            else:
                print(f"No annotations for {image_name}, copying original")
            
            # Save image (anonymized or original)
            output_path = os.path.join(output_folder, image_name)
            cv2.imwrite(output_path, image)
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {image_name}: {str(e)}")
            continue
    
    print(f"Processed {processed_count} images total")
    return processed_count, None