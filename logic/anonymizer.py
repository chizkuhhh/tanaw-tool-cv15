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

def anonymize_images(input_folder, output_folder, model_path, conf=0.05, progress_callback=None):
    """
    Anonymize faces and license plates in images
    
    Args:
        input_folder: Directory containing input images
        output_folder: Directory to save anonymized images
        model_path: Path to YOLO model (.pt file)
        conf: Confidence threshold for detections
        progress_callback: Optional function(current, total) called to report progress
    
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
    
    # Get list of images
    image_files = sorted([f for f in os.listdir(input_folder) 
                          if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    total_images = len(image_files)
    
    if total_images == 0:
        return 0, "No images found in input folder"
    
    # Get image dimensions from first image
    sample_img = cv2.imread(os.path.join(input_folder, image_files[0]))
    if sample_img is None:
        return 0, "Could not read sample image"
    height_output_images, width_output_images = sample_img.shape[:2]
    
    # Process each image with YOLO
    processed_count = 0
    
    for idx, image_name in enumerate(image_files):
        image_path = os.path.join(input_folder, image_name)
        
        # Run YOLO on single image
        results = det_model(image_path, verbose=True, conf=conf, device='cpu')
        
        # Convert detections to VOC format
        txt_filename = os.path.splitext(image_name)[0] + '.txt'
        annot_file_path = os.path.join(annot_txt_dir, txt_filename)
        
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                with open(annot_file_path, "w") as f:
                    for box in boxes:
                        # Get normalized coordinates (YOLO format)
                        x_center, y_center, w, h = box.xywhn[0].tolist()
                        
                        # Convert to VOC (pixel coordinates)
                        bbox_voc = yolo_to_voc([x_center, y_center, w, h], 
                                              (width_output_images, height_output_images))
                        data_string = " ".join(str(num) for num in bbox_voc)
                        f.write(data_string + "\n")
        
        # Apply anonymization
        image = cv2.imread(image_path)
        if image is not None:
            if os.path.exists(annot_file_path):
                with open(annot_file_path, 'r') as f:
                    lines = f.readlines()
                
                bboxes = []
                for line in lines:
                    values = line.strip().split()
                    if len(values) >= 4:
                        x_min, y_min, x_max, y_max = map(float, values[:4])
                        x_min, y_min, x_max, y_max = int(round(x_min)), int(round(y_min)), int(round(x_max)), int(round(y_max))
                        bboxes.append([x_min, y_min, x_max, y_max])
                
                if bboxes:
                    image = anonymize(image, bboxes)
            
            # Save image
            output_path = os.path.join(output_folder, image_name)
            cv2.imwrite(output_path, image)
            processed_count += 1
        
        # Report progress (YOLO already printed "image X/Y", we just update our UI)
        if progress_callback:
            progress_callback(idx + 1, total_images)
    
    return processed_count, None