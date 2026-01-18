import os
import cv2
import numpy as np
import math


def extract_histogram_based(video_path, output_dir, target_distance_m, speed_kph, threshold):
    """
    Extract frames using histogram-based method
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames
        target_distance_m: Target distance between frames in meters
        speed_kph: Assumed driving speed in km/h
        threshold: Histogram difference threshold (0.05-0.5)
    
    Returns:
        tuple: (list of saved frame paths, error message if any)
    """
    
    cap = cv2.VideoCapture(video_path)
    success, prev_frame = cap.read()
    
    if not success:
        return None, f"Error: Cannot read video file {video_path}"
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25  # fallback default
    
    # calculate frame gap
    speed_mps = speed_kph * 1000 / 3600
    distance_per_frame = speed_mps / fps
    min_frame_gap = max(1, math.ceil(target_distance_m / distance_per_frame))
    
    frame_counter = 0
    keyframe_counter = 0
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_hist = cv2.calcHist([prev_gray], [0], None, [256], [0, 256])
    
    last_keyframe_idx = 0
    saved_frames = []
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    
    while True:
        success, curr_frame = cap.read()
        if not success:
            break
        
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        curr_hist = cv2.calcHist([curr_gray], [0], None, [256], [0, 256])
        diff = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA)
        
        if diff > threshold and (frame_counter - last_keyframe_idx) >= min_frame_gap:
            keyframe_filename = os.path.join(output_dir, f"{video_name}_{keyframe_counter:04d}.jpg")
            cv2.imwrite(keyframe_filename, curr_frame)
            saved_frames.append(keyframe_filename)
            keyframe_counter += 1
            
            prev_hist = curr_hist
            last_keyframe_idx = frame_counter
        
        prev_gray = curr_gray
        frame_counter += 1
    
    cap.release()
    return saved_frames, None