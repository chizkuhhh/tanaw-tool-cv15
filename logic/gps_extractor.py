import os
import cv2
import numpy as np
import gpxpy
from haversine import haversine

def extract_gps_based(video_path, gpx_path, output_dir, interval):
    """
    Extract frames using GPS-based method
    
    Args:
        video_path: Path to video file
        gpx_path: Path to corresponding GPX file
        output_dir: Directory to save extracted frames
        interval: Distance interval between frames in meters
    
    Returns:
        tuple: (list of saved frame paths, error message if any)
    """

    # parse GPX
    try:
        with open(gpx_path) as gpx_file:
            gpx = gpxpy.parse(gpx_file)
    except Exception as e:
        return None, f"Error parsing GPX file: {str(e)}"
    
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append((point.latitude, point.longitude, point.time))
    
    if not points:
        return None, "No GPS points found in GPX file"
    
    # calculate cumulative distances and timestamps
    distances = [0.0]
    timestamps = [points[0][2]]
    for i in range(1, len(points)):
        dist = haversine(points[i-1][:2], points[i][:2]) * 1000  # convert to meters
        distances.append(distances[-1] + dist)
        timestamps.append(points[i][2])
    
    query_distances = np.arange(0, distances[-1], interval)
    query_times = np.interp(query_distances, distances, [t.timestamp() for t in timestamps])
    
    # extract frames
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps <= 0:
        cap.release()
        return None, "Could not determine video FPS"
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    
    saved_frames = []
    last_frame_num = None
    
    for idx, sec in enumerate(query_times):
        frame_num = int(fps * (sec - query_times[0]))
        
        # skip duplicate frames
        if frame_num == last_frame_num:
            continue
        last_frame_num = frame_num
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        success, frame = cap.read()
        if success:
            out_fname = os.path.join(output_dir, f"{video_name}_frame_{idx:04d}.jpg")
            cv2.imwrite(out_fname, frame)
            saved_frames.append(out_fname)
    
    cap.release()
    return saved_frames, None