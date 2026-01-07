"""
Computer Vision Utility Functions
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image to target size
    
    Args:
        image: Input image
        target_size: (width, height)
        
    Returns:
        resized: Resized image
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def preprocess_frame(frame: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
    """
    Preprocess video frame for detection
    
    Args:
        frame: Input frame
        target_size: Optional target size
        
    Returns:
        processed: Preprocessed frame
    """
    processed = frame.copy()
    
    # Resize if needed
    if target_size:
        processed = resize_image(processed, target_size)
    
    # Convert to RGB if needed
    if len(processed.shape) == 2:
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    
    return processed


def draw_zones(frame: np.ndarray, zones: List[Dict], 
               colors: List[Tuple] = None) -> np.ndarray:
    """
    Draw zones on frame
    
    Args:
        frame: Input frame
        zones: List of zone dictionaries with 'bbox' key
        colors: Optional list of BGR colors for each zone
        
    Returns:
        annotated: Frame with zones drawn
    """
    annotated = frame.copy()
    
    if colors is None:
        colors = [(0, 255, 0)] * len(zones)
    
    for zone, color in zip(zones, colors):
        x1, y1, x2, y2 = zone['bbox']
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        label = zone.get('name', f"Zone {zone['id']}")
        cv2.putText(annotated, label, (x1 + 5, y1 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return annotated


def draw_trajectory(frame: np.ndarray, points: List[Tuple[int, int]],
                    color: Tuple[int, int, int] = (0, 255, 255),
                    thickness: int = 2) -> np.ndarray:
    """
    Draw trajectory line on frame
    
    Args:
        frame: Input frame
        points: List of (x, y) coordinates
        color: Line color (BGR)
        thickness: Line thickness
        
    Returns:
        annotated: Frame with trajectory
    """
    annotated = frame.copy()
    
    if len(points) < 2:
        return annotated
    
    # Draw lines between consecutive points
    for i in range(len(points) - 1):
        pt1 = tuple(points[i])
        pt2 = tuple(points[i + 1])
        cv2.line(annotated, pt1, pt2, color, thickness)
    
    # Draw circles at each point
    for pt in points:
        cv2.circle(annotated, tuple(pt), 3, color, -1)
    
    return annotated


def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two boxes
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        iou: IoU score
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def non_max_suppression(boxes: List[List[int]], scores: List[float],
                        threshold: float = 0.5) -> List[int]:
    """
    Apply Non-Maximum Suppression
    
    Args:
        boxes: List of bounding boxes [x1, y1, x2, y2]
        scores: Confidence scores for each box
        threshold: IoU threshold
        
    Returns:
        keep_indices: Indices of boxes to keep
    """
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]
    
    return keep


def create_heatmap(detections: List[Dict], frame_shape: Tuple[int, int],
                   kernel_size: int = 51) -> np.ndarray:  # default to odd number
    """
    Create density heatmap from detections
    """
    height, width = frame_shape[:2]
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    # Add Gaussian blur at each detection center
    for det in detections:
        x, y = det['center']
        if 0 <= x < width and 0 <= y < height:
            heatmap[y, x] = 1
    
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Apply Gaussian blur
    heatmap = cv2.GaussianBlur(heatmap, (kernel_size, kernel_size), 0)
    
    # Normalize
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap



def apply_colormap(heatmap: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Apply colormap to heatmap
    
    Args:
        heatmap: Grayscale heatmap (0-1 or 0-255)
        colormap: OpenCV colormap constant
        
    Returns:
        colored: Colored heatmap
    """
    if heatmap.max() <= 1.0:
        heatmap = (heatmap * 255).astype(np.uint8)
    else:
        heatmap = heatmap.astype(np.uint8)
    
    colored = cv2.applyColorMap(heatmap, colormap)
    return colored


def overlay_heatmap(frame: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.5) -> np.ndarray:
    """
    Overlay heatmap on frame
    
    Args:
        frame: Original frame
        heatmap: Heatmap to overlay
        alpha: Transparency (0-1)
        
    Returns:
        overlayed: Frame with heatmap overlay
    """
    # Ensure heatmap is colored
    if len(heatmap.shape) == 2:
        heatmap = apply_colormap(heatmap)
    
    # Resize heatmap if needed
    if heatmap.shape[:2] != frame.shape[:2]:
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    
    # Blend
    overlayed = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
    
    return overlayed


def count_vehicles_in_roi(detections: List[Dict], roi: List[int]) -> int:
    """
    Count vehicles in region of interest
    
    Args:
        detections: List of detection dictionaries
        roi: [x1, y1, x2, y2] region
        
    Returns:
        count: Number of vehicles in ROI
    """
    x1, y1, x2, y2 = roi
    count = 0
    
    for det in detections:
        cx, cy = det['center']
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            count += 1
    
    return count


def annotate_frame_with_info(frame: np.ndarray, info: Dict,
                             position: Tuple[int, int] = (10, 30)) -> np.ndarray:
    """
    Annotate frame with information text
    
    Args:
        frame: Input frame
        info: Dictionary of information to display
        position: Starting position (x, y)
        
    Returns:
        annotated: Annotated frame
    """
    annotated = frame.copy()
    x, y = position
    
    for key, value in info.items():
        text = f"{key}: {value}"
        cv2.putText(annotated, text, (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, text, (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y += 30
    
    return annotated


def extract_frames(video_path: str, output_dir: str, frame_interval: int = 1):
    """
    Extract frames from video
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        frame_interval: Extract every Nth frame
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            output_path = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    logger.info(f"Extracted {saved_count} frames from {video_path}")


def batch_process_images(image_paths: List[str], process_func, **kwargs):
    """
    Process multiple images with a function
    
    Args:
        image_paths: List of image file paths
        process_func: Function to apply to each image
        **kwargs: Additional arguments for process_func
        
    Returns:
        results: List of processing results
    """
    results = []
    
    for img_path in image_paths:
        try:
            image = cv2.imread(img_path)
            if image is None:
                logger.warning(f"Could not read image: {img_path}")
                results.append(None)
                continue
            
            result = process_func(image, **kwargs)
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            results.append(None)
    
    return results


if __name__ == "__main__":
    # Test utilities
    print("Testing CV utilities...")
    
    # Create test image
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Test zone drawing
    zones = [
        {'id': 0, 'name': 'Zone 1', 'bbox': [0, 0, 320, 240]},
        {'id': 1, 'name': 'Zone 2', 'bbox': [320, 0, 640, 240]}
    ]
    annotated = draw_zones(test_img, zones)
    print(f"Drew {len(zones)} zones")
    
    # Test heatmap
    detections = [
        {'center': [100, 100]},
        {'center': [200, 200]},
        {'center': [150, 150]}
    ]
    heatmap = create_heatmap(detections, test_img.shape)
    print(f"Created heatmap with max value: {heatmap.max():.4f}")
    
    # Test IoU
    box1 = [0, 0, 100, 100]
    box2 = [50, 50, 150, 150]
    iou = calculate_iou(box1, box2)
    print(f"IoU between overlapping boxes: {iou:.4f}")
    
    print("All tests passed!")