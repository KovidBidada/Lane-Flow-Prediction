"""
Vehicle Detection Module using YOLO
Detects vehicles in traffic images/videos with comprehensive features
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import yaml
from typing import List, Tuple, Dict, Optional
import logging
from collections import defaultdict
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VehicleDetector:
    """Advanced vehicle detection using YOLOv8"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize vehicle detector
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.detection_config = self.config['vehicle_detection']
        self.paths = self.config['paths']
        
        # Initialize model
        self.model = self._load_model()
        
        # Vehicle class IDs in COCO dataset
        # 2: car, 3: motorcycle, 5: bus, 7: truck
        self.target_classes = self.detection_config['target_classes']
        
        # Class names mapping
        self.class_names = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        
        # Statistics tracking
        self.detection_stats = defaultdict(int)
        
        logger.info(f"Vehicle detector initialized with {self.detection_config['model_type']}")
    
    def _load_model(self) -> YOLO:
        """Load YOLO model with error handling"""
        model_path = self.paths['yolo_weights']
        
        try:
            if Path(model_path).exists():
                logger.info(f"Loading custom weights from {model_path}")
                model = YOLO(model_path)
            else:
                logger.info(f"Loading pretrained {self.detection_config['model_type']}")
                model = YOLO(f"{self.detection_config['model_type']}.pt")
        except Exception as e:
            logger.warning(f"Error loading model: {e}. Using default yolov8n.pt")
            model = YOLO('yolov8n.pt')
        
        return model
    
    def detect_vehicles(
        self, 
        image: np.ndarray,
        return_annotated: bool = True,
        draw_boxes: bool = True
    ) -> Tuple[List[Dict], Optional[np.ndarray]]:
        """
        Detect vehicles in an image
        
        Args:
            image: Input image (BGR format)
            return_annotated: Whether to return annotated image
            draw_boxes: Whether to draw bounding boxes
            
        Returns:
            detections: List of detection dictionaries
            annotated_image: Image with bounding boxes (if return_annotated=True)
        """
        if image is None or image.size == 0:
            logger.error("Invalid image provided")
            return [], None
        
        # Run inference
        results = self.model(
            image,
            conf=self.detection_config['confidence_threshold'],
            iou=self.detection_config['iou_threshold'],
            device=self.detection_config['device'],
            verbose=False
        )
        
        detections = []
        annotated_image = image.copy() if return_annotated else None
        
        # Process results
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                cls = int(box.cls[0])
                
                # Filter for vehicle classes only
                if cls in self.target_classes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    
                    # Calculate center point
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # Calculate area
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': conf,
                        'class': cls,
                        'class_name': self.class_names.get(cls, 'vehicle'),
                        'center': (center_x, center_y),
                        'area': int(area),
                        'dimensions': (int(width), int(height))
                    }
                    detections.append(detection)
                    
                    # Update statistics
                    self.detection_stats[self.class_names.get(cls, 'vehicle')] += 1
                    
                    # Draw bounding box
                    if return_annotated and draw_boxes:
                        # Color coding by vehicle type
                        color_map = {
                            'car': (0, 255, 0),      # Green
                            'motorcycle': (255, 0, 0),  # Blue
                            'bus': (0, 165, 255),    # Orange
                            'truck': (0, 0, 255)     # Red
                        }
                        color = color_map.get(detection['class_name'], (0, 255, 0))
                        
                        # Draw rectangle
                        cv2.rectangle(
                            annotated_image,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            color,
                            2
                        )
                        
                        # Draw center point
                        cv2.circle(
                            annotated_image,
                            (center_x, center_y),
                            5,
                            color,
                            -1
                        )
                        
                        # Add label with background
                        label = f"{detection['class_name']} {conf:.2f}"
                        (label_w, label_h), baseline = cv2.getTextSize(
                            label,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            2
                        )
                        
                        # Draw label background
                        cv2.rectangle(
                            annotated_image,
                            (int(x1), int(y1) - label_h - 10),
                            (int(x1) + label_w + 5, int(y1)),
                            color,
                            -1
                        )
                        
                        # Draw label text
                        cv2.putText(
                            annotated_image,
                            label,
                            (int(x1) + 2, int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2
                        )
        
        # Sort detections by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        return detections, annotated_image
    
    def detect_video(
        self, 
        video_path: str, 
        output_path: str = None,
        show_progress: bool = True,
        skip_frames: int = 0
    ) -> List[Dict]:
        """
        Detect vehicles in video with frame skipping option
        
        Args:
            video_path: Path to input video
            output_path: Path to save annotated video
            show_progress: Whether to show progress bar
            skip_frames: Process every Nth frame (0 = all frames)
            
        Returns:
            frame_detections: List of detections per frame
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        frame_detections = []
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        # Setup video writer
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        processed_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if specified
            if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                frame_idx += 1
                continue
            
            # Detect vehicles
            detections, annotated = self.detect_vehicles(frame, return_annotated=bool(output_path))
            
            # Store detection info
            frame_info = {
                'frame': frame_idx,
                'timestamp': frame_idx / fps,
                'detections': detections,
                'vehicle_count': len(detections),
                'vehicle_types': self._count_vehicle_types(detections)
            }
            frame_detections.append(frame_info)
            
            # Write annotated frame
            if output_path and annotated is not None:
                # Add frame information overlay
                info_text = [
                    f"Frame: {frame_idx}/{total_frames}",
                    f"Vehicles: {len(detections)}",
                    f"Time: {frame_idx/fps:.1f}s"
                ]
                
                y_offset = 30
                for i, text in enumerate(info_text):
                    cv2.putText(
                        annotated,
                        text,
                        (10, y_offset + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2
                    )
                
                out.write(annotated)
            
            processed_frames += 1
            frame_idx += 1
            
            # Progress logging
            if show_progress and processed_frames % 100 == 0:
                progress = (frame_idx / total_frames) * 100
                logger.info(f"Processed {processed_frames} frames ({progress:.1f}%)")
        
        cap.release()
        if output_path:
            out.release()
        
        logger.info(f"Video processing complete. Processed {processed_frames} frames")
        
        return frame_detections
    
    def batch_detect_images(
        self, 
        image_dir: str, 
        output_dir: str = None,
        file_extensions: List[str] = None
    ) -> Dict[str, List[Dict]]:
        """
        Detect vehicles in a directory of images
        
        Args:
            image_dir: Directory containing images
            output_dir: Directory to save annotated images
            file_extensions: List of file extensions to process
            
        Returns:
            all_detections: Dictionary mapping image names to detections
        """
        if file_extensions is None:
            file_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        image_path = Path(image_dir)
        if not image_path.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        
        all_detections = {}
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Collect all image files
        image_files = []
        for ext in file_extensions:
            image_files.extend(image_path.glob(f'*{ext}'))
            image_files.extend(image_path.glob(f'*{ext.upper()}'))
        
        logger.info(f"Found {len(image_files)} images to process")
        
        for img_file in image_files:
            image = cv2.imread(str(img_file))
            if image is None:
                logger.warning(f"Could not read image: {img_file}")
                continue
            
            detections, annotated = self.detect_vehicles(
                image, 
                return_annotated=bool(output_dir)
            )
            
            all_detections[img_file.name] = {
                'detections': detections,
                'vehicle_count': len(detections),
                'vehicle_types': self._count_vehicle_types(detections)
            }
            
            if output_dir and annotated is not None:
                output_file = output_path / f"detected_{img_file.name}"
                cv2.imwrite(str(output_file), annotated)
        
        logger.info(f"Batch detection complete. Processed {len(all_detections)} images")
        
        return all_detections
    
    def get_vehicle_count(self, image: np.ndarray) -> int:
        """Quick vehicle count without annotation"""
        detections, _ = self.detect_vehicles(image, return_annotated=False)
        return len(detections)
    
    def _count_vehicle_types(self, detections: List[Dict]) -> Dict[str, int]:
        """Count vehicles by type"""
        type_counts = defaultdict(int)
        for det in detections:
            type_counts[det['class_name']] += 1
        return dict(type_counts)
    
    def get_detection_statistics(self) -> Dict[str, int]:
        """Get accumulated detection statistics"""
        return dict(self.detection_stats)
    
    def reset_statistics(self):
        """Reset detection statistics"""
        self.detection_stats.clear()
        logger.info("Detection statistics reset")
    
    def save_detections_to_json(self, detections: List[Dict], output_path: str):
        """Save detections to JSON file"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python types
        json_detections = []
        for det in detections:
            json_det = {
                'bbox': [int(x) for x in det['bbox']],
                'confidence': float(det['confidence']),
                'class': int(det['class']),
                'class_name': det['class_name'],
                'center': [int(x) for x in det['center']],
                'area': int(det['area']),
                'dimensions': [int(x) for x in det['dimensions']]
            }
            json_detections.append(json_det)
        
        with open(output_path, 'w') as f:
            json.dump(json_detections, f, indent=2)
        
        logger.info(f"Saved {len(json_detections)} detections to {output_path}")
    
    def detect_in_roi(
        self, 
        image: np.ndarray, 
        roi: Tuple[int, int, int, int]
    ) -> List[Dict]:
        """
        Detect vehicles only in specified region of interest
        
        Args:
            image: Input image
            roi: Region of interest (x1, y1, x2, y2)
            
        Returns:
            detections: Filtered detections within ROI
        """
        detections, _ = self.detect_vehicles(image, return_annotated=False)
        
        x1_roi, y1_roi, x2_roi, y2_roi = roi
        
        roi_detections = []
        for det in detections:
            center_x, center_y = det['center']
            
            # Check if center is within ROI
            if x1_roi <= center_x <= x2_roi and y1_roi <= center_y <= y2_roi:
                roi_detections.append(det)
        
        return roi_detections


if __name__ == "__main__":
    # Example usage
    detector = VehicleDetector()
    
    # Test on single image
    test_image_path = "data/raw/DETRAC-UPLOAD/images/train/sample.jpg"
    if Path(test_image_path).exists():
        image = cv2.imread(test_image_path)
        detections, annotated = detector.detect_vehicles(image)
        
        print(f"\nDetected {len(detections)} vehicles:")
        for i, det in enumerate(detections, 1):
            print(f"  {i}. {det['class_name']} (confidence: {det['confidence']:.2f})")
        
        # Save annotated image
        output_dir = Path("outputs/detections")
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / "test_detection.jpg"), annotated)
        print(f"\nAnnotated image saved to {output_dir / 'test_detection.jpg'}")
    else:
        print(f"Test image not found: {test_image_path}")
        print("Please ensure you have sample images in the correct location")