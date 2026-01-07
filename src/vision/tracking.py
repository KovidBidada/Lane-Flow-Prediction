"""
Vehicle Tracking Module
Tracks vehicles across frames using simple centroid tracking
"""

import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VehicleTracker:
    """Simple centroid-based vehicle tracker"""
    
    def __init__(self, max_disappeared: int = 50, max_distance: int = 50):
        """
        Initialize tracker
        
        Args:
            max_disappeared: Max frames object can be lost before deregistration
            max_distance: Max distance for matching centroids
        """
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        # Track vehicle trajectories
        self.trajectories = OrderedDict()
        
        logger.info("Vehicle tracker initialized")
    
    def register(self, centroid: np.ndarray, bbox: list):
        """Register new object with unique ID"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.trajectories[self.next_object_id] = {
            'centroids': [centroid],
            'bboxes': [bbox],
            'frames': [0]
        }
        self.next_object_id += 1
    
    def deregister(self, object_id: int):
        """Remove object from tracking"""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, detections: list, frame_idx: int = 0):
        """
        Update tracker with new detections
        
        Args:
            detections: List of detection dicts with 'bbox' key
            frame_idx: Current frame number
            
        Returns:
            tracked_objects: Dict mapping object_id to centroid and bbox
        """
        # If no detections, mark all as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            return self.objects
        
        # Calculate centroids from bboxes
        input_centroids = np.zeros((len(detections), 2), dtype="int")
        bboxes = []
        
        for i, det in enumerate(detections):
            bbox = det['bbox']
            bboxes.append(bbox)
            cX = int((bbox[0] + bbox[2]) / 2.0)
            cY = int((bbox[1] + bbox[3]) / 2.0)
            input_centroids[i] = (cX, cY)
        
        # If no existing objects, register all
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], bboxes[i])
        
        else:
            # Get existing object IDs and centroids
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Compute distances between existing and new centroids
            D = dist.cdist(np.array(object_centroids), input_centroids)
            
            # Find minimum distance for each existing object
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            # Match existing objects to new detections
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                # Check if distance is within threshold
                if D[row, col] > self.max_distance:
                    continue
                
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                
                # Update trajectory
                self.trajectories[object_id]['centroids'].append(
                    input_centroids[col]
                )
                self.trajectories[object_id]['bboxes'].append(bboxes[col])
                self.trajectories[object_id]['frames'].append(frame_idx)
                
                used_rows.add(row)
                used_cols.add(col)
            
            # Handle unmatched existing objects
            unused_rows = set(range(D.shape[0])) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Register new detections
            unused_cols = set(range(D.shape[1])) - used_cols
            for col in unused_cols:
                self.register(input_centroids[col], bboxes[col])
        
        return self.objects
    
    def get_tracked_objects_with_bboxes(self):
        """Get current tracked objects with their bounding boxes"""
        tracked = {}
        for object_id in self.objects.keys():
            if object_id in self.trajectories:
                tracked[object_id] = {
                    'centroid': self.objects[object_id],
                    'bbox': self.trajectories[object_id]['bboxes'][-1]
                }
        return tracked
    
    def get_trajectory(self, object_id: int):
        """Get trajectory for specific object"""
        if object_id in self.trajectories:
            return self.trajectories[object_id]
        return None
    
    def get_all_trajectories(self):
        """Get all tracked trajectories"""
        return self.trajectories
    
    def calculate_speed(self, object_id: int, fps: float = 30.0):
        """
        Estimate speed of object (in pixels per second)
        
        Args:
            object_id: ID of tracked object
            fps: Frames per second
            
        Returns:
            speed: Estimated speed in pixels/second
        """
        if object_id not in self.trajectories:
            return 0.0
        
        traj = self.trajectories[object_id]
        centroids = traj['centroids']
        
        if len(centroids) < 2:
            return 0.0
        
        # Calculate distance between last two points
        last_centroid = np.array(centroids[-1])
        prev_centroid = np.array(centroids[-2])
        
        distance = np.linalg.norm(last_centroid - prev_centroid)
        speed = distance * fps
        
        return speed
    
    def count_vehicles_in_zone(self, zone_bbox: list):
        """
        Count vehicles currently in a zone
        
        Args:
            zone_bbox: [x1, y1, x2, y2] defining the zone
            
        Returns:
            count: Number of vehicles in zone
        """
        count = 0
        tracked = self.get_tracked_objects_with_bboxes()
        
        x1, y1, x2, y2 = zone_bbox
        
        for obj_id, obj_data in tracked.items():
            centroid = obj_data['centroid']
            
            if x1 <= centroid[0] <= x2 and y1 <= centroid[1] <= y2:
                count += 1
        
        return count
    
    def reset(self):
        """Reset tracker state"""
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.trajectories = OrderedDict()
        logger.info("Tracker reset")


if __name__ == "__main__":
    # Example usage
    tracker = VehicleTracker(max_disappeared=30, max_distance=50)
    
    # Simulate detections across frames
    frame1_detections = [
        {'bbox': [100, 100, 200, 200]},
        {'bbox': [300, 300, 400, 400]}
    ]
    
    frame2_detections = [
        {'bbox': [110, 105, 210, 205]},
        {'bbox': [310, 305, 410, 405]},
        {'bbox': [500, 500, 600, 600]}
    ]
    
    # Update tracker
    tracker.update(frame1_detections, frame_idx=0)
    print(f"Frame 0: {len(tracker.objects)} tracked objects")
    
    tracker.update(frame2_detections, frame_idx=1)
    print(f"Frame 1: {len(tracker.objects)} tracked objects")
    
    # Get trajectories
    for obj_id, traj in tracker.get_all_trajectories().items():
        print(f"Object {obj_id}: {len(traj['centroids'])} positions tracked")