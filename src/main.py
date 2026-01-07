"""
main.py - Integrated Traffic Management System
Complete Production Version

Pipeline:
Vehicle Detection (YOLO) → Tracking → Density Estimation → 
Traffic Prediction (LSTM/GCN) → Intelligent Lane Decision (Rule-based + AI)

Author: Traffic Management System Team
Version: 1.0
Date: 2025
"""

import sys
import os
import argparse
import yaml
import logging
import time
import traceback
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

warnings.filterwarnings('ignore')

# Add src to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# Import vision modules
from .vision.vehicle_detection import VehicleDetector
from .vision.density_estimation import DensityEstimator
from .vision.tracking import VehicleTracker
from .vision.utils_cv import (
    draw_zones, draw_trajectory, annotate_frame_with_info, 
    create_heatmap, overlay_heatmap
)

# Import prediction modules
from .decision.visualize_decision import DecisionVisualizer
from .visualization.statistical_visualizer import StatisticalVisualizer
from .prediction.preprocess_data import DataPreprocessor
from .prediction.traffic_predictors import TrafficPredictor, GCNTrafficPredictor
from .prediction.evaluation import ModelEvaluator
from .prediction.utils_pred import (
    load_pickle, save_pickle, smooth_series, 
    calculate_metrics, create_sequences
)

# Import decision modules
from .decision.rule_based import RuleBasedDecision, AdaptiveRuleEngine
from .decision.ai_decision import AIDecisionEngine, LaneDecisionNetwork
from .decision.visualize_decision import DecisionVisualizer

# Configure logging
Path('outputs').mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class IntegratedTrafficManagementSystem:
    """
    Complete Integrated Traffic Management System
    
    Features:
    - Real-time vehicle detection and tracking (YOLO)
    - Multi-zone traffic density estimation
    - LSTM-based traffic flow prediction
    - Hybrid decision engine (rule-based + AI)
    - Comprehensive visualization and reporting
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the complete integrated system"""
        self._print_banner()
        logger.info("="*80)
        logger.info("INITIALIZING INTEGRATED TRAFFIC MANAGEMENT SYSTEM")
        logger.info("="*80)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Create directory structure
        self._create_directories()
        
        # Initialize all components
        self._initialize_all_components()
        
        # System state
        self.frame_count = 0
        self.current_decision = None
        self.decision_history = []
        self.traffic_history = []
        self.zones = None
        
        # Performance metrics
        self.performance_metrics = {
            'total_frames_processed': 0,
            'total_vehicles_detected': 0,
            'total_decisions_made': 0,
            'lane_switches': 0,
            'avg_processing_time': 0.0,
            'avg_fps': 0.0,
            'processing_times': []
        }
        
        logger.info("✓ System initialization complete!")
        logger.info("="*80 + "\n")
    
    def _print_banner(self):
        """Print system banner"""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║           INTEGRATED TRAFFIC MANAGEMENT SYSTEM v1.0                          ║
║                                                                              ║
║  Detection → Tracking → Density → Prediction → Decision                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"✓ Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"⚠ Config file not found: {config_path}, using defaults")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"✗ Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'paths': {
                'data_raw': 'data/raw/',
                'data_processed': 'data/processed/',
                'models': 'models/',
                'outputs': 'outputs/'
            },
            'detection': {
                'model_type': 'yolov8n',
                'confidence_threshold': 0.5,
                'iou_threshold': 0.45,
                'device': 'cpu',
                'vehicle_classes': [2, 3, 5, 7]
            },
            'tracking': {
                'max_disappeared': 50,
                'max_distance': 50
            },
            'density': {
                'roi_zones': 4,
                'density_levels': {'low': 10, 'medium': 25, 'high': 40},
                'smoothing_window': 5
            },
            'prediction': {
                'sequence_length': 12,
                'prediction_horizon': 12,
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.2,
                'epochs': 50,
                'batch_size': 32,
                'learning_rate': 0.001
            },
            'decision': {
                'mode': 'hybrid',
                'total_lanes': 4,
                'min_lanes_per_direction': 1,
                'decision_interval': 30,
                'ai_weight': 0.7,
                'thresholds': {
                    'high_traffic': 0.7,
                    'medium_traffic': 0.4,
                    'low_traffic': 0.2
                },
                'lane_configs': [
                    {'name': 'balanced', 'inbound': 2, 'outbound': 2},
                    {'name': 'inbound_priority', 'inbound': 3, 'outbound': 1},
                    {'name': 'outbound_priority', 'inbound': 1, 'outbound': 3}
                ]
            },
            'decision_engine': {
                'num_lanes': 4,
                'congestion_threshold': 70,
                'hysteresis': 0.1
            },
            'output_dirs': {
                'detections': 'outputs/detections',
                'density_graphs': 'outputs/density_graphs',
                'predictions': 'outputs/predictions',
                'decisions': 'outputs/decisions'
            },
            'visualization': {
                'plot_style': 'seaborn',
                'figure_size': [12, 8],
                'dpi': 100,
                'show_plots': False
            }
        }
    
    def _create_directories(self):
        """Create necessary output directories"""
        directories = [
            'data/raw/DETRAC-UPLOAD/images/train',
            'data/raw/DETRAC-UPLOAD/images/val',
            'data/raw/DETRAC-UPLOAD/labels/train',
            'data/processed',
            'models',
            'outputs/detections',
            'outputs/density_graphs',
            'outputs/predictions',
            'outputs/decisions',
            'outputs/videos',
            'outputs/reports'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("✓ Directory structure verified/created")
    
    def _initialize_all_components(self):
        """Initialize all system components"""
        logger.info("\nInitializing system components:")
        
        try:
            # 1. Vehicle Detection
            logger.info("  [1/9] Vehicle Detection Module...")
            self.vehicle_detector = VehicleDetector(config_path='config.yaml')
            
            # 2. Density Estimation
            logger.info("  [2/9] Density Estimation Module...")
            self.density_estimator = DensityEstimator(config_path='config.yaml')
            
            # 3. Vehicle Tracker
            logger.info("  [3/9] Vehicle Tracker...")
            self.vehicle_tracker = VehicleTracker(
                max_disappeared=self.config['tracking']['max_disappeared'],
                max_distance=self.config['tracking']['max_distance']
            )
            
            # 4. Data Preprocessor
            logger.info("  [4/9] Data Preprocessor...")
            self.data_preprocessor = DataPreprocessor(config_path='config.yaml')
            
            # 5. Traffic Predictor
            logger.info("  [5/9] Traffic Predictor (LSTM)...")
            self.traffic_predictor = TrafficPredictor(
                input_size=self.config['traffic_prediction']['input_size'],
                hidden_size=self.config['traffic_prediction'].get('hidden_size', 128),
                num_layers=self.config['traffic_prediction'].get('num_layers', 3),
                output_size=self.config['traffic_prediction'].get('output_size', 1),
                sequence_length=self.config['traffic_prediction'].get('sequence_length', 12),
                prediction_horizon=self.config['traffic_prediction'].get('prediction_horizon', 3),
                dropout=self.config['traffic_prediction'].get('dropout', 0.2),
                bidirectional=self.config['traffic_prediction'].get('bidirectional', True),
                use_attention=self.config['traffic_prediction'].get('use_attention', True)
                )
            
            # 6. Model Evaluator
            logger.info("  [6/9] Model Evaluator...")
            self.model_evaluator = ModelEvaluator(config_path='config.yaml')
            
            # 7. Rule-Based Decision Engine
            logger.info("  [7/9] Rule-Based Decision Engine...")
            self.rule_engine = RuleBasedDecision(config_path='config.yaml')
            self.adaptive_engine = AdaptiveRuleEngine(self.rule_engine)
            
            # 8. AI Decision Engine
            logger.info("  [8/9] AI Decision Engine...")
            self.ai_engine = AIDecisionEngine(config_path='config.yaml')
            
            # Try to load trained AI model
            try:
                model_path = Path(self.config['paths']['models']) / 'lane_decision_model.pth'
                if model_path.exists():
                    self.ai_engine.load_model(str(model_path))
                    logger.info("  ✓ Loaded trained AI decision model")
            except Exception as e:
                logger.info(f"  ℹ AI model not loaded: {e}")
            
            # 9. Decision Visualizer
            logger.info("  [9/10] Decision Visualizer...")
            self.visualizer = DecisionVisualizer(config_path='config.yaml')
            
            
            logger.info("\n✓ All components initialized successfully!")
            
        except Exception as e:
            logger.error(f"✗ Error initializing components: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def process_video(self, 
                     video_path: str, 
                     output_path: str = None,
                     decision_mode: str = 'rule',
                     show_live: bool = False) -> Dict:
        """
        Process video through complete pipeline
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            decision_mode: 'rule', 'ai', or 'hybrid'
            show_live: Display video while processing
            
        Returns:
            results: Processing results and metrics
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"PROCESSING VIDEO: {video_path}")
        logger.info(f"Decision Mode: {decision_mode.upper()}")
        logger.info(f"{'='*80}\n")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"✗ Could not open video: {video_path}")
            return None
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {total_frames} frames @ {fps} FPS ({width}x{height})")
        
        # Video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Define zones
        self.zones = self.density_estimator.define_zones((height, width))
        
        # Processing loop
        self.frame_count = 0
        decision_interval = self.config['decision']['decision_interval']
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame through pipeline
                processed_frame, frame_data = self._process_frame(
                    frame, decision_mode, decision_interval
                )
                
                # Write output
                if writer:
                    writer.write(processed_frame)
                
                # Display
                if show_live:
                    cv2.imshow('Traffic Management System', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("⚠ User requested quit")
                        break
                
                # Update metrics
                self.frame_count += 1
                self.performance_metrics['total_frames_processed'] = self.frame_count
                
                processing_time = time.time() - start_time
                self.performance_metrics['processing_times'].append(processing_time)
                
                # Progress update
                if self.frame_count % 100 == 0:
                    avg_time = np.mean(self.performance_metrics['processing_times'][-100:])
                    fps_actual = 1.0 / avg_time if avg_time > 0 else 0
                    progress = (self.frame_count / total_frames) * 100
                    logger.info(
                        f"Progress: {progress:.1f}% | "
                        f"Frames: {self.frame_count}/{total_frames} | "
                        f"FPS: {fps_actual:.1f}"
                    )
        
        except KeyboardInterrupt:
            logger.info("⚠ Processing interrupted by user")
        
        except Exception as e:
            logger.error(f"✗ Error during video processing: {e}")
            logger.error(traceback.format_exc())
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if show_live:
                cv2.destroyAllWindows()
        
        # Calculate final metrics
        if self.performance_metrics['processing_times']:
            self.performance_metrics['avg_processing_time'] = np.mean(
                self.performance_metrics['processing_times']
            )
            self.performance_metrics['avg_fps'] = 1.0 / self.performance_metrics['avg_processing_time']
        
        # Print summary
        self._print_processing_summary()
        
        # Generate report
        report_path = self._generate_session_report(video_path, output_path)
        
        return {
            'metrics': self.performance_metrics,
            'decision_history': self.decision_history,
            'report_path': report_path
        }
    
    def _process_frame(self, frame: np.ndarray, decision_mode: str,
                      decision_interval: int) -> Tuple[np.ndarray, Dict]:
        """Process a single frame through the pipeline"""
        
        # STEP 1: Detect vehicles
        annotated_frame, detections = self.vehicle_detector.detect_frame(frame)
        self.performance_metrics['total_vehicles_detected'] += len(detections)
        
        # STEP 2: Track vehicles
        self.vehicle_tracker.update(detections, self.frame_count)
        tracked_objects = self.vehicle_tracker.get_tracked_objects_with_bboxes()
        
        # STEP 3: Estimate density
        zone_densities = self.density_estimator.calculate_zone_density(
            detections, self.zones
        )
        zone_densities = self.density_estimator.smooth_density(zone_densities)
        
        # Add required metrics for AI engine
        for zone_id in zone_densities:
            zone_densities[zone_id]['occupancy_ratio'] = min(100,
                zone_densities[zone_id]['density'] * 2)
            zone_densities[zone_id]['congestion_score'] = min(100,
                zone_densities[zone_id]['vehicle_count'] * 2)
        
        # STEP 4: Predict traffic (simple moving average)
        predictions = self._predict_traffic(zone_densities)
        
        # STEP 5: Make decision (periodically)
        if self.frame_count % decision_interval == 0:
            decision = self._make_intelligent_decision(
                zone_densities, predictions, decision_mode
            )
            
            self.current_decision = decision
            self.performance_metrics['total_decisions_made'] += 1
            
            # Check for lane switch
            if len(self.decision_history) > 0:
                if self._has_lane_switched(self.decision_history[-1], decision):
                    self.performance_metrics['lane_switches'] += 1
            
            self.decision_history.append({
                'timestamp': datetime.now(),
                'frame': self.frame_count,
                'config': decision,
                'traffic_density': zone_densities,
                'predictions': predictions,
                'method': decision_mode
            })
            
            logger.info(
                f"Frame {self.frame_count}: {decision.get('config_name', 'N/A')} | "
                f"In: {decision.get('inbound_lanes', 0)} | "
                f"Out: {decision.get('outbound_lanes', 0)}"
            )
        
        # STEP 6: Annotate frame
        annotated_frame = self._annotate_frame(
            annotated_frame, detections, tracked_objects,
            zone_densities, self.current_decision
        )
        
        frame_data = {
            'frame_number': self.frame_count,
            'vehicle_count': len(detections),
            'zone_densities': zone_densities,
            'decision': self.current_decision
        }
        
        return annotated_frame, frame_data
    
    def _predict_traffic(self, zone_densities: Dict) -> Dict:
        """Simple traffic prediction based on recent history"""
        self.traffic_history.append(zone_densities)
        
        window = 10
        if len(self.traffic_history) > window:
            self.traffic_history = self.traffic_history[-window:]
        
        predictions = {}
        if len(self.traffic_history) >= 3:
            for zone_id in zone_densities.keys():
                recent = [h[zone_id]['density'] for h in self.traffic_history[-3:]]
                predictions[zone_id] = np.mean(recent) * 1.05
        else:
            predictions = {k: v['density'] for k, v in zone_densities.items()}
        
        return predictions
    
    def _make_intelligent_decision(self, zone_densities: Dict, 
                                   predictions: Dict,
                                   mode: str = 'rule') -> Dict:
        """Make intelligent lane allocation decision"""
        
        # Calculate aggregated metrics
        num_zones = len(zone_densities)
        inbound_zones = list(zone_densities.values())[:num_zones//2]
        outbound_zones = list(zone_densities.values())[num_zones//2:]
        
        inbound_density = np.mean([z.get('density', 0) for z in inbound_zones]) if inbound_zones else 0
        outbound_density = np.mean([z.get('density', 0) for z in outbound_zones]) if outbound_zones else 0
        
        # Normalize to 0-1 scale
        inbound_norm = min(inbound_density / 100.0, 1.0)
        outbound_norm = min(outbound_density / 100.0, 1.0)
        
        # Make decision based on mode
        try:
            if mode == 'ai':
                decision = self.ai_engine.make_decision(
                    zone_densities=zone_densities,
                    traffic_predictions=predictions.get(0) if predictions else None,
                    time_of_day=float(datetime.now().hour),
                    weather_factor=1.0,
                    use_model=False
                )
                # Convert AI decision format to standard format
                decision['inbound_lanes'] = decision.get('direction_a_lanes', 2)
                decision['outbound_lanes'] = decision.get('direction_b_lanes', 2)
                decision['config_name'] = f"ai_{decision['inbound_lanes']}i_{decision['outbound_lanes']}o"
                
            elif mode == 'hybrid':
                rule_decision = self.rule_engine.determine_lane_allocation(
                    inbound_norm, outbound_norm
                )
                ai_decision = self.ai_engine.make_decision(
                    zone_densities=zone_densities,
                    traffic_predictions=predictions.get(0) if predictions else None,
                    time_of_day=float(datetime.now().hour),
                    weather_factor=1.0,
                    use_model=False
                )
                
                # Combine decisions
                ai_weight = self.config['decision'].get('ai_weight', 0.7)
                rule_weight = 1.0 - ai_weight
                
                ai_inbound = ai_decision.get('direction_a_lanes', 2)
                rule_inbound = rule_decision.get('inbound_lanes', 2)
                
                inbound_lanes = int(ai_weight * ai_inbound + rule_weight * rule_inbound)
                outbound_lanes = self.config['decision']['total_lanes'] - inbound_lanes
                
                decision = {
                    'config_name': f'hybrid_{inbound_lanes}i_{outbound_lanes}o',
                    'inbound_lanes': inbound_lanes,
                    'outbound_lanes': outbound_lanes,
                    'ai_suggestion': ai_decision,
                    'rule_suggestion': rule_decision,
                    'method': 'hybrid'
                }
            else:  # rule-based
                decision = self.rule_engine.determine_lane_allocation(
                    inbound_norm, outbound_norm
                )
        
        except Exception as e:
            logger.error(f"✗ Error making decision: {e}")
            decision = {
                'config_name': 'balanced',
                'inbound_lanes': 2,
                'outbound_lanes': 2,
                'method': 'fallback'
            }
        
        # Add metadata
        decision['frame'] = self.frame_count
        decision['timestamp'] = datetime.now()
        decision['inbound_density'] = inbound_density
        decision['outbound_density'] = outbound_density
        decision['method'] = mode
        
        return decision
    
    def _has_lane_switched(self, prev_decision: Dict, current_decision: Dict) -> bool:
        """Check if lane configuration changed"""
        prev_config = prev_decision.get('config', prev_decision)
        curr_config = current_decision
        
        prev_in = prev_config.get('inbound_lanes', 2)
        curr_in = curr_config.get('inbound_lanes', 2)
        
        return prev_in != curr_in
    
    def _annotate_frame(self, frame: np.ndarray, detections: List,
                       tracked_objects: Dict, zone_densities: Dict,
                       decision: Optional[Dict]) -> np.ndarray:
        """Annotate frame with all information"""
        
        # Visualize zones with density
        annotated = self.density_estimator.visualize_zones(
            frame, self.zones, zone_densities
        )
        
        # Draw vehicle trajectories
        for obj_id in tracked_objects.keys():
            trajectory = self.vehicle_tracker.get_trajectory(obj_id)
            if trajectory and len(trajectory.get('centroids', [])) > 1:
                points = trajectory['centroids'][-10:]
                for i in range(len(points) - 1):
                    pt1 = tuple(map(int, points[i]))
                    pt2 = tuple(map(int, points[i + 1]))
                    cv2.line(annotated, pt1, pt2, (255, 255, 0), 2)
        
        # Add decision overlay
        if decision:
            self._draw_decision_overlay(annotated, decision, len(detections))
        
        # Add system info
        self._draw_system_info(annotated)
        
        return annotated
    
    def _draw_decision_overlay(self, frame: np.ndarray, decision: Dict, vehicle_count: int):
        """Draw decision information panel"""
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (420, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y_offset = 35
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Title
        cv2.putText(frame, "LANE CONTROL", (20, y_offset),
                   font, 0.7, (0, 255, 255), 2)
        y_offset += 30
        
        # Decision info
        info_lines = [
            f"Config: {decision.get('config_name', 'N/A')}",
            f"Method: {decision.get('method', 'N/A').upper()}",
            f"Inbound: {decision.get('inbound_lanes', 0)} lanes",
            f"Outbound: {decision.get('outbound_lanes', 0)} lanes",
            f"Vehicles: {vehicle_count}"
        ]
        
        for line in info_lines:
            cv2.putText(frame, line, (20, y_offset),
                       font, 0.55, (255, 255, 255), 2)
            y_offset += 25
    
    def _draw_system_info(self, frame: np.ndarray):
        """Draw system statistics"""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        fps_val = self.performance_metrics.get('avg_fps', 0)
        info_lines = [
            f"Frame: {self.frame_count}",
            f"FPS: {fps_val:.1f}",
            f"Total Vehicles: {self.performance_metrics['total_vehicles_detected']}",
            f"Decisions: {self.performance_metrics['total_decisions_made']}"
        ]
        
        y_offset = h - 100
        for line in info_lines:
            # Shadow
            cv2.putText(frame, line, (w - 198, y_offset),
                       font, 0.5, (0, 0, 0), 3)
            # Text
            cv2.putText(frame, line, (w - 200, y_offset),
                       font, 0.5, (0, 255, 0), 2)
            y_offset += 22
    
    def _print_processing_summary(self):
        """Print processing summary"""
        logger.info(f"\n{'='*80}")
        logger.info("VIDEO PROCESSING COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total frames: {self.performance_metrics['total_frames_processed']}")
        logger.info(f"Avg processing time: {self.performance_metrics['avg_processing_time']:.3f}s/frame")
        logger.info(f"Avg FPS: {self.performance_metrics['avg_fps']:.1f}")
        logger.info(f"Total vehicles: {self.performance_metrics['total_vehicles_detected']}")
        logger.info(f"Total decisions: {self.performance_metrics['total_decisions_made']}")
        logger.info(f"Lane switches: {self.performance_metrics['lane_switches']}")
        logger.info(f"{'='*80}\n")
    
    def _generate_session_report(self, video_path: str = None, 
                                output_path: str = None) -> str:
        """Generate comprehensive session report"""
        logger.info("Generating session report...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = Path("outputs/reports") / f"session_{timestamp}.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(report_path, 'w') as f:
                f.write("="*80 + "\n")
                f.write("INTEGRATED TRAFFIC MANAGEMENT SYSTEM - SESSION REPORT\n")
                f.write("="*80 + "\n\n")
                
                f.write(f"Session Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if video_path:
                    f.write(f"Input Video: {video_path}\n")
                if output_path:
                    f.write(f"Output Video: {output_path}\n")
                f.write("\n")
                
                f.write("-"*80 + "\n")
                f.write("PERFORMANCE METRICS\n")
                f.write("-"*80 + "\n")
                for key, value in self.performance_metrics.items():
                    if key != 'processing_times':
                        f.write(f"  {key.replace('_', ' ').title()}: {value}\n")
                
                f.write("\n" + "-"*80 + "\n")
                f.write("DECISION SUMMARY\n")
                f.write("-"*80 + "\n")
                f.write(f"  Total Decisions: {len(self.decision_history)}\n")
                
                if self.decision_history:
                    methods = [d.get('method', 'unknown') for d in self.decision_history]
                    f.write(f"  AI Decisions: {methods.count('ai')}\n")
                    f.write(f"  Rule-based Decisions: {methods.count('rule')}\n")
                    f.write(f"  Hybrid Decisions: {methods.count('hybrid')}\n")
                
                f.write("\n" + "="*80 + "\n")
            
            logger.info(f"✓ Session report saved to: {report_path}")
            
            # Generate visualization timeline
            if len(self.decision_history) > 0:
                try:
                    timeline_path = Path("outputs/decisions") / f"timeline_{timestamp}.html"
                    formatted_history = []
                    for d in self.decision_history:
                        decision = d['config']
                        formatted_history.append({
                            'timestamp': d['timestamp'],
                            'config': {
                                'lanes': (['inbound'] * decision.get('inbound_lanes', 2) +
                                        ['outbound'] * decision.get('outbound_lanes', 2))
                            },
                            'traffic_density': {
                                'inbound': decision.get('inbound_density', 0),
                                'outbound': decision.get('outbound_density', 0)
                            }
                        })
                    
                    self.visualizer.create_interactive_timeline(
                        formatted_history, save_path=str(timeline_path)
                    )
                except Exception as e:
                    logger.warning(f"⚠ Could not create timeline: {e}")
            
            return str(report_path)
        
        except Exception as e:
            logger.error(f"✗ Error generating session report: {e}")
            return None
    
    def train_prediction_model(self):
        """Train traffic prediction model"""
        logger.info("\n" + "="*80)
        logger.info("TRAINING PREDICTION MODEL")
        logger.info("="*80 + "\n")
        
        # Preprocess data
        try:
            data_dict = self.data_preprocessor.load_preprocessed_data()
            logger.info("✓ Loaded preprocessed data")
        except:
            logger.info("Preprocessing data...")
            try:
                data = self.data_preprocessor.preprocess_metr_la()
            except:
                logger.warning("⚠ Using synthetic data...")
                data = self.data_preprocessor.generate_synthetic_data()
            
            data_dict = self.data_preprocessor.create_train_test_split(data)
            self.data_preprocessor.save_preprocessed_data(data_dict)
        
        # Train
        logger.info("Training LSTM model...")
        train_losses, val_losses = self.traffic_predictor.train(
            data_dict['X_train'],
            data_dict['y_train'],
            data_dict['X_val'],
            data_dict['y_val']
        )
        
        # Evaluate
        logger.info("Evaluating model...")
        y_pred = self.traffic_predictor.predict(data_dict['X_test'])
        metrics = self.model_evaluator.evaluate_predictions(data_dict['y_test'], y_pred)
        
        logger.info("\nPrediction Model Metrics:")
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  MAE: {metrics['mae']:.4f}")
        logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        logger.info(f"  R²: {metrics.get('r2', 0):.4f}")
        
        # Generate report
        self.model_evaluator.generate_evaluation_report(
            metrics, data_dict['y_test'], y_pred, "prediction_model"
        )
        
        logger.info("\n✓ Prediction model training complete!")
        logger.info("="*80 + "\n")
    
    def train_ai_decision_model(self):
        """Train AI decision model"""
        logger.info("\n" + "="*80)
        logger.info("TRAINING AI DECISION MODEL")
        logger.info("="*80 + "\n")
        
        # Build model if not exists
        if self.ai_engine.model is None:
            self.ai_engine.build_model()
        
        # Generate training data
        logger.info("Generating training data...")
        training_data = []
        
        for _ in range(2000):
            zone_densities = {}
            for i in range(4):
                zone_densities[i] = {
                    'vehicle_count': np.random.randint(5, 50),
                    'density': np.random.uniform(10, 90),
                    'occupancy_ratio': np.random.uniform(10, 90),
                    'congestion_score': np.random.uniform(20, 90),
                    'zone_name': f'Zone_{i+1}'
                }
            
            features = self.ai_engine.extract_features(
                zone_densities,
                traffic_predictions=None,
                time_of_day=float(np.random.randint(0, 24)),
                weather_factor=1.0
            )
            
            avg_congestion = np.mean([z['congestion_score'] for z in zone_densities.values()])
            allocation = {i: {'direction': 'A' if i < 2 else 'B'} for i in range(4)}
            
            training_data.append((features, allocation))
        
        # Train
        logger.info("Training neural network...")
        self.ai_engine.train_model(training_data, epochs=100, batch_size=32)
        
        # Save model
        self.ai_engine.save_model()
        
        logger.info("\n✓ AI decision model training complete!")
        logger.info("="*80 + "\n")
    
    def run_demo(self, mode: str = 'synthetic'):
        """Run system demonstration"""
        logger.info(f"\n{'='*80}")
        logger.info(f"RUNNING DEMO: {mode.upper()}")
        logger.info(f"{'='*80}\n")
        
        try:
            if mode == 'synthetic':
                self._run_synthetic_demo()
            elif mode == 'image':
                self._run_image_demo()
            else:
                logger.error(f"✗ Unknown demo mode: {mode}")
        
        except Exception as e:
            logger.error(f"✗ Error running demo: {e}")
            logger.error(traceback.format_exc())
    
    def _run_synthetic_demo(self):
        """Run 24-hour synthetic traffic simulation"""
        logger.info("Running 24-hour synthetic traffic simulation...\n")
        
        for hour in range(24):
            # Simulate traffic patterns
            if 7 <= hour <= 9:  # Morning rush
                inbound = 0.8 + np.random.rand() * 0.1
                outbound = 0.3 + np.random.rand() * 0.1
            elif 17 <= hour <= 19:  # Evening rush
                inbound = 0.3 + np.random.rand() * 0.1
                outbound = 0.8 + np.random.rand() * 0.1
            else:  # Normal traffic
                inbound = 0.4 + np.random.rand() * 0.2
                outbound = 0.4 + np.random.rand() * 0.2
            
            # Make decision
            decision = self.rule_engine.determine_lane_allocation(inbound, outbound)
            
            # Log decision
            logger.info(
                f"Hour {hour:02d}:00 | {decision.get('config_name', 'Unknown'):20s} | "
                f"Inbound: {decision.get('inbound_lanes', 0)} | "
                f"Outbound: {decision.get('outbound_lanes', 0)} | "
                f"Traffic: {inbound:.2f}/{outbound:.2f}"
            )
        
        logger.info("\n✓ Demo complete!")
    
    def _run_image_demo(self):
        """Process sample images"""
        image_dir = Path("data/raw/DETRAC-UPLOAD/images/train")
        if not image_dir.exists():
            logger.warning(f"⚠ Image directory not found: {image_dir}")
            return
        
        image_paths = list(image_dir.glob("*.jpg"))[:5]
        
        if not image_paths:
            logger.warning("⚠ No images found in directory")
            return
        
        for img_path in image_paths:
            logger.info(f"\nProcessing {img_path.name}...")
            try:
                annotated, detections = self.vehicle_detector.detect_image(str(img_path))
                
                # Save result
                output_path = Path("outputs/detections") / img_path.name
                cv2.imwrite(str(output_path), annotated)
                
                logger.info(f"  ✓ Detected {len(detections)} vehicles")
                logger.info(f"  ✓ Saved to {output_path}")
            except Exception as e:
                logger.error(f"  ✗ Error processing {img_path.name}: {e}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Integrated Traffic Management System v1.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video with rule-based decisions
  python main.py --video traffic.mp4 --mode rule
  
  # Process video with AI decisions and live display
  python main.py --video traffic.mp4 --mode ai --show
  
  # Process video with hybrid decisions
  python main.py --video traffic.mp4 --mode hybrid --output result.mp4
  
  # Train all models
  python main.py --train all
  
  # Train only prediction model
  python main.py --train prediction
  
  # Run synthetic demo
  python main.py --demo synthetic
  
  # Run image demo
  python main.py --demo image
        """
    )
    
    # Main operations
    parser.add_argument('--video', type=str, 
                       help='Path to input video file')
    parser.add_argument('--output', type=str, 
                       help='Path to output video file')
    parser.add_argument('--demo', type=str, choices=['synthetic', 'image'],
                       help='Run demo mode')
    parser.add_argument('--train', type=str, choices=['all', 'prediction', 'decision'],
                       help='Train models')
    
    # Decision mode
    parser.add_argument('--mode', type=str, default='rule',
                       choices=['rule', 'ai', 'hybrid'],
                       help='Decision making mode (default: rule)')
    
    # Display options
    parser.add_argument('--show', action='store_true',
                       help='Display video while processing')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save output video')
    
    # Configuration
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file (default: config.yaml)')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()
    
    try:
        # Initialize system
        system = IntegratedTrafficManagementSystem(config_path=args.config)
        
        # Training mode
        if args.train:
            if args.train == 'all':
                system.train_prediction_model()
                system.train_ai_decision_model()
            elif args.train == 'prediction':
                system.train_prediction_model()
            elif args.train == 'decision':
                system.train_ai_decision_model()
            return 0
        
        # Demo mode
        if args.demo:
            system.run_demo(mode=args.demo)
            return 0
        
        # Video processing mode
        if args.video:
            # Determine output path
            if args.output:
                output_path = args.output
            elif not args.no_save:
                video_name = Path(args.video).stem
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f"outputs/videos/{video_name}_{args.mode}_{timestamp}.mp4"
            else:
                output_path = None
            
            # Process video
            results = system.process_video(
                video_path=args.video,
                output_path=output_path,
                decision_mode=args.mode,
                show_live=args.show
            )
            
            if results:
                logger.info("\n✓ Processing complete!")
                logger.info(f"  Report: {results['report_path']}")
                if output_path:
                    logger.info(f"  Video: {output_path}")
                return 0
            else:
                logger.error("\n✗ Processing failed!")
                return 1
        
        # No operation specified
        else:
            logger.error("✗ No operation specified!")
            logger.info("Run 'python main.py --help' for usage information")
            return 1
    
    except KeyboardInterrupt:
        logger.info("\n⚠ Interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"\n✗ Fatal error: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())