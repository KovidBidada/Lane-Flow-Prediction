"""
AI-Based Lane Decision Module
Uses ML/DL techniques for intelligent lane allocation based on traffic patterns
"""

import numpy as np
import torch
import torch.nn as nn
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from collections import deque
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LaneDecisionNetwork(nn.Module):
    """Neural network for lane allocation decisions"""
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [128, 64, 32],
        num_lanes: int = 4,
        dropout: float = 0.2
    ):
        """
        Initialize lane decision network
        
        Args:
            input_size: Input feature dimension
            hidden_sizes: List of hidden layer sizes
            num_lanes: Number of lanes to allocate
            dropout: Dropout rate
        """
        super(LaneDecisionNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Output layer - softmax over possible lane configurations
        self.feature_extractor = nn.Sequential(*layers)
        self.output = nn.Linear(prev_size, num_lanes * 2)  # Allocation for each direction
        
        self.num_lanes = num_lanes
    
    def forward(self, x):
        """Forward pass"""
        features = self.feature_extractor(x)
        output = self.output(features)
        
        # Reshape to (batch, num_lanes, 2) for bidirectional allocation
        output = output.view(-1, self.num_lanes, 2)
        
        # Apply softmax on directions
        output = torch.softmax(output, dim=-1)
        
        return output


class AIDecisionEngine:
    """AI-based intelligent lane control decision engine"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize AI decision engine
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.decision_config = self.config['decision_engine']
        self.paths = self.config['paths']
        
        self.num_lanes = self.decision_config['num_lanes']
        self.congestion_threshold = self.decision_config['congestion_threshold']
        self.hysteresis = self.decision_config['hysteresis']
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model
        self.model = None
        
        # State tracking
        self.previous_allocation = None
        self.allocation_history = deque(maxlen=10)
        self.state_history = deque(maxlen=50)
        
        # Decision statistics
        self.decision_count = 0
        self.lane_switches = 0
        
        logger.info(f"AI Decision Engine initialized on {self.device}")
    
    def build_model(self, input_size: int = 20):
        """
        Build decision network
        
        Args:
            input_size: Size of input features
        """
        self.model = LaneDecisionNetwork(
            input_size=input_size,
            hidden_sizes=[128, 64, 32],
            num_lanes=self.num_lanes,
            dropout=0.2
        ).to(self.device)
        
        logger.info(f"Built decision network with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def extract_features(
        self,
        zone_densities: Dict[int, Dict],
        traffic_predictions: Optional[np.ndarray] = None,
        time_of_day: Optional[float] = None,
        weather_factor: float = 1.0
    ) -> np.ndarray:
        """
        Extract features for decision making
        
        Args:
            zone_densities: Current zone density metrics
            traffic_predictions: Predicted future traffic
            time_of_day: Time in hours (0-24)
            weather_factor: Weather impact factor (0.5-1.5)
            
        Returns:
            features: Feature vector for model input
        """
        features = []
        
        # Zone-based features
        for zone_id in sorted(zone_densities.keys()):
            zone_data = zone_densities[zone_id]
            features.extend([
                zone_data['vehicle_count'] / 50.0,  # Normalized count
                zone_data['density'] / 100.0,        # Normalized density
                zone_data['occupancy_ratio'] / 100.0,
                zone_data['congestion_score'] / 100.0
            ])
        
        # Prediction features (if available)
        if traffic_predictions is not None:
            pred_mean = np.mean(traffic_predictions)
            pred_std = np.std(traffic_predictions)
            pred_trend = traffic_predictions[-1] - traffic_predictions[0]
            features.extend([pred_mean / 50.0, pred_std / 20.0, pred_trend / 30.0])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Temporal features
        if time_of_day is not None:
            # Encode time cyclically
            hour_sin = np.sin(2 * np.pi * time_of_day / 24)
            hour_cos = np.cos(2 * np.pi * time_of_day / 24)
            features.extend([hour_sin, hour_cos])
        else:
            features.extend([0.0, 0.0])
        
        # Environmental features
        features.append(weather_factor)
        
        # Pad or trim to fixed size
        target_size = 20
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]
        
        return np.array(features, dtype=np.float32)
    
    def make_decision(
        self,
        zone_densities: Dict[int, Dict],
        traffic_predictions: Optional[np.ndarray] = None,
        time_of_day: Optional[float] = None,
        weather_factor: float = 1.0,
        use_model: bool = True
    ) -> Dict[str, any]:
        """
        Make lane allocation decision
        
        Args:
            zone_densities: Current zone densities
            traffic_predictions: Future traffic predictions
            time_of_day: Current time
            weather_factor: Weather impact
            use_model: Whether to use neural network (vs. heuristic)
            
        Returns:
            decision: Dictionary with allocation and metadata
        """
        self.decision_count += 1
        
        if use_model and self.model is not None:
            decision = self._model_based_decision(
                zone_densities, traffic_predictions, time_of_day, weather_factor
            )
        else:
            decision = self._heuristic_decision(zone_densities)
        
        # Apply hysteresis to prevent frequent switching
        decision = self._apply_hysteresis(decision)
        
        # Track history
        self.allocation_history.append(decision['allocation'])
        self.state_history.append({
            'zone_densities': zone_densities,
            'allocation': decision['allocation'],
            'timestamp': self.decision_count
        })
        
        # Update previous allocation
        if self.previous_allocation is not None:
            if decision['allocation'] != self.previous_allocation:
                self.lane_switches += 1
                decision['lane_switched'] = True
        
        self.previous_allocation = decision['allocation'].copy()
        
        return decision
    
    def _model_based_decision(
        self,
        zone_densities: Dict[int, Dict],
        traffic_predictions: Optional[np.ndarray],
        time_of_day: Optional[float],
        weather_factor: float
    ) -> Dict[str, any]:
        """Make decision using neural network"""
        # Extract features
        features = self.extract_features(
            zone_densities, traffic_predictions, time_of_day, weather_factor
        )
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            output = self.model(features_tensor)
        
        # Get allocation probabilities
        allocation_probs = output[0].cpu().numpy()  # Shape: (num_lanes, 2)
        
        # Determine allocation (0: direction A, 1: direction B)
        allocation = {}
        total_vehicles_a = 0
        total_vehicles_b = 0
        
        for zone_id in sorted(zone_densities.keys()):
            # Assign based on probability
            prob_a = allocation_probs[zone_id][0]
            prob_b = allocation_probs[zone_id][1]
            
            # Consider current congestion
            congestion = zone_densities[zone_id]['congestion_score']
            
            # Weighted decision
            if congestion > 70:
                # High congestion: favor balanced allocation
                direction = 'A' if prob_a > 0.45 else 'B'
            else:
                # Normal: use model prediction
                direction = 'A' if prob_a > prob_b else 'B'
            
            allocation[zone_id] = {
                'direction': direction,
                'confidence': max(prob_a, prob_b),
                'zone_name': zone_densities[zone_id]['zone_name']
            }
            
            if direction == 'A':
                total_vehicles_a += zone_densities[zone_id]['vehicle_count']
            else:
                total_vehicles_b += zone_densities[zone_id]['vehicle_count']
        
        decision = {
            'allocation': allocation,
            'method': 'ai_model',
            'direction_a_lanes': sum(1 for v in allocation.values() if v['direction'] == 'A'),
            'direction_b_lanes': sum(1 for v in allocation.values() if v['direction'] == 'B'),
            'direction_a_vehicles': total_vehicles_a,
            'direction_b_vehicles': total_vehicles_b,
            'balance_score': self._calculate_balance_score(allocation, zone_densities),
            'lane_switched': False
        }
        
        return decision
    
    def _heuristic_decision(
        self,
        zone_densities: Dict[int, Dict]
    ) -> Dict[str, any]:
        """Make decision using heuristic rules"""
        allocation = {}
        
        # Calculate total congestion per direction assumption
        zones = sorted(zone_densities.keys())
        midpoint = len(zones) // 2
        
        # Assume first half = Direction A, second half = Direction B
        direction_a_congestion = np.mean([
            zone_densities[z]['congestion_score'] for z in zones[:midpoint]
        ])
        direction_b_congestion = np.mean([
            zone_densities[z]['congestion_score'] for z in zones[midpoint:]
        ])
        
        # Calculate imbalance
        imbalance = abs(direction_a_congestion - direction_b_congestion)
        
        total_vehicles_a = 0
        total_vehicles_b = 0
        
        if imbalance > 30:
            # Significant imbalance: allocate more lanes to congested direction
            if direction_a_congestion > direction_b_congestion:
                # More lanes for A
                lanes_a = min(self.num_lanes - 1, int(self.num_lanes * 0.75))
                lanes_b = self.num_lanes - lanes_a
            else:
                # More lanes for B
                lanes_b = min(self.num_lanes - 1, int(self.num_lanes * 0.75))
                lanes_a = self.num_lanes - lanes_b
            
            # Assign lanes
            for i, zone_id in enumerate(zones):
                if i < lanes_a:
                    direction = 'A'
                    total_vehicles_a += zone_densities[zone_id]['vehicle_count']
                else:
                    direction = 'B'
                    total_vehicles_b += zone_densities[zone_id]['vehicle_count']
                
                allocation[zone_id] = {
                    'direction': direction,
                    'confidence': 1.0 - (imbalance / 100.0),
                    'zone_name': zone_densities[zone_id]['zone_name']
                }
        else:
            # Balanced: equal allocation
            lanes_per_direction = self.num_lanes // 2
            
            for i, zone_id in enumerate(zones):
                if i < lanes_per_direction:
                    direction = 'A'
                    total_vehicles_a += zone_densities[zone_id]['vehicle_count']
                else:
                    direction = 'B'
                    total_vehicles_b += zone_densities[zone_id]['vehicle_count']
                
                allocation[zone_id] = {
                    'direction': direction,
                    'confidence': 0.9,
                    'zone_name': zone_densities[zone_id]['zone_name']
                }
        
        decision = {
            'allocation': allocation,
            'method': 'heuristic',
            'direction_a_lanes': sum(1 for v in allocation.values() if v['direction'] == 'A'),
            'direction_b_lanes': sum(1 for v in allocation.values() if v['direction'] == 'B'),
            'direction_a_vehicles': total_vehicles_a,
            'direction_b_vehicles': total_vehicles_b,
            'balance_score': self._calculate_balance_score(allocation, zone_densities),
            'lane_switched': False,
            'imbalance': imbalance
        }
        
        return decision
    
    def _apply_hysteresis(self, decision: Dict[str, any]) -> Dict[str, any]:
        """Apply hysteresis to prevent frequent lane switching"""
        if self.previous_allocation is None:
            return decision
        
        # Check if allocation changed
        changed_zones = []
        for zone_id, alloc in decision['allocation'].items():
            if zone_id in self.previous_allocation:
                if alloc['direction'] != self.previous_allocation[zone_id]['direction']:
                    changed_zones.append(zone_id)
        
        # If change is minor and confidence is low, revert
        if len(changed_zones) <= 1 and len(changed_zones) > 0:
            avg_confidence = np.mean([
                decision['allocation'][z]['confidence'] for z in changed_zones
            ])
            
            if avg_confidence < (0.5 + self.hysteresis):
                # Revert to previous allocation
                logger.info(f"Hysteresis applied: reverting {len(changed_zones)} zone changes")
                for zone_id in changed_zones:
                    decision['allocation'][zone_id] = self.previous_allocation[zone_id].copy()
                
                # Recalculate metrics
                decision['direction_a_lanes'] = sum(
                    1 for v in decision['allocation'].values() if v['direction'] == 'A'
                )
                decision['direction_b_lanes'] = sum(
                    1 for v in decision['allocation'].values() if v['direction'] == 'B'
                )
        
        return decision
    
    def _calculate_balance_score(
        self,
        allocation: Dict[int, Dict],
        zone_densities: Dict[int, Dict]
    ) -> float:
        """Calculate how balanced the allocation is (0-100)"""
        vehicles_a = sum(
            zone_densities[z]['vehicle_count']
            for z, alloc in allocation.items() if alloc['direction'] == 'A'
        )
        vehicles_b = sum(
            zone_densities[z]['vehicle_count']
            for z, alloc in allocation.items() if alloc['direction'] == 'B'
        )
        
        total = vehicles_a + vehicles_b
        if total == 0:
            return 100.0
        
        # Perfect balance = 50/50
        ratio = min(vehicles_a, vehicles_b) / total
        balance_score = (ratio / 0.5) * 100
        
        return min(100.0, balance_score)
    
    def train_model(
        self,
        training_data: List[Tuple[np.ndarray, Dict]],
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ):
        """
        Train the decision network
        
        Args:
            training_data: List of (features, optimal_allocation) tuples
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        if self.model is None:
            self.build_model()
        
        # Prepare data
        X = np.array([item[0] for item in training_data])
        y = np.array([self._allocation_to_tensor(item[1]) for item in training_data])
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                # Calculate loss for each lane
                loss = 0
                for lane_idx in range(self.num_lanes):
                    lane_output = outputs[:, lane_idx, :]
                    lane_target = batch_y[:, lane_idx].long()
                    loss += criterion(lane_output, lane_target)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / (len(X_tensor) / batch_size)
                logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        logger.info("Model training complete")
    
    def _allocation_to_tensor(self, allocation: Dict) -> np.ndarray:
        """Convert allocation dictionary to tensor format"""
        tensor = np.zeros(self.num_lanes)
        for zone_id, alloc in allocation.items():
            tensor[zone_id] = 0 if alloc['direction'] == 'A' else 1
        return tensor
    
    def save_model(self, path: str = None):
        """Save trained model"""
        if path is None:
            path = str(Path(self.paths['models_dir']) / 'lane_decision_model.pth')
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.decision_config,
            'decision_count': self.decision_count,
            'lane_switches': self.lane_switches
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str = None):
        """Load trained model"""
        if path is None:
            path = str(Path(self.paths['models_dir']) / 'lane_decision_model.pth')
        
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        if self.model is None:
            self.build_model()
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info(f"Model loaded from {path}")
    
    def get_statistics(self) -> Dict[str, any]:
        """Get decision engine statistics"""
        return {
            'total_decisions': self.decision_count,
            'lane_switches': self.lane_switches,
            'switch_rate': self.lane_switches / max(1, self.decision_count),
            'history_length': len(self.allocation_history)
        }


if __name__ == "__main__":
    # Example usage
    engine = AIDecisionEngine()
    
    # Sample zone densities
    zone_densities = {
        0: {'vehicle_count': 15, 'density': 25, 'occupancy_ratio': 30, 
            'congestion_score': 45, 'zone_name': 'Lane_1'},
        1: {'vehicle_count': 25, 'density': 40, 'occupancy_ratio': 50,
            'congestion_score': 65, 'zone_name': 'Lane_2'},
        2: {'vehicle_count': 10, 'density': 18, 'occupancy_ratio': 20,
            'congestion_score': 30, 'zone_name': 'Lane_3'},
        3: {'vehicle_count': 20, 'density': 35, 'occupancy_ratio': 45,
            'congestion_score': 55, 'zone_name': 'Lane_4'}
    }
    
    # Make decision (heuristic mode)
    decision = engine.make_decision(zone_densities, use_model=False)
    
    print("\n" + "="*60)
    print("LANE ALLOCATION DECISION")
    print("="*60)
    print(f"Method: {decision['method']}")
    print(f"Direction A Lanes: {decision['direction_a_lanes']}")
    print(f"Direction B Lanes: {decision['direction_b_lanes']}")
    print(f"Balance Score: {decision['balance_score']:.1f}")
    
    print("\nAllocation Details:")
    for zone_id, alloc in decision['allocation'].items():
        print(f"  {alloc['zone_name']}: Direction {alloc['direction']} "
              f"(confidence: {alloc['confidence']:.2f})")
    print("="*60)
    
    
    
    
    # decision/hybrid_decision.py
class HybridDecisionEngine:
    def __init__(self, rule_engine, ai_engine, ai_weight=0.7):
        self.rule_engine = rule_engine
        self.ai_engine = ai_engine
        self.ai_weight = ai_weight

    def decide(self, traffic_state):
        rule_decision = self.rule_engine.determine_lane_allocation(
            traffic_state['inbound_density'],
            traffic_state['outbound_density']
        )
        ai_decision = self.ai_engine.make_decision(
            zone_densities=traffic_state['zone_densities'],
            use_model=True
        )

        # Simple weighted combination
        combined = {}
        for zone_id in ai_decision['allocation']:
            if ai_decision['allocation'][zone_id]['direction'] == 'A':
                combined_direction = 'A' if np.random.rand() < self.ai_weight else 'B'
            else:
                combined_direction = 'B' if np.random.rand() < self.ai_weight else 'A'

            combined[zone_id] = {
                'direction': combined_direction,
                'confidence': ai_decision['allocation'][zone_id]['confidence'],
                'zone_name': ai_decision['allocation'][zone_id]['zone_name']
            }
        return {'allocation': combined}
