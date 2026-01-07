"""
Rule-Based Lane Control Decision Module
Makes lane allocation decisions based on traffic density rules
"""

import numpy as np
import yaml
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RuleBasedDecision:
    """Rule-based lane control system"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize rule-based decision engine"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.decision_config = self.config['decision']
        self.total_lanes = self.decision_config['total_lanes']
        self.min_lanes = self.decision_config['min_lanes_per_direction']
        self.thresholds = self.decision_config['thresholds']
        self.lane_configs = self.decision_config['lane_configs']
        
        logger.info(f"RuleBasedDecision initialized with {self.total_lanes} lanes")
    
    def calculate_traffic_ratio(self, inbound_density: float,
                                outbound_density: float) -> float:
        """
        Calculate traffic ratio between directions
        
        Args:
            inbound_density: Inbound traffic density (0-1)
            outbound_density: Outbound traffic density (0-1)
            
        Returns:
            ratio: Inbound to outbound ratio
        """
        total = inbound_density + outbound_density
        if total == 0:
            return 0.5  # Equal split if no traffic
        
        return inbound_density / total
    
    def determine_lane_allocation(self, inbound_density: float,
                                  outbound_density: float) -> Dict:
        """
        Determine optimal lane allocation based on traffic density
        
        Args:
            inbound_density: Inbound traffic density (0-1 normalized)
            outbound_density: Outbound traffic density (0-1 normalized)
            
        Returns:
            decision: Dictionary with lane allocation details
        """
        ratio = self.calculate_traffic_ratio(inbound_density, outbound_density)
        
        # Classify traffic levels
        inbound_level = self._classify_density(inbound_density)
        outbound_level = self._classify_density(outbound_density)
        
        # Determine lane split
        inbound_lanes, outbound_lanes, config_name = self._allocate_lanes(
            ratio, inbound_level, outbound_level
        )
        
        decision = {
            'inbound_lanes': inbound_lanes,
            'outbound_lanes': outbound_lanes,
            'config_name': config_name,
            'inbound_density': inbound_density,
            'outbound_density': outbound_density,
            'inbound_level': inbound_level,
            'outbound_level': outbound_level,
            'ratio': ratio,
            'total_lanes': self.total_lanes
        }
        
        return decision
    
    def _classify_density(self, density: float) -> str:
        """Classify density level"""
        if density >= self.thresholds['high_traffic']:
            return 'high'
        elif density >= self.thresholds['medium_traffic']:
            return 'medium'
        else:
            return 'low'
    
    def _allocate_lanes(self, ratio: float, inbound_level: str,
                       outbound_level: str) -> Tuple[int, int, str]:
        """
        Allocate lanes based on traffic ratio and levels
        
        Returns:
            (inbound_lanes, outbound_lanes, config_name)
        """
        # Emergency cases: one direction very high, other very low
        if inbound_level == 'high' and outbound_level == 'low':
            if ratio > 0.8:
                return self.total_lanes, 0, 'heavy_inbound'
            else:
                return 3, 1, 'inbound_priority'
        
        if outbound_level == 'high' and inbound_level == 'low':
            if ratio < 0.2:
                return 0, self.total_lanes, 'heavy_outbound'
            else:
                return 1, 3, 'outbound_priority'
        
        # Both high or both medium
        if inbound_level == 'high' and outbound_level == 'high':
            # Fair split when both are high
            return 2, 2, 'balanced'
        
        # Normal cases based on ratio
        if ratio > 0.7:
            return 3, 1, 'inbound_priority'
        elif ratio < 0.3:
            return 1, 3, 'outbound_priority'
        else:
            return 2, 2, 'balanced'
    
    def decide_with_prediction(self, current_inbound: float, current_outbound: float,
                              predicted_inbound: float, predicted_outbound: float,
                              prediction_weight: float = 0.3) -> Dict:
        """
        Make decision considering predicted future traffic
        
        Args:
            current_inbound: Current inbound density
            current_outbound: Current outbound density
            predicted_inbound: Predicted inbound density
            predicted_outbound: Predicted outbound density
            prediction_weight: Weight for prediction (0-1)
            
        Returns:
            decision: Lane allocation decision
        """
        # Weighted combination of current and predicted
        combined_inbound = (1 - prediction_weight) * current_inbound + \
                          prediction_weight * predicted_inbound
        combined_outbound = (1 - prediction_weight) * current_outbound + \
                           prediction_weight * predicted_outbound
        
        decision = self.determine_lane_allocation(combined_inbound, combined_outbound)
        decision['prediction_used'] = True
        decision['predicted_inbound'] = predicted_inbound
        decision['predicted_outbound'] = predicted_outbound
        
        return decision
    
    def decide_multi_zone(self, zone_densities: Dict[int, Dict]) -> List[Dict]:
        """
        Make decisions for multiple zones
        
        Args:
            zone_densities: Dictionary mapping zone_id to density info
            
        Returns:
            decisions: List of decisions for each zone
        """
        decisions = []
        
        for zone_id, zone_info in zone_densities.items():
            # Assume inbound/outbound split based on vehicle count
            total_vehicles = zone_info['vehicle_count']
            
            # Simple heuristic: split based on zone position
            # (in reality, would need direction detection)
            inbound_ratio = 0.6 if zone_id < 2 else 0.4
            
            inbound_density = zone_info['density'] * inbound_ratio
            outbound_density = zone_info['density'] * (1 - inbound_ratio)
            
            decision = self.determine_lane_allocation(inbound_density, outbound_density)
            decision['zone_id'] = zone_id
            decision['zone_name'] = zone_info['zone_name']
            
            decisions.append(decision)
        
        return decisions
    
    def get_lane_change_recommendation(self, current_config: Dict,
                                      new_config: Dict) -> Dict:
        """
        Generate lane change recommendation
        
        Args:
            current_config: Current lane configuration
            new_config: Recommended new configuration
            
        Returns:
            recommendation: Change details and justification
        """
        inbound_change = new_config['inbound_lanes'] - current_config.get('inbound_lanes', 2)
        outbound_change = new_config['outbound_lanes'] - current_config.get('outbound_lanes', 2)
        
        should_change = abs(inbound_change) > 0
        
        if not should_change:
            return {
                'should_change': False,
                'message': 'Current configuration is optimal'
            }
        
        # Generate change message
        if inbound_change > 0:
            action = f"Add {inbound_change} lane(s) to inbound direction"
        elif inbound_change < 0:
            action = f"Remove {abs(inbound_change)} lane(s) from inbound direction"
        else:
            action = "No change needed"
        
        justification = self._generate_justification(new_config)
        
        return {
            'should_change': should_change,
            'action': action,
            'inbound_change': inbound_change,
            'outbound_change': outbound_change,
            'new_inbound_lanes': new_config['inbound_lanes'],
            'new_outbound_lanes': new_config['outbound_lanes'],
            'justification': justification,
            'priority': self._calculate_priority(new_config)
        }
    
    def _generate_justification(self, config: Dict) -> str:
        """Generate human-readable justification"""
        inbound_level = config['inbound_level']
        outbound_level = config['outbound_level']
        ratio = config['ratio']
        
        justification = f"Traffic ratio is {ratio:.2f} (inbound/total). "
        justification += f"Inbound traffic: {inbound_level}, Outbound traffic: {outbound_level}. "
        
        if inbound_level == 'high' and outbound_level == 'low':
            justification += "Heavy inbound traffic requires more lanes."
        elif outbound_level == 'high' and inbound_level == 'low':
            justification += "Heavy outbound traffic requires more lanes."
        elif inbound_level == 'high' and outbound_level == 'high':
            justification += "Both directions busy, balanced allocation recommended."
        else:
            justification += "Traffic levels moderate, standard allocation sufficient."
        
        return justification
    
    def _calculate_priority(self, config: Dict) -> str:
        """Calculate change priority (low/medium/high)"""
        inbound_level = config['inbound_level']
        outbound_level = config['outbound_level']
        
        if inbound_level == 'high' or outbound_level == 'high':
            return 'high'
        elif inbound_level == 'medium' or outbound_level == 'medium':
            return 'medium'
        else:
            return 'low'


class AdaptiveRuleEngine:
    """Adaptive rule engine that learns from historical patterns"""
    
    def __init__(self, base_engine: RuleBasedDecision):
        """Initialize adaptive engine"""
        self.base_engine = base_engine
        self.decision_history = []
        self.performance_history = []
        
        logger.info("AdaptiveRuleEngine initialized")
    
    def record_decision(self, decision: Dict, actual_outcome: Dict):
        """Record decision and outcome for learning"""
        self.decision_history.append(decision)
        self.performance_history.append(actual_outcome)
    
    def adjust_thresholds(self):
        """Adjust thresholds based on historical performance"""
        if len(self.performance_history) < 10:
            return
        
        # Analyze last N decisions
        recent_performance = self.performance_history[-20:]
        
        # Calculate average congestion after decisions
        avg_congestion = np.mean([p.get('congestion_index', 0.5) 
                                  for p in recent_performance])
        
        # Adjust thresholds if performance is poor
        if avg_congestion > 0.7:
            logger.info("High congestion detected, adjusting thresholds")
            self.base_engine.thresholds['high_traffic'] *= 0.95
            self.base_engine.thresholds['medium_traffic'] *= 0.95
        elif avg_congestion < 0.3:
            logger.info("Low congestion detected, relaxing thresholds")
            self.base_engine.thresholds['high_traffic'] *= 1.05
            self.base_engine.thresholds['medium_traffic'] *= 1.05
    
    def get_statistics(self) -> Dict:
        """Get decision statistics"""
        if not self.decision_history:
            return {}
        
        config_counts = {}
        for decision in self.decision_history:
            config_name = decision['config_name']
            config_counts[config_name] = config_counts.get(config_name, 0) + 1
        
        return {
            'total_decisions': len(self.decision_history),
            'config_distribution': config_counts,
            'avg_inbound_density': np.mean([d['inbound_density'] 
                                           for d in self.decision_history]),
            'avg_outbound_density': np.mean([d['outbound_density'] 
                                            for d in self.decision_history])
        }


def main():
    """Demo function"""
    engine = RuleBasedDecision()
    
    # Test scenarios
    scenarios = [
        ("Morning rush (heavy inbound)", 0.8, 0.3),
        ("Evening rush (heavy outbound)", 0.3, 0.8),
        ("Balanced traffic", 0.5, 0.5),
        ("Light traffic", 0.2, 0.2),
        ("One-way surge", 0.9, 0.1)
    ]
    
    print("\n" + "="*60)
    print("RULE-BASED LANE CONTROL DECISIONS")
    print("="*60)
    
    for name, inbound, outbound in scenarios:
        decision = engine.determine_lane_allocation(inbound, outbound)
        
        print(f"\n{name}")
        print(f"  Inbound density: {inbound:.2f}, Outbound density: {outbound:.2f}")
        print(f"  → Allocation: {decision['inbound_lanes']} inbound, "
              f"{decision['outbound_lanes']} outbound")
        print(f"  → Configuration: {decision['config_name']}")
        print(f"  → Traffic levels: Inbound={decision['inbound_level']}, "
              f"Outbound={decision['outbound_level']}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()