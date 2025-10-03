# fakeacc/heuristics.py
"""
Advanced heuristic analysis for wallet behavior patterns.
Provides detailed explanations and risk assessment based on transaction patterns.
"""

import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timedelta


class HeuristicAnalyzer:
    """
    Analyzes wallet features and provides detailed heuristic explanations.
    """
    
    def __init__(self):
        self.risk_thresholds = {
            'very_high_frequency': 100,  # tx per day
            'high_frequency': 20,
            'low_tx_count': 5,
            'very_low_tx_count': 2,
            'high_repeated_ratio': 0.7,
            'medium_repeated_ratio': 0.4,
            'low_entropy': 1.5,
            'very_low_entropy': 0.5,
            'high_weekend_activity': 0.4,
            'high_failed_ratio': 0.1,
            'high_contract_ratio': 0.8,
            'low_value_variance': 0.001,
        }
    
    def analyze_features(self, features: Dict, df: pd.DataFrame = None) -> Dict:
        """
        Perform comprehensive heuristic analysis on wallet features.
        
        Args:
            features: Dictionary of extracted wallet features
            df: Optional DataFrame of transactions for additional analysis
            
        Returns:
            Dictionary containing risk assessment and explanations
        """
        analysis = {
            'risk_score': 0.0,
            'risk_level': 'LOW',
            'flags': [],
            'explanations': [],
            'recommendations': [],
            'behavioral_patterns': []
        }
        
        # Analyze each feature category
        self._analyze_transaction_frequency(features, analysis)
        self._analyze_transaction_patterns(features, analysis)
        self._analyze_timing_patterns(features, analysis)
        self._analyze_interaction_patterns(features, analysis)
        self._analyze_value_patterns(features, analysis)
        
        # Additional analysis if transaction data is available
        if df is not None and not df.empty:
            self._analyze_transaction_data(df, analysis)
        
        # Calculate overall risk score
        analysis['risk_score'] = min(1.0, len(analysis['flags']) * 0.15)
        
        # Determine risk level
        if analysis['risk_score'] >= 0.7:
            analysis['risk_level'] = 'HIGH'
        elif analysis['risk_score'] >= 0.4:
            analysis['risk_level'] = 'MEDIUM'
        else:
            analysis['risk_level'] = 'LOW'
        
        return analysis
    
    def _analyze_transaction_frequency(self, features: Dict, analysis: Dict):
        """Analyze transaction frequency patterns."""
        tx_count = features.get('tx_count', 0)
        tx_freq = features.get('tx_freq_per_day', 0)
        lifetime = features.get('lifetime_days', 0)
        
        if tx_freq > self.risk_thresholds['very_high_frequency']:
            analysis['flags'].append('VERY_HIGH_FREQUENCY')
            analysis['explanations'].append(
                f"Extremely high transaction frequency ({tx_freq:.1f} tx/day) - "
                "typical of automated systems or bots"
            )
            analysis['behavioral_patterns'].append('bot_like_activity')
            
        elif tx_freq > self.risk_thresholds['high_frequency']:
            analysis['flags'].append('HIGH_FREQUENCY')
            analysis['explanations'].append(
                f"High transaction frequency ({tx_freq:.1f} tx/day) - "
                "may indicate automated trading or bot activity"
            )
            
        if tx_count < self.risk_thresholds['very_low_tx_count']:
            analysis['flags'].append('VERY_NEW_ACCOUNT')
            analysis['explanations'].append(
                f"Very few transactions ({tx_count}) - new or dormant account"
            )
            analysis['behavioral_patterns'].append('new_account')
            
        elif tx_count < self.risk_thresholds['low_tx_count'] and lifetime > 30:
            analysis['flags'].append('DORMANT_ACCOUNT')
            analysis['explanations'].append(
                f"Few transactions ({tx_count}) over {lifetime:.0f} days - "
                "dormant or inactive account"
            )
            analysis['behavioral_patterns'].append('dormant_account')
    
    def _analyze_transaction_patterns(self, features: Dict, analysis: Dict):
        """Analyze transaction behavioral patterns."""
        repeated_ratio = features.get('repeated_ratio', 0)
        unique_counterparties = features.get('unique_counterparties', 0)
        tx_count = features.get('tx_count', 0)
        
        if repeated_ratio > self.risk_thresholds['high_repeated_ratio']:
            analysis['flags'].append('HIGH_REPETITION')
            analysis['explanations'].append(
                f"High repetition ratio ({repeated_ratio:.2f}) - "
                "frequently interacts with the same addresses"
            )
            analysis['behavioral_patterns'].append('repetitive_behavior')
            
        elif repeated_ratio > self.risk_thresholds['medium_repeated_ratio']:
            analysis['explanations'].append(
                f"Moderate repetition ratio ({repeated_ratio:.2f}) - "
                "some repeated interactions with same addresses"
            )
        
        # Analyze counterparty diversity
        if tx_count > 10:
            diversity_ratio = unique_counterparties / tx_count
            if diversity_ratio < 0.1:
                analysis['flags'].append('LOW_DIVERSITY')
                analysis['explanations'].append(
                    f"Low counterparty diversity ({unique_counterparties} unique from {tx_count} tx) - "
                    "limited interaction scope"
                )
    
    def _analyze_timing_patterns(self, features: Dict, analysis: Dict):
        """Analyze timing and temporal patterns."""
        hour_entropy = features.get('hour_entropy', 0)
        weekend_activity = features.get('weekend_activity', 0)
        
        if hour_entropy < self.risk_thresholds['very_low_entropy']:
            analysis['flags'].append('VERY_REGULAR_TIMING')
            analysis['explanations'].append(
                f"Very low hour entropy ({hour_entropy:.2f}) - "
                "extremely regular transaction timing, typical of bots"
            )
            analysis['behavioral_patterns'].append('automated_timing')
            
        elif hour_entropy < self.risk_thresholds['low_entropy']:
            analysis['flags'].append('REGULAR_TIMING')
            analysis['explanations'].append(
                f"Low hour entropy ({hour_entropy:.2f}) - "
                "regular transaction timing patterns"
            )
        
        if weekend_activity > self.risk_thresholds['high_weekend_activity']:
            analysis['flags'].append('HIGH_WEEKEND_ACTIVITY')
            analysis['explanations'].append(
                f"High weekend activity ({weekend_activity:.2f}) - "
                "active during weekends, possible automated system"
            )
            analysis['behavioral_patterns'].append('24_7_activity')
    
    def _analyze_interaction_patterns(self, features: Dict, analysis: Dict):
        """Analyze contract and interaction patterns."""
        contract_ratio = features.get('contract_interaction_ratio', 0)
        failed_ratio = features.get('failed_tx_ratio', 0)
        
        if contract_ratio > self.risk_thresholds['high_contract_ratio']:
            analysis['flags'].append('HIGH_CONTRACT_INTERACTION')
            analysis['explanations'].append(
                f"High contract interaction ratio ({contract_ratio:.2f}) - "
                "primarily interacts with smart contracts"
            )
            analysis['behavioral_patterns'].append('contract_focused')
        
        if failed_ratio > self.risk_thresholds['high_failed_ratio']:
            analysis['flags'].append('HIGH_FAILURE_RATE')
            analysis['explanations'].append(
                f"High transaction failure rate ({failed_ratio:.2f}) - "
                "many failed transactions, possible bot testing or MEV activity"
            )
            analysis['behavioral_patterns'].append('high_failure_rate')
    
    def _analyze_value_patterns(self, features: Dict, analysis: Dict):
        """Analyze transaction value patterns."""
        value_variance = features.get('value_variance', 0)
        avg_value = features.get('avg_value_eth', 0)
        gas_efficiency = features.get('gas_efficiency', 0)
        
        if value_variance < self.risk_thresholds['low_value_variance'] and avg_value > 0:
            analysis['flags'].append('UNIFORM_VALUES')
            analysis['explanations'].append(
                f"Very low value variance ({value_variance:.6f}) - "
                "transactions use very similar amounts, typical of bots"
            )
            analysis['behavioral_patterns'].append('uniform_amounts')
        
        if avg_value < 0.001 and features.get('tx_count', 0) > 10:
            analysis['flags'].append('DUST_TRANSACTIONS')
            analysis['explanations'].append(
                f"Very small average transaction value ({avg_value:.6f} ETH) - "
                "dust transactions or micro-payments"
            )
            analysis['behavioral_patterns'].append('dust_activity')
    
    def _analyze_transaction_data(self, df: pd.DataFrame, analysis: Dict):
        """Perform additional analysis on raw transaction data."""
        if 'timeStamp' in df.columns:
            # Analyze time gaps between transactions
            df_sorted = df.sort_values('timeStamp')
            time_diffs = df_sorted['timeStamp'].diff().dt.total_seconds().dropna()
            
            if len(time_diffs) > 1:
                # Check for very regular intervals
                std_dev = time_diffs.std()
                mean_diff = time_diffs.mean()
                
                if std_dev < mean_diff * 0.1 and len(time_diffs) > 5:
                    analysis['flags'].append('REGULAR_INTERVALS')
                    analysis['explanations'].append(
                        f"Very regular transaction intervals (std: {std_dev:.0f}s) - "
                        "automated scheduling detected"
                    )
                    analysis['behavioral_patterns'].append('scheduled_transactions')
        
        # Analyze gas price patterns
        if 'gasPrice' in df.columns:
            gas_prices = df['gasPrice'].dropna()
            if len(gas_prices) > 5:
                gas_variance = gas_prices.var()
                if gas_variance < gas_prices.mean() * 0.01:
                    analysis['flags'].append('UNIFORM_GAS_PRICES')
                    analysis['explanations'].append(
                        "Very consistent gas prices - possible automated gas price setting"
                    )
    
    def generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        if 'VERY_HIGH_FREQUENCY' in analysis['flags']:
            recommendations.append(
                "⚠️ Monitor for bot activity - consider rate limiting or additional verification"
            )
        
        if 'VERY_NEW_ACCOUNT' in analysis['flags']:
            recommendations.append(
                "🔍 New account - implement additional KYC checks for high-value transactions"
            )
        
        if 'HIGH_REPETITION' in analysis['flags']:
            recommendations.append(
                "📊 High repetition detected - verify legitimate business relationship"
            )
        
        if 'VERY_REGULAR_TIMING' in analysis['flags']:
            recommendations.append(
                "🤖 Automated behavior detected - flag for manual review"
            )
        
        if 'HIGH_FAILURE_RATE' in analysis['flags']:
            recommendations.append(
                "⚡ High failure rate - possible MEV bot or testing activity"
            )
        
        if not analysis['flags']:
            recommendations.append("✅ No significant risk indicators detected")
        
        return recommendations


def analyze_wallet_heuristics(features: Dict, df: pd.DataFrame = None) -> Dict:
    """
    Convenience function to perform heuristic analysis.
    
    Args:
        features: Dictionary of wallet features
        df: Optional DataFrame of transactions
        
    Returns:
        Dictionary containing analysis results
    """
    analyzer = HeuristicAnalyzer()
    analysis = analyzer.analyze_features(features, df)
    analysis['recommendations'] = analyzer.generate_recommendations(analysis)
    return analysis
