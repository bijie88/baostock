import pandas as pd
import numpy as np

class ChipCore:
    """
    Chip Distribution (CYQ - Chou Ma Fen Bu) estimation based on Turnover Rate.
    """
    
    @staticmethod
    def calculate_chip_distribution(df: pd.DataFrame, decay_factor: float = 1.0) -> dict:
        """
        Estimate the cost distribution of chips.
        
        Logic:
        - Daily Turnover Rate (TR) means TR% of chips changed hands at today's average price.
        - (1 - TR%) of chips remained at their old costs.
        - We iterate through history to build a probability mass function of cost.
        
        Args:
            df: DataFrame containing 'close' (or 'avg_price' if avail) and 'turnover' (percentage 0-100 or 0-1).
            decay_factor: Adjuster for turnover (usually 1.0).
            
        Returns:
            dict: {
                'profit_ratio': float, # % of chips in profit (Cost < Current Price)
                'avg_cost': float,     # Average cost of all chips
                'concentration_90': float # Price range covering 90% of chips / avg_cost (width)
            }
        """
        if len(df) < 100:
            return {'profit_ratio': 0, 'avg_cost': 0, 'concentration_90': 1.0}

        # Need turnover as fraction (0.0 - 1.0)
        # Check if turnover is 0-100 or 0-1. AkShare usually returns % (e.g., 2.5 for 2.5%)
        # But we need to be careful.
        # Let's assume input 'turnover' is in percentage, so divide by 100.
        
        turnover = df['turnover'].values / 100.0 * decay_factor
        prices = df['close'].values # Use Close approximation
        
        # We model the chip distribution as a list of buckets or just arrays of (Price, Weight)
        # But iterating day by day and updating weights is better.
        # To avoid infinite growing arrays, we can stick to "Price Buckets" or "Recent N Days history".
        # Simplest accurate method: 
        # Chips[t] = Chips[t-1] * (1 - Turnover[t]) + NewChips(Price[t], Weight=Turnover[t])
        # Since sum(Weights) should always be 1 (100% of shares), 
        # Wait, if we start with 100% shares at Day 0 price.
        # Day 1: T% turnover. T% becomes Price[1], (1-T)% remains at Day 0 Price.
        # Day N: ...
        
        # For efficiency, we only look back e.g. 250 days (approx 1 year). Shares older than that are negligible if turnover is normal.
        
        history_len = min(len(df), 300)
        recent_prices = prices[-history_len:]
        recent_turnover = turnover[-history_len:]
        
        # Weights vector represents how much of the Total Capital is held at each day's price.
        # Initialize weights assuming all chips were bought at start of window? Or just evolve.
        
        # Vectorized simulation of decay:
        # Cost at Day i (Price[i]) has remaining weight:
        # Weight[i] = Turnover[i] * Product_{j=i+1 to End} (1 - Turnover[j])
        
        # The last day's turnover = shares bought today. They haven't decayed.
        # The day before last = bought then, decayed by today's turnover.
        
        # We can calculate these "Remaining Weights" for each past day.
        
        # 1 - Turnover
        retention = 1.0 - recent_turnover
        # Clip to [0, 1] to avoid errors
        retention = np.clip(retention, 0.0, 1.0)
        
        # Cumulative Retention from end to start (reverse cumprod)
        # cum_retention[k] = product(retention[k:]) -- wait
        # Weight[i] = Turnover[i] * Product(Retention[i+1:])
        
        # Let's compute suffix products
        suffix_prod = np.cumprod(retention[::-1])[::-1]
        # Shift suffix_prod to align: suffix for i starts at i+1
        # suffix_prod[i] includes retention[i], we want retention[i+1]...
        
        valid_weights = np.zeros(history_len)
        valid_weights[:-1] = recent_turnover[:-1] * suffix_prod[1:]
        valid_weights[-1] = recent_turnover[-1] # The last day
        
        # Normalize weights to sum to 1 (handling initial condition or truncation)
        current_sum = np.sum(valid_weights)
        if current_sum > 0:
            valid_weights /= current_sum
            
        # Now we have a distribution: Price -> Weight
        # Calculate Profit Ratio: Sum weights where Price < Current Price
        current_price = prices[-1]
        
        profit_mask = recent_prices < current_price
        profit_ratio = np.sum(valid_weights[profit_mask])
        
        # Average Cost
        avg_cost = np.sum(recent_prices * valid_weights)
        
        # Concentration (90%)
        # Sort by price
        sorted_indices = np.argsort(recent_prices)
        sorted_prices = recent_prices[sorted_indices]
        sorted_weights = valid_weights[sorted_indices]
        
        cum_weights = np.cumsum(sorted_weights)
        
        # Find price at 5% and 95%
        p05_idx = np.searchsorted(cum_weights, 0.05)
        p95_idx = np.searchsorted(cum_weights, 0.95)
        
        # Clamp indices
        p05_idx = min(p05_idx, len(sorted_prices)-1)
        p95_idx = min(p95_idx, len(sorted_prices)-1)
        
        p05 = sorted_prices[p05_idx]
        p95 = sorted_prices[p95_idx]
        
        concentration = (p95 - p05) / (p05 + p95) * 2 # Standard formula might vary
        # Or simply (p95 - p05) / avg_cost
        
        return {
            'profit_ratio': profit_ratio,
            'avg_cost': avg_cost,
            'concentration': concentration,
            'cost95': p95,
            'cost05': p05
        }
