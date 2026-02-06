import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Tuple

class RSRSCore:
    """
    RSRS (Resistance Support Relative Strength) Strategy.
    Based on OLS regression of High vs Low prices.
    """
    
    @staticmethod
    def calculate_rsrs(df: pd.DataFrame, n: int = 18, m: int = 600) -> dict:
        """
        Calculate RSRS, Beta, R2 and Standardized RSRS Score.
        
        Args:
            df: DataFrame with 'high' and 'low' columns.
            n: Regression period (days), default 18.
            m: Standardization window (days), default 600.
            
        Returns:
            dict containing latest 'beta', 'r2', 'rsrs_score'.
        """
        if len(df) < n + 1:
            return {'beta': 0, 'r2': 0, 'rsrs_score': 0}
            
        # We need to calculate Beta and R2 on a rolling basis to standardize them.
        # However, rolling OLS is slow.
        # For scanning current state, we only strictly need the *latest* Beta/R2, 
        # BUT we need history of Beta to calculate Z-Score.
        
        # Optimization:
        # 1. Calculate Betas for the last M days.
        # 2. Calculate Z-Score from these M Betas.
        
        # Work with arrays for speed
        highs = df['high'].values
        lows = df['low'].values
        
        # Limit processing to needed window (M + N) to save time
        if len(highs) > (m + n + 10):
            highs = highs[-(m + n + 10):]
            lows = lows[-(m + n + 10):]
            
        betas = []
        r2s = []
        
        # Rolling OLS - can be vectorized using stride_tricks? 
        # Simple loop for 'm' times likely acceptable for simple scan (~600 iters).
        # For full history backtest it's slow, but for "Scanner", we just need the end result.
        
        for i in range(len(highs) - n + 1):
            y = highs[i : i+n]
            x = lows[i : i+n]
            x = sm.add_constant(x)
            
            try:
                model = sm.OLS(y, x)
                results = model.fit()
                betas.append(results.params[1]) # slope matches Low
                r2s.append(results.rsquared)
            except:
                betas.append(0)
                r2s.append(0)
                
        if not betas:
            return {'beta': 0, 'r2': 0, 'rsrs_score': 0}

        # Latest values
        latest_beta = betas[-1]
        latest_r2 = r2s[-1]
        
        # Standardize Beta (RSRS Score)
        # using the last M betas (excluding current one? usually inclusive)
        recent_betas = np.array(betas[-m:])
        mean_beta = np.mean(recent_betas)
        std_beta = np.std(recent_betas)
        
        if std_beta == 0:
            z_score = 0
        else:
            z_score = (latest_beta - mean_beta) / std_beta
            
        # RSRS_Std = Z_Score * R2 (Weighted by R2)
        rsrs_std = z_score * latest_r2
        
        return {
            'beta': latest_beta,
            'r2': latest_r2,
            'z_score': z_score,
            'rsrs_score': rsrs_std
        }
