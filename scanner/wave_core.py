import pandas as pd
import numpy as np

class WaveCore:
    """
    Elliott Wave helper based on Awesome Oscillator (AO).
    AO = SMA(5) - SMA(34) (Median Price).
    """
    
    @staticmethod
    def calculate_ao(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Median Price
        df['mp'] = (df['high'] + df['low']) / 2
        df['sma5'] = df['mp'].rolling(window=5).mean()
        df['sma34'] = df['mp'].rolling(window=34).mean()
        df['ao'] = df['sma5'] - df['sma34']
        return df

    @staticmethod
    def check_wave_structure(df: pd.DataFrame) -> dict:
        """
        Analyze current wave status using AO.
        
        Wave 3 identification:
        - Usually corresponds to the highest peak (or lowest trough) in AO.
        - Strong momentum.
        
        Wave 4:
        - AO retraces towards zero line (often crosses it slightly or touches it).
        
        Wave 5:
        - Price makes new High, but AO makes Lower High (Divergence).
        """
        if len(df) < 50:
            return {'status': 'Unknown', 'desc': 'Not enough data'}
            
        if 'ao' not in df.columns:
            df = WaveCore.calculate_ao(df)
            
        # Get last few AO values
        last_ao = df['ao'].iloc[-1]
        prev_ao = df['ao'].iloc[-2]
        
        # Check for Zero Cross (Transition)
        zero_cross_up = (prev_ao < 0) and (last_ao > 0)
        
        # Find local peaks of AO in recent window (e.g., 50 bars)
        recent_window = df.iloc[-50:]
        max_ao = recent_window['ao'].max()
        min_ao = recent_window['ao'].min()
        
        res = {'status': 'Neutral', 'desc': '', 'is_wave3': False, 'divergence': False}
        
        # 1. Potential Wave 3 (Strong Momentum)
        # If current AO is very positive and close to recent Max.
        if last_ao > 0 and last_ao > (max_ao * 0.8):
            res['status'] = 'Bullish'
            res['is_wave3'] = True
            res['desc'] += "Strong AO Momentum (Potential Wave 3). "
            
        # 2. Divergence (Potential Wave 5 Top)
        # Price High > Prev Price High, but AO High < Prev AO High
        # Need peak detection logic. Simplified:
        # Just check if Price is Max in window, but AO is NOT Max in window.
        
        price_max = recent_window['close'].max()
        curr_price = df['close'].iloc[-1]
        
        if (curr_price >= price_max * 0.98) and (last_ao < max_ao * 0.7) and (last_ao > 0):
             res['divergence'] = True
             res['desc'] += "Bearish Divergence (Price High but AO Weak). "

        # 3. Wave 4 Pullback
        # AO was strong positive, now retracing near zero.
        # Hard to code simply without peak tracking.
        
        return res
