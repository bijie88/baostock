import pandas as pd
import numpy as np
from enum import Enum
from typing import List, Optional, Tuple

class BiType(Enum):
    UP = 1
    DOWN = -1

class ChanCore:
    """
    Simplified Chan Lun implementation focusing on Fractals (Fen Xing) and Bi (Strokes).
    """

    @staticmethod
    def find_fractals(df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify Top and Bottom Fractals.
        Top Fractal: High[i] > High[i-1] and High[i] > High[i+1]
        Bottom Fractal: Low[i] < Low[i-1] and Low[i] < Low[i+1]
        
        Note: This is a simplified version without strict 'inclusion' (Baohan) processing.
        """
        df = df.copy()
        df['type'] = 0  # 1: Top, -1: Bottom
        
        highs = df['high'].values
        lows = df['low'].values
        
        # Vectorized search for simple fractals (3 bars)
        # We need at least 3 bars. 
        # For strict Chan Lun, we need 5 bars? Let's use 3 for sensitivity, or 5?
        # Standard: Look at 3 bars (Left, Mid, Right).
        
        # Top Fractal
        is_top = (highs[1:-1] > highs[:-2]) & (highs[1:-1] > highs[2:])
        # Bottom Fractal
        is_bottom = (lows[1:-1] < lows[:-2]) & (lows[1:-1] < lows[2:])
        
        # Fill types (offset by 1 because we sliced 1:-1)
        # Note: We prioritize validity. If Top and Bottom share bars, it's noise, but for now mark them.
        
        # We prefer to iterate to handle "shared" bars or complex patterns if needed, 
        # but vectorized is faster.
        
        top_indices = np.where(is_top)[0] + 1
        bottom_indices = np.where(is_bottom)[0] + 1
        
        df.iloc[top_indices, df.columns.get_loc('type')] = 1
        df.iloc[bottom_indices, df.columns.get_loc('type')] = -1
        
        return df

    @staticmethod
    def get_bi_list(df: pd.DataFrame) -> List[dict]:
        """
        Generate Bi (Strokes) from fractals.
        Rules:
        1. Start from a fractal.
        2. Top -> Bottom -> Top ...
        3. Determine direction based on price.
        4. (Simplified) Ignore "number of bars in between" constraint for now, 
           or enforce min 1 bar between Top and Bottom.
        """
        if 'type' not in df.columns:
            df = ChanCore.find_fractals(df)
            
        fractals = df[df['type'] != 0].copy()
        if fractals.empty:
            return []
        
        bi_list = []
        curr_bi = None
        
        # Iterate through fractals
        # We need to find a sequence of Top -> Bottom or Bottom -> Top
        
        last_f = None
        
        for idx, row in fractals.iterrows():
            f_type = row['type']
            f_price = row['high'] if f_type == 1 else row['low']
            f_date = row['date']
            
            if last_f is None:
                last_f = {'index': idx, 'type': f_type, 'price': f_price, 'date': f_date}
                continue
            
            # If we have a Last Fractal
            if last_f['type'] == f_type:
                # Same type, keep the more extreme one (Higher Top or Lower Bottom)
                if f_type == 1: # Top
                    if f_price > last_f['price']:
                        last_f = {'index': idx, 'type': f_type, 'price': f_price, 'date': f_date}
                else: # Bottom
                    if f_price < last_f['price']:
                        last_f = {'index': idx, 'type': f_type, 'price': f_price, 'date': f_date}
            else:
                # Different type, form a Bi
                # Check validity (e.g. price difference validation? not strictly needed for raw Bi)
                bi = {
                    'start_index': last_f['index'],
                    'start_date': last_f['date'],
                    'start_price': last_f['price'],
                    'end_index': idx,
                    'end_date': f_date,
                    'end_price': f_price,
                    'type': BiType.DOWN if last_f['type'] == 1 else BiType.UP
                }
                bi_list.append(bi)
                last_f = {'index': idx, 'type': f_type, 'price': f_price, 'date': f_date}
                
        return bi_list

    @staticmethod
    def check_buys(bi_list: List[dict], current_price: float) -> dict:
        """
        Check for Buy Points based on Bi list.
        Args:
            bi_list: List of historical Bi.
            current_price: Current price to check potential dynamic setup.
            
        Returns:
            dict: {'buy1': bool, 'buy2': bool, 'buy3': bool, 'desc': str}
        """
        if len(bi_list) < 4:
            return {'buy1': False, 'buy2': False, 'buy3': False, 'desc': ''}
        
        # Consider the last completed Bi
        last_bi = bi_list[-1]
        prev_bi = bi_list[-2]
        prev_prev_bi = bi_list[-3]
        
        res = {'buy1': False, 'buy2': False, 'buy3': False, 'desc': ''}
        
        # 1. First Buy (Trend Reversal logic - Simplified)
        # Usually requires divergence. We'll leave strict 1-Buy to MACD combination in Strategy.
        # But chart-wise: A sharp Down Bi followed by a failure to continue down? 
        # Hard to judge 1st buy solely on Bi without divergence.
        
        # 2. Second Buy (Pullback logic)
        # Situation: Up Bi (prev), then Down Bi (last).
        # Logic: Last Down Bi's Low > Previous Low (from where Up Bi started).
        # Structure: ... -> Down(pp) -> Up(p) -> Down(last)
        if last_bi['type'] == BiType.DOWN:
            # The low of the last Down Bi
            last_low = last_bi['end_price']
            # The low of the previous Down Bi (start of the Up Bi)
            prev_low = prev_bi['start_price'] # prev_bi is UP, so start is Low
            
            if last_low > prev_low:
                # Potential 2nd Buy
                res['buy2'] = True
                res['desc'] += f"Buy 2: Higher Low formed ({last_low} > {prev_low}). "

        # 3. Third Buy (Trend Continuation/Breakout logic)
        # Situation: Large Up Trend, consolidation (Zoo), then Breakout, then Pullback not entering Zoo.
        # Simplified: Up -> Down (High Low) -> Up (Break High) -> Down (Stay above High)
        # This needs more history. Let's look at relative strength.
        # If Last Bi is DOWN, and its Low is significantly higher than previous Pivot High.
        
        if last_bi['type'] == BiType.DOWN:
             # prev_bi is UP. prev_bi start is Low, end is High.
             # prev_prev_bi is DOWN. 
             
             # Pivot High: prev_bi['end_price']
             # Pivot Low: prev_bi['start_price']
             
             # Actually for 3rd Buy, we typically compare with a "Center" (ZhongShu).
             # Simplified: If Last Low > The High of the 'prev_prev_bi' (The peak before the valley)
             # Structure: Top1 -> Low1 -> Top2 -> Low2. 
             # If Low2 > Top1, it's very strong. (Gap up-ish structure).
             
             # prev_prev_bi is DOWN. Its start is a High (Top1).
             top1 = prev_prev_bi['start_price']
             last_low = last_bi['end_price']
             
             if last_low > top1:
                 res['buy3'] = True
                 res['desc'] += f"Buy 3: Strong Trend (Low {last_low} > Prev High {top1}). "

        return res
