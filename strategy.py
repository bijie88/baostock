import pandas as pd
import pandas_ta as ta  # optional, if installed
from .chan_core import ChanCore
from .wave_core import WaveCore
from .rsrs_core import RSRSCore
from .chip_core import ChipCore
from .micro_structure import MicroStructure

class Strategy:
    """
    Composite Strategy Engine.
    Combines Macro (RSRS, Chip) -> Pattern (Chan, Wave) -> Micro (OFI, VPIN).
    """

    @staticmethod
    def analyze_daily(df: pd.DataFrame) -> dict:
        """
        Stage 1: Daily Analysis.
        Returns a dictionary of signals and scores.
        """
        if df.empty or len(df) < 60:
            return {'signal': False, 'reason': 'Insufficient Data'}
            
        result = {
            'symbol': '', # Caller fills this
            'date': df['date'].iloc[-1],
            'close': df['close'].iloc[-1],
            'signal_buy': False,
            'score': 0,
            'details': []
        }
        
        # 1. Macro: RSRS
        # Using 18 days regression, 600 days normalization (or less if not avail)
        rsrs_res = RSRSCore.calculate_rsrs(df)
        score_rsrs = rsrs_res['rsrs_score']
        
        result['rsrs'] = score_rsrs
        if score_rsrs > 0.7:
            result['score'] += 30
            result['details'].append(f"RSRS Strong ({score_rsrs:.2f})")
        elif score_rsrs < -0.7:
            result['score'] -= 30
            result['details'].append(f"RSRS Weak ({score_rsrs:.2f})")
            
        # 2. Macro: Chip Distribution
        chip_res = ChipCore.calculate_chip_distribution(df)
        profit_ratio = chip_res['profit_ratio']
        concentration = chip_res['concentration'] # Lower is better? No, formula: (p95-p05)/(p05+p95). Smaller = higher concentration.
        
        # Concentration usually < 0.1 is highly concentrated?
        # Let's say < 0.15 is good.
        
        result['chip_profit'] = profit_ratio
        if profit_ratio > 0.9:
            result['score'] += 20
            result['details'].append("Chip: High Profit Ratio (>90%)")
        elif profit_ratio > 0.5:
             result['score'] += 10
             
        if concentration < 0.15:
            result['score'] += 10
            result['details'].append("Chip: Highly Concentrated")
            
        # 3. Pattern: Chan Lun
        # Calc Fractals and Bi
        df = ChanCore.find_fractals(df)
        bi_list = ChanCore.get_bi_list(df)
        chan_res = ChanCore.check_buys(bi_list, df['close'].iloc[-1])
        
        if chan_res['buy2']:
            result['score'] += 25
            result['signal_buy'] = True # Potential Trigger
            result['details'].append("Chan: Buy 2 Point")
            
        if chan_res['buy3']:
            result['score'] += 30
            result['signal_buy'] = True
            result['details'].append("Chan: Buy 3 Point (Strong Breakout)")
            
        # 4. Pattern: Elliott Wave (AO)
        wave_res = WaveCore.check_wave_structure(df)
        if wave_res['is_wave3']:
            result['score'] += 15
            result['details'].append("Wave: Potential Wave 3")
        if wave_res['divergence']:
             result['score'] -= 20
             result['details'].append("Wave: Bearish Divergence (Risk)")

        # 5. Volume/Trend Filter
        # MA Alignment
        ma20 = df['close'].rolling(20).mean().iloc[-1]
        ma60 = df['close'].rolling(60).mean().iloc[-1]
        if ma20 > ma60:
             result['score'] += 5
        
        # Final Coarse Filter
        # e.g. Score > 60 implies a good candidate
        
        return result

    @staticmethod
    def analyze_intraday(ticks_df: pd.DataFrame, daily_res: dict = None) -> dict:
        """
        Stage 2: Micro-structure verification.
        Args:
            ticks_df: DataFrame with tick data (Bid/Ask/Vol).
            daily_res: Result from Stage 1.
        """
        result = {'confirmed': False, 'vpin': 0, 'ofi': 0, 'details': []}
        
        # 1. OFI Check
        ofi_val = MicroStructure.calculate_ofi_l1(ticks_df)
        result['ofi'] = ofi_val
        
        if ofi_val > 0:
            result['details'].append(f"OFI Positive ({ofi_val:.2f}) - Buying Pressure")
            result['confirmed'] = True
        else:
            result['details'].append(f"OFI Negative ({ofi_val:.2f}) - Selling Pressure")

        # 2. VPIN Check (if we have trades)
        # Assuming ticks_df has 'price', 'volume' columns for trade log if available, 
        # or we derived it.
        # If ticks_df is L1 snapshot, we can't do exact VPIN easily unless we infer trades.
        # Let's assume we pass in a derived 'trades' df or ticks_df has volume info.
        
        # For simplified snapshot usage, we skip VPIN or use trivial volume change.
        
        return result
