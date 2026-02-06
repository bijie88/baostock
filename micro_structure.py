import pandas as pd
import numpy as np

class MicroStructure:
    """
    Micro-structure indicators: OFI (Order Flow Imbalance) and VPIN.
    Note: Limitations apply due to lack of L2 data.
    """
    
    @staticmethod
    def calculate_ofi_l1(ticks: pd.DataFrame) -> float:
        """
        Calculate Net OFI (Order Flow Imbalance) using Level-1 Snapshot data.
        Requirement: `ticks` DataFrame must contain time-series of:
        ['bid1_price', 'bid1_volume', 'ask1_price', 'ask1_volume']
        
        OFI_t = e_b_t - e_a_t
        
        Accumulates OFI over the provided ticks window.
        """
        if ticks.empty or len(ticks) < 2:
            return 0.0
            
        # Vectorized calculation
        
        # Bid Side Flow (e_b)
        b_p = ticks['bid1_price'].values
        b_q = ticks['bid1_volume'].values
        
        # Shifted (prev)
        b_p_prev = np.roll(b_p, 1)
        b_q_prev = np.roll(b_q, 1)
        b_p_prev[0] = b_p[0] # Handle first element
        b_q_prev[0] = b_q[0]
        
        # Logic:
        # P > P_prev: flow = q
        # P == P_prev: flow = q - q_prev
        # P < P_prev: flow = -q_prev
        
        # We can implement this with condition masks
        e_b = np.where(b_p > b_p_prev, b_q,
                       np.where(b_p == b_p_prev, b_q - b_q_prev, -b_q_prev))
        # First element is invalid/zero
        e_b[0] = 0
        
        # Ask Side Flow (e_a)
        a_p = ticks['ask1_price'].values
        a_q = ticks['ask1_volume'].values
        
        a_p_prev = np.roll(a_p, 1)
        a_q_prev = np.roll(a_q, 1)
        a_p_prev[0] = a_p[0]
        a_q_prev[0] = a_q[0]
        
        # Ask logic (similar but impact is opposite? No, formula definition for inflow):
        # P < P_prev: flow = q (Price improved/lowered -> new stack)
        # P == P_prev: flow = q - q_prev
        # P > P_prev: flow = -q_prev (Price rose -> old stack gone)
        
        # Wait, Rama Cont definition:
        # e_n_t (Ask side):
        # If P < P_prev (Best Ask Dropped): q (New limit orders)
        # If P == P_prev: q - q_prev
        # If P > P_prev (Best Ask Rose): -q_prev (Orders consumed/cancelled)
        
        e_a = np.where(a_p < a_p_prev, a_q,
                       np.where(a_p == a_p_prev, a_q - a_q_prev, -a_q_prev))
        e_a[0] = 0
        
        # Net OFI
        ofi = e_b - e_a
        
        # Return Cumulative OFI
        return np.sum(ofi)

    @staticmethod
    def calculate_vpin(trades: pd.DataFrame, bucket_volume: int = 10000, window_buckets: int = 50) -> float:
        """
        VPIN (Volume-Synchronized Probability of Informed Trading).
        Requires `trades` DataFrame with ['price', 'volume'] and ideally ['type'] (Buy/Sell) 
        or we use Lee-Ready.
        
        Args:
            trades: Tick-by-tick trades (not snapshots).
            bucket_volume: Volume size per bucket.
            window_buckets: Number of buckets for VPIN calculation.
        """
        if trades.empty:
            return 0.0
            
        # 1. Trade Classification (Lee-Ready)
        # If we don't have Bid/Ask mid in 'trades', we use Tick Rule (Price Change).
        # trades['price'] vs trades['price'].shift(1)
        
        price_diff = trades['price'].diff()
        
        # 0 diff handling: use last non-zero. simpler: ffill the 'sign'.
        # sign: 1 (Buy), -1 (Sell)
        signs = np.sign(price_diff)
        signs = signs.replace(0, np.nan).ffill().fillna(0) # 0 for start
        
        # Buy Vol, Sell Vol
        # If sign > 0: Buy Volume = Volume, Sell = 0
        # If sign < 0: Sell Volume = Volume, Buy = 0
        
        trades = trades.copy()
        trades['buy_vol'] = np.where(signs > 0, trades['volume'], 0)
        trades['sell_vol'] = np.where(signs < 0, trades['volume'], 0)
        
        # 2. Bucketing
        # We need to aggregate into volume buckets
        # This is strictly sequential.
        
        cum_vol = trades['volume'].cumsum()
        # Determine bucket indices
        # bucket_id = cum_vol // bucket_volume
        
        trades['bucket'] = (cum_vol // bucket_volume).astype(int)
        
        # Group by bucket
        # We want OI (Order Imbalance) per bucket = |Buy - Sell|
        
        bucket_stats = trades.groupby('bucket').agg({
            'buy_vol': 'sum',
            'sell_vol': 'sum'
        })
        
        bucket_stats['OI'] = np.abs(bucket_stats['buy_vol'] - bucket_stats['sell_vol'])
        bucket_stats['Total'] = bucket_stats['buy_vol'] + bucket_stats['sell_vol'] 
        # Note: Total should be approx bucket_volume (except last one)
        
        # 3. VPIN Calculation (Rolling mean of OI / Average Volume)
        # VPIN = Sum(OI over N buckets) / (N * bucket_volume)
        
        if len(bucket_stats) < window_buckets:
            # Not enough data
            return 0.0
            
        recent_oi_sum = bucket_stats['OI'].tail(window_buckets).sum()
        total_vol = bucket_stats['Total'].tail(window_buckets).sum()
        
        if total_vol == 0:
            return 0.0
            
        vpin = recent_oi_sum / total_vol
        return vpin
