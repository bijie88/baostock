import sys
import os
import concurrent.futures
import pandas as pd
import datetime
import time
from typing import List

# Ensure we can import modules from local dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scanner.data_loader import DataLoader
from scanner.strategy import Strategy

class MainScanner:
    def __init__(self, stock_pool: List[str] = None):
        self.stock_pool = stock_pool if stock_pool else DataLoader.get_hs300_stocks()
        if not self.stock_pool:
            print("Warning: Failed to fetch HS300, using fallback list.")
            self.stock_pool = ['000001', '600519', '000858'] # Fallback
            
    def run_daily_scan(self, lookback_days=700):
        """
        Run daily scan on the stock pool.
        """
        print(f"Starting scan for {len(self.stock_pool)} stocks...")
        
        today_str = datetime.date.today().strftime("%Y%m%d")
        start_date_str = (datetime.date.today() - datetime.timedelta(days=lookback_days)).strftime("%Y%m%d")
        
        results = []
        
        # ThreadPool for I/O bound tasks (Network fetching)
        # Limit workers to avoid IP ban
        max_workers = 5 
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_stock = {
                executor.submit(self._process_single_stock, stock, start_date_str, today_str): stock 
                for stock in self.stock_pool[:50] # Limit to 50 for demo/testing speed
            }
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_stock):
                stock = future_to_stock[future]
                completed += 1
                try:
                    res = future.result()
                    if res:
                        results.append(res)
                    print(f"[{completed}/{len(future_to_stock)}] Processed {stock}", end='\r')
                except Exception as e:
                    print(f"Error processing {stock}: {e}")
                    
        print("\nScan complete.")
        
        # Convert to DataFrame
        if not results:
            print("No results found.")
            return
            
        df_res = pd.DataFrame(results)
        
        # Filter signals
        # Logic: Score > 50 OR signal_buy is True
        candidates = df_res[ (df_res['score'] > 50) | (df_res['signal_buy'] == True) ].copy()
        
        # Sort by score
        candidates.sort_values(by='score', ascending=False, inplace=True)
        
        print("\n=== Top Candidates ===")
        print(candidates[['symbol', 'score', 'signal_buy', 'details']].head(10))
        
        # Save to CSV
        output_file = f"scan_results_{today_str}.csv"
        candidates.to_csv(output_file, index=False)
        print(f"Saved candidate list to {output_file}")
        
    def _process_single_stock(self, symbol, start_date, end_date):
        """
        Worker function.
        """
        try:
            # Add basic sleep to be nice to API
            # time.sleep(0.1) 
            
            # Fetch Data
            df = DataLoader.get_stock_daily(symbol, start_date, end_date)
            if df.empty:
                return None
                
            # Analyze
            res = Strategy.analyze_daily(df)
            res['symbol'] = symbol
            
            return res
            
        except Exception as e:
            # print(f"Err {symbol}: {e}")
            return None

if __name__ == "__main__":
    print("Initializing A-Share Scanner (Chan + Elliott + RSRS + OFI)...")
    scanner = MainScanner()
    scanner.run_daily_scan()
