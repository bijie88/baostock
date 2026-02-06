import sys
import os
import concurrent.futures
import pandas as pd
import datetime
import time
from typing import List
import baostock as bs # 引入 baostock 以便在子进程中登录
import random

# Ensure we can import modules from local dir
# Add parent dir to sys.path so 'scanner' package is found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scanner.data_loader import DataLoader
from scanner.strategy import Strategy

# === 关键修改 1：Worker 函数必须定义在类外面 (Top-level) ===
# 这样才能被多进程 (pickle) 正确序列化
def process_worker_task(task_args):
    """
    独立的进程工作函数
    """
    symbol, start_date, end_date = task_args
    
    # 每个进程必须有自己的登录会话
    # 为了避免每次请求都登录，通常做法是进程初始化时登录
    # 但为了简单稳健，这里我们在函数内处理，虽然有一点点开销，但比报错强
    lg = bs.login()
    # if lg.error_code != '0':
    #     return None
    
    try:
        # 使用你现有的 DataLoader (它里面可能也有 login，没关系，BaoStock 允许重复登录)
        # 注意：这里我们直接调用 get_stock_daily
        df = DataLoader.get_stock_daily(symbol, start_date, end_date)
        
        if df.empty:
            bs.logout()
            return None
            
        # Analyze
        res = Strategy.analyze_daily(df)
        if res:
            res['symbol'] = symbol
        
        bs.logout()
        return res
        
    except Exception as e:
        bs.logout()
        # print(f"Error {symbol}: {e}")
        return None

class MainScanner:
    def __init__(self, stock_pool: List[str] = None, test_mode: bool = False):
        if test_mode:
            print("Running in TEST MODE (Small stock pool)...")
            self.stock_pool = ['sh.600519', 'sz.000858', 'sz.000001', 'sh.601318', 'sz.300059']
        else:
            # Main process needs login to fetch all stocks list if not provided
             # But NOTE: user code snippet removed global login from main. 
             # So this only works if cached or if we add temporary login here.
             # Or relies on user externally logging in?
             # Let's try to be safe: Initialize just for this call.
            DataLoader.initialize()
            self.stock_pool = stock_pool if stock_pool else DataLoader.get_all_stocks()
            DataLoader.logout() # Logout after fetching list, workers will login themselves
            
        if not self.stock_pool:
            print("Warning: Failed to fetch stock list. Using small fallback.")
            self.stock_pool = ['sh.600519', 'sz.000858'] 
            
    def run_daily_scan(self, lookback_days=700):
        print(f"Starting scan for {len(self.stock_pool)} stocks...")
        
        # === 关键修改 2：使用 ProcessPoolExecutor ===
        # 一般设为 CPU 核心数，比如 8 核电脑设为 6 或 8
        count = os.cpu_count()
        max_workers = min(count, 8) if count else 4
        print(f"Performance Mode: ON (Multiprocessing, Workers={max_workers})")
        
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        start_date = datetime.date.today() - datetime.timedelta(days=lookback_days)
        start_date_str = start_date.strftime("%Y-%m-%d")
        
        results = []
        
        # 准备任务参数列表
        tasks = [(stock, start_date_str, today_str) for stock in self.stock_pool]
        
        # 使用 ProcessPoolExecutor (多进程)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            future_to_stock = {
                executor.submit(process_worker_task, task): task[0] 
                for task in tasks
            }
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_stock):
                stock = future_to_stock[future]
                completed += 1
                try:
                    res = future.result()
                    if res:
                        results.append(res)
                    print(f"[{completed}/{len(self.stock_pool)}] Found {len(results)} candidates... Processing {stock}    ", end='\r')
                except Exception as e:
                    print(f"Error processing {stock}: {e}")
                    
        print("\nScan complete.")
        
        if not results:
            print("No results found.")
            return
            
        df_res = pd.DataFrame(results)
        
        if 'score' in df_res.columns:
            candidates = df_res[ (df_res['score'] > 50) | (df_res['signal_buy'] == True) ].copy()
            candidates.sort_values(by='score', ascending=False, inplace=True)
            
            top_n = 20
            print(f"\n=== Top {top_n} Candidates ===")
            pd.set_option('display.max_colwidth', 100)
            pd.set_option('display.unicode.east_asian_width', True)
            print(candidates[['symbol', 'score', 'signal_buy', 'details']].head(top_n))
            
            output_file = f"scan_results_{today_str}.csv"
            candidates.to_csv(output_file, index=False)
            print(f"Saved candidate list to {output_file}")

if __name__ == "__main__":
    print("Initializing A-Share Scanner (Chan + Elliott + RSRS + OFI)...")
    
    # ⚠️ Windows 下使用多进程必须放在 if __name__ == "__main__": 下
    
    # 第一次建议先 True 测试一下，如果不报错再改为 False
    TEST_MODE = False
    if len(sys.argv) > 1 and sys.argv[1] == '--all': # If --all passed, disable test mode
        TEST_MODE = False
    
    scanner = MainScanner(test_mode=TEST_MODE)
    scanner.run_daily_scan()
