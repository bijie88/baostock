import baostock as bs
import pandas as pd
import datetime
import os
import threading
from typing import List, Optional

class DataLoader:
    """
    Data Loader utilizing BaoStock for A-share market data.
    Features:
    - Global Login/Logout
    - Local CSV Caching
    - High Concurrency Support
    """
    _login_lock = threading.Lock()
    _is_logged_in = False
    CACHE_DIR = "data_cache"

    @classmethod
    def initialize(cls):
        """Global login to BaoStock."""
        with cls._login_lock:
            if not cls._is_logged_in:
                lg = bs.login()
                if lg.error_code == '0':
                    print(f"BaoStock login success: {lg.error_msg}")
                    cls._is_logged_in = True
                    # Ensure cache dir exists
                    if not os.path.exists(cls.CACHE_DIR):
                        os.makedirs(cls.CACHE_DIR)
                else:
                    print(f"BaoStock login failed: {lg.error_msg}")

    @classmethod
    def logout(cls):
        """Global logout."""
        bs.logout()
        cls._is_logged_in = False

    @staticmethod
    def get_stock_daily(symbol: str, start_date: str, end_date: str, adjust: str = "2") -> pd.DataFrame:
        """
        Get daily historical stock data with Caching.
        
        Args:
            symbol (str): Stock code (e.g., 'sh.600000'). NOTE: BaoStock needs 'sh.'/'sz.' prefix.
            start_date (str): 'YYYY-MM-DD'
            end_date (str): 'YYYY-MM-DD'
            adjust (str): '1': hfq, '2': qfq (default), '3': no adjust.
            
        Returns:
            pd.DataFrame
        """
        # Normalize symbol for filename (e.g. sh.600000 -> sh_600000.csv)
        file_symbol = symbol.replace('.', '_')
        cache_file = os.path.join(DataLoader.CACHE_DIR, f"{file_symbol}.csv")
        
        # 1. Try Cache
        if os.path.exists(cache_file):
            try:
                # We assume cache contains enough data or use it as is.
                # Ideally check last date, but for simplicity read all.
                df = pd.read_csv(cache_file)
                df['date'] = pd.to_datetime(df['date']).dt.date
                # Filter by date range? Optional, mostly we want everything ending today.
                return df
            except Exception as e:
                print(f"Error reading cache for {symbol}: {e}")

        # 2. Download from BaoStock
        # BaoStock requires 'YYYY-MM-DD' format.
        # Input might be 'YYYYMMDD', convert if needed.
        if len(start_date) == 8:
            start_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
        if len(end_date) == 8:
            end_date = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"

        try:
            # fields: date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST
            rs = bs.query_history_k_data_plus(
                symbol,
                "date,code,open,high,low,close,volume,amount,turn,pctChg",
                start_date=start_date, end_date=end_date,
                frequency="d", adjustflag=adjust
            )
            
            if rs.error_code != '0':
                return pd.DataFrame()

            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())
                
            if not data_list:
                return pd.DataFrame()
                
            df = pd.DataFrame(data_list, columns=rs.fields)
            
            # Convert types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # Rename for compatibility
            df.rename(columns={'turn': 'turnover', 'pctChg': 'pct_change'}, inplace=True)
            df['date'] = pd.to_datetime(df['date']).dt.date
            
            # 3. Save to Cache
            df.to_csv(cache_file, index=False)
            
            return df
            
        except Exception as e:
            print(f"Error downloading {symbol}: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_all_stocks() -> List[str]:
        """
        Get all A-share stocks.
        """
        today = datetime.date.today().strftime("%Y-%m-%d")
        rs = bs.query_all_stock(day=today)
        
        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
            
        if not data_list:
            # Fallback to previous day if today is holiday/weekend?
            # BaoStock query_all_stock works on business days.
            return []
            
        df = pd.DataFrame(data_list, columns=rs.fields)
        # return symbols like 'sh.600000'
        return df['code'].tolist()

    @staticmethod
    def get_realtime_quotes():
        # BaoStock doesn't provide real-time tick in the same way as AkShare.
        # We might skip Micro-structure for now or use yesterday's close.
        pass
