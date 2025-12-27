import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf

class MarketEnv(gym.Env):
    """
    Market Environment for Robust Market Making.
    Simulates a Limit Order Book (LOB) using OHLCV data.
    """
    def __init__(self, ticker='GC=F', start_date='2020-01-01', end_date='2023-01-01', window_size=20):
        super(MarketEnv, self).__init__()
        
        self.ticker = ticker
        self.window_size = window_size
        
        # Load Data
        self.df = self._load_data(ticker, start_date, end_date)
        
        # Action Space: 0: Hold, 1: Buy@Bid, 2: Sell@Ask, 3: Clear Inventory
        self.action_space = spaces.Discrete(4)
        
        # Observation Space: [Mid_Price, Bid_Ask_Spread, Volume_Imbalance, RSI, MACD, Holdings_Inventory, Cash]
        # We normalize these for the agent strictly speaking, but for now we provide raw values or normalized
        # Adding Cash/PnL info might be useful but PRD says specifically:
        # [Mid_Price, Bid_Ask_Spread, Volume_Imbalance, RSI, MACD, Holdings_Inventory]
        # I will stick to PRD but adding Normalized Time/Step might help. Sticking to PRD for now.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        self.reset()

    def _load_data(self, ticker, start, end):
        import os
        if os.path.exists('gold_data.csv'):
            data = pd.read_csv('gold_data.csv', index_col=0, parse_dates=True)
            # Ensure index is datetime
            data.index = pd.to_datetime(data.index)
        else:
            data = yf.download(ticker, start=start, end=end, progress=False, interval="1h")
            if len(data) == 0:
                 print(f"Warning: No data found for {ticker}. Generating synthetic data.")
                 dates = pd.date_range(start, end, freq='h')
                 data = pd.DataFrame(index=dates)
                 data['Close'] = 100 + np.random.randn(len(dates)).cumsum()
                 data['Volume'] = np.random.randint(100, 1000, size=len(dates))
            else:
                # Save to CSV for stability
                data.to_csv('gold_data.csv')
             
        # Feature Engineering
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        
        # Synthetic Bid/Ask Spread
        data['Spread'] = data['Close'] * np.random.uniform(0.0001, 0.0005, size=len(data))
        
        # Synthetic Volume Imbalance (-1 to 1)
        data['Imbalance'] = np.random.uniform(-1, 1, size=len(data))
        
        # Mid Price is Approx Close
        data['Mid'] = data['Close']
        
        data.fillna(0, inplace=True)
        return data.reset_index(drop=True)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = self.window_size 
        self.inventory = 0
        self.cash = 10000.0 
        self.entry_price = 0 
        self.done = False
        self.initial_price = self.df.iloc[self.current_step]['Mid'] # Store for normalization
        return self._get_observation(), {}

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        
        # Normalization (Improved based on Expert Review)
        # Relative scaling: (Current / Ref) - 1.0 gives % change centered at 0
        # Use initial price of the episode as reference
        ref_price = self.initial_price if self.initial_price > 0 else 1.0
        
        norm_price = (row['Mid'] / ref_price) - 1.0
        norm_spread = row['Spread'] / row['Mid'] # Spread as % of price
        norm_rsi = row['RSI'] / 100.0
        norm_macd = row['MACD'] / ref_price # MACD relative to price scale
        
        obs = np.array([
            norm_price * 10.0,   # Scale up slightly
            norm_spread * 1000.0, # Scale up small spread
            row['Imbalance'], 
            norm_rsi,
            norm_macd * 10.0,
            self.inventory / 10.0 # Normalize inventory (assuming soft limit ~10)
        ], dtype=np.float32)
        return obs

    def step(self, action):
        row = self.df.iloc[self.current_step]
        mid = row['Mid']
        half_spread = row['Spread'] / 2.0
        bid = mid - half_spread
        ask = mid + half_spread
        
        reward = 0
        transaction_cost = 0.0001 * mid # Transaction fees etc.
        
        # 0: Hold
        if action == 0:
            pass
            
        # 1: Buy Limit Order (at Best Bid) -> Executed immediately for simplicity or probabilistic?
        # PRD implies Market Making logic. We assume fill probability or immediate fill at Bid.
        # For simplicity in Phase 1: Immediate fill at Bid.
        elif action == 1:
            self.inventory += 1
            self.cash -= bid
            # Update entry price? weighted avg
            
        # 2: Sell Limit Order (at Best Ask)
        elif action == 2:
            self.inventory -= 1
            self.cash += ask
            
        # 3: Clear Inventory (Market Order)
        elif action == 3:
            if self.inventory > 0:
                # Sell all at Bid (Market Sell)
                self.cash += self.inventory * bid
                self.inventory = 0
            elif self.inventory < 0:
                # Buy back at Ask (Market Buy)
                self.cash -= abs(self.inventory) * ask
                self.inventory = 0
        
        # Calculate PnL Change (Reward)
        # Mark to Market PnL calculation
        # Net Wealth = Cash + Inventory * Mid
        current_wealth = self.cash + self.inventory * mid
        
        # To calculate step reward, we check change in wealth or realized pnl.
        # Usually for RL, change in portfolio value is a good dense reward.
        # But we also need Inventory Pnealty.
        
        # Let's say we track previous wealth.
        # For efficiency, let's just approximate Reward = Spread Captured - Penality
        # Realistically: Reward = (Value_t - Value_{t-1}) - InventoryPenalty
        if not hasattr(self, 'prev_wealth'):
            self.prev_wealth = 10000.0
            
        step_pnl = current_wealth - self.prev_wealth
        self.prev_wealth = current_wealth
        
        # Inventory Penalty (Risk Aversion)
        # Gamma * Inventory^2
        penalty = 0.01 * (self.inventory ** 2)
        
        reward = step_pnl - penalty
        
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True
            
        truncated = False
        info = {'pnl': step_pnl, 'inventory': self.inventory, 'wealth': current_wealth}
        
        return self._get_observation(), reward, self.done, truncated, info

    def render(self):
        pass
