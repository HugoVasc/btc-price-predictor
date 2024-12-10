import pandas as pd
import numpy as np

data = pd.read_parquet('./btc_hist_partitioned.parquet')

class RLTradingEnv:
    def __init__(self, data, initial_balance=1000):
        self.data = data
        self.n = len(data)
        self.reset()
        self.initial_balance = initial_balance
        
    def reset(self):
        self.t = 0
        self.done = False
        self.profits = 0
        self.positions = []
        self.position_value = self.initial_balance
        self.cripto_percent = 0
        return self.get_state()
    
    def get_state(self):
        window = self.data.iloc[self.t:self.t+self.window_size]
        return np.array([
            self.position_value,
            self.profits,
            self.data['close'].iloc[self.t],
        ])
    
    def step(self, action): # 0: Hold / 1: Buy / 2: Sell
        if action == 0 and self.position_value == 0:
            pass
        elif action == 1: # Buy
            percent = self.position_value / self.data['close'].iloc[self.t]
            self.position_value = self.data['close'].iloc[self.t] * percent
            self.positions.append(self.position_value)
        



        # prev_position_value = self.position_value
        # if action == 1 and self.position_value == 0: # Buy
        #     self.positions.append(self.data['close'].iloc[self.t])
        #     self.position_value = self.data['close'].iloc[self.t]
        # elif action == 2 and self.position_value != 0: # Sell
        #     profits = self.data['close'].iloc[self.t] - self.position_value
        #     self.profits += profits
        #     self.position_value = 0
        # else:
        #     self.position_value = 0
        # # self.history.pop(0)
        # # self.history.append(self.data['close'].iloc[self.t])
        # self.t += 1
        # if self.t == self.n:
        #     self.done = True
        # next_state = self.get_state()
        # reward = self.position_value - prev_position_value
        # return next_state, reward, self.done