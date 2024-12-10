from crypto_trading_env import CryptoTradingEnv
import numpy as np
import pandas as pd


if __name__ == "__main__":
    # Exemplo de dataset fictício
    # data = {
    #     'Date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
    #     'Open': np.random.uniform(100, 200, 10),
    #     'High': np.random.uniform(200, 300, 10),
    #     'Low': np.random.uniform(50, 100, 10),
    #     'Close': np.random.uniform(100, 200, 10),
    #     'Volume': np.random.uniform(1000, 5000, 10),
    #     'm_avg_7': np.random.uniform(100, 200, 10),
    #     'm_avg_25': np.random.uniform(100, 200, 10),
    #     'm_avg_99': np.random.uniform(100, 200, 10),
    #     'close_diff': np.random.uniform(-0.05, 0.05, 10),
    # }
    # df = pd.DataFrame(data)
    df = pd.read_parquet('./btc_hist_partitioned.parquet')

    # Cria o ambiente
    env = CryptoTradingEnv(dataset=df, initial_balance=1000, max_lose_percent=0.2)
    state = env.reset()

    # Loop de interação
    done = False
    for _ in range(len(df)):
        action = np.random.choice([0, 1, 2])  # Escolhe uma ação aleatória
        state, profit, done = env.step(action)
        print(f"Step: {env.current_step}, Action: {action}, Profit: {profit}, Done: {done}")
