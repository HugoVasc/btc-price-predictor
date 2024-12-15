import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from crypto_trading_env import CryptoTradingEnv

np.random.seed(42)

#Disable warnings
import warnings
warnings.filterwarnings('ignore')

data = pd.read_parquet('./btc_hist_partitioned.parquet')
data = data.reset_index(drop=True).drop(columns=['Date'])
data.head()

env = CryptoTradingEnv(
    dataset=data,
    initial_balance=1000,
    max_lose_percent=0.5
)

n_inputs = data.shape[1]

if tf.config.experimental.list_physical_devices('GPU'):
    print("GPU is available and will be used")
else:
    print("GPU is not available")

model = keras.models.Sequential([
    keras.layers.Dense(20, activation='elu', input_shape=[n_inputs]), # try reku, elu, leaky relu
    # keras.layers.Dense(32, activation='elu'),
    # keras.layers.Dense(4, activation='elu'),
    keras.layers.Dense(1, activation='sigmoid') # 2 actions: buy or sell
])

data_scaler = MinMaxScaler()
data_scaler.fit(data)


def play_one_step(env, obs, model, loss_fn):
    with tf.GradientTape() as tape:
        if np.isnan(obs).any() or np.isinf(obs).any():
            raise ValueError("obs contém valores NaN ou Inf")

        # proba = model(obs.values[np.newaxis]) # predict the probability of going left
        proba = model(
            data_scaler.transform(obs.values[np.newaxis])
        )
        if np.isnan(proba.numpy()).any():
            print(f"obs: {obs}")
            print(f"probs: {proba}")
            raise ValueError("proba = NaN")
        action = (tf.random.uniform([1, 1]) > proba) # action: 0 or 1
        y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
        loss = tf.reduce_mean(loss_fn(y_target, proba)) # loss function
        if tf.math.is_nan(loss):
            raise ValueError("Loss gerou NaN")
    grads = tape.gradient(loss, model.trainable_variables) # compute the gradients
    action = tf.cast(action, tf.int32).numpy()[0][0]
    obs, reward, done = env.step(action) # apply the action
    print(f"Action: {action}\tReward: {reward}")
    return obs, reward, done, grads

def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn, iteration = 0):
    all_rewards = []
    all_grads = []
    max_steps = []
    for episode in range(n_episodes):
        # print(f"Iteration: {iteration} | Episode {episode+1}")
        current_rewards = []
        current_grads = []
        current_max_steps = 0
        obs = env.reset()
        for step in range(n_max_steps):
            print(f"Iteration: {iteration+1} | Episode: {episode} | Step: {step+1}:")
            obs, reward, done, grads = play_one_step(env, obs, model, loss_fn)
            current_rewards.append(reward)
            current_grads.append(grads)
            if done:
                if step > current_max_steps:
                    current_max_steps = step
                break
        else: # executed if the loop ended without break
            current_max_steps = n_max_steps
        max_steps.append(current_max_steps)
        all_rewards.append(current_rewards)
        all_grads.append(current_grads)
    return all_rewards, all_grads, max_steps

def discount_rewards(rewards, discount_rate):
    discounted = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1): # iterate over the rewards in reverse order
        discounted[step] += discounted[step + 1] * discount_rate # add the discounted reward from the next step
    return discounted

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]


n_iterations = 1000
n_episodes_per_update = 5
n_max_steps = env.max_steps
discount_rate = 0.995

optimizer = keras.optimizers.Adam(learning_rate=0.01)
loss_fn = keras.losses.binary_crossentropy

last_rewards = []

params = {
    "layers": [20, 1],
    "activation": ['elu', 'sigmoid'],
    "optimizer": 'Adam',
    "learning_rate": 0.01,
    "loss": 'binary_crossentropy',
    "discount_rate": discount_rate,
    "n_iterations": n_iterations,
    "n_episodes_per_update": n_episodes_per_update,
    "n_max_steps": n_max_steps
}

import mlflow
from mlflow.models import infer_signature
from datetime import datetime, date

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("Reinforcement Learning for Bitcoin Trading")

with mlflow.start_run():
    mlflow.log_params(params)
    

    for iteration in range(n_iterations):
        all_rewards, all_grads, max_steps = play_multiple_episodes(env, n_episodes_per_update, n_max_steps, model, loss_fn, iteration)
        print(f"Iteration {iteration+1}:\n\tmean rewards = {np.mean([rewards[-1] for rewards in all_rewards])};\n")
        all_final_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)
        all_mean_grads = []
        for var_index in range(len(model.trainable_variables)): # iterate over all trainable variables
            mean_grads = tf.reduce_mean([final_reward * all_grads[episode_index][step][var_index] # mean gradient for this variable
                                        for episode_index, final_rewards in enumerate(all_final_rewards)
                                        for step, final_reward in enumerate(final_rewards)], axis=0)
            all_mean_grads.append(mean_grads)
        optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables)) # apply the gradients
        
        last_rewards = [rewards[-1] for rewards in all_rewards]
        mean_last_rewards = np.mean(last_rewards)
        last_reward = last_rewards[-1]
        all_rewards_concat = np.concatenate(all_rewards)
        
        max_balance = np.max(all_rewards_concat)
        min_balance = np.min(all_rewards_concat)
        drawdown = (max_balance - min_balance) / max_balance
        mlflow.log_metrics(
            step=iteration+1,
            metrics={
                "max_last_rewards": np.max(last_rewards),
                "min_last_rewards": np.min(last_rewards),
                "mean_last_rewards": mean_last_rewards,
                "last_reward": last_rewards[-1],
                "drawdown": drawdown
            },
            timestamp=np.datetime64(datetime.now()).astype(int)
        )
    
    mlflow.set_tag("Training Info", """
                   Increase to 16 the number of neurons in the first layer/
                   Increase the number of episodes per update to 5""")
    
    signature = infer_signature(data, model.predict(data_scaler.transform(data)))
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="reinforcement_learning_model_for_crypto_trading_v1",
        signature=signature,
        input_example=data.iloc[0:1],
        registered_model_name=f"reinforcement_learning_model_for_crypto_trading_v1_{datetime.now().strftime('%Y-%m-%d_%H.%M.%S')}",

    )