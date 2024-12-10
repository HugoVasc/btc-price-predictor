import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from crypto_trading_env import CryptoTradingEnv


data = pd.read_parquet('btc_hist_partitioned.parquet')

env = CryptoTradingEnv(
    data=data,
    initial_balance=1000,
    max_lose_percent=0.5
)

n_inputs = 11

model = keras.models.Sequential([
    keras.layers.Dense(5, activation='elu', input_shape=[n_inputs]),
    keras.layers.Dense(3, activation='sigmoid')
])

def play_one_step(env, obs, model, loss_fn):
    with tf.GradientTape() as tape:
        probas = model(obs[np.newaxis]) # predict the probability of going left
        action = tf.random.categorical(probas, 1) # sample random action
        y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32) # target probability
        loss = tf.reduce_mean(loss_fn(y_target, left_proba)) # loss function
    grads = tape.gradient(loss, model.trainable_variables) # compute the gradients
    obs, reward, done, _, info = env.step(int(action[0, 0].numpy())) # apply the action
    return obs, reward, done, grads

def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
    all_rewards = []
    all_grads = []
    max_steps = []
    for episode in range(n_episodes):
        current_rewards = []
        current_grads = []
        current_max_steps = 0
        obs = env.reset()[0]
        for step in range(n_max_steps):
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


n_iterations = 150
n_episodes_per_update = 10
n_max_steps = 200
discount_rate = 0.95

optimizer = keras.optimizers.Adam(learning_rate=0.01)
loss_fn = keras.losses.categorical_crossentropy

for iteration in range(n_iterations):
    all_rewards, all_grads, max_steps = play_multiple_episodes(env, n_episodes_per_update, n_max_steps, model, loss_fn)
    print(f"Iteration {iteration}:\n\tmean rewards = {np.mean([sum(rewards) for rewards in all_rewards])};\n\tmax steps = {np.max(max_steps)};\n\tmean steps = {np.mean(max_steps)}")
    all_final_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)
    all_mean_grads = []
    for var_index in range(len(model.trainable_variables)): # iterate over all trainable variables
        mean_grads = tf.reduce_mean([final_reward * all_grads[episode_index][step][var_index] # mean gradient for this variable
                                     for episode_index, final_rewards in enumerate(all_final_rewards)
                                     for step, final_reward in enumerate(final_rewards)], axis=0)
        all_mean_grads.append(mean_grads)
    optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables)) # apply the gradients