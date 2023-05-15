from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from tqdm import tqdm

import gymnasium as gym
import tensorflow as tf
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# Let's start by creating the blackjack environment.
# Note: We are going to follow the rules from Sutton & Barto.
# Other versions of the game can be found below for you to experiment.

env = gym.make("Breakout-v0")


class BreakoutAgent:
    def __init__(self, env, learning_rate=0.001):
        self.env = env

        # Initialize a simple neural network model
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(210, 160, 3)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.env.action_space.n)
        ])

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='mse')

    def get_action(self, obs):
        q_values = self.model.predict(np.expand_dims(obs, axis=0))
        return np.argmax(q_values[0])

    def update(self, obs, action, reward, next_obs, done):
        q_values = self.model.predict(np.expand_dims(obs, axis=0))
        next_q_values = self.model.predict(np.expand_dims(next_obs, axis=0))

        q_values[0][action] = reward + 0.95 * np.max(next_q_values[0]) * (1 - done)
        self.model.fit(np.expand_dims(obs, axis=0), q_values, verbose=0)

n_episodes = 5

agent = BreakoutAgent(env)

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    obs = obs.reshape((210, 160, 3))
    done = False

    # play one episode
    while not done:
        obs = obs.reshape((210, 160, 3))
        action = agent.get_action(obs)
        #res = env.step(action)
        #print(res)
        next_obs, reward, done, trunc, info = env.step(action)
        next_obs = next_obs.reshape((210, 160, 3))
        print(f'Done: {done}, Trunc: {trunc}')
        # update the agent
        agent.update(obs, action, reward, next_obs, done)

        # update if the environment is done and the current obs
        done = done or trunc
        obs = next_obs

    agent.decay_epsilon()


env = gym.make("Breakout-v0", render_mode='human')
observation, info = env.reset()
agent.set_env(env)

for _ in range(1000):
    action = agent.get_action(observation)  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()