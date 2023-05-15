import gym
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
import random

import gc
from tensorflow.keras import backend as k
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU
from tensorflow.keras.callbacks import Callback

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
#tf.logging.set_verbosity(tf.logging.INFO)


class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        k.clear_session()


class DQNAgent:
    def __init__(self, state_shape, action_space, replay_memory_size=50000):
        self.state_shape = state_shape
        self.action_space = action_space
        self.memory = []
        self.replay_memory_size = replay_memory_size
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        model = Sequential()

        model.add(Conv2D(32, (8, 8), strides=(4, 4), padding='valid', 
                        activation='relu', input_shape=self.state_shape))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='valid', 
                        activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', 
                        activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_space))
        model.compile(loss='mse', optimizer=Adam(), run_eagerly=True)
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) > self.replay_memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        '''states = np.squeeze(states)
        next_states = np.squeeze(next_states)'''
        states = np.reshape(states, (-1, *self.state_shape))
        next_states = np.reshape(next_states, (-1, *self.state_shape))


        targets = rewards + self.gamma*(np.amax(self.target_model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0, callbacks=ClearMemory())
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        self.model.save(name)

    def load(self, name):
        self.model = tf.keras.models.load_model(name)
# 1 for train, 0 for play
train_play = 1

if train_play:
    # Initialize gym environment and the agent
    env = gym.make('BreakoutDeterministic-v4')
    agent = DQNAgent(env.observation_space.shape, env.action_space.n)

    #print(f'LOOK OBERSVATION SPACE: {env.observation_space.shape}')
    # Iterate the game

    num_eps = 1000
    for e in range(num_eps):
        # reset state in the beginning of each game
        state, info = env.reset()
        #print(f'THIS IS THE STATE: {state}')
        state = np.reshape(state, [1, *state.shape])

        # time ticks
        for time in range(5000):

            # turn this on if you want to render
            '''if(time>4000):
                env.render()'''

            # Decide action
            action = agent.act(state)

            # Advance the game to the next frame based on the action.
            #print(f'THIS IS THE ACTION: {env.step(action)}')
            next_state, reward, done, trunc, info  = env.step(action)
            #print(f'LOOK NEXT STATE: {next_state.shape}')


            # Remember the previous state, action, reward, and done
            next_state = np.reshape(next_state, [1, *next_state.shape])
            #print(f'LOOK NEXT STATE RESHAPED: {next_state.shape}')
            agent.remember(state, action, reward, next_state, done)

            # make next_state the new current state for the next frame.
            state = next_state

            # done becomes True when the game ends
            # ex) The agent drops the pole
            if done:
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}"
                    .format(e, num_eps, time))
                break

            # train the agent with the experience of the episode
            if len(agent.memory) > 32:
                agent.replay(32)

        # update target model weights every episode
        agent.update_target_model()

    # Save the model after training
    agent.save("dqn_model_bigly.h5")

else:
    # Initialize gym environment and the agent
    env = gym.make('BreakoutDeterministic-v4', render_mode='human')
    agent = DQNAgent(env.observation_space.shape, env.action_space.n)
    agent.load('dqn_model_bigly.h5')

    #print(f'LOOK OBERSVATION SPACE: {env.observation_space.shape}')
    # Iterate the game

    num_eps = 10
    for e in range(num_eps):
        # reset state in the beginning of each game
        state, info = env.reset()
        #print(f'THIS IS THE STATE: {state}')
        state = np.reshape(state, [1, *state.shape])

        # time ticks
        for time in range(5000):

            # turn this on if you want to render
            env.render()

            # Decide action
            action = agent.act(state)

            # Advance the game to the next frame based on the action.
            #print(f'THIS IS THE ACTION: {env.step(action)}')
            next_state, reward, done, trunc, info  = env.step(action)
            #print(f'LOOK NEXT STATE: {next_state.shape}')


            # Remember the previous state, action, reward, and done
            next_state = np.reshape(next_state, [1, *next_state.shape])
            #print(f'LOOK NEXT STATE RESHAPED: {next_state.shape}')
            #agent.remember(state, action, reward, next_state, done)

            # make next_state the new current state for the next frame.
            state = next_state

            # done becomes True when the game ends
            # ex) The agent drops the pole
            if done:
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}"
                    .format(e, num_eps, time))
                break

            # train the agent with the experience of the episode

        # update target model weights every episode