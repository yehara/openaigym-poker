import numpy as np
import gym
from gym import wrappers

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import rl.callbacks

import pokerenv

ENV_NAME = 'PokerEnv-v1'
env = gym.make(ENV_NAME)

np.random.seed(123)
env.seed(123)

nb_actions = 32

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

nb_pre_episodes = 0
nb_episodes = 10000

if nb_pre_episodes > 0:
    dqn.load_weights('weights/dqn_{}_weights_{}.h5f'.format(ENV_NAME, nb_pre_episodes))

if nb_episodes > 0:
    dqn.fit(env, nb_steps=nb_episodes, visualize=True, verbose=1, log_interval=1000)
    dqn.save_weights('weights/dqn_{}_weights_{}.h5f'.format(ENV_NAME, nb_episodes + nb_pre_episodes), overwrite=True)

class EpisodeAccumulator(rl.callbacks.Callback):
    def __init__(self):
        self.reward_sum = 0
        self.episode_count = 0

    def on_episode_end(self, episode, logs={}):
        self.reward_sum += logs['episode_reward']
        self.episode_count += 1

    def reward_average(self):
        return self.reward_sum / self.episode_count

accumulator = EpisodeAccumulator()

dqn.test(env, nb_episodes=10000, visualize=True, callbacks=[accumulator])

print("avarage score: " + str(accumulator.reward_average()))



