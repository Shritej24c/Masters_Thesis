import numpy as np
import gym

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, ELU, BatchNormalization
from keras.optimizers import Adam, SGD
from keras import backend as K

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess


from Inv_pendulum import InvPendulumEnv
from Test_Env import PendulumEnv

Testenv = 'Test_Inv_pendulum-v0'
Env = 'Inverted_Pendulum-v0'
Og_Env = 'Pendulum-v0'

env = gym.make(Env)


# Get the environment and extract the number of actions.

np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(16, activation="relu"))
actor.add(Dense(16, activation="relu"))
actor.add(Dense(16, activation="relu"))
actor.add(Dense(nb_actions, activation="linear"))
print(actor.summary())


action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!

memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
agent.fit(env, nb_steps=60000, visualize=True, verbose=1, nb_max_episode_steps=300)

# After training is done, we save the final weights.
actor.save('No_filtering_step.h5', overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=300)

