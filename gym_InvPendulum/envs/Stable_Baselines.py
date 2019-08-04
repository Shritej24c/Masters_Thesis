import gym
import pandas as pd
import numpy as np
import os
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, DQN, DDPG, TRPO
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise, ActionNoise, AdaptiveParamNoiseSpec, NormalActionNoise

from Inv_pendulum import InvPendulumEnv
from Test_Env import PendulumEnv

Testenv = 'Test_Inv_pendulum-v0'
Env = 'Inverted_Pendulum-v0'
Og_Env = 'Pendulum-v0'

env = gym.make(Env)

env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = AdaptiveParamNoiseSpec(desired_action_stddev=.2)
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=.95)

model = DDPG(MlpPolicy, env, verbose=1, action_noise=action_noise)

model.learn(total_timesteps=400000)

directory = 'Pickle models/'

model_name = 'ddpg_param2_sig95_m56_a99_c20_k4'

model.save(directory + model_name)

models = os.listdir(directory)

print(models.index(model_name + '.pkl'))

