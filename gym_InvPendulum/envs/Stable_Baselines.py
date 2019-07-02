import gym
import pandas as pd
import numpy as np

from stable_baselines.ddpg.policies import MlpPolicy
#from stable_baselines.common.policies import MlpPolicy
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
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.90) * np.ones(n_actions), theta=0.2)

model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)

#model = TRPO(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=400000)

#model.save("ddpg_nd_f")



