import numpy as np
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import matplotlib.pyplot as plt
import Inv_pendulum
from stable_baselines import DDPG, TRPO
from stable_baselines.common.vec_env import DummyVecEnv
'''
import sys
import pkg_resources

import stable_baselines

# Fix for breaking change for DDPG buffer in v2.6.0
if pkg_resources.get_distribution("stable_baselines").version >= "2.6.0":
    sys.modules['stable_baselines.ddpg.memory'] = stable_baselines.deepq.replay_buffer
    stable_baselines.deepq.replay_buffer.Memory = stable_baselines.deepq.replay_buffer.ReplayBuffer

'''
metadata_ = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }


Xnew = [np.array([[[0.29466096, 0.30317302]]])]


X = np.array([0.29466096, 0.30317302])


env_test = gym.make('Inverted_Pendulum-v0')
env_ = DummyVecEnv([lambda: env_test])  # The algorithms require a vectorized environment to run

obs_ = env_.reset()

print("Xnew", Xnew)
print("X",X)
print("obs", obs_)


a_model_name = "ddpg_sig95_buff"

model_ = DDPG.load(a_model_name)
l = 110

theta = []
theta_dot = []
actions = []


def plot_(env, model, timesteps):
    obs = env.reset()
    print(obs)
    for t in range(timesteps):
        action, states = model.predict(obs)
        obs, rew, done, info = env.step(action)
        theta.append(obs[0][0])
        theta_dot.append(obs[0][1])
        actions.append(action[0][0])
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
        elif t == timesteps - 1:
            print("The episode done with all the timesteps")
            break
    env.close()
    global cop
    cop = l*np.sin(np.array(theta))
    return cop

plt.su
plt.plot(plot_(env=env_, model=model_, timesteps=1000))
plt.show()


def play(env, model, video_path, num_episodes, timesteps, metadata):
    for i_episodes in range(num_episodes):
        video_recorder = VideoRecorder(
            env=env, path=video_path, metadata=metadata, enabled=video_path is not None)
        obs = env.reset()
        print(obs)
        for t in range(timesteps):
            video_recorder.capture_frame()
            action, states = model.predict(obs)
            obs, rew, done, info = env.step(action)
            env.render()
            theta.append(obs[0][0])
            theta_dot.append(obs[0][1])
            actions.append(action[0][0])
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                num_episodes += 1
                # save video of first episode
                print("Saved video.")
                video_recorder.close()
                video_recorder.enabled = False
                break
            elif t == timesteps-1:
                print("The episode done with all the timesteps")
                print("Saved video")
                video_recorder.close()
                video_recorder.enabled = False
                break
    env.close()
    return theta


#plt.plot(l*np.sin(np.array(play(env_, model_, "DDPG_c9(1).mp4", 1, 1000, metadata_))))
#plt.show()

from scipy.fftpack import fft
cop = cop - np.mean(cop)
abs_ = fft(cop)
freq = np.fft.fftfreq(cop.size, 0.01)
cop1 = np.abs(abs_)
plt.plot(freq, cop1)
plt.xlim(-5, 5)
plt.show()
