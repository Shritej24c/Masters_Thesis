import os
import numpy as np
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import matplotlib.pyplot as plt
import Inv_pendulum
from stable_baselines import DDPG, TRPO
from stable_baselines.common.vec_env import DummyVecEnv
from scipy.fftpack import fft


import sys
import pkg_resources

import stable_baselines

# Fix for breaking change for DDPG buffer in v2.6.0
if pkg_resources.get_distribution("stable_baselines").version >= "2.6.0":
    sys.modules['stable_baselines.ddpg.memory'] = stable_baselines.deepq.replay_buffer
    stable_baselines.deepq.replay_buffer.Memory = stable_baselines.deepq.replay_buffer.ReplayBuffer


metadata_ = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }


env_test = gym.make('Inverted_Pendulum-v0')
env_ = DummyVecEnv([lambda: env_test])  # The algorithms require a vectorized environment to run

obs_ = env_.reset()


print("obs", obs_)

directory = "Pickle models/"
graph_directory = "Sway/"

models = os.listdir(directory)
model_ = DDPG.load(directory + models[4])
l = 110
theta = []
theta_dot = []
actions = []


print("Number of pretrained models = {}".format(len(models)))

print("Tested model is" + models[4])


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


fig, ax = plt.subplots(2, 1)

ax[0].plot(plot_(env=env_, model=model_, timesteps=1000))
ax[0].set(ylabel="Sway (cm)",  xlabel="Timesteps (dt = 0.01 secs)")
ax[0].set_axisbelow(True)
ax[0].minorticks_on()
ax[0].grid(which='major', linestyle="-", linewidth='0.5', color='red')
ax[0].grid(which='minor', linestyle=":", linewidth='0.5', color='blue')
# Turn off the display of all ticks.
ax[0].tick_params(which='both',  # Options for both major and minor ticks
                   top='off',  # turn off top ticks
                   left='off',  # turn off left ticks
                   right='off',  # turn off right ticks
                   bottom='off')  # turn off bottom ticks


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

cop = cop - np.mean(cop)
abs_ = fft(cop)
freq = np.fft.fftfreq(cop.size, 0.01)
cop1 = np.abs(abs_)
ax[1].plot(freq, cop1)
ax[1].set_axisbelow(True)
ax[1].minorticks_on()
ax[1].grid(which='major', linestyle="-", linewidth='0.5', color='red')
ax[1].grid(which='minor', linestyle=":", linewidth='0.5', color='blue')
# Turn off the display of all ticks.
ax[1].tick_params(which='both',  # Options for both major and minor ticks
                   top='off',  # turn off top ticks
                   left='off',  # turn off left ticks
                   right='off',  # turn off right ticks
                   bottom='off')  # turn off bottom ticks
plt.xlabel("Freq (Hz)")
plt.ylabel("|Y(freq)|")
plt.xlim(-3, 3)
plt.savefig("Model 5_1.png")
plt.show()


'''

for x in range(len(models)-2):
    model_ = DDPG.load(directory + models[x])
    l = 110

    theta = []
    theta_dot = []
    actions = []
    fig, ax = plt.subplots(2, 1)

    ax[0].plot(plot_(env=env_, model=model_, timesteps=1000))
    ax[0].set(ylabel="Sway (cm)", xlabel="Timesteps (dt = 0.01 secs)")
    ax[0].set_axisbelow(True)
    ax[0].minorticks_o n()
    ax[0].grid(which='major', linestyle="-", linewidth='0.5', color='red')
    ax[0].grid(which='minor', linestyle=":", linewidth='0.5', color='blue')
    # Turn off the display of all ticks.
    ax[0].tick_params(which='both',  # Options for both major and minor ticks
                   top='off',  # turn off top ticks
                   left='off',  # turn off left ticks
                   right='off',  # turn off right ticks
                   bottom='off')  # turn off bottom ticks
    cop = cop - np.mean(cop)
    abs_ = fft(cop)
    freq = np.fft.fftfreq(cop.size, 0.01)
    cop1 = np.abs(abs_)
    ax[1].plot(freq, cop1)
    ax[1].set(ylabel="|Y(freq)|", xlabel="Freq (Hz)")
    ax[1].set_axisbelow(True)
    ax[1].minorticks_on()
    ax[1].grid(which='major', linestyle="-", linewidth='0.5', color='red')
    ax[1].grid(which='minor', linestyle=":", linewidth='0.5', color='blue')
    # Turn off the display of all ticks.
    ax[1].tick_params(which='both',  # Options for both major and minor ticks
                   top='off',  # turn off top ticks
                   left='off',  # turn off left ticks
                   right='off',  # turn off right ticks
                   bottom='off')  # turn off bottom ticks
    plt.xlim(-3, 3)
    plt.savefig(graph_directory + "Model {}.png".format(x+1))
    plt.show()

'''