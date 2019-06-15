import numpy as np
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import matplotlib.pyplot as plt
import Inv_pendulum
from stable_baselines import DDPG
from stable_baselines.common.vec_env import DummyVecEnv


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

model_ = DDPG.load("DDPG_baselines")


theta = []
theta_dot = []
actions = []


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


plt.plot(np.array(play(env_, model_, "DDPG_nd.mp4", 1, 1000, metadata_)))
plt.show()

