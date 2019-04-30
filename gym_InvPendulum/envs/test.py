import numpy as np
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import os
import os.path
import json
import subprocess
import tempfile
import distutils.spawn,distutils.version
from six import StringIO
import six
from gym import error, logger
from Inv_pendulum import InvPendulumEnv

metadata_ = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }


def render(mode='human'):
    from gym.envs.classic_control import rendering
    viewer = rendering.Viewer(500, 500)
    viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)

    surface = rendering.Line(start=(-1.2, -0.05), end=(1.2, -0.05))

    viewer.add_geom(surface)

    hide = rendering.FilledPolygon([(-2.2, -0.05), (-2.2,-2.2), (2.2,-2.2), (2.2,0.05)])
    hide.set_color(0.4,0.5,.9)
    #viewer.add_geom(hide)

    bob = rendering.make_circle(0.15,filled=True)
    bob.set_color(.8,.3,.2)
    attributes = rendering.Transform(translation=(0.0,1.0), rotation=np.pi/6)
    bob.add_attr(attributes)
    #viewer.add_geom(bob)
    #viewer.draw_circle(0.5,30,filled=True).set_color(0.1,0.4,.5)
    #viewer.draw_line(0,3)

    support = rendering.FilledPolygon([(0.15, 0.05), (0.15,-0.05), (-0.15, -0.05), (-0.15, 0.05)])
    support.set_color(0.7,0.7,0.7)
    #viewer.add_geom(support)

    rod = rendering.FilledPolygon([(-0.025,0), (-0.025,1.0-0.15), (0.025,1.0 - 0.15), (0.025,0)])
    rod.set_color(0.2, 0.2, 0.7)

    pendulum = rendering.Compound([bob, rod])
    pendulum.set_color(0.4, 0.5, 1)
    translate = rendering.Transform(translation=(0.0,-0.05))
    pendulum.add_attr(translate)
    viewer.add_geom(pendulum)

    axle_fill = rendering.make_circle(radius=.1,res=30,filled=True)
    axle_fill.set_color(1,1,1)

    axle = rendering.make_circle(radius=0.1, res=30, filled=False)
    semi = rendering.Transform(translation=(0.0, -0.05))
    axle_fill.add_attr(semi)
    axle.add_attr(semi)
    axle.set_color(0, 0, 0)

    viewer.add_geom(axle_fill)
    viewer.add_geom(axle)

    pivot = rendering.make_circle(0.02, filled=True)
    viewer.add_geom(pivot)

    hide = rendering.FilledPolygon([(-2.2, -0.07), (-2.2, -2.2), (2.2, -2.2), (2.2, -0.07)])
    hide.set_color(1, 1, 1)
    viewer.add_geom(hide)

    return viewer.render(return_rgb_array=mode == 'rgb_array')


#render()

import h5py
import h5df

from keras.models import model_from_json

Xnew = [np.array([[[0.29466096, 0.30317302]]])]

json_file = open('default_reward_wo_tor.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

#loaded_model.predict(Xnew)

# load weights into new model
loaded_model.load_weights("default_reward_wo_tor.h5")
#array = loaded_model.get_weights
weights = np.array(loaded_model.get_weights())


#print(loaded_model.predict(Xnew))

#print("Loaded model from disk")
#print(weights.shape)


def relu(output):
    for i in range(len(output)):
        if output[i] >= 0:
            pass
        else:
            output[i] = 0
    return output


def actor(obs, weights):
    op_1 = np.dot(weights[0].transpose(), obs)           # wx + b
    op_1 = relu(op_1 + weights[1])                     # relu activation

    op_2 = np.dot(weights[2].transpose(), op_1)
    op_2 = relu(op_2 + weights[3])

    op_3 = np.dot(weights[4].transpose(), op_2)
    op_3 = relu(op_3 + weights[5])

    op_4 = np.dot(weights[6].transpose(), op_3)
    op_4 = (op_4 + weights[7])

    return op_4


#action = actor(Xnew[0], weights)

#print(actor(Xnew[0], weights))

X = np.array([0.29466096, 0.30317302])
#print(actor(X,weights))

env = gym.make('Inverted_Pendulum-v0')


def play_og(env, act, stochastic, video_path):
    num_episodes = 0
    video_recorder = None
    video_recorder = VideoRecorder(
        env, video_path, enabled=video_path is not None)
    obs = env.reset()
    while True:
        env.unwrapped.render()
        video_recorder.capture_frame()
        action = act(np.array(obs)[None], stochastic=stochastic)[0]
        obs, rew, done, info = env.step(action)
        if done:
            obs = env.reset()
        if len(info["rewards"]) > num_episodes:
            if len(info["rewards"]) == 1 and video_recorder.enabled:
                # save video of first episode
                print("Saved video.")
                video_recorder.close()
                video_recorder.enabled = False
            print(info["rewards"][-1])
            num_episodes = len(info["rewards"])


videos = ["test.mp4"]

for i in range(10):
    videos.append("test"+ str(i) + ".mp4")

theta = []
theta_dot = []
actions = []

sin_theta = []
cos_theta = []
theta_dot = []


def play(env, model, video_path, num_episodes, timesteps, metadata):
    for i_episodes in range(num_episodes):
        video_recorder = VideoRecorder(
            env=env, path=video_path, metadata=metadata, enabled=video_path is not None)
        obs = env.reset()
        for t in range(timesteps):
            obs = [np.array([[list(obs)]])]
            video_recorder.capture_frame()
            action = model.predict(obs)[0]
            obs, rew, done, info = env.step(action)
            env.render()
            theta.append(obs[0])
            theta_dot.append(obs[1])
            actions.append(action[0])
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


import matplotlib.pyplot as plt


plt.plot(play(env, loaded_model, "D_reward_no_tor_5.mp4", 1, 300, metadata_))
plt.show()






