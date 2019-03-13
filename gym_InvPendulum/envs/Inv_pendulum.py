ddimport gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from os import path
import random



class InvPendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.max_theta = np.pi / 8  # rad
        self.max_thetadot = 0.5     # rad/sec
        self.max_torque = 120       # N-m
        self.dt = 0.01
        self.viewer = None

        #bounds = np.array([self.max_theta, self.max_thetadot])

        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0, -np.sin(self.max_theta), -self.max_thetadot]), high=np.array([np.cos(self.max_theta), np.sin(self.max_theta), self.max_thetadot]), dtype=np.float32)
        self.seed()

    def seed(self, seed=None):
        _, seed = seeding.np_random(seed)
        return [seed]

    def step(self, tor):
        th, thdot = self.state
        tor_prev = self.action      # Action at time t-1

        g = 9.8             # acceleration due to gravity
        m = 65              # Mass
        l = 1.1             # length
        dt = self.dt        # Time step
        a = 0.83            # Filtering factor
        b = 0.8             # damping constant
        k = 8               # stiffness constant
        c = np.sqrt(40)     # noise amplitude

        tor_con = np.clip(tor, -self.max_torque, self.max_torque)[0] + c*np.random.normal(0, 1, 1)
        # Torque applied by the controller with additive white gaussian noise

        tor_t = a * tor_con + (1 - a) * tor_prev
        # Torque at time t with filtering

        I = m * (l ** 2)
        # Moment of Inertia

        newthdot = thdot + (tor_t + m * g * l * np.sin(th) - b * thdot - k * thdot) / I * dt
        # dynamical equation solved by euler method

        newth = th + newthdot * dt

        newthdot = np.clip(newthdot, -self.max_thetadot, self.max_thetadot)
        #Clipping the value of angular velocity

        self.state = np.array([newth, newthdot])

        self.action = tor_t

        if newth > np.pi/8 or newth < -np.pi/8:
            newth, newthdot = self.reset()

        if 0.0078 > newthdot > -0.0078 and 2.9504e4 > newth > -2.9504e-4:
            reward = 1
        else:
            reward = 0

        return self._get_obs(), reward, False, {}

    def reset(self):
        init_th = ((random.random() - 0.5) * 2) * 5
        init_thr = init_th * np.pi / 180
        init_thdotr = ((random.random() - 0.5) * 2) * 0.0625
        self.state = np.array([init_thr, init_thdotr])
        self.action = 0
        return self._get_obs()

    def _get_obs(self):
        th, thdot = self.state
        return np.array([np.cos(th), np.sin(th), thdot])


    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)

            rod = rendering.make_capsule(1, .2)
            rod.set_color(.3, .3, .8)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)

            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)

            fname = path.join(path.dirname(__file__), "clockwise.png")
            self.img = rendering.Image(fname, 0.5, 0.5)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.action != 0:
            self.imgtrans.scale = (-self.action / 2, np.abs(self.action) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None





