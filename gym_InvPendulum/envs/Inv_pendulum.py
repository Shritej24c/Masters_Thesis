import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from os import path
import random
from gym.envs.registration import register


class InvPendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.max_theta = np.pi / 8  # rad
        self.max_thetadot = 0.5     # rad/sec
        self.max_torque = 300       # N-m
        self.dt = 0.01
        self.viewer = None

        bounds = np.array([self.max_theta, self.max_thetadot])

        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-bounds, high=bounds, dtype=np.float32)
        self.seed()

    def seed(self, seed=None):
        _, seed = seeding.np_random(seed)
        return [seed]

    def step(self, tor):

        th, thdot = self.state

        tor_prev = self.action  # Action at time t-1

        g = 9.8             # acceleration due to gravity
        m = 65              # Mass
        l = 1.1             # length
        dt = self.dt        # Time step
        a = 0.83            # Filtering factor
        b = 0.8             # damping constant
        k = 8               # stiffness constant
        c = np.sqrt(40)     # noise amplitude

        tor_con = np.clip(tor, -self.max_torque, self.max_torque)[0] + c*np.random.normal(0, 1, 1)[0]
        # Torque applied by the controller with additive white gaussian noise

        tor_t = a * tor_con + (1 - a) * tor_prev
        # Torque at time t with filtering

        I = m * (l ** 2)
        # Moment of Inertia

        newthdot = thdot + (tor_con + m * g * l * np.sin(th) - k*thdot - b*thdot) / I * dt
        # dynamical equation solved by euler method

        newth = th + newthdot * dt

        newthdot = np.clip(newthdot, -self.max_thetadot, self.max_thetadot)
        # Clipping the value of angular velocity

        self.state = np.array([newth, newthdot])

        self.action = tor_t

        done = bool(newth > np.pi/8 or newth < -np.pi/8)

        costs = th**2 + 0.1*thdot**2

        return self.state, -costs, done, {}

    def reset(self):
        init_th = ((random.random() - 0.5) * 2) * 5
        init_thr = init_th * np.pi / 180
        init_thdotr = ((random.random() - 0.5) * 2) * 0.0625
        self.state = np.array([init_thr, init_thdotr])
        self.action = 0
        return self.state

    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)        
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)

            surface = rendering.Line(start=(-1.2, -0.05), end=(1.2, -0.05))

            self.viewer.add_geom(surface)

            bob = rendering.make_circle(0.15, filled=True)
            bob.set_color(.8, .3, .2)
            attributes = rendering.Transform(translation=(0.0, 1.0))
            bob.add_attr(attributes)

            rod = rendering.FilledPolygon([(-0.025, 0), (-0.025, 1.0 - 0.15), (0.025, 1.0 - 0.15), (0.025, 0)])
            rod.set_color(0.2, 0.2, 0.7)

            pendulum = rendering.Compound([bob, rod])
            pendulum.set_color(0.4, 0.5, 1)
            translate = rendering.Transform(translation=(0.0, -0.05))
            pendulum.add_attr(translate)
            self.pole_transform = rendering.Transform()
            pendulum.add_attr(self.pole_transform)
            self.viewer.add_geom(pendulum)

            axle_fill = rendering.make_circle(radius=.1, res=30, filled=True)
            axle_fill.set_color(1, 1, 1)

            axle = rendering.make_circle(radius=0.1, res=30, filled=False)
            semi = rendering.Transform(translation=(0.0, -0.05))
            axle_fill.add_attr(semi)
            axle.add_attr(semi)
            axle.set_color(0, 0, 0)

            self.viewer.add_geom(axle_fill)
            self.viewer.add_geom(axle)

            pivot = rendering.make_circle(0.02, filled=True)
            self.viewer.add_geom(pivot)

            hide = rendering.FilledPolygon([(-2.2, -0.07), (-2.2, -2.2), (2.2, -2.2), (2.2, -0.07)])
            hide.set_color(1, 1, 1)
            self.viewer.add_geom(hide)

            fname = path.join(path.dirname(__file__), "clockwise.png")
            self.img = rendering.Image(fname, 0.5, 0.5)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0])
        if self.action != 0:
            self.imgtrans.scale = (-self.action / 8, np.abs(self.action) / 8)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


register(
    id='Inverted_Pendulum-v0',
    entry_point='Inv_pendulum:InvPendulumEnv',
    kwargs={}
)

