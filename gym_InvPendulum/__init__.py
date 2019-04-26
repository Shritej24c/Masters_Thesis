from gym.envs.registration import register

register(
    id='Inverted_Pendulum-v0',
    entry_point='gym_InvPendulum.envs.Inv_pendulum:InvPendulumEnv'

)
