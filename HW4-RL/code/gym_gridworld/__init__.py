from gym.envs.registration import register

register(
    id='gridworld-v0',
    entry_point='gym_gridworld.envs:GridworldEnv',
    kwargs={'map_file': 'plan0.txt'},
)
register(
    id='gridworldnoisy-v0',
    entry_point='gym_gridworld.envs:GridworldEnv',
    kwargs={'map_file': 'plan0.txt', 'transition_noise': 0.2},
)
register(
    id='gridworld-v1',
    entry_point='gym_gridworld.envs:GridworldEnv',
    kwargs={'map_file': 'plan1.txt'},
)
register(
    id='gridworldnoisy-v1',
    entry_point='gym_gridworld.envs:GridworldEnv',
    kwargs={'map_file': 'plan1.txt', 'transition_noise': 0.2},
)
