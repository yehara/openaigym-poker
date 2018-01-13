from gym.envs.registration import register

register(
    id='PokerEnv-v1',
    entry_point='pokerenv.pokerenv:PokerEnv'
)
