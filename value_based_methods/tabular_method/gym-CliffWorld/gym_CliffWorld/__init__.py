from gym.envs.registration import register

register(
    id='cliff_world-v0',
    entry_point='gym_CliffWorld.envs:CliffWorldEnv',
)
