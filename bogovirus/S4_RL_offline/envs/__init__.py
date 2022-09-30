# module for a gym environment
# that wraps an off-line data set. 
from gym.envs.registration import register

register(
    id='BogoEnv-v0',
    entry_point='envs.bogo_world:BogoEnv',
    max_episode_steps=300
)