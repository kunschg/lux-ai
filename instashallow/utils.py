import gym
from gym.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from instashallow.phase1 import zero_bid
from instashallow.phase2 import place_near_random_ice, place_best_ice
from instashallow.actions import SimpleUnitDiscreteController
from instashallow.observations import SimpleUnitObservationWrapper
from instashallow.environment import InvalidActionWrapper, CustomEnvWrapper, StartWrapper

def make_env(env_id: str, rank: int, seed: int, max_episode_steps=100):
    def _init() -> gym.Env:
        
        env = gym.make(env_id, verbose=0, collect_stats=True, MAX_FACTORIES=2)

        #env = StartWrapper(env,
        #    factory_placement_policy=place_near_random_ice,
        #    controller=SimpleUnitDiscreteController(env.env_cfg),
        #)

        env = InvalidActionWrapper(env, 
            bid_policy=zero_bid,
            factory_placement_policy=place_best_ice,
            controller=SimpleUnitDiscreteController(env.env_cfg),
        )

        env = SimpleUnitObservationWrapper(env)
        
        env = CustomEnvWrapper(env)  
        
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        set_random_seed(seed + rank)
        return env

    return _init
