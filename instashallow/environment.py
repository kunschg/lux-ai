from typing import Callable, Dict

import gym
import copy
import numpy as np
import numpy.typing as npt

from luxai_s2.env import LuxAI_S2
from luxai_s2.state import ObservationStateDict, StatsStateDict
from luxai_s2.unit import BidActionType, FactoryPlacementActionType
from luxai_s2.utils import my_turn_to_place_factory

from luxai_s2.wrappers.controllers import Controller

class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, full_pow=0) -> None:
        """Adds a custom reward and turns the LuxAI_S2 environment into a single-agent environment for easy training"""
        super().__init__(env)
        self.prev_step_metrics = None
        self.full_pow = full_pow

    def step(self, action):
        agent = "player_0"
        opp_agent = "player_1"

        opp_factories = self.env.state.factories[opp_agent]
        for k in opp_factories.keys():
            factory = opp_factories[k]
            # set enemy factories to have 1000 water to keep them alive the whole around and treat the game as single-agent
            factory.cargo.water = 1000

        # submit actions for just one agent to make it single-agent
        # and save single-agent versions of the data below
        action = {agent: action}
        obs, _, done, info = self.env.step(action)
        obs = obs[agent]
        done = done[agent]
        
        stats: StatsStateDict = self.env.state.stats[agent]

        info = dict()
        metrics = dict()
        metrics["ice_dug"] = (stats["generation"]["ice"]["HEAVY"] + stats["generation"]["ice"]["LIGHT"])
        metrics["water_produced"] = stats["generation"]["water"]

        metrics["action_queue_updates_success"] = stats["action_queue_updates_success"]
        metrics["action_queue_updates_total"] = stats["action_queue_updates_total"]

        #metrics["ice_fullness"] =  -np.power(obs[4], self.full_pow)

        reward = 0
        if self.prev_step_metrics is not None:
            # we check how much ice and water is produced and reward the agent for generating both
            ice_dug_this_step = metrics["ice_dug"] - self.prev_step_metrics["ice_dug"]
            water_produced_this_step = (metrics["water_produced"] - self.prev_step_metrics["water_produced"])
            
            # we reward water production more as it is the most important resource for survival
            reward = ice_dug_this_step / 100 + water_produced_this_step 
            #reward += metrics["ice_fullness"]
        
        # we can save the metrics to info so we can use tensorboard to log them to get a glimpse into how our agent is behaving
        info["metrics"] = metrics

        self.prev_step_metrics = copy.deepcopy(metrics)
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)["player_0"]
        self.prev_step_metrics = None
        return obs


class StartWrapper(gym.Wrapper):
    def __init__(self, env: LuxAI_S2, bid_policy, factory_placement_policy, controller: Controller = None) -> None:

        gym.Wrapper.__init__(self, env)
        self.env = env
        
        assert controller is not None
        assert bid_policy is not None
        assert factory_placement_policy is not None

        self.controller = controller
        self.action_space = controller.action_space
        self.bid_policy = bid_policy
        self.factory_placement_policy = factory_placement_policy
        self.prev_obs = None

    def step(self, action: Dict[str, npt.NDArray]):
        
        lux_action = dict()
        for agent in self.env.agents:
            if agent in action:
                lux_action[agent] = self.controller.action_to_lux_action(agent=agent, obs=self.prev_obs, action=action[agent])
            else:
                lux_action[agent] = dict()
        
        obs, reward, done, info = self.env.step(lux_action)
        self.prev_obs = obs
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        
        # Phase 1
        action = dict()
        for agent in self.env.agents:
            action[agent] = self.bid_policy(agent, obs[agent])
        obs, _, _, _ = self.env.step(action)
        
        # Phase 2
        while self.env.state.real_env_steps < 0:
            action = dict()
            for agent in self.env.agents:
                if my_turn_to_place_factory(
                    obs["player_0"]["teams"][agent]["place_first"],
                    self.env.state.env_steps,
                ):
                    action[agent] = self.factory_placement_policy(agent, obs[agent])
                else:
                    action[agent] = dict()
            obs, _, _, _ = self.env.step(action)
        self.prev_obs = obs
        
        # Phase 3 from now on
        return obs

class InvalidActionWrapper(StartWrapper):
    def __init__(self, env: LuxAI_S2, bid_policy, factory_placement_policy, controller: Controller = None) -> None:
        super().__init__(env, bid_policy, factory_placement_policy, controller)
    
    def action_masks(self):
        return self.controller.action_masks('player_0', self.prev_obs)
