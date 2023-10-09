
import os.path as osp
import sys
import numpy as np
import torch as th
from stable_baselines3.ppo import PPO
from lux.config import EnvConfig

from instashallow.actions import SimpleUnitDiscreteController
from instashallow.observations  import SimpleUnitObservationWrapper

from instashallow.phase1 import zero_bid
from instashallow.phase2 import place_near_random_ice, place_best_ice

#from stable_baselines3.common.type_aliases import Schedule

MODEL_WEIGHTS_RELATIVE_PATH = "./best_model"
FACTORY_POLICY_FUNC = place_best_ice
BIDDING_POLICY = zero_bid

class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg

        directory = osp.dirname(__file__)
        self.policy = PPO.load(osp.join(directory, MODEL_WEIGHTS_RELATIVE_PATH))
        #, custom_objects={'lr_schedule': Schedule, 'clip_range': Schedule})
        
        self.controller = SimpleUnitDiscreteController(self.env_cfg)

    def bid_policy(self, step: int, obs, remainingOverageTime: int = 60):
        return BIDDING_POLICY(self.player, obs)

    def factory_placement_policy(self, step: int, obs, remainingOverageTime: int = 60):
        return FACTORY_POLICY_FUNC(self.player, obs)

    def act(self, step: int, obs, remainingOverageTime: int = 60):

        raw_obs = dict(player_0=obs, player_1=obs)
        obs = SimpleUnitObservationWrapper.convert_obs(raw_obs, env_cfg=self.env_cfg)
        obs = obs[self.player]

        obs = th.from_numpy(obs).float()
        with th.no_grad():

            action_mask = (th.from_numpy(self.controller.action_masks(self.player, raw_obs)).unsqueeze(0).bool())
            
            # Invalid Action Masking
            features = self.policy.policy.features_extractor(obs.unsqueeze(0))
            x = self.policy.policy.mlp_extractor.shared_net(features)
            logits = self.policy.policy.action_net(x)

            logits[~action_mask] = -1e8 # mask out invalid actions
            dist = th.distributions.Categorical(logits=logits)
            actions = dist.sample().cpu().numpy() # shape (1, 1)

        lux_action = self.controller.action_to_lux_action(self.player, raw_obs, actions[0])

        shared_obs = raw_obs[self.player]
        factories = shared_obs["factories"][self.player]
        for unit_id in factories.keys():
            factory = factories[unit_id]
            
            if 1000 - step < factory["cargo"]["water"] and step > 400 and factory["cargo"]["water"] > 50:
            #if factory["cargo"]["water"] > 250:
                lux_action[unit_id] = 2

        return lux_action