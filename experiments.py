import os
import time
import json
import argparse

from sb3_contrib import MaskablePPO

from stable_baselines3.ppo import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from instashallow.utils import make_env
from instashallow.callbacks import TensorboardCallback

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default='todo')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_envs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=800)

    parser.add_argument('--train_max_ep_steps', type=int, default=200)
    parser.add_argument('--eval_max_ep_steps', type=int, default=1000)
    parser.add_argument('--total_timesteps', type=int, default=5_000_000)
    parser.add_argument('--rollout_steps', type=int, default=4000)

    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--target_kl', type=float, default=0.05)

    parser.add_argument('--full_pow', type=float, default=0)

    parser.add_argument('--invalid_action_masking', '-iam', action='store_true', default=False)

    args = parser.parse_args()
    params = vars(args)

    set_random_seed(params['seed'])
    log_path = "logs/" + params['exp_name'] + "-" + time.strftime("%d-%m-%Y_%H-%M-%S")
    os.makedirs(log_path)
    
    num_envs = params['num_envs']
    invalid_action_masking = params['invalid_action_masking']

    if invalid_action_masking:
        print("Running Invalid Action Masking")

    train_envs = [make_env("LuxAI_S2-v0", i, params['seed'], params['train_max_ep_steps']) for i in range(num_envs)]
    eval_envs = [make_env("LuxAI_S2-v0", i, params['seed'], params['eval_max_ep_steps']) for i in range(4)]

    env = DummyVecEnv(train_envs) if invalid_action_masking else SubprocVecEnv(train_envs)
    env.reset()

    eval_env = DummyVecEnv(eval_envs) if invalid_action_masking else SubprocVecEnv(eval_envs)
    eval_env.reset()
    
    policy_kwargs = dict(net_arch=(128, 128))
    model_params = {
            'policy' : "MlpPolicy",
            'env' : env,
            'n_steps': params['rollout_steps'] // num_envs,
            'batch_size': params['batch_size'],
            'learning_rate': params['learning_rate'],
            'policy_kwargs': policy_kwargs,
            'verbose': 1,
            'n_epochs': 2,
            'target_kl': params['target_kl'],
            'gamma': params['gamma'],
            'tensorboard_log': os.path.join(log_path)
        }

    model = MaskablePPO(**model_params) if invalid_action_masking else PPO(**model_params)

    print("Training model: {}".format(type(model)))

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_path, "models"),
        log_path=os.path.join(log_path, "eval_logs"),
        eval_freq=24_000,
        deterministic=False,
        render=False,
        n_eval_episodes=5,
    )

    with open(os.path.join(log_path, "params.json"), "w") as write_file:
        json.dump(params, write_file, indent=4, sort_keys=True)
        print("Saved parameters in params.json")

    model.learn(params['total_timesteps'], callback=[TensorboardCallback(tag="train_metrics"), eval_callback])
    model.save(os.path.join(log_path, "models/latest_model"))
