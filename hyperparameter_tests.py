import os

from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

from ai import *
import stable_baselines3.common.callbacks as callbacks
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
import numpy as np
import random
import cProfile
import optuna


def mask_fn(env: UnoAIEnvironment):
    return env.action_mask()

def optimize_maskable_ppo(trial):
    """ Learning hyperparamters we want to optimise"""
    return {
        'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
        # 'cliprange': trial.suggest_uniform('cliprange', 0.1, 0.4),
        # 'noptepochs': int(trial.suggest_loguniform('noptepochs', 1, 48)),
        # 'lam': trial.suggest_uniform('lam', 0.8, 1.),
    }


def optimize_agent(trial):
    """ Train the model and optimize
        Optuna maximises the negative log likelihood, so we
        need to negate the reward here
    """
    model_params = optimize_maskable_ppo(trial)
    env = UnoAIEnvironment()
    env = ActionMasker(env, mask_fn)  # Wrap to enable masking
    n_layers_value = trial.suggest_int('n_layers_value', 1, 3)
    n_layers_policy = trial.suggest_int('n_layers_policy', 1, 3)
    policy_kwargs = dict(net_arch=[trial.suggest_int("first_layer", 32, 512, log=True),
                                   dict(pi=[trial.suggest_int('policy_layers', 32, 512, log=True) for i in range(n_layers_policy)],
                                        vf=[trial.suggest_int('value_layers', 32, 512, log=True) for i in range(n_layers_value)])])
    model = MaskablePPO('MlpPolicy', env, verbose=0, policy_kwargs=policy_kwargs, **model_params)
    model.learn(10000)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)

    return -1 * mean_reward

study = optuna.create_study()
study.optimize(optimize_agent, n_trials=100)

print(study.best_params)








# cProfile.run("model.learn(total_timesteps=20000)", sort="tottime")