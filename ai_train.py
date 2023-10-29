# Imports from ai (from this project), os, stable_baselines3, sb3_contrib, numpy, and random (all open source libraries)
from ai import *
import os
import stable_baselines3.common.callbacks as callbacks
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
import numpy as np
import random
import cProfile

# A masking with the correct naming for the sb3_contrib model to use
def mask_fn(env: UnoAIEnvironment):
    return env.action_mask()

# Sets up the environment
env = UnoAIEnvironment()
env = ActionMasker(env, mask_fn)  # Wrap to enable masking

# Sets up the neural network architecture
policy_kwargs = dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])])


# Defines the path for the training logs, which can be viewed using tensorboard
log_path = os.path.join("Training", "Logs")

# Defines the path that the model will be stored after training
save_path = os.path.join("Training", "Uno_Model_maker_portfolio_demonstration")


# def speed_distribution():
#     # Creates the model
#     model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=2, tensorboard_log=log_path,
#                         policy_kwargs=policy_kwargs, batch_size=2048)
#
#     # Makes it so the model saves every 500000 steps, so that if your computer crashes, you don't lose all your progress
#     # checkpoint = callbacks.CheckpointCallback(save_freq=500000, save_path=save_path, name_prefix="uno_model")
#
#     # Trains the model
#     model.learn(total_timesteps=65536)
#
#     # Saves the final model
#     model.save(save_path)

# Creates the model
model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=2, tensorboard_log=log_path,
                    policy_kwargs=policy_kwargs, batch_size=2048)
print(model.policy)

# Makes it so the model saves every 500000 steps, so that if your computer crashes, you don't lose all your progress
# checkpoint = callbacks.CheckpointCallback(save_freq=500000, save_path=save_path, name_prefix="uno_model")

# Trains the model
model.learn(total_timesteps=65536)

# Saves the final model
model.save(save_path)










# cProfile.run("speed_distribution()", sort="cumtime")
