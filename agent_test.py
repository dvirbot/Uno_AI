# Imports from ai (from this project), os, stable_baselines3, sb3_contrib, numpy, and random (all open source libraries)
from ai import *
import os
import stable_baselines3.common.callbacks as callbacks
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
import numpy as np
import random

# A masking with the correct naming for the sb3_contrib model to use
def mask_fn(env: UnoAIEnvironment):
    return env.action_mask()

# Sets up the environment and load the model that is to be tested
env = UnoAIEnvironment()
env = ActionMasker(env, mask_fn)  # Wrap to enable masking

model_path = os.path.join("Training", "Uno_Model_MaskablePPO_10M")
model = MaskablePPO.load(model_path)
print(model.policy)

# Sets the number of games to test for and initializes other required variables
episodes = 100000
action_masks = []
score = 0

# Runs episodes number of games and prints the total reward and cumulative win rate of the ai every 100 games.
for episode in range(1, episodes + 1):
    obs = env.reset()
    done = False
    while not done:
        action_masks = env.action_masks()
        action = model.predict(obs, action_masks=action_masks)[0]
        obs, reward, done, info = env.step(action)
        score += reward
    if episode % 100 == 0:
        print(f"Episode: {episode}, Score: {score}, Cumulative Win Rate: {(50*score)/episode + 50}%")
env.close()
