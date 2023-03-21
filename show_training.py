import gym
import torch
from pathlib import Path
import datetime, os

from gym.wrappers import FrameStack

import gym_super_mario_bros as smb
from gym_super_mario_bros import actions

from nes_py.wrappers import JoypadSpace

# Files
from wrappers import SkipFrame, GrayScaleObservation, ResizeObservation
from agent import Mario


# Setting up the enviornment
env = smb.make("SuperMarioBros-1-1-v0", render_mode = 'human', apply_api_compatibility = True)

env = JoypadSpace(env, smb.actions.SIMPLE_MOVEMENT)
env = SkipFrame(env, skip = 4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape = 84)
env = FrameStack(env, num_stack = 4)
env.reset()

# Enabling saving the output for graphs and updated nn structure
save_dir = None

# Loading the checkpoint
checkpoint = Path('mario_net_3.chkpt')
# Initializing mario and the logger
mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint = checkpoint)
mario.exploration_rate = mario.exploration_rate_min

# Starting the game
episodes = 40000
for episode in range(episodes):

    state = env.reset()

    # Play the game!
    while True:

        # Run agent on the state
        action = mario.act(state)

        # Agent performs action
        next_state, reward, done, trunc, info = env.step(action)

        # Update state
        state = next_state

        # Check if end of game
        if done or info["flag_get"]:
            break
