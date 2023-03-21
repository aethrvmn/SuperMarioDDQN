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
from logger import MetricLogger


# Setting up the enviornment
env = smb.make("SuperMarioBros-1-1-v0", render_mode = None, apply_api_compatibility = True)

env = JoypadSpace(env, smb.actions.SIMPLE_MOVEMENT)
env.reset()

next_state, reward, done, trunc, info = env.step(action=0)
print(f"{next_state.shape},\n {reward},\n {done},\n {info}")

env = SkipFrame(env, skip = 4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape = 84)
env = FrameStack(env, num_stack = 4)

# Enabling saving the output for graphs and updated nn structure
save_dir = save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

# Loading the checkpoint
checkpoint = Path('mario_net.chkpt')
# Initializing mario and the logger
mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint = checkpoint)

logger = MetricLogger(save_dir)

use_cuda = torch.cuda.is_available()
use_mps = torch.backends.mps.is_available()
print(f"CUDA is available: {use_cuda}")
print(f"MPS is available: {use_mps}")
print(f"Using {mario.device}")

# Starting the game
episodes = 10000
for episode in range(episodes):

    state = env.reset()

    # Play the game!
    while True:

        # Run agent on the state
        action = mario.act(state)

        # Agent performs action
        next_state, reward, done, trunc, info = env.step(action)

        # Remember
        mario.cache(state, next_state, action, reward, done)

        # Learn
        q, loss = mario.learn()

        # Logging
        logger.log_step(reward, loss, q)

        # Update state
        state = next_state

        # Check if end of game
        if done or info["flag_get"]:
            break

    logger.log_episode()

    if episode % 20 == 0:
        logger.record(episode=episode, epsilon=mario.exploration_rate, step=mario.curr_step)
    #This ensures mario is saved at the end of training
    if episode % episodes == 0 or episode == episodes-1:
        mario.save()
