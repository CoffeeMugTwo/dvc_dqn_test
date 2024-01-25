"""Module containing the training logic.
"""
import random
import math
from itertools import count

import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim

import dvc.api

from replay_memory import ReplayMemory
from replay_memory import Transition
from model import DQN

# Get Parameter
PARAM_DICT = dvc.api.params_show()

EPS_START = PARAM_DICT["training"]["eps_start"]
EPS_END = PARAM_DICT["training"]["eps_end"]
EPS_DECAY = PARAM_DICT["training"]["eps_decay"]

BATCH_SIZE = PARAM_DICT["training"]["batch_size"]
GAMMA = PARAM_DICT["training"]["gamma"]
TAU = PARAM_DICT["training"]["tau"]
LR = PARAM_DICT["training"]["lr"]

REPLAY_MEMORY_SIZE = PARAM_DICT["training"]["memory_size"]

N_EPISODES = PARAM_DICT["training"]["n_episodes"]

# Get correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define environment etc.
env = gym.make("MountainCar-v0")
n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

# Define policy and target net
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

# Miscellaneous
optimizer = optim.Adam(
    policy_net.parameters(),
    lr=LR,
    amsgrad=True
)
memory = ReplayMemory(REPLAY_MEMORY_SIZE)
steps_done = 0
episode_durations = []
rewards_list = []


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor(
            [[env.action_space.sample()]],
            device=device,
            dtype=torch.long
        )
    

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(
            map(
                lambda s: s is not None,
                batch.next_state
            )
        ),
        device=device,
        dtype=torch.bool
    )

    non_final_next_states = torch.cat(
        [s for s in batch.next_state if s is not None]
    )

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # print(f"{expected_state_action_values=}")
    # print(f"{state_action_values=}")
    # print(f"{reward_batch=}")

    criterion = nn.SmoothL1Loss()
    loss = criterion(
        state_action_values,
        expected_state_action_values.unsqueeze(1)
    )

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def main():

    for i_episode in range(N_EPISODES):
        # Initialize the environment and get it's state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        min_distance = 99
        for t in count():
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            distance_to_goal = np.abs(0.5 - observation[0])
            reward = torch.tensor([reward], device=device)
            if distance_to_goal < min_distance:
                min_distance = distance_to_goal
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                # episode_durations.append(t + 1)
                rewards_list.append(min_distance)

                # debug
                print(f"Episode {i_episode} | Min. Distance = {min_distance}")
                # plot_rewards()
                break


if __name__ == "__main__":
    main()