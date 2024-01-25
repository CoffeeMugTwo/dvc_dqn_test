"""Module implementing the replay memory.
"""

import random
from collections import namedtuple
from collections import deque


Transition = namedtuple(
    "Transition",
    ("state", "action", "next_state", "reward")
)

class ReplayMemory(object):
    """Replay memory class"""

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
