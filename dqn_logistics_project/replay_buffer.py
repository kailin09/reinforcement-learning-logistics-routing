import random
import numpy as np
from collections import deque


class ReplayBuffer:

    def __init__(self, capacity=10000):

        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):

        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)

        s, a, r, s2, d = zip(*batch)

        return (
            np.array(s),
            np.array(a),
            np.array(r),
            np.array(s2),
            np.array(d),
        )

    def __len__(self):

        return len(self.buffer)