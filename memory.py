# code from openai
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

import random
from collections import deque
import operator
import numpy as np
from segment_tree_ctypes import SumSegmentTree, MinSegmentTree

class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


# class SumSegmentTree(SegmentTree):
#     def __init__(self, capacity):
#         super(SumSegmentTree, self).__init__(
#             capacity=capacity,
#             operation=operator.add,
#             neutral_element=0.0
#         )

#     def sum(self, start=0, end=None):
#         """Returns arr[start] + ... + arr[end]"""
#         return super(SumSegmentTree, self).reduce(start, end)

#     def find_prefixsum_idx(self, prefixsum):
#         """Find the highest index `i` in the array such that
#             sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
#         if array values are probabilities, this function
#         allows to sample indexes according to the discrete
#         probability efficiently.
#         Parameters
#         ----------
#         perfixsum: float
#             upperbound on the sum of array prefix
#         Returns
#         -------
#         idx: int
#             highest index satisfying the prefixsum constraint
#         """
#         assert 0 <= prefixsum <= self.sum() + 1e-5
#         idx = 1
#         while idx < self._capacity:  # while non-leaf
#             if self._value[2 * idx] > prefixsum:
#                 idx = 2 * idx
#             else:
#                 prefixsum -= self._value[2 * idx]
#                 idx = 2 * idx + 1
#         return idx - self._capacity


# class MinSegmentTree(SegmentTree):
#     def __init__(self, capacity):
#         super(MinSegmentTree, self).__init__(
#             capacity=capacity,
#             operation=min,
#             neutral_element=float('inf')
#         )

#     def min(self, start=0, end=None):
#         """Returns min(arr[start], ...,  arr[end])"""

#         return super(MinSegmentTree, self).reduce(start, end)


class ReplayBuffer(object):
    def __init__(self, size, obs_shape=(4, 84, 84)):
        """Create Replay buffer with pre-allocated NumPy arrays to prevent memory leaks.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer.
        obs_shape: tuple
            The shape of the observation (e.g., stacked Atari frames).
        """
        self._maxsize = size
        self._next_idx = 0
        self._size = 0
        
        # Pre-allocate exact memory limit
        self._states = np.zeros((size, *obs_shape), dtype=np.uint8)
        self._actions = np.zeros(size, dtype=np.int64)
        self._rewards = np.zeros(size, dtype=np.float32)
        self._next_states = np.zeros((size, *obs_shape), dtype=np.uint8)
        self._dones = np.zeros(size, dtype=np.float32)

    def __len__(self):
        return self._size

    def add(self, obs_t, action, reward, obs_tp1, done):
        idx = self._next_idx
        
        self._states[idx] = obs_t
        self._actions[idx] = action
        self._rewards[idx] = reward
        self._next_states[idx] = obs_tp1
        self._dones[idx] = done

        self._next_idx = (self._next_idx + 1) % self._maxsize
        self._size = min(self._size + 1, self._maxsize)

    def _encode_sample(self, idxes):
        # Slice directly from pre-allocated arrays (O(1) allocation)
        return (
            self._states[idxes],
            self._actions[idxes],
            self._rewards[idxes],
            self._next_states[idxes],
            self._dones[idxes]
        )

    def sample(self, batch_size):
        idxes = [random.randint(0, self._size - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha, obs_shape=(4, 84, 84)):
        super(PrioritizedReplayBuffer, self).__init__(size, obs_shape)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, self._size - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * self._size) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * self._size) ** (-beta)
            weights.append(weight / max_weight)
            
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < self._size
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


class CustomPrioritizedReplayBuffer(PrioritizedReplayBuffer):
    """
    Customized PrioritizedReplayBuffer class
    1. Edited add method to receive priority as input. This enables entering priority when adding sample.
    This efficiently merges two methods (add, update_priorities) which enables less shared memory lock.
    2. Uses pre-allocated numpy arrays via parent to eliminate LazyFrame caching memory explosions.
    """
    def __init__(self, size, alpha, obs_shape=(4, 84, 84)):
        super(CustomPrioritizedReplayBuffer, self).__init__(size, alpha, obs_shape)

    def add(self, state, action, reward, next_state, done, priority):
        idx = self._next_idx
        
        # Overwrite pre-allocated slots directly (No list appends)
        self._states[idx] = state
        self._actions[idx] = action
        self._rewards[idx] = reward
        self._next_states[idx] = next_state
        self._dones[idx] = done

        self._next_idx = (self._next_idx + 1) % self._maxsize
        self._size = min(self._size + 1, self._maxsize)

        self._it_sum[idx] = priority ** self._alpha
        self._it_min[idx] = priority ** self._alpha
        self._max_priority = max(self._max_priority, priority)

    def _encode_sample(self, idxes):
        # Slice directly from the pre-allocated arrays
        return (
            self._states[idxes],
            self._actions[idxes],
            self._rewards[idxes],
            self._next_states[idxes],
            self._dones[idxes]
        )


class BatchStorage:
    """
    Storage for actors to support multi-step learning and efficient priority calculation.
    Saving Q values with experiences enables td-error priority calculation
    without re-calculating Q-values for each state.
    """
    def __init__(self, n_steps, gamma=0.99):
        self.state_deque = deque(maxlen=n_steps)
        self.action_deque = deque(maxlen=n_steps)
        self.reward_deque = deque(maxlen=n_steps)
        self.q_values_deque = deque(maxlen=n_steps)
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.q_values = []
        self.next_q_values = []

        self.n_steps = n_steps
        self.gamma = gamma

    def add(self, state, reward, action, done, q_values):
        if len(self.state_deque) == self.n_steps or done:
            t0_state = self.state_deque[0]
            t0_reward = self.multi_step_reward(*self.reward_deque, reward)
            t0_action = self.action_deque[0]
            t0_q_values = self.q_values_deque[0]
            tp_n_state = state
            tp_n_q_values = q_values
            done = np.float32(done)
            self.states.append(t0_state)
            self.actions.append(t0_action)
            self.rewards.append(t0_reward)
            self.next_states.append(tp_n_state)
            self.dones.append(done)
            self.q_values.append(t0_q_values)
            self.next_q_values.append(tp_n_q_values)

        if done:
            self.state_deque.clear()
            self.reward_deque.clear()
            self.action_deque.clear()
        else:
            self.state_deque.append(state)
            self.reward_deque.append(reward)
            self.action_deque.append(action)
            self.q_values_deque.append(q_values)

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.q_values = []
        self.next_q_values = []

    def compute_priorities(self):
        # np.array(..., copy=False) is broken in NumPy 2.x — use np.asarray instead.
        actions = np.asarray(self.actions)
        rewards = np.asarray(self.rewards)
        dones = np.asarray(self.dones)
        q_values = np.stack(self.q_values)
        next_q_values = np.stack(self.next_q_values)

        q_a_values = q_values[(range(len(q_values)), actions)]
        next_q_a_values = next_q_values.max(1)
        expected_q_a_values = rewards + (self.gamma ** self.n_steps) * next_q_a_values * (1 - dones)
        td_error = expected_q_a_values - q_a_values
        prios = np.abs(td_error) + 1e-6
        return prios

    def make_batch(self):
        prios = self.compute_priorities()
        batch = [self.states, self.actions, self.rewards, self.next_states, self.dones]
        return batch, prios

    def multi_step_reward(self, *rewards):
        ret = 0.
        for idx, reward in enumerate(rewards):
            ret += reward * (self.gamma ** idx)
        return ret

    def __len__(self):
        return len(self.states)