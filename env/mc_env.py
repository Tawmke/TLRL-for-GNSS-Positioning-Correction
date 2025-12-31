import numpy as np
import gym

class mountaincar():
    def __init__(self, normalized=True, max_step=1001, sparse_reward=False, register=False):
        if register:
            self._register(max_step)
        self.mc_env = gym.make('MountainCarLong-v0')
        self.num_action = 3
        self.num_state = 2
        self.normalized = normalized
        self.sparse_reward = sparse_reward

    def _register(self, max_step):
        gym.envs.register(
            id='MountainCarLong-v0',
            entry_point='gym.envs.classic_control:MountainCarEnv',
            max_episode_steps=max_step
        )

    def reset(self):
        observation = self.mc_env.reset()
        if self.normalized:
            return self._normalization(observation)
        else:
            return observation

    def step(self, a):
        observation, reward, done, info = self.mc_env.step(a)
        if self.sparse_reward:
            if done:
                reward = 1
            else:
                reward = 0
        if self.normalized:
            return (self._normalization(observation), reward, done, info)
        else:
            return (observation, reward, done, info)

    def close(self):
        return

    def _normalization(self, state):
        '''
        normalize to [-1, 1]
        '''
        state[0] = (state[0] + 0.35) / 0.85
        state[1] = (state[1]) / 0.07
        return state
