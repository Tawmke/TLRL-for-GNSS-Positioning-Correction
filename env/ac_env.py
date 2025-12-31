import numpy as np
# from numpy import sin, cos
import gym


class Acrobot():
    def __init__(self, normalized=True, max_step=None):
        self.env_name = 'AC'
        self.num_action = 3
        self.num_state = 4
        self.state = None
        self.normalized = normalized

        self.MAX_VEL_1 = 4 * np.pi
        self.MAX_VEL_2 = 9 * np.pi
        self.MAX_THETA_1 = np.pi
        self.MAX_THETA_2 = np.pi
        self.m1 = 1.0
        self.m2 = 1.0
        self.l1 = 1.0
        self.l2 = 1.0
        self.lc1 = 0.5
        self.lc2 = 0.5
        self.I1 = 1.0
        self.I2 = 1.0
        self.g = 9.8
        self.dt = 0.05
        self.acrobotGoalPosition = 1.0
        self.max_step = max_step
        self.now_step = 0

    def reset(self):
        self.state = np.random.uniform(low=-0.5, high=0.5, size=(4,))
        self.now_step = 0
        return self._get_ob()

    def reset_state(self, init_state):
        self.reset()
        self.state = init_state.copy()
        self.now_step = 0
        return self._get_ob()

    def _get_ob(self):
        '''
        normalize to [-1, 1]
        '''
        if self.normalized:
            s = self.state
            s0 = s[0] / (1 * np.pi)
            s1 = s[1] / (1 * np.pi)
            s2 = s[2] / (4 * np.pi)
            s3 = s[3] / (9 * np.pi)
            return np.array([s0, s1, s2, s3])
        else:
            s = self.state
            return np.array([s[0], s[1], s[2], s[3]])

    def _terminal(self):
        s = self.state
        firstJointEndHeight = self.l1 * np.cos(s[0])
        secondJointEndHeight = self.l2 * np.sin(np.pi / 2 - s[1] - s[2])
        feet_height = -(firstJointEndHeight + secondJointEndHeight);
        return bool(feet_height > self.acrobotGoalPosition or self.now_step >= self.max_step)

    def step(self, a):
        s = self.state
        self.now_step += 1
        torque = a - 1.0
        count = 0
        theta1 = s[0]
        theta2 = s[1]
        theta1Dot = s[2]
        theta2Dot = s[3]

        while count < 4:
            d1 = self.m1 * np.power(self.lc1, 2) + self.m2 * (np.power(self.l1, 2) + np.power(self.lc2, 2) + 2 * self.l1 * self.lc2 * np.cos(theta2)) + self.I1 + self.I2;
            d2 = self.m2 * (np.power(self.lc2, 2) + self.l1 * self.lc2 * np.cos(theta2)) + self.I2;
            phi_2 = self.m2 * self.lc2 * self.g * np.cos(theta1 + theta2 - np.pi / 2.0);
            phi_1 = -(self.m2 * self.l1 * self.lc2 * np.power(theta2Dot, 2) * np.sin(theta2) - 2 * self.m2 * self.l1 * self.lc2 * theta1Dot * theta2Dot * np.sin(theta2)) + (self.m1 * self.lc1 + self.m2 * self.l1) * self.g * np.cos(theta1 - np.pi / 2.0) + phi_2;
            theta2_ddot = (torque + (d2 / d1) * phi_1 - self.m2 * self.l1 * self.lc2 * np.power(theta1Dot, 2) * np.sin(theta2) - phi_2) / (self.m2 * np.power(self.lc2, 2) + self.I2 - np.power(d2, 2) / d1);
            theta1_ddot = -(d2 * theta2_ddot + phi_1) / d1;
            theta1Dot += theta1_ddot * self.dt;
            theta2Dot += theta2_ddot * self.dt;
            theta1 += theta1Dot * self.dt;
            theta2 += theta2Dot * self.dt;
            count += 1

        if np.fabs(theta1Dot)>self.MAX_VEL_1:
            theta1Dot = np.sign(theta1Dot) * self.MAX_VEL_1
        if np.fabs(theta2Dot)>self.MAX_VEL_2:
            theta2Dot = np.sign(theta2Dot) * self.MAX_VEL_2
        if np.fabs(theta1) > self.MAX_THETA_1:
            theta1 = np.sign(theta1) * self.MAX_THETA_1
            theta1Dot = 0
        if np.fabs(theta2) > self.MAX_THETA_2:
            theta2 = np.sign(theta2) * self.MAX_THETA_2
            theta2Dot = 0

        s[0] = theta1
        s[1] = theta2
        s[2] = theta1Dot
        s[3] = theta2Dot
        self.state = s

        terminal = self._terminal()
        reward = -1.0

        return (self._get_ob(), reward, terminal, {})


class AcrobotV1():
    def __init__(self, normalized, max_step, render=False):
        assert normalized, 'set normalized to True for AC_V1.'
        self.env = gym.make('Acrobot-v1')
        self.env_name = 'AC'
        self.env._max_episode_steps = max_step + 10
        self.num_action = 3
        self.num_state = 6
        self.normalized = normalized
        self.render = render

    def reset(self):
        observation = self.env.reset()
        return self._normalization(observation)

    # def random_reset(self):
    #     # call reset first
    #     self.env.reset()
    #     # change env.state
    #     self.env.env.state = np.random.uniform(low=-1, high=1, size=(4,))
    #     # return normalized state
    #     observation = np.asarray(self.env.env._get_ob())
    #     return self._normalization(observation)

    def step(self, a):
        # if self.render:
        #     self.env.render()
        observation, reward, done, info = self.env.step(a)
        return self._normalization(observation), reward, done, info

    def _normalization(self, state):
        """
        normalize to [-1, 1] if self.normalized is True
        """
        # state[4] = (state[4]) / self.env.MAX_VEL_1
        # state[5] = (state[5]) / self.env.MAX_VEL_2
        return state

    def close(self):
        self.env.close()


def init_env(normalized, max_step, params):
    env = AcrobotV1(normalized, max_step)
    return env
