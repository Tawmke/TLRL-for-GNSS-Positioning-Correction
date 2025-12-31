import numpy as np
import gym
import csv
import os

def actStandardEnergyPumping(state, randomness):
    if np.random.rand() < randomness:
        return np.random.randint(low=0, high=3)
    if state[1] >= 0:
        return 2
    if state[1] < 0:
        return 0


def normalize(state):
    res = [0, 0]
    res[0] = (state[0] + 0.35) / 0.85
    res[1] = state[1] / 0.07
    return res


def getSingleTrajectory(env, randomness, normalized):
    res = []
    observation = env.reset()
    done = False
    while not done:
        oldObservation = observation
        action = actStandardEnergyPumping(oldObservation, randomness)
        observation, reward, done, info = env.step(action)
        if normalized:
            oldObservation = normalize(oldObservation)
            observation = normalize(observation)
        res.append((oldObservation, reward, observation, done))
    return res


class DataCollector:
    def __init__(self, TrajectoriesNum, randomness, normalized, load=False):
        self.env = gym.make('MountainCar-v0')
        if load:
            self.exps = []
            for i in range(20):
                filename = "./data/" + str(i+1) + ".csv"
                # filename = "./data/MC/" + str(i+1) + ".csv"
                b = np.loadtxt(filename, delimiter=",")
                c = [[b[i][:2], b[i][2], b[i][3:5], b[i][5]] for i in range(len(b))]
                self.exps.append(c)
            return
        self.exps = self.getExps(TrajectoriesNum, randomness, normalized)

    def getExps(self, TrajectoriesNum, randomness, normalized):
        res = []
        for i in range(TrajectoriesNum):
            singleRes = getSingleTrajectory(self.env, randomness, normalized)
            res.append(singleRes)
        return res

    def getTrueValueOfStateOnce(self, state, randomness, gamma):
        self.env.reset()
        self.env.env.state = state
        res = 0
        done = False
        observation = state
        rewards = []
        while not done:
            action = actStandardEnergyPumping(observation, randomness)
            observation, reward, done, info = self.env.step(action)
            rewards.append(reward)
        for reward in rewards[::-1]:
            res = res * gamma + reward
        return res

    def getTrueValueOfState(self, state, randomness, gamma, rollOutNum):
        res = 0
        for i in range(rollOutNum):
            res = res + self.getTrueValueOfStateOnce(state, randomness, gamma)
        res = res / rollOutNum
        return res

    def getSampleReturn(self, index0, index1, gamma):
        res = 0
        for i in range(index1, len(self[index0])):
            res = res + np.power(gamma, i - index1) * self[index0][i][1]
        return res

    def __getitem__(self, item):
        return self.exps[item]

    def __len__(self):
        return len(self.exps)


if __name__ == '__main__':
    dataCollector = DataCollector(0, 0, 0, load=True)
    # dataCollector = DataCollector(TrajectoriesNum=20, randomness=0.1, normalized=False)
    # for i in range(len(dataCollector)):
    #     file_name = "./data/" + str(i+1) + ".csv"
    #     if os.path.exists(file_name):
    #         os.remove(file_name)
    #     with open(file_name, "w") as f:
    #         csv_writer = csv.writer(f)
    #         for j in range(len(dataCollector[i])):
    #             temp = [dataCollector[i][j][0][0], dataCollector[i][j][0][1],
    #                     dataCollector[i][j][1], dataCollector[i][j][2][0], dataCollector[i][j][2][1],
    #                     1 if dataCollector[i][j][3] else 0]
    #             csv_writer.writerow(temp)
    # print(dataCollector[0][0][0])
    # value = dataCollector.getTrueValueOfState(dataCollector[0][0][0], randomness=0.1, gamma=1, rollOutNum=20)
    # print(value)
    # print(len(dataCollector[0]))
