import numpy as np
from SolverLogNormBatchMagnifyVelocity import *
import gym
import logging
import torch
import scipy.io
import time
import sys
import os
import shutil
import threading
import multiprocessing
from multiprocessing import Process
import numpy as np
import matplotlib.pyplot as plt


class ControlLayer(nn.Module):
    """docstring for Net"""
    def __init__(self, pre_num, next_num):
        super(ControlLayer, self).__init__()
        self.out = nn.Linear(pre_num, next_num, bias=False)
        #self.out.weight.copy_(weights)

    def forward(self, x):
        action_prob = self.out(x)
        return action_prob

class ActionLearner:
    def __init__(self, actionRandomness, decay, stepSize, preTrainTimes=None, maxIter=None, gamma=None, lossMode=None,
                 wordNums=None, betaB=None, betaW=None, betaFi=None,
                 lineSearchW=None, lineSearchB=None, thresholdW=None, thresholdB=None, thresholdFi=None,
                 thresholdLoss=None, lineAlphaW=None, lineBetaW=None,
                 lineAlphaB=None,
                 lineBetaB=None,
                 dataSource=None, randomness=None,
                 normalized=None, lipSafe=None, loadBasic=None, index=1, outputFile=None):

        self.outputFile = outputFile
        self.basic = None
        if loadBasic is None:
            self.basic = DLSRSolver(
                maxIter, gamma, lossMode, wordNums, betaB, betaW, betaFi,
                lineSearchW, lineSearchB, thresholdW, thresholdB, thresholdFi, thresholdLoss, lineAlphaW, lineBetaW,
                lineAlphaB,
                lineBetaB,
                dataSource, randomness,
                normalized, lipSafe
            )
            # self.gamma = gamma
            self.preTrain(preTrainTimes)
        else:
            filename = "./matlab_Model/" + loadBasic  + "/" + loadBasic + '_' + str(index) +'/'
            self.basic = DLSRSolver(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, skip=True)
            self.basic.B = scipy.io.loadmat(filename + 'B.mat')['B']
            self.basic.wordNums = self.basic.B.shape[0]
            self.basic.w = scipy.io.loadmat(filename + 'W.mat')['W']
            self.basic.weightReconstruct = scipy.io.loadmat(filename + 'weightReconstruct.mat')[
                'weightReconstruct'].item()
            self.basic.gamma = scipy.io.loadmat(filename + 'gamma.mat')['gamma'].item()
            self.basic.lossMode = 'MeanSquaredReturnError'
            self.basic.weightPredict = scipy.io.loadmat(filename + 'weightPredict.mat')['weightPredict'].item()
            self.basic.maxIter = scipy.io.loadmat(filename + 'maxIter.mat')['maxIter'].item()
            self.basic.delta = scipy.io.loadmat(filename + 'delta.mat')['delta'].item()
            self.basic.betaFi = scipy.io.loadmat(filename + 'betaFi.mat')['betaFi'].item()
            self.basic.betaW = scipy.io.loadmat(filename + 'betaW.mat')['betaW'].item()
            self.basic.thresholdFi = scipy.io.loadmat(filename + 'thresholdFi.mat')['thresholdFi'].item()
            if lossMode == 'MeanSquaredReturnError':
                self.basic.gammaHat = 0
            else:
                self.basic.gammaHat = self.basic.gamma
            # with open(filename, "rb") as file:
            #     self.basic = pickle.loads(file.read())
            # self.gamma = self.basic.gamma

        self.env = gym.make('MountainCar-v0')
        self.weights = np.array([self.basic.w.copy().reshape(-1) for _ in range(self.env.action_space.n)])
        self.stepSize = stepSize
        self.actionRandomness = actionRandomness
        self.book = {}
        self.decay = decay
        self.num_action = self.env.action_space.n  # 动作数
        self.w = ControlLayer(64, self.num_action)
        self.w_target = ControlLayer(64, self.num_action)
        self.w.state_dict()['out.weight'].copy_(torch.FloatTensor(self.weights))
        self.w_target.load_state_dict(self.w.state_dict())
        self.epsilon = 0.1
        self.optimizer = torch.optim.SGD(self.w.parameters(), lr=1e-6)
        self.loss_func = nn.MSELoss()
        self.gamma = 1.0

    def quickGetSparseForm(self, state):
        # start_time=time.time()
        # L = np.linalg.norm(np.dot(self.basic.B, self.basic.B.T), 2) * 2 * self.basic.weightReconstruct
        L = np.power(np.linalg.norm(self.basic.B, 2), 2) * 2 * self.basic.weightReconstruct
        stepSize = 1 / L
        stepSize = 0.8 * stepSize
        # stepSize = 0.5 / (
        #         (1 + np.power(self.basic.gammaHat, 2)) * np.power(np.linalg.norm(self.basic.w, 2),
        #                                                           2) * self.basic.weightPredict + np.power(
        #     np.linalg.norm(self.basic.B, 2), 2) * self.basic.weightReconstruct
        # ) * 0.8
        resFi = np.zeros((self.basic.wordNums, 1))
        for _ in range(100000):
            oldFi = resFi.copy()
            resFi = resFi - stepSize * np.dot(
                self.basic.B, np.dot(self.basic.B.T, resFi) - state.reshape(-1, 1)
            ) * self.basic.weightReconstruct * 2
            t = self.basic.betaFi * stepSize
            condition1 = resFi > np.sqrt(4*t) - self.basic.delta
            condition2 = resFi < -(np.sqrt(4 * t) - self.basic.delta)
            condition3 = np.abs(resFi) <= np.sqrt(4 * t) - self.basic.delta
            resFi[condition1] = 0.5 * ((resFi[condition1] - self.basic.delta) + np.sqrt(np.power(resFi[condition1] + self.basic.delta, 2) - 4 * t))
            resFi[condition2] = 0.5 * ((resFi[condition2] + self.basic.delta) - np.sqrt(
                np.power(resFi[condition2] - self.basic.delta, 2) - 4 * t))
            resFi[condition3] = 0
            # resFi = np.sign(resFi) * np.maximum(np.abs(resFi) - stepSize * self.basic.betaFi, 0)
            if np.abs(oldFi - resFi).max() < self.basic.thresholdFi:
                break
        # print(time.time()-start_time)
        return resFi

    def getSparseForm(self, observation):
        # begin matlab
        # start_time = time.time()
        # matlabResult = eng.getSparseForm(self.basic.maxIter, matlab.double(observation.reshape(1, -1).tolist()), matlab.double(self.basic.w.tolist()), matlab.double(self.basic.B.tolist()), self.basic.weightPredict, self.basic.weightReconstruct, self.basic.betaFi, self.basic.gamma, self.basic.lossMode, self.basic.delta, self.basic.thresholdFi)
        # result = np.array(matlabResult._data).reshape(-1, 1)
        # print("time cost:{}".format(time.time()-start_time))
        # return result
        # end matlab

        # return self.quickGetSparseForm(observation)
        tempObservation = observation.copy()
        for index, element in enumerate(tempObservation):
            tempObservation[index] = round(element, 7)
        temp = "*".join(tempObservation.astype('str').tolist())

        if temp in self.book.keys():
            # print("hit")
            # print(self.book[temp].reshape(-1))
            return self.book[temp]
        with open(self.outputFile, "a") as f:
            w_str = ','.join(str(x) for x in self.basic.w)
            # f.write('ohno')
            f.write(w_str)
            f.write('\n')
            # print(self.basic.w)
        result = self.quickGetSparseForm(observation)
        # matlabResult = eng.getSparseForm(self.basic.maxIter, matlab.double(observation.reshape(1, -1).tolist()),
        #                                  matlab.double(self.basic.w.tolist()), matlab.double(self.basic.B.tolist()),
        #                                  self.basic.weightPredict, self.basic.weightReconstruct, self.basic.betaFi,
        #                                  self.basic.gamma, self.basic.lossMode, self.basic.delta,
        #                                  self.basic.thresholdFi)
        # result = np.array(matlabResult._data).reshape(-1, 1)
        self.book[temp] = result
        # print("time cost:{}".format(time.time() - start_time))
        return self.book[temp]

    def _encoder(self, observation):
        observation_ = self.basic.net(torch.from_numpy(observation.copy())).detach().numpy()
        observation_encode = self.getSparseForm(observation_)
        return np.squeeze(observation_encode)

    def decide(self, s):
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(self.num_action)
        else:
            z = self._encoder(s)
            z = torch.unsqueeze(torch.FloatTensor(z), 0)  # get a 1D array
            Q_value = self.w.forward(z)
            greed_action = torch.max(Q_value, 1)[1].data.numpy()
            greed_action = greed_action[0]
            action = greed_action
        return action

    def learn_sarsa(self, s, a, s_t, a_t, reward, done):
        z = self._encoder(s)
        z = torch.unsqueeze(torch.FloatTensor(z), 0)
        z_t = self._encoder(s_t)
        z_t = torch.unsqueeze(torch.FloatTensor(z_t), 0)
        a = torch.LongTensor(np.expand_dims([a], axis=0))
        a_t = torch.LongTensor(np.expand_dims([a_t], axis=0))
        reward = torch.LongTensor(np.expand_dims([reward], axis=0))
        q_eval = self.w(z).gather(1,a)
        q_next = self.w_target(z_t).detach().gather(1, a_t)
        q_target = reward + self.gamma * q_next * (1-done)
        loss = self.loss_func(q_eval, q_target)
        #print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.w_target.load_state_dict(self.w.state_dict())
        return

    def play_sarsa(self, train=False, render=False):
        env = self.env
        episode_reward = 0
        observation = env.reset()
        action = self.decide(observation)
        while True:
            if render:
                env.render()
            next_observation, reward, done, _ = env.step(action)
            #print(action, reward)
            episode_reward += reward
            next_action = self.decide(next_observation) # 终止状态时此步无意义
            if train:
                self.learn_sarsa(observation, action, next_observation, next_action, reward, done)
            if done:
                break
            observation, action = next_observation, next_action
        return episode_reward

if __name__ == '__main__':
    k=0
    betaFi = 10
    weightReconstruct = 20000
    delta = 1e-3
    decay = 1.0
    folder =  './output/' + str(k + 1)
    stepSizes = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    if not os.path.exists(folder):
        os.makedirs(folder)
    plot_folder = './plots_wjq/MC_control'
    for stepSize in stepSizes:
        for j in range(1):
            outputFile = folder + '/' + str(j + 1) + '.log'
            if os.path.exists(outputFile):
                os.remove(outputFile)
            loadBasic = "419Fast_Repete_Reconstruct_" + str(weightReconstruct) + "_BetaFi_" + str(betaFi) + "_delta_" + str(delta)
            actionLearner = ActionLearner(actionRandomness=0.1, decay=decay, stepSize=stepSize, loadBasic=loadBasic,
                                          index=k + 1, outputFile=outputFile)
            episode_rewards = []
            for episode in range(50):
                total_reward = actionLearner.play_sarsa(train=True, render=False)
                episode_rewards.append(total_reward)
                print("episode: ", episode, "rewards: ", total_reward)
            plt.plot(episode_rewards)
            plt.xlabel("episodes")
            plt.ylabel("rewards")
            plt.title("MountainCar")
            plt.savefig(plot_folder + '/decay={:.4f}_control_lr={:.7f}.png'.format(decay, stepSize))
            plt.clf()
