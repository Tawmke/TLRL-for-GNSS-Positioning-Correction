# from funcs.preNet_folder import *
from funcs.DataCollectorModule import *
import time
import pickle
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import scipy.io
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def getSingleLogNorm(x, delta):
    res = np.sum(np.log(1 + np.abs(x) / delta))
    return res

# def loadNet_preNet(filename):
#     data_folder = './newNet/'
#     pkl_route = data_folder + 'new.pkl'
#     with open(pkl_route, "rb") as file:
#         params = pickle.load(file)
#     file.close()
#     net = Net(params)
#     return net

class Net_preNet(nn.Module):
    def __init__(self, PreNet, Bias, preprocess, dataset):
        super(Net_preNet, self).__init__()
        self.preprocess=preprocess
        self.dataset=dataset
        self.fc = nn.Linear(2, 32, bias=True).double()
        # self.fc.data = torch.DoubleTensor(params)

        if self.preprocess == 'DQN':
            data_folder = './newNet/'
            pkl_route = data_folder + 'new.pkl'
            with open(pkl_route, "rb") as file:
                params = pickle.load(file)
            file.close()
            self.fc.weight.data = torch.DoubleTensor(params['PreNet'].T)
            self.fc.bias.data = torch.DoubleTensor(params['Bias'].reshape(-1).T)
        elif self.preprocess == 'DQN_norm_only':
            Model_folder=PreNet
            PreNet1 = scipy.io.loadmat(Model_folder+'_layer1_weights.mat')['W1']
            Bias1 = scipy.io.loadmat(Model_folder+'_layer1_bias.mat')['B1']
            self.fc.weight.data = torch.DoubleTensor(PreNet1.T)
            self.fc.bias.data = torch.DoubleTensor(Bias1.reshape(-1).T)
            self.fc1 = nn.Linear(32, 256, bias=True).double()
            PreNet2 = scipy.io.loadmat(Model_folder+'_layer2_weights.mat')['W2']
            Bias2 = scipy.io.loadmat(Model_folder+'_layer2_bias.mat')['B2']
            self.fc1.weight.data = torch.DoubleTensor(PreNet2.T)
            self.fc1.bias.data = torch.DoubleTensor(Bias2.reshape(-1).T)
            # self.fc2 = nn.Linear(256, 1, bias=False).double()
            # PreNet2 = scipy.io.loadmat(Model_folder+'_output_weights.mat')['Wa']
            # self.fc2.weight.data = torch.DoubleTensor(PreNet2.T)
        else:
            self.fc.weight.data = torch.DoubleTensor(PreNet.T)
            self.fc.bias.data = torch.DoubleTensor(Bias.reshape(-1).T)
        pass

    def forward(self, x):
        res = x.detach()
        data= res
        if self.preprocess == 'randn' or self.preprocess == 'DCT_IV':
            if self.dataset == 'MC':
                data.T[0] = (data.T[0] + 0.35) / 0.85
                data.T[1] = (data.T[1]) / 0.07
            elif self.dataset == 'CP':
                data.T[0] = (data.T[0]) / 0.5
                data.T[1] = (data.T[1]) / 2
                data.T[2] = (data.T[2]) / 0.2
                data.T[3] = (data.T[3]) / 2.7
            # elif self.dataset == 'PW':
            #     data.T[0] = (data.T[0] - 0.5) * 2.0
            #     data.T[1] = (data.T[1] - 0.5) * 2.0
            # elif self.dataset == 'AC' or dataset == 'AC2':
            #     data.T[0] = (data.T[0]) / (1 * np.pi)
            #     data.T[1] = (data.T[1]) / (1 * np.pi)
            #     data.T[2] = (data.T[2]) / (4 * np.pi)
            #     data.T[3] = (data.T[3]) / (9 * np.pi)
            # elif self.dataset == 'CA3':
            #     data.T[0] = (data.T[0] - 26) / 26
            #     data.T[1] = (data.T[1]) / 8
            #     data.T[2] = (data.T[2] - 26) / 26
            #     data.T[3] = (data.T[3] - 20) / 45

            res=data
            if self.preprocess == 'randn':
                res = torch.tanh(10 * res) * 0.7
                # res[1] = torch.tanh(10 * res[1]) * 0.7
                res = self.fc(res)
                res = F.relu(res)
            elif self.preprocess == 'DCT_IV':
                res = self.fc(res)
        elif self.preprocess == 'DQN_norm':
            res = self.fc(res)
            res = F.relu(res)
        elif self.preprocess == 'DQN_norm_re':
            if self.dataset == 'MC':
                data.T[0] = (data.T[0] + 0.35) / 0.85
                data.T[1] = (data.T[1]) / 0.07
            elif self.dataset == 'CP':
                data.T[0] = (data.T[0]) / 0.5
                data.T[1] = (data.T[1]) / 2
                data.T[2] = (data.T[2]) / 0.2
                data.T[3] = (data.T[3]) / 2.7
            res = self.fc(res)
            res = F.relu(res)
        elif self.preprocess == 'DQN':
            if self.dataset == 'MC':
                res[1] = torch.tanh(10 * res[1]) * 0.7
            res = self.fc(res)
            res = F.relu(res)
        elif self.preprocess == 'DQN_norm_only':
            if self.dataset == 'MC':
                data.T[0] = (data.T[0] + 0.35) / 0.85
                data.T[1] = (data.T[1]) / 0.07
            elif self.dataset == 'CP':
                data.T[0] = (data.T[0]) / 0.5
                data.T[1] = (data.T[1]) / 2
                data.T[2] = (data.T[2]) / 0.2
                data.T[3] = (data.T[3]) / 2.7
            res = self.fc(res)
            res = F.relu(res)
            res = self.fc1(res)
            res = F.relu(res)
        return res

class DLSRSolver_preNet:
    def __init__(self, filename_net, filename, preprocess, dataset, maxIter, gamma, lossMode, wordNums, betaB, betaW, betaFi,
                 lineSearchW, lineSearchB, thresholdW, thresholdB, thresholdFi, thresholdLoss, weightPredict,
                 weightReconstruct, delta,
                 lineAlphaW=None,lineBetaW=None,lineAlphaB=None,lineBetaB=None,dataSource=None, randomness=None,
                 normalized=None, lipSafe=None, skip=False):

        if preprocess == 'DQN_norm_only':
            self.PreNet = filename_net
            self.Bias = 0
        else:
            self.PreNet = scipy.io.loadmat(filename_net + 'PreNet.mat')['PreNet']
            self.Bias = scipy.io.loadmat(filename_net + 'Bias.mat')['Bias']
        self.net = Net_preNet(self.PreNet, self.Bias, preprocess, dataset)
        if skip:
            return
        assert dataSource is not None or (
                dataSource is None and randomness is not None and normalized is not None)
        assert not lineSearchW or (lineSearchW and lineAlphaW is not None and lineBetaW is not None)
        assert not lineSearchB or (lineSearchB and lineAlphaB is not None and lineBetaB is not None)
        assert lineSearchW or (not lineSearchW and lipSafe is not None)
        assert lineSearchB or (not lineSearchB and lipSafe is not None)
        self.maxIter = maxIter
        self.gamma = gamma
        self.gammaHat = 0
        self.lossMode = lossMode
        self.wordNums = wordNums
        self.w = np.zeros((self.wordNums, 1))
        self.betaB = betaB
        self.betaW = betaW
        self.betaFi = betaFi
        self.B = None
        self.x = None
        self.y = None
        self.t = None
        self.fi = None
        self.dataCollector = None
        self.mask = None
        self.lineSearchW = lineSearchW
        self.lineSearchB = lineSearchB
        self.thresholdW = thresholdW
        self.thresholdB = thresholdB
        self.thresholdFi = thresholdFi
        self.thresholdLoss = thresholdLoss
        self.weightPredict = weightPredict
        self.weightReconstruct = weightReconstruct
        self.lineAlphaW = lineAlphaW
        self.lineBetaW = lineBetaW
        self.lineAlphaB = lineAlphaB
        self.lineBetaB = lineBetaB
        self.lipSafe = lipSafe
        self.delta = delta
        self.__adjustGammaHat()
        self.changeDataSource(dataSource) if dataSource is not None \
            else self.changeDataSource(DataCollector(TrajectoriesNum=20, randomness=randomness, normalized=normalized))
        # print(self.x)
        # print(self.x.shape)

    def __adjustGammaHat(self):
        if self.lossMode == "MeanSquaredReturnError":
            self.gammaHat = 0
        elif self.lossMode == "BellmanError":
            self.gammaHat = self.gamma

    def changeDataSource(self, dataSource):
        self.dataCollector = dataSource
        self.__extractFromDataCollector()

    def __extractFromDataCollector(self):
        self.__extractX()
        self.__extractY()
        self.__extractMASK()
        self.__adjustY()

    def __extractX(self):
        assert self.dataCollector is not None
        assert self.B is None or self.B.shape[1] == self.x.shape[1]
        self.x = None
        for i in range(len(self.dataCollector)):
            tempArray = np.array([self.net(torch.from_numpy(Tuple[0])).detach().numpy() for Tuple in
                                  self.dataCollector[i]])
            if self.x is None:
                self.x = tempArray
            else:
                self.x = np.append(self.x, tempArray, axis=0)
            self.x = self.x = np.vstack(
                [self.x, self.net(torch.from_numpy(self.dataCollector[i][-1][2])).detach().numpy()])
        self.__buildB()
        self.__rebuildFi()

    def __extractY(self):
        assert self.dataCollector is not None
        if self.lossMode == "MeanSquaredReturnError":
            self.y = None
            for trajectoryIndex in range(len(self.dataCollector)):
                tempArray = np.array([
                    self.dataCollector.getSampleReturn(trajectoryIndex, i, self.gamma) for i in
                    range(len(self.dataCollector[trajectoryIndex]))
                ])
                if self.y is None:
                    self.y = tempArray
                else:
                    self.y = np.append(self.y, tempArray, axis=0)
        elif self.lossMode == "BellmanError":
            self.y = None
            for trajectoryIndex in range(len(self.dataCollector)):
                tempArray = np.array([Tuple[1] for Tuple in self.dataCollector[trajectoryIndex]])
                if self.y is None:
                    self.y = tempArray
                else:
                    self.y = np.append(self.y, tempArray, axis=0)

    def __extractMASK(self):
        assert self.dataCollector is not None
        self.mask = None
        for trajectoryIndex in range(len(self.dataCollector)):
            tempArray = np.array([Tuple[3] for Tuple in self.dataCollector[trajectoryIndex]])
            if self.mask is None:
                self.mask = tempArray
            else:
                self.mask = np.append(self.mask, tempArray, axis=0)

    def __adjustY(self):
        cntPos = 0
        for i in range(len(self.mask) - 1):
            if self.mask[i]:
                self.y = np.insert(self.y, cntPos + 1, 0)
                cntPos = cntPos + 2
            else:
                cntPos = cntPos + 1

    def __buildB(self):
        if self.B is not None:
            return
        self.B = np.random.rand(self.wordNums, self.x.shape[1])

    def __rebuildFi(self):
        self.fi = np.zeros((self.x.shape[0], self.wordNums))

    def getUpdateWMethod(self, b, A):
        updateW = -b - np.dot(A, self.w).reshape(-1) * self.weightPredict - (self.betaW * self.w).reshape(-1)
        return updateW

    def getDerivativeOverW(self):
        # res = np.mean(
        #     2 * (self.y.reshape(-1, 1) + self.gamma * np.dot(self.fi[1:], self.w) - np.dot(self.fi[:-1], self.w)) *
        #     (self.gamma * self.fi[1:] - self.fi[:-1]) * (self.y != 0).reshape(-1, 1), axis=0
        # )
        res = np.mean(
            2 * (self.y.reshape(-1, 1) + self.gammaHat * np.dot(self.fi[1:], self.w) - np.dot(self.fi[:-1], self.w)) *
            (self.gammaHat * self.fi[1:] - self.fi[:-1]) * (self.y != 0).reshape(-1, 1), axis=0
        )
        res = res * (len(self.x) - 1) / len(self.mask)
        res = res.reshape(-1, 1) * self.weightPredict + 2 * self.betaW * self.w
        res = res.reshape(-1)
        return res

    def getItemsW(self, nextW, stepSize, updateW):
        temp = self.w.copy()
        self.w = nextW
        item1 = np.sum(self.getLoss())
        self.w = temp
        item2 = np.sum(self.getLoss()) + stepSize * self.lineAlphaW * np.dot(
            updateW.T, self.getDerivativeOverW()
        ).item()
        return item1, item2

    def LineSearchW(self, b, A):
        updateW = self.getUpdateWMethod(b, A)
        updateW = updateW.reshape(-1, 1)
        stepSize = 1
        nextW = self.w + stepSize * updateW
        item1, item2 = self.getItemsW(nextW, stepSize, updateW)
        while item1 > item2:
            stepSize = stepSize * self.lineBetaW
            nextW = self.w + stepSize * updateW
            item1, item2 = self.getItemsW(nextW, stepSize, updateW)
        return stepSize

    def lipschitzStepW(self, b, A):
        res = self.betaW + np.max(
            np.linalg.eig(A)[0]
        )
        res = 1 / res
        res = res * self.lipSafe
        return res

    def getUpdateW(self, b, A):
        updateW = self.getUpdateWMethod(b, A)
        # updateW = -b - np.dot(A, self.w).reshape(-1)
        if self.lineSearchW:
            updateW = self.LineSearchW(b, A) * updateW
        else:
            updateW = self.lipschitzStepW(b, A) * updateW
        updateW = updateW.reshape(-1, 1)
        return updateW

    def learnW(self):
        b = np.mean(
            self.y.reshape(-1, 1) * (self.gammaHat * self.fi[1:] - self.fi[:-1]), axis=0
        ) * self.weightPredict
        b = b * (len(self.x) - 1) / len(self.mask)
        A = np.array(
            [np.dot(self.gammaHat * np.expand_dims(self.fi[i + 1], axis=1) - np.expand_dims(self.fi[i], axis=1),
                    (self.gammaHat * np.expand_dims(self.fi[i + 1], axis=1) - np.expand_dims(self.fi[i],
                                                                                             axis=1)).T) * (
                     self.y[i] != 0) for i in
             range(len(self.fi) - 1)])
        A = np.mean(A, axis=0) * self.weightPredict
        A = A * (len(self.x) - 1) / len(self.mask)
        count = 0
        updateW = self.getUpdateW(b, A)
        self.w = self.w + updateW
        updateW = self.getUpdateW(b, A)
        while count < self.maxIter and np.abs(updateW).max() > self.thresholdW:
            # print("Update W Minal: {}".format(np.sum(self.getLoss())))
            self.w = self.w + updateW
            updateW = self.getUpdateW(b, A)
            count = count + 1

    def getUpdateBMethod(self):
        updateB = - np.dot(
            self.fi.T, np.dot(self.fi, self.B) - self.x
        ) / self.x.shape[0]
        updateB = updateB * self.weightReconstruct - self.betaB * self.B
        return updateB

    def getItemsB(self, nextB, stepSize, updateB):
        temp = self.B.copy()
        self.B = nextB
        item1 = np.sum(self.getLoss())
        self.B = temp
        item2 = np.sum(self.getLoss()) + stepSize * self.lineAlphaB * np.sum(
            updateB * (-updateB)
        )
        return item1, item2

    def LineSearchB(self):
        updateB = self.getUpdateBMethod()
        stepSize = 1
        nextB = self.B + stepSize * updateB
        item1, item2 = self.getItemsB(nextB, stepSize, updateB)
        while item1 > item2 and stepSize > 0:
            stepSize = stepSize * self.lineBetaB
            nextB = self.B + stepSize * updateB
            item1, item2 = self.getItemsB(nextB, stepSize, updateB)
        return stepSize

    def getUpdateB(self):
        updateB = self.getUpdateBMethod()
        if self.lineSearchB:
            updateB = self.LineSearchB() * updateB
        else:
            updateB = self.lipschitzStepB() * updateB
        return updateB

    def lipschitzStepB(self):
        pass

    def learnB(self):
        count = 0
        updateB = self.getUpdateB()
        self.B = self.B + updateB
        updateB = self.getUpdateB()
        while count < self.maxIter and np.abs(updateB).max() > self.thresholdB:
            self.B = self.B + updateB
            updateB = self.getUpdateB()
            count = count + 1

    def learnFi(self):
        stepSize = 0.5 / (
                (1 + np.power(self.gammaHat, 2)) * np.power(np.linalg.norm(self.w, 2),
                                                            2) * self.weightPredict + np.power(
            np.linalg.norm(self.B, 2), 2) * self.weightReconstruct
        ) * 0.8
        count = 0
        while count < self.maxIter:
            print("Minor Fi: Loss Part1: {:<22}  Loss Part2: {:<22}  Loss Part3: {:<22}  Loss Part4: {:<20}  Loss "
                  "Part5: "
                  "{:<20}".format(
                *tuple(self.getLoss())), "Total Loss: {}".format(np.sum(self.getLoss())))
            oldFiOuter = self.fi.copy()
            oldLossOuter = np.sum(self.getLoss())
            a = np.dot(self.fi[:-1], self.w) - self.y.reshape(-1, 1)
            a = np.append([np.dot(self.fi[0], self.w) * self.gammaHat], a, axis=0)
            c = self.gammaHat * np.dot(self.fi[1:], self.w) + self.y.reshape(-1, 1)
            c = np.append(c, [np.dot(self.fi[-1], self.w)], axis=0)
            for k in range(self.maxIter):
                # print("Little Fi: Loss Part1: {:<22}  Loss Part2: {:<22}  Loss Part3: {:<22}  Loss Part4: {:<20}  Loss "
                #       "Part5: "
                #       "{:<20}".format(
                #     *tuple(self.getLoss())), "Total Loss: {}".format(np.sum(self.getLoss())))
                oldFiInner = self.fi.copy()
                oldLossInner = np.sum(self.getLoss())
                b = np.dot(self.fi, self.w)
                self.fi = self.fi - stepSize * np.dot(
                    np.dot(self.fi, self.B) - self.x, self.B.T
                ) * self.weightReconstruct * 2
                tempY1 = np.append([0], self.y != 0)
                tempY2 = np.append(self.y != 0, [0])
                self.fi = self.fi - stepSize * (b - c) * self.w.reshape(-1) * tempY2.reshape(-1,
                                                                                             1) * self.weightPredict * 2
                self.fi = self.fi - stepSize * self.gammaHat * (self.gammaHat * b - a) * self.w.reshape(
                    -1) * tempY1.reshape(-1, 1) * self.weightPredict * 2
                # self.fi = np.sign(self.fi) * np.maximum(np.abs(self.fi) - stepSize * self.betaFi, 0)
                # t = self.betaFi * stepSize
                t = self.betaFi * stepSize
                condition1 = self.fi > np.sqrt(4 * t) - self.delta
                condition2 = self.fi < -(np.sqrt(4 * t) - self.delta)
                condition3 = np.abs(self.fi) < np.sqrt(4 * t) - self.delta
                self.fi[condition1] = 0.5 * ((self.fi[condition1] - self.delta) + np.sqrt(np.power(self.fi[condition1] + self.delta, 2) - 4 * t))
                self.fi[condition2] = 0.5 * ((self.fi[condition2] + self.delta) - np.sqrt(np.power(self.fi[condition2] - self.delta, 2) - 4 * t))
                self.fi[condition3] = 0
                # g = lambda fi: np.array([0.5 * ((self.fi - self.delta) + np.sqrt(
                #     np.power(self.fi + self.delta, 2) - 4 * t)) if element > np.sqrt(4 * t) - self.delta \
                #                              else 0.5 * (
                #             (self.fi + self.delta) - np.sqrt(np.power(self.fi - self.delta, 2) - 4 * t)) if element < -(
                #             np.sqrt(4 * t) - self.delta) else 0 for element in fi])
                # self.fi = np.array([g(fi) for fi in self.fi])
                # self.fi = 0.5 * ((self.fi - self.delta) + np.sqrt(np.power(self.fi + self.delta, 2) - 4 * t)) if condition1 else 0
                # self.fi = self.fi + 0.5 * ((self.fi + self.delta) - np.sqrt(np.power(self.fi - self.delta, 2) - 4 * t)) if condition2 else 0
                # self.fi = condition1 * 0.5 * ((self.fi - self.delta) + np.sqrt(np.power(self.fi + self.delta, 2) - 4 * t)) \
                #             + condition2 * 0.5 * ((self.fi + self.delta) - np.sqrt(np.power(self.fi - self.delta, 2) - 4 * t))
                if np.abs(self.fi - oldFiInner).max() < self.thresholdFi or np.abs(
                        np.sum(self.getLoss()) - oldLossInner).max() < self.thresholdLoss:
                    break
            # print("Update Minor Fi With: {}".format(np.abs(self.fi - oldFiOuter).max()))
            if np.abs(self.fi - oldFiOuter).max() < self.thresholdFi or np.abs(
                    np.sum(self.getLoss()) - oldLossOuter).max() < self.thresholdLoss:
                break

    def Learn(self):
        print("Initial : Loss Part1: {:<22}  Loss Part2: {:<22}  Loss Part3: {:<22}  Loss Part4: {:<20}  Loss Part5: {"
              ":<20}".format(
            *tuple(self.getLoss())), "Total Loss: {}".format(np.sum(self.getLoss())))
        for _ in range(self.maxIter):
            oldLoss = np.sum(self.getLoss())
            self.learnW()
            print("Learn W : Loss Part1: {:<22}  Loss Part2: {:<22}  Loss Part3: {:<22}  Loss Part4: {:<20}  Loss "
                  "Part5: "
                  "{:<20}".format(
                *tuple(self.getLoss())), "Total Loss: {}".format(np.sum(self.getLoss())))
            self.learnB()
            print("Learn B : Loss Part1: {:<22}  Loss Part2: {:<22}  Loss Part3: {:<22}  Loss Part4: {:<20}  Loss "
                  "Part5: "
                  "{:<20}".format(
                *tuple(self.getLoss())), "Total Loss: {}".format(np.sum(self.getLoss())))
            self.learnFi()
            print("Learn Fi: Loss Part1: {:<22}  Loss Part2: {:<22}  Loss Part3: {:<22}  Loss Part4: {:<20}  Loss "
                  "Part5: "
                  "{:<20}".format(
                *tuple(self.getLoss())), "Total Loss: {}".format(np.sum(self.getLoss())))
            if np.abs(oldLoss - np.sum(self.getLoss())).max() < self.thresholdLoss:
                break

    def getLoss(self):
        # start_time = time.time()
        part1 = np.mean(np.power(
            self.y.reshape(-1, 1) + self.gammaHat * np.dot(self.fi[1:], self.w) - np.dot(self.fi[:-1], self.w)
            , 2) * (self.y != 0)
                        )
        part1 = self.weightPredict * part1 * (len(self.x) - 1) / len(self.mask)
        # part2 = np.mean(np.power(
        #     [np.linalg.norm(temp) for temp in np.dot(self.fi, self.B) - self.x], 2
        # )) * self.weightReconstruct
        part2 = np.mean(np.power(
            np.linalg.norm(np.dot(self.fi, self.B) - self.x, 2, axis=1), 2
        )) * self.weightReconstruct
        part3 = self.betaB * np.power(np.linalg.norm(self.B), 2)
        part4 = self.betaW * np.power(np.linalg.norm(self.w), 2)
        part5 = (self.betaFi / (len(self.fi) - 1)) * np.sum(
            [getSingleLogNorm(temp, self.delta) for temp in self.fi]
        )
        # part5 = (self.betaFi / (len(self.fi) - 1)) * np.sum(
        #     [np.linalg.norm(temp, 1) for temp in self.fi]
        # )
        # print("time cost: {}".format(time.time()-start_time))
        return part1, part2, part3, part4, part5

    def getSparseForm(self, state):
        L = np.linalg.norm(np.dot(self.B, self.B.T), 2) * 2 * self.weightReconstruct
        stepSize = 1 / L
        stepSize = 0.5 / (
                (1 + np.power(self.gammaHat, 2)) * np.power(np.linalg.norm(self.w, 2),
                                                            2) * self.weightPredict + np.power(
            np.linalg.norm(self.B, 2), 2) * self.weightReconstruct
        ) * 0.8
        resFi = np.zeros((self.wordNums, 1))
        for _ in range(self.maxIter):
            oldFi = resFi.copy()
            resFi = resFi - stepSize * np.dot(
                self.B, np.dot(self.B.T, resFi) - state.reshape(-1, 1)
            ) * self.weightReconstruct * 2
            t = self.betaFi * stepSize
            condition1 = resFi > np.sqrt(4 * t) - self.delta
            condition2 = resFi < -(np.sqrt(4 * t) - self.delta)
            condition3 = np.abs(resFi) < np.sqrt(4 * t) - self.delta
            resFi[condition1] = 0.5 * ((resFi[condition1] - self.delta) + np.sqrt(
                np.power(resFi[condition1] + self.delta, 2) - 4 * t))
            resFi[condition2] = 0.5 * ((resFi[condition2] + self.delta) - np.sqrt(
                np.power(resFi[condition2] - self.delta, 2) - 4 * t))
            resFi[condition3] = 0
            if np.abs(oldFi - resFi).max() < self.thresholdFi:
                break
        return resFi

def checkAccuracy(solverCheck):
    print('-' * 80)
    for index, x in enumerate(solverCheck.x):
        print('x :        ', x)
        print('predict x: ', np.dot(solverCheck.fi[index], solverCheck.B))
        print('y :        ', solverCheck.y[index] if index != len(solverCheck.x) - 1 else 0)
        print('predict y: ', np.dot(solverCheck.fi[index], solverCheck.w))
        print('fi:        ', solverCheck.fi[index])
        print('-' * 80)
    print("Word Similarity:")
    temp = np.array([word / np.linalg.norm(word) for word in solverCheck.B])
    temp = np.dot(temp, temp.T)
    for i in range(temp.shape[0]):
        for j in range(i, temp.shape[1]):
            print("Similarity between {} and {}: {}".format(i, j, np.abs(temp[i][j])))
    print("Weights:")
    print(solverCheck.w)


def checkGetSparseForm(solverCheck):
    print('-' * 80)
    for index, x in enumerate(solverCheck.x):
        sparseForm = solverCheck.getSparseForm(x)
        print("x            :{}".format(x))
        print("Sparse       :{}".format(sparseForm.reshape(-1)))
        print("Reconstruct x:{}".format(np.dot(solverCheck.B.T, sparseForm).reshape(-1)))
        print("Predict:      {}".format(np.dot(solverCheck.w.T, sparseForm).item()))
        print('-' * 80)


def SaveModel(solverSave, filename):
    filename1 = "./model/" + filename + ".pkl"
    outPutModel = open(filename1, "wb")
    strDumps = pickle.dumps(solverSave)
    outPutModel.write(strDumps)
    outPutModel.close()

    attributes = dir(solverSave)
    filename2 = "./model/" + filename + ".txt"
    txtFile = open(filename2, "w")
    for attribute in attributes:
        if '_' not in attribute and not callable(getattr(solver, attribute)):
            txtFile.writelines("{}:{}".format(attribute, getattr(solver, attribute)))
            txtFile.writelines('\n')
    txtFile.close()


if __name__ == '__main__':
    dataCollector = DataCollector(0, 0, 0, load=True)

    np.set_printoptions(threshold=np.inf)
    solver = DLSRSolver(maxIter=10000, lineSearchW=True, lineSearchB=True, thresholdW=1e-4, thresholdB=1e-3,
                        thresholdFi=1e-5, thresholdLoss=1e-2, weightPredict=5e-1, weightReconstruct=15000, delta=1e-3,
                        gamma=1,
                        lossMode="MeanSquaredReturnError", wordNums=64,
                        betaB=0.1,
                        betaW=1e-6, betaFi=1, randomness=0.1, normalized=False, lineAlphaW=0.8, lineBetaW=0.5,
                        lineAlphaB=0.5, lineBetaB=0.5, dataSource=dataCollector)
    solver.Learn()
    # solver = None
    # filename = "./model/" + 'Train20_PreNEt_LogNorm_Batch' + ".pkl"
    # with open(filename, "rb") as file:
    #     solver = pickle.loads(file.read())
    checkAccuracy(solver)
    checkGetSparseForm(solver)
    print("Final  : Loss Part1: {:<22}  Loss Part2: {:<22}  Loss Part3: {:<22}  Loss Part4: {:<20}  Loss "
          "Part5: "
          "{:<20}".format(
        *tuple(solver.getLoss())), "Total Loss: {}".format(np.sum(solver.getLoss())))
    # SaveModel(solver, "Train20_PreNEt_LogNorm_Magnify_Batch")
