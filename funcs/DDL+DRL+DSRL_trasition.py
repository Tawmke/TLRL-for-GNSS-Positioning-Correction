import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import copy
import scipy.io as scio
#from sklearn.svm import SVC
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import FashionMNIST, EMNIST,MNIST
#from matplotlib import pyplot as plt
#from svm_fashion_mnist import svm_train,svm_test
#from sklearn.neighbors import KNeighborsClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
layer_final_fig = 2638
# Deep Dictionary Configurations
input_dim = 2  # the input dimensions to be expected
dd_layer_config = [layer_final_fig//2, layer_final_fig//4, layer_final_fig]  # the layer configuration for the deep dictionary

epoch_per_level = 10  # the number of epochs to train for each layer of deep dictionary
torch.manual_seed(2) #随机种子


# prepare data loaders
trainXdata = scio.loadmat('trainX.mat')
trainX = trainXdata['trainX']
trainX = torch.tensor(trainX) #状态数据X
trainX = (trainX.T).to(torch.float32)

trainYdata = scio.loadmat('trainY.mat')
trainY = trainYdata['trainY']
trainY = torch.tensor(trainY) #状态数据X
trainY = (trainY.T).to(torch.float32)

#tensorboard的设置
writer = SummaryWriter(log_dir='explog')

#更新W时使用的一些参数
lineSearchW = 1 # 没用的
lineSearchB = 1 # 没用的
lineAlphaB = 0.5 #求字典用，没用
weightReconstruct = 900
lineAlphaW = 0.5 # 计算W时，用回溯线搜索算法的搜索精度指标，α
lineBetaW = 0.5 # 求w线性步长时用的β
lineBetaB = 0.5
betaW = 1e-6 # β_w, 预测权重约束强度
betaB = 0.1 # 字典约束强度
wordNum = 2 #状态维度
weightpredict = 0.06
maxlter = 450
maxltertest = 800

def Sparse(z): #求稀疏表示的稀疏度
    x = z.detach().numpy()
    zero_idx = np.nonzero(x)
    h,w = np.shape(zero_idx)
    idx = np.size(x)
    sparsety = w/idx
    return sparsety

class ELUInv:

    alpha = 1.

    @staticmethod
    def forward(x):
        #return (x > 0).float() * x + (x <= 0).float() * ELUInv.alpha * (torch.exp(x) - 1)
        return torch.where(x > 0, x, torch.log(x + 1.))

    @staticmethod
    def inverse(x):
        #return (x > 0).float() * x + (x <= 0).float() * torch.log((x / ELUInv.alpha) + 1)
        return torch.where(x > 0, x, ELUInv.alpha * (torch.exp(x) - 1.))

class TanhInv:
    alpha = 1.

    @staticmethod
    def forward(x):
        return TanhInv.alpha * torch.atan(x)

    @staticmethod
    def inverse(x):
        return torch.tanh(x / TanhInv.alpha)

#--------------------定义可学习正则子-------------------------
class LearnableThresholding(nn.Module):
    def __init__(self, ):
        super(LearnableThresholding, self).__init__()
        self.w1 = torch.nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)  # w11,b11,w22,b22
        self.b1 = torch.nn.Parameter(torch.FloatTensor([0.0005]), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor([1.]), requires_grad=True)
        self.b2 = torch.nn.Parameter(torch.FloatTensor([0.03]), requires_grad=True)
        self.Lip = torch.nn.Parameter(torch.FloatTensor([15000.]), requires_grad=True)
        #self.Lip=2.
        self.epsilon = 0.1

    def activate(self, x):
        """
        d1 = x >=self.weight[3]
        d2 = (x < self.weight[3]) & (x >= self.weight[1])
        d3 = (x < self.weight[1]) & (x >= -self.weight[1])
        d4 = (x < -self.weight[1]) & (x >= -self.weight[3])
        d5 = x < -self.weight[3]
        """
        # print(self.weight[0], self.weight[1], self.weight[2], self.weight[3])
        zz = torch.zeros_like(x).to(device) #生成和括号内变量维度维度一致的全是零的内容
        #print(x.is_cuda,self.w1.is_cuda,self.w2.is_cuda,self.b2.is_cuda,self.b1.is_cuda,zz.is_cuda)
        aa = torch.where(x >= self.b2, self.w2 * (x - self.b2) + self.w1 * (self.b2 - self.b1), zz)
        aa = torch.where((x < self.b2) & (x >= self.b1), self.w1 * (x - self.b1), aa)
        aa = torch.where((x >= (-self.b1)) & (x < self.b1), zz, aa)
        aa = torch.where((x < -self.b1) & (x >= -self.b2), self.w1 * (x + self.b1), aa)
        x = torch.where(x < -self.b2, self.w2 * (x + self.b2) + self.w1 * (self.b1 - self.b2), aa)
        """
        aa = torch.where(x >= self.b1, self.w1 * (x - self.b1), zz)
        aa = torch.where((x >= (-self.b1)) & (x < self.b1), zz, aa)
        x = torch.where(x < -self.b1, self.w1 * (x + self.b1), aa)
        """

        return x

    def forward(self, z1,z_prev):
        d_i = dd.dictionary_layers[2]
        z2 = self.activate(z1 - (1. / self.Lip) * (torch.mm(torch.mm(torch.t(d_i),d_i),z1) - torch.mm(torch.t(d_i),z_prev)))
        out = self.activate(z2 - (1. / self.Lip) * (torch.mm(torch.mm(torch.t(d_i),d_i),z2) - torch.mm(torch.t(d_i),z_prev)))
        #z4 = self.activate(z3 - (1. / self.Lip) *(torch.mm(torch.mm(torch.t(d_i),d_i),z3) - torch.mm(torch.t(d_i),z_prev)))
        #z5 = self.activate(z4 - (1. / self.Lip)*(torch.mm(torch.mm(torch.t(d_i),d_i),z4) - torch.mm(torch.t(d_i),z_prev)))
        #z6 = self.activate(z5 - (1. / self.Lip) *(torch.mm(torch.mm(torch.t(d_i),d_i),z5) - torch.mm(torch.t(d_i),z_prev)))
        #z7 = self.activate(z6 - (1. / self.Lip) *(torch.mm(torch.mm(torch.t(d_i),d_i),z6) - torch.mm(torch.t(d_i),z_prev)))
        #out = self.activate(z7 - (1. / self.Lip) * (torch.mm(torch.mm(torch.t(d_i), d_i), z7) - torch.mm(torch.t(d_i), z_prev)))
        #z9 = self.activate(z8 - (1. / self.Lip) * (torch.mm(torch.mm(torch.t(d_i), d_i), z8) - torch.mm(torch.t(d_i), z_prev)))
        #out = self.activate(z9 -(1. / self.Lip)*(torch.mm(torch.mm(torch.t(d_i),d_i),z9) - torch.mm(torch.t(d_i),z_prev)))
        return out

    def project(self):
        with torch.no_grad():
            if 0 < self.w1.data <= 1.:
                self.w1.data=torch.max(self.w1.data, torch.tensor(self.epsilon,device=device))
                # if self.w2.data > 1:
                if 1. >= self.w2.data > 0:
                    self.w2.data = self.w2.data
                else:
                    self.w2.data = torch.min(torch.max(self.w1.data, torch.tensor(self.epsilon,device=device)), torch.tensor(1.0,device=device))
                # self.w2.data = torch.min(torch.max(self.w1.data, torch.tensor(self.epsilon,device=device)), torch.tensor(1.0,device=device))
                # w22 = torch.min(torch.max(w1, torch.tensor(0.01,device=device)), torch.tensor(1.0,device=device))
                if (self.b1.data >= 0) & (self.b2.data >= self.b1.data):
                    self.b1.data = self.b1.data
                    self.b2.data = self.b2.data
                elif (self.b1.data < 0) & (self.b2.data > 0):
                    self.b1.data = torch.tensor(0.,device=device)
                    self.b2.data = self.b2.data
                elif self.b2.data <= torch.min(torch.tensor(0.,device=device), -self.b1.data):
                    self.b1.data = self.b2.data = torch.tensor(0.,device=device)
                elif self.b1.data >= torch.abs(self.b2.data):
                    self.b1.data = self.b2.data = (self.b1.data + self.b2.data) / 2

            if self.w1.data > 1.:
                p1 = ((self.w1.data - 1.) ** 2) / (self.w1.data ** 2 + (self.w1.data - 1.) ** 2)
                p2 = (self.w1.data * (self.w1.data - 1.)) / (self.w1.data ** 2 + (self.w1.data - 1.) ** 2)
                p3 = (self.w1.data ** 2) / (self.w1.data ** 2 + (self.w1.data - 1.) ** 2)
                # if self.w2.data > 1.:
                # print('the value of w22 is bigger than 1.0')
                if 1. >= self.w2.data > 0:
                    self.w2.data = self.w2.data
                else:
                    self.w2.data = torch.min(torch.max(self.w1.data, torch.tensor(self.epsilon,device=device)), torch.tensor(1.0,device=device))
                # print(self.w2.data)
                if (self.b2.data >= 0.) & (
                        ((self.w1.data - 1.) / self.w1.data) * self.b2.data <= self.b1.data <= self.b2.data):
                    self.b1.data = self.b1.data
                    self.b2.data = self.b2.data
                elif ((self.w1.data / (1. - self.w1.data)) * self.b2.data) < self.b1.data < (
                        (self.w1.data - 1.) / self.w1.data) * self.b2.data:
                    self.b1.data = p1 * self.b1.data + p2 * self.b2.data
                    self.b2.data = p2 * self.b1.data + p3 * self.b2.data
                elif (self.b2.data >= 0.) & (self.b1.data <= ((self.w1.data / (1. - self.w1.data)) * self.b2.data)):
                    self.b1.data = self.b2.data = torch.tensor(0.,device=device)
                elif self.b2.data <= torch.min(torch.tensor(0.,device=device), -self.b1.data):
                    self.b1.data = self.b2.data = torch.tensor(0.,device=device)
                elif self.b1.data >= torch.abs(self.b2.data):
                    self.b1.data = self.b2.data = (self.b1.data + self.b2.data) / 2.

DSRL = LearnableThresholding().to(device)

class DeepDictionary():

    def __init__(self, input_dim, **kwargs):
        """
        :param input_dim: the input dimension to be exp
        ected
        :param layer_config: the configuration for the layer dimensions of each dictionary layer
        :param activation: the activation function (default: None)
        :param sparseness_coeff: the sparseness coefficient
        """

        self.lter = 1 #近端迭代次数
        self.input_dim = input_dim

        self.layer_config = kwargs.get('layer_config', [self.input_dim, self.input_dim//2])  # list of layer dimensions
        if self.layer_config[0] != self.input_dim:  # in case where the layer config is given as hidden dimensions only
            self.layer_config = [self.input_dim] + self.layer_config
        assert len(self.layer_config) >= 2, "Error in specifying layer configuration, not enough layers"

        #self.activation = kwargs.get('activation', Identity)    # default implies linear
        self.activation = kwargs.get('activation', TanhInv)  # default implies linear
        self.spar_cff = kwargs.get('sparseness_coeff', 0)   # default sparseness coefficient

        # construct dictionary
        k = len(trainX)
        self.dictionary_layers = [nn.init.kaiming_normal_(
            torch.rand((self.layer_config[i], self.layer_config[i + 1]), requires_grad=False, device=device),
            mode='fan_in', nonlinearity='relu') for i in range(len(self.layer_config) - 1)]

        self.z_layers = [15 * nn.init.kaiming_normal_(
            torch.rand((self.layer_config[i + 1], layer_final_fig), requires_grad=False, device=device),
            mode='fan_in', nonlinearity='relu') for i in range(len(self.layer_config) - 1)]

        # self.dictionary_layers = [torch.rand((self.layer_config[i], self.layer_config[i+1]), requires_grad=False).to(device)
        #                           for i in range(len(self.layer_config)-1)]
        # self.z_layers = [torch.rand((self.layer_config[i+1], len(trainX)), requires_grad=False).to(device)
        #                           for i in range(len(self.layer_config)-1)]

    def eval_layer(self, x, layer, trainY, W, weightPredict, weightReconstruct, concat_prev=False):
        """
        :param x: input of dimension (batch x dim)
        :param layer: the layer to be evaluated at (value between 0 and len(self.dictionary_layers)-1)
        :return: (Z_i, Z_i-1) 'Z' at layer 'i' and 'i-1' (batch x dim[layer])
        """
        assert layer in range(0, len(self.dictionary_layers)), "Error with layer specified (out of range)"

        if layer == 0:
            with torch.no_grad():
                d_0 = self.dictionary_layers[layer]

                DT = torch.mm(torch.t(d_0),d_0)
                DT = DT + 0.0001 * torch.eye(DT.shape[0])  # 矩阵每个元素加上一个小值，防止为奇异矩阵

                z_0 =torch.mm( torch.mm(torch.inverse(DT),torch.t(d_0)),x)  # first obtain Z_0

                ZT = torch.mm(z_0,torch.t(z_0))
                ZT = ZT + 0.0001 * torch.eye(ZT.shape[0]) # 矩阵每个元素加上一个小值，防止为奇异矩阵

                self.dictionary_layers[layer] = torch.mm(torch.mm(x,torch.t(z_0)),torch.inverse(ZT)) # get optimal dictionary
                z_layer_rec = torch.mm(self.dictionary_layers[layer],z_0)  # if first layer, linear activation
                self.z_layers[layer] = z_0
                lat_rec_loss = torch.sum((x - z_layer_rec) ** 2)

        elif layer == 1:
            with torch.no_grad():
                d_1 = self.dictionary_layers[layer]
                z_i_prev_ia1 = self.activation.forward(self.z_layers[layer-1])

                DT = torch.mm(torch.t(d_1),d_1)
                DT = DT + 0.0001 * torch.eye(DT.shape[0])  # 矩阵每个元素加上一个小值，防止为奇异矩阵

                z_1 =torch.mm( torch.mm(torch.inverse(DT),torch.t(d_1)),z_i_prev_ia1)   # otherwise, treat as regular

                ZT = torch.mm(z_1,torch.t(z_1))
                ZT = ZT + 0.0001 * torch.eye(ZT.shape[0]) # 矩阵每个元素加上一个小值，防止为奇异矩阵

                self.dictionary_layers[layer] =torch.mm(torch.mm(z_i_prev_ia1,torch.t(z_1)),torch.inverse(ZT))
                z_layer_rec = self.activation.inverse(torch.mm(self.dictionary_layers[layer],z_1))  # otherwise, non-linear activation
                self.z_layers[layer] = z_1
                lat_rec_loss = torch.sum((self.z_layers[layer-1]- z_layer_rec) ** 2)

        elif layer==2:
            d_2 = self.dictionary_layers[layer]
            z_i_prev_ia1 = self.activation.forward(self.z_layers[layer - 1])
            #z_i_pre = z_i_prev_ia1.numpy() #上一层表示转化为矩阵
            #D = d_2.detach().numpy() #字典转化为矩阵

        #--------------稀疏表示更新: 加入DSRL+强化学习的目标函数处理-----------------
            z_2 = self.z_layers[2]
            print(z_2)
            trainY = torch.t(trainY)
            with torch.no_grad():
                W = torch.from_numpy(W)
            #trainY = trainY.numpy().T
            tempY2 = np.append(trainY != 0, np.array([1]).reshape(-1,1), 0) #在trainY的第一个元素后加一个1的元素:2438*1
            tempY2 = torch.from_numpy(tempY2)

            #optimizer.zero_grad()
            DSRL.project()
            #1、计算梯度
            for ep in range(self.lter):
                z_prev = z_2
                trainFi = z_2
                #trainFi = trainFi.detach().numpy()
                c = trainY
                #c = np.append(c, trainFi[-1,:].reshape(1,-1) @ W, 0) # 2638*1
                c = torch.cat((c,trainFi[-1,:].reshape(1,-1) @ W),dim=0) #拼接
                b = torch.t(trainFi) @ W

                #trainFi_gra = trainFi - D.T @ (D @ trainFi - z_i_prev_ia1) * weightReconstruct * 2 #重构的梯度
                trainFi_gra = trainFi - torch.t(d_2) @ (d_2 @ trainFi - z_i_prev_ia1) * weightReconstruct * 2  # 重构的梯度
                #trainFi = trainFi - stepSize * torch.mm(d_2,(torch.mm(d_2,trainFi) - z_i_prev_ia1)) * weightReconstruct * 2 # 重构的梯度
                trainFi_gra = trainFi_gra - (b - c) * torch.t(W) * weightPredict * 2 * tempY2 #预测的梯度
                #trainFi = trainFi - stepSize * (b - c) * torch.T(W) * tempY2 * weightPredict * 2

                #print(trainFi_gra)

            #2 可学习正则子近端算子求出稀疏系数
                #trainFi_gra = torch.from_numpy(trainFi_gra)
                with torch.no_grad():
                    trainFi = DSRL.activate(
                        z_prev - (1. / DSRL.Lip) * trainFi_gra)

                #z_2 = trainFi.to(torch.float32) #转为32位浮点型的张量，不然数据会不匹配
                z_2 = DSRL.forward(trainFi, z_i_prev_ia1) #前向传播更新fi
                self.z_layers[layer] = z_2

                #lat_rec_loss = torch.sum((self.z_layers[layer - 1] - z_layer_rec) ** 2)
                #print(z_2)

         # -----------------字典更新-----------------------------
            with torch.no_grad():

                ZT = torch.mm(z_2, torch.t(z_2))
                ZT = ZT + 0.0001 * torch.eye(ZT.shape[0])  # 矩阵每个元素加上一个小值，防止为奇异矩阵
                self.dictionary_layers[layer] = torch.mm(torch.mm(z_i_prev_ia1, torch.t(z_2)), torch.inverse(ZT))

            z_layer_rec = self.activation.inverse(
                torch.mm(self.dictionary_layers[layer], z_2))  # otherwise, non-linear activation

            lat_rec_loss =  torch.sum((self.z_layers[layer - 1] - z_layer_rec) ** 2)  # 最后一层的重构误差

            #计算预测的价值函数
            Wi = W.numpy()
            trainFi = z_2.detach().numpy()
            trainFi = trainFi.T  # 2637 * 2
            predictY = trainFi[0:-1,:] @ Wi
            PredictY = torch.from_numpy(predictY)

            #计算预测价值函数的误差MSRE
            # nonGapRow = trainY != 0
            # nonGapRow = nonGapRow.numpy()
            tempDiff = trainY - PredictY
            MSRE_loss =  torch.sum(tempDiff ** 2) * weightpredict

            loss = MSRE_loss + lat_rec_loss * weightReconstruct
            loss.backward()  # 反向传播更新参数
            optimizer.step()

            print(DSRL.w1, DSRL.b1, DSRL.w2, DSRL.b2, DSRL.Lip)

            sparsity = Sparse(z_2) #计算稀疏度

            writer.add_scalar('Sparsety', sparsity, epoch)
            print('Sparsety={},latent Loss={}'.format(sparsity, lat_rec_loss))

        return lat_rec_loss

    def reconstruction(self, z_2):
        """
        Performs total reconstruction of the input image
        :param x: input of dimension (batch x dim)
        :return: a reconstruction of 'x' based on learned dictionaries
        """
        with torch.no_grad():
            z_1 = self.activation.inverse(torch.mm(self.dictionary_layers[2], z_2))
            z_0 = self.activation.inverse(torch.mm(self.dictionary_layers[1], z_1))
            x_rec = torch.mm(self.dictionary_layers[0], z_0)

        return x_rec

def updateW(W, trainX, trainY, z_2, D, weightReconstruct, weightpredict, reconstruct_loss, betaW, betaB, lineSearchW, lineAlphaW, lineBetaW):
    # 训练
    trainY = trainY.detach().numpy()
    trainY = trainY.T
    trainFi = z_2.detach().numpy() #是否需要转置

    # nonGapRow = trainY != 0 # 2637 * 1 ---> 2618 * 1
    tempDiff =  - trainFi[0:-1,:] #2637 * 2638
    #
    #
    # tempDiff1 = tempDiff[:, 0].reshape(-1, 1)  # 第一列的元素索引
    # tempDiff1 = tempDiff1[nonGapRow].reshape(-1, 1)
    # for idx in range(layer_final_fig-1):
    #     tempDiff2 = tempDiff[:,idx+1].reshape(-1,1)
    #     tempDiff1 = np.append(tempDiff1,tempDiff2[nonGapRow].reshape(-1,1),axis=1) #2618 * 2638
    # tempDiff = tempDiff1

    # b的更新
    #b = trainY[nonGapRow].reshape(-1,1) * tempDiff  # 2618 * 2638
    b = trainY.reshape(-1, 1) * tempDiff  # 2618 * 2638
    b = weightpredict * np.mean(b, 0).reshape(1,layer_final_fig)  # 0是指每一列的均值，1 * 2638

    #A的更新
    for i in range(np.size(tempDiff, 0)): # size返回temDiff的行数，2618
        if i == 0:
            A =  tempDiff[i, :].reshape(1, -1) @ tempDiff[i, :].T.reshape(-1, 1) # 2638 * 2638 这地方是不是有问题
        else:
            temp = A
            temp2 = tempDiff[i, :].reshape(1, -1) @ tempDiff[i, :].T.reshape(-1, 1)
            A = np.add(temp, temp2)               # 将对应位置的元素
    A = weightpredict * A / np.size(tempDiff, 0)
    A = A[0,0]
    #更新W
    updateW =  - b.T - A * W * weightpredict - betaW * W #梯度
    updateW = updateW * 0.8 #lineSearchWF(weightpredict, weightReconstruct, trainX, trainY, trainFi, W, reconstruct_loss, betaW, betaB, updateW, lineAlphaW, lineBetaW) #看看是否有需要使用回溯直线搜索法
    W = W + updateW

    for i in range(1):
        updateW = - b.T - A * W * weightpredict - betaW * W  # 梯度
        updateW = updateW * 0.8#lineSearchWF(weightpredict, weightReconstruct, trainX, trainY, trainFi, W, reconstruct_loss, betaW, betaB, updateW, lineAlphaW, lineBetaW)  # 看看是否有需要使用回溯直线
        W = W + updateW

    print(W)
    return W

def lineSearchWF(weightpredict, weightReconstruct, trainX, trainY, trainFi, W, reconstruct_loss, betaW, betaB, updateW, lineAlphaW, lineBetaW):#回溯线性法
    stepSize = 1
    nextW = W + updateW * stepSize # 下一步的W
    tempW = W # 当前的W
    trainFi = trainFi[0:-1, :]  # 2637 * 2
    # 把下一步的W赋值，然后item1就是对总损失函数求和
    W = nextW

    PredictY = trainFi @ W
    nonGapRow = trainY != 0 #求损失函数
    tempDiff = trainY - PredictY
    tempDiff = tempDiff[nonGapRow]
    MSRE_loss = weightpredict * np.mean(tempDiff ** 2)
    w_loss = betaW * (np.linalg.norm(W, ord=None, axis=None, keepdims=False) ** 2)

    item1 = MSRE_loss + w_loss#求出item1第一种情况

    #把当前的W赋值，然后item2
    W = tempW

    PredictY = trainFi @ W
    nonGapRow = trainY != 0 #求损失函数
    tempDiff = trainY - PredictY
    tempDiff = tempDiff[nonGapRow]
    MSRE_loss = weightpredict * np.mean(tempDiff ** 2)
    w_loss = betaW * (np.linalg.norm(W, ord=None, axis=None, keepdims=False) ** 2)
    item2 = MSRE_loss + w_loss + lineAlphaW * stepSize * updateW.T @ getDerivativeOverW(weightpredict, trainX, trainY, trainFi, W, betaW)
    item2 = item2[0,0]

    while item1 > item2:
        stepSize = stepSize * lineBetaW

        nextW = W + updateW * stepSize
        tempW = W

        W = nextW

        PredictY = trainFi @ W
        tempDiff = trainY - PredictY
        tempDiff = tempDiff[nonGapRow]
        MSRE_loss = weightpredict * np.mean(tempDiff ** 2)
        w_loss = betaW * (np.linalg.norm(W, ord=None, axis=None, keepdims=False) ** 2)
        item1 = MSRE_loss + w_loss  # 求出item1第一种情况
        # 把当前的W赋值，然后item2
        W = tempW

        PredictY = trainFi @ W
        tempDiff = trainY - PredictY
        tempDiff = tempDiff[nonGapRow]
        w_loss = betaW * (np.linalg.norm(W, ord=None, axis=None, keepdims=False) ** 2)
        item2 = MSRE_loss + w_loss + lineAlphaW * stepSize * updateW.T @ getDerivativeOverW(weightpredict, trainX,
                                                                                            trainY, trainFi, W, betaW)
        item2 = item2[0,0]

    return stepSize

def getDerivativeOverW(weightpredict, trainX, trainY, trainFi, W, betaW):
    nonGapRow = trainY != 0

    PredictY = trainFi @ W
    res = 2 * (trainY - PredictY)
    res = res * (-1*trainFi)

    # res1 = res[:, 0].reshape(-1, 1)  # 第一列的元素索引
    # res1 = res1[nonGapRow].reshape(-1, 1)
    # for idx in range(layer_final_fig - 1):
    #     res2 = res[:, idx + 1].reshape(-1, 1)
    #     res1 = np.append(res1, res2[nonGapRow].reshape(-1, 1), axis=1)  # 2618 * 2638
    # res = res1

    res = np.mean(res,axis=0).reshape(-1,layer_final_fig) #计算每一列的均值
    res = res.T * weightpredict + 2 * betaW * W

    return res

def YReconstruct(z_2,W): #重构预测的Y值
    trainFi = z_2.detach().numpy()
    trainFi = trainFi[:,0:-1]# 2637 * 2
    predictY = trainFi.T @ W
    predictY = torch.from_numpy(predictY)
    return predictY

def getloss(trainY,trainFi,trainX,trainX_rec,PredictY,weightpredict,betaW,W):
    #1、字典重构的损失函数loss
    reconstruct_loss = np.linalg.norm(trainX - trainX_rec, ord=None, axis=1, keepdims=False) ** 2  # 求出二范数
    # print(trainX.T)
    # print(trainX_rec)
    reconstruct_loss = weightReconstruct * np.mean(reconstruct_loss)
    reconstruct_loss = torch.tensor(reconstruct_loss)  # 用二模定义的loss

    #2、预测值误差MSRE
    nonGapRow = trainY != 0
    nonGapRow = nonGapRow.numpy().T
    tempDiff = trainY.T - PredictY
    tempDiff = tempDiff.numpy() #转化为矩阵操作
    tempDiff = tempDiff[nonGapRow]
    MSRE_loss = weightpredict * np.mean(tempDiff ** 2)

    #3、W的二模范数约束
    w_loss = betaW * (np.linalg.norm(W, ord=None, axis=None, keepdims=False) ** 2)

    #4、trainFi的一模范数约束
    Fi = trainFi.detach().numpy()
    Fi_loss = np.linalg.norm(Fi, ord=1, axis=1, keepdims=False)
    Fi_loss =  np.mean(Fi_loss)

    #5、总的损失函数
    total_loss = reconstruct_loss + MSRE_loss + w_loss + Fi_loss

    return reconstruct_loss,MSRE_loss,w_loss,Fi_loss,total_loss

# define the models of the Deep Dictionary

dd = DeepDictionary(input_dim=input_dim, layer_config=dd_layer_config,
                          activation=TanhInv)

#z_2 = torch.rand((layer_final_fig,len(trainX)),requires_grad=False).to(device)
#z_2 = torch.torch.zeros((layer_final_fig,2),requires_grad=False).to(device) #fi初始化为0矩阵
W = nn.init.kaiming_normal_(15 *
            torch.rand((layer_final_fig, 1), requires_grad=False, device=device),
            mode='fan_in', nonlinearity='relu')
W = W.numpy()

mlp_lr_para = 0.000008
optimizer = torch.optim.Adam(DSRL.parameters(), mlp_lr_para) #确立优化器

# begin model trainings
print('BEGIN TRAINING THE DEEP DICTIONARY MODEL')
trainX = trainX.to(device)  #将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行。
for layer_i in range(len(dd.dictionary_layers)):
    epoch_per_level = maxlter if layer_i == 2 else epoch_per_level  # 更新最后一层的时候迭代多次
    for epoch in range(epoch_per_level):

        # 计算估测值trainY和trainX

        # optimization 得到更新的W和FI
        #lat_rec_loss, _ = dd.eval_layer(torch.t(trainX), layer_i, z_2, trainY, W, weightpredict, weightReconstruct)  # 最后一层的重构误差
        lat_rec_loss = dd.eval_layer(trainX, layer_i, trainY, W, weightpredict,
                                        weightReconstruct)  # 最后一层的重构误差

        z_2 = dd.z_layers[2]
        PredictY = YReconstruct(z_2,W)
        trainX_rec = dd.reconstruction(z_2)

        if layer_i == 2 : #最后一层才更新W
            W = updateW(W, trainX, trainY, z_2, dd.dictionary_layers, weightReconstruct, weightpredict, reconstruct_loss, betaW, betaB, lineSearchW, lineAlphaW, lineBetaW)

        #计算loss

        reconstruct_loss,MSRE_loss, w_loss, Fi_loss, total_loss = getloss(trainY,z_2,trainX,trainX_rec,PredictY,weightpredict,betaW,W)
        writer.add_scalar('reconstruct_loss', reconstruct_loss, epoch)
        writer.add_scalar('MSRE_loss', MSRE_loss, epoch)
        writer.add_scalar('total_loss', total_loss, epoch)
        print(f'Layer: {layer_i} | Epoch:{epoch} - total loss:{total_loss} - '
              f'| Latent Loss: {lat_rec_loss:.4f} - MSRE Loss:{MSRE_loss} - reconstruct loss:{reconstruct_loss:.4F}')
              #f'|  Latent Loss: {lat_rec_loss:.4f}')

writer.close()
#
print(f'---------------------------- FINAL TEST RESULTS ----------------------------------------')
# print(f'[TEST] | Loss: {test_loss_avg:.4f} | ACC: {test_acc:.4f}')
print('---------BEGIN TEST THE DDL+DRL MODEL-------------')

def YReconstructtest(z_2,W): #重构预测的Y值
    trainFi = z_2.numpy().T
    predictY = trainFi @ W
    predictY = torch.from_numpy(predictY)
    return predictY

class DeepDictionarytest(): #定义新的作为测试

    def __init__(self, lambuda, input_dim, **kwargs):
        """
        :param input_dim: the input dimension to be expected
        :param layer_config: the configuration for the layer dimensions of each dictionary layer
        :param activation: the activation function (default: None)
        :param sparseness_coeff: the sparseness coefficient
        """
        self.lambuda= lambuda #L1正则项系数
        #self.lambuda1=0.001
        #self.mu=0.000005 #近端梯度步长
        self.lter = 70 #近端迭代次数
        self.input_dim = input_dim

        self.layer_config = kwargs.get('layer_config', [self.input_dim, self.input_dim//2])  # list of layer dimensions
        if self.layer_config[0] != self.input_dim:  # in case where the layer config is given as hidden dimensions only
            self.layer_config = [self.input_dim] + self.layer_config
        assert len(self.layer_config) >= 2, "Error in specifying layer configuration, not enough layers"

        #self.activation = kwargs.get('activation', Identity)    # default implies linear
        self.activation = kwargs.get('activation', TanhInv)  # default implies linear

        # construct dictionary
        self.dictionary_layers = [nn.init.kaiming_normal_(
            torch.rand((self.layer_config[i], self.layer_config[i + 1]), requires_grad=False, device=device),
            mode='fan_in', nonlinearity='relu') for i in range(len(self.layer_config) - 1)]

        self.dictionary_layers[0] = dd.dictionary_layers[0] #学习过的字典赋予给测试的模型
        self.dictionary_layers[1] = dd.dictionary_layers[1]
        self.dictionary_layers[2] = dd.dictionary_layers[2]

        self.z_layers = [ 0.1 * nn.init.kaiming_normal_(
            torch.rand((self.layer_config[i + 1], 1), requires_grad=False, device=device),
            mode='fan_in', nonlinearity='relu') for i in range(len(self.layer_config) - 1)]

        # self.dictionary_layers = [torch.rand((self.layer_config[i], self.layer_config[i+1]), requires_grad=False).to(device)
        #                           for i in range(len(self.layer_config)-1)]
        # self.z_layers = [torch.rand((self.layer_config[i+1], len(trainX)), requires_grad=False).to(device)
        #                           for i in range(len(self.layer_config)-1)]

    def eval_layer(self, x, layer,z_2, trainY, W, weightPredict, weightReconstruct, concat_prev=False):
        """
        :param x: input of dimension (batch x dim)
        :param layer: the layer to be evaluated at (value between 0 and len(self.dictionary_layers)-1)
        :return: (Z_i, Z_i-1) 'Z' at layer 'i' and 'i-1' (batch x dim[layer])
        """
        assert layer in range(0, len(self.dictionary_layers)), "Error with layer specified (out of range)"

        if layer == 0:
            d_0 = self.dictionary_layers[layer]

            DT = torch.mm(torch.t(d_0),d_0)
            DT = DT + 0.0001 * torch.eye(DT.shape[0])  # 矩阵每个元素加上一个小值，防止为奇异矩阵

            z_0 =torch.mm( torch.mm(torch.inverse(DT),torch.t(d_0)),x)  # first obtain Z_0

            z_layer_rec = torch.mm(self.dictionary_layers[layer],z_0)  # if first layer, linear activation
            self.z_layers[layer] = z_0
            lat_rec_loss = torch.sum((x - z_layer_rec) ** 2)

        elif layer == 1:
            d_1 = self.dictionary_layers[layer]
            z_i_prev_ia1 = self.activation.forward(self.z_layers[layer-1])

            DT = torch.mm(torch.t(d_1),d_1)
            DT = DT + 0.0001 * torch.eye(DT.shape[0])  # 矩阵每个元素加上一个小值，防止为奇异矩阵

            z_1 =torch.mm( torch.mm(torch.inverse(DT),torch.t(d_1)),z_i_prev_ia1)   # otherwise, treat as regular

            z_layer_rec = self.activation.inverse(torch.mm(self.dictionary_layers[layer],z_1))  # otherwise, non-linear activation
            self.z_layers[layer] = z_1
            lat_rec_loss = torch.sum((self.z_layers[layer-1]- z_layer_rec) ** 2)

        elif layer==2:
            d_2 = self.dictionary_layers[layer]
            z_i_prev_ia1 = self.activation.forward(self.z_layers[layer - 1])
            z_i_pre = z_i_prev_ia1.numpy() #上一层表示转化为矩阵
            D = d_2.numpy() #字典转化为矩阵

        #--------------稀疏表示更新: 加入强化学习的目标函数处理-----------------
            z_2 = self.z_layers[2]
            trainFi = z_2.numpy()
            trainY = trainY.numpy().T
            #1 求利普希茨系数作为步长
            w_norm2 = np.linalg.norm(W, ord=None, axis=None, keepdims=False) ** 2 #ord默认为二模，axis：处理类型，默认矩阵处理，
            D_norm2 = np.linalg.norm(D, ord=None, axis=None, keepdims=False) ** 2

            stepSize = 0.5 / ( D_norm2 * weightReconstruct)  # 测试字典重构的时候，没有权重的加入
            stepSize = 0.2 * stepSize #for safety

            #2、计算梯度
            for ep in range(self.lter):

                trainFi = trainFi - stepSize * D.T @ (D @ trainFi - z_i_pre) * weightReconstruct * 2 #重构的梯度

                toc = self.lambuda * stepSize

            #3 软阈值算法求出稀疏系数
                trainFi = np.sign(trainFi) * np.maximum(np.abs(trainFi) - toc, 0)

            z_2 = torch.from_numpy(trainFi).to(torch.float32) #转为32位浮点型的张量，不然数据会不匹配
            self.z_layers[layer] = z_2

            z_layer_rec = self.activation.inverse(torch.mm(self.dictionary_layers[layer],z_2))  # otherwise, non-linear activation

            lat_rec_loss = torch.sum((self.z_layers[layer-1] - z_layer_rec) ** 2) #最后一层的重构误差

        return lat_rec_loss, z_2

    def reconstruction(self, z_2):
        """
        Performs total reconstruction of the input image
        :param x: input of dimension (batch x dim)
        :return: a reconstruction of 'x' based on learned dictionaries
        """

        z_1 = self.activation.inverse(torch.mm(self.dictionary_layers[2], z_2))
        z_0 = self.activation.inverse(torch.mm(self.dictionary_layers[1], z_1))
        x_rec = torch.mm(self.dictionary_layers[0], z_0)

        return x_rec

def getlosstest(trainY,trainFi,trainX,trainX_rec,PredictY,lamubda,weightpredict,betaW,W):
    #1、字典重构的损失函数loss
    reconstruct_loss = np.linalg.norm(trainX - trainX_rec, ord=None, axis=1, keepdims=False) ** 2  # 求出二范数
    # print(trainX.T)
    # print(trainX_rec)
    reconstruct_loss = weightReconstruct * np.mean(reconstruct_loss)
    reconstruct_loss = torch.tensor(reconstruct_loss)  # 用二模定义的loss

    #2、预测值误差MSRE
    print(trainY)
    print(PredictY)
    tempDiff = trainY - PredictY
    tempDiff = tempDiff.numpy() #转化为矩阵操作
    MSRE_loss = weightpredict * np.mean(tempDiff ** 2)

    #3、预测的正确率
    acc = np.abs(tempDiff)/PredictY

    #5、总的损失函数
    total_loss = reconstruct_loss + MSRE_loss + w_loss + Fi_loss

    return reconstruct_loss,MSRE_loss,acc

for i in range(2638):
    testX = trainX[:,i].reshape(-1,1) #提取出每个状态作为输入测试数据
    ddtest = DeepDictionarytest(lambuda, input_dim=input_dim, layer_config=dd_layer_config,
                                activation=TanhInv)
    for layer_i in range(len(dd.dictionary_layers)):
        epoch_per_level = maxltertest if layer_i == 2 else epoch_per_level  # 更新最后一层的时候迭代多次
        for epoch in range(epoch_per_level):

            testX = testX.to(device)  # 将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行。

            z_2 = ddtest.z_layers[2]
            # 计算估测值trainY和trainX
            PredictY = YReconstructtest(z_2, W)
            testX_rec = ddtest.reconstruction(z_2)

            # optimization 得到更新的W和FI
            # lat_rec_loss, _ = dd.eval_layer(torch.t(trainX), layer_i, z_2, trainY, W, weightpredict, weightReconstruct)  # 最后一层的重构误差
            lat_rec_loss, _ = ddtest.eval_layer(testX, layer_i, z_2, trainY, W, weightpredict,
                                            weightReconstruct)  # 最后一层的重构误差
            z_2 = ddtest.z_layers[2]

        # 计算测试的loss
        Ytest = trainY[0,i].reshape(-1,1)
        reconstruct_loss, MSRE_loss, acc = getlosstest(Ytest, z_2, testX, testX_rec, PredictY,
                                                                           lambuda, weightpredict, betaW, W)
        sparsity = Sparse(z_2)  # 计算稀疏度
        #writer.add_scalar('Sparsety', sparsity, epoch)
        print('Sparsety={}, Ysample={}'.format(sparsity, i))
        # writer.add_scalar('reconstruct_loss', reconstruct_loss, epoch)
        # writer.add_scalar('MSRE_loss', MSRE_loss, epoch)
        # writer.add_scalar('total_loss', total_loss, epoch)
        print(f'Layer: {layer_i} | Epoch:{epoch}  - '
              f'| Latent Loss: {lat_rec_loss:.4f} - test MSRE Loss:{MSRE_loss} test accuracy:{acc}- test reconstruct loss:{reconstruct_loss:.4F}')
        # f'|  Latent Loss: {lat_rec_loss:.4f}')

    del ddtest









