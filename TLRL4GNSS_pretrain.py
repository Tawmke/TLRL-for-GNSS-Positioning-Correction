import numpy as np
import gym
import logging
import torch
import scipy.io
import time
import sys
import os
import copy
import shutil
import threading
import multiprocessing
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch.nn as nn
from multiprocessing import Process
import time
import pickle

from torch.optim import lr_scheduler

device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )

def Sparse(z): #求稀疏表示的稀疏度
    # x = z.detach().numpy()
    x = z
    zero_idx = np.nonzero(x)
    h,w = np.shape(zero_idx)
    idx = np.size(x)
    sparsety = 1-w/idx
    return sparsety

def ProxSparseL0(u,theta):
    for idx in range(u.shape[1]):
        col = u[:,idx]
        col[abs(col) < theta] = 0
        u[: , idx] = col
    return u

def projtf(B):
    try:
        U, S, V = np.linalg.svd(B)
        return U @ np.eye(B.shape[0], B.shape[1]) @ V.T
    except:
        return B
    # M, N = np.size(B)


def train_DTL(trainX,trainY,wordNum,maxIter,WeightPred,weightRec,betaFi,reg_method,**kwargs):
    weightReconstruct = weightRec
    weightPredict = WeightPred
    innerloopW = 10
    innerloopB = 10
    innerloopFi = 10
    thresholdW = 5e-4
    thresholdB = 5e-3
    thresholdFi = 1e-5
    thresholdLoss = 3e-5 #这里改了，gsdcurban原来是5e-5
    betaB = 0.1
    betaW = 1e-6
    Wnormc = 100
    eachiterprint = 0
    iterprint = 1
    iterprintiter = 50
    finprint = 1
    B_init = 'Xrand'
    Dnorm = 'normSn'
    Fi_init = 'HardSC'
    W_init = 'zeros'
    wupdate = kwargs.get('wupdate')
    momstep = 0
################ initialization ##################
    #字典初始化：Xrand
    if B_init == 'Xrand':
        pos = torch.randperm(np.size(trainX,0))
        B = np.linalg.pinv(trainX[pos[0:wordNum],:].T).T
    #字典标准化：normSn
    B = projtf(B)
    normvD = np.sqrt(np.size(B, 0)) # sqrt m
    if Dnorm == 'normSn':
        normD = np.maximum(normvD,np.sqrt(np.sum(B*B,axis=0)))
        B = B @ np.diag(normvD / normD)
    elif Dnorm == 'normS':
        B = B @ np.diag(normvD / np.sqrt(np.sum(B * B, axis=0)))

    #稀疏表示初始化
    if Fi_init == 'pinv':
        trainFi = (np.linalg.pinv(B.T) @ trainX.T).T
        trainFi_ = copy.deepcopy(trainFi.T)
        iniFi = np.size(trainX, 0)
        for idx in range(iniFi):
            trainFi_[:, idx] = trainFi_[torch.randperm(wordNum), idx]
        trainFi = trainFi_.T
    elif Fi_init == 'HardSC':
        trainFi = trainX @ B
        val = np.sort(abs(trainFi.T),axis=0)
        thre = val[int(np.ceil(wordNum / 6))-1,:].T
        trainFi = ProxSparseL0(trainFi,thre)

    #权值初始化
    if W_init == 'norm':
        W = np.random.randn(wordNum, 1)
        normvD = np.sqrt(np.size(W, 0)) * Wnormc
        normD = np.maximum(normvD, np.sqrt(np.sum(W * W, axis=0)))
        W = W @ np.diag(normvD / normD)
    elif W_init == 'zeros':
        W = np.zeros([wordNum, 1])
############### Algorithm Start ####################
    MSTDE = []
    objerr = []
    sparsitylist = []
    losslist = []
    loss = getloss(trainY,trainFi,trainX,weightPredict,weightReconstruct,betaW,betaB,B,W,betaFi)
    MSTDE.append(loss[1])
    objerr.append(loss[0])
    # Wnorm.append(loss[2])
    # Bnorm.append(loss[3])
    losslist.append(loss[5])
    oldlossall = loss[5]
    print("Initial total Loss:", oldlossall)
    # 设置动量值
    tau = 1
    tau_old = tau
    lipschitzStepB = np.linalg.norm(trainX.T @ trainX)

    for iter in range(maxIter):
        # 设置动量值，这里没有设置 #
        ####  update W  ####
        if wupdate == 'yzt_re':
            tempDiff = - trainFi #- trainFi[0:-1, :]
            b = 2 * weightPredict * tempDiff.T @ trainY
            A = tempDiff.T @ tempDiff *2*weightPredict
            L_w = np.linalg.norm(A+np.eye(np.size(A,0))*betaW, 2)
            for e1 in range(innerloopW):
                updateW = b + A @ W + betaW * W
                W = W - updateW / L_w
                if np.abs(updateW).max() < thresholdW:
                    break
        elif wupdate == 'yzt': # TL here
            tempDiff = - trainFi  # - trainFi[0:-1, :]
            b = trainY * tempDiff
            b = weightPredict * np.mean(b,0)
            A = []
            for e1 in range(tempDiff.shape[0]):
                A.append(tempDiff[e1,:].reshape(-1,1) @ tempDiff[e1,:].reshape(1,-1))
            A = np.mean(np.stack(A,axis=2),2) * weightPredict
            L_w = np.linalg.norm(A) + betaW
            for e1 in range(innerloopW):
                updateW = -b.reshape(-1,1) - A @ W * weightPredict - betaW * W
                W = W + updateW/L_w
                if np.max(np.abs(updateW)) < thresholdW:
                    break

        ####  update Fi  ####
        tempY1 = copy.deepcopy(trainY)  # 源代码是补充维度？
        tempY2 = copy.deepcopy(trainY) # 这个地方bug
        tmp = weightPredict * W @ W.T
        L_Fi = np.linalg.norm(2 * (np.eye(len(W),1) * weightReconstruct + tmp), 2)
        stepSize = 1 / L_Fi
        for e1 in range(innerloopFi):
            trainFi_old = copy.deepcopy(trainFi)
            a = trainFi @ W - trainY
            c = copy.deepcopy(trainY)
            b = trainFi @ W
            U = trainFi - stepSize * ((trainFi - trainX @ B) * weightReconstruct * 2 +
                                            (b-c) * W.T * weightPredict * 2)
            if reg_method == 'L0':
                trainFi = ProxSparseL0(U,stepSize*betaFi)
                # ProxSparseL0(np.array([[0.1,0.5,3,6,7],[0.3,5,3,0.6,7]]), 2) # test
            sparsity = Sparse(trainFi)

        ####  update B  ####
        lipschitzStepB = np.linalg.norm(trainX.T @ trainX,2) + 1e-8
        for e1 in range(innerloopB):
            if reg_method == 'log':
                B_T = B.T
                E = trainFi.T @ trainFi
                HH = copy.deepcopy(E)
                HI = copy.deepcopy(trainFi)
                Eab = copy.deepcopy(E)
                P = trainX.T @ HI
                for bj in range(52):
                    Eab[bj,bj] = 0
                    B_T[:,bj] = (P[:,bj]-B_T @ Eab[:,bj])/HH[bj,bj]
                    nw = np.sqrt(np.sum(B_T[:,bj]*B_T[:,bj]))
                    B_T[:,bj] = B_T[:,bj] / nw
                updateB = B_T.T
                B = updateB
                if np.abs(updateB).max() < thresholdB:
                    break
            else:
                B_old = B
                updateB = trainX.T @ (trainX @ B - trainFi)/lipschitzStepB # PO
                B = B - updateB
                B = projtf(B)
                if Dnorm == 'normSn':
                    normD = np.maximum(normvD, np.sqrt(np.sum(B * B, axis=0)))
                    B = B @ np.diag(normvD / normD)
                elif Dnorm == 'normS':
                    B = B @ np.diag(normvD / np.sqrt(np.sum(B * B, axis=0)))
            if np.abs(updateB).max() < thresholdB:
                break

        loss = getloss(trainY, trainFi, trainX, weightPredict, weightReconstruct, betaW, betaB, B, W, betaFi)
        MSTDE.append(loss[1]) # 预测回报误差
        objerr.append(loss[0]) # 重构输入state误差
        losslist.append(loss[5])
        sparsitylist.append(sparsity)
        lossall = loss[5]

        if iter % iterprintiter == 0:
            print('Iter:{},MSTDE:{},RectrErr:{},Wnorm:{},Bnorm:{},'
                  'regloss:{},all_loss:{},sparsity:{}'.format(iter,loss[1],loss[0],loss[2],loss[3],loss[4],loss[5],sparsity))

        if np.abs(lossall) > 1000:
            break

        if np.abs((oldlossall-lossall)/np.abs(oldlossall)).max() < thresholdLoss or np.isnan(lossall):
            break
        oldlossall = lossall

    return B,W,trainFi,losslist,MSTDE,objerr,sparsitylist

def getloss(trainY,trainFi,trainX,weightPredict,weightReconstruct,betaW,betaB,B,W,betaFi):
    # 1、字典重构的损失函数loss
    reconstruct_loss = np.linalg.norm(trainFi - np.dot(trainX, B), 2,1) ** 2  # 求出二范数
    reconstruct_loss = weightReconstruct * np.mean(reconstruct_loss)

    # 2、预测值误差MSRE
    # td_err = (trainY - trainFi[0:-1, :] @ W) ** 2
    td_err = (trainY - trainFi @ W) ** 2
    MSRE_loss = weightPredict * np.mean(td_err)

    #3、W的二模范数约束
    w_loss = betaW * np.linalg.norm(W, 2) ** 2

    #4、B的二模范数约束
    B_loss = betaB * np.linalg.norm(B, 2) ** 2

    #5 L0约束
    tmp = np.abs(trainFi) > 0
    reg_loss = np.mean(np.sum(tmp,1)) * betaFi

    #5、总的损失函数
    total_loss = reconstruct_loss + MSRE_loss

    return [reconstruct_loss,MSRE_loss,w_loss,B_loss,reg_loss,total_loss]

def tensorloss(tensorY,tensorFi,tensorX,weightPredict,weightReconstruct,tensorB,tensorW,trainFi):
    # msre error
    len = np.size(trainFi,0)
    predictY = torch.narrow(tensorFi,0,0,len-1) @ tensorW
    tempDiff = tensorY - predictY
    MSRE_loss = torch.mean(tempDiff ** 2) * weightPredict
    # rec loss
    rec_loss = torch.mm(tensorFi, tensorB) - tensorX
    rec_loss = torch.norm(rec_loss,p=2,dim=1)**2
    rec_loss = weightReconstruct * torch.mean(rec_loss)

    losstensor = MSRE_loss + rec_loss
    return losstensor

def trainFi_12PO(U, trainFi, stepSize, betaFi):
    condition1 = np.abs(U) > np.power(2 * stepSize * betaFi, 2 / 3) * 3 / 2  # L1/2的PO算子
    condition2 = np.abs(U) <= np.power(2 * stepSize * betaFi, 2 / 3) * 3 / 2
    temp = 1 + np.cos(2 / 3 * np.arccos(
        -1 * np.power(3, 3 / 2) / 4 * 2 * stepSize * betaFi * np.power(
            np.abs(U[condition1]), -3 / 2)))
    trainFi[condition1] = (2 / 3) * U[condition1] * temp
    trainFi[condition2] = 0
    return trainFi

if __name__ == '__main__':

    datasets= 'TDurbancanyon'# 'GSDC2022urban' 'TDurbancanyon'
    dir_path = '/mnt/sdb/home/tangjh/DLRL4GNSSpos/'

    reg_method = 'L0'# 'L0','L0relu' 'log'
    layer = 1 #设定多层网络
    gamma = 1
    lossMode = 'MeanSquaredReturnError'
    thresholdFi = 5e-4

    maxIter = 3000 #训练迭代次数
    wupdate = 'yzt_re' # yzt_re_betaW yzt_re yzt
    testdate = '240802' # 240802
    wupdate_param=1.00e-06
    mean_times = 5
    modelname='preDTL_reinit'# preDTL_reW preDTL_ori g
    # preprocesses=['DQN_norm'] # 'DCT_IV','randn' DQN

    if datasets == 'GSDC2022urban':
        WeightPredlist = [8e-1,5e-1,1e-1,1e-2,1e-3,1e-4,5e-5,1e-5,5e-6]  # [8e-1,5e-1,1e-1,1e-2,1e-3,1e-4,5e-5,1e-5]
        weightReconstructlist = [1e-1] # [1e-2,5e-2,1e-3,1e-4]
        betaFilist = [1.3,1,8e-1,5e-1,3e-1,1e-1,5e-2,1e-2] # [8e-1,5e-1,3e-1,1e-1,5e-2,1e-2]
        with open(dir_path + f'trainX_data.pkl', "rb") as file:
            trainX = pickle.load(file)
        file.close()
        with open(dir_path + f'trainY_data.pkl', "rb") as file:
            trainY = pickle.load(file)
        file.close()
        Datanumslist = [9000]  # [100,500,1000,2000,4000,5000]

    elif datasets == 'TDurbancanyon':
        WeightPredlist = [1,8e-1,5e-1,1e-1,1e-2,1e-3,1e-4,5e-5,1e-5,5e-6]  # [1,8e-1,5e-1,1e-1,1e-2,1e-3,1e-4,5e-5,1e-5,5e-6]
        weightReconstructlist = [1e-1] # [1e-2,5e-2,1e-3,1e-4]
        betaFilist = [5e-4] # [5e-1,3e-1,1e-1,5e-2,1e-2,5e-3,1e-3,5e-4]
        with open(dir_path + f'TDcanyon_trainX_data.pkl', "rb") as file:
            trainX = pickle.load(file)
        file.close()
        with open(dir_path + f'TDcanyon_trainY_data.pkl', "rb") as file:
            trainY = pickle.load(file)
        file.close()
        Datanumslist = [14000]  # [100,500,1000,2000,4000,5000]


    betaB = 0.1
    betaW = 1e-6
    wordNumlayer = 128 # default = 64  # 第二层字典数 [32,64,128,256]

    for Datanums in Datanumslist:
        trainX = trainX[0:Datanums,:]
        trainY = trainY[0:Datanums]
        for WeightPred in WeightPredlist:
            for betaF in betaFilist:
                for weightRec in weightReconstructlist:
                    for runtime in range(mean_times):
                        #runtime = 5
                        t0 = time.time()
                        ###############################
                        wordNum = wordNumlayer #字典原子数
                        Dicname = 'Dic{}_{}'.format(np.size(trainX, 1),wordNum)
                        model_file_name = f'/Datanum{Datanums}_Dic{wordNumlayer}_WeightPred{WeightPred:.2e}_{reg_method}' \
                                          f'_WeightRec{weightRec:.2e}_beta{betaF:.2e}_{wupdate}/Run_{runtime+1}/'
                        savepath = dir_path + 'matlab_Model/' + modelname + '/' + datasets + testdate + model_file_name
                        print(model_file_name)
                        if not os.path.exists(savepath):
                            print('----------------LOOP:{}------------------'.format(runtime+1))
                            B,W,trainFi,losslist,MSTDE,objerr,sparsitylist = train_DTL(trainX,trainY,wordNum,maxIter,WeightPred,weightRec,betaF,reg_method,
                                                                                                 wupdate=wupdate)
                            os.makedirs(savepath)
                            t1 = time.time()
                            timecost = t1-t0

                            # 保存数据到mat文件
                            scipy.io.savemat(savepath + 'B.mat', {'B': B})
                            scipy.io.savemat(savepath + 'W.mat', {'W': W})
                            scipy.io.savemat(savepath + 'weightReconstruct.mat', {'weightReconstruct': weightRec})
                            scipy.io.savemat(savepath + 'gamma.mat', {'gamma': gamma})
                            scipy.io.savemat(savepath + 'lossMode.mat', {'lossMode': lossMode})
                            scipy.io.savemat(savepath + 'weightPredict.mat', {'weightPredict': WeightPred})
                            scipy.io.savemat(savepath + 'maxIter.mat', {'maxIter': maxIter})
                            scipy.io.savemat(savepath + 'betaW.mat', {'betaW': betaW})
                            # scipy.io.savemat(savepath + 'PreNet.mat', {'PreNet': PreNet})
                            # scipy.io.savemat(savepath + 'Bias.mat', {'Bias': Bias})
                            scipy.io.savemat(savepath + 'thresholdFi.mat', {'thresholdFi': thresholdFi})
                            scipy.io.savemat(savepath + 'record_loss.mat', {'losslist': losslist})
                            scipy.io.savemat(savepath + 'MSTDE.mat', {'MSTDE': MSTDE})
                            scipy.io.savemat(savepath + 'betaFi.mat', {'betaFi': betaF})
                            scipy.io.savemat(savepath + 'objerr.mat', {'objerr': objerr})
                            scipy.io.savemat(savepath + 'sparsity.mat', {'sparsity': sparsitylist})
                            scipy.io.savemat(savepath + 'timecost.mat', {'timecost': timecost})
                            scipy.io.savemat(savepath + 'trainFi.mat', {'trainFi': trainFi})

                            # txtpath = dir_path + 'matlab_Model/' + modelname + '/' + datasets + testdate + f'/Layer{layer}_Datanum{Datanums}_Dic{wordNumlayer}_WeightPred{WeightPred}_{reg_method}' \
                            #             f'_WeightRec{weightRec}_beta{betaF}_{wupdate}_{wupdate_param}/'
                            txtpath = savepath
                            with open(txtpath + f'all_loss={losslist[-1]},msre={MSTDE[-1]},rec={objerr[-1]}.txt', "w") as file:
                                file.write(f'all_loss={losslist[-1]},msre={MSTDE[-1]},rec={objerr[-1]}')

print('finish pretraining')














