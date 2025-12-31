import numpy as np
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
import matplotlib.pyplot as plt

from multiprocessing import Process
from funcs.control_model_DDL_preNet import *
from funcs.control_model_DDL_preNet_SGD import *
from funcs.control_model_DL_preNet_DSRL import *

def CheckSimilarity(B):
    print("Word Similarity:")
    temp = np.array([word / np.linalg.norm(word) for word in B])
    temp = np.dot(temp, temp.T)
    for i in range(temp.shape[0]):
        for j in range(i, temp.shape[1]):
            print("Similarity between {} and {}: {}".format(i, j, np.abs(temp[i][j])))

def multiThreadFun(k, epinum, model_file_name, stepSize, outputFile, weightRec, reg_method, Model_folder, adapstep, revise, dataset, preprocess, printtrig, Wcontrolmode):
    decay = 0.99
    print("outputFile:{}".format(outputFile))
    actionLearner = ActionLearner_DSRL(actionRandomness=0.1, decay=decay, stepSize=stepSize, loadBasic=model_file_name,
                                       index=k, outputFile=outputFile, weightRec=weightRec, WeightPred=WeightPred,
                                       reg_method=reg_method,
                                       Model_folder=Model_folder, dataset=dataset, preprocess=preprocess,
                                       printtrig=printtrig, Wcontrolmode=Wcontrolmode)
    # epinum=100
    episode_rewards, episode_loss = actionLearner.learn(epinum)


if __name__ == '__main__':
    #betaFis = [2.15e-,4.64e-,1e-  ,3.16e-,1e-]
    datasets= ['CP']# 'MC','PW','AC','CA3','CP'
    reg_methods = ['DSRL']# 'L0','L0relu''DSRL'
    Datanums = [5000]
    stepSizes = [3e-1,5e-1,6e-1,6.5e-1,7e-1]#[4e-1,3e-1,1e-1,5e-2] [4e-1,5e-1,6e-1] [3.5e-1,4e-1,4.5e-1]
    betaFis_indexs = [1.0]
    reW = 1
    epinum = 5000

    wupdate = 'yzt_re_betaW'#yzt_re_betaW
    wupdate_param=1.00e-06
    mean_times = 10
    modelname='preDDL'# preDTL_reW preDTL_orig
    preprocesses=['DQN_norm'] # 'DCT_IV','randn' DQN
    revise = 1
    Wcontrolmode = 4
    adapstep = 0
    complete_run = 5
    printtrig=1
    singlemultimode=1

    ts = []

    for dataset in datasets:
        if dataset=='MC':
            epinum = 100
            run_date='1226'
            WeightPreds = [1e-3,0.5*1e-2,1e-2]
            betaFis = [1e-3,0.5*1e-2,1e-2]
            # WeightPreds = [1e-7,1e-6,1e-5]
            # betaFis = [1e-2,3.16e-2,1e-1,3.16e-1,1e-0]
            #[1e-4,3.16e-4,1e-3,3.16e-3,1e-2,3.16e-2,1e-1]
        elif dataset=='PW':
            epinum = 250
            run_date='0000'
            outputdate = '1111'
            WeightPreds = [1e-1] # [1e-5,1e-4,1e-3] [1e-5,1e-4,1e-3,5e-2,1e-2,1e-1] [2e-1,3e-1,5e-1]
            WeightReconstructlist = [1e-2] # [1e-4,5e-3,1e-3,1e-2] [1e-2,5e-2,1e-1,5e-1]
        elif dataset=='AC':
            epinum = 250
            run_date='0000'
            outputdate = '0413'
            WeightPreds = [1e-1] # [1e-5,1e-4,1e-3,5e-2,1e-2]  [5e-2,1e-1]
            WeightReconstructlist = [5e-3] # [0.05,0.5] [1e-4,5e-3]
            delta = 1e-3
        elif dataset=='CP':
            epinum = 250
            run_date='0000'
            outputdate = '0000'
            WeightPreds = [5e-2,1e-2,1e-1]
            WeightReconstructlist = [1e-4,5e-3,1e-3,1e-2]
        elif dataset=='CA3':
            epinum = 500
            run_date='1201'
            WeightPreds = [1e-5,1e-4,1e-3,1e-2]
            betaFis = [1e-4,3.16e-4,1e-3,3.16e-3,1e-2,3.16e-2,1e-1]

        if reW > 0.1:
            Model_folder = '../pre_train/matlab_Model/' + modelname + '/' + dataset + run_date
        else:
            Model_folder = '../pre_train/matlab_Model/' + modelname + '/' + dataset + run_date

        for preprocess in preprocesses:
            for reg_method in reg_methods:
                if reg_method == 'L0relu':
                    reg_method1='L0'
                else:
                    reg_method1=reg_method
                outputFloder = './Outputs' +'/'+ modelname + '/'+ reg_method+'/'+'Revised_wcontrol{}'.format(Wcontrolmode)+'_' + dataset +'_' +preprocess+ '_epi{}_'.format(epinum) + outputdate
                if not os.path.exists(outputFloder):
                    os.makedirs(outputFloder)
                for Datanum in Datanums:
                    for WeightPred in WeightPreds:
                        for weightRec in WeightReconstructlist:
                                if reW > 0.1:
                                    model_file_name = "Layer1_Datanum{}_Dic64_WeightPred{:.2e}_{}_WeightRec{:.2e}_{}_{:.2e}".\
                                    format(Datanum, WeightPred,reg_method1, weightRec, wupdate, wupdate_param)
                                else:
                                    model_file_name = "Layer1_Datanum{}_Dic64_WeightPred{:.2e}_{}_WeightRec{:.2e}_delta_{:.2e}_{}_{:.2e}".\
                                    format(Datanum, WeightPred,reg_method1, weightRec, delta, wupdate, wupdate_param)
                                folder = "{}/{}".format(Model_folder, model_file_name)
                                print(folder)
                                if os.path.exists(folder):
                                    print(folder)
                                    tmp=os.listdir(folder)
                                    complete_run=len(tmp)
                                    for betaFis_index in betaFis_indexs:
                                        for stepSize in stepSizes:
                                            for time in range(mean_times):
                                                index = time % complete_run + 1

                                                outputFile = '{}/{}_betaFiIndex{}_lr={:.2e}_run={}'.format(outputFloder, model_file_name, betaFis_index, stepSize, time+1)
                                                if singlemultimode==1:
                                                    # single run
                                                    if os.path.exists(outputFile):
                                                        os.remove(outputFile)
                                                    multiThreadFun(index,epinum,model_file_name,stepSize,outputFile,weightRec*betaFis_index,reg_method,Model_folder,adapstep,revise,dataset,preprocess,printtrig,Wcontrolmode)
                                                elif singlemultimode==2:
                                                    # multi run remove existed
                                                    if os.path.exists(outputFile):
                                                        os.remove(outputFile)
                                                    t = Process(target=multiThreadFun, args=(index,epinum,model_file_name,stepSize,outputFile,betaFi*betaFis_index,reg_method,Model_folder,adapstep,revise,dataset,preprocess,printtrig,Wcontrolmode))
                                                    t.start()
                                                    ts.append(t)
                                                elif singlemultimode==3:
                                                    #  multi run keep existed
                                                    if not os.path.exists(outputFile):
                                                        t = Process(target=multiThreadFun, args=(index,epinum,model_file_name,stepSize,outputFile,betaFi*betaFis_index,reg_method,Model_folder,adapstep,revise,dataset,preprocess,printtrig,Wcontrolmode))
                                                        t.start()
                                                        ts.append(t)
                                            for t in ts:
                                                t.join()
