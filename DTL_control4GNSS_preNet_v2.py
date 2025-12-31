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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from multiprocessing import Process
from funcs.control_model_DTL_preNet import *
from funcs.control_model_DTL_preNet_TDdataset import *
# from funcs.control_model_DDL_preNet_SGD import *

def CheckSimilarity(B):
    print("Word Similarity:")
    temp = np.array([word / np.linalg.norm(word) for word in B])
    temp = np.dot(temp, temp.T)
    for i in range(temp.shape[0]):
        for j in range(i, temp.shape[1]):
            print("Similarity between {} and {}: {}".format(i, j, np.abs(temp[i][j])))

def multiThreadFun(k, epinum, model_file_name, stepSize, outputFile, betaFiControl, reg_method, Model_folder, adapstep, revise, dataset, preprocess, printtrig, Wcontrolmode):
    decay = 1.0
    if Wcontrolmode < 3.1:
        actionLearner = ActionLearner_SGD(actionRandomness=0.1, decay=decay, stepSize=stepSize, loadBasic=model_file_name,
                              index=k, outputFile=outputFile, betaFi_control = betaFiControl, reg_method=reg_method,
                              Model_folder=Model_folder, dataset=dataset, preprocess=preprocess, printtrig=printtrig, Wcontrolmode=Wcontrolmode)
    else:
        if 'GSDC2022' in dataset:
            actionLearner = ActionLearner_GSDC2022(actionRandomness=0.1, decay=decay, stepSize=stepSize, loadBasic=model_file_name,
                                  index=k, outputFile=outputFile, betaFi_control = betaFiControl, reg_method=reg_method,
                                  Model_folder=Model_folder, dataset=dataset, preprocess=preprocess, printtrig=printtrig, Wcontrolmode=Wcontrolmode)
        elif 'TD' in dataset:
            actionLearner = ActionLearner_TD(actionRandomness=0.1, decay=decay, stepSize=stepSize, loadBasic=model_file_name,
                                  index=k, outputFile=outputFile, betaFi_control = betaFiControl, reg_method=reg_method,
                                  Model_folder=Model_folder, dataset=dataset, preprocess=preprocess, printtrig=printtrig, Wcontrolmode=Wcontrolmode)
    # epinum=100
    episode_rewards, episode_loss = actionLearner.learn(epinum)

if __name__ == '__main__':
    datasets= ['TDurbancanyon']# 'GSDC2022urban' 'GSDC2022highway' 'TDurbancanyon'
    reg_method1 = 'wo_sparse'# 'L0', 'L1' 'wo_sparse'
    reg_method = reg_method1
    # stepSizes = [1e-2]#,1e-5,1e-4,1e-3,1e-2,1e-1  0.33*1e-1,0.66*1e-1,1e-1,0.33
    betaFis_indexs = [1.0]
    reW = 1
    # epinum = 100
    delta = 1e-3

    wupdate = 'yzt_re' #yzt_re_betaW
    wupdate_param = 1.00e-06
    # mean_times = 5  # Number of runs per parameter
    modelname='preDTL_reinit'# preDTL_reW preDTL_orig
    preprocess= 'DQN_norm' # 'DCT_IV','randn' DQN
    revise = 1
    Wcontrolmode = 4  # 修改后的4
    adapstep = 0
    complete_run = 5
    printtrig=1
    singlemultimode = 3
    ts = []

    for dataset in datasets:
        if dataset=='GSDC2022urban':
            Datanums = [9000]
            epinum = 50
            # output_date = '0518_3'  # Final1
            preindex = 0 # pretrained model index
            fix_pre = False # fix pretrained para
            beginruntime = 0 # continume repeat exp
            """
            通常地址：240721traj1_6_kf_acspace31: 最佳 WeightPreds = [1e-4]  stepSizes = [3e-2] betaFis = [5e-1]
            噪声灵敏度：240721traj1_6_kf_acspace31_noise
            消融实验：240721traj1_6_kf_acspace31_ablation 
            """
            if reg_method == 'L0':
                run_date = '241212woBnorm' # 通常地址：240715，不进行字典标准化：241212woBnorm
                output_date = '240721traj1_6_kf_acspace31_woBnorm_noise' # 240717traj0_6_acspace41_sc2e 240721traj1_6_kf_acspace31
                mean_times = 5
                stepSizes = [3e-2] # [1e-4,5e-4,1e-3,5e-3,1e-2,8e-3,1e-2,3e-2,8e-2]
                WeightRecs = [1e-1]  # [1e-1]
                WeightPreds = [1e-3] # [8e-1,5e-1,1e-1,1e-2,1e-3,1e-4,5e-5,1e-5,5e-6]
                betaFis = [5e-1] # [1.3,1,8e-1,5e-1,3e-1,1e-1,5e-2,1e-2] [8e-1,5e-1,3e-1,1e-1,5e-2]
            elif reg_method == 'L1' or reg_method == 'wo_sparse':
                run_date = '241212'
                output_date = '240721traj1_6_kf_acspace31_ablation'
                mean_times = 5
                stepSizes = [8e-2]  # [1e-4,5e-4,1e-3,5e-3,1e-2,8e-3,1e-2,3e-2,8e-2]
                WeightRecs = [1e-1]  # [1e-1]
                WeightPreds = [5e-5]  # [8e-1,5e-1,1e-1,1e-2,1e-3,1e-4,5e-5,1e-5,5e-6]
                betaFis = [5e-1]  # [1.3,1,8e-1,5e-1,3e-1,1e-1,5e-2,1e-2] [8e-1,5e-1,3e-1,1e-1,5e-2]
        elif dataset=='TDurbancanyon':
            Datanums = [14000]
            epinum = 100
            preindex = 0  # pretrained model index
            fix_pre = False  # fix pretrained para
            beginruntime = 0  # continume repeat exp
            if reg_method == 'L0':
                run_date='241212woBnorm' #0913
                # output_date = '0518_3'  # Final1
                output_date = '240812_rtk_acspace41_woBnorm' # 240805_rtk_acspace31 240812_rtk_acspace41
                mean_times = 5
                stepSizes = [1e-2,5e-2,1e-1] # [5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1]
                WeightRecs = [1e-1] # [1e-1]
                WeightPreds =  [1,8e-1,5e-1]   # [1,8e-1,5e-1,1e-1,1e-2,1e-3,1e-4,5e-5,1e-5,5e-6]
                betaFis = [1e-4] # [5e-1,3e-1,1e-1,5e-2,1e-2,5e-3,1e-3,5e-4] best： [1e-3] [1e-2,1e-3,1e-4]
            elif reg_method == 'L1'or reg_method == 'wo_sparse':
                run_date='241212' #0913
                # output_date = '0518_3'  # Final1
                output_date = '241212_rtk_acspace41_ablation' # 240805_rtk_acspace31 240812_rtk_acspace41
                mean_times = 5
                stepSizes = [1e-2] # [5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1]
                WeightRecs = [1e-1] # [1e-1]
                WeightPreds = [5e-1]  # [1,8e-1,5e-1,1e-1,1e-2,1e-3,1e-4,5e-5,1e-5,5e-6]
                betaFis = [1e-1] # [5e-1,3e-1,1e-1,5e-2,1e-2,5e-3,1e-3,5e-4] best： [1e-3]

        elif dataset=='GSDC2022highway':
            epinum = 100
            run_date='240715' #0913
            # output_date = '0518_3'  # Final1
            output_date = '240722traj0_7_kf_acspace31' # 240717traj0_6_acspace41_sc2e
            mean_times = 5
            stepSizes = [5e-4,1e-3] # [5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1]
            WeightRecs = [1e-1]  # [1e-1]
            WeightPreds = [8e-1,5e-1,1e-1,1e-2,1e-3,1e-4,5e-5,1e-5,5e-6]  # [8e-1,5e-1,1e-1,1e-2,1e-3,1e-4,5e-5,1e-5,5e-6]
            betaFis = [8e-1,5e-1,3e-1,1e-1,5e-2,1e-2] # [8e-1,5e-1,3e-1,1e-1,5e-2,1e-2]

        # 好参数：lr=1e-4,wr=1e-4,wp=5e-2,betafi=1e-3,run1

        if reW > 0.1:
            Model_folder = '/mnt/sdb/home/tangjh/DLRL4GNSSpos/matlab_Model/' + modelname + '/' + dataset + run_date
        else:
            Model_folder = '/mnt/sdb/home/tangjh/DLRL4GNSSpos/matlab_Model/' + modelname + '/' + dataset

        outputFloder = '/mnt/sdb/home/tangjh/DLRL4GNSSpos/Outputs' +'/'+ modelname + '/'+ reg_method+'/'+dataset+'/'+'Revised_wcontrol{}'\
            .format(Wcontrolmode)+'_' + dataset +'_' +preprocess+ '_epi{}_'.format(epinum) + output_date
        # outputFloder = './Outputs' +'/'+ modelname + '/'+ reg_method+'/'+'Revised_wcontrol{}'.format(Wcontrolmode)+'_' + dataset +'_' +preprocess+ '_epi{}_'.format(epinum) + output_date
        # print(outputFloder)
        if not os.path.exists(outputFloder):
            os.makedirs(outputFloder)
        for Datanum in Datanums:
            for WeightRec in WeightRecs:
                for WeightPred in WeightPreds:
                    for betaFi in betaFis:
                        if reW > 0.1:
                            model_file_name = f"Datanum{Datanum}_Dic128_WeightPred{WeightPred:.2e}_{reg_method1}_WeightRec{WeightRec:.2e}_beta{betaFi:.2e}_{wupdate}"
                        else:
                            model_file_name = "Layer1_Datanum{}_Dic128_WeightPred{:.2e}_{}_betaFi{:.2e}".\
                            format(Datanum, WeightPred, reg_method1, betaFi)

                        folder = "{}/{}".format(Model_folder, model_file_name)
                        # print(folder)
                        if os.path.exists(folder):
                            tmp = os.listdir(folder)
                            complete_run = len(tmp)
                            for betaFis_index in betaFis_indexs:
                                for stepSize in stepSizes:
                                    for time in range(mean_times):
                                        recordtime = 9
                                        # time = 0 # time for run
                                        file_name = "WP{:.2e}_WR{:.2e}_{}_betaFi{:.2e}". \
                                            format(WeightPred, WeightRec, reg_method1, betaFi)
                                        if fix_pre:
                                            index = preindex
                                        else:
                                            index = time % complete_run + 1

                                        time = time + beginruntime
                                        outputFile = '{}/{}_lr={:.2e}_run={}'.format(outputFloder, file_name, stepSize, time)
                                        # if not os.path.exists(outputFile):
                                        #     os.makedirs(outputFile)
                                        # print(outputFile)
                                        if singlemultimode == 1:
                                            # single run
                                            # if os.path.exists(outputFile):
                                            #     os.remove(outputFile)
                                            multiThreadFun(index, epinum, model_file_name, stepSize, outputFile, betaFi*betaFis_index, reg_method, Model_folder, adapstep, revise, dataset, preprocess,printtrig,Wcontrolmode)
                                        elif singlemultimode == 2:
                                            # multi run remove existed
                                            if os.path.exists(outputFile):
                                                os.remove(outputFile)
                                            t = Process(target=multiThreadFun, args=(index,epinum,model_file_name,stepSize,outputFile,betaFi*betaFis_index,reg_method,Model_folder,adapstep,revise,dataset,preprocess,printtrig,Wcontrolmode))
                                            t.start()
                                            ts.append(t)
                                        elif singlemultimode==3:
                                            #  multi run keep existed
                                            # if not os.path.exists(outputFile+ f'acspace={discrete_actionspace},sc={action_scale}'):
                                            t = Process(target=multiThreadFun, args=(index,epinum,model_file_name,stepSize,outputFile,betaFi*betaFis_index,reg_method,Model_folder,adapstep,revise,dataset,preprocess,printtrig,Wcontrolmode))
                                            t.start()
                                            ts.append(t)
                                for t in ts:
                                    t.join()

print('control finish!')
