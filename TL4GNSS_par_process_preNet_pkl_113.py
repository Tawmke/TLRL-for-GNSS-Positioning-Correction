import re
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd

def plotMean(inputFile):
    pattern = re.compile('.*Cumulative.Reward.*')
    reader = open(inputFile, 'r')
    rewards = []
    while True:
        line = reader.readline()
        if (len(line) == 0):
            break
        m = pattern.match(line)
        if (m):
            ngs = [w for w in m.group().split(' ')]
            rewards.append(float(ngs[-1]))
    return rewards

def exp_average(data, expFactor=0.1):
    expRawRewards = np.zeros(data.shape)
    for i in range(data.shape[0]):
        expRaw = 0.0
        J = 0.0
        for j in range(data.shape[1]):
            J *= (1.0-expFactor)
            J += (expFactor)
            rate = expFactor/J
            expRaw = (1-rate)*expRaw
            expRaw += rate*data[i][j]
            expRawRewards[i, j] = expRaw
    return expRawRewards

if __name__ == '__main__':
    LogFolder = './newOutput'
    datasets = ['TDurbancanyon']  # 'GSDC2022urban' 'GSDC2022highway' TDurbancanyon
    reg_methods = ['L0']  # 'L0','L0relu'
    Datanums = [9000]
    betaFis_indexs = [1.0]
    reW = 1
    more = 4

    wupdate = 'yzt_re_betaW'  # yzt_re_betaW
    wupdate_param = 1.00e-06
    modelname = 'preDTL_reinit'  # preDTL_reW preDTL_orig
    preprocesses = ['DQN_norm']  # 'DCT_IV','randn'
    revise = 1
    Wcontrolmodes = [4]
    needsmooth = 1
    needlimit = 1
    noise = 0

    if more > 1:
        #method_list = ['L1/2','L1','L2/3','dropout','NN','SRNN','DSRL']
        model_file_name_list = []
        if datasets[0] == 'GSDC2022urban':
            method_list = ['A3C', 'Multi-LSTMPPO', 'DLRL-$\mathregular{log}$','TLRL (DDPG)', 'TLRL (PPO)', 'TLRL'] #'MVDRL-SC',
            model_file_name_DLRLlog = "./Outputs/preDDL/log/GSDC2022urban/Revised_wcontrol4_GSDC2022urban_DQN_norm_epi80_0919gsdc/" \
                                 "WP1.00e-02_WR1.00e-02_log_betaFi1.00e-03_delta1.00e-03_lr=1.00e-01"
            model_file_name_TLRLPPO = "./Outputs/preDTL_reinit/L0/GSDC2022urban/Revised_wcontrol4_GSDC2022urban_DQN_norm_epi85234_251208traj1_6_kf_ppo/" \
                                 "WP1.00e-04_WR1.00e-01_L0_betaFi1.00e-01_lr=1.00e-02"
            model_file_name_TLRLDDPG = "./Outputs/preDTL_reinit/L0/GSDC2022urban/Revised_wcontrol4_GSDC2022urban_DQN_norm_epi85234_251208traj1_6_kf_ddpg/" \
                                 "WP1.00e-04_WR1.00e-01_L0_betaFi5.00e-01_lr=1.00e-02"
            model_file_name_MVDRLSR = "./records_values/0725_MVDRL_SC/continuous_lstmATF1_SCL1mh_c_urban1_n1_t1_kf_igst_lam=0.5_lr=0.005/MVDRLSR"
            model_file_name_A3C = "./records_values/robustRL_1111a2cpos/source=urban/discrete_A2C_1_urban_ecef_kf_igst_lr=0.001_pos=50/MVDRLSR"
            model_file_name_LSTMPPO = "./records_values/0725_PPO_Urban/continuous_lstmATF1_SCL1mh_c_urban1_n1_t1_kf_igst_lam=0_lr=0.001/MVDRLSR"
            model_file_name_list.append(model_file_name_A3C)
            model_file_name_list.append(model_file_name_LSTMPPO)
            # model_file_name_list.append(model_file_name_MVDRLSR)
            model_file_name_list.append(model_file_name_DLRLlog)
            model_file_name_list.append(model_file_name_TLRLDDPG)
            model_file_name_list.append(model_file_name_TLRLPPO)

        if datasets[0] == 'TDurbancanyon':
            method_list = ['A3C','Multi-LSTMPPO','DLRL-$\mathregular{log}$','TLRL (DDPG)', 'TLRL (PPO)','TLRL']
            model_file_name_TLRLPPO = "./Outputs/preDTL_reinit/L0/TDurbancanyon/Revised_wcontrol4_TDurbancanyon_DQN_norm_epi111693_251209_rtk_sc2/" \
                                 "WP8.00e-01_WR1.00e-01_L0_betaFi1.00e-03_lr=1.00e-01"
            model_file_name_TLRLDDPG = "./Outputs/preDTL_reinit/L0/TDurbancanyon/Revised_wcontrol4_TDurbancanyon_DQN_norm_epi111693_251209_rtk_sc2_ddpg/" \
                                 "WP8.00e-01_WR1.00e-01_L0_betaFi1.00e-03_lr=5.00e-03"
            model_file_name_LSTMPPO = "./records_values/0808_PPO_TDurbancanyon/continuous_lstmATF1_SCL1mh_c_canyon_n1_t1_rtk_lam=0_lr=0.001/MVDRLSR"
            model_file_name_A3C = "./records_values/robustRL_0814TDA2C/source=canyon/discrete_A2C_1_canyon_ecef_rtk_lr=0.01_pos=40/MVDRLSR"
            model_file_name_DLRLLog = "./Outputs/preDDL/log/TDurbancanyon/Revised_wcontrol4_TDurbancanyon_DQN_norm_epi100_TDurbancanyon0814/" \
                                      "WP1.00e-05_WR1.00e-01_log_betaFi1.00e-02_delta1.00e-03_lr=1.00e-01"
            model_file_name_list.append(model_file_name_A3C)
            model_file_name_list.append(model_file_name_LSTMPPO)
            model_file_name_list.append(model_file_name_DLRLLog)
            model_file_name_list.append(model_file_name_TLRLDDPG)
            model_file_name_list.append(model_file_name_TLRLPPO)


    for dataset in datasets:
        if dataset == 'GSDC2022urban':
            epinum = 50
            run_date = '240721traj1_6_kf_acspace31' # 240715
            plot_date = '240721traj1_6_kf_acspace31' #
            mean_times = 15
            more = 6 # 所有方法
            env = 'GSDC2022urban'
            envsetting = 'acspace=31,sc=0.2'
            rewardlimit = -1000
            """
            当前最优：WP1.00e-04_WR1.00e-01_L0_betaFi5.00e-01_mean_times=15_lr=0.03_reward=1547.98_diserr=3.10.pdf
            """
            stepSizes = [3e-2] # [1e-4,5e-4,1e-3,5e-3,1e-2,8e-3,1e-2,3e-2,8e-2,1e-1]
            WeightPreds = [1e-4] # [8e-1,5e-1,1e-1,1e-2,1e-3,1e-4,5e-5,1e-5,5e-6]
            WeightReconstructlist = [1e-1] #[1e-5,1e-4,1e-3,1e-2]
            # WeightPreds = [1e-7,1e-6,1e-5]
            betaFis =  [5e-1] # [1.3,1,8e-1,5e-1,3e-1,1e-1,5e-2,1e-2]
        elif dataset == 'GSDC2022highway':
            epinum = 100
            run_date = '240722traj0_7_kf_acspace31'  # 240715
            plot_date = '240722traj0_7_kf_acspace31'  #
            mean_times = 5
            more = 1
            env = 'GSDC2022urban'
            envsetting = 'acspace=31,sc=0.2'
            rewardlimit = -1000
            stepSizes = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 8e-3, 1e-2, 3e-2]
            WeightPreds = [1,8e-1, 5e-1, 1e-1, 1e-2, 1e-3, 1e-4, 5e-5, 1e-5]  # [1e-4,1e-3,1e-2,1e-1]
            WeightReconstructlist = [1e-1]  # [1e-5,1e-4,1e-3,1e-2]
            # WeightPreds = [1e-7,1e-6,1e-5]
            betaFis = [8e-1, 5e-1, 3e-1, 1e-1, 5e-2, 1e-2]
            # [1e-4,3.16e-4,1e-3,3.16e-3,1e-2,3.16e-2,1e-1]
        elif dataset == 'TDurbancanyon':
            """
            当前最优：WP8.00e-01_WR1.00e-01_L0_betaFi1.00e-03_mean_times=10_lr=0.1_reward=3931.18_diserr=4.75.pdf
            """
            epinum = 100
            run_date = '240812_rtk_acspace41'  # 240805_rtk_acspace31
            plot_date = '240812_rtk_acspace41'  # 240805_rtk_acspace31
            mean_times = 10
            more = 6
            # env = 'GSDC2022urban'
            envsetting = 'acspace=41,sc=0.2'
            rewardlimit = -1000
            stepSizes = [1e-1] # [5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1]
            WeightPreds = [8e-1]  # [1,8e-1,5e-1,1e-1,1e-2,1e-3,1e-4,5e-5,1e-5,5e-6]
            WeightReconstructlist = [1e-1]  # [1e-5,1e-4,1e-3,1e-2]
            betaFis = [1e-3] # [5e-1,3e-1,1e-1,5e-2,1e-2,5e-3,1e-3,5e-4]

        parmsk1k2 = np.zeros((len(WeightPreds), len(WeightReconstructlist)))
        for preprocess in preprocesses:
            for Wcontrolmode in Wcontrolmodes:
                lr_dic = {}
                lr_dic_std = {}
                Datanum_dic = {}
                WeightPred_dic = {}
                WeightPred_dic_std = {}
                weightRecs_dic = {}
                weightRecs_dic_std = {}
                regMethod_dic = {}
                betaFisIndex_dic = {}
                betaFisIndex_std_dic = {}
                stepSize_dic = {}
                min_diserr = 100
                max_reward = 0
                min_avg_diserr = 100

                for reg_method in reg_methods:
                    if reg_method == 'L0relu':
                        reg_method1 = 'L0'
                    else:
                        reg_method1 = reg_method
                    outputFloder = './Outputs/' + modelname + '/' + reg_method +  f'/{dataset}' '/Revised_wcontrol{}'.format(
                        Wcontrolmode) + '_' + dataset + '_' + preprocess + '_epi{}_'.format(epinum) + run_date
                    plot_folder = './plots/' + reg_method + '/Revised_wcontrol{}'.format(Wcontrolmode) + '_' \
                                  + dataset + '_' + preprocess + '_control_' + modelname + '_epi{}_loop{}_'. \
                                      format(epinum, mean_times) + plot_date

                    detail_folder = plot_folder + '/Details'
                    if not os.path.exists(plot_folder):
                        os.makedirs(plot_folder)
                        os.makedirs(detail_folder)

                    for Datanum in Datanums:
                        for WeightPred in WeightPreds:
                            for weightRec in WeightReconstructlist:
                                for betaFi in betaFis:
                                    if dataset == 'CP': # ADD 0727
                                        maxmeanreward = 0
                                    else:
                                        maxmeanreward = -10000
                                    for stepSize in stepSizes:
                                        rewards_DSRL = []
                                        diserr = []
                                        if more > 1:
                                            rewards_all = []
                                            for _ in range(more-1):
                                                rewards_all.append([])

                                        for i in range(mean_times):
                                            if reW > 0.1:
                                                model_file_name = "WP{:.2e}_WR{:.2e}_{}_betaFi{:.2e}".format\
                                                    (WeightPred, weightRec, reg_method1, betaFi)
                                            else:
                                                model_file_name = "Layer1_Datanum{}_Dic64_WeightPred{:.2e}_{}_betaFi{:.2e}". \
                                                    format(Datanum, WeightPred, reg_method1, betaFi)

                                            outputFile = '{}/{}_lr={:.2e}_run={}_rewards.pkl'. \
                                                format(outputFloder, model_file_name, stepSize, i)

                                            # 提取error信息
                                            trajfolder = f'{outputFloder}/{model_file_name}_lr={stepSize:.2e}_run={i}{envsetting}'
                                            trajerrfile = f'{trajfolder}/train_rl_distances.csv'

                                            if more > 1:
                                                outputFilelist = []
                                                for modelname_base in model_file_name_list:
                                                    outputFilelist.append('{}_run={}_rewards.pkl'.format(modelname_base,i + 1))

                                            checkexist = False # 检查是否该参数存在
                                            if os.path.exists(outputFile):
                                                check = True
                                                with open(outputFile, "rb") as file:
                                                    reward = pickle.load(file) #读取字符串并重构回去，效率更高？
                                                # traj err
                                                pderr = pd.read_csv(trajerrfile)
                                                tranerr = pderr.loc[1,'Avg']
                                                if tranerr < min_diserr:
                                                    min_diserr = tranerr
                                                    min_diserr_file = trajfolder
                                                # reward=np.array(reward)
                                                if not reward:
                                                    continue
                                                file.close()
                                                len_reward = len(reward)
                                                if len_reward < epinum:
                                                    n = epinum - len_reward
                                                    reward_append = reward[-1]
                                                    for e1 in range(n):
                                                        reward.append(reward_append)
                                                elif len_reward > epinum:
                                                    reward = reward[0:epinum]

                                                if needlimit == 1:
                                                    if reward[-1] > rewardlimit:
                                                        rewards_DSRL.append(reward)
                                                else:
                                                    rewards_DSRL.append(reward)

                                                diserr.append(tranerr)

                                            if more > 1:
                                                # for 12PO
                                                for idx in range(more-1):
                                                    outputFile = outputFilelist[idx]
                                                    if os.path.exists(outputFile):
                                                        with open(outputFile, "rb") as file:
                                                            reward = pickle.load(file)

                                                        file.close()
                                                        len_reward = len(reward)
                                                        if len_reward < epinum:
                                                            n = epinum - len_reward
                                                            reward_append = reward[-1]
                                                            for e1 in range(n):
                                                                reward.append(reward_append)
                                                        elif len_reward > epinum:
                                                            reward = reward[0:epinum]

                                                        if needlimit == 1:
                                                            if idx < more-4:
                                                                if reward[-1] > rewardlimit:
                                                                    rewards_all[idx].append(reward)
                                                            else: #神经网络
                                                                if reward[-1] > rewardlimit:#- 1100:
                                                                    # print(reward)
                                                                    rewards_all[idx].append(reward)
                                                        else:
                                                            rewards_all[idx].append(reward)

                                        if not rewards_DSRL:
                                            if dataset == 'MC':
                                                rewards_DSRL.append(-500*np.ones((250)))
                                            else:
                                                continue
                                        if more > 1:
                                            rewards_mean_list=[]
                                            rewards_std_list=[]
                                            meanRewards_list=[]
                                            stdRewards_list = []
                                            for idx in range(more-1):
                                                rewards = rewards_all[idx]
                                                rewards_list = rewards
                                                rewards = np.array(rewards)
                                                rewards_mean_list.append(rewards.mean(axis=0))
                                                rewards_std_list.append(np.std(rewards, axis=0))  # 计算矩阵每一列的标准差
                                                meanRewards_list.append(rewards.mean(axis=0)[-1])
                                                stdRewards_list.append(np.std(rewards, axis=0)[-1])
                                                print(method_list[idx]+':rewards={}+{}'.format(rewards.mean(axis=0)[-1],np.std(rewards, axis=0)[-1]))
                                                if needsmooth == 1:
                                                    a0 = rewards[:, 0].mean(axis=0)
                                                    a1 = rewards[:, -1].mean(axis=0)
                                                    rewards = exp_average(rewards)
                                                    rewards_mean = rewards.mean(axis=0)
                                                    # b0 = rewards_mean[0]
                                                    # b1 = rewards_mean[-1]
                                                    # rewards_mean = [(x - b0) * (a1 - a0) / (b1 - b0) + a0 for x in rewards_mean]
                                                    rewards_mean_list[idx] = rewards_mean
                                                    rewards_std_list[idx] = np.std(rewards, axis=0)
                                                print(method_list[idx]+':smooth_rewards={}+{}'.format(rewards.mean(axis=0)[-1],np.std(rewards, axis=0)[-1]))

                                        rewards_DSRL = np.array(rewards_DSRL)
                                        diserr = np.array(diserr)
                                        diserr_mean = diserr.mean(axis=0)
                                        diserr_std = diserr.std(axis=0)
                                        #print(rewards_DSRL)
                                        rewards_mean = rewards_DSRL.mean(axis=0)
                                        rewards_std = np.std(rewards_DSRL, axis=0) #计算矩阵每一列的标准差
                                        meanRewards = rewards_mean[-1]
                                        stdRewards = rewards_std[-1]
                                        # print(rewards_DSRL[:, -1])
                                        # print(rewards_mean)
                                        if needsmooth == 1:
                                            a0 = rewards_DSRL[:,0].mean(axis=0)
                                            a1 = rewards_DSRL[:,-1].mean(axis=0)
                                            rewards_DSRL = exp_average(rewards_DSRL)
                                            rewards_mean = rewards_DSRL.mean(axis=0)
                                            rewards_std = np.std(rewards_DSRL, axis=0)
                                        # print(rewards_DSRL[:, -1])
                                        meanRewards = rewards_mean[-1]
                                        stdRewards = rewards_std[-1]

                                        if stepSize in lr_dic:
                                            lr_dic[stepSize].append(diserr_mean)
                                            lr_dic_std[stepSize].append(diserr_std)
                                        else:
                                            lr_dic[stepSize] = [diserr_mean]
                                            lr_dic_std[stepSize] = [diserr_std]
                                        if Datanum in Datanum_dic:
                                            Datanum_dic[Datanum].append(diserr_mean)
                                        else:
                                            Datanum_dic[Datanum] = [diserr_mean]
                                        if WeightPred in WeightPred_dic:
                                            WeightPred_dic[WeightPred].append(diserr_mean)
                                            WeightPred_dic_std[WeightPred].append(diserr_std)
                                        else:
                                            WeightPred_dic[WeightPred] = [diserr_mean]
                                            WeightPred_dic_std[WeightPred] = [diserr_std]
                                        if weightRec in weightRecs_dic:
                                            weightRecs_dic[weightRec].append(diserr_mean)
                                            weightRecs_dic_std[weightRec].append(diserr_std)
                                        else:
                                            weightRecs_dic[weightRec] = [diserr_mean]
                                            weightRecs_dic_std[weightRec] = [diserr_std]
                                        if betaFi in betaFisIndex_dic:
                                            betaFisIndex_dic[betaFi].append(diserr_mean)
                                            betaFisIndex_std_dic[betaFi].append(diserr_std)
                                        else:
                                            betaFisIndex_dic[betaFi] = [diserr_mean]
                                            betaFisIndex_std_dic[betaFi] = [diserr_std]
                                        if reg_method in regMethod_dic:
                                            regMethod_dic[reg_method].append(diserr_mean)
                                        else:
                                            regMethod_dic[reg_method] = [diserr_mean]
                                        if stepSize in stepSize_dic:
                                            stepSize_dic[stepSize].append(diserr_mean)
                                        else:
                                            stepSize_dic[stepSize] = [diserr_mean]

                                        if check:
                                            if meanRewards > maxmeanreward:
                                                parmsk1k2[WeightPreds.index(WeightPred), WeightReconstructlist.index(weightRec)] = meanRewards  # add 0727
                                                maxmeanreward = meanRewards

                                        print("{}_mean:{:.2f}+{:.2f}".format(model_file_name, meanRewards,stdRewards))
                                        print("{}_mean_smooth:{:.2f}+{:.2f}".format(model_file_name, rewards_mean[-1], rewards_std[-1]))
                                        plot_file = '{}/{}_mean_times={}_lr={}_reward={:.2f}_diserr={:.2f}.pdf'.format(detail_folder,
                                                                                                          model_file_name,
                                                                                                          mean_times,
                                                                                                          stepSize,
                                                                                                          rewards_mean[-1],
                                                                                                          diserr_mean)

                                        if rewards_mean[-1] > max_reward:
                                            max_reward = rewards_mean[-1]
                                            max_reward_file = plot_file
                                        if diserr_mean < min_avg_diserr:
                                            min_avg_diserr = diserr_mean
                                            min_avg_diserr_file = plot_file

                                        plt.figure(figsize=(7, 6))
                                        width = 3
                                        rewards_mean_dsrl = rewards_mean
                                        rewards_std_dsrl = rewards_std
                                        if more < 2:
                                            plt.plot([x for x in range(1, epinum + 1)], rewards_mean)
                                            plt.fill_between([x for x in range(1, epinum + 1)], rewards_mean + rewards_std,
                                                             rewards_mean - rewards_std, color='b', alpha=.1) #填充方差的区域
                                            plt.xlabel("episodes",fontsize=18)
                                            plt.ylabel("rewards",fontsize=18)
                                            plt.title(datasets)
                                            plt.savefig(plot_file)
                                            plt.clf() #清除东西
                                        else:
                                            rewards_mean_dsrl = rewards_mean
                                            rewards_std_dsrl = rewards_std

                                            for idx,rewards_mean in enumerate(rewards_mean_list):
                                                if idx == more-6:
                                                    plt.plot([x for x in range(1, epinum + 1)], rewards_mean,'darkcyan',linestyle='-.',linewidth=width)
                                                elif idx == more-5:
                                                    plt.plot([x for x in range(1, epinum + 1)], rewards_mean,'purple',linestyle=(0, (5, 5)),linewidth=width)
                                                elif idx == more-4:
                                                    plt.plot([x for x in range(1, epinum + 1)], rewards_mean,'firebrick',linestyle=':',linewidth=width)
                                                elif idx == more-3:
                                                    plt.plot([x for x in range(1, epinum + 1)], rewards_mean,'olive',linestyle=(0, (5, 1)),linewidth=width)
                                                elif idx == more-2:
                                                    plt.plot([x for x in range(1, epinum + 1)], rewards_mean,'gold',linestyle='--',linewidth=width)
                                                else:
                                                    plt.plot([x for x in range(1, epinum + 1)], rewards_mean,',:',linewidth=width+1)
                                            plt.plot([x for x in range(1, epinum + 1)], rewards_mean_dsrl, 'mediumblue',linewidth=width)
                                            if dataset == 'GSDC2022urban':
                                                plt.legend(method_list,fontsize=18)

                                            plt.fill_between([x for x in range(1, epinum + 1)],rewards_mean_dsrl + rewards_std_dsrl,
                                                                     rewards_mean_dsrl - rewards_std_dsrl,color='mediumblue',alpha=.1)  # 填充方差的区域

                                            for idx in range(more-1):
                                                rewards_mean = rewards_mean_list[idx]
                                                rewards_std = rewards_std_list[idx]
                                                if idx == more-6:
                                                    plt.fill_between([x for x in range(1, epinum + 1)],rewards_mean + rewards_std,
                                                                     rewards_mean - rewards_std, color = 'darkcyan', alpha=.2)  # 填充方差的区域
                                                elif idx == more-5:
                                                    plt.fill_between([x for x in range(1, epinum + 1)],rewards_mean + rewards_std,
                                                                     rewards_mean - rewards_std, color = 'purple', alpha=.2)  # 填充方差的区域
                                                elif idx == more-4:
                                                    plt.fill_between([x for x in range(1, epinum + 1)],rewards_mean + rewards_std,
                                                                     rewards_mean - rewards_std, color = 'firebrick', alpha=.2)  # 填充方差的区域
                                                elif idx == more-3:
                                                    plt.fill_between([x for x in range(1, epinum + 1)],rewards_mean + rewards_std,
                                                                     rewards_mean - rewards_std, color = 'olive', alpha=.2)  # 填充方差的区域
                                                elif idx == more-2:
                                                    plt.fill_between([x for x in range(1, epinum + 1)],rewards_mean + rewards_std,
                                                                     rewards_mean - rewards_std, color = 'gold', alpha=.2)  # 填充方差的区域
                                                else:
                                                    plt.fill_between([x for x in range(1, epinum + 1)], rewards_mean + rewards_std,
                                                                 rewards_mean - rewards_std, alpha=.1)  # 填充方差的区域

                                            plt.xlabel("Episodes",fontsize=19)
                                            plt.ylabel("Rewards",fontsize=19)
                                            plt.tick_params(labelsize=18)

                                            if dataset == 'PW':
                                                plt.ylim(-500,100)
                                            elif dataset == 'CP':
                                                plt.ylim(0,300)
                                            #plt.title(env,fontsize=15)
                                            #plt.show()
                                            plt.savefig(plot_file, bbox_inches='tight')
                                            plt.clf()  # 清除东西

                                        results_txt = open(detail_folder + '/{}_mean_times={}_reward={:.2f}+{:.2f}.txt'.
                                                           format(model_file_name, mean_times, rewards_mean_dsrl[-1], rewards_std_dsrl[-1]),
                                                           'w')
                                        results_txt.write('Rewards: \n {}'.format(rewards_DSRL[:,-1]))
                                        results_txt.write('distance error: \n {}'.format(diserr))
                                        results_txt.close()

                    plt.figure(figsize=(20,20))
                    plt.subplot(131)
                    x = []
                    y = []
                    std = []
                    for key in sorted(WeightPred_dic): #对参数排序
                        x.append(key)
                        y.append(np.min(WeightPred_dic[key]))
                        idx = np.argmin(WeightPred_dic[key])
                        std.append(WeightPred_dic_std[key][idx])
                    plot_file = '{}/a_WeightPred_par_mean_times={}.pdf'.format(plot_folder, mean_times)
                    #plt.plot(x, y)
                    plt.errorbar(x,y,yerr=std,capsize=2,capthick=2,linewidth=2)
                    # plt.axhline(baseline, color='grey', linestyle='--')
                    plt.legend(['TLRL'])
                    plt.xscale('log')
                    plt.xlabel("Weight predict parameter")
                    plt.ylabel("distance error")
                    # plt.xlabel("${k_2}$",fontsize=55)
                    plt.tick_params()
                    # plt.savefig(plot_file,bbox_inches='tight')
                    # plt.show()
                    #plt.ylabel("Rewards",fontsize=17)
                    # plt.savefig(plot_file)
                    # plt.clf()

                    # plt.figure()
                    # x = []
                    # y = []
                    # std = []
                    # for key in sorted(weightRecs_dic):
                    #     x.append(key)
                    #     y.append(np.max(weightRecs_dic[key]))
                    #     idx = np.argmax(weightRecs_dic[key])
                    #     std.append(weightRecs_dic_std[key][idx])
                    # plot_file = '{}/betaFis_par_mean_times={}.pdf'.format(plot_folder, mean_times)
                    # plot_file = '{}/k1k2in_{}.pdf'.format(plot_folder, dataset)
                    # #plt.plot(x, y)
                    # plt.errorbar(x, y, yerr=std, capsize=3, capthick=2, linewidth=4)
                    # # plt.axhline(baseline, color='grey', linestyle='--')
                    # plt.legend(['Baseline','DLRL-ASR'],fontsize=25)
                    # plt.xscale('log')
                    # plt.xlabel("Weight reconstruct parameter",fontsize=55)
                    # plt.ylabel("Rewards",fontsize=55)
                    # plt.tick_params(labelsize=20)
                    # # plt.subplots_adjust(left=0.08, bottom=None, right=None, top=None, \
                    # #                     wspace=0.2, hspace=0.05)
                    # plt.savefig(plot_file)
                    # plt.clf()
                    # plt.show()

                    plt.subplot(132)
                    x = []
                    y = []
                    std = []
                    for key in sorted(lr_dic):
                        x.append(key)
                        y.append(np.min(lr_dic[key]))
                        idx = np.argmin(lr_dic[key])
                        std.append(lr_dic_std[key][idx])
                    plot_file = '{}/lr_{}.pdf'.format(plot_folder, dataset)
                    #plt.plot(x, y)
                    plt.errorbar(x, y, yerr=std, capsize=3, capthick=2, linewidth=2)
                    # plt.axhline(baseline, color='grey', linestyle='--',linewidth=6)
                    plt.legend(['TLRL'])
                    plt.xscale('log')
                    plt.xlabel("StepSizes")
                    # plt.ylabel("distance error")
                    plt.tick_params()
                    # plt.savefig(plot_file, bbox_inches='tight')
                    # plt.clf()
                    # plt.show()

                    # plt.figure(figsize=(17.5, 16.5))
                    plt.subplot(133)
                    x = []
                    y = []
                    std = []
                    for key in sorted(betaFisIndex_dic):
                        x.append(key)
                        y.append(np.min(betaFisIndex_dic[key]))
                        idx = np.argmin(betaFisIndex_dic[key])
                        std.append(betaFisIndex_std_dic[key][idx])
                    plot_file = '{}/betafi_{}.pdf'.format(plot_folder, dataset)
                    #plt.plot(x, y)
                    plt.errorbar(x, y, yerr=std, capsize=3, capthick=2, linewidth=2)
                    # plt.axhline(baseline, color='grey', linestyle='--',linewidth=6)
                    plt.legend(['TLRL'])
                    plt.xscale('log')
                    plt.xlabel("betafi")
                    # plt.ylabel("distance error")
                    plt.tick_params()
                    plot_file = '{}/para_selection_{}.pdf'.format(plot_folder, dataset)
                    plt.savefig(plot_file, bbox_inches='tight')
                    plt.clf()
                    plt.show()

print(f'min_diserr={min_diserr},min_diserr file:{min_diserr_file}')
print(f'max_meandis_file:{min_avg_diserr_file}')
print(f'max_reward file:{max_reward_file}')