import pandas as pd
#from env.GSDC_2022_LOSPOS import *
from env.GSDC_2022_POS import * # RL环境
# from env.dummy_cec_env_custom import *
import gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import A2C
from model.a2c import A2C
# from stable_baselines3 import PPO
# from sb3_contrib import RecurrentPPO
from model.ppo import PPO
from env.env_param import *
from funcs.utilis import *
#from funcs.PPO_SR import *
from model.model_ATF import *
from collections import deque
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# parameter settings
learning_rate_list = [1e-4] #[5e-4,1e-4 8e-5,3e-5]
triptype = 'urban'

if triptype == 'urban':
    #learning_rate_list = [3e-4]  # [5e-4,1e-4 8e-5,3e-5]
    postraj_num_list = [50] # [10,20,30,40]
    traj_type_target_train = [0,1] # 轨迹数据的比例
    traj_type_target_test = [0.1]
    tripIDlist = traj_else
    training_stepnum = 85234  # urban[1,6] 50epi 总步数 85234
    trajdata_range = [1,6]

trajdata_sort='sorted' # 'randint' 'sorted'
# baseline for RL: bl, wls, kf, kf_igst
baseline_mod='kf_igst' # baseline方法
# around 37.52321	-122.35447 scale 1e-5 lat 1.11m, lon 0.88m
# continuous action settings
max_action=100 #最大动作的范围
continuous_action_scale=20e-1 # 动作的尺度
continuous_actionspace=[-max_action,max_action]
# discrete action settings
discrete_actionspace = 31
action_scale = 2e-1 # scale in meters

# select network and environment
discrete_lists=['discrete','discrete_A2C','discrete_lstm','ppo_discrete']
continuous_lists=['continuous','continuous_lstm','continuous_lstm_custom','ppo','continuous_custom','continuous_lstmATF1']
custom_lists=['ppo_discrete','ppo']
networkmod='discrete_A2C'
# select environment type
envlists=['latlon','ned','ecef','los','lospos','losllAcat']
envmod='ecef'
# recording parameters
running_date = '0727a3cpos'
reward_setting='RMSEadv' # 'RMSE' ‘RMSEadv'
# data cata: KF: LatitudeDegrees; robust WLS: LatitudeDegrees_wls; standard WLS: LatitudeDegrees_bl
# parameters for customized ppo
# test settings
moretests=False #True False

if networkmod in discrete_lists:
    print(f'Action scale {action_scale:8.2e}, discrete action space {discrete_actionspace}')
elif networkmod in continuous_lists:
    print(f'Action scale {continuous_action_scale:8.2e}, contiuous action space from {continuous_actionspace[0]} to {continuous_actionspace[1]}')

for learning_rate in learning_rate_list:
    for posnum in postraj_num_list:
        tensorboard_log = f'{project_path}records_values/robustRL_{running_date}/source={triptype}/{networkmod}_{traj_type_target_train[1]}_{triptype}_{envmod}_{baseline_mod}_lr={learning_rate}_pos={posnum}'
        if envmod == 'ecef':
            if networkmod=='discrete_A2C':
                env = DummyVecEnv([lambda: GPSPosition_discrete_pos(trajdata_range, traj_type_target_train, triptype, action_scale, discrete_actionspace,
                                                                       reward_setting,trajdata_sort,baseline_mod, posnum)])

                model = A2C("MlpPolicy", env, verbose=2, tensorboard_log=tensorboard_log, learning_rate=learning_rate)
                model.learn(total_timesteps=training_stepnum, eval_log_path=tensorboard_log)

        #print and save training results
        logdirname=model.logger.dir+'/train_'
        # logdirname='./'
        print('Training finished.')

        #record model
        # params=model.get_parameters()
        if networkmod in discrete_lists:
            model.save(model.logger.dir+f"/{networkmod}_{reward_setting}_action{discrete_actionspace}_{action_scale:0.1e}_trainingnum{training_stepnum:0.1e}"
                                        f"_env_{baseline_mod}{envmod}range{trajdata_range[0]}_{trajdata_range[-1]}{trajdata_sort}_lr{learning_rate:0.1e}")
        elif networkmod in continuous_lists:
            model.save(model.logger.dir+f"/{networkmod}_{reward_setting}_action{continuous_actionspace[0]}_{continuous_actionspace[1]}"
                                        f"_{continuous_action_scale:0.1e}_trainingnum{training_stepnum:0.1e}"
                                        f"_env_{baseline_mod}{envmod}range{trajdata_range[0]}_{trajdata_range[-1]}{trajdata_sort}_lr{learning_rate:0.1e}")

        rl_distance = recording_results_ecef(data_truth_dic,trajdata_range,tripIDlist,logdirname,baseline_mod,traj_record=True)
        with open(model.logger.dir[:-5] + f'/A2C_run={model.logger.dir[-1]}_dis={rl_distance:.3}.txt', 'w') as file:
            file.write(f'dis={rl_distance}\n')
        file.close()

        # more tests
        if moretests:
            for testtype in moreteststypelist:
                print(f'more test for {testtype} env begin here')
                # 获取不同testtype的轨迹列表ID
                tripIDlist_test = trip_lists.get(testtype, [])

                more_test_trajrange = [0, len(tripIDlist_test) - 1]
                if testtype == triptype:
                    traj_type = traj_type_target_test  # 独立同分布测试
                else:
                    traj_type = [0, 1]  # 域外分布测试范围

                test_trajlist=range(more_test_trajrange[0],more_test_trajrange[-1]+1)#[0,1,2,3,4,5]
                for test_traj in test_trajlist:
                    test_trajdata_range = [test_traj, test_traj]
                    if networkmod in discrete_lists:
                        env = DummyVecEnv([lambda: GPSPosition_discrete_pos(test_trajdata_range, traj_type, testtype, action_scale, discrete_actionspace,
                                                                           reward_setting,trajdata_sort,baseline_mod, posnum)])
                    elif networkmod in continuous_lists:
                        env = DummyVecEnv([lambda: GPSPosition_continuous_lospos(test_trajdata_range, traj_type, testtype, continuous_action_scale,
                                                                                 continuous_actionspace,reward_setting, trajdata_sort, baseline_mod, posnum)])
                    obs = env.reset()
                    maxiter = 100000
                    for iter in range(maxiter):
                        action, _states = model.predict(obs)
                        obs, rewards, done, info = env.step(action)
                        tmp = info[0]['tripIDnum']
                        if iter <= 1 or iter % np.ceil(maxiter / 10) == 0:
                            # print(f'Iter {:.1f} reward is {:.2e}'.format(iter, rewards))
                            print(f'Iter {iter}, traj {tmp} reward is {rewards}')
                        elif done:
                            print(f'Iter {iter}, traj {tmp} reward is {rewards}, done')
                            break

                    pd_train = data_truth_dic[tripIDlist_test[int(info[0]['tripIDnum'])]]
                    # pd_train=info[0]['baseline']
                    if baseline_mod == 'bl':
                        test = pd_train.loc[:, ['ecefX', 'ecefY', 'ecefZ',
                                                'X_RLpredict', 'Y_RLpredict', 'Z_RLpredict',
                                                'XEcefMeters_bl', 'YEcefMeters_bl', 'ZEcefMeters_bl']]
                    elif baseline_mod == 'wls':
                        test = pd_train.loc[:, ['ecefX', 'ecefY', 'ecefZ',
                                                'X_RLpredict', 'Y_RLpredict', 'Z_RLpredict',
                                                'XEcefMeters_wls', 'YEcefMeters_wls', 'ZEcefMeters_wls']]
                    elif baseline_mod == 'kf':
                        test = pd_train.loc[:, ['ecefX', 'ecefY', 'ecefZ',
                                                'X_RLpredict', 'Y_RLpredict', 'Z_RLpredict',
                                                'XEcefMeters_kf', 'YEcefMeters_kf', 'ZEcefMeters_kf']]
                    elif baseline_mod == 'kf_igst':
                        test = pd_train.loc[:, ['ecefX', 'ecefY', 'ecefZ',
                                                'X_RLpredict', 'Y_RLpredict', 'Z_RLpredict',
                                                'XEcefMeters_kf_igst', 'YEcefMeters_kf_igst', 'ZEcefMeters_kf_igst']]
                    test['rl_distance'] = test.apply(lambda test: cal_distance_ecef(test,baseline_mod)[0], axis=1) # 对test DataFrame 中的每一行使用 cal_distance_ecef 函数
                    test['or_distance'] = test.apply(lambda test: cal_distance_ecef(test,baseline_mod)[1], axis=1)
                    test['error'] = test['rl_distance'].astype(float) - test['or_distance'].astype(float)
                    test['count_rl_distance'] = test['rl_distance'].astype(float)
                    test['count_or_distance'] = test['or_distance'].astype(float)
                    print(test['error'].describe())
                    print(test['count_rl_distance'].describe())
                    print(test['count_or_distance'].describe())

                logdirname = model.logger.dir + f'/testmore_{testtype}_'
                rl_distance = recording_results_ecef(data_truth_dic,[test_trajlist[0],test_trajlist[-1]],tripIDlist_test,logdirname,baseline_mod,traj_record=True)
                with open(model.logger.dir[:-5] + f'MVDRLSR_run={model.logger.dir[-1]}_dis={rl_distance:.3}.txt','w') as file:
                    file.write(f'dis={rl_distance}\n')
                file.close()
        cnt=1

print('More Test for different areas finished.')
