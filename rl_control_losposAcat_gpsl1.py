import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from env.GSDC_2022_LOSPOS_gpsl1 import *
# from env.GSDC_2022_LOSPOS_gpsl1 import *
# from env.dummy_cec_env_custom import *
from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3 import PPO
# from sb3_contrib import RecurrentPPO
from sb3_161.ppo import PPO
from sb3_161.ppo_recurrent import RecurrentPPO
from funcs.utilis import *
from model.model_custom_demo import CustomActorCriticPolicy
from model.model_custom_lstm import *
from model.model_transformer import *
from model.model_ATF import *
import time
import socket
hostname=socket.gethostname()
pid = os.getpid()

# traj type: full 79 urban 32 highway 47 losangel 21 bayarea 58
traj_type='urban'
traj_ratio=1
trajnum_ratio=1
trajnum_ratio_test=1
trajdata_range=[0,50]
trajdata_sort='sorted' # 'randint' 'sorted'
# baseline for RL: bl, wls, kf, kf_igst
baseline_mod='kf_igst'
# around 37.52321 -122.35447 scale 1e-5 lat 1.11m, lon 0.88m
# continuous action settings
max_action=100
continuous_action_scale=20e-1 #
continuous_actionspace=[-max_action,max_action]
# discrete action settings
discrete_actionspace=7
action_scale = 10e1 # scale in meters
# training settings
learning_rate_list=[1e-3] # [5e-2,1e-2,5e-3,1e-3,5e-4]
# learning_rate_list=[1e-5,1e-3]#[3e-4]#
# learning_rate = 1e-4
# param for SC 1: lambda L1mh_c Hoyermh_c
regulation_method='L1mh_c'
regulation_param1_list=[0] #[1e-4,1e-3,1e-2,5e-2,1e-1]
regulation_param2_list=[1e-5]
regulation_param3_list=[1e-3]
regulation_param4_list=[0]
# regulation_param1 = 1e-8 # sp for a_lambda
# regulation_param2 = 1e-1 # sp for c_lambda
# regulation_param3 = 1e-1 # log a_delta, skl a_beta
# regulation_param4 = 1e-0 # log c_delta, skl c_beta
#state-dependent exploration (SDE) sde_sample_freq default=8 in paper
sde_sample_freq_list=[1]
# sde_sample_freq=16
# pos num for lospos cat envir
pos_num_list=[10]#[5,10,20,40,80]
# pos_num_list=[10,5]
# pos_num=10
# select network and environment
discrete_lists=['discrete','discrete_A2C','discrete_lstm','ppo_discrete']
continuous_lists=['continuous','continuous_lstm','continuous_lstm_custom','ppo','continuous_custom',
                  'continuous_demo','continuous_customlstm','continuous_customlstm1','continuous_customlstm2',
                  'continuous_customGNN','continuous_customtransformer','continuous_ATN','continuous_ATNlstm',
                  'continuous_deepset','continuous_ATN1','continuous_ATN2','continuous_deepset1',
                  'continuous_lstmSC','continuous_lstmATF1','continuous_lstmSATF1','continuous_lstmCSATF1',
                  'continuous_lstmCoSATF1','continuous_lstmsATF1','continuous_lstmCoATF1','continuous_lstmCATF1',
                  'continuous_lstmCfATF1','continuous_lstmCpATF1','continuous_lstmFATF1','continuous_lstmSATFre1',
                  'continuous_lstmATF1_SC','continuous_lstmCoATF1_SC','continuous_lstmATF1_sdeSC','continuous_lstmATF1_sde',
                  'continuous_septransformerATF1', 'continuous_sepGNNATF1']
custom_lists=['ppo_discrete','ppo'] #continuous_lstmATF1_SC
networkmod='continuous_lstmATF1_SC' #
# select environment type
envlists=['latlon','ned','ecef','los','los_gpsl1_ex','losposcat_gpsl1_ex','losposAcat_gpsl1']
envmod='losposAcat_gpsl1'

if traj_type == 'highway':
    tripIDlist = traj_highway
elif traj_type == 'urban':
    tripIDlist = traj_else
    training_stepnum = 85234  # urban[1,6] 50epi 总步数 85234
    trajdata_range = [1, 6]

# if traj_ratio >= 1:
#     trajdata_range = [0, len(tripIDlist) - 1]
# else:
#     trajdata_range = [0, int(np.floor(len(tripIDlist) * traj_ratio))]

more_test_trajrange = [0, len(tripIDlist) - 1]

# recording parameters ############## ATTENTION ##########
running_date='test' # 0725_PPO_Urban
verbose=0 #Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for debug messages
######## ATTENTION ################## ATTENTION ##########
# test settings
moretests=0 #True False
for regulation_param1 in regulation_param1_list:
    for regulation_param2 in regulation_param2_list:
        for regulation_param3 in regulation_param3_list:
            for regulation_param4 in regulation_param4_list:
                for sde_sample_freq in sde_sample_freq_list:
                    for learning_rate in learning_rate_list:
                        for pos_num in pos_num_list:
                            if networkmod in discrete_lists:
                                print(f'Action scale {action_scale:8.2e}, discrete action space {discrete_actionspace}')
                            elif networkmod in continuous_lists:
                                print(f'Action scale {continuous_action_scale:8.2e}, contiuous action space from {continuous_actionspace[0]} to {continuous_actionspace[1]}, '
                                      f'pos num {pos_num}')
                            if 'SC' in networkmod:
                                if 'sp' in regulation_method:
                                    if 'Log' in regulation_method or 'SKL' in regulation_method:
                                        print(f'Learning rate: {learning_rate:8.2e}, lambda_a: {regulation_param1:8.2e}, delta_a: {regulation_param3:8.2e}, lambda_c: {regulation_param2:8.2e}, delta_c: {regulation_param4:8.2e}')
                                        tensorboard_log=f'./records_values/{running_date}/{networkmod}{regulation_method}a{regulation_param3:8.2e}_c{regulation_param4:8.2e}_{traj_type}{traj_ratio}_{envmod}_{baseline_mod}'
                                    else:
                                        print(f'Learning rate: {learning_rate:8.2e}, actor lambda: {regulation_param1:8.2e}, critic lambda: {regulation_param2:8.2e}')
                                        tensorboard_log=f'./records_values/{running_date}/{networkmod}{regulation_method}_{traj_type}{traj_ratio}_n{trajnum_ratio}_t{trajnum_ratio_test}_{envmod}_{baseline_mod}'
                                else:
                                    if 'Log' in regulation_method or 'SKL' in regulation_method:
                                        print(f'Learning rate: {learning_rate:8.2e}, lambda: {regulation_param1:8.2e}, delta: {regulation_param3:8.2e}')
                                        tensorboard_log=f'./records_values/{running_date}/{networkmod}{regulation_method}{regulation_param3:8.2e}_{traj_type}{traj_ratio}_n{trajnum_ratio}_t{trajnum_ratio_test}_{envmod}_{baseline_mod}'
                                    else:
                                        print(f'Learning rate: {learning_rate:8.2e}, lambda: {regulation_param1:8.2e}')
                                        tensorboard_log=f'{project_path}/records_values/{running_date}/{networkmod}{regulation_method}_{traj_type}{traj_ratio}_n{trajnum_ratio}_t{trajnum_ratio_test}_{baseline_mod}_lam={regulation_param1}_lr={learning_rate}'
                            else:
                                print(f'Learning rate: {learning_rate:8.2e} ')
                                tensorboard_log=f'./records_values/{running_date}/{networkmod}_{traj_type}{traj_ratio}_n{trajnum_ratio}_t{trajnum_ratio_test}_{envmod}_{baseline_mod}'
                            if 'sde' in networkmod and sde_sample_freq > 0:
                                print(f'sde_sample_freq: {sde_sample_freq}')

                            t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                            print(f'Log path: {tensorboard_log}, on {hostname}, pid {pid}, time {t}')
                            reward_setting='RMSEadv' # 'RMSE' ‘RMSEadv'
                            # data cata: KF: LatitudeDegrees; robust WLS: LatitudeDegrees_wls; standard WLS: LatitudeDegrees_bl
                            # parameters for customized ppo

                            #default param
                            customlayer_num = ''
                            features_dim = ''
                            if envmod=='losposAcat_gpsl1':
                                if networkmod=='continuous_septransformerATF1':
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_losposAcat(trajdata_range, traj_type, continuous_action_scale, continuous_actionspace,
                                                                                           reward_setting,trajdata_sort,baseline_mod,pos_num,trajnum_ratio)])
                                    obs = env.reset()
                                    features_dim = 0
                                    for tmp in obs:
                                        features_dim+=obs[tmp].shape[-1]
                                    policy_kwargs = dict(
                                        features_extractor_class=CustomATF1, #CustomCNN CustomMLP
                                        features_extractor_kwargs=dict(features_dim=features_dim),
                                        ATF_trig=networkmod
                                    )
                                    model = RecurrentPPO("TransformerLstmPolicy", env, verbose=verbose, policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log, learning_rate=learning_rate)
                                    # action, _states = model.predict(obs)
                                    # obs, rewards, done, info = env.step(action)
                                    # save initial params
                                    model.learn(total_timesteps=training_stepnum, eval_log_path=tensorboard_log)
                                elif networkmod=='continuous_lstmATF1':
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_losposAcat(trajdata_range, traj_type, continuous_action_scale, continuous_actionspace,
                                                                                           reward_setting,trajdata_sort,baseline_mod,pos_num,trajnum_ratio)])
                                    obs = env.reset()
                                    features_dim = 0
                                    for tmp in obs:
                                        features_dim+=obs[tmp].shape[-1]
                                    policy_kwargs = dict(
                                        features_extractor_class=CustomATF1, #CustomCNN CustomMLP
                                        features_extractor_kwargs=dict(features_dim=features_dim),
                                        ATF_trig=networkmod
                                    )
                                    model = RecurrentPPO("MlpLstmPolicy", env, verbose=verbose, policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log, learning_rate=learning_rate)
                                    # action, _states = model.predict(obs)
                                    # obs, rewards, done, info = env.step(action)
                                    # save initial params
                                    model.learn(total_timesteps=training_stepnum, eval_log_path=tensorboard_log)
                                elif networkmod=='continuous_lstmATF1_sde':
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_losposAcat(trajdata_range, traj_type, continuous_action_scale, continuous_actionspace,
                                                                                           reward_setting,trajdata_sort,baseline_mod,pos_num,trajnum_ratio)])
                                    obs = env.reset()
                                    features_dim = 0
                                    for tmp in obs:
                                        features_dim+=obs[tmp].shape[-1]
                                    policy_kwargs = dict(
                                        features_extractor_class=CustomATF1, #CustomCNN CustomMLP
                                        features_extractor_kwargs=dict(features_dim=features_dim),
                                        ATF_trig=networkmod
                                    )
                                    model = RecurrentPPO("MlpLstmPolicy", env, verbose=verbose, policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log, learning_rate=learning_rate, use_sde=True, sde_sample_freq=sde_sample_freq)
                                    # action, _states = model.predict(obs)
                                    # obs, rewards, done, info = env.step(action)
                                    # save initial params
                                    model.learn(total_timesteps=training_stepnum, eval_log_path=tensorboard_log)
                                elif networkmod=='continuous_lstmsATF1':
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_losposAcat(trajdata_range, traj_type, continuous_action_scale, continuous_actionspace,
                                                                                           reward_setting,trajdata_sort,baseline_mod,pos_num,trajnum_ratio)])
                                    obs = env.reset()
                                    features_dim = 0
                                    for tmp in obs:
                                        features_dim+=obs[tmp].shape[-1]
                                    policy_kwargs = dict(
                                        features_extractor_class=CustomsATF1, #CustomCNN CustomMLP
                                        features_extractor_kwargs=dict(features_dim=features_dim),
                                        ATF_trig=networkmod
                                    )
                                    model = RecurrentPPO("MlpLstmPolicy", env, verbose=verbose, policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log, learning_rate=learning_rate)
                                    # action, _states = model.predict(obs)
                                    # obs, rewards, done, info = env.step(action)
                                    # save initial params
                                    model.learn(total_timesteps=training_stepnum, eval_log_path=tensorboard_log)
                                elif networkmod=='continuous_lstmSATF1':
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_losposAcat(trajdata_range, traj_type, continuous_action_scale, continuous_actionspace,
                                                                                           reward_setting,trajdata_sort,baseline_mod,pos_num,trajnum_ratio)])
                                    obs = env.reset()
                                    features_dim = 0
                                    for tmp in obs:
                                        features_dim+=obs[tmp].shape[-1]
                                    policy_kwargs = dict(
                                        features_extractor_class=CustomSATF1, #CustomCNN CustomMLP
                                        features_extractor_kwargs=dict(features_dim=features_dim),
                                        ATF_trig=networkmod
                                    )
                                    model = RecurrentPPO("MlpLstmPolicy", env, verbose=verbose, policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log, learning_rate=learning_rate)
                                    # action, _states = model.predict(obs)
                                    # obs, rewards, done, info = env.step(action)
                                    # save initial params
                                    model.learn(total_timesteps=training_stepnum, eval_log_path=tensorboard_log)
                                elif networkmod=='continuous_lstmsSATF1':
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_losposAcat(trajdata_range, traj_type, continuous_action_scale, continuous_actionspace,
                                                                                           reward_setting,trajdata_sort,baseline_mod,pos_num,trajnum_ratio)])
                                    obs = env.reset()
                                    features_dim = 0
                                    for tmp in obs:
                                        features_dim+=obs[tmp].shape[-1]
                                    policy_kwargs = dict(
                                        features_extractor_class=CustomsSATF1, #CustomCNN CustomMLP
                                        features_extractor_kwargs=dict(features_dim=features_dim),
                                        ATF_trig=networkmod
                                    )
                                    model = RecurrentPPO("MlpLstmPolicy", env, verbose=verbose, policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log, learning_rate=learning_rate)
                                    # action, _states = model.predict(obs)
                                    # obs, rewards, done, info = env.step(action)
                                    # save initial params
                                    model.learn(total_timesteps=training_stepnum, eval_log_path=tensorboard_log)
                                elif networkmod=='continuous_lstmSATFre1':
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_losposAcat(trajdata_range, traj_type, continuous_action_scale, continuous_actionspace,
                                                                                           reward_setting,trajdata_sort,baseline_mod,pos_num,trajnum_ratio)])
                                    obs = env.reset()
                                    features_dim = 0
                                    for tmp in obs:
                                        features_dim+=obs[tmp].shape[-1]
                                    policy_kwargs = dict(
                                        features_extractor_class=CustomSATFre1, #CustomCNN CustomMLP
                                        features_extractor_kwargs=dict(features_dim=features_dim),
                                        ATF_trig=networkmod
                                    )
                                    model = RecurrentPPO("MlpLstmPolicy", env, verbose=verbose, policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log, learning_rate=learning_rate)
                                    # action, _states = model.predict(obs)
                                    # obs, rewards, done, info = env.step(action)
                                    # save initial params
                                    model.learn(total_timesteps=training_stepnum, eval_log_path=tensorboard_log)
                                elif networkmod=='continuous_lstmCSATF1':
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_losposAcat(trajdata_range, traj_type, continuous_action_scale, continuous_actionspace,
                                                                                           reward_setting,trajdata_sort,baseline_mod,pos_num,trajnum_ratio)])
                                    obs = env.reset()
                                    features_dim = 0
                                    for tmp in obs:
                                        features_dim+=obs[tmp].shape[-1]
                                    hidden_size=64
                                    features_dim +=hidden_size
                                    policy_kwargs = dict(
                                        features_extractor_class=CustomCSATF1, #CustomCNN CustomMLP
                                        features_extractor_kwargs=dict(features_dim=features_dim),
                                        ATF_trig=networkmod
                                    )
                                    model = RecurrentPPO("MlpLstmPolicy", env, verbose=verbose, policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log, learning_rate=learning_rate)
                                    # action, _states = model.predict(obs)
                                    # obs, rewards, done, info = env.step(action)
                                    # save initial params
                                    model.learn(total_timesteps=training_stepnum, eval_log_path=tensorboard_log)
                                elif networkmod=='continuous_lstmCoSATF1':
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_losposAcat(trajdata_range, traj_type, continuous_action_scale, continuous_actionspace,
                                                                                           reward_setting,trajdata_sort,baseline_mod,pos_num,trajnum_ratio)])
                                    obs = env.reset()
                                    features_dim = 0
                                    for tmp in obs:
                                        features_dim+=obs[tmp].shape[-1]
                                    hidden_size=64
                                    features_dim +=hidden_size
                                    policy_kwargs = dict(
                                        features_extractor_class=CustomCoSATF1, #CustomCNN CustomMLP
                                        features_extractor_kwargs=dict(features_dim=features_dim),
                                        ATF_trig=networkmod
                                    )
                                    model = RecurrentPPO("MlpLstmPolicy", env, verbose=verbose, policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log, learning_rate=learning_rate)
                                    # action, _states = model.predict(obs)
                                    # obs, rewards, done, info = env.step(action)
                                    # save initial params
                                    model.learn(total_timesteps=training_stepnum, eval_log_path=tensorboard_log)
                                elif networkmod=='continuous_lstmCATF1':
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_losposAcat(trajdata_range, traj_type, continuous_action_scale, continuous_actionspace,
                                                                                           reward_setting,trajdata_sort,baseline_mod,pos_num,trajnum_ratio)])
                                    obs = env.reset()
                                    features_dim = 0
                                    for tmp in obs:
                                        features_dim+=obs[tmp].shape[-1]
                                    hidden_size=64
                                    features_dim +=hidden_size
                                    policy_kwargs = dict(
                                        features_extractor_class=CustomCATF1, #CustomCNN CustomMLP
                                        features_extractor_kwargs=dict(features_dim=features_dim),
                                        ATF_trig=networkmod
                                    )
                                    model = RecurrentPPO("MlpLstmPolicy", env, verbose=verbose, policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log, learning_rate=learning_rate)
                                    # action, _states = model.predict(obs)
                                    # obs, rewards, done, info = env.step(action)
                                    # save initial params
                                    model.learn(total_timesteps=training_stepnum, eval_log_path=tensorboard_log)
                                elif networkmod=='continuous_lstmCoATF1':
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_losposAcat(trajdata_range, traj_type, continuous_action_scale, continuous_actionspace,
                                                                                           reward_setting,trajdata_sort,baseline_mod,pos_num,trajnum_ratio)])
                                    obs = env.reset()
                                    features_dim = 0
                                    for tmp in obs:
                                        features_dim+=obs[tmp].shape[-1]
                                    hidden_size=16
                                    features_dim +=hidden_size
                                    policy_kwargs = dict(
                                        features_extractor_class=CustomCoATF1, #CustomCNN CustomMLP
                                        features_extractor_kwargs=dict(features_dim=features_dim),
                                        ATF_trig=networkmod
                                    )
                                    model = RecurrentPPO("MlpLstmPolicy", env, verbose=verbose, policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log, learning_rate=learning_rate)
                                    # action, _states = model.predict(obs)
                                    # obs, rewards, done, info = env.step(action)
                                    # save initial params
                                    model.learn(total_timesteps=training_stepnum, eval_log_path=tensorboard_log)
                                elif networkmod=='continuous_lstmCfATF1':
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_losposAcat(trajdata_range, traj_type, continuous_action_scale, continuous_actionspace,
                                                                                           reward_setting,trajdata_sort,baseline_mod,pos_num,trajnum_ratio)])
                                    obs = env.reset()
                                    features_dim = 0
                                    for tmp in obs:
                                        features_dim+=obs[tmp].shape[-1]
                                    hidden_size=32
                                    features_dim +=hidden_size
                                    policy_kwargs = dict(
                                        features_extractor_class=CustomCfATF1, #CustomCNN CustomMLP
                                        features_extractor_kwargs=dict(features_dim=features_dim),
                                        ATF_trig=networkmod
                                    )
                                    model = RecurrentPPO("MlpLstmPolicy", env, verbose=verbose, policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log, learning_rate=learning_rate)
                                    # action, _states = model.predict(obs)
                                    # obs, rewards, done, info = env.step(action)
                                    # save initial params
                                    model.learn(total_timesteps=training_stepnum, eval_log_path=tensorboard_log)
                                elif networkmod=='continuous_lstmCpATF1':
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_losposAcat(trajdata_range, traj_type, continuous_action_scale, continuous_actionspace,
                                                                                           reward_setting,trajdata_sort,baseline_mod,pos_num,trajnum_ratio)])
                                    obs = env.reset()
                                    features_dim = 0
                                    for tmp in obs:
                                        features_dim+=obs[tmp].shape[-1]
                                    hidden_size=32
                                    features_dim +=hidden_size
                                    policy_kwargs = dict(
                                        features_extractor_class=CustomCpATF1, #CustomCNN CustomMLP
                                        features_extractor_kwargs=dict(features_dim=features_dim),
                                        ATF_trig=networkmod
                                    )
                                    model = RecurrentPPO("MlpLstmPolicy", env, verbose=verbose, policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log, learning_rate=learning_rate)
                                    # action, _states = model.predict(obs)
                                    # obs, rewards, done, info = env.step(action)
                                    # save initial params
                                    model.learn(total_timesteps=training_stepnum, eval_log_path=tensorboard_log)
                                elif networkmod=='continuous_lstmFATF1':
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_losposAcat(trajdata_range, traj_type, continuous_action_scale, continuous_actionspace,
                                                                                           reward_setting,trajdata_sort,baseline_mod,pos_num,trajnum_ratio)])
                                    obs = env.reset()
                                    hidden_size1=64
                                    hidden_size2=32
                                    features_dim = hidden_size1+hidden_size2
                                    policy_kwargs = dict(
                                        features_extractor_class=CustomFATF1, #CustomCNN CustomMLP
                                        features_extractor_kwargs=dict(features_dim=features_dim),
                                        ATF_trig=networkmod
                                    )
                                    model = RecurrentPPO("MlpLstmPolicy", env, verbose=verbose, policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log, learning_rate=learning_rate)
                                    # action, _states = model.predict(obs)
                                    # obs, rewards, done, info = env.step(action)
                                    # save initial params
                                    model.learn(total_timesteps=training_stepnum, eval_log_path=tensorboard_log)
                                elif networkmod=='continuous_lstmSC':
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_losposAcat(trajdata_range, traj_type, continuous_action_scale, continuous_actionspace,
                                                                                           reward_setting,trajdata_sort,baseline_mod,pos_num,trajnum_ratio)])
                                    # obs = env.reset()
                                    if 'L1' in regulation_method or 'L0' in regulation_method or 'Hoyer' in regulation_method:
                                        if 'sp' in regulation_method:
                                            policy_kwargs = dict(
                                                regulation_method=regulation_method,
                                                regulation_param1=regulation_param1,
                                                regulation_param2=regulation_param2
                                            )
                                        else:
                                            policy_kwargs = dict(
                                                regulation_method=regulation_method,
                                                regulation_param1=regulation_param1
                                            )
                                    elif 'Log' in regulation_method or 'SKL' in regulation_method:
                                        if 'sp' in regulation_method:
                                            policy_kwargs = dict(
                                                regulation_method=regulation_method,
                                                regulation_param1=regulation_param1,
                                                regulation_param2=regulation_param2,
                                                regulation_param3=regulation_param3,
                                                regulation_param4=regulation_param4
                                            )
                                        else:
                                            policy_kwargs = dict(
                                                regulation_method=regulation_method,
                                                regulation_param1=regulation_param1,
                                                regulation_param3=regulation_param3
                                            )
                                    model = RecurrentPPO("MlpLstmPolicy", env, verbose=verbose, policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log, learning_rate=learning_rate)
                                    # action, _states = model.predict(obs)
                                    # obs, rewards, done, info = env.step(action)
                                    # save initial params
                                    model.learn(total_timesteps=training_stepnum, eval_log_path=tensorboard_log)
                                elif networkmod=='continuous_lstmATF1_SC':
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_losposAcat(trajdata_range, traj_type, continuous_action_scale, continuous_actionspace,
                                                                                           reward_setting,trajdata_sort,baseline_mod,pos_num,trajnum_ratio)])
                                    obs = env.reset()
                                    features_dim = 0
                                    for tmp in obs:
                                        features_dim+=obs[tmp].shape[-1]

                                    if 'L1' in regulation_method or 'L0' in regulation_method or 'Hoyer' in regulation_method:
                                        if 'sp' in regulation_method:
                                            policy_kwargs = dict(
                                                regulation_method=regulation_method,
                                                regulation_param1=regulation_param1,
                                                regulation_param2=regulation_param2
                                            )
                                        else:
                                            policy_kwargs = dict(
                                                regulation_method=regulation_method,
                                                regulation_param1=regulation_param1
                                            )
                                    elif 'Log' in regulation_method or 'SKL' in regulation_method:
                                        if 'sp' in regulation_method:
                                            policy_kwargs = dict(
                                                regulation_method=regulation_method,
                                                regulation_param1=regulation_param1,
                                                regulation_param2=regulation_param2,
                                                regulation_param3=regulation_param3,
                                                regulation_param4=regulation_param4
                                            )
                                        else:
                                            policy_kwargs = dict(
                                                regulation_method=regulation_method,
                                                regulation_param1=regulation_param1,
                                                regulation_param3=regulation_param3
                                            )

                                    policy_kwargs['features_extractor_class']=CustomATF1
                                    policy_kwargs['features_extractor_kwargs']=dict(features_dim=features_dim)
                                    policy_kwargs['ATF_trig']=networkmod

                                    model = RecurrentPPO("MlpLstmPolicy", env, verbose=verbose, policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log, learning_rate=learning_rate)
                                    # action, _states = model.predict(obs)
                                    # obs, rewards, done, info = env.step(action)
                                    # save initial params
                                    model.learn(total_timesteps=training_stepnum, eval_log_path=tensorboard_log)
                                elif networkmod=='continuous_lstmATF1_sdeSC':
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_losposAcat(trajdata_range, traj_type, continuous_action_scale, continuous_actionspace,
                                                                                           reward_setting,trajdata_sort,baseline_mod,pos_num,trajnum_ratio)])
                                    obs = env.reset()
                                    features_dim = 0
                                    for tmp in obs:
                                        features_dim+=obs[tmp].shape[-1]

                                    if 'L1' in regulation_method or 'L0' in regulation_method or 'Hoyer' in regulation_method:
                                        if 'sp' in regulation_method:
                                            policy_kwargs = dict(
                                                regulation_method=regulation_method,
                                                regulation_param1=regulation_param1,
                                                regulation_param2=regulation_param2
                                            )
                                        else:
                                            policy_kwargs = dict(
                                                regulation_method=regulation_method,
                                                regulation_param1=regulation_param1
                                            )
                                    elif 'Log' in regulation_method or 'SKL' in regulation_method:
                                        if 'sp' in regulation_method:
                                            policy_kwargs = dict(
                                                regulation_method=regulation_method,
                                                regulation_param1=regulation_param1,
                                                regulation_param2=regulation_param2,
                                                regulation_param3=regulation_param3,
                                                regulation_param4=regulation_param4
                                            )
                                        else:
                                            policy_kwargs = dict(
                                                regulation_method=regulation_method,
                                                regulation_param1=regulation_param1,
                                                regulation_param3=regulation_param3
                                            )

                                    policy_kwargs['features_extractor_class']=CustomATF1
                                    policy_kwargs['features_extractor_kwargs']=dict(features_dim=features_dim)
                                    policy_kwargs['ATF_trig']=networkmod

                                    model = RecurrentPPO("MlpLstmPolicy", env, verbose=verbose, policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log, learning_rate=learning_rate, use_sde=True, sde_sample_freq=sde_sample_freq)
                                    # action, _states = model.predict(obs)
                                    # obs, rewards, done, info = env.step(action)
                                    # save initial params
                                    model.learn(total_timesteps=training_stepnum, eval_log_path=tensorboard_log)
                                elif networkmod=='continuous_lstmCoATF1_SC':
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_losposAcat(trajdata_range, traj_type, continuous_action_scale, continuous_actionspace,
                                                                                           reward_setting,trajdata_sort,baseline_mod,pos_num,trajnum_ratio)])
                                    obs = env.reset()

                                    if 'L1' in regulation_method or 'L0' in regulation_method or 'Hoyer' in regulation_method:
                                        if 'sp' in regulation_method:
                                            policy_kwargs = dict(
                                                regulation_method=regulation_method,
                                                regulation_param1=regulation_param1,
                                                regulation_param2=regulation_param2
                                            )
                                        else:
                                            policy_kwargs = dict(
                                                regulation_method=regulation_method,
                                                regulation_param1=regulation_param1
                                            )
                                    elif 'Log' in regulation_method or 'SKL' in regulation_method:
                                        if 'sp' in regulation_method:
                                            policy_kwargs = dict(
                                                regulation_method=regulation_method,
                                                regulation_param1=regulation_param1,
                                                regulation_param2=regulation_param2,
                                                regulation_param3=regulation_param3,
                                                regulation_param4=regulation_param4
                                            )
                                        else:
                                            policy_kwargs = dict(
                                                regulation_method=regulation_method,
                                                regulation_param1=regulation_param1,
                                                regulation_param3=regulation_param3
                                            )

                                    features_dim = 0
                                    for tmp in obs:
                                        features_dim+=obs[tmp].shape[-1]
                                    hidden_size=64
                                    features_dim +=hidden_size
                                    policy_kwargs = dict(
                                        features_extractor_class=CustomCoATF1, #CustomCNN CustomMLP
                                        features_extractor_kwargs=dict(features_dim=features_dim),
                                        ATF_trig=networkmod
                                    )
                                    policy_kwargs['features_extractor_class']=CustomCoATF1
                                    policy_kwargs['features_extractor_kwargs']=dict(features_dim=features_dim)
                                    policy_kwargs['ATF_trig']=networkmod

                                    model = RecurrentPPO("MlpLstmPolicy", env, verbose=verbose, policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log, learning_rate=learning_rate)
                                    # action, _states = model.predict(obs)
                                    # obs, rewards, done, info = env.step(action)
                                    # save initial params
                                    model.learn(total_timesteps=training_stepnum, eval_log_path=tensorboard_log)

                            #print and save training results
                            logdirname=model.logger.dir+'/train_'
                            # logdirname='./'
                            t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                            print(f'Training finished. time {t}')

                            #record model
                            # params=model.get_parameters()
                            if networkmod in discrete_lists:
                                model.save(model.logger.dir+f"/{networkmod}{customlayer_num}_f{features_dim}_{reward_setting}_action{discrete_actionspace}_{action_scale:0.1e}_trgnum{training_stepnum:0.1e}"
                                                            f"_env_{baseline_mod}{envmod}p{pos_num}range{trajdata_range[0]}_{trajdata_range[-1]}{trajdata_sort}_lr{learning_rate:0.1e}")
                            elif networkmod in continuous_lists:
                                if sde_sample_freq>0:
                                    sde_txt=f'{sde_sample_freq}'
                                else:
                                    sde_txt=''
                                if 'SC' in networkmod:
                                    if 'sp' in regulation_method:
                                        if 'Log' in regulation_method or 'SKL' in regulation_method:
                                            regulation_txt=f'{regulation_method}_a{regulation_param1:0.2e}_{regulation_param3:0.2e}_c{regulation_param2:0.2e}_{regulation_param4:0.2e}'
                                        else:
                                            regulation_txt=f'{regulation_method}_a{regulation_param1:0.2e}_c{regulation_param2:0.2e}'
                                    else:
                                        if 'Log' in regulation_method or 'SKL' in regulation_method:
                                            regulation_txt=f'{regulation_method}{regulation_param1:0.2e}_{regulation_param3:0.2e}'
                                        else:
                                            regulation_txt=f'{regulation_method}{regulation_param1:0.2e}'
                                    model.save(model.logger.dir+f"/{networkmod}{sde_txt}_{customlayer_num}_{regulation_txt}_f{features_dim}_{reward_setting}_action{continuous_actionspace[0]}_{continuous_actionspace[1]}"
                                                            f"_{continuous_action_scale:0.1e}_trgnum{training_stepnum:0.1e}"
                                                            f"_env_{baseline_mod}{envmod}p{pos_num}range{trajdata_range[0]}_{trajdata_range[-1]}{trajdata_sort}_lr{learning_rate:0.1e}")
                                else:
                                    model.save(model.logger.dir+f"/{networkmod}{sde_txt}_{customlayer_num}_f{features_dim}_{reward_setting}_action{continuous_actionspace[0]}_{continuous_actionspace[1]}"
                                                            f"_{continuous_action_scale:0.1e}_trgnum{training_stepnum:0.1e}"
                                                            f"_env_{baseline_mod}{envmod}p{pos_num}range{trajdata_range[0]}_{trajdata_range[-1]}{trajdata_sort}_lr{learning_rate:0.1e}")

                            rl_distance = recording_results_ecef(data_truth_dic,trajdata_range,tripIDlist,logdirname,baseline_mod,traj_record=True)
                            with open(model.logger.dir[:-14] + f'MVDRLSR_run={model.logger.dir[-1]}_dis={rl_distance:.3}.txt', 'w') as file:
                                file.write(f'dis={rl_distance}\n')
                            file.close()

                            # more tests
                            iterprint=False
                            if moretests>0:
                                for testcnt in range(moretests):
                                    test_trajlist=range(more_test_trajrange[0],more_test_trajrange[-1]+1)#[0,1,2,3,4,5]
                                    for test_traj in test_trajlist:
                                        test_trajdata_range = [test_traj, test_traj]
                                        # if networkmod in discrete_lists:
                                        #     env = DummyVecEnv([lambda: GPSPosition_discrete_los(test_trajdata_range, traj_type, action_scale, discrete_actionspace,
                                        #                                                        reward_setting,trajdata_sort,baseline_mod,pos_num,trajnum_ratio_test)])
                                        if networkmod in continuous_lists:
                                            env = DummyVecEnv([lambda: GPSPosition_continuous_losposAcat(test_trajdata_range, traj_type, continuous_action_scale, continuous_actionspace,
                                                                                               reward_setting,trajdata_sort,baseline_mod,pos_num,trajnum_ratio_test)])
                                        obs = env.reset()
                                        maxiter = 10000
                                        for iter in range(maxiter):
                                            action, _states = model.predict(obs)
                                            obs, rewards, done, info = env.step(action)
                                            tmp = info[0]['tripIDnum']
                                            if iterprint:
                                                t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                                                if iter <= 1 or iter % 1000 == 0:
                                                    print(f'Iter {iter}, traj {tmp} reward is {rewards}, time {t}')
                                            if done:
                                                if iterprint:
                                                    print(f'Iter {iter}, traj {tmp} reward is {rewards}, done, time {t}')
                                                break

                                        pd_train = data_truth_dic[tripIDlist[int(info[0]['tripIDnum'])]]
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
                                        test['rl_distance'] = test.apply(lambda test: cal_distance_ecef(test,baseline_mod)[0], axis=1)
                                        test['or_distance'] = test.apply(lambda test: cal_distance_ecef(test,baseline_mod)[1], axis=1)
                                        test['error'] = test['rl_distance'].astype(float) - test['or_distance'].astype(float)
                                        test['count_rl_distance'] = test['rl_distance'].astype(float)
                                        test['count_or_distance'] = test['or_distance'].astype(float)
                                        error_mean=np.nanmean(test['error'])
                                        error_std=np.nanstd(test['error'])
                                        rl_mean=np.nanmean(test['rl_distance'])
                                        rl_std=np.nanstd(test['rl_distance'])
                                        or_mean=np.nanmean(test['or_distance'])
                                        or_std=np.nanstd(test['or_distance'])
                                        traj_name=tripIDlist[int(info[0]['tripIDnum'])]
                                        print(f'{traj_name}, error: {error_mean:1.3f}+{error_std:1.3f}, rl_distance: {rl_mean:1.3f}+{rl_std:1.3f}, '
                                              f'or_distance: {or_mean:1.3f}+{or_std:1.3f}')

                                    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                                    print(f'More Test {testcnt} finished. time {t}')
                                    logdirname=model.logger.dir+f'/testmore{testcnt}_'
                                    recording_results_ecef_rectraj(data_truth_dic,[test_trajlist[0],test_trajlist[-1]],tripIDlist,logdirname,baseline_mod,False)

                            cnt=1
                            # randnum=[]
                            # for i in range(168):
                            #     randnum.append(random.randint(0,168))
                            # if networkmod in discrete_lists:
                            #     predict_lat, predict_lng = action // discrete_actionspace, action % discrete_actionspace
                            #     predict_lat = (predict_lat - discrete_actionspace // 2) * action_scale  # RL调节范围 1e-6对应cm
                            #     predict_lng = (predict_lng - discrete_actionspace // 2) * action_scale
                            #     print(f'predict_lat: {predict_lat}, predict_lng: {predict_lng}')
                            #
                            #     action, _states = model.predict(obs)
                            #
                            #     predict_lat, predict_lng = action // discrete_actionspace, action % discrete_actionspace
                            #     predict_lat = (predict_lat - discrete_actionspace // 2) * action_scale  # RL调节范围 1e-6对应cm
                            #     predict_lng = (predict_lng - discrete_actionspace // 2) * action_scale
                            #     print(f'Another predict_lat: {predict_lat}, predict_lng: {predict_lng}')
                            # elif networkmod in continuous_lists:
                            #     print(f'action: {action}')
                            #     action, _states = model.predict(obs)
                            #     print(f'Another action: {action}')