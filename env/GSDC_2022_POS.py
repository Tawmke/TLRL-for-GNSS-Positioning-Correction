#强化学习定位环境构建
import gym
from gym import spaces
import random
import pickle
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import glob as gl
from env.env_param import *
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# import lightgbm as lgb
# from sklearn.metrics import mean_absolute_error
import simdkalman
step_print=False
#导入数据
dir_path = '/mnt/sdb/home/tangjh/smartphone-decimeter-2022/'#'/home/tangjianhao/smartphone-decimeter-2022/' # '/home/tangjh/smartphone-decimeter-2022/''D:/jianhao/smartphone-decimeter-2022/'
project_path= '/mnt/sdb/home/tangjh/DLRL4GNSSpos/'
# load raw data
with open(dir_path+'env/raw_baseline_gpsl1.pkl', "rb") as file:
    data_truth_dic = pickle.load(file)
file.close() # 导入数据的文件
gnss_trig=True
if gnss_trig:
    with open(dir_path+'env/raw_gnss_gpsl1.pkl', "rb") as file:
        gnss_dic = pickle.load(file)
    file.close()
with open(dir_path + 'env/raw_tripID_gpsl1.pkl', "rb") as file:
    tripIDlist_full = pickle.load(file)
file.close()
with open(dir_path + 'env/processed_features_gpsl1_ecef_id.pkl', "rb") as file:
    losfeature = pickle.load(file)
file.close()
random.seed(0)

satnum_df = pd.read_csv(f'{dir_path}/env/raw_satnum_gpsl1.csv')
traj_sum_df = pd.read_csv(f'{project_path}/env/traj_summary.csv')
traj_highway=traj_sum_df.loc[traj_sum_df['Type']=='highway']['tripId'].values.tolist()
traj_else=traj_sum_df.loc[traj_sum_df['Type']=='urban']['tripId'].values.tolist()

class GPSPosition_discrete_pos(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,trajdata_range, traj_type, triptype, action_scale, discrete_actionspace, reward_setting, trajdata_sort, baseline_mod, traj_len):
        super(GPSPosition_discrete_pos, self).__init__()
        self.max_visible_sat=13
        self.traj_len = traj_len
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(self.max_visible_sat, 4), dtype=np.float)#shape=(2, 1)

        self.observation_space = spaces.Box(low=0, high=1, shape=(3, self.traj_len), dtype=np.float)

        if triptype == 'urban':
            self.tripIDlist = traj_else

        self.traj_type = traj_type
        if trajdata_range=='full':
            self.trajdata_range = [0, len(self.tripIDlist)-1]
        else:
            self.trajdata_range = trajdata_range
        # discrete action
        self.discrete_actionspace = discrete_actionspace
        self.action_space = spaces.Discrete(discrete_actionspace**3)
        self.action_scale = action_scale
        self.total_reward = 0
        self.reward_setting=reward_setting
        self.trajdata_sort=trajdata_sort
        self.baseline_mod=baseline_mod
        if self.trajdata_sort == 'sorted':
            self.tripIDnum = self.trajdata_range[0]
            # continuous action
        # self.action_space = spaces.Box(low=-1, high=1, dtype=np.float)

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        if self.trajdata_sort=='randint':
            # self.tripIDnum=random.randint(0,len(self.tripIDlist)-1)
            self.tripIDnum=random.randint(self.trajdata_range[0],self.trajdata_range[1])
        elif self.trajdata_sort=='sorted':
            self.tripIDnum = self.tripIDnum+1
            if self.tripIDnum>self.trajdata_range[1]:
                self.tripIDnum = self.trajdata_range[0]

        # self.tripIDnum=tripIDnum
        # self.info['tripIDnum']=self.tripIDnum
        self.baseline=data_truth_dic[self.tripIDlist[self.tripIDnum]].copy()
        self.losfeature=losfeature[self.tripIDlist[self.tripIDnum]].copy()
        self.datatime=self.baseline['UnixTimeMillis']
        self.timeend=self.baseline.loc[len(self.baseline.loc[:, 'UnixTimeMillis'].values)-1, 'UnixTimeMillis']
        #normalize baseline
        # self.baseline['LatitudeDegrees_norm'] = (self.baseline['LatitudeDegrees']-lat_min)/(lat_max-lat_min)
        # self.baseline['LongitudeDegrees_norm'] = (self.baseline['LongitudeDegrees']-lon_min)/(lon_max-lon_min)
        # gen pred
        if self.baseline_mod == 'bl':
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_bl']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_bl']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_bl']
        elif self.baseline_mod == 'wls':
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_wls']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_wls']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_wls']
        elif self.baseline_mod == 'kf':
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_kf']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_kf']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_kf']
        elif self.baseline_mod == 'kf_igst':
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_kf_igst']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_kf_igst']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_kf_igst']

        if gnss_trig:
            self.gnss=gnss_dic[self.tripIDlist[self.tripIDnum]]
        # Set the current step to a random point within the data frame
        # self.current_step = random.randint(0, len(self.df.loc[:, 'latDeg_norm'].values) - (traj_len-1))
        self.visible_sat=satnum_df.loc[satnum_df.loc[:,self.tripIDlist[self.tripIDnum]]>0,'Svid'].to_numpy()
        # revise 1017: need the specific percent of the traj
        self.current_step = np.ceil(len(self.baseline) * self.traj_type[0])  # self.current_step = 0
        if self.traj_type[0] > 0:  # 只要剩下部分轨迹的定位结果
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['X_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['Y_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['Z_RLpredict']] = None

        obs=self._next_observation()
        # must return in observation scale
        return obs#self.tripIDnum#, obs#, {}

    def _normalize_pos(self,state):
        state[0]=(state[0]-xecef_min) / (xecef_max - xecef_min)
        state[1]=(state[1]-yecef_min) / (yecef_max - yecef_min)
        state[2]=(state[2]-zecef_min) / (zecef_max - zecef_min)
        return state

    def _next_observation(self):
        obs = np.array([
            self.baseline.loc[self.current_step: self.current_step + (self.traj_len-2), 'X_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.traj_len-2), 'Y_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.traj_len-2), 'Z_RLpredict'].values
        ])
        if self.baseline_mod == 'bl':
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.traj_len-1), 'XEcefMeters_bl']],
                                 [self.baseline.loc[self.current_step + (self.traj_len-1), 'YEcefMeters_bl']],
                                 [self.baseline.loc[self.current_step + (self.traj_len-1), 'ZEcefMeters_bl']]],axis=1)
        elif self.baseline_mod == 'wls':
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.traj_len-1), 'XEcefMeters_wls']],
                                 [self.baseline.loc[self.current_step + (self.traj_len-1), 'YEcefMeters_wls']],
                                 [self.baseline.loc[self.current_step + (self.traj_len-1), 'ZEcefMeters_wls']]],axis=1)
        elif self.baseline_mod == 'kf':
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.traj_len-1), 'XEcefMeters_kf']],
                                 [self.baseline.loc[self.current_step + (self.traj_len-1), 'YEcefMeters_kf']],
                                 [self.baseline.loc[self.current_step + (self.traj_len-1), 'ZEcefMeters_kf']]],axis=1)
        elif self.baseline_mod == 'kf_igst':
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.traj_len-1), 'XEcefMeters_kf_igst']],
                                 [self.baseline.loc[self.current_step + (self.traj_len-1), 'YEcefMeters_kf_igst']],
                                 [self.baseline.loc[self.current_step + (self.traj_len-1), 'ZEcefMeters_kf_igst']]],axis=1)

        obs=self._normalize_pos(obs)

        # TODO latDeg lngDeg ... latDeg lngDeg
        return obs

    def step(self, action):
        # judge if end #
        done = (self.current_step >= len(self.baseline.loc[:, 'UnixTimeMillis'].values) * self.traj_type[-1] - (self.traj_len) - outlayer_in_end_ecef)
        timestep = self.baseline.loc[self.current_step + (self.traj_len - 1), 'UnixTimeMillis']
        # action for new prediction
        predict_x=action % self.discrete_actionspace
        predict_yz=action // self.discrete_actionspace
        predict_z, predict_y = predict_yz // self.discrete_actionspace, predict_yz % self.discrete_actionspace
        predict_x = (predict_x - self.discrete_actionspace//2) * self.action_scale# RL调节范围 1e-6对应cm
        predict_y = (predict_y - self.discrete_actionspace//2) * self.action_scale
        predict_z = (predict_z - self.discrete_actionspace//2) * self.action_scale
        if self.baseline_mod == 'bl':
            obs_x = self.baseline.loc[self.current_step + (self.traj_len-1), 'XEcefMeters_bl']
            obs_y = self.baseline.loc[self.current_step + (self.traj_len-1), 'YEcefMeters_bl']
            obs_z = self.baseline.loc[self.current_step + (self.traj_len-1), 'ZEcefMeters_bl']
        elif self.baseline_mod == 'wls':
            obs_x = self.baseline.loc[self.current_step + (self.traj_len-1), 'XEcefMeters_wls']
            obs_y = self.baseline.loc[self.current_step + (self.traj_len-1), 'YEcefMeters_wls']
            obs_z = self.baseline.loc[self.current_step + (self.traj_len-1), 'ZEcefMeters_wls']
        elif self.baseline_mod == 'kf':
            obs_x = self.baseline.loc[self.current_step + (self.traj_len-1), 'XEcefMeters_kf']
            obs_y = self.baseline.loc[self.current_step + (self.traj_len-1), 'YEcefMeters_kf']
            obs_z = self.baseline.loc[self.current_step + (self.traj_len-1), 'ZEcefMeters_kf']
        elif self.baseline_mod == 'kf_igst':
            obs_x = self.baseline.loc[self.current_step + (self.traj_len-1), 'XEcefMeters_kf_igst']
            obs_y = self.baseline.loc[self.current_step + (self.traj_len-1), 'YEcefMeters_kf_igst']
            obs_z = self.baseline.loc[self.current_step + (self.traj_len-1), 'ZEcefMeters_kf_igst']
        gro_x = self.baseline.loc[self.current_step + (self.traj_len-1), 'ecefX']
        gro_y = self.baseline.loc[self.current_step + (self.traj_len-1), 'ecefY']
        gro_z = self.baseline.loc[self.current_step + (self.traj_len-1), 'ecefZ']
        rl_x = obs_x + predict_x
        rl_y = obs_y + predict_y
        rl_z = obs_z + predict_z
        self.baseline.loc[self.current_step + (self.traj_len-1), ['X_RLpredict']] = rl_x
        self.baseline.loc[self.current_step + (self.traj_len-1), ['Y_RLpredict']] = rl_y
        self.baseline.loc[self.current_step + (self.traj_len-1), ['Z_RLpredict']] = rl_z
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.traj_len-1), ['X_RLpredict']] = rl_x
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.traj_len-1), ['Y_RLpredict']] = rl_y
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.traj_len-1), ['Z_RLpredict']] = rl_z
        # reward function
        if self.reward_setting=='RMSE':
            # reward = np.mean(-((rl_lat - gro_lat) ** 2 + (rl_lng - gro_lng) ** 2))
            reward = -np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2))*1e0#*1e5
        elif self.reward_setting=='RMSEadv':
            reward = np.sqrt(((obs_x - gro_x) ** 2 + (obs_y - gro_y) ** 2 + (obs_z - gro_z) ** 2))*1e0 - \
                     np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2))*1e0
        if step_print:
            print(f'{self.tripIDlist[self.tripIDnum]}, Time {timestep}/{self.timeend} Baseline dist: [{np.abs(obs_x - gro_x):.2f}, {np.abs(obs_y - gro_y):.2f}, {np.abs(obs_z - gro_z):.2f}] m, '
                  f'RL dist: [{np.abs(rl_x - gro_x):.2f}, {np.abs(rl_y - gro_y):.2f}, {np.abs(rl_z - gro_z):.2f}] m, RMSEadv: {reward:0.2e} m.')
        self.total_reward += reward
        # Execute one time step within the environment
        self.current_step += 1
        if done:
            obs = []
        else:
            obs = self._next_observation()
        return obs, reward, done, {'tripIDnum':self.tripIDnum, 'current_step':self.current_step, 'baseline':self.baseline} #self.info#, {}# , 'data_truth_dic':data_truth_dic

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')
        #  print(f'reward: {self.reward}')
        print(f'total_reward: {self.total_reward}')

