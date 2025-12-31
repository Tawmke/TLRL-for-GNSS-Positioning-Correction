#强化学习定位环境构建
import gym
from gym import spaces
import torch
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
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import simdkalman
step_print=False
#导入数据
dir_path = '/mnt/sdb/home/tangjh/smartphone-decimeter-2022/'
project_path = '/mnt/sdb/home/tangjh/DLRL4GNSSpos/'
# load raw data
with open(dir_path+'env/raw_baseline_gpsl1.pkl', "rb") as file:
    data_truth_dic = pickle.load(file)
file.close()
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

class GPSPosition_continuous_losposAcat(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,trajdata_range, traj_type, continuous_action_scale, continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, pos_num, trajnum_ratio):
    # def __init__(self,trajdata_range, action_scale, discrete_actionspace, reward_setting, trajdata_sort, baseline_mod):
        super(GPSPosition_continuous_losposAcat, self).__init__()
        self.max_visible_sat=13
        self.pos_num=pos_num #50
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(1, 3*self.pos_num+self.max_visible_sat*4), dtype=np.float)#shape=(2, 1)
        self.observation_space = spaces.Dict({'gnss':spaces.Box(low=-1, high=1, shape=(1, self.max_visible_sat*4)),
                                              'pos':spaces.Box(low=0, high=1, shape=(1, 3*self.pos_num), dtype=np.float)})
        if traj_type=='highway':
            self.tripIDlist=traj_highway
        elif traj_type=='full':
            self.tripIDlist=tripIDlist_full
        elif traj_type=='urban':
            self.tripIDlist = traj_else
        elif traj_type=='losangel':
            self.tripIDlist = traj_losangel
        elif traj_type=='bayarea':
            self.tripIDlist = traj_bayarea

        # continuous action
        if trajdata_range=='full':
            self.trajdata_range = [0, len(self.tripIDlist)-1]
        else:
            self.trajdata_range = trajdata_range
        self.continuous_actionspace = continuous_actionspace
        self.continuous_action_scale = continuous_action_scale
        self.action_space = spaces.Box(low=continuous_actionspace[0], high=continuous_actionspace[1], shape=(1, 3), dtype=np.float)#shape=(2, 1)
        self.total_reward = 0
        self.reward_setting=reward_setting
        self.trajdata_sort=trajdata_sort
        self.trajnum_ratio=trajnum_ratio
        self.baseline_mod=baseline_mod
        if self.trajdata_sort == 'sorted':
            self.tripIDnum = self.trajdata_range[0]
            # continuous action
        # self.action_space = spaces.Box(low=-1, high=1, dtype=np.float)

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = traj_len-self.pos_num
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
        # self.current_step = random.randint(0, len(self.df.loc[:, 'latDeg_norm'].values) - (self.pos_num-1))
        self.visible_sat=satnum_df.loc[satnum_df.loc[:,self.tripIDlist[self.tripIDnum]]>0,'Svid'].to_numpy()
        obs=self._next_observation()
        # must return in observation scale
        return obs#self.tripIDnum#, obs#, {}

    def _normalize_pos(self,state):
        state[0]=(state[0]-xecef_min) / (xecef_max - xecef_min)
        state[1]=(state[1]-yecef_min) / (yecef_max - yecef_min)
        state[2]=(state[2]-zecef_min) / (zecef_max - zecef_min)
        return state

    def _normalize_los(self,gnss):
        # gnss[:,0]=(gnss[:,0]-res_min) / (res_max - res_min)*2-1
        # gnss[:,1]=(gnss[:,1]-losx_min) / (losx_max - losx_min)*2-1
        # gnss[:,2]=(gnss[:,2]-losy_min) / (losy_max - losy_min)*2-1
        # gnss[:,3]=(gnss[:,3]-losz_min) / (losz_max - losz_min)*2-1
        gnss[:,1]=(gnss[:,1]) / max(res_max, np.abs(res_min))
        gnss[:,2]=(gnss[:,2]) / max(losx_max, np.abs(losx_min))
        gnss[:,3]=(gnss[:,3]) / max(losy_max, np.abs(losy_min))
        gnss[:,4]=(gnss[:,4]) / max(losz_max, np.abs(losz_min))
        return gnss

    def _next_observation(self):
        obs = np.array([
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num-2), 'X_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num-2), 'Y_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num-2), 'Z_RLpredict'].values
        ])
        if self.baseline_mod == 'bl':
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_bl']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_bl']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_bl']]],axis=1)
        elif self.baseline_mod == 'wls':
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_wls']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_wls']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_wls']]],axis=1)
        elif self.baseline_mod == 'kf':
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_kf']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_kf']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_kf']]],axis=1)
        elif self.baseline_mod == 'kf_igst':
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_kf_igst']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_kf_igst']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_kf_igst']]],axis=1)

        obs=self._normalize_pos(obs)
        # obs_f=self.losfeature[self.datatime[self.current_step + (self.pos_num-1)]]
        feature_tmp=self.losfeature[self.datatime[self.current_step + (self.pos_num-1)]]['features']
        # obs_feature = np.zeros([len(self.visible_sat), 4])
        feature_tmp = self._normalize_los(feature_tmp)
        obs_feature = np.zeros([(self.max_visible_sat), 4])
        for i in range(len(self.visible_sat)):
            if self.visible_sat[i] in feature_tmp[:,0]:
                obs_feature[i,:]=feature_tmp[feature_tmp[:,0]==self.visible_sat[i],1:]

        # obs_feature = np.array([np.where(self.visible_sat[i] in feature_tmp[:,0],feature_tmp[feature_tmp[:,0]==self.visible_sat[i],1:]
        #                         ,np.zeros_like(feature_tmp[0,1:])) for i in range(len(self.visible_sat))])
        # obs_all=np.concatenate((obs.reshape(1,3*self.pos_num, order='F'),
        #                        obs_feature.reshape(1,4*self.max_visible_sat,order='C')), axis=1)
        obs_all={'pos':obs.reshape(1,3*self.pos_num, order='F'), 'gnss':obs_feature.reshape(1,4*self.max_visible_sat,order='C')}
        # obs_all = self._normalize(obs_all)

        # obs = obs.reshape(-1, 1) # + (self.pos_num-1)  + (self.pos_num-1)
        # obs=np.array([self.baseline.loc[self.current_step, 'LatitudeDegrees_norm'],self.baseline.loc[self.current_step, 'LongitudeDegrees_norm']])
        # obs=obs.reshape(2,1)
        # TODO latDeg lngDeg ... latDeg lngDeg
        return obs_all

    def step(self, action):
        # judge if end #
        done=(self.current_step >= len(self.baseline.loc[:, 'UnixTimeMillis'].values)*self.trajnum_ratio - (self.pos_num) -outlayer_in_end_ecef)
        timestep=self.baseline.loc[self.current_step + (self.pos_num-1), 'UnixTimeMillis']
        # action for new prediction
        action=np.reshape(action,[1,3])
        predict_x = action[0,0]*self.continuous_action_scale
        predict_y = action[0,1]*self.continuous_action_scale
        predict_z = action[0,2]*self.continuous_action_scale
        if self.baseline_mod == 'bl':
            obs_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_bl']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_bl']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_bl']
        elif self.baseline_mod == 'wls':
            obs_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_wls']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_wls']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_wls']
        elif self.baseline_mod == 'kf':
            obs_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_kf']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_kf']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_kf']
        elif self.baseline_mod == 'kf_igst':
            obs_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_kf_igst']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_kf_igst']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_kf_igst']
        gro_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'ecefX']
        gro_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'ecefY']
        gro_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ecefZ']
        rl_x = obs_x + predict_x
        rl_y = obs_y + predict_y
        rl_z = obs_z + predict_z
        self.baseline.loc[self.current_step + (self.pos_num-1), ['X_RLpredict']] = rl_x
        self.baseline.loc[self.current_step + (self.pos_num-1), ['Y_RLpredict']] = rl_y
        self.baseline.loc[self.current_step + (self.pos_num-1), ['Z_RLpredict']] = rl_z
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['X_RLpredict']] = rl_x
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['Y_RLpredict']] = rl_y
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['Z_RLpredict']] = rl_z
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

class GPSPosition_continuous_lospos(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,trajdata_range, traj_type, continuous_action_scale, continuous_actionspace, reward_setting, trajdata_sort, baseline_mod):
    # def __init__(self,trajdata_range, action_scale, discrete_actionspace, reward_setting, trajdata_sort, baseline_mod):
        super(GPSPosition_continuous_lospos, self).__init__()
        self.max_visible_sat=13
        self.pos_num=50
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(self.max_visible_sat, 4), dtype=np.float)#shape=(2, 1)
        self.observation_space = spaces.Dict({'gnss':spaces.Box(low=-1, high=1, shape=(self.max_visible_sat, 4)),
                                              'pos':spaces.Box(low=0, high=1, shape=(self.pos_num, 3), dtype=np.float)})
        if traj_type=='highway':
            self.tripIDlist=traj_highway
        elif traj_type=='full':
            self.tripIDlist=tripIDlist_full
        elif traj_type=='urban':
            self.tripIDlist = traj_else

        # continuous action
        if trajdata_range=='full':
            self.trajdata_range = [0, len(self.tripIDlist)-1]
        else:
            self.trajdata_range = trajdata_range
        self.continuous_actionspace = continuous_actionspace
        self.continuous_action_scale = continuous_action_scale
        self.action_space = spaces.Box(low=continuous_actionspace[0], high=continuous_actionspace[1], shape=(1, 3), dtype=np.float)#shape=(2, 1)
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
        obs=self._next_observation()
        # must return in observation scale
        return obs#self.tripIDnum#, obs#, {}

    def _normalize_pos(self,state):
        state[0]=(state[0]-xecef_min) / (xecef_max - xecef_min)
        state[1]=(state[1]-yecef_min) / (yecef_max - yecef_min)
        state[2]=(state[2]-zecef_min) / (zecef_max - zecef_min)
        return state

    def _normalize_los(self,gnss):
        # gnss[:,0]=(gnss[:,0]-res_min) / (res_max - res_min)*2-1
        # gnss[:,1]=(gnss[:,1]-losx_min) / (losx_max - losx_min)*2-1
        # gnss[:,2]=(gnss[:,2]-losy_min) / (losy_max - losy_min)*2-1
        # gnss[:,3]=(gnss[:,3]-losz_min) / (losz_max - losz_min)*2-1
        gnss[:,1]=(gnss[:,1]) / max(res_max, np.abs(res_min))
        gnss[:,2]=(gnss[:,2]) / max(losx_max, np.abs(losx_min))
        gnss[:,3]=(gnss[:,3]) / max(losy_max, np.abs(losy_min))
        gnss[:,4]=(gnss[:,4]) / max(losz_max, np.abs(losz_min))
        return gnss

    def _next_observation(self):
        obs = np.array([
            self.baseline.loc[self.current_step: self.current_step + (traj_len-2), 'X_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (traj_len-2), 'Y_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (traj_len-2), 'Z_RLpredict'].values
        ])
        if self.baseline_mod == 'bl':
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (traj_len-1), 'XEcefMeters_bl']],
                                 [self.baseline.loc[self.current_step + (traj_len-1), 'YEcefMeters_bl']],
                                 [self.baseline.loc[self.current_step + (traj_len-1), 'ZEcefMeters_bl']]],axis=1)
        elif self.baseline_mod == 'wls':
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (traj_len-1), 'XEcefMeters_wls']],
                                 [self.baseline.loc[self.current_step + (traj_len-1), 'YEcefMeters_wls']],
                                 [self.baseline.loc[self.current_step + (traj_len-1), 'ZEcefMeters_wls']]],axis=1)
        elif self.baseline_mod == 'kf':
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (traj_len-1), 'XEcefMeters_kf']],
                                 [self.baseline.loc[self.current_step + (traj_len-1), 'YEcefMeters_kf']],
                                 [self.baseline.loc[self.current_step + (traj_len-1), 'ZEcefMeters_kf']]],axis=1)
        elif self.baseline_mod == 'kf_igst':
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (traj_len-1), 'XEcefMeters_kf_igst']],
                                 [self.baseline.loc[self.current_step + (traj_len-1), 'YEcefMeters_kf_igst']],
                                 [self.baseline.loc[self.current_step + (traj_len-1), 'ZEcefMeters_kf_igst']]],axis=1)

        obs=self._normalize_pos(obs)
        # obs_f=self.losfeature[self.datatime[self.current_step + (traj_len-1)]]
        feature_tmp=self.losfeature[self.datatime[self.current_step + (traj_len-1)]]['features']
        # obs_feature = np.zeros([len(self.visible_sat), 4])
        feature_tmp = self._normalize_los(feature_tmp)
        obs_feature = np.zeros([(self.max_visible_sat), 4])
        for i in range(len(self.visible_sat)):
            if self.visible_sat[i] in feature_tmp[:,0]:
                obs_feature[i,:]=feature_tmp[feature_tmp[:,0]==self.visible_sat[i],1:]

        # obs_feature = np.array([np.where(self.visible_sat[i] in feature_tmp[:,0],feature_tmp[feature_tmp[:,0]==self.visible_sat[i],1:]
        #                         ,np.zeros_like(feature_tmp[0,1:])) for i in range(len(self.visible_sat))])
        obs_all={'pos':obs, 'gnss':obs_feature}
        # obs_all = self._normalize(obs_all)

        # obs = obs.reshape(-1, 1) # + (traj_len-1)  + (traj_len-1)
        # obs=np.array([self.baseline.loc[self.current_step, 'LatitudeDegrees_norm'],self.baseline.loc[self.current_step, 'LongitudeDegrees_norm']])
        # obs=obs.reshape(2,1)
        # TODO latDeg lngDeg ... latDeg lngDeg
        return obs_all

    def step(self, action):
        # judge if end #
        done=(self.current_step >= len(self.baseline.loc[:, 'UnixTimeMillis'].values) - (traj_len) -outlayer_in_end_ecef)
        timestep=self.baseline.loc[self.current_step + (traj_len-1), 'UnixTimeMillis']
        # action for new prediction
        action=np.reshape(action,[1,3])
        predict_x = action[0,0]*self.continuous_action_scale
        predict_y = action[0,1]*self.continuous_action_scale
        predict_z = action[0,2]*self.continuous_action_scale
        if self.baseline_mod == 'bl':
            obs_x = self.baseline.loc[self.current_step + (traj_len-1), 'XEcefMeters_bl']
            obs_y = self.baseline.loc[self.current_step + (traj_len-1), 'YEcefMeters_bl']
            obs_z = self.baseline.loc[self.current_step + (traj_len-1), 'ZEcefMeters_bl']
        elif self.baseline_mod == 'wls':
            obs_x = self.baseline.loc[self.current_step + (traj_len-1), 'XEcefMeters_wls']
            obs_y = self.baseline.loc[self.current_step + (traj_len-1), 'YEcefMeters_wls']
            obs_z = self.baseline.loc[self.current_step + (traj_len-1), 'ZEcefMeters_wls']
        elif self.baseline_mod == 'kf':
            obs_x = self.baseline.loc[self.current_step + (traj_len-1), 'XEcefMeters_kf']
            obs_y = self.baseline.loc[self.current_step + (traj_len-1), 'YEcefMeters_kf']
            obs_z = self.baseline.loc[self.current_step + (traj_len-1), 'ZEcefMeters_kf']
        elif self.baseline_mod == 'kf_igst':
            obs_x = self.baseline.loc[self.current_step + (traj_len-1), 'XEcefMeters_kf_igst']
            obs_y = self.baseline.loc[self.current_step + (traj_len-1), 'YEcefMeters_kf_igst']
            obs_z = self.baseline.loc[self.current_step + (traj_len-1), 'ZEcefMeters_kf_igst']
        gro_x = self.baseline.loc[self.current_step + (traj_len-1), 'ecefX']
        gro_y = self.baseline.loc[self.current_step + (traj_len-1), 'ecefY']
        gro_z = self.baseline.loc[self.current_step + (traj_len-1), 'ecefZ']
        rl_x = obs_x + predict_x
        rl_y = obs_y + predict_y
        rl_z = obs_z + predict_z
        self.baseline.loc[self.current_step + (traj_len-1), ['X_RLpredict']] = rl_x
        self.baseline.loc[self.current_step + (traj_len-1), ['Y_RLpredict']] = rl_y
        self.baseline.loc[self.current_step + (traj_len-1), ['Z_RLpredict']] = rl_z
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (traj_len-1), ['X_RLpredict']] = rl_x
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (traj_len-1), ['Y_RLpredict']] = rl_y
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (traj_len-1), ['Z_RLpredict']] = rl_z
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

class GPSPosition_continuous_los(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,trajdata_range, traj_type, continuous_action_scale, continuous_actionspace, reward_setting, trajdata_sort, baseline_mod):
    # def __init__(self,trajdata_range, action_scale, discrete_actionspace, reward_setting, trajdata_sort, baseline_mod):
        super(GPSPosition_continuous_los, self).__init__()
        self.max_visible_sat=13
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.max_visible_sat, 4), dtype=np.float)#shape=(2, 1)
        if traj_type=='highway':
            self.tripIDlist=traj_highway
        elif traj_type=='full':
            self.tripIDlist=tripIDlist_full
        elif traj_type=='urban':
            self.tripIDlist = traj_else
        elif traj_type=='losangel':
            self.tripIDlist = traj_losangel
        elif traj_type=='bayarea':
            self.tripIDlist = traj_bayarea

        # continuous action
        if trajdata_range=='full':
            self.trajdata_range = [0, len(self.tripIDlist)-1]
        else:
            self.trajdata_range = trajdata_range
        self.continuous_actionspace = continuous_actionspace
        self.continuous_action_scale = continuous_action_scale
        self.action_space = spaces.Box(low=continuous_actionspace[0], high=continuous_actionspace[1], shape=(3, 1), dtype=np.float)#shape=(2, 1)
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
        obs=self._next_observation()
        # must return in observation scale
        return obs#self.tripIDnum#, obs#, {}

    def _normalize(self,gnss):
        # gnss[:,0]=(gnss[:,0]-res_min) / (res_max - res_min)*2-1
        # gnss[:,1]=(gnss[:,1]-losx_min) / (losx_max - losx_min)*2-1
        # gnss[:,2]=(gnss[:,2]-losy_min) / (losy_max - losy_min)*2-1
        # gnss[:,3]=(gnss[:,3]-losz_min) / (losz_max - losz_min)*2-1
        gnss[:,1]=(gnss[:,1]) / max(res_max, np.abs(res_min))
        gnss[:,2]=(gnss[:,2]) / max(losx_max, np.abs(losx_min))
        gnss[:,3]=(gnss[:,3]) / max(losy_max, np.abs(losy_min))
        gnss[:,4]=(gnss[:,4]) / max(losz_max, np.abs(losz_min))
        return gnss

    def _next_observation(self):
        # obs_f=self.losfeature[self.datatime[self.current_step + (traj_len-1)]]
        feature_tmp=self.losfeature[self.datatime[self.current_step + (traj_len-1)]]['features']
        # obs_feature = np.zeros([len(self.visible_sat), 4])
        feature_tmp = self._normalize(feature_tmp)
        obs_feature = np.zeros([(self.max_visible_sat), 4])
        for i in range(len(self.visible_sat)):
            if self.visible_sat[i] in feature_tmp[:,0]:
                obs_feature[i,:]=feature_tmp[feature_tmp[:,0]==self.visible_sat[i],1:]

        return obs_feature

    def step(self, action):
        # judge if end #
        done=(self.current_step >= len(self.baseline.loc[:, 'UnixTimeMillis'].values) - (traj_len) -outlayer_in_end_ecef)
        timestep=self.baseline.loc[self.current_step + (traj_len-1), 'UnixTimeMillis']
        # action for new prediction
        predict_x = action[0]*self.continuous_action_scale
        predict_y = action[1]*self.continuous_action_scale
        predict_z = action[2]*self.continuous_action_scale
        if self.baseline_mod == 'bl':
            obs_x = self.baseline.loc[self.current_step + (traj_len-1), 'XEcefMeters_bl']
            obs_y = self.baseline.loc[self.current_step + (traj_len-1), 'YEcefMeters_bl']
            obs_z = self.baseline.loc[self.current_step + (traj_len-1), 'ZEcefMeters_bl']
        elif self.baseline_mod == 'wls':
            obs_x = self.baseline.loc[self.current_step + (traj_len-1), 'XEcefMeters_wls']
            obs_y = self.baseline.loc[self.current_step + (traj_len-1), 'YEcefMeters_wls']
            obs_z = self.baseline.loc[self.current_step + (traj_len-1), 'ZEcefMeters_wls']
        elif self.baseline_mod == 'kf':
            obs_x = self.baseline.loc[self.current_step + (traj_len-1), 'XEcefMeters_kf']
            obs_y = self.baseline.loc[self.current_step + (traj_len-1), 'YEcefMeters_kf']
            obs_z = self.baseline.loc[self.current_step + (traj_len-1), 'ZEcefMeters_kf']
        elif self.baseline_mod == 'kf_igst':
            obs_x = self.baseline.loc[self.current_step + (traj_len-1), 'XEcefMeters_kf_igst']
            obs_y = self.baseline.loc[self.current_step + (traj_len-1), 'YEcefMeters_kf_igst']
            obs_z = self.baseline.loc[self.current_step + (traj_len-1), 'ZEcefMeters_kf_igst']
        gro_x = self.baseline.loc[self.current_step + (traj_len-1), 'ecefX']
        gro_y = self.baseline.loc[self.current_step + (traj_len-1), 'ecefY']
        gro_z = self.baseline.loc[self.current_step + (traj_len-1), 'ecefZ']
        rl_x = obs_x + predict_x
        rl_y = obs_y + predict_y
        rl_z = obs_z + predict_z
        self.baseline.loc[self.current_step + (traj_len-1), ['X_RLpredict']] = rl_x
        self.baseline.loc[self.current_step + (traj_len-1), ['Y_RLpredict']] = rl_y
        self.baseline.loc[self.current_step + (traj_len-1), ['Z_RLpredict']] = rl_z
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (traj_len-1), ['X_RLpredict']] = rl_x
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (traj_len-1), ['Y_RLpredict']] = rl_y
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (traj_len-1), ['Z_RLpredict']] = rl_z
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

class GPSPosition_continuous_losposcat(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,trajdata_range, traj_type, continuous_action_scale, continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, pos_num):
    # def __init__(self,trajdata_range, action_scale, discrete_actionspace, reward_setting, trajdata_sort, baseline_mod):
        super(GPSPosition_continuous_losposcat, self).__init__()
        self.max_visible_sat=13
        self.pos_num=pos_num #50
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1, 3*self.pos_num+self.max_visible_sat*4), dtype=np.float)#shape=(2, 1)
        # self.observation_space = spaces.Dict({'gnss':spaces.Box(low=-1, high=1, shape=(self.max_visible_sat, 4)),
        #                                       'pos':spaces.Box(low=0, high=1, shape=(3, self.pos_num), dtype=np.float)})
        if traj_type=='highway':
            self.tripIDlist=traj_highway
        elif traj_type=='full':
            self.tripIDlist=tripIDlist_full
        elif traj_type=='urban':
            self.tripIDlist = traj_else
        elif traj_type=='losangel':
            self.tripIDlist = traj_losangel
        elif traj_type=='bayarea':
            self.tripIDlist = traj_bayarea

        # continuous action
        if trajdata_range=='full':
            self.trajdata_range = [0, len(self.tripIDlist)-1]
        else:
            self.trajdata_range = trajdata_range
        self.continuous_actionspace = continuous_actionspace
        self.continuous_action_scale = continuous_action_scale
        self.action_space = spaces.Box(low=continuous_actionspace[0], high=continuous_actionspace[1], shape=(1, 3), dtype=np.float)#shape=(2, 1)
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
        # self.current_step = random.randint(0, len(self.df.loc[:, 'latDeg_norm'].values) - (self.pos_num-1))
        self.visible_sat=satnum_df.loc[satnum_df.loc[:,self.tripIDlist[self.tripIDnum]]>0,'Svid'].to_numpy()
        obs=self._next_observation()
        # must return in observation scale
        return obs#self.tripIDnum#, obs#, {}

    def _normalize_pos(self,state):
        state[0]=(state[0]-xecef_min) / (xecef_max - xecef_min)
        state[1]=(state[1]-yecef_min) / (yecef_max - yecef_min)
        state[2]=(state[2]-zecef_min) / (zecef_max - zecef_min)
        return state

    def _normalize_los(self,gnss):
        # gnss[:,0]=(gnss[:,0]-res_min) / (res_max - res_min)*2-1
        # gnss[:,1]=(gnss[:,1]-losx_min) / (losx_max - losx_min)*2-1
        # gnss[:,2]=(gnss[:,2]-losy_min) / (losy_max - losy_min)*2-1
        # gnss[:,3]=(gnss[:,3]-losz_min) / (losz_max - losz_min)*2-1
        gnss[:,1]=(gnss[:,1]) / max(res_max, np.abs(res_min))
        gnss[:,2]=(gnss[:,2]) / max(losx_max, np.abs(losx_min))
        gnss[:,3]=(gnss[:,3]) / max(losy_max, np.abs(losy_min))
        gnss[:,4]=(gnss[:,4]) / max(losz_max, np.abs(losz_min))
        return gnss

    def _next_observation(self):
        obs = np.array([
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num-2), 'X_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num-2), 'Y_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num-2), 'Z_RLpredict'].values
        ])
        if self.baseline_mod == 'bl':
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_bl']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_bl']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_bl']]],axis=1)
        elif self.baseline_mod == 'wls':
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_wls']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_wls']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_wls']]],axis=1)
        elif self.baseline_mod == 'kf':
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_kf']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_kf']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_kf']]],axis=1)
        elif self.baseline_mod == 'kf_igst':
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_kf_igst']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_kf_igst']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_kf_igst']]],axis=1)

        obs=self._normalize_pos(obs)
        # obs_f=self.losfeature[self.datatime[self.current_step + (self.pos_num-1)]]
        feature_tmp=self.losfeature[self.datatime[self.current_step + (self.pos_num-1)]]['features']
        # obs_feature = np.zeros([len(self.visible_sat), 4])
        feature_tmp = self._normalize_los(feature_tmp)
        obs_feature = np.zeros([(self.max_visible_sat), 4])
        for i in range(len(self.visible_sat)):
            if self.visible_sat[i] in feature_tmp[:,0]:
                obs_feature[i,:]=feature_tmp[feature_tmp[:,0]==self.visible_sat[i],1:]

        # obs_feature = np.array([np.where(self.visible_sat[i] in feature_tmp[:,0],feature_tmp[feature_tmp[:,0]==self.visible_sat[i],1:]
        #                         ,np.zeros_like(feature_tmp[0,1:])) for i in range(len(self.visible_sat))])
        obs_all=np.concatenate((obs.reshape(1,3*self.pos_num, order='F'),
                               obs_feature.reshape(1,4*self.max_visible_sat,order='C')), axis=1)
        # obs_all={'pos':obs, 'gnss':obs_feature}
        # obs_all = self._normalize(obs_all)

        # obs = obs.reshape(-1, 1) # + (self.pos_num-1)  + (self.pos_num-1)
        # obs=np.array([self.baseline.loc[self.current_step, 'LatitudeDegrees_norm'],self.baseline.loc[self.current_step, 'LongitudeDegrees_norm']])
        # obs=obs.reshape(2,1)
        # TODO latDeg lngDeg ... latDeg lngDeg
        return obs_all

    def step(self, action):
        # judge if end #
        done=(self.current_step >= len(self.baseline.loc[:, 'UnixTimeMillis'].values) - (self.pos_num) -outlayer_in_end_ecef)
        timestep=self.baseline.loc[self.current_step + (self.pos_num-1), 'UnixTimeMillis']
        # action for new prediction
        action=np.reshape(action,[1,3])
        predict_x = action[0,0]*self.continuous_action_scale
        predict_y = action[0,1]*self.continuous_action_scale
        predict_z = action[0,2]*self.continuous_action_scale
        if self.baseline_mod == 'bl':
            obs_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_bl']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_bl']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_bl']
        elif self.baseline_mod == 'wls':
            obs_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_wls']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_wls']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_wls']
        elif self.baseline_mod == 'kf':
            obs_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_kf']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_kf']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_kf']
        elif self.baseline_mod == 'kf_igst':
            obs_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_kf_igst']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_kf_igst']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_kf_igst']
        gro_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'ecefX']
        gro_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'ecefY']
        gro_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ecefZ']
        rl_x = obs_x + predict_x
        rl_y = obs_y + predict_y
        rl_z = obs_z + predict_z
        self.baseline.loc[self.current_step + (self.pos_num-1), ['X_RLpredict']] = rl_x
        self.baseline.loc[self.current_step + (self.pos_num-1), ['Y_RLpredict']] = rl_y
        self.baseline.loc[self.current_step + (self.pos_num-1), ['Z_RLpredict']] = rl_z
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['X_RLpredict']] = rl_x
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['Y_RLpredict']] = rl_y
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['Z_RLpredict']] = rl_z
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
