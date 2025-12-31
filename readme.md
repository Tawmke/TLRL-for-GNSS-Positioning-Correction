---
author: Jianhao Tang
date: 2025-02-08
changes:
  - "2025-02-08: Jianhao Tang  初始文档创建"
---
# 强化学习卫星定位修正代码说明文档（GSDC dataset）

## 前提说明
***实验室资料使用严禁外传！！***

 如有个别文件和代码需要或者缺失，可以在服务器`/mnt/sdb/home/tangjh`或`/mnt/sdb/home/dingweizu/lihui/MAML`文件夹找一下或者问问师兄。

## 代码结构
```
project-root/
├── README.md     # 本文件
├── plot/         # 画图代码文件夹
├── env/          # 环境文件夹，包括数据处理文件和代码
├── model/        # 模型文件夹
├── funcs/        # 函数库文件夹
├── sb3_custom/   # stable_baseline3库的一些函数修改
├── train/        # 原始下载数据(请在服务器下载)
├── ...
carrier-smoothing-robust-wls-kalman-smoother_single.py
rl_control_custom_lospos.py
rl_control_custom_lospos_diff_area.py
testonly_gsdc_LOSLLAFT_diffarea.py
rl_control_graphlos_compactmultic.py
```
基础模型有如下示例版本，其他的改进方法请询问对应的作者。
1) [Multi-LSTMPPO（rl_control_custom_lospos.py）](./rl_control_custom_lospos.py)
   > 参考文献 《Fusing Vehicle Trajectories and GNSS  Measurements to improve GNSS Positioning
Correction based on Actor-Critic Learning》，仅使用GNSS特征和位置序列作为模型观测。
2) [AWRL（rl_control_custom_lospos_diff_area.py）](./rl_control_custom_lospos_diff_area.py)
   > 参考文献 《Improving performances of GNSS positioning correction using 
multiview deep reinforcement learning with sparse representation》,《Deep Reinforcement Learning with Robust Augmented Reward Sequence Prediction for Improving GNSS Positioning》，在上面结构的基础上的升级，加上注意力模块学习不同观测的权重，
   > 是后期论文的常用结构。此外，该文件使用多场景的数据切分进行实验。
3) [WLS+KF（carrier-smoothing-robust-wls-kalman-smoother_single.py）](./carrier-smoothing-robust-wls-kalman-smoother_single.py)
   > 参考谷歌Taro的WLS+KF代码，需要原始数据集。
4) [GNNRL（rl_control_graphlos_compactmultic.py）](rl_control_graphlos_compactmultic.py)
   > 参考论文《Efficient Graph Neural Network Driven Recurrent Reinforcement Learning for GNSS Position Correction》，图强化学习代码，特征提取器使用GNN，使用多系统的卫星特征，
   > 以及边矩阵，后续用得较少。

## 一、主函数代码使用说明
***以AWRL的代码rl_control_custom_lospos_diff_area.py为例进行说明，其他是大同小异。不需要拘泥于固定的形式，环境、模型的设置是可以自由调节的。***
### 1）参数设置
不同代码的参数命名可能不相同，但是道理是相似的，下面只介绍一些关键的参数。
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 使用服务器的哪张卡，不使用服务器默认使用cpu
training_stepnum = 16000 # 训练的迭代步数，可根据情况调整
learning_rate_list = [1e-3] # 模型学习率，1e-4附近较佳
postraj_num_list = [10] # pos序列的长度
traj_type_target_train=[0,0.7] # 使用每条轨迹的百分之多少进行训练，此例为使用前70%的数据进行训练
traj_type_target_test=[0.7,1] # 使用每条轨迹的百分之多少进行测试，此例为使用后30%的数据进行测试
triptype = 'urban2' # 用于训练的环境标签，此例为使用urban2数据进行训练
baseline_mod='kf_igst' # baseline方法
max_action=100 # 最大动作的范围，默认为100不调整
continuous_action_scale=20e-1 # 动作的尺度，需要根据不同数据集或模型调整
envmod = 'losllAcat' # 选择环境，默认为连续的gnss+pos的环境，可以根据需求换成离散的
networkmod = 'continuous_lstmATF1' # 选择模型，根据需求更改
running_date # 数据保存的主文件夹，根据情况设置
tripIDlist # 从env文件夹导入需要进行实验的轨迹名称的list
trajdata_range = [0, len(tripIDlist) - 1] # 用于构建（训练/测试）环境的轨迹的索引范围，默认使用全部轨迹
```

### 2）训练流程说明
1. 确定实验结果的存储路径。其中变量dir_path为***基础路径***，在环境代码[GSDC_2022_LOSPOS_diff_area](./env/GSDC_2022_LOSPOS_diff_area.py)中
设置为使用者当前项目的路径，如`dir_path = '/mnt/sdb/home/tangjh/smartphone-decimeter-2022/'`。（建议在存储的文件夹命名中添加重要的调参信息，方便消融实验查看）
```python
tensorboard_log = f'{dir_path}records_values/robustRL_{running_date}/source={triptype}/{networkmod}_{traj_type_target_train[1]}_{triptype}_{envmod}_{baseline_mod}_lr={learning_rate}_pos={posnum}'
```
2. 构建训练环境。当前使用最基本的以LOS向量+伪距残差（以下简称为LOS）和位置序列（简称为POS）作观测的环境
 ```python
env = DummyVecEnv([lambda: GPSPosition_continuous_lospos(trajdata_range, traj_type_target_train, triptype, continuous_action_scale, continuous_actionspace,
                         reward_setting,trajdata_sort,baseline_mod, posnum)])
 ```
3. 定义特征提取器。此处[CustomATF1](./model/model_ATF.py)使用注意力模块，如果是最基本的LSTMPPO模型此处可以忽略
 ```python
policy_kwargs = dict(features_extractor_class=CustomATF1,
                     features_extractor_kwargs=dict(features_dim=features_dim),ATF_trig=networkmod)
 ```
4. 定义DRL模型和训练。此处使用基本的开源stable_baseline3中的模型结构和训练方式，但稍微有所更改。模型文件位于model文件夹。
 ```python
model = RecurrentPPO("MlpLstmPolicy", env, verbose=2, policy_kwargs=policy_kwargs, 
                     tensorboard_log=tensorboard_log, learning_rate=learning_rate)
model.learn(total_timesteps=training_stepnum, eval_log_path=tensorboard_log)
 ```
5. 模型保存和数据记录。
 ```python
model.save(model.logger.dir+f"/{networkmod}_{reward_setting}_action{continuous_actionspace[0]}_{continuous_actionspace[1]}"
        f"_{continuous_action_scale:0.1e}_trainingnum{training_stepnum:0.1e}"
         f"_env_{baseline_mod}{envmod}range{trajdata_range[0]}_{trajdata_range[-1]}{trajdata_sort}_lr{learning_rate:0.1e}")
recording_results_ecef(data_truth_dic,trajdata_range,tripIDlist,logdirname,baseline_mod,traj_record=True)
 ```
`model.save()`函数会保存模型参数到对应的文件夹，一般命名为很长一串，方便后续重新导入（下面会说）。

在[recording_results_ecef](./funcs/utilis.py)函数中记录了训练或测试数据的统计结果，当`traj_record=True`会保存DRL预测的轨迹坐标的csv文件（..._rl_traj_....csv）。
最终会根据训练或测试的预测数据生成四个主要结果文件，例：
1) train_or_distances.csv / testmore_or_distances.csv：记录作为baseline的基于模型方法的平均定位结果，
2) train_rl_distances.csv / testmore_rl_distances.csv：记录使用的DRL方法的平均定位结果，
3) train_errors.csv / testmore_errors.csv：记录RL平均误差-baseline平均误差，
4) train_xyz_distances.csv / testmore_xyz_distances.csv：记录baseline和RL方法在不同方向上的平均误差结果。

### 3）测试流程说明
1. 构建环境并进行测试。注意，在测试阶段模型不进行参数更新，并且LSTM的隐藏状态输入设置为全0（在model.predict函数文件可以查看）。
 ```python
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
 ```
2. 数据统计和保存。同上。
 ```python
logdirname=model.logger.dir + f'/testmore_{testtype}_'
recording_results_ecef(data_truth_dic,[test_trajlist[0],test_trajlist[-1]],
                       tripIDlist_test,logdirname,baseline_mod,traj_record=True)
 ```


## 二、环境使用说明
***以代码”rl_control_custom_lospos_diff_area.py“使用的强化学习环境[GPSPosition_continuous_lospos](./env/GSDC_2022_LOSPOS_diff_area.py)为例进行说明***
### 1）预处理文件导入
预处理的数据通过代码诸如[env_processing_gpsl1.py](./env/env_processing_gpsl1.py)等文件使用原始数据进行处理，如果项目已经有处理好的数据，则不需要再重新生成。
1. 导入位置坐标的文件
 ```python
with open(dir_path+'env/raw_baseline_gpsl1.pkl', "rb") as file:
    data_truth_dic = pickle.load(file)
file.close()
 ```
2. 导入处理后的gnss特征数据（这里只有los和pr两种特征）
 ```python
with open(dir_path + 'env/processed_features_gpsl1_ecef_id.pkl', "rb") as file:
    losfeature = pickle.load(file)
file.close()
 ```
3. 导入卫星统计数据（后续有一些代码这个文件没再用了）
 ```python
satnum_df = pd.read_csv(f'{dir_path}/env/raw_satnum_gpsl1.csv')
 ```
4. 导入所有轨迹的名称，并根据某些定义划分不同的类型，此例把相同地点（Area）采集的轨迹归为一类，从而构建一个强化学习环境
 ```python
traj_sum_df = pd.read_csv(f'{dir_path}/env/traj_summary_differtype.csv')
trip_lists = {'open1': traj_sum_df.loc[(traj_sum_df['Area']==1)]['tripId'].values.tolist(),
    'open2': traj_sum_df.loc[(traj_sum_df['Area']==5)]['tripId'].values.tolist(),
    'open3': traj_sum_df.loc[(traj_sum_df['Area']==20)]['tripId'].values.tolist(),
    'semi1': traj_sum_df.loc[(traj_sum_df['Area']==15)]['tripId'].values.tolist(),
    'semi2': traj_sum_df.loc[(traj_sum_df['Area']==18)]['tripId'].values.tolist(),
    'semi3': traj_sum_df.loc[(traj_sum_df['Area']==21)]['tripId'].values.tolist(),
    'urban1': traj_sum_df.loc[(traj_sum_df['Area']==2)]['tripId'].values.tolist(),
    'urban2': traj_sum_df.loc[(traj_sum_df['Area']==3)]['tripId'].values.tolist(),
    'urban3': traj_sum_df.loc[(traj_sum_df['Area']==4)]['tripId'].values.tolist(),
    'urban4': traj_sum_df.loc[(traj_sum_df['Area']==93)]['tripId'].values.tolist(),
    'source': traj_sum_df.loc[(traj_sum_df['Area']==19)|(traj_sum_df['Area']==17)|(traj_sum_df['Area']==94)]['tripId'].values.tolist()}
 ```
### 2）环境类的构建
以环境类”GPSPosition_continuous_lospos“为例进行说明，源代码已写得比较清晰，这里只提一些注意事项
1. 初始化环境类。包括定义观测空间（observation_space）为字典形式，连续动作空间的维度等
 ```python
def __init__(self,trajdata_range, traj_type, triptype, continuous_action_scale, continuous_actionspace,
                 reward_setting, trajdata_sort, baseline_mod, traj_len, excludenum=0):
    ...
    self.observation_space = spaces.Dict({'gnss':spaces.Box(low=-1, high=1, shape=(1, self.max_visible_sat * 4)),
                                              'pos':spaces.Box(low=0, high=1, shape=(1, 3 * self.pos_num), dtype=np.float)})
    self.action_space = spaces.Box(low=continuous_actionspace[0], high=continuous_actionspace[1], shape=(1, 3), dtype=np.float)
 ```
2. 输入观测构建。根据不同的baseline的设置，使用不同的基准方法生成的原始坐标（谷歌数据集一般使用kf_igst或wls），并构建POS序列。
 ```python
def _next_observation(self):
   ...
   if self.baseline_mod == 'bl':
       obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_bl']],
                            [self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_bl']],
                            [self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_bl']]],axis=1)
   elif self.baseline_mod == 'wls':
       obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_wls']],
                            [self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_wls']],
                            [self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_wls']]],axis=1)
   elif self.baseline_mod == 'kf_igst':
       obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_kf_igst']],
                            [self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_kf_igst']],
                            [self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_kf_igst']]],axis=1)
   obs=self._normalize_pos(obs)
 ```
构建gnss特征，两种特征都要使用***归一化***处理：
 ```python
feature_tmp=self.losfeature[self.datatime[self.current_step + (self.pos_num-1)]]['features']
feature_tmp = self._normalize_los(feature_tmp)
obs_feature = np.zeros([(self.max_visible_sat), 4])
for i in range(len(self.visible_sat)):
   if self.visible_sat[i] in feature_tmp[:,0]:
       obs_feature[i,:]=feature_tmp[feature_tmp[:,0]==self.visible_sat[i],1:]
 ```
3. 执行修正动作。注意，模型输出的动作需要乘上属性***self.continuous_action_scale***才是最终的位置修正！
 ```python
def step(self, action):
    predict_x = action[0,0]*self.continuous_action_scale 
    predict_y = action[0,1]*self.continuous_action_scale
    predict_z = action[0,2]*self.continuous_action_scale
 ```

## 三、强化学习模型文件使用说明
> 以代码”rl_control_custom_lospos_diff_area.py“使用的模型文件[ppo_recurrent_ATF1.py](./model/ppo_recurrent_ATF1.py)为例进行说明，模型主要参考stable_baseline3库中的模型结构（[https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)）。这里包含了模型的结构定义，
> 训练流程，模型导入等，比较复杂，有需要的同学需要花点心思去看。
1. 策略模型如下代码在函数”_setup_model(self)“中进行定义，该模型包括了actor和critic网络。
 ```python
self.policy = self.policy_class(
   self.observation_space,
   self.action_space,
   self.lr_schedule,
   use_sde=self.use_sde,
   **self.policy_kwargs,  # pytype:disable=not-instantiable
)
 ```
但是注意直接ctrl+单击不能跳转到对应的模型代码，如需要查看具体的模型结构和forward流程，需要先跳转下面代码中的`MlpLstmPolicy`字样；
 ```python
 policy_aliases: Dict[str, Type[BasePolicy]] = {
     "MlpLstmPolicy": MlpLstmPolicy,
     "CnnLstmPolicy": CnnLstmPolicy,
     "MultiInputLstmPolicy": MultiInputLstmPolicy,
 }
 ```
随后再跳转`RecurrentActorCriticPolicy`即可查看模型结构和forward等（如有不明白，可询问师兄）：

 ```python
def _init__():
   ...
    self.lstm_actor = nn.LSTM(self.features_dim,lstm_hidden_size,
      num_layers=n_lstm_layers, **self.lstm_kwargs,
                              )
def forward：
    ...
 ```
2. 环境交互数据采集。policy会先进行环境交互采集一批数据，存储到`self.rollout_buffer`属性中，当到达最大的容量 n_rollout_steps（默认设置为128步）停止采集并开始训练。
 ```python
def collect_rollouts:
   ...
    while n_steps < n_rollout_steps:
        actions, values, log_probs, lstm_states = self.policy.forward(obs_tensor, lstm_states, episode_starts)
        rollout_buffer.add()
 ```
3. 模型训练。从`self.rollout_buffer`中随机抽样batch数据进行训练，其中PPO（近端策略优化算法）的loss也在这里定义，
 ```python
def train(self) -> None:
   for epoch in range(self.n_epochs):
        for rollout_data in self.rollout_buffer.get(self.batch_size):
            ...
 ```
4. 模型导入。对于AWRL结构的模型，在重新导入保存的模型进行测试的时候直接使用`RecurrentPPO.load(model_filenamepath,env=env)`会报错，需要加入以下代码：
 ```python
def set_parameters(): ...
def load(): ...
 ```
## 四、模型导入测试
有时候需要重新将保存的模型导入然后进行测试（或训练），则使用代码如[testonly_gsdc_LOSLLAFT_diffarea.py](testonly_gsdc_LOSLLAFT_diffarea.py)进行导入：
 ```python
if networkmod in {'continuous_lstmATF1'}: # 带注意力模块的要使用这个
    model = RecurrentPPO("MlpLstmPolicy", env, policy_kwargs=policy_kwargs)
    model.policy.features_extractor.attention1.attwts.weight = torch.nn.Parameter(model.policy.features_extractor.attention1.attwts.weight.squeeze())
    model.policy.features_extractor.attention2.attwts.weight = torch.nn.Parameter(model.policy.features_extractor.attention2.attwts.weight.squeeze())
    model.load(model_filenamepath,env=env)
elif networkmod in {'continuous_lstm'}: # 不带注意力模块直接load就行
    model = RecurrentPPO.load(model_filenamepath,env=env)
 ```
其中，`model_filenamepath`为模型的保存的路径，模型导入后，再重新进行测试。

## 五、环境配置
本项目使用的环境要求不高，可使用[environment.yml](environment.yml)进行一步到位的配置，或者根据缺失的包一步一步进行配置，或者参考服务器85中dingweizu文件夹的环境配置。需要注意，stable_baseline3的版本只能是1.6.2，python版本为3.8.10。


## Reference
 ```
@article{tang2024improving,
  title={Improving GNSS positioning correction using deep reinforcement learning with an adaptive reward augmentation method},
  author={Tang, Jianhao and Li, Zhenni and Hou, Kexian and Li, Peili and Zhao, Haoli and Wang, Qianming and Liu, Ming and Xie, Shengli},
  journal={NAVIGATION: Journal of the Institute of Navigation},
  volume={71},
  number={4},
  year={2024},
  publisher={Institute of Navigation}
}
@article{zhao2024improving,
  title={Improving performances of GNSS positioning correction using multiview deep reinforcement learning with sparse representation},
  author={Zhao, Haoli and Li, Zhenni and Wang, Qianming and Xie, Kan and Xie, Shengli and Liu, Ming and Chen, Ci},
  journal={GPS Solutions},
  volume={28},
  number={3},
  pages={98},
  year={2024},
  publisher={Springer}
}
 ```

