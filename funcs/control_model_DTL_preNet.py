import numpy as np
import torch

from funcs.pretrain_model_preNet import *
import scipy.io
from env.GSDC_los_env import *
#from env.ca_env import *
from gym.envs.classic_control.cartpole import *
from funcs.utilis import *

# discrete action settings
discrete_actionspace = 31 # 默认18
action_scale = 2e-1 # scale in meters #默认5e-1
envmod='los'
reward_setting='RMSEadv'
traj_ratio = 1
trajdata_sort='sorted' # 'randint' 'sorted'
# baseline for RL: bl, wls, kf, kf_igst
baseline_mod='kf_igst' # baseline方法:kf_igst

traj_sum_df = pd.read_csv(f'{project_path}/env/traj_summary.csv')
tripIDlist_full=traj_sum_df['tripId'].values.tolist()
traj_highway=traj_sum_df.loc[traj_sum_df['Type']=='highway']['tripId'].values.tolist()
traj_else=traj_sum_df.loc[traj_sum_df['Type']=='urban']['tripId'].values.tolist()
traj_type_urban = 'urban'
traj_type_highway = 'highway'

# trajdata_range=[0,int(np.ceil(len(tripIDlist)*traj_ratio))]
trajdata_range_urban = [1,6]
trajdata_range_highway = [0,7]
trajdata_range_highway_2 = [0,7]
# Sensitivity analysis of noise
noisedB = None # add noise to observations
noise_sample = 5000 # 采集噪声样本量
noise_levels = [None,0,10,20,30,40] # [None,0,10,20,30,40]
noise_phi = []
for _ in range(len(noise_levels)+1):
    noise_phi.append([])

##
def sigma_compute(train_data, SNRdB=10):
    #SNRdB= 0
    # data_mean = [np.mean(train_data[0], axis=0), np.mean(train_data[1], axis=0)]
    # data_std = [np.std(train_data[0], axis=0), np.std(train_data[1], axis=0)]
    data_mean = np.mean(train_data, axis=0)
    data_std = np.std(train_data, axis=0)
    a_noise = np.sqrt(np.power(10, (SNRdB / 10)))
    SNRsigma = np.sqrt(np.power(data_std, 2) + np.power(data_mean, 2)) / a_noise

    # datasize = len(train_data)
    # noise = np.random.normal(0,SNRsigma,datasize)
    # noise2 = np.random.normal(0, 1, datasize) * SNRsigma
    # tmp1 = train_data+noise
    # SNRmeasure = 10 * np.log10(sum(np.power(train_data,2))/sum(np.power(noise,2)))
    # SNRmeasure2 = 10 * np.log10(sum(np.power(train_data, 2)) / sum(np.power(noise2, 2)))
    # print(SNRmeasure,SNRmeasure2)

    return SNRsigma

class ActionLearner_GSDC2022:
    def __init__(self, actionRandomness, decay, stepSize, preTrainTimes=None, maxIter=None, gamma=None, lossMode=None,
                 wordNums=None, betaB=None, betaW=None, betaFi=None,
                 lineSearchW=None, lineSearchB=None, thresholdW=None, thresholdB=None, thresholdFi=None,
                 thresholdLoss=None, lineAlphaW=None, lineBetaW=None,
                 lineAlphaB=None, lineBetaB=None, dataSource=None, randomness=None,
                 normalized=None, lipSafe=None, loadBasic=None, index=1,
                 outputFile=None, betaFi_control =None, reg_method =None, Model_folder=None, dataset=None,
                 preprocess=None, printtrig=None, Wcontrolmode=None, PreLfolders=[]):

        self.outputFile = outputFile
        if noisedB is not None:
            self.outputFile = self.outputFile + f'_db={noisedB}'
        self.logdirname = self.outputFile+ f'acspace={discrete_actionspace},sc={action_scale}' + '/train_'
        if not os.path.exists(self.outputFile+ f'acspace={discrete_actionspace},sc={action_scale}'):
            os.makedirs(self.outputFile+ f'acspace={discrete_actionspace},sc={action_scale}')
        print(self.logdirname)

        self.dataset = dataset
        self.basic = None
        self.gamma = None
        self.printtrig =printtrig
        self.Wcontrolmode=Wcontrolmode

        # filename = "./matlab_Model/" + loadBasic  + "/" + loadBasic + '_' + str(index) +'/'
        filename = '{}/{}/Run_{}/'.format(Model_folder, loadBasic, index)
        filename_net = '/mnt/sdb/home/tangjh/DLRL4GNSSpos/'
        self.basic = DLSRSolver_preNet(filename_net, filename, preprocess, dataset, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, skip=True)
        self.basic.B = []

        for preLfolder in PreLfolders:
            tempfilename = '{}/{}/Run_{}/'.format(Model_folder, preLfolder, index)
            self.basic.B.append(scipy.io.loadmat(tempfilename + 'B.mat')['B'])
        
        self.basic.B = scipy.io.loadmat(filename + 'B.mat')['B']
        self.basic.wordNums = self.basic.B[-1].shape[0]
        self.basic.w = scipy.io.loadmat(filename + 'W.mat')['W']
        self.basic.weightReconstruct = scipy.io.loadmat(filename + 'weightReconstruct.mat')[
            'weightReconstruct'].item()
        self.basic.gamma = scipy.io.loadmat(filename + 'gamma.mat')['gamma'].item()
        self.basic.lossMode = 'MeanSquaredReturnError'
        self.basic.weightPredict = scipy.io.loadmat(filename + 'weightPredict.mat')['weightPredict'].item()
        self.basic.maxIter = scipy.io.loadmat(filename + 'maxIter.mat')['maxIter'].item()
        #self.basic.delta = 1e-3 #scipy.io.loadmat(filename + 'delta.mat')['delta'].item()
        self.basic.betaFi = scipy.io.loadmat(filename + 'betaFi.mat')['betaFi'].item()
        self.basic.betaW = scipy.io.loadmat(filename + 'betaW.mat')['betaW'].item()
        self.basic.thresholdFi = scipy.io.loadmat(filename + 'thresholdFi.mat')['thresholdFi'].item()
        self.reg_method = reg_method

        if self.reg_method == 'POw_v':
            self.basic.theta = scipy.io.loadmat(filename + 'theta.mat')['theta'].item()
        #self.basic.sparseness = scipy.io.loadmat(filename + 'Sp.mat')['sp'].item()
        
        # self.basic.PreNet = scipy.io.loadmat(filename + 'PreNet.mat')['PreNet'].item()
        if lossMode == 'MeanSquaredReturnError':
            self.basic.gammaHat = 0
        else:
            self.basic.gammaHat = self.basic.gamma
        # with open(filename, "rb") as file:
        #     self.basic = pickle.loads(file.read())
        self.gamma = self.basic.gamma

        if dataset == 'GSDC2022urban':
            self.max_step = 5000
            self.env = GPSPosition_discrete_los(trajdata_range_urban, traj_type_urban, action_scale, discrete_actionspace,
                                                reward_setting,trajdata_sort,baseline_mod)
        if dataset == 'GSDC2022highway':
            self.max_step = 5000
            self.env = GPSPosition_discrete_los(trajdata_range_highway, traj_type_highway, action_scale, discrete_actionspace,
                                                reward_setting, trajdata_sort, baseline_mod)


        self.num_action = self.env.num_action
        self.weights = np.array([self.basic.w.copy().reshape(-1) for _ in range(self.num_action)])
        self.stepSize = stepSize
        self.actionRandomness = actionRandomness
        self.book = {}
        self.decay = decay
        self.loss_func = nn.MSELoss()
        self.betaFi_control = betaFi_control
        self.cnt = 0 # count noise samples
        # CheckSimilarity(self.basic.B)

        # featuresize = np.size(self.weights.shape)
        featuresize = self.weights.shape[1] # change for DLRL gnss

        if self.Wcontrolmode>3.1:
            self.w = ControlLayer(featuresize, self.num_action).cuda()
            self.w_target = ControlLayer(featuresize, self.num_action).cuda()
            self.w.state_dict()['out.weight'].copy_(torch.FloatTensor(self.weights))
            self.w_target.load_state_dict(self.w.state_dict())
            self.epsilon = 0.1
            self.optimizer = torch.optim.SGD(self.w.parameters(), lr=stepSize)
            self.loss_func = nn.MSELoss()

    def getSparseForm(self, observation):
        tempObservation = observation.copy()
        for index, element in enumerate(tempObservation):
            tempObservation[index] = round(element, 7) # 将每个元素四舍五入到小数点后7位
        temp = "*".join(tempObservation.astype('str').tolist()) # 将数值数组转换为字符串数组,用星号连接所有字符串
        if temp in self.book.keys():
            return self.book[temp] # 如果状态已经存在，则不需要重新进行稀疏表示
        # result = self.quickGetSparseForm(observation)
        # result = self.quickGetSparseForm_L1(observation)
        if self.reg_method == 'L0':
            SparseForm = np.dot(observation, self.basic.B).T
            zeros = np.zeros_like(SparseForm)
            result = np.where(np.abs(SparseForm) > self.betaFi_control, SparseForm, zeros)
        elif self.reg_method == 'L1':
            SparseForm = np.dot(observation, self.basic.B).T
            result = np.sign(SparseForm) * np.maximum(np.abs(SparseForm) - self.betaFi_control, 0)
        elif self.reg_method == 'wo_sparse':
            SparseForm = np.dot(observation, self.basic.B).T
            result = SparseForm

        self.book[temp] = result
        return self.book[temp]

    def optimizeW_resp(self, done, reward, action, nextAction, oldObservation, observation, flagObservation):
        oldFi = self.getSparseForm(oldObservation)
        fi = self.getSparseForm(observation)
        # 500 & 10 * reward
        if done:
            q_eval = np.dot(self.weights[action].T, oldFi)
            q_target = 0

            gradient = ((q_eval - q_target)* oldFi)
            gradient = (gradient * oldFi).reshape(-1) + (2 * self.basic.betaW * self.weights[action])
            self.weights[action] -= self.stepSize * gradient.reshape(-1)

        else:
            q_eval = np.dot(self.weights[action].T, oldFi)
            q_target = reward + self.gamma * np.dot(self.weights[nextAction].T, fi)
            q_target_cal=q_target

            gradient = None

            # if q_target>0:
            #     q_target_cal=-10
            oldweights = self.weights.copy()
            if self.Wcontrolmode==1:
                gradient = ((q_eval - q_target_cal) * oldFi).reshape(-1)
                # stepSize_adap = 1/np.dot(oldFi.T,oldFi)#*stepSize_adap[0]
            elif self.Wcontrolmode==2:
                if action == nextAction:
                    fi_dif=(oldFi - self.gamma * fi)
                    gradient = ((q_eval - q_target_cal) * fi_dif).reshape(-1)
                else:
                    gradient = ((q_eval - q_target_cal) * oldFi).reshape(-1)
            elif self.Wcontrolmode==3:
                if action == nextAction:
                    fi_dif=(oldFi - self.gamma * fi)
                    gradient = ((q_eval - q_target_cal) * fi_dif).reshape(-1)
                else:
                    gradient = ((q_eval - q_target_cal) * oldFi).reshape(-1)
                    gradient_next = ((q_eval - q_target_cal) * fi).reshape(-1)
                    self.weights[nextAction] = self.weights[nextAction] \
                                               - self.stepSize*gradient_next.reshape(-1)

            self.weights[action] = self.weights[action] - self.stepSize * gradient.reshape(-1)
            pass
        loss_old = np.power(q_eval-q_target,2)#self.loss_func(q_eval, q_target)
        q_eval_new = np.dot(self.weights[action].T, oldFi)
        q_target_new = reward + self.gamma * np.dot(self.weights[nextAction].T, fi)
        loss = np.power(q_eval_new - q_target_new, 2)
        loss_dif = loss-loss_old
        # print('Loss dif: {:.5e}, Loss_old: {:.5e}, Loss after update: {:.5e}'.format(loss_dif[-1], loss_old[-1],loss[-1]))
        # debug=1
        return loss, loss_dif

    def play(self, times):
        sumReward = 0
        for time in range(times):
            observation = self.env.reset()
            observation = self.basic.net(torch.from_numpy(observation)).detach().numpy()
            done = False
            action = self.act(observation)
            culReward = 0
            count = 0
            while not done:
                count = count + 1
                flag = False
                # if actStandardEnergyPumping(observation, 0) != action:
                if False:
                    flag = True
                    print(observation)
                    print(self.getSparseForm(observation).reshape(-1))
                    for tempAction in range(self.num_action):
                        print(np.dot(self.weights[tempAction].T, self.getSparseForm(observation)).item())
                    print("action {}".format(action))
                oldObservation = observation
                oldAction = action
                observation, reward, done, info = self.env.step(action)
                observation = self.basic.net(torch.from_numpy(observation)).detach().numpy()
                # self.env.render()
                if observation[0] < 0.5 and count < 1500:
                    done = False
                if not done:
                    action = self.act(observation)
                if flag:
                    print('\n')
                    for tempAction in range(self.num_action):
                        print(np.dot(self.weights[tempAction].T, self.getSparseForm(oldObservation)).item())
                    print('-' * 50)
                culReward += reward
            print("Action Play: {}/{}: Cumulative Reward {}".format(time, times, culReward))
            for index, weight in enumerate(self.weights):
                print('Weight {}: {}'.format(index, weight.reshape(-1)))
            sumReward += culReward
            print("Average Cumulative Reward {}".format(sumReward / (time + 1)))
            print('-' * 150)

    def learn(self, times):
        # if self.basic.sparseness == 1.0:
        #     return 0, 0
        tick = 0
        dead_count = 0
        txtflag = False
        episode_rewards = []
        episode_loss=[]
        episode_times=[]
        values_list_all = []
        for time in range(times):
            if self.Wcontrolmode > 3.1:
                culReward, record_times = self.play_sarsa(train=True)
                # culReward, record_times= self.play_sarsa_noise(train=True)  #noise
                #  NO USEFULL culReward= self.play_sarsa(train=True)
                episode_times.append(record_times)
                if self.printtrig > 0.1:
                    print("Action Learn: {}/{}: Cumulative Reward {:.2f}".format(time, times, culReward))

            self.stepSize = self.stepSize * self.decay
            episode_rewards.append(culReward)

        print('train finish !')
        # with open(self.outputFile+'.pkl', 'wb') as value_file:
        #     pickle.dump(values_list_all, value_file, True)
        # value_file.close()
        # with open(self.outputFile+'_rewards.pkl', 'wb') as rewards_file:
        #     pickle.dump(episode_rewards, rewards_file, True)
        # rewards_file.close()
        # with open(self.outputFile+'_times.pkl', 'wb') as times_file:
        #     pickle.dump(episode_times, times_file, True)
        # times_file.close()
        with open(self.outputFile + '_rewards.pkl', 'wb') as rewards_file:
            pickle.dump(episode_rewards, rewards_file, True)
        rewards_file.close()

        if self.dataset == 'GSDC2022urban':
            error = recording_results_ecef(data_truth_dic,trajdata_range_urban,traj_else,self.logdirname,baseline_mod,
                                           traj_record=False)
        elif self.dataset == 'GSDC2022highway':
            error = recording_results_ecef(data_truth_dic, trajdata_range_highway, traj_highway, self.logdirname, baseline_mod,
                                           traj_record=False)
        with open(self.outputFile + f'_diserror={error}.txt', 'w') as file:
            file.write(f'average reward={np.mean(np.array(episode_rewards))}\n')
            file.write(f'converge reward={episode_rewards[-1]}\n')
            file.write(f'average time={np.mean(np.array(episode_times))}\n')
        file.close()
        return episode_rewards, episode_loss

    def act(self, state):
        temp = 0
        if np.abs(state[1]) < 1e-3:
            temp = 0
        if np.random.rand() < self.actionRandomness + temp:
            action = np.random.randint(low=0, high=self.num_action)
            return action
            # return action if action != 1 else (2 if np.random.rand() < 0.5 else 0)
        action = None
        maxQ = None
        for index, weight in enumerate(self.weights):
            tmpSparse = self.getSparseForm(state)
            tempQ = np.dot(weight.T, tmpSparse).item()
            tempQ = tempQ if tempQ <= 0 else 0
            if maxQ is None or tempQ > maxQ:
                maxQ = tempQ
                action = index
        return action

    def preLearn(self, times):
        for time in range(times):
            observation = self.env.reset()
            done = False
            action = actStandardEnergyPumping(observation, 0)
            culReward = 0
            while not done:
                print(observation)
                print(self.getSparseForm(observation).reshape(-1))
                for tempAction in range(self.env.action_space.n):
                    print(np.dot(self.weights[tempAction].T, self.getSparseForm(observation)).item())
                for index, weight in enumerate(self.weights):
                    print('Weight {}: {}'.format(index, weight.reshape(-1)))
                oldObservation = observation
                oldAction = action
                observation, reward, done, info = self.env.step(action)
                self.env.render()
                if not done:
                    action = actStandardEnergyPumping(observation, 0)
                self.optimizeW_resp(done, reward, oldAction, action, oldObservation, observation)
                print('\n')
                for tempAction in range(self.num_action):
                    print(np.dot(self.weights[tempAction].T, self.getSparseForm(oldObservation)).item())
                print('-' * 50)
                culReward += reward
            print('Prelearning')
            print("Action Learn: {}/{}: Cumulative Reward {}".format(time, times, culReward))
            for index, weight in enumerate(self.weights):
                print('Weight {}: {}'.format(index, weight.reshape(-1)))
            print('-' * 150)

    def _encoder(self, observation):
        # observation_ = self.basic.net(torch.from_numpy(observation.copy()).to(torch.float64)).detach().numpy()
        observation_  = observation.flatten() # flatten
        observation_encode = self.getSparseForm(observation_)
        return np.squeeze(observation_encode)

    def decide(self, s):
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(self.num_action)
        else:
            z = self._encoder(s)
            z = torch.unsqueeze(torch.FloatTensor(z), 0).cuda()  # get a 1D array
            Q_value = self.w.forward(z)
            Q_value = Q_value.detach().cpu()
            greed_action = torch.max(Q_value, 1)[1].data.numpy()
            greed_action = greed_action[0]
            action = greed_action
        return action

    def learn_sarsa(self, s, a, s_t, a_t, reward, done):
        z = self._encoder(s)
        z = torch.unsqueeze(torch.FloatTensor(z), 0).cuda()
        z_t = self._encoder(s_t)
        z_t = torch.unsqueeze(torch.FloatTensor(z_t), 0).cuda()
        a = torch.LongTensor(np.expand_dims([a], axis=0)).cuda()
        a_t = torch.LongTensor(np.expand_dims([a_t], axis=0)).cuda()
        reward = torch.LongTensor(np.expand_dims([reward], axis=0))
        q_eval = self.w(z).gather(1,a)
        q_next = self.w_target(z_t).detach().gather(1, a_t) # 收集每个样本选择的动作对应的Q值
        q_target = reward.cuda() + self.gamma * q_next * (1-done)
        loss = self.loss_func(q_eval, q_target)
        #print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.w_target.load_state_dict(self.w.state_dict())
        return

    def play_sarsa(self, train=False, render=False):
        env = self.env
        episode_reward = 0
        count=0
        observation = env.reset()
        action = self.decide(observation)
        record_times=[]
        while True:
            start_time=time.time()
            if render:
                env.render()
            next_observation, reward, done, _ = env.step(action)

            # 添加噪声实验
            if noisedB is not None:
                mean, std = 0, 1  # mean and standard deviation
                sigma = sigma_compute(next_observation,SNRdB=noisedB)
                next_observation = next_observation + np.random.normal(mean, std, size=next_observation.shape) * sigma

            episode_reward += reward
            next_action = self.decide(next_observation)
            count=count+1
            # if (reward < - 5) | (reward > 5) :# (count % 200) == 0 |
            #     print(f'step={count},reward={reward}')
            if count >= self.max_step:#flagObservation[0] < 0.5 and
                done = True
            if train:
                self.learn_sarsa(observation, action, next_observation, next_action, reward, done)
            if done:
                break
            observation, action = next_observation, next_action
            end_time=time.time()
            record_times.append(end_time-start_time)

            # 采集不同噪声下的稀疏表示样本
            # self.cnt += 1
            # noise_phi[0].append(observation.flatten())
            # for idx, noisedB in enumerate(noise_levels):
            #     mean, std = 0, 1  # mean and standard deviation
            #     if noisedB is not None:
            #         sigma = sigma_compute(observation, SNRdB=noisedB)
            #         observation = observation + np.random.normal(mean, std, size=observation.shape) * sigma
            #     z = self._encoder(observation)
            #     noise_phi[idx+1].append(z)
            #     if self.cnt > noise_sample:
            #         with open(self.outputFile + f'_noise_{self.reg_method}.pkl', 'wb') as file:
            #             pickle.dump(noise_phi, file, True)
            #         file.close()
            #         print('Save noise representation samples')
            #         break

        for param_group in self.optimizer.param_groups: # 衰减学习率
            param_group['lr'] *= 0.9
        return episode_reward, np.mean(record_times)

## noise
    def play_sarsa_noise(self, train=False, render=False):
        env = self.env
        episode_reward = 0
        count=0
        observation = env.reset()
        mean, std = 0, 1  # mean and standard deviation
        size = len(observation)
        sigma = sigma_compute(observation)
        observation = observation + np.random.normal(mean, std, size) * sigma
        action = self.decide(observation)
        record_times=[]
        while True:
            start_time=time.time()
            if render:
                env.render()
            next_observation, reward, done, _ = env.step(action)
            clean_signal = next_observation
            sigma = sigma_compute(next_observation)
            next_observation = next_observation + np.random.normal(mean, std, size) * sigma

            #print(action, reward)
            episode_reward += reward
            next_action = self.decide(next_observation) # 终止状态时此步无意义
            count=count+1
            if count >= self.max_step:#flagObservation[0] < 0.5 and
                done = True
            if train:
                self.learn_sarsa(observation, action, next_observation, next_action, reward, done)
            if done:
                break
            observation, action = next_observation, next_action
            end_time=time.time()
            record_times.append(end_time-start_time)
        return episode_reward, np.mean(record_times)

class ControlLayer(nn.Module):
    """docstring for Net"""

    def __init__(self, pre_num, next_num):
        super(ControlLayer, self).__init__()
        self.out = nn.Linear(pre_num, next_num, bias=False)
        # self.out.weight.copy_(weights)

    def forward(self, x):
        action_prob = self.out(x)
        return action_prob
