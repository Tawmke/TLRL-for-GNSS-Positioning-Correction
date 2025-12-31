import numpy as np
from funcs.pretrain_model_preNet import *
import scipy.io
from env.mc_env import *
from env.pw_env import *
from env.ac_env import *
from env.ca_env import *
from gym.envs.classic_control.cartpole import *

class ActionLearner:
    def __init__(self, actionRandomness, decay, stepSize, preTrainTimes=None, maxIter=None, gamma=None, lossMode=None,
                 wordNums=None, betaB=None, betaW=None, betaFi=None,
                 lineSearchW=None, lineSearchB=None, thresholdW=None, thresholdB=None, thresholdFi=None,
                 thresholdLoss=None, lineAlphaW=None, lineBetaW=None,
                 lineAlphaB=None, lineBetaB=None, dataSource=None, randomness=None,
                 normalized=None, lipSafe=None, loadBasic=None, index=1,
                 outputFile=None, betaFi_control =None, reg_method =None, Model_folder=None, dataset=None,
                 preprocess=None, printtrig=None, Wcontrolmode=None):

        self.outputFile = outputFile
        self.dataset = dataset
        self.basic = None
        self.gamma = None
        self.printtrig =printtrig
        self.Wcontrolmode=Wcontrolmode

        if dataset == 'MC':
            self.max_step=500
            self.env = gym.make('MountainCar-v0')
            # env = mountaincar(max_step=max_step, register=register)
        elif dataset == 'PW':
            self.max_step=500
            self.env = finite_puddleworld(max_step=self.max_step)
        elif dataset == 'PW2' or dataset == 'PW2T':
            self.max_step=500
            self.env = finite_puddleworld2(normalized=True, max_step=self.max_step)
        elif dataset == 'AC' or dataset == 'AC2':
            self.max_step=500
            self.env = Acrobot(max_step=self.max_step)
        elif dataset == 'CA3':
            self.max_step=5000
            self.env = catcher3(init_lives=1)
        elif dataset == 'CA':
            self.env = catcher(init_lives=1)
        elif dataset == 'CP' or dataset == 'CP_nrm':
            self.max_step=500
            self.env = CartPoleEnv()

        if dataset == 'CP' or dataset == 'CP_nrm':
            self.num_state = self.env.observation_space.shape[0]
            self.num_action = self.env.action_space.n
        elif dataset == 'MC':
            self.num_action = self.env.action_space.n
        else:
            self.num_state = self.env.num_state
            self.num_action = self.env.num_action

        # filename = "./matlab_Model/" + loadBasic  + "/" + loadBasic + '_' + str(index) +'/'
        filename = '{}/{}/Run_{}/'.format(Model_folder, loadBasic, index)
        if preprocess == 'DQN_norm_only':
            self.basic = DLSRSolver_preNet(Model_folder, preprocess, dataset, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, skip=True)
            PreNet2 = scipy.io.loadmat(Model_folder+'_output_weights.mat')['Wa']
            self.weights = np.array([PreNet2.copy().reshape(-1) for _ in range(self.num_action)])
            self.gamma=1
            featuresize = np.max(self.weights.shape)#self.weights.shape[1]
        else:
            self.basic = DLSRSolver_preNet(filename, preprocess, dataset, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, skip=True)
            self.basic.B = scipy.io.loadmat(filename + 'B.mat')['B']
            self.basic.wordNums = self.basic.B.shape[0]
            self.basic.w = scipy.io.loadmat(filename + 'W.mat')['W']
            self.basic.weightReconstruct = scipy.io.loadmat(filename + 'weightReconstruct.mat')[
                'weightReconstruct'].item()
            self.basic.gamma = scipy.io.loadmat(filename + 'gamma.mat')['gamma'].item()
            self.basic.weightPredict = scipy.io.loadmat(filename + 'weightPredict.mat')['weightPredict'].item()
            self.basic.maxIter = scipy.io.loadmat(filename + 'maxIter.mat')['maxIter'].item()
            self.basic.delta = scipy.io.loadmat(filename + 'delta.mat')['delta'].item()
            self.basic.betaFi = scipy.io.loadmat(filename + 'betaFi.mat')['betaFi'].item()
            self.basic.betaW = scipy.io.loadmat(filename + 'betaW.mat')['betaW'].item()
            self.basic.thresholdFi = scipy.io.loadmat(filename + 'thresholdFi.mat')['thresholdFi'].item()
            # self.basic.PreNet = scipy.io.loadmat(filename + 'PreNet.mat')['PreNet'].item()
            self.gamma = self.basic.gamma
            self.weights = np.array([self.basic.w.copy().reshape(-1) for _ in range(self.num_action)])
            featuresize = np.max(self.weights.shape)#self.weights.shape[1]
        self.basic.lossMode = 'MeanSquaredReturnError'
        if lossMode == 'MeanSquaredReturnError':
            self.basic.gammaHat = 0
        else:
            self.basic.gammaHat = self.gamma
        # with open(filename, "rb") as file:
        #     self.basic = pickle.loads(file.read())

        self.stepSize = stepSize
        self.actionRandomness = actionRandomness
        self.book = {}
        self.decay = decay
        self.loss_func = nn.MSELoss()
        self.betaFi_control = betaFi_control
        self.reg_method = reg_method
        # CheckSimilarity(self.basic.B)

        if self.Wcontrolmode>3.1:
            self.w = ControlLayer(featuresize, self.num_action)
            self.w_target = ControlLayer(featuresize, self.num_action)
            self.w.state_dict()['out.weight'].copy_(torch.FloatTensor(self.weights))
            self.w_target.load_state_dict(self.w.state_dict())
            self.epsilon = 0.1
            self.optimizer = torch.optim.SGD(self.w.parameters(), lr=stepSize)
            self.loss_func = nn.MSELoss()


    def quickGetSparseForm(self, state):
        state_tmp = np.expand_dims(state, 0)
        resFi = np.dot(state_tmp, self.basic.B).T
        return resFi

    def L0Process(self, SparseForm):
        zeros = np.zeros_like(SparseForm)
        result = np.where(np.abs(SparseForm)>self.betaFi_control,SparseForm,zeros)
        return result

    def getSparseForm(self, observation):
        tempObservation = observation.copy()
        for index, element in enumerate(tempObservation):
            tempObservation[index] = round(element, 7)
        temp = "*".join(tempObservation.astype('str').tolist())

        if temp in self.book.keys():
            return self.book[temp]
        result = self.quickGetSparseForm(observation)
        if(self.reg_method=='L0'):
            result = self.L0Process(result)
        elif self.reg_method == 'L0relu':
            result = self.L0Process(result)
            result[result < 0] = 0
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
        tick = 0
        dead_count = 0
        txtflag = False
        episode_rewards = []
        episode_loss=[]
        values_list_all = []
        for time in range(times):
            if self.Wcontrolmode>3.1:
                culReward= self.play_sarsa(train=True)
                if self.printtrig > 0.1:
                    print("Action Learn: {}/{}: Cumulative Reward {:.2f}".format(time, times, culReward))
            else:
                observation = self.env.reset()
                flagObservation = observation.copy()
                temp = torch.from_numpy(observation.copy()).to(torch.float64)
                observation = self.basic.net(torch.from_numpy(observation.copy()).to(torch.float64)).detach().numpy()
                done = False
                action = self.act(observation)
                culReward = 0
                count = 0
                tick = tick + 1
                losstmp=[]
                values_list = []
                loss_diff_cnt=0
                while not done:
                    count = count + 1
                    oldObservation = observation
                    oldAction = action
                    observation, reward, done, info = self.env.step(action)
                    flagObservation = observation.copy()
                    observation = self.basic.net(torch.from_numpy(observation.copy()).to(torch.float64)).detach().numpy()
                    # self.env.render()
                    if count >= self.max_step:#flagObservation[0] < 0.5 and
                        done = True
                    if not done:
                        action = self.act(observation)
                    if txtflag:
                        with open(self.outputFile, "a") as f:
                            f.write('\n')
                            # print('\n')
                            values=np.dot(self.weights, self.getSparseForm(observation))
                            values_list.append(np.vstack((values,action)))

                            for tempAction in range(self.num_action):
                                f.write(str(values[tempAction].item()))
                                f.write('\n')
                            f.write("action {}".format(action))
                            f.write('\n')
                    loss,loss_diff = self.optimizeW_resp(done, reward, oldAction, action, oldObservation, observation, flagObservation)
                    culReward += reward
                    if loss_diff<0:
                        loss_diff_cnt+=1
                    losstmp.append(loss_diff)
                if self.printtrig>0.1:
                    print("Action Learn: {}/{}: Cumulative Reward {:.2f}, Loss_dif_sum {:.2e}, Loss_dif_per {:.2f}".format(time, times, culReward, sum(losstmp)[0], loss_diff_cnt/count*100))
                if txtflag:
                    with open(self.outputFile, "a") as f:
                        f.write("Action Learn: {}/{}: Cumulative Reward {:.2f}".format(time, times, culReward))
                        f.write('\n')
                episode_loss.append(losstmp)
                values_list_all.append(np.array(values_list).reshape(-1,4))
            self.stepSize = self.stepSize * self.decay
            if self.dataset == 'MC' and culReward == -self.max_step:
                dead_count = dead_count + 1
            else:
                dead_count = 0
            episode_rewards.append(culReward)
            if dead_count >= 10:
                with open(self.outputFile, "a") as f:
                    f.write("Dead Game")
                    f.write('\n')
                break
                # return

        with open(self.outputFile+'.pkl', 'wb') as value_file:
            pickle.dump(values_list_all, value_file, True)
        value_file.close()
        with open(self.outputFile+'_rewards.pkl', 'wb') as rewards_file:
            pickle.dump(episode_rewards, rewards_file, True)
        rewards_file.close()
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

    # def act_re(self, state):
    #     z = self.getSparseForm(state)
    #     Q_value = tf.matmul(z, self.W_v)
    #     greed_action = tf.argmax(Q_value, axis=1, output_type=tf.int32)
    #     greed_action = greed_action.numpy()[0]
    #     if np.random.uniform() < self.epsilon:
    #         action = np.random.randint(self.num_action)
    #     else:
    #         action = greed_action
    #     return action

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

    def optimizeW_resp_dnn(self, done, reward, action, nextAction, oldObservation, observation):
        oldFi = oldObservation
        fi = observation
        # 500 & 10 * reward
        if done:
            q_eval = np.dot(self.weights[action].T, oldFi)
            q_target = 0

            gradient = ((q_eval - q_target)* oldFi)
            gradient = (gradient * oldFi).reshape(-1) #+ (2 * self.basic.betaW * self.weights[action])
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

    def learn_only(self, times):
        tick = 0
        dead_count = 0
        episode_rewards = []
        episode_loss=[]
        values_list_all = []
        for time in range(times):
            if self.Wcontrolmode>3.1:
                culReward= self.play_sarsa_only(train=True)
                if self.printtrig > 0.1:
                    print("Action Learn: {}/{}: Cumulative Reward {:.2f}".format(time, times, culReward))
            else:
                observation = self.env.reset()
                observation = self.basic.net(torch.from_numpy(observation.copy()).to(torch.float64)).detach().numpy()
                done = False
                action = self.act_dnn(observation)
                culReward = 0
                count = 0
                tick = tick + 1
                losstmp=[]
                values_list = []
                loss_diff_cnt=0
                while not done:
                    count = count + 1
                    oldObservation = observation
                    oldAction = action
                    observation, reward, done, info = self.env.step(action)
                    flagObservation = observation.copy()
                    observation = self.basic.net(torch.from_numpy(observation.copy()).to(torch.float64)).detach().numpy()
                    # self.env.render()
                    if count >= self.max_step:#flagObservation[0] < 0.5 and
                        done = True
                    if not done:
                        action = self.act_dnn(observation)
                    loss,loss_diff = self.optimizeW_resp_dnn(done, reward, oldAction, action, oldObservation, observation)
                    culReward += reward
                    if loss_diff<0:
                        loss_diff_cnt+=1
                    losstmp.append(loss_diff)
                if self.printtrig>0.1:
                    print("Action Learn: {}/{}: Cumulative Reward {:.2f}, Loss_dif_sum {:.2e}, Loss_dif_per {:.2f}".format(time, times, culReward, sum(losstmp), loss_diff_cnt/count*100))
                values_list_all.append(np.array(values_list).reshape(-1,4))
            self.stepSize = self.stepSize * self.decay
            if self.dataset == 'MC' and culReward == -self.max_step:
                dead_count = dead_count + 1
            else:
                dead_count = 0
            episode_rewards.append(culReward)
            if dead_count >= 10:
                with open(self.outputFile, "a") as f:
                    f.write("Dead Game")
                    f.write('\n')
                break
                # return

        with open(self.outputFile+'.pkl', 'wb') as value_file:
            pickle.dump(values_list_all, value_file, True)
        value_file.close()
        with open(self.outputFile+'_rewards.pkl', 'wb') as rewards_file:
            pickle.dump(episode_rewards, rewards_file, True)
        rewards_file.close()
        return episode_rewards

    def act_dnn(self, state):
        if np.random.rand() < self.actionRandomness:
            action = np.random.randint(low=0, high=self.num_action)
            return action
        action = None
        maxQ = None
        for index, weight in enumerate(self.weights):
            tempQ = np.dot(weight.T, state).item()
            tempQ = tempQ if tempQ <= 0 else 0
            if maxQ is None or tempQ > maxQ:
                maxQ = tempQ
                action = index
        return action

    def decide_only(self, s):
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(self.num_action)
        else:
            z = self._encoder_only(s)
            z = torch.unsqueeze(torch.FloatTensor(z), 0)  # get a 1D array
            Q_value = self.w.forward(z)
            greed_action = torch.max(Q_value, 1)[1].data.numpy()
            greed_action = greed_action[0]
            action = greed_action
        return action

    def _encoder_only(self, observation):
        observation_ = self.basic.net(torch.from_numpy(observation.copy()).to(torch.float64)).detach().numpy()
        # observation_ = self.basic.net(torch.from_numpy(observation.copy())).detach().numpy()
        return np.squeeze(observation_)

    def learn_sarsa_only(self, s, a, s_t, a_t, reward, done):
        z = self._encoder_only(s)
        z = torch.unsqueeze(torch.FloatTensor(z), 0)
        z_t = self._encoder_only(s_t)
        z_t = torch.unsqueeze(torch.FloatTensor(z_t), 0)
        a = torch.LongTensor(np.expand_dims([a], axis=0))
        a_t = torch.LongTensor(np.expand_dims([a_t], axis=0))
        reward = torch.LongTensor(np.expand_dims([reward], axis=0))
        q_eval = self.w(z).gather(1,a)
        q_next = self.w_target(z_t).detach().gather(1, a_t)
        q_target = reward + self.gamma * q_next * (1-done)
        loss = self.loss_func(q_eval, q_target)
        #print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.w_target.load_state_dict(self.w.state_dict())
        return

    def play_sarsa_only(self, train=False, render=False):
        env = self.env
        episode_reward = 0
        count=0
        observation = env.reset()
        action = self.decide_only(observation)
        while True:
            if render:
                env.render()
            next_observation, reward, done, _ = env.step(action)
            #print(action, reward)
            episode_reward += reward
            next_action = self.decide_only(next_observation) # 终止状态时此步无意义
            count=count+1
            if count >= self.max_step:#flagObservation[0] < 0.5 and
                done = True
            if train:
                self.learn_sarsa_only(observation, action, next_observation, next_action, reward, done)
            if done:
                break
            observation, action = next_observation, next_action
        return episode_reward

    def decide(self, s):
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(self.num_action)
        else:
            z = self._encoder(s)
            z = torch.unsqueeze(torch.FloatTensor(z), 0)  # get a 1D array
            Q_value = self.w.forward(z)
            greed_action = torch.max(Q_value, 1)[1].data.numpy()
            greed_action = greed_action[0]
            action = greed_action
        return action

    def _encoder(self, observation):
        observation_ = self.basic.net(torch.from_numpy(observation.copy()).to(torch.float64)).detach().numpy()
        observation_encode = self.getSparseForm(observation_)
        return np.squeeze(observation_encode)

    def learn_sarsa(self, s, a, s_t, a_t, reward, done):
        z = self._encoder(s)
        z = torch.unsqueeze(torch.FloatTensor(z), 0)
        z_t = self._encoder(s_t)
        z_t = torch.unsqueeze(torch.FloatTensor(z_t), 0)
        a = torch.LongTensor(np.expand_dims([a], axis=0))
        a_t = torch.LongTensor(np.expand_dims([a_t], axis=0))
        reward = torch.LongTensor(np.expand_dims([reward], axis=0))
        q_eval = self.w(z).gather(1,a)
        q_next = self.w_target(z_t).detach().gather(1, a_t)
        q_target = reward + self.gamma * q_next * (1-done)
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
        while True:
            if render:
                env.render()
            next_observation, reward, done, _ = env.step(action)
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
        return episode_reward

class ControlLayer(nn.Module):
    """docstring for Net"""

    def __init__(self, pre_num, next_num):
        super(ControlLayer, self).__init__()
        self.out = nn.Linear(pre_num, next_num, bias=False)
        # self.out.weight.copy_(weights)

    def forward(self, x):
        action_prob = self.out(x)
        return action_prob