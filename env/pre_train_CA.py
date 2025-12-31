import tensorflow as tf
import matplotlib.pyplot as plt
import random
import os
import pickle
import time
from models.nn_model import *
from env.mc_env import *
from env.pw_env import *
from env.ac_env import *
from utils.utils1 import *
from data.gen_exp import *
from plots.plot import *
import numpy as np
from env.ca_env import *


def pre_train(arg, register):
    # absolute_route = '/home/wjq/code/pyProject/SR_in_DRL/'
    absolute_route = '/zhao/code/SR_in_DRL/'
    #absolute_route = '/root/code/SR_in_DRL/'
    batch_size = arg.batch_size
    gamma = arg.gamma
    pre_lr = arg.pre_lr
    epoch = arg.epoch
    lambda_list = arg.lambda_list
    beta_list = arg.beta_list
    num_run = arg.num_run
    dataset = arg.dataset
    max_step = arg.max_step
    reg_method = arg.reg_method
    reg_nn = arg.reg_nn
    reg_vf = arg.reg_vf
    use_target = arg.use_target
    scale = arg.scale
    tar_fre = arg.tar_fre
    nn_shape = arg.nn_shape
    episode = arg.episode
    epsilon = arg.epsilon
    thread = arg.thread
    if reg_method == 'DSRLogGroupPo':
        arg.reg_nn[1] *= arg.p_norm

    if dataset == 'MC':
        env = mountaincar(max_step=max_step, register=register)
    elif dataset == 'PW':
        env = finite_puddleworld(max_step=max_step)
    elif dataset == 'AC' or dataset == 'AC2':
        env = Acrobot(max_step=max_step)
    elif dataset == 'CA3':
        env = catcher3(init_lives=1)
    elif dataset == 'CA':
        env = catcher(init_lives=1)
    num_state = env.num_state
    num_action = env.num_action
    Agent_lr_ls = [0.4, 0.1, 0.04, 0.01, 0.004]
    #.............读入训练数据..............  , 0.001
    """
    exp = []
    for i in range(1, num_run+1):
        exp_in = open(absolute_route + 'data/{}/exp{}.pkl'.format(dataset, i), 'rb')
        expReplayer = pickle.loads(exp_in.read())
        exp.append(expReplayer)
        exp_in.close()
    """
    if dataset=='AC2':
        dataset1 = 'AC'
    else:
        dataset1 = dataset

    # .............读入作者训练数据..............
    train_data = []
    for i in range(1, num_run + 1):
        data_train = create_train_dataset(absolute_route + 'data/{}/10ktrain{}.txt'.format(dataset1, i + 50), dataset1,
                                       normalize=True, scale=scale)
        train_data.append(data_train)

    """
    data_temp = train_data[0][0]
    temp1 = np.sqrt(np.sum(np.square(data_temp), axis=1))
    for n in range(10000):
        for m in range(2):
            data_temp[n][m]=data_temp[n][m]/temp1[n]

    temp = np.dot(data_temp, data_temp.T)
    temp = np.abs(temp)
    stat = [0 for i in range(10)]
    for i in range(temp.shape[0]):
        for j in range(i, temp.shape[0]):
            if(temp[i, j]>=1):
                temp[i, j] = 0.9999
            index_ = int(np.floor(temp[i, j] * 10))
            # print(index_, temp[i, j])
            stat[index_] += 1
    plt.bar(range(10), stat)
    plt.show()
    """

    #.............读入测试数据..............
    test_data = create_test_dataset(absolute_route + 'data/{}/test.txt'.format(dataset1), dataset1, normalize=True,
                                    scale=scale)


    #................同一个表示参数要训练多次...................
    for i in range(1, num_run+1):
        if reg_method == 'SR' or reg_method == 'NN' or reg_method == 'DSR':
            Model = SrnnModel(nn_shape, num_action=num_action, reg_method=reg_method, lr_rate=pre_lr, beta_list=beta_list,
                              lambdaKL_list=lambda_list, reg_vf=reg_vf, reg_nn=reg_nn, use_target=use_target,
                              tar_fre=tar_fre, gamma=gamma, epsilon=epsilon)
        elif reg_method == 'L1w' or reg_method == 'L1a' or reg_method == 'L2w' or reg_method == 'L2a':
            Model = nnModel(nn_shape, num_action=num_action, reg_method=reg_method, lr_rate=pre_lr, beta_list=beta_list,
                              lambdaKL_list=lambda_list, reg_vf=reg_vf, reg_nn=reg_nn, use_target=use_target,
                              tar_fre=tar_fre, gamma=gamma, epsilon=epsilon)
        elif reg_method == 'Hoyer' or reg_method == 'Hoyer_a':
            Model = HoyerModel(nn_shape, num_action=num_action, reg_method=reg_method, lr_rate=pre_lr, beta_list=beta_list,
                            lambdaKL_list=lambda_list, reg_vf=reg_vf, reg_nn=reg_nn, use_target=use_target,
                            tar_fre=tar_fre, gamma=gamma, epsilon=epsilon)
        elif reg_method == 'DSRHoyer1' or reg_method=='DSRHoyer3':
            Model = SrnnHoyerPo1Model(nn_shape, num_action=num_action, reg_method=reg_method, lr_rate=pre_lr, beta_list=beta_list,
                            lambdaKL_list=lambda_list, reg_vf=reg_vf, reg_nn=reg_nn, use_target=use_target,
                            tar_fre=tar_fre, gamma=gamma, epsilon=epsilon)
        elif reg_method == 'DSRL1wPo2' or reg_method == 'DSRPnormPo23' or reg_method == 'DSRPnormPo12' or reg_method == 'DSRLogPo2' or \
                reg_method == 'DSRHoyer2' or reg_method== 'DSRHoyer4' or reg_method == 'DSRLogGroupPo':
            Model = SrnnPoModel(nn_shape, num_action=num_action, reg_method=reg_method, lr_rate=pre_lr, beta_list=beta_list,
                            lambdaKL_list=lambda_list, reg_vf=reg_vf, reg_nn=reg_nn, use_target=use_target,
                            tar_fre=tar_fre, gamma=gamma, epsilon=epsilon, log_norm=arg.log_pra, p_norm=arg.p_norm)
        elif reg_method == 'DSRLogPo1':
            Model = SrnnLogPo1Model(nn_shape, num_action=num_action, reg_method=reg_method, lr_rate=pre_lr, beta_list=beta_list,
                            lambdaKL_list=lambda_list, reg_vf=reg_vf, reg_nn=reg_nn, use_target=use_target, tar_fre=tar_fre, gamma=gamma,
                            epsilon=epsilon, log_pra=arg.log_pra)
        elif reg_method == 'DSRL1wPo1' or reg_method == 'DSRL1w' or reg_method=='L1wPo1':
            Model = SrnnL1wPoModel(nn_shape, num_action=num_action, reg_method=reg_method, lr_rate=pre_lr, beta_list=beta_list,
                            lambdaKL_list=lambda_list, reg_vf=reg_vf, reg_nn=reg_nn, use_target=use_target,
                            tar_fre=tar_fre, gamma=gamma, epsilon=epsilon)
        elif reg_method == 'DSRL1w':
            Model = SrnnL1wModel(nn_shape, num_action=num_action, reg_method=reg_method, lr_rate=pre_lr, beta_list=beta_list,
                            lambdaKL_list=lambda_list, reg_vf=reg_vf, reg_nn=reg_nn, use_target=use_target,
                            tar_fre=tar_fre, gamma=gamma, epsilon=epsilon)
        elif reg_method == 'DSRL2w':
            Model = SrnnL2wModel(nn_shape, num_action=num_action, reg_method=reg_method, lr_rate=pre_lr, beta_list=beta_list,
                            lambdaKL_list=lambda_list, reg_vf=reg_vf, reg_nn=reg_nn, use_target=use_target,
                            tar_fre=tar_fre, gamma=gamma, epsilon=epsilon)
        elif reg_method == 'dropout':
            Model = dropout_Model(nn_shape, num_action=num_action, reg_method=reg_method, lr_rate=pre_lr, beta_list=beta_list,
                                lambdaKL_list=lambda_list, reg_vf=reg_vf, reg_nn=reg_nn, use_target=use_target,
                                tar_fre=tar_fre, gamma=gamma, epsilon=epsilon)
        elif reg_method == 'log':
            Model = LogModel(nn_shape, num_action=num_action, reg_method=reg_method, lr_rate=pre_lr,
                              beta_list=beta_list,
                              lambdaKL_list=lambda_list, reg_vf=reg_vf, reg_nn=reg_nn, use_target=use_target,
                              tar_fre=tar_fre, gamma=gamma, epsilon=epsilon, log_pra=arg.log_pra)
        elif reg_method == 'P_norm':
            Model = P_norm_Model(nn_shape, num_action=num_action, reg_method=reg_method, lr_rate=pre_lr,
                             beta_list=beta_list,
                             lambdaKL_list=lambda_list, reg_vf=reg_vf, reg_nn=reg_nn, use_target=use_target,
                             tar_fre=tar_fre, gamma=gamma, epsilon=epsilon, p_norm=arg.p_norm)
        elif reg_method == 'P_normPo':
            Model = P_normPo_Model(nn_shape, num_action=num_action, reg_method=reg_method, lr_rate=pre_lr,
                            beta_list=beta_list,
                            lambdaKL_list=lambda_list, reg_vf=reg_vf, reg_nn=reg_nn, use_target=use_target,
                            tar_fre=tar_fre, gamma=gamma, epsilon=epsilon, p_norm=arg.p_norm)
        elif reg_method == 'DSRL1A' or reg_method == 'DSRL2A':
            Model = SrnnL1AModel(nn_shape, num_action=num_action, reg_method=reg_method, lr_rate=pre_lr,
                                   beta_list=beta_list,
                                   lambdaKL_list=lambda_list, reg_vf=reg_vf, reg_nn=reg_nn, use_target=use_target,
                                   tar_fre=tar_fre, gamma=gamma, epsilon=epsilon)
        elif reg_method =='DSRLog_a':
            Model = DSRLog_aModel(nn_shape, num_action=num_action, reg_method=reg_method, lr_rate=pre_lr, beta_list=beta_list,
                            lambdaKL_list=lambda_list, reg_vf=reg_vf, reg_nn=reg_nn, use_target=use_target, tar_fre=tar_fre, gamma=gamma,
                            epsilon=epsilon, log_pra=arg.log_pra)
        elif reg_method == 'L2aLogPo2':
            Model = L2aPoModel(nn_shape, num_action=num_action, reg_method=reg_method, lr_rate=pre_lr,
                                  beta_list=beta_list,
                                  lambdaKL_list=lambda_list, reg_vf=reg_vf, reg_nn=reg_nn, use_target=use_target,
                                  tar_fre=tar_fre, gamma=gamma,
                                  epsilon=epsilon, log_norm=arg.log_pra)
        elif reg_method == 'myL1wPo1':
            Model = MyL1wPoModel(nn_shape, num_action=num_action, reg_method=reg_method, lr_rate=pre_lr,
                                   beta_list=beta_list,
                                   lambdaKL_list=lambda_list, reg_vf=reg_vf, reg_nn=reg_nn, use_target=use_target,
                                   tar_fre=tar_fre, gamma=gamma, epsilon=epsilon)
        elif reg_method == 'DSRHoyer5':
            Model = SrnnHoyer5Model(nn_shape, num_action=num_action, reg_method=reg_method, lr_rate=pre_lr, beta_list=beta_list,
                            lambdaKL_list=lambda_list, reg_vf=reg_vf, reg_nn=reg_nn, use_target=use_target,
                            tar_fre=tar_fre, gamma=gamma, epsilon=epsilon, p_norm=arg.p_norm)
        elif reg_method == 'MCP':
            Model = MCPModel(nn_shape, num_action=num_action, reg_method=reg_method, lr_rate=pre_lr,
                                    beta_list=beta_list,reg_vf=reg_vf, reg_nn=reg_nn, use_target=use_target,
                                    tar_fre=tar_fre, gamma=gamma, epsilon=epsilon, p_norm=arg.p_norm, lambdaKL_list=lambda_list)
        elif reg_method == 'Hoyer5a':
            Model = Hoyer5aModel(nn_shape, num_action=num_action, reg_method=reg_method, lr_rate=pre_lr,
                             beta_list=beta_list, reg_vf=reg_vf, reg_nn=reg_nn, use_target=use_target,
                             tar_fre=tar_fre, gamma=gamma, epsilon=epsilon, p_norm=arg.p_norm, lambdaKL_list=lambda_list)
        Model.build(input_shape=(None, num_state))  # 目标网路
        Model.gen_var_list()
        Model.summary()
        #.................一个epoch训练得结果保存.................
        ae_loss_list = []  #all loss sum
        pred_loss_list = []  #MSTDE loss
        SKL_loss_list = []  #SKL loss sum
        reg_loss_list = []  #reg in nn loss
        SKL_layers_list = []  #SKL of each layer
        test_loss_list = []
        pop_sparse_list = []
        num_active_list = []

        ## ............作者训练数据处理.............

        data = train_data[i-1]

        DSR_setting=''
        regnn_setting = ''
        for j in range(1, 1 + len(nn_shape)):
            DSR_setting += ' lambda{}={} beta{}={}'.format(j, lambda_list[j-1], j, beta_list[j-1])
            regnn_setting += ' regnn{}={}'.format(j, reg_nn[j-1])

        setting = 'epoch={} use_target={}. p_norm={} log_pra={}'.format(epoch, use_target, arg.p_norm, arg.log_pra) + regnn_setting + DSR_setting
        print(setting)

        folder_name = absolute_route + 'results/{}/{}'.format(dataset, reg_method)

        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        folder_name = folder_name + '/{}'.format(setting)
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        trailnum=(thread-1)*num_run + i
        folder_name = folder_name+'/'+str(trailnum)
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        logger_txt = folder_name+'/logger.txt'
        logger = open(logger_txt, 'a')

        # expReplayer = exp[i-1]
        # iteration = int(expReplayer.count / batch_size)
        iteration = int(len(data[0])/ batch_size)



        train_list = []
        for k in range(len(data[0])):
            train_list.append(k)
        # 表示网络训练
        for j in range(epoch):
            random.shuffle(train_list)
            ae_loss = 0
            pred_loss = 0
            SKL_loss = 0
            reg_loss = 0
            SKL_list = np.zeros(len(nn_shape))

            for iter in range(iteration):
                list_temp = train_list[iter * batch_size:(iter + 1) * batch_size]

                au_current_state = data[0][list_temp]
                au_next_state = data[1][list_temp]
                au_reward = data[2][list_temp]
                au_dones = np.abs(data[3][list_temp]-1)
                """
                observations, actions, rewards, next_observations, dones = \
                    expReplayer.list_sample(list_temp)  # 经验回放
                observations = tf.convert_to_tensor(observations, dtype=tf.float32)
                next_observations = tf.convert_to_tensor(next_observations, dtype=tf.float32)
                rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
                dones = tf.convert_to_tensor(dones, dtype=tf.float32)
                dones = np.reshape(dones, [batch_size, 1])
                rewards = np.reshape(rewards, [batch_size, 1])
                """
                observations = tf.convert_to_tensor(au_current_state, dtype=tf.float32)
                next_observations = tf.convert_to_tensor(au_next_state, dtype=tf.float32)
                rewards = tf.convert_to_tensor(au_reward, dtype=tf.float32)
                dones = tf.convert_to_tensor(au_dones, dtype=tf.float32)
                # start = time.clock() #........测试开始时间.........
                l, l1, l2, l3, l_list = Model.network_learn(observations, next_observations, rewards, dones)
                # end = time.clock()  #.........测试结束时间.........
                # print("learn time ", end - start)
                ae_loss += l
                pred_loss += l1
                SKL_loss += l2
                reg_loss += l3
                for n in range(len(l_list)):
                    l_list[n] = l_list[n].numpy()
                    skl = l_list[n]
                    SKL_list[n] += skl
            #..............测试网络.............
            x_test = test_data[0]
            x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
            y_test = test_data[1]
            y_test = np.expand_dims(y_test, axis=1)
            y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

            # start = time.clock() #........测试开始时间.........
            y_pred = Model.call(x_test)
            test_loss = RMSE(y_pred, y_test)

            rep = Model.getReprestation(x_test, len(nn_shape))
            # end = time.clock()  # .........测试结束时间.........
            # print("Model call time ", end - start)
            rep = rep.numpy()
            pop_sparse, num_active_units, _ = sparsity_stat(rep, print_out=False)


            ae_loss = ae_loss/iteration
            pred_loss = pred_loss/iteration
            SKL_loss = SKL_loss/iteration
            reg_loss = reg_loss/iteration
            SKL_list = SKL_list/iteration
            skl_result = ''
            for n in range(len(SKL_list)):
                skl_result += 'SKL{}={:3.5f} '.format(n+1, SKL_list[n])
            SKL_layers_list.append(SKL_list)
            ae_loss_list.append(ae_loss)
            pred_loss_list.append(pred_loss)
            SKL_loss_list.append(SKL_loss)
            reg_loss_list.append(reg_loss)
            test_loss_list.append(test_loss)
            pop_sparse_list.append(pop_sparse)
            num_active_list.append(num_active_units)

            epoch_result = 'Epoch {:3d}: Loss={:9.5f}. pred_loss={:8.5f}. reg_loss={:8.5f}. '.format(j, ae_loss,
                            pred_loss, reg_loss) + skl_result + \
                           'Test_loss={:6.4f}. Sparsity={:.4f}/{:3d}   '.format(test_loss, pop_sparse.mean(), num_active_units)
            t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            logger_mes = epoch_result + t
            print(logger_mes)
            logger.write(logger_mes+'\n')
        #..........训练完一个表示...........
        Model_route = folder_name + '/pre_train_model'
        Model.save_weights(Model_route)
        pre_train_result=[]
        pre_train_result.append(ae_loss_list)
        pre_train_result.append(pred_loss_list)
        pre_train_result.append(SKL_loss_list)
        pre_train_result.append(reg_loss_list)
        pre_train_result.append(SKL_layers_list)
        pre_train_result.append(pop_sparse_list)
        pre_train_result.append(num_active_list)
        pre_train_result.append(test_loss_list)

        result_route = folder_name + '/pre_train_result.pkl'
        with open(result_route, 'wb') as result_file:
            pickle.dump(pre_train_result, result_file, True)
        result_file.close()
        #.................Agent 控制.....................
        for k in range(len(Agent_lr_ls)):
            lr = Agent_lr_ls[k]
            Model.W_v_initial()
            rewards_list = []
            print('sarsa Agent control with lr = {:.5f}'.format(lr))
            logger.write('sarsa Agent control with lr = {:.5f}'.format(lr) + '\n')
            for epi in range(1, episode+1):
                nan_check = Model.check()
                if nan_check > 0:
                    print('Model.w contains nan: break the loop')
                    logger.write('Model.w contains nan: break the loop\n')
                    # rewards_list.append(-1000)
                    break
                #start = time.clock() #........测试开始时间.........
                total_reward = play_sarsa(env, Model, lr=lr, train=True, render=False)
                """
                #下降3
                if epi % 20 == 0:
                    Model.epsilon *= 0.1
                    lr *= 0.1
                """
                """
                # 下降2
                if epi % 25 == 0:
                    Model.epsilon *= 0.4
                    lr *= 0.4
                """
                """
                # 下降1
                Model.epsilon *= 0.97
                lr *= 0.97
                """
                """
                # 下降4
                Model.epsilon *= 0.96
                lr *= 0.96
                """
                """
                # 下降5
                Model.epsilon *= 0.95
                lr *= 0.95
                """
                #end = time.clock()  # .........测试结束时间.........
                #print("play episode ", end - start)

                rewards_list.append(total_reward)
                t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print('Episode {:4d}:  env:{}, rewards={:.2f} '.format(epi, dataset, total_reward), t)
                logger.write('Episode {:4d}:  env:{}, rewards={:.2f} '.format(epi, dataset, total_reward) + t +'\n')
                if use_target and epi % tar_fre == 0:
                    Model.update_target()

            rewards_route = folder_name + '/lr={:.5f} rewards.pkl'.format(Agent_lr_ls[k])
            with open(rewards_route, 'wb') as rewards_file:
                pickle.dump(rewards_list, rewards_file, True)
            rewards_file.close()
        logger.close()
    if num_run >= 10: #循环大于10才画结果图
        plot_result(arg, Agent_lr_ls, absolute_route)
    elif trailnum == 20:
        arg.num_run=20
        arg.thread=1
        plot_result(arg, Agent_lr_ls, absolute_route)






















