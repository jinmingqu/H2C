#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-12-03 20:44:24
# @Author  : ${QU JINMING}
# @Email    : ${qjming97@163.com}

import argparse
import pickle
from collections import namedtuple
from torch.autograd import Variable
from itertools import count
from torch.distributions import Categorical
import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import random
import time
import logging
import copy
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from layers import GraphAttentionLayer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter
import math
from Order import *
from City import *
from Object import *
from ReplayBuffer import *

SEED = 3
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

NUM_AMBS_REAL = 50
NUM_AMBS      = 15
NUM_TIMESTEP  = 144
NUM_NODES     = 37
NUM_ACTIONS   = 5

BUFFER_WEIGHT_1 = 4
BUFFER_WEIGHT_2 = 10
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class EncoderPri(nn.Module):

    def __init__(self, cuda):
        super(EncoderPri, self).__init__()
        #encoder the private state of each node

        self.fc1 = nn.Sequential(
            nn.Linear(NUM_AMBS*10+1, 128),
            nn.LeakyReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(),

        )
        if cuda:
            self.cuda()

    def forward(self, x1, x2):

        x1 = x1.view(-1, 1, NUM_AMBS*10)  # (batch,channel,NUM_AMBS*state_size)
        x2 = x2.view(-1, 1, 1)
        x = torch.cat([x1, x2], dim=-1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout, alpha, nheads, cuda):
        """Dense version of GAT. Reference from https://arxiv.org/abs/1710.10903"""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(
            nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.dropout = nn.Dropout(dropout)
        self.out_att = GraphAttentionLayer(
            nhid * nheads, nout, dropout=dropout, alpha=alpha, concat=False)
        if cuda:
            self.cuda()

    def forward(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = self.dropout(x)
        x = self.out_att(x, adj)
        return x


class LowerActionLayer(nn.Module):
    def __init__(self, cuda):
        super(LowerActionLayer, self).__init__()
        len_state      = 100+NUM_NODES+NUM_TIMESTEP
        len_state_out  = NUM_ACTIONS
        hiddennum      = NUM_ACTIONS

        self.LSTM = nn.LSTM(
            len_state,
            len_state_out,
            num_layers=hiddennum,
            batch_first=True,
        )
        self.softmax = nn.Softmax(dim=2)
        self.fc      = nn.Sequential(
                        nn.Linear(NUM_AMBS*NUM_ACTIONS, 100),
                        nn.LeakyReLU(),
                        nn.Linear(100, NUM_NODES)
        )
        self.use_cuda = cuda
        if self.use_cuda:
            self.cuda()

    def forward(self, x):

        x = x.repeat(1, NUM_AMBS, 1)
        out_n, hidden = self.LSTM(x)
        x = out_n
        x = self.softmax(x)
        return x


class LowerActionModule(nn.Module):

    def __init__(self, nfeat, nhid, nout, dropout, alpha, nheads, cuda):
        super(LowerActionModule, self).__init__()
   
        self.use_cuda = cuda
        # GAT parameters
        self.nfeat = nfeat
        self.nhid = nhid
        self.nout = nout
        self.dropout = dropout
        self.alpha = alpha
        self.nheads = nheads
        ###

        # sub_module initial
        self.encoder_pri = EncoderPri(cuda)
        self.gat         = GAT(nfeat, nhid, nout, dropout, alpha, nheads, cuda)
        self.act_net     = [LowerActionLayer(self.use_cuda) for i in range(NUM_NODES)]
        for i, act in enumerate(self.act_net):
            self.add_module('act_net_{}'.format(i), act)

        if self.use_cuda:
            self.cuda()

    def forward(self, state_pri, state_order, goal, adj, t):

        state_order = torch.FloatTensor(state_order).view(NUM_NODES, -1, 1)
        t           = torch.FloatTensor(t).cuda().view(1, -1, NUM_TIMESTEP).repeat(NUM_NODES, 1, 1)

        #encode the private state
        state_pri_after_code = [self.encoder_pri(
            state_pri[i].cuda(), state_order[i].cuda()) for i in range(len(state_pri))]
        state_pri_after_code = torch.cat(
            [x.view(x.size(0), 1, -1) for x in state_pri_after_code], dim=1)

        # inter-hospital communication by GAT
        state_pri_after_aggragate = torch.cat([self.gat(x, Variable(torch.FloatTensor(
            adj), requires_grad=True).cuda()).view(NUM_NODES, -1, 100) for x in state_pri_after_code], dim=1)
        state_pri_after_aggragate = torch.cat(
            [state_pri_after_aggragate, t], dim=-1)

        #concatenate the aggagated state and goal, then input then into policy network which incudes in-hospital communication 
        value_list = torch.cat([self.act_net[i](torch.cat([state_pri_after_aggragate[i].squeeze(0), Variable(torch.FloatTensor(goal).squeeze(
            0), requires_grad=True).cuda()], dim=-1).view(-1, 1, 100+NUM_NODES+NUM_TIMESTEP)).unsqueeze(3) for i in range(state_pri_after_aggragate.size(0))], dim=3)

        return value_list


class HigherActionLayer(nn.Module):
    def __init__(self, cuda):
        super(HigherActionLayer, self).__init__()
        self.len_state = NUM_NODES+NUM_TIMESTEP  
        self.len_action = NUM_NODES
        self.len_value = 1
        self.len_next_state = NUM_NODES
        self.use_cuda = cuda
        if self.use_cuda:
            self.cuda()

        self.layers_1 = nn.Sequential(
            nn.Linear(self.len_state+self.len_action, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, self.len_next_state)
        )
        self.layers_2 = nn.Sequential(
            nn.Linear(self.len_next_state+self.len_state+self.len_action, 30),
            nn.LeakyReLU(),
            nn.Linear(30, 30),
            nn.LeakyReLU(),
            nn.Linear(30, self.len_value)
        )

    def forward(self, x, y):

        x = x.view(-1, self.len_state)
        y = y.view(-1, self.len_action)

        data   = torch.cat([x, y], dim=-1)
        state  = self.layers_1(data)
        data_2 = torch.cat([data, state], dim=-1)
        value  = self.layers_2(data_2)
        return state, value


class HigherActionModule(nn.Module):
    def __init__(self, cuda):

        super(HigherActionModule, self).__init__()

        self.act_net      = HigherActionLayer(cuda)
        self.chosen_goals = {} # save the chosen goals for each timestep
        self.use_cuda     = cuda
        if self.use_cuda:
            self.cuda()

    def forward(self, x, y):
        x, y = self.act_net(x, y)
        return x, y


class H2C(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout, alpha, nheads, cuda, learning_rate_H, learning_rate_L, momentum, gamma):
        super(H2C, self).__init__()

        self.HigherActor, self.HigherCriticor = HigherActionModule(
            cuda), HigherActionModule(cuda)
        self.LowerActor = LowerActionModule(
            nfeat, nhid, nout, dropout, alpha, nheads, cuda)

        self.learning_rate_H = learning_rate_H
        self.learning_rate_L = learning_rate_L
        self.alpha = 1
        self.lamda = 2
        self.momentum = momentum
        self.gamma    = gamma
        self.explore  = 0.8 #initial exploration rate

        self.optimizer_H = optim.Adam(
            self.HigherActor.parameters(), self.learning_rate_H, betas=[0.9, 0.999])
        self.optimizer_L = optim.SGD(
            self.LowerActor.parameters(), self.learning_rate_L, self.momentum)

        self.loss_func_L = nn.CosineEmbeddingLoss()
        self.loss_func_H = nn.MSELoss()

        self.update_count_H = 0
        self.update_count_L = 0
        #record initialize
        path = "./logging/400_500_12/"
        self.writer_L = SummaryWriter(
            path+'lower/loss/'+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())))
        self.writer_H = SummaryWriter(
            path+'higher/loss/'+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())))
        self.writer_WR = SummaryWriter(
            path+'reward/whole/'+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())))
        self.writer_R = SummaryWriter(
            path+'reward/step/'+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())))
        self.writer_RA = SummaryWriter(
            path+'reward/idle/'+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())))


    def load_model(self,model_path):

        if os.path.exists(model_path):
            print("load model!")
            check_point = torch.load(model_path)
            self.HigherActor.load_state_dict(check_point['HigherActor'])
            self.LowerActor.load_state_dict(check_point['LowerActor'])
            self.HigherCriticor.load_state_dict(
                check_point['HigherCriticor'])
            self.HigherActor.chosen_goals = check_point['chosen_goals']

    def save_model(self, model_path):
 
        if not os.path.exists("./trained_model/"):
            os.mkdir("./trained_model/")
        check_point = {'HigherActor': self.HigherActor.state_dict(),
                       'LowerActor': self.LowerActor.state_dict(),
                       'HigherCriticor': self.HigherCriticor.state_dict(),
                       'chosen_goals': self.HigherActor.chosen_goals,
                       }
        torch.save(check_point, model_path)

    def reshape_input(self, inputs):

        results = []
        for node in inputs.keys():
            results.append(inputs[node])

        return results

    def state_reshape(self, state):
        # reshape private state into same dimension 
        state_final = []
        batch_size = len(state)
        for batch in range(batch_size):
            state_final_one_batch = []
            state_one_batch = state[batch]
            for item in state_one_batch:
                state_node = self.reshape_for_input_step(item)
                state_final_one_batch.append(state_node)
                
            state_final.append(state_final_one_batch)
        return Variable(torch.FloatTensor(state_final).permute(1, 0, 2, 3, 4), requires_grad=True)

    def reshape_for_input_step(self, state_private):
        amb_count = [len(state_private) if len(
            state_private) < NUM_AMBS else NUM_AMBS][0]
        final_state = np.zeros((NUM_AMBS, 10))
        if amb_count != 0:
            final_state[:amb_count, :] = state_private[:amb_count]
        else:
            pass
        final_state = final_state.reshape((1, NUM_AMBS, 10))
        return final_state

    def select_higher_action(self, state_now, bufferhigher, t, goal, flag):
        #select the optimal distribution state as higher action from replay buffer in training process.
        #two pattern could be chosen select.For online training, select the goal not the state.
        update_count = bufferhigher.counts()
        if bufferhigher.counts() == 0:
            return goal
        else:
            state, reward, new_state, done, T, next_T, real_goal = bufferhigher.getBatch(update_count)
            
            states, values = self.HigherActor(Variable(torch.cat([torch.FloatTensor(state_now).view(-1, NUM_NODES),
                                                                  torch.FloatTensor(t).view(-1, NUM_TIMESTEP)], dim=1).repeat(torch.FloatTensor(real_goal).shape[0], 1).cuda().view(-1, NUM_NODES+NUM_TIMESTEP, 1)), Variable(torch.FloatTensor(real_goal).cuda().view(-1, NUM_NODES, 1)))
            states2, values2 = self.HigherActor(Variable(torch.cat([torch.FloatTensor(state_now).view(-1, NUM_NODES),
                                                                    torch.FloatTensor(t).view(-1, NUM_TIMESTEP)], dim=1).repeat(torch.FloatTensor(state).shape[0], 1).cuda().view(-1, NUM_NODES+NUM_TIMESTEP, 1)), Variable(torch.FloatTensor(state).cuda().view(-1, NUM_NODES, 1)))
            max_index = values.max(0)[1].data[0].cpu().numpy().tolist()
            max_index_2 = values2.max(0)[1].data[0].cpu().numpy().tolist()

            if np.random.random(1) > 0:
                goal = state[max_index_2]
            else:
                goal = real_goal[max_index]

            if flag: #flag=1 if online training
                goal = real_goal[max_index]

            return goal

    def update_learningrate(self):

        self.explore = self.explore*1.01


    def select_lower_action(self, state_pri, state_order, adj, idle_num, goal, n_round, online_times, flag, T):
        # preprocesing for private state
        state_node = [state_pri]
        state_pri  = self.state_reshape(state_node)
        # output the probability of actions for agents in each node
        value_list = self.LowerActor(
            state_pri, state_order, goal, adj, T)
        #select the optimal actions
        action = value_list.max(2)[1].permute(2, 1, 0).data.cpu(
        ).numpy().tolist()
        # cut out the idle agents' action
        action_list = [action[i][:idle_num[i]] if idle_num[i] <=
                       NUM_AMBS else action[i][:idle_num[i]] for i in range(len(idle_num))]
        # exploration with 1- self.explore
        if n_round < online_times:
            for j in range(len(action_list)):
                if len(action_list[j]) > 0:
                    for z in range(len(action_list[j])):
                        if np.random.random(1) >= self.explore:
                            action_list[j][z] = [
                                np.random.choice(range(NUM_ACTIONS), 1).item()]
                        else:
                            pass
                else:
                    pass
        else:
            pass

        return action_list

    def update_actor(self, buffers, batch_size, adj):
        state_pri, idle_driver, action, goal, state_higher, next_state_higher, reward, done, T, next_T, order_num, next_goal = buffers.getBatch(
            batch_size)
        #sample one batch of tuple to update
        loss = self.update_higher_actor(
            state_higher, reward, next_state_higher, done, T, next_T, goal, next_goal)
        self.update_lower_actor(state_pri, order_num, idle_driver, action, 
                                goal, state_higher, next_state_higher, adj, loss, batch_size, T)

        if self.update_count_L %288 ==0:
            self.update_learningrate()

    def update_higher_actor(self, state, reward, new_state, done, T, next_T, goal, next_goal):

 
        state      = torch.FloatTensor(state).cuda().view(-1, NUM_NODES, 1)
        reward     = Variable(torch.FloatTensor(
            reward).cuda().view(-1, 1), requires_grad=True)
        done       = Variable(torch.FloatTensor(
            done).cuda().view(-1, 1), requires_grad=True)
        next_state = Variable(torch.FloatTensor(
            new_state).cuda().view(-1, NUM_NODES, 1), requires_grad=True)
        GAMMA      = torch.Tensor(
            np.ones((next_state.size(0), 1))*self.gamma).cuda()

        T          = torch.FloatTensor(T).cuda().view(-1, NUM_TIMESTEP, 1)
        next_T     = torch.FloatTensor(next_T).cuda().view(-1, NUM_TIMESTEP, 1)

        state      = Variable(
            torch.cat([state, T], dim=1).view(-1, NUM_TIMESTEP+NUM_NODES, 1), requires_grad=True)
        new_state  = Variable(
            torch.cat([next_state, next_T], dim=1).view(-1, NUM_TIMESTEP+NUM_NODES, 1), requires_grad=True)

        goal       = Variable(torch.FloatTensor(
            goal).cuda().view(-1, NUM_NODES, 1), requires_grad=True)
        next_goal  = Variable(torch.FloatTensor(
            next_goal).cuda().view(-1, NUM_NODES, 1), requires_grad=True)

        predict_state, value           = self.HigherActor(state, goal)
        predict_next_state, next_value = self.HigherCriticor(
            new_state, next_goal)

        expect_value  = reward + torch.mul(torch.mul(GAMMA, next_value), done)
        TD_ERROR      = expect_value - value
        predict_state = predict_state.view(-1, NUM_NODES, 1)
        target        = torch.ones_like(reward).cuda()
        
        loss_value = self.loss_func_H(value, expect_value)
        loss_state = self.loss_func_L(predict_state, next_state, target)
        loss = loss_state*self.alpha +loss_value

        self.writer_H.add_scalar(
            'loss/value_loss_higher', loss, self.update_count_H)
        self.writer_H.add_scalar(
            'loss/value_loss_value', loss_value, self.update_count_H)
        self.writer_H.add_scalar(
            'loss/value_loss_state', loss_state, self.update_count_H)

        self.optimizer_H.zero_grad()
        loss.backward()
        self.optimizer_H.step()
        self.update_count_H += 1

        if self.update_count_H % 50 == 0:
            self.HigherCriticor.load_state_dict(self.HigherActor.state_dict())

        return TD_ERROR

    def update_lower_actor(self, state_pri, order_num, idle_driver, action, goal, state_higher, next_state_higher, adj, loss_higher, batch_size, T):

        

        state_pri= self.state_reshape(state_pri)
        value_list = self.LowerActor(
            state_pri, order_num, goal, adj, T).permute(0, 1, 3, 2)

        next_state_higher        = np.array(next_state_higher) - np.array(state_higher)
        goal                     = np.array(goal) - np.array(state_higher)
       
        next_state_higher_vec = Variable(torch.FloatTensor(
            next_state_higher).cuda().view(-1, NUM_NODES, 1), requires_grad=True)
        goal_vec = Variable(torch.FloatTensor(
            goal).cuda().view(-1, NUM_NODES, 1), requires_grad=True)
        target = torch.ones_like(goal_vec).cuda()[:, 1, :].view(-1, 1)
        grad_loss = self.loss_func_L(next_state_higher_vec, goal_vec, target)

        #mask
        action_space = []
        for batch in range(batch_size):
            action_space_batch = []
            action_batch = action[batch]
            for act_node in action_batch:
                action_space_node = np.zeros(NUM_AMBS*NUM_ACTIONS)
                for i in range(len(act_node)):
                    action_space_node[i*NUM_ACTIONS+act_node[i][0]] += 1
                action_space_node = [True if x ==
                                     1 else False for x in action_space_node]
                action_space_batch.append(action_space_node)
            action_space.append(action_space_batch)
        action_space = torch.ByteTensor(action_space).cuda()
        action_space = action_space.squeeze(2)

        log_action = torch.log(value_list).view(-1, NUM_NODES, NUM_AMBS*NUM_ACTIONS)

        loss = -1*(log_action.masked_select(action_space) *
                   (loss_higher.detach()-self.lamda*grad_loss.detach())).mean()

        self.writer_L.add_scalar(
            'loss/value_loss_lower', loss, self.update_count_L)
        self.writer_L.add_scalar(
            'loss/value_loss_lower_TD', loss_higher.mean(), self.update_count_L)
        self.writer_L.add_scalar(
            'loss/value_loss_lower_GRAD', grad_loss, self.update_count_L)

        self.optimizer_L.zero_grad()
        loss.backward()
        self.optimizer_L.step()
        self.update_count_L += 1



def test(env):

    n_test_round = 20
    step_higher = 12

    model_path = "./trained_model/model_400_50_12checkpoint.pth"
    writer = SummaryWriter('./logging/400_500_12/reward/test/' +
                           time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())))
    model = H2C(nfeat=128, nhid=100, nout=100, dropout=0.1, alpha=0.05,
                                     nheads=2, cuda=True, learning_rate_H=0.005, learning_rate_L=0.01, momentum=0.78, gamma=0.8)
    model.load_model(model_path=model_path)
    public_state_last_round = np.random.randint(1, 20, size=(NUM_TIMESTEP))

    for j in range(n_test_round):

        print("round {}".format(j))

        env.reset()
        env.step_ini()
        state_private = env.get_observation_private()
        state_statistic = env.get_observation_statistic()

        select_higher_action_T = [
            env.StatetoOneHot([i], NUM_TIMESTEP) for i in range(NUM_TIMESTEP)]

        for i in range(NUM_TIMESTEP-1):
            # print(state_private)
            if i % 30 == 0:
                print("step {}".format(i))
            else:
                pass

            T = select_higher_action_T[i]
            next_T = select_higher_action_T[i+1]

            if i % step_higher == 0:

                goal = model.HigherActor.chosen_goals[math.floor(
                    i/step_higher)]
            else:
                pass

            joint_action = model.select_lower_action(state_private['state_pri'], state_private['order_num'],
                                                    env.adj, state_private['idle_driver'], goal, j, online_times=0, flag=False, T=T)
       
            reward, next_state_pri, next_state_sta, done = env.step(
                joint_action)
            state_private = next_state_pri
            state_statistic = next_state_sta

        whole_reward = sum(
            env.final_response_rate['finish'])/sum(env.final_response_rate['all'])
        writer.add_scalar('Whole_value/reward', whole_reward, j)


def train(env):

    batch_size = 50
    n_round = 200
    step_higher = 12
    reward_count = 0
    step_flag = 0
    online_times = 60
    model = H2C(nfeat=128, nhid=100, nout=100, dropout=0.1, alpha=0.05,
                                     nheads=2, cuda=True, learning_rate_H=0.005, learning_rate_L=0.01, momentum=0.78, gamma=0.8)

    replay_buffer_lower = ReplayBufferLowLayer(288)
    replay_buffer_better = ReplayBufferLowLayer(288)
    replay_buffer_higher = [ReplayBufferHighLayer(
        40) for i in range(int(NUM_TIMESTEP/step_higher))]

    amb_index = env.amb_index()
    node_index = env.node_index()
    select_higher_action_T = [env.StatetoOneHot([i], NUM_TIMESTEP) for i in range(NUM_TIMESTEP)]

    for j in range(n_round):
        tauple_temp = []
        print("round {}".format(j))

        env.reset()
        env.step_ini()
        state_private = env.get_observation_private()
        state_statistic = env.get_observation_statistic()

        goal = state_statistic+np.ones_like(state_statistic)*0.0002

        if j > online_times:
            action_flag = True
        else:
            action_flag = False

        for i in range(NUM_TIMESTEP-1):
    
            if i % 30 == 0:
                print("step {}".format(i))
            else:
                pass

            T = select_higher_action_T[i]
            next_T = select_higher_action_T[i+1]
            ## higher action selection
            if i % step_higher == 0:
                goal = model.select_higher_action(state_statistic, replay_buffer_higher[math.floor(
                    i/step_higher)], select_higher_action_T[i+step_higher-1], goal, action_flag)
                next_goal = goal
                if math.floor(i/step_higher) in model.HigherActor.chosen_goals.keys():
                    goal = model.HigherActor.chosen_goals[math.floor(
                        i/step_higher)]
                    next_goal = goal
                else:
                    pass
            elif i % step_higher == step_higher-1 and i != 143:
                if math.floor(i/step_higher)+1 in model.HigherActor.chosen_goals.keys():
                    next_goal = model.HigherActor.chosen_goals[math.floor(
                        i/step_higher)+1]
                else:
                    next_goal = model.select_higher_action(state_statistic, replay_buffer_higher[math.floor(
                        i/step_higher)+1], select_higher_action_T[i+step_higher], goal, action_flag)
            else:
                pass
            #lower action selection
            joint_action = model.select_lower_action(state_private['state_pri'], state_private['order_num'],
                                                      env.adj, state_private['idle_driver'], goal, j, online_times, flag=False, T=T)

            reward, next_state_pri, next_state_sta, done = env.step(joint_action)

            if type(reward) != bool:
                model.writer_R.add_scalar('value/reward', reward, reward_count)
                model.writer_RA.add_scalar(
                    'value/busy_rate', 1-env.geting_idle_rate(), reward_count)
                model.writer_R.add_histogram('car_distribution', np.array(
                    state_statistic)*NUM_AMBS_REAL, reward_count)

                reward_count += 1
                replay_buffer_lower.add(state_private['state_pri'], state_private['idle_driver'], joint_action,
                                 goal, state_statistic, next_state_sta, reward, done, T, next_T, state_private['order_num'], next_goal)
                tauple_temp.append([state_private['state_pri'], state_private['idle_driver'], joint_action,
                                     goal, state_statistic, next_state_sta, reward, done, T, next_T, state_private['order_num'], next_goal])
                #future distribution state
                if i % step_higher == step_higher-1:
                    replay_buffer_higher[math.floor(
                        i/step_higher)].add(state_statistic, reward, next_state_sta, done, T, next_T, goal)

            state_private = next_state_pri
            state_statistic = next_state_sta

            if replay_buffer_lower.counts() > batch_size:
                model.update_actor(replay_buffer_lower, batch_size, env.adj)

        whole_reward = sum(
            env.final_response_rate['finish'])/sum(env.final_response_rate['all'])
        step_flag = max([whole_reward, step_flag])

        if whole_reward >= step_flag:
            ##weighted update for optimal experience
            count = 0
            if j > online_times:
                model.save_model(
                    model_path="./trained_model/model_400_50_12{}checkpoint.pth".format(step_flag))

            for item in tauple_temp:

                for h in range(BUFFER_WEIGHT_1):
                    replay_buffer_better.add(item[0], item[1], item[2], item[3], item[4], item[5],
                                             item[6], item[7], item[8], item[9], item[10], item[11])
                    if count % step_higher == step_higher-1:
                        for w in range(BUFFER_WEIGHT_2):
                            replay_buffer_higher[math.floor(
                                count/step_higher)].add(item[4], item[6], item[5], item[7], item[8], item[9], item[3])

                count += 1

            for step in range(BUFFER_WEIGHT_1*BUFFER_WEIGHT_2):
                if replay_buffer_better.counts() > batch_size:
                    model.update_actor(replay_buffer_better,
                                       batch_size, env.adj)
        for step in range(NUM_TIMESTEP-1):
            if step % step_higher == 0:
                goal = model.select_higher_action(tauple_temp[step][4], replay_buffer_higher[math.floor(
                    step/step_higher)], select_higher_action_T[step+step_higher-1], goal, action_flag)
                model.HigherActor.chosen_goals[math.floor(
                    step/step_higher)] = goal
            else:
                pass

        model.writer_WR.add_scalar('Whole_value/reward', whole_reward, j)


    model.save_model(
        model_path="./trained_model/model_400_50_12checkpoint.pth")


def main():
    env = Nanjing()
    print("*"*10, "start training model!", "*"*10)
    train(env)
    print("*"*10, "start testing model!", "*"*10)
    test(env)


if __name__ == '__main__':
    main()

