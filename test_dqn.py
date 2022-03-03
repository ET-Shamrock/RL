from collections import deque
import torch
import torch.tensor as tensor
import gym
import torch.nn as nn
import numpy as np

episodes = 4000
batch_size = 32
gamma = 0.9
lr = 0.01
epsilon = 0.9
mem_capacity = 2000  #memory的容量
upd_num = 100        #更新参数次数

env = gym.make('CartPole-v0').unwrapped
state_num = env.observation_space.shape[0]
action_num = env.action_space.n 


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_num, 50),
            nn.ReLU(),
            nn.Linear(50, action_num)
        )
        nn.init.normal_(self.fc[0].weight, 0, 0.1) 
        nn.init.normal_(self.fc[2].weight, 0, 0.1) 

    def forward(self, x):
        x = self.fc(x)
        return x

class DQN(object):
    def __init__(self) -> None:
        self.net = torch.load('./model.pth')
    
    def choose_act(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        actions = self.net(s)
        action = torch.max(actions, 1)[1].numpy()
        action = action[0]
        return action

def main():
    dqn = DQN()
    reward_sum = 0
    s = env.reset()
    while True:
        env.render()
        a = dqn.choose_act(s)
        ss, r, done, info = env.step(a)

        x, x_dot, theta, theta_dot = ss
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2
        reward_sum += r
        s = ss

        if done:
            print('reward_sum: %s' % (round(reward_sum, 2)))
            break  

if __name__ == '__main__':
    main()