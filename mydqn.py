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
        self.net, self.target_net = Net(), Net()
        self.memory = np.zeros((mem_capacity, state_num * 2 + 2))
        self.learn_step = 0
        self.mem_cnt = 0
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()
    
    def choose_act(self, s):
        if np.random.uniform() < epsilon:
            s = torch.unsqueeze(torch.FloatTensor(s), 0)
            actions = self.net(s)
            action = torch.max(actions, 1)[1].numpy()
            action = action[0]
        else:
            action = np.random.randint(0, action_num)
        return action
    
    def store(self, s, a, r, ss):
        transition = np.hstack((s, a, r, ss))
        self.memory[self.mem_cnt % mem_capacity, :] = transition
        self.mem_cnt += 1
    
    def learn(self):
        if self.learn_step % upd_num == 0:
            self.target_net.load_state_dict(self.net.state_dict())
        self.learn_step += 1

        index = np.random.choice(mem_capacity, batch_size)
        transitions = self.memory[index, :]
        s = torch.FloatTensor(transitions[:, :state_num])
        a = torch.LongTensor(transitions[:, state_num].astype(int))
        r = torch.FloatTensor(transitions[:, state_num + 1])
        ss = torch.FloatTensor(transitions[:, -state_num:])
        
        a = torch.unsqueeze(a, 1)
        r = torch.unsqueeze(r, 1)
        q = self.net(s).gather(1, a)
        q_next = self.target_net(ss).detach()
        q_target = r + gamma * q_next.max(1)[0].view(batch_size, 1)
        loss = self.loss_func(q, q_target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

def main():
    dqn = DQN()
    score = deque(maxlen=50)
    for episode in range(1, episodes + 1):
        # print('Episode: \t %s' % episode)
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
            dqn.store(s, a, r, ss)
            reward_sum += r
            s = ss
            
            if dqn.mem_cnt > batch_size:
                dqn.learn()

            if done:
                score.append(reward_sum)
                if episode % 50 == 0:
                    print('episode_50---reward_mean: %s' % (round(np.mean(score), 2)))
                break        

        if np.mean(score) > 550 and episode > 400:
            # torch.save(dqn.net, './model.pth')
            break
    print('----------finished---------')

if __name__ == '__main__':
    main()