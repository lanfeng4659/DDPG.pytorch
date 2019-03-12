import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
import gym
import time
DDPG_CONFIG={
'MAX_EP_STEPS':200,
'LR_ACTOR':0.001,
'LR_CRITIC':0.002,
'GAMMA':0.9,
'TAU':0.01,
'MEMORY_CAPACITY':10000,
'BATCH_SIZE':32,

}
MAX_EPISODES = 200
RENDER = False
ENV_NAME = 'Pendulum-v0'
class ActionNet(nn.Module):
    def __init__(self,feature_len, action_range):
        super(ActionNet, self).__init__()
        self.action =  nn.Sequential(
                    nn.Linear(feature_len, 30, bias = False),
                    nn.ReLU(),
                    nn.Linear(30, 1, bias = False),
                    nn.Tanh()
                    )
        self.action_range = action_range
    def forward(self, features):
        return self.action(features).mul(float(self.action_range))
class CriticNet(nn.Module):
    def __init__(self,state_len, action_len):
        super(CriticNet, self).__init__()
        self.state_branch =  nn.Sequential(
                    nn.Linear(state_len, 30, bias = False),
                    )
        self.action_branch =  nn.Sequential(
                    nn.Linear(action_len, 30, bias = False),
                    )
        self.bias = Parameter(torch.Tensor(30))
        self.critic =  nn.Sequential(
                    nn.Linear(30, 1),
                    )
    def forward(self, states, actions):
        values = F.relu(self.state_branch(states) + self.action_branch(actions) + self.bias)
        return self.critic(values)
class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.memory = np.zeros((DDPG_CONFIG['MEMORY_CAPACITY'], s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound

        self.actor_eval = ActionNet(s_dim, a_bound[0])
        self.critic_eval = CriticNet(s_dim, a_dim)

        self.actor_target = ActionNet(s_dim, a_bound[0])    #should not be trained
        self.critic_target = CriticNet(s_dim, a_dim)    #should not be trained

        self.actor_op = torch.optim.Adam(self.actor_eval.parameters(), lr=DDPG_CONFIG['LR_ACTOR'])
        self.critic_op = torch.optim.Adam(self.critic_eval.parameters(), lr=DDPG_CONFIG['LR_CRITIC'])

    def choose_action(self, states):
        states = torch.Tensor(states)
        return self.actor_eval(states).detach().numpy()
    def store_transition(self, states, actions, rewards, states_next):
        transitions = np.hstack((states, actions, [rewards], states_next))
        index = self.pointer % DDPG_CONFIG['MEMORY_CAPACITY']
        self.memory[index, :] = transitions
        self.pointer += 1
    def learn(self):
        indices = np.random.choice(DDPG_CONFIG['MEMORY_CAPACITY'], size = DDPG_CONFIG['BATCH_SIZE'])
        bt = torch.Tensor(self.memory[indices, :])
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]


        action_eval = self.actor_eval(bs)
        q_value_eval = self.critic_eval(bs, action_eval)
        action_loss = -torch.mean(q_value_eval)

        self.actor_op.zero_grad()
        action_loss.backward()
        self.actor_op.step()

        self.actor_target.load_state_dict(self.update_parameters(self.actor_eval.state_dict(), self.actor_target.state_dict(), DDPG_CONFIG['TAU']))
        self.critic_target.load_state_dict(self.update_parameters(self.critic_eval.state_dict(), self.critic_target.state_dict(), DDPG_CONFIG['TAU']))

        action_next    = self.actor_target(bs_)#the target nets' parameters are not in optimizier, it's not neccery to use detach.
        q_value_next   = self.critic_target(bs_, action_next)#the target nets' parameters are not in optimizier, it's not neccery to use detach.
        q_value_target = br + DDPG_CONFIG['GAMMA'] * q_value_next
        q_value_eval = self.critic_eval(bs, ba)
        td_error = torch.mean((q_value_target - q_value_eval)**2)

        self.critic_op.zero_grad()
        td_error.backward()
        self.critic_op.step()
    def update_parameters(self, p_eval, p_target, ratio):
        return {name:p_target[name]*(1-ratio)+p_eval[name]*ratio for name in p_target.keys()}






def main():
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high

    ddpg = DDPG(a_dim, s_dim, a_bound)
    var = 3
    RENDER = False
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        if i > 100:
            RENDER = True
        for j in range(DDPG_CONFIG['MAX_EP_STEPS']):
            if RENDER:
                env.render()

            # Add exploration noise
            a = ddpg.choose_action(s)
            #print(a)
            a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
            s_, r, done, info = env.step(a)

            ddpg.store_transition(s, a, r / 10, s_)

            if ddpg.pointer > DDPG_CONFIG['MEMORY_CAPACITY']:
                var *= .9995    # decay the action randomness
                ddpg.learn()

            s = s_
            ep_reward += r
            if j == DDPG_CONFIG['MAX_EP_STEPS']-1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
                # if ep_reward > -300:RENDER = True
                break

    #print('Running time: ', time.time() - t1)
if __name__ == '__main__':
    main()
