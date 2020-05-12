from environment import MountainCar
import sys
import numpy as np

class Q_learning():
    def __init__(self,mode):
        self.env = MountainCar(mode)
        # self.env.seed(1)

    def action_wise_state(self,state,action):
        l = len(state)
        res = []
        for i in range(3):
            if i==action:
                res.append(state)
            else:
                res.append([0]*l)
        return np.array(res)

    def sparse_to_dense(self,d):
        if self.env.mode == 'raw':
            res = [0]*self.env.state_space
        if self.env.mode == 'tile':
            res = [0]*self.env.state_space
        for k,v in d.items():
            res[k] = v

        return res

    def grad(self,ep_reward, state, next_state, theta, action, bias, gamma, learning_rate):
        dot = np.dot(state[action],theta[action]) + bias
        target = ep_reward + gamma * max([np.dot(next_state[action],theta[0])+bias,np.dot(next_state[action],theta[1])+bias,np.dot(next_state[action],theta[2])+bias])
        td_err = learning_rate*(dot-target)
        td_arr = [[0]*len(state[0])]*3
        td_arr[action] = [td_err]*len(state[0])
        res = np.array(td_arr)*state
        return res,td_err


    def take_epilon_greedy_action(self, q_s_a_theta_p, epsilon):
        if np.random.random() < epsilon:
            action = np.random.randint(0,3)
        else:
            rewards = np.array(q_s_a_theta_p)
            action = np.argmax(rewards)
        # print(action)
        return action

    def q_learning(self,episodes,max_iterations,epsilon,gamma,learning_rate):
        # initialize theta
        l = self.env.state_space
        theta = np.array([[0]*l]*3)
        # print(theta)
        bias = 0 # bias term
        fi_reward = []
        for i in range(episodes):
            ep_reward = 0
            state = self.sparse_to_dense(self.env.reset())
            for j in range(max_iterations):
                q_s_a_theta_p = [np.dot(state,theta[0])+bias,np.dot(state,theta[1])+bias,np.dot(state,theta[2])+bias]
                action = self.take_epilon_greedy_action(q_s_a_theta_p, epsilon)
                next_state, reward, done= self.env.step(action) # receive example
                ep_reward += reward

                grad_theta = self.grad(reward,self.action_wise_state(state,action),self.action_wise_state(self.sparse_to_dense(next_state),action),theta,action,bias, gamma, learning_rate)
                theta = theta - grad_theta[0]
                bias = bias - grad_theta[1]
                state = self.sparse_to_dense(next_state)
                if done:
                    break
            fi_reward.append(ep_reward)

        # print(theta[0])
        # print(fi_reward)
        return fi_reward,theta,bias

    def write_result(self,result,weight_out,returns_out):
        with open (returns_out,'w') as f:
            for item in result[0]:
                f.write('{}\n'.format(item))
        with open(weight_out,'w') as f:
            f.write('{}\n'.format(result[2]))
            for row in result[1].T:
                for col in row:
                    f.write('{}\n'.format(col))

def main(args):
    mode = args[1]
    weight_out = args[2]
    returns_out = args[3]
    episodes = int(args[4])
    max_iterations = int(args[5])
    epsilon = float(args[6])
    gamma = float(args[7])
    learning_rate = float(args[8])
    qlearning = Q_learning(mode)
    result = qlearning.q_learning(episodes=episodes,max_iterations=max_iterations,epsilon=epsilon,gamma=gamma,learning_rate=learning_rate)
    qlearning.write_result(result,weight_out=weight_out,returns_out=returns_out)

if __name__ == "__main__":
    main(sys.argv)