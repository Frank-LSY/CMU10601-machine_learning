from environment import MountainCar
from q_learning import Q_learning
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    x1 = range(2000)
    x2 = range(25,2000)
    qlearning = Q_learning('raw')
    raw_reward,_,_ = qlearning.q_learning(episodes=2000, max_iterations=200, epsilon=0.05, gamma=0.999,learning_rate=0.001)
    mean_raw = []
    for i in range(25,2000):
        mean_raw.append(sum(raw_reward[i-25:i])/25)
    plt.figure(figsize=(25, 6))
    plt.plot(x1,raw_reward,ls='--', marker='o', ms = 2.0,lw=1.2,label='sum of all rewards in an episode')
    plt.plot(x2,mean_raw,ls='-.', marker='v', ms = 2.0,lw=1.2,label='rolling mean over a 25 episode window')
    plt.xlabel('episodes')
    plt.ylabel('sum of all rewards')
    plt.title('Raw rewards plot',fontsize=36)
    plt.legend(loc='upper left',prop = {'size':30})
    plt.xticks(np.arange(0, 2001, 10),rotation=90,fontsize=10)
    plt.savefig('./1.4.1.png',bbox_inches='tight',pad_inches=0.0)
    # plt.show()

    plt.cla()
    x3 = range(400)
    x4 = range(25,400)
    qlearning = Q_learning('tile')
    tile_reawrd,_,_ = qlearning.q_learning(episodes=400, max_iterations=200, epsilon=0.05, gamma=0.99, learning_rate=0.00005)
    mean_tile = []
    for i in range(25,400):
        mean_tile.append(sum(tile_reawrd[i-25:i])/25)
    plt.figure(figsize=(16, 10))
    plt.plot(x3,tile_reawrd,ls='--', marker='o', ms = 2.0,lw=1.2,label='sum of all rewards in an episode')
    plt.plot(x4,mean_tile,ls='-.', marker='v', ms = 2.0,lw=1.2,label='rolling mean over a 25 episode window')
    plt.xlabel('episodes')
    plt.ylabel('sum of all rewards')
    plt.title('Tile rewards plot',fontsize=36)
    plt.legend(loc='upper left',prop = {'size':30})
    plt.xticks(np.arange(0, 401, 5),rotation=90)
    plt.savefig('./1.4.2.png',bbox_inches='tight',pad_inches=0.0)
    # plt.show()