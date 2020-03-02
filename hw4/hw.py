import numpy as np
import math
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

if __name__ == '__main__':
    theta = [1.5,2,1,2,3]
    theta = np.array(theta)
    x = [[0,0,1,0,1],
         [0,1,0,0,0],
         [0,1,1,0,0],
         [1,0,0,1,0]]
    x = np.array(x)
    y = [0,1,1,0]
    y = np.array(y)

    # sum = 0
    # for i in range(4):
    #     pos = sigmoid(np.dot(theta.T,x[i]))**y[i]
    #     neg = (1-sigmoid(np.dot(theta.T,x[i])))**(1-y[i])
    #     print('pos',pos)
    #     print('neg',neg)
    #     sum += math.log(pos)+math.log(neg)
    # sum = -sum
    # #
    # print(sum)
    # a = math.log(1-sigmoid(4))+math.log(sigmoid(2))+math.log(sigmoid(3))+math.log(1-sigmoid(3.5))
    # print(a)
    # sum = 0
    # for i in range(4):
    #     print(i,x[i][4]*(sigmoid(np.dot(theta.T,x[i]))-y[i]))
    #     sum += x[i][4]*(sigmoid(np.dot(theta.T,x[i]))-y[i])
    # print(sum)

    # print(sigmoid(1*2.1666+1*0.0654))
    mortality = np.arange(0,1.01,0.02)
    a_s_i = 0.076
    n_f_i = 0.004
    a_s_l = 0.09
    die_i_tPA = np.zeros(51)
    die_w_tPA = np.zeros(51)
    for i,p_ami in enumerate(mortality):
        # print(mortality)
        die_i_tPA[i] = p_ami*a_s_i + (1-p_ami)*n_f_i
        die_w_tPA[i] = p_ami*a_s_l

    plt.plot(mortality,die_i_tPA,ls='--', marker='o', ms = 1.5,lw=1,label='immediate tPA')
    plt.plot(mortality,die_w_tPA,ls='-.', marker='v', ms = 1.5,lw=1,label='wait and see')
    plt.xlabel('AMI probability')
    plt.ylabel('Die probability')
    plt.title('AMI-Die')
    plt.legend()
    plt.savefig('./1.png')
    plt.show()