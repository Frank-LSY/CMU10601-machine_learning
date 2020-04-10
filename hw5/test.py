import numpy as np
from neuralnet import Dataprocess,Neuralnet
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':
    train_input = './handout/largeTrain.csv'
    test_input = './handout/largeTest.csv'
    train_out = './a.csv'
    test_out = './b.csv'
    metrics_out = './me.txt'
    num_epoch = 100
    hidden_units = 50
    init_flag = '1'
    learning_rate = 0.01

    ld = Dataprocess()
    data_train = ld.load_data(train_input)
    x_train, y_train = ld.parse_data(data_train)
    data_test = ld.load_data(test_input)
    x_test, y_test = ld.parse_data(data_test)

    nn = Neuralnet()
    len_x = len(x_test[0])-1
    len_y = len(y_test[0])

    x = [0.1,0.01,0.001]
    train = []
    test = []
    for learning_rate in x:
        weight_xh,weight_hy, yhat_train, yhat_test, j_train, j_test = nn.model(x_train=x_train, y_train=y_train,
                                                          x_test=x_test, y_test=y_test,
                                                          num_hidden_units=hidden_units,
                                                          init_strategy=init_flag, num_epochs=num_epoch,
                                                          learning_rate=learning_rate,
                                                          metrics_out=metrics_out, len_x= len_x, len_y=len_y)
        train.append(j_train)
        test.append(j_test)

    with open('./plot_lr.pkl','wb') as f:
        pickle.dump([x,train,test],f)
    plt.plot(x,train,ls='--', marker='o', ms = 1.5,lw=1,label='train cross-entropy')
    plt.plot(x,test,ls='-.', marker='v', ms = 1.5,lw=1,label='test cross-entropy')
    plt.xlabel('Learning Rate')
    plt.ylabel('cross-entropy')
    plt.title('NN Cross Entropy')
    plt.legend()
    plt.savefig('./1.2.3.png')
    plt.show()