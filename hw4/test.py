import matplotlib.pyplot as plt
import lr
import numpy as np
import math
import pickle

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

if __name__ == '__main__':
    mod1_train_input = './handout/largeoutput/model1_formatted_train.tsv'
    mod1_validation_input = './handout/largeoutput/model1_formatted_valid.tsv'
    mod1_test_input = './handout/largeoutput/model1_formatted_test.tsv'

    mod2_train_input = './handout/largeoutput/model2_formatted_train.tsv'
    mod2_validation_input = './handout/largeoutput/model2_formatted_valid.tsv'
    mod2_test_input = './handout/largeoutput/model2_formatted_test.tsv'

    train = []
    valid = []
    x_axis = []
    for num_epoch in range(1,201):
        model = lr.Linear_regression()
        train_data = lr.Parse_data(mod1_train_input)
        train_X = train_data.X
        train_Y = train_data.Y
        theta, train_err_rate,train_Yhat = model.lr_sgd(train_X, train_Y, int(num_epoch))

        valid_data = lr.Parse_data(mod1_validation_input)
        valid_X = valid_data.X
        valid_Y = valid_data.Y
        valid_Yhat = model.predict(valid_X,theta)

        j_train = 0
        for i in range(len(train_Y)):
            pos = sigmoid(model.sparse_dot(train_X[i],theta))**train_Y[i]
            neg = (1-sigmoid(model.sparse_dot(train_X[i],theta)))**(1-train_Y[i])
            # print('pos:',pos)
            # print('neg:',neg)
            j_train += math.log(pos)+math.log(neg)
        j_train = -j_train/len(train_Y)
        # print('train: ',j_train)
        train.append(j_train)

        j_valid = 0
        for i in range(len(valid_Y)):
            pos = sigmoid(model.sparse_dot(valid_X[i],theta))**valid_Y[i]
            neg = (1-sigmoid(model.sparse_dot(valid_X[i],theta)))**(1-valid_Y[i])
            j_valid += math.log(pos)+math.log(neg)
        j_valid = -j_valid/len(valid_Y)
        valid.append(j_valid)

        x_axis.append(num_epoch)

    with open('./mod1.pkl','wb') as f:
        pickle.dump([x_axis,train,valid],f)
    plt.plot(x_axis,train,ls='--', marker='o', ms = 1.5,lw=1,label='train neg-log-likelihood')
    plt.plot(x_axis,valid,ls='-.', marker='v', ms = 1.5,lw=1,label='valid neg-log-likelihood')
    plt.xlabel('Epoch')
    plt.ylabel('neg-log-lokelihood')
    plt.title('model1 neg-log-likelihood')
    plt.legend()
    plt.savefig('../1.4.1.png')
    plt.show()