import matplotlib.pyplot as plt
import numpy as np

from loadData import LoadData
from decisionTree import Node, DecisionTree, Evaluate
from inspection import Inspection

if __name__ == '__main__':
    train_input = '../handout/education_train.tsv'
    test_input = '../handout/education_test.tsv'
    train_output = '../result/education_train.labels'
    test_output = '../result/education_test.labels'

    ld = LoadData()
    dataset = ld.load_data(train_input)
    dt = DecisionTree(ld)
    tr_err = []
    te_err = []
    x_arr = []
    print(ld.head)
    for i in range(len(ld.head)):
        root = dt.construct(dataset, i)
        # dt.traverse(root)
        dt.classify(ld.load_data(train_input),root,train_output)
        dt.classify(ld.load_data(test_input), root, test_output)
        with open(train_output,'r') as f:
            predcol = f.read().splitlines()
        realcol = np.loadtxt(train_input, dtype=str, delimiter='\t', skiprows=1)[:,-1]
        eva_train = Evaluate(realcol,predcol)
        train_errate = eva_train.error_rate()
        with open(test_output,'r') as f:
            predcol = f.read().splitlines()
        realcol = np.loadtxt(test_input, dtype=str, delimiter='\t', skiprows=1)[:, -1]
        eva_test = Evaluate(realcol,predcol)
        test_errate = eva_test.error_rate()
        print('max depth:',i)
        print('train:',train_errate)
        print('test: ',test_errate)
        print('\n')
        x_arr.append(i)
        tr_err.append(train_errate)
        te_err.append(test_errate)

    # plt.plot(x_arr,tr_err, ls='--', marker='o', label='Train Error Rate')
    # plt.plot(x_arr,te_err, ls='-.', marker='v', label='Test Error Rate')
    # plt.xlabel('Depth')
    # plt.ylabel('Error Rate')
    # plt.title('Politician Train/Test Error Rate')
    # plt.legend()
    # plt.savefig('./1.3.2.png')
    # plt.show()

