import numpy as np
import sys

class Parse_data():
    def __init__(self,file_in):
        self.Y = []
        self.X = []
        with open (file_in,'r') as f:
            line = f.readline()
            while line:
                line = line.strip('\n')
                self.Y.append(float(line[0]))
                raw_X = line.split('\t')[1:]
                for i in range(len(raw_X)):
                    raw_X[i] = raw_X[i].split(':')
                dict_raw_X = {'0':1} # bias term
                for key in dict(raw_X):
                    dict_raw_X[str(int(key)+1)] = float(dict(raw_X)[key])
                self.X.append(dict_raw_X)
                line = f.readline()



class Linear_regression():
    def __init__(self):
        print('Bi-Linear regression!')

    def sparse_dot(self,X,theta):
        product = 0.0
        for i,v in theta.items():
            if i not in X:
                continue
            else:
                product += X[i]*v
        return product

    def sparse_add(self,g, theta):
        for i, v in g.items():
            if i not in theta:
                theta[i] = v
            else:
                theta[i] += v
        return theta

    def sigmoid(self,x):
        return 1.0 / (1.0 + np.exp(-x))

    def classify(self,y):
        if y>0.5:
            return 1.0
        else:
            return 0.0

    def simple_classify(self,y):
        if y>0:
            return 1.0
        else:
            return 0.0

    def grad(self,xi,yi,theta):
        dot = self.sparse_dot(theta,xi)
        return yi-self.sigmoid(dot) #

    def lr_sgd(self,X,Y,epoch,learning_rate=0.1):
        theta = {'0':0} # bias term
        for count in range(epoch):
            print("epoch: ",count+1)
            for i in range(len(X)):
                g = {} # bias term
                # for our specific time, the gradient for every feature are the same as value always equals to 1
                grad_all = learning_rate*self.grad(X[i],Y[i],theta)
                for key in X[i]:
                    g[key] = grad_all
                # print(g)
                self.sparse_add(g,theta)
        yhat = self.predict(X,theta)
        err_rate = self.metrics(Y,yhat)
        return theta, err_rate,yhat

    def predict(self,X,theta):
        yhat_arr = []
        for item in X:
            ycal = self.sparse_dot(item,theta)
            yhat = self.simple_classify(ycal)
            yhat_arr.append(yhat)
        return yhat_arr

    def metrics(self,y,yhat):
        err_rate = 0
        for i in range(len(y)):
            if y[i] != yhat[i]:
                err_rate += 1
        err_rate = err_rate/len(y)
        # print(err_rate)
        return err_rate

if __name__ == '__main__':
    formatted_train_input = sys.argv[1]
    formatted_validation_input = sys.argv[2]
    formatted_test_input = sys.argv[3]
    dict_input = sys.argv[4]
    train_out = sys.argv[5]
    test_out = sys.argv[6]
    metrics_out = sys.argv[7]
    num_epoch = sys.argv[8]


    lr = Linear_regression()
    train_data = Parse_data(formatted_train_input)
    train_X = train_data.X
    train_Y = train_data.Y
    theta, train_err_rate,train_Yhat = lr.lr_sgd(train_X, train_Y, int(num_epoch))

    valid_data = Parse_data(formatted_validation_input)
    valid_X = valid_data.X
    valid_Y = valid_data.Y
    valid_Yhat = lr.predict(valid_X,theta)
    valid_err_rate = lr.metrics(valid_Y,valid_Yhat)

    test_data = Parse_data(formatted_test_input)
    test_X = test_data.X
    test_Y = test_data.Y
    test_Yhat = lr.predict(test_X,theta)
    test_err_rate = lr.metrics(test_Y,test_Yhat)

    print('error(train): {}'.format(train_err_rate))
    print('error(test): {}'.format(test_err_rate))

    with open(train_out,'w') as f:
        for item in train_Yhat:
            f.write('{}\n'.format(item))

    with open(test_out,'w') as f:
        for item in test_Yhat:
            f.write('{}\n'.format(item))

    with open(metrics_out,'w') as f:
        f.write('error(train): {}\n'.format(train_err_rate))
        f.write('error(test): {}\n'.format(test_err_rate))