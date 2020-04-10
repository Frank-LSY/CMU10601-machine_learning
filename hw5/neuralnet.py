import numpy as np
import sys

class Dataprocess():
    def load_data(self,file_in):
        with open(file_in, 'r', encoding='utf-8') as f:
            file = np.loadtxt(f, dtype=float, delimiter=',')
        return file

    def parse_data(self,dataset):
        self.x = dataset[:,1:]
        self.y = dataset[:,0]

        # embedidng bias into x
        bias_x = np.ones([len(dataset),1])
        self.x = np.hstack((bias_x,self.x))

        # one-hot coding y
        self.y = np.eye(10)[self.y.astype('int64')]
        return self.x, self.y

class Neuralnet():
    def sigmoid(self,x):
        return 1.0 / (1.0 + np.exp(-x))

    def softmax(self,x):
        exp_x = np.exp(x)
        softmax_x = exp_x/sum(exp_x)
        return softmax_x

    def init_params(self,num_hidden_units,init_strategy,len_x,len_y):
        # embedded the bias term into the weight
        if init_strategy== '1':
            weight_xh = np.random.uniform(-0.1,0.1,size=(len_x+1,num_hidden_units))
            weight_hy = np.random.uniform(-0.1,0.1,size=(num_hidden_units+1,len_y))
        elif init_strategy== '2':
            weight_xh = np.zeros([len_x+1,num_hidden_units])
            weight_hy = np.zeros([num_hidden_units+1,len_y])

        return weight_xh,weight_hy

    def J(self,y,y_hat):
        return -np.dot(y,np.log(y_hat))

    def cross_entropy(self,y,y_hat):
        mul = np.dot(y,np.log(y_hat).T)
        add = sum(mul.diagonal())
        return -add/len(y)

    def nnforward(self,x,weight_xh,weight_hy):
        # α.T*X+b
        h_ = np.dot(weight_xh.T,x)

        # sigmoid
        h = self.sigmoid(h_)

        # insert bias for hidden layer
        h_b = np.insert(h, 0, 1, 0)

        # β.T*h+b
        y_ = np.dot(weight_hy.T, h_b)
        # softmax
        y_hat = self.softmax(y_)
        return y_hat,h_b


    def nnbackward(self,x,h_b,weight_hy,y_hat,y):
        djdy_ = np.array([y_hat-y]) # δJ/δy
        # print(djdy_)
        dy_dw_hy = np.array([h_b])
        # print('djdy:',djdy_.shape)
        g_weight_hy= np.dot(djdy_.T,dy_dw_hy).T # δJ/δweight_hy
        # print(g_weight_hy.shape)
        dy_dh_b = weight_hy # δy/δh_b
        dh_bdh = h_b[1:]*(1-h_b[1:]) # δh_b/δh: d_sigmoid
        dh_dw_xh = np.array([x]) # δh/δweight_xh

        djdh_b = np.dot(djdy_,dy_dh_b[1:].T)
        djdh_ = djdh_b * dh_bdh
        g_weight_xh = np.dot(djdh_.T,dh_dw_xh).T # δJ/δweight_xh
        # print(g_weight_xh.shape)
        return g_weight_xh, g_weight_hy

    def model(self, x_train, y_train, x_test, y_test, num_hidden_units,init_strategy, num_epochs,learning_rate, metrics_out, len_x = 128,len_y = 10):
        weight_xh, weight_hy= self.init_params(num_hidden_units,init_strategy,len_x,len_y)
        n = len(x_train)
        with open (metrics_out,'w') as f:
            for _ in range(num_epochs):
                for i in range(n):
                    # feed forward
                    y_hat,h_b = self.nnforward(x_train[i],weight_xh,weight_hy)

                    # j = self.J(y_train[i],y_hat) # cross entropy
                    # print(j)

                    g_weight_xh, g_weight_hy= self.nnbackward(x_train[i],h_b,weight_hy,y_hat,y_train[i])# gradient
                    weight_xh = weight_xh-learning_rate*g_weight_xh
                    weight_hy = weight_hy-learning_rate*g_weight_hy

                y_train_hat_arr = self.valid(x_train,y_train,weight_xh,weight_hy)
                j_train = self.cross_entropy(y_train,y_train_hat_arr)

                y_test_hat_arr = self.valid(x_test, y_test, weight_xh, weight_hy)
                y_test_hat_arr = np.array(y_test_hat_arr)
                j_test = self.cross_entropy(y_test, y_test_hat_arr)
                print('epoch={} crossentropy(train):{}'.format(_+1,j_train))
                print('epoch={} crossentropy(test): {}'.format(_+1, j_test))
                f.write('epoch={} crossentropy(train):{}\n'.format(_+1,j_train))
                f.write('epoch={} crossentropy(test): {}\n'.format(_+1, j_test))

            err_train = self.err_rate(y_train, y_train_hat_arr)
            err_test = self.err_rate(y_test, y_test_hat_arr)
            print('error (train): ',err_train)
            print('error (test): ', err_test)
            f.write('error (train): {}\n'.format(err_train))
            f.write('error (test): {}\n'.format(err_test))

        return weight_xh,weight_hy,self.predict(y_train_hat_arr),self.predict(y_test_hat_arr),j_train,j_test

    def valid(self,x,y,weight_xh,weight_hy):
        y_hat_arr = []
        for i in range(len(y)):
            y_hat = self.nnforward(x[i], weight_xh, weight_hy)[0]
            y_hat_arr.append(y_hat)
        y_hat_arr = np.array(y_hat_arr)
        return y_hat_arr

    def predict(self,y_hat):
        return np.argmax(y_hat,axis=1)

    def err_rate(self,y,y_hat):
        y = np.argmax(y,axis=1)
        y_hat = np.argmax(y_hat,axis=1)
        count = 0
        for i in range(len(y)):
            count += 1 if y[i]!=y_hat[i] else 0
        return count/len(y)

if __name__ == '__main__':
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metrics_out = sys.argv[5]
    num_epoch = int(sys.argv[6])
    hidden_units = int(sys.argv[7])
    init_flag = sys.argv[8]
    learning_rate = float(sys.argv[9])

    ld = Dataprocess()
    data_train = ld.load_data(train_input)
    x_train, y_train = ld.parse_data(data_train)
    data_test = ld.load_data(test_input)
    x_test, y_test = ld.parse_data(data_test)

    nn = Neuralnet()
    len_x = len(x_test[0])-1
    len_y = len(y_test[0])

    weight_xh,weight_hy, yhat_train, yhat_test = nn.model(x_train=x_train, y_train=y_train,
                                                          x_test=x_test, y_test=y_test,
                                                          num_hidden_units=hidden_units,
                                                          init_strategy=init_flag, num_epochs=num_epoch,
                                                          learning_rate=learning_rate,
                                                          metrics_out=metrics_out, len_x= len_x, len_y=len_y)

    with open(train_out,'w') as f:
        for label in yhat_train:
            f.write('{}\n'.format(label))

    with open(test_out,'w') as f:
        for label in yhat_test:
            f.write('{}\n'.format(label))