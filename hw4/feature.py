import sys
import numpy as np
import time

class Loaddata():
    def __init__(self,file_in):
        self.file = file_in

    def load_data(self):
        with open(self.file, 'r', encoding='utf-8') as f:
            file = np.loadtxt(f, str, delimiter='\t',comments=None)
        return file

    def load_dict(self):
        with open(self.file,'r') as f:
            dict = f.read()
        dict = dict.split('\n')

        new_dict = {}
        for item in dict:
            tmp = item.split()
            if len(tmp)==2:
                new_dict[tmp[0]] = tmp[1]
            else:
                continue

        return new_dict

class Model():
    def __init__(self,file_in,dict):
        self.data = Loaddata(file_in).load_data()
        self.label = self.data[:,0]
        self.txt = self.data[:,1]

        self.dict = Loaddata(dict).load_dict()

    def bow_count(self,txt_lst):
        bow_dict = {}
        for token in txt_lst:
            if token not in self.dict:
                continue
            else:
                if self.dict[token] not in bow_dict:
                    bow_dict[self.dict[token]] = 1
                else:
                    bow_dict[self.dict[token]] += 1
        # print(bow_dict)
        return bow_dict

    def model1(self):
        txt_arr = []
        for i in range(len(self.txt)):
            review_str = self.label[i]
            review = self.txt[i].split()
            bow_dict = self.bow_count(review)
            for key,value in bow_dict.items():
                kv = ':'.join([key,'1'])
                review_str += '\t{}'.format(kv)
            review_str += '\n'
            txt_arr.append(review_str)
        return txt_arr

    def model2(self):
        txt_arr = []
        for i in range(len(self.txt)):
            review_str = self.label[i]
            review = self.txt[i].split()
            bow_dict = self.bow_count(review)
            for key,value in bow_dict.items():
                if value<4:
                    kv = ':'.join([key,'1'])
                    review_str += '\t{}'.format(kv)
                else:
                    continue
            review_str += '\n'
            txt_arr.append(review_str)
        return txt_arr

if __name__ == '__main__':
    train_input = sys.argv[1]
    validation_input = sys.argv[2]
    test_input = sys.argv[3]
    dict_input = sys.argv[4]
    formatted_train_out = sys.argv[5]
    formatted_validation_out = sys.argv[6]
    formatted_test_out = sys.argv[7]
    feature_flag = sys.argv[8]

    start = time.time()
    model_train = Model(train_input,dict_input)
    model_validation = Model(validation_input, dict_input)
    model_test = Model(test_input, dict_input)
    # end = time.time()
    # print("model time: ",end-start)

    # start = time.time()
    if feature_flag=='1':
        print('Model 1...')
        train_out = model_train.model1()
        validation_out = model_validation.model1()
        test_out = model_test.model1()
    elif feature_flag=='2':
        print('Model 2...')
        train_out = model_train.model2()
        validation_out = model_validation.model2()
        test_out = model_test.model2()
    end = time.time()
    print("construct time: ",end-start)

    with open(formatted_train_out,'w') as f:
        f.writelines(train_out)
    with open(formatted_validation_out,'w') as f:
        f.writelines(validation_out)
    with open(formatted_test_out,'w') as f:
        f.writelines(test_out)

    # ld = Loaddata('./handout/dict.txt')
    # dict = ld.load_dict()
    # print(len(dict))
    # print(dict['films'])
    # print(dict['re-recorded'])
