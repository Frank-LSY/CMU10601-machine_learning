import numpy as np
import sys
import random as rd

# dealing with data
class LoadData():
    def __init__(self, infile, attribute):
        self.infile = infile
        self.attribute = attribute
        return

    # load dataset from files
    def load_data(self):
        dataset = np.loadtxt(self.infile,dtype=str,delimiter='\t',skiprows=1)
        return dataset

    # get specific column from the dataset
    def get_col(self,colIndex):
        dataset = self.load_data()
        # print(colIndex)
        col = []
        for row in dataset:
            # print(row[colIndex])
            col.append(row[colIndex])
        return col

    #get values of specific column (binary)
    def get_value(self,colIndex):
        dataset = self.load_data()
        label1 = dataset[0][colIndex]
        for item in dataset:
            if item[colIndex]!= label1:
                label2 = item[colIndex]
                break
        return label1, label2


#Decision Stump Logic
class DecisionStump():
    def __init__(self,infile,outfile,attribute):
        self.ld = LoadData(infile,attribute)
        self.attribute = attribute
        self.outfile = outfile
        self.dataset = self.ld.load_data()
        self.attrCol = self.ld.get_col(self.attribute)

    # train
    def train(self):
        dataset = self.dataset
        attrCol = self.attrCol
        label = self.ld.get_value(self.attribute)
        data0 = []
        data1 = []
        for index,row in enumerate(dataset):
            if attrCol[index] ==label[0]:
                data0.append(row)
            else:
                data1.append(row)

        vote0 = self.majority_vote(data0)
        vote1 = self.majority_vote(data1)

        return vote0, vote1

    # majority vote
    def majority_vote(self,dataset):
        count1 = 0
        label = self.ld.get_value(-1)
        for row in dataset:
            if row[-1] == label[0]:
                count1 += 1
            else:
                continue
        count2 = len(dataset)-count1

        if count1>count2:
            return label[0]
        elif count2>count1:
            return label[1]
        else:
            return rd.choice(label[0],label[1])

    # predict
    def h(self, vote_result):
        predict_label = []
        label = self.ld.get_value(self.attribute)
        for row in self.dataset:
            if row[self.attribute] == label[0]:
                predict_label.append(vote_result[0])
            else:
                predict_label.append(vote_result[1])
        predict_label = np.array(predict_label)
        np.savetxt(self.outfile,predict_label,fmt='%s',delimiter='\n')
        return predict_label

    # evaluate
    def evaluate(self):
        mr = Metrics(self.ld.get_col(-1),self.h(self.train()))
        return mr.error_rate()

class Metrics():
    def __init__(self,realcol,predcol):
        self.realcol = realcol
        self.predcol = predcol
        return

    def error_rate(self):
        count = 0
        for i in range(len(self.realcol)):
            if self.realcol[i] != self.predcol[i]:
                count += 1

        er = count/len(self.realcol)
        return er

if __name__ == '__main__':
    train_infile = sys.argv[1]
    test_infile = sys.argv[2]
    attr = sys.argv[3]
    train_outfile = sys.argv[4]
    test_outfile = sys.argv[5]
    metrics_out = sys.argv[6]
    # print("train_in:{}".format(train_infile))
    # print("test_in:{}".format(test_infile))
    # print("attr:{}".format(attr))
    # print("train_out:{}".format(train_outfile))
    # print("test_out:{}".format(test_outfile))
    # print("metrics:{}".format(metrics_out))

    train_ds = DecisionStump(train_infile,train_outfile,int(attr))
    train_train = train_ds.train()
    train_predict = train_ds.h(train_train)
    train_errorrate = train_ds.evaluate()

    test_ds = DecisionStump(test_infile,test_outfile,int(attr))
    test_train = test_ds.train()
    test_predict = test_ds.h(test_train)
    test_errorrate = test_ds.evaluate()

    with open(metrics_out,'w') as f:
        f.writelines("error(train): {}\n".format(train_errorrate))
        f.writelines("error(test): {}".format(test_errorrate))