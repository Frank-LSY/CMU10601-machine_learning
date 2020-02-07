import numpy as np
import sys
import random as rd

from loadData import LoadData

class Inspection():
    def __init__(self,ori_dataset):
        self.ori_dataset = ori_dataset
        self.ld = LoadData()

        self.er = 0
        self.gi = 0

    # majority vote
    def majority_vote(self,dataset):
        count1 = 0
        label = self.ld.get_value(dataset,-1)
        for row in dataset:
            if row[-1] == label[0]:
                count1 += 1
            else:
                continue
        count2 = len(dataset) - count1
        if count1 > count2:
            return label[0]
        elif count2 > count1:
            return label[1]
        elif count2==0:
            return label[0]
        else:
            return label[1]

    # error rate
    def error_rate(self,dataset):
        label = self.majority_vote(dataset)
        count = 0
        for row in dataset:
            if row[-1] != label:
                count += 1
        self.er = count/len(dataset)
        return self.er

    #gini impurity
    def gini_impurity(self,dataset):
        if len(dataset)==0:
            self.gi=0
        else:
            count1 = 0
            for item in dataset:
                if item[-1]==dataset[0][-1]:
                    count1+=1
            count2 = len(dataset)-count1
            self.gi = (count1/len(dataset))*(count2/len(dataset))+(count2/len(dataset))*(count1/len(dataset))
        return self.gi

    # evaluate with error_rate and gini_impurity
    def evaluate(self):
        err_rate = self.error_rate(self.ori_dataset)
        gini_impurity = self.gini_impurity(self.ori_dataset)
        return err_rate,gini_impurity


if __name__ == '__main__':
    infile = sys.argv[1]
    outfile = sys.argv[2]
    ld = LoadData()
    ori_dataset = ld.load_data(infile)
    ins = Inspection(ori_dataset)
    eva = ins.evaluate()
    err_rate = eva[0]
    gini_impurity = eva[1]
    with open(outfile, 'w') as f:
        f.writelines("gini_impurity: {}\n".format(gini_impurity))
        f.writelines("error: {}\n".format(err_rate))
    # print(err_rate)
    # print(gini_impurity)