import numpy as np
import sys
import random as rd
import csv


class LoadData:
    def __init__(self, infile, outfile, attribute):
        self.infile = infile
        self.outfile = outfile
        self.attribute = attribute
        return

    def load_data(self):
        with open(self.infile,'r') as f:
            datafile = csv.reader(f,delimiter='\t')
            dataset = []
            attrCol = []
            for index,row in enumerate(datafile):
                if index==0:
                    continue
                dataset.append(row)
                attrCol.append(row[self.attribute])
        return dataset, attrCol

    def get_value(self,dataset,colIndex):
        label1 = dataset[0][colIndex]
        for item in dataset:
            if item[colIndex]!= label1:
                label2 = item[colIndex]
                break
        return label1, label2



class DecisionStump:

    def __init__(self,dataset,attr_col):
        self.dataset = dataset
        self.attrCol = attr_col
        self.ld = LoadData()


    def train(self,attribute):
        dataset = LoadData.load_data(DecisionStump)[0]
        attrCol = LoadData.load_data(DecisionStump)[1]
        label = LoadData.get_value(dataset,attribute)
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


    def majority_vote(self,dataset):
        count1 = 0
        label = self.get_value(dataset,-1)
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

    def h(self,dataset,attribute):
        return


class Metrics:
    def __init__(self):
        return

    def error_rate(self,realcol, predcol):
        count = 0
        for i in range(len(realcol)):
            if realcol[i] != predcol[i]:
                count += 1

        er = count/len(realcol)
        return er

if __name__ == '__main__':
    # infile = sys.argv[1]
    # outfile = sys.argv[2]
    # print("The input file is: %s" % (infile))
    # print("The output file is: %s" % (outfile))
    ds = DecisionStump('./handout/small_train.tsv','./tmp',0)
    data = ds.load_data()
    label = ds.get_value(data[0],0)
    maj = ds.majority_vote(data[0])
    tr = ds.train(0)
    print(tr)