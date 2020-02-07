import numpy as np
import sys
import random as rd
import time

from loadData import LoadData
from inspection import Inspection


class Node():
    def __init__(self,col_index_used,col_index_touse, dataset, head, left = None, right = None, depth=0):
        self.left = left
        self.right = right
        self.message = ''
        self.col_index_touse = col_index_touse
        self.col_index_used = col_index_used
        self.attribute = head[col_index_used]  # the attribute to divide
        self.dataset = dataset      # the dataset divided by attribute
        self.depth = depth          # current depth of the node

        self.ld = LoadData()
        self.ins = Inspection(self.dataset)
        self.result = self.ins.majority_vote(dataset) # majority_vote for each node


class DecisionTree():
    def __init__(self,ori_dataset):
        self.col = []
        self.ld = ori_dataset
        self.dataset = ori_dataset.dataset
        self.ins = Inspection(self.dataset)
        self.head = ori_dataset.head
        self.label = self.ld.get_value(self.dataset,-1)

    # divide the dataset with certain attribute
    def divide_dataset(self,dataset,col_index):
        label = self.ld.get_value(dataset,col_index)
        dataset0 = []
        dataset1 = []
        for row in dataset:
            if row[col_index] == label[0]:
                dataset0.append(row)
            else:
                dataset1.append(row)
        dataset0 = np.array(dataset0)
        dataset1 = np.array(dataset1)

        return dataset0,dataset1

    # calculate the gini impurity given attribute
    def gini_impurity(self,dataset,col_index=-1):

        if col_index==-1:
            gi = self.ins.gini_impurity(dataset)
        else:
            ds = self.divide_dataset(dataset,col_index)
            gi_left = self.ins.gini_impurity(ds[0])
            gi_right = self.ins.gini_impurity(ds[1])
            gi = (len(ds[0])/len(dataset))*gi_left+(len(ds[1])/len(dataset))*gi_right

        return gi

    # calculate the gini gain given attribute
    def gini_gain(self,dataset,col_index):
        ori_gi = self.gini_impurity(dataset)
        new_gi = self.gini_impurity(dataset,col_index)
        gg = ori_gi-new_gi

        return gg

    def get_attribute(self,dataset,used_col):
        gg_arr = {}
        col_arr = [i for i in range(len(dataset[0])-1)]
        for item in list(set(col_arr).difference(set(used_col))):
            gg_arr[item] = self.gini_gain(dataset,item)

        gg_arr = sorted(gg_arr.items(),key=lambda d:d[0], reverse=True)
        gg_lst = []
        gg_index = []
        for item in gg_arr:
            gg_index.append(item[0])
            gg_lst.append(item[1])
        col_index = gg_index[gg_lst.index(max(gg_lst))]
        return col_index


    # construct the decision tree
    def construct(self,dataset,max_depth, depth=0, col_index=-1):
        # print('used col: {}'.format(self.col))

        # reach the max depth
        if depth>max_depth-1:
            # print('depth reach max depth')
            node = Node(col_index, -1, dataset, self.head, depth=depth)
            node.message = 'Gini Impurity: {}'.format(self.gini_impurity(dataset))
            return node

        # after divide the dataset is empty
        elif len(dataset)==0:
            # print('dataset is empty.')
            node = Node(col_index, -1, dataset, self.head, depth=depth)
            node.message = 'Gini Impurity: {}'.format(self.gini_impurity(dataset))
            return node

        # No more attribute to divide
        elif len(dataset[0])==len(self.col)+1:
            # print('all the attributes have been used.')
            node = Node(col_index, -1, dataset, self.head, depth=depth)
            node.message = 'Gini Impurity: {}'.format(self.gini_impurity(dataset))
            return node

        # after divide the gini-impurity of dataset is 0
        elif self.gini_impurity(dataset)==0:
            # print('no need to do more division!-gini impurity')
            node = Node(col_index, -1, dataset, self.head, depth=depth)
            node.message = 'Gini Impurity: {}'.format(self.gini_impurity(dataset))
            return node

        # no more gini-gain for further division
        elif self.gini_gain(dataset,self.get_attribute(dataset,self.col))==0:
            # print('no need to do more division!-gini-gain')
            node = Node(col_index, -1, dataset, self.head, depth=depth)
            node.message = 'Gini Impurity: {}'.format(self.gini_impurity(dataset))

            return node

        # recursively construct the left and right node
        else:
            # construct the current node
            node = Node(col_index, self.get_attribute(dataset, self.col), dataset, self.head, depth=depth)
            node.message = 'Gini Impurity: {}'.format(self.gini_impurity(dataset))
            col_index = self.get_attribute(dataset, self.col)
            self.col.append(col_index)

            # divide the dataset according to max gini-gain
            new_dataset = self.divide_dataset(dataset,col_index)
            depth += 1
            #recurse the left branch
            # print('left')
            left_branch = self.construct(new_dataset[0], max_depth, depth,col_index)
            node.left = left_branch
            #recurse the right branch
            # print('right')
            right_branch = self.construct(new_dataset[1], max_depth, depth,col_index)
            node.right = right_branch
            self.col.remove(col_index)

            return node

    # traverse the constructed tree and print the tree
    def traverse(self,node):
        if node:
            result = node.dataset[:,-1]
            count={self.label[0]:0,self.label[1]:0}
            for item in result:
                    count[item] += 1

            for i in range(node.depth):
                print('| ',end='')

            if node.col_index_used != -1:
                print(node.attribute,'=',node.dataset[0][node.col_index_used],': ',end='')
            print(count)
            self.traverse(node.left)
            self.traverse(node.right)



    def classify_row(self,row,node,outfile_stream):
        if node.left==None and node.right==None:
            outfile_stream.write('{}\n'.format(node.result))
            return node.result
        else:
            if row[node.col_index_touse] == node.left.dataset[0][node.col_index_touse]:
                self.classify_row(row, node.left,outfile_stream)
            elif row[node.col_index_touse] == node.right.dataset[0][node.col_index_touse]:
                self.classify_row(row, node.right,outfile_stream)

    def classify(self,dataset,root,outfile):
        with open(outfile,'w') as f:
            for row in dataset:
                self.classify_row(row,root,f)
        return

class Evaluate():
    def __init__(self, realcol, predcol):
        self.realcol = realcol
        self.predcol = predcol

    def error_rate(self):
        count = 0
        for i in range(len(self.realcol)):
            if self.realcol[i] != self.predcol[i]:
                count += 1

        er = count / len(self.realcol)
        return er

if __name__ == '__main__':
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    max_depth = sys.argv[3]
    train_output = sys.argv[4]
    test_output = sys.argv[5]
    metrics_out = sys.argv[6]

    ld = LoadData()
    dataset = ld.load_data(train_input)
    dt = DecisionTree(ld)

    root = dt.construct(dataset, max_depth=int(max_depth))

    dt.traverse(root)

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

    print('train:',train_errate)
    print('test: ',test_errate)

    with open(metrics_out,'w') as f:
        f.write('error(train): {}\n'.format(train_errate))
        f.write('error(test): {}\n'.format(test_errate))
