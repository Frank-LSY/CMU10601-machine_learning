import numpy as np
import sys
import random as rd

from loadData import LoadData
from inspection import Inspection


class Node():
    def __init__(self,col_index,dataset, left = None, right = None, depth=0):
        self.left = left
        self.right = right
        # self.choice = choice
        self.attribute = col_index  # the attribute to divide
        self.dataset = dataset      # the dataset divided by attribute
        self.depth = depth          # current depth of the node


class DecisionTree():
    def __init__(self,ori_dataset,max_depth):
        self.col = []

        self.max_depth = max_depth
        self.dataset = ori_dataset

        self.ld = LoadData()
        self.ins = Inspection(ori_dataset)

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
            # print('dataset:\n', dataset)
            # print('col index:',col_index)
            ds = self.divide_dataset(dataset,col_index)
            # print('ds0:\n',ds[0])
            # print('ds1:\n', ds[1])
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
        col_index = max(gg_arr,key=gg_arr.get)

        return col_index


    # 记录下路径，再进行搞
    def construct(self,dataset,col_index=-1,depth=0):
        # print('\nlen:',len(dataset))
        # print(dataset)
        # print('used_col:',self.col)
        # print('depth:', depth)
        # reach the max depth
        if depth>self.max_depth:
            print('depth reach max depth')
            # self.col.pop(col_index)
            return None

        # after divide the dataset is empty
        elif len(dataset)==0:
            print('dataset is empty.')
            # self.col.pop(col_index)
            return None

        # No more attribute to divide
        elif len(dataset[0])==len(self.col)+1:
            # print(self.col)
            print('all the attributes have been used.')
            # self.col.pop(col_index)
            # print(depth)
            return None

        # after divide the gini-impurity of dataset is 0
        elif self.gini_impurity(dataset)==0:
            print('no need to do more division!')
            # self.col.pop(col_index)
            return None

        # recursively construct the left and right node
        else:
            col_index = self.get_attribute(dataset,self.col)
            # construct the current node
            node = Node(col_index, dataset, depth=depth)
            self.col.append(col_index)

            # divide the dataset according to max gini-gain
            new_dataset = self.divide_dataset(dataset,col_index)
            depth += 1
            #recurse the left branch
            left_branch = self.construct(new_dataset[0], col_index, depth)
            node.left = left_branch
            # self.col.pop(col_index)
            #recurse the right branch
            right_branch = self.construct(new_dataset[1], col_index, depth)
            node.right = right_branch
            # print('col_index:',col_index)
            self.col.remove(col_index)

            return node

    def traverse(self,node):
        if node:
            # print(node.dataset,'\n')
            print(node.depth,'\t',node.attribute)
            self.traverse(node.left)
            self.traverse(node.right)


if __name__ == '__main__':
    ld = LoadData()
    dataset = ld.load_data('../handout/small_train.tsv')
    dt = DecisionTree(dataset,0)
    ds = dt.divide_dataset(dataset,1)
    # gini = dt.gini_impurity(dataset,1)
    giga = dt.gini_gain(dataset,1)
    # col = dt.get_attribute(dataset)
    root = dt.construct(dataset)
    # print(root.left.left.left.right.depth)
    dt.traverse(root)
    # print(root.dataset)

    # print(dataset)
    # print(ds[0])
    # print(ds[1])