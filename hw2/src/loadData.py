import numpy as np


# dealing with data
class LoadData():
    def __init__(self):
        self.dataset = None
        self.head = None
        self.col = []
        self.label1 = ''
        self.label2 = ''

    # load dataset from files
    def load_data(self,infile):
        self.dataset = np.loadtxt(infile,dtype=str,delimiter='\t',skiprows=1)
        with open(infile,'r') as f:
            self.head = f.readline().strip(' \n').split('\t')
        return self.dataset

    # get specific column from the dataset
    def get_col(self,dataset,col_index):
        for row in dataset:
            self.col.append(row[col_index])

        return self.col

    # get values of specific column (binary)
    def get_value(self,dataset,col_index):
        label = np.unique(dataset[:,col_index])
        self.label1 = label[0]
        if len(label)==2:
            self.label2 = label[1]
        else:
            self.label2 = ''
        return self.label1,self.label2

if __name__ == '__main__':
    ld = LoadData()
    dataset = ld.load_data('../handout/education_test.tsv')
    print(type(dataset))
    col = ld.get_col(dataset,0)
    val = ld.get_value(dataset,-1)
    print(val)