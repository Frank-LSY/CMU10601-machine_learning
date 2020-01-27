import numpy as np


# dealing with data
class LoadData():
    def __init__(self):
        self.dataset = None
        self.col = []
        self.label1 = ''
        self.label2 = ''

    # load dataset from files
    def load_data(self,infile):
        self.dataset = np.loadtxt(infile,dtype=str,delimiter='\t',skiprows=1)
        return self.dataset

    # get specific column from the dataset
    def get_col(self,dataset,col_index):
        # print(colIndex)
        for row in dataset:
            # print(row[colIndex])
            self.col.append(row[col_index])

        return self.col

    # get values of specific column (binary)
    def get_value(self,dataset,col_index):
        self.label1 = dataset[0][col_index]
        for item in dataset:
            if item[col_index]!= self.label1:
                self.label2 = item[col_index]
                break
            else:
                self.label2 = ''
        return self.label1,self.label2

if __name__ == '__main__':
    ld = LoadData()
    dataset = ld.load_data('../handout/small_test.tsv')
    print(type(dataset))
    col = ld.get_col(dataset,0)
    val = ld.get_value(dataset,0)
    print(val)