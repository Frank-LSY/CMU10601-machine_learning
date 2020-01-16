import numpy as np
import csv



a= np.loadtxt('./handout/small_train.tsv', delimiter='\t',dtype=str,skiprows=1)
for index,rows in enumerate(a):
    print(rows)
    print(index)
# print(a)

# with open ('./handout/small_train.tsv','r') as f: