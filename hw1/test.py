import numpy as np
import csv

with open ('./handout/small_train.tsv','r') as f:

    a = csv.reader(f,delimiter='\t')

    for index,row in enumerate(a):
        if index==0:
            continue
        print(row[0])