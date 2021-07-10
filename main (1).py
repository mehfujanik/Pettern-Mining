import pandas as pd
from pandas import read_csv
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
import csv
import time
import scipy
acc_list_of_arrays=[]
with open('chess.csv', 'r') as file:
    dataset = csv.reader(file, delimiter=' ')
    for row in dataset:
        temp=[element for element in row]
        temp= temp[:-1]
        #print(temp)
        acc_list_of_arrays.append(temp)
ts= pd.Series(pd.arrays.SparseArray(acc_list_of_arrays))
print(ts)
start_time = time.process_time()
te = TransactionEncoder()
te_ary = te.fit(ts).transform(ts)
df = pd.DataFrame(te_ary, columns=te.columns_)
print(df.head(10))
result= fpgrowth(df, min_support=0.9, use_colnames=True)
print (result)
print ((time.process_time() - start_time),"seconds")



