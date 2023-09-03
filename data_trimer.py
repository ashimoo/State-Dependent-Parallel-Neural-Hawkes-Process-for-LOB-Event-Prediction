import pandas as pd
import numpy as np

path_lob = './LOBSTER/JPM_2012-06-20_2012-06-26_10/JPM_2012-06-26_34200000_57600000_orderbook_10.csv'
path_msb = './LOBSTER/JPM_2012-06-20_2012-06-26_10/JPM_2012-06-26_34200000_57600000_message_10.csv'

with open(path_lob) as f:
    data_lob = pd.read_csv(f, header=None)
with open(path_msb) as f:
    data_msb = pd.read_csv(f, header=None).iloc[:,:-1]
    data_msb.columns = ['time', 'type', 'index', 'quantity', 'price', 'direction']

data_combined = data_lob.join(data_msb)
data_combined = data_combined[(data_combined['time']>36000) & (data_combined['time']<55800)]
data_combined = np.array(data_combined)
trimmed_list = []

for i in range(data_combined.shape[0]):
    if (data_combined[i,-2] <= data_combined[i,16]) and (data_combined[i,-2] >= data_combined[i,18]):
        trimmed_list.append(data_combined[i,:])
trimmed_array = np.stack(trimmed_list)
data_lob_5 = pd.DataFrame(trimmed_array[:,list(range(20))])
data_msb_5 = pd.DataFrame(trimmed_array[:,40:])

data_lob_5.to_csv('./LOBSTER/JPM/JPM_orderbook_part_5.csv',header=0,index=0)
data_msb_5.to_csv('./LOBSTER/JPM/JPM_message_part_5.csv',header=0,index=0)