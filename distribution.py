import torch
import pandas as pd
import numpy as np
import utils
import os
from matplotlib import pyplot as plt
from scipy.stats.mstats import winsorize
import scipy

from utils import *


def truncated_power_law(a, m):
    x = np.arange(0, m, dtype='float')
    pmf = 1/torch.pow(torch.tensor(x+1).cuda(),a)
    pmf /= pmf.sum()
    return pmf

def dist_MLE(empirical_dist,dist_name = 'None',rounds = 10000, lr = 1e-3):
    pctg_neg = False
    if (empirical_dist[0] < empirical_dist[1]) & ('vol' not in dist_name):
        pctg_neg = empirical_dist[0] / sum(empirical_dist)
        empirical_dist = empirical_dist[1:]
    length = len(empirical_dist)
    a = torch.rand(1, requires_grad=True, device="cuda")
    optimizer = torch.optim.RMSprop([a], lr=lr)
    loss_list = []
    param_list = []
    for i in range(rounds):
        optimizer.zero_grad()
        log_sum_neg = 0
        d = truncated_power_law(a=a, m=length)
        log_sum_neg = -torch.sum(d.log()*torch.tensor(empirical_dist).cuda())
        log_sum_neg.backward()
        optimizer.step()
        loss_list.append(log_sum_neg.item())
        param_list.append(a.item())
        print(log_sum_neg.item())
    return [pctg_neg, param_list[np.argmin(np.array(loss_list))],length]


df_ = pd.read_csv('./LOBSTER/parsed_event_df/INTC_event_df_5.csv',index_col=0)
df_.loc[df_.event=='A','event'] = 1
df_.loc[df_.event=='B','event'] = 2
df_.loc[df_.event=='C','event'] = 3
df_.loc[df_.event=='D','event'] = 4
delta_time = np.array(df_.time[1:]) - np.array(df_.time[:-1])
# a = len(delta_time[delta_time<0.01])
# delta_time = np.clip(delta_time,a_min = 0.01,a_max=100)
#
# plt.hist(delta_time, bins =  list(np.linspace(0,1,10000)))
# plt.title("histogram")
# plt.show()
df = df_[1:]
df.time = delta_time
df['state'] = (df['3'] - df['1'])/(df['3']+df['1'])
df.loc[df.state>0.4,'state'] = 3
df.loc[df.state<-0.4,'state'] = 1
df.loc[(df.state>=-0.4)&(df.state<=0.4),'state'] = 2

df.loc[df.event==1,'price'] = (df.loc[df.event==1,'price'] - df.loc[df.event==1,'2']) / 100 * (-1)
df.loc[df.event==2,'price'] = (df.loc[df.event==2,'price'] - df.loc[df.event==2,'2']) / 100 * (-1)
df.loc[df.event==3,'price'] = (df.loc[df.event==3,'price'] - df.loc[df.event==3,'0']) / 100
df.loc[df.event==4,'price'] = (df.loc[df.event==4,'price'] - df.loc[df.event==4,'0']) / 100
df['price'] = df['price'].clip(lower=-2)
df['volume'] = (df['volume']/100).round().astype(int)



key_list = ['pctg_market_bid_1','pctg_market_bid_2','pctg_market_ask_1','pctg_market_ask_2',
            'limit_bid_sub_price_1','limit_bid_sub_price_2','limit_ask_sub_price_1','limit_ask_sub_price_2',
            'limit_bid_cancel_price_1','limit_bid_cancel_price_2','limit_ask_cancel_price_1','limit_ask_cancel_price_2',
            'market_bid_sub_vol_1','market_bid_sub_vol_2','market_ask_sub_vol_1','market_ask_sub_vol_2',
            'limit_bid_sub_vol_1','limit_bid_sub_vol_2','limit_ask_sub_vol_1','limit_ask_sub_vol_2',
            'limit_bid_cancel_vol_1','limit_bid_cancel_vol_2','limit_ask_cancel_vol_1','limit_ask_cancel_vol_2'
            ]
param_dict = {}

'''
account for market orders
'''

num_bid_market_order_1 = len(df.loc[(df.event==1)&(df.price==-1)&(df['0']-df['2']==100),'price'])
num_bid_market_order_2 = len(df.loc[(df.event==1)&(df.price==-2)&(df['0']-df['2']==200),'price'])
num_ask_market_order_1 = len(df.loc[(df.event==3)&(df.price==-1)&(df['0']-df['2']==100),'price'])
num_ask_market_order_2 = len(df.loc[(df.event==3)&(df.price==-2)&(df['0']-df['2']==200),'price'])
sum_market_order = num_bid_market_order_1 + num_bid_market_order_2 + num_ask_market_order_1 + num_ask_market_order_2
pctg_market = sum_market_order / len(df[(df.event==1) | (df.event==3)])
num_bid_1 = len(df.loc[(df.event==1)&(df['0']-df['2']==100),'price'])
num_bid_2 = len(df.loc[(df.event==1)&(df['0']-df['2']==200),'price'])
num_ask_1 = len(df.loc[(df.event==3)&(df['0']-df['2']==100),'price'])
num_ask_2 = len(df.loc[(df.event==3)&(df['0']-df['2']==200),'price'])
pctg_market_bid_1 = num_bid_market_order_1 / num_bid_1
pctg_market_bid_2 = num_bid_market_order_2 / num_bid_2
pctg_market_ask_1 = num_ask_market_order_1 / num_ask_1
pctg_market_ask_2 = num_ask_market_order_2 / num_ask_2

'''
account for limit orders submission
'''
limit_bid_sub_price_1 = [len(df[(df.price==i) & (df.event==1) & (df['0']-df['2']==100)]) for i in range(5)]
limit_bid_sub_price_2 = [len(df[(df.price==i-1) & (df.event==1) & (df['0']-df['2']==200)]) for i in range(6)]
limit_ask_sub_price_1 = [len(df[(df.price==i) & (df.event==3) & (df['0']-df['2']==100)]) for i in range(5)]
limit_ask_sub_price_2 = [len(df[(df.price==i-1) & (df.event==3) & (df['0']-df['2']==200)]) for i in range(6)]

'''
account for limit orders cancellation
'''
limit_bid_cancel_price_1 = [len(df[(df.price==i) & (df.event==2) & (df['0']-df['2']==100)]) for i in range(5)]
limit_bid_cancel_price_2 = [len(df[(df.price==i) & (df.event==2) & (df['0']-df['2']==200)]) for i in range(5)]
limit_ask_cancel_price_1 = [len(df[(df.price==i) & (df.event==4) & (df['0']-df['2']==100)]) for i in range(5)]
limit_ask_cancel_price_2 = [len(df[(df.price==i) & (df.event==4) & (df['0']-df['2']==200)]) for i in range(5)]

'''
account for market order volume
'''
market_bid_sub_vol_1 = [len(df[(df.volume==(i+1)) & (df.event==1) & (df.price == -1) & (df['0']-df['2']==100)]) for i in range(round(np.quantile(np.array(df[(df.event==1) & (df.price == -1) & (df['0']-df['2']==100)].volume),0.95)))]
market_bid_sub_vol_2 = [len(df[(df.volume==(i+1)) & (df.event==1) & (df.price == -2) & (df['0']-df['2']==200)]) for i in range(round(np.quantile(np.array(df[(df.event==1) & (df.price == -2) & (df['0']-df['2']==200)].volume),0.95)))]
market_ask_sub_vol_1 = [len(df[(df.volume==(i+1)) & (df.event==3) & (df.price == -1) & (df['0']-df['2']==100)]) for i in range(round(np.quantile(np.array(df[(df.event==3) & (df.price == -1) & (df['0']-df['2']==100)].volume),0.95)))]
market_ask_sub_vol_2 = [len(df[(df.volume==(i+1)) & (df.event==3) & (df.price == -2) & (df['0']-df['2']==200)]) for i in range(round(np.quantile(np.array(df[(df.event==3) & (df.price == -2) & (df['0']-df['2']==200)].volume),0.95)))]

'''
account for limit order submission volume
'''
limit_bid_sub_vol_1 = [[len(df[(df.volume==(i+1)) & (df.event==1) & (df.price == j) & (df['0']-df['2']==100)]) for i in range(round(np.quantile(np.array(df[(df.event==1) & (df.price == j) & (df['0']-df['2']==100)].volume),0.95)))] for j in range(5)]
limit_bid_sub_vol_2 = [[len(df[(df.volume==(i+1)) & (df.event==1) & (df.price == j-1) & (df['0']-df['2']==200)]) for i in range(round(np.quantile(np.array(df[(df.event==1) & (df.price == j-1) & (df['0']-df['2']==200)].volume),0.95)))] for j in range(6)]
limit_ask_sub_vol_1 = [[len(df[(df.volume==(i+1)) & (df.event==3) & (df.price == j) & (df['0']-df['2']==100)]) for i in range(round(np.quantile(np.array(df[(df.event==3) & (df.price == j) & (df['0']-df['2']==100)].volume),0.95)))] for j in range(5)]
limit_ask_sub_vol_2 = [[len(df[(df.volume==(i+1)) & (df.event==3) & (df.price == j-1) & (df['0']-df['2']==200)]) for i in range(round(np.quantile(np.array(df[(df.event==3) & (df.price == j-1) & (df['0']-df['2']==200)].volume),0.95)))] for j in range(6)]

'''
account for limit order cancellation volume
'''
limit_bid_cancel_vol_1 = [[len(df[(df.volume==(i+1)) & (df.event==2) & (df.price == j) & (df['0']-df['2']==100)]) for i in range(round(np.quantile(np.array(df[(df.event==2) & (df.price == j) & (df['0']-df['2']==100)].volume),0.95)))] for j in range(5)]
limit_bid_cancel_vol_2 = [[len(df[(df.volume==(i+1)) & (df.event==2) & (df.price == j) & (df['0']-df['2']==200)]) for i in range(round(np.quantile(np.array(df[(df.event==2) & (df.price == j) & (df['0']-df['2']==200)].volume),0.95)))] for j in range(5)]
limit_ask_cancel_vol_1 = [[len(df[(df.volume==(i+1)) & (df.event==4) & (df.price == j) & (df['0']-df['2']==100)]) for i in range(round(np.quantile(np.array(df[(df.event==4) & (df.price == j) & (df['0']-df['2']==100)].volume),0.95)))] for j in range(5)]
limit_ask_cancel_vol_2 = [[len(df[(df.volume==(i+1)) & (df.event==4) & (df.price == j) & (df['0']-df['2']==200)]) for i in range(round(np.quantile(np.array(df[(df.event==4) & (df.price == j) & (df['0']-df['2']==200)].volume),0.95)))] for j in range(5)]


'''
calculate dist parameters using MLE and store parameters in a dict
'''
for key in key_list:
    print(key)
    exec('empirical_dist = {}'.format(key))
    param = []
    if isinstance(empirical_dist,list):
        if isinstance(empirical_dist[0],list):
            for item in empirical_dist:
                param.append(dist_MLE(item,dist_name ='vol'))
        else:
            param = param + dist_MLE(empirical_dist,dist_name=key)
    else:
        param = param + [empirical_dist]
    param_dict[key] = param
np.save('param_dict_95_INTC_5thday.npy',param_dict)



plt.figure()

# fig, axs = plt.subplots(2, 2)
# axs[0, 0].hist(np.array(df.loc[df.event==1,'price']).astype(int),bins=[-1,0,1,2,3,4])
# axs[0, 0].set_title('Axis [0, 0]')
# axs[0, 1].hist(np.array(df.loc[df.event==2,'price']).astype(int),bins=[-1,0,1,2,3,4])
# axs[0, 1].set_title('Axis [0, 1]')
# axs[1, 0].hist(np.array(df.loc[df.event==3,'price']).astype(int),bins=[-1,0,1,2,3,4])
# axs[1, 0].set_title('Axis [1, 0]')
# axs[1, 1].hist(np.array(df.loc[df.event==4,'price']).astype(int),bins=[-1,0,1,2,3,4])
# axs[1, 1].set_title('Axis [1, 1]')
# plt.show()

# fig, axs = plt.subplots(2, 2)
# axs[0, 0].hist(np.array(a.loc[a.state==1,'price']).astype(int),bins=[0,1,2,3,4,5])
# axs[0, 0].set_title('Axis [0, 0]')
# axs[0, 1].hist(np.array(a.loc[a.state==2,'price']).astype(int),bins=[0,1,2,3,4,5])
# axs[0, 1].set_title('Axis [0, 1]')
# axs[1, 0].hist(np.array(a.loc[a.state==3,'price']).astype(int),bins=[0,1,2,3,4,5])
# axs[1, 0].set_title('Axis [1, 0]')
# plt.show()

# df['volume'] = (df['volume']/100).astype(int)
#
# fig, axs = plt.subplots(2, 2)
# axs[0, 0].hist(np.array(df.loc[df.state==1,'volume']).astype(int),bins=list(np.linspace(0,20,21,endpoint=True)))
# axs[0, 0].set_title('Axis [0, 0]')
# axs[0, 1].hist(np.array(df.loc[df.state==2,'volume']).astype(int),bins=list(np.linspace(0,20,21,endpoint=True)))
# axs[0, 1].set_title('Axis [0, 1]')
# axs[1, 0].hist(np.array(df.loc[df.state==3,'volume']).astype(int),bins=list(np.linspace(0,20,21,endpoint=True)))
# axs[1, 0].set_title('Axis [1, 0]')
# plt.show()

