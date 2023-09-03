import torch
import pandas as pd
import numpy as np
import utils
import os
from matplotlib import pyplot as plt
from scipy.stats.mstats import winsorize

from utils import *
from model.model_CT4LSTM_PP import CT4LSTM_PP

a = torch.load('./state_trans_prob0.torch')

np.random.seed(10)
device = torch.device('cuda:0')

len_of_record = 50
batch_size = 512
niter = 500
lr = 2e-3
df_ = pd.read_csv('./LOBSTER/parsed_event_df/INTC_event_df_5.csv',index_col=0)
df_.loc[df_.event=='A','event'] = 1
df_.loc[df_.event=='B','event'] = 2
df_.loc[df_.event=='C','event'] = 3
df_.loc[df_.event=='D','event'] = 4
delta_time = np.array(df_.time[1:]) - np.array(df_.time[:-1])
df = df_[1:]
df.time = delta_time

df['score'] = (df['3'] - df['1'])/(df['3']+df['1'])
df['state'] = 0
df.loc[df.score>0.4,'state'] = 2
df.loc[df.score<-0.4,'state'] = 0
df.loc[(df.score>=-0.4)&(df.score<=0.4),'state'] = 1

itr = 5000
length = len(df)-1

state_before = df.state[:-1].astype(int).reset_index(drop=True)
state_after = df.state[1:].astype(int).reset_index(drop=True)
event = df.event[:-1].astype(int).reset_index(drop=True) - 1
selected_df = pd.DataFrame({'event':event,'state_before':state_before,'state_after':state_after})
params = torch.rand((4,3,3),requires_grad=True,device='cuda:0')
optimizer = torch.optim.RMSprop([params], lr=lr)

for i in range(itr):
    optimizer.zero_grad()
    prob = torch.softmax(params,dim=-1)
    log_prob = 0
    for m in range(4):
        for n in range(3):
            for h in range(3):
                log_prob = log_prob - torch.log(prob[m,n,h]) * len(selected_df[(selected_df.event==m)&(selected_df.state_before==n)&(selected_df.state_after==h)])
    log_prob.backward()
    optimizer.step()
    print(log_prob.item())
print(prob)
torch.save(prob,'./state_trans_prob0.torch')





