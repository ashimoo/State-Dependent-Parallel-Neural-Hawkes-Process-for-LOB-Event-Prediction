import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.stats import weibull_min
from scipy.stats import kstest
import scipy
import random
from scipy.stats.mstats import winsorize
from scipy.stats import pearsonr
from scipy.stats import kurtosis

random.seed(0)
np.random.seed(0)
msg = pd.read_csv('./LOBSTER/INTC_message_part.csv', header=None, names=['time', 'event', 'index', 'volume', 'price', 'side'])
lob = pd.read_csv('./LOBSTER/INTC_orderbook_part.csv', header=None)
msg_lob = msg.join(lob)
msg_lob.iloc[:, 6:] = msg_lob.iloc[:, 6:].shift(1)
msg_lob = msg_lob.dropna()
msg_lob['mid_price'] = (msg_lob.iloc[:,6] + msg_lob.iloc[:,8]) / 2
msg_lob['volume'] = msg_lob['volume'] // 100
msg_lob.loc[msg_lob.event==4,'price'] = 'market'
target_real = np.array(msg_lob.loc[:,['time','mid_price','event','price','volume']])
target_real[:,1] =  target_real[:,1]/1000

msg = pd.read_csv('./fake_lob/msg_400k2.csv',index_col=0)
lob = pd.read_csv('./fake_lob/lob_400k2.csv',index_col=0)
lob['mid_price'] = (lob.iloc[:,1] + lob.iloc[:,3]) / 2
lob['time'] = lob['time'] / 100 + 36000
target = lob.loc[:,['time','mid_price']]
target = target.join(msg.iloc[:,[1,2,3]])
target_syn = np.array(target)
target_syn[:,1] = target_syn[:,1]



def resample_regular(irregular_ts, interval, start = 36000):
    if irregular_ts[-1,0]>55800:
        end = 55800
    else:
        end = irregular_ts[-1,0]
    num = int((end-start)/interval)
    end = start + num*interval
    time_sequence = np.linspace(start=start, stop=end, endpoint=True, num=num+1)
    price_sequence = np.zeros(np.shape(time_sequence))
    traded_sequence = np.zeros(np.shape(time_sequence)[0])
    for i in range(len(time_sequence)):
        price_sequence[i] = irregular_ts[np.argmax(irregular_ts[:,0]/time_sequence[i]>=1),1]
        if i > 0:
            traded_sequence[i] = np.sum(irregular_ts[(irregular_ts[:,0]>=time_sequence[i-1]) & (irregular_ts[:,0]<=time_sequence[i]) & (irregular_ts[:,3]=='market'),4])
    price_sequence[-1] = irregular_ts[-1,1]
    new_ts = np.concatenate((time_sequence[:,None],price_sequence[:,None]),axis=1)
    ret_ts = np.log(new_ts[1:,1]/new_ts[:-1,1])
    return new_ts, np.concatenate((time_sequence[1:,None],ret_ts[:,None], traded_sequence[1:,None]),axis=1)

def cal_corr(ts, delay_in_secs = 10*60,length=10):
    index = np.argmax((ts[:,0]-36000)/delay_in_secs>=1)+1
    delayed_ts = ts[index:(length+index),1]
    nondelayed_ts = ts[:length,1]
    corr = np.corrcoef(delayed_ts,nondelayed_ts)[0,1]
    p = pearsonr(delayed_ts,nondelayed_ts)[1]
    return corr


def cal_corr_volatility_volume(ts,averaging=10):
    array = np.zeros(shape=(len(ts)//averaging,2))
    for i in range(len(ts)//averaging):
        array[i,0] = np.std(ts[i*averaging:(i+1)*averaging,1])
        array[i,1] = np.mean(ts[i*averaging:(i+1)*averaging,2])
    corr = np.corrcoef(array[:,0],array[:,1])[0,1]
    return corr, array[:,0], array[:,1]


def cal_corr_volatility_return(ts,averaging=10):
    array = np.zeros(shape=(len(ts)//averaging,2))
    for i in range(len(ts)//averaging):
        array[i,0] = np.sum(ts[i*averaging:(i+1)*averaging,1])
        array[i,1] = np.std(ts[i*averaging:(i+1)*averaging,1])
    corr, p = pearsonr(array[:,0],array[:,1])
    return corr, array[:,0], array[:,1]

resampled_price_lowf_syn, resampled_ret_lowf_syn = resample_regular(target_syn,interval=1*60)
kur1 = kurtosis(resampled_ret_lowf_syn[:,1])
resampled_price_hif_syn, resampled_ret_hif_syn = resample_regular(target_syn,interval=1)
kur2 = kurtosis(resampled_ret_hif_syn[:,1])
resampled_price_lowf_real, resampled_ret_lowf_real = resample_regular(target_real,interval=1*60)
kur3 = kurtosis(resampled_ret_lowf_real[:,1])
resampled_price_hif_real, resampled_ret_hif_real = resample_regular(target_real,interval=1)
kur4 = kurtosis(resampled_ret_hif_real[:,1])

# price fluctuation

plt.figure()
length = len(resampled_price_lowf_syn)
f, axes = plt.subplots(1,3,figsize=(16,6))
sns.lineplot(x=resampled_price_lowf_syn[:int(length*0.3),0],y=resampled_price_lowf_syn[:int(length*0.3),1],color='g',ax=axes[0])
axes[0].set_xlabel('time (seconds)',fontsize=12)
axes[0].set_ylabel('price ($)',fontsize=12)
axes[0].set_title('Stage 1',fontsize=16)
sns.lineplot(x=resampled_price_lowf_syn[int(length*0.3):int(length*0.6),0],y=resampled_price_lowf_syn[int(length*0.3):int(length*0.6),1],color='g',ax=axes[1])
axes[1].set_xlabel('time (seconds)',fontsize=12)
axes[1].set_ylabel('price ($)',fontsize=12)
axes[1].set_title('Stage 2',fontsize=16)
sns.lineplot(x=resampled_price_lowf_syn[int(length*0.6):,0],y=resampled_price_lowf_syn[int(length*0.6):,1],color='g',ax=axes[2])
axes[2].set_xlabel('time (seconds)',fontsize=12)
axes[2].set_ylabel('price ($)',fontsize=12)
axes[2].set_title('Stage 3',fontsize=16)
plt.show()


# time distribution

time = np.unique(target_real[:,0])
real = (time[1:] - time[:-1]).astype(float)
syn = (target_syn[1:,0] - target_syn[:-1,0]).astype(float)
dt_list = [syn,real]
name_list = ['Simulated data','Real data']
plt.figure()
f, axes = plt.subplots(1,2,figsize=(12,6))
for index,dt in enumerate(dt_list):
    hist, bins = np.histogram(dt, bins=200, normed=True)
    bin_centers = (bins[1:]+bins[:-1])*0.5
    bin_centers = bin_centers[bin_centers<=1]
    sns.lineplot(bin_centers, np.log(hist[:len(bin_centers)]),label='empirical',ax=axes[index])
    dist_names = ["exponweib",'weibull_min','expon']
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(scipy.stats, dist_name)
        param = dist.fit(dt)
        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = scipy.stats.kstest(dt[list(np.random.randint(0,len(dt),50))], dist_name, args=param)
        print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))
        x = np.linspace(0, 1, 100)
        p = dist.pdf(x, *param)
        sns.lineplot(x,np.log(p),label=dist_name,ax=axes[index])
    axes[index].set_ylabel('log density',fontsize=12)
    axes[index].set_xlabel('time (seconds)',fontsize=12)
    axes[index].legend(fontsize=12)
    axes[index].set_ylim(-6, 4)
    axes[index].set_title(name_list[index],fontsize=16)
plt.show()


# aggregational normality test

plt.figure()
f, axes = plt.subplots(1,2,figsize=(12,6))
sns.distplot(resampled_ret_lowf_syn[:,1],color='r', fit=norm, fit_kws={"color":"red"}, kde=False, bins=8, label='1 minute return',ax=axes[0],)
sns.distplot(resampled_ret_hif_syn[:,1],color='g', fit=norm, fit_kws={"color":"green"}, kde=False, bins=18, label='1 second return',ax=axes[0])
axes[0].set_xlabel('log return',fontsize=12)
axes[0].set_ylabel('frequency',fontsize=12)
axes[0].legend(fontsize=12)
axes[0].set_title('Simulated data',fontsize=16)
axes[0].set_xticks(np.arange(-0.002,0.003,0.001))
sns.distplot(resampled_ret_lowf_real[:,1],color='r', fit=norm, fit_kws={"color":"red"}, kde=False, bins=10, label='1 minute return',ax=axes[1])
sns.distplot(resampled_ret_hif_real[:,1],color='g', fit=norm, fit_kws={"color":"green"}, kde=False, bins=22, label='1 second return',ax=axes[1])
axes[1].set_xlabel('log return',fontsize=12)
axes[1].set_ylabel('frequency',fontsize=12)
axes[1].legend(fontsize=12)
axes[1].set_title('Real data',fontsize=16)
axes[0].set_xticks(np.arange(-0.002,0.003,0.001))
plt.show()


# volatility clustering test

plt.figure()
color_list = ['g','r']
label_list = ['simulated','real']
for index,data in enumerate([target_syn,target_real]):
    corr_list = []
    _, resampled_ret = resample_regular(data, interval=1)
    resampled_ret[:,1] = np.square(resampled_ret[:,1])
    num_samples = 100
    for i in range(num_samples):
        corr_list.append((cal_corr(resampled_ret,delay_in_secs=1*(i+1),length=len(resampled_ret)-num_samples)))
    sns.lineplot(list(range(1,num_samples+1,1)),corr_list, color=color_list[index], label=label_list[index])
plt.xlabel('lambda (seconds)',fontsize=12)
plt.ylabel('correlation',fontsize=12)
plt.title('Volatility clustering',fontsize=16)
plt.legend(fontsize=12)
plt.show()

# volatility and traded volume
plt.figure()
corr_list = []
color_list = ['g','r']
label_list = ['simulated','real']
f, axes = plt.subplots(1,2,figsize=(12,6))
for index,data in enumerate([target_syn,target_real]):
    corr, x, y = cal_corr_volatility_volume(resample_regular(data,interval=1)[1],averaging=300)
    corr_list.append(corr)
    sns.regplot(10000*x,y,color=color_list[index],ax=axes[index])
    axes[index].set_xlabel('standard deviation (1e-4)',fontsize=12)
    axes[index].set_ylabel('average traded volume',fontsize=12)
axes[0].text(2.0,2.7, "corr={}".format(round(corr_list[0],2)), horizontalalignment='left', size='large', color='black', weight='medium',fontsize=12)
axes[0].set_title('Simulated data',fontsize=16)
axes[1].text(1.2,1.5, "corr={}".format(round(corr_list[1],2)), horizontalalignment='left', size='large', color='black', weight='medium',fontsize=12)
axes[1].set_title('Real data',fontsize=16)
plt.show()

# volatility and return

plt.figure()
corr_list = []
color_list = ['g','r']
label_list = ['simulated','real']
f, axes = plt.subplots(1,2,figsize=(12,6))
for index,data in enumerate([target_syn,target_real]):
    corr, x, y = cal_corr_volatility_return(resample_regular(data,interval=1)[1],averaging=300)
    corr_list.append(corr)
    sns.regplot(10000*x,10000*y,color=color_list[index],ax=axes[index])
    axes[index].set_xlabel('log return (1e-4)',fontsize=12)
    axes[index].set_ylabel('standard deviation (1e-4)',fontsize=12)

axes[0].text(15,0.9, "corr={}".format(round(corr_list[0],2)), horizontalalignment='left', size='large', color='black', weight='medium')
axes[0].set_title('Simulated data',fontsize=16)
axes[1].text(15,0.1, "corr={}".format(round(corr_list[1],2)), horizontalalignment='left', size='large', color='black', weight='medium')
axes[1].set_title('Real data',fontsize=16)
plt.show()


