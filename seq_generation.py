import torch
import numpy as np
import utils
import os

from utils import *
from model.model_SAHP import SAHP
from model.model_CT4LSTM_PPP import CT4LSTM_PPP
from exchange_NEW import Exchange
from sklearn.cluster import KMeans
from tqdm import tqdm
import joblib
np.random.seed(0)

# a = pd.read_csv('./fake_lob/lob_____.csv')
# b = a.std()
# b1 = b[[3,5]].mean()
# b2 = b[[7,9,11,13,15,17,19,21]].mean()
# c = a.mean()
# c1 = c[[3,5]].mean()
# c2 = c[[7,9,11,13,15,17,19,21]].mean()
# aa = np.sum(np.round((np.array(a.ask_1) - np.array(a.bid_1))*100)>=2)


device = torch.device('cuda:0')
len_of_record = 50
batch_size = 1
df_ = pd.read_csv('./LOBSTER/parsed_event_df/INTC_event_df_5.csv',index_col=0)
df_.loc[df_.event=='A','event'] = 1
df_.loc[df_.event=='B','event'] = 2
df_.loc[df_.event=='C','event'] = 3
df_.loc[df_.event=='D','event'] = 4
delta_time = np.array(df_.time[1:]) - np.array(df_.time[:-1])
df = df_[1:]
df.time = delta_time
a = len(df[df.event==1])
b = len(df[df.event==2])
c = len(df[df.event==3])
d = len(df[df.event==4])
df['score'] = (df['3'] - df['1'])/(df['3']+df['1'])
df['state'] = 0
df.loc[df.score>0.4,'state'] = 2
df.loc[df.score<-0.4,'state'] = 0
df.loc[(df.score>=-0.4)&(df.score<=0.4),'state'] = 1
df.iloc[:,list(range(5,25,2))] = df.iloc[:,list(range(5,25,2))]//100
print(len(df[df.state==0]))
print(len(df[df.state==1]))
print(len(df[df.state==2]))
state_list = []


stat_avg = np.array(df.mean())[[5,7,9,11,13,15,17,19,21,23]]
avg1 = stat_avg[[0,1]].mean()
avg2 = stat_avg[[2,3,4,5,6,7,8,9]].mean()
stat_std = np.array(df.std())[[5,7,9,11,13,15,17,19,21,23]]
std1 = stat_std[[0,1]].mean()
std2 = stat_std[[2,3,4,5,6,7,8,9]].mean()

def from_param_to_cdf(a,m):
    x = np.arange(0, m, dtype='float')
    pmf = 1/(x+1)**a
    pmf /= pmf.sum()
    return pmf.cumsum()

def sample_from_dist(cdf):
    return np.argmax(cdf/np.random.random(1)[0]>=1)

def truncated_power_law(a, m):
    x = np.arange(0, m, dtype='float')
    pmf = 1/(x+1)**a
    pmf /= pmf.sum()
    return pmf

def gen_from_dist_given_total_vol(avg_vol,dist_param_list):
    list_of_the_side = []
    for index,avg_vol_each_level in enumerate(avg_vol):
        order_list = []
        cdf = from_param_to_cdf(a=dist_param_list[index][1], m=dist_param_list[index][2])
        while True:
            order_list = order_list + [sample_from_dist(cdf)+1]
            if (sum(order_list) > avg_vol_each_level):
                break
        np.random.shuffle(order_list)
        list_of_the_side.append(np.array(order_list))
    return list_of_the_side




dist_dict = np.load('./param_dict_95_INTC_5thday.npy',allow_pickle=True).item()
ask_side_order_list = gen_from_dist_given_total_vol(avg_vol = [130,155,165,170,160], dist_param_list= dist_dict['limit_ask_sub_vol_1'])
bid_side_order_list = gen_from_dist_given_total_vol(avg_vol = [130,155,165,170,160], dist_param_list= dist_dict['limit_bid_sub_vol_1'])

exchange = Exchange(ask_side_order_list= ask_side_order_list, bid_side_order_list = bid_side_order_list, initial_quotes = [27.01,27.00], stat_avg=stat_avg,stat_std=stat_std, dist_dict = dist_dict)
record_list = df_to_list(df,len_of_record)
data_obj = parse_datasets(device,batch_size,record_list)

batch_dict = get_next_batch(data_obj["train_dataloader"])

batch_dict['time_step'] = batch_dict['time_step'].to(torch.float32)
batch_dict['mkt_state'] = batch_dict['mkt_state'].to(torch.float32)

gen_num = 5000
model = torch.load('./models/INTC_CT4LSTM_PPP_mkt-True_coef-0_0.02rms0.002.mdl')

gen_types_list = []
gen_times_list = []


# spread_indicator = np.concatenate((np.array(df.iloc[:,25]==1)[:,None],np.array(df.iloc[:,25]==2)[:,None]),axis=-1)
# ask_vol_array = (np.array(df.iloc[:,list(range(5,25,4))]) - stat_avg[[0,2,4,6,8]]) / stat_std[[0,2,4,6,8]]
# # ask_vol_array = np.array(df.iloc[:,list(range(5,25,4))])
# ask_vol_array = np.concatenate((ask_vol_array,spread_indicator),axis=-1)
# bid_vol_array = (np.array(df.iloc[:,list(range(7,27,4))]) - stat_avg[[1,3,5,7,9]]) / stat_std[[1,3,5,7,9]]
# # bid_vol_array = np.array(df.iloc[:,list(range(7,27,4))])
# bid_vol_array = np.concatenate((bid_vol_array,spread_indicator),axis=-1)
#
# ask_state_label = KMeans(n_clusters=8,random_state=0).fit(ask_vol_array)
# bid_state_label = KMeans(n_clusters=8,random_state=0).fit(bid_vol_array)

# ask_state_label = joblib.load('./state_clus_model/ask_clustering_model.pkl')
# bid_state_label = joblib.load('./state_clus_model/bid_clustering_model.pkl')
#
# bid_price_sub_model = torch.load('./price_param_model/bid_price_sub_model.mdl')
# bid_price_cancel_model = torch.load('./price_param_model/bid_price_cancel_model.mdl')
# ask_price_sub_model = torch.load('./price_param_model/ask_price_sub_model.mdl')
# ask_price_cancel_model = torch.load('./price_param_model/ask_price_cancel_model.mdl')

dist_dict['pctg_market_bid_1'] = dist_dict['pctg_market_ask_1']
dist_dict['pctg_market_bid_2'] = dist_dict['pctg_market_ask_2']
dist_dict['limit_bid_sub_price_1'] = dist_dict['limit_ask_sub_price_1']
dist_dict['limit_bid_sub_price_2'] = dist_dict['limit_ask_sub_price_2']
dist_dict['limit_bid_cancel_price_2'] = dist_dict['limit_ask_cancel_price_2']
dist_dict['limit_bid_cancel_price_1'] = dist_dict['limit_ask_cancel_price_1']
dist_dict['limit_bid_sub_vol_1'] = dist_dict['limit_ask_sub_vol_1']
dist_dict['limit_bid_sub_vol_2'] = dist_dict['limit_ask_sub_vol_2']
dist_dict['limit_bid_cancel_vol_2'] = dist_dict['limit_ask_cancel_vol_2']
dist_dict['limit_bid_cancel_vol_1'] = dist_dict['limit_ask_cancel_vol_1']
dist_dict['market_bid_sub_vol_1'] = dist_dict['market_ask_sub_vol_1']
dist_dict['market_bid_sub_vol_2'] = dist_dict['market_ask_sub_vol_2']

state_tran = torch.load('./state_trans_prob0.torch').detach().cpu().numpy()
event_count_list = [0,0,0,0]
count=0
buy_neutral_sell_pctg = np.array([0.3,0.6,1])
buy_mkt_pctg = [[0.05,0.01],[0.025,0.005],[0.025,0.005]]
sell_mkt_pctg = [[0.025,0.005],[0.025,0.005],[0.05,0.01]]
for i in tqdm(range(gen_num)):
    current_index = np.argmax(buy_neutral_sell_pctg/((i+1)/gen_num)>=1)
    exchange.update_order_book()
    spread = exchange.spread
    mkt = exchange.status_score
    pred_time_list, pred_type_list = model.read_iterative_sampling(batch_dict,mkt_state=True)
    if i>100:
        if (pred_type_list[0]==0 and event_count_list[1]*1.05<=event_count_list[0]):
            pred_type=1
            pred_time=pred_time_list[np.nonzero(pred_type_list==1)[0][0]]
            count = count+1
        elif (pred_type_list[0] == 2 and event_count_list[3] * 1.05 <= event_count_list[2]):
            pred_type = 3
            pred_time = pred_time_list[np.nonzero(pred_type_list == 3)[0][0]]
            count = count + 1
        else:
            pred_type=pred_type_list[0]
            pred_time=pred_time_list[0]
    else:
        pred_type = pred_type_list[0]
        pred_time = pred_time_list[0]

    if spread == 1:
        if pred_type == 0:
            if np.random.random(1)[0] <= buy_mkt_pctg[current_index][0]:
                price = 'market'
                volume = sample_from_dist(from_param_to_cdf(dist_dict['market_bid_sub_vol_1'][1],m=dist_dict['market_bid_sub_vol_1'][2])) + 1
            else:
                price = sample_from_dist(from_param_to_cdf(dist_dict['limit_bid_sub_price_1'][1],m=5))
                volume = sample_from_dist(from_param_to_cdf(dist_dict['limit_bid_sub_vol_1'][price][1],m=dist_dict['limit_bid_sub_vol_1'][price][2])) + 1
        elif pred_type == 1:
            price = sample_from_dist(from_param_to_cdf(dist_dict['limit_bid_cancel_price_1'][1],m=5))
            volume = sample_from_dist(from_param_to_cdf(dist_dict['limit_bid_cancel_vol_1'][price][1],m=dist_dict['limit_bid_cancel_vol_1'][price][2])) + 1
        elif pred_type == 2:
            if np.random.random(1)[0] <= sell_mkt_pctg[current_index][0]:
                price = 'market'
                volume = sample_from_dist(from_param_to_cdf(dist_dict['market_ask_sub_vol_1'][1],m=dist_dict['market_ask_sub_vol_1'][2])) + 1
            else:
                price = sample_from_dist(from_param_to_cdf(dist_dict['limit_ask_sub_price_1'][1],m=5))
                volume = sample_from_dist(from_param_to_cdf(dist_dict['limit_ask_sub_vol_1'][price][1],m=dist_dict['limit_ask_sub_vol_1'][price][2])) + 1
        elif pred_type == 3:
            price = sample_from_dist(from_param_to_cdf(dist_dict['limit_ask_cancel_price_1'][1], m=5))
            volume = sample_from_dist(from_param_to_cdf(dist_dict['limit_ask_cancel_vol_1'][price][1],m=dist_dict['limit_ask_cancel_vol_1'][price][2])) + 1
    elif spread == 2:
        if pred_type == 0:
            if np.random.random(1)[0] <= buy_mkt_pctg[current_index][1]:
                price = 'market'
                volume = sample_from_dist(from_param_to_cdf(dist_dict['market_bid_sub_vol_2'][1],m=dist_dict['market_bid_sub_vol_2'][2])) + 1
            else:
                if np.random.random(1)[0] <= 0.1:
                    price = -1
                    volume = sample_from_dist(from_param_to_cdf(dist_dict['limit_bid_sub_vol_2'][0][1], m=dist_dict['limit_bid_sub_vol_2'][0][2])) + 1
                else:
                    price = sample_from_dist(from_param_to_cdf(dist_dict['limit_bid_sub_price_2'][1],m=5))
                    volume = sample_from_dist(from_param_to_cdf(dist_dict['limit_bid_sub_vol_2'][price+1][1],m=dist_dict['limit_bid_sub_vol_2'][price+1][2])) + 1
        elif pred_type == 1:
            price = sample_from_dist(from_param_to_cdf(dist_dict['limit_bid_cancel_price_2'][1],m=5))
            volume = sample_from_dist(from_param_to_cdf(dist_dict['limit_bid_cancel_vol_2'][price][1],m=dist_dict['limit_bid_cancel_vol_2'][price][2])) + 1
        elif pred_type == 2:
            if np.random.random(1)[0] <= sell_mkt_pctg[current_index][1]:
                price = 'market'
                volume = sample_from_dist(from_param_to_cdf(dist_dict['market_ask_sub_vol_2'][1],m=dist_dict['market_ask_sub_vol_2'][2])) + 1
            else:
                if np.random.random(1)[0] <= 0.1:
                    price = -1
                    volume = sample_from_dist(from_param_to_cdf(dist_dict['limit_ask_sub_vol_2'][0][1], m=dist_dict['limit_ask_sub_vol_2'][0][2])) + 1
                else:
                    price = sample_from_dist(from_param_to_cdf(dist_dict['limit_ask_sub_price_2'][1],m=5))
                    volume = sample_from_dist(from_param_to_cdf(dist_dict['limit_ask_sub_vol_2'][price+1][1],m=dist_dict['limit_ask_sub_vol_2'][price+1][2])) + 1
        elif pred_type == 3:
            price = sample_from_dist(from_param_to_cdf(dist_dict['limit_ask_cancel_price_2'][1], m=5))
            volume = sample_from_dist(from_param_to_cdf(dist_dict['limit_ask_cancel_vol_2'][price][1],m=dist_dict['limit_ask_cancel_vol_2'][price][2])) + 1
    else:
        if pred_type == 0:
            price = -1
            volume = sample_from_dist(from_param_to_cdf(dist_dict['limit_bid_sub_vol_2'][0][1], m=dist_dict['limit_bid_sub_vol_2'][0][2])) + 1
        elif pred_type == 1:
            price = sample_from_dist(from_param_to_cdf(dist_dict['limit_bid_cancel_price_2'][1],m=5))
            volume = sample_from_dist(from_param_to_cdf(dist_dict['limit_bid_cancel_vol_2'][price][1],m=dist_dict['limit_bid_cancel_vol_2'][price][2])) + 1
        elif pred_type == 2:
            price = -1
            volume = sample_from_dist(from_param_to_cdf(dist_dict['limit_ask_sub_vol_2'][0][1], m=dist_dict['limit_ask_sub_vol_2'][0][2])) + 1
        elif pred_type == 3:
            price = sample_from_dist(from_param_to_cdf(dist_dict['limit_ask_cancel_price_2'][1],m=5))
            volume = sample_from_dist(from_param_to_cdf(dist_dict['limit_ask_cancel_vol_2'][price][1],m=dist_dict['limit_ask_cancel_vol_2'][price][2])) + 1
    event_count_list[pred_type] = event_count_list[pred_type] + 1
    gen_types_list.append(pred_type)
    # print(pred_type,format(pred_time,'.4f'),price,volume)
    event_type_, price_, volume_ = exchange.read_event(event_type=pred_type, delta_t = pred_time, price=price, volume=volume)
    exchange.update_message_book(event_type_, price_, volume_)
    batch_dict['event'][:,-1] = pred_type
    batch_dict['event'][:,:-1] = batch_dict['event'].clone()[:,1:]
    batch_dict['event'][:,-1] = 0
    batch_dict['time_step'][:,:-1] = batch_dict['time_step'].clone()[:,1:]
    batch_dict['time_step'][:,-1] = pred_time
    batch_dict['mkt_state'][:,-1] = exchange.get_mkt_score()
    batch_dict['mkt_state'][:,:-1] = batch_dict['mkt_state'].clone()[:,1:]
    batch_dict['mkt_state'][:,-1] = 0.1
    # state_prob = state_tran[pred_type]
    # a = np.cumsum(state_prob[int(batch_dict['mkt_state'][0,-2].item())])/np.random.random(1)[0]
    # state = np.argmax(np.cumsum(state_prob[int(batch_dict['mkt_state'][0,-2].item())])/np.random.random(1)[0] >=1)
    # batch_dict['mkt_state'][:, -1] = state
    # batch_dict['mkt_state'][:,:-1] = batch_dict['mkt_state'].clone()[:,1:]
    # batch_dict['mkt_state'][:,-1] = 0.1
    # state_list.append(exchange.get_mkt_score())
    if i%25000==1:
        exchange.print_lob().to_csv('./fake_lob/lob_{}.csv'.format(i))
        exchange.print_msg().to_csv('./fake_lob/msg_{}.csv'.format(i))
        print(count)
print(np.sum(np.array(state_list)==0))
print(np.sum(np.array(state_list)==1))
print(np.sum(np.array(state_list)==2))
LOB = exchange.print_lob()
MSG = exchange.print_msg()
LOB.to_csv('./fake_lob/lob_____.csv')
MSG.to_csv('./fake_lob/msg_____.csv')


print(np.sum(np.array(gen_types_list)==0))
print(np.sum(np.array(gen_types_list)==1))
print(np.sum(np.array(gen_types_list)==2))
print(np.sum(np.array(gen_types_list)==3))
print(count)




