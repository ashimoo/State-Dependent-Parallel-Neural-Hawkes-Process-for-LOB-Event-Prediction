import numpy as np
import pandas as pd
import torch
import random
np.random.seed(0)

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

class Exchange():
    def __init__(self,ask_side_order_list, bid_side_order_list, initial_quotes, stat_avg, stat_std, dist_dict):

        self.ask_side_order_list_copy = ask_side_order_list.copy() # list of array
        self.bid_side_order_list_copy = bid_side_order_list.copy()
        self.ask_side_order_list = ask_side_order_list # list of array
        self.bid_side_order_list = bid_side_order_list
        self.ask_side_vol_per_level = np.array([np.sum(array) for array in ask_side_order_list])
        self.bid_side_vol_per_level = np.array([np.sum(array) for array in bid_side_order_list])
        self.ask_side_num_orders_per_level = np.array([len(array) for array in ask_side_order_list])
        self.bid_side_num_orders_per_level = np.array([len(array) for array in bid_side_order_list])
        self.quotes = initial_quotes
        self.spread = int((initial_quotes[0] - initial_quotes[1])*100)
        self.status_score = int((initial_quotes[0] - initial_quotes[1]) / (initial_quotes[0] + initial_quotes[1]))
        self.time = 0
        self.message_book = pd.DataFrame(columns=['time','type','price','volume'])
        self.order_book = pd.DataFrame(columns=['time','ask_1','ask_1_vol','bid_1','bid_1_vol','ask_2','ask_2_vol','bid_2','bid_2_vol','ask_3','ask_3_vol','bid_3','bid_3_vol','ask_4','ask_4_vol','bid_4','bid_4_vol','ask_5','ask_5_vol','bid_5','bid_5_vol'])
        self.stat_avg = stat_avg
        self.stat_std = stat_std
        self.dist_dict = dist_dict

    def get_shuffled_order_list(self,side):
        if side == 'ask':
            target = self.ask_side_order_list_copy
        else:
            target = self.bid_side_order_list_copy
        for array in target:
            np.random.shuffle(array)
        return target


    def read_event(self, event_type, delta_t, price, volume):
        self.time = self.time + delta_t
        if event_type == 2: # ask sub
            if isinstance(price,str):
                index_price = np.argmax(np.cumsum(self.bid_side_vol_per_level) / volume > 1)
                residual_on_new_top = np.cumsum(self.bid_side_vol_per_level)[index_price] - volume
                removed_from_new_top = self.bid_side_vol_per_level[index_price] - residual_on_new_top
                self.quotes[1] = self.quotes[1] - index_price * 0.01
                index_order = np.argmax((np.cumsum(self.bid_side_order_list[index_price]) / removed_from_new_top) > 1)
                residual_on_top_order = np.cumsum(self.bid_side_order_list[index_price])[index_order] - removed_from_new_top
                new_order_array_at_that_level = np.insert(self.bid_side_order_list[index_price][(index_order + 1):], 0,
                                                          residual_on_top_order)
                if index_price > 0:
                    self.bid_side_order_list = [new_order_array_at_that_level] + self.bid_side_order_list[(index_price+1):] + self.bid_side_order_list_copy[(5-index_price):]
                else:
                    self.bid_side_order_list[0] = new_order_array_at_that_level

            else:
                if price >= 0:
                    self.ask_side_order_list[price] = np.append(self.ask_side_order_list[price], volume)
                elif price == -1:
                    self.ask_side_order_list = [np.array([volume])] + self.ask_side_order_list[:4]
                    self.quotes[0] = self.quotes[0] - 0.01

        elif event_type == 3:

            if len(self.ask_side_order_list[price]) == 1:
                if price == 0:
                    volume = self.ask_side_order_list[0][0]
                    self.ask_side_order_list = self.ask_side_order_list[1:] + [self.ask_side_order_list_copy[4]]
                    self.quotes[0] = self.quotes[0] + 0.01
                else:
                    price = np.argmax(self.ask_side_vol_per_level)
                    if len(self.ask_side_order_list[price])>1:
                        volume = self.ask_side_order_list[price][0]
                        self.ask_side_order_list[price] = self.ask_side_order_list[price][1:]
                    else:
                        volume = 1
                        self.ask_side_order_list[price][0] = self.ask_side_order_list[price][0] - volume
            else:
                if (np.random.random(1)[0] > 0.9) & (self.ask_side_order_list[price][0]>=5):
                    volume = np.random.randint(low=self.ask_side_order_list[price][0]//2,high=self.ask_side_order_list[price][0]-1,size=1)[0]
                    self.ask_side_order_list[price] = np.insert(self.ask_side_order_list[price][1:],0,self.ask_side_order_list[price][0]-volume)
                else:
                    volume = self.ask_side_order_list[price][0]
                    self.ask_side_order_list[price] = self.ask_side_order_list[price][1:]

        elif event_type == 0:
            if isinstance(price, str):
                index_price = np.argmax(np.cumsum(self.ask_side_vol_per_level) / volume > 1)
                residual_on_new_top = np.cumsum(self.ask_side_vol_per_level)[index_price] - volume
                removed_from_new_top = self.ask_side_vol_per_level[index_price] - residual_on_new_top
                self.quotes[0] = self.quotes[0] + index_price * 0.01
                index_order = np.argmax((np.cumsum(self.ask_side_order_list[index_price]) / removed_from_new_top) > 1)
                residual_on_top_order = np.cumsum(self.ask_side_order_list[index_price])[
                                            index_order] - removed_from_new_top
                new_order_array_at_that_level = np.insert(self.ask_side_order_list[index_price][(index_order + 1):], 0,
                                                          residual_on_top_order)
                if index_price > 0:
                    self.ask_side_order_list = [new_order_array_at_that_level] + self.ask_side_order_list[(index_price + 1):] + self.ask_side_order_list_copy[(5 - index_price):]
                else:
                    self.ask_side_order_list[0] = new_order_array_at_that_level

            else:
                if price >= 0:
                    self.bid_side_order_list[price] = np.append(self.bid_side_order_list[price], volume)
                elif price == -1:
                    self.bid_side_order_list = [np.array([volume])] + self.bid_side_order_list[:4]
                    self.quotes[1] = self.quotes[1] + 0.01

        elif event_type == 1:

            if len(self.bid_side_order_list[price]) == 1:
                if price == 0:
                    volume = self.bid_side_order_list[0][0]
                    self.bid_side_order_list = self.bid_side_order_list[1:] + [self.bid_side_order_list_copy[4]]
                    self.quotes[1] = self.quotes[1] - 0.01
                else:
                    price = np.argmax(self.bid_side_num_orders_per_level)
                    if len(self.bid_side_order_list[price])>1:
                        volume = self.bid_side_order_list[price][0]
                        self.bid_side_order_list[price] = self.bid_side_order_list[price][1:]
                    else:
                        volume = 1
                        self.bid_side_order_list[price][0] = self.bid_side_order_list[price][0] - volume
            else:
                if (np.random.random(1)[0] > 0.9) & (self.bid_side_order_list[price][0]>=5):
                    volume = np.random.randint(low=self.bid_side_order_list[price][0]//2,high=self.bid_side_order_list[price][0]-1,size=1)[0]
                    self.bid_side_order_list[price] = np.insert(self.bid_side_order_list[price][1:],0,self.bid_side_order_list[price][0]-volume)
                else:
                    volume = self.bid_side_order_list[price][0]
                    self.bid_side_order_list[price] = self.bid_side_order_list[price][1:]
        self.bid_side_vol_per_level = np.array([np.sum(array) for array in self.bid_side_order_list])
        self.bid_side_num_orders_per_level = np.array([len(array) for array in self.bid_side_order_list])
        self.ask_side_vol_per_level = np.array([np.sum(array) for array in self.ask_side_order_list])
        self.ask_side_num_orders_per_level = np.array([len(array) for array in self.ask_side_order_list])

        return event_type, price, volume


    def update_order_book(self):
        self.order_book = self.order_book.append({
        'time': self.time,
        'ask_1': self.quotes[0],
        'ask_1_vol': self.ask_side_vol_per_level[0],
        'bid_1': self.quotes[1],
        'bid_1_vol': self.bid_side_vol_per_level[0],
        'ask_2': self.quotes[0] + 0.01,
        'ask_2_vol': self.ask_side_vol_per_level[1],
        'bid_2': self.quotes[1] - 0.01,
        'bid_2_vol': self.bid_side_vol_per_level[1],
        'ask_3': self.quotes[0] + 0.02,
        'ask_3_vol': self.ask_side_vol_per_level[2],
        'bid_3': self.quotes[1] - 0.02,
        'bid_3_vol': self.bid_side_vol_per_level[2],
        'ask_4':  self.quotes[0] + 0.03,
        'ask_4_vol': self.ask_side_vol_per_level[3],
        'bid_4': self.quotes[1] - 0.03,
        'bid_4_vol': self.bid_side_vol_per_level[3],
        'ask_5':  self.quotes[0] + 0.04,
        'ask_5_vol': self.ask_side_vol_per_level[4],
        'bid_5': self.quotes[1] - 0.04,
        'bid_5_vol': self.bid_side_vol_per_level[4],
        },ignore_index=True)
        a = self.order_book.iloc[-1, 1] - self.order_book.iloc[-1,3]
        self.spread = round((self.order_book.iloc[-1,1] - self.order_book.iloc[-1,3])*100)


    def update_message_book(self,event_type,price,volume):
        self.message_book = self.message_book.append({
            'time': self.time,
            'type': event_type,
            'price': price,
            'volume': volume,
        },ignore_index=True)

    def get_mkt_score(self):
        self.status_score = (self.order_book.iloc[-1, 4] - self.order_book.iloc[-1, 2]) / \
                                  (self.order_book.iloc[-1, 2] + self.order_book.iloc[-1, 4])
        if self.status_score > 0.4:
            score = 2
        elif self.status_score < -0.4:
            score = 0
        else:
            score = 1
        return score

    def print_lob(self):
        return self.order_book

    def print_msg(self):
        return self.message_book

    def pred_cluster(self,clustering_model_bid,clustering_model_ask):
        bid_vol = (np.array(self.order_book.iloc[-1,list(range(4,24,4))])-self.stat_avg[[1,3,5,7,9]])/self.stat_std[[1,3,5,7,9]]
        ask_vol = (np.array(self.order_book.iloc[-1,list(range(2,22,4))])-self.stat_avg[[0,2,4,6,8]])/self.stat_std[[0,2,4,6,8]]
        spread_indicator = np.concatenate(
            (np.array(self.spread == 1)[None,None], np.array(self.spread == 2)[None,None]), axis=-1)
        bid_cluster_index = clustering_model_bid.predict(np.concatenate((bid_vol[None,:],spread_indicator),axis=-1))
        ask_cluster_index = clustering_model_ask.predict(np.concatenate((ask_vol[None,:], spread_indicator), axis=-1))
        bid_cluster_index = torch.nn.functional.one_hot(torch.Tensor(bid_cluster_index).cuda().type(torch.int64),num_classes=8).type(torch.float)
        ask_cluster_index = torch.nn.functional.one_hot(torch.Tensor(ask_cluster_index).cuda().type(torch.int64),num_classes=8).type(torch.float)
        return [bid_cluster_index,ask_cluster_index]













