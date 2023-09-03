import torch
import pandas as pd
import numpy as np

for i in range(5):
    path_msg_1 = './LOBSTER/JPM/JPM_message_part_{}.csv'.format(i+1)
    path_lob_1 = './LOBSTER/JPM/JPM_orderbook_part_{}.csv'.format(i+1)

    msg = pd.read_csv(path_msg_1, header = None, names = ['time','event','index','volume','price','side'])
    lob = pd.read_csv(path_lob_1, header = None)
    msg_lob = msg.join(lob)
    msg_lob.iloc[:,6:] = msg_lob.iloc[:,6:].shift(1)
    msg_lob = msg_lob.dropna()
    msg_lob = msg_lob.loc[(msg_lob['time']>36000.0)&(msg_lob['time']<55800.0),:]
    msg_lob = msg_lob.loc[msg_lob['volume']>=100,:]
    msg_lob = msg_lob.loc[msg_lob['price']%100==0,:]
    msg_lob['time'] = msg_lob['time'] * 1000

    # -1 is orders on ask side; +1 is orders on bid side; -1 market order is order being bought on ask side.
    msg_lob = msg_lob.drop(msg_lob[(msg_lob.iloc[:,1]==5)].index)   # do not consider hidden order execution
    msg_lob.loc[(msg_lob['event'] == 4) & (msg_lob['side'] == -1),'event'] = 'A' # buy market order
    msg_lob.loc[(msg_lob['event'] == 4) & (msg_lob['side'] == 1),'event'] = 'C'  # sell market order
    msg_lob.loc[msg_lob['event'] == 3,'event'] = 2
    msg_lob.loc[(msg_lob['event'] == 2) & (msg_lob['side'] == -1),'event'] = 'D'  # cancel ask side order
    msg_lob.loc[(msg_lob['event'] == 2) & (msg_lob['side'] == 1),'event'] = 'B'  # cancel bid side order
    msg_lob.loc[(msg_lob['event'] == 1) & (msg_lob['side'] == -1),'event'] = 'C'  # submit ask side order
    msg_lob.loc[(msg_lob['event'] == 1) & (msg_lob['side'] == 1),'event'] = 'A'  # submit bid side order

    event_list = ['A','B','C','D']
    msg_lob = msg_lob.drop(['index','side'], axis=1, inplace=False)
    time_index = msg_lob['time'].unique()
    event_df = []

    count_overlapping = 0
    for index,time in enumerate(time_index):
        target = msg_lob[msg_lob.time == time]
        time_list = None
        if len(target.event.unique())>1:
            count_overlapping = count_overlapping + 1
            time_1 = time
            if index+1 != len(time_index):
                time_2 = time_index[index+1]
            else:
                time_2 = time + 1
            time_list = np.linspace(time_1,time_2,num=len(target.event.unique()),endpoint=False)
        count = 0
        for event in target.event.unique():
            sub_target = target[target.event == event]
            if len(sub_target) > 1:
                sub_target.iloc[0,2] = sub_target.volume.sum()
                event_df.append(np.array(sub_target.iloc[0,:]).reshape(-1))
                if time_list is not None:
                    event_df[-1][0] = time_list[count]
                    count = count + 1
            else:
                event_df.append(np.array(sub_target).reshape(-1))
                if time_list is not None:
                    event_df[-1][0] = time_list[count]
                    count = count + 1
        print(index)

    event_df = pd.DataFrame((np.array(event_df)))
    event_df.columns = ['time','event','volume','price'] + list(range(20))
    event_df.loc[event_df.event == 'A', 'event'] = 1
    event_df.loc[event_df.event == 'B', 'event'] = 2
    event_df.loc[event_df.event == 'C', 'event'] = 3
    event_df.loc[event_df.event == 'D', 'event'] = 4

    event_df['score'] = (event_df[3] - event_df[1]) / (event_df[3] + event_df[1])
    event_df['state'] = 0
    event_df.loc[event_df.score > 0.4, 'state'] = 2
    event_df.loc[event_df.score < -0.4, 'state'] = 0
    event_df.loc[(event_df.score >= -0.4) & (event_df.score <= 0.4), 'state'] = 1

    event_df.to_csv('./LOBSTER/parsed_event_df/INTC_event_df_{}.csv'.format(i+1))