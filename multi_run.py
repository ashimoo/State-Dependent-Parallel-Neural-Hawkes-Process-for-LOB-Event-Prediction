from joblib import Parallel, delayed
from run_model import main
from itertools import combinations

combined=[]
for dataset in ['MSFT','INTC','JPM']:
    for main_module in ['Hawkes','LSTM_PP','SAHP','CTLSTM_PP','CT4LSTM_PPP']:
        for seed in range(5):
            combined.append((dataset,0,main_module,True,seed))

Parallel(n_jobs=15)(delayed(main)(dataset=d, class_loss_weight=w, model=mm, mkt_state=m, seed=seed, gpu=seed%3) for i,(d,w,mm,m,seed) in enumerate(combined))