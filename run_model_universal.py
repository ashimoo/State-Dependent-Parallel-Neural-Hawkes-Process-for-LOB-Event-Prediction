import torch
import pandas as pd
import numpy as np
import utils
import os
from matplotlib import pyplot as plt
from scipy.stats.mstats import winsorize
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import random
import argparse
from utils import *
from model.model_SAHP import SAHP
from model.model_Hawkes import Hawkes
from model.model_LSTM_PP import LSTM_PP
from model.model_CTLSTM_PP import CTLSTM_PP
from model.model_CT4LSTM_PPP import CT4LSTM_PPP

def define_args():
    parser = argparse.ArgumentParser('LOB event prediction')
    parser.add_argument('--dataset',  type=str, default="INTC", help="dataset used")
    parser.add_argument('--class_loss_weight', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--model', type=str, default='CT4LSTM_PPP')
    parser.add_argument('--mkt_state', action='store_true', default=True)
    parser.add_argument('--seed',type=int, default=0)
    return parser.parse_args()

def main(dataset,class_loss_weight,model,mkt_state,seed,gpu):
    args = define_args()
    args.dataset = dataset
    args.class_loss_weight = class_loss_weight
    args.model = model
    args.mkt_state = mkt_state
    args.seed = seed
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda')
    len_of_record = 50
    batch_size = 512
    niter = 200
    lr = args.lr
    model_path_known_t = './models/{}_{}_mkt-{}_coef-{}_{}rms{}_new.mdl'.format(args.dataset,args.model,args.mkt_state,args.class_loss_weight,args.lr,2e-3)
    log_path = './logs/{}_{}_mkt-{}_coef-{}_{}rms{}_new.log'.format(args.dataset,args.model,args.mkt_state,args.class_loss_weight,args.lr,2e-3)
    logger = open(log_path, "w")

    train_record_list = []
    val_record_list = []
    test_record_list = []
    for i in range(5):
        df_ = pd.read_csv('./LOBSTER/parsed_event_df/{}_event_df_{}.csv'.format(args.dataset,i+1),index_col=0)
        delta_time = np.array(df_.time[1:]) - np.array(df_.time[:-1])
        df = df_[1:]
        df.time = delta_time
        if i < 3:
            train_record_list = df_to_list(df,len_of_record) + train_record_list
        elif i == 3:
            val_record_list = df_to_list(df,len_of_record)
        elif i == 4:
            test_record_list = df_to_list(df,len_of_record)

    data_obj = parse_datasets_separate(device,batch_size,train_record_list[:(len(train_record_list)//batch_size)*batch_size],val_record_list[:(len(val_record_list)//batch_size)*batch_size],test_record_list[:(len(test_record_list)//batch_size)*batch_size])

    if args.model == 'CTLSTM_PP':
        model = CTLSTM_PP(num_events = 4, hidden_dim = 16, input_embed_dim = 16)
    elif args.model == 'LSTM_PP':
        model = LSTM_PP(num_events = 4, hidden_dim = 16, input_embed_dim = 16)
    elif args.model == 'SAHP':
        model = SAHP(num_events = 4, hidden_dim = 16, input_embed_dim = 16)
    elif args.model == 'Hawkes':
        model = Hawkes(num_events = 4, hidden_dim = 16, input_embed_dim = 16)
    elif args.model == 'CT4LSTM_PPP':
        model = CT4LSTM_PPP(num_events = 4, hidden_dim = 16, input_embed_dim = 16)

    model = model.cuda()

    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    average_loss = []
    accuracy_list = []
    val_accuracy_list_known_t = []
    for itr in tqdm(range(1, data_obj['n_train_batches'] * (niter + 1))):
        optimizer.zero_grad()
        batch_dict = get_next_batch(data_obj["train_dataloader"])
        update_learning_rate(optimizer, decay_rate=0.999, lowest=2e-3)
        intens, cells, cell_targets, outputs, decays = model(batch_dict, args.mkt_state, device = torch.device("cuda"))
        loss, hawkes_loss, pred_accuracy = model.compute_loss(batch_dict, intens, cells, cell_targets, outputs, decays, args.class_loss_weight)
        loss.backward()
        optimizer.step()
        average_loss.append(hawkes_loss)
        accuracy_list.append(pred_accuracy)
        if itr % data_obj['n_train_batches'] == 0:
            torch.cuda.empty_cache()
            logger.write('Epoch {:04d} Hawkes_loss: {:.4f} Accuracy_score: {:.4f}'.format(itr // data_obj['n_train_batches'],sum(average_loss) / len(average_loss),sum(accuracy_list) / len(accuracy_list)))
            if itr % (10 * data_obj['n_train_batches']) == 0:
                target_array = []
                pred_array_known_time = []
                hawkes_loss_val_list = []
                count_sample_val = 0
                for i in tqdm(range(data_obj['n_val_batches'])):
                    batch_dict = get_next_batch(data_obj['val_dataloader'])
                    target, pred_known_time, hawkes_loss_val= model.read_predict(batch_dict, args.mkt_state, metrics=False)
                    target_array.append(target)
                    pred_array_known_time.append(pred_known_time)
                    hawkes_loss_val_list.append(hawkes_loss_val)
                target_array = np.hstack(target_array).reshape(-1)
                pred_array_known_time = np.hstack(pred_array_known_time).reshape(-1)
                message = 'Validation Hawkes_loss {:.4f} Accuracy_score_known_t {:.4f}'.format((sum(hawkes_loss_val_list) / len(hawkes_loss_val_list)), accuracy_score(target_array,pred_array_known_time))
                logger.write(message)
                val_accuracy_list_known_t.append(sum(hawkes_loss_val_list) / len(hawkes_loss_val_list))
                if val_accuracy_list_known_t[-1] == np.min(val_accuracy_list_known_t):
                    torch.save(model,model_path_known_t)
            accuracy_list = []
            average_loss = []

    model = torch.load(model_path_known_t)
    target_array = []
    pred_array_known_time = []
    pred_array_unknown_time = []
    hawkes_loss_test_list = []
    error_dt_test_list = []
    for i in tqdm(range(data_obj['n_test_batches'])):
        batch_dict = get_next_batch(data_obj['test_dataloader'])
        target, pred_known_time, pred_unknown_time, hawkes_loss_test, error_dt_test = model.read_predict(batch_dict, args.mkt_state,metrics=True)
        target_array.append(target)
        pred_array_known_time.append(pred_known_time)
        pred_array_unknown_time.append(pred_unknown_time)
        hawkes_loss_test_list.append(hawkes_loss_test)
        error_dt_test_list.append(error_dt_test)
    target_array = np.hstack(target_array).reshape(-1)
    pred_array_known_time = np.hstack(pred_array_known_time).reshape(-1)
    pred_array_unknown_time = np.hstack(pred_array_unknown_time).reshape(-1)
    message = 'Test Hawkes_loss {:.4f} Accuracy_score_known_t {:.4f} Accuracy_score_unknown_t {:.4f} Time error {:.4f}'.format((sum(hawkes_loss_test_list) / len(hawkes_loss_test_list)), accuracy_score(target_array,pred_array_known_time), accuracy_score(target_array,pred_array_unknown_time), (sum(error_dt_test_list) / len(error_dt_test_list)))
    logger.write(message)
    print(message)
    logger.close()
if __name__ == '__main__':
    main()








