from torch.utils.data import DataLoader
from sklearn import model_selection
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import logging
from torch.autograd import Variable

class MaskBatch():
    "object for holding a batch of data with mask during training"

    def __init__(self, src, pad, device):
        self.src = src
        self.src_mask = self.make_std_mask(self.src, pad, device)

    @staticmethod
    def make_std_mask(tgt, pad, device):
        "create a mask to hide padding and future input"
        # torch.cuda.set_device(device)
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)).to(device)
        return tgt_mask

def subsequent_mask(size):
    "mask out subsequent positions"
    atten_shape = (1, size, size)
    # np.triu: Return a copy of a matrix with the elements below the k-th diagonal zeroed.
    mask = np.triu(np.ones(atten_shape), k=1).astype('uint8')
    aaa = torch.from_numpy(mask) == 0
    return aaa

def df_to_list(df,len_of_record):
    records = []
    df = np.array(df)
    num_batches = len(df)//len_of_record

    for i in range(len(df) - len_of_record + 1):
        records.append(df[i:i + len_of_record, [0,1,-1]])
    return(records)

def parse_datasets(device,batch_size,dataset,train_percentage=0.8):
    total_dataset = dataset

    # Shuffle and split
    if train_percentage > 0:
        train_data, test_data = model_selection.train_test_split(total_dataset, train_size= train_percentage, shuffle = False)
    else:
        test_data = total_dataset

    if train_percentage > 0:
        train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=False,
            collate_fn = lambda batch: variable_time_collate_fn(batch, device))
        test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle=False,
            collate_fn= lambda batch: variable_time_collate_fn(batch, device))
    else:
        test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle=False,
            collate_fn= lambda batch: variable_time_collate_fn(batch, device))
    if train_percentage > 0:
        data_objects = {"dataset_obj": total_dataset,
                        "train_dataloader": inf_generator(train_dataloader),
                        "test_dataloader": inf_generator(test_dataloader),
                        "n_train_batches": len(train_dataloader),
                        "n_test_batches": len(test_dataloader)}
    else:
        data_objects = {"dataset_obj": total_dataset,
                        "test_dataloader": inf_generator(test_dataloader),
                        "n_test_batches": len(test_dataloader)}
    return data_objects

def parse_datasets_separate(device,batch_size,train_dataset,val_dataset,test_dataset):

    train_dataloader = DataLoader(train_dataset, batch_size= batch_size, shuffle=False,
        collate_fn = lambda batch: variable_time_collate_fn(batch, device))
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False,
        collate_fn= lambda batch: variable_time_collate_fn(batch, device))
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False,
        collate_fn= lambda batch: variable_time_collate_fn(batch, device))

    data_objects = {"train_dataloader": inf_generator(train_dataloader),
                    "val_dataloader": inf_generator(val_dataloader),
                    "test_dataloader": inf_generator(test_dataloader),
                    "n_train_batches": len(train_dataloader),
                    "n_val_batches": len(val_dataloader),
                    "n_test_batches": len(test_dataloader)}

    return data_objects

def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

def variable_time_collate_fn(batch, device=torch.device("cuda")):
    D = 4
    T = 50

    batch = np.array(batch)

    data_dict = {
        "time_step": torch.Tensor(batch[:,:,0] / 10).to(device),
        "event": torch.LongTensor(batch[:,:,1] - 1).to(device),
        "mkt_state": torch.Tensor(batch[:,:,-1]).to(device)}

    return data_dict

def get_next_batch(dataloader):
    # Make the union of all time points and perform normalization across the whole dataset
    data_dict = dataloader.__next__()
    return data_dict

def predict_from_hidden(intens_at_samples, seq_dt, timestep, t_max, samples, batch_length, device='cuda'):

    intens_t_vals_sum = intens_at_samples.sum(dim=-1)
    integral_ = torch.cumsum(timestep * intens_t_vals_sum, dim=-1)
    # density for the time-until-next-event law
    density = intens_t_vals_sum * torch.exp(-integral_)
    taus = torch.tensor(
        np.linspace(0, 100 * t_max, 100 * samples, endpoint=False).astype(np.float32)).repeat(batch_length, 1).to(
        device)
    t_pit = taus * density  # integrand for the time estimator
    ratio = intens_at_samples / intens_t_vals_sum[:, :, None]
    prob_type = ratio * density[:, :, None]
    estimate_dt = (timestep * 0.5 * (t_pit[:, 1:] + t_pit[:, :-1])).sum(dim=-1)
    next_dt = seq_dt[:, -1]
    error_dt = torch.abs(torch.log10(estimate_dt) - torch.log10(next_dt)).mean()
    estimate_type_prob = (timestep * 0.5 * (prob_type[:, 1:, :] + prob_type[:, :-1, :])).sum(dim=1)
    next_pred_event_unknown_time = np.argmax(estimate_type_prob.cpu().numpy(), axis=-1)
    return next_pred_event_unknown_time, error_dt

def update_learning_rate(optimizer, decay_rate=0.999, lowest=1e-3):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lr = max(lr * decay_rate, lowest)
        param_group['lr'] = lr

def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode='w')
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)

    for f in package_files:
        logger.info(f)
        with open(f, 'r') as package_f:
            logger.info(package_f.read())

    return logger
