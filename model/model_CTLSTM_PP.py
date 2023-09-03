import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, accuracy_score
import random
from utils import *


class HawkesLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int):
        super(HawkesLSTMCell, self).__init__()
        self.input_g = nn.Sequential(nn.Linear(input_dim + hidden_size, hidden_size),nn.Tanh(),nn.Linear(hidden_size,hidden_size))
        self.forget_g = nn.Sequential(nn.Linear(input_dim + hidden_size, hidden_size),nn.Tanh(),nn.Linear(hidden_size,hidden_size))
        self.output_g = nn.Sequential(nn.Linear(input_dim + hidden_size, hidden_size),nn.Tanh(),nn.Linear(hidden_size,hidden_size))
        self.input_target = nn.Sequential(nn.Linear(input_dim + hidden_size, hidden_size),nn.Tanh(),nn.Linear(hidden_size,hidden_size))
        self.forget_target = nn.Sequential(nn.Linear(input_dim + hidden_size, hidden_size),nn.Tanh(),nn.Linear(hidden_size,hidden_size))
        # activation will be tanh
        self.z_gate = nn.Sequential(nn.Linear(input_dim + hidden_size, hidden_size),nn.Tanh(),nn.Linear(hidden_size,hidden_size))
        # Cell decay factor, identical for all hidden dims
        '''optional:
           the input_dim can be removed '''
        self.decay_layer = nn.Sequential(nn.Linear(input_dim + hidden_size, hidden_size),nn.Tanh(),nn.Linear(hidden_size,hidden_size),nn.Softplus(1.))


    def forward(self, x, h_t, c_t, c_target):
        """
        Compute the updated LSTM paramters.

        Args:s
            x: event type embedding
            h_t:
            c_t:
            c_target:

        Returns:
            h_i: just-updated hidden state
            h_t: hidden state just before next event
            cell_i: just-updated cell state
            c_t: cell state decayed to before next event
            c_target_i: cell state target before the next event
            output: LSTM output
            decay_i: rate of decay for the cell state
        """
        v = torch.cat((x, h_t), dim=1)
        inpt = torch.sigmoid(self.input_g(v))
        forget = torch.sigmoid(self.forget_g(v))
        input_target = torch.sigmoid(self.input_target(v))
        forget_target = torch.sigmoid(self.forget_target(v))
        output = torch.sigmoid(self.output_g(v))  # compute the LSTM network output
        # Not-quite-c
        z_i = torch.tanh(self.z_gate(v))
        # Compute the decay parameter
        decay = self.decay_layer(v)
        # Update the cell state to c(t_i+)
        c_i = forget * c_t + inpt * z_i
        # h_i = output * torch.tanh(c_i)  # hidden state just after event
        # Update the cell state target
        c_target = forget_target * c_target + input_target * z_i
        return c_i, c_target, output, decay

class CTLSTM_PP(nn.Module):
    def __init__(self, num_events, hidden_dim, input_embed_dim, dropout=0):
        super(CTLSTM_PP, self).__init__()
        self.num_events = num_events
        self.hidden_dim = hidden_dim
        self.lstm = HawkesLSTMCell(input_embed_dim, hidden_dim).cuda()
        self.intensity_layer = nn.Sequential(nn.Linear(hidden_dim,4),nn.Softplus(beta=1.)).cuda()
        self.input_embedding_layer = nn.Sequential(
            nn.Linear(num_events,input_embed_dim),
            nn.Tanh(),
            nn.Linear(input_embed_dim,input_embed_dim),
            nn.Tanh())

        self.state_embedding_layer = nn.Sequential(
            nn.Linear(3,input_embed_dim),
            nn.Tanh(),
            nn.Linear(input_embed_dim,input_embed_dim),
            nn.Tanh())


    def forward(self, batch, mkt_state, device = torch.device('cuda')):
        (h_t, c_t, c_target_t) = (torch.zeros(len(batch['event']), self.hidden_dim).to(device),
                                  torch.zeros(len(batch['event']), self.hidden_dim).to(device),
                                  torch.zeros(len(batch['event']), self.hidden_dim).to(device))
        seq_dt = batch['time_step'][:,1:]
        seq_event = batch['event']
        seq_state = batch['mkt_state']
        batch_size = seq_state.shape[0]
        outputs = []  # output from each LSTM pass
        cells = []  # cell states at event times
        cell_targets = []  # target cell states for each interval
        decays = []
        intens_after_decay = []
        seq_length = seq_event.shape[1]
        type_embedding = self.input_embedding_layer(torch.nn.functional.one_hot(seq_event,num_classes=4).type(torch.float))
        state_embedding = self.state_embedding_layer(torch.nn.functional.one_hot(seq_state.long(),num_classes=3).type(torch.float))
        if mkt_state:
            x = type_embedding + state_embedding
        else:
            x = type_embedding

        for i in range(seq_length-1):
            dt = seq_dt[:,i]
            cell_i, c_target_t, output, decay_i = self.lstm(x[:,i], h_t, c_t, c_target_t)
            c_t = c_target_t + (cell_i - c_target_t) * torch.exp(-(decay_i) * dt[:, None])
            h_t = output * torch.tanh(c_t)
            inten_after_decay = self.intensity_layer(h_t)

            outputs.append(output)
            decays.append(decay_i)
            cells.append(cell_i)
            intens_after_decay.append(inten_after_decay)
            cell_targets.append(c_target_t)

        outputs = torch.stack(outputs).permute(1,0,2)
        cells = torch.stack(cells).permute(1,0,2)
        intens_after_decay = torch.stack(intens_after_decay).permute(1,0,2)
        cell_targets = torch.stack(cell_targets).permute(1,0,2)
        decays = torch.stack(decays).permute(1,0,2)

        return intens_after_decay, cells, cell_targets, outputs, decays

    def compute_loss(self, batch, intens_after, cells, cell_targets, outputs, decays, class_loss_weight=0, device=torch.device('cuda')):
        """
        Compute the negative log-likelihood as a loss function.

        Args:
            seq_times: event occurrence timestamps
            seq_onehot_types: types of events in the sequence, one hot encoded
            batch_sizes: batch sizes for each event sequence tensor, by length
            hiddens_ti: hidden states just before the events occur.
            cells: entire cell state history
            cell_targets: cell state target values history
            outputs: entire output history
            decays: entire decay history
            tmax: temporal horizon

        Returns:
            log-likelihood of the event times under the learned parameters

        Shape:
            one-element tensor
        """

        seq_dt = batch['time_step'][:,1:]
        seq_event = batch['event']
        batch_size = seq_event.shape[0]
        n_times = seq_event.shape[1]-1
        intens_at_evs = torch.log(intens_after + 1e-6)
        log_intensities = intens_at_evs
        log_sum = (log_intensities * torch.nn.functional.one_hot(seq_event[:,1:],num_classes=4).type(torch.float)).sum(-1).mean(-1)

        n_mc_samples = 50
        taus = torch.Tensor(np.linspace(0, 1, n_mc_samples, endpoint=False)).repeat(batch_size, n_times, 1, 1).cuda()
        taus = seq_dt[:, :, None, None] * taus # bs*t*1*100

        c_t = cell_targets.unsqueeze(-1) + (cells.unsqueeze(-1) - cell_targets.unsqueeze(-1)) * torch.exp(-decays.unsqueeze(-1) * taus)
        h_t = outputs.unsqueeze(-1) * torch.tanh(c_t)
        intens_at_samples = self.intensity_layer(h_t.transpose(2, 3)) # bs*t*4*mc
        total_intens_samples = intens_at_samples.sum(dim=-1) #bs*t*mc
        partial_integrals = seq_dt * total_intens_samples.mean(dim=-1)
        integral_ = partial_integrals.mean(dim=1)

        hawkes_loss = (-log_sum + integral_).mean()
        class_loss = torch.nn.CrossEntropyLoss()(intens_after.reshape(intens_after.shape[0] * intens_after.shape[1] ,-1),(seq_event[:,1:].reshape(-1)))

        next_target_event = seq_event[:,-1,]
        next_pred_event = torch.argmax(intens_after[:,-1,:],dim=1)
        pred_accuracy = (torch.sum(next_target_event==next_pred_event)).item()/len(next_pred_event)

        if class_loss_weight:
            total_loss = hawkes_loss + class_loss_weight * class_loss
        else:
            total_loss = hawkes_loss
        return total_loss, hawkes_loss.cpu().item(), pred_accuracy

    def read_predict(self, batch, mkt_state, device = torch.device("cuda"),metrics=False):
        """
        Read an event sequence and predict the next event time and type.

        Args:
            sequence:
            seq_types:
            seq_lengths:
            hmax:
            plot:
            print_info:

        Returns:

        """
        with torch.no_grad():
            seq_event = batch['event']
            seq_time = batch['time_step']
            seq_states = batch['mkt_state']
            batch_length = seq_time.shape[0]
            seq_dt = seq_time[:,1:]
            next_target_event = seq_event[:, -1, ]
            intens, cells, cell_targets, outputs, decays = self.forward(batch, mkt_state)
            next_pred_event_known_time = np.argmax(intens[:,-1,:].cpu().numpy(), axis=1)
            hawkes_loss,_,_ = self.compute_loss(batch, intens, cells, cell_targets, outputs, decays)
            if metrics:
                intens_at_samples = None
                for i in range(100):
                    samples = 1000
                    t_max = 1
                    timestep = t_max/samples
                    taus = torch.tensor(np.linspace(i*t_max,(i+1)*t_max,samples,endpoint=False).astype(np.float32)).repeat(batch_length,1,1).to(device)
                    cell_tau_ = cell_targets[:, -1, :, None] + (cells[:, -1, :, None] - cell_targets[:, -1, :, None]) * torch.exp(-decays[:, -1, :, None] * taus)
                    cell_tau_ = cell_tau_.transpose(1, 2)
                    if intens_at_samples is not None:
                        intens_at_samples = torch.cat((intens_at_samples,self.intensity_layer(outputs[:, -1, None, :] * torch.tanh(cell_tau_))),dim=1) #bs*N*4
                    else:
                        intens_at_samples = self.intensity_layer(outputs[:, -1, None, :] * torch.tanh(cell_tau_))

                next_pred_event_unknown_time, error_dt = predict_from_hidden(intens_at_samples, seq_dt, timestep, t_max, samples, batch_length, device='cuda')
            if metrics:
                return next_target_event.cpu().numpy(),next_pred_event_known_time,next_pred_event_unknown_time, hawkes_loss.cpu().item(), error_dt.item()
            else:
                return next_target_event.cpu().numpy(), next_pred_event_known_time, hawkes_loss.cpu().item()








