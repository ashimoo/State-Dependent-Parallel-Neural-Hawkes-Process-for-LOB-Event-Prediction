import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.distributions.negative_binomial import NegativeBinomial
import random
from utils import *

class LSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int):
        super(LSTMCell, self).__init__()
        self.input_g = nn.Sequential(nn.Linear(input_dim + hidden_size, hidden_size),nn.Tanh(),nn.Linear(hidden_size,hidden_size))
        self.forget_g = nn.Sequential(nn.Linear(input_dim + hidden_size, hidden_size),nn.Tanh(),nn.Linear(hidden_size,hidden_size))
        self.output_g = nn.Sequential(nn.Linear(input_dim + hidden_size, hidden_size),nn.Tanh(),nn.Linear(hidden_size,hidden_size))
        # activation will be tanh
        self.z_gate = nn.Sequential(nn.Linear(input_dim + hidden_size, hidden_size),nn.Tanh(),nn.Linear(hidden_size,hidden_size))
        # Cell decay factor, identical for all hidden dims
        '''optional:
           the input_dim can be removed '''

    def forward(self, x, h_t, c_t):
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
        output = torch.sigmoid(self.output_g(v))
        z_i = torch.tanh(self.z_gate(v))
        c_i = forget * c_t + inpt * z_i
        return c_i, output

class LSTM_PP(nn.Module):
    def __init__(self, num_events, hidden_dim, input_embed_dim, dropout=0):
        super(LSTM_PP, self).__init__()
        self.num_events = num_events
        self.hidden_dim = hidden_dim
        self.lstm = LSTMCell(input_embed_dim, hidden_dim).cuda()

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

        self.decoding_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,self.num_events),
            nn.Softmax(-1))

    def forward(self, batch, mkt_state, device = torch.device('cuda')):
        (h_t, c_t) = (torch.zeros(len(batch['event']), self.hidden_dim).to(device),
                      torch.zeros(len(batch['event']), self.hidden_dim).to(device))
        seq_dt = batch['time_step'][:,1:]
        seq_event = batch['event']
        seq_state = batch['mkt_state']
        batch_size = seq_state.shape[0]

        seq_length = seq_event.shape[1]
        type_embedding = self.input_embedding_layer(torch.nn.functional.one_hot(seq_event,num_classes=4).type(torch.float))
        state_embedding = self.state_embedding_layer(torch.nn.functional.one_hot(seq_state.long(),num_classes=3).type(torch.float))
        if mkt_state:
            x = type_embedding + state_embedding
        else:
            x = type_embedding
        hidden_state = []
        for i in range(seq_length-1):
            c_t, output, = self.lstm(x[:,i], h_t, c_t)
            h_t = output * torch.tanh(c_t)
            hidden_state.append(h_t)
        hidden_state = torch.stack(hidden_state).permute(1,0,2)

        return None, hidden_state, None, None, None

    def compute_loss(self, batch, intens_after, cells, cell_targets, outputs, decays, class_loss_weight=0, weights=torch.FloatTensor([1.,1.,1.,1.]).cuda(), device=torch.device('cuda')):
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
        prob = self.decoding_layer(cells.reshape(cells.shape[0]*cells.shape[1],-1))
        class_loss = torch.nn.CrossEntropyLoss(weight=weights)(prob,(seq_event[:,1:].reshape(-1)))

        next_target_event = seq_event[:,-1,]
        next_pred_event = torch.argmax(self.decoding_layer(cells[:,-1,:]),dim=1)
        pred_accuracy = (torch.sum(next_target_event==next_pred_event)).item()/len(next_pred_event)

        return class_loss, 0, pred_accuracy

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
            _, cells, _, _, _ = self.forward(batch, mkt_state)
            prob = self.decoding_layer(cells[:,-1])
            next_pred_event_known_time = torch.argmax(prob, dim=-1)
            if metrics:
                return next_target_event.cpu().numpy(),next_pred_event_known_time.cpu().numpy(),next_pred_event_known_time.cpu().numpy(),0,0
            else:
                return next_target_event.cpu().numpy(), next_pred_event_known_time.cpu().numpy(), 0



