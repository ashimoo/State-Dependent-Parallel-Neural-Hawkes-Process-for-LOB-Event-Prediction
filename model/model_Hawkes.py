import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.nn.functional import softplus
import random
from utils import *


class Hawkes(nn.Module):
    def __init__(self, num_events, hidden_dim, input_embed_dim, dropout=0):
        super(Hawkes, self).__init__()
        self.num_events = num_events
        self.hidden_dim = hidden_dim
        self.mu_layer = nn.Sequential(nn.Linear(4,16),nn.ReLU(),nn.Linear(16,1),nn.Softplus())
        self.alpha_layer = nn.Sequential(nn.Linear(11,16),nn.ReLU(),nn.Linear(16,1),nn.Softplus())
        self.delta_layer = nn.Sequential(nn.Linear(11,16),nn.ReLU(),nn.Linear(16,1),nn.Softplus())


    def forward(self, batch, mkt_state, device = torch.device('cuda')):

        seq_dt = batch['time_step'][:,1:]
        seq_event = batch['event'].long()
        seq_state = batch['mkt_state'].long()
        batch_size = seq_state.shape[0]
        seq_length = seq_event.shape[1]

        timestep = torch.cat((torch.zeros(batch_size,1).cuda(),torch.cumsum(seq_dt,dim=-1)),dim=-1)
        event_matrix = nn.functional.one_hot(seq_event,num_classes=4).type(torch.float)
        state_matrix = nn.functional.one_hot(seq_state,num_classes=3).type(torch.float)
        relation_matrix = torch.zeros((batch_size,seq_length-1,4,11)).cuda()
        dt_matrix = torch.zeros((batch_size,seq_length-1,seq_length)).cuda()
        for i in range(seq_length-1):
            for j in range(4):
                relation_matrix[:,i,j] = torch.cat((event_matrix[:,i],torch.cat((state_matrix[:,i],nn.functional.one_hot(torch.ones(batch_size).long()*j,num_classes=4).cuda()),dim=-1)),dim=-1)

        dt_matrix = timestep.unsqueeze(1).repeat(1,seq_length-1,1) - timestep[:,:-1].unsqueeze(-1).repeat(1,1,seq_length)
        dt_matrix[dt_matrix<=0] = 1e9
        mu = self.mu_layer(nn.functional.one_hot(torch.tensor([0,1,2,3]).long()).cuda().repeat(batch_size,seq_length,1,1).type(torch.float)) #bs*t*4*1
        alpha = self.alpha_layer(relation_matrix).transpose(2,3) #bs*t*1*4
        delta = self.delta_layer(relation_matrix).transpose(2,3)
        # intens_matrix = mu[:,1:].squeeze(-1) + torch.triu(alpha * torch.exp(-delta*dt_matrix.unsqueeze(-1)),diagonal=1).sum(1)[:,1:]
        intens_matrix = mu[:, 1:].squeeze(-1) + (alpha * torch.exp(-delta * dt_matrix.unsqueeze(-1)[:,:,1:])*torch.triu(torch.ones(seq_length-1,seq_length-1).cuda()).unsqueeze(0).unsqueeze(-1).repeat(batch_size,1,1,1)).sum(1)


        return intens_matrix, mu, alpha, delta, None

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
        intens_matrix = intens_after
        alpha = cell_targets
        delta = outputs
        seq_dt = batch['time_step'][:,1:]
        seq_event = batch['event'].long()
        seq_state = batch['mkt_state'].long()
        batch_size = seq_event.shape[0]

        n_times = seq_event.shape[1]-1
        log_intensities = torch.log(intens_after + 1e-6)
        seq_mask = torch.nn.functional.one_hot(seq_event[:,1:],num_classes=4).type(torch.float)
        log_sum = (log_intensities * seq_mask).sum(dim=-1).mean(dim=-1)

        n_mc_samples = 50
        mu = cells.squeeze(-1)[:,:-1].repeat(1,n_mc_samples,1)

        taus = torch.Tensor(np.linspace(0, 1, n_mc_samples, endpoint=False)).repeat(batch_size, n_times, 1).cuda()
        taus = seq_dt[:, :, None].repeat(1, 1, n_mc_samples) * taus
        timestep = torch.cat((torch.zeros(batch_size,1).cuda(),torch.cumsum(seq_dt,dim=-1)),dim=-1).unsqueeze(-1)
        timestep_taus = (timestep[:,:-1] + taus).reshape(batch_size,-1).unsqueeze(1).repeat(1,n_times,1)
        time_matrix = timestep_taus - timestep[:,:-1]
        time_matrix[time_matrix<0] = 1e9
        mask = torch.triu(torch.ones(n_times,n_times).cuda()).unsqueeze(-1).repeat(batch_size,1,1,n_mc_samples).reshape(batch_size,n_times,-1)
        intens_matrix_taus = mu + ((alpha * torch.exp(-delta*time_matrix.unsqueeze(-1))) * mask.unsqueeze(-1)).sum(1)

        partial_integrals = seq_dt * intens_matrix_taus.reshape(batch_size,n_times,n_mc_samples,-1).sum(-1).mean(dim=-1)
        integral_ = partial_integrals.mean(1)

        hawkes_loss = (-log_sum + integral_).mean()
        class_loss = torch.nn.CrossEntropyLoss(weight=weights)(intens_after.reshape(intens_after.shape[0] * intens_after.shape[1] ,-1),(seq_event[:,1:].reshape(-1)))

        next_target_event = seq_event[:,-1,]
        next_pred_event = torch.argmax(intens_after[:,-1,:],dim=-1)
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
            n_times = seq_time.shape[1]-1
            seq_dt = seq_time[:,1:]
            next_target_event = seq_event[:, -1, ]
            intens, mu, alpha, delta, _ = self.forward(batch, mkt_state)
            next_pred_event_known_time = np.argmax(intens[:,-1,:].cpu().numpy(), axis=1)
            hawkes_loss,_,_ = self.compute_loss(batch, intens, mu, alpha, delta, None)

            if metrics:
                intens_at_samples = None
                for i in range(50):
                    samples = 1000
                    t_max = 1
                    timestep = t_max/samples
                    taus = torch.tensor(np.linspace(i*t_max,(i+1)*t_max,samples,endpoint=False).astype(np.float)).repeat(batch_length,1,1).to(device)
                    timesteps = torch.cat((torch.zeros(batch_length, 1).cuda(), torch.cumsum(seq_dt, dim=-1)),dim=-1).unsqueeze(-1)  # bs*(t+1)*1
                    timestep_taus = (timesteps[:, -2,None,:] + taus).repeat(1, n_times,1)  # bs*t*(mc)
                    time_matrix = timestep_taus - timesteps[:,:-1]  # distance between mc points and past event points. size #bs*t*(mc)
                    time_matrix[time_matrix < 0] = 1e9
                    mu = mu.squeeze(-1)[:,-2,None,:].repeat(1, samples, 1)  # bs*mc*4
                    intens_matrix_taus = mu + ((alpha * torch.exp(-delta * time_matrix.unsqueeze(-1)))).sum(1)  # bs*mc*4
                    if intens_at_samples is not None:
                        intens_at_samples = torch.cat((intens_at_samples,intens_matrix_taus),dim=1) #bs*N*4
                    else:
                        intens_at_samples = intens_matrix_taus

                next_pred_event_unknown_time, error_dt = predict_from_hidden(intens_at_samples, seq_dt, timestep, t_max, samples, batch_length, device='cuda')

            if metrics:
                return next_target_event.cpu().numpy(),next_pred_event_known_time,next_pred_event_unknown_time, hawkes_loss.cpu().item(), error_dt.item()
            else:
                return next_target_event.cpu().numpy(), next_pred_event_known_time, hawkes_loss.cpu().item()



