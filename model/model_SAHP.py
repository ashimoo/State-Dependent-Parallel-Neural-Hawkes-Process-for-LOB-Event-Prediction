import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
import math
import random
from utils import *


class SAHP(nn.Module):
    def __init__(self, num_events, hidden_dim, input_embed_dim, dropout = 0):
        super(SAHP, self).__init__()
        self.num_events = num_events
        self.hidden_dim = hidden_dim
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

        self.intensity_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, 4, bias = True),
            nn.Softplus(beta=1.0)
        )

        self.attention = MultiHeadedAttention(h=4, d_model=self.hidden_dim)
        self.feed_forward = PositionwiseFeedForward(d_model=self.hidden_dim, d_ff=self.hidden_dim * 4, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=self.hidden_dim, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=self.hidden_dim, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.position_emb = BiasedPositionalEmbedding(d_model=self.hidden_dim,max_len = 50)
        self.gelu = GELU()
        self.nLayers = 3
        self.start_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
        )

        self.converge_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
        )

        self.decay_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
            nn.Softplus(beta=1.)
        )


    def forward(self, batch, mkt_state, device = torch.device('cuda')):

        seq_dt = batch['time_step'][:,1:]
        seq_event = batch['event']
        seq_states = batch['mkt_state']
        type_embedding = self.input_embedding_layer(torch.nn.functional.one_hot(seq_event,num_classes=4).type(torch.float))
        position_embedding = self.position_emb(seq_event,torch.cat((torch.ones(seq_dt.shape[0],1).cuda(),seq_dt),dim=-1))
        state_embedding = self.state_embedding_layer(torch.nn.functional.one_hot(seq_states.long(),num_classes=3).type(torch.float))
        mask = MaskBatch(seq_event, pad=4, device=device).src_mask
        if mkt_state:
            x = type_embedding + state_embedding + position_embedding
        else:
            x = type_embedding + position_embedding
        for i in range(self.nLayers):
            x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
            x = self.dropout(self.output_sublayer(x, self.feed_forward))

        embed_info = x

        self.start_point = self.start_layer(embed_info)
        self.converge_point = self.converge_layer(embed_info)
        self.omega = self.decay_layer(embed_info)
        cell = self.converge_point[:,:-1,:] + (self.start_point[:,:-1,:] - self.converge_point[:,:-1,:] ) * torch.exp(
            -(self.omega[:,:-1,:] ) * seq_dt[:,:,None])
        intens = self.intensity_layer(torch.tanh(cell))
        return intens, cell, None, None, None

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

        Returns:
            log-likelihood of the event times under the learned parameters

        Shape:
            one-element tensor
        """

        seq_dt = batch['time_step'][:,1:]
        seq_event = batch['event']
        n_times = seq_dt.shape[1]
        batch_size = seq_dt.shape[0]

        intens_at_evs = torch.log(intens_after + 1e-6)
        log_intensities = intens_at_evs.squeeze(-1) # log intensities
        log_sum = (log_intensities * torch.nn.functional.one_hot(seq_event[:,1:],num_classes=4).type(torch.float)).sum(-1).mean(-1) # input bs*t*4 output bs
        # log_sum = torch.log(intens_after.sum(-1) + 1e-6).mean(-1)
        n_mc_samples_ = 50
        taus = torch.Tensor(np.linspace(0, 1, n_mc_samples_, endpoint=False)[1:]).repeat(batch_size, n_times, 1, 1).cuda()
        taus = seq_dt[:, :, None, None] * taus  # inter-event times samples)

        cell_tau = self.converge_point[:,:-1,:,None] + (self.start_point[:,:-1,:,None] - self.converge_point[:,:-1,:,None])*torch.exp(-self.omega[:,:-1,:,None] * taus)
        intens_at_samples = self.intensity_layer(torch.tanh(cell_tau.transpose(2, 3))).transpose(2,3)

        total_intens_samples = intens_at_samples.sum(dim=2)  # shape batch * t * MC
        partial_integrals = seq_dt * total_intens_samples.mean(dim=-1) # batch*t
        integral = partial_integrals.mean(dim=1) # mean on t

        hawkes_loss = (- log_sum + integral).mean()
        class_loss = torch.nn.CrossEntropyLoss()(intens_after.reshape(intens_after.shape[0] * intens_after.shape[1] ,-1),(seq_event[:,1:].reshape(-1)))

        next_target_event = seq_event[:,-1,]
        next_pred_event = torch.argmax(intens_after[:,-1,:],dim=1)
        pred_accuracy = (torch.sum(next_target_event==next_pred_event)).item()/batch_size
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
            seq_event:
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
            n_times =len(seq_dt[0])
            next_target_event = seq_event[:, -1, ]
            intens, _, _, _, _ = self.forward(batch, mkt_state)
            hawkes_loss, _, _ = self.compute_loss(batch, intens, None, None, None, None)
            next_pred_event_known_time = np.argmax(intens[:,-1,:].cpu().numpy(), axis=1)

            if metrics:
                intens_at_samples = None
                for i in range(100):
                    samples = 1000
                    t_max = 1
                    timestep = t_max/samples
                    taus = torch.tensor(np.linspace(i*t_max,(i+1)*t_max,samples,endpoint=False).astype(np.float32)).repeat(batch_length,1,1).to(device)
                    cell = self.converge_point[:,-2,:].unsqueeze(-1) + (self.start_point[:,-2,:].unsqueeze(-1) - self.converge_point[:,-2,:].unsqueeze(-1)) * torch.exp(
                        -(self.omega[:,-2,:].unsqueeze(-1))*taus) #bs*hs*N
                    if intens_at_samples is not None:
                        intens_at_samples = torch.cat((intens_at_samples,self.intensity_layer(torch.tanh(cell).permute(0,2,1))),dim=1) #bs*N*4
                    else:
                        intens_at_samples = self.intensity_layer(torch.tanh(cell).permute(0,2,1))
                next_pred_event_unknown_time, error_dt = predict_from_hidden(intens_at_samples, seq_dt, timestep, t_max, samples, batch_length, device='cuda')

            if metrics:
                return next_target_event.cpu().numpy(),next_pred_event_known_time,next_pred_event_unknown_time, hawkes_loss.cpu().item(), error_dt.item()
            else:
                return next_target_event.cpu().numpy(), next_pred_event_known_time, hawkes_loss.cpu().item()


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def forward(self, query, key, value, mask=None, dropout=None):

        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        # scores = torch.exp(torch.matmul(query, key.transpose(-2, -1))) \
        #          / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    """
    Take in models size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model, bias=True) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model, bias=True)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # the same mask applies to all heads
            # unsqueeze Returns a new tensor with a dimension of size one
            # inserted at the specified position.
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention.forward(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation.forward(self.w_1(x))))

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class BiasedPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        self.register_buffer('position', position)
        self.register_buffer('div_term', div_term)

        self.Wt = nn.Linear(1, d_model // 2, bias=False)

    def forward(self, x, interval):
        phi = self.Wt(interval.unsqueeze(-1))
        aa = len(x.size())
        if aa > 1:
            length = x.size(1)
        else:
            length = x.size(0)

        arc = (self.position[:length] * self.div_term).unsqueeze(0)

        pe_sin = torch.sin(arc + phi)
        pe_cos = torch.cos(arc + phi)
        pe = torch.cat([pe_sin, pe_cos], dim=-1)

        return pe


