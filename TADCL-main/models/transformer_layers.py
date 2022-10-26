import torch
import torch.nn as nn
from .utils import get_activation_fn


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1, dim_feedforward=2048, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        # Custom method to return attn outputs. Otherwise same as nn.TransformerEncoderLayer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):

        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead=4, dropout=0.1, dim_feedforward=2048, activation="relu"):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,  memory_key_padding_mask=None):

        tgt2, sim_mat_1 = self.self_attn(tgt, tgt, tgt,
                                         attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2, sim_mat_2 = self.multihead_attn(tgt, memory, memory,
                                              attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class SeBlock(nn.Module):
    def __init__(self, out_channels=512, feature_size=[7, 7], labels=14):
        super(SeBlock, self).__init__()

        self.shortcut = nn.Sequential(
            nn.Linear(feature_size[0]*feature_size[1], out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels)
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.se = nn.Sequential(
            nn.Linear(labels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, labels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)

        x_shortcut = self.shortcut(x)

        x_tmp = self.avgpool(x_shortcut)
        x_tmp = x_tmp.permute(0, 2, 1)
        x_tmp = self.se(x_tmp)
        x_tmp = x_tmp.permute(0, 2, 1)

        y = x_shortcut * x_tmp

        return y
