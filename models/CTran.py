 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pdb import set_trace as stop
from .transformer_layers import TransformerEncoderLayer, TransformerDecoderLayer, SeBlock
from .backbone import Backbone
from .utils import custom_replace, weights_init, intraClass_Sim
from .position_enc import PositionEmbeddingSine, positionalencoding2d

 
class CTranModel(nn.Module):
    def __init__(self, num_labels, pos_emb=False, layers=4, heads=8, dropout=0.1):
        super(CTranModel, self).__init__()

        # ResNet backbone
        self.backbone = Backbone()
        hidden = 2048 # this should match the backbone output feature size 2048

        self.seblock = SeBlock(hidden, [7, 7], num_labels)

        self.conv_downsample1 = torch.nn.Conv2d(hidden, hidden, (1, 1))
        self.conv_downsample2 = torch.nn.Conv2d(hidden, num_labels, (1, 1))
        
        # Label Embeddings
        self.label_input = torch.Tensor(np.arange(num_labels)).view(1, -1).long()
        self.label_lt = torch.nn.Embedding(num_labels, hidden, padding_idx=None)

        # Position Embeddings (for image features)
        self.use_pos_enc = pos_emb
        if self.use_pos_enc:
            # self.position_encoding = PositionEmbeddingSine(int(hidden/2), normalize=True)
            self.position_encoding = positionalencoding2d(hidden, 18, 18).unsqueeze(0)

        # Transformer
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(hidden, heads, dropout) for _ in range(layers)])
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(hidden, heads, dropout) for _ in range(layers)])

        # Classifier
        # Output is of size num_labels because we want a separate classifier for each label
        self.output_linear1 = torch.nn.Linear(hidden, num_labels)
        self.output_linear2 = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )

        # Other
        self.LayerNorm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

        # Init all except pretrained backbone
        self.label_lt.apply(weights_init)
        self.LayerNorm.apply(weights_init)
        self.conv_downsample1.apply(weights_init)
        self.conv_downsample2.apply(weights_init)
        self.seblock.apply(weights_init)
        self.encoder_layers.apply(weights_init)
        self.decoder_layers.apply(weights_init)
        self.output_linear1.apply(weights_init)
        self.output_linear2.apply(weights_init)

    def forward(self, images):

        const_label_input = self.label_input.repeat(images.size(0), 1).cuda()
        init_label_embeddings = self.label_lt(const_label_input)
        # print('init_label_embeddings:', init_label_embeddings.size())

        features = self.backbone(images)
        # print('backbone image feature shape:', features.size())

        features1 = self.conv_downsample1(features)

        features2 = self.conv_downsample2(features)
        features3 = self.seblock(features2)
        # print('seblock output shape:', features3.size())

        if self.use_pos_enc:
            pos_encoding = self.position_encoding(features1, torch.zeros(features.size(0), 18, 18, dtype=torch.bool).cuda())
            features1 = features1 + pos_encoding

        features1 = features1.view(features1.size(0), features1.size(1), -1).permute(0, 2, 1)
        # print('resized feature shape:', features.size())

        init_label_embeddings = init_label_embeddings + features3

        embeddings = torch.cat((features1, init_label_embeddings), 1)

        # print('transformer input shape:', embeddings.size())

        # Feed image and label embeddings through Transformer
        embeddings = self.LayerNorm(embeddings)        
        # attns = []
        for layer in self.encoder_layers:
            embeddings = layer(embeddings)
            # attns += attn.detach().unsqueeze(0).data

        # print('transformer output shape:', embeddings.size())

        image_embeddings = embeddings[:, 0:features1.size(1), :]
        label_embeddings = embeddings[:, -init_label_embeddings.size(1):, :]
        # print('')
        # print('encoder image embeddings shape:', image_embeddings.size())
        # print('encoder label embeddings shape:', label_embeddings.size())

        label_embeddings = label_embeddings + features3

        for layer in self.decoder_layers:
            label_embeddings = layer(label_embeddings, image_embeddings)
            # attns += attn.detach().unsqueeze(0).data
        # print('decoder label embeddings shape:', label_embeddings.size())

        output1 = self.output_linear1(label_embeddings)
        # print('output shape:', output.size())
        diag_mask = torch.eye(output1.size(1)).unsqueeze(0).repeat(output1.size(0), 1, 1).cuda()
        # print('diag_mask shape:', diag_mask.size())
        output1 = (output1*diag_mask).sum(-1)
        # print('output1 shape:', output1.size())

        output2 = self.output_linear2(features2)
        # print('output2 shape:', output2.size())
        output2 = torch.squeeze(output2)
        # print('output2 shape:', output2.size())

        return output1, output2, label_embeddings

