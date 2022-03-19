import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from speech_to_music.modules.net_modules import MusicTaggingTransformer
from speech_to_music.constants import (DATASET, AUDIOSET_TAGS)

class MusicReg(nn.Module):
    def __init__(self, pretrained_path):
        super().__init__()
        self.feature_extractor = self._load_pretrained(pretrained_path = pretrained_path)
        self.feature_extractor.train()
        self.feature_dim = 64
        self.n_class = len(AUDIOSET_TAGS)
        self.mlp_head = nn.Linear(self.feature_dim, 2)
        self.to_latent = nn.Identity()
        self.dropout = nn.Dropout(0.5)

    def forward(self, wav):
        feature = self.feature_extractor(wav)
        feature = self.to_latent(feature[:, 0])
        feature = self.dropout(feature)
        logit = self.mlp_head(feature)
        return logit

    def extractor(self, wav):
        feature = self.feature_extractor(wav)
        feature = self.to_latent(feature[:, 0])
        return feature

    def _load_pretrained(self, pretrained_path):
        pretrained = torch.load(pretrained_path)
        student_ckpt = {k[8:]: v for k, v in pretrained.items() if (k[:7] != 'teacher')}
        model = MusicTaggingTransformer(conv_ndim=128, attention_ndim=64, n_seq_cls=50)
        model.load_state_dict(student_ckpt)
        return nn.Sequential(*list(model.children())[:-3]) # to_latent
        
    # https://github.com/facebookresearch/moco-v3/blob/main/moco/builder.py
    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim
            mlp.append(nn.Linear(dim1, dim2, bias=False))
            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                mlp.append(nn.BatchNorm1d(dim2, affine=False))
        return nn.Sequential(*mlp)