import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, DistilBertModel
from speech_to_music.constants import (DATASET)

class EmbModel(nn.Module):
    def __init__(self, fusion_type):
        super().__init__()
        self.fusion_type = fusion_type
        self.music_feature_dim = 64
        self.speech_audio_feature_dim = 768
        self.speech_text_feature_dim = 768
        self.tag_feature_dim = 300
        self.projection_dim = 64
        self.speech_feature_dim = int(self.speech_audio_feature_dim + self.speech_text_feature_dim)
        self.disentangle_dim = int(self.projection_dim / 2) # half is linguistic, half is acoustic

        if self.fusion_type == "audio":
            self.speech_audio_mlp = self._build_mlp(num_layers=2, input_dim=self.speech_audio_feature_dim, mlp_dim=self.speech_audio_feature_dim, output_dim=self.projection_dim)
        elif self.fusion_type == "text":
            self.speech_text_mlp = self._build_mlp(num_layers=2, input_dim=self.speech_text_feature_dim, mlp_dim=self.speech_text_feature_dim, output_dim=self.projection_dim)
        elif (self.fusion_type == "early_fusion"):
            self.speech_fusion_mlp = self._build_mlp(num_layers=2, input_dim=self.speech_feature_dim, mlp_dim=self.speech_feature_dim, output_dim=self.projection_dim)
        elif (self.fusion_type == "late_fusion") or (self.fusion_type == "disentangle"):
            self.audio_projector = self._build_mlp(num_layers=2, input_dim=self.speech_audio_feature_dim, mlp_dim=self.speech_audio_feature_dim, output_dim=self.disentangle_dim)
            self.text_projector = self._build_mlp(num_layers=2, input_dim=self.speech_audio_feature_dim, mlp_dim=self.speech_audio_feature_dim, output_dim=self.disentangle_dim)
        self.music_mlp = self._build_mlp(num_layers=2, input_dim=self.music_feature_dim, mlp_dim=self.music_feature_dim, output_dim=self.projection_dim)
        self.tag_mlp = self._build_mlp(num_layers=2, input_dim=self.tag_feature_dim, mlp_dim=self.tag_feature_dim, output_dim=self.projection_dim)

        self.to_latent = nn.Identity()
        self.dropout = nn.Dropout(0.1)
    
    def speech_to_emb(self, speech_emb, random_idx):
        if self.fusion_type == "audio":
            speech_emb = self.dropout(speech_emb)
            speech_emb = self.speech_audio_mlp(speech_emb)
        elif self.fusion_type == "text":
            speech_emb = self.dropout(speech_emb)
            speech_emb = self.speech_text_mlp(speech_emb)
        elif self.fusion_type == "early_fusion":
            audio_emb, text_emb = speech_emb
            feature = torch.cat((audio_emb, text_emb),1)
            feature = self.dropout(feature)
            speech_emb = self.speech_fusion_mlp(feature)
        elif self.fusion_type == "late_fusion":
            audio_emb, text_emb = speech_emb
            audio_emb = self.dropout(audio_emb)
            text_emb = self.dropout(text_emb)
            audio_emb = self.audio_projector(audio_emb)
            text_emb = self.text_projector(text_emb)
            speech_emb = torch.cat((audio_emb, text_emb),1)
        elif self.fusion_type == "disentangle":
            audio_emb, text_emb = speech_emb  # batch x dim,  batch x dim  max seq 1307, max seq 127, mean seq 12x, 15
            device = audio_emb.device
            audio_emb = self.dropout(audio_emb)
            text_emb = self.dropout(text_emb)
            audio_emb = self.audio_projector(audio_emb)
            text_emb = self.text_projector(text_emb)
            audio_masking = torch.zeros(audio_emb.shape).to(device)
            text_masking = torch.zeros(text_emb.shape).to(device)
            if random_idx == 0:
                speech_emb = torch.cat((audio_emb, text_masking),1)
            elif random_idx == 1:
                speech_emb = torch.cat((audio_masking, text_emb),1)
            elif random_idx == 2:
                speech_emb = torch.cat((audio_emb, text_emb),1)
        else:
            speech_emb = None
        return speech_emb
        
    def music_to_emb(self, music):
        music = self.dropout(music)
        return self.music_mlp(music)

    def tag_to_emb(self, tag):
        tag = self.dropout(tag)
        return self.tag_mlp(tag)

    def forward(self, speech_emb, music, tag, random_idx):
        speech_emb = self.speech_to_emb(speech_emb, random_idx)
        music_emb = self.music_to_emb(music)
        tag_emb = self.tag_to_emb(tag)
        return speech_emb, music_emb, tag_emb

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