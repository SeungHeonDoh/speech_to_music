import torch
import torch.nn as nn

from speech_to_music.constants import (IEMOCAP_TAGS)
from transformers import Wav2Vec2Model, DistilBertForSequenceClassification
from speech_to_music.modules.net_modules import Transformer, PositionalEncoding

class FusionModel(nn.Module):
    def __init__(self, data_type, freeze_type, fusion_type):
        super().__init__()
        self.text_encoder = TextModel(data_type, freeze_type)  
        self.audio_encoder = AudioModel(data_type, freeze_type)  
        self.text_dim = self.text_encoder.feature_dim
        self.audio_dim = self.audio_encoder.feature_dim
        self.feature_dim = self.text_dim + self.audio_dim
        self.late_feature_dim = int(self.feature_dim/2)
        self.fusion_type = fusion_type
        self.text_projector = self._build_mlp(num_layers=2, input_dim=self.text_dim, mlp_dim=self.text_dim, output_dim=self.late_feature_dim)
        self.audio_projector = self._build_mlp(num_layers=2, input_dim=self.audio_dim, mlp_dim=self.audio_dim, output_dim=self.late_feature_dim)
        self.classifier = nn.Linear(self.feature_dim, len(IEMOCAP_TAGS))
        self.dropout = nn.Dropout(0.5)

    def forward(self, audio, audio_mask, token, mask, random_idx):
        device = audio.device
        text_emb = self.text_encoder.pooling_extractor(token, mask)
        audio_emb = self.audio_encoder.pooling_extractor(audio, audio_mask)
        if self.fusion_type == "early_fusion":
            feature = torch.cat((text_emb, audio_emb),1)
        elif self.fusion_type == "late_fusion":
            text_feature = self.text_projector(text_emb)
            audio_feature = self.audio_projector(audio_emb)
            feature = torch.cat((text_feature, audio_feature),1)
        elif self.fusion_type == "disentangle":
            text_feature = self.text_projector(text_emb)
            audio_feature = self.audio_projector(audio_emb)
            text_masking = torch.zeros(text_emb.shape).to(device)
            audio_masking = torch.zeros(audio_feature.shape).to(device)
            if random_idx == 0:
                feature = torch.cat((text_feature, audio_masking),1)
            elif random_idx == 1:
                feature = torch.cat((text_masking, audio_feature),1)
            elif random_idx == 2:
                feature = torch.cat((text_feature, audio_feature),1)
        else:
            feature = None
        dropout_feature = self.dropout(feature)
        logit = self.classifier(dropout_feature)
        return logit
    
    def extractor(self, audio, audio_mask, token, mask, random_idx):
        device = audio.device
        text_emb = self.text_encoder.pooling_extractor(token, mask)
        audio_emb = self.audio_encoder.pooling_extractor(audio, audio_mask)
        if self.fusion_type == "early_fusion":
            feature = torch.cat((text_emb, audio_emb),1)
        elif self.fusion_type == "late_fusion":
            text_feature = self.text_projector(text_emb)
            audio_feature = self.audio_projector(audio_emb)
            feature = torch.cat((text_feature, audio_feature),1)
        elif self.fusion_type == "disentangle":
            text_feature = self.text_projector(text_emb)
            audio_feature = self.audio_projector(audio_emb)
            text_masking = torch.zeros(text_emb.shape).to(device)
            audio_masking = torch.zeros(audio_feature.shape).to(device)
            if random_idx == 0:
                feature = torch.cat((text_feature, audio_masking),1)
            elif random_idx == 1:
                feature = torch.cat((text_masking, audio_feature),1)
            elif random_idx == 2:
                feature = torch.cat((text_feature, audio_feature),1)
        else:
            feature = None
        return feature

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim
            mlp.append(nn.Linear(dim1, dim2, bias=False))
            if l < num_layers - 1:
                mlp.append(nn.LayerNorm(dim2))
                mlp.append(nn.ReLU(inplace=True))
        return nn.Sequential(*mlp)

class TextModel(nn.Module):
    def __init__(self, data_type, freeze_type):
        super().__init__()
        self.data_type = data_type
        self.bert = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', problem_type="multi_label_classification", num_labels=len(IEMOCAP_TAGS))
        self.feature_dim = 768
        if freeze_type == "feature":
            for param in self.bert.distilbert.parameters():
                param.requires_grad = False

    def forward(self, token, mask, labels):
        output = self.bert(token, mask, labels=labels)
        prediction = output['logits']
        loss = output['loss']
        return prediction, loss

    def infer(self, token, mask):
        output = self.bert(token, mask)
        prediction = output['logits']
        return prediction

    def pooling_extractor(self, token, mask):
        distilbert_output = self.bert.distilbert(input_ids=token, attention_mask=mask)
        hidden_state = distilbert_output[0]  # (b, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (b, dim)
        return pooled_output

    def sequence_extractor(self, token, mask):
        distilbert_output = self.bert.distilbert(input_ids=token, attention_mask=mask)
        hidden_state = distilbert_output[0]  # (b, seq_len, dim)
        return hidden_state

class AudioModel(nn.Module):
    def __init__(self, data_type, freeze_type):
        super().__init__()
        self.data_type = data_type
        self.wav2vec2 = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
        self.feature_dim = 768
        self.projector = nn.Linear(self.feature_dim, self.feature_dim)
        self.classifier = nn.Linear(self.feature_dim, len(IEMOCAP_TAGS))
        self.dropout = nn.Dropout(0.5)
        if freeze_type == "feature":
            self.wav2vec2.feature_extractor._freeze_parameters()
        else:
            print("not freeze params")

    def forward(self, wav, mask=None):
        feature = self.wav2vec2(
            wav,
            attention_mask=mask,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None
            )
        last_hidden = feature['last_hidden_state']
        last_hidden = self.projector(last_hidden) # use last hidden layer
        projection_feature = last_hidden.mean(1,False) # average pooling
        dropout_feature = self.dropout(projection_feature)
        logit = self.classifier(dropout_feature)
        return logit

    def pooling_extractor(self, wav, mask=None):
        feature = self.wav2vec2(
            wav,
            attention_mask=mask,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None
            )
        last_hidden = feature['last_hidden_state']
        last_hidden = self.projector(last_hidden) # use last hidden layer
        projection_feature = last_hidden.mean(1,False) # average pooling
        return projection_feature
    
    def sequence_extractor(self, wav, mask=None):
        feature = self.wav2vec2(
            wav,
            attention_mask=mask,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None
            )
        last_hidden = feature['last_hidden_state']
        last_hidden = self.projector(last_hidden) # use last hidden layer
        return last_hidden