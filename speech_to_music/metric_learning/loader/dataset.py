import os
import torch
import random
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2Processor, DistilBertTokenizer
from speech_to_music.constants import (DATASET, AUDIOSET_TAGS, IEMOCAP_TAGS, SPEECH_SAMPLE_RATE, IEMOCAP_TO_AUDIOSET)

class Emotion_Dataset(Dataset):
    def __init__(self, split, fusion_type, word_model, is_augmentation):
        self.split = split
        self.fusion_type = fusion_type
        self.speech_tags = IEMOCAP_TAGS
        self.music_tags = AUDIOSET_TAGS
        self.stm_mapper = IEMOCAP_TO_AUDIOSET
        self.speech_meta = pd.read_csv(os.path.join(DATASET, f"split/IEMOCAP/annotation.csv"), index_col=0).set_index("wav_file_name")
        self.w2v = torch.load(os.path.join(DATASET, f"pretrained/word/{word_model}.pt"))
        if split == 'TRAIN':
            self.fl_speech = pd.read_csv(os.path.join(DATASET, f"split/IEMOCAP/train.csv"), index_col=0)
            self.fl_music = pd.read_csv(os.path.join(DATASET, f"split/Audioset/train.csv"), index_col=0)
        elif split == 'VALID':
            self.fl_speech = pd.read_csv(os.path.join(DATASET, f"split/IEMOCAP/valid.csv"), index_col=0)
            self.fl_music = pd.read_csv(os.path.join(DATASET, f"split/Audioset/test.csv"), index_col=0)
        elif split == 'TEST':
            self.fl_speech = pd.read_csv(os.path.join(DATASET, f"split/IEMOCAP/test.csv"), index_col=0)
            self.fl_music = pd.read_csv(os.path.join(DATASET, f"split/Audioset/test.csv"), index_col=0)

    def get_train_item(self, index):
        tts_tag_emb, tts_speech_emb, tts_binary = self.get_tag_to_speech()
        ttm_tag_emb, ttm_music_emb, ttm_binary = self.get_tag_to_music()
        stm_speech_emb, stm_music_emb, stm_binary = self.get_speech_to_music()
        return tts_tag_emb, tts_speech_emb, tts_binary, \
                ttm_tag_emb, ttm_music_emb, ttm_binary, \
                stm_speech_emb, stm_music_emb, stm_binary
    
    def get_speech_emb(self, fname):
        if self.fusion_type == "audio":
            speech_emb = torch.load(os.path.join(DATASET, f"feature/IEMOCAP/pretrained/audio_feature_False/{fname}.pt"))
        elif self.fusion_type == "text":
            speech_emb = torch.load(os.path.join(DATASET, f"feature/IEMOCAP/pretrained/text_feature_False/{fname}.pt"))
        elif (self.fusion_type == "early_fusion") or (self.fusion_type == "late_fusion") or (self.fusion_type == "disentangle"):
            audio_emb = torch.load(os.path.join(DATASET, f"feature/IEMOCAP/pretrained/audio_feature_False/{fname}.pt"))
            text_emb = torch.load(os.path.join(DATASET, f"feature/IEMOCAP/pretrained/text_feature_False/{fname}.pt"))
            speech_emb = (audio_emb, text_emb)
        else:
            speech_emb = None
        return speech_emb

    def get_tag_to_speech(self):
        i = random.randrange(len(self.speech_tags))
        tag = self.speech_tags[i]
        tag_emb = self.w2v[tag]
        item_list = self.fl_speech[tag]
        pos_pool = item_list[item_list != 0].index
        fname = random.choice(pos_pool)
        speech_emb = self.get_speech_emb(fname)
        speech_binary = self.fl_speech.loc[fname].to_numpy()
        return tag_emb.astype('float32'), speech_emb, speech_binary.astype('float32')

    def get_tag_to_music(self):
        i = random.randrange(len(self.music_tags))
        tag = self.music_tags[i]
        tag_emb = self.w2v[tag]
        item_list = self.fl_music[tag]
        pos_pool = item_list[item_list != 0].index
        fname = random.choice(pos_pool)
        music_wav = torch.load(os.path.join(DATASET, f"feature/Audioset/pretrained/{fname}.pt"))
        music_binary = self.fl_music.loc[fname].to_numpy()
        return tag_emb.astype('float32'), music_wav, music_binary.astype('float32')

    def get_speech_to_music(self):
        i = random.randrange(len(self.speech_tags))
        speech_tag = self.speech_tags[i]
        s_item_list = self.fl_speech[speech_tag]
        s_pos_pool = s_item_list[s_item_list != 0].index
        s_fname = random.choice(s_pos_pool)
        speech_emb = self.get_speech_emb(s_fname)
        speech_binary = self.fl_speech.loc[s_fname].to_numpy()

        # get music audio
        music_tags = self.stm_mapper[speech_tag]
        music_tag = random.choice(music_tags)
        m_item_list = self.fl_music[music_tag]
        m_pos_pool = m_item_list[m_item_list != 0].index
        m_fname = random.choice(m_pos_pool)
        music_wav = torch.load(os.path.join(DATASET, f"feature/Audioset/pretrained/{m_fname}.pt"))
        return speech_emb, music_wav, speech_binary.astype('float32')

    def get_eval_item(self, index):
        speech_tag = self.speech_tags[index % len(self.speech_tags)]
        speech_tag_emb = self.w2v[speech_tag]
        speech_tag_binary = self.fl_speech[speech_tag].to_numpy()

        music_tag = self.music_tags[index % len(self.music_tags)]
        music_tag_emb = self.w2v[music_tag]
        music_tag_binary = self.fl_music[music_tag].to_numpy()
        
        speech_idx = index % len(self.fl_speech)
        speech_item = self.fl_speech.iloc[speech_idx]
        speech_fnames = list(self.fl_speech.index)
        speech_fname = speech_item.name
        speech_emb = self.get_speech_emb(speech_fname)
        speech_binary = speech_item.to_numpy()

        music_idx = index % len(self.fl_music)
        music_item = self.fl_music.iloc[music_idx]
        music_fnames = list(self.fl_music.index)
        music_fname = music_item.name
        music_wav = torch.load(os.path.join(DATASET, f"feature/Audioset/pretrained/{music_fname}.pt"))
        music_binary = music_item.to_numpy()

        return music_tag_emb, music_tag_binary.astype('float32'), music_tag, \
                speech_tag_emb, speech_tag_binary.astype('float32'), speech_tag, \
                music_wav, music_binary.astype('float32'), music_fname, \
                speech_emb, speech_binary.astype('float32'), speech_fname, \
                speech_fnames, music_fnames

    def __getitem__(self, index):
        if (self.split=='TRAIN') or (self.split=='VALID'):
            return self.get_train_item(index)
        else:
            return self.get_eval_item(index)

    def __len__(self):
        return max(len(self.fl_speech), len(self.fl_music))