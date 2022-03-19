import os
import torch
import random
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2Processor, DistilBertTokenizer
from speech_to_music.constants import (CLSREG_DATASET, SPEECH_SAMPLE_RATE)
from torchaudio_augmentations import (
    RandomResizedCrop,
    RandomApply,
    PolarityInversion,
    Noise,
    Gain,
    HighLowPass,
    Delay,
    PitchShift,
    Reverb,
    Compose,
)
class Emotion_Dataset(Dataset):
    def __init__(self, split, data_type, is_augmentation):
        self.split = split
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
        self.maxseqlen = SPEECH_SAMPLE_RATE * 16
        self.is_augmentation = is_augmentation
        self.data_type = data_type
        self.df_meta = pd.read_csv(os.path.join(CLSREG_DATASET, f"split/{self.data_type}/annotation.csv"), index_col=0).set_index("wav_file_name")
        if split == 'TRAIN':
            self.fl = pd.read_csv(os.path.join(CLSREG_DATASET, f"split/{self.data_type}/train.csv"), index_col=0)
        elif split == 'VALID':
            self.fl = pd.read_csv(os.path.join(CLSREG_DATASET, f"split/{self.data_type}/valid.csv"), index_col=0)
        elif split == 'TEST':
            self.fl = pd.read_csv(os.path.join(CLSREG_DATASET, f"split/{self.data_type}/test.csv"), index_col=0)
        if self.is_augmentation:
            self._get_augmentations()

    def _get_augmentations(self):
        transforms = [
            RandomApply([Noise(min_snr=0, max_snr=0.1)], p=0.2),
            RandomApply([Gain()], p=0.5),
            RandomApply([HighLowPass(sample_rate=SPEECH_SAMPLE_RATE)], p=0.8),
            RandomApply([Reverb(sample_rate=SPEECH_SAMPLE_RATE)], p=0.5),
            RandomApply([PitchShift(n_samples=self.maxseqlen, sample_rate=SPEECH_SAMPLE_RATE)], p=0.5),
        ] 
        self.augmentation = Compose(transforms=transforms)
            
    def __getitem__(self, indices):
        audios, labels, texts, fnames, binarys = [], [], [], [], []
        for index in indices:
            item = self.fl.iloc[index]
            fname = item.name
            binary = item.values
            text = self.df_meta.loc[fname]['transcription']
            if self.is_augmentation:
                audio = np.load(os.path.join(CLSREG_DATASET, f"feature/{self.data_type}/npy/{fname}.npy"), mmap_mode='r')
                audio = self.augmentation(torch.from_numpy(np.array(audio))).squeeze(0).numpy()
            else:
                audio = np.load(os.path.join(CLSREG_DATASET, f"feature/{self.data_type}/npy/{fname}.npy"), mmap_mode='r')
                audio = np.array(audio).squeeze(0)
            fnames.append(fname)
            texts.append(text)
            audios.append(audio)
            binarys.append(binary)
            labels.append(torch.from_numpy(binary))
        # audios = self.processor_fn(audios) # one batch
        audio_encoding = self.processor(audios, padding="max_length", max_length=self.maxseqlen, truncation=True, return_tensors='pt',sampling_rate=SPEECH_SAMPLE_RATE, return_attention_mask=True)
        text_encoding = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        
        audios = audio_encoding['input_values']
        audio_mask = audio_encoding['attention_mask']
        token = text_encoding['input_ids']
        mask = text_encoding['attention_mask']
        labels = torch.stack(labels, dim=0).float()
        return {"audios" : audios, "audio_mask": audio_mask,"labels" : labels, "token" : token, "mask": mask, "fnames": fnames, 'binarys':binarys}

    def __len__(self):
        return len(self.fl)