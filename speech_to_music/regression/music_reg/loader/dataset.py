import os
import torch
import random
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from speech_to_music.constants import (CLSREG_DATASET, VA_MAP)

class Emotion_Dataset(Dataset):
    def __init__(self, split, data_type):
        self.split = split
        self.data_type = data_type
        self.va_map = VA_MAP
        if split == 'TRAIN':
            fl_train = pd.read_csv(os.path.join(CLSREG_DATASET, "split", self.data_type, "train.csv"), index_col=0)
            fl_valid = pd.read_csv(os.path.join(CLSREG_DATASET, "split", self.data_type, "valid.csv"), index_col=0)
            self.fl = pd.concat([fl_train,fl_valid])
        elif split == 'VALID':
            self.fl = pd.read_csv(os.path.join(CLSREG_DATASET, "split", self.data_type, "test.csv"), index_col=0)
        elif split == 'TEST':
            self.fl = pd.read_csv(os.path.join(CLSREG_DATASET, "split", self.data_type, "test.csv"), index_col=0)
            
    def __getitem__(self, indices):
        audios = []
        labels = []
        fnames = []
        binarys = []
        for index in indices:
            item = self.fl.iloc[index]
            fname = item.name
            binary = item.to_numpy()
            av = np.array(self.va_map[item.idxmax()])
            audio = np.load(os.path.join(CLSREG_DATASET, "feature", self.data_type, 'npy', fname + ".npy"))
            audios.append(torch.from_numpy(audio))
            labels.append(torch.from_numpy(av))
            fnames.append(fname)
            binarys.append(binary)
        audios = torch.stack(audios, dim=0).float()
        labels = torch.stack(labels, dim=0).float()
        return {"audios" : audios,"labels" : labels, "fnames":fnames, "binarys":binarys}

    def __len__(self):
        return len(self.fl)