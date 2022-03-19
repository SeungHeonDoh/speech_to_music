# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import numpy as np
import torch
import torchaudio
import augment
import argparse
from tqdm import tqdm
from dataclasses import dataclass
from speech_to_music.constants import (
    DATASET, SPEECH_SAMPLE_RATE
    )
import noisereduce as nr


if __name__ == '__main__':
    os.makedirs(os.path.join(DATASET, 'feature', "IEMOCAP", 'denoise'), exist_ok=True)
    dir_path = os.path.join(DATASET, 'feature', "IEMOCAP", 'npy')
    for fname in tqdm(os.listdir(dir_path)):
        sample = np.load(os.path.join(dir_path, fname))
        reduced_noise = nr.reduce_noise(y=sample, sr=SPEECH_SAMPLE_RATE)
        sample = reduced_noise.astype(np.float32)
        save_name = os.path.join(DATASET, 'feature', "IEMOCAP", 'denoise', fname)
        np.save(save_name, sample)