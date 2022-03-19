import os
import torch
import librosa
import pandas as pd
import numpy as np
from pytorch_lightning import Trainer, seed_everything
from argparse import ArgumentParser, Namespace, ArgumentTypeError

from speech_to_music.classification.music_cls.model.emb_model import MusicModel
from speech_to_music.classification.music_cls.model.lightning_model import ClsRunner
from speech_to_music.constants import (CLSREG_DATASET)

def embedding_extractor(runner, DEVICE):
    fl = pd.read_csv(os.path.join(CLSREG_DATASET, "split", "Audioset", "annotation.csv"), index_col=0)
    for idx in range(len(fl)):
        item = fl.iloc[idx]
        fname = item.name
        audio = np.load(os.path.join(CLSREG_DATASET, "feature", "Audioset", 'npy', fname + ".npy"))
        audio = torch.from_numpy(audio)
        with torch.no_grad():
            audio = runner.model.extractor(audio.unsqueeze(0).to(DEVICE))
        audio = audio.squeeze(0).detach().cpu()
        torch.save(audio, os.path.join(CLSREG_DATASET, "feature", "Audioset", 'pretrained', fname + ".pt"))

def inference(audio_dir, runner, DEVICE):
    audio, sr = librosa.load(audio_dir)
    audio = torch.from_numpy(audio)
    with torch.no_grad():
        logit = runner.model(audio.unsqueeze(0).to(DEVICE))
    logit = logit.squeeze(0).detach().cpu()
    return logit

def main(args) -> None:
    if args.reproduce:
        seed_everything(42)
    DEVICE = f"cuda:{args.gpus[0]}"
    save_path = f"exp/{args.data_type}/{args.cls_type}_{args.freeze_type}"
    model = MusicModel(
            pretrained_path=os.path.join(CLSREG_DATASET, "pretrained/music/compact_student.ckpt")
        )
    runner = ClsRunner(
            model = model,
            lr = args.lr, 
            max_epochs = args.max_epochs,
            batch_size = args.batch_size
    )
    state_dict = torch.load(os.path.join(save_path, "best.ckpt"))
    runner.load_state_dict(state_dict.get("state_dict"))
    runner = runner.to(DEVICE)
    embedding_extractor(runner, DEVICE)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tid", default="debug", type=str)    
    parser.add_argument("--cls_type", default="music", type=str)
    parser.add_argument("--data_type", default="Audioset", type=str)
    parser.add_argument("--freeze_type", default="none", type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=16, type=int)
    # runner 
    parser.add_argument("--lr", default=1e-3, type=float)
    # trainer
    parser.add_argument("--max_epochs", default=200, type=int)
    parser.add_argument("--gpus", default="3", type=str)
    parser.add_argument("--accelerator", default="ddp", type=str)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument("--reproduce", default=True, type=bool)

    args = parser.parse_args()
    main(args)