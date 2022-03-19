import os
import torch
import random
import pandas as pd
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from argparse import ArgumentParser, Namespace, ArgumentTypeError
from transformers import Wav2Vec2Processor, DistilBertTokenizer

from speech_to_music.metric_learning.model.emb_model import EmbModel
from speech_to_music.metric_learning.model.lightning_model import TripletRunner

from speech_to_music.classification.speech_cls.model.emb_model import AudioModel, TextModel
from speech_to_music.classification.speech_cls.model.lightning_model import SpeechCls

from speech_to_music.classification.music_cls.model.emb_model import MusicModel
from speech_to_music.classification.music_cls.model.lightning_model import ClsRunner

from speech_to_music.preprocessing.audio_utils import load_audio
from speech_to_music.constants import (
    DATASET, SPEECH_SAMPLE_RATE, MUSIC_SAMPLE_RATE, STR_CH_FIRST, INPUT_LENGTH)



def load_audio_backbone(args):
    save_path = f"../classification/speech_cls/exp/IEMOCAP/{args.cv_split}/audio_feature"
    config = OmegaConf.load(os.path.join(save_path, "hparams.yaml"))
    DEVICE = f"cuda:{args.gpus[0]}"
    model = AudioModel(data_type="IEMOCAP", freeze_type="feature")
    runner = SpeechCls(
            model = model,
            fusion_type = config.fusion_type,
            lr = config.lr, 
            max_epochs = config.max_epochs,
            batch_size = config.batch_size
    )
    state_dict = torch.load(os.path.join(save_path, "best.ckpt"))
    runner.load_state_dict(state_dict.get("state_dict"))
    runner = runner.eval().to(DEVICE)
    return runner

def load_text_backbone(args):
    save_path = f"../classification/speech_cls/exp/IEMOCAP/{args.cv_split}/text_feature"
    config = OmegaConf.load(os.path.join(save_path, "hparams.yaml"))
    DEVICE = f"cuda:{args.gpus[0]}"
    model = TextModel(data_type="IEMOCAP", freeze_type="feature")
    runner = SpeechCls(
            model = model,
            fusion_type = config.fusion_type,
            lr = config.lr, 
            max_epochs = config.max_epochs,
            batch_size = config.batch_size
    )
    state_dict = torch.load(os.path.join(save_path, "best.ckpt"))
    runner.load_state_dict(state_dict.get("state_dict"))
    runner = runner.eval().to(DEVICE)
    return runner

def load_music_backbone(args):
    DEVICE = f"cuda:{args.gpus[0]}"
    save_path = f"../classification/music_cls/exp/Audioset/music_none"
    config = OmegaConf.load(os.path.join(save_path, "hparams.yaml"))
    model = MusicModel(
            pretrained_path=os.path.join(DATASET, "pretrained/music/compact_student.ckpt")
        )
    runner = ClsRunner(
            model = model,
            lr = config.lr, 
            max_epochs = config.max_epochs,
            batch_size = config.batch_size
    )
    state_dict = torch.load(os.path.join(save_path, "best.ckpt"))
    runner.load_state_dict(state_dict.get("state_dict"))
    runner = runner.eval().to(DEVICE)
    return runner

def projection_model(args):
    DEVICE = f"cuda:{args.gpus[0]}"
    save_path = f"exp/Audioset_IEMOCAP_{args.cv_split}/{args.branch_type}/{args.word_model}_{args.fusion_type}"
    config = OmegaConf.load(os.path.join(save_path, "hparams.yaml"))
    model = EmbModel(
        fusion_type = config.fusion_type
    )

    runner = TripletRunner(
        model = model,
        branch_type = config.branch_type,
        lr = config.lr, 
        max_epochs = config.max_epochs,
        batch_size = config.batch_size
    )
    state_dict = torch.load(os.path.join(save_path, "best.ckpt"))
    runner.load_state_dict(state_dict.get("state_dict"))
    runner = runner.eval().to(DEVICE)
    return runner

def speech_resampler(fname):
    src, _ = load_audio(
        path=fname,
        ch_format= STR_CH_FIRST,
        sample_rate= SPEECH_SAMPLE_RATE,
        downmix_to_mono= True)
    return torch.from_numpy(src.astype(np.float32))

def music_resampler(fname):
    src, _ = load_audio(
        path=fname,
        ch_format= STR_CH_FIRST,
        sample_rate= MUSIC_SAMPLE_RATE,
        downmix_to_mono= True)
    if src.shape[1] < 220500:
        pad = np.zeros((1,220500))
        pad[0,:src.shape[1]] = src[0,:]
        wav = torch.from_numpy(pad.astype(np.float32))
    else:
        input_length = 220500
        random_idx = random.randint(0, src.shape[1] - input_length)
        wav = torch.from_numpy(src[:,random_idx:random_idx+input_length].astype(np.float32))
    return wav

def music_extractor(args, music_backbone, joint_backbone):
    DEVICE = f"cuda:{args.gpus[0]}"
    audio = music_resampler(args.music_path)
    with torch.no_grad():
        print(audio.shape)
        audio_emb = music_backbone.model.extractor(audio.unsqueeze(0).to(DEVICE))
        audio_emb = joint_backbone.model.music_mlp(audio_emb)
    emb = audio_emb.squeeze(0).detach().cpu()
    return emb

def speech_extractor(args, audio_backbone, joint_backbone):
    DEVICE = f"cuda:{args.gpus[0]}"
    audio = speech_resampler(args.speech_path)
    with torch.no_grad():
        embs = audio_backbone.model.pooling_extractor(audio.to(DEVICE))
        embs = joint_backbone.model.speech_audio_mlp(embs)
    emb = embs.squeeze(0).detach().cpu()
    return emb


def main(args) -> None:
    if args.reproduce:
        seed_everything(42)
    audio_backbone = load_audio_backbone(args)
    music_backbone = load_music_backbone(args)
    joint_backbone = projection_model(args)
    if args.inference_type == "music_extractor":
        emb = music_extractor(args, music_backbone, joint_backbone)
    elif args.inference_type == "speech_extractor":
        emb = speech_extractor(args, audio_backbone, joint_backbone)
    print(emb.shape)
    return emb


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--inference_type", default="music_extractor", type=str)
    parser.add_argument("--music_path", default="../../dataset/raw/Audioset/wav/_1T2uagaTuw.mp3", type=str)
    parser.add_argument("--speech_path", default="../../dataset/raw/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav", type=str)
    parser.add_argument("--branch_type", default="3branch", type=str)
    parser.add_argument("--fusion_type", default="audio", type=str)
    parser.add_argument("--word_model", default="glove", type=str)
    parser.add_argument("--freeze_type", default="feature", type=str)
    parser.add_argument("--cv_split", default="01F", type=str)
    parser.add_argument("--gpus", default=[0], type=list)
    parser.add_argument("--reproduce", default=True, type=bool)

    args = parser.parse_args()
    main(args)