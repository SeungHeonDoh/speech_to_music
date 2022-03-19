import os
import torch
import librosa
import pandas as pd
import numpy as np
from pytorch_lightning import Trainer, seed_everything
from argparse import ArgumentParser, Namespace, ArgumentTypeError
from transformers import Wav2Vec2Processor, DistilBertTokenizer

from speech_to_music.classification.speech_cls.model.emb_model import AudioModel, TextModel, FusionModel
from speech_to_music.classification.speech_cls.model.lightning_model import SpeechCls
from speech_to_music.constants import (CLSREG_DATASET, SPEECH_SAMPLE_RATE)
from tqdm import tqdm

def embedding_extractor(args, runner, tokenizer, processor, DEVICE):
    fl_tr = pd.read_csv(os.path.join(CLSREG_DATASET, "split", "IEMOCAP", "train.csv"), index_col=0)
    fl_va = pd.read_csv(os.path.join(CLSREG_DATASET, "split", "IEMOCAP", "valid.csv"), index_col=0)
    fl_te = pd.read_csv(os.path.join(CLSREG_DATASET, "split", "IEMOCAP", "test.csv"), index_col=0)
    fl = pd.concat([fl_tr, fl_va, fl_te])
    print("start inference total: " ,len(fl))
    df_meta = pd.read_csv(os.path.join(CLSREG_DATASET, f"split/IEMOCAP/annotation.csv"), index_col=0).set_index("wav_file_name")
    save_dir = os.path.join(CLSREG_DATASET, f"feature/{args.data_type}/pretrained/{args.fusion_type}_{args.freeze_type}_{args.is_augmentation}")
    os.makedirs(save_dir, exist_ok=True)
    for idx in tqdm(range(len(fl))):
        item = fl.iloc[idx]
        fname = item.name
        audio = np.load(os.path.join(CLSREG_DATASET, "feature", "IEMOCAP", 'npy', fname + ".npy"))
        audio = torch.from_numpy(audio).to(DEVICE)
        text = df_meta.loc[fname]['transcription']
        encoding = tokenizer(text, return_tensors='pt')
        token = encoding['input_ids'].to(DEVICE)
        mask = encoding['attention_mask'].to(DEVICE)
        with torch.no_grad():
            if args.fusion_type == "audio":
                embs = runner.model.pooling_extractor(audio)
                feature = embs.squeeze(0).detach().cpu()
            elif args.fusion_type == "text":
                embs = runner.model.pooling_extractor(token, mask)
                feature = embs.squeeze(0).detach().cpu()
            elif (args.fusion_type == "early_fusion") or (args.fusion_type == "late_fusion") or (args.fusion_type == "disentangle"):
                embs = runner.model.extractor(audio, token, mask, random_idx=2) 
                feature = embs.squeeze(0).detach().cpu()
            else:
                embs = None
        torch.save(feature, os.path.join(save_dir, f"{fname}.pt"))
    
def load_model(args, DEVICE):
    save_path = f"exp/{args.data_type}/{args.fusion_type}_{args.freeze_type}_{args.is_augmentation}"
    if args.fusion_type == "audio":
        model = AudioModel(args.data_type, args.freeze_type)
    elif args.fusion_type == "text":
        model = TextModel(args.data_type, args.freeze_type)
    elif (args.fusion_type == "early_fusion") or (args.fusion_type == "late_fusion") or (args.fusion_type == "disentangle"):
        model = FusionModel(args.data_type, args.freeze_type, args.fusion_type)
    else:
        model = None

    runner = SpeechCls(
            model = model,
            fusion_type = args.fusion_type,
            lr = args.lr, 
            max_epochs = args.max_epochs,
            batch_size = args.batch_size
    )
    state_dict = torch.load(os.path.join(save_path, "best.ckpt"), map_location="cpu")
    runner.load_state_dict(state_dict.get("state_dict"))
    runner = runner.to(DEVICE)
    return runner

def main(args) -> None:
    if args.reproduce:
        seed_everything(42)
    DEVICE = f"cuda:{args.gpus[0]}"
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
    runner = load_model(args, DEVICE)
    embedding_extractor(args, runner, tokenizer, processor, DEVICE)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_type", default="IEMOCAP", type=str)
    parser.add_argument("--fusion_type", default="audio", type=str)
    parser.add_argument("--freeze_type", default="feature", type=str)
    parser.add_argument("--is_augmentation", default=False, type=bool) 
    # runner 
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    # trainer
    parser.add_argument("--max_epochs", default=200, type=int)
    parser.add_argument("--gpus", default=[0], nargs="+", type=list)
    parser.add_argument("--reproduce", default=True, type=bool)

    args = parser.parse_args()
    main(args)