import os
import torch
import json
import numpy as np
from sklearn import metrics
import pandas as pd
from argparse import ArgumentParser
from speech_to_music.constants import (CLSREG_DATASET, AV_MAP, IEMOCAP_TO_AUDIOSET)
from speech_to_music.modules.visualize_modules import music_viz, speech_viz

def embedding_dict(speech_results, music_results):
    music_fname = [i for i in music_results['fnames']]
    speech_fname = [i for i in speech_results['fnames']]
    music_emb = music_results['y_pred'].numpy()
    speech_emb = speech_results['y_pred'].numpy()
    tags = [i for i in AV_MAP.keys()]
    tag_emb = np.stack([AV_MAP[i] for i in tags])
    total_emb = np.vstack([music_emb, speech_emb, tag_emb])
    df_music = pd.DataFrame(music_emb, index=music_fname)
    df_speech = pd.DataFrame(speech_emb, index=speech_fname)
    return total_emb, tag_emb, df_music, df_speech, tags

def main(args) -> None:
    music_gt= pd.read_csv(os.path.join(CLSREG_DATASET, "split/Audioset/test.csv"), index_col=0)
    speech_gt= pd.read_csv(os.path.join(CLSREG_DATASET, f"split/IEMOCAP/cv_split/iemocap_{args.cv_split}.test.csv"), index_col=0)
    music_results = torch.load(os.path.join(CLSREG_DATASET,f"inference/regression/{args.music_type}_music_none.pt"))
    speech_results = torch.load(os.path.join(CLSREG_DATASET,f"inference/regression/{args.speech_type}_{args.cv_split}/{args.fusion_type}_{args.freeze_type}.pt"))
    total_emb, tag_emb, df_music, df_speech, tags = embedding_dict(speech_results, music_results)
    save_path = os.path.join(CLSREG_DATASET,f"visualize/regression/{args.speech_type}_{args.cv_split}/{args.fusion_type}_{args.freeze_type}/")
    os.makedirs(save_path, exist_ok=True)
    music_viz(music_gt, df_music, tag_emb, tags, save_path)
    speech_viz(speech_gt, df_speech,tag_emb, tags, save_path)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--speech_type", default="IEMOCAP", type=str)
    parser.add_argument("--music_type", default="Audioset", type=str)
    parser.add_argument("--fusion_type", default="audio", type=str)
    parser.add_argument("--freeze_type", default="feature", type=str)
    parser.add_argument("--cv_split", default="01F", type=str)
    args = parser.parse_args()
    main(args)