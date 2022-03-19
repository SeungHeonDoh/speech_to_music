import os
import torch
import json
import numpy as np
import pandas as pd

from sklearn import metrics
from tqdm import tqdm
from argparse import ArgumentParser
from query_by_speech.constants import (CLSREG_DATASET, IEMOCAP_TO_AUDIOSET, AUDIOSET_TAGS, IEMOCAP_TO_AUDIOSET)
from query_by_speech.modules.loss_modules import get_mrr, get_prec_k, ndcg_score

def _get_retrieval_success(df_speech, df_music, speech_gt, music_gt, mapping_dict):
    tag_sim = pd.read_csv(os.path.join(CLSREG_DATASET, "va_sim.csv"), index_col=0)
    gt_sims, pred_sims, retrieval_success = [], [], []
    for idx in tqdm(range(len(df_speech))):
        item = df_speech.iloc[idx]
        gt_query = speech_gt.iloc[idx].idxmax()
        query = item['predict']
        gt_sim, pred_sim, success = [], [], []
        try:
            sim_music = df_music[query]
            for music_fname in sim_music.index:
                music_tag = music_gt.loc[music_fname].idxmax()
                pred_sim.append(tag_sim[query].loc[music_tag])
                gt_sim.append(tag_sim[gt_query].loc[music_tag])
        except:
            pred_sim = [0 for _ in range(len(df_music))]
            gt_sim = [0 for _ in range(len(df_music))]

        # sort case
        try:
            sim_music = df_music[query].sort_values(ascending=False)
            for music_fname in sim_music.index:
                music_tag = music_gt.loc[music_fname].idxmax() # get tag
                if music_tag == gt_query:
                    success.append(1)
                else:
                    success.append(0)
        except:
            success = [0 for _ in range(len(df_music))]

        pred_sims.append(pred_sim)
        gt_sims.append(gt_sim)
        retrieval_success.append(success)

    return np.array(pred_sims), np.array(gt_sims), np.array(retrieval_success)

def get_result(music_results, speech_results, speech_gt, music_gt):
    music_fname = [i for i in music_results['fnames']]
    music_logit = music_results['logit'].numpy()
    df_music = pd.DataFrame(music_logit, index=music_fname, columns=AUDIOSET_TAGS)

    speech_fname = [i for i in speech_results['fnames']]
    speech_predict = speech_results['y_pred']
    df_speech = pd.DataFrame([speech_gt.columns[i] for i in speech_predict], index=speech_fname, columns=['predict'])
    pred_sims, gt_sims, retrieval_success = _get_retrieval_success(df_speech, df_music, speech_gt, music_gt, IEMOCAP_TO_AUDIOSET)
    return {
        "mrr": get_mrr(retrieval_success),
        "prec_k": get_prec_k(retrieval_success),
        "sim_ndcg_k": ndcg_score(gt_sims, pred_sims, k=5),
    }
    

def main(args) -> None:
    music_gt= pd.read_csv(os.path.join(CLSREG_DATASET, "split/Audioset/test.csv"), index_col=0)
    speech_gt= pd.read_csv(os.path.join(CLSREG_DATASET, "split/IEMOCAP/test.csv"), index_col=0)
    music_results = torch.load(os.path.join(f"../music_cls/exp/{args.music_type}/music_none/inference.pt"))
    speech_results = torch.load(os.path.join(f"../speech_cls/exp/{args.speech_type}/{args.fusion_type}_feature_{args.is_augmentation}/inference.pt"))
    results = get_result(music_results, speech_results, speech_gt, music_gt)

    save_path = f"exp/{args.speech_type}_{args.music_type}/"
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, f"{args.fusion_type}_{args.freeze_type}_{args.is_augmentation}.json"), mode="w") as io:
        json.dump(results, io, indent=4)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--speech_type", default="IEMOCAP", type=str)
    parser.add_argument("--music_type", default="Audioset", type=str)
    parser.add_argument("--fusion_type", default="audio", type=str)
    parser.add_argument("--freeze_type", default="feature", type=str)
    parser.add_argument("--is_augmentation", default=True, type=bool)
    args = parser.parse_args()
    main(args)