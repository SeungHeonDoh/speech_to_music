import os
import torch
import json
import wandb
import numpy as np
from sklearn import metrics
from tqdm import tqdm
import pandas as pd
from argparse import ArgumentParser
from query_by_speech.constants import (CLSREG_DATASET, IEMOCAP_TO_AUDIOSET)
from query_by_speech.modules.loss_modules import get_mrr, get_prec_k
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score, average_precision_score, ndcg_score, precision_recall_fscore_support


def get_sim_table(music_results, speech_results, sim_type):
    music_fname = [i for i in music_results['fnames']]
    speech_fname = [i for i in speech_results['fnames']]
    music_emb = music_results['y_pred'].numpy()
    speech_emb = speech_results['y_pred'].numpy()
    if sim_type == "euclidean":
        sim_matrix = metrics.pairwise.euclidean_distances(speech_emb, music_emb)
    elif sim_type == "cosine":
        sim_matrix = metrics.pairwise.cosine_distances(speech_emb, music_emb)
    else:
        sim_matrix = None
    df_predict = pd.DataFrame(sim_matrix, index=speech_fname, columns=music_fname)
    return df_predict

def _get_retrieval_success(df_predict, speech_gt, music_gt, mapping_dict):
    tag_sim = pd.read_csv(os.path.join(CLSREG_DATASET, "va_sim.csv"), index_col=0)
    s_tags = list(speech_gt.columns)
    m_tags = list(music_gt.columns)
    sims, matchs, sort_sims, sort_matchs = [], [], [], []
    confusion_matrix = np.zeros((len(speech_gt.columns), len(music_gt.columns)))

    for speech_fname in tqdm(df_predict.index):
        speech_tag = speech_gt.loc[speech_fname].idxmax()
        candiate_pool = mapping_dict[speech_tag]
        item = df_predict.loc[speech_fname]
        sim, match = [], []
        for idx, music_fname in enumerate(item.index):
            music_tag = music_gt.loc[music_fname].idxmax() # matching 
            sim.append(tag_sim[speech_tag].loc[music_tag])
            if music_tag in candiate_pool:
                match.append(1)
            else:
                match.append(0)
        sort_sim, sort_match = [], []
        sim_music = item.sort_values(ascending=False)
        for idx, music_fname in enumerate(sim_music.index):
            music_tag = music_gt.loc[music_fname].idxmax() # matching 
            sort_sim.append(tag_sim[speech_tag].loc[music_tag])
            if music_tag in candiate_pool:
                sort_match.append(1)
            else:
                sort_match.append(0)
            if idx < 5: # check @5 case
                confusion_matrix[s_tags.index(speech_tag)][m_tags.index(music_tag)] += 1
        
        sims.append(sim)
        matchs.append(match)
        sort_sims.append(sort_sim)
        sort_matchs.append(sort_match)
    results = {
        "mrr": get_mrr(np.array(sort_matchs)),
        "prec_k": get_prec_k(np.array(sort_matchs)),
        "sim_k": get_prec_k(np.array(sort_sims)),
        "match_ndcg_k": ndcg_score(np.array(matchs), df_predict.to_numpy(), k=5),
        "sim_ndcg_k": ndcg_score(np.array(sims), df_predict.to_numpy(), k=5),
    }
    return results, pd.DataFrame(confusion_matrix, index=s_tags, columns=m_tags)


def get_result(music_results, speech_results, speech_gt, music_gt):
    df_euclidean = get_sim_table(music_results, speech_results, sim_type="euclidean")
    df_euclidean = df_euclidean.loc[speech_gt.index][music_gt.index]
    df_predict = 1 - df_euclidean
    euclidean_results, cm = _get_retrieval_success(df_predict, speech_gt, music_gt, IEMOCAP_TO_AUDIOSET)
    return euclidean_results, cm

def main(args) -> None:
    wandb.init(config=args)
    wandb.run.name = f"regression_{args.speech_type}_{args.music_type}/{args.fusion_type}_{args.freeze_type}"
    music_gt= pd.read_csv(os.path.join(CLSREG_DATASET, "split/Audioset/test.csv"), index_col=0)
    speech_gt= pd.read_csv(os.path.join(CLSREG_DATASET, "split/IEMOCAP/test.csv"), index_col=0)
    music_results = torch.load(os.path.join(f"../music_reg/exp/{args.music_type}/music_none/inference.pt"))
    speech_results = torch.load(os.path.join(f"../speech_reg/exp/{args.speech_type}/{args.fusion_type}_feature_{args.is_augmentation}/inference.pt"))
    results, cm = get_result(music_results, speech_results, speech_gt, music_gt)
    wandb.log(results)

    # for save
    results['confusion_matrix'] = cm.to_dict()
    save_path = f"exp/"
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, f"{args.fusion_type}_{args.freeze_type}_{args.is_augmentation}.json"), mode="w") as io:
        json.dump(results, io, indent=4)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--speech_type", default="IEMOCAP", type=str)
    parser.add_argument("--music_type", default="Audioset", type=str)
    parser.add_argument("--fusion_type", default="audio", type=str)
    parser.add_argument("--freeze_type", default="feature", type=str)
    parser.add_argument("--is_augmentation", default=False, type=bool)
    args = parser.parse_args()
    main(args)