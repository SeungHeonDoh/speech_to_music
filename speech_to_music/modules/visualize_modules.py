import os
import umap
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from speech_to_music.constants import (EMOTION_TO_COLOR)


def _get_xy(df_gt, df_pred, label, c):
    fnames = df_gt[df_gt[label] != 0].index
    x = df_pred.loc[fnames].to_numpy()[:,0]
    y = df_pred.loc[fnames].to_numpy()[:,1]
    plt.scatter(x, y, marker="o", color=c,alpha=0.6, s=30, linewidths=1, label=label)

def music_viz(music_gt, df_music, tag_emb, tags, save_path, is_label=True):
    plt.figure(figsize=(8,8), dpi=150)
    for label in ["tender","happy","funny","exciting","angry","scary","sad","noise"]:
        _get_xy(music_gt, df_music, label, EMOTION_TO_COLOR[label])
    if is_label:
        plt.scatter(tag_emb[:,0], tag_emb[:,1], marker="x", color="orange", alpha=1, s=50, linewidths=1, label="emotion")
        for x,y,tag in zip(tag_emb[:,0], tag_emb[:,1], tags):
            plt.text(x + 0.01, y - 0.005, tag, fontsize=10, alpha=1, backgroundcolor='#FFFFFF50')
    plt.legend(scatterpoints=1)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path,"music.png"))

def speech_viz(speech_gt, df_speech,tag_emb, tags, save_path, is_label=True):
    plt.figure(figsize=(8,8), dpi=150)
    for label in ['neutral', 'angry','happy', 'sad']:
        _get_xy(speech_gt, df_speech, label, EMOTION_TO_COLOR[label])
    if is_label:
        plt.scatter(tag_emb[:,0], tag_emb[:,1], marker="x", color="orange", alpha=1, s=50, linewidths=1, label="emotion")
        for x,y,tag in zip(tag_emb[:,0], tag_emb[:,1], tags):
            plt.text(x + 0.01, y - 0.005, tag, fontsize=10, alpha=1, backgroundcolor='#FFFFFF50')
    plt.legend(scatterpoints=1)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path,"speech.png"))

def umap_viz(music_emb, speech_emb, tag_emb, music_fname, speech_fname, music_gt, speech_gt, tags, save_path, branch_type):
    embeddings = torch.cat([music_emb,tag_emb], dim=0)
    m_umap_model = umap.UMAP(n_neighbors=100,
          min_dist=0.5,
          metric='cosine',
          verbose=1,
          n_jobs=8).fit(embeddings)
    music_umap = m_umap_model.transform(music_emb)
    speech_umap = m_umap_model.transform(speech_emb)
    tag_umap = m_umap_model.transform(tag_emb)
    # parsing
    df_music = pd.DataFrame(music_umap, index=music_fname)
    df_speech = pd.DataFrame(speech_umap, index=speech_fname)
    tag_emb = np.stack(tag_umap)
    if branch_type == "2branch":
        music_viz(music_gt, df_music, tag_emb, tags, save_path, is_label=False)
        speech_viz(speech_gt, df_speech, tag_emb, tags, save_path, is_label=False)
    else:
        music_viz(music_gt, df_music, tag_emb, tags, save_path, is_label=True)
        speech_viz(speech_gt, df_speech, tag_emb, tags, save_path, is_label=True)