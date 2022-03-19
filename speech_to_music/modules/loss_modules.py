import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score, average_precision_score, ndcg_score, precision_recall_fscore_support, balanced_accuracy_score
from speech_to_music.constants import (DATASET, IEMOCAP_TO_AUDIOSET, RAVDESS_TO_AUDIOSET)

class SyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]
        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)
        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)
        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]

def get_prec_k(retrieval_success):
    p_5 = retrieval_success[:, :5].mean(axis=1)
    return p_5.mean()


def get_mrr(retrieval_success):
    # mrrs = [1/(line.argmax()+1) for line in retrieval_success]
    mrrs = []
    for line in retrieval_success:
        collect = False
        for idx, success in enumerate(line):
            if success:
                mrrs.append(1/(idx+1))
                collect = True
                break
        if not collect:
            mrrs.append(1/(len(line))) # in fail case we use last index of model
    return np.mean(mrrs)


def get_metric_result(speech_emb, music_emb, speech_fname, music_fname, speech_gt, music_gt, mapping_dict):
    tag_sim = pd.read_csv(os.path.join(DATASET, "va_sim.csv"), index_col=0)
    stm_sim = _get_similarity(speech_emb, music_emb)
    df_predict = pd.DataFrame(stm_sim, index=speech_fname, columns=music_fname)
    confusion_matrix = np.zeros((len(speech_gt.columns), len(music_gt.columns)))
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

def unseen_retrieval_success(speech_emb, music_emb, speech_fname, music_fname, speech_gt, music_gt, mapping_dict):
    stm_sim = _get_similarity(speech_emb, music_emb)
    df_predict = pd.DataFrame(stm_sim, index=speech_fname, columns=music_fname)
    retrieval_success = []
    confusion_matrix = np.zeros((len(speech_gt.columns), len(music_gt.columns)))
    s_tags = list(speech_gt.columns)
    m_tags = list(music_gt.columns)
    for speech_fname in df_predict.index:
        speech_tag = speech_gt.loc[speech_fname].idxmax()
        candiate_pool = mapping_dict[speech_tag]
        item = df_predict.loc[speech_fname]
        sim_music = item.sort_values(ascending=False)
        success = []
        for idx, music_fname in enumerate(sim_music.index):
            music_tag = music_gt.loc[music_fname].idxmax()
            if music_tag in candiate_pool:
                success.append(1)
            else:
                success.append(0)
            if idx < 5: # check @5 case
                confusion_matrix[s_tags.index(speech_tag)][m_tags.index(music_tag)] += 1
        retrieval_success.append(success)
    return np.array(retrieval_success), pd.DataFrame(confusion_matrix, index=s_tags, columns=m_tags)

def _get_similarity(embs_a, embs_b):
    sim_scores = np.zeros((len(embs_a), len(embs_b)))
    for i in range(len(embs_a)):
        sim_scores[i] = np.array(nn.CosineSimilarity(dim=-1)(embs_a[i], embs_b))
    return sim_scores

def get_r2(predict, label):
    y_pred = predict.detach().cpu()
    y_true = label.detach().cpu()
    rv, ra = r2_score(y_true.numpy(), y_pred.numpy(), multioutput='raw_values')
    r2 = np.mean([rv, ra])
    return r2, y_pred, y_true

def accuracy(predict, label):
    predict = predict.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    y_pred = np.argmax(predict, axis=1)
    y_true = np.argmax(label, axis=1)
    acc = accuracy_score(y_true, y_pred)
    return acc

def get_scores(predict, label):
    predict = predict.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    y_pred = np.argmax(predict, axis=1)
    y_true = np.argmax(label, axis=1)
    WA = accuracy_score(y_true, y_pred)
    UA = balanced_accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(label, predict, average='macro')
    pr_auc = average_precision_score(label, predict, average='macro')
    pre_rec_f1_macro = precision_recall_fscore_support(y_true, y_pred, average='macro')
    pre_rec_f1_micro = precision_recall_fscore_support(y_true, y_pred, average='micro')
    return WA, UA, roc_auc, pr_auc, pre_rec_f1_macro, pre_rec_f1_micro, y_pred, y_true

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)      # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))       # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class OneHotToCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_hat = F.softmax(y_hat, dim=-1)
        return self.loss(y_hat, y)

class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.relu = nn.ReLU()

    def forward(self, anchor, positive, negative, size_average=True):
        cosine_positive = nn.CosineSimilarity(dim=-1)(anchor, positive)
        cosine_negative = nn.CosineSimilarity(dim=-1)(anchor, negative)
        losses = self.relu(self.margin - cosine_positive + cosine_negative)
        return losses.mean()