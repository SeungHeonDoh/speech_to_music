import os
import random
import torch
import torch.nn as nn
import pandas as pd
from pytorch_lightning import LightningModule
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from speech_to_music.modules.loss_modules import TripletLoss, SyncFunction, get_metric_result
from speech_to_music.modules.opt_modules import CosineAnnealingWarmupRestarts
from speech_to_music.modules.visualize_modules import umap_viz
from speech_to_music.constants import (DATASET, IEMOCAP_TO_AUDIOSET)

class TripletRunner(LightningModule):
    def __init__(self, model, branch_type, lr, batch_size, max_epochs):
        super().__init__()
        self.model = model
        self.branch_type = branch_type
        self.criterion = TripletLoss(margin=0.4)
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.2
        )
        # Source: https://github.com/openai/CLIP/issues/107
        num_training_steps = len(self.trainer.datamodule.train_dataloader()) # single-gpu case
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=num_training_steps,
            cycle_mult=1.0,
            max_lr=self.lr,
            min_lr=1e-8,
            warmup_steps=int(0.1*num_training_steps),
            gamma=1.0
        )
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
    
    def shared_step(self, batch, types):
        tts_tag, tts_speech, tts_binary, \
        ttm_tag, ttm_music, ttm_binary, \
        stm_speech, stm_music, stm_binary = batch

        # 2branch case
        if types == "train":
            random_idx = torch.randint(0,3,(1,))
            stm_speech_emb, stm_music_emb, ttm_tag_emb = self.model(stm_speech, stm_music, ttm_tag, random_idx)      
            tts_speech_emb, ttm_music_emb, tts_tag_emb = self.model(tts_speech, ttm_music, tts_tag, random_idx)
        else:
            stm_speech_emb, stm_music_emb, ttm_tag_emb = self.model(stm_speech, stm_music, ttm_tag, random_idx=2)
            tts_speech_emb, ttm_music_emb, tts_tag_emb = self.model(tts_speech, ttm_music, tts_tag, random_idx=2)
        stm_anchor, stm_pos, stm_neg = self.triplet_sampling(stm_speech_emb, stm_music_emb, stm_binary)
        stm_loss = self.criterion(stm_anchor, stm_pos, stm_neg) 
        if self.branch_type == "3branch":
            tts_anchor, tts_pos, tts_neg = self.triplet_sampling(tts_tag_emb, tts_speech_emb, tts_binary)
            ttm_anchor, ttm_pos, ttm_neg = self.triplet_sampling(ttm_tag_emb, ttm_music_emb, ttm_binary)
            tts_loss = self.criterion(tts_anchor, tts_pos, tts_neg)
            ttm_loss = self.criterion(ttm_anchor, ttm_pos, ttm_neg)
            loss = (0.3 * tts_loss) + (0.3 * ttm_loss) + (0.4 * stm_loss)
        else:
            loss = stm_loss
        return loss
   
    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, types="train")
        self.log_dict(
            {"train_loss": loss},
            prog_bar=False,
            # logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True
            )
        return loss

    def training_step_end(self, step_output):
        return step_output

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, types="eval")
        return {"val_loss": loss}

    def validation_step_end(self, step_output):
        return step_output

    def validation_epoch_end(self, outputs):
        val_loss = torch.mean(torch.stack([output["val_loss"] for output in outputs]))
        self.log_dict(
            {
                "val_loss": val_loss
            },
            prog_bar=True,
            # logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )

    def test_step(self, batch, batch_idx):
        music_tag, music_tag_binary, m_tag, \
        speech_tag, speech_tag_binary, s_tag, \
        music_wav, music_binary, music_fname, \
        speech, speech_binary, speech_fname, \
        speech_fnames, music_fnames= batch

        music_tag_emb = self.model.tag_to_emb(music_tag)
        speech_tag_emb = self.model.tag_to_emb(speech_tag)
        music_emb = self.model.music_to_emb(music_wav)
        speech_emb = self.model.speech_to_emb(speech, random_idx=2)

        return {
            # embedding
            "music_tag_emb":music_tag_emb,
            "speech_tag_emb":speech_tag_emb,
            "music_emb":music_emb,
            "speech_emb":speech_emb,
            # label
            "music_tag_binary":music_tag_binary,
            "speech_tag_binary":speech_tag_binary,
            "music_binary":music_binary,
            "speech_binary":speech_binary,
            # fname
            "m_tag":m_tag,
            "s_tag":s_tag,
            "music_fname":music_fname,
            "speech_fname":speech_fname,
            "music_fnames": music_fnames,
            "speech_fnames": speech_fnames
        }

    def test_step_end(self, batch_parts):
        return batch_parts

    def test_epoch_end(self, outputs):
        music_tag_emb = torch.stack([i for output in outputs for i in output['music_tag_emb']]).detach().cpu()
        speech_tag_emb = torch.stack([i for output in outputs for i in output['speech_tag_emb']]).detach().cpu()
        music_emb = torch.stack([i for output in outputs for i in output['music_emb']]).detach().cpu()
        speech_emb = torch.stack([i for output in outputs for i in output['speech_emb']]).detach().cpu()
        music_tag_binary = torch.stack([i for output in outputs for i in output['music_tag_binary']]).detach().cpu()
        speech_tag_binary = torch.stack([i for output in outputs for i in output['speech_tag_binary']]).detach().cpu()
        music_binary = torch.stack([i for output in outputs for i in output['music_binary']]).detach().cpu()
        speech_binary = torch.stack([i for output in outputs for i in output['speech_binary']]).detach().cpu()
        m_tag = [i for output in outputs for i in output['m_tag']]
        s_tag = [i for output in outputs for i in output['s_tag']]
        music_fname = [i for output in outputs for i in output['music_fname']]
        speech_fname = [i for output in outputs for i in output['speech_fname']]
        # get embedding dictionary
        m_to_emb = {tag:emb for tag, emb in zip(music_fname, music_emb)}
        s_to_emb = {tag:emb for tag, emb in zip(speech_fname, speech_emb)}
        m_tag_to_emb = {tag:emb for tag, emb in zip(m_tag, music_tag_emb)}
        s_tag_to_emb = {tag:emb for tag, emb in zip(s_tag, speech_tag_emb)}
        all_tag_emb = {}
        all_tag_emb.update(m_tag_to_emb)
        all_tag_emb.update(s_tag_to_emb)
        # get embeddings
        music_fname = [i for i in m_to_emb.keys()]
        music_emb = torch.stack([m_to_emb[fname] for fname in music_fname])
        speech_fname = [i for i in s_to_emb.keys()]
        speech_emb = torch.stack([s_to_emb[fname] for fname in speech_fname])
        tags = [i for i in all_tag_emb.keys()]
        tags_emb = torch.stack([all_tag_emb[tag] for tag in tags])
        music_fnames = [i[0] for i in outputs[0]['music_fnames']]
        speech_fnames = [i[0] for i in outputs[0]['speech_fnames']]
        m_tag_to_binary = {tag:binary for tag, binary in zip(m_tag, music_tag_binary)}
        s_tag_to_binary = {tag:binary for tag, binary in zip(s_tag, speech_tag_binary)}
        music_gt = pd.DataFrame(m_tag_to_binary, index=music_fnames)
        speech_gt = pd.DataFrame(s_tag_to_binary, index=speech_fnames)
        
        # get score
        results, cm = get_metric_result(speech_emb, music_emb, speech_fname, music_fname, speech_gt, music_gt, IEMOCAP_TO_AUDIOSET)
        self.log_dict(results)
        self.confusion_matrix = cm.to_dict()
        self.results = results
        self.inference = {
            "music_emb":music_emb,
            "speech_emb":speech_emb,
            "tags_emb":tags_emb,
            "music_fname":music_fname,
            "speech_fname":speech_fname,
            "music_gt":music_gt,
            "speech_gt":speech_gt,
            "tags":tags
        }


    def triplet_sampling(self, anchor_emb, positive_emb, binary, is_weighted=True):
        if torch.distributed.get_world_size() > 1:
            anchor_emb = SyncFunction.apply(anchor_emb)
            positive_emb = SyncFunction.apply(positive_emb)
            binary = SyncFunction.apply(binary)
        num_batch = len(anchor_emb)
        if is_weighted:
            # get distance weights
            anchor_norm = anchor_emb / anchor_emb.norm(dim=1)[:, None]
            positive_norm = positive_emb / positive_emb.norm(dim=1)[:, None]
            dot_sim = torch.matmul(anchor_norm, positive_norm.T)
            weights = (dot_sim + 1) / 2

            # masking
            mask = 1 - torch.matmul(binary, binary.T)
            masked_weights = weights * mask

            # sampling
            index_array = torch.arange(num_batch)
            negative_ix = [random.choices(index_array, weights=masked_weights[i], k=1)[0].item() for i in range(num_batch)]
            negative_emb = positive_emb[negative_ix]

        else:
            num_batch = len(anchor_emb)

            # masking
            mask = 1 - torch.matmul(binary, binary.T)

            # sampling
            index_array = torch.arange(num_batch)
            negative_ix = [random.choices(index_array, weights=mask[i], k=1)[0].item() for i in range(num_batch)]
            negative_emb = positive_emb[negative_ix]
        return anchor_emb, positive_emb, negative_emb
