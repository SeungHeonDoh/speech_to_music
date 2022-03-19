import torch
import torch.nn as nn

from pytorch_lightning import LightningModule
from torch.optim import AdamW
from speech_to_music.modules.loss_modules import get_scores, accuracy
from speech_to_music.modules.opt_modules import CosineAnnealingWarmupRestarts

class SpeechCls(LightningModule):
    def __init__(self, model, fusion_type, lr, batch_size, max_epochs):
        super().__init__()
        self.model = model
        self.fusion_type = fusion_type
        self.criterion = nn.CrossEntropyLoss()
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
            warmup_steps=int(0.2*num_training_steps),
            gamma=1.0
        )
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        
    def shared_step(self, batch, types):
        item_dict = batch
        audios = item_dict['audios']
        audio_mask = item_dict['audio_mask']
        labels = item_dict['labels']
        token = item_dict['token']
        mask = item_dict['mask']

        if self.fusion_type == "audio":
            prediction = self.model(audios, audio_mask)        
            loss = self.criterion(prediction, labels.argmax(dim=-1))
        elif self.fusion_type == "text":
            prediction, loss = self.model(token, mask, labels)        
        elif (self.fusion_type == "early_fusion") or (self.fusion_type == "late_fusion") or (self.fusion_type == "disentangle") or (self.fusion_type == "attention") or (self.fusion_type == "attention_mask"):
            if types == "train":
                random_idx = torch.randint(0,3,(1,))
                prediction = self.model(audios, audio_mask, token, mask, random_idx)
                loss = self.criterion(prediction, labels.argmax(dim=-1))
            else:
                prediction = self.model(audios, audio_mask, token, mask, random_idx=2) # use full dimension
                loss = self.criterion(prediction, labels.argmax(dim=-1))
        else:
            prediction = None
            loss = None 
        return loss, prediction, labels

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.shared_step(batch, types="train")
        self.log_dict(
            {"train_loss": loss},
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        return loss

    def training_step_end(self, step_output):
        return step_output

    def validation_step(self, batch, batch_idx):
        loss, prediction, labels = self.shared_step(batch, types="eval")
        return {
            "val_loss": loss,
            "prediction": prediction,
            "labels": labels
            }

    def validation_step_end(self, step_output):
        return step_output

    def validation_epoch_end(self, outputs):
        val_loss = torch.mean(torch.stack([output["val_loss"] for output in outputs]))
        val_prediction = torch.stack([item for output in outputs for item in output["prediction"]])
        val_labels = torch.stack([item for output in outputs for item in output["labels"]])
        val_acc = accuracy(val_prediction, val_labels)
        self.log_dict(
            {
                "val_loss": val_loss,
                "val_acc": val_acc
            },
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def test_step(self, batch, batch_idx):
        fnames = batch['fnames']
        loss, prediction, labels = self.shared_step(batch, types="eval")
        return {
            "fnames" : fnames,
            "val_loss": loss,
            "prediction": prediction,
            "labels": labels
            }

    def test_step_end(self, batch_parts):
        return batch_parts

    def test_epoch_end(self, outputs):
        fnames =[output["fnames"][0] for output in outputs]
        val_loss = torch.mean(torch.stack([output["val_loss"] for output in outputs]))
        val_prediction = torch.stack([item for output in outputs for item in output["prediction"]])
        val_labels = torch.stack([item for output in outputs for item in output["labels"]])
        WA, UA, roc_auc, pr_auc, pre_rec_f1_macro, pre_rec_f1_micro, y_pred, y_true = get_scores(val_prediction, val_labels)
        result = {
            "test_loss": float(val_loss.detach().cpu()), 
            "WA": WA, 
            "UA": UA, 
            "roc_auc": roc_auc, 
            "pr_auc":pr_auc,
            "pre_rec_f1_macro":pre_rec_f1_macro,
            "pre_rec_f1_micro":pre_rec_f1_micro
        }
        inference = {
            "logit": val_prediction.detach().cpu(),
            "val_labels": val_labels.detach().cpu(),
            "fnames": fnames,
            "y_pred": y_pred,
            "y_true": y_true
        }
        self.test_results = result
        self.inference = inference