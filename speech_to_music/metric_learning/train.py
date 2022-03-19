import os
import json
import torch
from argparse import ArgumentParser, Namespace, ArgumentTypeError
from pathlib import Path

import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from transformers import Wav2Vec2Model, DistilBertModel

from speech_to_music.metric_learning.model.emb_model import EmbModel
from speech_to_music.metric_learning.model.lightning_model import TripletRunner
from speech_to_music.metric_learning.loader.dataloader import DataPipeline
from speech_to_music.constants import (DATASET)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

def get_checkpoint_callback(save_path) -> ModelCheckpoint:
    prefix = save_path
    suffix = "best"
    checkpoint_callback = ModelCheckpoint(
        dirpath=prefix,
        filename=suffix,
        save_top_k=1,
        save_last= False,
        monitor="val_loss",
        mode="min",
        save_weights_only=True,
        verbose=True,
    )
    return checkpoint_callback

def get_early_stop_callback() -> EarlyStopping:
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=20, verbose=True, mode="min"
    )
    return early_stop_callback

def save_hparams(args, save_path):
    save_config = OmegaConf.create(vars(args))
    os.makedirs(save_path, exist_ok=True)
    OmegaConf.save(config=save_config, f= Path(save_path, "hparams.yaml"))


def main(args) -> None:
    if args.reproduce:
        seed_everything(42)

    wandb.init(config=args)
    wandb.run.name = f"{args.music_type}_{args.speech_type}/{args.branch_type}/{args.word_model}_{args.fusion_type}_{args.is_augmentation}"
    save_path = f"exp/{args.music_type}_{args.speech_type}/{args.branch_type}/{args.word_model}_{args.fusion_type}_{args.is_augmentation}"
    save_hparams(args, save_path)
    args = wandb.config

    pipeline = DataPipeline(
        fusion_type = args.fusion_type,        
        word_model = args.word_model,
        is_augmentation = args.is_augmentation,
        batch_size = args.batch_size,
        num_workers = args.num_workers
    )

    model = EmbModel(
        fusion_type = args.fusion_type
    )

    runner = TripletRunner(
        model = model,
        branch_type = args.branch_type,
        lr = args.lr, 
        max_epochs = args.max_epochs,
        batch_size = args.batch_size
    )

    logger = WandbLogger()
    checkpoint_callback = get_checkpoint_callback(save_path)
    early_stop_callback = get_early_stop_callback()
    trainer = Trainer(
                    max_epochs= args.max_epochs,
                    num_nodes=args.num_nodes,
                    gpus= args.gpus,
                    strategy = DDPPlugin(find_unused_parameters=True),
                    logger=logger,
                    sync_batchnorm=True,
                    resume_from_checkpoint=None,
                    reload_dataloaders_every_epoch=True,
                    replace_sampler_ddp=True,
                    callbacks=[
                        early_stop_callback,
                        checkpoint_callback
                    ],
                )

    trainer.fit(runner, datamodule=pipeline)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--music_type", default="Audioset", type=str)
    parser.add_argument("--speech_type", default="IEMOCAP", type=str)
    parser.add_argument("--branch_type", default="3branch", type=str)
    parser.add_argument("--fusion_type", default="audio" , type=str)
    parser.add_argument("--word_model", default="glove", type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--is_augmentation", default=False, type=str2bool)    
    # runner 
    parser.add_argument("--lr", default=1e-3, type=float)
    # trainer
    parser.add_argument("--max_epochs", default=200, type=int)
    parser.add_argument("--gpus", default=[0], type=list)
    parser.add_argument("--strategy", default="ddp", type=str)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument("--reproduce", default=False, type=str2bool)    

    args = parser.parse_args()
    main(args)