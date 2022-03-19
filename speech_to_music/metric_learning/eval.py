import json
import os
from argparse import ArgumentParser, Namespace, ArgumentTypeError
from pathlib import Path

import wandb
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from speech_to_music.classification.speech_cls.model.emb_model import AudioModel, TextModel
from speech_to_music.classification.speech_cls.model.lightning_model import SpeechCls

from speech_to_music.metric_learning.model.emb_model import EmbModel
from speech_to_music.metric_learning.model.lightning_model import TripletRunner
from speech_to_music.metric_learning.loader.dataloader import DataPipeline

from speech_to_music.constants import (DATASET, IEMOCAP_TO_AUDIOSET)
from speech_to_music.modules.visualize_modules import umap_viz


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

def get_wandb_logger(model):
    logger = WandbLogger()
    logger.watch(model)
    return logger 


def load_speech_encoder(args):
    save_path = f"../classification/speech_cls/exp/{args.speech_type}/{args.cv_split}/audio_feature"
    config = OmegaConf.load(os.path.join(save_path, "hparams.yaml"))
    model = AudioModel(config.data_type, config.freeze_type)
    runner = SpeechCls(model = model, fusion_type = config.fusion_type, lr = config.lr, max_epochs = config.max_epochs, batch_size = config.batch_size)
    state_dict = torch.load(os.path.join(save_path, "best.ckpt"), map_location="cpu")
    runner.load_state_dict(state_dict.get("state_dict"))
    return runner.model

def main(args) -> None:
    if args.reproduce:
        seed_everything(42)
    wandb.init(config=args)
    wandb.run.name = f"eval/{args.music_type}_{args.speech_type}/{args.branch_type}/{args.word_model}_{args.fusion_type}_{args.is_augmentation}"
    save_path = f"exp/{args.music_type}_{args.speech_type}/{args.branch_type}/{args.word_model}_{args.fusion_type}_{args.is_augmentation}"
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
    state_dict = torch.load(os.path.join(save_path, "best.ckpt"), map_location="cpu")
    runner.load_state_dict(state_dict.get("state_dict"))

    logger = WandbLogger()
    
    trainer = Trainer(
                    max_epochs= args.max_epochs,
                    num_nodes=args.num_nodes,
                    gpus= args.gpus,
                    strategy = DDPPlugin(find_unused_parameters=True),
                    logger=logger,
                    sync_batchnorm=True,
                    resume_from_checkpoint=None,
                    reload_dataloaders_every_epoch=True,
                    replace_sampler_ddp=True
                )


    trainer.test(runner, datamodule=pipeline)
    results = runner.results 
    # viz_path = os.path.join(DATASET, f"visualize/{args.music_type}_{args.speech_type}_{args.cv_split}/{args.branch_type}/{args.word_model}_{args.fusion_type}")
    # umap_viz(
    #     runner.visualize['music_emb'], 
    #     runner.visualize['speech_emb'], 
    #     runner.visualize['tags_emb'], 
    #     runner.visualize['music_fname'], 
    #     runner.visualize['speech_fname'], 
    #     runner.visualize['music_gt'], 
    #     runner.visualize['speech_gt'], 
    #     runner.visualize['tags'], 
    #     viz_path, 
    #     branch_type=args.branch_type
    #     )
    # wandb.log(results)
    torch.save(runner.inference, Path(save_path, "inference.pt"))
    results['confusion_matrix'] = runner.confusion_matrix
    with open(Path(save_path, f"results.json"), mode="w") as io:
        json.dump(results, io, indent=4)
    print("finish save")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tid", default="test", type=str)    
    parser.add_argument("--music_type", default="Audioset", type=str)
    parser.add_argument("--speech_type", default="IEMOCAP", type=str)
    parser.add_argument("--branch_type", default="3branch", type=str)
    parser.add_argument("--fusion_type", default="audio" , type=str)
    parser.add_argument("--is_augmentation", default=False , type=str2bool)
    parser.add_argument("--word_model", default="glove", type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    # runner 
    parser.add_argument("--lr", default=1e-3, type=float)
    # trainer
    parser.add_argument("--max_epochs", default=200, type=int)
    parser.add_argument("--gpus", default=[0], type=list)
    parser.add_argument("--accelerator", default="ddp", type=str)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument("--reproduce", default=True, type=str2bool)
    # parser.add_argument("--deterministic", default=True, type=str2bool)
    # parser.add_argument("--benchmark", default=False, type=str2bool)

    args = parser.parse_args()
    main(args)