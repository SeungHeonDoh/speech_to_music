import json
import os
from argparse import ArgumentParser, Namespace, ArgumentTypeError
from pathlib import Path
import torch
import wandb
import torchaudio

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from speech_to_music.regression.speech_reg.model.emb_model import AudioModel, TextModel, FusionModel
from speech_to_music.regression.speech_reg.model.lightning_model import RegRunner
from speech_to_music.regression.speech_reg.loader.dataloader import DataPipeline
from speech_to_music.constants import (CLSREG_DATASET)

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

def main(args) -> None:
    if args.reproduce:
        seed_everything(42)

    save_path = f"exp/{args.data_type}/{args.fusion_type}_{args.freeze_type}_{args.is_augmentation}"
    wandb.init(config=args)
    wandb.run.name = f"{args.data_type}/{args.fusion_type}_{args.freeze_type}_{args.is_augmentation}"
    args = wandb.config

    pipeline = DataPipeline(
            data_type = args.data_type,
            is_augmentation = args.is_augmentation,
            batch_size = args.batch_size,
            num_workers = args.num_workers
    )

    if args.fusion_type == "audio":
        model = AudioModel(args.data_type, args.freeze_type)
    elif args.fusion_type == "text":
        model = TextModel(args.data_type, args.freeze_type)
    elif (args.fusion_type == "early_fusion") or (args.fusion_type == "late_fusion") or (args.fusion_type == "disentangle"):
        model = FusionModel(args.data_type, args.freeze_type, args.fusion_type)
    else:
        model = None

    runner = RegRunner(
            model = model,
            fusion_type = args.fusion_type,
            lr = args.lr, 
            max_epochs = args.max_epochs,
            batch_size = args.batch_size
    )

    state_dict = torch.load(os.path.join(save_path, "best.ckpt"), map_location="cpu")
    runner.load_state_dict(state_dict.get("state_dict"))

    logger = get_wandb_logger(runner)
    
    trainer = Trainer(
                    max_epochs= args.max_epochs,
                    num_nodes=args.num_nodes,
                    gpus= args.gpus,
                    strategy = DDPPlugin(find_unused_parameters=True),
                    logger=logger,
                    sync_batchnorm=True,
                    resume_from_checkpoint=None,
                    reload_dataloaders_every_epoch=True,
                    replace_sampler_ddp=False,
                )

    trainer.test(runner, datamodule=pipeline)

    # os.makedirs(os.path.join(CLSREG_DATASET, f"inference/regression/{args.data_type}_{args.cv_split}/"), exist_ok=True)
    torch.save(runner.inference, os.path.join(save_path, "inference.pt"))
    with open(Path(save_path, "results.json"), mode="w") as io:
        json.dump(runner.test_results, io, indent=4)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_type", default="IEMOCAP", type=str)
    parser.add_argument("--fusion_type", default="audio", type=str)
    parser.add_argument("--freeze_type", default="feature", type=str) 
    parser.add_argument("--is_augmentation", default=False , type=str2bool)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    # runner 
    parser.add_argument("--lr", default=5e-5, type=float)
    # trainer
    parser.add_argument("--max_epochs", default=200, type=int)
    parser.add_argument("--gpus", default=[0], type=list)
    parser.add_argument("--strategy", default="ddp", type=str)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument("--reproduce", default=False, type=str2bool) 
    # parser.add_argument("--deterministic", default=True, type=str2bool)
    # parser.add_argument("--benchmark", default=False, type=str2bool)

    args = parser.parse_args()
    main(args)