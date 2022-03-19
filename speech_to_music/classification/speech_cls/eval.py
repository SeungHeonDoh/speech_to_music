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

from speech_to_music.classification.speech_cls.model.emb_model import AudioModel, TextModel, FusionModel
from speech_to_music.classification.speech_cls.model.lightning_model import SpeechCls
from speech_to_music.classification.speech_cls.loader.dataloader import DataPipeline

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from speech_to_music.constants import (CLSREG_DATASET, AUDIOSET_TAGS, IEMOCAP_TAGS)

def save_cm(predict, label, label_name, save_path):
    predict_ = [label_name[i] for i in predict]
    label_ = [label_name[i] for i in label]
    cm = confusion_matrix(label_, predict_, labels=label_name)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_name)
    disp.plot(xticks_rotation="vertical")
    plt.savefig(os.path.join(save_path, 'cm.png'), dpi=150)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

def main(args) -> None:
    if args.reproduce:
        seed_everything(42)

    save_path = f"exp/{args.data_type}/{args.fusion_type}_{args.freeze_type}_{args.is_augmentation}"
    wandb.init(config=args)
    wandb.run.name = f"{args.data_type}/{args.fusion_type}_{args.freeze_type}_{args.is_augmentation}"
    args = wandb.config


    pipeline = DataPipeline(
            data_type = args.data_type,
            batch_size = args.batch_size,
            num_workers = args.num_workers
    )

    if args.fusion_type == "audio":
        model = AudioModel(args.data_type, args.freeze_type)
    elif args.fusion_type == "text":
        model = TextModel(args.data_type, args.freeze_type)
    elif (args.fusion_type == "early_fusion") or (args.fusion_type == "late_fusion") or (args.fusion_type == "disentangle"):
        model = FusionModel(args.data_type, args.freeze_type, args.fusion_type)
    elif (args.fusion_type == "attention") or (args.fusion_type == "attention_mask"):
        model = AttentionModel(args.data_type, args.freeze_type, args.fusion_type)
    else:
        model = None

    runner = SpeechCls(
            model = model,
            fusion_type = args.fusion_type,
            lr = args.lr, 
            max_epochs = args.max_epochs,
            batch_size = args.batch_size
    )

    state_dict = torch.load(os.path.join(save_path, "best.ckpt"))
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
                    replace_sampler_ddp=False,
                )


    trainer.test(runner, datamodule=pipeline)

    with open(Path(save_path, "results.json"), mode="w") as io:
        json.dump(runner.test_results, io, indent=4)

    save_cm(runner.inference['y_pred'], runner.inference['y_true'], IEMOCAP_TAGS, save_path)
    os.makedirs(os.path.join(CLSREG_DATASET, f"inference/classification/{args.data_type}"), exist_ok=True)
    torch.save(runner.inference, os.path.join(CLSREG_DATASET, f"inference/classification/{args.data_type}/{args.fusion_type}_{args.freeze_type}_{args.is_augmentation}.pt"))
    print(classification_report(runner.inference['y_true'], runner.inference['y_pred'], target_names=IEMOCAP_TAGS))
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_type", default="IEMOCAP", type=str)
    parser.add_argument("--fusion_type", default="audio", type=str)
    parser.add_argument("--freeze_type", default="feature", type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=12, type=int)
    # runner 
    # runner 
    parser.add_argument("--lr", default=5e-5, type=float)
    # trainer
    parser.add_argument("--max_epochs", default=200, type=int)
    parser.add_argument("--is_augmentation", default=False , type=str2bool)
    parser.add_argument("--gpus", default=[3], type=list)
    parser.add_argument("--strategy", default="ddp", type=str)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument("--reproduce", default=True, type=str2bool) 
    # parser.add_argument("--deterministic", default=True, type=str2bool)
    # parser.add_argument("--benchmark", default=False, type=str2bool)

    args = parser.parse_args()
    main(args)