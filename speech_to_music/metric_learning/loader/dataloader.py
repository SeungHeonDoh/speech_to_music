from typing import Callable, Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from speech_to_music.metric_learning.loader.dataset import Emotion_Dataset

class DataPipeline(LightningDataModule):
    def __init__(self, fusion_type, word_model, is_augmentation, batch_size, num_workers) -> None:
        super(DataPipeline, self).__init__()
        self.dataset_builder = Emotion_Dataset  
        self.fusion_type = fusion_type
        self.word_model = word_model
        self.is_augmentation = is_augmentation
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = DataPipeline.get_dataset(
                self.dataset_builder,
                split = "TRAIN",
                fusion_type = self.fusion_type,
                word_model = self.word_model,
                is_augmentation = self.is_augmentation,
            )

            self.val_dataset = DataPipeline.get_dataset(
                self.dataset_builder,                
                split = "VALID",
                fusion_type = self.fusion_type,
                word_model = self.word_model,
                is_augmentation = False,
            )

        if stage == "test" or stage is None:
            self.test_dataset = DataPipeline.get_dataset(
                self.dataset_builder,
                split = "TEST",
                fusion_type = self.fusion_type,
                word_model = self.word_model,
                is_augmentation = False,
            )

    def train_dataloader(self) -> DataLoader:
        return DataPipeline.get_dataloader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle = True
        )

    def val_dataloader(self) -> DataLoader:
        return DataPipeline.get_dataloader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle = False
        )

    def test_dataloader(self) -> DataLoader:
        return DataPipeline.get_dataloader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle = False
        )

    @classmethod
    def get_dataset(cls, dataset_builder: Callable, split, fusion_type, word_model, is_augmentation) -> Dataset:
        dataset = dataset_builder(split, fusion_type, word_model, is_augmentation)
        return dataset

    @classmethod
    def get_dataloader(cls, dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool, **kwargs) -> DataLoader:
        # dist_sampler = DistributedSampler(dataset, shuffle=shuffle)
        return DataLoader(
            dataset,
            # sampler = dist_sampler,
            batch_size = batch_size,
            num_workers = num_workers,
            shuffle=shuffle,
            persistent_workers=False,
            **kwargs
        )