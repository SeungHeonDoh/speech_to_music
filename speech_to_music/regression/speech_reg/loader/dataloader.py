from typing import Callable, Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, BatchSampler

from speech_to_music.regression.speech_reg.loader.dataset import Emotion_Dataset

class DataPipeline(LightningDataModule):
    def __init__(self, data_type, is_augmentation, batch_size, num_workers) -> None:
        super(DataPipeline, self).__init__()
        self.dataset_builder = Emotion_Dataset        
        self.data_type = data_type
        self.is_augmentation = is_augmentation
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = DataPipeline.get_dataset(
                self.dataset_builder,
                split = "TRAIN",
                data_type = self.data_type,
                is_augmentation = self.is_augmentation
            )

            self.val_dataset = DataPipeline.get_dataset(
                self.dataset_builder,
                split = "VALID",
                data_type = self.data_type,
                is_augmentation = self.is_augmentation
            )

        if stage == "test" or stage is None:
            self.test_dataset = DataPipeline.get_dataset(
                self.dataset_builder,
                split = "TEST",
                data_type = self.data_type,
                is_augmentation = self.is_augmentation
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
    def get_dataset(cls, dataset_builder: Callable, split, data_type, is_augmentation) -> Dataset:
        dataset = dataset_builder(split, data_type, is_augmentation)
        return dataset

    @classmethod
    def get_dataloader(cls, dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool, **kwargs) -> DataLoader:
        all_indices = list(range(len(dataset)))
        sampler = SubsetRandomSampler(all_indices)
        batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=True)
        return DataLoader(
            dataset,
            sampler=batch_sampler,
            batch_size=None,
            num_workers = num_workers, 
            persistent_workers=False,
            **kwargs
        )
