"""DataModules for DeepChem molecular datasets (QM7, QM8, QM9, PDBbind)."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Literal

import torch
from lightning import LightningDataModule
from rdkit import RDLogger
from torch.utils.data import DataLoader, TensorDataset

from reg_transfo.utils.env_vars import DATA_DIR, NUM_WORKERS

# Suppress all deepchem warnings and logging BEFORE import
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("deepchem").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Suppress RDKit C-level logging (deprecation warnings, sanitization errors, etc.)
RDLogger.DisableLog("rdApp.*")


class DeepChemDataModule(LightningDataModule):
    """Base DataModule for DeepChem molecular datasets."""

    def __init__(
        self,
        dataset_name: Literal["qm7", "qm8", "qm9", "pdbbind"],
        data_dir: str | Path = DATA_DIR,
        batch_size: int = 32,
        num_workers: int = NUM_WORKERS,
        featurizer: str = "ECFP",
        splitter: str = "random",
        pin_memory: bool = True,
        shuffle: bool = True,
    ):
        """
        Args:
            dataset_name: Name of the dataset to load (qm7, qm8, qm9, or pdbbind)
            data_dir: Directory to store/load data
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            featurizer: Featurizer to use (ECFP, GraphConv, etc.)
            splitter: Splitting method (random, scaffold, etc.)
            pin_memory: Pin memory for faster GPU transfer
            shuffle: Shuffle training data
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir) / "deepchem" / dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.featurizer = featurizer
        self.splitter = splitter
        self.pin_memory = pin_memory
        self.shuffle = shuffle

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.tasks = None

        self.save_hyperparameters()

    def prepare_data(self):
        """Download dataset if needed."""
        from deepchem import molnet

        self.data_dir.mkdir(parents=True, exist_ok=True)

        loader_map = {
            "qm7": molnet.load_qm7,
            "qm8": molnet.load_qm8,
            "qm9": molnet.load_qm9,
            "pdbbind": molnet.load_pdbbind,
        }

        loader = loader_map[self.dataset_name]
        loader(
            featurizer=self.featurizer,
            splitter=self.splitter,
            data_dir=str(self.data_dir),
        )

    def setup(self, stage: str | None = None):
        """Load and setup datasets."""
        from deepchem import molnet

        loader_map = {
            "qm7": molnet.load_qm7,
            "qm8": molnet.load_qm8,
            "qm9": molnet.load_qm9,
            "pdbbind": molnet.load_pdbbind,
        }

        loader = loader_map[self.dataset_name]
        self.tasks, datasets, _ = loader(
            featurizer=self.featurizer,
            splitter=self.splitter,
            data_dir=str(self.data_dir),
        )

        train_dc, valid_dc, test_dc = datasets

        if stage == "fit" or stage is None:
            self.train_dataset = self._to_torch_dataset(train_dc)
            self.val_dataset = self._to_torch_dataset(valid_dc)

        if stage == "test" or stage is None:
            self.test_dataset = self._to_torch_dataset(test_dc)

    def _to_torch_dataset(self, dc_dataset):
        """Convert DeepChem dataset to PyTorch TensorDataset."""
        X = torch.from_numpy(dc_dataset.X).float()
        y = torch.from_numpy(dc_dataset.y).float()
        return TensorDataset(X, y)

    def train_dataloader(self):
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class QM7DataModule(DeepChemDataModule):
    """DataModule for QM7 dataset."""

    def __init__(self, **kwargs):
        super().__init__(dataset_name="qm7", **kwargs)


class QM8DataModule(DeepChemDataModule):
    """DataModule for QM8 dataset."""

    def __init__(self, **kwargs):
        super().__init__(dataset_name="qm8", **kwargs)


class QM9DataModule(DeepChemDataModule):
    """DataModule for QM9 dataset."""

    def __init__(self, **kwargs):
        super().__init__(dataset_name="qm9", **kwargs)


class PDBbindDataModule(DeepChemDataModule):
    """DataModule for PDBbind dataset."""

    def __init__(self, **kwargs):
        super().__init__(dataset_name="pdbbind", **kwargs)
