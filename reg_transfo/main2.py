import lightning.pytorch as pl
import yaml
from lightning.pytorch.loggers import WandbLogger

from reg_transfo.algorithms.molecule_gnn_transformer import MoleculeGNNTransformer
from reg_transfo.datamodules.molecule_data import MOL_DIR, CREMPDataModule
from reg_transfo.utils.mem_utils import print_peak_memory

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

wandb_logger = WandbLogger(project="molecule-topo-transfo")
dm = CREMPDataModule(mol_dir=MOL_DIR, batch_size=64, num_workers=7)
model = MoleculeGNNTransformer(**cfg['model'])
trainer = pl.Trainer(fast_dev_run=cfg['debug'], accelerator="auto", devices="auto",
                     max_epochs=10, logger=wandb_logger)
trainer.fit(model, datamodule=dm)

print_peak_memory()
