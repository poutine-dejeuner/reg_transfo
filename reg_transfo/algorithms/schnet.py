from collections.abc import Callable

import torch
from torch_geometric.nn import SchNet

from reg_transfo.algorithms.molecule_base import MoleculeRegressor


class MoleculeSchNet(MoleculeRegressor):
    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 3,
        num_gaussians: int = 50,
        datamodule=None,
        loss_fn: Callable | None = None,
        metrics: dict[str, Callable] | None = None,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay, loss_fn=loss_fn, metrics=metrics)
        self.save_hyperparameters(ignore=["datamodule", "loss_fn", "metrics"])

        self.atom_map = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16}
        self.schnet = SchNet(hidden_channels=hidden_channels, num_filters=num_filters,
                             num_interactions=num_interactions, num_gaussians=num_gaussians,
                             cutoff=10.0)

    def forward(self, batch_graph, batch_images=None):
        atom_indices = batch_graph.x.argmax(dim=1)
        z_values = torch.tensor([1, 6, 7, 8, 16], device=self.device)[atom_indices]

        # SchNet returns [batch_size, 1] by default if using standard readout
        # It performs global pooling internally
        out = self.schnet(z_values, batch_graph.pos, batch_graph.batch)

        return out.squeeze(-1)
