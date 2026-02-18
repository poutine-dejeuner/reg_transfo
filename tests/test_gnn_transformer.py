import torch
from torch_geometric.data import Batch, Data

from reg_transfo.algorithms.molecule_gnn_transformer import MoleculeGNNTransformer


def test_molecule_gnn_transformer_forward():
    model = MoleculeGNNTransformer()

    # Create fake batch
    # 2 molecules
    # Mol 1: 3 atoms
    # Mol 2: 2 atoms

    # Pos (N, 3)
    pos1 = torch.rand(3, 3)
    pos2 = torch.rand(2, 3)

    # X (one-hot) (N, 5)
    x1 = torch.tensor([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]], dtype=torch.float)
    x2 = torch.tensor([[0, 0, 0, 1, 0], [0, 0, 0, 0, 1]], dtype=torch.float)

    y1 = torch.tensor([1.0])
    y2 = torch.tensor([2.0])

    d1 = Data(x=x1, pos=pos1, y=y1)
    d2 = Data(x=x2, pos=pos2, y=y2)

    batch_graph = Batch.from_data_list([d1, d2])

    # Images (B, 32, 32)
    batch_images = torch.rand(2, 32, 32)

    # Forward
    out = model(batch_graph, batch_images)

    print(f"Output shape: {out.shape}")
    assert out.shape == (2,)

    # Check if features were captured
    assert 'last_interaction' in model.schnet_features
    features = model.schnet_features['last_interaction']
    print(f"Captured features shape: {features.shape}")
    # Should be (Total atoms, hidden_channels) -> (5, 128)
    assert features.shape == (5, 128)

if __name__ == "__main__":
    test_molecule_gnn_transformer_forward()
