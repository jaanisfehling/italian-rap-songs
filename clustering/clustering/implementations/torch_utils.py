import torch
from torch.utils.data import Dataset


def build_block(
    layers: list,
    activation_fn: torch.nn.Module = torch.nn.LeakyReLU,
    output_fn: torch.nn.Module = torch.nn.LeakyReLU,
    bias: bool = True,
    batch_norm: bool = False,
    dropout: float = None,
):
    block_list = []
    for i in range(len(layers) - 1):
        block_list.append(torch.nn.Linear(layers[i], layers[i + 1], bias=bias))
        if batch_norm:
            block_list.append(torch.nn.BatchNorm1d(layers[i + 1]))
        if dropout is not None:
            block_list.append(torch.nn.Dropout(dropout))
        if activation_fn is not None:
            if i != len(layers) - 2:
                block_list.append(activation_fn())
            else:
                if output_fn is not None:
                    block_list.append(output_fn())
    return torch.nn.Sequential(*block_list)


def build_autoencoder(
    input_dim: int,
    output_dim: int,
    layer_per_block: int,
    hidden_dim: int = None,
    activation_fn: torch.nn.Module = torch.nn.LeakyReLU,
    output_fn: torch.nn.Module = torch.nn.LeakyReLU,
    bias: bool = True,
    batch_norm: bool = False,
    dropout: float = None,
):
    if not hidden_dim:
        hidden_dim = max(1, min(round(input_dim / 4), round(output_dim / 4)))

    encoder_layer_list = list(
        range(input_dim, hidden_dim - 1, min(-1, -round((input_dim - hidden_dim) / layer_per_block)))
    )
    encoder_layer_list[-1] = hidden_dim
    encoder = build_block(encoder_layer_list, activation_fn, output_fn, bias, batch_norm, dropout)

    decoder_layer_list = list(
        range(output_dim, hidden_dim - 1, min(-1, -round((output_dim - hidden_dim) / layer_per_block)))
    )
    decoder_layer_list[-1] = hidden_dim
    decoder = build_block(decoder_layer_list[::-1], activation_fn, output_fn, bias, batch_norm, dropout)

    return encoder, decoder


class PytorchMixedTypeDataset(Dataset):
    def __init__(self, df, cat_cols, num_cols):
        self.cat = torch.tensor(df[cat_cols].values, dtype=torch.int)
        self.cont = torch.tensor(df[num_cols].values, dtype=torch.float)

    def __getitem__(self, idx):
        return self.cat[idx], self.cont[idx]

    def __len__(self):
        return self.cat.shape[0]
