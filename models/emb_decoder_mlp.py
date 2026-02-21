import torch.nn as nn


def _resolve_activation(activation):
    if isinstance(activation, str):
        name = activation.strip()
        if "." in name:
            module_path, attr = name.rsplit(".", 1)
            try:
                module = __import__(module_path, fromlist=[attr])
                return getattr(module, attr)
            except (ImportError, AttributeError) as exc:
                raise ValueError(f"Unknown activation '{activation}'") from exc
        if not hasattr(nn, name):
            raise ValueError(f"Unknown activation '{activation}'")
        return getattr(nn, name)
    return activation


class EmbDecoderMLP(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim=None,
        num_layers=2,
        dropout=0.0,
        activation=nn.ReLU,
    ):
        super().__init__()
        hidden_dim = hidden_dim or 2 * out_dim
        activation = _resolve_activation(activation)

        layers = []
        prev_dim = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, out_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
