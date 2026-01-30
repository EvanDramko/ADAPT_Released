# imports
import torch
import torch.nn as nn

#=========================================================================================================================================
#============================================================= NN structure ==============================================================
#=========================================================================================================================================
class ResidualLinear(nn.Module):
    """
    One residual block: y = norm( drop(act(Wx + b)) + skip(x) )
    - If in_dim == out_dim, skip(x) = x
    - Else, skip(x) = P x with a learned linear projection P
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation=nn.ReLU,
        use_proj_skip: bool = True,
        use_norm: bool = True,
        dropout: float = 0.1,          # NEW: dropout prob on the main path
    ):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.act = activation()
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()
        self.skip = (
            nn.Identity() if in_dim == out_dim
            else (nn.Linear(in_dim, out_dim) if use_proj_skip else None)
        )
        self.norm = nn.LayerNorm(out_dim) if use_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc(x)
        y = self.act(y)
        y = self.drop(y)  # dropout BEFORE residual add, AFTER activation
        if self.skip is None:
            out = y
        else:
            out = y + self.skip(x)
        return self.norm(out)


class nrgMLP(nn.Module):
    def __init__(
        self,
        layer_sizes,
        activation=nn.ReLU,
        use_proj_skip: bool = True,
        use_norm: bool = True,
        dropout: float = 0.1,
        input_dropout: float = 0.0,
    ):
        """
        layer_sizes: [input_dim, h1, h2, ..., output_dim]
        - Residual connections on all hidden transitions.
        - Final layer is linear only (no residual, no activation, no dropout).
        """
        super().__init__()
        layers = [nn.Flatten()]
        if input_dropout and input_dropout > 0.0:
            layers.append(nn.Dropout(input_dropout))

        pairs = list(zip(layer_sizes[:-1], layer_sizes[1:]))
        for i, (in_dim, out_dim) in enumerate(pairs):
            is_last = (i == len(pairs) - 1)
            if is_last:
                layers.append(nn.Linear(in_dim, out_dim))
            else:
                layers.append(
                    ResidualLinear(
                        in_dim, out_dim,
                        activation=activation,
                        use_proj_skip=use_proj_skip,
                        use_norm=use_norm,
                        dropout=dropout,           # pass through
                    )
                )

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


#=========================================================================================================================================
#=========================================================== End NN structure ============================================================
#=========================================================================================================================================

#=========================================================================================================================================
#============================================================ Training Loop ==============================================================
#=========================================================================================================================================

# create train function
def train(dataloader, model, loss_fn, optimizer):
    device = next(model.parameters()).device
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    totalLoss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error        
        pred = model(X)
        loss = loss_fn(pred.squeeze(-1), y)
        totalLoss += loss.item()

        # Backpropagation
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return (totalLoss/num_batches) # store batch mean loss for training loss


@torch.inference_mode()
def test(dataloader, model, loss_fn):
    device = next(model.parameters()).device
    num_batches = len(dataloader)
    test_loss = 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred.squeeze(-1), y)
            test_loss += loss.item()

    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f}")
    return(test_loss)

#=========================================================================================================================================
#========================================================== End Training Loop ============================================================
#=========================================================================================================================================
