# imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#=========================================================================================================================================
#============================================================= NN structure ==============================================================
#=========================================================================================================================================

class ScalarDecoderHead(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout=0.1):
        super().__init__()
        # A single learnable query token (1, 1, d_model) -> expand to (B, 1, d_model)
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, 1)  # produce a single scalar

    def forward(self, memory, key_padding_mask=None):
        """
        memory: (B, N, K) encoder outputs
        key_padding_mask: (B, N) True for pads
        returns: (B, 1) scalar
        """
        if key_padding_mask is not None:
            assert key_padding_mask.dtype == torch.bool
            pad_mask = ~key_padding_mask # attention wants true where padded, false elsewhere
        else:
            pad_mask = None

        B = memory.size(0)
        q = self.query.expand(B, -1, -1)                    # (B, 1, K)
        attn_out, _ = self.cross_attn(q, memory, memory,
                                      key_padding_mask=pad_mask)
        x = self.norm1(q + self.drop1(attn_out))            # (B, 1, K)

        ff = self.ff2(F.relu(self.ff1(x)))
        x = self.norm2(x + self.drop2(ff))                  # (B, 1, K)

        scalar = self.out(x).squeeze(-1)                    # (B, 1)
        return scalar

class ResidualLinearMap(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation=F.relu,
    ):
        """
        One-hidden-layer MLP with residual connection:
            y = f2( act(f1(x)) ) + R(x)
        where R(x) is either identity (if dims match) or a learnable projection.

        Args:
            input_dim:  input feature size
            hidden_dim: hidden width
            output_dim: output feature size
            activation: callable nonlinearity (default: ReLU)
            residual_projection: if True and dims differ, use Linear(input_dim->output_dim) for skip
            bias: whether to use bias terms in the main linear layers
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=True)
        self.activation = activation

        # Residual path
        if input_dim == output_dim:
            self.residual = nn.Identity()
        else:
            # Match shapes to allow for adding
            self.residual = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.fc1(x))
        out = self.fc2(h)
        return out + self.residual(x)


#=========================================================================================================================================
#=========================================================== End NN structure ============================================================
#=========================================================================================================================================

#=========================================================================================================================================
#============================================================ Training Loop ==============================================================
#=========================================================================================================================================

# create train function
if __name__ == "__main__":
    # Setup
    # Set device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    # device = "cuda:0"
    print(f"Using {device} device")

    ValUpScale = ResidualLinearMap(15, 512, 256).to(device)
    model = ScalarDecoderHead(256, 512, 8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.MSELoss()

    # Training loop
    num_batches = 2000
    batch_size = 64
    input_shape = (217, 15)
    label_value = 0.077

    for batch_idx in range(num_batches):
        # Generate random noise input
        inputs = torch.randn(batch_size, *input_shape, device=device)
        
        # Generate labels (same for all in batch)
        labels = torch.full((batch_size, 1), label_value, device=device)
        
        # Forward
        inputs2 = ValUpScale(inputs)
        outputs = model(inputs2)
        loss = criterion(outputs, labels)
        
        # Backward + optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (batch_idx + 1) % 100 == 0:
            print(f"Batch {batch_idx+1}/{num_batches}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "static_7_decoder.pth")
    torch.save(ValUpScale.state_dict(), "static_7_upScale.pth")
#=========================================================================================================================================
#========================================================== End Training Loop ============================================================
#=========================================================================================================================================
