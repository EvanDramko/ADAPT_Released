# imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

#=========================================================================================================================================
#============================================================= NN structure ==============================================================
#=========================================================================================================================================

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate, num_heads):
        super(TransformerEncoderLayer, self).__init__()
        # Multi-head self-attention block
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout_rate, batch_first=True)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: Tensor typed input sequence of shape (batch, seq_length, d_model).
            src_mask: Tensor typed mask for the attention mechanism.
            src_key_padding_mask: Tensor typed padding mask. (True at real, False at padding)
        Returns:
            Tensor: Output of shape (batch, seq_length, d_model)
        """
        pad_mask = None
        assert src.dim() == 3
        if src_key_padding_mask is not None:
            assert src_key_padding_mask.shape[:2] == src.shape[:2]
            assert src_key_padding_mask.dtype == torch.bool
            pad_mask = ~src_key_padding_mask # attention wants true where padded, false elsewhere

        # Self-attention sub-layer
        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask,
                                          key_padding_mask=pad_mask)
        # Residual connection and normalization
        src2 = self.dropout1(attn_output)
        src = self.norm1(src + src2)

        # Feed-forward sub-layer
        ff_output = self.linear2((F.relu(self.linear1(src))))
        src2 = self.dropout2(ff_output)
        x = self.norm2(src + src2)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, num_layers, d_out, dropout_rate, num_heads, vecRepLength=12):
        super(TransformerEncoder, self).__init__()
        # create linear layers as "embedding" for the data (the same embedding is applied indep. to each token)
        self.lin1 = nn.Linear(vecRepLength, 128)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(128, d_model)
        self.relu2 = nn.ReLU()
        self.lin3 = nn.Linear(d_model, d_model)
        self.embed_norm = nn.LayerNorm(d_model)
        self.embed_drop = nn.Dropout(dropout_rate)
        self.native_token_dim = vecRepLength

        # Stack multiple encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, d_ff, dropout_rate, num_heads)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
        # add linear layer to project into target output space
        self.output_linear = nn.Linear(d_model, d_out)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, return2ndlast=False):
        """
        Args:
            src (Tensor): Input sequence (batch_size, seq_length, d_model).
            src_mask (Tensor, optional): Mask for attention.
            src_key_padding_mask (Tensor, optional): Padding mask.
        Returns:
            Tensor: Output sequence (batch_size, seq_length, d_out)
        """
        src = self.lin1(src)
        src = self.relu1(src)
        src = self.lin2(src)
        src = self.relu2(src)
        src = self.lin3(src)
        src = self.embed_norm(src)
        src = self.embed_drop(src)
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.norm(output)
        if(return2ndlast):
            outputLin = self.output_linear(output)
            return outputLin, output
        output = self.output_linear(output)
        return output


#=========================================================================================================================================
#=========================================================== End NN structure ============================================================
#=========================================================================================================================================

#=========================================================================================================================================
#============================================================ Training Loop ==============================================================
#=========================================================================================================================================

# create train function
def train(dataloader, model, loss_fn, optimizer, rotation=None, translation=None, include_weighting_pattern=True):
    start = time.time()
    device = next(model.parameters()).device
    model.train()
    running_loss = 0
    count = 0
    total_samples = len(dataloader.dataset)
    for batch_idx, (X, y, mask) in enumerate(dataloader):
        X = X.to(device, non_blocking=True)  # (B, n, 13) # (B, n, k-1) features, (B, n, 1) atom importance weighting pattern
        y = y.to(device, non_blocking=True)  # (B, n, 3)
        mask = mask.to(device, non_blocking=True) # (B, n)
        if include_weighting_pattern:
            Xdata = X[:, :, :-1]
            pattern = X[:, :, -1]  # pull the loss weighting pattern out of the model
        else:
            Xdata = X

        # check for dimension errors
        assert X.dim() == 3 and y.dim() == 3 and mask.dim() == 2
        B, n, Dx = Xdata.shape
        B, ny, Dy = y.shape
        assert n == ny and Dy == 3 and Dx == model.native_token_dim
        assert mask.shape == (B, n) and mask.dtype == torch.bool

        # perform rotation augmentation
        if (rotation != None):
            Xdata, y = rotation(Xdata, y)

        # perform translation augmentation
        if (translation != None):
            Xdata = translation(Xdata)

        # Compute prediction error 
        optimizer.zero_grad(set_to_none=True)
        pred = model(Xdata, src_key_padding_mask=mask)
        loss = loss_fn(pred, y, pattern) if include_weighting_pattern else loss_fn(pred, y)
        running_loss += loss.detach()
        count += 1

        # Backpropagation
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if (batch_idx) % 100 == 0:
            # single sync here
            avg = (running_loss / count).item()
            print(f"loss: {avg:.6f}  [{(batch_idx+1)*X.size(0)}]/{total_samples}")
            running_loss = 0.0
            count = 0
    
    end = time.time()
    print("Time taken for this epoch was: ", end-start)
    

# create test function
@torch.no_grad()
def test(dataloader, model, loss_fn, rotation=None, translation=None, include_weighting_pattern=True):
    device = next(model.parameters()).device
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    for batch_idx, (X, y, mask) in enumerate(dataloader):
        X = X.to(device)  # (B, n, 13) # (B, n, k-1) features, (B, n, 1) atom importance weighting pattern
        y = y.to(device)  # (B, n, 3)
        mask = mask.to(device) # (B, n)
        if include_weighting_pattern:
            Xdata = X[:, :, :-1]
            pattern = X[:, :, -1]  # pull the loss weighting pattern out of the model
        else:
            Xdata = X

        # perform rotation augmentation
        if (rotation != None):
            Xdata, y = rotation(Xdata, y)

        # perform translation augmentation
        if (translation != None):
            Xdata = translation(Xdata)

        # Compute prediction error
        pred = model(Xdata, src_key_padding_mask=mask)
        loss = loss_fn(pred, y, pattern) if include_weighting_pattern else loss_fn(pred, y)
        test_loss += loss.item()
                

    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f}")

#=========================================================================================================================================
#========================================================== End Training Loop ============================================================
#=========================================================================================================================================
