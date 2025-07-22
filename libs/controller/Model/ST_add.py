import os
import time
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import pynvml 
from typing import List, Union 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
nodes_index_matrix = None
OVERRIDE_SEC = 290              
CONGEST_CITES = "cites_26.cites"   
CONGEST_NODES = [0, 1]               
_current_sec = None                

def _load_allowed_mask(cites_file: str, N: int):
    """
    Loads and processes the cites file to create an allowed_mask.
    The mask indicates allowed connections for traffic transfer.
    """
    df = pd.read_csv(cites_file, header=None, sep=" ")
    df = df[df[1] != df[0]]
    # Ensure the indices are 0-based for direct use with tensor indexing later
    # And filter to only include nodes relevant to the current N
    df = df[(df[0] <= N) & (df[1] <= N)]

    # Create a pivot table to get distances, fill missing with a large value
    dist = df.pivot(index=0, columns=1, values=2).fillna(9999)

    # Reindex dist to cover all N nodes from 1 to N, filling missing with 9999
    # This ensures consistent size for argmin operation even if some nodes are isolated in the file
    all_nodes_index = pd.RangeIndex(start=1, stop=N + 1)
    dist = dist.reindex(index=all_nodes_index, columns=all_nodes_index, fill_value=9999)

    # For each row, get the indices of the two smallest distances
    # Handle cases where a row might have all 9999s (e.g., node has no connections in the filtered df)
    # If a row is all 9999s, argsort will return [0,1] for those, but we filter later
    nearest = dist.apply(lambda row: np.argsort(row.values)[:2] if row.min() != 9999 else [-1, -1], axis=1) # Use -1 as placeholder for no valid connections

    mask = torch.zeros((N, N), dtype=torch.bool, device=device)
    for idx in nearest.index:
        i = idx - 1 # Convert to 0-based index (node ID in cites file is 1-based)
        for j_offset in nearest.loc[idx]:
            if j_offset == -1: # Skip if no valid connections found for this node
                continue
            # Convert offset back to 0-based column index relative to the sorted values
            j = dist.columns[j_offset] - 1 # This converts sorted index to original column node ID, then to 0-based index
            if 0 <= i < N and 0 <= j < N: # Ensure within bounds of current N
                mask[i, j] = True
    return mask

def _orig_iterative_transfer_projection(base_scores, available, deficit, allowed_mask, iterations=10):
    """
    Performs iterative projection to calculate traffic transfers based on scores,
    available capacity, deficit, and allowed connections.
    """
    # Mask out disallowed connections by setting their scores very low
    masked = base_scores.masked_fill(~allowed_mask, -1e9)
    # Convert scores to probabilities (distribution)
    dist = torch.softmax(masked, dim=2)
    # Initial transfer proportional to available capacity and distribution
    T = available.unsqueeze(2) * dist

    for _ in range(iterations):
        # Adjust transfers based on column sums (deficit at destination)
        col_sum = T.sum(dim=1)
        col_scale = deficit / (col_sum + 1e-6) # Add epsilon for numerical stability
        col_scale = torch.clamp(col_scale, max=1.0) # Ensure scale doesn't increase transfers
        T = T * col_scale.unsqueeze(1) # Apply column-wise scaling

        # Adjust transfers based on row sums (available at source)
        row_sum = T.sum(dim=2)
        row_scale = available / (row_sum + 1e-6)
        row_scale = torch.clamp(row_scale, max=1.0) # Ensure scale doesn't increase transfers
        T = T * row_scale.unsqueeze(2) # Apply row-wise scaling
    return T

def iterative_transfer_projection(base_scores, available, deficit, allowed_mask, iterations=10):
    """
    Monkey-patched version of the projection function to introduce congestion
    at a specific time point (OVERRIDE_SEC).
    """
    global _current_sec
    # If current second matches OVERRIDE_SEC, modify the allowed_mask
    if _current_sec == OVERRIDE_SEC:
        N = allowed_mask.shape[-1] # Get the current number of nodes
        new_mask = _load_allowed_mask(CONGEST_CITES, N) # Load base mask for current N
        # Apply congestion: set all incoming and outgoing connections for CONGEST_NODES to False
        for node_idx_to_congest in CONGEST_NODES: # CONGEST_NODES are 0-based indices
            if 0 <= node_idx_to_congest < N: # Ensure node index is within current N
                new_mask[:, node_idx_to_congest] = False # Block incoming to congested node
                new_mask[node_idx_to_congest, :] = False # Block outgoing from congested node
        allowed_mask = new_mask # Use the modified mask
    # Call the original projection logic with the potentially modified mask
    return _orig_iterative_transfer_projection(base_scores, available, deficit, allowed_mask, iterations)

def load_data(cites_file, data_file, fps: int = 10, random_noise: bool = False, label_file: str = None):
    """
    Loads traffic data and optional labels, and processes the cites file
    to create an initial adjacency matrix for all nodes found in cites_file.
    """
    global nodes_index_matrix
    nodes_df = pd.read_csv(cites_file, header=None, sep=" ")
    nodes_df = nodes_df[nodes_df[1] != nodes_df[0]] # Remove self-loops
    
    # Determine the maximum node ID to set the full matrix size
    max_node_id = max(nodes_df[0].max(), nodes_df[1].max())
    num_nodes_full = int(max_node_id) 

    # Create a full adjacency matrix based on the nearest two connections from cites_file
    mat = np.zeros((num_nodes_full, num_nodes_full), dtype=float)
    nodes_distance = pd.pivot_table(nodes_df, index=0, columns=1, values=2, fill_value=9999)
    # Ensure index and columns cover all nodes from 1 to max_node_id
    all_nodes_range = pd.RangeIndex(start=1, stop=max_node_id + 1)
    nodes_distance = nodes_distance.reindex(index=all_nodes_range, columns=all_nodes_range, fill_value=9999)

    # For each node, find its two nearest neighbors (based on distance)
    # If a node has no connections (all 9999s), use [-1, -1] as placeholders
    nodes_nearest_index = nodes_distance.apply(lambda x: np.argsort(x.values)[:2] if x.min() != 9999 else [-1, -1], axis=1)

    # Populate the adjacency matrix based on the nearest neighbors
    for idx in nodes_nearest_index.index:
        i = idx - 1 # Convert to 0-based
        for j_offset in nodes_nearest_index.loc[idx]:
            if j_offset == -1: # Skip if no valid connections
                continue
            j = nodes_distance.columns[j_offset] - 1 # Convert sorted index to original node ID, then to 0-based
            if 0 <= i < num_nodes_full and 0 <= j < num_nodes_full:
                mat[i, j] = 1 # Set connection to 1
    nodes_index_matrix = torch.from_numpy(mat).float().to(device) # Store as global tensor

    # Load traffic data from JSON
    with open(data_file, 'r') as f:
        raw = json.load(f)
    # Sort node keys numerically (e.g., 'node_1', 'node_10')
    node_keys = sorted(raw.keys(), key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    per_node = []
    for key in node_keys:
        arr = np.array(raw[key]['person'], dtype=float)
        total_secs = arr.shape[0] // fps
        if total_secs == 0:
            raise ValueError("less then 1s...")
        # Aggregate frames per second
        sec_vals = arr[fps-1 :: fps] # Take the last frame of each second
        per_node.append(sec_vals)
    data_array = np.stack(per_node, axis=1) # Shape: (total_seconds, num_nodes)
    data_seconds = torch.from_numpy(data_array.astype(np.float32)).to(device)
    
    if random_noise:
        data_seconds += torch.randn_like(data_seconds) * 1e-3 # Add small noise if specified
    
    # Load labels if provided
    if label_file:
        df = pd.read_csv(label_file)
        lbl = df.values[:, 1:].astype(np.float32) # Assuming first column is ID, rest are labels
        labels = torch.from_numpy(lbl).to(device)
    else:
        labels = None
    return data_seconds, labels

class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to input embeddings to inject sequence order information.
    """
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        # Register as buffer, meaning it's part of the state_dict but not trainable
        self.register_buffer('pe', pe.unsqueeze(0)) 
    def forward(self, x):
        return x + self.pe[:, :x.size(1)] # Add positional encoding to input

class SimpleSTBlock(nn.Module):
    """
    A simplified Spatio-Temporal Attention block.
    Combines temporal and spatial attention with a gating mechanism.
    """
    def __init__(self, embed_dim, heads):
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        self.spatial_attn = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        self.gate = nn.Linear(embed_dim * 2, embed_dim) # Gate to weigh temporal vs spatial
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # Apply temporal attention (across time steps for each node)
        t_out, _ = self.temporal_attn(x, x, x)
        # Apply spatial attention (across nodes for each time step)
        s_out, _ = self.spatial_attn(x, x, x)
        
        # Gated fusion of temporal and spatial outputs
        fused = torch.cat([t_out, s_out], dim=-1)
        gate_v = torch.sigmoid(self.gate(fused))
        fusion = gate_v * t_out + (1 - gate_v) * s_out
        
        # Residual connection and layer normalization
        x2 = self.norm1(x + fusion)
        # Feed-forward network with residual connection and layer normalization
        return self.norm2(x2 + self.ffn(x2))

class ResidualBlock(nn.Module):
    """
    A simple Residual Block for MLP networks.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_features, out_features)
        # Shortcut connection: if input/output features differ, use a linear layer
        self.shortcut = nn.Identity()
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
            
    def forward(self, x):
        identity = self.shortcut(x) # Store identity for residual connection
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out + identity # Add identity back

class Generator(nn.Module):
    """
    The main Generator model for predicting traffic load and computing transfer schedules.
    """
    def __init__(self, num_nodes=16, embed_dim=32, threshold=10, num_layers=8, heads=8, forecast_horizon=8, history_steps=32):
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.threshold = threshold
        self.forecast_horizon = forecast_horizon
        self.history_steps = history_steps

        # Input projection layer
        self.input_proj = nn.Linear(1, embed_dim) # Projects scalar input to embedding dim
        # Positional encoding
        self.pos_enc = PositionalEncoding(embed_dim)
        # Encoder: stack of Spatio-Temporal Attention Blocks
        self.encoder = nn.Sequential(*[SimpleSTBlock(embed_dim, heads) for _ in range(num_layers)])
        
        # Decoder linear layers for main and auxiliary predictions
        self.dec_main = nn.Linear(embed_dim, forecast_horizon)
        self.dec_aux = nn.Linear(embed_dim, forecast_horizon)
        # Learnable scaling and bias for decoder outputs
        self.out_scale = nn.Parameter(torch.tensor(10.0))
        self.out_bias = nn.Parameter(torch.tensor(10.0))

        # Spatial Embedding: Learnable embedding for each node
        # Initialize as zeros to easily distinguish pre-trained vs new nodes
        self.spatial_emb = nn.Parameter(torch.zeros(num_nodes, embed_dim)) 
        
        # Struct Embedding (phi_graph): Learns an embedding from degree (structure)
        self.struct_emb = nn.Linear(1, embed_dim) 
        
        # Temp FC (Workload): Learns features from node loads
        self.temp_fc = nn.Sequential(nn.Linear(num_nodes, embed_dim), nn.GELU()) 

        # MLP for computing transfer scores (offloading-related MLP)
        # Input dimension: num_nodes * (1 (for pred) + embed_dim (for combined embeddings))
        mlp_in_dim = num_nodes * (1 + embed_dim) 
        self.net = nn.Sequential(
            ResidualBlock(mlp_in_dim, 128),
            ResidualBlock(128, 256),
            nn.Linear(256, num_nodes * num_nodes) # Output scores for N x N transfers
        )

    def predict_load(self, hist: torch.Tensor):
        """
        Predicts future traffic load based on historical data.
        """
        B, P, N = hist.shape # Batch, History_steps, Num_nodes
        if N != self.num_nodes:
            raise ValueError(f"Input history nodes ({N}) mismatch with model's num_nodes ({self.num_nodes})")
        
        # Prepare input for encoder
        x = hist.unsqueeze(-1) # Add feature dimension (1)
        x = self.input_proj(x) # Project to embedding dimension
        x = x.permute(0, 2, 1, 3).reshape(B*N, P, self.embed_dim) # Reshape for batching nodes
        x = self.pos_enc(x) # Add positional encoding
        
        x = self.encoder(x) # Pass through ST encoder
        x_enc = x[:, -1, :] # Take the last hidden state from the encoder (representing current state)

        # Decode to forecast horizon for main and auxiliary predictions
        m = self.dec_main(x_enc) * self.out_scale + self.out_bias
        a = self.dec_aux(x_enc) * self.out_scale + self.out_bias

        # Reshape outputs to (Batch, Forecast_horizon, Num_nodes)
        m = m.view(B, N, self.forecast_horizon).transpose(1, 2)
        a = a.view(B, N, self.forecast_horizon).transpose(1, 2)
        return m, a

    def compute_schedule(self, pred, orig_cap, adj_matrix):
        """
        Computes traffic transfer schedules based on predicted load and network state.
        """
        # Calculate available capacity (above threshold) and deficit (below threshold)
        available = torch.clamp(orig_cap - self.threshold, min=0)
        deficit = torch.clamp(self.threshold - orig_cap, min=0)
        B, N = pred.shape # Batch, Num_nodes

        # Generate temporal features (Workload related) from predicted load
        tf = self.temp_fc(pred).unsqueeze(1).expand(-1, N, -1)
        
        # Calculate node degrees from adjacency matrix for structural features
        if adj_matrix.dim() == 2: # Single adjacency matrix (N,N)
            deg = adj_matrix.sum(1, keepdim=True) # Sum rows for out-degree (or sum cols for in-degree, depends on interpretation)
        else: # Batch of adjacency matrices (B, N, N)
            deg = adj_matrix.sum(2, keepdim=True) # Sum along node dimension (dim 2)
        
        # Generate structural features (phi_graph)
        sf = self.struct_emb(deg).unsqueeze(0).expand(B, -1, -1) # Expand to match batch size
        if sf.shape[1] != N: # Adjust sf's dimensions if deg was (N,1) and sf became (B,1,embed_dim)
            sf = sf.permute(0,2,1) # To (B, embed_dim, N)
            sf = sf.unsqueeze(1).expand(-1, N, -1, -1).squeeze(2) # To (B, N, embed_dim)

        # Combine spatial, structural, and temporal embeddings
        se = (self.spatial_emb.unsqueeze(0) + sf) + tf

        # Concatenate predicted load with combined embeddings and flatten for MLP input
        inp = torch.cat([pred.unsqueeze(2), se], dim=2).view(B, -1)
        
        # Compute raw transfer scores using the MLP
        scores = self.net(inp).view(B, N, N) # Reshape to N x N transfer matrix scores

        # Create a mask for allowed transfers:
        # 1. Source node must have capacity above threshold (`send`)
        # 2. Destination node must have deficit below threshold (`recv`)
        # 3. Connection must exist in the adjacency matrix (`adj_matrix.bool()`)
        send = (orig_cap > self.threshold).unsqueeze(2) # (B, N, 1)
        recv = (orig_cap < self.threshold).unsqueeze(1) # (B, 1, N)
        
        if adj_matrix.dim() == 2: # (N,N)
            mask = send & recv & adj_matrix.unsqueeze(0).bool() # Add batch dimension to adj_matrix
        else: # (B, N, N)
            mask = send & recv & adj_matrix.bool()

        # Perform iterative projection to get integer transfers
        T = iterative_transfer_projection(scores, available, deficit, mask, iterations=10)
        T_int = torch.ceil(T) # Round up transfers to integer values
        
        # Calculate incoming and outgoing traffic based on transfers
        inc = T_int.sum(1) # Sum columns for incoming traffic to each node
        out = T_int.sum(2) # Sum rows for outgoing traffic from each node
        
        # Calculate new capacity after transfers
        new_cap = torch.clamp(orig_cap + inc - out, 0, self.threshold) # Clamp between 0 and threshold
        return T_int, new_cap

    def forward(self, hist, adj_matrix, schedule_from_aux=True, orig_cap=None):
        """
        Forward pass of the Generator model.
        """
        main, aux = self.predict_load(hist) # Predict main and auxiliary loads
        H = main.size(1) # Forecast horizon

        Ts, loads = [], []
        # Iterate through each forecast step to compute schedules
        for h in range(H):
            # Choose which prediction (aux or main) to use for scheduling
            p = aux[:, h, :] if schedule_from_aux else main[:, h, :]
            # Use provided original capacity or the predicted 'p' for the first step
            o = orig_cap if h == 0 and orig_cap is not None else p
            
            T_h, l_h = self.compute_schedule(p, o, adj_matrix) # Compute transfers and new loads
            Ts.append(T_h)
            loads.append(l_h)
        
        # Stack results over the forecast horizon
        return torch.stack(Ts, dim=1), torch.stack(loads, dim=1), main, aux

def discriminator_loss(orig_cap,T):
    """
    Calculates a loss to encourage new capacities to stay within bounds and be non-negative.
    """
    inc=T.sum(1) # Incoming traffic
    out=T.sum(2) # Outgoing traffic
    new_cap=orig_cap+inc-out # New capacity after transfers
    
    pen_ex=F.relu(new_cap-10).mean() # Penalty for exceeding threshold (10)
    pen_ng=F.relu(-new_cap).sum()    # Penalty for negative capacity
    rms=torch.sqrt((new_cap**2).mean()) # Root Mean Square of new capacity (to keep it non-zero but controlled)
    
    return pen_ex+pen_ng+rms

def train_extended(model: Generator,
                   data: torch.Tensor,
                   adj_matrix: torch.Tensor,
                   labels: torch.Tensor = None,
                   num_epochs: int = 1000,
                   aux_w: float = 1.0, # Weight for auxiliary loss
                   main_w: float = 2.0, # Weight for main prediction loss
                   cons_w: float = 0.5, # Weight for consistency loss between steps
                   sched_w: float = 1.0, # Weight for schedule prediction loss (against labels)
                   fine_tune_specific_parts: bool = False,
                   old_N: int = None): # Pass the previous N for loading old spatial_emb
    """
    Trains the Generator model with options for full training or specific fine-tuning.
    """
    model.train() # Set model to training mode
    T_len = data.size(0) # Total number of time steps in the data
    H = model.history_steps # History window size
    FH = model.forecast_horizon # Forecast horizon size
    
    # Check if there's enough data for at least one full history and forecast window
    if T_len - FH < H:
        print(f"[Warning] no enough data, Data length: {T_len}, History: {H}, Forecast: {FH}")
        return

    # --- OPTIMIZER INITIALIZATION CONTROL ---
    if fine_tune_specific_parts:
        print("[Info] Fine-tuning: Updating Spatial Embedding, Encoder, Decoder, and Offloading MLP.")
        
        trainable_params = []
        
        # 1. Spatial Embedding (model.spatial_emb)
        trainable_params.append(model.spatial_emb)
        
        # 2. Encoder (model.encoder - STAtten Blocks)
        trainable_params.extend(model.encoder.parameters())
        
        # 3. Decoder layers (dec_main, dec_aux, out_scale, out_bias) - NEWLY INCLUDED
        trainable_params.extend(model.dec_main.parameters())
        trainable_params.extend(model.dec_aux.parameters())
        trainable_params.append(model.out_scale)
        trainable_params.append(model.out_bias)

        # 4. Offloading MLP (model.net) - NEWLY INCLUDED
        trainable_params.extend(model.net.parameters())
        
        optimizer = optim.Adam(trainable_params, lr=1e-3)

        # Explicitly set requires_grad=False for all other modules to freeze them
        # (e.g., input_proj, struct_emb, temp_fc)
        for name, param in model.named_parameters():
            # Check if parameter is trainable (not already frozen by model init or manual copy)
            # and if it's NOT in our list of explicitly trainable parameters
            if param.requires_grad and all(param is not p for p in trainable_params):
                param.requires_grad = False
                # print(f"Frozen parameter: {name}") # Uncomment for debugging to see what's frozen
        
    else: # Full training
        print("[Info] Training ALL model parameters (initial training or N changed significantly).")
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        # Ensure all parameters have requires_grad=True when training everything
        for param in model.parameters():
            param.requires_grad = True
    # --- END OPTIMIZER INITIALIZATION CONTROL ---

    for epoch in range(1,num_epochs+1):
        # Randomly select a history window from the data
        idx = random.randint(H, T_len - FH) # Ensure enough future data for FH
        hist = data[idx-H:idx].unsqueeze(0) # History input: (1, H, N)
        target = data[idx:idx+FH].unsqueeze(0) # Target for prediction: (1, FH, N)

        # Predict main and auxiliary loads
        main_p, aux_p = model.predict_load(hist)
        
        # Calculate prediction losses
        loss_main = F.mse_loss(main_p, target)
        loss_aux = F.mse_loss(aux_p, target)
        
        # Compute transfer schedule and new loads.
        T_seq, loads, _, _ = model(hist, adj_matrix, schedule_from_aux=True, orig_cap=target[:,0,:])
        
        # Calculate discriminator loss based on the first step's transfers and target load
        loss_D = discriminator_loss(target[:,0,:], T_seq[:,0,:,:])
        
        # Calculate schedule prediction loss if labels are available
        loss_sched = F.mse_loss(loads, labels[idx:idx+FH].unsqueeze(0)) if labels is not None else torch.tensor(0.0,device=device)
        
        # Calculate consistency loss between consecutive main predictions
        cons = torch.tensor(0.0, device=device)
        if FH > 1:
            cons = sum(F.mse_loss(main_p[:,h,:], main_p[:,h-1,:]) for h in range(1,FH))/(FH-1)
        
        # Total loss combining all components with their weights
        total = loss_D + main_w*loss_main + aux_w*loss_aux + cons_w*cons + sched_w*loss_sched
        
        # Optimization step
        optimizer.zero_grad() # Clear previous gradients
        total.backward()      # Backpropagate to compute gradients
        optimizer.step()      # Update model parameters

        # Print progress
        if epoch == 1 or epoch % 500 == 0:
            print(f"[Ext Train] Epoch {epoch}/{num_epochs}, Loss_D={loss_D:.4f}, MSE_main={loss_main:.4f}, Loss_Aux={loss_aux:.4f}, Cons={cons:.4f}, Sched={loss_sched:.4f}, Total={total:.4f}")

def predict_and_aggregate(model_file, cites_file, data_file, num_nodes, adj_matrix, end_second):
    """
    Loads a trained model and predicts traffic transfers and aggregated percentages
    at a specific time point.
    """
    # Load full data for prediction
    data_sec, _ = load_data(cites_file, data_file) 
    
    # Adjust data_sec to match the current num_nodes (pad with zeros or trim)
    if num_nodes > data_sec.size(1):
        pad = torch.zeros(data_sec.size(0), num_nodes - data_sec.size(1), device=device)
        data_sec = torch.cat([data_sec, pad], dim=1)
    elif num_nodes < data_sec.size(1):
        data_sec = data_sec[:, :num_nodes]

    # Initialize and load the Generator model
    gen = Generator(num_nodes=num_nodes).to(device)
    gen.load_state_dict(torch.load(model_file, map_location=device))
    gen.eval() # Set model to evaluation mode (disables dropout, batchnorm updates, etc.)

    T_data = data_sec.size(0) # Total time steps in the data
    
    history_steps = gen.history_steps
    # Forecast horizon is not directly used for the single prediction point here,
    # but forecast_horizon impacts the output size of dec_main/aux.

    # Determine the start and end index for the history window
    # The history should end at `end_second - 1` to predict for `end_second` onwards.
    end_idx_for_history = min(end_second - 1, T_data - 1)
    start_idx_for_history = max(0, end_idx_for_history - history_steps + 1)

    # Prepare history data slice for prediction
    if (end_idx_for_history - start_idx_for_history + 1) < history_steps:
        print(f"[Warning] Not enough history data for prediction at second {end_second}. "
              f"Required: {history_steps}, Available: {end_idx_for_history - start_idx_for_history + 1}. Padding with zeros.")
        actual_history_len = end_idx_for_history - start_idx_for_history + 1
        hist_data_slice = data_sec[start_idx_for_history : end_idx_for_history + 1]
        
        if actual_history_len < history_steps:
            padding_needed = history_steps - actual_history_len
            zero_padding = torch.zeros(padding_needed, num_nodes, device=device)
            hist = torch.cat([zero_padding, hist_data_slice], dim=0).unsqueeze(0) # Pad at the beginning
        else: # Should not happen if condition is checked correctly, but for safety
            hist = hist_data_slice.unsqueeze(0)
    else:
        hist = data_sec[start_idx_for_history : end_idx_for_history + 1].unsqueeze(0) # Take the slice
    
    # Perform prediction without gradient calculation
    with torch.no_grad():
        # Predict main and aux for the forecast horizon starting from end_second
        main_p, aux_p = gen.predict_load(hist)
        
        # Use the first forecasted main prediction (at t=end_second) as the `orig_cap` for scheduling
        orig_cap_for_schedule = main_p[:, 0, :] # Batch 0, First Forecast Step, All Nodes
        
        # Call model.forward to get T_seq (transfers) and loads (new_capacities)
        T_seq, _, _, _ = gen(hist, adj_matrix, schedule_from_aux=False, orig_cap=orig_cap_for_schedule)
    
    # Aggregate transfer percentages
    agg, cnt = {}, {}
    
    # We are interested in transfers at the `end_second`, which corresponds to the
    # first forecast step (index 0) of `T_seq`.
    if T_seq.size(1) > 0: # Ensure T_seq has at least one forecast step
        step_at_end_second = T_seq[0, 0] # Batch 0, First Forecast Step (which is `end_second`)
        main_p_at_end_second = main_p[0, 0] # Predicted load at `end_second`

        for f_node_idx in range(step_at_end_second.size(0)): # Iterate through 'from' nodes
            orig_load = main_p_at_end_second[f_node_idx].item()
            if orig_load <= 0: # Only consider transfers from nodes with positive load
                continue
            for t_node_idx in range(step_at_end_second.size(1)): # Iterate through 'to' nodes
                transfer_amount = step_at_end_second[f_node_idx, t_node_idx].item()
                if transfer_amount <= 0: # Only consider positive transfers
                    continue
                
                # Calculate percentage of transfer relative to original load
                percentage = (transfer_amount / orig_load) * 100
                key = (f_node_idx, t_node_idx) # (from_node, to_node) tuple as key
                agg[key] = agg.get(key, 0) + percentage
                cnt[key] = cnt.get(key, 0) + 1
    
    # Return average percentage for each (from, to) pair that had transfers
    return {k: round(agg[k] / cnt[k], 2) for k in agg}


def dynamic_train_and_predict_AIcity(device_counts: List[int], change_times: List[int], end_seconds: Union[int, List[int]],
                                     cites_file: str, data_file: str, fps: int = 10, label_file: str = None):
    """
    Manages dynamic training and prediction based on changing device counts and specific time points.
    """
    global nodes_index_matrix, _current_sec
    # Convert single end_second to a list for consistent iteration
    secs = end_seconds if isinstance(end_seconds, list) else [end_seconds]
    
    # Load full dataset and precompute full adjacency matrix based on cites_file
    data_full, labels_full = load_data(cites_file, data_file, fps, False, label_file)
    full_adj = nodes_index_matrix.clone() # Use the global full adjacency matrix

    results = {} # Dictionary to store results for each queried second
    
    # Store the previous N for loading spatial_emb
    previous_N_map = {} # Maps N to the N that was used to save the last model

    for sec in secs:
        _current_sec = sec # Set global current second for monkey-patching
        
        # Determine the current number of nodes (N) based on change_times
        idx = max([i for i, t in enumerate(change_times) if sec >= t] + [0])
        N = device_counts[idx]
        print(f"\n=== t={sec}s: use {N} nodes ===")
        
        # Create an adjacency matrix trimmed to the current N nodes
        cur_adj = full_adj[:N, :N].clone().to(device)

        # Define the model file path for the current N
        path = f"AIcity_{N}nodes.pt"
        
        # Determine if training is needed:
        # 1. If the model file for this N doesn't exist.
        # 2. If it's the OVERRIDE_SEC, which forces a re-training/fine-tuning cycle.
        need_train = not os.path.exists(path) or sec == OVERRIDE_SEC

        current_old_N = None
        if path in previous_N_map:
            current_old_N = previous_N_map[path] # Get the N that was used to save this file

        if need_train:
            model = Generator(num_nodes=N).to(device) # Initialize a new model for current N

            should_fine_tune_specific_parts = False 
            if os.path.exists(path):
                print(f"[Info] Loading existing model for {N} nodes from {path} for potential finetuning.")
                checkpoint = torch.load(path, map_location=device)
                
                # --- Manual loading for spatial_emb and other flexible layers ---
                # This handles N changes gracefully, ensuring old values are kept.
                model_state_dict = model.state_dict()
                for key in checkpoint:
                    if key in model_state_dict:
                        if checkpoint[key].shape == model_state_dict[key].shape:
                            model_state_dict[key].copy_(checkpoint[key])
                        elif key == 'spatial_emb' and checkpoint[key].size(0) < model_state_dict[key].size(0):
                            # Copy existing embeddings for old nodes, new nodes remain zero-initialized
                            model_state_dict[key][:checkpoint[key].size(0)].copy_(checkpoint[key])
                            print(f"  Copied old spatial_emb for {checkpoint[key].size(0)} nodes. New nodes remain zero-initialized.")
                        # --- For MLP (self.net) and Decoder layers (dec_main, dec_aux) ---
                        # If N changed, the size of input/output for these layers might change.
                        # We specifically handle them here to allow strict=True load afterwards.
                        elif (key.startswith('net.') or key.startswith('dec_main.') or key.startswith('dec_aux.')) and checkpoint[key].dim() == 2:
                            # For Linear layers, if dimensions change, they will be re-initialized naturally
                            # by the new Generator() call, and we don't copy old values if sizes mismatch.
                            # So, no specific copy logic here beyond the default strict=True behavior,
                            # which means they'll get new random weights if sizes don't match.
                            # The 'fine_tune_specific_parts' will then update these re-initialized parts.
                            pass # Let them be handled by load_state_dict(strict=True) or re-init
                        else:
                            print(f"  [Warning] Skipping {key}: shape mismatch ({checkpoint[key].shape} vs {model_state_dict[key].shape}). Will be re-initialized by Generator init.")
                    else:
                        print(f"  [Warning] Skipping {key}: not in current model state_dict.")
                model.load_state_dict(model_state_dict, strict=True) # Now use strict=True after manual copy
                # --- End Manual loading ---

                if sec == OVERRIDE_SEC: 
                    should_fine_tune_specific_parts = True # Then we want to fine-tune specific parts
            else:
                print(f"[Info] Training a new model for {N} nodes as {path} does not exist.")
                should_fine_tune_specific_parts = False 
            
            # Use data and labels truncated/padded to current N nodes for training
            train_data = data_full[:, :N]
            train_labels = labels_full[:, :N] if labels_full is not None else None

            # Call the extended training function
            train_extended(model,
                           train_data,
                           cur_adj, # Pass the current N-nodes adjacency matrix
                           train_labels,
                           num_epochs=1000, # Number of epochs for training/fine-tuning
                           fine_tune_specific_parts=should_fine_tune_specific_parts,
                           old_N=current_old_N) # Pass the previous N

            # Save the trained/finetuned model state
            torch.save(model.state_dict(), path)
            previous_N_map[path] = N # Update map with the N used to save this model

        # Load the (just trained or existing) model for prediction and aggregate results
        decs = predict_and_aggregate(path, cites_file, data_file, N, cur_adj, sec)
        results[sec] = decs # Store aggregated results
    return results


if __name__ == "__main__":
    # Clean up previous model files for a fresh start during testing/debugging
    # This ensures models are re-trained if you make changes and want to see the effect.
    for N_val in [8, 12, 16]:
        model_path = f"AIcity_{N_val}nodes.pt"
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"Removed old model: {model_path}")

    # Define device counts and times when they change
    device_counts = [8, 12, 16] # Example: 8 nodes for 0-59s, 12 for 60-119s, 16 for 120s+
    change_times = [60, 120, 180] # Seconds at which device counts change

    # --- Test with a regular time point (180s, N=16) ---
    # This should trigger training a new 16-node model if none exists,
    # or load an existing one and not fine-tune specific parts (unless OVERRIDE_SEC).
    print("\n" + "="*50)
    print("--- Testing with a regular time point (180s) ---")
    print("="*50)
    res_regular = dynamic_train_and_predict_AIcity(
        device_counts, change_times, 180,
        "cites_26.cites", "AIcity.json", 10, "AIcity_balance.csv"
    )
    print("\nResults for 180s:")
    print(res_regular)

    # --- Test with OVERRIDE_SEC (290s, N=16) ---
    # This will trigger the specific fine-tuning (updating spatial_emb, STAtten, decoders, MLP)
    # if a model for N=16 already exists. If not, it will train a full new model.
    print("\n" + "="*50)
    print(f"--- Testing with OVERRIDE_SEC ({OVERRIDE_SEC}s) to trigger specific fine-tuning/congestion ---")
    print("="*50)
    res_override = dynamic_train_and_predict_AIcity(
        device_counts, change_times, OVERRIDE_SEC,
        "cites_26.cites", "AIcity.json", 10, "AIcity_balance.csv"
    )
    print(f"\nResults for {OVERRIDE_SEC}s:")
    print(res_override)

    print("\n--- Simulation Complete ---")
