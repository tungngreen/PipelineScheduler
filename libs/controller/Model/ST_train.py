import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import argparse
import json

node, nodes_distance, nodes_nearest_index, data, nodes_index_matrix, labels = None, None, None, None, None, None
global_best_loss = 10000

def calculate_transfers(input_vector1, input_vector2):
    diff = input_vector2.flatten() - input_vector1.flatten()
    surplus = [(i, -diff[i].item()) for i in range(len(diff)) if diff[i].item() < 0]
    deficit = [(i, diff[i].item()) for i in range(len(diff)) if diff[i].item() > 0]

    transfers = []
    i, j = 0, 0

    while i < len(surplus) and j < len(deficit):
        from_node, surplus_amount = surplus[i]
        to_node, deficit_amount = deficit[j]
        transfer_amount = min(surplus_amount, deficit_amount)
        transfers.append((from_node, to_node, transfer_amount))
        surplus[i] = (from_node, surplus_amount - transfer_amount)
        deficit[j] = (to_node, deficit_amount - transfer_amount)
        if surplus[i][1] == 0:
            i += 1
        if deficit[j][1] == 0:
            j += 1

    while i < len(surplus):
        from_node, remaining = surplus[i]
        transfers.append((from_node, None, remaining))
        i += 1

    return transfers

def rule_based_method(tensor: torch.tensor) -> (torch.tensor, torch.tensor):
    balanced_tensor = tensor.clone()
    n = len(balanced_tensor)
    final_strategy = torch.zeros((n, n), dtype=torch.float32)
    while (balanced_tensor > 10).any() and (balanced_tensor < 10).any():
        overload_indices = torch.where(balanced_tensor > 10)[0]
        underload_indices = torch.where(balanced_tensor < 10)[0]
        i = overload_indices[torch.argmax(balanced_tensor[overload_indices] - 10)].item()
        j = underload_indices[torch.argmax(10 - balanced_tensor[underload_indices])].item()
        transfer_amount = min(balanced_tensor[i] - 10, 10 - balanced_tensor[j])
        balanced_tensor[i] -= transfer_amount
        balanced_tensor[j] += transfer_amount
        final_strategy[i, j] += transfer_amount
    return balanced_tensor, final_strategy

def get_final_transfers(input_vector1, final_vector, max_iter=10):
    iteration = 0
    transfers = calculate_transfers(input_vector1, final_vector)
    while any(t[1] is None for t in transfers) and iteration < max_iter:
        print("Unassigned detected, applying rule-based remedy...")
        final_vector, _ = rule_based_method(final_vector)
        transfers = calculate_transfers(input_vector1, final_vector)
        iteration += 1
    return transfers, final_vector

def load_data(cites_file, data_file, random_noise=False):
    global node, nodes_distance, nodes_nearest_index, data, labels, nodes_index_matrix
    
    nodes = pd.read_csv(cites_file, header=None, sep=" ")
    nodes = nodes[nodes[1] != nodes[0]]
    nodes_distance = pd.pivot(index=0, columns=1, values=2, data=nodes).fillna(9999)
    nodes_nearest_index = nodes_distance.apply(lambda x: np.argsort(x.values)[:2], axis=1)

    if data_file.endswith('.csv'):
        data = pd.read_csv(data_file).dropna().values[:, 1:].astype(np.float32)
    elif data_file.endswith('.json'):
        with open(data_file, 'r') as f:
            json_data = json.load(f)
        if 'workloads' in json_data:
            data = np.array(json_data['workloads'], dtype=np.float32)
        else:
            raise ValueError(f"JSON file '{data_file}' does not contain a 'workloads' key.")
    else:
        raise ValueError(f"Unsupported data file type: {data_file}. Only .csv and .json are supported.")

    data = torch.from_numpy(data)

    try:
        labels = pd.read_csv('workloads_rebalanced.csv').dropna().values[:, 1:].astype(np.float32)
        labels = torch.from_numpy(labels)
    except FileNotFoundError:
        print("Warning: workloads_rebalanced.csv not found. Supervised labels not used.")
        labels = None
    except Exception as e:
        print(f"Error loading workloads_rebalanced.csv: {e}. Supervised labels not used.")
        labels = None

    nodes_index_matrix = np.zeros((16, 16))
    for index in nodes_nearest_index.index:
        if (index - 1) < 16:
            nodes_index_matrix[index-1][nodes_nearest_index.loc[index]] = 1
    nodes_index_matrix = torch.from_numpy(nodes_index_matrix).float()
    return node, nodes_distance, nodes_nearest_index, data, labels, nodes_index_matrix

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SimpleSTBlock(nn.Module):
    def __init__(self, embed_dim, heads):
        super(SimpleSTBlock, self).__init__()
        self.spatial_attn = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        self.temporal_attn = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x2, _ = self.temporal_attn(x, x, x)
        x = self.norm1(x + x2)
        x2, _ = self.spatial_attn(x, x, x)
        x = self.norm2(x + x2)
        x2 = self.ffn(x)
        x = self.norm3(x + x2)
        return x

def iterative_transfer_projection(base_scores, available, deficit, allowed_mask, iterations=10):
    masked_scores = base_scores.masked_fill(~allowed_mask, -1e9)
    distribution = torch.softmax(masked_scores, dim=2)
    T = available.unsqueeze(2) * distribution
    for _ in range(iterations):
        col_sum = T.sum(dim=1)
        col_scale = deficit / (col_sum + 1e-6)
        col_scale = torch.clamp(col_scale, max=1.0)
        T = T * col_scale.unsqueeze(1)
        row_sum = T.sum(dim=2)
        row_scale = available / (row_sum + 1e-6)
        row_scale = torch.clamp(row_scale, max=1.0)
        T = T * row_scale.unsqueeze(2)
    return T

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_features, out_features)
        
        self.shortcut = nn.Identity()
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out + identity

class Generator(nn.Module):
    def __init__(self,
                 num_nodes: int = 16,
                 embed_dim: int = 32,
                 threshold: float = 10,
                 num_layers: int = 3,
                 heads: int = 4,
                 forecast_horizon: int = 1,
                 history_steps: int = 16):
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.threshold = threshold
        self.forecast_horizon = forecast_horizon
        self.history_steps = history_steps

        self.input_proj = nn.Linear(1, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim)
        self.encoder = nn.Sequential(*[SimpleSTBlock(embed_dim, heads)
                                       for _ in range(num_layers)])
        self.dec_main = nn.Linear(embed_dim, forecast_horizon)
        self.dec_aux  = nn.Linear(embed_dim, forecast_horizon)
        self.out_scale = nn.Parameter(torch.tensor(10.0))
        self.out_bias  = nn.Parameter(torch.tensor(10.0))

        self.spatial_emb = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.struct_emb  = nn.Linear(1, embed_dim)
        self.temp_fc     = nn.Sequential(nn.Linear(num_nodes, embed_dim), nn.GELU())

        mlp_in_dim = num_nodes * (1 + embed_dim)
        
        self.net = nn.Sequential(
            ResidualBlock(mlp_in_dim, 128),
            ResidualBlock(128, 256),
            nn.Linear(256, num_nodes * num_nodes)
        )
        self.latest_prediction = None
        self.adjusted_load = None

    def predict_load(self, hist: torch.Tensor):
        B, P, N = hist.shape
        if N != self.num_nodes:
            raise ValueError(f"Input history nodes ({N}) mismatch with model's num_nodes ({self.num_nodes})")
        
        if P != self.history_steps:
             raise ValueError(f"Input history steps ({P}) mismatch with model's history_steps ({self.history_steps})")

        x = hist.unsqueeze(-1)
        x = self.input_proj(x)
        
        x = x.permute(0, 2, 1, 3).reshape(B * N, P, self.embed_dim)
        
        x = self.pos_enc(x)
        x = self.encoder(x)
        
        x_enc = x[:, -1, :]
        
        m = self.dec_main(x_enc) * self.out_scale + self.out_bias
        a = self.dec_aux(x_enc) * self.out_scale + self.out_bias
        
        m = m.view(B, N, self.forecast_horizon).transpose(1, 2)
        a = a.view(B, N, self.forecast_horizon).transpose(1, 2)
        
        return m, a

    def compute_schedule(self, pred, orig_cap, adj_matrix):
        available = torch.clamp(orig_cap - self.threshold, min=0)
        deficit   = torch.clamp(self.threshold - orig_cap, min=0)
        B, N = pred.shape

        if N != self.num_nodes:
            raise ValueError(f"Prediction nodes ({N}) mismatch with model's num_nodes ({self.num_nodes})")
        
        current_adj_matrix = adj_matrix.unsqueeze(0) if adj_matrix.dim() == 2 else adj_matrix

        if current_adj_matrix.size(0) != B:
            if current_adj_matrix.size(0) == 1:
                current_adj_matrix = current_adj_matrix.expand(B, -1, -1)
            else:
                raise ValueError(f"Adjacency matrix batch size ({current_adj_matrix.size(0)}) mismatch with current prediction batch size ({B})")

        if current_adj_matrix.shape[1] != N or current_adj_matrix.shape[2] != N:
            raise ValueError(f"Adjacency matrix shape {current_adj_matrix.shape} mismatch with current num_nodes ({N})")
        
        tf = self.temp_fc(pred).unsqueeze(1).expand(-1, N, -1)
        
        deg = current_adj_matrix.sum(2, keepdim=True)
        sf = self.struct_emb(deg).expand(-1, -1, self.embed_dim)
        
        se = (self.spatial_emb.unsqueeze(0).expand(B, -1, -1) + sf) + tf
        inp = torch.cat([pred.unsqueeze(2), se], dim=2).view(B, -1)
        scores = self.net(inp).view(B, N, N)

        send = (orig_cap > self.threshold).unsqueeze(2)
        recv = (orig_cap < self.threshold).unsqueeze(1)
        
        mask = send & recv & current_adj_matrix.bool()

        T = iterative_transfer_projection(scores, available, deficit, mask, iterations=10)
        T_int = torch.ceil(T) 

        inc = T_int.sum(1)
        out = T_int.sum(2)
        
        new_cap = orig_cap + inc - out
        new_cap = torch.clamp(new_cap, 0, self.threshold)
        
        return T_int, new_cap

    def forward(self, hist, adj_matrix, schedule_from_aux=True, orig_cap=None):
        main_preds, aux_preds = self.predict_load(hist)
        
        self.latest_prediction = main_preds[:, 0, :].detach()

        all_transfers = []
        all_adjusted_loads = []

        current_orig_capacity = orig_cap

        for h in range(self.forecast_horizon):
            pred_h = aux_preds[:, h, :] if schedule_from_aux else main_preds[:, h, :]
            T_h, l_h = self.compute_schedule(pred_h, current_orig_capacity, adj_matrix)
            
            all_transfers.append(T_h)
            all_adjusted_loads.append(l_h)
            
            current_orig_capacity = l_h

        self.adjusted_load = all_adjusted_loads[0].detach()

        return torch.stack(all_transfers, dim=1), torch.stack(all_adjusted_loads, dim=1), main_preds, aux_preds

def discriminator_loss(original_capacity, scheduling_strategy):
    first_step_strategy = scheduling_strategy[:, 0, :, :]
    
    incoming = first_step_strategy.sum(dim=1)
    outgoing = first_step_strategy.sum(dim=2)
    
    new_capacity = original_capacity + incoming - outgoing
    
    capacity_excess = torch.relu(new_capacity - 10)
    capacity_penalty = capacity_excess.mean()
    negative_capacity = torch.relu(-new_capacity)
    negative_penalty = negative_capacity.sum()
    rms = torch.sqrt(torch.mean(new_capacity ** 2))
    loss = capacity_penalty + negative_penalty + rms
    return loss, capacity_penalty, negative_penalty, rms

def train(args):
    generator = Generator(num_nodes=16, embed_dim=args.embed_dim, 
                          history_steps=args.history_steps, forecast_horizon=args.forecast_horizon)
    if args.load_model:
        try:
            generator.load_state_dict(torch.load(args.load_model, weights_only=True))
            print(f"Loaded model from {args.load_model}")
        except FileNotFoundError:
            print(f"Warning: Model file {args.load_model} not found. Starting training from scratch.")
    
    optimizer_G = optim.SGD(generator.parameters(), lr=args.lr)
    global global_best_loss, data, labels, nodes_index_matrix
    
    _, _, _, data, labels, nodes_index_matrix = load_data(args.cites_file, args.data_file)

    min_data_points = args.history_steps + args.forecast_horizon
    if len(data) < min_data_points:
        raise ValueError(f"Not enough data points ({len(data)}) for training with history length {args.history_steps} and forecast horizon {args.forecast_horizon}. Need at least {min_data_points} points.")

    for epoch in range(args.epochs):
        possible_start_indices = list(range(args.history_steps, len(data) - args.forecast_horizon))
        
        np.random.shuffle(possible_start_indices)

        for i in range(0, len(possible_start_indices), args.batch_size):
            batch_indices = possible_start_indices[i:i + args.batch_size]
            if not batch_indices:
                continue

            hist_x_batch = []
            original_capacity_batch = []
            target_main_load_batch = []
            target_aux_load_batch = []
            target_final_load_batch = []

            for start_idx in batch_indices:
                hist_x_batch.append(data[start_idx - args.history_steps:start_idx].unsqueeze(0))
                original_capacity_batch.append(data[start_idx].unsqueeze(0))
                
                target_main_load_batch.append(data[start_idx + 1 : start_idx + 1 + args.forecast_horizon].unsqueeze(0))
                target_aux_load_batch.append(data[start_idx + 1 : start_idx + 1 + args.forecast_horizon].unsqueeze(0))

                if labels is not None:
                    target_final_load_batch.append(labels[start_idx].unsqueeze(0))

            hist_x = torch.cat(hist_x_batch, dim=0)
            original_capacity = torch.cat(original_capacity_batch, dim=0)
            
            target_main_load = torch.cat(target_main_load_batch, dim=0)
            target_aux_load = torch.cat(target_aux_load_batch, dim=0)

            scheduling_strategy, adjusted_loads, main_preds, aux_preds = \
                generator(hist_x, nodes_index_matrix, schedule_from_aux=True, orig_cap=original_capacity)

            loss_D, capacity_penalty, negative_penalty, _ = discriminator_loss(original_capacity, scheduling_strategy)
            
            loss_pred_main = F.mse_loss(main_preds, target_main_load)
            loss_pred_aux = F.mse_loss(aux_preds, target_aux_load)
            
            loss_sup_adj_load = torch.tensor(0.0)
            if labels is not None and target_final_load_batch:
                target_final_load = torch.cat(target_final_load_batch, dim=0)
                loss_sup_adj_load = F.mse_loss(adjusted_loads[:, 0, :], target_final_load)

            total_loss = loss_D + loss_pred_main + loss_pred_aux + 2.0 * loss_sup_adj_load

            optimizer_G.zero_grad()
            total_loss.backward()
            optimizer_G.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{args.epochs}, Total Loss: {total_loss.item():.4f}, Unsupervised: {loss_D.item():.4f}, Pred Main: {loss_pred_main.item():.4f}, Pred Aux: {loss_pred_aux.item():.4f}, Adj Load Sup: {loss_sup_adj_load.item():.4f}, Best: {global_best_loss:.4f}')

        if total_loss.item() < global_best_loss:
            global_best_loss = total_loss.item()
            torch.save(generator.state_dict(), args.output_model)
            print(f"Model saved to {args.output_model} with new best loss: {global_best_loss:.4f}")

def predict(args):
    generator = Generator(num_nodes=16, embed_dim=args.embed_dim, 
                          history_steps=args.history_steps, forecast_horizon=args.forecast_horizon)
    try:
        generator.load_state_dict(torch.load(args.output_model, weights_only=True))
    except FileNotFoundError:
        print(f"Error: Model file '{args.output_model}' not found. Please train the model first.")
        sys.exit(1)
    generator.eval()

    global data, labels, nodes_index_matrix
    _, _, _, data, labels, nodes_index_matrix = load_data(args.cites_file, args.data_file)

    min_data_points = args.history_steps + args.forecast_horizon 
    if len(data) < min_data_points:
        print(f"Error: Not enough data for prediction. Need at least {min_data_points} data points (history + forecast horizon).")
        sys.exit(1)
        
    start_index = torch.randint(args.history_steps, len(data) - args.forecast_horizon, size=(1,))
    idx = start_index.item()
    print("Current index for prediction (data[idx]):", idx)
    print("Original vector at this index:", data[idx].tolist())

    hist_x = data[idx - args.history_steps:idx].unsqueeze(0)
    original_capacity_for_scheduling = data[idx].unsqueeze(0)

    with torch.no_grad():
        scheduling_strategies, adjusted_loads, main_preds, aux_preds = \
            generator(hist_x, nodes_index_matrix, schedule_from_aux=True, orig_cap=original_capacity_for_scheduling)
        
        pred_next_step = main_preds[0, 0, :]
        adjusted_load_next_step = adjusted_loads[0, 0, :]
        transfer_strategy_next_step = scheduling_strategies[0, 0, :, :]

    print("\nPredicted next frame (from history):")
    print(pred_next_step.tolist())

    print("\nAdjusted final vector after transfers (for next frame):")
    print(adjusted_load_next_step.tolist())

    print("\nTotal Capacity Transferred Between Nodes (for the next frame):")
    transfers = transfer_strategy_next_step 
    for from_node in range(transfers.shape[0]):
        for to_node in range(transfers.shape[1]):
            amount = transfers[from_node, to_node].item()
            if amount > 0:
                orig_capacity_node = original_capacity_for_scheduling[0, from_node].item()
                percentage = (amount / orig_capacity_node) * 100 if orig_capacity_node > 0 else 0.0
                print(f"Node {from_node} -> Node {to_node}: {amount:.2f} (from original {orig_capacity_node:.2f}, {percentage:.2f}%)")
    print('-----\n\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or predict with the ST-GNN model for workload balancing.")
    parser.add_argument('--cites_file', type=str, required=True, help='Path to the bandwidth.cites file.')
    parser.add_argument('--data_file', type=str, required=True, help='Path to the data file (e.g., Data2.csv or s01_person.json).')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for the optimizer.')
    parser.add_argument('--output_model', type=str, default='model_super_factory.pt', help='Path to save the trained model.')
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], default='train', 
                        help='Mode to run: "train" for training, "predict" for prediction.')
    parser.add_argument('--load_model', type=str, default=None, 
                        help='Path to a pre-trained model to load before training. If not provided, training starts from scratch.')
    
    parser.add_argument('--num_nodes', type=int, default=16, help='Number of nodes in the graph.')
    parser.add_argument('--embed_dim', type=int, default=32, help='Embedding dimension for the model.')
    parser.add_argument('--threshold', type=float, default=10.0, help='Workload balancing threshold.')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of ST-GNN blocks in the encoder.')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads.')
    parser.add_argument('--forecast_horizon', type=int, default=1, help='Number of future steps to forecast and schedule for.')
    parser.add_argument('--history_steps', type=int, default=16, help='Number of historical steps to consider for prediction.')

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "predict":
        predict(args)