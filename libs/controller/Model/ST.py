import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Tuple
import torch.nn.functional as F
import torch.optim as optim
import sys
import json
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(f"Using device: {device}")  


node, nodes_distance, nodes_nearest_index, data, nodes_index_matrix = None, None, None, None, None
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


def load_data(cites_file,
              data_file,
              fps: int = 15,
              random_noise: bool = False):

    global nodes_distance, nodes_nearest_index, nodes_index_matrix


    nodes_df = pd.read_csv(cites_file, header=None, sep=" ")
    nodes_df = nodes_df[nodes_df[1] != nodes_df[0]]
    nodes_distance = pd.pivot(index=0, columns=1, values=2, data=nodes_df).fillna(9999)
    nodes_nearest_index = nodes_distance.apply(lambda x: np.argsort(x.values)[:2], axis=1)


    num_nodes = len(nodes_distance)
    mat = np.zeros((num_nodes, num_nodes), dtype=float)
    for idx in nodes_nearest_index.index:
        i = idx - 1
        for j in nodes_nearest_index.loc[idx]:
            mat[i, j] = 1
    nodes_index_matrix = torch.from_numpy(mat).float().to(device)


    with open(data_file, 'r') as f:
        raw = json.load(f)
    node_keys = sorted(raw.keys(),
                       key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))

    per_node = []
    for key in node_keys:
        arr = np.array(raw[key]['person'], dtype=float)
        n = arr.shape[0]

        total_secs = n // fps
        if total_secs == 0:
            raise ValueError("less then 1s...")
        arr = arr[: total_secs * fps]          
        sec_vals = arr[fps-1 :: fps]           
        per_node.append(sec_vals)


    data_array = np.stack(per_node, axis=1)   
    data_seconds = torch.from_numpy(data_array.astype(np.float32)).to(device)

    if random_noise:
        data_seconds += torch.randn_like(data_seconds) * 1e-3

    return None, nodes_distance, nodes_nearest_index, data_seconds, nodes_index_matrix




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

# ------------------------- Positional Encoding -------------------------
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
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        self.spatial_attn = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        self.gate = nn.Linear(embed_dim * 2, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        t_out, _ = self.temporal_attn(x, x, x)
        s_out, _ = self.spatial_attn(x, x, x)
        fused = torch.cat([t_out, s_out], dim=-1)
        gate = torch.sigmoid(self.gate(fused))
        fused = gate * t_out + (1 - gate) * s_out
        x2 = self.norm1(x + fused)
        ffn = self.ffn(x2)
        return self.norm2(x2 + ffn)

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_features, out_features)
        
        # Identity or projection for skip connection if dimensions change
        self.shortcut = nn.Identity()
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out + identity # Residual connection


class Generator(nn.Module):
    def __init__(self, embed_dim=32, threshold=10, num_encoder_layers=8, heads=8, forecast_horizon=8, history_steps=32):
        super(Generator, self).__init__()
        self.threshold = threshold
        self.embed_dim = embed_dim
        self.forecast_horizon = forecast_horizon
        self.history_steps = history_steps


        self.input_proj = nn.Linear(1, embed_dim)
        self.position_encoding = PositionalEncoding(embed_dim)
        self.encoder = nn.Sequential(*[SimpleSTBlock(embed_dim, heads) for _ in range(num_encoder_layers)])
        self.decoder_forecast = nn.Linear(embed_dim, forecast_horizon)
        self.aux_decoder_forecast = nn.Linear(embed_dim, forecast_horizon)
        self.output_scale = nn.Parameter(torch.tensor(10.0))
        self.output_bias = nn.Parameter(torch.tensor(10.0))



        self.spatial_embedding = nn.Parameter(torch.randn(16, embed_dim))
        self.structural_embedding = nn.Linear(1, embed_dim)
        self.temporal_fc = nn.Sequential(
            nn.Linear(16, embed_dim),
            nn.GELU()
        )
        mlp_in_dim = 16 * (1 + embed_dim)
        self.net = nn.Sequential(
            ResidualBlock(mlp_in_dim, 128),
            ResidualBlock(128, 256),
            nn.Linear(256, 16 * 16)
        )

    def predict_load(self, hist_x):
        B, P, N = hist_x.shape
        x = hist_x.unsqueeze(-1)
        x = self.input_proj(x)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B * N, P, self.embed_dim)
        x = self.position_encoding(x)
        x = self.encoder(x)
        x_enc = x[:, -1, :]
        main_forecast = self.decoder_forecast(x_enc)
        main_forecast = main_forecast * self.output_scale + self.output_bias
        main_forecast = main_forecast.view(B, N, self.forecast_horizon)
        aux_forecast = self.aux_decoder_forecast(x_enc)
        aux_forecast = aux_forecast * self.output_scale + self.output_bias
        aux_forecast = aux_forecast.view(B, N, self.forecast_horizon)
        main_pred = main_forecast.transpose(1, 2)
        aux_pred = aux_forecast.transpose(1, 2)
        self.latest_prediction = main_pred.detach()
        self.aux_prediction = aux_pred.detach()
        return main_pred, aux_pred

    def compute_schedule(self, pred, original_capacity):
        available = torch.clamp(original_capacity - self.threshold, min=0)
        deficit = torch.clamp(self.threshold - original_capacity, min=0)
        B, N = pred.shape
        temporal_feat = self.temporal_fc(pred)
        temporal_feat_expanded = temporal_feat.unsqueeze(1).expand(-1, N, -1)
        degree = nodes_index_matrix.sum(dim=1, keepdim=True)
        structure_feat = self.structural_embedding(degree)
        effective_spatial_embedding = (self.spatial_embedding + structure_feat).unsqueeze(0).expand(B, -1, -1)
        st_feat = effective_spatial_embedding + temporal_feat_expanded
        x_expanded = pred.unsqueeze(2)
        enriched_node_info = torch.cat([x_expanded, st_feat], dim=2)
        enriched_input = enriched_node_info.view(B, -1)
        base_scores = self.net(enriched_input).view(B, N, N)
        sender_mask = (original_capacity > self.threshold).unsqueeze(2)
        receiver_mask = (original_capacity < self.threshold).unsqueeze(1)
        allowed_mask = sender_mask & receiver_mask & nodes_index_matrix.unsqueeze(0).bool()
        T = iterative_transfer_projection(base_scores, available, deficit, allowed_mask, iterations=10)
        T_int = torch.ceil(T)
        incoming = T_int.sum(dim=1)
        outgoing = T_int.sum(dim=2)
        final_load = original_capacity + incoming - outgoing
        final_load = torch.clamp(final_load, 0, self.threshold)
        self.adjusted_load = final_load.detach()
        return T_int, final_load

    def forward(self, hist_x, schedule_from_aux=True, original_capacity=None):
        main_pred_seq, aux_pred_seq = self.predict_load(hist_x)
        H = main_pred_seq.shape[1]
        T_list, final_load_list = [], []
        for h in range(H):
            pred_h = aux_pred_seq[:, h, :] if schedule_from_aux else main_pred_seq[:, h, :]
            orig_cap = original_capacity if original_capacity is not None else pred_h
            T_h, final_load_h = self.compute_schedule(pred_h, orig_cap)
            T_list.append(T_h)
            final_load_list.append(final_load_h)
        T_all = torch.stack(T_list, dim=1)
        final_load_all = torch.stack(final_load_list, dim=1)
        return T_all, final_load_all, main_pred_seq, aux_pred_seq


def discriminator_loss(original_capacity, scheduling_strategy):
    incoming = scheduling_strategy.sum(dim=1)
    outgoing = scheduling_strategy.sum(dim=2)
    new_capacity = original_capacity + incoming - outgoing
    capacity_excess = torch.relu(new_capacity - 10)
    capacity_penalty = capacity_excess.mean()
    negative_capacity = torch.relu(-new_capacity)
    negative_penalty = negative_capacity.sum()
    rms = torch.sqrt(torch.mean(new_capacity ** 2))
    loss = capacity_penalty + negative_penalty + rms
    return loss, capacity_penalty, negative_penalty, rms




def predict_and_aggregate(model_file: str,
                           cites_file: str,
                           data_file: str,
                           end_second: int = None) -> Dict[Tuple[int,int], float]:

    gen = Generator(forecast_horizon=8, history_steps=32).to(device)
    gen.load_state_dict(torch.load(model_file))

    _, _, _, data_sec, _ = load_data(cites_file, data_file, fps=15)
    data_sec = data_sec.to(device)
    T = data_sec.shape[0]

 
    if end_second is None:
        end_idx = T - 1
    else:
        end_idx = max(gen.history_steps - 1, min(end_second, T - 1))
    start_idx = end_idx - (gen.history_steps - 1)
    hist = data_sec[start_idx:end_idx+1].unsqueeze(0)

    with torch.no_grad():
        main_pred, aux_pred = gen.predict_load(hist)
        T_seq, _, _, _ = gen(hist, schedule_from_aux=False)


    agg = {}
    cnt = {}
    H = T_seq.shape[1]
    for h in range(H):
        step = T_seq[0, h]
        for f in range(step.shape[0]):
            for t in range(step.shape[1]):
                amt = step[f, t].item()
                if amt > 0:
                    orig = main_pred[0, h, f].item()
                    if orig > 0:
                        pct = round(amt / orig * 100, 2)
                        key = (f, t)
                        agg[key] = agg.get(key, 0) + pct
                        cnt[key] = cnt.get(key, 0) + 1

    result = {k: round(agg[k] / cnt[k], 2) for k in agg}


    return result



if __name__ == "__main__":

    decs = predict_and_aggregate("model_super_school.pt", "bandwidth.cites", "s36_person.json", end_second=10)
    print("decs: ",decs)

